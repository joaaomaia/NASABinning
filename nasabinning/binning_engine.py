# binning_engine.py
"""
binning_engine.py
Orquestra a escolha de estratégia (supervised / unsupervised),
aplica refinamentos e expõe interface scikit-learn-compatível.
"""
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .strategies import get_strategy  # factory function (ver abaixo)
from .refinement import refine_bins   # ainda será implementado
from .metrics import iv               # placeholder

from .temporal_stability import event_rate_by_time
from .visualizations import plot_event_rate_stability

class NASABinner(BaseEstimator, TransformerMixin):
    """Binner compatível com scikit-learn."""

    def __init__(
        self,
        strategy: str = "supervised",
        # parâmetros comuns
        min_event_rate_diff: float = 0.02,
        monotonic: Optional[str] = None,  # "ascending", "descending" ou None
        check_stability: bool = False,
        use_optuna: bool = False,
        **strategy_kwargs,
    ):
        self.strategy = strategy
        self.strategy_kwargs = strategy_kwargs
        self.min_event_rate_diff = min_event_rate_diff
        self.monotonic = monotonic
        self.check_stability = check_stability
        self.use_optuna = use_optuna
        # internos
        self._fitted_strategy = None
        self._bin_summary_ = None

    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y: pd.Series, *, time_col: str = None):
        """Treina o binner nos dados."""
        assert isinstance(X, pd.DataFrame), "X deve ser um DataFrame"
        assert isinstance(y, pd.Series), "y deve ser uma Series"

        X = X.copy()

        # -------------------------------------------------- #
        # se optuna estiver ligado, delega a otimização
        if self.use_optuna:
            from .optuna_optimizer import optimize_bins
            best_params, opt_binner = optimize_bins(
                X, y,
                time_col=time_col,
                n_trials=self.strategy_kwargs.pop("n_trials", 20),
                strategy=self.strategy,
                min_event_rate_diff=self.min_event_rate_diff,
                monotonic=self.monotonic,
                check_stability=self.check_stability,
            )
            # clona atributos do melhor
            self.__dict__.update(opt_binner.__dict__)
            
            #Guardar melhor os hiperparâmetros otimizados
            self.best_params_ = best_params
            return self
        # -------------------------------------------------- #

        self._fitted_strategy = get_strategy(
            self.strategy, **self.strategy_kwargs
        )
        self._fitted_strategy.fit(X, y, monotonic_trend=self.monotonic)

        # aplica refinamentos (delta event-rate, estabilidade, etc.)
        self._bin_summary_ = refine_bins(
            self._fitted_strategy.bin_summary_,
            min_er_delta=self.min_event_rate_diff,
            trend=self.monotonic,
            time_col=time_col,
            check_stability=self.check_stability,
        )

        # calcula pivot de estabilidade temporal **apenas se a coluna realmente existir**
        if time_col and time_col in self._bin_summary_.columns:
            self._pivot_ = event_rate_by_time(self._bin_summary_, time_col)
        else:
            self._pivot_ = None

        # Calcula IV como exemplo de métrica armazenada
        self.iv_ = iv(self._bin_summary_)

        return self


    # ------------------------------------------------------------------ #
    def transform(self, X: pd.DataFrame, *, return_woe: bool = False):
        if self._fitted_strategy is None:
            raise RuntimeError("Call fit before transform.")
        Xt = self._fitted_strategy.transform(X, return_woe=return_woe)
        return Xt

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series, **fit_params
    ):
        return self.fit(X, y, **fit_params).transform(X)

    # ------------------------------------------------------------------ #
    def bin_summary_(self):
        """Retorna DataFrame com cortes, event-rate, WoE, IV."""
        return self._bin_summary_

    # -------------------------------------------------- #
    def stability_over_time(self, X: pd.DataFrame, y, time_col: str):
        """
        Calcula event-rate por bin ao longo das safras e devolve DataFrame pivotado.
        Necessita que 'time_col' exista em X.
        """

        if time_col not in X.columns:
            raise KeyError(
                f"time_col='{time_col}' não está em X. "
                "Inclua a coluna de safra no DataFrame passado a stability_over_time."
            )

        # if self._fitted_strategy is None:
        #     raise RuntimeError("Fit the binner first.")

        # aplica cortes (ordinal) em todas as variáveis
        X_bins = self._fitted_strategy.transform(X)

        # junta target e safra
        df_aux = pd.concat([X_bins, y.rename("target"), X[time_col]], axis=1)

        # para cada variável, cada bin e cada safra → event-rate
        out = []
        for var in X_bins.columns:
            grp = (
                df_aux.groupby([time_col, var])["target"]
                .agg(["sum", "count"])
                .reset_index()
                .rename(columns={"sum": "event", "count": "total", var: "bin"})
            )
            grp["event_rate"] = grp["event"] / grp["total"]
            grp["variable"] = var
            out.append(grp)

        df_rate = pd.concat(out, ignore_index=True)

        pivot = (
            df_rate.pivot_table(
                index=["variable", "bin"],
                columns=time_col,
                values="event_rate",
                fill_value=0
            )
            .sort_index(axis=1)
            .sort_index()
        )
        return pivot

    # ------------------------------------------------------------------ #
    def plot_event_rate_stability(self, pivot=None, *, title=None, **plot_kwargs):
        """
        Gera gráfico de event-rate por bin ao longo das safras.

        Parâmetros
        ----------
        pivot : pd.DataFrame | None
            DataFrame pivotado, tal como retornado por stability_over_time.
            Se None, tenta usar self._pivot_.
        title : str | None
            Título opcional para o gráfico.
        **plot_kwargs : dict
            Argumentos extras repassados ao visualizador.

        Retorna
        -------
        (fig, ax) : Matplotlib Figure e Axes
        """
        from .visualizations import plot_event_rate_stability as _plot

        if pivot is None:
            if getattr(self, "_pivot_", None) is None:
                raise ValueError(
                    "Nenhum pivot encontrado. "
                    "Passe um pivot explícito ou chame stability_over_time primeiro."
                )
            pivot = self._pivot_

        return _plot(pivot, title=title, **plot_kwargs)
