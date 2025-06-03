# 
"""
binning_engine.py
Orquestra a escolha de estratégia (supervised / unsupervised),
aplica refinamentos e expõe interface scikit-learn-compatível.
"""
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .strategies import get_strategy  # factory function (ver abaixo)
from .refinement import refine_bins   # ainda será implementado
from .metrics import iv               # placeholder

from .temporal_stability import event_rate_by_time
from .visualizations import plot_event_rate_stability
import woodwork as ww


class NASABinner(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        strategy: str = "supervised",
        # hiper-parâmetros globais
        max_bins: int = 6,                       #  ←  NOVO
        min_event_rate_diff: float = 0.02,
        monotonic: str | None = None,
        check_stability: bool = False,
        use_optuna: bool = False,
        time_col: str | None = None,
        force_categorical: list[str] | None = None,
        force_numeric: list[str] | None = None,
        strategy_kwargs: dict | None = None,
    ):
        self.strategy = strategy
        self.max_bins = max_bins                #  ←  guarda
        self.min_event_rate_diff = min_event_rate_diff
        self.monotonic = monotonic
        self.check_stability = check_stability
        self.use_optuna = use_optuna
        self.time_col = time_col
        self.force_categorical = force_categorical or []
        self.force_numeric = force_numeric or []

        # ——— normaliza strategy_kwargs ———
        strategy_kwargs = strategy_kwargs or {}
        if "strategy_kwargs" in strategy_kwargs:
            nested = strategy_kwargs.pop("strategy_kwargs")
            for k, v in nested.items():
                strategy_kwargs.setdefault(k, v)

        # garante que max_bins chegue à strategy caso ela use
        strategy_kwargs.setdefault("max_bins", self.max_bins)
        self.strategy_kwargs = strategy_kwargs

        # internos
        self._fitted_strategy = None
        self._bin_summary_ = None


    def _apply_overrides(self, df_ww):
        """Força tags conforme usuário."""
        for col in self.force_categorical:
            if col in df_ww.columns:
                df_ww.ww.set_semantic_tags(col, {"category"})
        for col in self.force_numeric:
            if col in df_ww.columns:
                df_ww.ww.set_semantic_tags(col, {"numeric"})
        return df_ww


    # ------------------------------------------------------------------ #
    def fit(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            *,
            time_col: str | None = None
            ):
        """Treina o binner.  Se use_optuna=True, otimiza coluna-a-coluna."""
        # ---------- validações -------------------------------------------
        assert isinstance(X, pd.DataFrame), "X deve ser um DataFrame"
        assert isinstance(y, pd.Series),   "y deve ser uma Series"

        # --------- time_col definido? (opcional) --------------------------
        time_col = time_col or self.time_col
        self.time_col = time_col  # garante que será armazenado mesmo que só apareça agora

        # --------- Woodwork: inferência e overrides -----------------------
        ww_df = X.copy()
        ww_df.ww.init()                     # cria schema padrão
        ww_df = self._apply_overrides(ww_df)
        self.schema_ = ww_df.ww.schema      # salva para debug

        # listas de colunas
        num_cols = [
            c for c in ww_df.columns if "numeric" in ww_df.ww.logical_types[c].standard_tags
        ]
        cat_cols = [
            c for c in ww_df.columns if "category" in ww_df.ww.logical_types[c].standard_tags
        ]
        self._ignored_cols = [c for c in ww_df.columns if c not in num_cols + cat_cols]

        X = ww_df                             # segue como DataFrame pandas+ww



        # ==================================================================
        # 1. OPTUNA → um estudo por variável
        # ==================================================================
        if self.use_optuna:
            from .optuna_optimizer import optimize_bins
            self._per_feature_binners = {}
            summaries, self.best_params_, self.iv_dict_ = [], {}, {}

            n_trials = self.strategy_kwargs.pop("n_trials", 20)
            base_kwargs = dict(
                strategy=self.strategy,
                min_event_rate_diff=self.min_event_rate_diff,
                monotonic=self.monotonic,
                check_stability=self.check_stability,
            )

            for col in num_cols + cat_cols:
                best, b_col = optimize_bins(
                    X[[col]], y, time_col=time_col, n_trials=n_trials, **base_kwargs
                )
                self._per_feature_binners[col] = b_col
                self.best_params_[col] = best
                self.iv_dict_[col] = b_col.iv_
                summaries.append(b_col._bin_summary_)

            self._bin_summary_ = pd.concat(summaries, ignore_index=True)
            self.iv_ = sum(self.iv_dict_.values())

            if time_col and time_col in self._bin_summary_.columns:
                from .temporal_stability import event_rate_by_time
                self._pivot_ = event_rate_by_time(self._bin_summary_, time_col)
            else:
                self._pivot_ = None

            self._fitted_strategy = None
            return self

        # ==================================================================
        # 2. Fluxo tradicional (sem Optuna)
        # ==================================================================

        # Achata eventual aninhamento strategy_kwargs
        if "strategy_kwargs" in self.strategy_kwargs:
            nested = self.strategy_kwargs.pop("strategy_kwargs")
            for k, v in nested.items():
                # mantém valor existente se já definido externamente
                self.strategy_kwargs.setdefault(k, v)

        self._fitted_strategy = get_strategy(self.strategy, **self.strategy_kwargs)
        self._fitted_strategy.fit(X, y, monotonic_trend=self.monotonic)

        from .refinement import refine_bins
        self._bin_summary_ = refine_bins(
            self._fitted_strategy.bin_summary_,
            min_er_delta=self.min_event_rate_diff,
            trend=self.monotonic,
            time_col=time_col,
            check_stability=self.check_stability,
        )

        # Pivot global (se aplicável)
        if time_col and time_col in self._bin_summary_.columns:
            from .temporal_stability import event_rate_by_time
            self._pivot_ = event_rate_by_time(self._bin_summary_, time_col)
        else:
            self._pivot_ = None

        # IV único
        from .metrics import iv
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
    # ------------------------------------------------------------------ #
    def stability_over_time(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        time_col: str,
    ) -> pd.DataFrame:
        """
        Calcula event-rate por bin ao longo das safras.

        Retorna um DataFrame pivotado:
            index   → (variable, bin_code_int)
            columns → valores únicos de `time_col` (yyyymm)
            values  → event_rate (0-1)
        """
        
        if time_col not in X.columns:
            raise KeyError(
                f"time_col='{time_col}' não está em X. "
                "Inclua a coluna de safra no DataFrame passado."
            )
        # -------------------------------------------------------------- #
        # 1) Gera DataFrame com códigos de bin para cada variável
        # -------------------------------------------------------------- #
        if getattr(self, "_fitted_strategy", None) is not None:
            # modo tradicional (um único strategy)
            X_bins = self._fitted_strategy.transform(X.drop(columns=[time_col], errors="ignore"))
        elif hasattr(self, "_per_feature_binners"):
            # modo Optuna por feature
            parts = []
            for col, binner_col in self._per_feature_binners.items():
                code = binner_col._fitted_strategy.transform(X[[col]]).rename(columns={col: col})
                parts.append(code)
            X_bins = pd.concat(parts, axis=1)
        else:
            raise RuntimeError("Binner ainda não foi treinado.  Chame .fit() antes.")

        # -------------------------------------------------------------- #
        # 2) Concatena target e safra
        # -------------------------------------------------------------- #
        df_aux = pd.concat(
            [X_bins, y.rename("target"), X[time_col]],
            axis=1,
        )

        # -------------------------------------------------------------- #
        # 3) Calcula event-rate por (var, bin, safra)
        # -------------------------------------------------------------- #
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

        # -------------------------------------------------------------- #
        # 4) Pivot final (index = (variable, bin), columns = safra)
        # -------------------------------------------------------------- #
        pivot = (
            df_rate.pivot_table(
                index=["variable", "bin"],
                columns=time_col,
                values="event_rate",
                fill_value=0,
            )
            .sort_index(axis=1)
            .sort_index()
        )
        return pivot


    def _bin_code_to_label(self, var: str) -> dict:
        """
        Retorna um dicionário {bin_code -> intervalo_textual} para a variável `var`.
        Funciona tanto se o código estiver salvo como int, float ou index posicional.
        """
        bs = self._bin_summary_.loc[self._bin_summary_["variable"] == var].copy()

        # Tenta adivinhar qual coluna guarda o código interno
        for cand in ("bin_code", "bin_code_float", "bin_code_int"):
            if cand in bs.columns:
                key_col = cand
                break
        else:  # fallback: usar a posição dos bins
            bs = bs.reset_index(drop=True)
            bs["__pos__"] = bs.index.astype(float)
            key_col = "__pos__"

        # Garante que as chaves sejam do mesmo tipo que chegará no pivot
        return {bs[key_col].iloc[i]: str(bs["bin"].iloc[i]) for i in range(len(bs))}


    # ------------------------------------------------------------------ #
    def plot_event_rate_stability(self, pivot, **kwargs):
        """
        Wrapper fino que delega ao módulo `visualizations`.
        Permite chamar:  binner.plot_event_rate_stability(pivot, ...)
        """
        from nasabinning.visualizations import plot_event_rate_stability
        return plot_event_rate_stability(pivot, binner=self, **kwargs)
