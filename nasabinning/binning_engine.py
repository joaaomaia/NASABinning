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
        X = X.copy()
        self._fitted_strategy = get_strategy(
            self.strategy, **self.strategy_kwargs
        )
        self._fitted_strategy.fit(X, y, monotonic_trend=self.monotonic)

        # aplica refinamentos (delta event-rate, estabilidade, etc.)
        self._bin_summary_ = refine_bins(
            self._fitted_strategy.bin_summary_,
            min_er_delta=self.min_event_rate_diff,
            time_col=time_col,
            check_stability=self.check_stability,
        )
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
