# equal-width, equal-freq, k-means, KBinsDiscretizer

"""
unsupervised.py
Binagem não supervisionada para qualquer número de colunas com
KBinsDiscretizer (uniform, quantile, k-means).
"""

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


class UnsupervisedBinning:
    def __init__(
        self,
        method: str = "quantile",    # "uniform" | "quantile" | "kmeans"
        n_bins: int = 10,
    ):
        if method not in {"uniform", "quantile", "kmeans"}:
            raise ValueError("method must be uniform, quantile or kmeans")
        self.method = method
        self.n_bins = n_bins
        self._kbd = None

    # -------------------------------------------------------------- #
    def fit(self, X: pd.DataFrame, y=None, **kwargs):
        self._kbd = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode="ordinal",
            strategy=self.method,
        )
        self._kbd.fit(X)
        return self

    # -------------------------------------------------------------- #
    def transform(self, X: pd.DataFrame, return_woe=False):
        if return_woe:
            raise NotImplementedError("WoE requer target supervisionado.")
        Xt = self._kbd.transform(X)
        return pd.DataFrame(Xt, columns=X.columns, index=X.index)
