# equal-width, equal-freq, k-means, KBinsDiscretizer
"""
Binagem não supervisionada: quantile, width, k-means.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

class UnsupervisedBinning:
    def __init__(self, method="quantile", n_bins=10):
        self.method = method
        self.n_bins = n_bins
        self._kbd = None

    def fit(self, X: pd.DataFrame, y=None, **kwargs):
        self._kbd = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode="ordinal",
            strategy=self.method,
        )
        self._kbd.fit(X.values)
        return self

    def transform(self, X: pd.DataFrame, **kwargs):
        Xt = self._kbd.transform(X.values)
        return pd.DataFrame(Xt, columns=X.columns, index=X.index)
