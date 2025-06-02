"""
Wrapper de OptimalBinning — apenas binagem supervisionada.
"""
from optbinning import OptimalBinning
import pandas as pd

class SupervisedBinning:
    def __init__(self, max_bins: int = 10, min_bin_size: float = 0.05):
        self.max_bins = max_bins
        self.min_bin_size = min_bin_size
        self.bin_summary_ = None
        self._ob = None

    # -------------------------------------------------------------- #
    def fit(self, X: pd.DataFrame, y, monotonic_trend=None):
        # Por enquanto, assume coluna única; depois tratamos multivariável
        col = X.columns[0]
        self._ob = OptimalBinning(
            name=col,
            solver="cp",
            monotonic_trend=monotonic_trend,
            max_n_bins=self.max_bins,
            min_bin_size=self.min_bin_size,
        )
        self._ob.fit(X[col].values, y.values)
        self.bin_summary_ = self._ob.binning_table.build()
        return self

    # -------------------------------------------------------------- #
    def transform(self, X: pd.DataFrame, return_woe=False):
        col = X.columns[0]
        if return_woe:
            X_trans = self._ob.transform(X[col].values, metric="woe")
            return pd.DataFrame({col: X_trans}, index=X.index)
        return pd.DataFrame(
            {col: self._ob.transform(X[col].values)}, index=X.index
        )
