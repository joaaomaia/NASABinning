"""
Executa várias configurações de NASABinner e consolida métricas
(IV, #bins, PSI se houver time_col).
"""

from __future__ import annotations
import pandas as pd
from typing import List, Dict, Any
from .binning_engine import NASABinner


class BinComparator:
    def __init__(self, configs: List[Dict[str, Any]], time_col: str | None = None):
        self.configs = configs
        self.time_col = time_col
        self.results_ = []

    # -------------------------------------------------------------- #
    def fit_compare(self, X: pd.DataFrame, y: pd.Series):
        for cfg in self.configs:
            name = cfg.pop("name", None) or cfg.get("strategy", "binner")
            binner = NASABinner(**cfg)
            binner.fit(X, y, time_col=self.time_col)
            self.results_.append(
                dict(
                    name=name,
                    strategy=binner.strategy,
                    iv=binner.iv_,
                    n_bins=len(binner._bin_summary_),
                    psi=binner._bin_summary_.attrs.get("psi_over_time", None),
                    binner=binner,
                )
            )
        return pd.DataFrame(self.results_).set_index("name")

    # -------------------------------------------------------------- #
    def to_excel(self, path: str):
        if not self.results_:
            raise RuntimeError("Run fit_compare first.")
        with pd.ExcelWriter(path) as writer:
            summary = self.fit_summary()
            summary.to_excel(writer, sheet_name="summary")
            # salva cada bin table em aba própria
            for res in self.results_:
                res["binner"]._bin_summary_.to_excel(writer, sheet_name=res["name"][:31])

    def fit_summary(self) -> pd.DataFrame:
        if not self.results_:
            raise RuntimeError("Run fit_compare first.")
        cols = ["strategy", "iv", "n_bins", "psi"]
        return pd.DataFrame(self.results_).set_index("name")[cols]
