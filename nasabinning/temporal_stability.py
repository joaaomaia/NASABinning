"""
temporal_stability.py
Avalia a robustez de cada bin ao longo do tempo.

Funções principais
------------------
event_rate_by_time(df, time_col)     -> DataFrame pivotado
stability_table(df_pivot)            -> métricas por bin
psi_over_time(df_pivot)              -> PSI entre 1ª e última safra
ks_over_time(df_pivot)               -> KS entre 1ª e última safra
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

# ------------------------------------------------------------------ #
def event_rate_by_time(bin_tbl: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Espera bin_tbl com colunas ['variable', 'bin', 'event', 'count', time_col].
    Retorna DataFrame pivotado: index = (variable, bin), columns = safra,
    values = event_rate.
    """
    df = bin_tbl.copy()
    df["event_rate"] = df["event"] / df["count"]
    pivot = (
        df.pivot_table(
            index=["variable", "bin"],
            columns=time_col,
            values="event_rate"
        )
        .sort_index(axis=1)          # ordena safras
        .sort_index()
    )
    return pivot

# ------------------------------------------------------------------ #
def stability_table(pivot: pd.DataFrame) -> pd.DataFrame:
    """Desvio padrão e amplitude por bin."""
    std = pivot.std(axis=1)
    rng = pivot.max(axis=1) - pivot.min(axis=1)
    return pd.DataFrame({"std": std, "range": rng})

# ------------------------------------------------------------------ #
def psi_over_time(pivot: pd.DataFrame) -> float:
    """PSI global entre primeira e última safra."""
    from .metrics import psi
    first, last = pivot.columns[0], pivot.columns[-1]
    df_tmp = pd.DataFrame(
        {
            "expected": pivot[first].values,
            "actual": pivot[last].values
        }
    )
    return psi(df_tmp)

# ------------------------------------------------------------------ #
def ks_over_time(pivot: pd.DataFrame) -> float:
    """KS global entre primeira e última safra (distribuição de event-rate)."""
    first, last = pivot.columns[0], pivot.columns[-1]
    return ks_2samp(pivot[first], pivot[last]).statistic
