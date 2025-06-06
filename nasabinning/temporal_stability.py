"""
temporal_stability.py
Avalia a robustez de cada bin ao longo do tempo e a separação entre suas curvas.

Funções principais
------------------
event_rate_by_time(df, time_col)     -> DataFrame pivotado
stability_table(df_pivot)            -> métricas por bin
psi_over_time(df_pivot)              -> PSI entre 1ª e última safra
ks_over_time(df_pivot)               -> KS entre 1ª e última safra
temporal_separability_score(df, variable, bin_col, target_col, time_col)
    -> escore médio de distância entre curvas de cada bin
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

# ------------------------------------------------------------------ #
def temporal_separability_score(
    df: pd.DataFrame,
    variable: str,
    bin_col: str,
    target_col: str,
    time_col: str,
    *,
    penalize_inversions: bool = False,
    penalize_low_freq: bool = False,
) -> float:
    """Escore médio de distância absoluta entre curvas dos bins."""
    tbl = (
        df.groupby([bin_col, time_col])[target_col]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "event", "count": "count"})
    )
    tbl["variable"] = variable
    pivot = event_rate_by_time(tbl, time_col)

    n_bins = pivot.shape[0]
    if n_bins < 2:
        return 0.0

    curves = pivot.to_numpy()
    dists = []
    for i in range(n_bins):
        for j in range(i + 1, n_bins):
            dists.append(np.abs(curves[i] - curves[j]).mean())

    score = float(np.mean(dists))

    if penalize_low_freq:
        freq = tbl.groupby(bin_col)["count"].min()
        low = (freq < 30).sum()
        score -= 0.1 * low

    if penalize_inversions:
        for idx in pivot.index:
            values = pivot.loc[idx].values
            trend = np.sign(np.diff(values))
            if np.unique(trend[trend != 0]).size > 1:
                score -= 0.1

    return score
