"""
refinement.py
Pós-processamento dos cortes:
1. Garante diferença mínima de event rate (Δ ER) entre bins vizinhos.
2. Mantém monotonicidade asc/desc se solicitado.
3. (opcional) armazena PSI de 1.ª × última safra.
"""

from __future__ import annotations
import pandas as pd


def _check_monotonic(series: pd.Series, trend: str) -> bool:
    if trend == "ascending":
        return series.is_monotonic_increasing
    if trend == "descending":
        return series.is_monotonic_decreasing
    raise ValueError("trend must be 'ascending', 'descending' or None")


def refine_bins(
    bin_tbl: pd.DataFrame,
    *,
    min_er_delta: float,
    trend: str | None = None,          # "ascending"/"descending"
    time_col: str | None = None,
    check_stability: bool = False,
) -> pd.DataFrame:
    """Retorna DataFrame com bins possivelmente fundidos."""
    tbl = bin_tbl.copy().reset_index(drop=True)

    # -------------------------------------------------- #
    # 1) Fusão por Δ event rate
    i = 0
    while i < len(tbl) - 1:
        delta = abs(tbl.at[i, "event_rate"] - tbl.at[i + 1, "event_rate"])
        if delta < min_er_delta:
            tbl = _merge(tbl, i, i + 1)
        else:
            i += 1

    # -------------------------------------------------- #
    # 2) Monotonicidade global (se houver)
    if trend is not None and not _check_monotonic(tbl["event_rate"], trend):
        # fusão iterativa do par que menos viola monotonicidade
        while not _check_monotonic(tbl["event_rate"], trend) and len(tbl) > 2:
            # encontra virada de sinal
            diff = tbl["event_rate"].diff().fillna(0)
            bad = diff[diff * (1 if trend == "ascending" else -1) < 0].index
            # funde o primeiro par que quebra a regra
            idx = bad[0] - 1
            tbl = _merge(tbl, idx, idx + 1)

    # -------------------------------------------------- #
    # 3) PSI ao longo do tempo (opcional)
    if check_stability and time_col is not None:
        from .temporal_stability import event_rate_by_time, psi_over_time
        pivot = event_rate_by_time(tbl, time_col)
        tbl.attrs["psi_over_time"] = psi_over_time(pivot)

    return tbl


def _merge(df: pd.DataFrame, i: int, j: int) -> pd.DataFrame:
    """Fundir linhas i e j, mantendo ordem."""
    df.at[i, ["count", "non_event", "event"]] = (
        df.loc[i, ["count", "non_event", "event"]].values +
        df.loc[j, ["count", "non_event", "event"]].values
    )
    df.at[i, "event_rate"] = df.at[i, "event"] / df.at[i, "count"]
    return df.drop(index=j).reset_index(drop=True)
