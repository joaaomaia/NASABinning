"""
refinement.py
Pós-processamento dos cortes:
1. Normaliza nomes de colunas vindos do OptimalBinning.
2. Garante diferença mínima de event rate (Δ ER) entre bins vizinhos.
3. Mantém monotonicidade asc/desc se solicitado.
4. (opcional) armazena PSI de 1.ª × última safra.
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
    """
    Recebe bin_tbl (output de OptimalBinning.binning_table.build()) e
    devolve um DataFrame normalizado com possíveis fusões.

    - Normaliza colunas: ["variable", "bin", "count", "event", "non_event", "event_rate"]
    - Fusão por Δ event-rate mínimo
    - Monotonicidade (se trend definido)
    - PSI entre primeira e última safra (opcional)

    Espera que bin_tbl venha com alguma combinação de:
      "Bin" ou "bin",
      "Count" ou "count",
      "Event" ou "event",
      "Non Event", "Non-Event", "Non-event" ou "non_event",
      "Event Rate" ou "event_rate",
      além de "variable" e possivelmente time_col.
    """
    # 1) Copiar e resetar índice
    tbl = bin_tbl.copy().reset_index(drop=True)

    # 2) Normalizar nomes de coluna

    # variable (presume existir)
    if "variable" not in tbl.columns:
        raise KeyError("nenhuma coluna 'variable' encontrada em bin_tbl")

    # bin
    if "Bin" in tbl.columns:
        tbl["bin"] = tbl["Bin"]
    elif "bin" not in tbl.columns:
        raise KeyError("nenhuma coluna 'Bin' ou 'bin' encontrada em bin_tbl")

    # count
    if "Count" in tbl.columns:
        tbl["count"] = tbl["Count"]
    elif "count" not in tbl.columns:
        raise KeyError("nenhuma coluna 'Count' ou 'count' encontrada em bin_tbl")

    # event
    if "Event" in tbl.columns:
        tbl["event"] = tbl["Event"]
    elif "event" not in tbl.columns:
        raise KeyError("nenhuma coluna 'Event' ou 'event' encontrada em bin_tbl")

    # non_event
    non_cols = [
        c for c in tbl.columns
        if c.lower().replace("-", " ").strip() in {"non event", "nonevent"}
    ]
    if non_cols:
        tbl["non_event"] = tbl[non_cols[0]]
    else:
        # Calcula se não existir explicitamente
        tbl["non_event"] = tbl["count"] - tbl["event"]

    # event_rate
    if "Event Rate" in tbl.columns:
        tbl["event_rate"] = tbl["Event Rate"]
    elif "event_rate" not in tbl.columns:
        # Se não estiver, calcula a partir de event/count
        tbl["event_rate"] = tbl["event"] / tbl["count"]

    # 3) Selecionar apenas colunas existentes
    base_cols = ["variable", "bin", "count", "event", "non_event", "event_rate"]
    if time_col and time_col in tbl.columns:
        base_cols.append(time_col)

    tbl = tbl[base_cols]

    # -------------------------------------------------- #
    # 4) Fusão por Δ event rate
    i = 0
    while i < len(tbl) - 1:
        delta = abs(tbl.at[i, "event_rate"] - tbl.at[i + 1, "event_rate"])
        if delta < min_er_delta:
            tbl = _merge(tbl, i, i + 1)
        else:
            i += 1

    # -------------------------------------------------- #
    # 5) Monotonicidade global (se houver)
    if trend is not None and not _check_monotonic(tbl["event_rate"], trend):
        # fusão iterativa do par que menos viola monotonicidade
        while not _check_monotonic(tbl["event_rate"], trend) and len(tbl) > 2:
            diff = tbl["event_rate"].diff().fillna(0)
            bad = diff[diff * (1 if trend == "ascending" else -1) < 0].index
            if len(bad) == 0:
                break
            idx = bad[0] - 1
            tbl = _merge(tbl, idx, idx + 1)

    # -------------------------------------------------- #
    # 6) PSI ao longo do tempo (opcional)
    if check_stability and time_col and time_col in tbl.columns:
        from .temporal_stability import event_rate_by_time, psi_over_time
        pivot = event_rate_by_time(tbl, time_col)
        tbl.attrs["psi_over_time"] = psi_over_time(pivot)

    return tbl


def _merge(df: pd.DataFrame, i: int, j: int) -> pd.DataFrame:
    """
    Fundir as linhas i e j, somando contagens e recálculando event_rate.
    Mantém a ordem original (linha i permanece, linha j é removida).
    """
    cols = ["count", "non_event", "event"]
    # Soma linha i + linha j para as colunas numéricas
    df.loc[i, cols] = df.loc[[i, j], cols].sum()

    # Recalcula event_rate na linha i
    df.loc[i, "event_rate"] = df.loc[i, "event"] / df.loc[i, "count"]

    # Descarta linha j e reajusta índices
    return df.drop(index=j).reset_index(drop=True)




# """
# refinement.py
# Pós-processamento dos cortes:
# 1. Garante diferença mínima de event rate (Δ ER) entre bins vizinhos.
# 2. Mantém monotonicidade asc/desc se solicitado.
# 3. (opcional) armazena PSI de 1.ª × última safra.
# """

# from __future__ import annotations
# import pandas as pd


# def _check_monotonic(series: pd.Series, trend: str) -> bool:
#     if trend == "ascending":
#         return series.is_monotonic_increasing
#     if trend == "descending":
#         return series.is_monotonic_decreasing
#     raise ValueError("trend must be 'ascending', 'descending' or None")


# def refine_bins(
#     bin_tbl: pd.DataFrame,
#     *,
#     min_er_delta: float,
#     trend: str | None = None,          # "ascending"/"descending"
#     time_col: str | None = None,
#     check_stability: bool = False,
# ) -> pd.DataFrame:
#     """Retorna DataFrame com bins possivelmente fundidos."""
#     tbl = bin_tbl.copy().reset_index(drop=True)

#     # -------------------------------------------------- #
#     # 1) Fusão por Δ event rate
#     i = 0
#     while i < len(tbl) - 1:
#         delta = abs(tbl.at[i, "event_rate"] - tbl.at[i + 1, "event_rate"])
#         if delta < min_er_delta:
#             tbl = _merge(tbl, i, i + 1)
#         else:
#             i += 1

#     # -------------------------------------------------- #
#     # 2) Monotonicidade global (se houver)
#     if trend is not None and not _check_monotonic(tbl["event_rate"], trend):
#         # fusão iterativa do par que menos viola monotonicidade
#         while not _check_monotonic(tbl["event_rate"], trend) and len(tbl) > 2:
#             # encontra virada de sinal
#             diff = tbl["event_rate"].diff().fillna(0)
#             bad = diff[diff * (1 if trend == "ascending" else -1) < 0].index
#             # funde o primeiro par que quebra a regra
#             idx = bad[0] - 1
#             tbl = _merge(tbl, idx, idx + 1)

#     # -------------------------------------------------- #
#     # 3) PSI ao longo do tempo (opcional)
#     if check_stability and time_col is not None:
#         from .temporal_stability import event_rate_by_time, psi_over_time
#         pivot = event_rate_by_time(tbl, time_col)
#         tbl.attrs["psi_over_time"] = psi_over_time(pivot)

#     return tbl


# def _merge(df: pd.DataFrame, i: int, j: int) -> pd.DataFrame:
#     """Fundir linhas i e j, mantendo ordem."""
#     df.at[i, ["count", "non_event", "event"]] = (
#         df.loc[i, ["count", "non_event", "event"]].values +
#         df.loc[j, ["count", "non_event", "event"]].values
#     )
#     df.at[i, "event_rate"] = df.at[i, "event"] / df.at[i, "count"]
#     return df.drop(index=j).reset_index(drop=True)
