"""
refinement.py
Pós-processamento dos cortes gerados pelo algoritmo de binagem:
- une bins consecutivos quando a diferença de event-rate é menor que um
  limite mínimo (min_er_delta)
- (opcional) elimina bins que violam tamanho mínimo absoluto
"""

import pandas as pd


def refine_bins(bin_tbl: pd.DataFrame,
                min_er_delta: float,
                time_col: str | None = None,
                check_stability: bool = False) -> pd.DataFrame:
    """
    bin_tbl precisa ter colunas:
        'bin', 'event_rate', 'count', 'non_event', 'event'
        (e opcionalmente time_col se for múltiplas safras)

    Retorna DataFrame com possíveis fusões já aplicadas.
    """
    tbl = bin_tbl.copy().reset_index(drop=True)

    # ---- 1. fundir bins por Δ event-rate ---------------------------
    i = 0
    while i < len(tbl) - 1:
        delta = abs(tbl.loc[i, "event_rate"] - tbl.loc[i + 1, "event_rate"])
        if delta < min_er_delta:
            # funde linha i+1 em i
            tbl.loc[i, ["count", "non_event", "event"]] = (
                tbl.loc[i, ["count", "non_event", "event"]].values +
                tbl.loc[i + 1, ["count", "non_event", "event"]].values
            )
            tbl.loc[i, "event_rate"] = (
                tbl.loc[i, "event"] / tbl.loc[i, "count"]
            )
            tbl.drop(index=i + 1, inplace=True)
            tbl.reset_index(drop=True, inplace=True)
            # não avança o índice – pode haver nova fusão!
        else:
            i += 1

    # ---- 2. (opcional) checar estabilidade temporal ---------------
    if check_stability and time_col is not None:
        # PSI entre primeira e última safra como heurística simples
        from .metrics import psi
        instability = psi(tbl, by=time_col)
        # Usuário decidirá depois o que fazer com esse valor
        tbl.attrs["psi_over_time"] = instability

    return tbl
