"""
Geração de relatórios auditáveis (.xlsx / .json) de binagem ou comparação.
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Union
from .binning_engine import NASABinner

PathLike = Union[str, Path]

# ------------------------------------------------------------------ #
def save_binner_report(binner: NASABinner, path: PathLike) -> None:
    """
    Salva tabela de bins, métricas e (opcional) pivot de estabilidade
    em um arquivo Excel ou JSON (detecta pela extensão).
    """
    p = Path(path)
    if p.suffix.lower() == ".json":
        _save_json(binner, p)
    else:  # padrão Excel
        _save_excel(binner, p.with_suffix(".xlsx"))


def _save_excel(binner: NASABinner, path: Path) -> None:
    with pd.ExcelWriter(path) as writer:
        binner._bin_summary_.to_excel(writer, sheet_name="bin_table", index=False)
        meta = pd.DataFrame(
            {
                "metric": ["IV", "n_bins", "PSI_over_time"],
                "value": [
                    binner.iv_,
                    len(binner._bin_summary_),
                    binner._bin_summary_.attrs.get("psi_over_time"),
                ],
            }
        )
        meta.to_excel(writer, sheet_name="metrics", index=False)

        if hasattr(binner, "_pivot_"):
            binner._pivot_.to_excel(writer, sheet_name="pivot_event_rate")

# ------------------------------------------------------------------ #
def _save_json(binner: NASABinner, path: Path) -> None:
    info = {
        "iv": binner.iv_,
        "n_bins": len(binner._bin_summary_),
        "psi_over_time": binner._bin_summary_.attrs.get("psi_over_time"),
        "bin_table": binner._bin_summary_.to_dict(orient="records"),
    }
    path.write_text(pd.json.dumps(info, indent=2))
