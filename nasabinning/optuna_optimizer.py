"""
optuna_optimizer.py
Busca hiperparâmetros ótimos para NASABinner via Optuna.

Função principal
----------------
optimize_bins(X, y, time_col=None, n_trials=20, **base_kwargs)
→ (best_params: dict, fitted_binner: NASABinner)
"""
from __future__ import annotations

from typing import Any, Tuple

import optuna
import pandas as pd

from .binning_engine import NASABinner


# --------------------------------------------------------------------------- #
def _objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    base_kwargs: dict[str, Any],
    time_col: str | None,
) -> float:
    """Função objetivo usada pelo Optuna."""
    params = {
        "max_bins": trial.suggest_int("max_bins", 3, 10),
        "min_bin_size": trial.suggest_float("min_bin_size", 0.01, 0.1),
        "min_event_rate_diff": trial.suggest_float(
            "min_event_rate_diff", 0.01, 0.1
        ),
    }

    # ------------------------------------------------------------------ #
    # remove possíveis chaves conflitantes antes de repassar
    cfg = dict(base_kwargs)
    cfg.pop("min_event_rate_diff", None)
    cfg.pop("strategy_kwargs", None)

    # cria NASABinner candidato
    binner = NASABinner(
        **cfg,
        max_bins=params["max_bins"],
        min_event_rate_diff=params["min_event_rate_diff"],
        strategy_kwargs=dict(min_bin_size=params["min_bin_size"]),
        use_optuna=False,  # evita recursão
    )
    binner.fit(X, y, time_col=time_col)

    # métrica composta que queremos MINIMIZAR
    iv = binner.iv_
    psi = binner._bin_summary_.attrs.get("psi_over_time", 0.0) or 0.0
    n_bins = len(binner._bin_summary_)

    if n_bins < 2:
        # penalização forte / descarta trial
        return 1e6

    cost = -(iv) + 0.5 * psi + 0.01 * n_bins

    # salva valores para análise posterior
    trial.set_user_attr("iv", iv)
    trial.set_user_attr("psi", psi)
    trial.set_user_attr("n_bins", n_bins)

    return cost


# --------------------------------------------------------------------------- #
def optimize_bins(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    time_col: str | None = None,
    n_trials: int = 20,
    **base_kwargs,
) -> Tuple[dict[str, Any], NASABinner]:
    """
    Executa Optuna para achar hiperparâmetros ideais.

    Retorna
    -------
    best_params : dict
        {'max_bins', 'min_bin_size', 'min_event_rate_diff'}
    fitted_binner : NASABinner
        Binner treinado com os melhores hiperparâmetros.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda tr: _objective(tr, X, y, base_kwargs, time_col),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best = study.best_trial
    best_params = {
        "max_bins": best.params["max_bins"],
        "min_bin_size": best.params["min_bin_size"],
        "min_event_rate_diff": best.params["min_event_rate_diff"],
    }

    # ------------------------------------------------------------------ #
    # treina binner final com melhores parâmetros
    cfg = dict(base_kwargs)
    cfg.pop("min_event_rate_diff", None)
    cfg.pop("strategy_kwargs", None)

    final_binner = NASABinner(
        **cfg,
        max_bins=best_params["max_bins"],
        min_event_rate_diff=best_params["min_event_rate_diff"],
        strategy_kwargs=dict(min_bin_size=best_params["min_bin_size"]),
        use_optuna=False,
    ).fit(X, y, time_col=time_col)

    # expõe best_params ao objeto para debug externo se desejado
    final_binner.best_params_ = best_params

    return best_params, final_binner
