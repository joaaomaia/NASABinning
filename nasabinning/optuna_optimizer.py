"""
optuna_optimizer.py
Busca hiperparâmetros ótimos para NASABinner via Optuna.

Função principal
----------------
optimize_bins(
    X, y, *, time_col=None, time_values=None, n_trials=20,
    alpha=0.7, beta=0.2, gamma=0.1, **base_kwargs)
→ (best_params: dict, fitted_binner: NASABinner)

O Optuna apenas ajusta os hiperparâmetros do OptimalBinning. O score retornado
pelo ``_objective`` prioriza a separabilidade temporal das curvas por safra,
computada por ``temporal_separability_score`` e ponderada com IV e KS segundo:

``score = α * separabilidade + β * IV + γ * KS``

onde ``α`` > ``β`` e ``γ`` (valores padrão: 0.7, 0.2 e 0.1).
"""
from __future__ import annotations

from typing import Any, Tuple, Optional

import optuna
import pandas as pd
import logging

from .binning_engine import NASABinner
from .temporal_stability import (
    temporal_separability_score,
    event_rate_by_time,
    ks_over_time,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
def _objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    base_kwargs: dict[str, Any],
    time_col: Optional[str],
    time_values: Optional[pd.Series],
    alpha: float,
    beta: float,
    gamma: float,
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
    df_fit = X.copy()
    if time_col and time_values is not None:
        df_fit[time_col] = time_values

    binner.fit(df_fit, y, time_col=time_col)

    # métricas para avaliação
    iv = binner.iv_
    n_bins = len(binner.bin_summary)

    if time_col and time_values is not None:
        bins = binner.transform(df_fit)[X.columns[0]]
        df_tmp = pd.DataFrame({
            'bin': bins,
            'target': y,
            'time': time_values,
        })
        sep = temporal_separability_score(
            df_tmp, X.columns[0], 'bin', 'target', 'time'
        )
        tbl = (
            df_tmp.groupby(['bin', 'time'])['target']
            .agg(['sum','count']).reset_index()
            .rename(columns={'sum':'event','count':'count'})
        )
        tbl['variable'] = X.columns[0]
        pivot = event_rate_by_time(tbl, 'time')
        ks = ks_over_time(pivot)
    else:
        sep = 0.0
        ks = 0.0

    score = alpha * sep + beta * iv + gamma * ks

    trial.set_user_attr('separability', sep)
    trial.set_user_attr('iv', iv)
    trial.set_user_attr('ks', ks)
    trial.set_user_attr('n_bins', n_bins)
    trial.set_user_attr('score', score)

    logger.info(
        f"Trial {trial.number}: score={score:.4f}, sep={sep:.4f}, iv={iv:.4f}, ks={ks:.4f}"
    )

    return score


# --------------------------------------------------------------------------- #
def optimize_bins(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    time_col: str | None = None,
    time_values: Optional[pd.Series] = None,
    n_trials: int = 20,
    alpha: float = 0.7,
    beta: float = 0.2,
    gamma: float = 0.1,
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
    study = optuna.create_study(direction="maximize")

    # ajustando verbose do optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(
        lambda tr: _objective(
            tr, X, y, base_kwargs, time_col, time_values, alpha, beta, gamma
        ),
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

    df_final = X.copy()
    if time_col and time_values is not None:
        df_final[time_col] = time_values

    final_binner = NASABinner(
        **cfg,
        max_bins=best_params["max_bins"],
        min_event_rate_diff=best_params["min_event_rate_diff"],
        strategy_kwargs=dict(min_bin_size=best_params["min_bin_size"]),
        use_optuna=False,
    ).fit(df_final, y, time_col=time_col)

    # expõe best_params ao objeto para debug externo se desejado
    final_binner.best_params_ = best_params

    return best_params, final_binner
