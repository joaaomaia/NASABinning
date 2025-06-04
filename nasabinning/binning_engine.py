# 
"""
binning_engine.py
Orquestra a escolha de estratégia (supervised / unsupervised),
aplica refinamentos e expõe interface scikit-learn-compatível.
"""
from __future__ import annotations
from typing import List, Optional, Dict
import inspect, pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .strategies import get_strategy  # factory function (ver abaixo)
from .refinement import refine_bins   # ainda será implementado
from .metrics import iv               # placeholder

from .temporal_stability import event_rate_by_time
from .visualizations import plot_event_rate_stability
from .utils.dtypes import search_dtypes
from .strategies import get_strategy


class NASABinner(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        strategy: str = "supervised",
        # hiper-parâmetros globais
        max_bins: int = 6,                       #  ←  NOVO
        min_event_rate_diff: float = 0.02,
        monotonic: str | None = None,
        check_stability: bool = False,
        use_optuna: bool = False,
        time_col: str | None = None,
        force_categorical: list[str] | None = None,
        force_numeric: list[str] | None = None,
        strategy_kwargs: dict | None = None,
    ):
        self.strategy = strategy
        self.max_bins = max_bins                #  ←  guarda
        self.min_event_rate_diff = min_event_rate_diff
        self.monotonic = monotonic
        self.check_stability = check_stability
        self.use_optuna = use_optuna
        self.time_col = time_col
        self.force_categorical = force_categorical or []
        self.force_numeric = force_numeric or []

        # ——— normaliza strategy_kwargs ———
        strategy_kwargs = strategy_kwargs or {}
        if "strategy_kwargs" in strategy_kwargs:
            nested = strategy_kwargs.pop("strategy_kwargs")
            for k, v in nested.items():
                strategy_kwargs.setdefault(k, v)

        # garante que max_bins chegue à strategy caso ela use
        strategy_kwargs.setdefault("max_bins", self.max_bins)
        self.strategy_kwargs = strategy_kwargs

        # internos
        self._fitted_strategy = None
        self.bin_summary = None


    def fit(self, X: pd.DataFrame, y: pd.Series, *, time_col: str | None = None):
        """Treina o binner. Se use_optuna=True, otimiza coluna-a-coluna."""
        assert isinstance(X, pd.DataFrame), "X deve ser um DataFrame"
        assert isinstance(y, pd.Series),    "y deve ser uma Series"

        time_col = time_col or self.time_col
        self.time_col = time_col            # garante persistência

        # ========= 1. Detectar tipos (substitui Woodwork) =============
        num_cols, cat_cols = search_dtypes(
            pd.concat([X, y.rename("target")], axis=1),
            target_col="target",
            limite_categorico=50,
            force_categorical=self.force_categorical,
            verbose=False
        )
        # armazenar para describe_schema()
        self.numeric_cols_ = num_cols
        self.cat_cols_     = cat_cols
        self.ignored_cols_ = [c for c in X.columns if c not in num_cols + cat_cols]

        # ========= 2. Fluxo com Optuna (por feature) ==================
        if self.use_optuna:
            from .optuna_optimizer import optimize_bins

            n_trials = self.strategy_kwargs.pop("n_trials", 20)
            base_kwargs = dict(
                strategy=self.strategy,
                min_event_rate_diff=self.min_event_rate_diff,
                monotonic=self.monotonic,
                check_stability=self.check_stability,
            )
            self._per_feature_binners = {}
            self.best_params_ = {}
            for col in num_cols + cat_cols:
                best, b_col = optimize_bins(
                    X[[col]], y,
                    time_col=time_col,
                    n_trials=n_trials,
                    **base_kwargs
                )
                self._per_feature_binners[col] = b_col
                self.best_params_[col] = best

            # monta bin_summary_ global
            self.bin_summary = pd.concat(
                [b.bin_summary for b in self._per_feature_binners.values()],
                ignore_index=True
            )
            # calcula IV global (soma dos IVs individuais)
            from .metrics import iv
            self.iv_ = self.bin_summary.groupby("variable").apply(iv).sum()
            return self
        
        # ========= 3. Fluxo tradicional  (sem Optuna) ======================
        self._per_feature_binners = {}          # sempre criamos o dicionário
        self.bin_summary       = []

        # ────────── fluxo numérico ──────────
        for col in num_cols:
            strat = get_strategy("supervised", **self.strategy_kwargs)
            strat.fit(X[[col]], y, monotonic_trend=self.monotonic)

            summary = refine_bins(          # <<<<<<
                strat.bin_summary_,         # usa bin_summary_ direto
                min_er_delta=self.min_event_rate_diff,
                trend=self.monotonic,
                time_col=time_col,
                check_stability=self.check_stability,
            )

            # ── limpa linhas indesejadas ───────────────────────────────────────
            summary = summary[
                (summary["count"] > 0) &          # exclui bins vazios
                (~summary["bin"].isin(["Total", "Special", "Missing"]))
            ].reset_index(drop=True)

            self._per_feature_binners[col] = strat
            self.bin_summary.append(summary)

        # ────────── fluxo categórico ────────
        for col in cat_cols:
            from .strategies.categorical import CategoricalBinning
            strat = CategoricalBinning()
            strat.fit(X[[col]], y)

            summary = strat.bin_summary_    # <<<<<< não há refine_bins (já é categórico)

            # ── limpa linhas indesejadas ───────────────────────────────────────
            summary = summary[
                (summary["count"] > 0) &          # exclui bins vazios
                (~summary["bin"].isin(["Total", "Special", "Missing"]))
            ].reset_index(drop=True)

            self._per_feature_binners[col] = strat
            self.bin_summary.append(summary)
           

        # junta tudo num único DataFrame
        self.bin_summary = pd.concat(self.bin_summary, ignore_index=True)

        # IV global  =  soma dos IVs individuais
        from .metrics import iv
        self.iv_ = self.bin_summary.groupby("variable").apply(iv).sum()
        return self


    # ----------------------------------------------------------------
    def transform(self, X: pd.DataFrame, *, return_woe: bool = False):
        out = {}
        for col, b in self._per_feature_binners.items():
            sig = inspect.signature(b.transform)
            kw  = {"return_woe": return_woe} if "return_woe" in sig.parameters else {}
            out[col] = b.transform(X[[col]], **kw)[col]
        return pd.DataFrame(out, index=X.index)


    # ----------------------------------------------------------------
    def describe_schema(self) -> pd.DataFrame:
        """Resumo simples do papel de cada coluna."""
        records = []
        for col in self.numeric_cols_:
            records.append({"col": col, "tipo": "numeric"})
        for col in self.cat_cols_:
            records.append({"col": col, "tipo": "categorical"})
        for col in getattr(self, "ignored_cols_", []):
            records.append({"col": col, "tipo": "ignored"})
        return pd.DataFrame(records)

    # ------------------------------------------------------------------ #
    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series, **fit_params
    ):
        return self.fit(X, y, **fit_params).transform(X)


    def get_bin_mapping(self, column: str) -> pd.DataFrame:
        """
        Retorna DataFrame categoria → bin para a coluna categórica `column`.

        Funciona em três cenários:
        1) NASABinner treinado apenas com essa coluna (categorical);
        2) NASABinner multi-feature + Optuna (usa _per_feature_binners);
        3) Encoder é "woe" (OptimalBinning) ou "ordinal" (fallback).
        """
        # ── localizar o binner da coluna ─────────────────────────────────
        if hasattr(self, "_per_feature_binners") and column in self._per_feature_binners:
            binner_col = self._per_feature_binners[column]
        else:
            if self._fitted_strategy is None:
                raise RuntimeError("O binner ainda não foi treinado.")
            binner_col = self._fitted_strategy        # fluxo simples

        # ── extrair encoder & tipo ───────────────────────────────────────
        if not hasattr(binner_col, "_encoder"):
            raise ValueError(f"A coluna '{column}' não passou por CategoricalBinning.")
        encoder, enc_type = binner_col._encoder

        # ── construir mapeamento ────────────────────────────────────────
        if enc_type == "woe":                     # OptimalBinning
            mapping = encoder.splits["mapping"]   # dict categoria → bin
        elif enc_type == "ordinal":               # fallback
            mapping = encoder.mapping[0]["mapping"]  # dict categoria → código
        else:
            raise RuntimeError("Tipo de encoder desconhecido.")

        return (
            pd.Series(mapping, name="bin")
            .reset_index()
            .rename(columns={"index": "categoria"})
            .sort_values("bin")
            .reset_index(drop=True)
        )


    # ------------------------------------------------------------------ #
    def bin_summary_(self):
        """Retorna DataFrame com cortes, event-rate, WoE, IV."""
        return self.bin_summary

    # ------------------------------------------------------------------ #
    def stability_over_time(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        time_col: str,
    ) -> pd.DataFrame:
        """
        Calcula event-rate por bin ao longo das safras.

        Retorna um DataFrame pivotado:
            index   → (variable, bin_code_int)
            columns → valores únicos de `time_col` (yyyymm)
            values  → event_rate (0-1)
        """
        
        if time_col not in X.columns:
            raise KeyError(
                f"time_col='{time_col}' não está em X. "
                "Inclua a coluna de safra no DataFrame passado."
            )
        # -------------------------------------------------------------- #
        # 1) Gera DataFrame com códigos de bin para cada variável
        # -------------------------------------------------------------- #
        if getattr(self, "_fitted_strategy", None) is not None:
            # modo tradicional (um único strategy)
            X_bins = self._fitted_strategy.transform(X.drop(columns=[time_col], errors="ignore"))
        elif hasattr(self, "_per_feature_binners"):
            # modo Optuna por feature
            parts = []
            for col, binner_col in self._per_feature_binners.items():
                code = binner_col._fitted_strategy.transform(X[[col]]).rename(columns={col: col})
                parts.append(code)
            X_bins = pd.concat(parts, axis=1)
        else:
            raise RuntimeError("Binner ainda não foi treinado.  Chame .fit() antes.")

        # -------------------------------------------------------------- #
        # 2) Concatena target e safra
        # -------------------------------------------------------------- #
        df_aux = pd.concat(
            [X_bins, y.rename("target"), X[time_col]],
            axis=1,
        )

        # -------------------------------------------------------------- #
        # 3) Calcula event-rate por (var, bin, safra)
        # -------------------------------------------------------------- #
        out = []
        for var in X_bins.columns:
            grp = (
                df_aux.groupby([time_col, var])["target"]
                .agg(["sum", "count"])
                .reset_index()
                .rename(columns={"sum": "event", "count": "total", var: "bin"})
            )
            grp["event_rate"] = grp["event"] / grp["total"]
            grp["variable"] = var
            out.append(grp)

        df_rate = pd.concat(out, ignore_index=True)

        # -------------------------------------------------------------- #
        # 4) Pivot final (index = (variable, bin), columns = safra)
        # -------------------------------------------------------------- #
        pivot = (
            df_rate.pivot_table(
                index=["variable", "bin"],
                columns=time_col,
                values="event_rate",
                fill_value=0,
            )
            .sort_index(axis=1)
            .sort_index()
        )
        return pivot


    def _bin_code_to_label(self, var: str) -> dict:
        """
        Retorna um dicionário {bin_code -> intervalo_textual} para a variável `var`.
        Funciona tanto se o código estiver salvo como int, float ou index posicional.
        """
        bs = self.bin_summary.loc[self.bin_summary["variable"] == var].copy()

        # Tenta adivinhar qual coluna guarda o código interno
        for cand in ("bin_code", "bin_code_float", "bin_code_int"):
            if cand in bs.columns:
                key_col = cand
                break
        else:  # fallback: usar a posição dos bins
            bs = bs.reset_index(drop=True)
            bs["__pos__"] = bs.index.astype(float)
            key_col = "__pos__"

        # Garante que as chaves sejam do mesmo tipo que chegará no pivot
        return {bs[key_col].iloc[i]: str(bs["bin"].iloc[i]) for i in range(len(bs))}

    # ------------------------------------------------------------------ #
    def plot_event_rate_stability(self, pivot, **kwargs):
        """
        Wrapper fino que delega ao módulo `visualizations`.
        Permite chamar:  binner.plot_event_rate_stability(pivot, ...)
        """
        from nasabinning.visualizations import plot_event_rate_stability
        return plot_event_rate_stability(pivot, binner=self, **kwargs)
