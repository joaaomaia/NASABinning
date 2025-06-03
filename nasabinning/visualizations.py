"""
visualizations.py
Gráficos de estabilidade temporal e WoE/event-rate.
Usa Matplotlib (sem cores explícitas, conforme guidelines).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

HEX_BASE = "#023059"   # azul escuro pedido
HEX_MIN  = "#B5C1CD"   # azul mais claro permitido


def _blend_palette(n: int):
    """Gera n tons de HEX_MIN → HEX_BASE (mais escuro = maior índice)."""
    return sns.blend_palette([HEX_MIN, HEX_BASE], n, as_cmap=False)


def plot_event_rate_stability(
    pivot: pd.DataFrame,
    *,
    label_mapper,
    title_prefix: str | None = "Estabilidade temporal",
    time_col_label: str | None = None,
    figsize=(12, 4),
) -> None:
    """
    Um gráfico por variável; linhas = bins (com legendas textuais).

    Parameters
    ----------
    pivot : DataFrame (index = (variable, bin_code_int), columns = yyyymm)
    label_mapper : Callable[var] -> dict[int, str]
        Traduz código do bin (0,1,2,…) para o intervalo textual
        '(-inf, 24.70)', …
    """

    if pivot is None or pivot.empty:
        raise ValueError("pivot vazio — calcule stability_over_time primeiro.")

    for var in pivot.index.get_level_values("variable").unique():
        # ---------- prepara dados longos ------------------------------------
        mapping = label_mapper(var)

        df_long = (
            pivot.loc[var]
            .reset_index()          # ['bin', safra1, safra2, ...]
            .astype({"bin": int})
            .melt(id_vars="bin", var_name="safra", value_name="event_rate")
            .replace({"bin": mapping})          # código → intervalo
            .rename(columns={"bin": "Bin"})
        )

        # garante 1 linha por (Bin, safra)  —— remove duplicados
        df_long = (
            df_long.groupby(["Bin", "safra"], as_index=False)
            .agg(event_rate=("event_rate", "mean"))
            .sort_values("safra")
        )

        # eixo-X string yyyymm
        df_long["safra"] = df_long["safra"].astype(str)

        # ---------- paleta coerente (claro→escuro) --------------------------
        means = df_long.groupby("Bin")["event_rate"].mean().sort_values()
        palette = {b: c for b, c in zip(means.index, _blend_palette(len(means)))}

        # ---------- plot ----------------------------------------------------
        plt.figure(figsize=figsize)
        sns.lineplot(
            data=df_long,
            x="safra",
            y="event_rate",
            hue="Bin",
            palette=palette,
            marker="o",
            linewidth=1.5,
            estimator=None,   #  ←  acrescentar
            errorbar=None,    #  ←  acrescentar
        )

        if title_prefix:
            plt.title(f"{title_prefix} – {var}")
        plt.xlabel(time_col_label or "Safra")
        plt.ylabel("Event Rate")
        plt.gca().set_facecolor("white")
        plt.grid(False)

        # legenda embaixo
        plt.legend(
            title="Bin",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=3,
            frameon=False,
            fontsize=8,
        )

        plt.tight_layout()

    # Não retorna nada → célula não exibirá lista de figuras
    plt.show()
