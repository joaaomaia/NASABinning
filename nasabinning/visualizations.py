"""
visualizations.py
Gráficos de estabilidade temporal e WoE/event-rate.
Usa Matplotlib (sem cores explícitas, conforme guidelines).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import FuncFormatter

HEX_BASE = "#023059"   # Azul escuro de referência
HEX_MIN  = "#B5C1CD"   # Azul claro mínimo permitido


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
    Gera um gráfico por variável, exibindo event rate em porcentagem (%).
    Linhas = bins (com legendas textuais).

    Parâmetros
    ----------
    pivot : DataFrame
        DataFrame pivotado (índice = (variable, bin_code_int), colunas = yyyymm).
    label_mapper : Callable[var] -> dict[int, str]
        Função que mapeia o código inteiro de cada bin para o intervalo textual,
        exatamente como aparece em `binner._bin_summary_`.
    title_prefix : str | None
        Título prefixo para cada figura.
    time_col_label : str | None
        Legenda a ser usada no eixo X (ex.: "AnoMesReferencia").
    figsize : tuple[int, int]
        Tamanho da figura em polegadas (largura, altura).
    """

    if pivot is None or pivot.empty:
        raise ValueError("pivot vazio — calcule stability_over_time primeiro.")

    for var in pivot.index.get_level_values("variable").unique():
        # ---------- prepara os dados longos ------------------------------------
        mapping = label_mapper(var)

        df_long = (
            pivot.loc[var]                             # seleciona a variável
            .reset_index()                              # ['bin', safra1, safra2, ...]
            .astype({"bin": int})                       # garante inteiro
            .melt(id_vars="bin", var_name="safra", value_name="event_rate")
            .replace({"bin": mapping})                  # código → intervalos textuais
            .rename(columns={"bin": "Bin"})
        )

        # Agrupa para garantir apenas um ponto por (Bin, safra)
        df_long = (
            df_long.groupby(["Bin", "safra"], as_index=False)
            .agg(event_rate=("event_rate", "mean"))
            .sort_values("safra")
        )

        # Converte event_rate para porcentagem
        df_long["event_rate"] = df_long["event_rate"] * 100

        # Converte eixo X para string “yyyymm” 
        df_long["safra"] = df_long["safra"].astype(str)

        # ---------- paleta coerente (claro → escuro) ----------------------------
        means = df_long.groupby("Bin")["event_rate"].mean().sort_values()
        palette = {
            b: c for b, c in zip(means.index, _blend_palette(len(means)))
        }

        # ---------- plot -------------------------------------------------------
        plt.figure(figsize=figsize)
        ax = sns.lineplot(
            data=df_long,
            x="safra",
            y="event_rate",
            hue="Bin",
            palette=palette,
            marker="o",
            linewidth=1.5,
            estimator=None,
            errorbar=None,
        )

        # Título e rótulos
        if title_prefix:
            plt.title(f"{title_prefix} – {var}")
        plt.xlabel(time_col_label or "Safra")
        plt.ylabel("Event Rate (%)")

        # Formata eixo Y para duas casas decimais
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.2f}"))

        # Remover gridlines de fundo
        ax.set_facecolor("white")
        ax.grid(False)

        # Legenda abaixo do gráfico
        plt.legend(
            title="Bin",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=3,
            frameon=False,
            fontsize=8,
        )

        plt.tight_layout()
        plt.show()
