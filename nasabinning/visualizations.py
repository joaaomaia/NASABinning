"""
visualizations.py
Gráficos de estabilidade temporal e WoE/event-rate.
Usa Matplotlib (sem cores explícitas, conforme guidelines).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# --------------------------------------------------------- #
# 1C. Obtém event-rate médio de cada bin (por variável)
# --------------------------------------------------------- #
def _bin_event_rate_map(binner, var: str) -> dict[str, float]:
    """
    Retorna {label_intervalo: event_rate_medio} para a variável `var`,
    excluindo Special / Missing.
    """
    bs = (
        binner._bin_summary_[binner._bin_summary_["variable"] == var]
        .loc[~binner._bin_summary_["bin"].isin(["Special", "Missing"])]
        .copy()
    )
    return {str(row["bin"]): row["event_rate"] for _, row in bs.iterrows()}

# --------------------------------------------------------- #
# Traduz códigos de bin (floats) → rótulos de intervalo
# --------------------------------------------------------- #
def _infer_bin_label_map(var: str, grp: pd.DataFrame, label_mapper):
    """
    Ajusta o dicionário {codigo_bin: label_intervalo} mesmo quando o pivot
    traz floats e o _bin_code_to_label usa índices inteiros.
    """
    base_map = label_mapper(var)                # mapa original (idx → texto)
    unique_codes = sorted(grp["bin"].unique())  # códigos que aparecem no gráfico

    # Caso raro: o pivot já tenha as chaves exatas do mapa original
    if all(code in base_map for code in unique_codes):
        return {code: base_map[code] for code in unique_codes}

    # Fallback seguro: associa pela posição ordenada
    return {code: base_map[idx] for idx, code in enumerate(unique_codes)}


# --------------------------------------------------------- #
# Remove completamente as linhas de grade do gráfico
# --------------------------------------------------------- #
def _remove_background_grid(ax):
    ax.grid(False)                    # desliga grid
    for spine in ("top", "right"):    # opcional: limpa bordas superiores
        ax.spines[spine].set_visible(False)


# --------------------------------------------------------- #
# 1A. Formata o eixo-X para exibir YYYYMM na ordem correta
# --------------------------------------------------------- #
def _format_time_axis(ax, ordered_safras: list[int | str], label: str | None):
    """
    Garante que o eixo-X use os valores YYYYMM na ordem crono­ló­gica
    recebida em `ordered_safras`, convertendo-os p/ string.
    """
    xticklabels = [str(v) for v in ordered_safras]
    ax.set_xticks(range(len(ordered_safras)))
    ax.set_xticklabels(xticklabels, rotation=0)
    if label:
        ax.set_xlabel(label)

# --------------------------------------------------------- #
# 1B. Posiciona a legenda na parte inferior, preservando rótulos
# --------------------------------------------------------- #
def _place_legend_bottom(ax, n_items: int):
    """
    Coloca a legenda centralizada abaixo do gráfico.
    Usa número de colunas proporcional à quantidade de itens.
    """
    ncol = max(1, min(n_items, 4))          # até 4 colunas p/ não ficar apertado
    ax.legend(
        title="Bin",
        bbox_to_anchor=(0.5, -0.25),
        loc="upper center",
        ncol=ncol,
        frameon=False,
    )

# ------------------------------------------------------------------ #
# Paleta de azuis (claro → escuro)                                   
# ------------------------------------------------------------------ #
HEX_BASE = "#023059"             # Azul mais escuro
HEX_MIN  = "#B5C1CD"             # Azul mais claro permitido

def _blend_palette(n: int) -> list[str]:
    """Gera `n` tons de azul do claro (HEX_MIN) ao escuro (HEX_BASE)."""
    if n <= 0:
        return []
    return sns.blend_palette([HEX_MIN, HEX_BASE], n, as_cmap=False)


# ------------------------------------------------------------------ #
# Gráfico principal de estabilidade temporal                         
# ------------------------------------------------------------------ #
def plot_event_rate_stability(
    pivot: pd.DataFrame,
    *,
    binner,
    label_mapper=None,
    title_prefix: str | None = "Estabilidade temporal",
    time_col_label: str | None = None,
    figsize=(12, 4),
):
    if label_mapper is None:
        label_mapper = lambda var: binner._bin_code_to_label(var)

    df_long = (
        pivot
        .reset_index()
        .melt(id_vars=["variable", "bin"], var_name="safra", value_name="event_rate")
        .dropna(subset=["event_rate"])
    )

    # loop por variável
    for var, grp in df_long.groupby("variable", sort=False):
        # ---------- mapeia código → texto ----------
        code2label = _infer_bin_label_map(var, grp, label_mapper)
        grp = grp.assign(BinLabel=grp["bin"].map(code2label))

        # ---------- remove Special / Missing ----------
        grp = grp[~grp["BinLabel"].str.contains("Special|Missing", case=False, na=False)]
        if grp.empty:
            continue  # nada a plotar

        # ---------- event-rate médio p/ ordenar cores ----------
        er_map = _bin_event_rate_map(binner, var)
        # usa somente labels que estão no grp
        er_map = {lbl: er_map.get(lbl, float("nan")) for lbl in grp["BinLabel"].unique()}

        # ordena pelo event-rate asc (NaN primeiro) ⇒ +escuro = +alto
        ordered_labels = sorted(er_map, key=lambda x: (pd.isna(er_map[x]), er_map[x]))
        palette = _blend_palette(len(ordered_labels))
        color_map = dict(zip(ordered_labels, palette))  # claro→escuro

        ordered_safras = sorted(grp["safra"].unique())
        fig, ax = plt.subplots(figsize=figsize)

        for lbl, g in grp.groupby("BinLabel"):
            g_plot = g.set_index("safra").reindex(ordered_safras).reset_index()
            ax.plot(
                range(len(ordered_safras)),
                g_plot["event_rate"] * 100,
                marker="o",
                linewidth=1.5,
                label=lbl,
                color=color_map[lbl],
            )

        prefix = f"{title_prefix} – " if title_prefix else ""
        ax.set_title(f"{prefix}{var}")
        ax.set_ylabel("Event Rate (%)")

        _format_time_axis(ax, ordered_safras, time_col_label or "Safra")
        _place_legend_bottom(ax, n_items=len(ordered_labels))
        _remove_background_grid(ax)

        plt.tight_layout()
        plt.show()
