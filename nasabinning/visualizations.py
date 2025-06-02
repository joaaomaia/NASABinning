"""
visualizations.py
Gráficos de estabilidade temporal e WoE/event-rate.
Usa Matplotlib (sem cores explícitas, conforme guidelines).
"""

import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------------------------ #
def plot_event_rate_stability(pivot: pd.DataFrame, *, title=None):
    """
    Desenha uma linha por bin mostrando a evolução do event-rate ao longo
    das safras. Entrada = DataFrame pivotado (mesmo formato do módulo
    temporal_stability).
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for (var, b), row in pivot.iterrows():
        ax.plot(pivot.columns, row.values, label=f"{var} | bin {b}")
    ax.set_xlabel("Safra")
    ax.set_ylabel("Event Rate")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.4, linestyle="--")
    # mostra apenas as quatro primeiras legendas para não poluir
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) <= 4:
        ax.legend(fontsize=8, frameon=False)
    return fig, ax
