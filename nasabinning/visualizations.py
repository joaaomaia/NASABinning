"""
visualizations.py
Gráficos de estabilidade temporal e WoE/event-rate.
Usa Matplotlib (sem cores explícitas, conforme guidelines).
"""

import matplotlib.pyplot as plt
import seaborn as sns

def plot_event_rate_stability(pivot, *, title=None):
    """
    Desenha linhas de event-rate por bin usando Seaborn.
    """
    if pivot is None or pivot.empty:
        raise ValueError("pivot vazio – calcule stability_over_time antes.")

    df_long = (
        pivot.reset_index()
        .melt(id_vars=["variable", "bin"], var_name="safra", value_name="event_rate")
    )

    plt.figure(figsize=(8, 4))
    sns.lineplot(
        data=df_long,
        x="safra",
        y="event_rate",
        hue="variable",
        style="bin",
        markers=False,
        dashes=True,
        linewidth=1
    )
    plt.xlabel("Safra")
    plt.ylabel("Event Rate")
    if title:
        plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    return plt.gcf(), plt.gca()

