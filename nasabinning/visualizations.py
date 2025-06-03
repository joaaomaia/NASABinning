"""
visualizations.py
Gráficos de estabilidade temporal e WoE/event-rate.
Usa Matplotlib (sem cores explícitas, conforme guidelines).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import FuncFormatter

HEX_BASE = "#023059"             # azul mais escuro
HEX_MIN  = "#B5C1CD"             # azul mais claro permitido


def _blend_palette(n: int):
    """Gera n tons de azul do claro (HEX_MIN) ao escuro (HEX_BASE)."""
    if n <= 0: # Se n for 0 ou negativo, retorna uma lista vazia
        return []
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
    Um gráfico por variável mostrando o Event-Rate (%) de cada bin.

    pivot
        index = (variable, bin_code_int) — gerado por stability_over_time
    label_mapper
        função var -> {bin_code_int: intervalo_texto}
    """

    if pivot is None or pivot.empty:
        print("A tabela pivot de entrada está vazia. Nenhum gráfico será gerado.")
        return

    for var in pivot.index.get_level_values("variable").unique():
        # -------- Extração de dados para a variável atual ---------------------
        try:
            df_var_slice = pivot.loc[var]
        except KeyError:
            print(f"Variável '{var}' não encontrada no pivot. Pulando.")
            continue

        if df_var_slice.empty:
            # print(f"Sem dados no pivot para a variável {var}. Gerando gráfico vazio.")
            plt.figure(figsize=figsize)
            ax = plt.gca()
            plot_title = f"{title_prefix} – {var}" if title_prefix else var
            plt.title(f"{plot_title} (Sem dados disponíveis)")
            plt.xlabel(time_col_label or "Safra")
            plt.ylabel("Event Rate (%)")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
            ax.set_facecolor("white")
            ax.grid(False)
            plt.tight_layout()
            plt.show()
            continue
            
        mapping = label_mapper(var)
        if not isinstance(mapping, dict):
            print(f"Atenção: label_mapper para a variável '{var}' não retornou um dicionário. Pulando esta variável.")
            continue

        # Assume que o índice de df_var_slice contém os códigos dos bins
        # e que reset_index() irá nomear a coluna correspondente como 'bin'
        # se o nível do índice se chamar 'bin', ou usa o nome do nível.
        # O código original usa .astype({"bin": int}), implicando que a coluna 'bin' é criada.
        bin_col_name_in_df = df_var_slice.index.name if df_var_slice.index.name is not None else 'bin'
        if df_var_slice.index.name is None : # Se o nível do índice não tiver nome, nomeia para 'bin'
             current_df = df_var_slice.copy() # Evitar SettingWithCopyWarning
             current_df.index.name = 'bin'
             df_long = current_df.reset_index()
        else:
             df_long = df_var_slice.reset_index()


        # -------- Transformação para formato longo (long format) --------------
        try:
            df_long = (
                df_long
                .astype({bin_col_name_in_df: int}) # Converte códigos dos bins para int
                .melt(id_vars=bin_col_name_in_df, var_name="safra", value_name="event_rate")
                .replace({bin_col_name_in_df: mapping}) # Troca códigos por labels textuais
                .rename(columns={bin_col_name_in_df: "Bin"}) # Renomeia coluna para "Bin"
            )
        except Exception as e:
            print(f"Erro ao processar dados para a variável '{var}' (possivelmente devido a nome de coluna de bin): {e}. Pulando.")
            continue
            
        if df_long.empty:
            # print(f"Sem dados para plotar para a variável {var} após processamento. Pulando.")
            plt.figure(figsize=figsize)
            ax = plt.gca()
            plot_title = f"{title_prefix} – {var}" if title_prefix else var
            plt.title(f"{plot_title} (Sem dados para exibir)")
            plt.xlabel(time_col_label or "Safra")
            plt.ylabel("Event Rate (%)")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
            ax.set_facecolor("white")
            ax.grid(False)
            plt.tight_layout()
            plt.show()
            continue

        # Converter taxa de evento para porcentagem
        df_long["event_rate"] = df_long["event_rate"] * 100
        
        # Preencher taxas de evento NaN com 0.0 para garantir que sejam plotadas
        df_long["event_rate"] = df_long["event_rate"].fillna(0.0)
        
        df_long["safra"] = df_long["safra"].astype(str)

        # -------- Consolida duplicados (se houver) ANTES do reindex -----------
        df_long = (
            df_long
            .groupby(["Bin", "safra"], as_index=False)
            .agg(event_rate=("event_rate", "mean"))
        )

        # -------- Completa combinações Bin × Safra que faltam ------------------
        bins_list = sorted(list(df_long["Bin"].unique()))
        safras_list = sorted(list(df_long["safra"].unique()))

        if not bins_list or not safras_list:
            # print(f"Sem bins ou safras para plotar para a variável {var} após processamento. Pulando.")
            # (Gráfico vazio já tratado se df_long estava vazio)
            continue

        full_index = pd.MultiIndex.from_product(
            [bins_list, safras_list], names=["Bin", "safra"]
        )

        df_long = (
            df_long
            .set_index(["Bin", "safra"])
            .reindex(full_index, fill_value=0.0) # Preenche combinações ausentes com 0.0
            .reset_index()
        )
        
        # -------- Paleta de cores ---------------------------------------------
        # Ordena os bins pela média da taxa de evento para uma paleta visualmente ordenada
        # ou usa a lista de bins já ordenada para consistência de cor.
        mean_event_rates = df_long.groupby("Bin")["event_rate"].mean().sort_values()
        ordered_bins_for_palette = mean_event_rates.index.tolist()
        
        palette_colors = _blend_palette(len(ordered_bins_for_palette))
        palette = {b: palette_colors[i] for i, b in enumerate(ordered_bins_for_palette)}

        # -------- Plot --------------------------------------------------------
        plt.figure(figsize=figsize)
        ax = sns.lineplot(
            data=df_long,
            x="safra",
            y="event_rate",
            hue="Bin",
            hue_order=ordered_bins_for_palette, # Garante a ordem da legenda
            palette=palette,
            marker="o",
            linewidth=1.5
        )
        
        plot_title_full = f"{title_prefix} – {var}" if title_prefix else var
        plt.title(plot_title_full)
        plt.xlabel(time_col_label or "Safra")
        plt.ylabel("Event Rate (%)")

        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax.set_facecolor("white")
        ax.grid(False)

        if ordered_bins_for_palette: # Apenas cria legenda se houver bins
            plt.legend(
                title="Bin",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=min(len(ordered_bins_for_palette), 3),
                frameon=False,
                fontsize=8,
            )
        
        plt.tight_layout()
        plt.show()
