import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from src.metricas.utils import *

pasta = "/home/camila/legal-doc-simplification-data/datasets"
arquivos = glob.glob(os.path.join(pasta, "*.parquet.final"))

if not arquivos:
    raise FileNotFoundError(f"Nenhum arquivo .parquet.final encontrado em {pasta}")

saida = "contagem/oficiais/individuais"
os.makedirs(saida, exist_ok=True)

# Configurações de fonte e estilo
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 24,
})

# Limites para o histograma
limites = np.arange(0, 1.5 + 0.1, 0.1)

# Lista de nomes de datasets para os títulos
datasets = [
    "acordaos_tcu",
    "datastf",
    "iudicium_textum",
    "tesemo_v2",
    "mlp_pt_CJPG",
    "mlp_pt_BRCAD-5"
]

# Ordena os arquivos para manter a correspondência com a lista
arquivos = sorted(arquivos)[:6]

for arquivo, titulo in zip(arquivos, datasets):
    print(f"Lendo {arquivo}...")

    df = pd.read_parquet(arquivo)

    if not all(col in df.columns for col in ["original_text", "paraphrase"]):
        print(f" Pulando {arquivo} — colunas não encontradas.")
        continue

    # Calcula compression ratio
    ratios = df.apply(
        lambda row: contar_caracteres(row["paraphrase"]) / contar_caracteres(row["original_text"]),
        axis=1
    )

    plt.figure(figsize=(13, 8))

    # Histograma de densidade
    counts, bins, patches = plt.hist(
        ratios, bins=limites, color='skyblue', edgecolor='black',
        alpha=0.6, density=True
    )

    # KDE
    pd.Series(ratios).plot(kind='kde', color='black', label='KDE')

    plt.xlim(0, 1.5)
    plt.xticks(limites)
    plt.xlabel("Compression Ratio")
    plt.ylabel("Density")
    plt.title(titulo)  # título do gráfico a partir da lista

    # Segundo eixo y: relative frequency
    ax2 = plt.gca().twinx()
    rel_freq = counts * np.diff(bins)
    ax2.set_ylim(0, rel_freq.max() * 1.1)
    ax2.set_ylabel("Relative Frequency")

    ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.grid(False)

    # Salva figura
    nome_figura = os.path.join(saida, f"compression_rate_{titulo}.png")
    plt.savefig(nome_figura, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico salvo em {nome_figura}")
