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

saida = "contagem/compression_rate"
os.makedirs(saida, exist_ok=True)

# Lista para armazenar todos os ratios
todos_ratios = []

for arquivo in arquivos:
    nome = os.path.basename(arquivo)
    print(f"Lendo {nome}...")

    df = pd.read_parquet(arquivo)

    if not all(col in df.columns for col in ["original_text", "paraphrase"]):
        print(f" Pulando {nome} — colunas não encontradas.")
        continue

    # Calcula compression ratio
    ratios = df.apply(
        lambda row: contar_caracteres(row["paraphrase"]) / contar_caracteres(row["original_text"]),
        axis=1
    )

    todos_ratios.extend(ratios.tolist())  # adiciona ao dataset único

# Converte para Series do pandas
todos_ratios = pd.Series(todos_ratios)
print(f"Total de ratios no dataset combinado: {len(todos_ratios)}")

# Limitar o gráfico a 1.5 e criar faixas de 0.1
limites = np.arange(0, 1.5 + 0.1, 0.1)
print(f"Número de faixas: {len(limites)-1}")

plt.figure(figsize=(12, 7))

# Cria o histograma de densidade
counts, bins, patches = plt.hist(
    todos_ratios, bins=limites, color='skyblue', edgecolor='black',
    alpha=0.6, density=True, label='Density'
)

plt.rcParams.update({
    'font.size': 16,         # tamanho base
    'axes.titlesize': 20,    # título do gráfico
    'axes.labelsize': 18,    # rótulos dos eixos
    'xtick.labelsize': 14,   # valores do eixo x
    'ytick.labelsize': 14,   # valores do eixo y
    'legend.fontsize': 14    # legenda
})

# Adiciona KDE
todos_ratios.plot(kind='kde', color='black', label='KDE')

plt.xlim(0, 1.5)
plt.xticks(limites)
plt.xlabel("Compression Ratio")
plt.ylabel("Density")
plt.title("LegalSim-PT")

# Segundo eixo y: relative frequency
ax2 = plt.gca().twinx()
# Relative frequency: counts normalizados pelo total
rel_freq = counts * np.diff(bins)  # counts * largura da bin
ax2.set_ylim(0, rel_freq.max() * 1.1)  # um pouco acima do máximo para não cortar
ax2.set_ylabel("Relative Frequency")

ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.grid(False)  # remove qualquer grid vertical

nome_figura = os.path.join(saida, "compression_ratio_2_combined_all_files.png")
plt.savefig(nome_figura, dpi=300, bbox_inches='tight')
plt.close()
print(f"Gráfico combinado de todos os arquivos salvo em {nome_figura}")
