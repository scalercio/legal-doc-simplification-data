import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from src.metricas.utils import *

# Pasta dos arquivos
pasta = "/home/camila/legal-doc-simplification-data/datasets"
arquivos = glob.glob(os.path.join(pasta, "*.parquet.final"))

if not arquivos:
    raise FileNotFoundError(f"Nenhum arquivo .parquet.final encontrado em {pasta}")

# Pasta de saída
saida = "contagem/compression_rate"
os.makedirs(saida, exist_ok=True)

# Define faixas fixas de 0.1 até 1.5
limites = np.arange(0, 1.5 + 0.1, 0.1)
print(f"Número de faixas: {len(limites)-1}")

# Inicializa figura compilada
plt.figure(figsize=(12, 8))

# Cores diferentes para cada arquivo
colors = plt.cm.tab20.colors  # até 20 cores diferentes
color_idx = 0

for arquivo in arquivos:
    nome = os.path.basename(arquivo)
    df = pd.read_parquet(arquivo)

    if not all(col in df.columns for col in ["original_text", "paraphrase"]):
        print(f" Pulando {nome} — colunas não encontradas.")
        continue

    # Calcula compression ratio
    ratios = df.apply(
        lambda row: contar_caracteres(row["paraphrase"]) / contar_caracteres(row["original_text"])
        if contar_caracteres(row["original_text"]) > 0 else 0,
        axis=1
    )

    # Histogramas normalizados (density=True)
    plt.hist(ratios, bins=limites, alpha=0.5, density=True,
             color=colors[color_idx % len(colors)], label=nome)

    color_idx += 1

plt.xlim(0, 1.5)
plt.xticks(limites)
plt.xlabel("Compression Ratio")
plt.ylabel("Density")
plt.title("Distribuição de compression ratios — Todos os arquivos")
plt.grid(True)
plt.legend(fontsize=8, loc='upper right')

# Salvar figura compilada
nome_figura = os.path.join(saida, "compression_ratio_all_files.png")
plt.savefig(nome_figura, dpi=300, bbox_inches='tight')
plt.close()
print(f"Gráfico compilado salvo em {nome_figura}")
