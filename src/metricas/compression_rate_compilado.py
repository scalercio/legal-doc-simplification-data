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
        lambda row: contar_caracteres(row["paraphrase"]) / contar_caracteres(row["original_text"])
        if contar_caracteres(row["original_text"]) > 0 else 0,
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
plt.hist(todos_ratios, bins=limites, color='skyblue', edgecolor='black', alpha=0.6, density=True)
todos_ratios.plot(kind='kde', color='black')  # KDE em preto

plt.xlim(0, 1.5)
plt.xticks(limites)

plt.xlabel("Compression Ratio")
plt.ylabel("Density")
plt.title("Distribuição de compression ratios — Todos os arquivos combinados")
plt.grid(True)

nome_figura = os.path.join(saida, "compression_ratio_combined_all_files.png")
plt.savefig(nome_figura, dpi=300, bbox_inches='tight')
plt.close()
print(f"Gráfico combinado de todos os arquivos salvo em {nome_figura}")
