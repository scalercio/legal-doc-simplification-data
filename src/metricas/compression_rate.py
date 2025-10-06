import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from src.metricas.utils import *

pasta = "/home/camila/legal-doc-simplification-data/datasets"
arquivos = glob.glob(os.path.join(pasta, "*.parquet.final"))

if not arquivos:
    raise FileNotFoundError(f"Nenhum arquivo .parquet.final encontrado em {pasta}")

saida = "contagem/compression_rate"
os.makedirs(saida, exist_ok=True)

for arquivo in arquivos:
    nome = os.path.basename(arquivo)
    print(f"Lendo {nome}...")

    df = pd.read_parquet(arquivo)

    if not all(col in df.columns for col in ["original_text", "paraphrase"]):
        print(f" Pulando {nome} — colunas não encontradas.")
        continue

    ratios = df.apply(lambda row: contar_caracteres(row["paraphrase"]) /
                                  contar_caracteres(row["original_text"]), axis=1)


    plt.figure(figsize=(10, 6))
    plt.hist(ratios, bins=30, color='skyblue', edgecolor='black', alpha=0.6, density=True)
    ratios.plot(kind='kde', color='black')  # KDE em preto


    plt.xlim(0, 3)
    plt.xticks([x * 0.5 for x in range(7)])

    plt.xlabel("Compression Ratio")
    plt.ylabel("Density")
    plt.title(f"Distribuição de compression ratios — {nome}")
    plt.grid(True)

    nome_figura = os.path.join(saida, f"compression_ratio_combined_{nome}.png")
    plt.savefig(nome_figura, dpi=300)
    plt.close()
    print(f"Gráfico combinado salvo em {nome_figura}")
