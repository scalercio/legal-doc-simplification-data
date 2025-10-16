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

saida = "contagem/flesch"
os.makedirs(saida, exist_ok=True)

ganhos_flesch = []
flesch_simples = []

for arquivo in arquivos:
    nome = os.path.basename(arquivo)
    print(f"Lendo {nome}...")

    df = pd.read_parquet(arquivo)

    # precisa ter colunas de Flesch
    if not all(col in df.columns for col in ["flesch_original", "flesch_paraphrase"]):
        print(f" Pulando {nome} — colunas Flesch não encontradas.")
        continue

    # calcula ganho de Flesch
    ganhos = df["flesch_paraphrase"] - df["flesch_original"]
    ganhos_flesch.extend(ganhos.tolist())

    # Flesch dos textos simples (paráfrases)
    flesch_simples.extend(df["flesch_paraphrase"].tolist())

# cria DataFrames
ganhos_flesch = pd.Series(ganhos_flesch).dropna()
flesch_simples = pd.Series(flesch_simples).dropna()

print(f"Total de amostras: ganhos={len(ganhos_flesch)}, simples={len(flesch_simples)}")

# Criação da figura
fig, ax1 = plt.subplots(figsize=(12, 7))

# Histograma e KDE do ganho de Flesch
ganhos_flesch.plot(kind='kde', color='blue', label='Ganho de Flesch', ax=ax1)
ax1.hist(
    ganhos_flesch, bins=30, color='skyblue', edgecolor='black', alpha=0.4,
    density=True
)

# KDE do Flesch dos textos simples
flesch_simples.plot(kind='kde', color='green', label='Flesch (textos simples)', ax=ax1, linestyle='--')

# Configurações de estilo
ax1.set_xlabel("Índice Flesch")
ax1.set_ylabel("Densidade")
ax1.set_title("Distribuição do Ganho de Flesch e do Índice Flesch (Textos Simples) — LeDocS-PT")

# grade apenas horizontal
ax1.grid(True, axis='y')
ax1.grid(False, axis='x')

# remove eixo X e bordas horizontais
ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
for spine in ['bottom', 'top']:
    ax1.spines[spine].set_visible(False)

ax1.legend()
plt.tight_layout()

# Salva figura
nome_figura = os.path.join(saida, "flesch_gain_and_flesch_simple_distribution.png")
plt.savefig(nome_figura, dpi=300, bbox_inches='tight')
plt.close()

print(f"Gráfico de Flesch salvo em {nome_figura}")
