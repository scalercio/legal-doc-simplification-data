import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches

pasta = "/home/camila/legal-doc-simplification-data/datasets"
arquivos = glob.glob(os.path.join(pasta, "*.parquet.final"))

if not arquivos:
    raise FileNotFoundError(f"Nenhum arquivo .parquet.final encontrado em {pasta}")

saida = "contagem/oficiais"
os.makedirs(saida, exist_ok=True)

# Configurações de fonte e estilo (antes de plotar)
plt.rcParams.update({
    'font.size': 18,         # tamanho base
    'axes.titlesize': 24,    # título do gráfico
    'axes.labelsize': 20,    # rótulos dos eixos
    'xtick.labelsize': 16,   # valores do eixo x
    'ytick.labelsize': 16,   # valores do eixo y
    'legend.fontsize': 16,   # legenda
    'figure.titlesize': 24,  # título da figura
})

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
        lambda row: (row["flesch_paraphrase"]),
        axis=1
    )
    print(ratios)
    print(max(ratios))
    print(min(ratios))
    print(ratios.mean())

    todos_ratios.extend(ratios.tolist())  # adiciona ao dataset único

# Converte para Series do pandas
todos_ratios = pd.Series(todos_ratios)
print(f"Total de ratios no dataset combinado: {len(todos_ratios)}")

# Limitar o gráfico a 1.5 e criar faixas de 0.1
limites = np.arange(-10, 105 + 5, 5)
print(f"Número de faixas: {len(limites)-1}")

plt.figure(figsize=(13, 8))

# Cria o histograma de densidade sem label
counts, bins, patches = plt.hist(
    todos_ratios, bins=limites, color='skyblue', edgecolor='black',
    alpha=0.6, density=True  # <- removido label
)

# Adiciona KDE com label

todos_ratios.plot(kind='kde', color='black', label='KDE')

plt.xlim(0, 50)
plt.xticks(limites)
plt.xlabel("Flesch Index" )
plt.ylabel("Density")
plt.title("LegalSim-PT Simplified Documents")

# Segundo eixo y: relative frequency
ax2 = plt.gca().twinx()
rel_freq = counts * np.diff(bins)
ax2.set_ylim(0, rel_freq.max() * 1.1)
ax2.set_ylabel("Relative Frequency")

ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.grid(False)

# Legenda apenas para KDE
nome_figura = os.path.join(saida, "simple_flesch.png")
plt.savefig(nome_figura, dpi=300, bbox_inches='tight')
plt.close()
print(f"Gráfico combinado de todos os arquivos salvo em {nome_figura}")