import pandas as pd
import numpy as np
import glob
from pathlib import Path

# ==============================================================
# Funções auxiliares (placeholders)
# ==============================================================
def flesch_reading_ease_pt(text: str) -> float:
    """Placeholder: retorna índice Flesch ajustado para português."""
    return np.random.uniform(0, 100)

def semantic_similarity_score(a: str, b: str) -> float:
    """Placeholder: retorna similaridade semântica entre original e simplificado."""
    return np.random.uniform(0, 1)

# ==============================================================
# 1. Carregar e combinar todos os arquivos Parquet
# ==============================================================
input_folder = "/home/arthurscalercio/repo/legal-doc-simplification-data/data/legal_pt"
output_folder = "splits_output"
Path(output_folder).mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 777

all_files = glob.glob(f"{input_folder}/*.parquet.final")
dfs = []

for file in all_files:
    df_temp = pd.read_parquet(file)
    df_temp["source_file"] = Path(file).stem  # nome do arquivo de origem (sem extensão)
    dfs.append(df_temp)

df = pd.concat(dfs, ignore_index=True)
print(f"Dataset combinado: {len(df):,} linhas de {len(all_files)} arquivos")

# ==============================================================
# 2. Calcular métricas
# ==============================================================
#print("Calculando métricas de similaridade e legibilidade...")
#df["similarity"] = df.apply(lambda r: semantic_similarity_score(r["original_text"], r["simplified_text"]), axis=1)
#df["flesch_original"] = df["original_text"].apply(flesch_reading_ease_pt)
#df["flesch_simplified"] = df["simplified_text"].apply(flesch_reading_ease_pt)
#df["flesch_gain"] = df["flesch_simplified"] - df["flesch_original"]

# ==============================================================
# 3. Criar splits estratificados por arquivo
# ==============================================================
print("Criando splits aleatórios com representatividade por arquivo...")

n_val = 10_000
n_test = 10_000

val_parts = []
test_parts = []
train_parts = []

# Amostragem proporcional por arquivo de origem
for src, group in df.groupby("source_file"):
    frac_val = n_val / len(df)
    frac_test = n_test / len(df)

    n_val_src = int(len(group) * frac_val)
    n_test_src = int(len(group) * frac_test)

    group = group.sample(frac=1, random_state=42)  # embaralhar localmente

    val_parts.append(group.iloc[:n_val_src])
    test_parts.append(group.iloc[n_val_src:n_val_src + n_test_src])
    train_parts.append(group.iloc[n_val_src + n_test_src:])

val_random = pd.concat(val_parts, ignore_index=True)
test_random = pd.concat(test_parts, ignore_index=True)
train_random = pd.concat(train_parts, ignore_index=True)

# Garantir exatamente 10k amostras nos splits de validação e teste
#val_random = val_random.sample(n=n_val, random_state=42)
#test_random = test_random.sample(n=n_test, random_state=42)

# ==============================================================
# 4. Criar val_challenges (seleção a partir do dataset completo)
# ==============================================================
print("Selecionando val_challenges...")

# Criar challenge sets com base em percentis
sim_p90 = df['sim'].quantile(0.9)
gain_p90 = df['flesch_diff'].quantile(0.9)
sim_p10 = df['sim'].quantile(0.1)
gain_p10 = df['flesch_diff'].quantile(0.1)

# challenge "fácil/bom": alta sim e alto ganho
challenge_good = df[(df['sim'] >= sim_p90) & (df['flesch_diff'] >= gain_p90)].sample(
    n=min(200, len(df[(df['sim'] >= sim_p90) & (df['flesch_diff'] >= gain_p90)])),
    random_state=RANDOM_SEED
)

# challenge "difícil": baixa sim OU ganho negativo
challenge_hard = df[(df['sim'] <= sim_p10) & (df['flesch_diff'] <= gain_p10)].sample(
    n=min(200, len(df[(df['sim'] <= sim_p10) & (df['flesch_diff'] <= gain_p10)])),
    random_state=RANDOM_SEED
)

#df["len_original"] = df["original_text"].str.split().apply(len)
#
#val_challenges = df.query(
#    "(similarity < 0.7) or (flesch_gain > 15) or (len_original > 300)"
#).sample(n=10_000, random_state=123, replace=False)

# ==============================================================
# 5. Salvar resultados
# ==============================================================
print("Salvando splits...")

train_random.to_parquet(f"{output_folder}/train_random.parquet", index=False)
val_random.to_parquet(f"{output_folder}/val_random.parquet", index=False)
test_random.to_parquet(f"{output_folder}/test_random.parquet", index=False)
challenge_good.to_parquet(f"{output_folder}/challenge_good.parquet", index=False)
challenge_hard.to_parquet(f"{output_folder}/challenge_hard.parquet", index=False)
challenge_good.to_csv("val_challenge_good.csv", index=False)
challenge_hard.to_csv("val_challenge_hard.csv", index=False)

# ==============================================================
# 6. Relatório resumido
# ==============================================================
print("\n✅ Splits criados e salvos com sucesso:")
for name, subset in {
    "train_random": train_random,
    "val_random": val_random,
    "test_random": test_random,
    "challenge_good": challenge_good,
    "challenge_hard": challenge_hard,
}.items():
    print(f"{name:<15}: {len(subset):>8,} registros — {subset['source_file'].nunique()} arquivos de origem")

