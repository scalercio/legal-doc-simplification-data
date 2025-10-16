import pandas as pd

# === 1️⃣ Carregar o CSV original ===
df = pd.read_csv("/home/camila/legal-doc-simplification-data/src/metricas/contagem/soma_metrica_datasets/individuais.csv")

# === 2️⃣ Calcular as médias por dataset ===
df["flesch_original_medio"] = df["total_flesch_original"] / df["numero_documentos"]
df["flesch_paraphrase_medio"] = df["total_flesch_paraphrase"] / df["numero_documentos"]
df["flesch_diff_medio"] = df["flesch_paraphrase_medio"] - df["flesch_original_medio"]

# === 3️⃣ Criar dataframe de resumo por dataset (sem std) ===
resumo = df[["dataset", "flesch_original_medio", "flesch_paraphrase_medio", "flesch_diff_medio"]].copy()

# === 4️⃣ Calcular totais gerais ===
flesch_original_total = df["total_flesch_original"].sum() / df["numero_documentos"].sum()
flesch_paraphrase_total = df["total_flesch_paraphrase"].sum() / df["numero_documentos"].sum()
flesch_diff_total = flesch_paraphrase_total - flesch_original_total

# === 5️⃣ Calcular desvios padrão globais (entre datasets) ===
flesch_original_std = df["flesch_original_medio"].std()
flesch_paraphrase_std = df["flesch_paraphrase_medio"].std()
flesch_diff_std = df["flesch_diff_medio"].std()

# Adiciona linha total com médias e desvios
resumo_total = pd.DataFrame({
    "dataset": ["__TOTAL__"],
    "flesch_original_medio": [flesch_original_total],
    "flesch_paraphrase_medio": [flesch_paraphrase_total],
    "flesch_diff_medio": [flesch_diff_total],
    "flesch_original_std": [flesch_original_std],
    "flesch_paraphrase_std": [flesch_paraphrase_std],
    "flesch_diff_std": [flesch_diff_std]
})

# === 6️⃣ Junta tudo ===
resumo_final = pd.concat([resumo, resumo_total], ignore_index=True)

# === 7️⃣ Salvar resultado ===
resumo_final.to_csv("flesch_resumo.csv", index=False)

print("Arquivo 'flesch_resumo.csv' criado com sucesso!")
print(resumo_final)
