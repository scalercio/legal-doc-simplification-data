import pandas as pd
import glob
import os
import gc
import numpy as np

from src.metricas.utils import contar_caracteres, contar_palavras, contar_sentencas

# Pasta com os arquivos
pasta = "/home/camila/legal-doc-simplification-data/dataset_gov"
arquivos = glob.glob(os.path.join(pasta, "*.parquet"))

if not arquivos:
    raise FileNotFoundError(f"Nenhum arquivo .parquet.final encontrado em {pasta}")

print(f"{len(arquivos)} arquivos encontrados:")
for a in arquivos:
    print("  -", os.path.basename(a))

resultados = []

# Acumuladores globais
acum = {
    "sentencas_origem": [],
    "sentencas_destino": [],
    "palavras_origem": [],
    "palavras_destino": [],
    "caracteres_origem": [],
    "caracteres_destino": [],
}

# --- PROCESSAMENTO INDIVIDUAL ---
for arquivo in arquivos:
    nome = os.path.basename(arquivo)
    print(f"\nLendo: {nome}")

    # Lê apenas as colunas necessárias
    df = pd.read_parquet(arquivo, columns=["original_text", "paraphrase"])

    if not all(col in df.columns for col in ["original_text", "paraphrase"]):
        print(f" Pulando {nome} — colunas não encontradas.")
        continue

    # Calcula métricas básicas (sem cópias desnecessárias)
    df["sentencas_origem"] = df["original_text"].map(contar_sentencas)
    df["sentencas_destino"] = df["paraphrase"].map(contar_sentencas)
    df["palavras_origem"] = df["original_text"].map(contar_palavras)
    df["palavras_destino"] = df["paraphrase"].map(contar_palavras)
    df["caracteres_origem"] = df["original_text"].map(contar_caracteres)
    df["caracteres_destino"] = df["paraphrase"].map(contar_caracteres)

    # Estatísticas individuais
    medias = {
        "dataset": nome,
        "media_sentencas_origem": df["sentencas_origem"].mean(),
        "desvio_sentencas_origem": df["sentencas_origem"].std(),
        "media_sentencas_destino": df["sentencas_destino"].mean(),
        "desvio_sentencas_destino": df["sentencas_destino"].std(),
        "media_palavras_origem": df["palavras_origem"].mean(),
        "desvio_palavras_origem": df["palavras_origem"].std(),
        "media_palavras_destino": df["palavras_destino"].mean(),
        "desvio_palavras_destino": df["palavras_destino"].std(),
        "media_caracteres_origem": df["caracteres_origem"].mean(),
        "desvio_caracteres_origem": df["caracteres_origem"].std(),
        "media_caracteres_destino": df["caracteres_destino"].mean(),
        "desvio_caracteres_destino": df["caracteres_destino"].std(),
        "numero_documentos": len(df),
    }

    resultados.append(medias)

    # --- PRINT INDIVIDUAL ---
    print(f"\nResumo de {nome}:")
    for k, v in medias.items():
        if k != "dataset":
            print(f"  {k}: {v:.2f}")

    # Acumula para o total combinado
    for k in acum:
        acum[k].append(df[k].to_numpy())

    # Libera memória
    del df
    gc.collect()

# --- ESTATÍSTICAS COMBINADAS ---
print("\nCalculando totais combinados...")

todos = {k: np.concatenate(v) for k, v in acum.items()}

totais = {
    "dataset": "TODOS_COMBINADOS",
    "media_sentencas_origem": todos["sentencas_origem"].mean(),
    "desvio_sentencas_origem": todos["sentencas_origem"].std(),
    "media_sentencas_destino": todos["sentencas_destino"].mean(),
    "desvio_sentencas_destino": todos["sentencas_destino"].std(),
    "media_palavras_origem": todos["palavras_origem"].mean(),
    "desvio_palavras_origem": todos["palavras_origem"].std(),
    "media_palavras_destino": todos["palavras_destino"].mean(),
    "desvio_palavras_destino": todos["palavras_destino"].std(),
    "media_caracteres_origem": todos["caracteres_origem"].mean(),
    "desvio_caracteres_origem": todos["caracteres_origem"].std(),
    "media_caracteres_destino": todos["caracteres_destino"].mean(),
    "desvio_caracteres_destino": todos["caracteres_destino"].std(),
    "numero_documentos": len(todos["sentencas_origem"]),
}

# --- PRINT FINAL ---
print("\nResumo total combinado (TODOS_COMBINADOS):")
for k, v in totais.items():
    if k != "dataset":
        print(f"  {k}: {v:.2f}")

resultados.append(totais)

# --- SALVA CSV FINAL ---
resultados_df = pd.DataFrame(resultados).round(2)
print("\n\nRESULTADOS FINAIS (por dataset e total combinado):")
print(resultados_df)

os.makedirs("contagem/tamanho_medio_datasets", exist_ok=True)
saida = "contagem/tamanho_medio_datasets/gov_br.csv"
resultados_df.to_csv(saida, index=False)
print(f"\nArquivo salvo: {saida}")
