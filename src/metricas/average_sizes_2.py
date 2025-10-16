import pandas as pd
import glob
import os
import gc
import numpy as np

# Pasta com os arquivos
pasta = "/home/camila/legal-doc-simplification-data/dataset_gov"
arquivos = glob.glob(os.path.join(pasta, "*.parquet"))

if not arquivos:
    raise FileNotFoundError(f"Nenhum arquivo .parquet.final encontrado em {pasta}")

print(f"{len(arquivos)} arquivos encontrados:")
for a in arquivos:
    print("  -", os.path.basename(a))

os.makedirs("contagem/soma_metrica_datasets", exist_ok=True)

# --- PROCESSAMENTO POR DATASET ---
for arquivo in arquivos:
    nome = os.path.basename(arquivo)
    print(f"\nLendo: {nome}")

    # Lê apenas as colunas necessárias
    df = pd.read_parquet(
        arquivo,
        columns=[
            "original_text", "paraphrase"
        ]
    )

    if not all(col in df.columns for col in [
        "original_text", "paraphrase",
    ]):
        print(f" Pulando {nome} — colunas não encontradas.")
        continue

    from src.metricas.utils import contar_caracteres, contar_palavras, contar_sentencas

    # Calcula métricas básicas
    df["sentencas_origem"] = df["original_text"].map(contar_sentencas)
    df["sentencas_destino"] = df["paraphrase"].map(contar_sentencas)
    df["palavras_origem"] = df["original_text"].map(contar_palavras)
    df["palavras_destino"] = df["paraphrase"].map(contar_palavras)
    df["caracteres_origem"] = df["original_text"].map(contar_caracteres)
    df["caracteres_destino"] = df["paraphrase"].map(contar_caracteres)

    # Soma de cada métrica
    soma = {
        "dataset": nome,
        "total_sentencas_origem": df["sentencas_origem"].sum(),
        "total_sentencas_destino": df["sentencas_destino"].sum(),
        "total_palavras_origem": df["palavras_origem"].sum(),
        "total_palavras_destino": df["palavras_destino"].sum(),
        "total_caracteres_origem": df["caracteres_origem"].sum(),
        "total_caracteres_destino": df["caracteres_destino"].sum(),
        "numero_documentos": len(df),
    }

    # Salva CSV individual
    saida = f"contagem/soma_metrica_datasets/{nome}.csv"
    pd.DataFrame([soma]).to_csv(saida, index=False)
    print(f"Arquivo salvo: {saida}")

    # Libera memória
    del df
    gc.collect()
