import pandas as pd
import glob
import os

from src.metricas.utils import contar_caracteres
from src.metricas.utils import contar_palavras, contar_sentencas

# Pasta com os arquivos
pasta = "/home/camila/legal-doc-simplification-data/datasets"
arquivos = glob.glob(os.path.join(pasta, "*.parquet.final"))

if not arquivos:
    raise FileNotFoundError(f"Nenhum arquivo .parquet.final encontrado em {pasta}")

print(f"{len(arquivos)} arquivos encontrados:")
for a in arquivos:
    print("  -", os.path.basename(a))

# Lista para armazenar resultados
resultados = []

# Processa cada arquivo
for arquivo in arquivos:
    print(f"\nLendo: {os.path.basename(arquivo)}")
    df = pd.read_parquet(arquivo)

    # Verifica se as colunas necessárias existem
    if not all(col in df.columns for col in ["original_text", "paraphrase"]):
        print(f" ⚠️ Pulando {os.path.basename(arquivo)} — colunas não encontradas.")
        continue

    # Calcula métricas para cada documento
    df["sentencas_origem"] = df["original_text"].apply(contar_sentencas)
    df["sentencas_destino"] = df["paraphrase"].apply(contar_sentencas)
    df["palavras_origem"] = df["original_text"].apply(contar_palavras)
    df["palavras_destino"] = df["paraphrase"].apply(contar_palavras)
    df["caracteres_origem"] = df["original_text"].apply(contar_caracteres)
    df["caracteres_destino"] = df["paraphrase"].apply(contar_caracteres)

    # Calcula médias e desvios padrão
    medias = {
        "dataset": os.path.basename(arquivo),
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
        "numero_documentos": len(df)
    }

    resultados.append(medias)

    # Mostra resumo do dataset
    print(f"\nResumo de {os.path.basename(arquivo)}:")
    for k, v in medias.items():
        if k != "dataset":
            print(f"  {k}: {v:.2f}")


final = pd.DataFrame(resultados)
final = final.sort_values(by="dataset").reset_index(drop=True)

print("\n Resultado combinado de todos os datasets:")
print(final.round(2))


saida = "contagem/tamanhos_medio_datasets/todos_datasets.csv"
final.to_csv(saida, index=False)
print(f"\n Arquivo salvo: {saida}")
