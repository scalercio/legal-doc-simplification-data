import pandas as pd
import glob
import os


pasta = "/home/camila/legal-doc-simplification-data/datasets"
arquivos = glob.glob(os.path.join(pasta, "*.parquet.final"))

if not arquivos:
    raise FileNotFoundError(f"Nenhum arquivo .parquet.final encontrado em {pasta}")

print(f"{len(arquivos)} arquivos encontrados:")
for a in arquivos:
    print("  -", os.path.basename(a))


for arquivo in arquivos:
    nome_dataset = os.path.basename(arquivo)
    print(f"\nLendo: {nome_dataset}")
    df = pd.read_parquet(arquivo)

    # Verifica se tem as colunas esperadas
    if not all(col in df.columns for col in ["original_text", "paraphrase"]):
        print(f"Pulando {nome_dataset} — colunas não encontradas.")
        continue

    # Amostra exatamente 50 linhas
    if len(df) >= 20:
        amostra = df.sample(n=20, random_state=42)
    else:
        amostra = df.sample(n=20, replace=True, random_state=42)


    amostra["rotulo"] = ""


    for idx, row in amostra.iterrows():
        print(f"--- Amostra {idx} ---")
        print("Original:")
        print(row["original_text"])
        print("Simplificado:")
        print(row["paraphrase"])
        print("\n")


    saida = f"contagem/rotulos/amostras_50_{nome_dataset}.csv"
    amostra.to_csv(saida, index=False)
    print(f"CSV salvo para {nome_dataset}: {saida}")
