import os
import pandas as pd

def gerar_parquet(pasta: str, output_file: str = "museum.parquet"):
    dados = {}

    # Itera sobre os arquivos da pasta
    for arquivo in os.listdir(pasta):
        caminho = os.path.join(pasta, arquivo)

        if not os.path.isfile(caminho):
            continue

        if arquivo.endswith(".original"):
            base = arquivo.replace(".original", "")
            with open(caminho, "r", encoding="utf-8") as f:
                texto = f.read()
            if base not in dados:
                dados[base] = {"original_text": None, "paraphrase": None}
            dados[base]["original_text"] = texto

        elif arquivo.endswith(".simple"):
            base = arquivo.replace(".simple", "")
            with open(caminho, "r", encoding="utf-8") as f:
                texto = f.read()
            if base not in dados:
                dados[base] = {"original_text": None, "paraphrase": None}
            dados[base]["paraphrase"] = texto

    # Constrói DataFrame
    rows = []
    for base, valores in dados.items():
        rows.append({
            "filename": base,
            "original_text": valores["original_text"],
            "paraphrase": valores["paraphrase"]
        })

    df = pd.DataFrame(rows)

    # Salva em parquet
    df.to_parquet(os.path.join(pasta, output_file), index=False)

    print(f"Arquivo salvo em: {os.path.join(pasta, output_file)}")

# Exemplo de uso
gerar_parquet("/home/arthur/nlp/repo/simplification/legal-doc-simplification-data/data/museum")
