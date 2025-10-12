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
#gerar_parquet("/home/arthur/nlp/repo/simplification/legal-doc-simplification-data/data/museum")

import pandas as pd
from pathlib import Path

def process_files(folder_path):
    """
    Lê dois arquivos de texto em paralelo e cria um DataFrame com duas colunas.
    
    Args:
        folder_path: Caminho da pasta contendo os arquivos
    """
    # Define os caminhos dos arquivos
    folder = Path(folder_path)
    complex_file = folder / "test.complex_filtered.txt"
    simple_file = folder / "test.simple_filtered.txt"
    
    # Verifica se os arquivos existem
    if not complex_file.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {complex_file}")
    if not simple_file.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {simple_file}")
    
    # Lê as linhas de ambos os arquivos
    with open(complex_file, 'r', encoding='utf-8') as f_complex:
        complex_lines = f_complex.readlines()
    
    with open(simple_file, 'r', encoding='utf-8') as f_simple:
        simple_lines = f_simple.readlines()
    
    # Verifica se ambos os arquivos têm a mesma quantidade de linhas
    if len(complex_lines) != len(simple_lines):
        raise ValueError(
            f"Os arquivos têm quantidades diferentes de linhas: "
            f"{complex_file.name} tem {len(complex_lines)} linhas, "
            f"{simple_file.name} tem {len(simple_lines)} linhas"
        )
    
    print(f"✓ Ambos os arquivos têm {len(complex_lines)} linhas")
    
    # Remove quebras de linha e cria o DataFrame
    data = {
        'original_text': [line.strip() for line in complex_lines],
        'paraphrase': [line.strip() for line in simple_lines]
    }
    
    df = pd.DataFrame(data)
    
    # Salva o DataFrame como arquivo Parquet
    output_file = Path.cwd() / "data/gov_lang_br/gov_lang_br.parquet"
    df.to_parquet(output_file, index=False)
    
    print(f"✓ DataFrame criado com {len(df)} linhas")
    print(f"✓ Arquivo salvo em: {output_file}")
    
    return df


# Exemplo de uso
if __name__ == "__main__":
    # Substitua pelo caminho da pasta contendo os arquivos
    folder_path = input("Digite o caminho da pasta contendo os arquivos: ").strip()
    
    try:
        df = process_files(folder_path)
        print("\nPrimeiras linhas do DataFrame:")
        print(df.head())
    except Exception as e:
        print(f"Erro: {e}")