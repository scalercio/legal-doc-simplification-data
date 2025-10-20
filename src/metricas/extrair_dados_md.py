import re
import pandas as pd
import os

def extrair_dados_markdown(caminho_md):
    """
    Extrai frases, paráfrases e contagem de operações de um arquivo Markdown.
    Retorna uma lista de dicionários.
    """
    with open(caminho_md, 'r', encoding='utf-8') as f:
        conteudo = f.read()

    blocos = re.split(r'###\s*Frase\s*\d+', conteudo)
    resultados = []

    for bloco in blocos:
        if not bloco.strip():
            continue

        # Extrai texto original
        original_match = re.search(r'\*\*Original:\*\*.*?>\s*(.+?)(?:\n\n|\Z)', bloco, re.S)
        original = original_match.group(1).strip() if original_match else ""

        # Extrai paráfrase
        parafrase_match = re.search(r'\*\*Paráfrase:\*\*.*?>\s*(.+?)(?:\n\n|\Z)', bloco, re.S)
        parafrase = parafrase_match.group(1).strip() if parafrase_match else ""

        # Inicializa contadores
        contagem = {'n_Delete': 0, 'n_Edit': 0, 'n_Insert': 0, 'n_Merge': 0, 'n_Split': 0}

        # Conta operações na tabela
        tabela_match = re.findall(r'\|\s*\*\*(.*?)\*\*\s*\|', bloco)
        for tipo in tabela_match:
            tipo = tipo.capitalize().strip()
            if tipo in ['Delete', 'Edit', 'Insert', 'Merge', 'Split']:
                contagem[f"n_{tipo}"] += 1

        resultados.append({
            'Frase_original': original,
            'Parafrase': parafrase,
            **contagem
        })

    return resultados

def processar_pasta_markdown(pasta_md, arquivo_saida="resultado_total.csv"):
    """
    Processa todos os arquivos Markdown da pasta e gera um único CSV.
    """
    todos_resultados = []

    for nome_arquivo in os.listdir(pasta_md):
        if nome_arquivo.endswith(".md"):
            caminho_completo = os.path.join(pasta_md, nome_arquivo)
            print(f"Processando: {caminho_completo}")
            dados = extrair_dados_markdown(caminho_completo)
            todos_resultados.extend(dados)

    df = pd.DataFrame(todos_resultados)
    df.to_csv(arquivo_saida, index=False, encoding='utf-8-sig')
    print(f"✅ CSV gerado: {arquivo_saida}")

# ============================
# Exemplo de uso
# ============================
pasta_markdown = "/home/camila/legal-doc-simplification-data/markdown"
processar_pasta_markdown(pasta_markdown, "resultado_total.csv")
