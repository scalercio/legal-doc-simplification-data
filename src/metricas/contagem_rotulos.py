import os

# Caminho da pasta com os arquivos
pasta = "/home/camila/legal-doc-simplification-data/src/metricas/contagem/rotulos_txt/"

# Dicionário total
contagem_total = {"D": 0, "M": 0, "S": 0, "E": 0, "I": 0}

# Função auxiliar para calcular porcentagens
def calcular_porcentagens(contagem):
    total = sum(contagem.values())
    if total == 0:
        return {letra: 0 for letra in contagem}
    return {letra: (qtd / total) * 100 for letra, qtd in contagem.items()}

# Percorre todos os arquivos .txt
for nome_arquivo in os.listdir(pasta):
    if nome_arquivo.endswith(".txt"):
        caminho = os.path.join(pasta, nome_arquivo)
        print(f"\nArquivo: {nome_arquivo}")

        contagem_local = {"D": 0, "M": 0, "S": 0, "E": 0, "I": 0}

        with open(caminho, "r", encoding="utf-8") as f:
            for linha in f:
                letra = linha.strip().upper()
                if letra in contagem_local:
                    contagem_local[letra] += 1
                    contagem_total[letra] += 1

        # Calcula porcentagens do arquivo atual
        porcentagens_local = calcular_porcentagens(contagem_local)

        # Exibe resultados do arquivo
        total_local = sum(contagem_local.values())
        print(f"  Total de linhas: {total_local}")
        for letra in contagem_local:
            print(f"  {letra}: {contagem_local[letra]} ({porcentagens_local[letra]:.2f}%)")

# Exibe o total geral da pasta
print("\n=== Total geral da pasta ===")
total_geral = sum(contagem_total.values())
porcentagens_total = calcular_porcentagens(contagem_total)

print(f"Total de linhas (todos os arquivos): {total_geral}")
for letra in contagem_total:
    print(f"{letra}: {contagem_total[letra]} ({porcentagens_total[letra]:.2f}%)")
