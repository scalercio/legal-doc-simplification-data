import re

from src.utils import legal_sentence_split


def contar_sentencas(texto):
    return len(legal_sentence_split(texto))

import re

def contar_palavras(texto):
    if not isinstance(texto, str) or not texto.strip():
        return 0
    # Substitui tudo que não seja letra (a-z, A-Z, letras com acento) ou hífen por espaço
    texto_limpo = re.sub(r"[^A-Za-zÀ-ÿ-]+", " ", texto)
    # Divide por espaços e conta palavras não vazias
    palavras = [p for p in texto_limpo.split(" ") if p]
    return len(palavras)


#tirar espaços
def contar_caracteres(texto):
    if not isinstance(texto, str):
        return 0
    # Remove tudo que não for letra ou número
    texto_limpo = re.sub(r'[^A-Za-z]', '', texto)
    return len(texto_limpo)

print(contar_sentencas("VISTOS, relatados e discutidos os recursos de reconsideração interpostos por Derli Antônio Donin, ex-prefeito de Toledo/PR, e pela empresa Castelo Comércio de Alimentos Ltda., contra o Acórdão 1.199/2014-TCU-Plenário, que rejeitou a tomada de contas especial realizada pelo FNDE. ACORDAM os Ministros do Tribunal de Contas da União: 9.1. Conhecer do recurso com base nos arts. 32, inciso I, e 33 da Lei 8.443/1992, mas negar-lhe provimento, mantendo o Acórdão 1.199/2014-TCU-Plenário; 9.2. Informar os recorrentes sobre a decisão completa."))
print(contar_palavras("VISTOS, relatados e discutidos 1234 * 2222 Hello H-A"))