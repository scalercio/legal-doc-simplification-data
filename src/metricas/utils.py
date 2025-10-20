import re

from src.utils import legal_sentence_split


def contar_sentencas(texto):
    return len(legal_sentence_split(texto))

import re


def contar_palavras(texto):
    if not isinstance(texto, str) or not texto.strip():
        return 0
    # Substitui todos os tipos de traços por hífen comum
    texto = texto.replace('–', '-').replace('—', '-').replace('−', '-').replace('‑', '-')
    texto_limpo = re.sub(r"[^A-Za-zÀ-ÿ-]+", " ", texto)
    # Remove hífens isolados
    palavras = [p for p in texto_limpo.split(" ") if p and p != "-"]
    return len(palavras)


def contar_caracteres(texto):
    if not isinstance(texto, str):
        return 0
    # Remove tudo que não for letra (incluindo letras acentuadas)
    texto_limpo = re.sub(r'[^A-Za-zÀ-ÿ]', '', texto)
    return len(texto_limpo)

print(contar_sentencas("VISTOS, relatados e discutidos os recursos de reconsideração interpostos por Derli Antônio Donin, ex-prefeito de Toledo/PR, e pela empresa Castelo Comércio de Alimentos Ltda., contra o Acórdão 1.199/2014-TCU-Plenário, que rejeitou a tomada de contas especial realizada pelo FNDE. ACORDAM os Ministros do Tribunal de Contas da União: 9.1. Conhecer do recurso com base nos arts. 32, inciso I, e 33 da Lei 8.443/1992, mas negar-lhe provimento, mantendo o Acórdão 1.199/2014-TCU-Plenário; 9.2. Informar os recorrentes sobre a decisão completa."))
print(contar_palavras("VISTOS, relatados e discutidos 1234 * 2222 Hello H-A"))

print(contar_caracteres("VISTOS, relatadós e discutidos"))
print(contar_caracteres("á"))