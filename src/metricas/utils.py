import re

from src.utils import legal_sentence_split


def contar_sentencas(texto):
    if not isinstance(texto, str) or not texto.strip():
        return 0
    return len(legal_sentence_split(texto))

def contar_palavras(texto):
    if not isinstance(texto, str) or not texto.strip():
        return 0
    return len(re.findall(r'\w+', texto))

def contar_caracteres(texto):
    if not isinstance(texto, str):
        return 0
    return len(texto.strip())

