import re
import pandas as pd
import nltk
from sentence_transformers import SentenceTransformer, util

def conta_silabas(palavra: str) -> int:
    """Conta sílabas de forma aproximada em português (heurística)."""
    return len(re.findall(r'[aeiouáéíóúâêôãõàü]', palavra.lower()))

def flesch_portugues(texto: str) -> float:
    """Calcula o índice de Flesch adaptado para o português (Cunha & Santos, 1985)."""
    # Divide em frases
    frases = re.split(r'[.!?]+', texto)
    frases = [f.strip() for f in frases if f.strip()]
    n_frases = len(frases)

    # Divide em palavras
    palavras = re.findall(r'\w+', texto.lower())
    n_palavras = len(palavras)

    # Conta sílabas aproximadas
    n_silabas = sum(conta_silabas(p) for p in palavras)

    # Evita divisão por zero
    ASL = n_palavras / max(1, n_frases)   # palavras por frase
    ASW = n_silabas / max(1, n_palavras)  # sílabas por palavra

    # Fórmula do Flesch adaptado ao português
    IFP = 248.835 - (1.015 * ASL) - (84.6 * ASW)
    return round(IFP, 2)

def interpretar_flesch(score: float) -> str:
    """Interpreta o índice de Flesch em português."""
    if score >= 75:
        return "Muito fácil (nível fundamental)"
    elif score >= 50:
        return "Médio (nível médio)"
    elif score >= 25:
        return "Difícil (nível superior)"
    else:
        return "Muito difícil (pós-graduação / textos técnicos)"

def avaliar_documentos(docs):
    """Recebe lista de documentos e devolve índice de Flesch + interpretação."""
    resultados = []
    for i, doc in enumerate(docs, start=1):
        score = flesch_portugues(doc)
        interpretacao = interpretar_flesch(score)
        resultados.append((f"Documento {i}", score, interpretacao))
    return resultados


# 🔹 Exemplo de uso
#documentos = [
#    "A leitura é essencial para o desenvolvimento humano. Livros simples ajudam crianças a aprender.",
#    "A fenomenologia transcendental husserliana apresenta uma estrutura complexa de intencionalidade da consciência, exigindo alto nível de abstração filosófica."
#]
#
#resultados = avaliar_documentos(documentos)
#
#for nome, score, interpretacao in resultados:
#    print(f"{nome}: Índice de Flesch = {score} → {interpretacao}")


# Baixar tokenizer do nltk (rodar uma vez)
nltk.download("punkt")
nltk.download('punkt_tab')

def chunk_sentences(text: str, max_sentences: int = 5, overlap: int = 2) -> list[str]:
    """
    Divide o texto em pedaços de até max_sentences sentenças,
    com sobreposição de 'overlap' sentenças entre chunks.
    """
    sentences = nltk.sent_tokenize(text, language="portuguese")
    chunks = []
    start = 0
    
    while start < len(sentences):
        end = min(start + max_sentences, len(sentences))
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)
        if end == len(sentences):
            break
        start += max_sentences - overlap
    
    return chunks


def calcular_similaridade_sliding(df: pd.DataFrame,
                                  max_sentences: int = 5,
                                  overlap: int = 2) -> list[float]:
    """
    Calcula similaridade semântica entre documentos longos
    (colunas 'original_text' e 'paraphrase') usando sliding window.
    
    - max_sentences: nº máximo de sentenças por chunk.
    - overlap: nº de sentenças que se repetem entre janelas.
    
    Retorna uma lista com as similaridades.
    """
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    similaridades = []
    
    for _, row in df.iterrows():
        doc1 = str(row["original_text"])
        doc2 = str(row["paraphrase"])
        
        # Quebrar documentos em chunks com janela deslizante
        chunks1 = chunk_sentences(doc1, max_sentences=max_sentences, overlap=overlap)
        chunks2 = chunk_sentences(doc2, max_sentences=max_sentences, overlap=overlap)
        
        # Gerar embeddings
        emb1 = model.encode(chunks1, convert_to_tensor=True)
        emb2 = model.encode(chunks2, convert_to_tensor=True)
        
        # Agregar com média
        emb1_mean = emb1.mean(dim=0)
        emb2_mean = emb2.mean(dim=0)
        
        # Similaridade
        sim = util.cos_sim(emb1_mean, emb2_mean).item()
        similaridades.append(sim)
    
    return similaridades

def calcular_delta_flesch(df: pd.DataFrame) -> list[float]:
    """
    Calcula a diferença entre o score flesch entre documentos longos
    (colunas 'original_text' e 'paraphrase').
    
    Retorna uma lista com as diferenças.
    """
    
    delta_scores = []
    
    for _, row in df.iterrows():
        doc1 = str(row["original_text"])
        doc2 = str(row["paraphrase"])
        
        # Quebrar documentos em chunks com janela deslizante
        flesch1 = flesch_portugues(doc1)
        flesch2 = flesch_portugues(doc2)
        
        # delta_scores
        delta_scores.append(flesch2-flesch1)
    
    return delta_scores
