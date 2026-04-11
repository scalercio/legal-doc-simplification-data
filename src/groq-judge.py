import os


import pandas as pd
from groq import Groq
import json
import re
from typing import List, Dict




# =========================
# Configurações
# =========================
GROQ_API_KEY = "sua-chave-aqui"  # configure sua chave de API Groq
MODEL = "llama-3.3-70b-versatile"
#MODEL = "openai/gpt-oss-120b"    # ou "mixtral-8x7b-32768"
TEMPERATURE = 0.0
MAX_ITEMS_PER_CALL = 20      # aumente/diminua conforme o tamanho médio dos textos


# =========================
# Instruções (uma vez só)
# =========================
SYSTEM_INSTRUCTIONS = """
Você é um avaliador linguístico especializado em simplificação de textos e preservação semântica.
Sua tarefa é avaliar o quão bem cada uma de três versões simplificadas atende a cinco critérios linguísticos.
Use SEMPRE a escala 1–5 (1=ruim, 5=excelente) e gere SAÍDA ESTRITAMENTE EM JSON.

Critérios:
P1. Simplicidade com preservação do significado e fluidez.
P2. Simplificação lexical (troca por termos mais simples).
P3. Simplificação estrutural (redução de complexidade sintática).
P4. Preservação do significado (sem omissões essenciais ou adições irrelevantes).
P5. Correção gramatical e fluência.

Para CADA item fornecido (com id), avalie as versões 1, 2 e 3 e produza o seguinte JSON por item:
{
  "id": "ID_DO_ITEM",
  "Versao_1": {
    "P1": {"nota": int, "justificativa": "string"},
    "P2": {"nota": int, "justificativa": "string"},
    "P3": {"nota": int, "justificativa": "string"},
    "P4": {"nota": int, "justificativa": "string"},
    "P5": {"nota": int, "justificativa": "string"},
    "Comentário_geral": "string"
  },
  "Versao_2": { ... mesmo formato ... },
  "Versao_3": { ... mesmo formato ... },
  "Ranking_geral": {
    "ordem_melhor_para_pior": ["Versao_X", "Versao_Y", "Versao_Z"],
    "justificativa": "string breve"
  }
}

A saída FINAL deve ser um ARRAY JSON com um objeto por item, na MESMA ORDEM de entrada.
Não inclua explicações fora do JSON. Não use markdown nem blocos de código.
Se algum texto for muito curto para avaliar, ainda assim dê notas e explique a limitação na justificativa.
Seja determinístico e consistente entre itens. Evite aleatoriedade.
"""

# =========================
# Montagem do payload do usuário
# =========================
def build_user_payload(items: List[Dict[str, str]]) -> str:
    """
    items: lista de dicts com chaves: id, original, v1, v2, v3
    Retorna um único texto compactando todos os itens.
    """
    parts = ["A seguir estão N itens. Para cada item, avalie as três versões conforme as instruções do sistema e retorne um ÚNICO array JSON com um objeto por item.\n"]
    for it in items:
        block = f"""### ITEM
ID: {it['id']}

TEXTO ORIGINAL:
{it['original']}

VERSÃO 1:
{it['v1']}

VERSÃO 2:
{it['v2']}

VERSÃO 3:
{it['v3']}
"""
        parts.append(block)
    parts.append("\nLembre-se: retorne APENAS um ARRAY JSON com os resultados, na mesma ordem dos itens acima.")
    return "\n".join(parts)

# =========================
# Chamada à API Groq
# =========================
def call_groq(items: List[Dict[str, str]]) -> List[Dict]:
    client = Groq(api_key=GROQ_API_KEY)
    user_text = build_user_payload(items)

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": user_text},
        ],
    )

    content = resp.choices[0].message.content.strip()
    

    # Tenta extrair JSON puro (às vezes modelos retornam texto extra)
    json_str = extract_json(content)
    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            return data
        else:
            # Se por algum motivo vier um objeto com chave "results" ou algo assim
            return data.get("results", [])
    except json.JSONDecodeError:
        raise ValueError(f"Falha ao decodificar JSON.\n---\n{content}\n---")

def extract_json(text: str) -> str:
    """
    Extrai o primeiro bloco JSON array válido no texto.
    Procura por algo iniciando com [ e terminando no colchete correspondente.
    """
    # Remove código, se houver
    text = text.strip().strip("`").strip()
    text = text.removeprefix("json").strip()
    text = text.removesuffix("```").strip()
    # Heurística: pegar o primeiro [ ... ] que pareça JSON
    match = re.search(r"$begin:math:display$\\s*{.*}\\s*$end:math:display$", text, flags=re.DOTALL)
    
    if match:
        return match.group(0)
    # fallback: se o modelo já retornou JSON puro
    return text

# Exibição resumida
def exibe_resumido(resultados: List[Dict]):
    for item in resultados:
        print(f"\n=== {item.get('id')} ===")
        for k in ("Versao_1", "Versao_2", "Versao_3"):
            v = item.get(k, {})
            if not v:
                continue
            p1 = v.get("P1", {})
            p2 = v.get("P2", {})
            p3 = v.get("P3", {})
            p4 = v.get("P4", {})
            p5 = v.get("P5", {})
            print(f"{k}: P1={p1.get('nota')} P2={p2.get('nota')} P3={p3.get('nota')} P4={p4.get('nota')} P5={p5.get('nota')}")
        rg = item.get("Ranking_geral", {})
        print("Ranking:", rg.get("ordem_melhor_para_pior"))
        print("Justificativa:", rg.get("justificativa"))

# =========================
# Execução em lotes (chunking)
# =========================
def evaluate_in_batches(all_items: List[Dict[str, str]], batch_size: int = MAX_ITEMS_PER_CALL) -> List[Dict]:
    results = []
   
    for i in range(0, len(all_items), batch_size):
    
        batch = all_items[i:i+batch_size]
        batch_results = call_groq(batch)
        #print(results)
        results.extend(batch_results)
    print("Avaliação concluída.")

    return results

# =========================
# Exemplo de uso
# =========================
if __name__ == "__main__":
    # 
    pq_file = pd.read_parquet('llm_as_a_judge.parquet')
    dados = []
    for index, row in pq_file.iterrows():
        print(str(index)+"\n")
        print(row)
        #if index not in range(1,22):
        #    continue
        dados.append({
            "id": index,
            "original": row['original_text'],
            "v1": row['paraphrase'],
            "v2": row['qwen2.5'],
            "v3": row['bode']
        })
        #break  # Apenas o primeiro registro para teste
    
    resultados = evaluate_in_batches(dados, batch_size=MAX_ITEMS_PER_CALL)

    # Salva e imprime
    with open("avaliacao_simplificacao-llama-r3.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    exibe_resumido(resultados)