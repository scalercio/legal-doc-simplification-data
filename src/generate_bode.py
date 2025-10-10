import torch
import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import evaluate
import numpy as np
from sentence_transformers import SentenceTransformer, util

# ======================
# 1. Carregar modelo Bode
# ======================
llm_model = 'recogna-nlp/bode-7b-alpaca-pt-br'
hf_auth = 'HF_KEY'  # Substitua pela sua chave do Hugging Face

print("🔧 Carregando modelo Bode...")

config = PeftConfig.from_pretrained(llm_model)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    trust_remote_code=True,
    return_dict=True,
    load_in_8bit=True,
    device_map='auto',
    token=hf_auth
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, token=hf_auth)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model = PeftModel.from_pretrained(model, llm_model)  # Adicione offload_folder="./offload_dir" se necessário
model.eval()

# ======================
# 2. Carregar test set
# ======================
print("📂 Carregando dataset de teste...")
test_df = pd.read_parquet("challenge_hard.parquet")

# ======================
# 3. Função de prompt (formato Alpaca)
# ======================
def generate_prompt(instruction, input_text=None):
    if input_text:
        return f"""Abaixo está uma instrução que descreve uma tarefa, juntamente com uma entrada que fornece mais contexto. Escreva uma resposta que complete adequadamente o pedido.

### Instrução:
{instruction}

### Entrada:
{input_text}

### Resposta:"""
    else:
        return f"""Abaixo está uma instrução que descreve uma tarefa. Escreva uma resposta que complete adequadamente o pedido.

### Instrução:
{instruction}

### Resposta:"""

def make_prompt(original_text):
    instruction = "Simplifique o texto a seguir, mas mantenha o sentido original. Retorne apenas o texto simplificado."
    return generate_prompt(instruction, original_text)

print("🧩 Gerando prompts formatados...")
test_df["prompt"] = test_df["original_text"].apply(make_prompt)

MAX_NEW_TOKENS = 1024
# ======================
# 4. Configuração de geração
# ======================
generation_config = GenerationConfig(
    temperature=0.2,
    top_p=0.75,
    num_beams=2,
    do_sample=True,
    max_new_tokens = MAX_NEW_TOKENS,  # Ajuste conforme necessário
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.1, # Evitar repetições
)

# ======================
# 5. Geração batched
# ======================
BATCH_SIZE = 2  # Reduza se tiver problemas de VRAM

predictions = []
references = test_df["paraphrase"].tolist()
prompts = test_df["prompt"].tolist()

print(f"\n🚀 Gerando saídas em batches de {BATCH_SIZE}...\n")

for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Generating"):
    batch_prompts = prompts[i : i + BATCH_SIZE]
    
    # Tokeniza lote
    batch_inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=3072,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
            generation_config=generation_config,
            #return_dict_in_generate=True,
            #output_scores=True,
        )

    # Decodifica apenas os tokens gerados (sem o prompt)
    gen_tokens = outputs[:, batch_inputs["input_ids"].shape[1]:]
    decoded_batch = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    
    # Extrai apenas a resposta (após "### Resposta:")
    for output in decoded_batch:
        if "### Resposta:" in output:
            response = output.split("### Resposta:")[1].strip()
        else:
            response = output.strip()
        predictions.append(response)

test_df["bode_output"] = predictions
print("\n✅ Geração concluída!\n")

# ======================
# 6. Avaliação
# ======================
print("\n📏 Calculando métricas...")

# --- SARI ---
print("  - Calculando D-SARI...")
sari_metric = evaluate.load("sari")

try:
    sari_result = sari_metric.compute(
        sources=test_df["original_text"],
        predictions=test_df["bode_output"],
        references=[[r] for r in test_df["paraphrase"]]
    )
    d_sari = sari_result["sari"]
except Exception as e:
    print(f"⚠️ Erro no cálculo em lote ({e}), calculando item a item...")
    sari_scores = []
    for src, pred, ref in tqdm(zip(test_df["original_text"], test_df["bode_output"], test_df["paraphrase"]),
                               total=len(test_df), desc="SARI"):
        try:
            score = sari_metric.compute(sources=[src], predictions=[pred], references=[[ref]])
            sari_scores.append(score["sari"])
        except:
            sari_scores.append(0.0)
    d_sari = np.mean(sari_scores)

print(f"  ✓ D-SARI: {d_sari:.2f}")

# --- Similaridade semântica ---
print("  - Calculando similaridade semântica...")
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBED_BATCH_SIZE = 32

src_embeddings = embedder.encode(
    test_df["original_text"].tolist(),
    batch_size=EMBED_BATCH_SIZE,
    show_progress_bar=True,
    convert_to_tensor=True
)
pred_embeddings = embedder.encode(
    test_df["bode_output"].tolist(),
    batch_size=EMBED_BATCH_SIZE,
    show_progress_bar=True,
    convert_to_tensor=True
)

sims = util.pytorch_cos_sim(src_embeddings, pred_embeddings)
semantic_similarity = torch.diagonal(sims).mean().item()
print(f"  ✓ Similaridade Semântica: {semantic_similarity:.4f}")

# ======================
# 7. Salvar resultados
# ======================
metrics = {
    "d_sari": d_sari,
    "semantic_similarity": semantic_similarity,
}

with open("test_metrics_bode_hard_v2.json", "w") as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)

test_df.to_csv("challenge_hard_bode_results_v2.csv", index=False)

print("\n✅ Métricas finais:")
print(json.dumps(metrics, indent=4, ensure_ascii=False))
print("\n📁 Resultados salvos em 'challenge_hard_bode_results_v2.csv'")