import torch
import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm
import evaluate
import numpy as np
from sentence_transformers import SentenceTransformer, util
from utils import flesch_portugues

# ======================
# 1. Carregar modelo Tucano
# ======================
#model_id = "TucanoBR/Tucano-2b4-Instruct"
#
#print("🔧 Carregando modelo Tucano...")
#
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#tokenizer = AutoTokenizer.from_pretrained(model_id)
#model = AutoModelForCausalLM.from_pretrained(
#    model_id,
#    device_map="auto",
#    torch_dtype=torch.float16,
#)
#model.eval()
#
#if tokenizer.pad_token is None:
#    tokenizer.pad_token = tokenizer.eos_token
#tokenizer.padding_side = "left"
#
## ======================
## 2. Carregar test set
## ======================
#print("📂 Carregando dataset de teste...")
#test_df = pd.read_parquet("data/gov_lang_br/gov_lang_br.parquet")
##test_df = test_df.iloc[:32]
#
## ======================
## 3. Função de prompt (formato Tucano)
## ======================
#def make_prompt(original_text):
#    """
#    O modelo Tucano segue o formato:
#    <instruction>texto da instrução</instruction>
#    <input>entrada opcional</input>
#    """
#    instruction = "Simplifique o texto a seguir, mantendo o sentido original. Retorne apenas o texto simplificado."
#    prompt = f"<instruction>{instruction}</instruction><input>{original_text}</input><output>"
#    return prompt
#
#print("🧩 Gerando prompts formatados...")
#test_df["prompt"] = test_df["original_text"].apply(make_prompt)
#
## ======================
## 4. Configuração de geração
## ======================
#MAX_NEW_TOKENS = 1024
#generation_config = GenerationConfig(
#    do_sample=True,
#    max_new_tokens=MAX_NEW_TOKENS,
#    renormalize_logits=True,
#    repetition_penalty=1.15,
#    temperature=0.2,
#    top_k=50,
#    top_p=0.9,
#    use_cache=True,
#)
#
## ======================
## 5. Geração batched
## ======================
#BATCH_SIZE = 16  # ajuste conforme sua VRAM
#predictions = []
#references = test_df["paraphrase"].tolist()
#prompts = test_df["prompt"].tolist()
#
#print(f"\n🚀 Gerando saídas em batches de {BATCH_SIZE}...\n")
#
#for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Generating"):
#    batch_prompts = prompts[i : i + BATCH_SIZE]
#
#    batch_inputs = tokenizer(
#        batch_prompts,
#        return_tensors="pt",
#        padding=True,
#        truncation=True,
#        max_length=2048,
#    ).to(model.device)
#
#    with torch.inference_mode():
#        outputs = model.generate(
#            **batch_inputs,
#            generation_config=generation_config,
#            pad_token_id=tokenizer.eos_token_id,
#            eos_token_id=tokenizer.eos_token_id,
#        )
#
#    gen_tokens = outputs[:, batch_inputs["input_ids"].shape[1]:]
#    decoded_batch = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
#
#    # remove qualquer prefixo redundante que o modelo possa repetir
#    cleaned = []
#    for out in decoded_batch:
#        if "<output>" in out:
#            out = out.split("<output>")[-1]
#        if "</output>" in out:
#            out = out.split("</output>")[0]
#        cleaned.append(out.strip())
#    #cleaned = [out.strip().replace("<output>", "").replace("</output>", "").strip() for out in decoded_batch]
#    #print(cleaned)
#    predictions.extend(cleaned)
#
#test_df["tucano_output"] = predictions
#print("\n✅ Geração concluída!\n")
#test_df.to_csv("test_tucano_gov_results.csv", index=False)
test_df = pd.read_csv("test_tucano_legal_results.csv")
# ======================
# 6. Avaliação
# ======================
print("\n📏 Calculando métricas...")

# --- D-SARI ---
print("  - Calculando D-SARI...")
from utils import calculate_d_sari

sari_scores = []
add_scores = []
keep_scores = []
del_scores = []

for src, pred, ref in tqdm(zip(test_df["original_text"], test_df["tucano_output"].astype(str), test_df["paraphrase"]),
                           total=len(test_df), desc="SARI"):

    score, add_score, keep_score, del_score = calculate_d_sari(src, pred, ref)
    sari_scores.append(score)
    add_scores.append(add_score)
    keep_scores.append(keep_score)
    del_scores.append(del_score)
d_sari = np.mean(sari_scores)
add_scores = np.mean(add_scores)
keep_scores = np.mean(keep_scores)
del_scores = np.mean(del_scores)


print(f"F1_add: {add_scores}")
print(f"F1_keep: {keep_scores}")
print(f"P_del: {del_scores}")


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
    test_df["tucano_output"].astype(str).tolist(),
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
    "d_add": add_scores,
    "d_keep":keep_scores,
    "d_del":del_scores,
    "semantic_similarity": semantic_similarity,
    "Flesch": test_df['tucano_output'].astype(str).apply(flesch_portugues).mean()
}

with open("test_metrics_tucano_legal.json", "w") as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)

#test_df.to_csv("test_tucano_legal_results.csv", index=False)

print("\n✅ Métricas finais:")
print(json.dumps(metrics, indent=4, ensure_ascii=False))
print("\n📁 Resultados salvos em 'test_tucano_legal_results.csv'")
