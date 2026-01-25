import torch
import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import evaluate
import numpy as np
from sentence_transformers import SentenceTransformer, util
from utils import flesch_portugues

# ======================
# 1. Carregar test set
# ======================
print("📂 Carregando dataset de teste...")
#test_df = pd.read_parquet("test_random.parquet")
#test_df = test_df.iloc[:32]
generate = False
file_name = "test_set_legal_qwen2.5-ft_sample_false_partial_70"

if generate:
    
    # ======================
    # 2. Carregar modelo treinado
    # ======================
    use_finetuned = True  # ✅ defina True se quiser usar o modelo LoRA ajustado
    model_dir = "./qwen-finetuned-chat2/checkpoint-52000"
    base_model = "Qwen/Qwen2.5-7B-Instruct"

    print(f"🔧 Carregando modelo ({'fine-tunado' if use_finetuned else 'base'})...")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )

    if use_finetuned:
        model = PeftModel.from_pretrained(model, model_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    model.eval()

    # ======================
    # 3. Preparar prompts
    # ======================
    print("🧩 Gerando prompts formatados...")

    def make_prompt(original_text):
        messages = [
            {
                "role": "system",
                "content": "Você é um assistente simplificador de textos."
            },
            {
                "role": "user",
                "content": f"Simplifique o texto a seguir, mas mantenha o sentido original. Retorne só o texto simplificado.\n\nTexto original: {original_text} \n\nTexto simplificado: "
            }
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    test_df["prompt"] = test_df["original_text"].apply(make_prompt)

    # ======================
    # 4. Geração batched
    # ======================
    BATCH_SIZE = 8  # ajuste conforme a VRAM disponível
    MAX_NEW_TOKENS = 2048

    predictions = []
    references = test_df["paraphrase"].tolist()
    prompts = test_df["prompt"].tolist()

    print(f"\n🚀 Gerando saídas em batches de {BATCH_SIZE}...\n")
    last_checkpoint = 0

    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Generating"):
        batch_prompts = prompts[i : i + BATCH_SIZE]
        # Tokeniza lote (padding à esquerda)
        batch_inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **batch_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1,
                top_p=0.9,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
            )

        gen_tokens = outputs.sequences[:, batch_inputs["input_ids"].shape[1]:]
        decoded_batch = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        predictions.extend([d.strip() for d in decoded_batch])
        
        # Salvar progresso a cada 10%
        progress = (i + BATCH_SIZE) / len(prompts)
        current_checkpoint = int(progress * 10)  # 0–10

        if current_checkpoint > last_checkpoint:
            test_df_partial = test_df.iloc[:len(predictions)].copy()
            test_df_partial["qwen2.5_output"] = predictions
            test_df_partial.to_csv(f"{file_name}_partial_{current_checkpoint*10}.csv", index=False)
            print(f"💾 Progresso salvo em {current_checkpoint*10}% ({len(predictions)} registros)")
            last_checkpoint = current_checkpoint

    test_df["qwen2.5_output"] = predictions
    print("\n✅ Geração concluída!\n")
else:
    test_df = pd.read_csv(file_name + ".csv",)
    
# ======================
# 5. Avaliação
# ======================
print("\n📏 Calculando métricas...")
#test_df.to_csv(file_name + ".csv", index=False)
# --- SARI ---
print("  - Calculando D-SARI...")

from easse.sari import corpus_sari
sari_result = corpus_sari(
    test_df["original_text"].tolist(),
    test_df["qwen2.5_output"].astype(str).tolist(),
    [test_df["paraphrase"].tolist()]
)
print("Qwen2.5 test:")
print(sari_result)

# To get individual components, you can use:
from easse.sari import get_corpus_sari_operation_scores
from utils import calculate_d_sari

sari_scores = []
add_scores = []
keep_scores = []
del_scores = []

for src, pred, ref in tqdm(zip(test_df["original_text"], test_df["qwen2.5_output"].astype(str), test_df["paraphrase"]),
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

#add_score, keep_score, del_score = get_corpus_sari_operation_scores(
#    test_df["original_text"].tolist(), test_df["qwen2.5_output"].tolist(), [test_df["paraphrase"].tolist()]
#)

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
    test_df["qwen2.5_output"].astype(str).tolist(),
    batch_size=EMBED_BATCH_SIZE,
    show_progress_bar=True,
    convert_to_tensor=True
)

sims = util.pytorch_cos_sim(src_embeddings, pred_embeddings)
semantic_similarity = torch.diagonal(sims).mean().item()
print(f"  ✓ Similaridade Semântica: {semantic_similarity:.4f}")

# ======================
# 6. Salvar resultados
# ======================
metrics = {
    "d_sari": d_sari,
    "d_add": add_scores,
    "d_keep":keep_scores,
    "d_del":del_scores,
    "semantic_similarity": semantic_similarity,
    "Flesch": test_df['qwen2.5_output'].astype(str).apply(flesch_portugues).mean()
}

with open(file_name + "_final.json", "w") as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)

#test_df.to_csv(file_name + ".csv", index=False)

print("\n✅ Métricas finais:")
print(json.dumps(metrics, indent=4, ensure_ascii=False))
