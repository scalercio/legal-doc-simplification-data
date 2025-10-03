import torch
import pandas as pd
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import evaluate
import numpy as np
from sentence_transformers import SentenceTransformer, util
import wandb

# ======================
# 1. Carregar modelo treinado
# ======================
use_finetuned = False
model_dir = "./qwen-finetuned-lora"  # ajuste para o path onde salvou o fine-tune
base_model = "Qwen/Qwen2.5-7B-Instruct"  # mesmo modelo base usado no treino

wandb.init(project="qwen-finetune", name="eval-test-finetuned" if use_finetuned else "eval-test-base")

if use_finetuned:
    model_path = model_dir
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, model_dir)  # carrega LoRA
else:
    model_path = base_model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )


tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model.eval()

# ======================
# 2. Carregar test set
# ======================
test_df = pd.read_parquet("./data/museum/museum.parquet")
test_dataset = Dataset.from_pandas(test_df)

# ======================
# 3. Preprocessamento com máscara
# ======================
def preprocess_function(example):
    # Prompt fixo
    prompt = f"Simplifique o seguinte texto:\n{example['original_text']}\nResposta: "
    target = example["paraphrase"]

    # Tokenização separada
    prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=4096)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False, truncation=True, max_length=4096)["input_ids"]

    # Input = prompt + resposta + <eos>
    input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]

    # Labels: ignora prompt, aprende só resposta
    labels = [-100] * len(prompt_ids) + target_ids + [tokenizer.eos_token_id]

    return {"input_ids": input_ids, "labels": labels}

test_dataset = test_dataset.map(preprocess_function, remove_columns=test_dataset.column_names)

# ======================
# 4. Métricas
# ======================
sari_metric = evaluate.load("sari")
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

decoded_preds, decoded_labels = [], []

# ======================
# 5. Geração no test set
# ======================
for i in range(len(test_dataset)):
    print(i)
    inputs = torch.tensor([test_dataset[i]["input_ids"]]).to(model.device)
    attention_mask = torch.ones_like(inputs).to(model.device) 

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,      # 👈 evita travamento
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_tokens = outputs[0][inputs.shape[1]:]
    pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    label_text = tokenizer.decode([id for id in test_dataset[i]["labels"] if id != -100], skip_special_tokens=True)

    # Isola só a resposta
    #pred_text = pred_text.split("Resposta:")[-1].strip() if "Resposta:" in pred_text else pred_text.strip()
    label_text = label_text.strip()

    decoded_preds.append(pred_text)
    decoded_labels.append(label_text)

# ======================
# 6. Calcular métricas
# ======================
# Perplexidade (aproximada pelo loss médio no test set)
losses = []
for i in range(len(test_dataset)):
    inputs = torch.tensor([test_dataset[i]["input_ids"]]).to(model.device)
    labels = torch.tensor([test_dataset[i]["labels"]]).to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=inputs, labels=labels)
        losses.append(outputs.loss.item())

avg_loss = np.mean(losses)
perplexity = np.exp(avg_loss) if avg_loss < 20 else float("inf")

# D-SARI
sari_scores = []
for src, pred, ref in zip(test_df["original_text"], decoded_preds, decoded_labels):
    try:
        sari_score = sari_metric.compute(sources=[src], predictions=[pred], references=[[ref]])
        sari_scores.append(sari_score["sari"])
    except:
        sari_scores.append(0.0)
d_sari = np.mean(sari_scores)

# Similaridade semântica (entrada vs saída)
sims = []
for src, pred in zip(test_df["original_text"], decoded_preds):
    emb_src = embedder.encode(src, convert_to_tensor=True)
    emb_pred = embedder.encode(pred, convert_to_tensor=True)
    sim = util.pytorch_cos_sim(emb_src, emb_pred).item()
    sims.append(sim)
semantic_similarity = np.mean(sims)

# ======================
# 7. Salvar métricas finais
# ======================
metrics = {
    "avg_loss": avg_loss,
    "perplexity": perplexity,
    "d_sari": d_sari,
    "semantic_similarity": semantic_similarity,
}

with open("test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)

print("✅ Métricas no test set:")
print(json.dumps(metrics, indent=4, ensure_ascii=False))
