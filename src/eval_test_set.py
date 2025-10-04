import torch
import pandas as pd
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import evaluate
import numpy as np
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# ======================
# 1. Carregar modelo treinado
# ======================
use_finetuned = False
model_dir = "./qwen-finetuned-lora"  # ajuste para o path onde salvou o fine-tune
base_model = "Qwen/Qwen2.5-7B-Instruct"  # mesmo modelo base usado no treino

#wandb.init(project="qwen-finetune", name="eval-test-finetuned" if use_finetuned else "eval-test-base")

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
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

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

    return {"input_ids": input_ids, "labels": labels, "target_text": target}

test_dataset = test_dataset.map(preprocess_function, remove_columns=test_dataset.column_names)

def collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    labels = []
    attention_mask = []
    target_texts = []

    for ex in batch:
        ids = ex["input_ids"]
        labs = ex["labels"]
        pad_len = max_len - len(ids)

        # LEFT pad with pad_token_id
        input_ids.append([tokenizer.pad_token_id] * pad_len + ids)
        # pad labels with -100 (ignored in loss)
        labels.append([-100] * pad_len + labs)
        # attention_mask: 0 for pads (left), 1 for real tokens
        attention_mask.append([0] * pad_len + [1] * len(ids))
        target_texts.append(ex["target_text"])  # 👈 NOVO

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "target_texts": target_texts  # 👈 NOVO
    }
test_loader = DataLoader(test_dataset, batch_size=24, collate_fn=collate_fn)
# ======================
# 4. Métricas
# ======================
sari_metric = evaluate.load("sari")
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

decoded_preds, decoded_labels = [], []

# 5. Geração batched
predictions, references = [], []
i=0
with torch.inference_mode():
    for batch in tqdm(test_loader, desc="Generating"):
        i=i+1
        print(i)
        batch_device = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
        #with torch.no_grad():
        outputs = model.generate(
            input_ids=batch_device["input_ids"],
            attention_mask=batch_device["attention_mask"],
            max_new_tokens=256,
            do_sample=False,    # mais rápido
            num_beams=1,        # reduzir custo
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            return_dict_in_generate=False,
        )
        gen_tokens = outputs[:, batch_device["input_ids"].shape[1]:]
        decoded_preds = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        #decoded_refs = tokenizer.batch_decode(batch["labels"].masked_fill(batch["labels"] == -100, tokenizer.pad_token_id), skip_special_tokens=True)

        decoded_refs = batch["target_texts"]
        predictions.extend(decoded_preds)
        references.extend(decoded_refs)

#    gen_tokens = outputs[0][inputs.shape[1]:]
#    pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
#    label_text = tokenizer.decode([id for id in test_dataset[i]["labels"] if id != -100], skip_special_tokens=True)
#
#    # Isola só a resposta
#    #pred_text = pred_text.split("Resposta:")[-1].strip() if "Resposta:" in pred_text else pred_text.strip()
#    label_text = label_text.strip()
#
#    decoded_preds.append(pred_text)
#    decoded_labels.append(label_text)

# ======================
# 6. Calcular métricas
# ======================
# Perplexidade (aproximada pelo loss médio no test set)
# Calcular perplexidade em batch
#losses = []
#for batch in tqdm(test_loader, desc="Perplexity"):
#    batch = {k: v.to(model.device) for k, v in batch.items()}
#    with torch.no_grad():
#        outputs = model(**batch)
#        losses.append(outputs.loss.item())
#
#avg_loss = np.mean(losses)
#perplexity = np.exp(avg_loss)
#
##wandb.log({"eval_loss": avg_loss, "perplexity": perplexity})
#
#print(f"Eval loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")

#losses = []
#for i in range(len(test_dataset)):
#    inputs = torch.tensor([test_dataset[i]["input_ids"]]).to(model.device)
#    labels = torch.tensor([test_dataset[i]["labels"]]).to(model.device)
#
#    with torch.no_grad():
#        outputs = model(input_ids=inputs, labels=labels)
#        losses.append(outputs.loss.item())
#
#avg_loss = np.mean(losses)
#perplexity = np.exp(avg_loss) if avg_loss < 20 else float("inf")

# D-SARI
sari_scores = []
for src, pred, ref in zip(test_df["original_text"], predictions, references):
    try:
        sari_score = sari_metric.compute(sources=[src], predictions=[pred], references=[[ref]])
        sari_scores.append(sari_score["sari"])
    except:
        sari_scores.append(0.0)
d_sari = np.mean(sari_scores)

# Similaridade semântica (entrada vs saída)
sims = []
for src, pred in zip(test_df["original_text"], predictions):
    emb_src = embedder.encode(src, convert_to_tensor=True)
    emb_pred = embedder.encode(pred, convert_to_tensor=True)
    sim = util.pytorch_cos_sim(emb_src, emb_pred).item()
    sims.append(sim)
semantic_similarity = np.mean(sims)

# ======================
# 7. Salvar métricas finais
# ======================
metrics = {
    #"avg_loss": avg_loss,
    #"perplexity": perplexity,
    "d_sari": d_sari,
    "semantic_similarity": semantic_similarity,
}

with open("test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)

print("✅ Métricas no test set:")
print(json.dumps(metrics, indent=4, ensure_ascii=False))
