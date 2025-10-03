import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import wandb
import evaluate
import numpy as np
from math import exp
from sentence_transformers import SentenceTransformer, util

# ======================
# 1. Configurar W&B
# ======================
#wandb.init(project="qwen-finetune", name="qwen2.5-7b-lora-masked")

# ======================
# 2. Carregar datasets
# ======================
train_df = pd.read_parquet("iudicium_textum_paraphrases_v2.parquet.final")
val_df = pd.read_parquet("iudicium_textum_paraphrases_v2.parquet.final")
#test_df = pd.read_parquet("acordaos_tcu_v4_intermediate_1000.parquet")

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
#test_dataset = Dataset.from_pandas(test_df)

# ======================
# 3. Modelo e tokenizer
# ======================
model_name = "Qwen/Qwen2.5-7B-Instruct"  # use o base adequado (não o quantizado Q4_K_M)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16,
)

# ======================
# 4. Configuração do LoRA
# ======================
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# ======================
# 5. Preprocessamento com máscara
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

train_dataset = train_dataset.map(preprocess_function, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(preprocess_function, remove_columns=val_dataset.column_names)
#test_dataset = test_dataset.map(preprocess_function, remove_columns=test_dataset.column_names)

# ======================
# 6. Métricas de avaliação
# ======================
sari_metric = evaluate.load("sari")
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Extrair só a resposta (após "Resposta:")
    decoded_preds = [pred.split("Resposta:")[-1].strip() if "Resposta:" in pred else pred.strip() for pred in decoded_preds]
    decoded_labels = [lab.strip() for lab in decoded_labels]

    # Perplexidade (aproximada pelo loss médio já logado)
    # Aqui calculamos só para logging extra
    loss = np.mean((logits - labels) ** 2)
    perplexity = exp(loss) if loss < 20 else float("inf")

    # D-SARI
    sari_scores = []
    for src, pred, ref in zip(val_df["original_text"], decoded_preds, decoded_labels):
        try:
            sari_score = sari_metric.compute(sources=[src], predictions=[pred], references=[[ref]])
            sari_scores.append(sari_score["sari"])
        except:
            sari_scores.append(0.0)
    sari_mean = np.mean(sari_scores)

    # Similaridade semântica (entrada vs saída gerada)
    sims = []
    for src, pred in zip(val_df["original_text"], decoded_preds):
        emb_src = embedder.encode(src, convert_to_tensor=True)
        emb_pred = embedder.encode(pred, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(emb_src, emb_pred).item()
        sims.append(sim)
    sim_mean = np.mean(sims)

    return {
        "perplexity": perplexity,
        "d_sari": sari_mean,
        "semantic_similarity": sim_mean,
    }

# ======================
# 7. Argumentos de treino
# ======================
training_args = TrainingArguments(
    output_dir="./qwen-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="steps",
    save_steps=600,
    eval_strategy="steps",
    eval_steps=200,
    #report_to="wandb",
    optim="paged_adamw_8bit",
    load_best_model_at_end=True,
)

# ======================
# 8. Trainer com métricas
# ======================
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args,
    compute_metrics=compute_metrics,
)

# ======================
# 9. Iniciar treino
# ======================
trainer.train()

# ======================
# 10. Salvar modelo final
# ======================
trainer.model.save_pretrained("./qwen-finetuned-lora")
tokenizer.save_pretrained("./qwen-finetuned-lora")
