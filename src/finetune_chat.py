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
wandb.init(project="qwen-finetune", name="qwen2.5-7b-lora-v2")

# ======================
# 2. Carregar datasets
# ======================
train_df = pd.read_parquet("splits_output/train_random.parquet")
#train_df = train_df.iloc[:10000]
val_df = pd.read_parquet("splits_output/val_random.parquet")
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
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16,
)

#model.gradient_checkpointing_enable()
#model.config.use_cache = False  # Necessário com gradient checkpointing

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

print(f"🔢 Parâmetros treináveis: {model.print_trainable_parameters()}")

# ======================
# 5. Análise de Comprimentos (IMPORTANTE!)
# ======================
MAX_LENGTH = 4096  # 👈 LIMITE GLOBAL

print("🔍 Analisando tamanhos de prompts e targets...")
prompt_lengths = []
target_lengths = []
# Mensagens template
SYSTEM_MSG = "Você é um assistente especializado em simplificar textos mantendo o significado original."

for idx in range(min(1000, len(train_df))):  # Analisa amostra
    messages_prompt = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": f"Simplifique o seguinte texto:\n{train_df['original_text'].iloc[idx]}"}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages_prompt,
        tokenize=False,
        add_generation_prompt=True
    )
    target = train_df["paraphrase"].iloc[idx]
    
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]
    
    prompt_lengths.append(len(prompt_tokens))
    target_lengths.append(len(target_tokens))

print(f"\n📊 PROMPTS:")
print(f"  Média: {np.mean(prompt_lengths):.0f} tokens")
print(f"  Mediana: {np.median(prompt_lengths):.0f} tokens")
print(f"  P95: {np.percentile(prompt_lengths, 95):.0f} tokens")
print(f"  Máximo: {max(prompt_lengths)} tokens")

print(f"\n📊 TARGETS (respostas):")
print(f"  Média: {np.mean(target_lengths):.0f} tokens")
print(f"  Mediana: {np.median(target_lengths):.0f} tokens")
print(f"  P95: {np.percentile(target_lengths, 95):.0f} tokens")
print(f"  Máximo: {max(target_lengths)} tokens")

# 👇 Define limites baseado nos dados
TARGET_MAX = int(np.percentile(target_lengths, 95))  # Cobre 95% dos casos
PROMPT_MAX = MAX_LENGTH - TARGET_MAX - 10  # -10 para margem de segurança

print(f"\n⚙️  LIMITES DEFINIDOS:")
print(f"  Prompt máximo: {PROMPT_MAX} tokens")
print(f"  Target máximo: {TARGET_MAX} tokens")
print(f"  Total: {PROMPT_MAX + TARGET_MAX + 1} tokens (+ 1 EOS)\n")

# ======================
# 6. Preprocessamento com Chat Template
# ======================
def preprocess_function(example):
    """
    Usa chat template para formatar com system + user + assistant
    """
    # Monta conversa completa (com resposta)
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": f"Simplifique o seguinte texto:\n{example['original_text']}"},
        {"role": "assistant", "content": example["paraphrase"]}
    ]
    
    # Aplica template com resposta completa
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False  # Já temos a resposta
    )
    
    # Tokeniza tudo
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False
    )
    input_ids = tokenized["input_ids"]
    
    # Agora mascara só o prompt (system + user)
    messages_prompt_only = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        messages_prompt_only,
        tokenize=False,
        add_generation_prompt=True
    )
    
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_length = len(prompt_ids)
    
     # Se o input_ids foi truncado, ajusta o prompt_length
    if len(input_ids) < prompt_length:
        # Texto tão grande que até o prompt foi cortado - descarta esse exemplo
        return {"input_ids": [], "labels": []}
    
    # Labels: -100 no prompt, valores reais na resposta
    labels = [-100] * prompt_length + input_ids[prompt_length:]
    
    # Verificação
    assert len(input_ids) <= MAX_LENGTH, f"Sequência muito longa: {len(input_ids)}"
    assert len(labels) == len(input_ids), f"Labels e input_ids com tamanhos diferentes"

    return {"input_ids": input_ids, "labels": labels}

train_dataset = train_dataset.map(preprocess_function, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(preprocess_function, remove_columns=val_dataset.column_names)

# Remove exemplos vazios (onde o prompt foi maior que MAX_LENGTH)
train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) > 0)
val_dataset = val_dataset.filter(lambda x: len(x["input_ids"]) > 0)


# ======================
# 7. Data Collator (com padding até MAX_LENGTH)
# ======================
class CustomDataCollator:
    """Collator que faz padding de input_ids e labels corretamente"""
    
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features):
        # Remove features extras que não precisamos
        batch = {
            "input_ids": [f["input_ids"] for f in features],
            "labels": [f["labels"] for f in features],
        }
        
        # Encontra o tamanho máximo no batch
        max_length = max(len(ids) for ids in batch["input_ids"])
        
        # Arredonda para múltiplo de 8 (otimização GPU)
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) 
                         // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        # Pad input_ids e attention_mask
        padded_input_ids = []
        attention_masks = []
        padded_labels = []
        
        for input_ids, labels in zip(batch["input_ids"], batch["labels"]):
            padding_length = max_length - len(input_ids)
            
            # LEFT padding (importante para geração)
            padded_input_ids.append(
                [self.tokenizer.pad_token_id] * padding_length + input_ids
            )
            attention_masks.append(
                [0] * padding_length + [1] * len(input_ids)
            )
            # Labels: pad com -100 (ignorado na loss)
            padded_labels.append(
                [-100] * padding_length + labels
            )
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

data_collator = CustomDataCollator(tokenizer, pad_to_multiple_of=8)


# ======================
# 8. Métricas de avaliação
# ======================
#sari_metric = evaluate.load("sari")
#embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def compute_metrics(eval_pred):
    return {}

# ======================
# 9. Argumentos de treino
# ======================
training_args = TrainingArguments(
    output_dir="./qwen-finetuned-chat",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=True,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="steps",
    save_steps=4000,
    eval_strategy="steps",
    eval_steps=4000,    
    eval_accumulation_steps=4,
    prediction_loss_only=True,
    report_to="wandb",
    optim="paged_adamw_8bit",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # 👈 EXPLÍCITO: salva baseado no loss
    greater_is_better=False,  # 👈 Menor loss é melhor
    save_total_limit=2,  # 👈 Mantém apenas os 2 melhores checkpoints
    warmup_steps=100,
    lr_scheduler_type = "cosine",
    # OTIMIZAÇÕES DE MEMÓRIA
    #gradient_checkpointing=True,  # Ativa gradient checkpointing
    #gradient_checkpointing_kwargs={"use_reentrant": False},
    # Configurações para múltiplas GPUs
    #ddp_find_unused_parameters=False,  # Otimização DDP
    dataloader_num_workers=4,  # Paraleliza carregamento de dados
    #dataloader_pin_memory=True,  # Acelera transferência GPU
    # Remove exemplos muito longos
    remove_unused_columns=False,
)

# ======================
# 10. Trainer com métricas
# ======================
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ======================
# 11. Iniciar treino
# ======================
print("🚀 Iniciando treino...")
trainer.train()

# ======================
# 10. Salvar modelo final
# ======================
print("💾 Salvando modelo...")
trainer.model.save_pretrained("./qwen-finetuned-lora-chat")
tokenizer.save_pretrained("./qwen-finetuned-lora-chat")

print("✅ Treino completo!")
print(f"📁 Modelo salvo em: ./qwen-finetuned-lora-chat")
