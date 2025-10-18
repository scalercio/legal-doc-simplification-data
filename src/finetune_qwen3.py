import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import pandas as pd
from datasets import Dataset
import wandb
import numpy as np
from math import exp
import math
from tqdm import tqdm

# ======================
# 1. Importar Unsloth
# ======================
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from transformers import Trainer, TrainingArguments

# ======================
# 2. Configurar W&B
# ======================
wandb.init(project="qwen-finetune", name="qwen3-1.7b-unsloth")

# ======================
# 3. Carregar datasets
# ======================
print("📂 Carregando datasets...")
train_df = pd.read_parquet("splits_output/train_random.parquet")
#train_df = train_df.iloc[:1000]
val_df = pd.read_parquet("splits_output/val_random.parquet")

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# ======================
# 4. Carregar modelo com Unsloth
# ======================
print("🚀 Carregando modelo com Unsloth...")

max_seq_length = 4096  # Suporta até 32K com RoPE scaling
dtype = None  # Auto-detecta. Float16 para T4/V100, bfloat16 para Ampere+
load_in_4bit = True  # Usa quantização 4bit

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-1.7B-bnb-4bit",  # ou "unsloth/Qwen3-1.7B-Instruct-bnb-4bit"
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token="hf_...",  # use caso precise de um token HF
)

# ======================
# 5. Adicionar adaptadores LoRA
# ======================
print("🔧 Configurando LoRA...")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # rank LoRA
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Otimização Unsloth (2x mais rápido)
    random_state=42,
    use_rslora=False,  # Rank stabilized LoRA
    loftq_config=None,  # LoftQ quantization
)
# 🔥 CRÍTICO: Forçar configurações após get_peft_model
model.config.use_cache = False  # Obrigatório com gradient checkpointing
if hasattr(model, 'enable_input_require_grads'):
    model.enable_input_require_grads()  # Necessário para alguns modelos quantizados

# ======================
# 6. Análise de Comprimentos
# ======================
MAX_LENGTH_TRAIN = 4096
MAX_LENGTH_EVAL = 2048
SYSTEM_MSG = "Você é um assistente simplificador de textos."

print("🔍 Analisando tamanhos de prompts e targets...")
prompt_lengths = []
target_lengths = []

for idx in range(min(1000, len(train_df))):
    messages_prompt = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": f"Simplifique o texto a seguir, mas mantenha o sentido original. Retorne só o texto simplificado.\n\nTexto original: {train_df['original_text'].iloc[idx]} \n\nTexto simplificado: "}
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

TARGET_MAX = int(np.percentile(target_lengths, 95))
PROMPT_MAX = MAX_LENGTH_TRAIN - TARGET_MAX - 10
MAX_LENGTH_TRAIN = PROMPT_MAX + TARGET_MAX + 1

print(f"\n⚙️  LIMITES DEFINIDOS:")
print(f"  Prompt máximo: {PROMPT_MAX} tokens")
print(f"  Target máximo: {TARGET_MAX} tokens")
print(f"  Total: {PROMPT_MAX + TARGET_MAX + 1} tokens (+ 1 EOS)\n")

# ======================
# 7. Preprocessamento com Chat Template (mantém controle total)
# ======================
def preprocess_function(example, max_len=4096):
    """
    Usa chat template para formatar com system + user + assistant
    """
    # Monta conversa completa (com resposta)
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": f"Simplifique o texto a seguir, mas mantenha o sentido original. Retorne só o texto simplificado.\n\nTexto original: {example['original_text']} \n\nTexto simplificado: "},
        {"role": "assistant", "content": example["paraphrase"]}
    ]
    
    # Aplica template com resposta completa
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False  # Já temos a resposta
    )
    
    # Tokeniza SEM truncation primeiro para ver o tamanho real
    full_tokenized = tokenizer(
        full_text,
        truncation=False,
        add_special_tokens=False
    )
    original_length = len(full_tokenized["input_ids"])
    if original_length > max_len * 1.5:  # Mais que 50% seria cortado
        return {"input_ids": [], "labels": []}
    
    # Tokeniza tudo
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_len,
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
    
    response_length = len(input_ids) - prompt_length
    
    # Exige mínimo de tokens na resposta
    MIN_RESPONSE_TOKENS = 64
    if response_length < MIN_RESPONSE_TOKENS:
        return {"input_ids": [], "labels": []}
    
    # Se o input_ids foi truncado, ajusta o prompt_length
    if len(input_ids) <= prompt_length:
        return {"input_ids": [], "labels": []}
    
    # Labels: -100 no prompt, valores reais na resposta
    labels = [-100] * prompt_length + input_ids[prompt_length:]
    
    # Validação extra de sanidade
    valid_labels = [l for l in labels if l != -100]
    
    if len(valid_labels) < MIN_RESPONSE_TOKENS:
        return {"input_ids": [], "labels": []}
    
    # Verificação
    assert len(input_ids) <= max_len, f"Sequência muito longa: {len(input_ids)}"
    assert len(labels) == len(input_ids), f"Labels e input_ids com tamanhos diferentes"
    assert len(valid_labels) >= MIN_RESPONSE_TOKENS, f"Poucos labels válidos: {len(valid_labels)}"

    return {"input_ids": input_ids, "labels": labels}

train_dataset = train_dataset.map(
    lambda x: preprocess_function(x, max_len=MAX_LENGTH_TRAIN), 
    remove_columns=train_dataset.column_names, 
    load_from_cache_file=False
)

val_dataset = val_dataset.map(
    lambda x: preprocess_function(x, max_len=MAX_LENGTH_EVAL), 
    remove_columns=val_dataset.column_names, 
    load_from_cache_file=False
)

# Remove exemplos vazios (onde o prompt foi maior que MAX_LENGTH)
def filter_valid_robust(example):
    """
    Filtro mais rigoroso que garante exemplos com conteúdo substancial
    """
    input_ids = example["input_ids"]
    labels = example["labels"]
    
    if len(input_ids) == 0:
        return False
    
    valid_labels = [l for l in labels if l != -100]
    
    if len(valid_labels) < 20:
        return False
    
    label_ratio = len(valid_labels) / len(labels)
    if label_ratio < 0.05:
        return False
    
    if len(input_ids) != len(labels):
        return False
    
    return True

train_dataset = train_dataset.filter(filter_valid_robust)
print(f"📊 Exemplos de treino após filtro: {len(train_dataset)}")
val_dataset = val_dataset.filter(filter_valid_robust)
print(f"📊 Exemplos de validação após filtro: {len(val_dataset)}")

# ======================
# 8. Data Collator (com padding até MAX_LENGTH)
# ======================
class CustomDataCollator:
    """Collator que faz padding de input_ids e labels corretamente"""
    
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features):
        batch = {
            "input_ids": [f["input_ids"] for f in features],
            "labels": [f["labels"] for f in features],
        }
        
        max_length = max(len(ids) for ids in batch["input_ids"])
        
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) 
                         // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        padded_input_ids = []
        attention_masks = []
        padded_labels = []
        
        for input_ids, labels in zip(batch["input_ids"], batch["labels"]):
            padding_length = max_length - len(input_ids)
            
            # LEFT padding (importante para geração)
            padded_input_ids.append(
                input_ids + [self.tokenizer.pad_token_id] * padding_length
            )
            attention_masks.append(
                [1] * len(input_ids) + [0] * padding_length
            )
            # Labels: pad com -100 (ignorado na loss)
            padded_labels.append(
                labels + [-100] * padding_length
            )
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

data_collator = CustomDataCollator(tokenizer, pad_to_multiple_of=8)

# ======================
# 9. Custom Evaluate Function
# ======================
from torch.utils.data import DataLoader

def custom_evaluate(trainer_self, ignore_keys=None, metric_key_prefix="eval", **kwargs):
    """
    Substitui completamente o evaluate() do Trainer.
    Versão robusta que garante estado limpo do modelo.
    """
    print("\n🔎 Rodando Custom Evaluation...")
    
    # Salva e limpa estado do modelo
    training_mode = trainer_self.model.training
    grad_checkpointing_enabled = (
        hasattr(trainer_self.model, 'is_gradient_checkpointing') 
        and trainer_self.model.is_gradient_checkpointing
    )
    use_cache_original = trainer_self.model.config.use_cache
    
    trainer_self.model.eval()
    
    # Limpa cache interno do modelo
    if hasattr(trainer_self.model, 'gradient_checkpointing_disable'):
        trainer_self.model.gradient_checkpointing_disable()
    
    # Habilita cache para inferência (mais rápido)
    trainer_self.model.config.use_cache = True
    
    torch.cuda.empty_cache()
    
    # Detecta precisão
    if trainer_self.args.bf16:
        dtype = "bf16"
        trainer_self.model.bfloat16()
    elif trainer_self.args.fp16:
        dtype = "fp16"
        trainer_self.model.half()
    else:
        dtype = "fp32"
        trainer_self.model.float()
    
    print(f"🧠 Avaliando em modo: {dtype}")
    
    loader = DataLoader(
        trainer_self.eval_dataset,
        batch_size=trainer_self.args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=trainer_self.data_collator,
        drop_last=False,
    )
    
    losses = []
    
    with torch.cuda.amp.autocast(enabled=False):
        with torch.no_grad():
            for i, batch in enumerate(loader):
                try:
                    batch = {k: v.to(trainer_self.model.device) for k, v in batch.items()}
                    
                    if (batch['labels'] != -100).sum() == 0:
                        print(f"⚠️ Batch {i}: todos labels são -100, pulando...")
                        continue
                    
                    outputs = trainer_self.model(**batch)
                    loss = outputs.loss
                    
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        losses.append(loss.item())
                        if i % 10 == 0:
                            print(f"✅ Batch {i}: loss = {loss.item():.4f}")
                    else:
                        print(f"⚠️ NaN/Inf detectado no batch {i}, ignorando...")
                        
                except torch.cuda.OutOfMemoryError:
                    print(f"💥 OOM no batch {i}, limpando cache...")
                    torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    print(f"❌ Erro no batch {i}: {e}")
                    continue
    
    print("\n🔄 Restaurando estado do modelo para treino...")
    
    if training_mode:
        trainer_self.model.train()
        
    # RESTAURA GRADIENT CHECKPOINTING
    if grad_checkpointing_enabled:
        print("✅ Reativando gradient checkpointing...")
        trainer_self.model.gradient_checkpointing_enable()
    
    # RESTAURA use_cache
    trainer_self.model.config.use_cache = use_cache_original
    
    torch.cuda.empty_cache()
    
    # Calcula métricas
    if len(losses) == 0:
        print("\n⚠️ NENHUMA LOSS VÁLIDA! Retornando NaN...")
        mean_loss = float("nan")
        ppl = float("nan")
    else:
        mean_loss = sum(losses) / len(losses)
        ppl = math.exp(mean_loss) if not math.isnan(mean_loss) and mean_loss < 100 else float("nan")
    
    print(f"\n✅ eval_loss_safe = {mean_loss:.4f}, perplexity = {ppl:.2f}")
    print(f"   Batches válidos: {len(losses)}/{i+1}")
    
    metrics = {
        f"{metric_key_prefix}_loss_safe": mean_loss,
        f"{metric_key_prefix}_ppl_safe": ppl,
        f"{metric_key_prefix}_valid_batches": len(losses),
    }
    
    trainer_self.log(metrics)
    
    trainer_self.control = trainer_self.callback_handler.on_evaluate(
        trainer_self.args, trainer_self.state, trainer_self.control, metrics=metrics
    )
    
    return metrics

# ======================
# 8. Configurar Trainer
# ======================
print("⚙️ Configurando trainer...")
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    max_seq_length=max_seq_length,
    data_collator=data_collator,
    dataset_num_proc=2,  # Multiprocessing para preparar dados
    packing=False,  # Pode ser True para treino mais rápido, mas cuidado com exemplos longos
    args=SFTConfig(
        # Diretório de saída
        output_dir="./qwen3-1.7b-unsloth-output",
        
        # Batch sizes
        per_device_train_batch_size=16,  # Unsloth permite batch maior
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=16,  # Reduzido já que batch é maior
        
        # Training schedule
        num_train_epochs=3,
        max_steps=-1,  # -1 = usar num_train_epochs
        
        # Learning rate
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        
        # Otimizador
        optim="adamw_8bit",  # Unsloth otimiza isso automaticamente
        weight_decay=0.01,
        max_grad_norm=0.5,
        
        # Precisão
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        
        # Logging
        logging_steps=5,
        logging_dir="./logs",
        logging_first_step=True,
        report_to="wandb",
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=500,
        eval_accumulation_steps=2,
        prediction_loss_only=True,
        
        # Saving
        save_strategy="steps",
        save_steps=500,
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss_safe",
        greater_is_better=False,
        
        # Otimizações
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        remove_unused_columns=False, 
        group_by_length=True,  # Agrupa exemplos de tamanho similar
        
        logging_nan_inf_filter=False,
        
        # Seed
        seed=42,
        data_seed=42,
    ),
)



# Substitui evaluate pelo custom
import types
trainer.evaluate = types.MethodType(custom_evaluate, trainer)
print("🚀 Iniciando treino...")
print(f"📊 Exemplos de treino: {len(train_dataset)}")
print(f"📊 Exemplos de validação: {len(val_dataset)}")

# Mostra estatísticas do modelo
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"💾 GPU: {gpu_stats.name}")
print(f"💾 Memória reservada: {start_gpu_memory} GB de {max_memory} GB")

trainer_stats = trainer.train()

# ======================
# 11. Treinar
# ======================
print("\n💾 Salvando modelo...")

# Salva o modelo fine-tuned
model.save_pretrained("./qwen3-1.7b-lora-final")
tokenizer.save_pretrained("./qwen3-1.7b-lora-final")

# Opcional: Salva versão merged (LoRA + base model)
# model.save_pretrained_merged(
#     "./qwen3-1.7b-merged",
#     tokenizer,
#     save_method="merged_16bit",  # ou "merged_4bit", "lora"
# )

# Opcional: Push para HuggingFace Hub
# model.push_to_hub_merged(
#     "seu-usuario/qwen3-1.7b-simplificador",
#     tokenizer,
#     save_method="merged_16bit",
#     token="hf_..."
# )

# Estatísticas finais
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)

print("\n" + "="*50)
print("✅ Treino completo!")
print("="*50)
print(f"⏱️  Tempo de treino: {trainer_stats.metrics['train_runtime']:.2f}s")
print(f"💾 Memória usada: {used_memory} GB ({used_percentage}%)")
print(f"🎯 Memória do LoRA: {used_memory_for_lora} GB")
print(f"📁 Modelo salvo em: ./qwen3-1.7b-lora-final")
print("="*50)

# ======================
# 11. Teste rápido de inferência
# ======================
print("\n🧪 Testando inferência...")

FastLanguageModel.for_inference(model)  # Ativa modo inferência otimizado

test_text = "Este é um texto de teste deveras complexo que precisa ser simplificado."
messages = [
    {"role": "system", "content": SYSTEM_MSG},
    {"role": "user", "content": f"Simplifique o texto a seguir, mas mantenha o sentido original. Retorne só o texto simplificado.\n\nTexto original: {test_text} \n\nTexto simplificado: "}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\n📝 Entrada: {test_text}")
print(f"✨ Saída: {result}")

print("\n✅ Script finalizado!")