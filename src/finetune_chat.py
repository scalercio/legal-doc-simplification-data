import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import wandb
import evaluate
import numpy as np
from math import exp
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import math

# ======================
# 1. Configurar W&B
# ======================
wandb.init(project="qwen-finetune", name="qwen2.5-7b-lora-trainer")

# ======================
# 2. Carregar datasets
# ======================
train_df = pd.read_parquet("train_random.parquet")
#train_df = train_df.iloc[:100]
val_df = pd.read_parquet("val_random.parquet")
#test_df = pd.read_parquet("acordaos_tcu_v4_intermediate_1000.parquet")
#val_df = val_df.iloc[:200]

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
#test_dataset = Dataset.from_pandas(test_df)

# ======================
# 3. Modelo e tokenizer
# ======================
model_name = "Qwen/Qwen2.5-7B-Instruct"  # use o base adequado (não o quantizado Q4_K_M)
# Configuração NF4 (Normal Float 4-bit) - MELHOR para finetune
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # 🔥 Quantização dupla (economiza ~0.4 bits/param)
    bnb_4bit_quant_type="nf4",  # 🔥 NF4 é melhor que FP4 para treino
    bnb_4bit_compute_dtype=torch.bfloat16,  # Computação em bf16
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,  # Reduz uso de RAM durante carregamento
    max_memory={0: "23GB"}, 
)

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,  # Ativa gradient checkpointing
)

# Ativa gradient checkpointing manualmente
model.gradient_checkpointing_enable()
model.config.use_cache = False  # OBRIGATÓRIO com gradient checkpointing

#model.gradient_checkpointing_enable()
#model.config.use_cache = False  # Necessário com gradient checkpointing

# ======================
# 4. Configuração do LoRA
# ======================
peft_config = LoraConfig(
    r=16,
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
MAX_LENGTH_TRAIN = 4096
MAX_LENGTH_EVAL  = 2048

print("🔍 Analisando tamanhos de prompts e targets...")
prompt_lengths = []
target_lengths = []
# Mensagens template
SYSTEM_MSG = "Você é um assistente simplificador de textos."

for idx in range(min(1000, len(train_df))):  # Analisa amostra
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

# 👇 Define limites baseado nos dados
TARGET_MAX = int(np.percentile(target_lengths, 95))  # Cobre 95% dos casos
PROMPT_MAX = MAX_LENGTH_TRAIN - TARGET_MAX - 10  # -10 para margem de segurança
MAX_LENGTH_TRAIN = PROMPT_MAX + TARGET_MAX + 1

print(f"\n⚙️  LIMITES DEFINIDOS:")
print(f"  Prompt máximo: {PROMPT_MAX} tokens")
print(f"  Target máximo: {TARGET_MAX} tokens")
print(f"  Total: {PROMPT_MAX + TARGET_MAX + 1} tokens (+ 1 EOS)\n")

# ======================
# 6. Preprocessamento com Chat Template
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
    
    # 🔥 NOVO: Tokeniza SEM truncation primeiro para ver o tamanho real
    full_tokenized = tokenizer(
        full_text,
        truncation=False,
        add_special_tokens=False
    )
    original_length = len(full_tokenized["input_ids"])
    if original_length > max_len * 1.5:  # Mais que 50% seria cortado. Pouco útil
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
    
    # 🔥 NOVO: Exige mínimo de tokens na resposta
    MIN_RESPONSE_TOKENS = 64  # Ajuste conforme necessário
    if response_length < MIN_RESPONSE_TOKENS:
        # Resposta muito curta após truncation - descarta
        return {"input_ids": [], "labels": []}
    
     # Se o input_ids foi truncado, ajusta o prompt_length
    if len(input_ids) <= prompt_length:
        # Texto tão grande que até o prompt foi cortado - descarta esse exemplo
        return {"input_ids": [], "labels": []}
    
    # Labels: -100 no prompt, valores reais na resposta
    labels = [-100] * prompt_length + input_ids[prompt_length:]
    
    # Validação extra de sanidade
    valid_labels = [l for l in labels if l != -100]
    
    if len(valid_labels) < MIN_RESPONSE_TOKENS:
        # Algo deu errado, descarta
        return {"input_ids": [], "labels": []}
    
    # Verificação
    assert len(input_ids) <= max_len, f"Sequência muito longa: {len(input_ids)}"
    assert len(labels) == len(input_ids), f"Labels e input_ids com tamanhos diferentes"
    assert len(valid_labels) >= MIN_RESPONSE_TOKENS, f"Poucos labels válidos: {len(valid_labels)}"

    return {"input_ids": input_ids, "labels": labels}

train_dataset = train_dataset.map(lambda x: preprocess_function(x, max_len=MAX_LENGTH_TRAIN), remove_columns=train_dataset.column_names, load_from_cache_file=False)
val_dataset = val_dataset.map(lambda x: preprocess_function(x, max_len=MAX_LENGTH_EVAL), remove_columns=val_dataset.column_names, load_from_cache_file=False)

# Remove exemplos vazios (onde o prompt foi maior que MAX_LENGTH)
# Filtro robusto
def filter_valid(example):
    return len(example["input_ids"]) > 0 and any(l != -100 for l in example["labels"])
def filter_valid_robust(example):
    """
    Filtro mais rigoroso que garante exemplos com conteúdo substancial
    """
    input_ids = example["input_ids"]
    labels = example["labels"]
    
    # 1. Verifica se não está vazio
    if len(input_ids) == 0:
        return False
    
    # 2. Conta labels válidos (não são -100)
    valid_labels = [l for l in labels if l != -100]
    
    # 🔥 CRÍTICO: Exige MÍNIMO de labels válidos
    # Se tem menos de 10 tokens válidos, descarta
    if len(valid_labels) < 20:
        return False
    
    # 3. Verifica proporção mínima de labels válidos
    # Pelo menos 5% do exemplo deve ser treinável
    label_ratio = len(valid_labels) / len(labels)
    if label_ratio < 0.05:
        return False
    
    # 4. Verifica tamanhos consistentes
    if len(input_ids) != len(labels):
        return False
    
    return True

train_dataset = train_dataset.filter(filter_valid_robust)
print(len(val_dataset))
val_dataset   = val_dataset.filter(filter_valid_robust)
print(len(val_dataset))


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

from torch.utils.data import DataLoader

def custom_evaluate(trainer_self, ignore_keys=None, metric_key_prefix="eval", **kwargs):
    """
    Substitui completamente o evaluate() do Trainer.
    Versão robusta que garante estado limpo do modelo.
    """
    print("\n🔎 Rodando Custom Evaluation...")
    
    # 🔥 CRÍTICO: Salva e limpa estado do modelo
    training_mode = trainer_self.model.training
    
    trainer_self.model.eval()
    torch.cuda.empty_cache()
    
    # 🔥 ADICIONA: Limpa cache interno do modelo (importante para LoRA)
    if hasattr(trainer_self.model, 'gradient_checkpointing_disable'):
        trainer_self.model.gradient_checkpointing_disable()
    
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
    
    # 🔥 MELHORA: Usa o mesmo batch_size do código que funciona
    loader = DataLoader(
        trainer_self.eval_dataset,
        batch_size=trainer_self.args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=trainer_self.data_collator,
        drop_last=False,  # 🔥 ADICIONA: Garante que não perde exemplos
    )
    
    losses = []
    
    # 🔥 ADICIONA: Desabilita autocast se estava ativo
    with torch.cuda.amp.autocast(enabled=False):
        with torch.no_grad():
            for i, batch in enumerate(loader):
                try:
                    # 🔥 MELHORA: Move para device igual ao seu código
                    batch = {k: v.to(trainer_self.model.device) for k, v in batch.items()}
                    
                    # 🔥 ADICIONA: Verifica se batch tem conteúdo válido
                    if (batch['labels'] != -100).sum() == 0:
                        print(batch)
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
    
    # 🔥 ADICIONA: Restaura estado original
    if training_mode:
        trainer_self.model.train()
    
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
    
    # Retorna no formato esperado pelo Trainer
    metrics = {
        f"{metric_key_prefix}_loss_safe": mean_loss,
        f"{metric_key_prefix}_ppl_safe": ppl,
        f"{metric_key_prefix}_valid_batches": len(losses),
    }
    
    # Loga as métricas no W&B
    trainer_self.log(metrics)
    
    # Notifica callbacks (importante para load_best_model_at_end)
    trainer_self.control = trainer_self.callback_handler.on_evaluate(
        trainer_self.args, trainer_self.state, trainer_self.control, metrics=metrics
    )
    
    return metrics

#from trl import SFTTrainer
#import types
#
## ======== CONFIGURAÇÃO ========
## Importa os objetos do seu pipeline principal
##from src.finetune import model, tokenizer, val_dataset, data_collator, custom_evaluate
#
#device = torch.device("cuda:0")  # força GPU
#model.to(device)
#model.eval()
#print(f"✅ Usando device: {device}")
#
## ======== DATA LOADER ========
#dl = DataLoader(
#    val_dataset,
#    batch_size=2,
#    shuffle=False,
#    collate_fn=data_collator,
#)
#
#print("🔍 Verificando labels diretamente do DataLoader...")
#for i, batch in enumerate(dl):
#    num_valid = (batch["labels"] != -100).sum().item()
#    num_total = batch["labels"].numel()
#    if num_valid == 0:
#        print(f"⚠️ DL Batch {i}: todos labels = -100")
#    elif num_valid / num_total < 0.05:
#        print(f"⚠️ DL Batch {i}: apenas {num_valid}/{num_total} labels válidos ({100*num_valid/num_total:.2f}%)")
#
#print("✅ DataLoader check concluído\n")
#
## ======== TRAINER ========
#training_args = TrainingArguments(
#    output_dir="./tmp_eval_diag",
#    per_device_eval_batch_size=2,
#    bf16=False,
#    fp16=False,
#    dataloader_num_workers=0,
#    report_to=[],
#    logging_dir="./logs_tmp",
#    no_cuda=False,  # permite GPU
#    save_strategy="no",
#    eval_strategy="no"
#)
#
#from transformers import Trainer
#dummy_train = val_dataset.select([0])
#trainer = Trainer(
#    model=model,
#    processing_class=tokenizer,
#    train_dataset=dummy_train,
#    eval_dataset=val_dataset,
#    args=training_args,
#    data_collator=data_collator,
#)
#
## substitui evaluate pelo custom
#trainer.evaluate = types.MethodType(custom_evaluate, trainer)
#
#print("🔍 Rodando avaliação via Trainer...")
#metrics = trainer.evaluate()
#
#print("\n===== RESULTADOS =====")
#print(metrics)
#    
#
## ======================
## 9. Argumentos de treino
## ======================
training_args = TrainingArguments(
    output_dir="./qwen-finetuned-chat2",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    #fp16_full_eval=False,
    bf16_full_eval=False,
    logging_dir="./logs",
    logging_steps=20,
    save_strategy="steps",
    save_steps=4000,
    eval_strategy="steps",
    eval_steps=2000,    
    eval_accumulation_steps=2,
    prediction_loss_only=True,
    report_to="wandb",
    optim="paged_adamw_8bit",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss_safe",  # 👈 EXPLÍCITO: salva baseado no loss
    greater_is_better=False,  # 👈 Menor loss é melhor
    save_total_limit=2,  # 👈 Mantém apenas os 2 melhores checkpoints
    warmup_steps=100,
    lr_scheduler_type = "cosine",
    # OTIMIZAÇÕES DE MEMÓRIA
    gradient_checkpointing=True,  # Ativa gradient checkpointing
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # Configurações para múltiplas GPUs
    #ddp_find_unused_parameters=False,  # Otimização DDP
    dataloader_num_workers=2,  # Paraleliza carregamento de dados
    #dataloader_pin_memory=True,  # Acelera transferência GPU
    # Remove exemplos muito longos
    remove_unused_columns=False,
)

## ======================
## 10. Trainer com métricas
## ======================
import types
trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args,
    data_collator=data_collator,
)
trainer.evaluate = types.MethodType(custom_evaluate, trainer)
#
## ======================
## 11. Iniciar treino
## ======================
print("🚀 Iniciando treino...")
trainer.train()

## ======================
## 10. Salvar modelo final
## ======================
print("💾 Salvando modelo...")
trainer.model.save_pretrained("./qwen-finetuned-lora-trainer")
tokenizer.save_pretrained("./qwen-finetuned-lora-trainer")

print("✅ Treino completo!")
print(f"📁 Modelo salvo em: ./qwen-finetuned-lora-trainer")

#from torch.utils.data import DataLoader
#print("\n🔎 Iniciando verificação de eval_loss (modo float32, sem quantização)...")
#
## 1️⃣ Coloca o modelo em modo float32 e eval
#model.eval()
#model.bfloat16()  # força precisão total
#torch.cuda.empty_cache()
#
## 2️⃣ Define DataLoader de validação
#safe_loader = DataLoader(
#    val_dataset,
#    batch_size=2,
#    shuffle=False,
#    collate_fn=data_collator
#)
#
## 3️⃣ Loop manual para detectar NaN
#nan_batches = []
#with torch.no_grad():
#    for i, batch in enumerate(safe_loader):
#        try:
#            batch = {k: v.to("cuda") for k, v in batch.items()}
#            outputs = model(**batch)
#            loss = outputs.loss
#
#            if torch.isnan(loss):
#                print(f"🚨 NaN detectado no batch {i}")
#                nan_batches.append(i)
#                break  # interrompe pra inspecionar
#            else:
#                if i % 50 == 0:
#                    print(f"✅ Batch {i}: loss = {loss.item():.4f}")
#
#        except Exception as e:
#            print(f"❌ Erro no batch {i}: {type(e).__name__} - {e}")
#            nan_batches.append(i)
#            break
#
## 4️⃣ Mostra resultados
#if not nan_batches:
#    print("\n✅ Nenhum NaN encontrado nos batches de validação.")
#else:
#    idx = nan_batches[0]
#    print(f"\n🚨 Problema detectado no batch {idx}. Inspecionando exemplo...\n")
#    # Mostra o primeiro exemplo do batch problemático
#    start = idx * 2
#    end = start + 2
#    for j in range(start, min(end, len(val_dataset))):
#        print(f"--- Exemplo {j} ---")
#        ex = val_dataset[j]
#        print(f"input_ids len: {len(ex['input_ids'])}")
#        print(f"labels len: {len(ex['labels'])}")
#        print(f"Qtd labels != -100: {sum(l != -100 for l in ex['labels'])}")
#        print(f"Texto original (primeiros 300 chars):")
#        decoded = tokenizer.decode([id for id in ex['input_ids'] if id != tokenizer.pad_token_id])
#        print(decoded[:300], "\n")
