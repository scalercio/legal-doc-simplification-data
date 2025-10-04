from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from unsloth.chat_templates import get_chat_template
import pandas as pd

max_seq_length = 2048
dtype = None
load_in_4bit = True

# Load datasets
train_df = pd.read_parquet("iudicium_textum_paraphrases_v2.parquet.final")
val_df = pd.read_parquet("tesemo_v2.parquet")
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# Get chat template for Qwen
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-2.5",  # Use Qwen's chat template
)

# Format dataset with conversation structure
def formatting_prompts_func(examples):
    conversations = []
    for original, paraphrase in zip(examples["original_text"], examples["paraphrase"]):
        convo = [
            {"role": "user", "content": f"Simplifique o seguinte texto:\n{original}"},
            {"role": "assistant", "content": paraphrase}
        ]
        conversations.append(convo)
    
    # Apply chat template to each conversation
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) 
             for convo in conversations]
    
    return {"text": texts}

# Apply formatting
train_dataset = train_dataset.map(
    formatting_prompts_func, 
    batched=True,
    remove_columns=train_dataset.column_names
)

val_dataset = val_dataset.map(
    formatting_prompts_func, 
    batched=True,
    remove_columns=val_dataset.column_names
)

# Train
from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
        eval_strategy = "steps",
        eval_steps = 20,
        save_steps = 30,
        save_total_limit = 2,
    ),
)

# Enable faster training
FastLanguageModel.for_training(model)

# Train the model
trainer_stats = trainer.train()

# Save the model
model.save_pretrained("qwen3_simplified_lora")
tokenizer.save_pretrained("qwen3_simplified_lora")

print("Training completed!")
print(f"Final loss: {trainer_stats.training_loss}")