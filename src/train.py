import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb

# Load processed dataset
dataset = load_from_disk("./processed_dataset")

# Model & tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# BitsAndBytes quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

# LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.print_trainable_parameters()

# TrainingArguments
training_args = TrainingArguments(
    output_dir="./llama_mathcot_qlora",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    bf16=True,
    fp16=False,
    optim="paged_adamw_8bit",
    evaluation_strategy="steps",
    eval_steps=200,
    eval_on_start=True,
    logging_steps=50,
    save_strategy="steps",
    save_steps=200,
    report_to="wandb",
    run_name="llama_mathcot_qlora",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# Trainer
wandb.login()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
)

trainer.train()

model.save_pretrained("./llama_mathcot_qlora_adapter")
print("Saved LoRA adapter ./llama_mathcot_qlora_adapter")
