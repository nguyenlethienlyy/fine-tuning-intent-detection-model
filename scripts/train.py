"""scripts/train.py"""

import yaml
import os
import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

# Dọn dẹp bộ nhớ đệm GPU
torch.cuda.empty_cache()

# ================= LOAD CONFIG =================
with open("configs/train.yaml", "r") as f:
    config = yaml.safe_load(f)

train_data_path = config["paths"]["train_data"]
output_dir = config["paths"]["output_dir"]
max_seq_length = config["model"]["max_seq_length"]

def format_prompt(example):
    """Định dạng dữ liệu thành prompt."""
    prompt = f"### Instruction:\nIdentify the intent label for the query.\n\n### Input:\n{example['text']}\n\n### Response:\n{example['label']}"
    return {"text_formatted": prompt}

def main():
    print("1. Loading dataset...")
    df_train = pd.read_csv(train_data_path)
    train_dataset = Dataset.from_pandas(df_train)
    train_dataset = train_dataset.map(format_prompt)

    print("2. Loading base model (Unsloth 4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config["model"]["name"],
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = config["model"]["load_in_4bit"],
    )

    print("3. Applying LoRA configuration...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = config["lora"]["r"],
        target_modules = config["lora"]["target_modules"],
        lora_alpha = config["lora"]["lora_alpha"],
        lora_dropout = config["lora"]["lora_dropout"], 
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    print("4. Setting up Trainer...")
    # Kiểm tra hỗ trợ bf16 (T4 thường trả về False -> dùng fp16)
    has_bf16 = is_bfloat16_supported()
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        dataset_text_field = "text_formatted",
        max_seq_length = max_seq_length,
        args = TrainingArguments(
            per_device_train_batch_size = config["training"]["batch_size"],
            gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"],
            warmup_steps = config["training"]["warmup_steps"],
            num_train_epochs = config["training"]["epochs"],
            learning_rate = config["training"]["learning_rate"],
            fp16 = not has_bf16,
            bf16 = has_bf16,
            optim = config["training"]["optimizer"],
            weight_decay = config["training"]["weight_decay"],
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            save_strategy = "no",
        ),
    )

    print("5. Starting fine-tuning...")
    trainer.train()

    print("6. Saving the model checkpoint...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved successfully to {output_dir}")

if __name__ == "__main__":
    main()
