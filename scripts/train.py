"""scripts/train.py"""

import argparse
import os
import sys

import pandas as pd
import torch
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

from config_utils import ConfigError, load_config, require, require_file

DEFAULT_CONFIG = "configs/train.yaml"
REQUIRED_COLUMNS = ("text", "label")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the intent detection model.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to the training config")
    return parser.parse_args()


def format_prompt(example):
    """Định dạng dữ liệu thành prompt."""
    prompt = f"### Instruction:\nIdentify the intent label for the query.\n\n### Input:\n{example['text']}\n\n### Response:\n{example['label']}"
    return {"text_formatted": prompt}


def load_train_dataset(train_data_path):
    require_file(train_data_path, "Training data")
    df_train = pd.read_csv(train_data_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df_train.columns]
    if missing:
        raise ValueError(
            f"Training data {train_data_path} is missing required column(s): {', '.join(missing)}"
        )
    if df_train.empty:
        raise ValueError(f"Training data {train_data_path} is empty")

    dataset = Dataset.from_pandas(df_train)
    return dataset.map(format_prompt)


def main():
    args = parse_args()
    config = load_config(args.config)

    train_data_path = require(config, ["paths", "train_data"], args.config)
    output_dir = require(config, ["paths", "output_dir"], args.config)
    max_seq_length = require(config, ["model", "max_seq_length"], args.config)

    torch.cuda.empty_cache()

    print("1. Loading dataset...")
    train_dataset = load_train_dataset(train_data_path)

    print("2. Loading base model (Unsloth 4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = require(config, ["model", "name"], args.config),
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = require(config, ["model", "load_in_4bit"], args.config),
    )

    print("3. Applying LoRA configuration...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = require(config, ["lora", "r"], args.config),
        target_modules = require(config, ["lora", "target_modules"], args.config),
        lora_alpha = require(config, ["lora", "lora_alpha"], args.config),
        lora_dropout = require(config, ["lora", "lora_dropout"], args.config),
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
            per_device_train_batch_size = require(config, ["training", "batch_size"], args.config),
            gradient_accumulation_steps = require(config, ["training", "gradient_accumulation_steps"], args.config),
            warmup_steps = require(config, ["training", "warmup_steps"], args.config),
            num_train_epochs = require(config, ["training", "epochs"], args.config),
            learning_rate = require(config, ["training", "learning_rate"], args.config),
            fp16 = not has_bf16,
            bf16 = has_bf16,
            optim = require(config, ["training", "optimizer"], args.config),
            weight_decay = require(config, ["training", "weight_decay"], args.config),
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            save_strategy = "no",
            report_to = "none",
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
    try:
        main()
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
