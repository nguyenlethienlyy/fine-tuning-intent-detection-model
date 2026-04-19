"""scripts/train.py"""

import yaml
import os
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# ================= LOAD CONFIG =================
with open("configs/train.yaml", "r") as f:
    config = yaml.safe_load(f)

# Lấy các thiết lập từ file cấu hình
train_data_path = config["paths"]["train_data"]
output_dir = config["paths"]["output_dir"]
max_seq_length = config["model"]["max_seq_length"]

def format_prompt(example):
    """
    Hàm định dạng dữ liệu thành prompt để huấn luyện mô hình sinh văn bản.
    """
    prompt = f"""Below is a customer query for a bank. Identify the correct intent category label (represented as a number) for this query.

### Query:
{example['text']}

### Intent Label:
{example['label']}"""
    return {"text_formatted": prompt}

def main():
    print("1. Loading dataset...")
    df_train = pd.read_csv(train_data_path)
    train_dataset = Dataset.from_pandas(df_train)
    
    # Map data theo format prompt
    train_dataset = train_dataset.map(format_prompt)

    print("2. Loading base model via Unsloth...")
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
        use_rslora = False,
    )

    print("4. Setting up Trainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        dataset_text_field = "text_formatted",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Set False for classification prompts
        args = TrainingArguments(
            per_device_train_batch_size = config["training"]["batch_size"],
            gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"],
            warmup_steps = config["training"]["warmup_steps"],
            num_train_epochs = config["training"]["epochs"],
            learning_rate = config["training"]["learning_rate"],
            fp16 = True,
            bf16 = False,
            logging_steps = 1,
            optim = config["training"]["optimizer"],
            weight_decay = config["training"]["weight_decay"],
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )

    print("5. Starting fine-tuning...")
    trainer_stats = trainer.train()

    print("6. Saving the model checkpoint...")
    os.makedirs(output_dir, exist_ok=True)
    # Lưu mô hình LoRA và tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved successfully to {output_dir}")

if __name__ == "__main__":
    main()
