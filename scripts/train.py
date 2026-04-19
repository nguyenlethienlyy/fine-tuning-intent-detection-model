"""
scripts/train.py
----------------
Train intent classification using Unsloth (LLM fine-tuning).
"""

import yaml
import json
import pandas as pd
from datasets import Dataset

from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer


# ================= LOAD CONFIG =================
with open("configs/train.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["model"]["name"]
OUTPUT_DIR = config["output"]["dir"]

TRAIN_PATH = config["data"]["train_path"]
TEST_PATH = config["data"]["test_path"]
LABEL_MAP_PATH = config["data"]["label_map_path"]

MAX_SEQ_LENGTH = config["training"]["max_seq_length"]
BATCH_SIZE = config["training"]["batch_size"]
EPOCHS = config["training"]["epochs"]


# ================= LOAD DATA =================
print("Loading data...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

id2label = {v: k for k, v in label_map.items()}

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


# ================= FORMAT PROMPT =================
def format_prompt(example):
    text = example["text"]
    label_id = example["label"]
    label_name = id2label[label_id]

    return {
        "text": f"""### Instruction:
Classify the banking intent

### Input:
{text}

### Response:
{label_name}"""
    }


train_dataset = train_dataset.map(format_prompt)
test_dataset = test_dataset.map(format_prompt)


# ================= LOAD MODEL =================
print("Loading model...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

# Enable LoRA (PEFT)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
)


# ================= TRAINING CONFIG =================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,

    num_train_epochs=EPOCHS,

    learning_rate=2e-4,  # LLM cần LR cao hơn BERT
    fp16=True,

    logging_steps=10,
    save_strategy="epoch",

    report_to="none",
)


# ================= TRAINER =================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,

    train_dataset=train_dataset,
    eval_dataset=test_dataset,

    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,

    args=training_args,
)


# ================= TRAIN =================
print("Training...")
trainer.train()


# ================= SAVE =================
print("Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Done.")