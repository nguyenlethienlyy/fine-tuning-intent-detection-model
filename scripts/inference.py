"""
scripts/inference.py
-------------------
Inference using trained intent classification model.
"""

import torch
import json
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class IntentClassification:
    def __init__(self, config_path):
        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.model_path = config["model_path"]
        self.label_map_path = config["label_map_path"]
        self.max_length = config.get("max_length", 64)

        # Load tokenizer & model
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()

        # Load label map
        print("Loading label map...")
        with open(self.label_map_path, "r") as f:
            label_map = json.load(f)

        # id → label
        self.id2label = {v: k for k, v in label_map.items()}

    def __call__(self, message: str):
        # Tokenize
        inputs = self.tokenizer(
            message,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length
        )

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=1).item()

        return self.id2label[pred_id]


if __name__ == "__main__":
    classifier = IntentClassification("configs/inference.yaml")

    example_message = "My card has not arrived yet."
    print("Input:", example_message)
    print("Predicted intent:", classifier(example_message))