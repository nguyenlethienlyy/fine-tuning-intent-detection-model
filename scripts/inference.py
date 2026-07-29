"""
scripts/inference.py
-------------------
Inference using trained intent classification model.
"""

import argparse
import json
import sys

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from .config_utils import ConfigError, load_config, require, require_file
except ImportError:
    from config_utils import ConfigError, load_config, require, require_file

DEFAULT_CONFIG = "configs/inference.yaml"


class IntentClassification:
    def __init__(self, config_path):
        config = load_config(config_path)

        self.model_path = require(config, ["model_path"], config_path)
        self.label_map_path = require(config, ["label_map_path"], config_path)
        self.max_length = config.get("max_length", 64)

        # Load tokenizer & model
        print("Loading model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to load model from '{self.model_path}' (configured in {config_path}): {exc}"
            ) from exc
        self.model.eval()

        # Load label map
        print("Loading label map...")
        require_file(self.label_map_path, "Label map")
        try:
            with open(self.label_map_path, "r") as f:
                label_map = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in label map {self.label_map_path}: {exc}") from exc

        if not isinstance(label_map, dict) or not label_map:
            raise ValueError(f"Label map {self.label_map_path} must be a non-empty label -> id mapping")

        # id → label
        self.id2label = {v: k for k, v in label_map.items()}

    def __call__(self, message: str):
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Input message must be a non-empty string")

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

        if pred_id not in self.id2label:
            raise KeyError(
                f"Model predicted id {pred_id}, which is absent from label map {self.label_map_path}. "
                "The checkpoint and the label map are out of sync."
            )

        return self.id2label[pred_id]


def parse_args():
    parser = argparse.ArgumentParser(description="Run intent classification inference.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to the inference config")
    parser.add_argument("--message", default="My card has not arrived yet.", help="Message to classify")
    return parser.parse_args()


def main():
    args = parse_args()
    classifier = IntentClassification(args.config)

    print("Input:", args.message)
    print("Predicted intent:", classifier(args.message))


if __name__ == "__main__":
    try:
        main()
    except (ConfigError, FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
