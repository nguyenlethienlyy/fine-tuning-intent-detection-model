"""
scripts/inference.py
-------------------
Inference using trained intent classification model.
"""

import torch

from scripts.common import (
    load_classifier,
    load_config,
    load_id2label,
    parse_config_path,
    tokenize,
)


class IntentClassification:
    def __init__(self, config_path):
        config = load_config(config_path)

        self.model_path = config["model_path"]
        self.label_map_path = config["label_map_path"]
        self.max_length = config.get("max_length", 64)

        self.tokenizer, self.model = load_classifier(self.model_path, eval_mode=True)
        self.id2label = load_id2label(self.label_map_path)

    def __call__(self, message: str):
        inputs = tokenize(
            self.tokenizer,
            message,
            self.max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        pred_id = torch.argmax(outputs.logits, dim=1).item()

        return self.id2label[pred_id]


if __name__ == "__main__":
    classifier = IntentClassification(parse_config_path("configs/inference.yaml"))

    example_message = "My card has not arrived yet."
    print("Input:", example_message)
    print("Predicted intent:", classifier(example_message))
