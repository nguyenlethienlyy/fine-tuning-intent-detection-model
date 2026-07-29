"""Shared helpers used by the preprocessing, training and inference scripts."""

import argparse
import json
import os

import pandas as pd
import yaml
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_config_path(default):
    """Return the config path from ``--config``, falling back to ``default``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default)
    args, _ = parser.parse_known_args()
    return args.config


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved {path}")


def load_label_map(path):
    """Return the ``label name -> id`` mapping stored at ``path``."""
    return load_json(path)


def load_id2label(path):
    """Return the ``id -> label name`` mapping stored at ``path``."""
    return {idx: name for name, idx in load_label_map(path).items()}


def load_csv_dataframe(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def load_csv_dataset(path):
    return Dataset.from_pandas(load_csv_dataframe(path))


def tokenize(tokenizer, texts, max_length, padding=True, return_tensors=None):
    return tokenizer(
        texts,
        truncation=True,
        padding=padding,
        max_length=max_length,
        return_tensors=return_tensors,
    )


def load_classifier(model_path, num_labels=None, eval_mode=False):
    """Load a sequence classification model and its tokenizer from ``model_path``."""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    kwargs = {} if num_labels is None else {"num_labels": num_labels}
    model = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)
    if eval_mode:
        model.eval()
    return tokenizer, model


def save_checkpoint(output_dir, tokenizer, model=None, trainer=None):
    """Persist a tokenizer plus either a raw ``model`` or a ``trainer``'s model."""
    os.makedirs(output_dir, exist_ok=True)
    if trainer is not None:
        trainer.save_model(output_dir)
    elif model is not None:
        model.save_pretrained(output_dir)
    else:
        raise ValueError("Either model or trainer must be provided")
    tokenizer.save_pretrained(output_dir)
    print(f"Saved checkpoint to {output_dir}")
