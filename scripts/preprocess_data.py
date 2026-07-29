import argparse
import os
import json
import sys
from collections import Counter, defaultdict

import pandas as pd
from datasets import load_dataset

from config_utils import ConfigError, load_config, require

DEFAULT_CONFIG = "configs/preprocess_data.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare the BANKING77 subset.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to the preprocessing config")
    return parser.parse_args()


def write_statistics(train, test, stats_path):
    train_counter = Counter(train["label"])
    test_counter = Counter(test["label"])

    with open(stats_path, "w") as f:
        f.write(f"Train size: {len(train)}\n")
        f.write(f"Test size: {len(test)}\n")
        f.write(f"Number of intents: {len(train_counter)}\n\n")

        f.write("Train distribution:\n")
        for k, v in sorted(train_counter.items()):
            f.write(f"{k}: {v}\n")

        f.write("\nTest distribution:\n")
        for k, v in sorted(test_counter.items()):
            f.write(f"{k}: {v}\n")

    print(f"Saved statistics to {stats_path}")


def filter_data(data, selected_labels, samples):
    grouped = defaultdict(list)

    for ex in data:
        if ex["label"] in selected_labels:
            grouped[ex["label"]].append(ex)

    result = []
    for label in selected_labels:
        examples = grouped[label]
        if not examples:
            raise ValueError(f"No examples found for label id {label}")
        result.extend(examples[:samples])

    return result


def to_df(data):
    return pd.DataFrame({
        "text": [x["text"] for x in data],
        "label": [x["label"] for x in data],
    })


def main():
    args = parse_args()
    config = load_config(args.config)

    dataset_name = require(config, ["data", "dataset_name"], args.config)
    output_dir = require(config, ["output", "output_dir"], args.config)
    stats_file = require(config, ["output", "stats_file"], args.config)
    label_map_file = require(config, ["output", "label_map_file"], args.config)
    num_intents = require(config, ["sampling", "num_intents"], args.config)
    sample_per_intent_train = require(config, ["sampling", "sample_per_intent_train"], args.config)
    sample_per_intent_test = require(config, ["sampling", "sample_per_intent_test"], args.config)

    os.makedirs(output_dir, exist_ok=True)

    # 1. load data
    try:
        dataset = load_dataset(dataset_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to load dataset '{dataset_name}': {exc}") from exc

    for split in ("train", "test"):
        if split not in dataset:
            raise KeyError(f"Dataset '{dataset_name}' has no '{split}' split (found: {list(dataset)})")

    train_data = dataset["train"]
    test_data = dataset["test"]

    # 2. label_map
    label_names = train_data.features["label"].names
    if num_intents > len(label_names):
        raise ValueError(
            f"num_intents={num_intents} exceeds the {len(label_names)} intents available in '{dataset_name}'"
        )

    label_map = {name: idx for idx, name in enumerate(label_names)}

    label_map_path = os.path.join(output_dir, label_map_file)
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"Saved {label_map_path}")

    # 3. statistics
    write_statistics(train_data, test_data, os.path.join(output_dir, stats_file))

    # 4. select intents
    selected_labels = list(range(num_intents))
    train_subset = filter_data(train_data, selected_labels, sample_per_intent_train)
    test_subset = filter_data(test_data, selected_labels, sample_per_intent_test)

    # 5/6. to dataframe and save
    df_train = to_df(train_subset)
    df_test = to_df(test_subset)

    df_train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print("Saved train.csv and test.csv")
    print("Train size:", len(df_train))
    print("Test size:", len(df_test))


if __name__ == "__main__":
    try:
        main()
    except (ConfigError, FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
