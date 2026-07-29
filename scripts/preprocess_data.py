import os
import json
import yaml
from collections import Counter, defaultdict
import pandas as pd
from datasets import load_dataset


DEFAULT_CONFIG_PATH = "configs/preprocess_data.yaml"


def load_config(config_path=DEFAULT_CONFIG_PATH):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_label_map(label_names):
    return {name: idx for idx, name in enumerate(label_names)}


def save_label_map(label_map, path):
    with open(path, "w") as f:
        json.dump(label_map, f, indent=2)


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


def filter_data(data, samples, selected_labels):
    grouped = defaultdict(list)

    for ex in data:
        if ex["label"] in selected_labels:
            grouped[ex["label"]].append(ex)

    result = []
    for label in selected_labels:
        result.extend(grouped[label][:samples])

    return result


def to_df(data):
    return pd.DataFrame({
        "text": [x["text"] for x in data],
        "label": [x["label"] for x in data]
    })


def main(config_path=DEFAULT_CONFIG_PATH):
    config = load_config(config_path)

    dataset_name = config["data"]["dataset_name"]
    output_dir = config["output"]["output_dir"]
    stats_file = config["output"]["stats_file"]
    label_map_file = config["output"]["label_map_file"]
    num_intents = config["sampling"]["num_intents"]
    sample_per_intent_train = config["sampling"]["sample_per_intent_train"]
    sample_per_intent_test = config["sampling"]["sample_per_intent_test"]

    os.makedirs(output_dir, exist_ok=True)

    # 1. load data
    dataset = load_dataset(dataset_name)
    train_data = dataset["train"]
    test_data = dataset["test"]

    # 2. label_map
    label_map = build_label_map(train_data.features["label"].names)
    save_label_map(label_map, os.path.join(output_dir, label_map_file))
    print("Saved label_map.json")

    # 3. statistics
    write_statistics(train_data, test_data, os.path.join(output_dir, stats_file))

    # 4. select intents
    selected_labels = list(range(num_intents))
    train_subset = filter_data(train_data, sample_per_intent_train, selected_labels)
    test_subset = filter_data(test_data, sample_per_intent_test, selected_labels)

    # 5. to dataframe
    df_train = to_df(train_subset)
    df_test = to_df(test_subset)

    # 6. save
    df_train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print("Saved train.csv and test.csv")
    print("Train size:", len(df_train))
    print("Test size:", len(df_test))


if __name__ == "__main__":
    main()
