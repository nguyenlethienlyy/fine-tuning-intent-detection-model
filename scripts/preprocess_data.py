import os
import json
import yaml
from collections import Counter, defaultdict
import pandas as pd
from datasets import load_dataset


# ================= LOAD CONFIG =================
with open("configs/preprocess_data.yaml", "r") as f:
    config = yaml.safe_load(f)

DATASET_NAME = config["data"]["dataset_name"]

OUTPUT_DIR = config["output"]["output_dir"]
STATS_FILE = config["output"]["stats_file"]
LABEL_MAP_FILE = config["output"]["label_map_file"]

NUM_INTENTS = config["sampling"]["num_intents"]
SAMPLE_PER_INTENT_TRAIN = config["sampling"]["sample_per_intent_train"]
SAMPLE_PER_INTENT_TEST = config["sampling"]["sample_per_intent_test"]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. load data
    dataset = load_dataset(DATASET_NAME)
    train_data = dataset["train"]
    test_data = dataset["test"]

    print(dataset)
    print(dataset["train"][0])

    # 2. label_map
    label_names = train_data.features["label"].names
    label_map = {name: idx for idx, name in enumerate(label_names)}

    with open(os.path.join(OUTPUT_DIR, LABEL_MAP_FILE), "w") as f:
        json.dump(label_map, f, indent=2)

    print("Saved label_map.json")

    # 3. statistics
    def statistic(train, test):
        train_labels = train["label"]
        test_labels = test["label"]

        train_counter = Counter(train_labels)
        test_counter = Counter(test_labels)

        stats_path = os.path.join(OUTPUT_DIR, STATS_FILE)
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

    statistic(train_data, test_data)

    # 4. select intents
    selected_labels = list(range(NUM_INTENTS))

    def filter_data(data, samples):
        grouped = defaultdict(list)

        for ex in data:
            if ex["label"] in selected_labels:
                grouped[ex["label"]].append(ex)

        result = []
        for label in selected_labels:
            result.extend(grouped[label][:samples])

        return result

    train_subset = filter_data(train_data, SAMPLE_PER_INTENT_TRAIN)
    test_subset = filter_data(test_data, SAMPLE_PER_INTENT_TEST)

    # 5. to dataframe
    def to_df(data):
        return pd.DataFrame({
            "text": [x["text"] for x in data],
            "label": [x["label"] for x in data]
        })

    df_train = to_df(train_subset)
    df_test = to_df(test_subset)

    # 6. save
    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    print("Saved train.csv and test.csv")
    print("Train size:", len(df_train))
    print("Test size:", len(df_test))


if __name__ == "__main__":
    main()