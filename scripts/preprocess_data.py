import os
from collections import Counter, defaultdict

import pandas as pd
from datasets import load_dataset

from scripts.common import load_config, parse_config_path, save_json


def main():
    config = load_config(parse_config_path("configs/preprocess_data.yaml"))

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

    print(dataset)
    print(dataset["train"][0])

    # 2. label_map
    label_names = train_data.features["label"].names
    label_map = {name: idx for idx, name in enumerate(label_names)}
    save_json(label_map, os.path.join(output_dir, label_map_file))

    # 3. statistics
    def statistic(train, test):
        train_counter = Counter(train["label"])
        test_counter = Counter(test["label"])

        stats_path = os.path.join(output_dir, stats_file)
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
    selected_labels = list(range(num_intents))

    def filter_data(data, samples):
        grouped = defaultdict(list)

        for ex in data:
            if ex["label"] in selected_labels:
                grouped[ex["label"]].append(ex)

        result = []
        for label in selected_labels:
            result.extend(grouped[label][:samples])

        return result

    train_subset = filter_data(train_data, sample_per_intent_train)
    test_subset = filter_data(test_data, sample_per_intent_test)

    # 5. to dataframe
    def to_df(data):
        return pd.DataFrame({
            "text": [x["text"] for x in data],
            "label": [x["label"] for x in data]
        })

    df_train = to_df(train_subset)
    df_test = to_df(test_subset)

    # 6. save
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    print("Saved train.csv and test.csv")
    print("Train size:", len(df_train))
    print("Test size:", len(df_test))


if __name__ == "__main__":
    main()
