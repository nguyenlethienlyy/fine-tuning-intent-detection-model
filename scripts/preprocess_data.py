"""
scripts/preprocess_data.py
"""

import os
from collections import Counter, defaultdict
import pandas as pd
from datasets import load_dataset

OUTPUT_DIR = "sample_data"
STATS_FILE = "stats.txt"
NUM_INTENTS = 10
SAMPLE_PER_INTENT_TRAIN = 100
SAMPLE_PER_INTENT_TEST = 20

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #1. load data
    dataset = load_dataset("banking77")
    train_data = dataset["train"]
    test_data = dataset["test"]

    #2. statistics
    def statistic(train, test):
        train_labels = train["label"]
        test_labels = test["label"]

        train_counter = Counter(train_labels)
        test_counter = Counter(test_labels)

        stats_path = os.path.join(OUTPUT_DIR, STATS_FILE)
        with open(stats_path, "a") as f:
            f.write(f"Train size: {len(train)}\n")
            f.write(f"Test size: {len(test)}\n")
            f.write(f"Number of intents: {len(train_counter)}\n\n")

            f.write("Train distribution:\n")
            for k, v in sorted(train_counter.items()):
                f.write(f"{k}: {v}\n")

            f.write("\nTest distribution:\n")
            for k, v in sorted(test_counter.items()):
                f.write(f"{k}: {v}\n")
        print(f"Saved statistics to {os.path.join(OUTPUT_DIR, STATS_FILE)}")

    statistic(train_data, test_data)

    #3. select intents 0-9
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

    #4. convert to dataframe
    def to_df(data):
        return pd.DataFrame({
            "text": [x["text"] for x in data],
            "label": [x["label"] for x in data]
        })
    
    df_train = to_df(train_subset)
    df_test = to_df(test_subset)

    #5. save
    df_train.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
    df_test.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)

    print("Saved train.csv and test.csv")
    print("Train size:", len(df_train))
    print("Test size:", len(df_test))


if __name__ == "__main__":
    main()