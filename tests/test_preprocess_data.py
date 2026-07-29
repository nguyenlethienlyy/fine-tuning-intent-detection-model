import json
from pathlib import Path

import pytest

from scripts import preprocess_data


def make_examples(spec):
    """spec maps label -> number of examples."""
    return [
        {"text": f"text-{label}-{i}", "label": label}
        for label, count in spec.items()
        for i in range(count)
    ]


class FakeSplit:
    """Minimal stand-in for a ``datasets.Dataset`` split."""

    def __init__(self, examples):
        self.examples = examples

    def __iter__(self):
        return iter(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, key):
        return [ex[key] for ex in self.examples]


def test_load_config_reads_yaml(tmp_path):
    config_path = tmp_path / "preprocess_data.yaml"
    config_path.write_text("data:\n  dataset_name: banking77\n")

    assert preprocess_data.load_config(str(config_path)) == {
        "data": {"dataset_name": "banking77"}
    }


def test_load_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        preprocess_data.load_config(str(tmp_path / "nope.yaml"))


def test_shipped_config_has_expected_sections():
    config_path = Path(__file__).resolve().parents[1] / preprocess_data.DEFAULT_CONFIG_PATH
    config = preprocess_data.load_config(str(config_path))

    assert config["data"]["dataset_name"]
    assert set(config["output"]) == {"output_dir", "stats_file", "label_map_file"}
    assert set(config["sampling"]) == {
        "num_intents",
        "sample_per_intent_train",
        "sample_per_intent_test",
    }


def test_build_label_map_uses_positional_ids():
    assert preprocess_data.build_label_map(["card_arrival", "card_linking"]) == {
        "card_arrival": 0,
        "card_linking": 1,
    }


def test_build_label_map_empty():
    assert preprocess_data.build_label_map([]) == {}


def test_save_label_map_writes_json(tmp_path):
    path = tmp_path / "label_map.json"
    preprocess_data.save_label_map({"a": 0, "b": 1}, str(path))

    assert json.loads(path.read_text()) == {"a": 0, "b": 1}


def test_write_statistics_reports_sizes_and_distributions(tmp_path):
    train = FakeSplit(make_examples({1: 2, 0: 3}))
    test = FakeSplit(make_examples({0: 1}))
    stats_path = tmp_path / "stats.txt"

    preprocess_data.write_statistics(train, test, str(stats_path))
    content = stats_path.read_text()

    assert "Train size: 5" in content
    assert "Test size: 1" in content
    assert "Number of intents: 2" in content
    train_section, test_section = content.split("Test distribution:")
    assert train_section.index("0: 3") < train_section.index("1: 2")
    assert test_section.strip() == "0: 1"


def test_filter_data_caps_samples_per_label_and_orders_by_label():
    data = make_examples({2: 5, 0: 5, 1: 5})

    result = preprocess_data.filter_data(data, samples=2, selected_labels=[0, 1])

    assert [ex["label"] for ex in result] == [0, 0, 1, 1]
    assert [ex["text"] for ex in result] == [
        "text-0-0",
        "text-0-1",
        "text-1-0",
        "text-1-1",
    ]


def test_filter_data_allows_labels_with_fewer_examples_than_requested():
    data = make_examples({0: 1, 1: 3})

    result = preprocess_data.filter_data(data, samples=2, selected_labels=[0, 1])

    assert [ex["label"] for ex in result] == [0, 1, 1]


def test_filter_data_ignores_unknown_labels():
    data = make_examples({5: 2})

    assert preprocess_data.filter_data(data, samples=2, selected_labels=[0]) == []


def test_to_df_keeps_columns_and_order():
    df = preprocess_data.to_df(make_examples({0: 2}))

    assert list(df.columns) == ["text", "label"]
    assert df["text"].tolist() == ["text-0-0", "text-0-1"]
    assert df["label"].tolist() == [0, 0]


def test_to_df_empty_input():
    df = preprocess_data.to_df([])

    assert list(df.columns) == ["text", "label"]
    assert df.empty


def test_main_end_to_end_with_fake_dataset(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    config_path = tmp_path / "preprocess_data.yaml"
    config_path.write_text(
        "data:\n"
        "  dataset_name: banking77\n"
        "output:\n"
        f"  output_dir: {output_dir}\n"
        "  stats_file: stats.txt\n"
        "  label_map_file: label_map.json\n"
        "sampling:\n"
        "  num_intents: 2\n"
        "  sample_per_intent_train: 2\n"
        "  sample_per_intent_test: 1\n"
    )

    train = FakeSplit(make_examples({0: 3, 1: 3, 2: 3}))
    test = FakeSplit(make_examples({0: 2, 1: 2, 2: 2}))
    train.features = {"label": type("Feat", (), {"names": ["a", "b", "c"]})()}

    requested = {}

    def fake_load_dataset(name):
        requested["name"] = name
        return {"train": train, "test": test}

    monkeypatch.setattr(preprocess_data, "load_dataset", fake_load_dataset)

    preprocess_data.main(str(config_path))

    assert requested["name"] == "banking77"
    assert json.loads((output_dir / "label_map.json").read_text()) == {
        "a": 0,
        "b": 1,
        "c": 2,
    }
    assert "Train size: 9" in (output_dir / "stats.txt").read_text()

    train_csv = (output_dir / "train.csv").read_text().splitlines()
    test_csv = (output_dir / "test.csv").read_text().splitlines()
    assert train_csv[0] == "text,label"
    assert len(train_csv) == 1 + 4  # 2 intents x 2 samples
    assert len(test_csv) == 1 + 2  # 2 intents x 1 sample
