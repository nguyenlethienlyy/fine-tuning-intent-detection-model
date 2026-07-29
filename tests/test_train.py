from pathlib import Path

import pandas as pd
import pytest

from scripts import train


def test_format_prompt_contains_instruction_input_and_response():
    result = train.format_prompt({"text": "card lost", "label": 3})

    assert result == {
        "text_formatted": (
            "### Instruction:\nIdentify the intent label for the query.\n\n"
            "### Input:\ncard lost\n\n"
            "### Response:\n3"
        )
    }


def test_format_prompt_only_returns_formatted_field():
    assert list(train.format_prompt({"text": "a", "label": 0})) == ["text_formatted"]


def test_format_prompt_requires_text_and_label():
    with pytest.raises(KeyError):
        train.format_prompt({"text": "no label"})


def test_load_config_reads_yaml(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text("paths:\n  train_data: sample_data/train.csv\n")

    assert train.load_config(str(config_path)) == {
        "paths": {"train_data": "sample_data/train.csv"}
    }


def test_shipped_config_has_keys_used_by_main():
    config_path = Path(__file__).resolve().parents[1] / train.DEFAULT_CONFIG_PATH
    config = train.load_config(str(config_path))

    assert {"train_data", "output_dir"} <= set(config["paths"])
    assert {"name", "max_seq_length", "load_in_4bit"} <= set(config["model"])
    assert {"r", "target_modules", "lora_alpha", "lora_dropout"} <= set(config["lora"])
    assert {
        "batch_size",
        "gradient_accumulation_steps",
        "warmup_steps",
        "epochs",
        "learning_rate",
        "optimizer",
        "weight_decay",
    } <= set(config["training"])


class FakeTrainer:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.trained = False
        FakeTrainer.instances.append(self)

    def train(self):
        self.trained = True


def test_main_wires_config_into_model_lora_trainer_and_saves(tmp_path, monkeypatch):
    train_csv = tmp_path / "train.csv"
    pd.DataFrame({"text": ["card lost", "atm broken"], "label": [0, 1]}).to_csv(
        train_csv, index=False
    )
    output_dir = tmp_path / "checkpoint"

    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "paths:\n"
        f"  train_data: {train_csv}\n"
        f"  output_dir: {output_dir}\n"
        "model:\n"
        "  name: unsloth/llama-3-8b-bnb-4bit\n"
        "  max_seq_length: 128\n"
        "  load_in_4bit: true\n"
        "lora:\n"
        "  r: 8\n"
        "  target_modules: [q_proj, v_proj]\n"
        "  lora_alpha: 16\n"
        "  lora_dropout: 0.05\n"
        "training:\n"
        "  batch_size: 2\n"
        "  gradient_accumulation_steps: 4\n"
        "  warmup_steps: 5\n"
        "  epochs: 1\n"
        "  learning_rate: 0.0002\n"
        "  optimizer: adamw_8bit\n"
        "  weight_decay: 0.01\n"
    )

    calls = {"empty_cache": 0, "saved": []}
    mapped_rows = []

    class FakeDataset:
        def __init__(self, rows):
            self.rows = rows

        @staticmethod
        def from_pandas(df):
            return FakeDataset(df.to_dict("records"))

        def map(self, fn):
            mapped_rows.extend(fn(row) for row in self.rows)
            return self

    class FakeModel:
        def save_pretrained(self, path):
            calls["saved"].append(("model", path))

    class FakeTokenizer:
        def save_pretrained(self, path):
            calls["saved"].append(("tokenizer", path))

    fake_model = FakeModel()
    fake_peft_model = FakeModel()
    fake_tokenizer = FakeTokenizer()

    class FakeFastLanguageModel:
        from_pretrained_kwargs = None
        peft_kwargs = None

        @classmethod
        def from_pretrained(cls, **kwargs):
            cls.from_pretrained_kwargs = kwargs
            return fake_model, fake_tokenizer

        @classmethod
        def get_peft_model(cls, model, **kwargs):
            cls.peft_kwargs = dict(kwargs, model=model)
            return fake_peft_model

    FakeTrainer.instances = []

    monkeypatch.setattr(train, "Dataset", FakeDataset)
    monkeypatch.setattr(train, "FastLanguageModel", FakeFastLanguageModel)
    monkeypatch.setattr(train, "is_bfloat16_supported", lambda: False)
    monkeypatch.setattr(train, "SFTTrainer", FakeTrainer)
    monkeypatch.setattr(train, "TrainingArguments", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        train.torch.cuda,
        "empty_cache",
        lambda: calls.__setitem__("empty_cache", calls["empty_cache"] + 1),
    )

    train.main(str(config_path))

    assert calls["empty_cache"] == 1
    assert [row["text_formatted"].endswith("### Response:\n0") for row in mapped_rows][0]
    assert len(mapped_rows) == 2

    assert FakeFastLanguageModel.from_pretrained_kwargs == {
        "model_name": "unsloth/llama-3-8b-bnb-4bit",
        "max_seq_length": 128,
        "dtype": None,
        "load_in_4bit": True,
    }
    assert FakeFastLanguageModel.peft_kwargs["model"] is fake_model
    assert FakeFastLanguageModel.peft_kwargs["r"] == 8
    assert FakeFastLanguageModel.peft_kwargs["target_modules"] == ["q_proj", "v_proj"]
    assert FakeFastLanguageModel.peft_kwargs["lora_alpha"] == 16
    assert FakeFastLanguageModel.peft_kwargs["lora_dropout"] == 0.05

    trainer = FakeTrainer.instances[0]
    assert trainer.trained is True
    assert trainer.kwargs["model"] is fake_peft_model
    assert trainer.kwargs["dataset_text_field"] == "text_formatted"
    assert trainer.kwargs["max_seq_length"] == 128
    args = trainer.kwargs["args"]
    assert args["per_device_train_batch_size"] == 2
    assert args["gradient_accumulation_steps"] == 4
    assert args["warmup_steps"] == 5
    assert args["num_train_epochs"] == 1
    assert args["learning_rate"] == 0.0002
    assert args["optim"] == "adamw_8bit"
    assert args["weight_decay"] == 0.01
    assert (args["fp16"], args["bf16"]) == (True, False)

    assert output_dir.is_dir()
    assert calls["saved"] == [("model", str(output_dir)), ("tokenizer", str(output_dir))]
