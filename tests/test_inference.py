import json

import pytest

from scripts import inference


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, message, **kwargs):
        self.calls.append((message, kwargs))
        return {"input_ids": [[1, 2, 3]]}


class FakeModel:
    def __init__(self, rows):
        self.rows = rows
        self.eval_called = False
        self.forward_kwargs = None

    def eval(self):
        self.eval_called = True

    def __call__(self, **kwargs):
        self.forward_kwargs = kwargs
        return type("Output", (), {"logits": FakeLogits(self.rows)})()


class FakeLogits:
    def __init__(self, rows):
        self.rows = rows


@pytest.fixture
def build_classifier(tmp_path, monkeypatch):
    """Builds an IntentClassification with the tokenizer/model layers faked out."""

    def _build(logits_rows=((0.1, 0.9),), label_map=None, config_extra="max_length: 32\n"):
        label_map = {"card_arrival": 0, "card_linking": 1} if label_map is None else label_map

        label_map_path = tmp_path / "label_map.json"
        label_map_path.write_text(json.dumps(label_map))

        config_path = tmp_path / "inference.yaml"
        config_path.write_text(
            f"model_path: {tmp_path / 'checkpoint'}\n"
            f"label_map_path: {label_map_path}\n"
            f"{config_extra}"
        )

        tokenizer = FakeTokenizer()
        model = FakeModel([list(row) for row in logits_rows])
        loaded = {}

        def fake_tokenizer_from_pretrained(path):
            loaded["tokenizer_path"] = path
            return tokenizer

        def fake_model_from_pretrained(path):
            loaded["model_path"] = path
            return model

        monkeypatch.setattr(
            inference,
            "AutoTokenizer",
            type("AutoTokenizer", (), {"from_pretrained": fake_tokenizer_from_pretrained}),
        )
        monkeypatch.setattr(
            inference,
            "AutoModelForSequenceClassification",
            type(
                "AutoModelForSequenceClassification",
                (),
                {"from_pretrained": fake_model_from_pretrained},
            ),
        )

        classifier = inference.IntentClassification(str(config_path))
        return classifier, tokenizer, model, loaded

    return _build


def test_init_loads_config_and_artifacts(build_classifier, tmp_path):
    classifier, _, model, loaded = build_classifier()

    assert classifier.model_path == str(tmp_path / "checkpoint")
    assert classifier.max_length == 32
    assert loaded["tokenizer_path"] == str(tmp_path / "checkpoint")
    assert loaded["model_path"] == str(tmp_path / "checkpoint")
    assert model.eval_called is True


def test_init_defaults_max_length_when_absent(build_classifier):
    classifier, _, _, _ = build_classifier(config_extra="")

    assert classifier.max_length == 64


def test_init_inverts_label_map(build_classifier):
    classifier, _, _, _ = build_classifier(
        label_map={"card_arrival": 0, "card_linking": 1, "atm_support": 2}
    )

    assert classifier.id2label == {0: "card_arrival", 1: "card_linking", 2: "atm_support"}


def test_init_raises_on_missing_required_config_key(tmp_path):
    config_path = tmp_path / "inference.yaml"
    config_path.write_text("label_map_path: whatever.json\n")

    with pytest.raises(KeyError):
        inference.IntentClassification(str(config_path))


def test_call_returns_label_of_highest_logit(build_classifier):
    classifier, _, _, _ = build_classifier(logits_rows=((0.1, 0.9),))
    assert classifier("my card has not arrived") == "card_linking"

    classifier, _, _, _ = build_classifier(logits_rows=((2.5, -1.0),))
    assert classifier("my card has not arrived") == "card_arrival"


def test_call_passes_tokenizer_arguments(build_classifier):
    classifier, tokenizer, model, _ = build_classifier()

    classifier("where is my card?")

    message, kwargs = tokenizer.calls[0]
    assert message == "where is my card?"
    assert kwargs == {
        "return_tensors": "pt",
        "truncation": True,
        "padding": True,
        "max_length": 32,
    }
    assert model.forward_kwargs == {"input_ids": [[1, 2, 3]]}


def test_call_raises_when_prediction_id_missing_from_label_map(build_classifier):
    classifier, _, _, _ = build_classifier(
        logits_rows=((0.1, 0.2, 5.0),), label_map={"card_arrival": 0, "card_linking": 1}
    )

    with pytest.raises(KeyError):
        classifier("unmapped class")
