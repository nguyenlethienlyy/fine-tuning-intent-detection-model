"""Shared test fixtures.

The scripts under test import heavy ML dependencies (torch, transformers,
datasets, trl, unsloth) that need a GPU-sized environment. The unit tests only
exercise the project's own logic, so those dependencies are replaced by minimal
stand-ins registered in ``sys.modules`` before the scripts are imported.
"""

import sys
import types
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


def _make_torch_stub():
    torch = types.ModuleType("torch")

    @contextmanager
    def no_grad():
        yield

    def argmax(tensor, dim=None):
        """Expects any object exposing ``rows`` as a 2D sequence of scores."""
        if dim != 1:
            raise AssertionError(f"unexpected dim: {dim}")
        row = tensor.rows[0]
        return FakeScalar(max(range(len(row)), key=row.__getitem__))

    cuda = types.SimpleNamespace(empty_cache=lambda: None)

    torch.no_grad = no_grad
    torch.argmax = argmax
    torch.cuda = cuda
    return torch


def _make_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _install_stubs():
    class _Unavailable:
        """Placeholder that fails loudly if a test relies on the real thing."""

        def __init__(self, name):
            self._name = name

        def __call__(self, *args, **kwargs):
            raise AssertionError(f"{self._name} must be patched by the test")

        def __getattr__(self, attr):
            return _Unavailable(f"{self._name}.{attr}")

    stubs = {
        "torch": _make_torch_stub(),
        "transformers": _make_module(
            "transformers",
            AutoTokenizer=_Unavailable("transformers.AutoTokenizer"),
            AutoModelForSequenceClassification=_Unavailable(
                "transformers.AutoModelForSequenceClassification"
            ),
            TrainingArguments=_Unavailable("transformers.TrainingArguments"),
        ),
        "datasets": _make_module(
            "datasets",
            load_dataset=_Unavailable("datasets.load_dataset"),
            Dataset=_Unavailable("datasets.Dataset"),
        ),
        "trl": _make_module("trl", SFTTrainer=_Unavailable("trl.SFTTrainer")),
        "unsloth": _make_module(
            "unsloth",
            FastLanguageModel=_Unavailable("unsloth.FastLanguageModel"),
            is_bfloat16_supported=_Unavailable("unsloth.is_bfloat16_supported"),
        ),
    }

    for name, module in stubs.items():
        sys.modules.setdefault(name, module)


_install_stubs()
