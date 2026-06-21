from __future__ import annotations

import os

from tools.qwen_controller_runtime_env import configure_pytorch_only_transformers_env


def test_configure_pytorch_only_transformers_env_sets_pytorch_only_flags(monkeypatch) -> None:
    monkeypatch.delenv("USE_TF", raising=False)
    monkeypatch.delenv("TRANSFORMERS_NO_TF", raising=False)

    report = configure_pytorch_only_transformers_env()

    assert os.environ["USE_TF"] == "0"
    assert os.environ["TRANSFORMERS_NO_TF"] == "1"
    assert report == {"USE_TF": "0", "TRANSFORMERS_NO_TF": "1"}


def test_configure_pytorch_only_transformers_env_forces_pytorch_only_flags(monkeypatch) -> None:
    monkeypatch.setenv("USE_TF", "1")
    monkeypatch.setenv("TRANSFORMERS_NO_TF", "0")

    report = configure_pytorch_only_transformers_env()

    assert os.environ["USE_TF"] == "0"
    assert os.environ["TRANSFORMERS_NO_TF"] == "1"
    assert report == {"USE_TF": "0", "TRANSFORMERS_NO_TF": "1"}
