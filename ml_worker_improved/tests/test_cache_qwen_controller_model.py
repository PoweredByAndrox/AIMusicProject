from __future__ import annotations

import json
import os
from pathlib import Path

import tools.cache_qwen_controller_model as cache_qwen
from tools.cache_qwen_controller_model import main, run_cache_qwen_controller_model


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_snapshot(tmp_path: Path) -> Path:
    snapshot = tmp_path / "hf" / "hub" / "snapshots" / "unit"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "model.safetensors").write_text("weights\n", encoding="utf-8")
    return snapshot


def test_cache_blocks_without_network_when_snapshot_missing(monkeypatch, tmp_path: Path) -> None:
    def fail_snapshot_download(**kwargs):
        raise RuntimeError("offline cache miss")

    monkeypatch.setattr(cache_qwen, "snapshot_download", fail_snapshot_download)

    manifest = run_cache_qwen_controller_model(
        output_dir=tmp_path / "out",
        allow_network=False,
    )

    assert manifest["schema"] == "qwen_controller_model_cache_manifest.v1"
    assert manifest["cache_status"] == "blocked"
    assert "model_snapshot_not_cached" in manifest["failed_checks"]
    assert manifest["download_attempted"] is False
    assert manifest["training_started"] is False
    assert manifest["fine_tune_started"] is False
    assert manifest["models_changed"] is False
    assert manifest["model_registry_edited"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert Path(manifest["outputs"]["manifest_json"]).name == "manifest.json"
    assert Path(manifest["outputs"]["progress_jsonl"]).name == "progress.jsonl"
    progress_rows = read_jsonl(Path(manifest["outputs"]["progress_jsonl"]))
    assert [row["event"] for row in progress_rows] == [
        "start",
        "snapshot_download_start",
        "snapshot_download_failed",
        "snapshot_verified",
        "finish",
    ]
    assert progress_rows[1]["local_files_only"] is True
    assert progress_rows[1]["allow_network"] is False


def test_cache_allows_explicit_network_and_validates_snapshot(monkeypatch, tmp_path: Path) -> None:
    snapshot = make_snapshot(tmp_path)
    calls: list[dict] = []
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hf" / "hub"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            assert os.environ["HF_HUB_OFFLINE"] == "0"
        return str(snapshot)

    monkeypatch.setattr(cache_qwen, "snapshot_download", fake_snapshot_download)

    manifest = run_cache_qwen_controller_model(
        output_dir=tmp_path / "out",
        allow_network=True,
        revision="main",
        max_workers=1,
    )

    assert manifest["cache_status"] == "ready"
    assert manifest["failed_checks"] == []
    assert manifest["download_attempted"] is True
    assert manifest["snapshot"]["cache_available"] is True
    assert manifest["snapshot"]["snapshot_path"] == str(snapshot)
    assert manifest["snapshot"]["required_files"] == {"config_json": True, "tokenizer": True, "weights": True}
    assert manifest["network"]["hf_hub_offline_before"] == "1"
    assert manifest["network"]["hf_hub_offline_during"] == "0"
    assert calls[0]["repo_id"] == cache_qwen.TARGET_MODEL_ID
    assert calls[0]["local_files_only"] is False
    assert calls[0]["revision"] == "main"
    assert calls[0]["max_workers"] == 1
    assert calls[1]["repo_id"] == cache_qwen.TARGET_MODEL_ID
    assert calls[1]["local_files_only"] is True
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False


def test_cache_blocks_snapshot_missing_required_files(monkeypatch, tmp_path: Path) -> None:
    snapshot = tmp_path / "hf" / "hub" / "snapshots" / "incomplete"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hf" / "hub"))
    monkeypatch.setattr(cache_qwen, "snapshot_download", lambda **kwargs: str(snapshot))

    manifest = run_cache_qwen_controller_model(
        output_dir=tmp_path / "out",
        allow_network=True,
    )

    assert manifest["cache_status"] == "blocked"
    assert "model_snapshot_missing_required_files" in manifest["failed_checks"]
    assert manifest["snapshot"]["missing_required_files"] == ["tokenizer", "weights"]
    assert manifest["release_ready"] is False


def test_cache_blocks_snapshot_outside_hf_cache_root(monkeypatch, tmp_path: Path) -> None:
    snapshot = make_snapshot(tmp_path / "outside")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hf" / "hub"))
    monkeypatch.setattr(cache_qwen, "snapshot_download", lambda **kwargs: str(snapshot))

    manifest = run_cache_qwen_controller_model(
        output_dir=tmp_path / "out",
        allow_network=True,
    )

    assert manifest["cache_status"] == "blocked"
    assert "model_snapshot_outside_cache_root" in manifest["failed_checks"]
    assert manifest["snapshot"]["cache_root_allowed"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False


def test_cli_checks_docker_runtime_before_parsing_args(monkeypatch) -> None:
    calls = []

    def reject_non_docker_runtime(**kwargs):
        calls.append(("require_docker_runtime", kwargs))
        raise RuntimeError("Docker-only test sentinel")

    def fail_if_args_are_parsed(self, argv=None):
        raise AssertionError("parse_args must not run before Docker runtime validation")

    monkeypatch.setattr(cache_qwen, "require_docker_runtime", reject_non_docker_runtime)
    monkeypatch.setattr(cache_qwen.argparse.ArgumentParser, "parse_args", fail_if_args_are_parsed)

    try:
        main(["--output-dir", "/tmp/out"])
    except RuntimeError as exc:
        assert str(exc) == "Docker-only test sentinel"
    else:
        raise AssertionError("main must fail through Docker runtime guard")

    assert calls == [
        (
            "require_docker_runtime",
            {
                "tool_name": "cache_qwen_controller_model.py",
                "entrypoint_path": "/app/tools/cache_qwen_controller_model.py",
            },
        )
    ]
