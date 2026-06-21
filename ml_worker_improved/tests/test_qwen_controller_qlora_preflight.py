from __future__ import annotations

import json
import sys
from pathlib import Path

import tools.qwen_controller_qlora_preflight as qwen_preflight
from tools.qwen_controller_qlora_preflight import main, run_qwen_controller_qlora_preflight


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sft_row(row_id: str, *, split: str, manifest_path: str, route_id: str = "route-a") -> dict:
    return {
        "schema": "qwen_controller_sft_example.v1",
        "id": row_id,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "{}"},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "accepted": True,
                        "action": "reuse_route_candidate",
                        "route_id": route_id,
                        "stem_type": "vocal",
                    },
                    sort_keys=True,
                ),
            },
        ],
        "metadata": {
            "split": split,
            "provenance": {
                "source_row_id": row_id,
                "track_id": f"track-{row_id}",
                "stem_type": "vocal",
                "route_id": route_id,
                "manifest_path": manifest_path,
            },
        },
        "delivery_allowed": False,
        "release_ready": False,
        "training_target": "qwen_controller_sft",
    }


def write_sft_dataset(
    tmp_path: Path,
    *,
    status: str = "ready",
    shared_eval_manifest: bool = False,
) -> Path:
    train_manifest = "/app/lab_artifacts/qwen/train/manifest.json"
    eval_manifest = train_manifest if shared_eval_manifest else "/app/lab_artifacts/qwen/eval/manifest.json"
    train_jsonl = write_jsonl(tmp_path / "dataset" / "train.jsonl", [sft_row("train-a", split="train", manifest_path=train_manifest)])
    eval_jsonl = write_jsonl(tmp_path / "dataset" / "eval.jsonl", [sft_row("eval-a", split="eval", manifest_path=eval_manifest)])
    excluded_jsonl = write_jsonl(tmp_path / "dataset" / "excluded.jsonl", [])
    return write_json(
        tmp_path / "dataset" / "manifest.json",
        {
            "schema": "qwen_controller_sft_dataset_manifest.v1",
            "manifest_type": "qwen_controller_sft_dataset",
            "dataset_status": status,
            "target_model_id": "Qwen/Qwen3-4B-Thinking-2507",
            "fingerprint": "dataset-fingerprint",
            "training_started": False,
            "fine_tune_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "delivery_allowed": False,
            "release_ready": False,
            "summary": {
                "train_examples": 1,
                "eval_examples": 1,
                "excluded_examples": 0,
            },
            "outputs": {
                "train_jsonl": str(train_jsonl),
                "eval_jsonl": str(eval_jsonl),
                "excluded_jsonl": str(excluded_jsonl),
                "manifest_json": str(tmp_path / "dataset" / "manifest.json"),
                "progress_jsonl": str(tmp_path / "dataset" / "progress.jsonl"),
            },
        },
    )


def install_green_runtime(monkeypatch) -> None:
    monkeypatch.setattr(qwen_preflight, "module_available", lambda module_name: True)
    monkeypatch.setattr(
        qwen_preflight,
        "qlora_api_report",
        lambda: {
            "compatible": True,
            "skipped": False,
            "pytorch_only_env": {"USE_TF": "0", "TRANSFORMERS_NO_TF": "1"},
            "checks": {
                "sft_config_has_max_length": True,
                "sft_config_has_packing": True,
                "sft_trainer_has_processing_class": True,
                "sft_trainer_has_peft_config": True,
                "bitsandbytes_config_imported": True,
            },
            "failed_checks": [],
        },
    )
    monkeypatch.setattr(
        qwen_preflight,
        "model_cache_report",
        lambda model_id: {
            "target_model_id": model_id,
            "cache_available": True,
            "snapshot_path": "/app/storage/cache/huggingface/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/unit-test",
            "local_files_only": True,
            "required_files": {"config_json": True, "tokenizer": True, "weights": True},
            "missing_required_files": [],
        },
    )
    monkeypatch.setattr(
        qwen_preflight,
        "cuda_report",
        lambda requested_device, min_cuda_free_memory_mb: {
            "cuda_available": True,
            "requested_device": requested_device,
            "device_count": 1,
            "device_name": "Unit Test GPU",
            "free_memory_mb": 24000,
            "total_memory_mb": 32000,
            "min_cuda_free_memory_mb": min_cuda_free_memory_mb,
            "memory_guard_passed": True,
        },
    )


def test_preflight_passes_ready_dataset_with_qwen_runtime(monkeypatch, tmp_path: Path) -> None:
    install_green_runtime(monkeypatch)
    dataset_manifest = write_sft_dataset(tmp_path)

    manifest = run_qwen_controller_qlora_preflight(
        sft_dataset_manifest_json=dataset_manifest,
        output_dir=tmp_path / "out",
        min_cuda_free_memory_mb=12000,
    )

    assert manifest["schema"] == "qwen_controller_qlora_preflight_manifest.v1"
    assert manifest["preflight_status"] == "passed"
    assert manifest["failed_checks"] == []
    assert manifest["training_started"] is False
    assert manifest["fine_tune_started"] is False
    assert manifest["models_changed"] is False
    assert manifest["model_registry_edited"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["execution_contract"]["docker_runtime_required"] is True
    assert manifest["execution_contract"]["gpu_training_required"] is True
    assert manifest["execution_contract"]["gpu_training_skipped"] is True
    assert manifest["execution_contract"]["expected_acceleration_mode"] == "gpu_qlora_training_preflight"
    assert manifest["cuda"]["device_name"] == "Unit Test GPU"
    assert manifest["dependency_report"]["missing_required_modules"] == []
    assert manifest["qlora_api"]["compatible"] is True
    assert manifest["model_cache"]["cache_available"] is True
    assert manifest["dataset"]["train_examples"] == 1
    assert manifest["dataset"]["eval_examples"] == 1
    assert manifest["dataset"]["train_eval_overlap_count"] == 0
    assert Path(manifest["outputs"]["manifest_json"]).name == "manifest.json"
    assert Path(manifest["outputs"]["preflight_json"]).name == "preflight.json"
    assert Path(manifest["outputs"]["progress_jsonl"]).name == "progress.jsonl"
    progress_rows = read_jsonl(Path(manifest["outputs"]["progress_jsonl"]))
    assert [row["event"] for row in progress_rows] == [
        "start",
        "dataset_checked",
        "runtime_checked",
        "qlora_api_checked",
        "model_cache_checked",
        "finish",
    ]


def test_preflight_blocks_missing_required_qwen_modules(monkeypatch, tmp_path: Path) -> None:
    missing = {"peft", "bitsandbytes"}
    monkeypatch.setattr(qwen_preflight, "module_available", lambda module_name: module_name not in missing)
    monkeypatch.setattr(
        qwen_preflight,
        "cuda_report",
        lambda requested_device, min_cuda_free_memory_mb: {
            "cuda_available": True,
            "requested_device": requested_device,
            "device_count": 1,
            "device_name": "Unit Test GPU",
            "free_memory_mb": 24000,
            "total_memory_mb": 32000,
            "min_cuda_free_memory_mb": min_cuda_free_memory_mb,
            "memory_guard_passed": True,
        },
    )
    dataset_manifest = write_sft_dataset(tmp_path)

    manifest = run_qwen_controller_qlora_preflight(
        sft_dataset_manifest_json=dataset_manifest,
        output_dir=tmp_path / "out",
    )

    assert manifest["preflight_status"] == "blocked"
    assert "missing_required_python_modules" in manifest["failed_checks"]
    assert manifest["dependency_report"]["missing_required_modules"] == ["bitsandbytes", "peft"]


def test_preflight_blocks_missing_offline_qwen_snapshot(monkeypatch, tmp_path: Path) -> None:
    install_green_runtime(monkeypatch)
    monkeypatch.setattr(
        qwen_preflight,
        "model_cache_report",
        lambda model_id: {
            "target_model_id": model_id,
            "cache_available": False,
            "snapshot_path": "",
            "local_files_only": True,
            "blocker": "model_snapshot_not_cached",
            "error": {"type": "LocalEntryNotFoundError", "message": "offline cache miss"},
        },
        raising=False,
    )
    dataset_manifest = write_sft_dataset(tmp_path)

    manifest = run_qwen_controller_qlora_preflight(
        sft_dataset_manifest_json=dataset_manifest,
        output_dir=tmp_path / "out",
        min_cuda_free_memory_mb=12000,
    )

    assert manifest["preflight_status"] == "blocked"
    assert "model_cache_unavailable" in manifest["failed_checks"]
    assert manifest["model_cache"]["cache_available"] is False
    assert manifest["model_cache"]["blocker"] == "model_snapshot_not_cached"
    assert manifest["execution_contract"]["fail_closed_on_model_cache_unavailable"] is True


def test_preflight_blocks_qlora_api_incompatibility(monkeypatch, tmp_path: Path) -> None:
    install_green_runtime(monkeypatch)
    monkeypatch.setattr(
        qwen_preflight,
        "qlora_api_report",
        lambda: {
            "compatible": False,
            "skipped": False,
            "pytorch_only_env": {"USE_TF": "0", "TRANSFORMERS_NO_TF": "1"},
            "checks": {
                "sft_config_has_max_length": False,
                "sft_config_has_packing": True,
                "sft_trainer_has_processing_class": False,
                "sft_trainer_has_peft_config": True,
                "bitsandbytes_config_imported": True,
            },
            "failed_checks": ["sft_config_missing_max_length", "sft_trainer_missing_processing_class"],
        },
    )
    dataset_manifest = write_sft_dataset(tmp_path)

    manifest = run_qwen_controller_qlora_preflight(
        sft_dataset_manifest_json=dataset_manifest,
        output_dir=tmp_path / "out",
    )

    assert manifest["preflight_status"] == "blocked"
    assert "qlora_runtime_api_incompatible" in manifest["failed_checks"]
    assert manifest["qlora_api"]["compatible"] is False
    assert manifest["execution_contract"]["fail_closed_on_qlora_api_incompatible"] is True


def test_preflight_blocks_dataset_status_and_train_eval_overlap(monkeypatch, tmp_path: Path) -> None:
    install_green_runtime(monkeypatch)
    dataset_manifest = write_sft_dataset(tmp_path, status="blocked", shared_eval_manifest=True)

    manifest = run_qwen_controller_qlora_preflight(
        sft_dataset_manifest_json=dataset_manifest,
        output_dir=tmp_path / "out",
    )

    assert manifest["preflight_status"] == "blocked"
    assert "sft_dataset_not_ready" in manifest["failed_checks"]
    assert "train_eval_manifest_overlap" in manifest["failed_checks"]
    assert manifest["dataset"]["train_eval_overlap_count"] == 1
    assert manifest["dataset"]["train_eval_overlap_manifest_paths"] == ["/app/lab_artifacts/qwen/train/manifest.json"]


def test_preflight_blocks_train_eval_overlap_after_path_normalization(monkeypatch, tmp_path: Path) -> None:
    install_green_runtime(monkeypatch)
    train_jsonl = write_jsonl(
        tmp_path / "dataset" / "train.jsonl",
        [sft_row("train-a", split="train", manifest_path=r"\app\lab\qwen\shared\..\shared\manifest.json")],
    )
    eval_jsonl = write_jsonl(
        tmp_path / "dataset" / "eval.jsonl",
        [sft_row("eval-a", split="eval", manifest_path="/app/lab/qwen/shared/manifest.json")],
    )
    dataset_manifest = write_json(
        tmp_path / "dataset" / "manifest.json",
        {
            "schema": "qwen_controller_sft_dataset_manifest.v1",
            "manifest_type": "qwen_controller_sft_dataset",
            "dataset_status": "ready",
            "target_model_id": "Qwen/Qwen3-4B-Thinking-2507",
            "fingerprint": "dataset-fingerprint",
            "training_started": False,
            "fine_tune_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "delivery_allowed": False,
            "release_ready": False,
            "outputs": {
                "train_jsonl": str(train_jsonl),
                "eval_jsonl": str(eval_jsonl),
                "manifest_json": str(tmp_path / "dataset" / "manifest.json"),
                "progress_jsonl": str(tmp_path / "dataset" / "progress.jsonl"),
            },
        },
    )

    manifest = run_qwen_controller_qlora_preflight(
        sft_dataset_manifest_json=dataset_manifest,
        output_dir=tmp_path / "out",
    )

    assert manifest["preflight_status"] == "blocked"
    assert "train_eval_manifest_overlap" in manifest["failed_checks"]
    assert manifest["dataset"]["train_eval_overlap_manifest_paths"] == ["/app/lab/qwen/shared/manifest.json"]


def test_cuda_report_blocks_non_cuda_and_out_of_range_devices(monkeypatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_name(device_index: int) -> str:
            return f"GPU {device_index}"

        @staticmethod
        def mem_get_info(device_index: int):
            return 24 * 1024 * 1024, 32 * 1024 * 1024

    fake_torch = type("FakeTorch", (), {"cuda": FakeCuda})()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    non_cuda = qwen_preflight.cuda_report("cpu", 0)
    out_of_range = qwen_preflight.cuda_report("cuda:9", 0)

    assert non_cuda["cuda_available"] is False
    assert non_cuda["blocker"] == "non_cuda_requested_device"
    assert non_cuda["memory_guard_passed"] is False
    assert out_of_range["cuda_available"] is False
    assert out_of_range["blocker"] == "cuda_device_index_out_of_range"
    assert out_of_range["memory_guard_passed"] is False


def test_cli_checks_docker_runtime_before_parsing_args(monkeypatch) -> None:
    calls = []

    def reject_non_docker_runtime(**kwargs):
        calls.append(("require_docker_runtime", kwargs))
        raise RuntimeError("Docker-only test sentinel")

    def fail_if_args_are_parsed(self, argv=None):
        raise AssertionError("parse_args must not run before Docker runtime validation")

    monkeypatch.setattr(qwen_preflight, "require_docker_runtime", reject_non_docker_runtime)
    monkeypatch.setattr(qwen_preflight.argparse.ArgumentParser, "parse_args", fail_if_args_are_parsed)

    try:
        main(["--sft-dataset-manifest-json", "/tmp/manifest.json", "--output-dir", "/tmp/out"])
    except RuntimeError as exc:
        assert str(exc) == "Docker-only test sentinel"
    else:
        raise AssertionError("main must fail through Docker runtime guard")

    assert calls == [
        (
            "require_docker_runtime",
            {
                "tool_name": "qwen_controller_qlora_preflight.py",
                "entrypoint_path": "/app/tools/qwen_controller_qlora_preflight.py",
            },
        )
    ]


def test_cli_writes_preflight_manifest(monkeypatch, tmp_path: Path) -> None:
    install_green_runtime(monkeypatch)
    monkeypatch.setattr(qwen_preflight, "require_docker_runtime", lambda **kwargs: None)
    dataset_manifest = write_sft_dataset(tmp_path)

    assert main(["--sft-dataset-manifest-json", str(dataset_manifest), "--output-dir", str(tmp_path / "out")]) == 0

    manifest = read_json(tmp_path / "out" / "manifest.json")
    assert manifest["preflight_status"] == "passed"
    assert (tmp_path / "out" / "preflight.json").exists()
    assert (tmp_path / "out" / "progress.jsonl").exists()
