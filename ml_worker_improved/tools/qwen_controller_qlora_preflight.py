#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import importlib.util
import json
import posixpath
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime  # noqa: E402
from tools.qwen_controller_runtime_env import configure_pytorch_only_transformers_env  # noqa: E402

MANIFEST_NAME = "manifest.json"
PREFLIGHT_NAME = "preflight.json"
PROGRESS_NAME = "progress.jsonl"
EXPECTED_DATASET_MANIFEST_TYPE = "qwen_controller_sft_dataset"
TARGET_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_MIN_CUDA_FREE_MEMORY_MB = 12_000
REQUIRED_MODULES = (
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "bitsandbytes",
    "trl",
    "sentencepiece",
    "google.protobuf",
)

START_TIME = time.monotonic()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "gpu_training_required": True,
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "qlora_preflight_only",
        "gpu_inference_required": False,
        "gpu_inference_skipped": True,
        "expected_acceleration_mode": "gpu_qlora_training_preflight",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def dependency_report() -> dict[str, Any]:
    modules = {
        module_name: {
            "available": module_available(module_name),
        }
        for module_name in REQUIRED_MODULES
    }
    missing = sorted(module_name for module_name, report in modules.items() if report["available"] is not True)
    return {
        "required_modules": list(REQUIRED_MODULES),
        "modules": modules,
        "missing_required_modules": missing,
        "all_required_modules_available": not missing,
    }


def qlora_api_report() -> dict[str, Any]:
    env_report = configure_pytorch_only_transformers_env()
    try:
        from transformers import BitsAndBytesConfig  # type: ignore
        from trl import SFTConfig, SFTTrainer  # type: ignore
    except Exception as exc:
        return {
            "compatible": False,
            "skipped": False,
            "pytorch_only_env": env_report,
            "checks": {},
            "failed_checks": ["qlora_api_import_failed"],
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }

    checks = {
        "sft_config_has_max_length": "max_length" in inspect.signature(SFTConfig).parameters,
        "sft_config_has_packing": "packing" in inspect.signature(SFTConfig).parameters,
        "sft_trainer_has_processing_class": "processing_class" in inspect.signature(SFTTrainer.__init__).parameters,
        "sft_trainer_has_peft_config": "peft_config" in inspect.signature(SFTTrainer.__init__).parameters,
        "bitsandbytes_config_imported": BitsAndBytesConfig is not None,
    }
    failed_checks = []
    if not checks["sft_config_has_max_length"]:
        failed_checks.append("sft_config_missing_max_length")
    if not checks["sft_config_has_packing"]:
        failed_checks.append("sft_config_missing_packing")
    if not checks["sft_trainer_has_processing_class"]:
        failed_checks.append("sft_trainer_missing_processing_class")
    if not checks["sft_trainer_has_peft_config"]:
        failed_checks.append("sft_trainer_missing_peft_config")
    if not checks["bitsandbytes_config_imported"]:
        failed_checks.append("bitsandbytes_config_import_failed")
    return {
        "compatible": not failed_checks,
        "skipped": False,
        "pytorch_only_env": env_report,
        "checks": checks,
        "failed_checks": failed_checks,
    }


def snapshot_file_report(snapshot_path: Path, *, target_model_id: str, local_files_only: bool) -> dict[str, Any]:
    required_files = {
        "config_json": (snapshot_path / "config.json").is_file(),
        "tokenizer": any((snapshot_path / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json")),
        "weights": any(snapshot_path.glob("*.safetensors")) or any(snapshot_path.glob("*.bin")),
    }
    missing_required_files = sorted(name for name, present in required_files.items() if not present)
    cache_available = snapshot_path.is_dir() and not missing_required_files
    report = {
        "target_model_id": target_model_id,
        "cache_available": cache_available,
        "snapshot_path": str(snapshot_path) if snapshot_path.is_dir() else "",
        "local_files_only": local_files_only,
        "required_files": required_files,
        "missing_required_files": missing_required_files,
    }
    if not cache_available:
        report["blocker"] = "model_snapshot_missing_required_files"
    return report


def model_cache_report(model_id: str) -> dict[str, Any]:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:
        return {
            "target_model_id": model_id,
            "cache_available": False,
            "snapshot_path": "",
            "local_files_only": True,
            "blocker": "huggingface_hub_import_failed",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }

    try:
        snapshot_path = Path(snapshot_download(repo_id=model_id, local_files_only=True))
    except Exception as exc:
        return {
            "target_model_id": model_id,
            "cache_available": False,
            "snapshot_path": "",
            "local_files_only": True,
            "blocker": "model_snapshot_not_cached",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }

    return snapshot_file_report(snapshot_path, target_model_id=model_id, local_files_only=True)


def cuda_report(requested_device: str, min_cuda_free_memory_mb: int) -> dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception as exc:
        return {
            "cuda_available": False,
            "requested_device": requested_device,
            "device_count": 0,
            "device_name": "",
            "free_memory_mb": 0,
            "total_memory_mb": 0,
            "min_cuda_free_memory_mb": min_cuda_free_memory_mb,
            "memory_guard_passed": False,
            "blocker": "torch_import_failed",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }

    if not requested_device.startswith("cuda:"):
        return {
            "cuda_available": False,
            "requested_device": requested_device,
            "device_count": int(torch.cuda.device_count()),
            "device_name": "",
            "free_memory_mb": 0,
            "total_memory_mb": 0,
            "min_cuda_free_memory_mb": min_cuda_free_memory_mb,
            "memory_guard_passed": False,
            "blocker": "non_cuda_requested_device",
        }

    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "requested_device": requested_device,
            "device_count": int(torch.cuda.device_count()),
            "device_name": "",
            "free_memory_mb": 0,
            "total_memory_mb": 0,
            "min_cuda_free_memory_mb": min_cuda_free_memory_mb,
            "memory_guard_passed": False,
            "blocker": "torch_cuda_unavailable",
        }

    device_count = int(torch.cuda.device_count())
    try:
        device_index = int(requested_device.split(":", 1)[1])
    except ValueError:
        device_index = 0
    if device_index < 0 or device_index >= device_count:
        return {
            "cuda_available": False,
            "requested_device": requested_device,
            "device_count": device_count,
            "device_name": "",
            "free_memory_mb": 0,
            "total_memory_mb": 0,
            "min_cuda_free_memory_mb": min_cuda_free_memory_mb,
            "memory_guard_passed": False,
            "blocker": "cuda_device_index_out_of_range",
        }
    device_name = str(torch.cuda.get_device_name(device_index))
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        free_mb = int(free_bytes // (1024 * 1024))
        total_mb = int(total_bytes // (1024 * 1024))
    except Exception:
        free_mb = 0
        total_mb = 0
    return {
        "cuda_available": True,
        "requested_device": requested_device,
        "device_count": device_count,
        "device_name": device_name,
        "free_memory_mb": free_mb,
        "total_memory_mb": total_mb,
        "min_cuda_free_memory_mb": min_cuda_free_memory_mb,
        "memory_guard_passed": free_mb >= min_cuda_free_memory_mb,
    }


def output_path(manifest: dict[str, Any], key: str) -> Path:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        return Path("")
    return Path(str(outputs.get(key) or ""))


def row_manifest_path(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        provenance = metadata.get("provenance")
        if isinstance(provenance, dict) and provenance.get("manifest_path"):
            return str(provenance["manifest_path"])
    return str(row.get("manifest_path") or "")


def canonical_manifest_path(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = posixpath.normpath(text.replace("\\", "/"))
    return "" if normalized == "." else normalized


def dataset_report(sft_dataset_manifest_json: Path) -> dict[str, Any]:
    manifest = read_json(sft_dataset_manifest_json)
    train_jsonl = output_path(manifest, "train_jsonl")
    eval_jsonl = output_path(manifest, "eval_jsonl")
    failed_checks: list[str] = []

    if str(manifest.get("manifest_type") or "") != EXPECTED_DATASET_MANIFEST_TYPE:
        failed_checks.append("unexpected_sft_dataset_manifest_type")
    if str(manifest.get("dataset_status") or "") != "ready":
        failed_checks.append("sft_dataset_not_ready")
    if manifest.get("delivery_allowed") is True or manifest.get("release_ready") is True:
        failed_checks.append("sft_dataset_claims_delivery_or_release")
    if not train_jsonl.is_file():
        failed_checks.append("train_jsonl_missing")
        train_rows: list[dict[str, Any]] = []
    else:
        train_rows = list(iter_jsonl(train_jsonl))
    if eval_jsonl and eval_jsonl.is_file():
        eval_rows = list(iter_jsonl(eval_jsonl))
    else:
        eval_rows = []
    if not train_rows:
        failed_checks.append("no_train_examples")

    train_paths = {canonical_manifest_path(row_manifest_path(row)) for row in train_rows if canonical_manifest_path(row_manifest_path(row))}
    eval_paths = {canonical_manifest_path(row_manifest_path(row)) for row in eval_rows if canonical_manifest_path(row_manifest_path(row))}
    overlap = sorted(train_paths.intersection(eval_paths))
    if overlap:
        failed_checks.append("train_eval_manifest_overlap")

    return {
        "manifest_json": str(sft_dataset_manifest_json),
        "manifest_type": str(manifest.get("manifest_type") or ""),
        "dataset_status": str(manifest.get("dataset_status") or ""),
        "target_model_id": str(manifest.get("target_model_id") or ""),
        "fingerprint": str(manifest.get("fingerprint") or ""),
        "train_jsonl": str(train_jsonl),
        "eval_jsonl": str(eval_jsonl),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "train_manifest_path_count": len(train_paths),
        "eval_manifest_path_count": len(eval_paths),
        "train_eval_overlap_count": len(overlap),
        "train_eval_overlap_manifest_paths": overlap[:20],
        "failed_checks": failed_checks,
    }


def run_qwen_controller_qlora_preflight(
    *,
    sft_dataset_manifest_json: Path,
    output_dir: Path,
    requested_device: str = "cuda:0",
    min_cuda_free_memory_mb: int = DEFAULT_MIN_CUDA_FREE_MEMORY_MB,
) -> dict[str, Any]:
    if min_cuda_free_memory_mb < 0:
        raise ValueError("min_cuda_free_memory_mb must be >= 0")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()

    progress(
        progress_path,
        "start",
        sft_dataset_manifest_json=str(sft_dataset_manifest_json),
        requested_device=requested_device,
        min_cuda_free_memory_mb=min_cuda_free_memory_mb,
    )
    dataset = dataset_report(sft_dataset_manifest_json)
    progress(
        progress_path,
        "dataset_checked",
        train_examples=dataset["train_examples"],
        eval_examples=dataset["eval_examples"],
        failed_checks=dataset["failed_checks"],
    )
    deps = dependency_report()
    qlora_api = (
        qlora_api_report()
        if deps["all_required_modules_available"]
        else {
            "compatible": False,
            "skipped": True,
            "pytorch_only_env": {},
            "checks": {},
            "failed_checks": ["skipped_due_to_missing_required_modules"],
        }
    )
    cuda = cuda_report(requested_device, min_cuda_free_memory_mb)
    model_cache = model_cache_report(TARGET_MODEL_ID)
    progress(
        progress_path,
        "runtime_checked",
        missing_required_modules=deps["missing_required_modules"],
        cuda_available=cuda["cuda_available"],
        memory_guard_passed=cuda["memory_guard_passed"],
    )
    progress(
        progress_path,
        "qlora_api_checked",
        compatible=qlora_api["compatible"],
        skipped=qlora_api["skipped"],
        failed_checks=qlora_api["failed_checks"],
    )
    progress(
        progress_path,
        "model_cache_checked",
        cache_available=model_cache["cache_available"],
        blocker=model_cache.get("blocker", ""),
        snapshot_path=model_cache.get("snapshot_path", ""),
    )

    failed_checks: list[str] = []
    failed_checks.extend(dataset["failed_checks"])
    if deps["missing_required_modules"]:
        failed_checks.append("missing_required_python_modules")
    if qlora_api["compatible"] is not True and qlora_api["skipped"] is not True:
        failed_checks.append("qlora_runtime_api_incompatible")
    if model_cache["cache_available"] is not True:
        failed_checks.append("model_cache_unavailable")
    if cuda["cuda_available"] is not True:
        failed_checks.append("cuda_unavailable")
    if cuda["memory_guard_passed"] is not True:
        failed_checks.append("cuda_memory_guard_failed")
    failed_checks = sorted(set(failed_checks))
    preflight_status = "passed" if not failed_checks else "blocked"

    execution_contract = {
        "docker_runtime_required": True,
        "expected_acceleration_mode": "gpu_qlora_training_preflight",
        "target_model_id": TARGET_MODEL_ID,
        "requested_device": requested_device,
        "gpu_training_required": True,
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "qlora_preflight_only",
        "gpu_inference_required": False,
        "gpu_inference_skipped": True,
        "fail_closed_on_missing_dependencies": True,
        "fail_closed_on_qlora_api_incompatible": True,
        "fail_closed_on_model_cache_unavailable": True,
        "fail_closed_on_cuda_unavailable": True,
    }
    preflight = {
        "schema": "qwen_controller_qlora_preflight_report.v1",
        "preflight_status": preflight_status,
        "failed_checks": failed_checks,
        "dependency_report": deps,
        "qlora_api": qlora_api,
        "model_cache": model_cache,
        "cuda": cuda,
        "dataset": dataset,
        "execution_contract": execution_contract,
    }
    manifest = {
        "schema": "qwen_controller_qlora_preflight_manifest.v1",
        "manifest_type": "qwen_controller_qlora_preflight",
        "preflight_status": preflight_status,
        "failed_checks": failed_checks,
        "sft_dataset_manifest_json": str(sft_dataset_manifest_json),
        "output_dir": str(output_dir),
        "target_model_id": TARGET_MODEL_ID,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "qwen_qlora_preflight_not_for_listen_pack",
        "release_status": "training_preflight_artifact_only",
        "release_ready": False,
        "dependency_report": deps,
        "qlora_api": qlora_api,
        "model_cache": model_cache,
        "cuda": cuda,
        "dataset": dataset,
        "execution_contract": execution_contract,
        "outputs": {
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "preflight_json": str(output_dir / PREFLIGHT_NAME),
            "progress_jsonl": str(progress_path),
        },
        "safety": {
            "metadata_only": True,
            "training_not_started": True,
            "not_release_artifact": True,
            "blocks_missing_qlora_dependencies": True,
            "blocks_qlora_api_incompatibility": True,
            "blocks_missing_offline_model_cache": True,
            "blocks_cuda_unavailable": True,
            "blocks_train_eval_overlap": True,
        },
        "recommended_next_step": (
            "run_qwen_controller_qlora_training"
            if preflight_status == "passed"
            else "rebuild_docker_image_cache_qwen_model_or_fix_dataset_before_training"
        ),
    }
    write_json(output_dir / PREFLIGHT_NAME, preflight)
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress(progress_path, "finish", preflight_status=preflight_status, failed_checks=failed_checks)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight Qwen controller QLoRA training runtime and dataset inputs.")
    parser.add_argument("--sft-dataset-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requested-device", default="cuda:0")
    parser.add_argument("--min-cuda-free-memory-mb", type=int, default=DEFAULT_MIN_CUDA_FREE_MEMORY_MB)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="qwen_controller_qlora_preflight.py",
        entrypoint_path="/app/tools/qwen_controller_qlora_preflight.py",
    )
    args = parse_args(argv)
    manifest = run_qwen_controller_qlora_preflight(
        sft_dataset_manifest_json=args.sft_dataset_manifest_json,
        output_dir=args.output_dir,
        requested_device=str(args.requested_device),
        min_cuda_free_memory_mb=int(args.min_cuda_free_memory_mb),
    )
    print(
        json.dumps(
            {
                "manifest_path": manifest["outputs"]["manifest_json"],
                "preflight_status": manifest["preflight_status"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["preflight_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
