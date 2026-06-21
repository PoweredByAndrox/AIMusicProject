from __future__ import annotations

import argparse
import json
import posixpath
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime
from tools.qwen_controller_runtime_env import configure_pytorch_only_transformers_env

MANIFEST_NAME = "manifest.json"
METRICS_NAME = "metrics.json"
PROGRESS_NAME = "progress.jsonl"
ADAPTER_DIR_NAME = "adapter"
TRAINER_DIR_NAME = "trainer"
EXPECTED_DATASET_MANIFEST_TYPE = "qwen_controller_sft_dataset"
EXPECTED_PREFLIGHT_MANIFEST_TYPE = "qwen_controller_qlora_preflight"
TARGET_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
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
        "expected_acceleration_mode": "gpu_qlora_training",
        "gpu_training_required": True,
        "gpu_inference_required": False,
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "qwen_controller_qlora_training_not_inference",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


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


def load_dataset_rows(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    failed_checks: list[str] = []
    train_jsonl = output_path(manifest, "train_jsonl")
    eval_jsonl = output_path(manifest, "eval_jsonl")
    if not train_jsonl.is_file():
        failed_checks.append("train_jsonl_missing")
        train_rows: list[dict[str, Any]] = []
    else:
        train_rows = list(iter_jsonl(train_jsonl))
    eval_rows = list(iter_jsonl(eval_jsonl)) if eval_jsonl and eval_jsonl.is_file() else []
    if not train_rows:
        failed_checks.append("no_train_examples")
    train_paths = {canonical_manifest_path(row_manifest_path(row)) for row in train_rows if canonical_manifest_path(row_manifest_path(row))}
    eval_paths = {canonical_manifest_path(row_manifest_path(row)) for row in eval_rows if canonical_manifest_path(row_manifest_path(row))}
    if train_paths.intersection(eval_paths):
        failed_checks.append("train_eval_manifest_overlap")
    return train_rows, eval_rows, failed_checks


def dataset_report(sft_dataset_manifest_json: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = read_json(sft_dataset_manifest_json)
    train_rows, eval_rows, failed_checks = load_dataset_rows(manifest)
    if str(manifest.get("manifest_type") or "") != EXPECTED_DATASET_MANIFEST_TYPE:
        failed_checks.append("unexpected_sft_dataset_manifest_type")
    if str(manifest.get("dataset_status") or "") != "ready":
        failed_checks.append("sft_dataset_not_ready")
    if manifest.get("delivery_allowed") is True or manifest.get("release_ready") is True:
        failed_checks.append("sft_dataset_claims_delivery_or_release")
    report = {
        "manifest_json": str(sft_dataset_manifest_json),
        "manifest_type": str(manifest.get("manifest_type") or ""),
        "dataset_status": str(manifest.get("dataset_status") or ""),
        "target_model_id": str(manifest.get("target_model_id") or ""),
        "fingerprint": str(manifest.get("fingerprint") or ""),
        "train_jsonl": str(output_path(manifest, "train_jsonl")),
        "eval_jsonl": str(output_path(manifest, "eval_jsonl")),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "failed_checks": sorted(set(failed_checks)),
    }
    return report, train_rows, eval_rows


def preflight_report(preflight_manifest_json: Path) -> dict[str, Any]:
    manifest = read_json(preflight_manifest_json)
    failed_checks: list[str] = []
    if str(manifest.get("manifest_type") or "") != EXPECTED_PREFLIGHT_MANIFEST_TYPE:
        failed_checks.append("unexpected_preflight_manifest_type")
    if str(manifest.get("preflight_status") or "") != "passed":
        failed_checks.append("preflight_not_passed")
    if manifest.get("delivery_allowed") is True or manifest.get("release_ready") is True:
        failed_checks.append("preflight_claims_delivery_or_release")
    dependency_report = manifest.get("dependency_report") if isinstance(manifest.get("dependency_report"), dict) else {}
    missing_modules = dependency_report.get("missing_required_modules")
    if isinstance(missing_modules, list) and missing_modules:
        failed_checks.append("missing_required_python_modules")
    cuda = manifest.get("cuda") if isinstance(manifest.get("cuda"), dict) else {}
    if cuda.get("cuda_available") is not True:
        failed_checks.append("cuda_unavailable")
    if cuda.get("memory_guard_passed") is not True:
        failed_checks.append("cuda_memory_guard_failed")
    return {
        "manifest_json": str(preflight_manifest_json),
        "manifest_type": str(manifest.get("manifest_type") or ""),
        "preflight_status": str(manifest.get("preflight_status") or ""),
        "failed_checks": sorted(set(failed_checks)),
        "source_failed_checks": manifest.get("failed_checks") if isinstance(manifest.get("failed_checks"), list) else [],
        "dependency_report": dependency_report,
        "cuda": cuda,
    }


def training_config(
    *,
    output_dir: Path,
    num_train_epochs: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    max_length: int,
    max_steps: int | None,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    compute_dtype: str,
    target_modules: Sequence[str],
) -> dict[str, Any]:
    config = {
        "target_model_id": TARGET_MODEL_ID,
        "trainer_output_dir": str(output_dir / TRAINER_DIR_NAME),
        "adapter_dir": str(output_dir / ADAPTER_DIR_NAME),
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": compute_dtype,
        },
        "lora": {
            "r": int(lora_r),
            "lora_alpha": int(lora_alpha),
            "lora_dropout": float(lora_dropout),
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": list(target_modules),
        },
        "sft": {
            "num_train_epochs": float(num_train_epochs),
            "per_device_train_batch_size": int(per_device_train_batch_size),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "learning_rate": float(learning_rate),
            "max_length": int(max_length),
            "max_steps": int(max_steps) if max_steps is not None else -1,
            "optim": "paged_adamw_8bit",
            "gradient_checkpointing": True,
            "report_to": [],
            "save_total_limit": 2,
            "logging_steps": 1,
            "save_steps": 25,
            "packing": False,
        },
    }
    return config


def load_qlora_runtime() -> SimpleNamespace:
    configure_pytorch_only_transformers_env()
    try:
        import torch  # type: ignore
        from datasets import Dataset  # type: ignore
        from peft import LoraConfig  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore
        from trl import SFTConfig, SFTTrainer  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Qwen QLoRA runtime import failed: {type(exc).__name__}: {exc}") from exc
    return SimpleNamespace(
        torch=torch,
        Dataset=Dataset,
        LoraConfig=LoraConfig,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
        BitsAndBytesConfig=BitsAndBytesConfig,
        SFTConfig=SFTConfig,
        SFTTrainer=SFTTrainer,
    )


def dtype_value(runtime: Any, compute_dtype: str) -> Any:
    if compute_dtype == "float16":
        return runtime.torch.float16
    return runtime.torch.bfloat16


def write_training_manifest(
    *,
    output_dir: Path,
    progress_path: Path,
    sft_dataset_manifest_json: Path,
    preflight_manifest_json: Path,
    dataset: dict[str, Any],
    preflight: dict[str, Any],
    config: dict[str, Any],
    training_status: str,
    failed_checks: Sequence[str],
    metrics: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trained = training_status == "trained"
    blocked = training_status == "blocked"
    manifest = {
        "schema": "qwen_controller_qlora_training_manifest.v1",
        "manifest_type": "qwen_controller_qlora_training",
        "training_status": training_status,
        "failed_checks": sorted(set(failed_checks)),
        "sft_dataset_manifest_json": str(sft_dataset_manifest_json),
        "preflight_manifest_json": str(preflight_manifest_json),
        "output_dir": str(output_dir),
        "target_model_id": TARGET_MODEL_ID,
        "training_started": trained or not blocked,
        "fine_tune_started": trained or not blocked,
        "models_changed": trained,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "qwen_qlora_training_not_for_listen_pack",
        "release_status": "training_artifact_only" if trained else "blocked_before_training",
        "release_ready": False,
        "summary": {
            "train_examples": int(dataset.get("train_examples") or 0),
            "eval_examples": int(dataset.get("eval_examples") or 0),
        },
        "dataset": dataset,
        "preflight": preflight,
        "training_config": config,
        "metrics": metrics or {},
        "promotion": {
            "status": "adapter_requires_external_eval" if trained else "blocked",
            "reason": (
                "external eval gate required before any registry or release action"
                if trained
                else "training did not start or did not complete"
            ),
            "can_promote_adapter": False,
            "can_edit_model_registry": False,
            "next_required_step": "external_eval_gate" if trained else "fix_training_blockers",
        },
        "outputs": {
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "adapter_dir": str(output_dir / ADAPTER_DIR_NAME),
            "trainer_dir": str(output_dir / TRAINER_DIR_NAME),
            "metrics_json": str(output_dir / METRICS_NAME),
            "progress_jsonl": str(progress_path),
        },
        "safety": {
            "docker_runtime_required": True,
            "gpu_training_required": True,
            "fail_closed_on_missing_runtime": True,
            "blocks_failed_preflight": True,
            "blocks_dataset_leakage": True,
            "not_release_artifact": True,
            "not_listen_artifact": True,
            "model_registry_updates_blocked": True,
        },
    }
    if error:
        manifest["error"] = error
    write_json(output_dir / MANIFEST_NAME, manifest)
    return manifest


def run_qwen_controller_qlora_training(
    *,
    sft_dataset_manifest_json: Path,
    preflight_manifest_json: Path,
    output_dir: Path,
    num_train_epochs: float = 1.0,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2.0e-4,
    max_length: int = 2048,
    max_steps: int | None = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    compute_dtype: str = "bfloat16",
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
) -> dict[str, Any]:
    if num_train_epochs <= 0:
        raise ValueError("num_train_epochs must be > 0")
    if per_device_train_batch_size <= 0:
        raise ValueError("per_device_train_batch_size must be > 0")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be > 0")
    if max_length <= 0:
        raise ValueError("max_length must be > 0")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be > 0 when provided")
    if compute_dtype not in {"bfloat16", "float16"}:
        raise ValueError("compute_dtype must be bfloat16 or float16")

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(
        progress_path,
        "start",
        sft_dataset_manifest_json=str(sft_dataset_manifest_json),
        preflight_manifest_json=str(preflight_manifest_json),
    )
    dataset, train_rows, eval_rows = dataset_report(sft_dataset_manifest_json)
    progress(
        progress_path,
        "dataset_checked",
        train_examples=dataset["train_examples"],
        eval_examples=dataset["eval_examples"],
        failed_checks=dataset["failed_checks"],
    )
    preflight = preflight_report(preflight_manifest_json)
    progress(
        progress_path,
        "preflight_checked",
        preflight_status=preflight["preflight_status"],
        failed_checks=preflight["failed_checks"],
    )
    config = training_config(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_length=max_length,
        max_steps=max_steps,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        compute_dtype=compute_dtype,
        target_modules=target_modules,
    )
    failed_checks = sorted(set(dataset["failed_checks"] + preflight["failed_checks"]))
    if failed_checks:
        manifest = write_training_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            sft_dataset_manifest_json=sft_dataset_manifest_json,
            preflight_manifest_json=preflight_manifest_json,
            dataset=dataset,
            preflight=preflight,
            config=config,
            training_status="blocked",
            failed_checks=failed_checks,
        )
        progress(progress_path, "finish", training_status="blocked", failed_checks=failed_checks)
        return manifest

    try:
        runtime = load_qlora_runtime()
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        failed_checks = ["qlora_runtime_import_failed"]
        manifest = write_training_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            sft_dataset_manifest_json=sft_dataset_manifest_json,
            preflight_manifest_json=preflight_manifest_json,
            dataset=dataset,
            preflight=preflight,
            config=config,
            training_status="blocked",
            failed_checks=failed_checks,
            error=error,
        )
        progress(progress_path, "finish", training_status="blocked", failed_checks=failed_checks, error=error)
        return manifest

    progress(progress_path, "runtime_loaded", train_examples=len(train_rows), eval_examples=len(eval_rows))
    dtype = dtype_value(runtime, compute_dtype)
    quantization_config = runtime.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = runtime.AutoTokenizer.from_pretrained(TARGET_MODEL_ID, trust_remote_code=True, local_files_only=True)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = runtime.AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
        local_files_only=True,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    progress(progress_path, "model_loaded", target_model_id=TARGET_MODEL_ID, device_map="auto")

    train_dataset = runtime.Dataset.from_list(train_rows)
    eval_dataset = runtime.Dataset.from_list(eval_rows) if eval_rows else None
    lora_config = runtime.LoraConfig(**config["lora"])
    sft_kwargs = dict(config["sft"])
    sft_kwargs["output_dir"] = str(output_dir / TRAINER_DIR_NAME)
    training_args = runtime.SFTConfig(**sft_kwargs)
    trainer = runtime.SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    progress(progress_path, "trainer_started", max_steps=config["sft"]["max_steps"])
    try:
        train_output = trainer.train()
        metrics = dict(getattr(train_output, "metrics", {}) or {})
        metrics["log_history"] = list(getattr(getattr(trainer, "state", None), "log_history", []) or [])
        adapter_dir = output_dir / ADAPTER_DIR_NAME
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        failed_checks = ["qlora_training_failed"]
        manifest = write_training_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            sft_dataset_manifest_json=sft_dataset_manifest_json,
            preflight_manifest_json=preflight_manifest_json,
            dataset=dataset,
            preflight=preflight,
            config=config,
            training_status="failed",
            failed_checks=failed_checks,
            error=error,
        )
        progress(progress_path, "finish", training_status="failed", failed_checks=failed_checks, error=error)
        return manifest

    progress(progress_path, "adapter_saved", adapter_dir=str(output_dir / ADAPTER_DIR_NAME))
    write_json(output_dir / METRICS_NAME, metrics)
    manifest = write_training_manifest(
        output_dir=output_dir,
        progress_path=progress_path,
        sft_dataset_manifest_json=sft_dataset_manifest_json,
        preflight_manifest_json=preflight_manifest_json,
        dataset=dataset,
        preflight=preflight,
        config=config,
        training_status="trained",
        failed_checks=[],
        metrics=metrics,
    )
    progress(progress_path, "finish", training_status="trained", failed_checks=[])
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gated Qwen controller QLoRA training from a reviewed SFT dataset.")
    parser.add_argument("--sft-dataset-manifest-json", type=Path, required=True)
    parser.add_argument("--preflight-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--compute-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="qwen_controller_qlora_train.py",
        entrypoint_path="/app/tools/qwen_controller_qlora_train.py",
    )
    args = parse_args(argv)
    manifest = run_qwen_controller_qlora_training(
        sft_dataset_manifest_json=args.sft_dataset_manifest_json,
        preflight_manifest_json=args.preflight_manifest_json,
        output_dir=args.output_dir,
        num_train_epochs=float(args.num_train_epochs),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        learning_rate=float(args.learning_rate),
        max_length=int(args.max_length),
        max_steps=args.max_steps,
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        compute_dtype=str(args.compute_dtype),
    )
    print(
        json.dumps(
            {
                "manifest_path": manifest["outputs"]["manifest_json"],
                "training_status": manifest["training_status"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["training_status"] == "trained" else 2


if __name__ == "__main__":
    raise SystemExit(main())
