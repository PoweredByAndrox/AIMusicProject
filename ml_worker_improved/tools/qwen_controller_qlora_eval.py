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
ROWS_NAME = "rows.jsonl"
PROGRESS_NAME = "progress.jsonl"
EXPECTED_DATASET_MANIFEST_TYPE = "qwen_controller_sft_dataset"
EXPECTED_TRAINING_MANIFEST_TYPE = "qwen_controller_qlora_training"
TARGET_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "gpu_qlora_adapter_eval",
        "gpu_training_required": False,
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "qwen_controller_qlora_eval_only",
        "gpu_inference_required": True,
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


def canonical_manifest_path(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = posixpath.normpath(text.replace("\\", "/"))
    return "" if normalized == "." else normalized


def row_manifest_path(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        provenance = metadata.get("provenance")
        if isinstance(provenance, dict) and provenance.get("manifest_path"):
            return str(provenance["manifest_path"])
    return str(row.get("manifest_path") or "")


def dataset_report(sft_dataset_manifest_json: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(sft_dataset_manifest_json)
    eval_jsonl = output_path(manifest, "eval_jsonl")
    train_jsonl = output_path(manifest, "train_jsonl")
    failed_checks: list[str] = []
    if str(manifest.get("manifest_type") or "") != EXPECTED_DATASET_MANIFEST_TYPE:
        failed_checks.append("unexpected_sft_dataset_manifest_type")
    if str(manifest.get("dataset_status") or "") != "ready":
        failed_checks.append("sft_dataset_not_ready")
    if manifest.get("delivery_allowed") is True or manifest.get("release_ready") is True:
        failed_checks.append("sft_dataset_claims_delivery_or_release")
    eval_rows = list(iter_jsonl(eval_jsonl)) if eval_jsonl and eval_jsonl.is_file() else []
    train_rows = list(iter_jsonl(train_jsonl)) if train_jsonl and train_jsonl.is_file() else []
    if not eval_rows:
        failed_checks.append("no_eval_examples")
    train_paths = {canonical_manifest_path(row_manifest_path(row)) for row in train_rows if canonical_manifest_path(row_manifest_path(row))}
    eval_paths = {canonical_manifest_path(row_manifest_path(row)) for row in eval_rows if canonical_manifest_path(row_manifest_path(row))}
    if train_paths.intersection(eval_paths):
        failed_checks.append("train_eval_manifest_overlap")
    return (
        {
            "manifest_json": str(sft_dataset_manifest_json),
            "manifest_type": str(manifest.get("manifest_type") or ""),
            "dataset_status": str(manifest.get("dataset_status") or ""),
            "target_model_id": str(manifest.get("target_model_id") or ""),
            "fingerprint": str(manifest.get("fingerprint") or ""),
            "eval_jsonl": str(eval_jsonl),
            "eval_examples": len(eval_rows),
            "train_eval_overlap_count": len(train_paths.intersection(eval_paths)),
            "failed_checks": sorted(set(failed_checks)),
        },
        eval_rows,
    )


def training_report(training_manifest_json: Path) -> dict[str, Any]:
    manifest = read_json(training_manifest_json)
    adapter_dir = output_path(manifest, "adapter_dir")
    failed_checks: list[str] = []
    if str(manifest.get("manifest_type") or "") != EXPECTED_TRAINING_MANIFEST_TYPE:
        failed_checks.append("unexpected_training_manifest_type")
    if str(manifest.get("training_status") or "") != "trained":
        failed_checks.append("training_not_trained")
    if manifest.get("delivery_allowed") is True or manifest.get("release_ready") is True:
        failed_checks.append("training_manifest_claims_delivery_or_release")
    if manifest.get("model_registry_edited") is True:
        failed_checks.append("training_manifest_claims_registry_edit")
    if not adapter_dir.is_dir():
        failed_checks.append("adapter_dir_missing")
    return {
        "manifest_json": str(training_manifest_json),
        "manifest_type": str(manifest.get("manifest_type") or ""),
        "training_status": str(manifest.get("training_status") or ""),
        "target_model_id": str(manifest.get("target_model_id") or ""),
        "adapter_dir": str(adapter_dir),
        "failed_checks": sorted(set(failed_checks)),
    }


def expected_payload(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return {}
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = str(message.get("content") or "")
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                return {}
            return payload if isinstance(payload, dict) else {}
    return {}


def prompt_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return []
    prompt: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            break
        if isinstance(message, dict):
            prompt.append(message)
    return prompt


def parse_json_prediction(text: str) -> tuple[dict[str, Any], str]:
    stripped = text.strip()
    if not stripped:
        return {}, "empty_generation"
    try:
        payload = json.loads(stripped)
        return (payload if isinstance(payload, dict) else {}, "" if isinstance(payload, dict) else "prediction_not_object")
    except json.JSONDecodeError:
        start = stripped.rfind("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(stripped[start : end + 1])
                return (payload if isinstance(payload, dict) else {}, "" if isinstance(payload, dict) else "prediction_not_object")
            except json.JSONDecodeError as exc:
                return {}, f"json_parse_failed:{exc.msg}"
        return {}, "json_object_not_found"


def normalized_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted": bool(payload.get("accepted")),
        "action": str(payload.get("action") or ""),
        "route_id": str(payload.get("route_id") or ""),
        "stem_type": str(payload.get("stem_type") or ""),
    }


def prediction_correct(expected: dict[str, Any], predicted: dict[str, Any]) -> bool:
    return normalized_payload(expected) == normalized_payload(predicted)


def load_eval_runtime() -> SimpleNamespace:
    configure_pytorch_only_transformers_env()
    try:
        import torch  # type: ignore
        from peft import PeftModel  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Qwen QLoRA eval runtime import failed: {type(exc).__name__}: {exc}") from exc
    return SimpleNamespace(
        torch=torch,
        PeftModel=PeftModel,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
        BitsAndBytesConfig=BitsAndBytesConfig,
    )


def dtype_value(runtime: Any, compute_dtype: str) -> Any:
    if compute_dtype == "float16":
        return runtime.torch.float16
    return runtime.torch.bfloat16


def release_cuda(runtime: Any, progress_path: Path, *, phase: str) -> None:
    try:
        cuda = getattr(runtime.torch, "cuda", None)
        if cuda is not None and cuda.is_available():
            cuda.synchronize()
            cuda.empty_cache()
            if hasattr(cuda, "ipc_collect"):
                cuda.ipc_collect()
            cuda.synchronize()
        progress(progress_path, "cuda_cleanup", phase=phase, cleanup_performed=True)
    except Exception as exc:
        progress(progress_path, "cuda_cleanup_failed", phase=phase, error={"type": type(exc).__name__, "message": str(exc)})


def decode_generated_text(tokenizer: Any, generated: Any, prompt_tokens: Any) -> str:
    try:
        output_tokens = generated[0][prompt_tokens.shape[-1] :]
    except Exception:
        try:
            output_tokens = generated[0]
        except Exception:
            output_tokens = generated
    return str(tokenizer.decode(output_tokens, skip_special_tokens=True)).strip()


def generate_predictions(
    *,
    runtime: Any,
    eval_rows: list[dict[str, Any]],
    model_role: str,
    adapter_dir: Path | None,
    max_new_tokens: int,
    compute_dtype: str,
) -> list[dict[str, Any]]:
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
    if adapter_dir is not None:
        model = runtime.PeftModel.from_pretrained(model, str(adapter_dir))
    if hasattr(model, "eval"):
        model.eval()
    predictions: list[dict[str, Any]] = []
    for row in eval_rows:
        prompts = prompt_messages(row)
        expected = expected_payload(row)
        prompt_tokens = tokenizer.apply_chat_template(prompts, add_generation_prompt=True, return_tensors="pt")
        if hasattr(prompt_tokens, "to"):
            prompt_tokens = prompt_tokens.to(getattr(model, "device", "cuda"))
        generated = model.generate(prompt_tokens, max_new_tokens=max_new_tokens, do_sample=False)
        raw_text = decode_generated_text(tokenizer, generated, prompt_tokens)
        predicted, parse_error = parse_json_prediction(raw_text)
        predictions.append(
            {
                "id": str(row.get("id") or ""),
                "expected": expected,
                "predicted": predicted,
                "raw_text": raw_text,
                "parse_error": parse_error,
                "model_role": model_role,
            }
        )
    return predictions


def score_predictions(predictions: Sequence[dict[str, Any]], eval_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    expected_by_id = {str(row.get("id") or ""): expected_payload(row) for row in eval_rows}
    scored_rows: list[dict[str, Any]] = []
    correct_count = 0
    parse_error_count = 0
    for prediction in predictions:
        row_id = str(prediction.get("id") or "")
        expected = expected_by_id.get(row_id, {})
        predicted = prediction.get("predicted") if isinstance(prediction.get("predicted"), dict) else {}
        parse_error = str(prediction.get("parse_error") or "")
        correct = not parse_error and prediction_correct(expected, predicted)
        if correct:
            correct_count += 1
        if parse_error:
            parse_error_count += 1
        scored_rows.append(
            {
                **prediction,
                "expected": expected,
                "correct": correct,
            }
        )
    evaluated = len(predictions)
    return {
        "evaluated_examples": evaluated,
        "correct_examples": correct_count,
        "parse_error_count": parse_error_count,
        "accuracy": round(correct_count / evaluated, 6) if evaluated else 0.0,
        "rows": scored_rows,
    }


def write_eval_manifest(
    *,
    output_dir: Path,
    progress_path: Path,
    sft_dataset_manifest_json: Path,
    training_manifest_json: Path,
    dataset: dict[str, Any],
    training: dict[str, Any],
    eval_status: str,
    failed_checks: Sequence[str],
    baseline_comparison: dict[str, Any],
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema": "qwen_controller_qlora_eval_manifest.v1",
        "manifest_type": "qwen_controller_qlora_eval",
        "eval_status": eval_status,
        "failed_checks": sorted(set(failed_checks)),
        "sft_dataset_manifest_json": str(sft_dataset_manifest_json),
        "training_manifest_json": str(training_manifest_json),
        "output_dir": str(output_dir),
        "target_model_id": TARGET_MODEL_ID,
        "dataset": dataset,
        "training": training,
        "baseline_comparison": baseline_comparison,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "qwen_qlora_eval_not_for_listen_pack",
        "release_status": "adapter_eval_artifact_only",
        "release_ready": False,
        "promotion": {
            "status": "adapter_eval_passed" if eval_status == "passed" else "blocked",
            "reason": (
                "adapter beat baseline but still requires manual registry review"
                if eval_status == "passed"
                else "eval gate failed or did not run"
            ),
            "can_promote_adapter": False,
            "can_edit_model_registry": False,
            "next_required_step": "manual_registry_review" if eval_status == "passed" else "fix_eval_blockers",
        },
        "outputs": {
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "rows_jsonl": str(output_dir / ROWS_NAME),
            "progress_jsonl": str(progress_path),
        },
        "safety": {
            "docker_runtime_required": True,
            "gpu_inference_required": True,
            "baseline_comparison_required": True,
            "blocks_training_data_eval": True,
            "not_release_artifact": True,
            "not_listen_artifact": True,
            "model_registry_updates_blocked": True,
        },
    }
    if error:
        manifest["error"] = error
    write_json(output_dir / MANIFEST_NAME, manifest)
    return manifest


def empty_comparison(
    *,
    min_eval_examples: int,
    min_adapter_accuracy: float,
    min_accuracy_improvement: float,
) -> dict[str, Any]:
    return {
        "eval_examples": 0,
        "min_eval_examples": int(min_eval_examples),
        "min_adapter_accuracy": float(min_adapter_accuracy),
        "min_accuracy_improvement": float(min_accuracy_improvement),
        "baseline_accuracy": 0.0,
        "adapter_accuracy": 0.0,
        "adapter_accuracy_improvement": 0.0,
        "baseline_parse_error_count": 0,
        "adapter_parse_error_count": 0,
    }


def run_qwen_controller_qlora_eval(
    *,
    sft_dataset_manifest_json: Path,
    training_manifest_json: Path,
    output_dir: Path,
    min_eval_examples: int = 20,
    min_adapter_accuracy: float = 0.7,
    min_accuracy_improvement: float = 0.0,
    max_new_tokens: int = 128,
    compute_dtype: str = "bfloat16",
) -> dict[str, Any]:
    if min_eval_examples <= 0:
        raise ValueError("min_eval_examples must be > 0")
    if not 0.0 <= min_adapter_accuracy <= 1.0:
        raise ValueError("min_adapter_accuracy must be between 0 and 1")
    if min_accuracy_improvement < 0:
        raise ValueError("min_accuracy_improvement must be >= 0")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
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
        training_manifest_json=str(training_manifest_json),
        min_eval_examples=min_eval_examples,
        min_adapter_accuracy=min_adapter_accuracy,
        min_accuracy_improvement=min_accuracy_improvement,
    )
    dataset, eval_rows = dataset_report(sft_dataset_manifest_json)
    progress(progress_path, "dataset_checked", eval_examples=dataset["eval_examples"], failed_checks=dataset["failed_checks"])
    training = training_report(training_manifest_json)
    progress(
        progress_path,
        "training_manifest_checked",
        training_status=training["training_status"],
        adapter_dir=training["adapter_dir"],
        failed_checks=training["failed_checks"],
    )
    failed_checks = sorted(set(dataset["failed_checks"] + training["failed_checks"]))
    if int(dataset["eval_examples"]) < min_eval_examples:
        failed_checks.append("insufficient_eval_examples")
    failed_checks = sorted(set(failed_checks))
    if failed_checks:
        write_jsonl(output_dir / ROWS_NAME, [])
        comparison = empty_comparison(
            min_eval_examples=min_eval_examples,
            min_adapter_accuracy=min_adapter_accuracy,
            min_accuracy_improvement=min_accuracy_improvement,
        )
        comparison["eval_examples"] = int(dataset.get("eval_examples") or 0)
        manifest = write_eval_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            sft_dataset_manifest_json=sft_dataset_manifest_json,
            training_manifest_json=training_manifest_json,
            dataset=dataset,
            training=training,
            eval_status="blocked",
            failed_checks=failed_checks,
            baseline_comparison=comparison,
        )
        progress(progress_path, "finish", eval_status="blocked", failed_checks=failed_checks)
        return manifest

    try:
        runtime = load_eval_runtime()
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        failed_checks = ["qlora_eval_runtime_import_failed"]
        write_jsonl(output_dir / ROWS_NAME, [])
        manifest = write_eval_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            sft_dataset_manifest_json=sft_dataset_manifest_json,
            training_manifest_json=training_manifest_json,
            dataset=dataset,
            training=training,
            eval_status="blocked",
            failed_checks=failed_checks,
            baseline_comparison=empty_comparison(
                min_eval_examples=min_eval_examples,
                min_adapter_accuracy=min_adapter_accuracy,
                min_accuracy_improvement=min_accuracy_improvement,
            ),
            error=error,
        )
        progress(progress_path, "finish", eval_status="blocked", failed_checks=failed_checks, error=error)
        return manifest

    progress(progress_path, "baseline_eval_start", eval_examples=len(eval_rows))
    try:
        baseline_predictions = generate_predictions(
            runtime=runtime,
            eval_rows=eval_rows,
            model_role="baseline",
            adapter_dir=None,
            max_new_tokens=max_new_tokens,
            compute_dtype=compute_dtype,
        )
        baseline_score = score_predictions(baseline_predictions, eval_rows)
        progress(progress_path, "baseline_eval_finish", accuracy=baseline_score["accuracy"])
        release_cuda(runtime, progress_path, phase="after_baseline_before_adapter")
        progress(progress_path, "adapter_eval_start", eval_examples=len(eval_rows))
        adapter_predictions = generate_predictions(
            runtime=runtime,
            eval_rows=eval_rows,
            model_role="adapter",
            adapter_dir=Path(training["adapter_dir"]),
            max_new_tokens=max_new_tokens,
            compute_dtype=compute_dtype,
        )
        adapter_score = score_predictions(adapter_predictions, eval_rows)
        progress(progress_path, "adapter_eval_finish", accuracy=adapter_score["accuracy"])
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        failed_checks = ["qwen_adapter_eval_failed"]
        write_jsonl(output_dir / ROWS_NAME, [])
        manifest = write_eval_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            sft_dataset_manifest_json=sft_dataset_manifest_json,
            training_manifest_json=training_manifest_json,
            dataset=dataset,
            training=training,
            eval_status="failed",
            failed_checks=failed_checks,
            baseline_comparison=empty_comparison(
                min_eval_examples=min_eval_examples,
                min_adapter_accuracy=min_adapter_accuracy,
                min_accuracy_improvement=min_accuracy_improvement,
            ),
            error=error,
        )
        progress(progress_path, "finish", eval_status="failed", failed_checks=failed_checks, error=error)
        return manifest

    rows = [
        {**row, "model_role": "baseline"}
        for row in baseline_score["rows"]
    ] + [
        {**row, "model_role": "adapter"}
        for row in adapter_score["rows"]
    ]
    write_jsonl(output_dir / ROWS_NAME, rows)
    improvement = round(float(adapter_score["accuracy"]) - float(baseline_score["accuracy"]), 6)
    failed_checks = []
    if int(dataset["eval_examples"]) < min_eval_examples:
        failed_checks.append("insufficient_eval_examples")
    if float(adapter_score["accuracy"]) < min_adapter_accuracy:
        failed_checks.append("adapter_accuracy_below_threshold")
    if improvement <= min_accuracy_improvement:
        failed_checks.append("adapter_not_better_than_baseline")
    if int(adapter_score["parse_error_count"] or 0) > 0:
        failed_checks.append("adapter_parse_errors")
    failed_checks = sorted(set(failed_checks))
    comparison = {
        "eval_examples": int(dataset["eval_examples"]),
        "min_eval_examples": int(min_eval_examples),
        "min_adapter_accuracy": float(min_adapter_accuracy),
        "min_accuracy_improvement": float(min_accuracy_improvement),
        "baseline_accuracy": float(baseline_score["accuracy"]),
        "adapter_accuracy": float(adapter_score["accuracy"]),
        "adapter_accuracy_improvement": improvement,
        "baseline_parse_error_count": int(baseline_score["parse_error_count"]),
        "adapter_parse_error_count": int(adapter_score["parse_error_count"]),
    }
    eval_status = "passed" if not failed_checks else "blocked"
    manifest = write_eval_manifest(
        output_dir=output_dir,
        progress_path=progress_path,
        sft_dataset_manifest_json=sft_dataset_manifest_json,
        training_manifest_json=training_manifest_json,
        dataset=dataset,
        training=training,
        eval_status=eval_status,
        failed_checks=failed_checks,
        baseline_comparison=comparison,
    )
    progress(progress_path, "finish", eval_status=eval_status, failed_checks=failed_checks)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Qwen controller QLoRA adapter against the base-model baseline.")
    parser.add_argument("--sft-dataset-manifest-json", type=Path, required=True)
    parser.add_argument("--training-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-eval-examples", type=int, default=20)
    parser.add_argument("--min-adapter-accuracy", type=float, default=0.7)
    parser.add_argument("--min-accuracy-improvement", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--compute-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="qwen_controller_qlora_eval.py",
        entrypoint_path="/app/tools/qwen_controller_qlora_eval.py",
    )
    args = parse_args(argv)
    manifest = run_qwen_controller_qlora_eval(
        sft_dataset_manifest_json=args.sft_dataset_manifest_json,
        training_manifest_json=args.training_manifest_json,
        output_dir=args.output_dir,
        min_eval_examples=int(args.min_eval_examples),
        min_adapter_accuracy=float(args.min_adapter_accuracy),
        min_accuracy_improvement=float(args.min_accuracy_improvement),
        max_new_tokens=int(args.max_new_tokens),
        compute_dtype=str(args.compute_dtype),
    )
    print(
        json.dumps(
            {
                "manifest_path": manifest["outputs"]["manifest_json"],
                "eval_status": manifest["eval_status"],
                "failed_checks": manifest["failed_checks"],
                "baseline_comparison": manifest["baseline_comparison"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["eval_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
