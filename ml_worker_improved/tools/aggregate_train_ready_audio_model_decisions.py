from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime


START_TIME = time.monotonic()
MANIFEST_NAME = "aggregate_train_ready_audio_model_decision_manifest.json"
SUMMARY_NAME = "aggregate_train_ready_audio_model_decision_summary.md"
PROGRESS_NAME = "aggregate_train_ready_audio_model_decision_progress.jsonl"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "metadata_only_aggregate_train_ready_decision",
        "expected_acceleration_mode": "external_only_cpu_metadata_validation",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def int_value(payload: dict[str, Any], key: str) -> int:
    try:
        return int(payload.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Aggregate Train Ready Audio Model Decision",
        "",
        f"- aggregate_decision: `{manifest['aggregate_decision']}`",
        f"- train_ready: `{str(manifest['train_ready']).lower()}`",
        f"- aggregate_success_rate: `{manifest['aggregate_success_rate']:.6f}`",
        f"- decision_count: `{manifest['decision_count']}`",
        f"- train_ready_child_count: `{manifest['train_ready_child_count']}`",
        f"- failed_checks: `{', '.join(manifest['failed_checks']) if manifest['failed_checks'] else 'none'}`",
        "- release_ready: `false`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def aggregate_train_ready_audio_model_decisions(
    *,
    decision_manifest_jsons: Sequence[Path],
    output_dir: Path,
    min_success_rate: float = 0.90,
    min_decision_count: int = 1,
) -> dict[str, Any]:
    if min_decision_count <= 0:
        raise ValueError("min_decision_count must be > 0")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(
        progress_path,
        "start",
        decision_manifest_jsons=[str(path) for path in decision_manifest_jsons],
        min_success_rate=min_success_rate,
        min_decision_count=min_decision_count,
    )
    children: list[dict[str, Any]] = []
    failed_checks: list[str] = []
    total_passed = 0
    total_eval = 0
    train_ready_child_count = 0
    for path in decision_manifest_jsons:
        child = read_json(path)
        total_passed += int_value(child, "passed_eval_count")
        total_eval += int_value(child, "holdout_eval_count")
        if child.get("train_ready") is True:
            train_ready_child_count += 1
        if child.get("release_ready") is True:
            failed_checks.append("child_release_ready_true")
        if child.get("delivery_allowed") is True:
            failed_checks.append("child_delivery_allowed_true")
        if child.get("promotion_allowed") is True:
            failed_checks.append("child_promotion_allowed_true")
        for check in child.get("failed_checks") or []:
            failed_checks.append(f"child_failed_check:{check}")
        children.append(
            {
                "path": str(path),
                "decision": child.get("decision"),
                "train_ready": child.get("train_ready") is True,
                "success_rate": float(child.get("success_rate") or 0.0),
                "passed_eval_count": int_value(child, "passed_eval_count"),
                "holdout_eval_count": int_value(child, "holdout_eval_count"),
                "failed_checks": list(child.get("failed_checks") or []),
            }
        )
    aggregate_success_rate = total_passed / total_eval if total_eval > 0 else 0.0
    if len(children) < min_decision_count:
        failed_checks.append("insufficient_decision_count")
    if total_eval <= 0:
        failed_checks.append("aggregate_eval_count_missing")
    if not aggregate_success_rate > float(min_success_rate):
        failed_checks.append("aggregate_success_rate_not_strictly_above_threshold")
    failed_checks = sorted(set(failed_checks))
    train_ready = not failed_checks
    manifest = {
        "schema": "aggregate_train_ready_audio_model_decision_manifest.v1",
        "manifest_type": "aggregate_train_ready_audio_model_decision",
        "decision_id": stable_hash({"children": children, "failed_checks": failed_checks})[:16],
        "aggregate_decision": "train_result_verified" if train_ready else "train_result_not_ready",
        "train_ready": train_ready,
        "failed_checks": failed_checks,
        "aggregate_success_rate": round(aggregate_success_rate, 12),
        "success_rate_rule": f"> {float(min_success_rate):.6f}",
        "min_success_rate": float(min_success_rate),
        "decision_count": len(children),
        "min_decision_count": int(min_decision_count),
        "train_ready_child_count": train_ready_child_count,
        "passed_eval_count": total_passed,
        "holdout_eval_count": total_eval,
        "children": children,
        "training_started": any(child.get("holdout_eval_count", 0) > 0 for child in children),
        "fine_tune_started": any(child.get("holdout_eval_count", 0) > 0 for child in children),
        "models_changed": False,
        "model_registry_edited": False,
        "runtime_gates_changed": False,
        "thresholds_relaxed": False,
        "delivery_allowed": False,
        "promotion_allowed": False,
        "presentation_status": "aggregate_train_ready_decision_artifact_not_for_listen_pack",
        "release_status": "training_decision_only",
        "release_ready": False,
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
    }
    write_json(output_dir / MANIFEST_NAME, manifest)
    write_summary(output_dir / SUMMARY_NAME, manifest)
    progress(progress_path, "finish", aggregate_decision=manifest["aggregate_decision"], failed_checks=failed_checks)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate train-ready decision manifests across folds.")
    parser.add_argument("--decision-manifest-json", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-success-rate", type=float, default=0.90)
    parser.add_argument("--min-decision-count", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="aggregate_train_ready_audio_model_decisions.py",
        entrypoint_path="/app/tools/aggregate_train_ready_audio_model_decisions.py",
    )
    args = parse_args(argv)
    manifest = aggregate_train_ready_audio_model_decisions(
        decision_manifest_jsons=args.decision_manifest_json,
        output_dir=args.output_dir,
        min_success_rate=args.min_success_rate,
        min_decision_count=args.min_decision_count,
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "aggregate_decision": manifest["aggregate_decision"],
                "aggregate_success_rate": manifest["aggregate_success_rate"],
                "train_ready": manifest["train_ready"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["train_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
