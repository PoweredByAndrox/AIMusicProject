from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime


START_TIME = time.monotonic()
MANIFEST_NAME = "train_ready_audio_model_decision_manifest.json"
SUMMARY_NAME = "train_ready_audio_model_decision_summary.md"
PROGRESS_NAME = "train_ready_audio_model_decision_progress.jsonl"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid jsonl row: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object row")
            yield row


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
        "gpu_inference_skip_reason": "metadata_only_train_ready_decision",
        "expected_acceleration_mode": "external_only_cpu_metadata_validation",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def bool_true(payload: dict[str, Any], key: str) -> bool:
    return payload.get(key) is True


def int_value(payload: dict[str, Any], key: str) -> int:
    try:
        return int(payload.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def load_matrix_rows(matrix_manifest: dict[str, Any], matrix_manifest_json: Path) -> list[dict[str, Any]]:
    rows_path_text = str(matrix_manifest.get("matrix_rows_jsonl") or "")
    if not rows_path_text:
        candidate = matrix_manifest_json.parent / "generic_audio_finetune_matrix_rows.jsonl"
        rows_path_text = str(candidate)
    rows_path = Path(rows_path_text)
    if not rows_path.exists():
        raise FileNotFoundError(f"matrix rows jsonl not found: {rows_path}")
    return list(iter_jsonl(rows_path))


def select_best_row(matrix_manifest: dict[str, Any], rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    best_config_id = str(matrix_manifest.get("best_config_id") or "")
    if best_config_id:
        for row in rows:
            if str(row.get("config_id") or "") == best_config_id:
                return row
    passed = [row for row in rows if row.get("status") == "passed"]
    return passed[0] if passed else None


def load_leakage_failed_checks(leak_safe_split_manifest_json: Path | None) -> tuple[list[str], dict[str, Any]]:
    if leak_safe_split_manifest_json is None:
        return [], {}
    payload = read_json(leak_safe_split_manifest_json)
    failed: list[str] = []
    critical_count = int_value(payload, "critical_leakage_count")
    if critical_count > 0:
        failed.append("critical_leakage_detected")
    if payload.get("locked") is False:
        failed.append("leak_safe_split_not_locked")
    if payload.get("release_ready") is True:
        failed.append("leak_safe_split_release_ready_true")
    return failed, payload


def holdout_failed_checks(holdout_rows_jsonls: Sequence[Path]) -> tuple[list[str], int]:
    failed: list[str] = []
    row_count = 0
    for path in holdout_rows_jsonls:
        for row in iter_jsonl(path):
            row_count += 1
            holdout_track = str(row.get("holdout_track_id") or "")
            training_tracks = {str(item) for item in row.get("checkpoint_training_track_ids") or [] if str(item)}
            checkpoint_track = str(row.get("checkpoint_track_id") or "")
            if checkpoint_track:
                training_tracks.add(checkpoint_track)
            if holdout_track and holdout_track in training_tracks:
                failed.append("holdout_track_leakage")
            if row.get("release_ready") is True:
                failed.append("holdout_row_release_ready_true")
            if row.get("delivery_allowed") is True:
                failed.append("holdout_row_delivery_allowed_true")
            if row.get("application_device_used") == 0.0:
                failed.append("holdout_application_not_gpu")
            if row.get("status") != "passed":
                failed.append("holdout_row_failed")
            for check in row.get("failed_checks") or []:
                if str(check).strip():
                    failed.append(f"holdout_row_failed_check:{check}")
    return sorted(set(failed)), row_count


def matrix_eval_counts(best_row: dict[str, Any] | None, rows: Sequence[dict[str, Any]]) -> tuple[int, int, int, str]:
    if best_row is not None:
        return (
            int_value(best_row, "holdout_eval_count"),
            int_value(best_row, "passed_eval_count"),
            int_value(best_row, "failed_eval_count"),
            "best_matrix_row",
        )
    return (
        sum(int_value(row, "holdout_eval_count") for row in rows),
        sum(int_value(row, "passed_eval_count") for row in rows),
        sum(int_value(row, "failed_eval_count") for row in rows),
        "all_matrix_rows_when_no_best",
    )


def matrix_safety_failed_checks(matrix_manifest: dict[str, Any], rows: Sequence[dict[str, Any]]) -> list[str]:
    failed: list[str] = []
    if matrix_manifest.get("matrix_validation_status") != "passed":
        failed.append("matrix_validation_not_passed")
    if not str(matrix_manifest.get("best_config_id") or ""):
        failed.append("missing_best_config_id")
    for key in ("release_ready", "delivery_allowed", "model_registry_edited", "runtime_gates_changed", "thresholds_relaxed"):
        if bool_true(matrix_manifest, key):
            failed.append(f"matrix_{key}_true")
    for row in rows:
        config_id = str(row.get("config_id") or "unknown")
        if row.get("release_ready") is True:
            failed.append(f"row_{config_id}_release_ready_true")
        if row.get("delivery_allowed") is True:
            failed.append(f"row_{config_id}_delivery_allowed_true")
    return failed


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Train Ready Audio Model Decision",
        "",
        f"- decision: `{manifest['decision']}`",
        f"- train_ready: `{str(manifest['train_ready']).lower()}`",
        f"- success_rate: `{manifest['success_rate']:.6f}`",
        f"- success_rate_rule: `{manifest['success_rate_rule']}`",
        f"- holdout_eval_count: `{manifest['holdout_eval_count']}`",
        f"- passed_eval_count: `{manifest['passed_eval_count']}`",
        f"- failed_checks: `{', '.join(manifest['failed_checks']) if manifest['failed_checks'] else 'none'}`",
        "- release_ready: `false`",
        "- promotion_allowed: `false`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_train_ready_audio_model_decision(
    *,
    matrix_manifest_json: Path,
    output_dir: Path,
    holdout_rows_jsonls: Sequence[Path] = (),
    leak_safe_split_manifest_json: Path | None = None,
    min_success_rate: float = 0.90,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(
        progress_path,
        "start",
        matrix_manifest_json=str(matrix_manifest_json),
        holdout_rows_jsonls=[str(path) for path in holdout_rows_jsonls],
        leak_safe_split_manifest_json=str(leak_safe_split_manifest_json) if leak_safe_split_manifest_json else "",
        min_success_rate=min_success_rate,
    )
    matrix_manifest = read_json(matrix_manifest_json)
    rows = load_matrix_rows(matrix_manifest, matrix_manifest_json)
    best_row = select_best_row(matrix_manifest, rows)
    progress(progress_path, "matrix_loaded", row_count=len(rows), best_config_id=str(matrix_manifest.get("best_config_id") or ""))

    holdout_eval_count, passed_eval_count, failed_eval_count, eval_count_source = matrix_eval_counts(best_row, rows)
    success_rate = passed_eval_count / holdout_eval_count if holdout_eval_count > 0 else 0.0
    failed_checks = matrix_safety_failed_checks(matrix_manifest, rows)
    if best_row is None:
        failed_checks.append("best_matrix_row_missing")
    else:
        if best_row.get("status") != "passed":
            failed_checks.append("best_matrix_row_not_passed")
        for check in best_row.get("failed_checks") or []:
            failed_checks.append(f"best_row_failed_check:{check}")
    if holdout_eval_count <= 0:
        failed_checks.append("holdout_eval_count_missing")
    if not success_rate > float(min_success_rate):
        failed_checks.append("success_rate_not_strictly_above_threshold")
    leakage_checks, leakage_manifest = load_leakage_failed_checks(leak_safe_split_manifest_json)
    failed_checks.extend(leakage_checks)
    holdout_checks, holdout_row_count = holdout_failed_checks(holdout_rows_jsonls)
    failed_checks.extend(holdout_checks)
    failed_checks = sorted(set(failed_checks))
    train_ready = not failed_checks
    decision = "train_result_verified" if train_ready else "train_result_not_ready"
    manifest = {
        "schema": "train_ready_audio_model_decision_manifest.v1",
        "manifest_type": "train_ready_audio_model_decision",
        "decision_id": stable_hash(
            {
                "matrix_manifest_json": str(matrix_manifest_json),
                "best_config_id": matrix_manifest.get("best_config_id"),
                "success_rate": success_rate,
                "failed_checks": failed_checks,
            }
        )[:16],
        "matrix_manifest_json": str(matrix_manifest_json),
        "output_dir": str(output_dir),
        "decision": decision,
        "train_ready": train_ready,
        "failed_checks": failed_checks,
        "success_rate": round(success_rate, 12),
        "success_rate_rule": f"> {float(min_success_rate):.6f}",
        "min_success_rate": float(min_success_rate),
        "holdout_eval_count": holdout_eval_count,
        "passed_eval_count": passed_eval_count,
        "failed_eval_count": failed_eval_count,
        "eval_count_source": eval_count_source,
        "holdout_rows_checked": holdout_row_count,
        "best_config_id": str(matrix_manifest.get("best_config_id") or ""),
        "best_matrix_row": best_row,
        "leak_safe_split_manifest_json": str(leak_safe_split_manifest_json) if leak_safe_split_manifest_json else "",
        "critical_leakage_count": int_value(leakage_manifest, "critical_leakage_count") if leakage_manifest else 0,
        "training_started": matrix_manifest.get("training_started") is True,
        "fine_tune_started": matrix_manifest.get("fine_tune_started") is True,
        "models_changed": matrix_manifest.get("models_changed") is True,
        "model_registry_edited": False,
        "runtime_gates_changed": False,
        "thresholds_relaxed": False,
        "delivery_allowed": False,
        "promotion_allowed": False,
        "presentation_status": "train_ready_decision_artifact_not_for_listen_pack",
        "release_status": "training_decision_only",
        "release_ready": False,
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "safety": {
            "metadata_only": True,
            "strict_success_rate_gt_threshold": True,
            "learned_or_trained_result_cannot_bypass_safety": True,
            "release_ready_forced_false": True,
            "promotion_not_performed": True,
        },
    }
    write_json(output_dir / MANIFEST_NAME, manifest)
    write_summary(output_dir / SUMMARY_NAME, manifest)
    progress(progress_path, "finish", decision=decision, train_ready=train_ready, failed_checks=failed_checks)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a strict train-ready decision manifest from matrix/holdout artifacts.")
    parser.add_argument("--matrix-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-rows-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--leak-safe-split-manifest-json", type=Path)
    parser.add_argument("--min-success-rate", type=float, default=0.90)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_train_ready_audio_model_decision.py",
        entrypoint_path="/app/tools/build_train_ready_audio_model_decision.py",
    )
    args = parse_args(argv)
    manifest = build_train_ready_audio_model_decision(
        matrix_manifest_json=args.matrix_manifest_json,
        output_dir=args.output_dir,
        holdout_rows_jsonls=args.holdout_rows_jsonl,
        leak_safe_split_manifest_json=args.leak_safe_split_manifest_json,
        min_success_rate=args.min_success_rate,
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "decision": manifest["decision"],
                "train_ready": manifest["train_ready"],
                "success_rate": manifest["success_rate"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["train_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
