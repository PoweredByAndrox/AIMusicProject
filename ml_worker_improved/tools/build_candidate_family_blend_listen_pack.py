#!/usr/bin/env python3
"""Build a listen-check pack from verified candidate-family blend holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402
from train_teacher_mask_separator import validate_stem_name  # noqa: E402
from artifact_name_policy import ARTIFACT_AUDIO_DIR, indexed_alias, indexed_audio_filename, relative_artifact_path  # noqa: E402


START_TIME = time.monotonic()
MANIFEST_NAME = "pack.json"
SUMMARY_NAME = "summary.md"
SELECTED_NAME = "selected.jsonl"
PROGRESS_NAME = "progress.jsonl"


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
                raise ValueError(f"{path}:{line_number}: expected JSON object row")
            yield row


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_metadata_packaging",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "listen_pack_copies_existing_passed_audio",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "listen_pack_metadata_only",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _decision_failed_checks(decision: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    if decision.get("decision") != "train_result_verified" or decision.get("train_ready") is not True:
        failed.append("train_ready_decision_not_verified")
    if float(decision.get("success_rate") or 0.0) <= 0.90:
        failed.append("success_rate_not_above_listen_threshold")
    if int(decision.get("critical_leakage_count") or 0) > 0:
        failed.append("critical_leakage_detected")
    for key in ("release_ready", "thresholds_relaxed", "runtime_gates_changed", "model_registry_edited"):
        if decision.get(key) is True:
            failed.append(f"decision_{key}_true")
    for check in decision.get("failed_checks") or []:
        failed.append(f"decision_failed_check:{check}")
    return sorted(set(failed))


def _row_failed_checks(row: dict[str, Any], *, stem: str) -> list[str]:
    failed: list[str] = []
    if row.get("status") != "passed":
        failed.append("holdout_row_not_passed")
    if str(row.get("stem") or row.get("stem_type") or "") != stem:
        failed.append("holdout_row_stem_mismatch")
    for check in row.get("failed_checks") or []:
        if str(check).strip():
            failed.append(f"holdout_row_failed_check:{check}")
    if row.get("release_ready") is True:
        failed.append("holdout_row_release_ready_true")
    if row.get("delivery_allowed") is True:
        failed.append("holdout_row_delivery_allowed_true")
    holdout_track = str(row.get("holdout_track_id") or "")
    training_tracks = {str(item) for item in row.get("checkpoint_training_track_ids") or [] if str(item)}
    if holdout_track and holdout_track in training_tracks:
        failed.append("holdout_track_leakage")
    audio_path = Path(str(row.get("candidate_audio_path") or ""))
    if not str(audio_path) or not audio_path.exists() or not audio_path.is_file():
        failed.append("candidate_audio_missing")
    return sorted(set(failed))


def _listen_row(row: dict[str, Any], *, output_dir: Path, stem: str, index: int) -> dict[str, Any]:
    source_path = Path(str(row["candidate_audio_path"]))
    candidate_id = stable_hash(
        {
            "config_id": row.get("config_id"),
            "holdout_track_id": row.get("holdout_track_id"),
            "source_path": str(source_path),
        }
    )[:16]
    artifact_alias = indexed_alias("v", index)
    name = indexed_audio_filename("v", index)
    listen_path = output_dir / ARTIFACT_AUDIO_DIR / name
    listen_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, listen_path)
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    return {
        "schema": "candidate_family_blend_listen_pack_audio.v1",
        "candidate_id": candidate_id,
        "stem_type": stem,
        "holdout_track_id": str(row.get("holdout_track_id") or ""),
        "source_holdout_track_id": str(row.get("holdout_track_id") or ""),
        "config_id": str(row.get("config_id") or ""),
        "source_config_id": str(row.get("config_id") or ""),
        "artifact_alias": artifact_alias,
        "artifact_filename": name,
        "artifact_relative_path": relative_artifact_path(listen_path, output_dir),
        "baseline_family": str(row.get("baseline_family") or ""),
        "baseline_policy": str(row.get("baseline_policy") or ""),
        "candidate_sdr_db": metrics.get("candidate_sdr_db"),
        "baseline_sdr_db": metrics.get("baseline_sdr_db"),
        "improvement_db": metrics.get("improvement_db"),
        "source_candidate_audio_path": str(source_path),
        "listen_audio_path": str(listen_path),
        "listen_audio_sha256": sha256_file(listen_path),
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "listenable": True,
        "deliverable": False,
    }


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Candidate Family Blend Listen Pack",
        "",
        f"- delivery_status: `{manifest['delivery_status']}`",
        f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
        f"- presentation_status: `{manifest['presentation_status']}`",
        f"- release_ready: `false`",
        f"- listen_audio_count: `{manifest['listen_audio_count']}`",
        f"- failed_checks: `{', '.join(manifest['failed_checks']) if manifest['failed_checks'] else 'none'}`",
        "",
    ]
    for item in manifest["listen_audio_files"]:
        lines.append(f"- `{item['holdout_track_id']}`: `{item['listen_audio_path']}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_candidate_family_blend_listen_pack(
    *,
    train_ready_decision_manifest_json: Path,
    holdout_rows_jsonls: Sequence[Path],
    output_dir: Path,
    stem: str,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    stem = validate_stem_name(stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    listen_audio_dir = output_dir / ARTIFACT_AUDIO_DIR
    if listen_audio_dir.exists():
        shutil.rmtree(listen_audio_dir)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(
        progress_path,
        "start",
        train_ready_decision_manifest_json=str(train_ready_decision_manifest_json),
        holdout_rows_jsonls=[str(path) for path in holdout_rows_jsonls],
        stem=stem,
    )
    decision = read_json(train_ready_decision_manifest_json)
    failed_checks = _decision_failed_checks(decision)
    decision_allows_packaging = not failed_checks
    selected: list[dict[str, Any]] = []
    rows_seen = 0
    for path in holdout_rows_jsonls:
        for row in iter_jsonl(path):
            rows_seen += 1
            row_checks = _row_failed_checks(row, stem=stem)
            if row_checks:
                failed_checks.extend(row_checks)
                continue
            if decision_allows_packaging:
                selected.append(_listen_row(row, output_dir=output_dir, stem=stem, index=len(selected) + 1))

    failed_checks = sorted(set(failed_checks))
    accepted = not failed_checks and bool(selected)
    if not selected and "listen_audio_missing" not in failed_checks:
        failed_checks.append("listen_audio_missing")
        accepted = False

    manifest = {
        "schema": "candidate_family_blend_listen_pack_manifest.v1",
        "manifest_type": "candidate_family_blend_listen_pack",
        "train_ready_decision_manifest_json": str(train_ready_decision_manifest_json),
        "holdout_rows_jsonls": [str(path) for path in holdout_rows_jsonls],
        "output_dir": str(output_dir),
        "selected_jsonl": str(output_dir / SELECTED_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "stem_type": stem,
        "delivery_status": "accepted" if accepted else "blocked",
        "delivery_allowed": accepted,
        "presentation_status": "accepted_for_listen_pack" if accepted else "blocked_hard_case",
        "release_status": "listen_preview_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "listenable": accepted,
        "deliverable": False,
        "failed_checks": failed_checks,
        "source_rows_seen": rows_seen,
        "listen_audio_count": len(selected) if accepted else 0,
        "listen_audio_files": selected if accepted else [],
        "artifact_name_policy": {
            "schema": "short_windows_safe_artifact_names.v1",
            "audio_dir": ARTIFACT_AUDIO_DIR,
            "filename_strategy": "indexed_alias",
            "raw_track_or_config_ids_in_path": False,
            "provenance_in_manifest": True,
        },
        "audio_path": selected[0]["listen_audio_path"] if accepted else "",
        "candidate_id": selected[0]["candidate_id"] if accepted else "",
        "model_id": "candidate_family_blend_adapter",
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "runtime_gates_changed": False,
        "thresholds_relaxed": False,
        "fallback_used": False,
        "command_args": command_args or [],
        "contract": {
            "source_train_ready_required": True,
            "only_passed_holdout_rows_packaged": True,
            "audio_copied_without_modification": True,
            "release_ready_forced_false": True,
            "can_publish_as_result": False,
            "short_windows_safe_artifact_names": True,
            "raw_ids_kept_in_manifest_only": True,
        },
    }
    write_jsonl(output_dir / SELECTED_NAME, selected if accepted else [])
    write_json(output_dir / MANIFEST_NAME, manifest)
    write_summary(output_dir / SUMMARY_NAME, manifest)
    progress(
        progress_path,
        "finish",
        delivery_status=manifest["delivery_status"],
        delivery_allowed=manifest["delivery_allowed"],
        listen_audio_count=manifest["listen_audio_count"],
        failed_checks=failed_checks,
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a listen-check pack from verified candidate-family blend holdouts.")
    parser.add_argument("--train-ready-decision-manifest-json", type=Path, required=True)
    parser.add_argument("--holdout-rows-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="vocal")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_candidate_family_blend_listen_pack.py",
        entrypoint_path="/app/tools/build_candidate_family_blend_listen_pack.py",
    )
    args = parse_args(argv)
    manifest = build_candidate_family_blend_listen_pack(
        train_ready_decision_manifest_json=args.train_ready_decision_manifest_json,
        holdout_rows_jsonls=args.holdout_rows_jsonl,
        output_dir=args.output_dir,
        stem=str(args.stem),
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "delivery_status": manifest["delivery_status"],
                "delivery_allowed": manifest["delivery_allowed"],
                "listen_audio_count": manifest["listen_audio_count"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["delivery_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
