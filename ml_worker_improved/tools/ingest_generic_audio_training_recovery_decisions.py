#!/usr/bin/env python3
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

from tools.docker_runtime import require_docker_runtime  # noqa: E402

APPROVED_NAME = "approved_audio_training_examples.jsonl"
REJECTED_NAME = "rejected_audio_training_recovery_rows.jsonl"
MANIFEST_NAME = "generic_audio_training_recovery_decision_manifest.json"
PROGRESS_NAME = "generic_audio_training_recovery_decision_progress.jsonl"

APPROVE_DECISIONS = {"approve_for_training", "approved_for_training"}
NON_APPROVE_DECISIONS = {"exclude", "needs_more_evidence", "reject", "rejected"}
TRAINABLE_PERMISSION_STATUSES = {"owned", "licensed", "cleared", "project_owned"}
NEGATIVE_RATING_BLOCKERS = {"negative_human_rating", "rating_not_positive"}

START_TIME = time.monotonic()


class RecoveryDecisionError(ValueError):
    pass


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RecoveryDecisionError(f"{path}:{line_number}: invalid jsonl row: {exc}") from exc
            if not isinstance(row, dict):
                raise RecoveryDecisionError(f"{path}:{line_number}: expected object row")
            yield row


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "decision_ingest_metadata_only",
        "expected_acceleration_mode": "external_only_cpu_metadata_curation",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
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


def load_decisions(path: Path) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        row_id = str(row.get("recovery_row_id") or row.get("id") or "").strip()
        if not row_id:
            raise RecoveryDecisionError("decision_row_missing_recovery_row_id")
        if row_id in decisions:
            raise RecoveryDecisionError(f"duplicate_recovery_row_decision:{row_id}")
        decision = str(row.get("decision") or "").strip().lower()
        if decision not in APPROVE_DECISIONS and decision not in NON_APPROVE_DECISIONS:
            raise RecoveryDecisionError(f"invalid_recovery_decision:{row_id}:{decision}")
        decisions[row_id] = row
    return decisions


def remaining_blockers(row: dict[str, Any], decision: dict[str, Any] | None) -> list[str]:
    blockers = list(row.get("blockers") or [])
    permission_status = str((decision or {}).get("permission_status") or row.get("permission_status") or "").strip().lower()
    if "missing_permission_status" in blockers and permission_status in TRAINABLE_PERMISSION_STATUSES:
        blockers.remove("missing_permission_status")
    if "permission_not_trainable" in blockers and permission_status in TRAINABLE_PERMISSION_STATUSES:
        blockers.remove("permission_not_trainable")
    if NEGATIVE_RATING_BLOCKERS.intersection(blockers):
        return blockers
    if not Path(str(row.get("source_audio_path") or "")).exists() and "source_audio_not_found" not in blockers:
        blockers.append("source_audio_not_found")
    if not Path(str(row.get("target_audio_path") or "")).exists() and "target_audio_not_found" not in blockers:
        blockers.append("target_audio_not_found")
    return blockers


def approved_example(row: dict[str, Any], decision: dict[str, Any] | None) -> dict[str, Any]:
    target_path = Path(str(row["target_audio_path"]))
    permission_status = str((decision or {}).get("permission_status") or row.get("permission_status") or "").strip().lower()
    return {
        "schema": "generic_finetuning_audio_training_example.v1",
        "id": stable_hash(["approved_recovery", row["id"], permission_status])[:16],
        "track_id": row.get("track_id"),
        "stem_type": row.get("stem_type"),
        "manifest_path": row.get("source_file"),
        "source_audio_path": row.get("source_audio_path"),
        "target_audio_path": row.get("target_audio_path"),
        "target_audio_sha256": sha256_file(target_path),
        "model_id": f"approved_recovery::{row.get('source_type')}",
        "rating": row.get("rating"),
        "human_score": None,
        "human_verified": row.get("human_verified") is True,
        "reference_verified": row.get("reference_verified") is True,
        "permission_status": permission_status,
        "split": str((decision or {}).get("split") or "train"),
        "notes": str((decision or {}).get("notes") or row.get("notes") or ""),
        "trainable": True,
        "training_target": "audio_separator_or_stem_adapter",
        "safety": {
            "not_release_artifact": True,
            "requires_eval_before_promotion": True,
            "approved_by_recovery_decision": bool(decision),
        },
    }


def ingest_generic_audio_training_recovery_decisions(
    *,
    recovery_queue_jsonl: Path,
    decisions_jsonl: Path | None = None,
    output_dir: Path,
    include_ready_without_decision: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    progress(
        progress_path,
        "start",
        recovery_queue_jsonl=str(recovery_queue_jsonl),
        decisions_jsonl=str(decisions_jsonl) if decisions_jsonl else "",
    )
    decisions = load_decisions(decisions_jsonl) if decisions_jsonl else {}
    approved_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    seen_queue_ids: set[str] = set()

    for row in iter_jsonl(recovery_queue_jsonl):
        row_id = str(row.get("id") or "")
        seen_queue_ids.add(row_id)
        decision = decisions.get(row_id)
        decision_value = str((decision or {}).get("decision") or "").strip().lower()
        should_consider = decision_value in APPROVE_DECISIONS or (include_ready_without_decision and row.get("ready_for_audio_training") is True)
        blockers = remaining_blockers(row, decision)
        if should_consider and not blockers:
            approved_rows.append(approved_example(row, decision))
        elif decision is not None or row.get("ready_for_audio_training") is True:
            rejected_rows.append(
                {
                    "schema": "generic_audio_training_recovery_rejected_row.v1",
                    "id": row_id,
                    "decision": decision_value or "implicit_ready_import_failed",
                    "remaining_blockers": blockers,
                    "source_file": row.get("source_file"),
                    "source_audio_path": row.get("source_audio_path"),
                    "target_audio_path": row.get("target_audio_path"),
                }
            )

    unknown = sorted(set(decisions) - seen_queue_ids)
    if unknown:
        raise RecoveryDecisionError(f"unknown_recovery_row_id:{unknown[0]}")

    approved_rows.sort(key=lambda item: (item["track_id"], item["stem_type"], item["id"]))
    rejected_rows.sort(key=lambda item: item["id"])
    manifest = {
        "schema": "generic_audio_training_recovery_decision_manifest.v1",
        "manifest_type": "generic_audio_training_recovery_decisions",
        "recovery_queue_jsonl": str(recovery_queue_jsonl),
        "decisions_jsonl": str(decisions_jsonl) if decisions_jsonl else "",
        "output_dir": str(output_dir),
        "fingerprint": stable_hash(
            {
                "approved": [
                    {
                        "id": row["id"],
                        "track_id": row["track_id"],
                        "stem_type": row["stem_type"],
                        "target_audio_sha256": row["target_audio_sha256"],
                    }
                    for row in approved_rows
                ],
                "rejected": rejected_rows,
            }
        ),
        "summary": {
            "decision_rows": len(decisions),
            "approved_audio_training_examples": len(approved_rows),
            "rejected_rows": len(rejected_rows),
            "include_ready_without_decision": include_ready_without_decision,
        },
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "recovery_decisions_not_for_listen_pack",
        "release_status": "training_dataset_artifact_only",
        "release_ready": False,
        "outputs": {
            "approved_audio_training_examples_jsonl": str(output_dir / APPROVED_NAME),
            "rejected_rows_jsonl": str(output_dir / REJECTED_NAME),
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "progress_jsonl": str(progress_path),
        },
        "safety": {
            "metadata_only": True,
            "no_training_started": True,
            "requires_explicit_approval_for_blocked_rows": True,
            "not_release_artifact": True,
        },
    }
    write_jsonl(output_dir / APPROVED_NAME, approved_rows)
    write_jsonl(output_dir / REJECTED_NAME, rejected_rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress(progress_path, "finish", **manifest["summary"])
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest recovery queue decisions into approved audio training examples.")
    parser.add_argument("--recovery-queue-jsonl", type=Path, required=True)
    parser.add_argument("--decisions-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-include-ready-without-decision", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="ingest_generic_audio_training_recovery_decisions.py",
        entrypoint_path="/app/tools/ingest_generic_audio_training_recovery_decisions.py",
    )
    args = parse_args(argv)
    manifest = ingest_generic_audio_training_recovery_decisions(
        recovery_queue_jsonl=args.recovery_queue_jsonl,
        decisions_jsonl=args.decisions_jsonl,
        output_dir=args.output_dir,
        include_ready_without_decision=not bool(args.no_include_ready_without_decision),
    )
    print(json.dumps({"summary": manifest["summary"], "fingerprint": manifest["fingerprint"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
