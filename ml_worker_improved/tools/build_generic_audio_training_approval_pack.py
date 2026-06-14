from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime


CANDIDATES_NAME = "generic_audio_training_approval_candidates.jsonl"
DRAFT_DECISIONS_NAME = "generic_audio_training_draft_decisions.jsonl"
MANIFEST_NAME = "generic_audio_training_approval_pack_manifest.json"
PROGRESS_NAME = "generic_audio_training_approval_pack_progress.jsonl"
START_TIME = time.monotonic()

ALLOWED_PERMISSION_STATUSES = ["owned", "licensed", "cleared", "project_owned"]
APPROVAL_PRIORITIES = {
    "P0_permission_only_verified_feedback_pair": 0,
    "P1_permission_only_verified_music_gt": 1,
}


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
        "gpu_inference_skip_reason": "metadata_only_audio_training_permission_review_pack",
        "expected_acceleration_mode": "external_only_cpu_metadata_curation",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_permission_only_candidate(row: dict[str, Any]) -> bool:
    priority = str(row.get("recovery_priority") or "")
    blockers = list(row.get("blockers") or [])
    if priority not in APPROVAL_PRIORITIES:
        return False
    if set(blockers) not in ({"missing_permission_status"}, {"permission_not_trainable"}):
        return False
    if row.get("ready_for_audio_training") is True:
        return False
    if str(row.get("source_domain") or "") == "synthetic_fixture":
        return False
    if not str(row.get("source_audio_path") or "") or not str(row.get("target_audio_path") or ""):
        return False
    if not Path(str(row.get("source_audio_path"))).exists():
        return False
    if not Path(str(row.get("target_audio_path"))).exists():
        return False
    return bool(row.get("human_verified") is True or row.get("reference_verified") is True)


def priority_sort_key(row: dict[str, Any]) -> tuple[int, str, str, str]:
    return (
        APPROVAL_PRIORITIES.get(str(row.get("recovery_priority") or ""), 99),
        str(row.get("track_id") or ""),
        str(row.get("stem_type") or ""),
        str(row.get("id") or ""),
    )


def candidate_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "generic_audio_training_approval_candidate.v1",
        "id": stable_hash(["approval_candidate", row.get("id")])[:16],
        "recovery_row_id": row.get("id"),
        "approval_candidate_status": "needs_permission_review",
        "track_id": row.get("track_id"),
        "stem_type": row.get("stem_type"),
        "source_domain": row.get("source_domain"),
        "recovery_priority": row.get("recovery_priority"),
        "source_file": row.get("source_file"),
        "source_audio_path": row.get("source_audio_path"),
        "target_audio_path": row.get("target_audio_path"),
        "rating": row.get("rating"),
        "human_verified": row.get("human_verified") is True,
        "reference_verified": row.get("reference_verified") is True,
        "current_blockers": list(row.get("blockers") or []),
        "recommended_actions": list(row.get("recommended_actions") or []),
        "decision": "needs_more_evidence",
        "permission_status": "",
        "allowed_permission_statuses": list(ALLOWED_PERMISSION_STATUSES),
        "notes": "Fill permission_status only if source/target rights are trainable; otherwise keep needs_more_evidence or exclude.",
    }


def draft_decision_row(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "generic_audio_training_recovery_decision_template_row.v1",
        "recovery_row_id": candidate["recovery_row_id"],
        "decision": "needs_more_evidence",
        "permission_status": "",
        "allowed_permission_statuses": list(ALLOWED_PERMISSION_STATUSES),
        "track_id": candidate["track_id"],
        "stem_type": candidate["stem_type"],
        "source_domain": candidate["source_domain"],
        "recovery_priority": candidate["recovery_priority"],
        "notes": candidate["notes"],
    }


def build_generic_audio_training_approval_pack(
    *,
    recovery_queue_jsonl: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    progress(progress_path, "start", recovery_queue_jsonl=str(recovery_queue_jsonl))
    queue_rows = list(iter_jsonl(recovery_queue_jsonl))
    candidate_rows = [candidate_row(row) for row in sorted(queue_rows, key=priority_sort_key) if is_permission_only_candidate(row)]
    draft_rows = [draft_decision_row(row) for row in candidate_rows]
    non_candidate_rows = len(queue_rows) - len(candidate_rows)
    priority_counts = dict(sorted(Counter(str(row["recovery_priority"]) for row in candidate_rows).items()))
    source_domain_counts = dict(sorted(Counter(str(row["source_domain"]) for row in candidate_rows).items()))
    stem_counts = dict(sorted(Counter(str(row["stem_type"]) for row in candidate_rows).items()))
    unique_tracks = sorted({str(row.get("track_id") or row.get("source_audio_path") or "") for row in candidate_rows})
    summary = {
        "queue_rows": len(queue_rows),
        "candidate_rows": len(candidate_rows),
        "non_candidate_rows": non_candidate_rows,
        "unique_track_count": len(unique_tracks),
        "recovery_priority_counts": priority_counts,
        "source_domain_counts": source_domain_counts,
        "stem_counts": stem_counts,
    }
    manifest = {
        "schema": "generic_audio_training_approval_pack_manifest.v1",
        "manifest_type": "generic_audio_training_approval_pack",
        "recovery_queue_jsonl": str(recovery_queue_jsonl),
        "recovery_queue_sha256": sha256_file(recovery_queue_jsonl),
        "output_dir": str(output_dir),
        "fingerprint": stable_hash(
            [
                {
                    "recovery_row_id": row["recovery_row_id"],
                    "decision": row["decision"],
                    "permission_status": row["permission_status"],
                }
                for row in candidate_rows
            ]
        ),
        "summary": summary,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "approval_pack_not_for_listen_pack",
        "release_status": "permission_review_artifact_only",
        "release_ready": False,
        "outputs": {
            "approval_candidates_jsonl": str(output_dir / CANDIDATES_NAME),
            "draft_decisions_jsonl": str(output_dir / DRAFT_DECISIONS_NAME),
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "progress_jsonl": str(progress_path),
        },
        "safety": {
            "metadata_only": True,
            "no_training_started": True,
            "not_release_artifact": True,
            "does_not_auto_approve_permission": True,
            "excludes_synthetic_fixture_rows": True,
            "requires_ingest_decisions_before_training": True,
        },
    }
    write_jsonl(output_dir / CANDIDATES_NAME, candidate_rows)
    write_jsonl(output_dir / DRAFT_DECISIONS_NAME, draft_rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress(progress_path, "finish", **summary)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build review pack for permission-only generic audio training recovery rows.")
    parser.add_argument("--recovery-queue-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_generic_audio_training_approval_pack.py",
        entrypoint_path="/app/tools/build_generic_audio_training_approval_pack.py",
    )
    args = parse_args(argv)
    manifest = build_generic_audio_training_approval_pack(
        recovery_queue_jsonl=args.recovery_queue_jsonl,
        output_dir=args.output_dir,
    )
    print(json.dumps({"summary": manifest["summary"], "fingerprint": manifest["fingerprint"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
