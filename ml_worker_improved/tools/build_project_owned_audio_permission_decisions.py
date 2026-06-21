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
MANIFEST_NAME = "project_owned_audio_permission_decision_manifest.json"
DECISIONS_NAME = "project_owned_audio_permission_decisions.jsonl"
REJECTED_NAME = "project_owned_audio_permission_rejected.jsonl"
PROGRESS_NAME = "project_owned_audio_permission_decision_progress.jsonl"
SHORT_MANIFEST_NAME = "perm.json"
SHORT_DECISIONS_NAME = "dec.jsonl"
SHORT_REJECTED_NAME = "rej.jsonl"
SHORT_PROGRESS_NAME = "progress.jsonl"
PERMISSION_ONLY_BLOCKERS = ({"missing_permission_status"}, {"permission_not_trainable"})
DEFAULT_ALLOWED_SOURCE_DOMAIN = "user_supplied_flstudio_music"
POSITIVE_RATINGS = {"accepted", "good", "great", "ideal", "better", "best", "reference_verified"}


def output_names(short_output_names: bool) -> dict[str, str]:
    if short_output_names:
        return {
            "manifest": SHORT_MANIFEST_NAME,
            "decisions": SHORT_DECISIONS_NAME,
            "rejected": SHORT_REJECTED_NAME,
            "progress": SHORT_PROGRESS_NAME,
        }
    return {
        "manifest": MANIFEST_NAME,
        "decisions": DECISIONS_NAME,
        "rejected": REJECTED_NAME,
        "progress": PROGRESS_NAME,
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


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "metadata_only_project_owned_permission_attestation",
        "expected_acceleration_mode": "external_only_cpu_metadata_curation",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def path_exists(path_text: Any) -> bool:
    path = str(path_text or "")
    return bool(path) and Path(path).exists()


def under_allowed_source_root(path_text: str, allowed_source_roots: Sequence[Path]) -> bool:
    if not allowed_source_roots:
        return True
    path = Path(path_text)
    try:
        resolved = path.resolve()
    except OSError:
        return False
    for root in allowed_source_roots:
        try:
            root_resolved = root.resolve()
        except OSError:
            continue
        if resolved == root_resolved or root_resolved in resolved.parents:
            return True
    return False


def attestation_rejection_reasons(
    row: dict[str, Any],
    *,
    allowed_source_domains: set[str],
    allowed_source_roots: Sequence[Path],
    allow_human_feedback_pairs: bool,
) -> list[str]:
    reasons: list[str] = []
    blockers = set(row.get("blockers") or [])
    source_file = str(row.get("source_file") or "")
    source_domain = str(row.get("source_domain") or "")
    source_type = str(row.get("source_type") or "")
    is_ground_truth = source_domain in allowed_source_domains and source_type == "ground_truth_manifest"
    is_human_feedback = (
        allow_human_feedback_pairs
        and source_domain == "human_feedback_audio_pair"
        and source_type == "human_feedback_source_target"
        and row.get("human_verified") is True
        and str(row.get("rating") or "").strip().lower() in POSITIVE_RATINGS
    )
    if row.get("schema") != "generic_audio_training_recovery_queue_row.v1":
        reasons.append("unexpected_schema")
    if not is_ground_truth and not is_human_feedback:
        reasons.append("source_domain_not_allowed")
    if not is_ground_truth and not is_human_feedback:
        reasons.append("source_type_not_ground_truth_manifest")
    if is_ground_truth and row.get("reference_verified") is not True:
        reasons.append("not_reference_verified")
    if is_human_feedback and row.get("human_verified") is not True:
        reasons.append("not_human_verified")
    if blockers not in PERMISSION_ONLY_BLOCKERS:
        reasons.append("not_permission_only_blocked")
    permission_status = str(row.get("permission_status") or "").strip().lower()
    if permission_status and blockers == {"missing_permission_status"}:
        reasons.append("permission_status_already_set")
    if str(row.get("track_id") or "") == "":
        reasons.append("missing_track_id")
    if not source_file:
        reasons.append("missing_source_file")
    elif not under_allowed_source_root(source_file, allowed_source_roots):
        reasons.append("source_file_not_under_allowed_root")
    if not path_exists(row.get("source_audio_path")):
        reasons.append("source_audio_not_found")
    if not path_exists(row.get("target_audio_path")):
        reasons.append("target_audio_not_found")
    if str(row.get("source_domain") or "") == "synthetic_fixture":
        reasons.append("synthetic_fixture_not_allowed")
    return reasons


def decision_row(row: dict[str, Any]) -> dict[str, Any]:
    source_domain = str(row.get("source_domain") or "")
    notes = (
        "project_owned_attestation:human_verified_project_audio_pair"
        if source_domain == "human_feedback_audio_pair"
        else "project_owned_attestation:verified_user_supplied_flstudio_ground_truth"
    )
    return {
        "recovery_row_id": str(row["id"]),
        "decision": "approve_for_training",
        "permission_status": "project_owned",
        "split": str(row.get("split") or "train"),
        "notes": notes,
    }


def build_project_owned_audio_permission_decisions(
    *,
    recovery_queue_jsonl: Path,
    output_dir: Path,
    allowed_source_roots: Sequence[Path] = (),
    allowed_source_domains: Sequence[str] = (DEFAULT_ALLOWED_SOURCE_DOMAIN,),
    allow_human_feedback_pairs: bool = False,
    short_output_names: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = output_names(short_output_names)
    manifest_path = output_dir / names["manifest"]
    decisions_path = output_dir / names["decisions"]
    rejected_path = output_dir / names["rejected"]
    progress_path = output_dir / names["progress"]
    if progress_path.exists():
        progress_path.unlink()
    allowed_domain_set = {str(domain) for domain in allowed_source_domains}
    progress(
        progress_path,
        "start",
        recovery_queue_jsonl=str(recovery_queue_jsonl),
        allowed_source_roots=[str(path) for path in allowed_source_roots],
        allowed_source_domains=sorted(allowed_domain_set),
        allow_human_feedback_pairs=allow_human_feedback_pairs,
    )
    decisions: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in iter_jsonl(recovery_queue_jsonl):
        reasons = attestation_rejection_reasons(
            row,
            allowed_source_domains=allowed_domain_set,
            allowed_source_roots=allowed_source_roots,
            allow_human_feedback_pairs=allow_human_feedback_pairs,
        )
        if not reasons:
            decisions.append(decision_row(row))
        else:
            rejected.append(
                {
                    "schema": "project_owned_audio_permission_rejected_row.v1",
                    "recovery_row_id": str(row.get("id") or ""),
                    "track_id": row.get("track_id"),
                    "stem_type": row.get("stem_type"),
                    "source_domain": row.get("source_domain"),
                    "source_type": row.get("source_type"),
                    "source_file": row.get("source_file"),
                    "rejection_reasons": reasons,
                }
            )
    decisions.sort(key=lambda item: (item["recovery_row_id"], item["split"]))
    rejected.sort(key=lambda item: item["recovery_row_id"])
    manifest = {
        "schema": "project_owned_audio_permission_decision_manifest.v1",
        "manifest_type": "project_owned_audio_permission_decisions",
        "recovery_queue_jsonl": str(recovery_queue_jsonl),
        "output_dir": str(output_dir),
        "short_output_names": bool(short_output_names),
        "fingerprint": stable_hash({"decisions": decisions, "rejected": rejected}),
        "decision_rows": len(decisions),
        "rejected_rows": len(rejected),
        "allowed_source_roots": [str(path) for path in allowed_source_roots],
        "allowed_source_domains": sorted(allowed_domain_set),
        "allow_human_feedback_pairs": bool(allow_human_feedback_pairs),
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "project_owned_permission_decisions_not_for_listen_pack",
        "release_status": "training_permission_attestation_only",
        "release_ready": False,
        "manifest_json": str(manifest_path),
        "decisions_jsonl": str(decisions_path),
        "rejected_jsonl": str(rejected_path),
        "progress_jsonl": str(progress_path),
        "safety": {
            "metadata_only": True,
            "attests_only_reference_verified_ground_truth": True,
            "human_feedback_pairs_require_explicit_opt_in": True,
            "synthetic_fixtures_not_allowed": True,
            "not_release_artifact": True,
        },
    }
    write_jsonl(decisions_path, decisions)
    write_jsonl(rejected_path, rejected)
    write_json(manifest_path, manifest)
    progress(progress_path, "finish", decision_rows=len(decisions), rejected_rows=len(rejected))
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build project-owned permission decisions for verified ground-truth recovery rows.")
    parser.add_argument("--recovery-queue-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allowed-source-root", type=Path, action="append", default=[])
    parser.add_argument("--allowed-source-domain", action="append", default=[DEFAULT_ALLOWED_SOURCE_DOMAIN])
    parser.add_argument("--allow-human-feedback-pairs", action="store_true")
    parser.add_argument("--short-output-names", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_project_owned_audio_permission_decisions.py",
        entrypoint_path="/app/tools/build_project_owned_audio_permission_decisions.py",
    )
    args = parse_args(argv)
    manifest = build_project_owned_audio_permission_decisions(
        recovery_queue_jsonl=args.recovery_queue_jsonl,
        output_dir=args.output_dir,
        allowed_source_roots=args.allowed_source_root,
        allowed_source_domains=args.allowed_source_domain,
        allow_human_feedback_pairs=args.allow_human_feedback_pairs,
        short_output_names=bool(args.short_output_names),
    )
    print(
        json.dumps(
            {
                "manifest_path": manifest["manifest_json"],
                "decision_rows": manifest["decision_rows"],
                "rejected_rows": manifest["rejected_rows"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
