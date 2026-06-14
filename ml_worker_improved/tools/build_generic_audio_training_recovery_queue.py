#!/usr/bin/env python3
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

from tools.docker_runtime import require_docker_runtime  # noqa: E402

QUEUE_NAME = "generic_audio_training_recovery_queue.jsonl"
DECISION_TEMPLATE_NAME = "generic_audio_training_recovery_decision_template.jsonl"
MANIFEST_NAME = "generic_audio_training_recovery_manifest.json"
PROGRESS_NAME = "generic_audio_training_recovery_progress.jsonl"

POSITIVE_RATINGS = {"ideal", "good", "accepted", "accept", "pass", "approved", "better", "prefer", "preferred"}
NEGATIVE_RATINGS = {"bad", "reject", "rejected", "fail", "failed", "poor", "worse"}
TRAINABLE_PERMISSION_STATUSES = {"owned", "licensed", "cleared", "project_owned"}
GROUND_TRUTH_PERMISSION_STATUSES = {"owned", "licensed", "cleared", "project_owned", "user_owned"}

SOURCE_KEYS = ("source_audio_path", "source_mix_path", "mix_path")
TARGET_KEYS = (
    "target_audio_path",
    "candidate_audio_path",
    "candidate_clip_path",
    "source_artifact",
    "selected_audio_path",
    "baseline_clip_path",
)

ACCEPTED_PRESENTATION_STATUSES = {"accepted_for_listen_pack"}

MANIFEST_SOURCE_KEYS = ("source_audio_path", "source_mix_path", "mix_path", "raw_audio_path")
STEM_MANIFEST_SOURCE_KEYS = ("source_audio_path", "source_mix_path", "mix_path", "raw_audio_path", "audio_path")
MANIFEST_GENERIC_TARGET_KEYS = ("selected_audio_path", "target_audio_path", "candidate_audio_path")
MANIFEST_TARGET_KEYS_BY_STEM = {
    "vocal": ("selected_vocal_path", "vocal_path", "listen_vocals_flac", "listen_selected_vocals_flac"),
    "minus": ("selected_minus_path", "minus_path", "instrumental_path", "listen_minus_flac", "listen_instrumental_flac"),
}

START_TIME = time.monotonic()


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


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


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
        "gpu_inference_skip_reason": "recovery_queue_metadata_scan_only",
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


def first_text(row: dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def nested_first_text(row: dict[str, Any], keys: Sequence[str]) -> str:
    value = first_text(row, keys)
    if value:
        return value
    for child_key in ("selected_artifacts", "evidence_paths"):
        child = row.get(child_key)
        if isinstance(child, dict):
            value = first_text(child, keys)
            if value:
                return value
    accepted_rows = row.get("accepted_listen_preview_rows")
    if isinstance(accepted_rows, list):
        for item in accepted_rows:
            if not isinstance(item, dict):
                continue
            value = nested_first_text(item, keys)
            if value:
                return value
    return ""


def normalized_stem(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "vocals":
        return "vocal"
    if text == "instrumental":
        return "minus"
    return text or "unknown"


def rating_of(row: dict[str, Any]) -> str:
    return str(
        row.get("rating")
        or row.get("human_rating")
        or row.get("human_ab_verdict")
        or row.get("verdict")
        or row.get("decision")
        or row.get("reviewer_decision")
        or ""
    ).strip().lower()


def feedback_verified(row: dict[str, Any]) -> bool:
    return row.get("human_verified") is True or row.get("listener_verified") is True


def permission_of(row: dict[str, Any]) -> str:
    return str(row.get("permission_status") or "").strip().lower()


def docker_visible_path(path: Any, *, base_dir: Path | None = None) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    if text.startswith("/"):
        return text
    if base_dir is not None:
        return str(base_dir / text).replace("\\", "/")
    return text


def normalized_path_text(path: str) -> str:
    return str(path or "").replace("\\", "/")


def manifest_path_from_feedback(feedback_path: Path, row: dict[str, Any]) -> Path | None:
    manifest_text = first_text(row, ("manifest_path", "listen_manifest", "listen_manifest_path"))
    if not manifest_text:
        return None
    manifest_path = Path(manifest_text)
    if manifest_path.is_absolute():
        return manifest_path
    return feedback_path.parent / manifest_path


def manifest_accepted_for_training_recovery(payload: dict[str, Any]) -> bool:
    return (
        payload.get("delivery_allowed") is True
        and str(payload.get("delivery_status") or "").lower() == "accepted"
        and str(payload.get("presentation_status") or "").lower() in ACCEPTED_PRESENTATION_STATUSES
    )


def audio_paths_from_listen_manifest(manifest_path: Path, stem_type: str) -> tuple[str, str, dict[str, Any]]:
    payload = read_json(manifest_path)
    if not payload or not manifest_accepted_for_training_recovery(payload):
        return "", "", {}
    source_audio_path = docker_visible_path(nested_first_text(payload, MANIFEST_SOURCE_KEYS), base_dir=manifest_path.parent)
    target_keys = (*MANIFEST_TARGET_KEYS_BY_STEM.get(stem_type, ()), *MANIFEST_GENERIC_TARGET_KEYS)
    target_audio_path = docker_visible_path(nested_first_text(payload, target_keys), base_dir=manifest_path.parent)
    return source_audio_path, target_audio_path, payload


def stem_type_from_stem_manifest(payload: dict[str, Any], target_audio_path: str, *, base_dir: Path) -> str:
    stems = payload.get("stems")
    if not isinstance(stems, dict):
        return "unknown"
    target_text = normalized_path_text(target_audio_path)
    for stem_name, stem_payload in stems.items():
        if not isinstance(stem_payload, dict):
            continue
        stem_path = docker_visible_path(stem_payload.get("path"), base_dir=base_dir)
        if normalized_path_text(stem_path) == target_text:
            return normalized_stem(stem_name)
    if len(stems) == 1:
        return normalized_stem(next(iter(stems.keys())))
    return "unknown"


def source_audio_path_from_stem_manifest(payload: dict[str, Any], *, base_dir: Path) -> str:
    source_audio_path = docker_visible_path(nested_first_text(payload, STEM_MANIFEST_SOURCE_KEYS), base_dir=base_dir)
    if source_audio_path:
        return source_audio_path
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        return ""
    for key in ("source_mix", "source_audio", "mix", "raw_audio"):
        value = inputs.get(key)
        if isinstance(value, dict):
            path_text = first_text(value, ("path", *STEM_MANIFEST_SOURCE_KEYS))
            if path_text:
                return docker_visible_path(path_text, base_dir=base_dir)
        elif isinstance(value, str) and value.strip():
            return docker_visible_path(value, base_dir=base_dir)
    return ""


def source_recovery_from_candidate_map(target_audio_path: str, expected_source_sha256: Any) -> dict[str, str]:
    if not target_audio_path:
        return {}
    target_path = Path(target_audio_path)
    candidate_map_path = target_path.parent / "candidate_map.json"
    candidate_map = read_json(candidate_map_path)
    if not candidate_map:
        return {}
    mapped_target = candidate_map.get(target_path.name)
    if not isinstance(mapped_target, str) or not mapped_target.strip():
        return {}
    mapped_target_path = docker_visible_path(mapped_target, base_dir=candidate_map_path.parent)
    stem_manifest_path = Path(mapped_target_path).parent / "stem_manifest.json"
    stem_manifest = read_json(stem_manifest_path)
    if not stem_manifest:
        return {}
    expected_sha = str(expected_source_sha256 or "").strip()
    manifest_sha = str(stem_manifest.get("source_audio_sha256") or "").strip()
    if expected_sha and manifest_sha and expected_sha != manifest_sha:
        return {}
    source_audio_path = source_audio_path_from_stem_manifest(stem_manifest, base_dir=stem_manifest_path.parent)
    if not source_audio_path:
        return {}
    recovered_stem = stem_type_from_stem_manifest(stem_manifest, mapped_target_path, base_dir=stem_manifest_path.parent)
    return {
        "source_audio_path": source_audio_path,
        "stem_type": recovered_stem,
        "source_recovery_method": "candidate_map_stem_manifest",
        "source_recovery_candidate_map_path": str(candidate_map_path),
        "source_recovery_manifest_path": str(stem_manifest_path),
        "source_recovery_mapped_target_path": mapped_target_path,
    }


def path_exists(path: str) -> bool:
    return bool(path) and Path(path).exists()


def blockers_for_candidate(
    *,
    source_audio_path: str,
    target_audio_path: str,
    human_verified: bool,
    reference_verified: bool,
    rating: str,
    permission_status: str,
) -> list[str]:
    blockers: list[str] = []
    if not human_verified and not reference_verified:
        blockers.append("missing_human_or_reference_verification")
    if rating in NEGATIVE_RATINGS:
        blockers.append("negative_human_rating")
    if rating and rating not in POSITIVE_RATINGS and not reference_verified:
        blockers.append("rating_not_positive")
    if not permission_status:
        blockers.append("missing_permission_status")
    elif permission_status not in TRAINABLE_PERMISSION_STATUSES and permission_status not in GROUND_TRUTH_PERMISSION_STATUSES:
        blockers.append("permission_not_trainable")
    if not source_audio_path:
        blockers.append("missing_source_audio_path")
    elif not path_exists(source_audio_path):
        blockers.append("source_audio_not_found")
    if not target_audio_path:
        blockers.append("missing_target_audio_path")
    elif not path_exists(target_audio_path):
        blockers.append("target_audio_not_found")
    return blockers


def recommended_actions(blockers: Sequence[str]) -> list[str]:
    if "synthetic_fixture_not_for_production_training" in blockers:
        return ["mark_synthetic_training_allowed_or_keep_as_fixture"]
    actions: list[str] = []
    if "missing_permission_status" in blockers or "permission_not_trainable" in blockers:
        actions.append("attach_trainable_permission_status_or_exclude")
    if "missing_human_or_reference_verification" in blockers or "rating_not_positive" in blockers:
        actions.append("attach_listener_or_reference_verification")
    if "missing_source_audio_path" in blockers or "source_audio_not_found" in blockers:
        actions.append("recover_docker_visible_source_audio_path")
    if "missing_target_audio_path" in blockers or "target_audio_not_found" in blockers:
        actions.append("recover_docker_visible_target_audio_path")
    if not actions:
        actions.append("import_to_audio_training_examples")
    return actions


def source_domain_for_feedback(row: dict[str, Any]) -> str:
    if row.get("human_verified") is True or row.get("listener_verified") is True:
        return "human_feedback_audio_pair"
    return "feedback_audio_pair"


def source_domain_for_ground_truth(contract: dict[str, Any]) -> str:
    source_text = str(contract.get("source") or "").lower()
    method_text = str(contract.get("verification_method") or "").lower()
    combined = f"{source_text} {method_text}"
    if "synthetic_deterministic_signal_generation" in combined or "run_ground_truth_quality_gate" in combined:
        return "synthetic_fixture"
    if "fl studio" in combined or "supplied by user" in combined:
        return "user_supplied_flstudio_music"
    return "ground_truth_manifest"


def recovery_priority(row: dict[str, Any]) -> str:
    blockers = set(row.get("blockers") or [])
    domain = str(row.get("source_domain") or "")
    permission_only = blockers in ({"missing_permission_status"}, {"permission_not_trainable"})
    if domain == "synthetic_fixture":
        return "P4_synthetic_fixture_not_production_training"
    if domain == "human_feedback_audio_pair" and permission_only:
        return "P0_permission_only_verified_feedback_pair"
    if domain == "user_supplied_flstudio_music" and permission_only:
        return "P1_permission_only_verified_music_gt"
    if "missing_source_audio_path" in blockers or "missing_target_audio_path" in blockers:
        return "P3_recover_audio_paths"
    return "P2_review_blockers"


def decision_template_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "generic_audio_training_recovery_decision_template_row.v1",
        "recovery_row_id": row["id"],
        "decision": "needs_more_evidence",
        "recommended_permission_status": "",
        "allowed_permission_statuses": ["owned", "licensed", "cleared", "project_owned"],
        "track_id": row["track_id"],
        "stem_type": row["stem_type"],
        "source_domain": row["source_domain"],
        "recovery_priority": row["recovery_priority"],
        "blockers": row["blockers"],
        "recommended_actions": row["recommended_actions"],
    }


def decision_template_sort_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["recovery_priority"]),
        str(row["track_id"]),
        str(row["stem_type"]),
        str(row["recovery_row_id"]),
    )


def feedback_recovery_row(path: Path, index: int, row: dict[str, Any]) -> dict[str, Any] | None:
    rating = rating_of(row)
    human_verified = feedback_verified(row)
    if not human_verified and rating not in POSITIVE_RATINGS:
        return None
    permission_status = permission_of(row)
    stem_type = normalized_stem(row.get("stem_type") or row.get("target_stem") or row.get("stem"))
    source_audio_path = docker_visible_path(first_text(row, SOURCE_KEYS))
    target_audio_path = docker_visible_path(first_text(row, TARGET_KEYS))
    source_manifest_path = manifest_path_from_feedback(path, row)
    source_manifest_payload: dict[str, Any] = {}
    if source_manifest_path and (not source_audio_path or not target_audio_path):
        manifest_source_path, manifest_target_path, source_manifest_payload = audio_paths_from_listen_manifest(
            source_manifest_path,
            stem_type,
        )
        source_audio_path = source_audio_path or manifest_source_path
        target_audio_path = target_audio_path or manifest_target_path
    source_recovery: dict[str, str] = {}
    if target_audio_path and not source_audio_path:
        source_recovery = source_recovery_from_candidate_map(target_audio_path, row.get("source_audio_sha256"))
        source_audio_path = source_recovery.get("source_audio_path") or source_audio_path
        recovered_stem_type = normalized_stem(source_recovery.get("stem_type"))
        if stem_type == "unknown" and recovered_stem_type != "unknown":
            stem_type = recovered_stem_type
    blockers = blockers_for_candidate(
        source_audio_path=source_audio_path,
        target_audio_path=target_audio_path,
        human_verified=human_verified,
        reference_verified=False,
        rating=rating,
        permission_status=permission_status,
    )
    candidate = {
        "schema": "generic_audio_training_recovery_queue_row.v1",
        "id": stable_hash(["feedback", str(path), index, source_audio_path, target_audio_path])[:16],
        "source_type": "human_feedback_source_target",
        "source_file": str(path),
        "source_row_index": index,
        "track_id": str(row.get("track_id") or source_manifest_payload.get("track_id") or row.get("source_audio_sha256") or ""),
        "stem_type": stem_type,
        "source_audio_path": source_audio_path,
        "target_audio_path": target_audio_path,
        "source_manifest_path": str(source_manifest_path) if source_manifest_path else "",
        "source_audio_exists": path_exists(source_audio_path),
        "target_audio_exists": path_exists(target_audio_path),
        "rating": rating,
        "human_verified": human_verified,
        "reference_verified": False,
        "permission_status": permission_status,
        "blockers": blockers,
        "ready_for_audio_training": not blockers,
        "notes": str(row.get("notes") or row.get("reason") or ""),
    }
    if source_recovery:
        candidate.update({key: value for key, value in source_recovery.items() if key != "stem_type"})
    candidate["source_domain"] = source_domain_for_feedback(row)
    candidate["recovery_priority"] = recovery_priority(candidate)
    candidate["recommended_actions"] = recommended_actions(candidate["blockers"])
    return candidate


def permission_from_ground_truth(payload: dict[str, Any]) -> str:
    report = payload.get("report_alignment_contract")
    if not isinstance(report, dict):
        report = {}
    value = str(report.get("rights_status") or payload.get("rights_status") or "").strip().lower()
    if value == "user_owned":
        return "project_owned"
    return value


def ground_truth_recovery_rows(path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    contract = payload.get("ground_truth_contract")
    if not isinstance(contract, dict):
        return []
    mix = payload.get("mix")
    stems = payload.get("stems")
    if not isinstance(mix, dict) or not isinstance(stems, dict):
        return []
    reference_verified = contract.get("verified") is True
    permission_status = permission_from_ground_truth(payload)
    source_audio_path = docker_visible_path(mix.get("path"), base_dir=path.parent)
    rows: list[dict[str, Any]] = []
    for stem_name, stem_payload in stems.items():
        if not isinstance(stem_payload, dict):
            continue
        stem_type = normalized_stem(stem_name)
        target_audio_path = docker_visible_path(stem_payload.get("path"), base_dir=path.parent)
        blockers = blockers_for_candidate(
            source_audio_path=source_audio_path,
            target_audio_path=target_audio_path,
            human_verified=False,
            reference_verified=reference_verified,
            rating="reference_verified" if reference_verified else "",
            permission_status=permission_status,
        )
        source_domain = source_domain_for_ground_truth(contract)
        if source_domain == "synthetic_fixture":
            blockers.append("synthetic_fixture_not_for_production_training")
        candidate = {
            "schema": "generic_audio_training_recovery_queue_row.v1",
            "id": stable_hash(["ground_truth", str(path), stem_type, target_audio_path])[:16],
            "source_type": "ground_truth_manifest",
            "source_file": str(path),
            "source_row_index": 0,
            "track_id": str(payload.get("track_name") or path.parent.name),
            "stem_type": stem_type,
            "source_audio_path": source_audio_path,
            "target_audio_path": target_audio_path,
            "source_audio_exists": path_exists(source_audio_path),
            "target_audio_exists": path_exists(target_audio_path),
            "rating": "reference_verified" if reference_verified else "",
            "human_verified": False,
            "reference_verified": reference_verified,
            "permission_status": permission_status,
            "blockers": blockers,
            "ready_for_audio_training": not blockers,
            "notes": f"ground_truth_contract_verified={reference_verified}",
            "source_domain": source_domain,
        }
        candidate["recovery_priority"] = recovery_priority(candidate)
        candidate["recommended_actions"] = recommended_actions(candidate["blockers"])
        rows.append(
            candidate
        )
    return rows


def feedback_paths_from_corpus_manifest(corpus_manifest_json: Path) -> list[Path]:
    payload = read_json(corpus_manifest_json) or {}
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        return []
    result: list[Path] = []
    for raw in inputs.get("all_human_feedback_jsonls") or inputs.get("human_feedback_jsonls") or []:
        if not isinstance(raw, str) or not raw:
            continue
        result.append(Path(raw))
    return result


def build_generic_audio_training_recovery_queue(
    *,
    artifact_root: Path,
    output_dir: Path,
    corpus_manifest_json: Path | None = None,
    feedback_jsonls: Sequence[Path] = (),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    progress(progress_path, "start", artifact_root=str(artifact_root))
    manifest_feedback_paths = feedback_paths_from_corpus_manifest(corpus_manifest_json) if corpus_manifest_json else []
    feedback_paths = sorted({path for path in [*manifest_feedback_paths, *feedback_jsonls]})

    rows: list[dict[str, Any]] = []
    feedback_rows_scanned = 0
    for path in feedback_paths:
        if not path.exists():
            continue
        for index, row in enumerate(iter_jsonl(path), start=1):
            feedback_rows_scanned += 1
            candidate = feedback_recovery_row(path, index, row)
            if candidate is not None:
                rows.append(candidate)

    ground_truth_paths = sorted(path for path in artifact_root.rglob("ground_truth_manifest.json") if path.is_file())
    for path in ground_truth_paths:
        payload = read_json(path)
        if payload is not None:
            rows.extend(ground_truth_recovery_rows(path, payload))

    rows.sort(key=lambda item: (item["ready_for_audio_training"] is False, item["source_type"], item["track_id"], item["stem_type"], item["id"]))
    decision_template_rows = [decision_template_row(row) for row in rows if not row["ready_for_audio_training"]]
    decision_template_rows.sort(key=decision_template_sort_key)
    summary = {
        "feedback_files": len(feedback_paths),
        "feedback_rows_scanned": feedback_rows_scanned,
        "ground_truth_manifests_scanned": len(ground_truth_paths),
        "recovery_rows": len(rows),
        "ready_for_audio_training": sum(1 for row in rows if row["ready_for_audio_training"]),
        "blocked_rows": sum(1 for row in rows if not row["ready_for_audio_training"]),
        "decision_template_rows": len(decision_template_rows),
        "missing_permission_status": sum(1 for row in rows if "missing_permission_status" in row["blockers"]),
        "missing_source_audio_path": sum(1 for row in rows if "missing_source_audio_path" in row["blockers"]),
        "missing_target_audio_path": sum(1 for row in rows if "missing_target_audio_path" in row["blockers"]),
        "source_audio_not_found": sum(1 for row in rows if "source_audio_not_found" in row["blockers"]),
        "target_audio_not_found": sum(1 for row in rows if "target_audio_not_found" in row["blockers"]),
        "source_domain_counts": dict(Counter(str(row["source_domain"]) for row in rows)),
        "recovery_priority_counts": dict(Counter(str(row["recovery_priority"]) for row in rows)),
    }
    manifest = {
        "schema": "generic_audio_training_recovery_manifest.v1",
        "manifest_type": "generic_audio_training_recovery_queue",
        "artifact_root": str(artifact_root),
        "output_dir": str(output_dir),
        "fingerprint": stable_hash(
            [
                {
                    "id": row["id"],
                    "ready_for_audio_training": row["ready_for_audio_training"],
                    "blockers": row["blockers"],
                }
                for row in rows
            ]
        ),
        "summary": summary,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "recovery_queue_not_for_listen_pack",
        "release_status": "planning_artifact_only",
        "release_ready": False,
        "inputs": {
            "corpus_manifest_json": str(corpus_manifest_json) if corpus_manifest_json else "",
            "feedback_jsonls": [str(path) for path in feedback_jsonls],
            "all_feedback_jsonls": [str(path) for path in feedback_paths],
        },
        "outputs": {
            "queue_jsonl": str(output_dir / QUEUE_NAME),
            "decision_template_jsonl": str(output_dir / DECISION_TEMPLATE_NAME),
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "progress_jsonl": str(progress_path),
        },
        "safety": {
            "metadata_only": True,
            "no_training_started": True,
            "not_release_artifact": True,
            "requires_permission_before_training": True,
        },
    }
    write_jsonl(output_dir / QUEUE_NAME, rows)
    write_jsonl(output_dir / DECISION_TEMPLATE_NAME, decision_template_rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress(progress_path, "finish", **summary)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build recovery queue for generic audio training source/target evidence.")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus-manifest-json", type=Path)
    parser.add_argument("--feedback-jsonl", type=Path, action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_generic_audio_training_recovery_queue.py",
        entrypoint_path="/app/tools/build_generic_audio_training_recovery_queue.py",
    )
    args = parse_args(argv)
    manifest = build_generic_audio_training_recovery_queue(
        artifact_root=args.artifact_root,
        output_dir=args.output_dir,
        corpus_manifest_json=args.corpus_manifest_json,
        feedback_jsonls=args.feedback_jsonl,
    )
    print(json.dumps({"summary": manifest["summary"], "fingerprint": manifest["fingerprint"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
