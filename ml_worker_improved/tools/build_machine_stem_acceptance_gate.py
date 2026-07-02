#!/usr/bin/env python3
"""Fail-closed machine acceptance gate for short full-stem listen packs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime  # noqa: E402


START_TIME = time.monotonic()
GATE_NAME = "gate.json"
BLOCKERS_NAME = "blockers.jsonl"
PROGRESS_NAME = "progress.jsonl"
SUMMARY_NAME = "summary.md"
DEFAULT_REQUIRED_STEMS = ("vocal", "minus", "drums", "bass", "other", "guitar", "piano")
AGGREGATE_VALIDATION_PENDING_STATUS = "aggregate_validation_pending"
AGGREGATE_VALIDATION_PENDING_PRESENTATION = "aggregate_validation_pending_not_for_listen_pack"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_contract_gate",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "machine_stem_acceptance_gate_reads_existing_manifests_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "machine_stem_acceptance_gate_no_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def as_bool(value: Any) -> bool:
    return value is True


def pack_items(pack: dict[str, Any]) -> list[dict[str, Any]]:
    items = pack.get("items")
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    stems = pack.get("stems")
    if isinstance(stems, list):
        return [row for row in stems if isinstance(row, dict)]
    rows = pack.get("listen_audio_files")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def blocker(reason: str, *, path: str = "", evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "reason": reason,
        "audio_path": path,
        "evidence": evidence or {},
    }


def normalize_stem(value: Any) -> str:
    stem = str(value or "").strip().lower().replace(" ", "_")
    if stem == "vocals":
        return "vocal"
    if stem == "instrumental":
        return "minus"
    return stem


def normalize_required_stems(values: Sequence[str] | None) -> tuple[str, ...]:
    raw_values = values or DEFAULT_REQUIRED_STEMS
    normalized: list[str] = []
    for value in raw_values:
        stem = normalize_stem(value)
        if not stem:
            raise ValueError("required stem cannot be empty")
        if stem not in normalized:
            normalized.append(stem)
    if not normalized:
        raise ValueError("at least one required stem is required")
    return tuple(normalized)


def missing_required_stems(items: Sequence[dict[str, Any]], required_stems: Sequence[str]) -> list[str]:
    present = {normalize_stem(item.get("stem_type") or item.get("artifact_kind") or "") for item in items}
    return sorted(stem for stem in required_stems if stem not in present)


def pack_audio_path(raw_path: Any, *, pack_manifest_json: Path) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = pack_manifest_json.parent / path
    return path


def stem_audio_file_checks(
    items: Sequence[dict[str, Any]],
    *,
    pack_manifest_json: Path,
    required_stems: Sequence[str],
) -> list[dict[str, Any]]:
    required = set(required_stems)
    checks: list[dict[str, Any]] = []
    for item in items:
        stem = normalize_stem(item.get("stem_type") or item.get("artifact_kind") or "")
        if stem not in required:
            continue
        raw_path = item.get("audio_path") or item.get("artifact_relative_path")
        resolved = pack_audio_path(raw_path, pack_manifest_json=pack_manifest_json)
        checks.append(
            {
                "stem_type": stem,
                "audio_path": str(raw_path or ""),
                "resolved_audio_path": str(resolved or ""),
                "exists": bool(resolved and resolved.is_file()),
            }
        )
    return checks


def normalize_train_ready_decision_stem(payload: dict[str, Any]) -> str:
    row = payload.get("best_matrix_row")
    if not isinstance(row, dict):
        row = {}
    return normalize_stem(payload.get("stem") or payload.get("stem_type") or row.get("stem") or "")


def job_stems(job: dict[str, Any]) -> list[str]:
    raw = job.get("stem_types")
    if isinstance(raw, list):
        stems = [normalize_stem(item) for item in raw]
    else:
        stems = [normalize_stem(job.get("stem_type") or job.get("stem") or "")]
    return sorted({stem for stem in stems if stem})


def evaluate_training_coverage(
    *,
    required_stems: Sequence[str],
    require_training_coverage: bool,
    training_plan_json: Path | None,
    train_ready_decision_jsons: Sequence[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evidence = {
        "required": bool(require_training_coverage),
        "training_plan_json": str(training_plan_json or ""),
        "train_ready_decision_jsons": [str(path) for path in train_ready_decision_jsons],
        "train_ready_stems": [],
        "training_ready_job_stems": [],
        "blocked_training_job_stems": {},
    }
    if not require_training_coverage:
        return [], evidence

    blockers: list[dict[str, Any]] = []
    train_ready_stems: set[str] = set()
    ready_job_stems: set[str] = set()
    blocked_job_stems: dict[str, list[dict[str, Any]]] = {}
    required_stem_set = set(required_stems)

    if training_plan_json is None:
        blockers.append(blocker("training_plan_missing"))
    elif not training_plan_json.exists():
        blockers.append(blocker("training_plan_file_missing", evidence={"training_plan_json": str(training_plan_json)}))
    else:
        plan = read_json(training_plan_json)
        for job in plan.get("jobs") if isinstance(plan.get("jobs"), list) else []:
            if not isinstance(job, dict):
                continue
            status = str(job.get("status") or "")
            stems = [stem for stem in job_stems(job) if stem in required_stem_set]
            if status == "ready_for_gpu_training":
                ready_job_stems.update(stems)
                continue
            for stem in stems:
                blocked_job_stems.setdefault(stem, []).append(
                    {
                        "scope": job.get("scope"),
                        "status": status or "unknown",
                        "unique_track_count": job.get("unique_track_count"),
                        "min_unique_tracks": job.get("min_unique_tracks"),
                        "blocker_reason": job.get("blocker_reason"),
                    }
                )

    if not train_ready_decision_jsons:
        blockers.append(blocker("train_ready_decision_missing"))
    for decision_json in train_ready_decision_jsons:
        if not decision_json.exists():
            blockers.append(blocker("train_ready_decision_file_missing", evidence={"train_ready_decision_json": str(decision_json)}))
            continue
        decision = read_json(decision_json)
        stem = normalize_train_ready_decision_stem(decision)
        safe = (
            decision.get("decision") == "train_result_verified"
            and decision.get("train_ready") is True
            and not decision.get("failed_checks")
            and decision.get("delivery_allowed") is not True
            and decision.get("release_ready") is not True
            and decision.get("promotion_allowed") is not True
            and decision.get("model_registry_edited") is not True
            and decision.get("runtime_gates_changed") is not True
            and decision.get("thresholds_relaxed") is not True
        )
        if stem in required_stem_set and safe:
            train_ready_stems.add(stem)
            continue
        blockers.append(
            blocker(
                f"train_ready_decision_not_verified:{stem or 'unknown'}",
                evidence={
                    "train_ready_decision_json": str(decision_json),
                    "stem": stem,
                    "decision": decision.get("decision"),
                    "train_ready": decision.get("train_ready"),
                    "failed_checks": decision.get("failed_checks"),
                },
            )
        )

    for stem in required_stems:
        if stem not in train_ready_stems:
            for job_evidence in blocked_job_stems.get(stem, []):
                status = str(job_evidence.get("status") or "unknown")
                blockers.append(blocker(f"training_job_blocked:{stem}:{status}", evidence=job_evidence))
            blockers.append(
                blocker(
                    f"required_stem_train_ready_missing:{stem}",
                    evidence={
                        "train_ready_stems": sorted(train_ready_stems),
                        "training_ready_job_stems": sorted(ready_job_stems),
                        "blocked_training_job_statuses": sorted({str(job.get("status") or "unknown") for job in blocked_job_stems.get(stem, [])}),
                    },
                )
            )

    evidence["train_ready_stems"] = sorted(train_ready_stems)
    evidence["training_ready_job_stems"] = sorted(ready_job_stems)
    evidence["blocked_training_job_stems"] = {
        stem: sorted({str(job.get("status") or "unknown") for job in jobs})
        for stem, jobs in sorted(blocked_job_stems.items())
    }
    return blockers, evidence


def load_semantic_review_evidence(
    semantic_review_json: Path | None,
    pack_manifest_json: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    if semantic_review_json is None:
        return [], {}, {}
    if not semantic_review_json.exists():
        return [blocker("semantic_review_file_missing", evidence={"semantic_review_json": str(semantic_review_json)})], {}, {}
    review = read_json(semantic_review_json)
    blockers: list[dict[str, Any]] = []
    if review.get("review_status") != "completed":
        blockers.append(blocker("semantic_review_not_completed", evidence={"review_status": str(review.get("review_status") or "")}))
    if review.get("failed_checks"):
        blockers.append(blocker("semantic_review_failed_checks_present", evidence={"failed_checks": review.get("failed_checks")}))
    if review.get("release_ready") is True or review.get("can_publish_as_result") is True:
        blockers.append(blocker("semantic_review_release_or_publish_true"))

    rows_path = Path(str(review.get("rows_jsonl") or ""))
    if not rows_path.exists():
        blockers.append(blocker("semantic_review_rows_missing", evidence={"rows_jsonl": str(rows_path)}))
        return blockers, {}, review

    evidence_by_stem_path: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(rows_path):
        if row.get("review_status") != "accepted":
            continue
        if row.get("semantic_quality_verified") is not True:
            continue
        if row.get("blockers"):
            continue
        for evidence_path_raw in row.get("evidence_paths") or []:
            evidence_path = Path(str(evidence_path_raw))
            if not evidence_path.exists():
                blockers.append(blocker("semantic_review_evidence_missing", evidence={"evidence_path": str(evidence_path)}))
                continue
            evidence = read_json(evidence_path)
            if str(evidence.get("source_manifest") or "") != str(pack_manifest_json):
                continue
            source_stem_id = str(evidence.get("source_stem_id") or "")
            stem = normalize_stem(evidence.get("stem_type") or row.get("stem_type") or "")
            if source_stem_id and stem:
                evidence_by_stem_path[(source_stem_id, stem)] = {
                    "review_row": row,
                    "evidence": evidence,
                    "evidence_path": str(evidence_path),
                }
    return blockers, evidence_by_stem_path, review


def has_machine_quality_evidence(
    item: dict[str, Any],
    source_payloads: dict[str, dict[str, Any]],
    semantic_evidence: dict[tuple[str, str], dict[str, Any]],
) -> bool:
    stem = normalize_stem(item.get("stem_type") or item.get("artifact_kind") or "")
    for raw_path in (item.get("audio_path"), item.get("artifact_relative_path")):
        item_path = str(raw_path or "")
        if item_path and (item_path, stem) in semantic_evidence:
            return True
    quality_status = str(item.get("quality_status") or "")
    if item.get("semantic_quality_verified") is True and not quality_status.endswith("not_semantic_verified"):
        return True
    if stem == "vocal":
        source = str(item.get("source_manifest") or "")
        payload = source_payloads.get(source) or {}
        return payload.get("manifest_type") == "no_reference_vocal_minus_master_listen_preview"
    return False


def unverified_required_stems(
    items: Sequence[dict[str, Any]],
    required_stems: Sequence[str],
    source_payloads: dict[str, dict[str, Any]],
    semantic_evidence: dict[tuple[str, str], dict[str, Any]],
) -> list[str]:
    unverified: list[str] = []
    for stem in required_stems:
        if stem == "minus":
            continue
        stem_items = [item for item in items if normalize_stem(item.get("stem_type") or item.get("artifact_kind") or "") == stem]
        if not stem_items:
            continue
        if not any(has_machine_quality_evidence(item, source_payloads, semantic_evidence) for item in stem_items):
            unverified.append(stem)
    return sorted(unverified)


def source_manifest_path(raw: str) -> Path:
    return Path(str(raw or ""))


def inspect_source_master(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        return [blocker("source_manifest_missing", evidence={"source_manifest": str(path)})], {}
    payload = read_json(path)
    blockers: list[dict[str, Any]] = []
    if payload.get("manifest_type") == "no_reference_vocal_minus_master_listen_preview":
        if payload.get("delivery_status") != "accepted" or payload.get("delivery_allowed") is not True:
            blockers.append(blocker("source_master_not_accepted", evidence={"source_manifest": str(path)}))
        if payload.get("presentation_status") != "accepted_for_listen_pack":
            blockers.append(blocker("source_master_presentation_not_accepted", evidence={"source_manifest": str(path)}))
        if payload.get("release_status") != "listen_preview_only":
            blockers.append(blocker("source_master_release_status_not_listen_preview", evidence={"source_manifest": str(path)}))
        if payload.get("release_ready") is True or payload.get("can_publish_as_result") is True or payload.get("can_publish_as_release") is True:
            blockers.append(blocker("source_master_release_or_publish_true", evidence={"source_manifest": str(path)}))
        if payload.get("pair_decision") != "pair_compatible":
            blockers.append(blocker("source_master_pair_not_compatible", evidence={"source_manifest": str(path)}))
        if int(payload.get("pair_window_count") or 0) < 5:
            blockers.append(blocker("source_master_pair_window_count_too_low", evidence={"source_manifest": str(path)}))
        safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
        for key in ("training_started", "fine_tune_started", "models_changed", "model_registry_edited", "thresholds_relaxed", "cleanup_routes_added"):
            if payload.get(key) is True or safety.get(key) is True:
                blockers.append(blocker(f"source_master_{key}_true", evidence={"source_manifest": str(path)}))
    return blockers, payload


def minus_audit_rows(minus_audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows_jsonl = str(minus_audit.get("rows_jsonl") or "")
    if rows_jsonl:
        rows_path = Path(rows_jsonl)
        if rows_path.exists():
            return read_jsonl(rows_path)
    return []


def evaluate_minus_audit(minus_items: list[dict[str, Any]], minus_audit_json: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not minus_items:
        return [blocker("minus_item_missing")], {}
    if minus_audit_json is None:
        return [blocker("minus_audit_missing")], {}
    if not minus_audit_json.exists():
        return [blocker("minus_audit_file_missing", evidence={"minus_audit_json": str(minus_audit_json)})], {}

    audit = read_json(minus_audit_json)
    blockers: list[dict[str, Any]] = []
    if audit.get("audit_status") != "completed":
        blockers.append(blocker("minus_audit_not_completed", evidence={"audit_status": str(audit.get("audit_status") or "")}))
    if audit.get("failed_checks"):
        blockers.append(blocker("minus_audit_failed_checks_present", evidence={"failed_checks": audit.get("failed_checks")}))
    if int(audit.get("blocked_minus_count") or 0) > 0:
        blockers.append(blocker("minus_audit_blocked_minus_present", evidence={"blocked_minus_count": audit.get("blocked_minus_count")}))
    if int(audit.get("warning_count") or 0) > 0:
        blockers.append(blocker("minus_audit_warnings_present", evidence={"warning_count": audit.get("warning_count")}))

    rows = {str(row.get("short_audio_path") or ""): row for row in minus_audit_rows(audit)}
    for item in minus_items:
        audio_path = str(item.get("audio_path") or item.get("artifact_relative_path") or "")
        row = rows.get(audio_path)
        if row is None:
            blockers.append(blocker("minus_audit_row_missing", path=audio_path))
            continue
        if row.get("audit_decision") != "passed":
            blockers.append(blocker("minus_audit_row_not_passed", path=audio_path, evidence={"audit_decision": row.get("audit_decision"), "audit_blockers": row.get("audit_blockers")}))
        if row.get("audit_blockers"):
            blockers.append(blocker("minus_audit_row_blockers_present", path=audio_path, evidence={"audit_blockers": row.get("audit_blockers")}))
        if row.get("quality_warning_reasons"):
            blockers.append(blocker("minus_audit_row_warnings_present", path=audio_path, evidence={"quality_warning_reasons": row.get("quality_warning_reasons")}))
    return blockers, audit


def build_machine_stem_acceptance_gate(
    *,
    pack_manifest_json: Path,
    validation_json: Path,
    minus_audit_json: Path | None,
    output_dir: Path,
    required_stems: Sequence[str] | None = DEFAULT_REQUIRED_STEMS,
    semantic_review_json: Path | None = None,
    require_training_coverage: bool = False,
    training_plan_json: Path | None = None,
    train_ready_decision_jsons: Sequence[Path] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    required_stem_list = normalize_required_stems(required_stems)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(
        progress_path,
        "start",
        pack_manifest_json=str(pack_manifest_json),
        validation_json=str(validation_json),
        minus_audit_json=str(minus_audit_json or ""),
        semantic_review_json=str(semantic_review_json or ""),
        require_training_coverage=bool(require_training_coverage),
        training_plan_json=str(training_plan_json or ""),
        train_ready_decision_jsons=[str(path) for path in (train_ready_decision_jsons or [])],
        required_stems=list(required_stem_list),
    )

    pack = read_json(pack_manifest_json)
    validation = read_json(validation_json)
    items = pack_items(pack)
    blockers: list[dict[str, Any]] = []

    if pack.get("delivery_status") != AGGREGATE_VALIDATION_PENDING_STATUS:
        blockers.append(blocker("pack_not_aggregate_validation_pending", evidence={"delivery_status": pack.get("delivery_status")}))
    if pack.get("delivery_allowed") is not False:
        blockers.append(blocker("pack_delivery_allowed_not_false", evidence={"delivery_allowed": pack.get("delivery_allowed")}))
    if pack.get("presentation_status") != AGGREGATE_VALIDATION_PENDING_PRESENTATION:
        blockers.append(blocker("pack_presentation_not_aggregate_validation_pending", evidence={"presentation_status": pack.get("presentation_status")}))
    if pack.get("release_status") != AGGREGATE_VALIDATION_PENDING_STATUS:
        blockers.append(blocker("pack_release_status_not_aggregate_validation_pending", evidence={"release_status": pack.get("release_status")}))
    if pack.get("aggregate_validation_ready") is not True:
        blockers.append(blocker("pack_aggregate_validation_ready_not_true", evidence={"aggregate_validation_ready": pack.get("aggregate_validation_ready")}))
    if pack.get("listenable") is not False:
        blockers.append(blocker("pack_listenable_not_false", evidence={"listenable": pack.get("listenable")}))
    if pack.get("release_ready") is True or pack.get("can_publish_as_result") is True:
        blockers.append(blocker("pack_release_or_publish_true"))
    if validation.get("validation_status") != "passed" or validation.get("failed_checks"):
        blockers.append(blocker("pack_validation_not_passed", evidence={"validation_status": validation.get("validation_status"), "failed_checks": validation.get("failed_checks")}))

    stem_types = [normalize_stem(item.get("stem_type") or item.get("artifact_kind") or "") for item in items]
    present_stems = sorted({stem for stem in stem_types if stem})
    missing_stems = missing_required_stems(items, required_stem_list)
    for stem in missing_stems:
        blockers.append(
            blocker(
                f"required_stem_missing:{stem}",
                evidence={"required_stems": list(required_stem_list), "present_stems": present_stems},
            )
        )
    audio_file_checks = stem_audio_file_checks(items, pack_manifest_json=pack_manifest_json, required_stems=required_stem_list)
    for check_row in audio_file_checks:
        if check_row["exists"] is not True:
            stem = check_row["stem_type"]
            blockers.append(
                blocker(
                    f"required_stem_audio_file_missing:{stem}",
                    path=str(check_row["audio_path"]),
                    evidence={"resolved_audio_path": check_row["resolved_audio_path"]},
                )
            )
    vocal_items = [item for item in items if normalize_stem(item.get("stem_type") or item.get("artifact_kind") or "") == "vocal"]
    minus_items = [item for item in items if normalize_stem(item.get("stem_type") or item.get("artifact_kind") or "") == "minus"]
    if not vocal_items:
        blockers.append(blocker("vocal_item_missing"))
    if not minus_items:
        blockers.append(blocker("minus_item_missing"))

    source_manifests = sorted({str(item.get("source_manifest") or "") for item in items if item.get("source_manifest")})
    source_evidence: list[dict[str, Any]] = []
    source_payloads: dict[str, dict[str, Any]] = {}
    for source in source_manifests:
        source_blockers, payload = inspect_source_master(source_manifest_path(source))
        blockers.extend(source_blockers)
        if payload:
            source_payloads[source] = payload
            source_evidence.append(
                {
                    "source_manifest": source,
                    "manifest_type": payload.get("manifest_type"),
                    "delivery_status": payload.get("delivery_status"),
                    "pair_decision": payload.get("pair_decision"),
                    "pair_window_count": payload.get("pair_window_count"),
                    "pair_compatibility_score": payload.get("pair_compatibility_score"),
                }
            )

    semantic_review_blockers, semantic_evidence, semantic_review = load_semantic_review_evidence(semantic_review_json, pack_manifest_json)
    blockers.extend(semantic_review_blockers)

    unverified_stems = unverified_required_stems(items, required_stem_list, source_payloads, semantic_evidence)
    for stem in unverified_stems:
        blockers.append(
            blocker(
                f"required_stem_quality_not_verified:{stem}",
                evidence={"required_stems": list(required_stem_list), "present_stems": present_stems},
            )
        )

    minus_blockers, minus_audit = evaluate_minus_audit(minus_items, minus_audit_json)
    blockers.extend(minus_blockers)

    training_blockers, training_evidence = evaluate_training_coverage(
        required_stems=required_stem_list,
        require_training_coverage=require_training_coverage,
        training_plan_json=training_plan_json,
        train_ready_decision_jsons=tuple(train_ready_decision_jsons or ()),
    )
    blockers.extend(training_blockers)

    blockers = [dict(row) for row in blockers]
    accepted = not blockers
    required_machine_gates = [
        "aggregate_pack_validation_passed",
        "full_required_stem_coverage",
        "required_stem_machine_quality_evidence",
        "source_vocal_minus_master_pair_compatible",
        "minus_quality_audit_passed_without_warnings",
    ]
    if require_training_coverage:
        required_machine_gates.append("required_stem_train_ready_model_evidence")
    manifest = {
        "schema": "machine_stem_acceptance_gate.v1",
        "gate_status": "accepted" if accepted else "blocked",
        "final_result_allowed": accepted,
        "delivery_status": "accepted" if accepted else "blocked",
        "delivery_allowed": accepted,
        "presentation_status": "accepted_for_listen_pack" if accepted else "blocked_from_listen_pack",
        "release_status": "listen_preview_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "pack_manifest_json": str(pack_manifest_json),
        "validation_json": str(validation_json),
        "minus_audit_json": str(minus_audit_json or ""),
        "semantic_review_json": str(semantic_review_json or ""),
        "training_plan_json": str(training_plan_json or ""),
        "train_ready_decision_jsons": [str(path) for path in (train_ready_decision_jsons or [])],
        "require_training_coverage": bool(require_training_coverage),
        "item_count": len(items),
        "stem_types": present_stems,
        "required_stems": list(required_stem_list),
        "missing_required_stems": missing_stems,
        "stem_audio_file_checks": audio_file_checks,
        "unverified_required_stems": unverified_stems,
        "train_ready_stems": training_evidence["train_ready_stems"],
        "training_ready_job_stems": training_evidence["training_ready_job_stems"],
        "blocked_training_job_stems": training_evidence["blocked_training_job_stems"],
        "semantic_review_status": semantic_review.get("review_status") if semantic_review else "",
        "semantic_review_evidence_count": len(semantic_evidence),
        "required_machine_gates": required_machine_gates,
        "source_evidence": source_evidence,
        "training_evidence": training_evidence,
        "minus_audit_status": minus_audit.get("audit_status") if minus_audit else "",
        "failed_checks": sorted({str(row["reason"]) for row in blockers}),
        "blocker_count": len(blockers),
        "blockers_jsonl": str(output_dir / BLOCKERS_NAME),
        "progress_jsonl": str(progress_path),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "contract": {
            "no_human_listen_required": True,
            "human_listen_pending_is_blocked": True,
            "audio_copied": False,
            "audio_exported": False,
            "training_started": False,
            "fine_tune_started": False,
            "separator_training_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "thresholds_relaxed": False,
            "cleanup_routes_added": False,
            "training_coverage_required": bool(require_training_coverage),
            "train_ready_model_evidence_required": bool(require_training_coverage),
        },
    }
    write_jsonl(output_dir / BLOCKERS_NAME, blockers)
    write_json(output_dir / GATE_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(build_summary(manifest, blockers), encoding="utf-8")
    progress(progress_path, "finish", gate_status=manifest["gate_status"], blocker_count=len(blockers), failed_checks=manifest["failed_checks"])
    return manifest


def build_summary(manifest: dict[str, Any], blockers: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Machine Stem Acceptance Gate",
        "",
        f"- gate_status: `{manifest['gate_status']}`",
        f"- final_result_allowed: `{str(manifest['final_result_allowed']).lower()}`",
        f"- item_count: `{manifest['item_count']}`",
        f"- blocker_count: `{manifest['blocker_count']}`",
        f"- release_ready: `{str(manifest['release_ready']).lower()}`",
        "",
        "## Blockers",
        "",
    ]
    if not blockers:
        lines.append("- none")
    for row in blockers:
        path = str(row.get("audio_path") or "")
        suffix = f" `{path}`" if path else ""
        lines.append(f"- `{row['reason']}`{suffix}")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fail-closed machine acceptance gate for a short full-stem pack.")
    parser.add_argument("--pack-manifest-json", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--minus-audit-json", type=Path)
    parser.add_argument("--semantic-review-json", type=Path)
    parser.add_argument("--training-plan-json", type=Path)
    parser.add_argument("--train-ready-decision-json", type=Path, action="append", default=[])
    parser.add_argument("--require-training-coverage", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--required-stem",
        action="append",
        default=None,
        help="Required stem type. Repeat to override the default full-stem set.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_machine_stem_acceptance_gate.py",
        entrypoint_path="/app/tools/build_machine_stem_acceptance_gate.py",
    )
    args = parse_args(argv)
    manifest = build_machine_stem_acceptance_gate(
        pack_manifest_json=args.pack_manifest_json,
        validation_json=args.validation_json,
        minus_audit_json=args.minus_audit_json,
        semantic_review_json=args.semantic_review_json,
        output_dir=args.output_dir,
        required_stems=args.required_stem,
        require_training_coverage=args.require_training_coverage,
        training_plan_json=args.training_plan_json,
        train_ready_decision_jsons=args.train_ready_decision_json,
    )
    print(
        json.dumps(
            {
                "gate_status": manifest["gate_status"],
                "final_result_allowed": manifest["final_result_allowed"],
                "blocker_count": manifest["blocker_count"],
                "failed_checks": manifest["failed_checks"],
                "manifest_path": str(args.output_dir / GATE_NAME),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["final_result_allowed"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
