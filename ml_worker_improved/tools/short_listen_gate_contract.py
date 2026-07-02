from __future__ import annotations

import json
from pathlib import Path
from typing import Any

AGGREGATE_SCHEMA = "aggregate_short_listen_pack_manifest.v1"
AGGREGATE_VALIDATION_PENDING_STATUS = "aggregate_validation_pending"
AGGREGATE_VALIDATION_PENDING_PRESENTATION = "aggregate_validation_pending_not_for_listen_pack"
MACHINE_GATE_SCHEMA = "machine_stem_acceptance_gate.v1"


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def stable_unique(items: list[str]) -> list[str]:
    return sorted({str(item) for item in items if str(item)})


def same_path(left: Any, right: Path) -> bool:
    raw = str(left or "")
    if not raw:
        return False
    try:
        return Path(raw).resolve(strict=False) == right.resolve(strict=False)
    except OSError:
        return raw.replace("\\", "/") == str(right).replace("\\", "/")


def resolve_gate_path(machine_gate_json: Path, raw_path: Any) -> Path | None:
    raw = str(raw_path or "")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = machine_gate_json.parent / path
    return path


def pack_rows(pack: dict[str, Any]) -> list[dict[str, Any]]:
    rows = pack.get("items")
    if not isinstance(rows, list):
        rows = pack.get("listen_audio_files")
    return [row for row in rows or [] if isinstance(row, dict)]


def has_minus_row(pack: dict[str, Any]) -> bool:
    return any("minus" in str(row.get("stem_type") or "").lower() for row in pack_rows(pack))


def staged_aggregate_pack_failures(pack: dict[str, Any], *, prefix: str = "pack") -> list[str]:
    failed: list[str] = []
    expected = {
        "schema": AGGREGATE_SCHEMA,
        "delivery_status": AGGREGATE_VALIDATION_PENDING_STATUS,
        "delivery_allowed": False,
        "presentation_status": AGGREGATE_VALIDATION_PENDING_PRESENTATION,
        "release_status": AGGREGATE_VALIDATION_PENDING_STATUS,
        "release_ready": False,
        "can_publish_as_result": False,
        "listenable": False,
        "deliverable": False,
        "aggregate_validation_ready": True,
    }
    for key, expected_value in expected.items():
        if pack.get(key) != expected_value:
            if key == "schema":
                failed.append(f"{prefix}_schema_unexpected")
            elif key in {"delivery_status", "presentation_status", "release_status"}:
                failed.append(f"{prefix}_{key}_not_{expected_value}")
            elif expected_value is False:
                failed.append(f"{prefix}_{key}_not_false")
            elif expected_value is True:
                failed.append(f"{prefix}_{key}_not_true")
    return failed


def accepted_machine_gate_failures(
    *,
    machine_gate_json: Path | None,
    pack_manifest_json: Path,
    pack: dict[str, Any],
    prefix: str = "machine_gate",
) -> list[str]:
    if machine_gate_json is None:
        return [f"{prefix}_json_missing"]
    if not machine_gate_json.exists():
        return [f"{prefix}_json_missing"]

    gate = read_json_object(machine_gate_json)
    failed: list[str] = []
    if gate.get("schema") != MACHINE_GATE_SCHEMA:
        failed.append(f"{prefix}_schema_unexpected")
    if gate.get("gate_status") != "accepted":
        failed.append(f"{prefix}_not_accepted")
    if gate.get("final_result_allowed") is not True:
        failed.append(f"{prefix}_final_result_allowed_not_true")
    if gate.get("delivery_status") != "accepted":
        failed.append(f"{prefix}_delivery_status_not_accepted")
    if gate.get("delivery_allowed") is not True:
        failed.append(f"{prefix}_delivery_allowed_not_true")
    if gate.get("presentation_status") != "accepted_for_listen_pack":
        failed.append(f"{prefix}_presentation_status_not_accepted")
    if gate.get("release_status") != "listen_preview_only":
        failed.append(f"{prefix}_release_status_not_listen_preview")
    if gate.get("release_ready") is True:
        failed.append(f"{prefix}_release_ready_true")
    if gate.get("can_publish_as_result") is True:
        failed.append(f"{prefix}_can_publish_as_result_true")
    if gate.get("failed_checks") not in ([], None):
        failed.append(f"{prefix}_failed_checks_not_empty")
    if not same_path(gate.get("pack_manifest_json"), pack_manifest_json):
        failed.append(f"{prefix}_pack_manifest_mismatch")
    validation_path = resolve_gate_path(machine_gate_json, gate.get("validation_json"))
    if validation_path is None:
        failed.append(f"{prefix}_validation_json_missing")
    elif not validation_path.exists():
        failed.append(f"{prefix}_validation_json_missing")
    else:
        validation = read_json_object(validation_path)
        if validation.get("schema") != "aggregate_short_listen_pack_validation.v1":
            failed.append(f"{prefix}_validation_schema_unexpected")
        if validation.get("validation_status") != "passed":
            failed.append(f"{prefix}_validation_status_not_passed")
        if validation.get("failed_checks") not in ([], None):
            failed.append(f"{prefix}_validation_failed_checks_not_empty")
        if validation.get("pack_manifest_json") and not same_path(validation.get("pack_manifest_json"), pack_manifest_json):
            failed.append(f"{prefix}_validation_pack_manifest_mismatch")
    semantic_path = resolve_gate_path(machine_gate_json, gate.get("semantic_review_json"))
    if semantic_path is None:
        failed.append(f"{prefix}_semantic_review_json_missing")
    elif not semantic_path.exists():
        failed.append(f"{prefix}_semantic_review_json_missing")
    else:
        semantic = read_json_object(semantic_path)
        if semantic.get("failed_checks") not in ([], None):
            failed.append(f"{prefix}_semantic_review_failed_checks_not_empty")
        if str(semantic.get("review_status") or "") not in {"completed", "passed"}:
            failed.append(f"{prefix}_semantic_review_not_passed")
        if semantic.get("pack_manifest_json") and not same_path(semantic.get("pack_manifest_json"), pack_manifest_json):
            failed.append(f"{prefix}_semantic_review_pack_manifest_mismatch")
    if int(gate.get("semantic_review_evidence_count") or 0) <= 0:
        failed.append(f"{prefix}_semantic_review_evidence_missing")
    if has_minus_row(pack):
        minus_audit_path = resolve_gate_path(machine_gate_json, gate.get("minus_audit_json"))
        if minus_audit_path is None:
            failed.append(f"{prefix}_minus_audit_json_missing")
        elif not minus_audit_path.exists():
            failed.append(f"{prefix}_minus_audit_json_missing")
        else:
            minus_audit = read_json_object(minus_audit_path)
            if minus_audit.get("schema") != "short_minus_quality_audit.v1":
                failed.append(f"{prefix}_minus_audit_schema_unexpected")
            if str(minus_audit.get("audit_status") or "") not in {"completed", "passed"}:
                failed.append(f"{prefix}_minus_audit_not_completed")
            if minus_audit.get("failed_checks") not in ([], None):
                failed.append(f"{prefix}_minus_audit_failed_checks_not_empty")
            if minus_audit.get("pack_manifest_json") and not same_path(minus_audit.get("pack_manifest_json"), pack_manifest_json):
                failed.append(f"{prefix}_minus_audit_pack_manifest_mismatch")
        if str(gate.get("minus_audit_status") or "") not in {"completed", "passed"}:
            failed.append(f"{prefix}_minus_audit_not_completed")
    return stable_unique(failed)
