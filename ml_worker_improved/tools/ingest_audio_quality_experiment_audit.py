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

from tools.audio_quality_critic_common import (
    BLOCKED_FILE,
    MANIFEST_FILE,
    PROGRESS_FILE,
    SAFETY_FLAGS,
    append_progress,
    read_json,
    write_json,
    write_jsonl,
)
from tools.build_audio_quality_experiment_audit_packet import PACKET_SCHEMA
from tools.docker_runtime import require_docker_runtime

INGEST_SCHEMA = "audio_quality_experiment_audit_ingest.v1"
AUDIT_JSON_FILE = "EXPERIMENT_AUDIT.json"
AUDIT_MD_FILE = "EXPERIMENT_AUDIT.md"
REQUIRED_CHECKS = (
    "gt_provenance",
    "score_normalization",
    "result_existence",
    "dead_code",
    "scope",
    "eval_type",
)


def _status(value: Any) -> str:
    return str(value or "").strip().lower()


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _blocked_row(reasons: Sequence[str]) -> dict[str, Any]:
    return {
        "schema": "audio_quality_experiment_audit_ingest_blocked_row.v1",
        "blocked_reasons": sorted(set(str(reason) for reason in reasons if reason)),
        **SAFETY_FLAGS,
    }


def _check_reasons(audit: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    checks = audit.get("checks")
    if not isinstance(checks, dict) or not checks:
        return ["experiment_audit_checks_missing"]
    for check_name in REQUIRED_CHECKS:
        check = checks.get(check_name)
        if not isinstance(check, dict):
            reasons.append("required_audit_check_missing")
            continue
        if _status(check.get("status")) not in {"pass", "warn", "fail"}:
            reasons.append("audit_check_status_invalid")
        evidence = _as_list(check.get("evidence"))
        if not evidence:
            reasons.append("audit_check_evidence_missing")
    return reasons


def _audit_reasons(packet_manifest: dict[str, Any], audit: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if audit.get("schema") == PACKET_SCHEMA or audit.get("manifest_type") == "audio_quality_experiment_audit_packet":
        reasons.append("audit_json_is_packet_manifest")
    if packet_manifest.get("schema") != PACKET_SCHEMA:
        reasons.append("packet_manifest_schema_unexpected")
    if packet_manifest.get("packet_status") != "ready_for_independent_review":
        reasons.append("packet_not_ready_for_independent_review")
    if packet_manifest.get("experiment_audit_pass") is True:
        reasons.append("packet_claims_experiment_audit_pass")
    if packet_manifest.get("delivery_allowed") is True:
        reasons.append("packet_delivery_allowed_true")
    if packet_manifest.get("release_ready") is True:
        reasons.append("packet_release_ready_true")

    verdict = _status(audit.get("overall_verdict"))
    integrity_status = _status(audit.get("integrity_status"))
    if verdict not in {"pass", "warn", "fail"}:
        reasons.append("overall_verdict_invalid")
    if integrity_status not in {"pass", "warn", "fail"}:
        reasons.append("integrity_status_invalid")
    if verdict != "pass" or integrity_status != "pass":
        reasons.append("experiment_audit_not_pass")
    if not str(audit.get("auditor") or "").strip():
        reasons.append("experiment_auditor_missing")
    if not str(audit.get("reviewer_trace_ref") or "").strip():
        reasons.append("reviewer_trace_ref_missing")
    reasons.extend(_check_reasons(audit))
    return sorted(set(reasons))


def _normalized_audit(
    *,
    packet_manifest_json: Path,
    audit_json: Path,
    packet_manifest: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    normalized = {
        **audit,
        "source_packet_manifest_json": str(packet_manifest_json),
        "source_audit_json": str(audit_json),
        "source_packet_path_count": int(packet_manifest.get("path_count") or 0),
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
    }
    return normalized


def _audit_markdown(normalized: dict[str, Any]) -> str:
    checks = normalized.get("checks") if isinstance(normalized.get("checks"), dict) else {}
    lines = [
        "# Experiment Audit Report",
        "",
        f"Overall Verdict: {normalized.get('overall_verdict', '')}",
        f"Integrity Status: {normalized.get('integrity_status', '')}",
        f"Auditor: {normalized.get('auditor', '')}",
        f"Reviewer Trace: {normalized.get('reviewer_trace_ref', '')}",
        "",
        "## Checks",
    ]
    for name in REQUIRED_CHECKS:
        check = checks.get(name, {}) if isinstance(checks.get(name), dict) else {}
        lines.append(f"- {name}: {check.get('status', 'missing')}")
    lines.extend(
        [
            "",
            "## Contract",
            "This report is normalized from an independent audit JSON. It is not a listen pack or release artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def ingest_audio_quality_experiment_audit(
    *,
    packet_manifest_json: Path,
    audit_json: Path,
    output_dir: Path,
) -> dict[str, Any]:
    start_time = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_FILE
    if progress_path.exists():
        progress_path.unlink()
    append_progress(
        progress_path,
        "start",
        start_time=start_time,
        packet_manifest_json=str(packet_manifest_json),
        audit_json=str(audit_json),
        expected_acceleration_mode="external_only_cpu_experiment_audit_ingest",
        gpu_inference_skipped=True,
        gpu_inference_skip_reason="experiment_audit_ingest_metadata_only",
    )

    errors: list[str] = []
    packet_manifest: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    if not packet_manifest_json.exists():
        errors.append("packet_manifest_missing")
    else:
        try:
            packet_manifest = read_json(packet_manifest_json)
        except Exception:
            errors.append("packet_manifest_invalid")
    if not audit_json.exists():
        errors.append("audit_json_missing")
    else:
        try:
            audit = read_json(audit_json)
        except Exception:
            errors.append("audit_json_invalid")
    if packet_manifest and audit:
        errors.extend(_audit_reasons(packet_manifest, audit))
    errors = sorted(set(errors))
    accepted = not errors

    if accepted:
        normalized = _normalized_audit(
            packet_manifest_json=packet_manifest_json,
            audit_json=audit_json,
            packet_manifest=packet_manifest,
            audit=audit,
        )
        write_json(output_dir / AUDIT_JSON_FILE, normalized)
        (output_dir / AUDIT_MD_FILE).write_text(_audit_markdown(normalized), encoding="utf-8")
        blocked_rows: list[dict[str, Any]] = []
    else:
        blocked_rows = [_blocked_row(errors)]
    write_jsonl(output_dir / BLOCKED_FILE, blocked_rows)

    manifest = {
        "schema": INGEST_SCHEMA,
        "manifest_type": "audio_quality_experiment_audit_ingest",
        "ingest_status": "accepted" if accepted else "blocked",
        "errors": errors,
        "packet_manifest_json": str(packet_manifest_json),
        "audit_json": str(audit_json),
        "experiment_audit_pass": bool(accepted),
        "promotion_allowed": False,
        "training_started": False,
        "presentation_status": "audio_quality_experiment_audit_ingest_not_for_listen_pack",
        "outputs": {
            "experiment_audit_json": str(output_dir / AUDIT_JSON_FILE) if accepted else "",
            "experiment_audit_md": str(output_dir / AUDIT_MD_FILE) if accepted else "",
            "blocked_jsonl": str(output_dir / BLOCKED_FILE),
            "manifest_json": str(output_dir / MANIFEST_FILE),
            "progress_jsonl": str(progress_path),
        },
        "safety": {
            "independent_reviewer_required": True,
            "packet_is_not_experiment_audit_pass": True,
            "no_self_issued_promotion": True,
            "not_release_artifact": True,
        },
        **SAFETY_FLAGS,
    }
    write_json(output_dir / MANIFEST_FILE, manifest)
    append_progress(
        progress_path,
        "finish",
        start_time=start_time,
        ingest_status=manifest["ingest_status"],
        experiment_audit_pass=bool(accepted),
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest independent AQC experiment-audit output.")
    parser.add_argument("--packet-manifest-json", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="ingest_audio_quality_experiment_audit.py",
        entrypoint_path="/app/tools/ingest_audio_quality_experiment_audit.py",
    )
    args = parse_args(argv)
    manifest = ingest_audio_quality_experiment_audit(
        packet_manifest_json=args.packet_manifest_json,
        audit_json=args.audit_json,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "manifest_json": str(args.output_dir / MANIFEST_FILE),
                "ingest_status": manifest["ingest_status"],
                "experiment_audit_pass": manifest["experiment_audit_pass"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["ingest_status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
