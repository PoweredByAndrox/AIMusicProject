#!/usr/bin/env python3
"""Validate candidate verified-reference bundles against full-pack corpus gaps."""

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
from tools.full_pack_artifact_scope import is_non_production_artifact_path  # noqa: E402


START_TIME = time.monotonic()
MANIFEST_NAME = "intake.json"
ROWS_NAME = "rows.jsonl"
AUDIO_NAME = "audio.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
TRAINABLE_PERMISSION_STATUSES = {"owned", "licensed", "cleared", "project_owned"}
GROUND_TRUTH_PERMISSION_STATUSES = {"owned", "licensed", "cleared", "project_owned", "user_owned"}
ALLOWED_SOURCE_DOMAINS = {"user_supplied_flstudio_music"}
BLOCKED_TRUE_FLAGS = (
    "release_ready",
    "can_publish_as_result",
    "deliverable",
    "delivery_allowed",
    "thresholds_relaxed",
    "runtime_gates_changed",
    "model_registry_edited",
    "models_changed",
    "training_started",
    "fine_tune_started",
    "separator_training_started",
    "cleanup_routes_added",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected object row")
            yield payload


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
        "expected_acceleration_mode": "external_only_cpu_reference_intake_validation",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "reference_intake_reads_metadata_and_path_existence_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "reference_intake_no_training_started",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized_stem(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "vocals":
        return "vocal"
    if text == "instrumental":
        return "minus"
    return text or "unknown"


def resolved_path(raw_path: Any, *, base_dir: Path) -> str:
    text = str(raw_path or "").strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    return str(path).replace("\\", "/")


def path_exists(path_text: str) -> bool:
    return bool(path_text) and Path(path_text).exists()


def collect_true_flag_failures(prefix: str, payload: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    for key in BLOCKED_TRUE_FLAGS:
        if payload.get(key) is True:
            failed.append(f"{prefix}_{key}_true")
    for nested_key in ("safety", "contract"):
        nested = payload.get(nested_key)
        if not isinstance(nested, dict):
            continue
        for key in BLOCKED_TRUE_FLAGS:
            if nested.get(key) is True:
                failed.append(f"{prefix}_{nested_key}_{key}_true")
    return failed


def stable_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def source_domain_for_ground_truth(payload: dict[str, Any]) -> str:
    contract = payload.get("ground_truth_contract")
    if not isinstance(contract, dict):
        contract = {}
    combined = f"{contract.get('source') or ''} {contract.get('verification_method') or ''}".lower()
    if "synthetic_deterministic_signal_generation" in combined or "run_ground_truth_quality_gate" in combined:
        return "synthetic_fixture"
    if "fl studio" in combined or "supplied by user" in combined:
        return "user_supplied_flstudio_music"
    return "ground_truth_manifest"


def permission_status_from_ground_truth(payload: dict[str, Any]) -> str:
    report = payload.get("report_alignment_contract")
    if not isinstance(report, dict):
        report = {}
    permission = str(report.get("rights_status") or payload.get("rights_status") or "").strip().lower()
    if permission == "user_owned":
        return "project_owned"
    return permission


def ground_truth_source_gate(payload: dict[str, Any]) -> dict[str, Any]:
    full_mix_contract = payload.get("full_mix_reconstruction_contract")
    if not isinstance(full_mix_contract, dict):
        full_mix_contract = {}
    reconstruction_status = str(full_mix_contract.get("status") or "").strip().lower()
    source_mix_1_to_1_claim_allowed = full_mix_contract.get("can_claim_source_mix_1_to_1") is True
    blocker_reasons: list[str] = []
    if reconstruction_status != "accepted":
        blocker_reasons.append("full_mix_reconstruction_not_accepted")
    if not source_mix_1_to_1_claim_allowed:
        blocker_reasons.append("source_mix_1_to_1_not_claimable")
    return {
        "reference_verified": True,
        "trainable_ground_truth": not blocker_reasons,
        "full_mix_reconstruction_status": reconstruction_status or "missing",
        "source_mix_1_to_1_claim_allowed": source_mix_1_to_1_claim_allowed,
        "blocker_reasons": blocker_reasons,
    }


def load_gap_rows(gap_plan_json: Path) -> tuple[list[dict[str, Any]], list[str]]:
    plan = read_json(gap_plan_json)
    failed = collect_true_flag_failures("gap_plan", plan)
    if plan.get("schema") != "full_pack_reference_corpus_gap_plan.v1":
        failed.append("gap_plan_schema_unexpected")
    if plan.get("failed_checks") not in ([], None):
        failed.append("gap_plan_failed_checks_not_empty")
    rows_path_text = ""
    outputs = plan.get("outputs")
    if isinstance(outputs, dict):
        rows_path_text = str(outputs.get("rows_jsonl") or "")
    rows_path_text = rows_path_text or str(plan.get("rows_jsonl") or "")
    rows_path = Path(rows_path_text)
    if rows_path_text and not rows_path.is_absolute():
        rows_path = gap_plan_json.parent / rows_path
    if not rows_path_text or not rows_path.exists():
        failed.append("gap_plan_rows_jsonl_missing")
        return [], stable_unique(failed)
    rows = list(iter_jsonl(rows_path))
    if any(row.get("schema") != "full_pack_reference_corpus_gap.v1" for row in rows):
        failed.append("gap_row_schema_unexpected")
    return rows if not failed else [], stable_unique(failed)


def required_stems_from_gaps(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        stem = normalized_stem(row.get("stem_type"))
        if not stem or stem == "unknown":
            continue
        result.setdefault(stem, row)
    return result


def existing_track_stems(path: Path) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    if not path.exists():
        return pairs
    for row in iter_jsonl(path):
        track_id = str(row.get("track_id") or "")
        stem = normalized_stem(row.get("stem_type"))
        if track_id and stem and stem != "unknown":
            pairs.add((track_id, stem))
    return pairs


def stem_payloads(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stems = payload.get("stems")
    if not isinstance(stems, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for raw_stem, stem_payload in stems.items():
        if isinstance(stem_payload, dict):
            result[normalized_stem(raw_stem)] = stem_payload
    return result


def candidate_row(
    *,
    manifest_path: Path,
    payload: dict[str, Any],
    stem: str,
    gap: dict[str, Any],
    stem_payload: dict[str, Any],
    seen_track_stems: set[tuple[str, str]],
) -> dict[str, Any]:
    contract = payload.get("ground_truth_contract")
    if not isinstance(contract, dict):
        contract = {}
    track_id = str(payload.get("track_name") or manifest_path.parent.name)
    source_domain = source_domain_for_ground_truth(payload)
    permission_status = permission_status_from_ground_truth(payload)
    source_audio_path = resolved_path((payload.get("mix") or {}).get("path") if isinstance(payload.get("mix"), dict) else "", base_dir=manifest_path.parent)
    target_audio_path = resolved_path(stem_payload.get("path"), base_dir=manifest_path.parent)
    source_gate = ground_truth_source_gate(payload)
    reasons: list[str] = []
    if contract.get("verified") is not True:
        reasons.append("not_reference_verified")
    if source_gate["trainable_ground_truth"] is not True:
        reasons.append("ground_truth_reconstruction_not_trainable")
    if source_domain not in ALLOWED_SOURCE_DOMAINS:
        reasons.append("source_domain_not_allowed")
    if source_domain == "synthetic_fixture":
        reasons.append("synthetic_fixture_not_allowed")
    if not permission_status:
        reasons.append("missing_permission_status")
    elif permission_status not in TRAINABLE_PERMISSION_STATUSES and permission_status not in GROUND_TRUTH_PERMISSION_STATUSES:
        reasons.append("permission_not_trainable")
    if not path_exists(source_audio_path):
        reasons.append("source_audio_not_found")
    if not path_exists(target_audio_path):
        reasons.append("target_audio_not_found")
    if is_non_production_artifact_path(manifest_path.parent):
        reasons.append("candidate_root_non_production_artifact")
    if (track_id, stem) in seen_track_stems:
        reasons.append("duplicate_existing_track_stem")
    intake_status = "rejected" if reasons else "accepted_for_corpus_import"
    return {
        "schema": "full_pack_reference_corpus_intake_row.v1",
        "id": stable_hash(["full_pack_reference_intake", str(manifest_path), stem, track_id])[:16],
        "intake_status": intake_status,
        "rejection_reasons": stable_unique(reasons),
        "gap_id": str(gap.get("gap_id") or ""),
        "queue_item_id": str(gap.get("queue_item_id") or ""),
        "stem_type": stem,
        "track_id": track_id,
        "manifest_path": str(manifest_path),
        "source_domain": source_domain,
        "permission_status": "project_owned" if permission_status == "user_owned" else permission_status,
        "source_audio_path": source_audio_path,
        "target_audio_path": target_audio_path,
        "target_audio_sha256": str(stem_payload.get("sha256") or ""),
        "reference_verified": contract.get("verified") is True,
        "source_gate": source_gate,
        "required_reference_stem_count": int(gap.get("required_reference_stem_count") or 0),
        "release_ready": False,
        "can_publish_as_result": False,
        "training_started": False,
    }


def audio_training_example(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "generic_finetuning_audio_training_example.v1",
        "id": stable_hash(["full_pack_reference_intake_audio", row["id"], row["permission_status"]])[:16],
        "track_id": row["track_id"],
        "stem_type": row["stem_type"],
        "manifest_path": row["manifest_path"],
        "source_audio_path": row["source_audio_path"],
        "target_audio_path": row["target_audio_path"],
        "target_audio_sha256": row["target_audio_sha256"],
        "model_id": "verified_ground_truth_bundle",
        "rating": "reference_verified",
        "human_score": None,
        "human_verified": False,
        "reference_verified": True,
        "permission_status": row["permission_status"],
        "split": "train",
        "notes": "full_pack_reference_corpus_intake",
        "trainable": True,
        "training_target": "audio_separator_or_stem_adapter",
        "source_gate": row["source_gate"],
        "safety": {
            "not_release_artifact": True,
            "requires_eval_before_promotion": True,
            "approved_by_full_pack_reference_intake": True,
        },
    }


def discover_candidate_manifests(candidate_roots: Sequence[Path]) -> list[Path]:
    manifests: list[Path] = []
    for root in candidate_roots:
        if root.is_file() and root.name == "ground_truth_manifest.json":
            manifests.append(root)
        elif root.is_dir():
            manifests.extend(path for path in root.rglob("ground_truth_manifest.json") if path.is_file())
    return sorted(set(manifests))


def build_full_pack_reference_corpus_intake(
    *,
    gap_plan_json: Path,
    existing_audio_training_examples_jsonl: Path,
    candidate_roots: Sequence[Path],
    output_dir: Path,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(
        progress_path,
        "start",
        gap_plan_json=str(gap_plan_json),
        existing_audio_training_examples_jsonl=str(existing_audio_training_examples_jsonl),
        candidate_roots=[str(path) for path in candidate_roots],
    )

    gap_rows, failed = load_gap_rows(gap_plan_json)
    required_by_stem = required_stems_from_gaps(gap_rows)
    seen_track_stems = existing_track_stems(existing_audio_training_examples_jsonl)
    rows: list[dict[str, Any]] = []
    if not failed:
        for manifest_path in discover_candidate_manifests(candidate_roots):
            payload = read_json(manifest_path)
            stems = stem_payloads(payload)
            for stem, gap in required_by_stem.items():
                stem_payload = stems.get(stem)
                if stem_payload is None:
                    continue
                row = candidate_row(
                    manifest_path=manifest_path,
                    payload=payload,
                    stem=stem,
                    gap=gap,
                    stem_payload=stem_payload,
                    seen_track_stems=seen_track_stems,
                )
                rows.append(row)
                if row["intake_status"] == "accepted_for_corpus_import":
                    seen_track_stems.add((row["track_id"], row["stem_type"]))

    rows.sort(key=lambda row: (row["intake_status"] != "accepted_for_corpus_import", row["stem_type"], row["track_id"], row["id"]))
    accepted_rows = [row for row in rows if row["intake_status"] == "accepted_for_corpus_import"]
    audio_rows = [audio_training_example(row) for row in accepted_rows]
    accepted_by_stem = Counter(row["stem_type"] for row in accepted_rows)
    required_by_stem_count = {
        stem: int(gap.get("required_reference_stem_count") or gap.get("missing_unique_track_count") or 0)
        for stem, gap in required_by_stem.items()
    }
    remaining_by_stem = {
        stem: max(required_count - int(accepted_by_stem.get(stem, 0)), 0)
        for stem, required_count in sorted(required_by_stem_count.items())
    }
    remaining_total = sum(remaining_by_stem.values())
    if failed:
        status = "blocked"
    elif accepted_rows and remaining_total:
        status = "accepted_with_remaining_gaps"
    elif accepted_rows:
        status = "accepted_reference_corpus_complete"
    else:
        status = "no_accepted_references"
    manifest = {
        "schema": "full_pack_reference_corpus_intake_manifest.v1",
        "manifest_type": "full_pack_reference_corpus_intake",
        "intake_status": status,
        "failed_checks": failed,
        "gap_plan_json": str(gap_plan_json),
        "existing_audio_training_examples_jsonl": str(existing_audio_training_examples_jsonl),
        "candidate_roots": [str(path) for path in candidate_roots],
        "candidate_manifest_count": len(discover_candidate_manifests(candidate_roots)) if not failed else 0,
        "accepted_reference_rows": len(accepted_rows),
        "rejected_reference_rows": len(rows) - len(accepted_rows),
        "accepted_audio_training_examples": len(audio_rows),
        "required_reference_stem_count": sum(required_by_stem_count.values()),
        "remaining_reference_stem_count": remaining_total,
        "accepted_by_stem": dict(sorted(accepted_by_stem.items())),
        "remaining_by_stem": remaining_by_stem,
        "rejection_reason_counts": dict(Counter(reason for row in rows for reason in row["rejection_reasons"])),
        "outputs": {
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "rows_jsonl": str(output_dir / ROWS_NAME),
            "accepted_audio_training_examples_jsonl": str(output_dir / AUDIO_NAME),
            "progress_jsonl": str(progress_path),
            "summary_md": str(output_dir / SUMMARY_NAME),
        },
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "reference_corpus_intake_not_for_listen_pack",
        "release_status": "training_reference_intake_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "safety": {
            "metadata_only": True,
            "audio_read": False,
            "audio_copied": False,
            "accepts_only_verified_ground_truth": True,
            "synthetic_fixtures_not_allowed": True,
            "not_release_artifact": True,
        },
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / ROWS_NAME, rows)
    write_jsonl(output_dir / AUDIO_NAME, audio_rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    summary_lines = [
        "# Full Pack Reference Corpus Intake",
        "",
        f"- intake_status: `{status}`",
        f"- accepted_reference_rows: `{len(accepted_rows)}`",
        f"- remaining_reference_stem_count: `{remaining_total}`",
        f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
        f"- release_ready: `{str(manifest['release_ready']).lower()}`",
        "",
    ]
    (output_dir / SUMMARY_NAME).write_text("\n".join(summary_lines), encoding="utf-8")
    progress(progress_path, "finish", intake_status=status, accepted_reference_rows=len(accepted_rows), remaining_reference_stem_count=remaining_total, failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate candidate ground-truth bundles against full-pack reference corpus gaps.")
    parser.add_argument("--gap-plan-json", type=Path, required=True)
    parser.add_argument("--existing-audio-training-examples-jsonl", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_full_pack_reference_corpus_intake.py",
        entrypoint_path="/app/tools/build_full_pack_reference_corpus_intake.py",
    )
    args = parse_args(argv)
    manifest = build_full_pack_reference_corpus_intake(
        gap_plan_json=args.gap_plan_json,
        existing_audio_training_examples_jsonl=args.existing_audio_training_examples_jsonl,
        candidate_roots=args.candidate_root,
        output_dir=args.output_dir,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "intake_status": manifest["intake_status"],
                "accepted_reference_rows": manifest["accepted_reference_rows"],
                "remaining_reference_stem_count": manifest["remaining_reference_stem_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
