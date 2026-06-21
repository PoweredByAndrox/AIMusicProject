#!/usr/bin/env python3
"""Audit available full-pack reference material before ARIS intake."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_full_pack_reference_corpus_intake import (  # noqa: E402
    BLOCKED_TRUE_FLAGS,
    candidate_row,
    collect_true_flag_failures,
    discover_candidate_manifests,
    existing_track_stems,
    load_gap_rows,
    read_json,
    required_stems_from_gaps,
    stable_unique,
    stem_payloads,
)
from tools.docker_runtime import require_docker_runtime  # noqa: E402


START_TIME = time.monotonic()
MANIFEST_NAME = "discovery.json"
ROWS_NAME = "rows.jsonl"
ELIGIBLE_NAME = "eligible.jsonl"
REJECTED_NAME = "rejected.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"


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
        "expected_acceleration_mode": "external_only_cpu_reference_material_discovery",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "reference_material_discovery_reads_metadata_and_path_existence_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "reference_material_discovery_never_starts_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def required_by_stem_counts(required_by_stem: dict[str, dict[str, Any]]) -> dict[str, int]:
    return {
        stem: int(gap.get("required_reference_stem_count") or gap.get("missing_unique_track_count") or 0)
        for stem, gap in sorted(required_by_stem.items())
    }


def collect_discovery_true_flag_failures(payload: dict[str, Any]) -> list[str]:
    failed = collect_true_flag_failures("ground_truth_manifest", payload)
    contract = payload.get("ground_truth_contract")
    if isinstance(contract, dict):
        for key in BLOCKED_TRUE_FLAGS:
            if contract.get(key) is True:
                failed.append(f"ground_truth_manifest_ground_truth_contract_{key}_true")
    return stable_unique(failed)


def discovery_row_from_intake(
    *,
    manifest_path: Path,
    payload: dict[str, Any],
    stem: str,
    gap: dict[str, Any],
    stem_payload: dict[str, Any],
    seen_track_stems: set[tuple[str, str]],
) -> dict[str, Any]:
    row = candidate_row(
        manifest_path=manifest_path,
        payload=payload,
        stem=stem,
        gap=gap,
        stem_payload=stem_payload,
        seen_track_stems=seen_track_stems,
    )
    reasons = list(row.get("rejection_reasons") or [])
    reasons.extend(collect_discovery_true_flag_failures(payload))
    reasons = stable_unique(reasons)
    eligible = not reasons
    return {
        **row,
        "schema": "full_pack_reference_material_discovery_row.v1",
        "source_intake_status": row["intake_status"],
        "discovery_status": "eligible_for_reference_intake" if eligible else "rejected",
        "rejection_reasons": reasons,
        "intake_status": "not_imported_discovery_only",
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "training_started": False,
    }


def build_summary(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Full Pack Reference Material Discovery",
            "",
            f"- discovery_status: `{manifest['discovery_status']}`",
            f"- candidate_manifest_count: `{manifest['candidate_manifest_count']}`",
            f"- eligible_candidate_rows: `{manifest['eligible_candidate_rows']}`",
            f"- rejected_candidate_rows: `{manifest['rejected_candidate_rows']}`",
            f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
            f"- release_ready: `{str(manifest['release_ready']).lower()}`",
            "",
        ]
    )


def audit_full_pack_reference_material_discovery(
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
    required_counts = required_by_stem_counts(required_by_stem)
    seen_track_stems = existing_track_stems(existing_audio_training_examples_jsonl)
    candidate_manifests = discover_candidate_manifests(candidate_roots) if not failed else []
    rows: list[dict[str, Any]] = []

    if not failed:
        for manifest_path in candidate_manifests:
            payload = read_json(manifest_path)
            stems = stem_payloads(payload)
            for stem, gap in required_by_stem.items():
                stem_payload = stems.get(stem)
                if stem_payload is None:
                    continue
                row = discovery_row_from_intake(
                    manifest_path=manifest_path,
                    payload=payload,
                    stem=stem,
                    gap=gap,
                    stem_payload=stem_payload,
                    seen_track_stems=seen_track_stems,
                )
                rows.append(row)
                if row["discovery_status"] == "eligible_for_reference_intake":
                    seen_track_stems.add((row["track_id"], row["stem_type"]))

    rows.sort(key=lambda row: (row["discovery_status"] != "eligible_for_reference_intake", row["stem_type"], row["track_id"], row["id"]))
    eligible_rows = [row for row in rows if row["discovery_status"] == "eligible_for_reference_intake"]
    rejected_rows = [row for row in rows if row["discovery_status"] != "eligible_for_reference_intake"]
    eligible_by_stem = Counter(row["stem_type"] for row in eligible_rows)
    remaining_after_eligible = {
        stem: max(required_count - int(eligible_by_stem.get(stem, 0)), 0)
        for stem, required_count in required_counts.items()
    }
    if failed:
        status = "blocked"
    elif eligible_rows:
        status = "eligible_reference_material_found"
    else:
        status = "blocked_no_eligible_reference_material"

    manifest = {
        "schema": "full_pack_reference_material_discovery_audit.v1",
        "manifest_type": "full_pack_reference_material_discovery_audit",
        "discovery_status": status,
        "failed_checks": failed,
        "gap_plan_json": str(gap_plan_json),
        "existing_audio_training_examples_jsonl": str(existing_audio_training_examples_jsonl),
        "candidate_roots": [str(path) for path in candidate_roots],
        "candidate_manifest_count": len(candidate_manifests),
        "candidate_row_count": len(rows),
        "eligible_candidate_rows": len(eligible_rows),
        "rejected_candidate_rows": len(rejected_rows),
        "required_by_stem": required_counts,
        "eligible_by_stem": dict(sorted(eligible_by_stem.items())),
        "remaining_after_eligible_by_stem": remaining_after_eligible,
        "rejection_reason_counts": dict(Counter(reason for row in rejected_rows for reason in row["rejection_reasons"])),
        "outputs": {
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "rows_jsonl": str(output_dir / ROWS_NAME),
            "eligible_jsonl": str(output_dir / ELIGIBLE_NAME),
            "rejected_jsonl": str(output_dir / REJECTED_NAME),
            "progress_jsonl": str(progress_path),
            "summary_md": str(output_dir / SUMMARY_NAME),
        },
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "reference_material_discovery_not_for_listen_pack",
        "release_status": "audit_artifact_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "safety": {
            "metadata_only": True,
            "audio_read": False,
            "audio_copied": False,
            "audio_decoded": False,
            "dataset_built": False,
            "training_started": False,
            "not_release_artifact": True,
            "accepted_rows_not_imported": True,
        },
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / ROWS_NAME, rows)
    write_jsonl(output_dir / ELIGIBLE_NAME, eligible_rows)
    write_jsonl(output_dir / REJECTED_NAME, rejected_rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(build_summary(manifest), encoding="utf-8")
    progress(
        progress_path,
        "finish",
        discovery_status=status,
        eligible_candidate_rows=len(eligible_rows),
        rejected_candidate_rows=len(rejected_rows),
        failed_checks=failed,
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit available reference material against full-pack corpus gaps.")
    parser.add_argument("--gap-plan-json", type=Path, required=True)
    parser.add_argument("--existing-audio-training-examples-jsonl", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="audit_full_pack_reference_material_discovery.py",
        entrypoint_path="/app/tools/audit_full_pack_reference_material_discovery.py",
    )
    args = parse_args(argv)
    manifest = audit_full_pack_reference_material_discovery(
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
                "discovery_status": manifest["discovery_status"],
                "candidate_manifest_count": manifest["candidate_manifest_count"],
                "eligible_candidate_rows": manifest["eligible_candidate_rows"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
