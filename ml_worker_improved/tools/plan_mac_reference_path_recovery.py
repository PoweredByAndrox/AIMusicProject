#!/usr/bin/env python3
"""Build a reference-path recovery plan from a blocked Mac readiness summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.plan_reference_path_recovery import (  # noqa: E402
    build_reference_path_recovery_plan,
    write_outputs,
)


INPUT_DIR_NAME = "inputs"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_must_be_object:{path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def _track_id(identity: dict[str, Any]) -> str:
    return str(identity.get("track_id") or identity.get("source_track_id") or identity.get("id") or "").strip()


def _case_id(summary: dict[str, Any]) -> str:
    candidate_id = str(summary.get("selected_candidate_id") or "").strip()
    source_id = _track_id(dict(summary.get("source_track_identity") or {}))
    suffix = candidate_id or source_id or "unknown"
    return f"mac-{suffix}-reference"


def _source_manifest(summary: dict[str, Any], path: Path) -> Path:
    return write_json(
        path,
        {
            "schema": "mac_reference_recovery_source_manifest.v1",
            "status": "blocked_reference_recovery_input",
            "selected_candidate_id": str(summary.get("selected_candidate_id") or ""),
            "source_audio_path": str(summary.get("source_audio_path") or ""),
            "source_track_identity": dict(summary.get("source_track_identity") or {}),
            "rejected_reference_track_identity": dict(summary.get("reference_track_identity") or {}),
            "rejected_reference_source_audio_path": str(summary.get("reference_source_audio_path") or ""),
            "reference_bundle_status": str(summary.get("reference_bundle_status") or summary.get("reference_audit_status") or ""),
            "reference_mismatch_status": "reference_bundle_mismatched_to_source"
            if str(summary.get("reference_audit_status") or "") == "mismatched_to_source"
            else "",
            "delivery_allowed": False,
            "listen_result_allowed": False,
            "presentation_status": "reference_recovery_input_not_for_listen_pack",
        },
    )


def _input_bundle(summary: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    input_dir = output_dir / INPUT_DIR_NAME
    case_id = _case_id(summary)
    source_identity = dict(summary.get("source_track_identity") or {})
    track_hash = _track_id(source_identity) or case_id
    source_manifest = _source_manifest(summary, input_dir / "src_manifest.json")
    source_audio_path = str(summary.get("source_audio_path") or "")
    reference_mismatch_status = (
        "reference_bundle_mismatched_to_source"
        if str(summary.get("reference_audit_status") or "") == "mismatched_to_source"
        else ""
    )
    mapping_row = {
        "schema": "source_path_mapping_row.v1",
        "case_id": case_id,
        "track_hash": track_hash,
        "stem_type": "minus",
        "split": "holdout",
        "source_manifest": str(source_manifest),
        "mapping_status": "blocked_missing_reference_path",
        "source_audio_path": source_audio_path,
        "source_track_identity": source_identity,
        "selected_candidate_id": str(summary.get("selected_candidate_id") or ""),
        "reference_bundle_status": str(summary.get("reference_bundle_status") or summary.get("reference_audit_status") or ""),
        "reference_mismatch_status": reference_mismatch_status,
        "rejected_reference_track_identity": dict(summary.get("reference_track_identity") or {}),
        "readiness_summary_json": str(summary.get("manifest") or ""),
        "original_paths_found": [
            {
                "type": "original_mix",
                "path": source_audio_path,
                "exists_in_docker": False,
                "source_key": "mac_readiness_summary.source_audio_path",
            }
        ]
        if source_audio_path
        else [],
        "forbidden_actions": ["no_audio_copy", "no_audio_read_or_hash", "no_promotion", "no_training_start"],
    }
    paths = {
        "source_path_mapping_jsonl": write_jsonl(input_dir / "source_path_mapping_plan.jsonl", [mapping_row]),
        "source_path_mapping_manifest_json": write_json(
            input_dir / "source_path_mapping_manifest.json",
            {
                "schema": "source_path_mapping_manifest.v1",
                "total_rows": 1,
                "source": "mac_candidate_readiness_summary",
                "delivery_allowed": False,
            },
        ),
        "docker_mount_verification_json": write_json(
            input_dir / "docker_mount_verification.json",
            {
                "schema": "docker_mount_verification.v1",
                "host_testing_stuff_music_visible": False,
                "source": "mac_candidate_readiness_summary",
            },
        ),
        "reference_recovery_plan_jsonl": write_jsonl(
            input_dir / "reference_bundle_recovery_plan.jsonl",
            [
                {
                    "case_id": case_id,
                    "track_hash": track_hash,
                    "stem_type": "minus",
                    "split": "holdout",
                    "source_manifest": str(source_manifest),
                    "recovery_status": "blocked_missing_reference",
                    "discovered_audio_inputs": [],
                    "discovered_reference_inputs": [],
                    "rejected_reference_track_identity": dict(summary.get("reference_track_identity") or {}),
                    "forbidden_actions": ["no_audio_copy", "no_audio_read_or_hash", "no_promotion"],
                }
            ],
        ),
        "collected_evidence_jsonl": write_jsonl(input_dir / "collected_evidence.jsonl", [{"case_id": case_id}]),
        "hard_case_benchmark_jsonl": write_jsonl(
            input_dir / "hard_case_benchmark.jsonl",
            [{"benchmark_case_id": case_id, "track_hash": track_hash, "stem_type": "minus"}],
        ),
        "golden_jsonl": write_jsonl(input_dir / "golden_corpus.jsonl", []),
    }
    return paths


def _attach_mac_context(rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[dict[str, Any]]:
    source_identity = dict(summary.get("source_track_identity") or {})
    rejected_identity = dict(summary.get("reference_track_identity") or {})
    for row in rows:
        row["source_audio_path"] = str(summary.get("source_audio_path") or "")
        row["source_track_identity"] = source_identity
        row["selected_candidate_id"] = str(summary.get("selected_candidate_id") or "")
        row["reference_bundle_status"] = str(summary.get("reference_bundle_status") or summary.get("reference_audit_status") or "")
        row["reference_mismatch_status"] = (
            "reference_bundle_mismatched_to_source"
            if str(summary.get("reference_audit_status") or "") == "mismatched_to_source"
            else ""
        )
        row["rejected_reference_track_identity"] = rejected_identity
        row["readiness_summary_json"] = str(summary.get("manifest") or "")
    return rows


def build_mac_reference_path_recovery_plan(
    *,
    mac_readiness_summary_json: Path,
    output_dir: Path,
    reference_search_roots: list[Path] | None = None,
) -> dict[str, Any]:
    summary = read_json(mac_readiness_summary_json)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary["manifest"] = str(mac_readiness_summary_json)
    input_paths = _input_bundle(summary, output_dir)
    result = build_reference_path_recovery_plan(
        source_path_mapping_jsonl=input_paths["source_path_mapping_jsonl"],
        source_path_mapping_manifest_json=input_paths["source_path_mapping_manifest_json"],
        docker_mount_verification_json=input_paths["docker_mount_verification_json"],
        reference_recovery_plan_jsonl=input_paths["reference_recovery_plan_jsonl"],
        collected_evidence_jsonl=input_paths["collected_evidence_jsonl"],
        hard_case_benchmark_jsonl=input_paths["hard_case_benchmark_jsonl"],
        golden_jsonl=input_paths["golden_jsonl"],
        reference_search_roots=reference_search_roots,
    )
    result["rows"] = _attach_mac_context(result["rows"], summary)
    result["summary"]["source"] = "mac_candidate_readiness_summary"
    result["summary"]["mac_readiness_summary_json"] = str(mac_readiness_summary_json)
    result["manifest"]["source"] = "mac_candidate_readiness_summary"
    result["manifest"]["mac_readiness_summary_json"] = str(mac_readiness_summary_json)
    result["manifest"]["delivery_allowed"] = False
    result["manifest"]["presentation_status"] = "reference_recovery_plan_not_for_listen_pack"
    result["manifest"]["input_files"]["mac_readiness_summary_json"] = str(mac_readiness_summary_json)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mac-readiness-summary-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-search-root", type=Path, action="append", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_mac_reference_path_recovery_plan(
        mac_readiness_summary_json=args.mac_readiness_summary_json,
        output_dir=args.output_dir,
        reference_search_roots=args.reference_search_root,
    )
    write_outputs(result, args.output_dir)
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / "reference_path_recovery_manifest.json"),
                "total_rows": result["summary"]["total_rows"],
                "recovery_status_counts": result["summary"]["recovery_status_counts"],
                "delivery_allowed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
