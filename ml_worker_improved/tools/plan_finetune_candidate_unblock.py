from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


ALLOWED_UNBLOCK_ACTION_TYPES = {
    "provide_reference_vocal_path",
    "provide_reference_minus_path",
    "repair_stem_type_metadata",
    "repair_manifest_provenance",
    "repair_docker_path_mapping",
    "prepare_gt_bundle_after_reference_fix",
    "keep_blocked_until_reference_available",
}

ALLOWED_UNBLOCK_STATUSES = {
    "blocked_waiting_reference_vocal",
    "blocked_waiting_reference_minus",
    "blocked_waiting_manifest_fix",
    "blocked_waiting_docker_path_fix",
    "blocked_waiting_gt_bundle",
    "blocked_manual_input_required",
    "ready_for_future_unblock_after_required_files",
}

DEFAULT_FORBIDDEN_ACTIONS = [
    "do_not_add_models",
    "do_not_add_weights",
    "do_not_start_training",
    "do_not_build_dataset",
    "do_not_build_gt_bundle",
    "do_not_copy_or_export_audio",
    "do_not_read_or_hash_audio",
    "do_not_add_cleanup_debleed_denoise",
    "do_not_relax_thresholds",
    "do_not_change_policy",
    "do_not_promote_artifacts",
]

SAFETY_FLAGS = {
    "audio_copied": False,
    "audio_read": False,
    "audio_hashed": False,
    "dataset_built": False,
    "gt_bundle_built": False,
    "training_started": False,
    "policy_changed": False,
    "models_changed": False,
    "weights_added": False,
    "cleanup_routes_added": False,
    "thresholds_changed": False,
}

METADATA_SUFFIXES = {".json", ".jsonl"}


def sha256_file(path: Path) -> str:
    if path.suffix.lower() not in METADATA_SUFFIXES:
        raise ValueError(f"Refusing to hash non-metadata input: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    if isinstance(value, set):
        return sorted(str(item) for item in value if str(item))
    text = str(value)
    return [text] if text else []


def stable_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def row_id(row: dict[str, Any]) -> str:
    for key in ("case_id", "benchmark_case_id", "source_hard_case_id"):
        value = row.get(key)
        if value:
            return str(value)
    raise ValueError(f"Row is missing case id: {row}")


def rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {row_id(row): row for row in iter_jsonl(path)}


def load_golden_track_hashes(golden_jsonl: Path) -> set[str]:
    return {str(row["track_hash"]) for row in iter_jsonl(golden_jsonl) if row.get("track_hash")}


def action_for_row(row: dict[str, Any]) -> str:
    exact_blocker = str(row.get("exact_blocker") or "")
    reasons = set(as_list(row.get("missing_evidence_reasons")))
    if exact_blocker == "missing_reference_vocal" or "missing_reference_vocal" in reasons:
        return "provide_reference_vocal_path"
    if exact_blocker == "missing_reference_minus" or "missing_reference_minus" in reasons:
        return "provide_reference_minus_path"
    if exact_blocker == "missing_stem_type" or "missing_stem_type" in reasons:
        return "repair_stem_type_metadata"
    if exact_blocker in {"missing_source_manifest", "insufficient_manifest_provenance"}:
        return "repair_manifest_provenance"
    if row.get("can_be_unblocked_by_docker_mount") or "missing_original_mix" in reasons:
        return "repair_docker_path_mapping"
    if exact_blocker in {"missing_gt_bundle", "missing_full_mix_reconstruction"}:
        return "prepare_gt_bundle_after_reference_fix"
    return "keep_blocked_until_reference_available"


def status_for_action(action: str) -> str:
    mapping = {
        "provide_reference_vocal_path": "blocked_waiting_reference_vocal",
        "provide_reference_minus_path": "blocked_waiting_reference_minus",
        "repair_stem_type_metadata": "blocked_waiting_manifest_fix",
        "repair_manifest_provenance": "blocked_waiting_manifest_fix",
        "repair_docker_path_mapping": "blocked_waiting_docker_path_fix",
        "prepare_gt_bundle_after_reference_fix": "blocked_waiting_gt_bundle",
        "keep_blocked_until_reference_available": "blocked_manual_input_required",
    }
    return mapping[action]


def required_reference_type(action: str) -> str:
    if action == "provide_reference_vocal_path":
        return "reference_vocal"
    if action == "provide_reference_minus_path":
        return "reference_minus"
    return ""


def required_manifest_fields(row: dict[str, Any], action: str) -> list[str]:
    fields: list[str] = []
    if action == "repair_stem_type_metadata" or "missing_stem_type" in as_list(row.get("missing_evidence_reasons")):
        fields.append("stem_type")
    if action == "repair_manifest_provenance" or "missing_source_manifest" in as_list(row.get("missing_evidence_reasons")):
        fields.append("source_manifest")
    if "insufficient_manifest_provenance" in as_list(row.get("missing_evidence_reasons")):
        fields.append("manifest_provenance")
    return stable_unique(fields)


def required_docker_mount(row: dict[str, Any], action: str) -> str:
    if action == "repair_docker_path_mapping" or row.get("can_be_unblocked_by_docker_mount"):
        return "source_audio_mount_or_manifest_path_mapping"
    return ""


def required_gt_bundle_inputs(row: dict[str, Any]) -> list[str]:
    inputs: list[str] = []
    missing_bundle_types = set(as_list(row.get("missing_bundle_types")))
    missing_metric_types = set(as_list(row.get("missing_metric_types")))
    if "gt_bundle" in missing_bundle_types or row.get("can_be_unblocked_by_gt_bundle_build"):
        inputs.extend(["original_mix", "generated_stem", "verified_reference_stem"])
    if "full_mix_reconstruction" in missing_bundle_types or "missing_full_mix_reconstruction" in missing_metric_types:
        inputs.append("full_mix_reconstruction_evidence")
    return stable_unique(inputs)


def can_apply_automatically(row: dict[str, Any], action: str, required_fields: list[str], gt_inputs: list[str]) -> bool:
    reasons = set(as_list(row.get("missing_evidence_reasons")))
    return (
        action in {"repair_stem_type_metadata", "repair_manifest_provenance"}
        and bool(required_fields)
        and reasons <= {"missing_stem_type", "missing_source_manifest", "insufficient_manifest_provenance"}
        and bool(row.get("source_manifest"))
        and not gt_inputs
    )


def merge_forbidden_actions(row: dict[str, Any], *, golden_protected: bool, holdout_protected: bool) -> list[str]:
    actions = list(DEFAULT_FORBIDDEN_ACTIONS)
    actions.extend(as_list(row.get("forbidden_actions")))
    if golden_protected:
        actions.append("golden_protected_no_training_or_tuning")
    if holdout_protected:
        actions.append("holdout_protected_no_training")
    return stable_unique(actions)


def unblock_row(
    row: dict[str, Any],
    *,
    golden_track_hashes: set[str],
    preflight_rows: dict[str, dict[str, Any]],
    root_rows: dict[str, dict[str, Any]],
    benchmark_rows: dict[str, dict[str, Any]],
    reference_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    case_id = row_id(row)
    preflight_row = preflight_rows.get(case_id, {})
    root_row = root_rows.get(case_id, {})
    benchmark_row = benchmark_rows.get(case_id, {})
    reference_row = reference_rows.get(case_id, {})

    track_hash = str(row.get("track_hash") or benchmark_row.get("track_hash") or preflight_row.get("track_hash") or "")
    split = str(row.get("split") or benchmark_row.get("split") or preflight_row.get("split") or "unknown")
    source_manifest = str(
        row.get("source_manifest")
        or benchmark_row.get("source_manifest")
        or preflight_row.get("source_manifest")
        or root_row.get("source_manifest")
        or reference_row.get("source_manifest")
        or ""
    )
    stem_type = str(row.get("stem_type") or benchmark_row.get("stem_type") or preflight_row.get("stem_type") or "unknown")
    enriched = dict(row)
    enriched.update({"case_id": case_id, "track_hash": track_hash, "split": split, "source_manifest": source_manifest, "stem_type": stem_type})

    action = action_for_row(enriched)
    status = status_for_action(action)
    required_fields = required_manifest_fields(enriched, action)
    docker_mount = required_docker_mount(enriched, action)
    gt_inputs = required_gt_bundle_inputs(enriched)
    reference_type = required_reference_type(action)
    requires_reference = bool(reference_type)
    requires_gt_bundle = bool(gt_inputs)
    golden_protected = track_hash in golden_track_hashes or "blocked_by_golden" in as_list(enriched.get("missing_evidence_reasons"))
    holdout_protected = split == "holdout" or "blocked_by_holdout" in as_list(enriched.get("missing_evidence_reasons"))
    automatic = can_apply_automatically(enriched, action, required_fields, gt_inputs) and not golden_protected and not holdout_protected
    can_enter_dataset_after_unblock = automatic and not requires_reference and not requires_gt_bundle and not docker_mount

    if action not in ALLOWED_UNBLOCK_ACTION_TYPES:
        raise ValueError(f"Invalid unblock action for {case_id}: {action}")
    if status not in ALLOWED_UNBLOCK_STATUSES:
        raise ValueError(f"Invalid unblock status for {case_id}: {status}")

    return {
        "case_id": case_id,
        "track_hash": track_hash,
        "stem_type": stem_type,
        "split": split,
        "source_manifest": source_manifest,
        "exact_blocker": str(enriched.get("exact_blocker") or ""),
        "required_to_unblock": as_list(enriched.get("required_to_unblock")),
        "unblock_action_type": action,
        "unblock_status": status,
        "required_reference_type": reference_type,
        "required_manifest_fields": required_fields,
        "required_docker_mount": docker_mount,
        "required_gt_bundle_inputs": gt_inputs,
        "can_apply_automatically": automatic,
        "requires_user_file": requires_reference,
        "requires_manifest_edit": bool(required_fields),
        "requires_reference_path": requires_reference,
        "requires_gt_bundle_build_later": requires_gt_bundle,
        "can_enter_dataset_after_unblock": can_enter_dataset_after_unblock,
        "can_start_training_now": False,
        "forbidden_actions": merge_forbidden_actions(
            enriched,
            golden_protected=golden_protected,
            holdout_protected=holdout_protected,
        ),
    }


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key) or "") for row in rows).items()))


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_rows": len(rows),
        "unblock_action_type_counts": count_by(rows, "unblock_action_type"),
        "unblock_status_counts": count_by(rows, "unblock_status"),
        "required_reference_vocal_count": sum(1 for row in rows if row["required_reference_type"] == "reference_vocal"),
        "required_reference_minus_count": sum(1 for row in rows if row["required_reference_type"] == "reference_minus"),
        "required_manifest_fix_count": sum(1 for row in rows if row["requires_manifest_edit"]),
        "required_docker_path_fix_count": sum(1 for row in rows if row["required_docker_mount"]),
        "required_gt_bundle_later_count": sum(1 for row in rows if row["requires_gt_bundle_build_later"]),
        "can_apply_automatically_count": sum(1 for row in rows if row["can_apply_automatically"]),
        "requires_user_file_count": sum(1 for row in rows if row["requires_user_file"]),
        "requires_manifest_edit_count": sum(1 for row in rows if row["requires_manifest_edit"]),
        "can_enter_dataset_after_unblock_count": sum(1 for row in rows if row["can_enter_dataset_after_unblock"]),
        "can_start_training_now_count": sum(1 for row in rows if row["can_start_training_now"]),
        "safety": dict(SAFETY_FLAGS),
    }


def build_manifest(input_paths: dict[str, Path], rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "finetune_candidate_unblock_plan.v1",
        "row_count": len(rows),
        "total_rows": len(rows),
        "input_files": {name: str(path) for name, path in sorted(input_paths.items())},
        "input_sha256": {name: sha256_file(path) for name, path in sorted(input_paths.items())},
        "outputs": [
            "finetune_candidate_unblock_plan.jsonl",
            "finetune_candidate_unblock_manifest.json",
            "finetune_candidate_unblock_summary.md",
        ],
        "summary": summary,
        "safety": dict(SAFETY_FLAGS),
    }


def plan_finetune_candidate_unblock(
    *,
    evidence_gap_report_jsonl: Path,
    evidence_gap_manifest_json: Path,
    finetune_preflight_jsonl: Path,
    root_cause_plan_jsonl: Path,
    hard_case_benchmark_jsonl: Path,
    golden_jsonl: Path,
    reference_path_recovery_jsonl: Path,
) -> dict[str, Any]:
    input_paths = {
        "evidence_gap_report_jsonl": evidence_gap_report_jsonl,
        "evidence_gap_manifest_json": evidence_gap_manifest_json,
        "finetune_preflight_jsonl": finetune_preflight_jsonl,
        "root_cause_plan_jsonl": root_cause_plan_jsonl,
        "hard_case_benchmark_jsonl": hard_case_benchmark_jsonl,
        "golden_jsonl": golden_jsonl,
        "reference_path_recovery_jsonl": reference_path_recovery_jsonl,
    }
    for path in input_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() not in METADATA_SUFFIXES:
            raise ValueError(f"Expected metadata input, got: {path}")

    evidence_manifest = read_json(evidence_gap_manifest_json)
    evidence_rows = list(iter_jsonl(evidence_gap_report_jsonl))
    manifest_row_count = evidence_manifest.get("row_count", evidence_manifest.get("total_rows"))
    if manifest_row_count is not None and int(manifest_row_count) != len(evidence_rows):
        raise ValueError(
            "evidence gap manifest row_count does not match JSONL rows: "
            f"{manifest_row_count} != {len(evidence_rows)}"
        )

    preflight_rows = rows_by_id(finetune_preflight_jsonl)
    root_rows = rows_by_id(root_cause_plan_jsonl)
    benchmark_rows = rows_by_id(hard_case_benchmark_jsonl)
    reference_rows = rows_by_id(reference_path_recovery_jsonl)
    golden_track_hashes = load_golden_track_hashes(golden_jsonl)

    rows = [
        unblock_row(
            row,
            golden_track_hashes=golden_track_hashes,
            preflight_rows=preflight_rows,
            root_rows=root_rows,
            benchmark_rows=benchmark_rows,
            reference_rows=reference_rows,
        )
        for row in sorted(evidence_rows, key=row_id)
    ]
    for row in rows:
        if row["can_start_training_now"]:
            raise ValueError(f"Training cannot start now for {row['case_id']}")
    summary = build_summary(rows)
    manifest = build_manifest(input_paths, rows, summary)
    return {"rows": rows, "summary": summary, "manifest": manifest}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Fine-tune Candidate Unblock Plan",
        "",
        "Metadata-only plan. It does not execute fixes, copy audio, build datasets, build GT bundles, train, or change policy.",
        "",
        f"- total_rows: {summary['total_rows']}",
        f"- required_reference_vocal_count: {summary['required_reference_vocal_count']}",
        f"- required_reference_minus_count: {summary['required_reference_minus_count']}",
        f"- required_manifest_fix_count: {summary['required_manifest_fix_count']}",
        f"- required_docker_path_fix_count: {summary['required_docker_path_fix_count']}",
        f"- required_gt_bundle_later_count: {summary['required_gt_bundle_later_count']}",
        f"- can_apply_automatically_count: {summary['can_apply_automatically_count']}",
        f"- requires_user_file_count: {summary['requires_user_file_count']}",
        f"- requires_manifest_edit_count: {summary['requires_manifest_edit_count']}",
        f"- can_enter_dataset_after_unblock_count: {summary['can_enter_dataset_after_unblock_count']}",
        f"- can_start_training_now_count: {summary['can_start_training_now_count']}",
        "",
        "## Unblock Actions",
        "",
    ]
    for key, value in summary["unblock_action_type_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Unblock Statuses", ""])
    for key, value in summary["unblock_status_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Safety", ""])
    for key, value in summary["safety"].items():
        lines.append(f"- {key}: {str(value).lower()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "finetune_candidate_unblock_plan.jsonl", result["rows"])
    (output_dir / "finetune_candidate_unblock_manifest.json").write_text(
        json.dumps(result["manifest"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary_md(output_dir / "finetune_candidate_unblock_summary.md", result["summary"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-gap-report-jsonl", required=True, type=Path)
    parser.add_argument("--evidence-gap-manifest-json", required=True, type=Path)
    parser.add_argument("--finetune-preflight-jsonl", required=True, type=Path)
    parser.add_argument("--root-cause-plan-jsonl", required=True, type=Path)
    parser.add_argument("--hard-case-benchmark-jsonl", required=True, type=Path)
    parser.add_argument("--golden-jsonl", required=True, type=Path)
    parser.add_argument("--reference-path-recovery-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    result = plan_finetune_candidate_unblock(
        evidence_gap_report_jsonl=args.evidence_gap_report_jsonl,
        evidence_gap_manifest_json=args.evidence_gap_manifest_json,
        finetune_preflight_jsonl=args.finetune_preflight_jsonl,
        root_cause_plan_jsonl=args.root_cause_plan_jsonl,
        hard_case_benchmark_jsonl=args.hard_case_benchmark_jsonl,
        golden_jsonl=args.golden_jsonl,
        reference_path_recovery_jsonl=args.reference_path_recovery_jsonl,
    )
    write_outputs(result, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "total_rows": result["summary"]["total_rows"],
                "can_start_training_now_count": result["summary"]["can_start_training_now_count"],
                "safety": result["summary"]["safety"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
