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


ALLOWED_MISSING_EVIDENCE_REASONS = {
    "missing_reference_vocal",
    "missing_reference_minus",
    "missing_original_mix",
    "missing_generated_vocal",
    "missing_generated_minus",
    "missing_gt_bundle",
    "missing_alignment_evidence",
    "missing_residual_metrics",
    "missing_leakage_metrics",
    "missing_full_mix_reconstruction",
    "missing_source_manifest",
    "missing_stem_type",
    "insufficient_manifest_provenance",
    "blocked_by_holdout",
    "blocked_by_golden",
    "blocked_by_policy",
    "blocked_by_unknown_reason",
}

EXACT_BLOCKER_PRIORITY = [
    "blocked_by_holdout",
    "blocked_by_golden",
    "blocked_by_policy",
    "missing_source_manifest",
    "missing_stem_type",
    "missing_reference_vocal",
    "missing_reference_minus",
    "missing_original_mix",
    "missing_generated_vocal",
    "missing_generated_minus",
    "missing_gt_bundle",
    "missing_alignment_evidence",
    "missing_residual_metrics",
    "missing_leakage_metrics",
    "missing_full_mix_reconstruction",
    "insufficient_manifest_provenance",
    "blocked_by_unknown_reason",
]

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
    hashes: set[str] = set()
    for row in iter_jsonl(golden_jsonl):
        track_hash = row.get("track_hash")
        if track_hash:
            hashes.add(str(track_hash))
    return hashes


def bool_value(row: dict[str, Any], key: str) -> bool:
    return bool(row.get(key))


def stem_reference_reason(stem_type: str) -> str:
    if stem_type == "minus":
        return "missing_reference_minus"
    if stem_type == "vocal":
        return "missing_reference_vocal"
    return "blocked_by_unknown_reason"


def reference_type_for_reason(reason: str) -> str | None:
    if reason == "missing_reference_vocal":
        return "reference_vocal"
    if reason == "missing_reference_minus":
        return "reference_minus"
    return None


def source_types_for_reasons(reasons: list[str]) -> list[str]:
    mapping = {
        "missing_original_mix": "original_mix",
        "missing_source_manifest": "source_manifest",
        "missing_stem_type": "stem_type",
        "missing_generated_vocal": "generated_vocal",
        "missing_generated_minus": "generated_minus",
    }
    return stable_unique(mapping[reason] for reason in reasons if reason in mapping)


def metric_types_for_reasons(reasons: list[str]) -> list[str]:
    metric_reasons = {
        "missing_alignment_evidence",
        "missing_residual_metrics",
        "missing_leakage_metrics",
        "missing_full_mix_reconstruction",
    }
    return [reason for reason in reasons if reason in metric_reasons]


def bundle_types_for_reasons(reasons: list[str]) -> list[str]:
    mapping = {
        "missing_gt_bundle": "gt_bundle",
        "missing_full_mix_reconstruction": "full_mix_reconstruction",
    }
    return stable_unique(mapping[reason] for reason in reasons if reason in mapping)


def exact_blocker(reasons: list[str]) -> str:
    for blocker in EXACT_BLOCKER_PRIORITY:
        if blocker in reasons:
            return blocker
    raise ValueError("At least one exact blocker is required")


def required_actions_for_reasons(reasons: list[str]) -> list[str]:
    actions: list[str] = []
    if "blocked_by_holdout" in reasons:
        actions.append("keep_as_holdout_only")
    if "blocked_by_golden" in reasons:
        actions.append("keep_as_golden_regression_only")
    if "blocked_by_policy" in reasons:
        actions.append("resolve_policy_issue_before_dataset_use")
    if "missing_source_manifest" in reasons or "insufficient_manifest_provenance" in reasons:
        actions.append("repair_manifest_provenance")
    if "missing_stem_type" in reasons:
        actions.append("resolve_stem_type")
    if "missing_reference_vocal" in reasons or "missing_reference_minus" in reasons:
        actions.append("recover_or_attach_verified_reference_path")
    if "missing_original_mix" in reasons:
        actions.append("recover_source_audio_path_or_docker_mount")
    if "missing_generated_vocal" in reasons or "missing_generated_minus" in reasons:
        actions.append("recover_generated_candidate_path")
    if "missing_gt_bundle" in reasons or "missing_full_mix_reconstruction" in reasons:
        actions.append("attach_or_build_verified_gt_bundle_in_later_approved_phase")
    if (
        "missing_alignment_evidence" in reasons
        or "missing_residual_metrics" in reasons
        or "missing_leakage_metrics" in reasons
    ):
        actions.append("collect_reference_metrics_after_prerequisites")
    if "blocked_by_unknown_reason" in reasons:
        actions.append("manual_manifest_inspection")
    actions.append("explicit_later_phase_approval_required")
    return stable_unique(actions)


def protection_reasons(
    row: dict[str, Any],
    *,
    golden_track_hashes: set[str],
) -> list[str]:
    status = str(row.get("preflight_status") or "")
    split = str(row.get("split") or "")
    track_hash = str(row.get("track_hash") or "")
    if status == "blocked_holdout" or split == "holdout":
        return ["blocked_by_holdout"]
    if status == "blocked_golden" or track_hash in golden_track_hashes:
        return ["blocked_by_golden"]
    if status == "blocked_policy_issue":
        return ["blocked_by_policy"]
    return []


def derive_missing_reasons(
    row: dict[str, Any],
    *,
    golden_track_hashes: set[str],
) -> list[str]:
    protected = protection_reasons(row, golden_track_hashes=golden_track_hashes)
    if protected:
        return protected

    reasons: list[str] = []
    stem_type = str(row.get("stem_type") or "unknown")

    if not row.get("source_manifest"):
        reasons.extend(["missing_source_manifest", "insufficient_manifest_provenance"])
    if stem_type not in {"vocal", "minus"}:
        reasons.append("missing_stem_type")
    if not bool_value(row, "has_source_audio_path"):
        reasons.append("missing_original_mix")
    if not bool_value(row, "has_reference_path"):
        reasons.append(stem_reference_reason(stem_type))
    if not bool_value(row, "has_gt_bundle"):
        reasons.extend(["missing_gt_bundle", "missing_full_mix_reconstruction"])
    if not bool_value(row, "has_alignment_evidence"):
        reasons.append("missing_alignment_evidence")
    if not bool_value(row, "has_residual_metrics"):
        reasons.append("missing_residual_metrics")
    if not bool_value(row, "has_leakage_metrics"):
        reasons.append("missing_leakage_metrics")

    status = str(row.get("preflight_status") or "")
    if status == "blocked_insufficient_evidence" and not reasons:
        reasons.append("insufficient_manifest_provenance")
    if not reasons:
        reasons.append("blocked_by_unknown_reason")

    return [reason for reason in stable_unique(reasons) if reason in ALLOWED_MISSING_EVIDENCE_REASONS]


def merge_forbidden_actions(row: dict[str, Any], reasons: list[str]) -> list[str]:
    actions = list(DEFAULT_FORBIDDEN_ACTIONS)
    actions.extend(as_list(row.get("forbidden_actions")))
    if "blocked_by_holdout" in reasons:
        actions.append("holdout_protected_no_training")
    if "blocked_by_golden" in reasons:
        actions.append("golden_protected_no_training_or_tuning")
    return stable_unique(actions)


def report_row(
    row: dict[str, Any],
    *,
    golden_track_hashes: set[str],
    root_rows: dict[str, dict[str, Any]],
    autonomous_rows: dict[str, dict[str, Any]],
    benchmark_rows: dict[str, dict[str, Any]],
    reference_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    case_id = row_id(row)
    root_row = root_rows.get(case_id, {})
    autonomous_row = autonomous_rows.get(case_id, {})
    benchmark_row = benchmark_rows.get(case_id, {})
    reference_row = reference_rows.get(case_id, {})

    source_manifest = (
        row.get("source_manifest")
        or benchmark_row.get("source_manifest")
        or autonomous_row.get("source_manifest")
        or root_row.get("source_manifest")
        or reference_row.get("source_manifest")
        or ""
    )
    stem_type = str(row.get("stem_type") or benchmark_row.get("stem_type") or root_row.get("stem_type") or "unknown")
    split = str(row.get("split") or benchmark_row.get("split") or root_row.get("split") or "unknown")
    track_hash = str(row.get("track_hash") or benchmark_row.get("track_hash") or root_row.get("track_hash") or "")

    enriched = dict(row)
    enriched.update(
        {
            "case_id": case_id,
            "source_manifest": source_manifest,
            "stem_type": stem_type,
            "split": split,
            "track_hash": track_hash,
        }
    )

    reasons = derive_missing_reasons(enriched, golden_track_hashes=golden_track_hashes)
    blocker = exact_blocker(reasons)
    reference_types = stable_unique(
        item
        for item in (reference_type_for_reason(reason) for reason in reasons)
        if item is not None
    )
    metric_types = metric_types_for_reasons(reasons)

    return {
        "case_id": case_id,
        "track_hash": track_hash,
        "stem_type": stem_type,
        "split": split,
        "source_manifest": source_manifest,
        "preflight_status": str(row.get("preflight_status") or ""),
        "original_root_cause_bucket": str(
            row.get("original_root_cause_bucket")
            or root_row.get("root_cause_bucket")
            or ""
        ),
        "original_safe_next_action": str(
            row.get("original_safe_next_action")
            or root_row.get("safe_next_action")
            or ""
        ),
        "missing_evidence_reasons": reasons,
        "missing_reference_types": reference_types,
        "missing_source_types": source_types_for_reasons(reasons),
        "missing_metric_types": metric_types,
        "missing_bundle_types": bundle_types_for_reasons(reasons),
        "exact_blocker": blocker,
        "required_to_unblock": required_actions_for_reasons(reasons),
        "can_be_unblocked_by_manifest_fix": bool(
            {"missing_source_manifest", "missing_stem_type", "insufficient_manifest_provenance", "blocked_by_unknown_reason"}
            & set(reasons)
        )
        and not {"blocked_by_holdout", "blocked_by_golden"} & set(reasons),
        "can_be_unblocked_by_reference_path_fix": bool(
            {"missing_reference_vocal", "missing_reference_minus"} & set(reasons)
        )
        and not {"blocked_by_holdout", "blocked_by_golden"} & set(reasons),
        "can_be_unblocked_by_docker_mount": bool(
            {"missing_original_mix", "missing_generated_vocal", "missing_generated_minus"} & set(reasons)
        )
        and not {"blocked_by_holdout", "blocked_by_golden"} & set(reasons),
        "can_be_unblocked_by_gt_bundle_build": bool(
            {"missing_gt_bundle", "missing_full_mix_reconstruction"} & set(reasons)
        )
        and not {"blocked_by_holdout", "blocked_by_golden"} & set(reasons),
        "can_be_unblocked_automatically": False,
        "can_start_training_now": False,
        "forbidden_actions": merge_forbidden_actions(enriched, reasons),
    }


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key) or "") for row in rows).items()))


def count_list_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(as_list(row.get(key)))
    return dict(sorted(counter.items()))


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_rows": len(rows),
        "preflight_status_counts": count_by(rows, "preflight_status"),
        "exact_blocker_counts": count_by(rows, "exact_blocker"),
        "missing_evidence_reason_counts": count_list_values(rows, "missing_evidence_reasons"),
        "missing_reference_type_counts": count_list_values(rows, "missing_reference_types"),
        "missing_metric_type_counts": count_list_values(rows, "missing_metric_types"),
        "missing_bundle_type_counts": count_list_values(rows, "missing_bundle_types"),
        "can_be_unblocked_by_manifest_fix_count": sum(1 for row in rows if row["can_be_unblocked_by_manifest_fix"]),
        "can_be_unblocked_by_reference_path_fix_count": sum(
            1 for row in rows if row["can_be_unblocked_by_reference_path_fix"]
        ),
        "can_be_unblocked_by_docker_mount_count": sum(1 for row in rows if row["can_be_unblocked_by_docker_mount"]),
        "can_be_unblocked_by_gt_bundle_build_count": sum(
            1 for row in rows if row["can_be_unblocked_by_gt_bundle_build"]
        ),
        "can_be_unblocked_automatically_count": sum(1 for row in rows if row["can_be_unblocked_automatically"]),
        "can_start_training_now_count": sum(1 for row in rows if row["can_start_training_now"]),
        "safety": dict(SAFETY_FLAGS),
    }


def build_manifest(
    *,
    input_paths: dict[str, Path],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "finetune_evidence_gap_report.v1",
        "row_count": len(rows),
        "total_rows": len(rows),
        "input_files": {name: str(path) for name, path in sorted(input_paths.items())},
        "input_sha256": {name: sha256_file(path) for name, path in sorted(input_paths.items())},
        "outputs": [
            "finetune_evidence_gap_report.jsonl",
            "finetune_evidence_gap_manifest.json",
            "finetune_evidence_gap_summary.md",
        ],
        "summary": summary,
        "safety": dict(SAFETY_FLAGS),
    }


def build_finetune_evidence_gap_report(
    *,
    finetune_preflight_jsonl: Path,
    finetune_preflight_manifest_json: Path,
    root_cause_plan_jsonl: Path,
    autonomous_decisions_jsonl: Path,
    hard_case_benchmark_jsonl: Path,
    golden_jsonl: Path,
    reference_path_recovery_jsonl: Path,
) -> dict[str, Any]:
    input_paths = {
        "finetune_preflight_jsonl": finetune_preflight_jsonl,
        "finetune_preflight_manifest_json": finetune_preflight_manifest_json,
        "root_cause_plan_jsonl": root_cause_plan_jsonl,
        "autonomous_decisions_jsonl": autonomous_decisions_jsonl,
        "hard_case_benchmark_jsonl": hard_case_benchmark_jsonl,
        "golden_jsonl": golden_jsonl,
        "reference_path_recovery_jsonl": reference_path_recovery_jsonl,
    }
    for path in input_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() not in METADATA_SUFFIXES:
            raise ValueError(f"Expected metadata input, got: {path}")

    preflight_manifest = read_json(finetune_preflight_manifest_json)
    preflight_rows = list(iter_jsonl(finetune_preflight_jsonl))
    manifest_row_count = preflight_manifest.get("row_count")
    if manifest_row_count is not None and int(manifest_row_count) != len(preflight_rows):
        raise ValueError(
            "finetune preflight manifest row_count does not match JSONL rows: "
            f"{manifest_row_count} != {len(preflight_rows)}"
        )

    root_rows = rows_by_id(root_cause_plan_jsonl)
    autonomous_rows = rows_by_id(autonomous_decisions_jsonl)
    benchmark_rows = rows_by_id(hard_case_benchmark_jsonl)
    reference_rows = rows_by_id(reference_path_recovery_jsonl)
    golden_track_hashes = load_golden_track_hashes(golden_jsonl)

    rows = [
        report_row(
            row,
            golden_track_hashes=golden_track_hashes,
            root_rows=root_rows,
            autonomous_rows=autonomous_rows,
            benchmark_rows=benchmark_rows,
            reference_rows=reference_rows,
        )
        for row in sorted(preflight_rows, key=row_id)
    ]
    for row in rows:
        if not row["exact_blocker"]:
            raise ValueError(f"Missing exact_blocker for {row['case_id']}")
        if not row["required_to_unblock"]:
            raise ValueError(f"Missing required_to_unblock for {row['case_id']}")
        if row["can_start_training_now"]:
            raise ValueError(f"Training cannot start now for {row['case_id']}")

    summary = build_summary(rows)
    manifest = build_manifest(input_paths=input_paths, rows=rows, summary=summary)
    return {"rows": rows, "summary": summary, "manifest": manifest}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Fine-tune Evidence Gap Report",
        "",
        "Metadata-only report for blocked fine-tune candidates.",
        "",
        f"- total_rows: {summary['total_rows']}",
        f"- can_be_unblocked_by_manifest_fix_count: {summary['can_be_unblocked_by_manifest_fix_count']}",
        f"- can_be_unblocked_by_reference_path_fix_count: {summary['can_be_unblocked_by_reference_path_fix_count']}",
        f"- can_be_unblocked_by_docker_mount_count: {summary['can_be_unblocked_by_docker_mount_count']}",
        f"- can_be_unblocked_by_gt_bundle_build_count: {summary['can_be_unblocked_by_gt_bundle_build_count']}",
        f"- can_be_unblocked_automatically_count: {summary['can_be_unblocked_automatically_count']}",
        f"- can_start_training_now_count: {summary['can_start_training_now_count']}",
        "",
        "## Exact Blockers",
        "",
    ]
    for key, value in summary["exact_blocker_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Missing Reference Types", ""])
    for key, value in summary["missing_reference_type_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Missing Metric Types", ""])
    for key, value in summary["missing_metric_type_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Safety", ""])
    for key, value in summary["safety"].items():
        lines.append(f"- {key}: {str(value).lower()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "finetune_evidence_gap_report.jsonl", result["rows"])
    (output_dir / "finetune_evidence_gap_manifest.json").write_text(
        json.dumps(result["manifest"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary_md(output_dir / "finetune_evidence_gap_summary.md", result["summary"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finetune-preflight-jsonl", required=True, type=Path)
    parser.add_argument("--finetune-preflight-manifest-json", required=True, type=Path)
    parser.add_argument("--root-cause-plan-jsonl", required=True, type=Path)
    parser.add_argument("--autonomous-decisions-jsonl", required=True, type=Path)
    parser.add_argument("--hard-case-benchmark-jsonl", required=True, type=Path)
    parser.add_argument("--golden-jsonl", required=True, type=Path)
    parser.add_argument("--reference-path-recovery-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    result = build_finetune_evidence_gap_report(
        finetune_preflight_jsonl=args.finetune_preflight_jsonl,
        finetune_preflight_manifest_json=args.finetune_preflight_manifest_json,
        root_cause_plan_jsonl=args.root_cause_plan_jsonl,
        autonomous_decisions_jsonl=args.autonomous_decisions_jsonl,
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
