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


PREFLIGHT_STATUSES = {
    "eligible_metadata_only",
    "blocked_holdout",
    "blocked_golden",
    "blocked_missing_reference",
    "blocked_missing_source_audio",
    "blocked_missing_gt_bundle",
    "blocked_insufficient_evidence",
    "blocked_policy_issue",
}
SAFETY_FLAGS = {
    "audio_copied": False,
    "dataset_built": False,
    "gt_bundle_built": False,
    "training_started": False,
    "policy_changed": False,
    "models_changed": False,
    "weights_added": False,
    "cleanup_routes_added": False,
    "thresholds_changed": False,
}
DEFAULT_FORBIDDEN_ACTIONS = [
    "no_audio_copy",
    "no_audio_export_as_result",
    "no_dataset_build_in_phase11",
    "no_gt_bundle_build_in_phase11",
    "no_training_start",
    "no_policy_change",
    "no_threshold_change",
    "no_cleanup_debleed_denoise",
    "no_model_or_weight_addition",
]
AUDIO_SUFFIXES = (".flac", ".wav", ".mp3", ".aiff", ".aif", ".ogg", ".m4a")
JSON_SUFFIXES = (".json", ".jsonl")
SOURCE_PATH_KEY_TOKENS = (
    "source_audio_path",
    "source_input",
    "input_mix",
    "source_mix",
    "mix.path",
    "mix_path",
)
REFERENCE_PATH_KEY_TOKENS = (
    "reference_vocal",
    "vocal_reference",
    "reference_minus",
    "minus_reference",
    "reference_instrumental",
    "instrumental_reference",
    "ground_truth_vocal",
    "ground_truth_minus",
    "ground_truth_instrumental",
)
GT_MANIFEST_KEY_TOKENS = ("ground_truth_manifest", "gt_manifest")
ALIGNMENT_TOKENS = ("alignment", "offset_samples", "polarity")
RESIDUAL_TOKENS = ("residual", "residual_rms", "residual_peak")
LEAKAGE_TOKENS = ("leakage", "bleed", "interference")
METRIC_REPORT_TOKENS = ("objective_report", "quality_report", "forensic_report")
POLICY_BUCKETS = {"policy_classification_problem"}
POLICY_ACTIONS = {"improve_taxonomy_mapping", "policy_fix_candidate"}
REFERENCE_RECOVERY_ACCEPTED_STATUSES = {"reference_found", "reference_candidate_found"}
REFERENCE_RECOVERY_BLOCKING_STATUSES = {
    "blocked_missing_reference",
    "blocked_ambiguous_reference",
    "blocked_reference_outside_allowed_roots",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"json_must_be_object:{path}")
    return payload


def try_read_json(path: Path | None) -> tuple[dict[str, Any] | None, str]:
    if path is None:
        return None, "manifest_path_missing"
    if not path.exists():
        return None, "manifest_not_found"
    try:
        return read_json(path), ""
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return None, f"manifest_unreadable:{type(exc).__name__}"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"jsonl_row_must_be_object:{path}:{line_number}")
            yield payload


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.replace(",", ";").split(";") if item.strip()]
    text = str(value).strip()
    return [text] if text else []


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("case_id") or row.get("benchmark_case_id") or "")


def rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {row_id(row): row for row in iter_jsonl(path) if row_id(row)}


def load_golden_track_hashes(golden_jsonl: Path) -> set[str]:
    return {str(row.get("track_hash") or "").strip() for row in iter_jsonl(golden_jsonl) if row.get("track_hash")}


def walk_values(payload: Any, prefix: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            current = prefix + (str(key),)
            yield current, value
            yield from walk_values(value, current)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            current = prefix + (str(index),)
            yield current, value
            yield from walk_values(value, current)


def resolve_manifest_path(manifest_path: Path | None, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute() or manifest_path is None:
        return candidate
    return (manifest_path.parent / candidate).resolve()


def path_text(value: Any) -> str:
    return str(value or "").strip()


def looks_like_path(value: str) -> bool:
    lower = value.lower()
    return "/" in value or "\\" in value or lower.endswith(AUDIO_SUFFIXES + JSON_SUFFIXES)


def existing_path_from_value(manifest_path: Path | None, value: str) -> str:
    if not value or not looks_like_path(value):
        return ""
    resolved = resolve_manifest_path(manifest_path, value)
    return str(resolved) if resolved.exists() else ""


def key_matches(key: str, tokens: tuple[str, ...]) -> bool:
    key_l = key.lower()
    return any(token in key_l for token in tokens)


def scan_manifest_payload(manifest_path: Path | None, payload: dict[str, Any] | None) -> dict[str, Any]:
    evidence = {
        "source_audio_paths": [],
        "reference_paths": [],
        "gt_bundle_paths": [],
        "metric_report_paths": [],
        "has_alignment_evidence": False,
        "has_residual_metrics": False,
        "has_leakage_metrics": False,
    }
    if payload is None:
        return evidence

    source_audio_paths: set[str] = set()
    reference_paths: set[str] = set()
    gt_bundle_paths: set[str] = set()
    metric_report_paths: set[str] = set()
    for path_parts, value in walk_values(payload):
        key = ".".join(path_parts)
        key_l = key.lower()
        value_s = path_text(value)
        value_l = value_s.lower()
        if any(token in key_l for token in ALIGNMENT_TOKENS):
            evidence["has_alignment_evidence"] = True
        if any(token in key_l for token in RESIDUAL_TOKENS):
            evidence["has_residual_metrics"] = True
        if any(token in key_l for token in LEAKAGE_TOKENS):
            evidence["has_leakage_metrics"] = True
        if not isinstance(value, str) or not value_s:
            continue
        existing = existing_path_from_value(manifest_path, value_s)
        if key_matches(key_l, GT_MANIFEST_KEY_TOKENS) and value_l.endswith(".json"):
            if existing:
                gt_bundle_paths.add(existing)
            continue
        if key_matches(key_l, SOURCE_PATH_KEY_TOKENS) and value_l.endswith(AUDIO_SUFFIXES):
            if existing:
                source_audio_paths.add(existing)
            continue
        if key_matches(key_l, REFERENCE_PATH_KEY_TOKENS) and value_l.endswith(AUDIO_SUFFIXES):
            if existing:
                reference_paths.add(existing)
            continue
        if key_matches(key_l, METRIC_REPORT_TOKENS) and value_l.endswith(".json"):
            if existing:
                metric_report_paths.add(existing)

    evidence["source_audio_paths"] = sorted(source_audio_paths)
    evidence["reference_paths"] = sorted(reference_paths)
    evidence["gt_bundle_paths"] = sorted(gt_bundle_paths)
    evidence["metric_report_paths"] = sorted(metric_report_paths)
    return evidence


def enrich_from_gt_manifest(evidence: dict[str, Any], gt_manifest_path: str, stem_type: str) -> None:
    payload, error = try_read_json(Path(gt_manifest_path))
    if payload is None:
        return
    gt_path = Path(gt_manifest_path)
    gt_evidence = scan_manifest_payload(gt_path, payload)
    evidence["source_audio_paths"] = sorted(set(evidence["source_audio_paths"]) | set(gt_evidence["source_audio_paths"]))
    evidence["reference_paths"] = sorted(set(evidence["reference_paths"]) | set(gt_evidence["reference_paths"]))
    for path_parts, value in walk_values(payload):
        key = ".".join(path_parts).lower()
        value_s = path_text(value)
        if not isinstance(value, str) or not value_s or not value_s.lower().endswith(AUDIO_SUFFIXES):
            continue
        if "source_mix" in key or key.endswith("mix.path"):
            existing = existing_path_from_value(gt_path, value_s)
            if existing:
                evidence["source_audio_paths"] = sorted(set(evidence["source_audio_paths"]) | {existing})
        if "stems" in key:
            if stem_type in {"unknown", ""} or stem_type in key or (stem_type == "vocal" and "vocals" in key):
                existing = existing_path_from_value(gt_path, value_s)
                if existing:
                    evidence["reference_paths"] = sorted(set(evidence["reference_paths"]) | {existing})
    evidence["gt_manifest_read_error"] = error


def enrich_from_metric_report(evidence: dict[str, Any], report_path: str) -> None:
    payload, _error = try_read_json(Path(report_path))
    if payload is None:
        return
    for path_parts, _value in walk_values(payload):
        key = ".".join(path_parts).lower()
        if any(token in key for token in ALIGNMENT_TOKENS):
            evidence["has_alignment_evidence"] = True
        if any(token in key for token in RESIDUAL_TOKENS):
            evidence["has_residual_metrics"] = True
        if any(token in key for token in LEAKAGE_TOKENS):
            evidence["has_leakage_metrics"] = True


def inspect_source_manifest(source_manifest: str, stem_type: str) -> dict[str, Any]:
    manifest_path = Path(source_manifest) if source_manifest else None
    payload, error = try_read_json(manifest_path)
    evidence = scan_manifest_payload(manifest_path, payload)
    for gt_path in list(evidence["gt_bundle_paths"]):
        enrich_from_gt_manifest(evidence, gt_path, stem_type)
    for report_path in list(evidence["metric_report_paths"]):
        enrich_from_metric_report(evidence, report_path)
    evidence["source_manifest_exists"] = bool(manifest_path and manifest_path.exists())
    evidence["source_manifest_error"] = error
    return evidence


def source_manifest_for_case(
    root_row: dict[str, Any],
    benchmark_row: dict[str, Any] | None,
    decision_row: dict[str, Any] | None,
) -> str:
    for row in (benchmark_row, decision_row, root_row):
        if isinstance(row, dict) and str(row.get("source_manifest") or "").strip():
            return str(row.get("source_manifest")).strip()
    return ""


def reference_recovery_blocks(reference_row: dict[str, Any] | None) -> bool:
    if not reference_row:
        return False
    status = str(reference_row.get("recovery_status") or "")
    if status in REFERENCE_RECOVERY_ACCEPTED_STATUSES:
        return False
    return status in REFERENCE_RECOVERY_BLOCKING_STATUSES


def status_for_row(
    *,
    root_row: dict[str, Any],
    track_hash: str,
    split: str,
    stem_type: str,
    golden_hashes: set[str],
    has_reference_path: bool,
    has_source_audio_path: bool,
    has_gt_bundle: bool,
    has_alignment_evidence: bool,
    has_residual_metrics: bool,
    has_leakage_metrics: bool,
    reference_row: dict[str, Any] | None,
) -> str:
    if split == "holdout":
        return "blocked_holdout"
    if track_hash in golden_hashes:
        return "blocked_golden"
    if (
        str(root_row.get("root_cause_bucket") or "") in POLICY_BUCKETS
        or str(root_row.get("safe_next_action") or "") in POLICY_ACTIONS
        or root_row.get("can_change_policy") is True
    ):
        return "blocked_policy_issue"
    if reference_recovery_blocks(reference_row):
        return "blocked_missing_reference"
    if not has_reference_path:
        return "blocked_missing_reference"
    if not has_source_audio_path:
        return "blocked_missing_source_audio"
    if not has_gt_bundle:
        return "blocked_missing_gt_bundle"
    if stem_type in {"", "unknown"} or not (has_alignment_evidence and has_residual_metrics and has_leakage_metrics):
        return "blocked_insufficient_evidence"
    return "eligible_metadata_only"


def required_before_training(status: str, *, has_alignment: bool, has_residual: bool, has_leakage: bool, stem_type: str) -> list[str]:
    requirements: list[str] = ["explicit_later_phase_approval_required"]
    if status == "eligible_metadata_only":
        requirements.extend(["build_dataset_in_later_phase", "run_training_preflight_in_later_phase"])
    elif status == "blocked_holdout":
        requirements.append("keep_as_holdout_only")
    elif status == "blocked_golden":
        requirements.append("keep_as_golden_regression_only")
    elif status == "blocked_missing_reference":
        requirements.append("recover_or_attach_verified_reference_path")
    elif status == "blocked_missing_source_audio":
        requirements.append("recover_source_audio_path")
    elif status == "blocked_missing_gt_bundle":
        requirements.append("attach_or_build_verified_gt_bundle_after_approval")
    elif status == "blocked_policy_issue":
        requirements.append("resolve_policy_issue_before_dataset_use")
    else:
        if stem_type in {"", "unknown"}:
            requirements.append("resolve_stem_type")
        if not has_alignment:
            requirements.append("collect_alignment_evidence")
        if not has_residual:
            requirements.append("collect_residual_metrics")
        if not has_leakage:
            requirements.append("collect_leakage_metrics")
    return sorted(set(requirements))


def preflight_row(
    root_row: dict[str, Any],
    *,
    benchmark_row: dict[str, Any] | None,
    decision_row: dict[str, Any] | None,
    reference_row: dict[str, Any] | None,
    golden_hashes: set[str],
) -> dict[str, Any]:
    case_id = str(root_row.get("case_id") or "")
    track_hash = str(root_row.get("track_hash") or (benchmark_row or {}).get("track_hash") or "")
    split = str(root_row.get("split") or (benchmark_row or {}).get("split") or "")
    stem_type = str(root_row.get("stem_type") or (benchmark_row or {}).get("stem_type") or "unknown")
    source_manifest = source_manifest_for_case(root_row, benchmark_row, decision_row)
    evidence = inspect_source_manifest(source_manifest, stem_type)
    has_reference_path = bool(evidence["reference_paths"]) or bool(reference_row and reference_row.get("can_recover_reference_audio"))
    has_source_audio_path = bool(evidence["source_audio_paths"])
    has_gt_bundle = bool(evidence["gt_bundle_paths"])
    has_alignment = bool(evidence["has_alignment_evidence"])
    has_residual = bool(evidence["has_residual_metrics"])
    has_leakage = bool(evidence["has_leakage_metrics"])
    status = status_for_row(
        root_row=root_row,
        track_hash=track_hash,
        split=split,
        stem_type=stem_type,
        golden_hashes=golden_hashes,
        has_reference_path=has_reference_path,
        has_source_audio_path=has_source_audio_path,
        has_gt_bundle=has_gt_bundle,
        has_alignment_evidence=has_alignment,
        has_residual_metrics=has_residual,
        has_leakage_metrics=has_leakage,
        reference_row=reference_row,
    )
    if status not in PREFLIGHT_STATUSES:
        raise ValueError(f"invalid_preflight_status:{status}")
    forbidden_actions = sorted(set(as_list(root_row.get("forbidden_actions")) + DEFAULT_FORBIDDEN_ACTIONS))
    if split == "holdout":
        forbidden_actions.append("holdout_protected")
    if track_hash in golden_hashes:
        forbidden_actions.append("golden_protected_no_training_or_tuning")
    eligible = status == "eligible_metadata_only"
    return {
        "case_id": case_id,
        "track_hash": track_hash,
        "stem_type": stem_type,
        "split": split,
        "source_manifest": source_manifest,
        "original_root_cause_bucket": str(root_row.get("root_cause_bucket") or ""),
        "original_safe_next_action": str(root_row.get("safe_next_action") or ""),
        "has_reference_path": has_reference_path,
        "has_source_audio_path": has_source_audio_path,
        "has_gt_bundle": has_gt_bundle,
        "has_alignment_evidence": has_alignment,
        "has_residual_metrics": has_residual,
        "has_leakage_metrics": has_leakage,
        "eligible_for_future_finetune": eligible,
        "preflight_status": status,
        "required_before_training": required_before_training(
            status,
            has_alignment=has_alignment,
            has_residual=has_residual,
            has_leakage=has_leakage,
            stem_type=stem_type,
        ),
        "forbidden_actions": sorted(set(forbidden_actions)),
        "can_build_dataset_later": eligible,
        "can_start_training_now": False,
    }


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key) or "") for row in rows).most_common())


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = count_by(rows, "preflight_status")
    return {
        "schema": "finetune_candidate_preflight_summary.v1",
        "total_preflight_rows": len(rows),
        "preflight_status_counts": status_counts,
        "stem_type_counts": count_by(rows, "stem_type"),
        "split_counts": count_by(rows, "split"),
        "eligible_metadata_only_count": status_counts.get("eligible_metadata_only", 0),
        "blocked_holdout_count": status_counts.get("blocked_holdout", 0),
        "blocked_golden_count": status_counts.get("blocked_golden", 0),
        "blocked_missing_reference_count": status_counts.get("blocked_missing_reference", 0),
        "blocked_missing_source_audio_count": status_counts.get("blocked_missing_source_audio", 0),
        "blocked_missing_gt_bundle_count": status_counts.get("blocked_missing_gt_bundle", 0),
        "blocked_insufficient_evidence_count": status_counts.get("blocked_insufficient_evidence", 0),
        "blocked_policy_issue_count": status_counts.get("blocked_policy_issue", 0),
        "can_build_dataset_later_count": sum(1 for row in rows if row["can_build_dataset_later"]),
        "can_start_training_now_count": sum(1 for row in rows if row["can_start_training_now"]),
        "safety": dict(SAFETY_FLAGS),
    }


def build_manifest(
    *,
    summary: dict[str, Any],
    root_cause_plan_jsonl: Path,
    root_cause_manifest_json: Path,
    autonomous_decisions_jsonl: Path,
    hard_case_benchmark_jsonl: Path,
    golden_jsonl: Path,
    reference_path_recovery_jsonl: Path,
    reference_path_recovery_manifest_json: Path,
) -> dict[str, Any]:
    input_files = {
        "root_cause_plan_jsonl": str(root_cause_plan_jsonl),
        "root_cause_manifest_json": str(root_cause_manifest_json),
        "autonomous_decisions_jsonl": str(autonomous_decisions_jsonl),
        "hard_case_benchmark_jsonl": str(hard_case_benchmark_jsonl),
        "golden_jsonl": str(golden_jsonl),
        "reference_path_recovery_jsonl": str(reference_path_recovery_jsonl),
        "reference_path_recovery_manifest_json": str(reference_path_recovery_manifest_json),
    }
    return {
        "schema": "finetune_candidate_preflight_manifest.v1",
        **summary,
        "input_files": input_files,
        "input_sha256": {
            key: sha256_file(Path(value))
            for key, value in input_files.items()
            if Path(value).exists()
        },
        "safety": dict(SAFETY_FLAGS),
    }


def build_finetune_candidate_preflight(
    *,
    root_cause_plan_jsonl: Path,
    root_cause_manifest_json: Path,
    autonomous_decisions_jsonl: Path,
    hard_case_benchmark_jsonl: Path,
    golden_jsonl: Path,
    reference_path_recovery_jsonl: Path,
    reference_path_recovery_manifest_json: Path,
) -> dict[str, Any]:
    _root_manifest = read_json(root_cause_manifest_json)
    _reference_manifest = read_json(reference_path_recovery_manifest_json)
    decisions = rows_by_id(autonomous_decisions_jsonl)
    benchmark = rows_by_id(hard_case_benchmark_jsonl)
    reference_rows = rows_by_id(reference_path_recovery_jsonl)
    golden_hashes = load_golden_track_hashes(golden_jsonl)
    rows = [
        preflight_row(
            row,
            benchmark_row=benchmark.get(str(row.get("case_id") or "")),
            decision_row=decisions.get(str(row.get("case_id") or "")),
            reference_row=reference_rows.get(str(row.get("case_id") or "")),
            golden_hashes=golden_hashes,
        )
        for row in iter_jsonl(root_cause_plan_jsonl)
        if row.get("can_enter_finetune_pool") is True
    ]
    rows.sort(key=lambda row: row["case_id"])
    summary = build_summary(rows)
    manifest = build_manifest(
        summary=summary,
        root_cause_plan_jsonl=root_cause_plan_jsonl,
        root_cause_manifest_json=root_cause_manifest_json,
        autonomous_decisions_jsonl=autonomous_decisions_jsonl,
        hard_case_benchmark_jsonl=hard_case_benchmark_jsonl,
        golden_jsonl=golden_jsonl,
        reference_path_recovery_jsonl=reference_path_recovery_jsonl,
        reference_path_recovery_manifest_json=reference_path_recovery_manifest_json,
    )
    return {"rows": rows, "summary": summary, "manifest": manifest}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Fine-tune Candidate Preflight Summary",
        "",
        f"- `total_preflight_rows`: {summary['total_preflight_rows']}",
        f"- `eligible_metadata_only_count`: {summary['eligible_metadata_only_count']}",
        f"- `blocked_holdout_count`: {summary['blocked_holdout_count']}",
        f"- `blocked_golden_count`: {summary['blocked_golden_count']}",
        f"- `blocked_missing_reference_count`: {summary['blocked_missing_reference_count']}",
        f"- `blocked_missing_source_audio_count`: {summary['blocked_missing_source_audio_count']}",
        f"- `blocked_missing_gt_bundle_count`: {summary['blocked_missing_gt_bundle_count']}",
        f"- `blocked_insufficient_evidence_count`: {summary['blocked_insufficient_evidence_count']}",
        f"- `blocked_policy_issue_count`: {summary['blocked_policy_issue_count']}",
        f"- `can_build_dataset_later_count`: {summary['can_build_dataset_later_count']}",
        f"- `can_start_training_now_count`: {summary['can_start_training_now_count']}",
        "",
        "## Preflight Status",
        "",
    ]
    for key, value in summary["preflight_status_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Safety", ""])
    for key, value in summary["safety"].items():
        lines.append(f"- `{key}`: {str(value).lower()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "finetune_candidate_preflight.jsonl", result["rows"])
    (output_dir / "finetune_candidate_preflight_manifest.json").write_text(
        json.dumps(result["manifest"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary_md(output_dir / "finetune_candidate_preflight_summary.md", result["summary"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Metadata-only preflight for future fine-tune candidate pool.")
    parser.add_argument("--root-cause-plan-jsonl", type=Path, required=True)
    parser.add_argument("--root-cause-manifest-json", type=Path, required=True)
    parser.add_argument("--autonomous-decisions-jsonl", type=Path, required=True)
    parser.add_argument("--hard-case-benchmark-jsonl", type=Path, required=True)
    parser.add_argument("--golden-jsonl", type=Path, required=True)
    parser.add_argument("--reference-path-recovery-jsonl", type=Path, required=True)
    parser.add_argument("--reference-path-recovery-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_finetune_candidate_preflight(
        root_cause_plan_jsonl=args.root_cause_plan_jsonl,
        root_cause_manifest_json=args.root_cause_manifest_json,
        autonomous_decisions_jsonl=args.autonomous_decisions_jsonl,
        hard_case_benchmark_jsonl=args.hard_case_benchmark_jsonl,
        golden_jsonl=args.golden_jsonl,
        reference_path_recovery_jsonl=args.reference_path_recovery_jsonl,
        reference_path_recovery_manifest_json=args.reference_path_recovery_manifest_json,
    )
    write_outputs(result, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
