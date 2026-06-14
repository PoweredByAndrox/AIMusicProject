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


MISSING_EVIDENCE_TYPES = {
    "missing_reference_metrics",
    "missing_alignment_check",
    "missing_full_mix_reconstruction",
    "missing_residual_metrics",
    "missing_leakage_metrics",
    "missing_stem_type",
    "missing_source_manifest",
    "missing_window_bounds",
    "missing_audio_path_for_internal_analysis",
    "missing_ground_truth_bundle",
    "unknown_manifest_inconsistency",
}
REQUIRED_TOOLS = {
    "run_ground_truth_quality_gate",
    "waveform_forensic_compare",
    "localize_reference_quality_failures",
    "stem_quality_control",
    "build_flstudio_ground_truth_bundle",
    "manifest_repair_only",
    "manual_source_manifest_inspection",
}
REFERENCE_METRIC_KEYS = {
    "sdr_db",
    "si_sdr_db",
    "sir_db",
    "sar_db",
    "reference_likeness_percent",
}
RESIDUAL_METRIC_KEYS = {
    "residual_rms_dbfs",
    "residual_peak_dbfs",
    "sample_residual_rms_dbfs",
    "sample_residual_peak_dbfs",
    "max_residual_rms_dbfs",
    "max_residual_peak_dbfs",
}
LEAKAGE_METRIC_KEYS = {
    "leakage_score",
    "max_residual_leakage_strength",
    "reference_stem_leakage",
    "micro_reference_stem_leakage",
    "non_target_leakage",
}
ALIGNMENT_KEYS = {
    "alignment",
    "alignment_status",
    "report_alignment_contract",
    "candidate_offset_samples",
    "estimated_offset_samples",
}
FULL_MIX_RECONSTRUCTION_KEYS = {
    "full_mix_reconstruction",
    "full_mix_reconstruction_contract",
    "reference_sum_check",
    "source_vs_sum_correlation",
}
SAFETY_FLAGS = {
    "audio_copied": False,
    "training_started": False,
    "policy_changed": False,
    "models_changed": False,
    "weights_added": False,
    "cleanup_routes_added": False,
    "thresholds_changed": False,
}
DEFAULT_FORBIDDEN_ACTIONS = [
    "no_audio_export_as_result",
    "no_audio_copy",
    "no_training_start",
    "no_policy_auto_change",
    "no_threshold_change",
    "no_cleanup_debleed_denoise",
    "no_model_or_weight_addition",
]
MANUAL_ONLY_TOOLS = {"manual_source_manifest_inspection", "manifest_repair_only"}
AUDIO_SUFFIXES = (".flac", ".wav", ".mp3", ".aiff", ".aif", ".ogg", ".m4a")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"json_must_be_object:{path}")
    return payload


def try_read_json(path: Path | None) -> tuple[dict[str, Any] | None, str]:
    if path is None:
        return None, "source_manifest_missing"
    if not path.exists():
        return None, "source_manifest_not_found"
    try:
        return read_json(path), ""
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return None, f"source_manifest_unreadable:{type(exc).__name__}"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
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


def as_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_golden_track_hashes(golden_jsonl: Path) -> set[str]:
    hashes: set[str] = set()
    for row in iter_jsonl(golden_jsonl):
        track_hash = str(row.get("track_hash") or "").strip()
        if track_hash:
            hashes.add(track_hash)
    return hashes


def index_by_case(path: Path, *, id_fields: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        for field in id_fields:
            case_id = str(row.get(field) or "").strip()
            if case_id:
                rows[case_id] = row
                break
    return rows


def should_include(row: dict[str, Any]) -> bool:
    return (
        row.get("root_cause_bucket") == "unknown_needs_more_evidence"
        or row.get("safe_next_action") == "collect_missing_reference_evidence"
    )


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


def key_matches(path_parts: tuple[str, ...], wanted: set[str]) -> bool:
    lower_path = ".".join(path_parts).lower()
    return any(key in lower_path for key in wanted)


def has_metric(payloads: Iterable[Any], keys: set[str]) -> bool:
    for payload in payloads:
        for path_parts, value in walk_values(payload):
            if key_matches(path_parts, keys) and value not in (None, "", "not_computed", "not_run"):
                return True
    return False


def string_path_hits(payloads: Iterable[Any], *, reference_only: bool = False, audio_only: bool = False) -> list[str]:
    hits: set[str] = set()
    for payload in payloads:
        for path_parts, value in walk_values(payload):
            if not isinstance(value, str) or not value.strip():
                continue
            key = ".".join(path_parts).lower()
            text = value.strip()
            lower_text = text.lower()
            looks_like_path = "/" in text or "\\" in text or lower_text.endswith((".json",) + AUDIO_SUFFIXES)
            if not looks_like_path:
                continue
            if reference_only and not any(token in key for token in ("reference", "ground_truth", "truth", "bundle")):
                continue
            if audio_only and not (lower_text.endswith(AUDIO_SUFFIXES) or "audio" in key or "vocals_path" in key):
                continue
            hits.add(text)
    return sorted(hits)


def has_key(payloads: Iterable[Any], keys: set[str]) -> bool:
    for payload in payloads:
        for path_parts, value in walk_values(payload):
            if key_matches(path_parts, keys) and value not in (None, "", [], {}):
                return True
    return False


def first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def merged_row_context(
    root_row: dict[str, Any],
    decision_row: dict[str, Any] | None,
    benchmark_row: dict[str, Any] | None,
) -> dict[str, Any]:
    decision_row = decision_row or {}
    benchmark_row = benchmark_row or {}
    window_start = (
        root_row.get("window_start_sec")
        if root_row.get("window_start_sec") is not None
        else benchmark_row.get("window_start_sec", decision_row.get("window_start_sec"))
    )
    window_end = (
        root_row.get("window_end_sec")
        if root_row.get("window_end_sec") is not None
        else benchmark_row.get("window_end_sec", decision_row.get("window_end_sec"))
    )
    return {
        "case_id": first_non_empty(root_row.get("case_id"), decision_row.get("case_id"), benchmark_row.get("benchmark_case_id")),
        "track_hash": first_non_empty(root_row.get("track_hash"), decision_row.get("track_hash"), benchmark_row.get("track_hash")),
        "stem_type": first_non_empty(root_row.get("stem_type"), decision_row.get("stem_type"), benchmark_row.get("stem_type"), "unknown"),
        "split": first_non_empty(root_row.get("split"), decision_row.get("split"), benchmark_row.get("split")),
        "source_manifest": first_non_empty(root_row.get("source_manifest"), decision_row.get("source_manifest"), benchmark_row.get("source_manifest")),
        "dominant_failure_reason": first_non_empty(
            root_row.get("dominant_failure_reason"),
            decision_row.get("dominant_failure_reason"),
            benchmark_row.get("dominant_failure_reason"),
        ),
        "failure_labels": sorted(
            set(
                as_list(root_row.get("failure_labels"))
                + as_list(decision_row.get("failure_labels"))
                + as_list(benchmark_row.get("failure_labels"))
            )
        ),
        "failed_checks": sorted(
            set(as_list(decision_row.get("failed_checks")) + as_list(benchmark_row.get("failed_checks")))
        ),
        "original_decision": first_non_empty(root_row.get("original_decision"), decision_row.get("decision")),
        "root_cause_bucket": str(root_row.get("root_cause_bucket") or ""),
        "forbidden_actions": sorted(
            set(as_list(root_row.get("forbidden_actions")) + as_list(decision_row.get("forbidden_actions")) + DEFAULT_FORBIDDEN_ACTIONS)
        ),
        "window_start_sec": as_optional_float(window_start),
        "window_end_sec": as_optional_float(window_end),
        "available_evidence": decision_row.get("available_evidence") if isinstance(decision_row.get("available_evidence"), dict) else {},
    }


def source_manifest_path(source_manifest: str) -> Path | None:
    text = str(source_manifest or "").strip()
    if not text:
        return None
    return Path(text)


def has_reference_paths(payloads: list[Any]) -> bool:
    return bool(string_path_hits(payloads, reference_only=True))


def has_audio_paths(payloads: list[Any]) -> bool:
    return bool(string_path_hits(payloads, audio_only=True))


def build_required_inputs(context: dict[str, Any], evidence_payloads: list[Any]) -> dict[str, Any]:
    return {
        "source_manifest": context["source_manifest"],
        "track_hash": context["track_hash"],
        "stem_type": context["stem_type"],
        "window_start_sec": context["window_start_sec"],
        "window_end_sec": context["window_end_sec"],
        "audio_paths_already_in_manifest": string_path_hits(evidence_payloads, audio_only=True)[:12],
        "reference_or_ground_truth_paths_already_in_manifest": string_path_hits(evidence_payloads, reference_only=True)[:12],
    }


def tools_for_missing(
    missing: set[str],
    *,
    reference_paths_present: bool,
    audio_paths_present: bool,
) -> set[str]:
    tools: set[str] = set()
    if "missing_source_manifest" in missing:
        tools.add("manual_source_manifest_inspection")
    if "missing_stem_type" in missing or "missing_window_bounds" in missing or "unknown_manifest_inconsistency" in missing:
        tools.add("manifest_repair_only")
    if "missing_ground_truth_bundle" in missing:
        tools.add("build_flstudio_ground_truth_bundle")
    if "missing_full_mix_reconstruction" in missing or "missing_alignment_check" in missing:
        tools.add("waveform_forensic_compare")
        tools.add("build_flstudio_ground_truth_bundle")
    if "missing_reference_metrics" in missing:
        if reference_paths_present:
            tools.add("run_ground_truth_quality_gate")
            tools.add("waveform_forensic_compare")
        else:
            tools.add("build_flstudio_ground_truth_bundle")
    if "missing_residual_metrics" in missing or "missing_leakage_metrics" in missing:
        tools.add("waveform_forensic_compare")
        tools.add("localize_reference_quality_failures")
    if "missing_audio_path_for_internal_analysis" in missing:
        tools.add("manifest_repair_only")
    if audio_paths_present and "missing_stem_type" not in missing:
        tools.add("stem_quality_control")
    return tools or {"manifest_repair_only"}


def blocked_reason(
    missing: set[str],
    *,
    source_manifest_error: str,
    stem_type: str,
    can_collect_automatically: bool,
) -> str:
    if can_collect_automatically:
        return ""
    if source_manifest_error:
        return source_manifest_error
    if "missing_source_manifest" in missing:
        return "source_manifest_missing"
    if stem_type in {"", "unknown"} or "missing_stem_type" in missing:
        return "stem_type_missing_or_unknown"
    if "missing_audio_path_for_internal_analysis" in missing:
        return "audio_path_missing_for_internal_analysis"
    if "unknown_manifest_inconsistency" in missing:
        return "manifest_inconsistency_requires_repair"
    return "manual_evidence_collection_required"


def plan_missing_evidence_row(
    root_row: dict[str, Any],
    decision_row: dict[str, Any] | None,
    benchmark_row: dict[str, Any] | None,
    golden_hashes: set[str],
) -> dict[str, Any]:
    context = merged_row_context(root_row, decision_row, benchmark_row)
    source_path = source_manifest_path(context["source_manifest"])
    source_payload, source_error = try_read_json(source_path)
    evidence_payloads: list[Any] = [context["available_evidence"]]
    if source_payload is not None:
        evidence_payloads.append(source_payload)
    if benchmark_row is not None:
        evidence_payloads.append(benchmark_row)

    missing: set[str] = set()
    if source_payload is None:
        missing.add("missing_source_manifest")
    stem_type = str(context["stem_type"] or "unknown")
    if stem_type in {"", "unknown", "null", "none"}:
        missing.add("missing_stem_type")
    if context["window_start_sec"] is None or context["window_end_sec"] is None:
        missing.add("missing_window_bounds")

    reference_paths_present = has_reference_paths(evidence_payloads)
    audio_paths_present = has_audio_paths(evidence_payloads)
    if source_payload is not None:
        if not audio_paths_present:
            missing.add("missing_audio_path_for_internal_analysis")
        if not reference_paths_present:
            missing.add("missing_ground_truth_bundle")
        if not has_metric(evidence_payloads, REFERENCE_METRIC_KEYS):
            missing.add("missing_reference_metrics")
        if not has_metric(evidence_payloads, RESIDUAL_METRIC_KEYS):
            missing.add("missing_residual_metrics")
        if not has_metric(evidence_payloads, LEAKAGE_METRIC_KEYS):
            missing.add("missing_leakage_metrics")
        if not has_key(evidence_payloads, ALIGNMENT_KEYS):
            missing.add("missing_alignment_check")
        if not has_key(evidence_payloads, FULL_MIX_RECONSTRUCTION_KEYS):
            missing.add("missing_full_mix_reconstruction")
    if not missing:
        missing.add("unknown_manifest_inconsistency")

    tools = tools_for_missing(missing, reference_paths_present=reference_paths_present, audio_paths_present=audio_paths_present)
    can_collect = (
        source_payload is not None
        and audio_paths_present
        and stem_type not in {"", "unknown", "null", "none"}
        and not missing.issubset({"unknown_manifest_inconsistency"})
        and bool(tools - MANUAL_ONLY_TOOLS)
    )
    is_golden = context["track_hash"] in golden_hashes
    split = str(context["split"] or "")
    can_use_for_finetune = can_collect and split == "train" and not is_golden and stem_type not in {"", "unknown", "null", "none"}
    row = {
        "case_id": context["case_id"],
        "track_hash": context["track_hash"],
        "stem_type": stem_type,
        "split": split,
        "source_manifest": context["source_manifest"],
        "dominant_failure_reason": context["dominant_failure_reason"],
        "failure_labels": context["failure_labels"],
        "original_decision": context["original_decision"],
        "root_cause_bucket": context["root_cause_bucket"],
        "missing_evidence_types": sorted(missing),
        "required_tools_to_run": sorted(tools),
        "required_inputs": build_required_inputs(context, evidence_payloads),
        "can_collect_automatically": can_collect,
        "can_use_for_finetune_after_collection": can_use_for_finetune,
        "blocked_reason": blocked_reason(
            missing,
            source_manifest_error=source_error,
            stem_type=stem_type,
            can_collect_automatically=can_collect,
        ),
        "forbidden_actions": context["forbidden_actions"],
    }
    invalid_missing = set(row["missing_evidence_types"]) - MISSING_EVIDENCE_TYPES
    invalid_tools = set(row["required_tools_to_run"]) - REQUIRED_TOOLS
    if invalid_missing:
        raise ValueError(f"invalid_missing_evidence_types:{sorted(invalid_missing)}")
    if invalid_tools:
        raise ValueError(f"invalid_required_tools:{sorted(invalid_tools)}")
    if not row["required_tools_to_run"]:
        raise ValueError(f"missing_required_tools:{row['case_id']}")
    if not row["missing_evidence_types"]:
        raise ValueError(f"missing_missing_evidence_types:{row['case_id']}")
    if not row["can_collect_automatically"] and not row["blocked_reason"]:
        raise ValueError(f"blocked_reason_required:{row['case_id']}")
    return row


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key) or "") for row in rows).most_common())


def count_list_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(as_list(row.get(key)))
    return dict(counter.most_common())


def build_summary(rows: list[dict[str, Any]], dashboard: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "missing_evidence_plan_summary.v1",
        "row_count": len(rows),
        "rows_by_missing_evidence_type": count_list_values(rows, "missing_evidence_types"),
        "rows_by_required_tool": count_list_values(rows, "required_tools_to_run"),
        "rows_by_stem_type": count_by(rows, "stem_type"),
        "rows_by_split": count_by(rows, "split"),
        "can_collect_automatically_count": sum(1 for row in rows if row["can_collect_automatically"]),
        "can_use_for_finetune_after_collection_count": sum(
            1 for row in rows if row["can_use_for_finetune_after_collection"]
        ),
        "blocked_evidence_collection_count": sum(1 for row in rows if not row["can_collect_automatically"]),
        "holdout_protected_count": sum(1 for row in rows if row["split"] == "holdout"),
        "golden_protected_count": sum(1 for row in rows if not row["can_use_for_finetune_after_collection"] and row["track_hash"]),
        "dashboard_summary": dashboard.get("summary", {}) if isinstance(dashboard, dict) else {},
        "safety": dict(SAFETY_FLAGS),
    }


def build_manifest(
    *,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    root_cause_plan_jsonl: Path,
    root_cause_manifest_json: Path,
    autonomous_decisions_jsonl: Path,
    hard_case_benchmark_jsonl: Path,
    dashboard_json: Path,
    golden_jsonl: Path,
) -> dict[str, Any]:
    input_files = {
        "root_cause_plan_jsonl": str(root_cause_plan_jsonl),
        "root_cause_manifest_json": str(root_cause_manifest_json),
        "autonomous_decisions_jsonl": str(autonomous_decisions_jsonl),
        "hard_case_benchmark_jsonl": str(hard_case_benchmark_jsonl),
        "dashboard_json": str(dashboard_json),
        "golden_jsonl": str(golden_jsonl),
    }
    return {
        "schema": "missing_evidence_plan_manifest.v1",
        "row_count": len(rows),
        "rows_by_missing_evidence_type": summary["rows_by_missing_evidence_type"],
        "rows_by_required_tool": summary["rows_by_required_tool"],
        "can_collect_automatically_count": summary["can_collect_automatically_count"],
        "can_use_for_finetune_after_collection_count": summary["can_use_for_finetune_after_collection_count"],
        "blocked_evidence_collection_count": summary["blocked_evidence_collection_count"],
        "holdout_protected_count": summary["holdout_protected_count"],
        "golden_protected_count": summary["golden_protected_count"],
        "input_files": input_files,
        "input_sha256": {
            key: sha256_file(Path(value))
            for key, value in input_files.items()
            if Path(value).exists()
        },
        "safety": dict(SAFETY_FLAGS),
    }


def build_missing_evidence_plan(
    *,
    root_cause_plan_jsonl: Path,
    root_cause_manifest_json: Path,
    autonomous_decisions_jsonl: Path,
    hard_case_benchmark_jsonl: Path,
    dashboard_json: Path,
    golden_jsonl: Path,
) -> dict[str, Any]:
    _root_manifest = read_json(root_cause_manifest_json)
    decisions = index_by_case(autonomous_decisions_jsonl, id_fields=("case_id",))
    benchmark_rows = index_by_case(hard_case_benchmark_jsonl, id_fields=("benchmark_case_id", "case_id"))
    dashboard = read_json(dashboard_json) if dashboard_json.exists() else {}
    golden_hashes = load_golden_track_hashes(golden_jsonl)
    rows = [
        plan_missing_evidence_row(
            row,
            decisions.get(str(row.get("case_id") or "")),
            benchmark_rows.get(str(row.get("case_id") or "")),
            golden_hashes,
        )
        for row in iter_jsonl(root_cause_plan_jsonl)
        if should_include(row)
    ]
    rows.sort(key=lambda row: row["case_id"])
    summary = build_summary(rows, dashboard)
    # Recompute the exact golden overlap after rows are built.
    summary["golden_protected_count"] = sum(1 for row in rows if row["track_hash"] in golden_hashes)
    manifest = build_manifest(
        rows=rows,
        summary=summary,
        root_cause_plan_jsonl=root_cause_plan_jsonl,
        root_cause_manifest_json=root_cause_manifest_json,
        autonomous_decisions_jsonl=autonomous_decisions_jsonl,
        hard_case_benchmark_jsonl=hard_case_benchmark_jsonl,
        dashboard_json=dashboard_json,
        golden_jsonl=golden_jsonl,
    )
    return {"rows": rows, "summary": summary, "manifest": manifest}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Missing Evidence Collection Plan Summary",
        "",
        f"- `total_rows`: {summary['row_count']}",
        f"- `can_collect_automatically_count`: {summary['can_collect_automatically_count']}",
        f"- `can_use_for_finetune_after_collection_count`: {summary['can_use_for_finetune_after_collection_count']}",
        f"- `blocked_evidence_collection_count`: {summary['blocked_evidence_collection_count']}",
        f"- `holdout_protected_count`: {summary['holdout_protected_count']}",
        f"- `golden_protected_count`: {summary['golden_protected_count']}",
        "",
        "## Missing Evidence Types",
        "",
    ]
    for key, value in summary["rows_by_missing_evidence_type"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Required Tools", ""])
    for key, value in summary["rows_by_required_tool"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Stem Types", ""])
    for key, value in summary["rows_by_stem_type"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Splits", ""])
    for key, value in summary["rows_by_split"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Safety", ""])
    for key, value in summary["safety"].items():
        lines.append(f"- `{key}`: {str(value).lower()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "missing_evidence_plan.jsonl", result["rows"])
    write_summary_md(output_dir / "missing_evidence_plan_summary.md", result["summary"])
    (output_dir / "missing_evidence_plan_manifest.json").write_text(
        json.dumps(result["manifest"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan missing evidence collection for unknown/root-reference-low cases.")
    parser.add_argument("--root-cause-plan-jsonl", type=Path, required=True)
    parser.add_argument("--root-cause-manifest-json", type=Path)
    parser.add_argument("--autonomous-decisions-jsonl", type=Path, required=True)
    parser.add_argument("--hard-case-benchmark-jsonl", type=Path, required=True)
    parser.add_argument("--dashboard-json", type=Path, required=True)
    parser.add_argument("--golden-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root_cause_manifest_json = args.root_cause_manifest_json or (
        args.root_cause_plan_jsonl.parent / "root_cause_plan_manifest.json"
    )
    result = build_missing_evidence_plan(
        root_cause_plan_jsonl=args.root_cause_plan_jsonl,
        root_cause_manifest_json=root_cause_manifest_json,
        autonomous_decisions_jsonl=args.autonomous_decisions_jsonl,
        hard_case_benchmark_jsonl=args.hard_case_benchmark_jsonl,
        dashboard_json=args.dashboard_json,
        golden_jsonl=args.golden_jsonl,
    )
    write_outputs(result, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
