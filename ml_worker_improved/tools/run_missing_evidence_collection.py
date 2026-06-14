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

from tools.artifact_status_contract import status_view_from_manifest  # noqa: E402
from tools.plan_missing_evidence_collection import (  # noqa: E402
    as_list,
    iter_jsonl,
    read_json,
    string_path_hits,
    walk_values,
)


COLLECTION_STATUSES = {"collected", "partially_collected", "failed", "skipped_blocked"}
SAFETY_FLAGS = {
    "audio_copied": False,
    "audio_exported": False,
    "training_started": False,
    "policy_changed": False,
    "models_changed": False,
    "weights_added": False,
    "cleanup_routes_added": False,
    "thresholds_changed": False,
}
AUDIO_SUFFIXES = (".flac", ".wav", ".mp3", ".aiff", ".aif", ".ogg", ".m4a")
REFERENCE_REQUIRED_TOOLS = {
    "waveform_forensic_compare",
    "localize_reference_quality_failures",
    "run_ground_truth_quality_gate",
}
FORBIDDEN_AUDIO_WRITING_TOOLS = {
    "build_flstudio_ground_truth_bundle",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_golden_track_hashes(golden_jsonl: Path) -> set[str]:
    return {str(row.get("track_hash") or "").strip() for row in iter_jsonl(golden_jsonl) if row.get("track_hash")}


def path_exists(value: str | Path | None) -> bool:
    if not value:
        return False
    return Path(value).exists()


def first_existing_path(paths: Iterable[str], *, audio_only: bool = False, json_only: bool = False) -> str:
    for raw_path in paths:
        text = str(raw_path or "").strip()
        if not text:
            continue
        lower = text.lower()
        if audio_only and not lower.endswith(AUDIO_SUFFIXES):
            continue
        if json_only and not lower.endswith(".json"):
            continue
        if Path(text).exists():
            return text
    return ""


def requested_audio_paths(row: dict[str, Any], source_payload: dict[str, Any] | None) -> list[str]:
    required_inputs = row.get("required_inputs") if isinstance(row.get("required_inputs"), dict) else {}
    paths = as_list(required_inputs.get("audio_paths_already_in_manifest"))
    if source_payload:
        paths.extend(string_path_hits([source_payload], audio_only=True))
    return sorted(dict.fromkeys(paths))


def requested_reference_paths(row: dict[str, Any], source_payload: dict[str, Any] | None) -> list[str]:
    required_inputs = row.get("required_inputs") if isinstance(row.get("required_inputs"), dict) else {}
    paths = as_list(required_inputs.get("reference_or_ground_truth_paths_already_in_manifest"))
    if source_payload:
        paths.extend(string_path_hits([source_payload], reference_only=True))
    return sorted(dict.fromkeys(paths))


def source_manifest_evidence(path: str, source_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not source_payload or not path_exists(path):
        return {"exists": False}
    evidence: dict[str, Any] = {"exists": True, "sha256": sha256_file(Path(path))}
    try:
        view = status_view_from_manifest(Path(path))
        evidence.update(
            {
                "delivery_status": view.delivery_status,
                "delivery_allowed": view.delivery_allowed,
                "presentation_status": view.presentation_status,
                "release_status": view.release_status,
                "dominant_failure_reason": view.dominant_failure_reason,
                "promotable": False,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive manifest provenance capture.
        evidence["status_contract_error"] = f"{type(exc).__name__}:{exc}"
        evidence["promotable"] = False
    evidence["top_level_keys"] = sorted(str(key) for key in source_payload.keys())[:40]
    return evidence


def collect_stem_quality_metrics(audio_path: str) -> dict[str, Any]:
    from tools.stem_quality_control import _audio_metrics, _read_audio  # noqa: PLC2701

    audio, sample_rate = _read_audio(Path(audio_path))
    metrics = _audio_metrics(audio, sample_rate)
    return {
        "status": "collected",
        "audio_path": audio_path,
        "audio_sha256": sha256_file(Path(audio_path)),
        "metrics": metrics,
    }


def collect_manifest_repair_scan(row: dict[str, Any]) -> dict[str, Any]:
    required_inputs = row.get("required_inputs") if isinstance(row.get("required_inputs"), dict) else {}
    return {
        "status": "metadata_scan_only",
        "stem_type_present": bool(str(row.get("stem_type") or "").strip())
        and str(row.get("stem_type") or "").strip() != "unknown",
        "window_start_sec_present": required_inputs.get("window_start_sec") is not None,
        "window_end_sec_present": required_inputs.get("window_end_sec") is not None,
    }


def collect_waveform_forensic_metrics(reference_path: str, candidate_path: str) -> dict[str, Any]:
    from tools.waveform_forensic_compare import _load_audio, compare_waveforms_forensically  # noqa: PLC2701

    reference_audio, reference_sample_rate = _load_audio(reference_path)
    candidate_audio, candidate_sample_rate = _load_audio(candidate_path)
    if reference_sample_rate != candidate_sample_rate:
        raise RuntimeError("reference_and_candidate_sample_rate_differ")
    report = compare_waveforms_forensically(
        reference_audio=reference_audio,
        candidate_audio=candidate_audio,
        sample_rate=reference_sample_rate,
    )
    return {
        "status": report.get("status"),
        "reference_path": reference_path,
        "candidate_path": candidate_path,
        "alignment": report.get("alignment", {}),
        "residual": report.get("residual", {}),
        "leakage_attribution": report.get("leakage_attribution", {}),
        "defect_range_count": len(report.get("defect_ranges") or []),
    }


def tool_skip_reason(
    tool: str,
    *,
    source_manifest_exists: bool,
    candidate_audio_path: str,
    reference_audio_path: str,
    reference_json_path: str,
) -> str:
    if not source_manifest_exists:
        return "missing_source_manifest"
    if tool == "stem_quality_control" and not candidate_audio_path:
        return "missing_audio_path_for_internal_analysis"
    if tool == "stem_quality_control":
        return ""
    if tool == "manifest_repair_only":
        return ""
    if tool in REFERENCE_REQUIRED_TOOLS:
        if tool == "localize_reference_quality_failures" and not reference_json_path:
            return "missing_ground_truth_manifest"
        if tool == "run_ground_truth_quality_gate" and not reference_json_path:
            return "missing_ground_truth_manifest"
        if not reference_audio_path and tool == "waveform_forensic_compare":
            return "missing_reference_audio_path"
        if not candidate_audio_path:
            return "missing_audio_path_for_internal_analysis"
    if tool in FORBIDDEN_AUDIO_WRITING_TOOLS:
        return "tool_writes_audio_or_bundle_artifacts_skipped_by_phase10b_safety"
    return "unsupported_or_unavailable_tool_contract"


def update_missing_after_tool(missing: set[str], tool: str, evidence: dict[str, Any]) -> None:
    if tool == "manifest_repair_only":
        if evidence.get("stem_type_present"):
            missing.discard("missing_stem_type")
        if evidence.get("window_start_sec_present") and evidence.get("window_end_sec_present"):
            missing.discard("missing_window_bounds")
    elif tool == "stem_quality_control":
        missing.discard("missing_audio_path_for_internal_analysis")
    elif tool == "waveform_forensic_compare":
        missing.discard("missing_alignment_check")
        missing.discard("missing_residual_metrics")
        missing.discard("missing_leakage_metrics")
    elif tool == "localize_reference_quality_failures":
        missing.discard("missing_residual_metrics")
        missing.discard("missing_leakage_metrics")
    elif tool == "run_ground_truth_quality_gate":
        missing.discard("missing_reference_metrics")
        missing.discard("missing_residual_metrics")
        missing.discard("missing_leakage_metrics")


def evidence_type_for_tool(tool: str) -> str:
    mapping = {
        "stem_quality_control": "stem_quality_metrics",
        "manifest_repair_only": "manifest_metadata_scan",
        "waveform_forensic_compare": "waveform_forensic_metrics",
        "localize_reference_quality_failures": "localized_reference_quality_failures",
        "run_ground_truth_quality_gate": "ground_truth_quality_gate_metrics",
        "build_flstudio_ground_truth_bundle": "ground_truth_bundle",
    }
    return mapping.get(tool, tool)


def collection_status(tools_executed: list[str], evidence_missing_after_run: list[str]) -> str:
    if not tools_executed:
        return "failed"
    if evidence_missing_after_run:
        return "partially_collected"
    return "collected"


def base_output_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": str(row.get("case_id") or ""),
        "track_hash": str(row.get("track_hash") or ""),
        "stem_type": str(row.get("stem_type") or "unknown"),
        "split": str(row.get("split") or ""),
        "source_manifest": str(row.get("source_manifest") or ""),
        "tools_requested": as_list(row.get("required_tools_to_run")),
        "tools_executed": [],
        "tools_skipped": [],
        "evidence_collected": {},
        "evidence_missing_after_run": as_list(row.get("missing_evidence_types")),
        "collection_status": "failed",
        "can_reclassify_after_collection": False,
        "can_enter_finetune_pool_after_collection": False,
        "can_promote_after_collection": False,
        "audio_exported": False,
        "training_started": False,
        "policy_changed": False,
    }


def collect_row(row: dict[str, Any], golden_hashes: set[str]) -> dict[str, Any]:
    output = base_output_row(row)
    source_manifest = output["source_manifest"]
    source_payload = read_json(Path(source_manifest)) if path_exists(source_manifest) else None
    candidate_audio_path = first_existing_path(requested_audio_paths(row, source_payload), audio_only=True)
    reference_paths = requested_reference_paths(row, source_payload)
    reference_audio_path = first_existing_path(reference_paths, audio_only=True)
    reference_json_path = first_existing_path(reference_paths, json_only=True)
    source_exists = bool(source_payload)
    output["evidence_collected"]["source_manifest"] = source_manifest_evidence(source_manifest, source_payload)
    output["tool_skip_reasons"] = {}

    missing_after = set(as_list(row.get("missing_evidence_types")))
    if source_exists:
        missing_after.discard("missing_source_manifest")
    for tool in sorted(output["tools_requested"]):
        skip_reason = tool_skip_reason(
            tool,
            source_manifest_exists=source_exists,
            candidate_audio_path=candidate_audio_path,
            reference_audio_path=reference_audio_path,
            reference_json_path=reference_json_path,
        )
        if skip_reason:
            output["tools_skipped"].append(tool)
            output["tool_skip_reasons"][tool] = skip_reason
            continue
        try:
            if tool == "stem_quality_control":
                evidence = collect_stem_quality_metrics(candidate_audio_path)
            elif tool == "manifest_repair_only":
                evidence = collect_manifest_repair_scan(row)
            elif tool == "waveform_forensic_compare":
                evidence = collect_waveform_forensic_metrics(reference_audio_path, candidate_audio_path)
            else:
                output["tools_skipped"].append(tool)
                output["tool_skip_reasons"][tool] = "execution_adapter_not_enabled_for_phase10b"
                continue
        except Exception as exc:  # pragma: no cover - exercised by real manifests when audio is unavailable.
            output["tools_skipped"].append(tool)
            output["tool_skip_reasons"][tool] = f"execution_failed:{type(exc).__name__}:{exc}"
            continue
        output["tools_executed"].append(tool)
        output["evidence_collected"][tool] = evidence
        update_missing_after_tool(missing_after, tool, evidence)

    output["tools_executed"] = sorted(output["tools_executed"])
    output["tools_skipped"] = sorted(output["tools_skipped"])
    output["evidence_missing_after_run"] = sorted(missing_after)
    output["collection_status"] = collection_status(output["tools_executed"], output["evidence_missing_after_run"])
    output["can_reclassify_after_collection"] = bool(output["tools_executed"]) and output["collection_status"] != "failed"
    output["can_enter_finetune_pool_after_collection"] = (
        output["can_reclassify_after_collection"]
        and bool(row.get("can_use_for_finetune_after_collection"))
        and output["split"] == "train"
        and output["track_hash"] not in golden_hashes
    )
    if output["collection_status"] not in COLLECTION_STATUSES:
        raise ValueError(f"invalid_collection_status:{output['collection_status']}")
    return output


def skipped_blocked_row(row: dict[str, Any]) -> dict[str, Any]:
    output = base_output_row(row)
    output["tools_skipped"] = sorted(output["tools_requested"])
    output["collection_status"] = "skipped_blocked"
    output["blocked_reason"] = str(row.get("blocked_reason") or "blocked_evidence_collection")
    return output


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key) or "") for row in rows).most_common())


def count_list_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(as_list(row.get(key)))
    return dict(counter.most_common())


def count_evidence_types(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        evidence = row.get("evidence_collected") if isinstance(row.get("evidence_collected"), dict) else {}
        for tool in evidence:
            counter[evidence_type_for_tool(tool)] += 1
    return dict(counter.most_common())


def build_summary(
    *,
    plan_rows: list[dict[str, Any]],
    collected_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    all_rows = collected_rows + failed_rows
    return {
        "schema": "collected_evidence_summary.v1",
        "total_rows_in_plan": len(plan_rows),
        "eligible_rows_processed": len(collected_rows),
        "blocked_rows_skipped": len(failed_rows),
        "collected_count": sum(1 for row in collected_rows if row["collection_status"] == "collected"),
        "partially_collected_count": sum(1 for row in collected_rows if row["collection_status"] == "partially_collected"),
        "failed_count": sum(1 for row in collected_rows if row["collection_status"] == "failed"),
        "tools_executed_counts": count_list_values(collected_rows, "tools_executed"),
        "tools_skipped_counts": count_list_values(all_rows, "tools_skipped"),
        "evidence_types_collected": count_evidence_types(collected_rows),
        "evidence_still_missing": count_list_values(collected_rows, "evidence_missing_after_run"),
        "collection_status_counts": count_by(all_rows, "collection_status"),
        "can_reclassify_after_collection_count": sum(
            1 for row in collected_rows if row["can_reclassify_after_collection"]
        ),
        "can_enter_finetune_pool_after_collection_count": sum(
            1 for row in collected_rows if row["can_enter_finetune_pool_after_collection"]
        ),
        "can_promote_after_collection_count": sum(1 for row in collected_rows if row["can_promote_after_collection"]),
        "safety": dict(SAFETY_FLAGS),
    }


def build_manifest(
    *,
    summary: dict[str, Any],
    missing_evidence_plan_jsonl: Path,
    missing_evidence_manifest_json: Path,
    root_cause_plan_jsonl: Path,
    autonomous_decisions_jsonl: Path,
    hard_case_benchmark_jsonl: Path,
    golden_jsonl: Path,
) -> dict[str, Any]:
    input_files = {
        "missing_evidence_plan_jsonl": str(missing_evidence_plan_jsonl),
        "missing_evidence_manifest_json": str(missing_evidence_manifest_json),
        "root_cause_plan_jsonl": str(root_cause_plan_jsonl),
        "autonomous_decisions_jsonl": str(autonomous_decisions_jsonl),
        "hard_case_benchmark_jsonl": str(hard_case_benchmark_jsonl),
        "golden_jsonl": str(golden_jsonl),
    }
    return {
        "schema": "collected_evidence_manifest.v1",
        "total_rows_in_plan": summary["total_rows_in_plan"],
        "eligible_rows_processed": summary["eligible_rows_processed"],
        "blocked_rows_skipped": summary["blocked_rows_skipped"],
        "collected_count": summary["collected_count"],
        "partially_collected_count": summary["partially_collected_count"],
        "failed_count": summary["failed_count"],
        "tools_executed_counts": summary["tools_executed_counts"],
        "tools_skipped_counts": summary["tools_skipped_counts"],
        "evidence_types_collected": summary["evidence_types_collected"],
        "evidence_still_missing": summary["evidence_still_missing"],
        "can_reclassify_after_collection_count": summary["can_reclassify_after_collection_count"],
        "can_enter_finetune_pool_after_collection_count": summary["can_enter_finetune_pool_after_collection_count"],
        "can_promote_after_collection_count": summary["can_promote_after_collection_count"],
        "input_files": input_files,
        "input_sha256": {
            key: sha256_file(Path(value))
            for key, value in input_files.items()
            if Path(value).exists()
        },
        "safety": dict(SAFETY_FLAGS),
    }


def run_missing_evidence_collection(
    *,
    missing_evidence_plan_jsonl: Path,
    missing_evidence_manifest_json: Path,
    root_cause_plan_jsonl: Path,
    autonomous_decisions_jsonl: Path,
    hard_case_benchmark_jsonl: Path,
    golden_jsonl: Path,
) -> dict[str, Any]:
    _missing_manifest = read_json(missing_evidence_manifest_json)
    # These source-of-truth inputs are intentionally loaded for provenance validation.
    _root_rows = list(iter_jsonl(root_cause_plan_jsonl))
    _autonomous_rows = list(iter_jsonl(autonomous_decisions_jsonl))
    _benchmark_rows = list(iter_jsonl(hard_case_benchmark_jsonl))
    golden_hashes = load_golden_track_hashes(golden_jsonl)
    plan_rows = list(iter_jsonl(missing_evidence_plan_jsonl))
    collected_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    for row in sorted(plan_rows, key=lambda item: str(item.get("case_id") or "")):
        if not row.get("can_collect_automatically"):
            failed_rows.append(skipped_blocked_row(row))
            continue
        collected_rows.append(collect_row(row, golden_hashes))
    summary = build_summary(plan_rows=plan_rows, collected_rows=collected_rows, failed_rows=failed_rows)
    manifest = build_manifest(
        summary=summary,
        missing_evidence_plan_jsonl=missing_evidence_plan_jsonl,
        missing_evidence_manifest_json=missing_evidence_manifest_json,
        root_cause_plan_jsonl=root_cause_plan_jsonl,
        autonomous_decisions_jsonl=autonomous_decisions_jsonl,
        hard_case_benchmark_jsonl=hard_case_benchmark_jsonl,
        golden_jsonl=golden_jsonl,
    )
    return {
        "collected_rows": collected_rows,
        "failed_rows": failed_rows,
        "summary": summary,
        "manifest": manifest,
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Collected Evidence Summary",
        "",
        f"- `total_rows_in_plan`: {summary['total_rows_in_plan']}",
        f"- `eligible_rows_processed`: {summary['eligible_rows_processed']}",
        f"- `blocked_rows_skipped`: {summary['blocked_rows_skipped']}",
        f"- `collected_count`: {summary['collected_count']}",
        f"- `partially_collected_count`: {summary['partially_collected_count']}",
        f"- `failed_count`: {summary['failed_count']}",
        f"- `can_reclassify_after_collection_count`: {summary['can_reclassify_after_collection_count']}",
        f"- `can_enter_finetune_pool_after_collection_count`: {summary['can_enter_finetune_pool_after_collection_count']}",
        f"- `can_promote_after_collection_count`: {summary['can_promote_after_collection_count']}",
        "",
        "## Tools Executed",
        "",
    ]
    for key, value in summary["tools_executed_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Tools Skipped", ""])
    for key, value in summary["tools_skipped_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Evidence Types Collected", ""])
    for key, value in summary["evidence_types_collected"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Evidence Still Missing", ""])
    for key, value in summary["evidence_still_missing"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Safety", ""])
    for key, value in summary["safety"].items():
        lines.append(f"- `{key}`: {str(value).lower()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "collected_evidence.jsonl", result["collected_rows"])
    write_jsonl(output_dir / "failed_evidence_collection.jsonl", result["failed_rows"])
    write_summary_md(output_dir / "collected_evidence_summary.md", result["summary"])
    (output_dir / "collected_evidence_manifest.json").write_text(
        json.dumps(result["manifest"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run automatic missing evidence collection for eligible rows only.")
    parser.add_argument("--missing-evidence-plan-jsonl", type=Path, required=True)
    parser.add_argument("--missing-evidence-manifest-json", type=Path, required=True)
    parser.add_argument("--root-cause-plan-jsonl", type=Path, required=True)
    parser.add_argument("--autonomous-decisions-jsonl", type=Path, required=True)
    parser.add_argument("--hard-case-benchmark-jsonl", type=Path, required=True)
    parser.add_argument("--golden-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_missing_evidence_collection(
        missing_evidence_plan_jsonl=args.missing_evidence_plan_jsonl,
        missing_evidence_manifest_json=args.missing_evidence_manifest_json,
        root_cause_plan_jsonl=args.root_cause_plan_jsonl,
        autonomous_decisions_jsonl=args.autonomous_decisions_jsonl,
        hard_case_benchmark_jsonl=args.hard_case_benchmark_jsonl,
        golden_jsonl=args.golden_jsonl,
    )
    write_outputs(result, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
