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


RECOVERY_STATUSES = {
    "recoverable",
    "partially_recoverable",
    "not_recoverable",
    "blocked_missing_source_audio",
    "blocked_missing_reference",
    "blocked_missing_manifest",
}
DISCOVERED_INPUT_TYPES = {
    "original_mix",
    "generated_vocal",
    "generated_minus",
    "reference_vocal",
    "reference_minus",
    "flstudio_stem_export",
    "full_mix_reference",
    "source_manifest",
    "validation_summary",
    "delivery_gate_manifest",
}
SAFETY_FLAGS = {
    "audio_copied": False,
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
    "no_gt_bundle_build_in_phase10c",
    "no_promotion",
    "no_training_start",
    "no_policy_change",
    "no_threshold_change",
    "no_cleanup_debleed_denoise",
    "no_model_or_weight_addition",
]
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


def load_golden_track_hashes(golden_jsonl: Path) -> set[str]:
    return {str(row.get("track_hash") or "").strip() for row in iter_jsonl(golden_jsonl) if row.get("track_hash")}


def path_like(value: str) -> bool:
    lower = value.lower()
    return "/" in value or "\\" in value or lower.endswith((".json",) + AUDIO_SUFFIXES)


def resolve_manifest_path(manifest_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (manifest_path.parent / candidate).resolve()


def classify_path(input_key: str, path_text: str, stem_type: str) -> str | None:
    key = input_key.lower()
    lower_path = path_text.lower()
    combined = f"{key} {lower_path}"
    is_audio = lower_path.endswith(AUDIO_SUFFIXES)
    is_json = lower_path.endswith(".json")

    if "validation_summary" in combined:
        return "validation_summary"
    if "delivery_gate" in combined or "selected_delivery_gate" in combined:
        return "delivery_gate_manifest"
    if "flstudio" in combined or "fl_studio" in combined or "split_mixer" in combined or "stem_export" in combined:
        return "flstudio_stem_export"
    if "full_mix" in combined or "reference_sum_mix" in combined:
        return "full_mix_reference"
    if "source_audio_path" in key or "input_mix" in combined or ("mix" in key and "reference" not in key):
        return "original_mix" if is_audio else None
    if ("ground_truth" in combined or "reference" in combined) and ("minus" in combined or "instrumental" in combined):
        return "reference_minus"
    if ("ground_truth" in combined or "reference" in combined) and ("vocal" in combined or stem_type == "vocal"):
        return "reference_vocal" if is_audio or is_json else None
    if "minus" in combined or "instrumental" in combined:
        return "generated_minus" if is_audio else None
    if "vocal" in combined or stem_type == "vocal":
        return "generated_vocal" if is_audio else None
    return None


def make_input_record(input_type: str, path_text: str, source_key: str, manifest_path: Path | None = None) -> dict[str, Any]:
    resolved_path = resolve_manifest_path(manifest_path, path_text) if manifest_path is not None else Path(path_text)
    return {
        "type": input_type,
        "path": str(resolved_path),
        "exists": resolved_path.exists(),
        "source_key": source_key,
    }


def discover_inputs(
    *,
    row: dict[str, Any],
    source_manifest_path: Path | None,
    source_payload: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stem_type = str(row.get("stem_type") or "unknown")
    audio_inputs: list[dict[str, Any]] = []
    reference_inputs: list[dict[str, Any]] = []
    manifest_inputs: list[dict[str, Any]] = []

    if source_manifest_path is not None:
        manifest_inputs.append(
            make_input_record("source_manifest", str(source_manifest_path), "source_manifest", None)
        )
    if source_payload is not None:
        for path_parts, value in walk_values(source_payload):
            if not isinstance(value, str) or not value.strip() or not path_like(value.strip()):
                continue
            source_key = ".".join(path_parts)
            input_type = classify_path(source_key, value.strip(), stem_type)
            if input_type is None or input_type not in DISCOVERED_INPUT_TYPES:
                continue
            record = make_input_record(input_type, value.strip(), source_key, source_manifest_path)
            if input_type in {"source_manifest", "validation_summary", "delivery_gate_manifest"}:
                manifest_inputs.append(record)
            elif input_type in {"reference_vocal", "reference_minus", "full_mix_reference"}:
                reference_inputs.append(record)
            else:
                audio_inputs.append(record)

    return dedupe_records(audio_inputs), dedupe_records(reference_inputs), dedupe_records(manifest_inputs)


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: (str(item["type"]), str(item["path"]), str(item["source_key"]))):
        key = (str(record["type"]), str(record["path"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def existing(records: list[dict[str, Any]], input_type: str) -> list[dict[str, Any]]:
    return [record for record in records if record["type"] == input_type and record["exists"]]


def any_existing(records: list[dict[str, Any]], input_types: set[str]) -> bool:
    return any(record["type"] in input_types and record["exists"] for record in records)


def required_bundle_inputs(
    *,
    audio_inputs: list[dict[str, Any]],
    reference_inputs: list[dict[str, Any]],
    manifest_inputs: list[dict[str, Any]],
    stem_type: str,
) -> dict[str, Any]:
    target_generated_type = "generated_minus" if stem_type == "minus" else "generated_vocal"
    target_reference_type = "reference_minus" if stem_type == "minus" else "reference_vocal"
    original_mix = existing(audio_inputs, "original_mix")
    flstudio_exports = existing(audio_inputs, "flstudio_stem_export")
    generated_target = existing(audio_inputs, target_generated_type)
    reference_target = existing(reference_inputs, target_reference_type)
    source_manifests = existing(manifest_inputs, "source_manifest")
    missing_for_gt_bundle = []
    if not original_mix:
        missing_for_gt_bundle.append("original_mix")
    if not flstudio_exports:
        missing_for_gt_bundle.append("flstudio_stem_export")
    missing_for_metrics = []
    if not generated_target:
        missing_for_metrics.append(target_generated_type)
    if not reference_target:
        missing_for_metrics.append(target_reference_type)
    return {
        "build_flstudio_ground_truth_bundle": {
            "source_mix_path": original_mix[0]["path"] if original_mix else None,
            "stem_dir": str(Path(flstudio_exports[0]["path"]).parent) if flstudio_exports else None,
            "flstudio_stem_export_count": len(flstudio_exports),
            "missing": missing_for_gt_bundle,
        },
        "waveform_forensic_compare": {
            "candidate_audio_path": generated_target[0]["path"] if generated_target else None,
            "reference_audio_path": reference_target[0]["path"] if reference_target else None,
            "missing": missing_for_metrics,
        },
        "run_ground_truth_quality_gate": {
            "candidate_manifest": source_manifests[0]["path"] if source_manifests else None,
            "reference_target_path": reference_target[0]["path"] if reference_target else None,
            "stem_type": stem_type,
            "missing": missing_for_metrics,
        },
    }


def decide_status(
    *,
    source_manifest_error: str,
    source_manifest_exists: bool,
    source_audio_exists: bool,
    generated_target_exists: bool,
    reference_target_exists: bool,
    can_build_gt_bundle: bool,
) -> tuple[str, str]:
    if not source_manifest_exists:
        return "blocked_missing_manifest", source_manifest_error or "source_manifest_missing"
    if not source_audio_exists:
        return "blocked_missing_source_audio", "original_mix_missing_or_unavailable"
    if generated_target_exists and (reference_target_exists or can_build_gt_bundle):
        return "recoverable", ""
    if generated_target_exists and not reference_target_exists:
        return "blocked_missing_reference", "reference_target_missing_or_unavailable"
    if source_audio_exists:
        return "partially_recoverable", "generated_candidate_or_reference_missing"
    return "not_recoverable", "insufficient_recovery_inputs"


def required_next_tool_for(
    *,
    can_build_gt_bundle: bool,
    can_run_reference_metrics: bool,
    can_run_alignment_check: bool,
    status: str,
) -> str:
    if status == "blocked_missing_manifest":
        return "manual_source_manifest_inspection"
    if can_build_gt_bundle:
        return "build_flstudio_ground_truth_bundle"
    if can_run_reference_metrics:
        return "run_ground_truth_quality_gate"
    if can_run_alignment_check:
        return "waveform_forensic_compare"
    if status == "blocked_missing_reference":
        return "manual_source_manifest_inspection"
    if status == "blocked_missing_source_audio":
        return "manual_source_manifest_inspection"
    return "manifest_repair_only"


def recovery_row(row: dict[str, Any], golden_hashes: set[str]) -> dict[str, Any]:
    source_manifest = str(row.get("source_manifest") or "")
    source_path = Path(source_manifest) if source_manifest else None
    source_payload, source_error = try_read_json(source_path)
    audio_inputs, reference_inputs, manifest_inputs = discover_inputs(
        row=row,
        source_manifest_path=source_path,
        source_payload=source_payload,
    )
    stem_type = str(row.get("stem_type") or "unknown")
    target_generated_type = "generated_minus" if stem_type == "minus" else "generated_vocal"
    target_reference_type = "reference_minus" if stem_type == "minus" else "reference_vocal"
    current_missing = as_list(row.get("evidence_missing_after_run"))
    source_manifest_exists = source_path is not None and source_path.exists()
    source_audio_exists = any_existing(audio_inputs + reference_inputs, {"original_mix", "full_mix_reference"})
    generated_target_exists = any_existing(audio_inputs, {target_generated_type})
    reference_target_exists = any_existing(reference_inputs, {target_reference_type})
    can_build_gt_bundle = source_audio_exists and any_existing(audio_inputs, {"flstudio_stem_export"})
    can_run_alignment_check = generated_target_exists and reference_target_exists
    can_run_reference_metrics = generated_target_exists and reference_target_exists
    can_run_residual_metrics = can_run_alignment_check
    can_run_leakage_metrics = can_run_alignment_check
    status, blocked_reason = decide_status(
        source_manifest_error=source_error,
        source_manifest_exists=source_manifest_exists,
        source_audio_exists=source_audio_exists,
        generated_target_exists=generated_target_exists,
        reference_target_exists=reference_target_exists,
        can_build_gt_bundle=can_build_gt_bundle,
    )
    if status not in RECOVERY_STATUSES:
        raise ValueError(f"invalid_recovery_status:{status}")
    forbidden_actions = sorted(set(as_list(row.get("forbidden_actions")) + DEFAULT_FORBIDDEN_ACTIONS))
    if str(row.get("split") or "") == "holdout":
        forbidden_actions.append("holdout_protected")
    if str(row.get("track_hash") or "") in golden_hashes:
        forbidden_actions.append("golden_protected_no_training_or_tuning")
    return {
        "case_id": str(row.get("case_id") or ""),
        "track_hash": str(row.get("track_hash") or ""),
        "stem_type": stem_type,
        "split": str(row.get("split") or ""),
        "source_manifest": source_manifest,
        "current_missing_evidence": current_missing,
        "discovered_audio_inputs": audio_inputs,
        "discovered_reference_inputs": reference_inputs,
        "discovered_manifest_inputs": manifest_inputs,
        "required_bundle_inputs": required_bundle_inputs(
            audio_inputs=audio_inputs,
            reference_inputs=reference_inputs,
            manifest_inputs=manifest_inputs,
            stem_type=stem_type,
        ),
        "recovery_status": status,
        "can_build_gt_bundle": can_build_gt_bundle,
        "can_run_alignment_check": can_run_alignment_check,
        "can_run_reference_metrics": can_run_reference_metrics,
        "can_run_residual_metrics": can_run_residual_metrics,
        "can_run_leakage_metrics": can_run_leakage_metrics,
        "required_next_tool": required_next_tool_for(
            can_build_gt_bundle=can_build_gt_bundle,
            can_run_reference_metrics=can_run_reference_metrics,
            can_run_alignment_check=can_run_alignment_check,
            status=status,
        ),
        "blocked_reason": blocked_reason,
        "forbidden_actions": sorted(set(forbidden_actions)),
    }


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key) or "") for row in rows).most_common())


def build_summary(rows: list[dict[str, Any]], failed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "reference_bundle_recovery_summary.v1",
        "total_recovery_rows": len(rows),
        "skipped_blocked_input_rows": len(failed_rows),
        "recovery_status_counts": count_by(rows, "recovery_status"),
        "required_next_tool_counts": count_by(rows, "required_next_tool"),
        "recoverable_count": sum(1 for row in rows if row["recovery_status"] == "recoverable"),
        "partially_recoverable_count": sum(1 for row in rows if row["recovery_status"] == "partially_recoverable"),
        "not_recoverable_count": sum(1 for row in rows if row["recovery_status"] == "not_recoverable"),
        "blocked_missing_source_audio_count": sum(
            1 for row in rows if row["recovery_status"] == "blocked_missing_source_audio"
        ),
        "blocked_missing_reference_count": sum(
            1 for row in rows if row["recovery_status"] == "blocked_missing_reference"
        ),
        "blocked_missing_manifest_count": sum(
            1 for row in rows if row["recovery_status"] == "blocked_missing_manifest"
        ),
        "can_build_gt_bundle_count": sum(1 for row in rows if row["can_build_gt_bundle"]),
        "can_run_alignment_check_count": sum(1 for row in rows if row["can_run_alignment_check"]),
        "can_run_reference_metrics_count": sum(1 for row in rows if row["can_run_reference_metrics"]),
        "can_run_residual_metrics_count": sum(1 for row in rows if row["can_run_residual_metrics"]),
        "can_run_leakage_metrics_count": sum(1 for row in rows if row["can_run_leakage_metrics"]),
        "safety": dict(SAFETY_FLAGS),
    }


def build_manifest(
    *,
    summary: dict[str, Any],
    collected_evidence_jsonl: Path,
    collected_evidence_manifest_json: Path,
    failed_evidence_jsonl: Path,
    missing_evidence_plan_jsonl: Path,
    hard_case_benchmark_jsonl: Path,
    golden_jsonl: Path,
) -> dict[str, Any]:
    input_files = {
        "collected_evidence_jsonl": str(collected_evidence_jsonl),
        "collected_evidence_manifest_json": str(collected_evidence_manifest_json),
        "failed_evidence_jsonl": str(failed_evidence_jsonl),
        "missing_evidence_plan_jsonl": str(missing_evidence_plan_jsonl),
        "hard_case_benchmark_jsonl": str(hard_case_benchmark_jsonl),
        "golden_jsonl": str(golden_jsonl),
    }
    return {
        "schema": "reference_bundle_recovery_manifest.v1",
        **summary,
        "input_files": input_files,
        "input_sha256": {
            key: sha256_file(Path(value))
            for key, value in input_files.items()
            if Path(value).exists()
        },
        "safety": dict(SAFETY_FLAGS),
    }


def build_reference_bundle_recovery_plan(
    *,
    collected_evidence_jsonl: Path,
    collected_evidence_manifest_json: Path,
    failed_evidence_jsonl: Path,
    missing_evidence_plan_jsonl: Path,
    hard_case_benchmark_jsonl: Path,
    golden_jsonl: Path,
) -> dict[str, Any]:
    _collected_manifest = read_json(collected_evidence_manifest_json)
    failed_rows = list(iter_jsonl(failed_evidence_jsonl))
    _missing_rows = list(iter_jsonl(missing_evidence_plan_jsonl))
    _benchmark_rows = list(iter_jsonl(hard_case_benchmark_jsonl))
    golden_hashes = load_golden_track_hashes(golden_jsonl)
    rows = [recovery_row(row, golden_hashes) for row in iter_jsonl(collected_evidence_jsonl)]
    rows.sort(key=lambda row: row["case_id"])
    summary = build_summary(rows, failed_rows)
    manifest = build_manifest(
        summary=summary,
        collected_evidence_jsonl=collected_evidence_jsonl,
        collected_evidence_manifest_json=collected_evidence_manifest_json,
        failed_evidence_jsonl=failed_evidence_jsonl,
        missing_evidence_plan_jsonl=missing_evidence_plan_jsonl,
        hard_case_benchmark_jsonl=hard_case_benchmark_jsonl,
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
        "# Reference Bundle Recovery Summary",
        "",
        f"- `total_recovery_rows`: {summary['total_recovery_rows']}",
        f"- `recoverable_count`: {summary['recoverable_count']}",
        f"- `partially_recoverable_count`: {summary['partially_recoverable_count']}",
        f"- `not_recoverable_count`: {summary['not_recoverable_count']}",
        f"- `blocked_missing_source_audio_count`: {summary['blocked_missing_source_audio_count']}",
        f"- `blocked_missing_reference_count`: {summary['blocked_missing_reference_count']}",
        f"- `can_build_gt_bundle_count`: {summary['can_build_gt_bundle_count']}",
        f"- `can_run_alignment_check_count`: {summary['can_run_alignment_check_count']}",
        f"- `can_run_reference_metrics_count`: {summary['can_run_reference_metrics_count']}",
        f"- `can_run_residual_metrics_count`: {summary['can_run_residual_metrics_count']}",
        f"- `can_run_leakage_metrics_count`: {summary['can_run_leakage_metrics_count']}",
        "",
        "## Recovery Status",
        "",
    ]
    for key, value in summary["recovery_status_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Required Next Tool", ""])
    for key, value in summary["required_next_tool_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Safety", ""])
    for key, value in summary["safety"].items():
        lines.append(f"- `{key}`: {str(value).lower()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "reference_bundle_recovery_plan.jsonl", result["rows"])
    (output_dir / "reference_bundle_recovery_manifest.json").write_text(
        json.dumps(result["manifest"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary_md(output_dir / "reference_bundle_recovery_summary.md", result["summary"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan reference and ground-truth bundle recovery from collected evidence.")
    parser.add_argument("--collected-evidence-jsonl", type=Path, required=True)
    parser.add_argument("--collected-evidence-manifest-json", type=Path, required=True)
    parser.add_argument("--failed-evidence-jsonl", type=Path, required=True)
    parser.add_argument("--missing-evidence-plan-jsonl", type=Path, required=True)
    parser.add_argument("--hard-case-benchmark-jsonl", type=Path, required=True)
    parser.add_argument("--golden-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_reference_bundle_recovery_plan(
        collected_evidence_jsonl=args.collected_evidence_jsonl,
        collected_evidence_manifest_json=args.collected_evidence_manifest_json,
        failed_evidence_jsonl=args.failed_evidence_jsonl,
        missing_evidence_plan_jsonl=args.missing_evidence_plan_jsonl,
        hard_case_benchmark_jsonl=args.hard_case_benchmark_jsonl,
        golden_jsonl=args.golden_jsonl,
    )
    write_outputs(result, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
