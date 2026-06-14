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


ROOT_CAUSE_BUCKETS = {
    "reference_or_alignment_problem",
    "policy_classification_problem",
    "current_model_insufficient",
    "candidate_space_exhausted",
    "no_accepted_candidate",
    "unknown_needs_more_evidence",
    "protected_holdout_only",
    "minus_specific_failure",
    "vocal_specific_failure",
}
SAFE_NEXT_ACTIONS = {
    "collect_missing_reference_evidence",
    "improve_taxonomy_mapping",
    "create_candidate_for_future_finetune_pool",
    "keep_as_hard_holdout",
    "rerun_existing_ladder_once",
    "blocked_no_action",
    "investigate_minus_specific_path",
    "investigate_vocal_specific_path",
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
    "no_audio_export",
    "no_repair_execution",
    "no_cleanup_debleed_denoise_escalation",
    "no_gate_relaxation",
    "no_new_model_or_weight_request_by_default",
    "no_training_start",
    "no_policy_auto_change",
    "no_mandatory_human_review",
]
REFERENCE_SIGNALS = {
    "reference_likeness_below_threshold",
    "alignment_failed",
    "alignment_status_failed",
    "exact_reference_threshold_failed",
    "sample_residual_above_threshold",
    "source_manifest_missing",
    "source_manifest_not_found",
    "insufficient_reference_evidence",
}
POLICY_SIGNALS = {
    "selected_candidate_has_unstable_vocal_consistency",
    "selected_candidate_still_has_clustered_issues",
    "selected_candidate_has_too_many_issue_windows",
    "candidate_proxy_regressed",
    "proxy_blocked",
    "experimental_proxy_not_verified_orchestral",
}
VOCAL_SIGNALS = {
    "selected_candidate_still_has_severe_issue",
    "thin_or_overcleaned",
    "robotic_or_noise_artifact",
    "vocal_tail_loss",
    "presence_loss",
    "local_vocal_body_dip",
    "vocal_dropout",
    "backing_vocal_loss_risk",
}
MINUS_SIGNALS = {
    "minus",
    "instrumental",
    "kick_punch_bleed",
    "low_band_bleed",
    "leaky_full_mix_risk",
}
STRONG_MODEL_INSUFFICIENCY_SIGNALS = {
    "selected_candidate_still_has_severe_issue",
    "thin_or_overcleaned",
    "robotic_or_noise_artifact",
    "candidate_space_exhausted",
    "no_accepted_candidate",
}
REQUIRED_EVIDENCE_BY_BUCKET = {
    "reference_or_alignment_problem": [
        "source_manifest",
        "reference_metrics_or_alignment_report",
        "failed_checks",
    ],
    "policy_classification_problem": [
        "policy_issue_labels",
        "autonomous_decision_row",
        "taxonomy_mapping_diff_before_execution",
    ],
    "current_model_insufficient": [
        "repeated_failure_evidence",
        "non_holdout_track_hash",
        "golden_exclusion_check",
        "objective_or_proxy_failure_labels",
    ],
    "candidate_space_exhausted": [
        "candidate_space_exhausted_label",
        "registered_candidate_inventory",
        "no_cleanup_escalation_confirmation",
    ],
    "no_accepted_candidate": [
        "no_accepted_candidate_label",
        "candidate_manifest_set",
        "delivery_gate_rejection_reasons",
    ],
    "unknown_needs_more_evidence": [
        "source_manifest",
        "failure_labels",
        "missing_reference_or_taxonomy_evidence",
    ],
    "protected_holdout_only": [
        "track_hash",
        "split_or_golden_protection",
        "holdout_exclusion_confirmation",
    ],
    "minus_specific_failure": [
        "stem_type_minus",
        "minus_failure_labels",
        "minus_route_manifest_evidence",
    ],
    "vocal_specific_failure": [
        "stem_type_vocal",
        "vocal_failure_labels",
        "vocal_route_manifest_evidence",
    ],
}


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


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_golden_track_hashes(golden_jsonl: Path) -> set[str]:
    hashes: set[str] = set()
    for row in iter_jsonl(golden_jsonl):
        track_hash = str(row.get("track_hash") or "").strip()
        if track_hash:
            hashes.add(track_hash)
    return hashes


def load_benchmark_rows(hard_case_benchmark_jsonl: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(hard_case_benchmark_jsonl):
        case_id = str(row.get("benchmark_case_id") or row.get("case_id") or "").strip()
        if case_id:
            rows[case_id] = row
    return rows


def normalized_signals(row: dict[str, Any], benchmark_row: dict[str, Any] | None = None) -> set[str]:
    signals: set[str] = set()
    sources = [row]
    if benchmark_row:
        sources.append(benchmark_row)
    for source in sources:
        for key in ("dominant_failure_reason", "decision", "judge_mode", "allowed_next_step"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                signals.add(value.strip())
        for key in ("failure_labels", "failed_checks", "reasons", "forbidden_actions"):
            signals.update(as_list(source.get(key)))
    evidence = row.get("available_evidence")
    if isinstance(evidence, dict):
        signals.update(as_list(evidence.get("proxy_signals")))
        contract = evidence.get("artifact_status_contract")
        if isinstance(contract, dict):
            for key in ("delivery_status", "presentation_status", "release_status", "dominant_failure_reason"):
                value = contract.get(key)
                if isinstance(value, str) and value.strip():
                    signals.add(value.strip())
        metrics = evidence.get("reference_metrics")
        if isinstance(metrics, dict):
            for key in metrics:
                signals.add(str(key))
    return {signal for signal in signals if signal}


def strong_evidence(row: dict[str, Any], signals: set[str]) -> bool:
    confidence = as_float(row.get("confidence"))
    return confidence >= 0.85 and bool(signals & STRONG_MODEL_INSUFFICIENCY_SIGNALS)


def classify_bucket(
    row: dict[str, Any],
    benchmark_row: dict[str, Any] | None,
    golden_hashes: set[str],
) -> tuple[str, list[str]]:
    signals = normalized_signals(row, benchmark_row)
    split = str(row.get("split") or (benchmark_row or {}).get("split") or "")
    stem_type = str(row.get("stem_type") or (benchmark_row or {}).get("stem_type") or "unknown")
    track_hash = str(row.get("track_hash") or (benchmark_row or {}).get("track_hash") or "")
    reasons: list[str] = []

    if split == "holdout" or track_hash in golden_hashes:
        reasons.append("protected_holdout_or_golden_track")
        return "protected_holdout_only", reasons
    if "candidate_space_exhausted" in signals:
        reasons.append("candidate_space_exhausted")
        if strong_evidence(row, signals) and stem_type == "unknown":
            return "current_model_insufficient", reasons
        return "candidate_space_exhausted", reasons
    if "no_accepted_candidate" in signals:
        reasons.append("no_accepted_candidate")
        if strong_evidence(row, signals) and stem_type == "unknown":
            return "current_model_insufficient", reasons
        return "no_accepted_candidate", reasons
    if "unknown_blocked_reason" in signals:
        if row.get("judge_mode") == "reference_aware" and signals & REFERENCE_SIGNALS:
            reasons.append("reference_aware_unknown_with_reference_signal")
            return "reference_or_alignment_problem", reasons
        reasons.append("unknown_blocked_reason_requires_more_evidence")
        return "unknown_needs_more_evidence", reasons
    if signals & REFERENCE_SIGNALS:
        reasons.append("reference_or_alignment_signal")
        return "reference_or_alignment_problem", reasons
    if stem_type == "minus":
        reasons.append("minus_stem_failure")
        return "minus_specific_failure", reasons
    if signals & POLICY_SIGNALS:
        reasons.append("policy_classification_signal")
        return "policy_classification_problem", reasons
    if stem_type == "vocal" and (signals & VOCAL_SIGNALS or str(row.get("dominant_failure_reason"))):
        reasons.append("vocal_stem_failure")
        return "vocal_specific_failure", reasons
    if strong_evidence(row, signals):
        reasons.append("strong_non_holdout_model_insufficiency_evidence")
        return "current_model_insufficient", reasons
    reasons.append("insufficient_root_cause_evidence")
    return "unknown_needs_more_evidence", reasons


def safe_next_action(bucket: str, row: dict[str, Any], signals: set[str]) -> str:
    if bucket == "reference_or_alignment_problem":
        return "collect_missing_reference_evidence"
    if bucket == "policy_classification_problem":
        return "improve_taxonomy_mapping"
    if bucket == "current_model_insufficient":
        return "create_candidate_for_future_finetune_pool"
    if bucket == "candidate_space_exhausted":
        return "rerun_existing_ladder_once"
    if bucket == "no_accepted_candidate":
        return "rerun_existing_ladder_once"
    if bucket == "protected_holdout_only":
        return "keep_as_hard_holdout"
    if bucket == "minus_specific_failure":
        return "investigate_minus_specific_path"
    if bucket == "vocal_specific_failure":
        return "investigate_vocal_specific_path"
    return "collect_missing_reference_evidence"


def can_enter_finetune_pool(bucket: str, row: dict[str, Any], golden_hashes: set[str]) -> bool:
    split = str(row.get("split") or "")
    track_hash = str(row.get("track_hash") or "")
    if split == "holdout" or track_hash in golden_hashes:
        return False
    return bucket in {"current_model_insufficient", "vocal_specific_failure", "minus_specific_failure"}


def can_change_policy(bucket: str, row: dict[str, Any], golden_hashes: set[str]) -> bool:
    split = str(row.get("split") or "")
    track_hash = str(row.get("track_hash") or "")
    return bucket == "policy_classification_problem" and split != "holdout" and track_hash not in golden_hashes


def can_request_new_model(bucket: str, row: dict[str, Any], signals: set[str], golden_hashes: set[str]) -> bool:
    split = str(row.get("split") or "")
    track_hash = str(row.get("track_hash") or "")
    return (
        bucket == "current_model_insufficient"
        and split != "holdout"
        and track_hash not in golden_hashes
        and strong_evidence(row, signals)
        and ("candidate_space_exhausted" in signals or "no_accepted_candidate" in signals)
    )


def plan_row(
    row: dict[str, Any],
    benchmark_row: dict[str, Any] | None,
    golden_hashes: set[str],
) -> dict[str, Any]:
    bucket, bucket_reasons = classify_bucket(row, benchmark_row, golden_hashes)
    signals = normalized_signals(row, benchmark_row)
    next_action = safe_next_action(bucket, row, signals)
    plan = {
        "case_id": str(row.get("case_id") or ""),
        "track_hash": str(row.get("track_hash") or ""),
        "stem_type": str(row.get("stem_type") or "unknown"),
        "split": str(row.get("split") or ""),
        "original_decision": str(row.get("decision") or ""),
        "dominant_failure_reason": str(row.get("dominant_failure_reason") or ""),
        "failure_labels": as_list(row.get("failure_labels")),
        "judge_mode": str(row.get("judge_mode") or ""),
        "confidence": as_float(row.get("confidence")),
        "root_cause_bucket": bucket,
        "safe_next_action": next_action,
        "forbidden_actions": sorted(set(as_list(row.get("forbidden_actions")) + DEFAULT_FORBIDDEN_ACTIONS)),
        "required_evidence_before_execution": REQUIRED_EVIDENCE_BY_BUCKET[bucket],
        "can_execute_automatically": False,
        "can_enter_finetune_pool": can_enter_finetune_pool(bucket, row, golden_hashes),
        "can_change_policy": can_change_policy(bucket, row, golden_hashes),
        "can_request_new_model": can_request_new_model(bucket, row, signals, golden_hashes),
        "human_review_required": False,
        "classification_reasons": bucket_reasons,
    }
    if plan["safe_next_action"] not in SAFE_NEXT_ACTIONS:
        raise ValueError(f"invalid_safe_next_action:{plan['safe_next_action']}")
    if plan["root_cause_bucket"] not in ROOT_CAUSE_BUCKETS:
        raise ValueError(f"invalid_root_cause_bucket:{plan['root_cause_bucket']}")
    return plan


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key) or "") for row in rows).most_common())


def build_summary(rows: list[dict[str, Any]], dashboard: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "root_cause_plan_summary.v1",
        "row_count": len(rows),
        "root_cause_bucket_counts": count_by(rows, "root_cause_bucket"),
        "safe_next_action_counts": count_by(rows, "safe_next_action"),
        "can_enter_finetune_pool_count": sum(1 for row in rows if row["can_enter_finetune_pool"]),
        "can_change_policy_count": sum(1 for row in rows if row["can_change_policy"]),
        "can_request_new_model_count": sum(1 for row in rows if row["can_request_new_model"]),
        "can_execute_automatically_count": sum(1 for row in rows if row["can_execute_automatically"]),
        "human_review_required_count": sum(1 for row in rows if row["human_review_required"]),
        "holdout_protected_count": sum(1 for row in rows if row["root_cause_bucket"] == "protected_holdout_only"),
        "minus_specific_count": sum(1 for row in rows if row["root_cause_bucket"] == "minus_specific_failure"),
        "vocal_specific_count": sum(1 for row in rows if row["root_cause_bucket"] == "vocal_specific_failure"),
        "unknown_needs_more_evidence_count": sum(
            1 for row in rows if row["root_cause_bucket"] == "unknown_needs_more_evidence"
        ),
        "dashboard_summary": dashboard.get("summary", {}) if isinstance(dashboard, dict) else {},
        "safety": dict(SAFETY_FLAGS),
    }


def build_manifest(
    *,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    autonomous_decisions_jsonl: Path,
    autonomous_manifest_json: Path,
    hard_case_benchmark_jsonl: Path,
    dashboard_json: Path,
    golden_jsonl: Path,
) -> dict[str, Any]:
    input_files = {
        "autonomous_decisions_jsonl": str(autonomous_decisions_jsonl),
        "autonomous_manifest_json": str(autonomous_manifest_json),
        "hard_case_benchmark_jsonl": str(hard_case_benchmark_jsonl),
        "dashboard_json": str(dashboard_json),
        "golden_jsonl": str(golden_jsonl),
    }
    return {
        "schema": "root_cause_plan_manifest.v1",
        "row_count": len(rows),
        "root_cause_bucket_counts": summary["root_cause_bucket_counts"],
        "safe_next_action_counts": summary["safe_next_action_counts"],
        "can_enter_finetune_pool_count": summary["can_enter_finetune_pool_count"],
        "can_change_policy_count": summary["can_change_policy_count"],
        "can_request_new_model_count": summary["can_request_new_model_count"],
        "can_execute_automatically_count": summary["can_execute_automatically_count"],
        "human_review_required_count": summary["human_review_required_count"],
        "input_files": input_files,
        "input_sha256": {
            key: sha256_file(Path(value))
            for key, value in input_files.items()
            if Path(value).exists()
        },
        "safety": dict(SAFETY_FLAGS),
    }


def plan_root_causes(
    *,
    autonomous_decisions_jsonl: Path,
    autonomous_manifest_json: Path,
    hard_case_benchmark_jsonl: Path,
    dashboard_json: Path,
    golden_jsonl: Path,
) -> dict[str, Any]:
    autonomous_manifest = read_json(autonomous_manifest_json)
    benchmark_rows = load_benchmark_rows(hard_case_benchmark_jsonl)
    dashboard = read_json(dashboard_json) if dashboard_json.exists() else {}
    golden_hashes = load_golden_track_hashes(golden_jsonl)
    rows = [
        plan_row(row, benchmark_rows.get(str(row.get("case_id") or "")), golden_hashes)
        for row in iter_jsonl(autonomous_decisions_jsonl)
        if str(row.get("decision") or "") in {"blocked_from_listen_pack", "hard_case"}
    ]
    expected_count = int(autonomous_manifest.get("row_count") or len(rows))
    if expected_count < len(rows):
        raise ValueError("root_cause_rows_exceed_autonomous_manifest_row_count")
    summary = build_summary(rows, dashboard)
    manifest = build_manifest(
        rows=rows,
        summary=summary,
        autonomous_decisions_jsonl=autonomous_decisions_jsonl,
        autonomous_manifest_json=autonomous_manifest_json,
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
        "# Root Cause Plan Summary",
        "",
        f"- `total_rows`: {summary['row_count']}",
        f"- `can_enter_finetune_pool_count`: {summary['can_enter_finetune_pool_count']}",
        f"- `can_change_policy_count`: {summary['can_change_policy_count']}",
        f"- `can_request_new_model_count`: {summary['can_request_new_model_count']}",
        f"- `can_execute_automatically_count`: {summary['can_execute_automatically_count']}",
        f"- `human_review_required_count`: {summary['human_review_required_count']}",
        f"- `holdout_protected_count`: {summary['holdout_protected_count']}",
        f"- `minus_specific_count`: {summary['minus_specific_count']}",
        f"- `vocal_specific_count`: {summary['vocal_specific_count']}",
        f"- `unknown_needs_more_evidence_count`: {summary['unknown_needs_more_evidence_count']}",
        "",
        "## Root Cause Buckets",
        "",
    ]
    for key, value in summary["root_cause_bucket_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Safe Next Actions", ""])
    for key, value in summary["safe_next_action_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Safety", ""])
    for key, value in summary["safety"].items():
        lines.append(f"- `{key}`: {str(value).lower()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "root_cause_plan.jsonl", result["rows"])
    write_summary_md(output_dir / "root_cause_plan_summary.md", result["summary"])
    (output_dir / "root_cause_plan_manifest.json").write_text(
        json.dumps(result["manifest"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan root-cause buckets for autonomous blocked/hard-case decisions.")
    parser.add_argument("--autonomous-decisions-jsonl", type=Path, required=True)
    parser.add_argument("--autonomous-manifest-json", type=Path, required=True)
    parser.add_argument("--hard-case-benchmark-jsonl", type=Path, required=True)
    parser.add_argument("--dashboard-json", type=Path, required=True)
    parser.add_argument("--golden-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    result = plan_root_causes(
        autonomous_decisions_jsonl=args.autonomous_decisions_jsonl,
        autonomous_manifest_json=args.autonomous_manifest_json,
        hard_case_benchmark_jsonl=args.hard_case_benchmark_jsonl,
        dashboard_json=args.dashboard_json,
        golden_jsonl=args.golden_jsonl,
    )
    write_outputs(result, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
