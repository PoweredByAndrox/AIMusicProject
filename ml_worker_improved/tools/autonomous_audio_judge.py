from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from tools.artifact_status_contract import (
    ArtifactStatusView,
    collect_failure_labels,
    infer_dominant_failure_reason,
    infer_stem_type,
    normalize_manifest_status,
)


DECISIONS = {
    "accepted_for_listen_pack",
    "release_ready",
    "blocked_from_listen_pack",
    "hard_case",
    "needs_reference_fix",
    "needs_policy_fix",
    "finetune_candidate",
}
JUDGE_MODES = {"reference_aware", "no_reference_proxy", "manifest_only"}
ACCEPTED_DECISIONS = {"accepted_for_listen_pack", "release_ready"}
TRAINING_OR_TUNING_DECISIONS = {"finetune_candidate", "needs_policy_fix"}
HIGH_CONFIDENCE_THRESHOLD = 0.85
MEDIUM_CONFIDENCE_THRESHOLD = 0.66

REFERENCE_METRIC_KEYS = {
    "sdr_db",
    "si_sdr_db",
    "sir_db",
    "sar_db",
    "reference_likeness_percent",
    "residual_rms_dbfs",
    "residual_peak_dbfs",
    "sample_residual_rms_dbfs",
    "sample_residual_peak_dbfs",
    "leakage_score",
    "max_residual_leakage_strength",
    "full_mix_reconstruction_status",
    "alignment_status",
}
SEVERE_ISSUE_SIGNALS = {
    "candidate_space_exhausted",
    "no_accepted_candidate",
    "selected_candidate_still_has_severe_issue",
    "selected_candidate_has_too_many_issue_windows",
}
POLICY_SIGNALS = {
    "selected_candidate_has_unstable_vocal_consistency",
    "selected_candidate_still_has_clustered_issues",
}
FINETUNE_SIGNALS = {
    "thin_or_overcleaned",
    "robotic_or_noise_artifact",
    "vocal_tail_loss",
    "presence_loss",
    "local_vocal_body_dip",
    "selected_candidate_still_has_severe_issue",
}
REFERENCE_FIX_SIGNALS = {"unknown_blocked_reason", "reference_likeness_below_threshold"}
DIAGNOSTIC_OR_BLOCKED_SIGNALS = {
    "diagnostic_only",
    "blocked",
    "blocked_for_listening",
    "blocked_from_delivery",
    "blocked_from_listen_pack",
    "rejected",
    "skipped",
}
BASE_FORBIDDEN_ACTIONS = [
    "no_audio_export",
    "no_diagnostic_audio_as_result",
    "no_cleanup_debleed_denoise_escalation",
    "no_gate_relaxation",
    "no_new_model_or_weight",
    "no_training_start",
    "no_policy_auto_change",
]
NEXT_STEP_BY_DECISION = {
    "accepted_for_listen_pack": "listen_pack",
    "release_ready": "release_delivery",
    "blocked_from_listen_pack": "do_not_export",
    "hard_case": "hard_case_queue",
    "needs_reference_fix": "reference_fix_queue",
    "needs_policy_fix": "policy_fix_candidate",
    "finetune_candidate": "finetune_candidate",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"json_must_be_object:{path}")
    return payload


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.replace(",", ";").split(";") if item.strip()]
    return [str(value).strip()] if str(value).strip() else []


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_golden_track_hashes(golden_jsonl: Path) -> set[str]:
    hashes: set[str] = set()
    if not golden_jsonl.exists():
        return hashes
    with golden_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict) and row.get("track_hash"):
                hashes.add(str(row["track_hash"]).strip())
    return hashes


def normalize_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def collect_reference_metrics(payload: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                normalized = normalize_key(str(key))
                if normalized in REFERENCE_METRIC_KEYS and normalized not in metrics:
                    metrics[normalized] = item
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    return metrics


def load_source_manifest(source_manifest: str) -> tuple[Path | None, dict[str, Any] | None, str | None]:
    if not source_manifest:
        return None, None, "source_manifest_missing"
    path = Path(source_manifest)
    if not path.exists():
        return path, None, "source_manifest_not_found"
    try:
        return path, read_json(path), None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return path, None, f"source_manifest_unreadable:{exc.__class__.__name__}"


def collect_signals(
    benchmark_row: dict[str, Any],
    review_row: dict[str, Any] | None,
    manifest_payload: dict[str, Any] | None,
    status_view: ArtifactStatusView | None,
) -> set[str]:
    signals: set[str] = set()
    for source in (benchmark_row, review_row or {}):
        signals.add(str(source.get("dominant_failure_reason") or "").strip())
        signals.update(as_list(source.get("failure_labels")))
        signals.update(as_list(source.get("failed_checks")))
        signals.update(as_list(source.get("review_priority")))
        signals.update(as_list(source.get("expected_audio_problem")))
    if manifest_payload:
        signals.update(collect_failure_labels(manifest_payload))
        signals.add(infer_dominant_failure_reason(manifest_payload))
        for key in ("delivery_status", "presentation_status", "release_status", "status"):
            value = manifest_payload.get(key)
            if isinstance(value, str) and value.strip():
                signals.add(value.strip())
    if status_view:
        signals.update(
            value
            for value in (
                status_view.delivery_status,
                status_view.presentation_status,
                status_view.release_status,
                status_view.dominant_failure_reason,
            )
            if value
        )
    return {signal for signal in signals if signal}


def confidence_band(confidence: float) -> str:
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


def can_finetune(split: str, track_hash: str, golden_track_hashes: set[str]) -> bool:
    return split != "holdout" and track_hash not in golden_track_hashes


def can_policy_fix(split: str, track_hash: str, golden_track_hashes: set[str]) -> bool:
    return split != "holdout" and track_hash not in golden_track_hashes


def has_severe_blocker(signals: set[str]) -> bool:
    return bool(signals & SEVERE_ISSUE_SIGNALS) or bool(signals & DIAGNOSTIC_OR_BLOCKED_SIGNALS)


def evidence_mode(
    *,
    manifest_payload: dict[str, Any] | None,
    reference_metrics: dict[str, Any],
    signals: set[str],
    review_row: dict[str, Any] | None,
) -> str:
    if reference_metrics:
        return "reference_aware"
    if manifest_payload is not None or signals or review_row is not None:
        if manifest_payload is None and review_row is None and signals <= {"unknown_blocked_reason"}:
            return "manifest_only"
        return "no_reference_proxy"
    return "manifest_only"


def base_available_evidence(
    *,
    benchmark_row: dict[str, Any],
    review_row: dict[str, Any] | None,
    source_manifest_path: Path | None,
    manifest_payload: dict[str, Any] | None,
    status_view: ArtifactStatusView | None,
    reference_metrics: dict[str, Any],
    signals: set[str],
    golden_track_hashes: set[str],
) -> dict[str, Any]:
    track_hash = str(benchmark_row.get("track_hash") or "")
    return {
        "benchmark_row": True,
        "review_pack_row": review_row is not None,
        "source_manifest": str(source_manifest_path) if source_manifest_path else str(benchmark_row.get("source_manifest") or ""),
        "source_manifest_loaded": manifest_payload is not None,
        "artifact_status_contract": asdict(status_view) if status_view else None,
        "reference_metrics": reference_metrics,
        "proxy_signals": sorted(signals),
        "golden_track_protected": track_hash in golden_track_hashes,
    }


def initial_decision(
    *,
    benchmark_row: dict[str, Any],
    review_row: dict[str, Any] | None,
    status_view: ArtifactStatusView | None,
    signals: set[str],
    mode: str,
    source_manifest_error: str | None,
    golden_track_hashes: set[str],
) -> tuple[str, float, list[str]]:
    reasons: list[str] = []
    split = str(benchmark_row.get("split") or "")
    track_hash = str(benchmark_row.get("track_hash") or "")
    dominant = str(benchmark_row.get("dominant_failure_reason") or "")
    review_next = str((review_row or {}).get("allowed_next_step") or "")

    if "candidate_space_exhausted" in signals or dominant == "candidate_space_exhausted":
        return "hard_case", 0.92, ["candidate_space_exhausted_stop"]

    if "no_accepted_candidate" in signals or dominant == "no_accepted_candidate":
        if review_next == "finetune_candidate" and can_finetune(split, track_hash, golden_track_hashes):
            return "finetune_candidate", 0.86, ["no_accepted_candidate_non_holdout_finetune_candidate"]
        return "hard_case", 0.9, ["no_accepted_candidate_no_audio_export"]

    if status_view and {
        status_view.delivery_status,
        status_view.presentation_status,
        status_view.release_status,
    } & {"diagnostic_only"}:
        return "blocked_from_listen_pack", 0.9, ["diagnostic_or_blocked_artifact"]

    if "selected_candidate_still_has_severe_issue" in signals or dominant == "selected_candidate_still_has_severe_issue":
        if can_finetune(split, track_hash, golden_track_hashes) and (
            review_next == "finetune_candidate" or bool(signals & FINETUNE_SIGNALS)
        ):
            return "finetune_candidate", 0.82, ["selected_candidate_still_has_severe_issue"]
        if split == "holdout":
            return "hard_case", 0.82, ["selected_candidate_still_has_severe_issue", "holdout_cannot_enter_finetune"]
        if track_hash in golden_track_hashes:
            return "hard_case", 0.82, [
                "selected_candidate_still_has_severe_issue",
                "golden_track_protected_from_training_or_tuning",
            ]
        return "hard_case", 0.82, ["selected_candidate_still_has_severe_issue"]

    if signals & POLICY_SIGNALS:
        if can_policy_fix(split, track_hash, golden_track_hashes):
            return "needs_policy_fix", 0.76, ["unstable_or_clustered_issue_policy_candidate"]
        if track_hash in golden_track_hashes:
            return "hard_case", 0.76, [
                "unstable_or_clustered_issue_but_protected_split",
                "golden_track_protected_from_training_or_tuning",
            ]
        if split == "holdout":
            return "hard_case", 0.76, [
                "unstable_or_clustered_issue_but_protected_split",
                "holdout_cannot_enter_finetune",
            ]
        return "hard_case", 0.76, ["unstable_or_clustered_issue_but_protected_split"]

    if "thin_or_overcleaned" in signals:
        if can_finetune(split, track_hash, golden_track_hashes):
            return "finetune_candidate", 0.75, ["thin_or_overcleaned_without_stronger_cleanup"]
        if can_policy_fix(split, track_hash, golden_track_hashes):
            return "needs_policy_fix", 0.72, ["thin_or_overcleaned_policy_candidate"]
        if split == "holdout":
            return "hard_case", 0.72, [
                "thin_or_overcleaned_protected_from_tuning",
                "holdout_cannot_enter_finetune",
            ]
        if track_hash in golden_track_hashes:
            return "hard_case", 0.72, [
                "thin_or_overcleaned_protected_from_tuning",
                "golden_track_protected_from_training_or_tuning",
            ]
        return "hard_case", 0.72, ["thin_or_overcleaned_protected_from_tuning"]

    if "robotic_or_noise_artifact" in signals or "high_noise_or_robotic_artifact" in signals:
        if can_finetune(split, track_hash, golden_track_hashes):
            return "finetune_candidate", 0.74, ["robotic_or_noise_artifact_without_more_denoise"]
        return "hard_case", 0.74, ["robotic_or_noise_artifact_hard_case"]

    if status_view and not status_view.delivery_allowed:
        return "blocked_from_listen_pack", 0.9, ["delivery_allowed_false"]

    if status_view and status_view.deliverable and not has_severe_blocker(signals):
        return "release_ready", 0.91 if mode == "reference_aware" else 0.78, ["artifact_contract_deliverable"]

    if status_view and status_view.listenable and not has_severe_blocker(signals):
        return "accepted_for_listen_pack", 0.9 if mode == "reference_aware" else 0.76, ["artifact_contract_listenable"]

    if signals & DIAGNOSTIC_OR_BLOCKED_SIGNALS:
        return "blocked_from_listen_pack", 0.88, ["diagnostic_or_blocked_artifact"]

    if dominant in REFERENCE_FIX_SIGNALS or signals & REFERENCE_FIX_SIGNALS:
        if source_manifest_error == "source_manifest_missing" or source_manifest_error == "source_manifest_not_found":
            return "hard_case", 0.64, ["insufficient_reference_evidence"]
        return "needs_reference_fix", 0.7, ["unknown_or_reference_quality_issue"]

    if source_manifest_error:
        return "hard_case", 0.62, [source_manifest_error]

    reasons.append("insufficient_autonomous_evidence")
    return "hard_case", 0.62, reasons


def enforce_safety(
    *,
    decision: str,
    confidence: float,
    reasons: list[str],
    mode: str,
    split: str,
    track_hash: str,
    golden_track_hashes: set[str],
    status_view: ArtifactStatusView | None,
) -> tuple[str, float, list[str]]:
    if decision in ACCEPTED_DECISIONS:
        if confidence < MEDIUM_CONFIDENCE_THRESHOLD:
            decision = "blocked_from_listen_pack"
            reasons.append("low_confidence_cannot_be_promoted")
        if mode == "manifest_only":
            decision = "blocked_from_listen_pack"
            confidence = min(confidence, 0.65)
            reasons.append("manifest_only_cannot_promote")
        if status_view is None:
            decision = "blocked_from_listen_pack"
            confidence = min(confidence, 0.65)
            reasons.append("artifact_contract_missing_for_promotion")
        elif not status_view.delivery_allowed:
            decision = "blocked_from_listen_pack"
            confidence = max(confidence, 0.86)
            reasons.append("delivery_allowed_false")

    if split == "holdout" and decision == "finetune_candidate":
        decision = "hard_case"
        confidence = min(max(confidence, 0.72), 0.84)
        reasons.append("holdout_cannot_enter_finetune")

    if track_hash in golden_track_hashes and decision in TRAINING_OR_TUNING_DECISIONS:
        decision = "hard_case"
        confidence = min(max(confidence, 0.72), 0.84)
        reasons.append("golden_track_protected_from_training_or_tuning")

    if confidence < MEDIUM_CONFIDENCE_THRESHOLD and decision in ACCEPTED_DECISIONS:
        decision = "blocked_from_listen_pack"
        reasons.append("low_confidence_cannot_be_promoted")

    if decision not in DECISIONS:
        decision = "hard_case"
        confidence = min(confidence, 0.65)
        reasons.append("invalid_decision_forced_to_hard_case")

    return decision, confidence, sorted(dict.fromkeys(reasons))


def judge_case(
    benchmark_row: dict[str, Any],
    review_row: dict[str, Any] | None = None,
    *,
    golden_track_hashes: set[str] | None = None,
) -> dict[str, Any]:
    golden_hashes = golden_track_hashes or set()
    source_manifest = str(benchmark_row.get("source_manifest") or "")
    manifest_path, manifest_payload, source_manifest_error = load_source_manifest(source_manifest)
    status_view: ArtifactStatusView | None = None
    status_error: str | None = None
    if manifest_payload is not None and manifest_path is not None:
        try:
            status_view = normalize_manifest_status(manifest_payload, manifest_path)
        except Exception as exc:  # Defensive: judge must block, not crash, on malformed manifests.
            status_error = f"artifact_status_contract_error:{exc.__class__.__name__}"

    reference_metrics = collect_reference_metrics(manifest_payload or {})
    signals = collect_signals(benchmark_row, review_row, manifest_payload, status_view)
    mode = evidence_mode(
        manifest_payload=manifest_payload,
        reference_metrics=reference_metrics,
        signals=signals,
        review_row=review_row,
    )
    decision, confidence, reasons = initial_decision(
        benchmark_row=benchmark_row,
        review_row=review_row,
        status_view=status_view,
        signals=signals,
        mode=mode,
        source_manifest_error=status_error or source_manifest_error,
        golden_track_hashes=golden_hashes,
    )
    if status_error:
        reasons.append(status_error)

    split = str(benchmark_row.get("split") or "")
    track_hash = str(benchmark_row.get("track_hash") or "")
    decision, confidence, reasons = enforce_safety(
        decision=decision,
        confidence=confidence,
        reasons=reasons,
        mode=mode,
        split=split,
        track_hash=track_hash,
        golden_track_hashes=golden_hashes,
        status_view=status_view,
    )
    band = confidence_band(confidence)
    optional_spotcheck = band != "high" or decision in ACCEPTED_DECISIONS or "unknown_blocked_reason" in signals

    return {
        "case_id": str(benchmark_row.get("benchmark_case_id") or benchmark_row.get("case_id") or ""),
        "source_manifest": source_manifest,
        "track_hash": track_hash,
        "stem_type": str(benchmark_row.get("stem_type") or (status_view.stem_type if status_view else infer_stem_type(benchmark_row))),
        "split": split,
        "dominant_failure_reason": str(benchmark_row.get("dominant_failure_reason") or infer_dominant_failure_reason(benchmark_row)),
        "failure_labels": as_list(benchmark_row.get("failure_labels")),
        "failed_checks": as_list(benchmark_row.get("failed_checks")),
        "available_evidence": base_available_evidence(
            benchmark_row=benchmark_row,
            review_row=review_row,
            source_manifest_path=manifest_path,
            manifest_payload=manifest_payload,
            status_view=status_view,
            reference_metrics=reference_metrics,
            signals=signals,
            golden_track_hashes=golden_hashes,
        ),
        "judge_mode": mode,
        "decision": decision,
        "confidence": round(float(confidence), 4),
        "confidence_band": band,
        "reasons": reasons,
        "forbidden_actions": list(BASE_FORBIDDEN_ACTIONS),
        "allowed_next_step": NEXT_STEP_BY_DECISION[decision],
        "human_review_required": False,
        "optional_human_spotcheck": optional_spotcheck,
    }
