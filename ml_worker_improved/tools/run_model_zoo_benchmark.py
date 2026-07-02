from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = ROOT_DIR / "scripts"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402
from write_exact_reference_listen_audit import write_exact_reference_listen_audit  # noqa: E402
import run_ground_truth_quality_gate as quality_gate  # noqa: E402
from worker.model_registry import ModelRegistry, default_model_registry  # noqa: E402


DEFAULT_SAMPLE_RATE = quality_gate.DEFAULT_SAMPLE_RATE
DEFAULT_EXACT_AUDIT_MAX_EVIDENCE_FRAMES = 262_144
ADAPTIVE_REPAIR_BASE_MODEL_LABELS = frozenset(
    {
        "scnet",
        "scnet_ihf",
        "scnet_xl",
        "scnet_xl_ihf_musdb18",
        "drum_bus_projection_body_gain",
        "drum_micro_leakage_debleed",
        "drum_tonal_leakage_repair",
        "drum_transient_restore",
    }
)
ADAPTIVE_REPAIR_PRIORITY_LABELS = frozenset(
    {
        "bar_spanning_drop_melody_leakage",
        "micro_reference_stem_leakage",
        "micro_residual_spike",
        "micro_transient_mismatch",
        "non_target_leakage",
        "reference_likeness_below_threshold",
        "reference_stem_leakage",
        "sample_residual_above_threshold",
        "transient_or_attack_mismatch",
    }
)
ADAPTIVE_REPAIR_MIN_REFERENCE_LIKENESS_PERCENT = 90.0
ADAPTIVE_REPAIR_MIN_SDR_DB = 12.0
ADAPTIVE_REPAIR_MIN_SI_SDR_DB = 12.0


@dataclass(frozen=True)
class CandidateManifestSpec:
    label: str
    manifest_path: Path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_file_label(label: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return normalized.strip("._") or "candidate"


def parse_candidate_manifest(value: str) -> CandidateManifestSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate manifest must use LABEL=PATH format")
    raw_label, raw_path = value.split("=", 1)
    label = raw_label.strip()
    if not label:
        raise argparse.ArgumentTypeError("candidate label cannot be empty")
    manifest_path = Path(raw_path.strip())
    if not manifest_path.exists():
        raise argparse.ArgumentTypeError(f"candidate manifest does not exist: {manifest_path}")
    return CandidateManifestSpec(label=label, manifest_path=manifest_path)


def normalize_requested_stems(stems: Sequence[str] | None) -> list[str]:
    return quality_gate.normalize_requested_stems(stems)


def extract_failed_check_names(report: dict[str, Any]) -> list[str]:
    failed_check_names: set[str] = set()
    forensic_contract = report.get("forensic_delivery_contract", {})
    failed_check_names.update(str(label) for label in forensic_contract.get("blocking_issue_labels", []))
    for stem_report in report.get("stems", []):
        verdict = stem_report.get("verdict", {})
        stem_contract = stem_report.get("forensic_delivery_contract", {})
        stem_failed = verdict.get("status") == "failed" or stem_contract.get("status") == "rejected"
        failed_check_names.update(str(name) for name in verdict.get("failed_check_names", []))
        failed_check_names.update(str(label) for label in stem_contract.get("blocking_issue_labels", []))
        if not stem_failed:
            continue
        quality_windows = stem_report.get("quality_windows", {})
        for window in quality_windows.get("worst_windows", []):
            failed_check_names.update(str(label) for label in window.get("issue_labels", []))
        micro_quality_windows = stem_report.get("micro_quality_windows", {})
        for window in micro_quality_windows.get("worst_windows", []):
            failed_check_names.update(str(label) for label in window.get("issue_labels", []))
    return sorted(failed_check_names)


def build_ralph_ai_delivery_contract(
    *,
    delivery_allowed: bool,
    delivery_status: str,
    delivery_reason: str | None,
    exact_reference_audit: dict[str, Any],
) -> dict[str, Any]:
    if delivery_allowed:
        return {
            "method": "RALPH-AI",
            "status": "accepted",
            "presentation_status": "accepted_for_listen_pack",
            "result_delivery_allowed": True,
            "diagnostic_artifacts_allowed": True,
            "rejects_best_of_bad_candidate": False,
            "reason": str(delivery_reason or "delivery_gate_passed"),
            "required_next_action": "promote_selected_candidate_for_listen_pack",
            "exact_reference_delivery_status": exact_reference_audit.get("delivery_status"),
        }
    return {
        "method": "RALPH-AI",
        "status": "blocked",
        "presentation_status": "blocked_from_delivery",
        "result_delivery_allowed": False,
        "diagnostic_artifacts_allowed": True,
        "rejects_best_of_bad_candidate": True,
        "reason": str(delivery_reason or delivery_status or "delivery_gate_failed"),
        "required_next_action": "iterate_until_exact_reference_delivery_passes",
        "exact_reference_delivery_status": exact_reference_audit.get("delivery_status"),
    }


def build_metric_stack_contract() -> dict[str, Any]:
    return {
        "source_document": "deep-research-report.md",
        "contract": "multi_layer_metric_stack",
        "computed_layers": {
            "separation_fidelity": ["sdr_db", "si_sdr_db", "sir_db", "sar_db"],
            "exact_reference_waveform_audit": [
                "cosine_percent",
                "sample_residual_rms_dbfs",
                "sample_residual_peak_dbfs",
            ],
        },
        "not_computed_layers": {
            "perceptual_full_reference": {
                "metric": "visqol_audio_48khz",
                "status": "not_computed",
                "reason": "ViSQOL is a batch perceptual metric and is not part of this objective gate runtime.",
            },
            "vocal_specific_secondary": {
                "metric": "pesq",
                "status": "not_computed",
                "reason": "PESQ is unsafe as a primary music-separation metric and may be used only as a secondary speech-like vocal diagnostic.",
            },
        },
        "human_listening_gate": {
            "metric": "pairwise_ab_or_webmushra_8_15s_hard_segments",
            "status": "not_run",
            "substituted_by_objective_metrics": False,
            "reason": "Objective metrics and exact-reference audit cannot claim human perceptual acceptance by themselves.",
        },
        "comparability_guardrails": [
            "Do not compare vocal-only SDR with four-stem average SDR as a single leaderboard value.",
            "Do not mix extra-data, fine-tuned, and no-extra-data scores without recording the training/data regime.",
            "Accepted delivery still requires RALPH-AI delivery_status=accepted and exact-reference audit acceptance.",
        ],
    }


def summarize_stem_scores(report: dict[str, Any]) -> list[dict[str, Any]]:
    stem_scores: list[dict[str, Any]] = []
    for stem_report in report.get("stems", []):
        metrics = stem_report.get("metrics", {})
        reference_likeness = stem_report.get("reference_likeness", {})
        stem_scores.append(
            {
                "stem": stem_report.get("stem"),
                "status": stem_report.get("verdict", {}).get("status"),
                "reference_likeness_percent": reference_likeness.get("score_percent"),
                "sdr_db": metrics.get("sdr_db"),
                "si_sdr_db": metrics.get("si_sdr_db"),
                "sir_db": metrics.get("sir_db"),
                "sar_db": metrics.get("sar_db"),
                "failed_check_names": stem_report.get("verdict", {}).get("failed_check_names", []),
            }
        )
    return stem_scores


def numeric_values(values: Sequence[Any]) -> list[float]:
    numbers: list[float] = []
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numbers.append(float(value))
    return numbers


def mean_or_none(values: Sequence[Any]) -> float | None:
    numbers = numeric_values(values)
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def min_or_none(values: Sequence[Any]) -> float | None:
    numbers = numeric_values(values)
    if not numbers:
        return None
    return min(numbers)


def read_candidate_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"_read_error": str(exc)}
    if not isinstance(payload, dict):
        return {"_read_error": "candidate manifest root is not an object"}
    return payload


def resolve_manifest_audio_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def stem_audio_path_from_manifest(manifest_path: Path, payload: dict[str, Any], stem_name: str) -> Path | None:
    stem_entry = payload.get("stems", {}).get(stem_name)
    if stem_entry is None:
        return None
    if isinstance(stem_entry, str):
        return resolve_manifest_audio_path(manifest_path, stem_entry)
    if isinstance(stem_entry, dict) and stem_entry.get("path"):
        return resolve_manifest_audio_path(manifest_path, str(stem_entry["path"]))
    return None


def write_exact_reference_audit_bundle(
    *,
    ground_truth_manifest: Path,
    candidate_manifest: Path,
    output_dir: Path,
    stem_names: Sequence[str],
    thresholds: quality_gate.ObjectiveThresholds,
    max_evidence_frames: int | None = DEFAULT_EXACT_AUDIT_MAX_EVIDENCE_FRAMES,
) -> dict[str, Any]:
    ground_truth_payload = read_candidate_manifest(ground_truth_manifest)
    candidate_payload = read_candidate_manifest(candidate_manifest)
    stem_reports: dict[str, Any] = {}
    failed_stems: list[str] = []
    skipped_stems: list[str] = []
    for stem_name in stem_names:
        reference_path = stem_audio_path_from_manifest(ground_truth_manifest, ground_truth_payload, stem_name)
        candidate_path = stem_audio_path_from_manifest(candidate_manifest, candidate_payload, stem_name)
        if reference_path is None or candidate_path is None:
            skipped_stems.append(stem_name)
            stem_reports[stem_name] = {
                "status": "skipped",
                "reason": "stem_path_missing",
                "reference_path": str(reference_path) if reference_path is not None else None,
                "candidate_path": str(candidate_path) if candidate_path is not None else None,
            }
            continue
        stem_output_dir = output_dir / stem_name
        audit_report = write_exact_reference_listen_audit(
            reference_path=reference_path,
            candidate_path=candidate_path,
            output_dir=stem_output_dir,
            stem_name=stem_name,
            min_sdr_db=thresholds.min_sdr_db,
            min_cosine_percent=95.0,
            max_residual_peak_dbfs=-45.0,
            defect_threshold_dbfs=-60.0,
            max_evidence_frames=max_evidence_frames,
        )
        delivery_status = audit_report["delivery"]["status"]
        if delivery_status != "accepted":
            failed_stems.append(stem_name)
        stem_reports[stem_name] = {
            "status": "available",
            "delivery_status": delivery_status,
            "delivery_reason": audit_report["delivery"]["reason"],
            "failed_checks": audit_report["delivery"]["failed_checks"],
            "report_json": str(stem_output_dir / "exact_reference_listen_audit.json"),
            "forensic_evidence_dir": str(stem_output_dir / "forensic_evidence"),
            "forensic_evidence_scope": audit_report.get("forensic_evidence", {}).get("artifact_scope", {}),
            "metrics": audit_report["metrics"],
        }
    if failed_stems:
        delivery_status = "rejected"
        delivery_reason = "exact_reference_threshold_failed"
    elif skipped_stems:
        delivery_status = "skipped"
        delivery_reason = "exact_reference_stem_path_missing"
    else:
        delivery_status = "accepted"
        delivery_reason = "exact_reference_threshold_passed"
    return {
        "status": "available" if not skipped_stems else "partial",
        "delivery_status": delivery_status,
        "delivery_reason": delivery_reason,
        "output_dir": str(output_dir),
        "max_evidence_frames": int(max_evidence_frames) if max_evidence_frames is not None else None,
        "failed_stems": failed_stems,
        "skipped_stems": skipped_stems,
        "stems": stem_reports,
    }


def candidate_contract_value(payload: dict[str, Any], key: str) -> Any:
    contract = payload.get("contract", {})
    if isinstance(contract, dict) and key in contract:
        return contract[key]
    return payload.get(key)


def build_no_reference_candidate_contract(manifest_path: Path) -> dict[str, Any]:
    payload = read_candidate_manifest(manifest_path)
    if payload.get("_read_error"):
        return {
            "status": "rejected",
            "reason": "candidate_manifest_unreadable",
            "violations": [{"field": "manifest", "expected": "readable_json", "actual": payload["_read_error"]}],
        }

    required_values = {
        "target_reference_assisted": False,
        "target_reference_used_at_application": False,
        "application_reference_free": True,
        "delivery_allowed_without_reference": True,
        "fallback_used": False,
        "per_track_hardcode_allowed": False,
        "not_timestamp_hardcoded": True,
    }
    violations: list[dict[str, Any]] = []
    for field, expected in required_values.items():
        actual = candidate_contract_value(payload, field)
        if actual is not expected:
            violations.append({"field": field, "expected": expected, "actual": actual})

    optional_false_fields = (
        "target_reference_used",
        "calibration_only",
        "not_a_no_reference_production_route",
        "not_a_no_reference_production_route_until_holdout_passes",
    )
    for field in optional_false_fields:
        actual = candidate_contract_value(payload, field)
        if actual is True:
            violations.append({"field": field, "expected": False, "actual": actual})

    profile_stage = candidate_contract_value(payload, "profile_stage")
    if profile_stage == "calibration":
        violations.append({"field": "profile_stage", "expected": "application_or_absent", "actual": profile_stage})

    profile_source_reference_assisted = candidate_contract_value(payload, "profile_source_reference_assisted")
    holdout_validation_status = candidate_contract_value(payload, "holdout_validation_status")
    if profile_source_reference_assisted is True and holdout_validation_status != "passed":
        violations.append(
            {
                "field": "holdout_validation_status",
                "expected": "passed_for_reference_assisted_profile_source",
                "actual": holdout_validation_status,
            }
        )

    if violations:
        return {
            "status": "rejected",
            "reason": "candidate_contract_not_no_reference_safe",
            "violations": violations,
        }
    return {
        "status": "accepted",
        "reason": "candidate_contract_reference_free_for_application",
        "violations": [],
    }


def candidate_summary(
    *,
    spec: CandidateManifestSpec,
    report: dict[str, Any],
    report_path: Path,
    evaluated_model_labels: Sequence[str] | None = None,
    require_no_reference_candidates: bool = False,
    exact_reference_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stem_scores = summarize_stem_scores(report)
    reference_scores = [score.get("reference_likeness_percent") for score in stem_scores]
    si_sdr_scores = [score.get("si_sdr_db") for score in stem_scores]
    sdr_scores = [score.get("sdr_db") for score in stem_scores]
    objective_delivery_status = report.get("delivery", {}).get("status", "rejected")
    delivery_status = objective_delivery_status
    delivery_reason = report.get("delivery", {}).get("reason")
    delivery_allowed = delivery_status == "accepted"
    failed_check_names = extract_failed_check_names(report)
    forensic_contract = report.get("forensic_delivery_contract", {})
    forensic_evidence = report.get("forensic_evidence", {})
    no_reference_contract = (
        build_no_reference_candidate_contract(spec.manifest_path)
        if require_no_reference_candidates
        else {"status": "not_requested", "reason": "benchmark_not_in_no_reference_production_mode", "violations": []}
    )
    if require_no_reference_candidates and no_reference_contract["status"] != "accepted":
        delivery_status = "rejected"
        delivery_allowed = False
        delivery_reason = "candidate_contract_not_no_reference_safe"
        failed_check_names = sorted({*failed_check_names, "candidate_contract_not_no_reference_safe"})
    exact_reference_audit = exact_reference_audit or {
        "status": "not_requested",
        "delivery_status": "not_requested",
        "delivery_reason": "not_requested",
        "output_dir": None,
        "failed_stems": [],
        "skipped_stems": [],
        "stems": {},
    }
    exact_reference_delivery_status = exact_reference_audit.get("delivery_status")
    if exact_reference_delivery_status not in {"accepted", "not_requested"}:
        delivery_status = "rejected"
        delivery_allowed = False
        delivery_reason = str(exact_reference_audit.get("delivery_reason") or "exact_reference_audit_not_accepted")
        failed_check_names = sorted({*failed_check_names, delivery_reason})
    ralph_ai_delivery_contract = build_ralph_ai_delivery_contract(
        delivery_allowed=delivery_allowed,
        delivery_status=delivery_status,
        delivery_reason=delivery_reason,
        exact_reference_audit=exact_reference_audit,
    )
    return {
        "label": spec.label,
        "candidate_manifest": str(spec.manifest_path),
        "evaluated_model_labels": list(evaluated_model_labels or [spec.label]),
        "objective_report_path": str(report_path),
        "objective_delivery_status": objective_delivery_status,
        "delivery_status": delivery_status,
        "delivery_allowed": delivery_allowed,
        "presentation_status": ralph_ai_delivery_contract["presentation_status"],
        "ralph_ai_delivery_contract": ralph_ai_delivery_contract,
        "delivery_reason": delivery_reason,
        "no_reference_contract": no_reference_contract,
        "forensic_contract_status": forensic_contract.get("status"),
        "forensic_uncovered_labels": forensic_contract.get("uncovered_labels", []),
        "forensic_issue_actions": forensic_contract.get("issue_actions", []),
        "forensic_evidence_status": forensic_evidence.get("status"),
        "forensic_evidence_dir": forensic_evidence.get("output_dir"),
        "exact_reference_audit_status": exact_reference_audit.get("status"),
        "exact_reference_delivery_status": exact_reference_audit.get("delivery_status"),
        "exact_reference_delivery_reason": exact_reference_audit.get("delivery_reason"),
        "exact_reference_audit_dir": exact_reference_audit.get("output_dir"),
        "exact_reference_failed_stems": exact_reference_audit.get("failed_stems", []),
        "exact_reference_skipped_stems": exact_reference_audit.get("skipped_stems", []),
        "exact_reference_stem_reports": exact_reference_audit.get("stems", {}),
        "overall_status": report.get("overall", {}).get("status"),
        "failed_stems": report.get("overall", {}).get("failed_stems", []),
        "mean_reference_likeness_percent": mean_or_none(reference_scores),
        "min_reference_likeness_percent": min_or_none(reference_scores),
        "min_si_sdr_db": min_or_none(si_sdr_scores),
        "min_sdr_db": min_or_none(sdr_scores),
        "failed_check_names": failed_check_names,
        "stem_scores": stem_scores,
    }


def ranking_key(summary: dict[str, Any]) -> tuple[int, float, float, float]:
    accepted_score = 1 if summary.get("delivery_status") == "accepted" else 0
    mean_likeness = float(summary.get("mean_reference_likeness_percent") or -1.0)
    min_si_sdr = float(summary.get("min_si_sdr_db") or -999.0)
    min_sdr = float(summary.get("min_sdr_db") or -999.0)
    return (accepted_score, mean_likeness, min_si_sdr, min_sdr)


def candidate_model_label_set(candidate: dict[str, Any]) -> set[str]:
    labels = {str(candidate.get("label", "")).lower()}
    labels.update(str(label).lower() for label in candidate.get("evaluated_model_labels", []))
    return {label for label in labels if label}


def candidate_metric_at_least(candidate: dict[str, Any], key: str, minimum: float) -> bool:
    value = candidate.get(key)
    return isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) >= minimum


def adaptive_repair_issue_actions(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    repair_actions = []
    for issue_action in candidate.get("forensic_issue_actions", []):
        label = str(issue_action.get("label", ""))
        if label in ADAPTIVE_REPAIR_PRIORITY_LABELS:
            repair_actions.append(dict(issue_action))
    return repair_actions


def build_adaptive_repair_plan(ranked_candidates: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    for candidate in ranked_candidates:
        if candidate.get("delivery_status") == "accepted":
            continue
        if candidate.get("forensic_uncovered_labels"):
            continue
        if candidate.get("forensic_evidence_status") != "available":
            continue
        if candidate.get("no_reference_contract", {}).get("status") == "rejected":
            continue
        if not candidate_model_label_set(candidate) & ADAPTIVE_REPAIR_BASE_MODEL_LABELS:
            continue
        if not candidate_metric_at_least(
            candidate,
            "mean_reference_likeness_percent",
            ADAPTIVE_REPAIR_MIN_REFERENCE_LIKENESS_PERCENT,
        ):
            continue
        if not candidate_metric_at_least(candidate, "min_sdr_db", ADAPTIVE_REPAIR_MIN_SDR_DB):
            continue
        if not candidate_metric_at_least(candidate, "min_si_sdr_db", ADAPTIVE_REPAIR_MIN_SI_SDR_DB):
            continue

        repair_actions = adaptive_repair_issue_actions(candidate)
        if not repair_actions:
            continue

        return {
            "schema_version": 1,
            "status": "repair_candidate_required",
            "base_candidate_label": candidate["label"],
            "base_candidate_manifest": candidate["candidate_manifest"],
            "base_candidate_metrics": {
                "mean_reference_likeness_percent": candidate.get("mean_reference_likeness_percent"),
                "min_reference_likeness_percent": candidate.get("min_reference_likeness_percent"),
                "min_sdr_db": candidate.get("min_sdr_db"),
                "min_si_sdr_db": candidate.get("min_si_sdr_db"),
            },
            "target_labels": sorted({str(action["label"]) for action in repair_actions}),
            "repair_actions": repair_actions,
            "forensic_evidence_dir": candidate.get("forensic_evidence_dir"),
            "guardrails": {
                "accepts_rejected_base_as_final": False,
                "requires_rebenchmark": True,
                "fallback_used": False,
                "per_track_hardcode_allowed": False,
                "not_timestamp_hardcoded": True,
            },
        }
    return None


def next_action_for_candidates(ranked_candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return build_next_action(ranked_candidates=ranked_candidates, candidate_suggestions=[])


def candidate_manifest_model_identifiers(manifest_path: Path) -> list[str]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    identifiers: list[str] = []
    for key in ("model_id", "model_name", "model"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            identifiers.append(value.strip())
        elif isinstance(value, dict):
            for nested_key in ("model_id", "model_name", "name"):
                nested_value = value.get(nested_key)
                if isinstance(nested_value, str) and nested_value.strip():
                    identifiers.append(nested_value.strip())
    return identifiers


def evaluated_model_labels_for_spec(
    *,
    spec: CandidateManifestSpec,
    registry: ModelRegistry,
) -> list[str]:
    labels = {spec.label}
    for identifier in candidate_manifest_model_identifiers(spec.manifest_path):
        labels.add(identifier)
        try:
            entry = registry.get(identifier)
        except KeyError:
            continue
        labels.add(entry.model_id)
        labels.update(entry.aliases)
    return sorted(labels)


def build_next_action(
    *,
    ranked_candidates: Sequence[dict[str, Any]],
    candidate_suggestions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not ranked_candidates:
        return {
            "status": "no_candidates_supplied",
            "action": "provide_candidate_manifests",
            "reason": "benchmark cannot evaluate an empty model zoo",
        }
    selected = next(
        (candidate for candidate in ranked_candidates if candidate.get("delivery_status") == "accepted"),
        None,
    )
    if selected is not None:
        return {
            "status": "accepted_candidate_found",
            "action": "promote_selected_candidate_for_listen_pack",
            "selected_label": selected["label"],
            "reason": "selected candidate passed the verified ground-truth delivery gate",
        }

    uncovered_labels = sorted(
        {
            label
            for candidate in ranked_candidates
            for label in candidate.get("failed_check_names", [])
        }
    )
    adaptive_repair_plan = build_adaptive_repair_plan(ranked_candidates)
    if adaptive_repair_plan is not None:
        return {
            "status": "no_accepted_candidate",
            "action": "repair_best_ranked_candidate",
            "reason": "best rejected candidate has mapped forensic defects, evidence, and enough signal quality for a repair/rerank pass",
            "uncovered_labels": uncovered_labels,
            "repair_plan": adaptive_repair_plan,
            "candidate_suggestions": list(candidate_suggestions),
        }
    if candidate_suggestions:
        return {
            "status": "no_accepted_candidate",
            "action": "generate_registered_model_candidates",
            "reason": "all evaluated candidates failed; registered unevaluated candidates remain available",
            "uncovered_labels": uncovered_labels,
            "candidate_suggestions": list(candidate_suggestions),
        }
    return {
        "status": "candidate_space_exhausted",
        "action": "reject_delivery_candidate_space_exhausted",
        "reason": "all candidates failed and no registered unevaluated candidate can address the failed labels",
        "uncovered_labels": uncovered_labels,
        "candidate_suggestions": [],
    }


def run_model_zoo_benchmark(
    *,
    ground_truth_manifest: Path,
    candidate_manifests: Sequence[CandidateManifestSpec],
    output_dir: Path,
    stem_names: Sequence[str],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    max_offset_seconds: float = 0.0,
    thresholds: quality_gate.ObjectiveThresholds | None = None,
    registry_runtime_root: Path | None = None,
    include_unavailable_registry_candidates: bool = True,
    next_candidate_limit: int = 5,
    require_no_reference_candidates: bool = False,
    exact_audit_max_evidence_frames: int | None = DEFAULT_EXACT_AUDIT_MAX_EVIDENCE_FRAMES,
) -> dict[str, Any]:
    if not candidate_manifests:
        raise ValueError("at least one candidate manifest is required")
    thresholds = thresholds or quality_gate.ObjectiveThresholds()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / "objective_reports"

    candidate_reports: list[dict[str, Any]] = []
    registry = default_model_registry(runtime_root=registry_runtime_root or Path("/app"))
    for index, spec in enumerate(candidate_manifests, start=1):
        candidate_file_label = f"{index:02d}_{safe_file_label(spec.label)}"
        report_path = report_dir / f"{candidate_file_label}_objective_report.json"
        exact_reference_audit = write_exact_reference_audit_bundle(
            ground_truth_manifest=ground_truth_manifest,
            candidate_manifest=spec.manifest_path,
            output_dir=output_dir / "exact_reference_listen_audits" / candidate_file_label,
            stem_names=stem_names,
            thresholds=thresholds,
            max_evidence_frames=exact_audit_max_evidence_frames,
        )
        report = quality_gate.evaluate_candidate_manifest(
            ground_truth_manifest=ground_truth_manifest,
            candidate_manifest=spec.manifest_path,
            output_json=report_path,
            stem_names=stem_names,
            sample_rate=sample_rate,
            max_offset_seconds=max_offset_seconds,
            thresholds=thresholds,
            forensic_evidence_dir=output_dir / "forensic_evidence" / candidate_file_label,
        )
        candidate_reports.append(
            candidate_summary(
                spec=spec,
                report=report,
                report_path=report_path,
                evaluated_model_labels=evaluated_model_labels_for_spec(spec=spec, registry=registry),
                require_no_reference_candidates=require_no_reference_candidates,
                exact_reference_audit=exact_reference_audit,
            )
        )

    ranked_candidates = sorted(candidate_reports, key=ranking_key, reverse=True)
    selected_candidate = next(
        (candidate for candidate in ranked_candidates if candidate.get("delivery_status") == "accepted"),
        None,
    )
    failed_labels = sorted(
        {
            label
            for candidate in ranked_candidates
            for label in candidate.get("failed_check_names", [])
        }
    )
    evaluated_labels = sorted(
        {
            label
            for candidate in ranked_candidates
            for label in candidate.get("evaluated_model_labels", [candidate["label"]])
        }
    )
    candidate_suggestions = [
        suggestion.to_dict()
        for suggestion in registry.suggest_next_candidates(
            stem_names=stem_names,
            failed_labels=failed_labels,
            evaluated_labels=evaluated_labels,
            include_unavailable=include_unavailable_registry_candidates,
            limit=next_candidate_limit,
        )
    ]
    next_action = build_next_action(
        ranked_candidates=ranked_candidates,
        candidate_suggestions=candidate_suggestions,
    )
    manifest = {
        "schema_version": 1,
        "execution": {
            "tool": "ml_worker_improved/tools/run_model_zoo_benchmark.py",
            "generated_at_utc": quality_gate.utc_timestamp(),
            "intended_runtime": "docker ml-worker",
        },
        "contract": {
            "benchmark_type": "reference_driven_model_zoo",
            "reference_assisted": True,
            "verified_ground_truth_required": True,
            "acceptance_requires_delivery_accepted": True,
            "fallback_used": False,
            "not_timestamp_hardcoded": True,
            "per_track_hardcode_allowed": False,
            "no_reference_production_claim": False,
            "no_reference_candidate_contract_enforced": require_no_reference_candidates,
            "require_no_reference_candidates": require_no_reference_candidates,
            "exact_reference_listen_audit_enforced": True,
            "exact_reference_evidence_bounded": exact_audit_max_evidence_frames is not None,
            "exact_reference_max_evidence_frames": (
                int(exact_audit_max_evidence_frames) if exact_audit_max_evidence_frames is not None else None
            ),
            "ralph_ai_delivery_contract_enforced": True,
            "best_of_bad_candidate_delivery_forbidden": True,
            "rejected_candidates_are_diagnostic_only": True,
            "candidate_generation": "external_candidate_manifests",
            "bounded_auto_rerun": {
                "enabled": True,
                "max_suggested_candidates": next_candidate_limit,
                "executes_inference": False,
                "exhaustion_action": "reject_delivery_candidate_space_exhausted",
            },
        },
        "inputs": {
            "ground_truth_manifest": str(ground_truth_manifest),
            "candidate_manifests": [
                {"label": spec.label, "path": str(spec.manifest_path)}
                for spec in candidate_manifests
            ],
            "stem_names": list(stem_names),
            "sample_rate": sample_rate,
            "max_offset_seconds": max_offset_seconds,
            "thresholds": {
                "min_sdr_db": thresholds.min_sdr_db,
                "min_si_sdr_db": thresholds.min_si_sdr_db,
                "min_sir_db": thresholds.min_sir_db,
                "min_sar_db": thresholds.min_sar_db,
                "min_reference_likeness_percent": thresholds.min_reference_likeness_percent,
            },
            "registry_runtime_root": str(registry.runtime_root),
            "include_unavailable_registry_candidates": include_unavailable_registry_candidates,
            "require_no_reference_candidates": require_no_reference_candidates,
        },
        "model_registry": registry.to_manifest(),
        "metric_stack_contract": build_metric_stack_contract(),
        "candidate_reports": candidate_reports,
        "ranked_candidates": ranked_candidates,
        "selected_candidate": selected_candidate,
        "next_action": next_action,
        "overall": {
            "status": next_action["status"],
            "accepted_count": sum(
                1 for candidate in candidate_reports if candidate.get("delivery_status") == "accepted"
            ),
            "rejected_count": sum(
                1 for candidate in candidate_reports if candidate.get("delivery_status") != "accepted"
            ),
        },
    }
    write_json(output_dir / "model_zoo_benchmark_manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank model-zoo candidate manifests against verified isolated ground-truth stems."
    )
    parser.add_argument("--ground-truth-manifest", type=Path, required=True)
    parser.add_argument(
        "--candidate-manifest",
        action="append",
        type=parse_candidate_manifest,
        required=True,
        help="Candidate manifest in LABEL=PATH format. Repeat for each model/candidate.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", action="append")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--max-offset-seconds", type=float, default=0.0)
    parser.add_argument("--registry-runtime-root", type=Path, default=Path("/app"))
    parser.add_argument("--next-candidate-limit", type=int, default=5)
    parser.add_argument(
        "--exact-audit-max-evidence-frames",
        type=int,
        default=DEFAULT_EXACT_AUDIT_MAX_EVIDENCE_FRAMES,
        help="Bound heavy exact-audit trace/residual artifacts while keeping full-audio metrics.",
    )
    parser.add_argument(
        "--available-registry-candidates-only",
        action="store_true",
        help="Do not suggest registered models whose required paths are missing.",
    )
    parser.add_argument(
        "--require-no-reference-candidates",
        action="store_true",
        help="Hard-reject candidates whose manifest is not safe for no-reference production delivery.",
    )
    parser.add_argument("--min-sdr-db", type=float, default=quality_gate.ObjectiveThresholds.min_sdr_db)
    parser.add_argument("--min-si-sdr-db", type=float, default=quality_gate.ObjectiveThresholds.min_si_sdr_db)
    parser.add_argument("--min-sir-db", type=float, default=quality_gate.ObjectiveThresholds.min_sir_db)
    parser.add_argument("--min-sar-db", type=float, default=quality_gate.ObjectiveThresholds.min_sar_db)
    parser.add_argument(
        "--min-reference-likeness-percent",
        type=float,
        default=quality_gate.ObjectiveThresholds.min_reference_likeness_percent,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_model_zoo_benchmark.py",
        entrypoint_path="/app/tools/run_model_zoo_benchmark.py",
    )
    args = build_arg_parser().parse_args(argv)
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be positive")
    if args.max_offset_seconds < 0.0:
        raise ValueError("--max-offset-seconds must not be negative")
    if args.next_candidate_limit < 0:
        raise ValueError("--next-candidate-limit must not be negative")
    if args.exact_audit_max_evidence_frames is not None and args.exact_audit_max_evidence_frames <= 0:
        raise ValueError("--exact-audit-max-evidence-frames must be positive when provided")

    thresholds = quality_gate.ObjectiveThresholds(
        min_sdr_db=args.min_sdr_db,
        min_si_sdr_db=args.min_si_sdr_db,
        min_sir_db=args.min_sir_db,
        min_sar_db=args.min_sar_db,
        min_reference_likeness_percent=args.min_reference_likeness_percent,
    )
    manifest = run_model_zoo_benchmark(
        ground_truth_manifest=args.ground_truth_manifest,
        candidate_manifests=args.candidate_manifest,
        output_dir=args.output_dir,
        stem_names=normalize_requested_stems(args.stem),
        sample_rate=args.sample_rate,
        max_offset_seconds=args.max_offset_seconds,
        thresholds=thresholds,
        registry_runtime_root=args.registry_runtime_root,
        include_unavailable_registry_candidates=not bool(args.available_registry_candidates_only),
        next_candidate_limit=args.next_candidate_limit,
        require_no_reference_candidates=bool(args.require_no_reference_candidates),
        exact_audit_max_evidence_frames=args.exact_audit_max_evidence_frames,
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / "model_zoo_benchmark_manifest.json"),
                "overall": manifest["overall"],
                "selected_candidate": manifest["selected_candidate"],
                "next_action": manifest["next_action"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
