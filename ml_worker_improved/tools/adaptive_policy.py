from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


FailureClass = str
ActionName = str


@dataclass(frozen=True)
class IssuePolicy:
    failure_class: FailureClass
    labels: tuple[str, ...]
    failed_checks: tuple[str, ...]
    allowed_actions: tuple[ActionName, ...]
    forbidden_actions: tuple[ActionName, ...]
    default_decision: str
    required_next_action: str
    rationale: str


# Priority matters. The first matched policy becomes the dominant failure class.
ISSUE_POLICIES: tuple[IssuePolicy, ...] = (
    IssuePolicy(
        failure_class="candidate_space_exhausted",
        labels=("candidate_space_exhausted", "no_accepted_candidate"),
        failed_checks=("candidate_space_exhausted", "no_accepted_candidate"),
        allowed_actions=("mark_hard_case", "send_to_finetune_set", "manual_hard_segment_ab"),
        forbidden_actions=("generate_more_ab_candidates", "switch_backbone", "switch_backbone_without_new_diagnostic", "stack_more_cleanup"),
        default_decision="hard_case",
        required_next_action="stop_candidate_generation_and_route_to_hard_case_dashboard",
        rationale="The current candidate space already failed; more blind candidates create noise, not progress.",
    ),
    IssuePolicy(
        failure_class="alignment_or_reference_failure",
        labels=(
            "sample_residual_above_threshold",
            "micro_residual_spike",
            "target_shape_mismatch",
            "full_mix_reconstruction_failed",
            "alignment_uncertain",
        ),
        failed_checks=(
            "sample_residual_above_threshold",
            "micro_residual_spike",
            "target_shape_mismatch",
            "full_mix_reconstruction_failed",
            "alignment_uncertain",
        ),
        allowed_actions=("fix_alignment", "reject_reference_bundle", "rebuild_ground_truth_bundle"),
        forbidden_actions=("switch_backbone", "generate_more_ab_candidates", "fine_tune_on_unverified_pair"),
        default_decision="reject",
        required_next_action="repair_or_rebuild_reference_before_any_model_change",
        rationale="Reference or sample-level contract is broken; model changes are meaningless until QC/alignment is fixed.",
    ),
    IssuePolicy(
        failure_class="robotic_or_noise_artifact",
        labels=("high_noise_or_robotic_artifact", "artifact_or_timbre_damage", "robotic_artifact", "musical_noise"),
        failed_checks=("high_noise_or_robotic_artifact", "artifact_or_timbre_damage", "min_sar_db"),
        allowed_actions=("rollback_to_cleaner_candidate", "perceptual_refiner_candidate", "lower_cleanup_strength"),
        forbidden_actions=("stack_more_cleanup", "stronger_denoise", "stronger_debleed", "hard_vad_gate"),
        default_decision="repair_once",
        required_next_action="try_one_artifact_safe_refiner_or_rollback_then_validate_sar_and_ab",
        rationale="Artifacts usually get worse when cleanup stages are stacked.",
    ),
    IssuePolicy(
        failure_class="thin_or_overcleaned",
        labels=("thin_or_overcleaned", "level_or_body_drift", "micro_level_or_body_drift", "presence_loss", "air_detail_loss", "local_vocal_body_dip"),
        failed_checks=("level_or_body_drift", "micro_level_or_body_drift", "reference_likeness_below_threshold"),
        allowed_actions=("body_repair", "presence_band_recovery", "rollback_cleanup_strength", "segment_wise_blend"),
        forbidden_actions=("stronger_debleed", "stronger_denoise", "hard_vad_gate", "switch_backbone"),
        default_decision="repair_once",
        required_next_action="restore_body_or_presence_once_then_validate_reference_likeness_and_artifacts",
        rationale="Thin vocal is usually caused by over-cleaning; stronger cleanup is the wrong direction.",
    ),
    IssuePolicy(
        failure_class="vocal_tail_loss",
        labels=("vocal_tail_loss", "reverb_tail_loss", "tail_loss", "phrase_tail_loss"),
        failed_checks=("vocal_tail_loss", "reverb_tail_loss"),
        allowed_actions=("tail_preserving_blend", "disable_hard_tail_gate", "segment_tail_blend"),
        forbidden_actions=("hard_vad_gate", "aggressive_cleanup", "stronger_debleed", "cut_silence_gate"),
        default_decision="repair_once",
        required_next_action="apply_tail_preserving_strategy_once_then_validate_tail_windows",
        rationale="Tail loss is made worse by hard gates and aggressive cleanup.",
    ),
    IssuePolicy(
        failure_class="vocal_dropout",
        labels=("vocal_dropout", "local_vocal_drop", "bar_spanning_vocal_drop"),
        failed_checks=("vocal_dropout", "reference_likeness_below_threshold"),
        allowed_actions=("segment_wise_blend", "donor_candidate_splice", "body_repair"),
        forbidden_actions=("global_cleanup", "hard_vad_gate", "switch_backbone_without_segment_evidence"),
        default_decision="repair_once",
        required_next_action="repair_only_the_dropped_segment_then_validate_no_new_leakage",
        rationale="Dropout is local; global model switching or cleanup can damage good sections.",
    ),
    IssuePolicy(
        failure_class="vocal_bleed_in_minus",
        labels=("vocal_bleed", "reference_stem_leakage", "micro_reference_stem_leakage", "wide_stereo_non_vocal_leakage"),
        failed_checks=("reference_stem_leakage", "micro_reference_stem_leakage"),
        allowed_actions=("vocal_activity_gate_soft", "frequency_aware_debleed", "reference_guided_residual_cleanup"),
        forbidden_actions=("global_subtractive_cleanup_without_ab", "hard_vad_gate_on_tails", "switch_backbone_first"),
        default_decision="repair_once",
        required_next_action="apply_one_vocal_safe_debleed_then_validate_false_vocal_energy_and_tail_integrity",
        rationale="Vocal bleed can be repaired, but hard/global gates often destroy tails and body.",
    ),
    IssuePolicy(
        failure_class="instrumental_bleed_in_vocal",
        labels=("instrumental_bleed", "drop_melody_leakage", "bar_spanning_drop_melody_leakage", "unsupported_anchor_leakage", "high_band_unsupported_leakage"),
        failed_checks=("min_sir_db", "reference_likeness_below_threshold"),
        allowed_actions=("segment_wise_blend", "frequency_aware_blend", "anchor_supported_cleanup"),
        forbidden_actions=("stronger_mask_globally", "full_track_denoise", "switch_backbone_without_leakage_map"),
        default_decision="repair_once",
        required_next_action="localize_leakage_then_apply_segment_or_band_limited_repair",
        rationale="Instrumental leakage is often local or band-specific; full-track cleanup causes timbre damage.",
    ),
    IssuePolicy(
        failure_class="transient_or_attack_mismatch",
        labels=("transient_or_attack_mismatch", "micro_transient_mismatch", "kick_punch_bleed", "transient_smearing"),
        failed_checks=("transient_or_attack_mismatch", "micro_transient_mismatch"),
        allowed_actions=("transient_preserving_blend", "reduce_overlap_damage", "localized_attack_repair"),
        forbidden_actions=("global_dereverb", "stronger_denoise", "waveform_average_without_alignment"),
        default_decision="repair_once",
        required_next_action="repair_transient_windows_only_then_validate_attack_metrics",
        rationale="Transient problems should be fixed locally; broad processing smears attacks further.",
    ),
    IssuePolicy(
        failure_class="backing_vocal_confusion",
        labels=("backing_vocal_loss_risk", "backing_vocal_confusion", "harmony_vocal_ambiguity"),
        failed_checks=("backing_vocal_loss_risk",),
        allowed_actions=("manual_hard_segment_ab", "mark_ambiguous_vocal_class", "send_to_finetune_set"),
        forbidden_actions=("treat_backing_vocals_as_always_leakage", "hard_remove_harmonies"),
        default_decision="hard_case",
        required_next_action="manual_ab_or_finetune_bucket_because_target_definition_is_ambiguous",
        rationale="Backing vocals may be desired vocal content or leakage depending on product target.",
    ),
    IssuePolicy(
        failure_class="mastered_or_compressed_failure",
        labels=("mastered_compressed_failure", "clipped_input", "limiter_pumping", "codec_or_mastering_artifact"),
        failed_checks=("clipped_input", "mastered_compressed_failure"),
        allowed_actions=("master_aware_bundle", "mastered_domain_finetune_bucket", "manual_hard_segment_ab"),
        forbidden_actions=("clean_bundle_thresholds", "fine_tune_as_clean_pair"),
        default_decision="fine_tune_candidate",
        required_next_action="route_to_master_aware_domain_adaptation_or_hard_case",
        rationale="Mastered/compressed tracks are a domain mismatch, not a simple post-processing failure.",
    ),
    IssuePolicy(
        failure_class="metric_failure_unspecified",
        labels=("low_sdr", "low_si_sdr", "min_sdr_db", "min_si_sdr_db", "min_sar_db", "reference_likeness_below_threshold"),
        failed_checks=("low_sdr", "low_si_sdr", "min_sdr_db", "min_si_sdr_db", "min_sar_db", "reference_likeness_below_threshold"),
        allowed_actions=("localize_failure_windows", "build_diagnostic_ab_pack", "manual_hard_segment_ab"),
        forbidden_actions=("switch_backbone_without_error_taxonomy", "generate_more_ab_candidates"),
        default_decision="reject",
        required_next_action="localize_failure_before_any_repair_or_model_change",
        rationale="Generic metric failure is not a diagnosis; localize the real issue first.",
    ),
)

POLICIES_BY_CLASS: dict[str, IssuePolicy] = {policy.failure_class: policy for policy in ISSUE_POLICIES}


def normalize_label(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def normalize_labels(values: Iterable[Any] | None) -> set[str]:
    if not values:
        return set()
    return {normalize_label(value) for value in values if str(value).strip()}


def classify_failure(*, issue_labels: Iterable[Any] | None = None, failed_checks: Iterable[Any] | None = None) -> IssuePolicy | None:
    labels = normalize_labels(issue_labels)
    checks = normalize_labels(failed_checks)
    for policy in ISSUE_POLICIES:
        policy_labels = normalize_labels(policy.labels)
        policy_checks = normalize_labels(policy.failed_checks)
        if labels.intersection(policy_labels) or checks.intersection(policy_checks):
            return policy
    return None


def decision_from_evidence(*, issue_labels: Iterable[Any] | None = None, failed_checks: Iterable[Any] | None = None, candidate_status: str | None = None) -> dict[str, Any]:
    labels = normalize_labels(issue_labels)
    checks = normalize_labels(failed_checks)
    status = normalize_label(candidate_status or "")
    if status in {"passed", "accepted", "release_ready", "accepted_for_listen_pack"} and not labels and not checks:
        return {
            "dominant_failure_class": "pass_or_no_dominant_issue",
            "decision": "promote",
            "allowed_actions": ["promote_without_modification"],
            "forbidden_actions": ["unnecessary_repair", "unnecessary_model_switch", "generate_more_ab_candidates"],
            "required_next_action": "preserve_as_regression_guard",
            "rationale": "No blocking issue was detected; do not modify a good result.",
        }
    policy = classify_failure(issue_labels=labels, failed_checks=checks)
    if policy is None:
        return {
            "dominant_failure_class": "unknown_failure",
            "decision": "reject",
            "allowed_actions": ["inspect_manifest_schema", "localize_failure_windows"],
            "forbidden_actions": ["switch_backbone", "generate_more_ab_candidates"],
            "required_next_action": "make_failure_observable_before_repair",
            "rationale": "No known policy matched the failure labels/checks.",
        }
    return {
        "dominant_failure_class": policy.failure_class,
        "decision": policy.default_decision,
        "allowed_actions": list(policy.allowed_actions),
        "forbidden_actions": list(policy.forbidden_actions),
        "required_next_action": policy.required_next_action,
        "rationale": policy.rationale,
    }


def action_allowed(action: str, decision: dict[str, Any]) -> bool:
    normalized = normalize_label(action)
    forbidden = normalize_labels(decision.get("forbidden_actions", []))
    allowed = normalize_labels(decision.get("allowed_actions", []))
    return normalized in allowed and normalized not in forbidden


def assert_action_not_forbidden(action: str, decision: dict[str, Any]) -> None:
    normalized = normalize_label(action)
    forbidden = normalize_labels(decision.get("forbidden_actions", []))
    if normalized in forbidden:
        raise ValueError(
            f"Action {action!r} is forbidden for {decision.get('dominant_failure_class')}: "
            f"{decision.get('rationale')}"
        )
