from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

@dataclass(frozen=True)
class VocalCandidateProfile:
    name: str
    candidate_mode: str
    prior_score: float
    intent: str
    tuning: dict[str, float | int | str]


@dataclass(frozen=True)
class VocalCandidateScore:
    profile_name: str
    score: float
    reasons: list[str]
    metrics: dict[str, float]
    manifest_path: str
    output_dir: str
    vocals_path: str


@dataclass(frozen=True)
class AnchorSuppressionProfile:
    name: str
    source_profile_name: str
    anchor_profile_name: str
    prior_score: float
    intent: str
    tuning: dict[str, float | int]


@dataclass(frozen=True)
class VocalCandidateRecord:
    profile: VocalCandidateProfile | AnchorSuppressionProfile | "VocalBodyRepairProfile" | "SegmentRouterProfile"
    manifest_path: Path
    output_dir: Path
    vocals_path: Path
    metrics: dict[str, float]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class VocalBodyRepairProfile:
    name: str
    base_profile_name: str
    donor_profile_names: tuple[str, ...]
    prior_score: float
    intent: str
    tuning: dict[str, float | int | str]


@dataclass(frozen=True)
class SegmentRouterProfile:
    name: str
    clean_profile_name: str
    strict_profile_name: str
    repair_profile_name: str
    preserve_profile_name: str
    anchor_profile_name: str
    prior_score: float
    intent: str
    tuning: dict[str, float | int | str]

def default_issue_aware_segment_selection_profile() -> VocalCandidateProfile:
    return VocalCandidateProfile(
        name="issue_aware_segment_selection",
        candidate_mode="issue_aware_segment_selected_vocals",
        prior_score=0.84,
        intent=(
            "Use segment issue diagnostics to select the least problematic existing vocal candidate "
            "per window without per-track timestamps or another model inference pass."
        ),
        tuning={
            "min_segment_issue_score": 0.42,
            "min_adjusted_improvement": 0.045,
            "global_score_penalty_weight": 0.012,
            "max_global_score_drop": 2.20,
            "leaky_full_mix_label_penalty": 0.28,
            "unsupported_anchor_label_penalty": 0.18,
            "high_band_unsupported_label_penalty": 0.12,
            "backing_vocal_loss_label_penalty": 0.12,
            "thin_or_overcleaned_label_penalty": 0.08,
            "crossfade_sec": 0.08,
        },
    )


def default_segment_issue_repair_profile() -> VocalCandidateProfile:
    return VocalCandidateProfile(
        name="segment_issue_repair",
        candidate_mode="segment_issue_repaired_vocals",
        prior_score=0.86,
        intent=(
            "Build a segment-level multi-issue resolution candidate for detector labels by combining "
            "vocal body/tail recovery with strict/anchor-supported debleed only inside diagnosed windows."
        ),
        tuning={
            "min_segment_issue_score": 0.50,
            "min_bleed_issue_score": 0.42,
            "min_recovery_improvement": 0.012,
            "candidate_score_penalty_weight": 0.004,
            "crossfade_sec": 0.10,
            "body_low_hz": 150.0,
            "body_high_hz": 5600.0,
            "core_debleed_low_hz": 65.0,
            "core_debleed_high_hz": 5600.0,
            "high_band_start_hz": 4500.0,
            "high_band_end_hz": 15000.0,
            "body_deficit_threshold": 0.08,
            "body_repair_strength": 0.88,
            "preserve_tail_strength": 0.26,
            "core_debleed_strength": 0.0,
            "bleed_core_debleed_strength": 0.58,
            "combined_core_debleed_strength": 0.36,
            "dropout_core_debleed_strength": 0.22,
            "preserve_max_ratio": 1.08,
            "max_body_fill_ratio": 1.12,
            "body_support_floor": 0.18,
            "dropout_body_support_floor": 0.12,
            "body_support_power": 0.72,
            "unsupported_busy_damping": 0.90,
            "dropout_unsupported_busy_damping": 0.74,
            "core_unsupported_tolerance": 1.02,
            "kick_core_unsupported_tolerance": 0.96,
            "core_unsupported_damping": 0.98,
            "strict_high_tolerance": 1.02,
            "strict_high_strength": 0.92,
            "activity_floor_ratio": 0.060,
            "activity_smoothing_frames": 11,
            "raw_busy_percentile": 76.0,
            "raw_busy_boost": 0.10,
            "target_rms_ratio": 0.98,
            "max_gain": 1.04,
            "phase_strategy": "clean_base",
        },
    )


def default_post_gate_residual_issue_repair_profile() -> VocalCandidateProfile:
    base_profile = default_segment_issue_repair_profile()
    return VocalCandidateProfile(
        name="post_gate_residual_issue_repair",
        candidate_mode="post_gate_residual_issue_repaired_vocals",
        prior_score=0.88,
        intent=(
            "Run a second-pass segment repair from the actually selected blocked candidate when "
            "the defect gate proves that severe audible issues remain after normal ranking."
        ),
        tuning={
            **base_profile.tuning,
            "min_segment_issue_score": 0.0,
            "min_bleed_issue_score": 0.0,
            "min_recovery_improvement": 0.0,
            "candidate_score_penalty_weight": 0.002,
        },
    )


def default_gate_blocker_best_segment_splice_profile() -> VocalCandidateProfile:
    return VocalCandidateProfile(
        name="gate_blocker_best_segment_splice",
        candidate_mode="gate_blocker_best_segment_spliced_vocals",
        prior_score=0.89,
        intent=(
            "Build a direct blocker-window splice candidate from the best objectively measured donor "
            "when the delivery gate proves that severe vocal defects remain after normal repair."
        ),
        tuning={
            "min_blocker_severity_score": 1.0,
            "min_donor_severity_improvement": 0.08,
            "crossfade_sec": 0.04,
        },
    )


def default_gate_blocker_presence_band_recovery_profile() -> VocalCandidateProfile:
    return VocalCandidateProfile(
        name="gate_blocker_presence_band_recovery",
        candidate_mode="gate_blocker_presence_band_recovered_vocals",
        prior_score=0.90,
        intent=(
            "Build a measured presence-band recovery candidate for delivery gate blockers by "
            "adding only donor presence-band delta and attenuating the base bleed band when the "
            "predicted segment issue score improves."
        ),
        tuning={
            "min_blocker_severity_score": 1.0,
            "min_predicted_segment_improvement": 0.08,
            "presence_low_hz": 900.0,
            "presence_high_hz": 4500.0,
            "bleed_low_hz": 250.0,
            "bleed_high_hz": 900.0,
            "presence_delta_gain": 2.0,
            "bleed_band_gain": 0.95,
            "crossfade_sec": 0.04,
            "band_filter_n_fft": 2048,
            "band_filter_hop_length": 512,
        },
    )


def default_clustered_issue_window_ratio_repair_profile() -> VocalCandidateProfile:
    return VocalCandidateProfile(
        name="clustered_issue_window_ratio_repair",
        candidate_mode="clustered_issue_window_ratio_repaired_vocals",
        prior_score=0.91,
        intent=(
            "Build a measured repair candidate for non-severe clustered issue windows by attenuating only "
            "the label-matched bleed/noise band when the predicted segment issue score improves."
        ),
        tuning={
            "min_segment_issue_score": 0.20,
            "min_predicted_segment_improvement": 0.02,
            "max_repaired_windows": 96,
            "kick_low_hz": 250.0,
            "kick_high_hz": 900.0,
            "kick_band_gain": 0.65,
            "kick_band_gain_strong": 0.45,
            "kick_band_gain_aggressive": 0.30,
            "kick_band_gain_extreme": 0.16,
            "low_low_hz": 35.0,
            "low_high_hz": 180.0,
            "low_band_gain": 0.70,
            "low_band_gain_strong": 0.52,
            "full_mix_low_hz": 120.0,
            "full_mix_high_hz": 900.0,
            "full_mix_band_gain": 0.82,
            "full_mix_band_gain_strong": 0.68,
            "high_low_hz": 4500.0,
            "high_high_hz": 9000.0,
            "high_band_gain": 0.86,
            "high_band_gain_strong": 0.72,
            "crossfade_sec": 0.04,
            "band_filter_method": "fft",
            "band_filter_n_fft": 2048,
            "band_filter_hop_length": 512,
        },
    )


def default_no_vocal_leakage_cleanup_profile() -> VocalCandidateProfile:
    return VocalCandidateProfile(
        name="no_vocal_leakage_cleanup",
        candidate_mode="no_vocal_leakage_cleaned_vocals",
        prior_score=0.92,
        intent=(
            "Suppress instrumental-only leakage in the selected vocal stem when candidate consensus "
            "does not provide reliable vocal evidence, without per-track timestamps."
        ),
        tuning={
            "min_consensus_issue_score": 1.15,
            "bridge_candidate_issue_score": 0.15,
            "bridge_rms_ratio_db": -22.0,
            "drop_melody_only_max_candidate_rms_dbfs": -24.0,
            "drop_melody_only_min_mid_band_ratio_db": -7.0,
            "drop_melody_only_max_presence_ratio_db": -3.0,
            "drop_melody_only_max_flatness": 0.080,
            "drop_melody_only_max_flux": 0.130,
            "drop_melody_only_min_peak_ratio": 0.25,
            "drop_melody_only_min_best_candidate_issue_score": 0.55,
            "drop_melody_only_suppression_gain": 0.055,
            "min_region_duration_sec": 5.0,
            "suppression_gain": 0.035,
            "bridge_suppression_gain": 0.08,
            "crossfade_sec": 0.08,
        },
    )


def default_vocal_harshness_guard_profile() -> VocalCandidateProfile:
    return VocalCandidateProfile(
        name="vocal_harshness_guard",
        candidate_mode="vocal_harshness_guarded_vocals",
        prior_score=0.91,
        intent=(
            "Build a high-band-only cleanup candidate from detector-marked harsh/robotic/unsupported "
            "windows while preserving the selected base vocal phase and untouched unlabelled windows."
        ),
        tuning={
            "anchor_profile": "ft_tail_safe",
            "high_band_start_hz": 6200.0,
            "high_band_end_hz": 15000.0,
            "anchor_tolerance": 1.04,
            "min_high_band_floor": 0.68,
            "base_strength": 0.62,
            "max_strength": 0.82,
            "vocal_loss_strength_scale": 0.42,
            "min_segment_severity": 0.55,
            "high_noise_flatness_threshold": 0.16,
            "fade_sec": 0.12,
            "n_fft": 4096,
            "hop_length": 512,
        },
    )


def default_candidate_profiles() -> tuple[VocalCandidateProfile, ...]:
    return (
        VocalCandidateProfile(
            name="hybrid_tail_preserve",
            candidate_mode="hybrid_vocals",
            prior_score=0.10,
            intent="Keep phrase tails and consonant endings; tolerate light bleed.",
            tuning={
                "low_bs_weight": 0.62,
                "high_bs_weight": 0.18,
                "inactive_low_bs_weight": 0.82,
                "inactive_high_bs_weight": 0.65,
                "crossover_hz": 3000.0,
                "activity_abs_floor_ratio": 0.045,
                "activity_rel_floor": 0.25,
                "activity_smoothing_frames": 15,
                "vocal_sharpen_alpha": 0.0,
                "tail_gate_strength": 0.0,
                "tail_gate_floor_ratio": 0.06,
                "tail_gate_smoothing_frames": 15,
                "subtractive_strength": 0.0,
                "subtractive_power": 1.0,
                "subtractive_residual_floor": 0.18,
                "subtractive_active_restore": 0.0,
                "subtractive_target_rms_ratio": 0.0,
                "subtractive_max_gain": 1.15,
                "hard_mask_threshold": 0.0,
                "hard_mask_power": 3.0,
                "hard_mask_residual_floor": 0.02,
            },
        ),
        VocalCandidateProfile(
            name="hybrid_balanced_clean",
            candidate_mode="hybrid_vocals",
            prior_score=0.06,
            intent="Balance tail preservation against backing bleed suppression.",
            tuning={
                "low_bs_weight": 0.75,
                "high_bs_weight": 0.25,
                "inactive_low_bs_weight": 0.92,
                "inactive_high_bs_weight": 0.78,
                "crossover_hz": 3000.0,
                "activity_abs_floor_ratio": 0.07,
                "activity_rel_floor": 0.32,
                "activity_smoothing_frames": 13,
                "vocal_sharpen_alpha": 0.0,
                "tail_gate_strength": 0.05,
                "tail_gate_floor_ratio": 0.07,
                "tail_gate_smoothing_frames": 15,
                "subtractive_strength": 0.32,
                "subtractive_power": 1.15,
                "subtractive_residual_floor": 0.20,
                "subtractive_active_restore": 0.22,
                "subtractive_target_rms_ratio": 0.96,
                "subtractive_max_gain": 1.15,
                "hard_mask_threshold": 0.0,
                "hard_mask_power": 3.0,
                "hard_mask_residual_floor": 0.02,
            },
        ),
        VocalCandidateProfile(
            name="ft_tail_safe",
            candidate_mode="ft_vocals",
            prior_score=0.02,
            intent="Use Demucs vocal continuity when BS-style extraction eats endings.",
            tuning={
                "low_bs_weight": 0.75,
                "high_bs_weight": 0.25,
                "inactive_low_bs_weight": 0.97,
                "inactive_high_bs_weight": 0.9,
                "crossover_hz": 3000.0,
                "activity_abs_floor_ratio": 0.08,
                "activity_rel_floor": 0.35,
                "activity_smoothing_frames": 11,
                "vocal_sharpen_alpha": 0.0,
                "tail_gate_strength": 0.03,
                "tail_gate_floor_ratio": 0.06,
                "tail_gate_smoothing_frames": 15,
                "subtractive_strength": 0.22,
                "subtractive_power": 1.1,
                "subtractive_residual_floor": 0.22,
                "subtractive_active_restore": 0.28,
                "subtractive_target_rms_ratio": 0.98,
                "subtractive_max_gain": 1.12,
                "hard_mask_threshold": 0.0,
                "hard_mask_power": 3.0,
                "hard_mask_residual_floor": 0.02,
            },
        ),
        VocalCandidateProfile(
            name="bs_artifact_clean",
            candidate_mode="bs_vocals",
            prior_score=0.0,
            intent="Use BS-RoFormer as a clean but stricter vocal isolation candidate.",
            tuning={
                "low_bs_weight": 0.75,
                "high_bs_weight": 0.25,
                "inactive_low_bs_weight": 0.97,
                "inactive_high_bs_weight": 0.9,
                "crossover_hz": 3000.0,
                "activity_abs_floor_ratio": 0.08,
                "activity_rel_floor": 0.35,
                "activity_smoothing_frames": 11,
                "vocal_sharpen_alpha": 0.0,
                "tail_gate_strength": 0.0,
                "tail_gate_floor_ratio": 0.08,
                "tail_gate_smoothing_frames": 11,
                "subtractive_strength": 0.0,
                "subtractive_power": 1.0,
                "subtractive_residual_floor": 0.12,
                "subtractive_active_restore": 0.0,
                "subtractive_target_rms_ratio": 0.0,
                "subtractive_max_gain": 1.35,
                "hard_mask_threshold": 0.0,
                "hard_mask_power": 3.0,
                "hard_mask_residual_floor": 0.02,
            },
        ),
    )


def default_anchor_suppression_profiles() -> tuple[AnchorSuppressionProfile, ...]:
    return (
        AnchorSuppressionProfile(
            name="ft_tail_safe_anchor_clean",
            source_profile_name="ft_tail_safe",
            anchor_profile_name="bs_artifact_clean",
            prior_score=0.34,
            intent="Keep the preferred FT vocal continuity while suppressing content not supported by the clean BS vocal anchor.",
            tuning={
                "support_threshold": 0.18,
                "mask_power": 1.35,
                "inactive_residual_floor": 0.06,
                "active_residual_floor": 0.40,
                "tail_residual_floor": 0.26,
                "anchor_activity_floor_ratio": 0.055,
                "anchor_activity_smoothing_frames": 15,
                "tail_memory_sec": 0.42,
                "high_band_start_hz": 6500.0,
                "high_band_extra_power": 1.25,
                "high_band_floor_scale": 0.72,
                "target_rms_ratio": 0.98,
                "max_gain": 1.08,
            },
        ),
        AnchorSuppressionProfile(
            name="ft_tail_safe_anchor_strict",
            source_profile_name="ft_tail_safe",
            anchor_profile_name="bs_artifact_clean",
            prior_score=0.20,
            intent="Stricter FT cleanup for tracks where uplifters, synths, or robot-like drop textures leak into the vocal stem.",
            tuning={
                "support_threshold": 0.24,
                "mask_power": 1.70,
                "inactive_residual_floor": 0.035,
                "active_residual_floor": 0.34,
                "tail_residual_floor": 0.20,
                "anchor_activity_floor_ratio": 0.065,
                "anchor_activity_smoothing_frames": 13,
                "tail_memory_sec": 0.34,
                "high_band_start_hz": 6000.0,
                "high_band_extra_power": 1.55,
                "high_band_floor_scale": 0.60,
                "target_rms_ratio": 0.95,
                "max_gain": 1.05,
            },
        ),
        AnchorSuppressionProfile(
            name="ir_residual_anchor_clean",
            source_profile_name="ir_residual_vocals",
            anchor_profile_name="bs_artifact_clean",
            prior_score=0.28,
            intent=(
                "Use the Instrumental Resurrection residual as a vocal-body candidate while suppressing "
                "content not supported by the clean BS vocal anchor."
            ),
            tuning={
                "support_threshold": 0.16,
                "mask_power": 1.45,
                "inactive_residual_floor": 0.045,
                "active_residual_floor": 0.38,
                "tail_residual_floor": 0.24,
                "anchor_activity_floor_ratio": 0.055,
                "anchor_activity_smoothing_frames": 15,
                "tail_memory_sec": 0.42,
                "high_band_start_hz": 6000.0,
                "high_band_extra_power": 1.45,
                "high_band_floor_scale": 0.62,
                "target_rms_ratio": 0.96,
                "max_gain": 1.06,
            },
        ),
    )


def default_vocal_body_repair_profiles() -> tuple[VocalBodyRepairProfile, ...]:
    return (
        VocalBodyRepairProfile(
            name="ft_anchor_clean_vocal_body_repair",
            base_profile_name="ft_tail_safe_anchor_clean",
            donor_profile_names=("bs_artifact_clean", "hybrid_tail_preserve", "hybrid_balanced_clean"),
            prior_score=0.50,
            intent="Repair audible vocal-body dips in the clean FT-anchor candidate using consensus vocal-core energy only.",
            tuning={
                "fill_low_hz": 140.0,
                "fill_high_hz": 4300.0,
                "fill_threshold": 0.18,
                "fill_strength": 0.42,
                "max_fill_ratio": 1.08,
                "donor_activity_floor_ratio": 0.07,
                "donor_activity_smoothing_frames": 11,
                "target_rms_ratio": 1.02,
                "max_gain": 1.06,
            },
        ),
        VocalBodyRepairProfile(
            name="ft_anchor_clean_phase_safe_drop_repair",
            base_profile_name="ft_tail_safe_anchor_clean",
            donor_profile_names=("bs_artifact_clean", "hybrid_tail_preserve", "hybrid_balanced_clean"),
            prior_score=0.62,
            intent=(
                "Repair crowded/drop vocal-body dips with donor consensus magnitude while preserving "
                "the clean base vocal phase to reduce phasey/flanger-like artifacts."
            ),
            tuning={
                "fill_low_hz": 130.0,
                "fill_high_hz": 4700.0,
                "fill_threshold": 0.14,
                "fill_strength": 0.54,
                "max_fill_ratio": 1.12,
                "donor_activity_floor_ratio": 0.06,
                "donor_activity_smoothing_frames": 15,
                "target_rms_ratio": 1.03,
                "max_gain": 1.08,
                "phase_strategy": "base",
            },
        ),
        VocalBodyRepairProfile(
            name="ir_residual_supported_body_repair",
            base_profile_name="ft_tail_safe_anchor_clean",
            donor_profile_names=("bs_artifact_clean", "ir_residual_anchor_clean", "hybrid_tail_preserve"),
            prior_score=0.66,
            intent=(
                "Use IR residual vocal evidence as an extra donor for crowded/drop vocal-body recovery, "
                "but preserve the clean base phase and keep BS support in the donor consensus."
            ),
            tuning={
                "fill_low_hz": 130.0,
                "fill_high_hz": 4700.0,
                "fill_threshold": 0.12,
                "fill_strength": 0.58,
                "max_fill_ratio": 1.10,
                "donor_activity_floor_ratio": 0.055,
                "donor_activity_smoothing_frames": 17,
                "target_rms_ratio": 1.025,
                "max_gain": 1.07,
                "phase_strategy": "base",
            },
        ),
    )


def default_bounded_thin_repair_body_profiles() -> tuple[VocalBodyRepairProfile, ...]:
    return (
        VocalBodyRepairProfile(
            name="ft_tail_safe_direct_body_presence_repair",
            base_profile_name="ft_tail_safe",
            donor_profile_names=("hybrid_tail_preserve", "hybrid_balanced_clean", "bs_artifact_clean"),
            prior_score=0.55,
            intent=(
                "Bounded thin/over-cleaned repair that restores vocal body and presence from consensus donors "
                "without requiring an anchor-clean or stricter debleed candidate."
            ),
            tuning={
                "fill_low_hz": 150.0,
                "fill_high_hz": 4300.0,
                "fill_threshold": 0.20,
                "fill_strength": 0.34,
                "max_fill_ratio": 1.06,
                "donor_activity_floor_ratio": 0.08,
                "donor_activity_smoothing_frames": 11,
                "target_rms_ratio": 1.01,
                "max_gain": 1.05,
                "phase_strategy": "base",
            },
        ),
    )


def default_segment_router_profiles() -> tuple[SegmentRouterProfile, ...]:
    return (
        SegmentRouterProfile(
            name="segment_adaptive_vocal_router",
            clean_profile_name="ft_tail_safe_anchor_clean",
            strict_profile_name="ft_tail_safe_anchor_strict",
            repair_profile_name="ft_anchor_clean_phase_safe_drop_repair",
            preserve_profile_name="hybrid_tail_preserve",
            anchor_profile_name="bs_artifact_clean",
            prior_score=0.78,
            intent=(
                "Route each time-frequency region between clean, strict, and body-preserving vocal "
                "candidates so busy/drop segments can be less aggressive while noisy high bands stay strict."
            ),
            tuning={
                "body_low_hz": 130.0,
                "body_high_hz": 4700.0,
                "high_band_start_hz": 6200.0,
                "high_band_end_hz": 15000.0,
                "body_deficit_threshold": 0.08,
                "body_repair_strength": 0.56,
                "preserve_tail_strength": 0.10,
                "preserve_max_ratio": 1.18,
                "max_body_fill_ratio": 1.16,
                "body_support_floor": 0.68,
                "body_support_power": 1.70,
                "unsupported_busy_damping": 0.78,
                "core_unsupported_tolerance": 1.12,
                "core_unsupported_damping": 0.92,
                "strict_high_tolerance": 1.22,
                "strict_high_strength": 0.72,
                "activity_floor_ratio": 0.055,
                "activity_smoothing_frames": 17,
                "raw_busy_percentile": 72.0,
                "raw_busy_boost": 0.0,
                "target_rms_ratio": 1.02,
                "max_gain": 1.06,
                "phase_strategy": "clean_base",
            },
        ),
        SegmentRouterProfile(
            name="busy_vocal_supported_router",
            clean_profile_name="ft_tail_safe_anchor_clean",
            strict_profile_name="ft_tail_safe_anchor_strict",
            repair_profile_name="ft_anchor_clean_phase_safe_drop_repair",
            preserve_profile_name="hybrid_tail_preserve",
            anchor_profile_name="bs_artifact_clean",
            prior_score=0.77,
            intent=(
                "Recover vocal body more assertively on dense drop frames when strict or anchor "
                "still carries partial speech evidence, while keeping unsupported synth/noise damped."
            ),
            tuning={
                "body_low_hz": 130.0,
                "body_high_hz": 4700.0,
                "high_band_start_hz": 6200.0,
                "high_band_end_hz": 15000.0,
                "body_deficit_threshold": 0.06,
                "body_repair_strength": 1.00,
                "preserve_tail_strength": 0.13,
                "preserve_max_ratio": 1.14,
                "max_body_fill_ratio": 1.22,
                "body_support_floor": 0.12,
                "body_support_power": 0.65,
                "unsupported_busy_damping": 0.82,
                "core_unsupported_tolerance": 1.06,
                "core_unsupported_damping": 0.95,
                "strict_high_tolerance": 1.16,
                "strict_high_strength": 0.78,
                "activity_floor_ratio": 0.050,
                "activity_smoothing_frames": 17,
                "raw_busy_percentile": 72.0,
                "raw_busy_boost": 0.18,
                "target_rms_ratio": 1.025,
                "max_gain": 1.08,
                "phase_strategy": "clean_base",
            },
        ),
        SegmentRouterProfile(
            name="ir_residual_supported_router",
            clean_profile_name="ft_tail_safe_anchor_clean",
            strict_profile_name="ft_tail_safe_anchor_strict",
            repair_profile_name="ir_residual_supported_body_repair",
            preserve_profile_name="hybrid_tail_preserve",
            anchor_profile_name="bs_artifact_clean",
            prior_score=0.79,
            intent=(
                "Route clean/strict vocals with an IR-residual-supported repair leg for dense sections "
                "where model vocals swallow syllables but unsupported synth/noise still needs damping."
            ),
            tuning={
                "body_low_hz": 130.0,
                "body_high_hz": 4700.0,
                "high_band_start_hz": 6200.0,
                "high_band_end_hz": 15000.0,
                "body_deficit_threshold": 0.055,
                "body_repair_strength": 0.82,
                "preserve_tail_strength": 0.12,
                "preserve_max_ratio": 1.14,
                "max_body_fill_ratio": 1.18,
                "body_support_floor": 0.22,
                "body_support_power": 0.82,
                "unsupported_busy_damping": 0.84,
                "core_unsupported_tolerance": 1.08,
                "core_unsupported_damping": 0.95,
                "strict_high_tolerance": 1.16,
                "strict_high_strength": 0.80,
                "activity_floor_ratio": 0.050,
                "activity_smoothing_frames": 17,
                "raw_busy_percentile": 72.0,
                "raw_busy_boost": 0.10,
                "target_rms_ratio": 1.025,
                "max_gain": 1.07,
                "phase_strategy": "clean_base",
            },
        ),
        SegmentRouterProfile(
            name="quality_consistent_ir_router",
            clean_profile_name="ft_tail_safe_anchor_clean",
            strict_profile_name="ft_tail_safe_anchor_strict",
            repair_profile_name="ir_residual_supported_body_repair",
            preserve_profile_name="hybrid_tail_preserve",
            anchor_profile_name="bs_artifact_clean",
            prior_score=0.805,
            intent=(
                "Use the IR-residual repair leg more conservatively than the dense-drop router so "
                "vocal body is recovered only when strict/anchor support is stable across the track."
            ),
            tuning={
                "body_low_hz": 130.0,
                "body_high_hz": 4700.0,
                "high_band_start_hz": 6200.0,
                "high_band_end_hz": 15000.0,
                "body_deficit_threshold": 0.065,
                "body_repair_strength": 0.62,
                "preserve_tail_strength": 0.10,
                "preserve_max_ratio": 1.12,
                "max_body_fill_ratio": 1.12,
                "body_support_floor": 0.38,
                "body_support_power": 1.10,
                "unsupported_busy_damping": 0.88,
                "core_unsupported_tolerance": 1.05,
                "core_unsupported_damping": 0.97,
                "strict_high_tolerance": 1.12,
                "strict_high_strength": 0.84,
                "activity_floor_ratio": 0.052,
                "activity_smoothing_frames": 19,
                "raw_busy_percentile": 72.0,
                "raw_busy_boost": 0.0,
                "target_rms_ratio": 1.015,
                "max_gain": 1.05,
                "phase_strategy": "clean_base",
            },
        ),
    )

def select_profiles(
    requested_profiles: list[str] | None,
    available_profiles: tuple[VocalCandidateProfile, ...],
) -> list[VocalCandidateProfile]:
    if not requested_profiles:
        return list(available_profiles)

    by_name = {profile.name: profile for profile in available_profiles}
    unknown_profiles = sorted(set(requested_profiles) - set(by_name))
    if unknown_profiles:
        raise ValueError(f"Unknown vocal profile(s): {', '.join(unknown_profiles)}")
    return [by_name[name] for name in requested_profiles]


def profiles_require_bs_roformer_runtime(profiles: Iterable[VocalCandidateProfile]) -> bool:
    bs_roformer_candidate_modes = {
        "bs_vocals",
        "hybrid_vocals",
    }
    return any(profile.candidate_mode in bs_roformer_candidate_modes for profile in profiles)
