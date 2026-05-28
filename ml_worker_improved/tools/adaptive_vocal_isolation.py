from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from adaptive_policy import decision_from_evidence  # noqa: E402

import librosa
import numpy as np
import soundfile as sf


DEFAULT_BS_ROFORMER_RUNTIME_REPO = Path("/tmp/msst_zfturbo")
DEFAULT_BS_ROFORMER_RUNTIME_WEIGHT_DIR = Path("/tmp/msst_zfturbo_weights")
DEFAULT_BS_ROFORMER_VENDOR_REPO = Path("/app/vendor/music_source_separation_training")
DEFAULT_BS_ROFORMER_VENDOR_CONFIG = Path("/app/vendor/bs_roformer_official/config_bs_roformer_384_8_2_485100.yaml")
DEFAULT_BS_ROFORMER_VENDOR_CKPT = Path("/app/vendor/bs_roformer_official/model_bs_roformer_ep_17_sdr_9.6568.ckpt")
RESERVED_EXTERNAL_CANDIDATE_NAMES = frozenset({"ir_residual_vocals"})
SEGMENT_NEAR_SILENT_RMS_DBFS = -90.0


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


class StageProfiler:
    def __init__(self, *, emit_heartbeat: bool = True, event_log_path: Path | None = None) -> None:
        self.emit_heartbeat = bool(emit_heartbeat)
        self.event_log_path = Path(event_log_path) if event_log_path is not None else None
        self._started_at = time.perf_counter()
        self._stages: dict[str, dict[str, Any]] = {}
        self._events: list[dict[str, Any]] = []
        if self.event_log_path is not None:
            self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.event_log_path.write_text("", encoding="utf-8")

    def start(self, stage: str, **metadata: Any) -> float:
        event = self._event(stage=stage, status="start", elapsed_sec=0.0, metadata=metadata)
        self._events.append(event)
        self._emit(event)
        return time.perf_counter()

    def finish(self, stage: str, started_at: float, **metadata: Any) -> float:
        elapsed_sec = max(0.0, time.perf_counter() - float(started_at))
        stage_entry = self._stages.setdefault(
            stage,
            {"call_count": 0, "elapsed_sec": 0.0, "max_elapsed_sec": 0.0},
        )
        stage_entry["call_count"] = int(stage_entry["call_count"]) + 1
        stage_entry["elapsed_sec"] = round(float(stage_entry["elapsed_sec"]) + elapsed_sec, 6)
        stage_entry["max_elapsed_sec"] = round(max(float(stage_entry["max_elapsed_sec"]), elapsed_sec), 6)
        event = self._event(stage=stage, status="finish", elapsed_sec=elapsed_sec, metadata=metadata)
        self._events.append(event)
        self._emit(event)
        return elapsed_sec

    @contextmanager
    def measure(self, stage: str, **metadata: Any):
        started_at = self.start(stage, **metadata)
        try:
            yield
        finally:
            self.finish(stage, started_at, **metadata)

    def mark(self, stage: str, *, status: str = "progress", **metadata: Any) -> dict[str, Any]:
        event = self._event(stage=stage, status=status, elapsed_sec=0.0, metadata=metadata)
        self._events.append(event)
        self._emit(event)
        return event

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mode": "adaptive_vocal_stage_profiler",
            "total_elapsed_sec": round(max(0.0, time.perf_counter() - self._started_at), 6),
            "stages": self._stages,
            "events": self._events,
        }

    def _event(self, *, stage: str, status: str, elapsed_sec: float, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "stage": str(stage),
            "status": str(status),
            "elapsed_sec": round(float(elapsed_sec), 6),
            "since_start_sec": round(max(0.0, time.perf_counter() - self._started_at), 6),
            "metadata": {str(key): value for key, value in metadata.items() if value is not None},
        }

    def _emit(self, event: dict[str, Any]) -> None:
        if self.event_log_path is not None:
            with self.event_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"type": "adaptive_vocal_stage", **event}, ensure_ascii=False) + "\n")
        if not self.emit_heartbeat:
            return
        print(json.dumps({"type": "adaptive_vocal_stage", **event}, ensure_ascii=False), file=sys.stderr, flush=True)


class AudioAnalysisCache:
    def __init__(self) -> None:
        self._audio_cache: dict[tuple[Any, ...], tuple[np.ndarray, int]] = {}
        self._source_window_cache: dict[tuple[Any, ...], tuple[np.ndarray, int]] = {}
        self._audio_reads = 0
        self._audio_cache_hits = 0
        self._source_window_reads = 0
        self._source_window_cache_hits = 0

    def read_audio(self, path: Path) -> tuple[np.ndarray, int]:
        key = self._file_key(path)
        cached = self._audio_cache.get(key)
        if cached is not None:
            self._audio_cache_hits += 1
            return cached
        audio, sample_rate = sf.read(str(path), always_2d=True)
        value = (np.asarray(audio, dtype=np.float32), int(sample_rate))
        self._audio_cache[key] = value
        self._audio_reads += 1
        return value

    def read_source_window(self, path: Path, *, start_sec: float, end_sec: float | None) -> tuple[np.ndarray, int]:
        key = (*self._file_key(path), round(float(start_sec), 6), None if end_sec is None else round(float(end_sec), 6))
        cached = self._source_window_cache.get(key)
        if cached is not None:
            self._source_window_cache_hits += 1
            return cached
        value = _read_source_window(path, start_sec=start_sec, end_sec=end_sec)
        value = (np.asarray(value[0], dtype=np.float32), int(value[1]))
        self._source_window_cache[key] = value
        self._source_window_reads += 1
        return value

    def stats(self) -> dict[str, int]:
        return {
            "audio_reads": int(self._audio_reads),
            "audio_cache_hits": int(self._audio_cache_hits),
            "source_window_reads": int(self._source_window_reads),
            "source_window_cache_hits": int(self._source_window_cache_hits),
            "audio_cache_entries": int(len(self._audio_cache)),
            "source_window_cache_entries": int(len(self._source_window_cache)),
        }

    @staticmethod
    def _file_key(path: Path) -> tuple[str, int, int]:
        resolved_path = Path(path)
        stat = resolved_path.stat()
        return (str(resolved_path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))


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


def _ensure_link_or_copy(source_path: Path, target_path: Path) -> None:
    if target_path.exists():
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source_path, target_path)
    except OSError:
        if source_path.is_dir():
            shutil.copytree(source_path, target_path)
        else:
            shutil.copyfile(source_path, target_path)


def ensure_bs_roformer_runtime() -> None:
    _ensure_link_or_copy(DEFAULT_BS_ROFORMER_VENDOR_REPO, DEFAULT_BS_ROFORMER_RUNTIME_REPO)
    DEFAULT_BS_ROFORMER_RUNTIME_WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_link_or_copy(
        DEFAULT_BS_ROFORMER_VENDOR_CONFIG,
        DEFAULT_BS_ROFORMER_RUNTIME_WEIGHT_DIR / "bs_roformer_vocals.yaml",
    )
    _ensure_link_or_copy(
        DEFAULT_BS_ROFORMER_VENDOR_CKPT,
        DEFAULT_BS_ROFORMER_RUNTIME_WEIGHT_DIR / "bs_roformer_vocals.ckpt",
    )


def audio_duration_sec(audio_path: Path) -> float:
    info = sf.info(str(audio_path))
    return float(info.frames) / float(info.samplerate)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return loaded


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _feedback_record_matches_context(
    record: dict[str, Any],
    *,
    source_audio_sha256: str | None,
    start_sec: float | None,
    end_sec: float | None,
) -> bool:
    requires_exact_context = (
        record.get("listener_verified") is True
        or record.get("human_verified") is True
    )
    record_source_sha256 = record.get("source_audio_sha256")
    if isinstance(record_source_sha256, str) and record_source_sha256.strip():
        if source_audio_sha256 is None:
            return False
        if record_source_sha256.strip().lower() != source_audio_sha256.strip().lower():
            return False
    elif requires_exact_context:
        return False

    window_sec = record.get("window_sec")
    if isinstance(window_sec, dict):
        record_start = window_sec.get("start")
        record_end = window_sec.get("end")
        if requires_exact_context and not (
            isinstance(record_start, (int, float))
            and isinstance(record_end, (int, float))
        ):
            return False
        if isinstance(record_start, (int, float)):
            if start_sec is None:
                return False
            if abs(float(record_start) - float(start_sec)) > 1e-3:
                return False
        if isinstance(record_end, (int, float)):
            if end_sec is None:
                return False
            if abs(float(record_end) - float(end_sec)) > 1e-3:
                return False
    elif requires_exact_context:
        return False
    return True


def _feedback_record_delta(record: dict[str, Any]) -> float:
    delta = 0.0
    rating = record.get("rating")
    if rating in {"accept", "good", "better"}:
        delta += 0.24
    elif rating in {"reject", "bad", "worse"}:
        delta -= 0.32

    numeric_score = record.get("score")
    if isinstance(numeric_score, (int, float)):
        delta += max(-1.0, min(1.0, float(numeric_score))) * 0.18
    if record.get("swallowed_endings") is True:
        delta -= 0.22
    if record.get("backing_bleed") is True:
        delta -= 0.18
    if record.get("backing_vocal_loss") is True:
        delta -= 0.18
    if record.get("watery_artifacts") is True:
        delta -= 0.14
    if record.get("non_vocal_bleed") is True:
        delta -= 0.24
    if record.get("uplifter_bleed") is True:
        delta -= 0.20
    if record.get("synth_bleed") is True:
        delta -= 0.20
    if record.get("robotic_drop_bleed") is True:
        delta -= 0.20

    if record.get("listener_verified") is True or record.get("human_verified") is True:
        confidence = record.get("confidence", 1.0)
        confidence_value = max(0.0, min(1.0, float(confidence) if isinstance(confidence, (int, float)) else 1.0))
        if rating in {"accept", "good", "better"}:
            delta += 6.0 * confidence_value
        elif rating in {"reject", "bad", "worse"}:
            delta -= 6.0 * confidence_value
    return delta


def _feedback_record_is_holdout(record: dict[str, Any]) -> bool:
    split = record.get("feedback_split") or record.get("split")
    return bool(record.get("holdout") is True or (isinstance(split, str) and split.lower() == "holdout"))


def load_feedback_adjustments(
    feedback_store_path: Path | None,
    *,
    source_audio_sha256: str | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> dict[str, float]:
    if feedback_store_path is None or not feedback_store_path.exists():
        return {}

    adjustments: dict[str, float] = {}
    for line_number, raw_line in enumerate(
        feedback_store_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSONL feedback record at {feedback_store_path}:{line_number}"
            ) from exc

        profile_name = record.get("profile") or record.get("selected_profile")
        if not isinstance(profile_name, str) or not profile_name:
            continue
        if _feedback_record_is_holdout(record):
            continue
        if not _feedback_record_matches_context(
            record,
            source_audio_sha256=source_audio_sha256,
            start_sec=start_sec,
            end_sec=end_sec,
        ):
            continue

        delta = _feedback_record_delta(record)

        adjustments[profile_name] = adjustments.get(profile_name, 0.0) + delta

    return adjustments


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


def build_vocal_export_command(
    *,
    python_executable: str,
    export_script: Path,
    source_audio_path: Path,
    output_dir: Path,
    start_sec: float,
    end_sec: float,
    device: str,
    profile: VocalCandidateProfile,
) -> list[str]:
    command = [
        python_executable,
        str(export_script),
        str(source_audio_path),
        "--candidate-mode",
        profile.candidate_mode,
        "--output-dir",
        str(output_dir),
        "--start-sec",
        str(start_sec),
        "--end-sec",
        str(end_sec),
        "--device",
        str(device),
    ]
    for key, value in profile.tuning.items():
        command.extend([f"--{key.replace('_', '-')}", str(value)])
    return command


def _run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def resolve_vocals_path(manifest: dict[str, Any]) -> Path:
    stems = manifest.get("stems")
    if isinstance(stems, dict):
        vocals = stems.get("vocals")
        if isinstance(vocals, dict) and vocals.get("path"):
            return Path(str(vocals["path"]))

    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, dict):
        for key in ("vocals_flac", "output_flac"):
            value = artifacts.get(key)
            if value:
                return Path(str(value))
    raise ValueError("Candidate manifest does not expose a vocals artifact path")


def _safe_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float32))))


def rms_dbfs(audio: np.ndarray) -> float:
    return 20.0 * math.log10(max(_safe_rms(audio), 1e-10))


def _mono(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        return array
    return array.mean(axis=1).astype(np.float32)


def _match_audio(raw_audio: np.ndarray, raw_sr: int, vocals_audio: np.ndarray, vocals_sr: int) -> tuple[np.ndarray, np.ndarray, int]:
    raw = np.asarray(raw_audio, dtype=np.float32)
    vocals = np.asarray(vocals_audio, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw[:, np.newaxis]
    if vocals.ndim == 1:
        vocals = vocals[:, np.newaxis]
    if raw_sr != vocals_sr:
        raw_channels = [
            librosa.resample(raw[:, channel_index], orig_sr=raw_sr, target_sr=vocals_sr)
            for channel_index in range(raw.shape[1])
        ]
        raw = np.stack(raw_channels, axis=1).astype(np.float32)
    frame_count = min(raw.shape[0], vocals.shape[0])
    return raw[:frame_count], vocals[:frame_count], int(vocals_sr)


def _limit_peak(audio: np.ndarray, *, peak: float = 0.999) -> np.ndarray:
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs <= float(peak):
        return audio.astype(np.float32)
    return (audio * (float(peak) / max_abs)).astype(np.float32)


def _audio_summary(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    channels = 1 if audio.ndim == 1 else int(audio.shape[1])
    frames = int(audio.shape[0])
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return {
        "channels": channels,
        "frames": frames,
        "duration_sec": round(frames / float(sample_rate), 3),
        "rms": round(_safe_rms(audio), 6),
        "peak": round(peak, 6),
    }


def _read_source_window(source_audio_path: Path, *, start_sec: float, end_sec: float | None) -> tuple[np.ndarray, int]:
    with sf.SoundFile(str(source_audio_path)) as handle:
        sample_rate = int(handle.samplerate)
        start_frame = max(0, int(round(float(start_sec) * sample_rate)))
        end_frame = handle.frames if end_sec is None else max(start_frame, int(round(float(end_sec) * sample_rate)))
        handle.seek(start_frame)
        frame_count = max(0, min(handle.frames, end_frame) - start_frame)
        audio = handle.read(frames=frame_count, dtype="float32", always_2d=True)
    return audio.astype(np.float32), sample_rate


def _float_values_match(left: float | int | None, right: float | int | None, *, tolerance: float = 1e-3) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= float(tolerance)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_is_within(child_path: Path, parent_path: Path) -> bool:
    try:
        child_path.resolve().relative_to(parent_path.resolve())
    except ValueError:
        return False
    return True


def _dependency_fingerprint(record: VocalCandidateRecord) -> dict[str, str]:
    return {
        "manifest_path": str(record.manifest_path),
        "manifest_sha256": _sha256_file(record.manifest_path),
        "vocals_path": str(record.vocals_path),
        "vocals_sha256": _sha256_file(record.vocals_path),
        "candidate_mode": str(record.manifest.get("pipeline", {}).get("candidate_mode")),
    }


def _candidate_manifest_reuse_rejection_reason(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    profile: VocalCandidateProfile | AnchorSuppressionProfile | VocalBodyRepairProfile | SegmentRouterProfile,
    source_audio_path: Path,
    start_sec: float,
    end_sec: float | None,
    expected_dependency_fingerprints: dict[str, dict[str, str]] | None = None,
) -> str | None:
    if str(manifest.get("audio_path")) != str(source_audio_path):
        return "source_audio_path_mismatch"
    source_audio_sha256 = manifest.get("source_audio_sha256")
    if not isinstance(source_audio_sha256, str) or not source_audio_sha256.strip():
        return "missing_source_audio_sha256"
    if _sha256_file(source_audio_path) != source_audio_sha256.strip().lower():
        return "source_audio_sha256_mismatch"

    try:
        raw_vocals_path = resolve_vocals_path(manifest)
    except ValueError:
        return "missing_vocals_path"
    vocals_path = raw_vocals_path if raw_vocals_path.is_absolute() else manifest_path.parent / raw_vocals_path
    if not vocals_path.exists():
        return "vocals_path_missing"
    if not _path_is_within(vocals_path, manifest_path.parent):
        return "vocals_path_outside_candidate_dir"

    preview_window = manifest.get("preview_window_sec")
    if not isinstance(preview_window, dict):
        return "missing_preview_window"
    if not _float_values_match(preview_window.get("start"), start_sec):
        return "preview_window_start_mismatch"
    if not _float_values_match(preview_window.get("end"), end_sec):
        return "preview_window_end_mismatch"

    manifest_tuning = manifest.get("tuning")
    if not isinstance(manifest_tuning, dict):
        return "missing_profile_tuning"
    for tuning_name, tuning_value in profile.tuning.items():
        if manifest_tuning.get(tuning_name) != tuning_value:
            return "profile_tuning_mismatch"
    if isinstance(profile, VocalCandidateProfile) and manifest.get("pipeline", {}).get("candidate_mode") != profile.candidate_mode:
        return "candidate_mode_mismatch"
    if isinstance(profile, SegmentRouterProfile) and manifest.get("pipeline", {}).get("candidate_mode") != "segment_adaptive_routed_vocals":
        return "candidate_mode_mismatch"
    if isinstance(profile, VocalBodyRepairProfile) and manifest.get("pipeline", {}).get("candidate_mode") != "vocal_body_repaired_vocals":
        return "candidate_mode_mismatch"
    if isinstance(profile, AnchorSuppressionProfile) and manifest.get("pipeline", {}).get("candidate_mode") != "anchor_suppressed_vocals":
        return "candidate_mode_mismatch"
    if expected_dependency_fingerprints is not None:
        pipeline = manifest.get("pipeline")
        if not isinstance(pipeline, dict):
            return "missing_pipeline"
        if pipeline.get("dependency_fingerprints") != expected_dependency_fingerprints:
            return "dependency_fingerprint_mismatch"
    return None


def load_reusable_candidate_manifest(
    *,
    manifest_path: Path,
    profile: VocalCandidateProfile | AnchorSuppressionProfile | VocalBodyRepairProfile | SegmentRouterProfile,
    source_audio_path: Path,
    start_sec: float,
    end_sec: float | None,
    reuse_existing_candidates: bool,
    reuse_rejected_profiles: list[dict[str, str]],
    expected_dependency_fingerprints: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any] | None:
    if not (bool(reuse_existing_candidates) and manifest_path.exists()):
        return None
    manifest = load_json(manifest_path)
    rejection_reason = _candidate_manifest_reuse_rejection_reason(
        manifest_path=manifest_path,
        manifest=manifest,
        profile=profile,
        source_audio_path=source_audio_path,
        start_sec=start_sec,
        end_sec=end_sec,
        expected_dependency_fingerprints=expected_dependency_fingerprints,
    )
    if rejection_reason is None:
        return manifest
    reuse_rejected_profiles.append(
        {
            "name": profile.name,
            "reason": rejection_reason,
            "manifest_path": str(manifest_path),
        }
    )
    return None


def _validate_candidate_name(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        raise argparse.ArgumentTypeError("external candidate name must not be empty")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if any(character not in allowed for character in normalized):
        raise argparse.ArgumentTypeError(
            "external candidate name may only contain letters, digits, '_' and '-'"
        )
    return normalized


def parse_external_vocal_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("external vocal candidate must use name=/path/to/vocals.flac")
    raw_name, raw_path = value.split("=", 1)
    name = _validate_candidate_name(raw_name)
    if name in RESERVED_EXTERNAL_CANDIDATE_NAMES:
        raise argparse.ArgumentTypeError(f"external vocal candidate name is reserved: {name}")
    path = Path(raw_path.strip())
    if not str(path):
        raise argparse.ArgumentTypeError("external vocal candidate path must not be empty")
    return name, path


def _external_candidate_profile(*, name: str, candidate_mode: str, source_path: Path, prior_score: float, intent: str) -> VocalCandidateProfile:
    return VocalCandidateProfile(
        name=name,
        candidate_mode=candidate_mode,
        prior_score=prior_score,
        intent=intent,
        tuning={
            "source_path": str(source_path),
            "source_contract": "proxy_candidate_not_ground_truth",
        },
    )


def _validate_external_candidate_coverage(
    *,
    profile_name: str,
    source_audio: np.ndarray,
    candidate_audio: np.ndarray,
    sample_rate: int,
    candidate_sample_rate: int,
    tolerance_sec: float = 0.25,
) -> None:
    source_frames = int(source_audio.shape[0])
    candidate_frames = int(candidate_audio.shape[0])
    source_duration_sec = source_frames / float(sample_rate)
    candidate_duration_sec = candidate_frames / float(candidate_sample_rate)
    if candidate_duration_sec + float(tolerance_sec) < source_duration_sec:
        raise ValueError(
            f"External vocal candidate {profile_name!r} is shorter than the requested analysis window: "
            f"candidate_duration_sec={candidate_duration_sec:.3f}, "
            f"required_duration_sec={source_duration_sec:.3f}. "
            "Use a full-window candidate or pass a matching --end-sec for diagnostic segment analysis."
        )


def build_external_vocal_candidate(
    *,
    profile: VocalCandidateProfile,
    source_audio_path: Path,
    candidate_audio_path: Path,
    output_dir: Path,
    start_sec: float,
    end_sec: float | None,
) -> dict[str, Any]:
    raw_audio, raw_sr = _read_source_window(source_audio_path, start_sec=start_sec, end_sec=end_sec)
    candidate_audio, candidate_sr = _read_source_window(candidate_audio_path, start_sec=start_sec, end_sec=end_sec)
    _validate_external_candidate_coverage(
        profile_name=profile.name,
        source_audio=raw_audio,
        candidate_audio=candidate_audio,
        sample_rate=int(raw_sr),
        candidate_sample_rate=int(candidate_sr),
    )
    raw_audio, candidate_audio, sample_rate = _match_audio(
        raw_audio,
        int(raw_sr),
        candidate_audio,
        int(candidate_sr),
    )
    candidate_audio = _limit_peak(candidate_audio)
    output_vocals_path = output_dir / "vocals.flac"
    output_vocals_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_vocals_path), candidate_audio, int(sample_rate), subtype="PCM_24")

    manifest = {
        "schema_version": 1,
        "audio_path": str(source_audio_path),
        "source_audio_sha256": _sha256_file(source_audio_path),
        "sample_rate": int(sample_rate),
        "export_subtype": "PCM_24",
        "preview_window_sec": {"start": float(start_sec), "end": None if end_sec is None else float(end_sec)},
        "pipeline": {
            "mode": "external_vocal_candidate",
            "candidate_mode": profile.candidate_mode,
            "source_candidate_path": str(candidate_audio_path),
            "ground_truth_contract": {
                "verified": False,
                "reason": "External vocal candidates are model/proxy inputs, not isolated ground truth.",
            },
        },
        "tuning": profile.tuning,
        "stems": {
            "vocals": {
                "path": str(output_vocals_path),
                "source_candidate_path": str(candidate_audio_path),
                **_audio_summary(candidate_audio, int(sample_rate)),
            }
        },
    }
    manifest_path = output_dir / "stem_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_ir_residual_vocal_candidate(
    *,
    profile: VocalCandidateProfile,
    source_audio_path: Path,
    instrumental_minus_path: Path,
    output_dir: Path,
    start_sec: float,
    end_sec: float | None,
) -> dict[str, Any]:
    raw_audio, raw_sr = _read_source_window(source_audio_path, start_sec=start_sec, end_sec=end_sec)
    instrumental_audio, instrumental_sr = _read_source_window(
        instrumental_minus_path,
        start_sec=start_sec,
        end_sec=end_sec,
    )
    _validate_external_candidate_coverage(
        profile_name=profile.name,
        source_audio=raw_audio,
        candidate_audio=instrumental_audio,
        sample_rate=int(raw_sr),
        candidate_sample_rate=int(instrumental_sr),
    )
    raw_audio, instrumental_audio, sample_rate = _match_audio(
        raw_audio,
        int(raw_sr),
        instrumental_audio,
        int(instrumental_sr),
    )
    instrumental_gain = float(profile.tuning.get("instrumental_gain", 1.0))
    residual_audio = raw_audio - instrumental_audio * instrumental_gain
    residual_audio = _limit_peak(residual_audio)
    output_vocals_path = output_dir / "vocals.flac"
    output_vocals_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_vocals_path), residual_audio, int(sample_rate), subtype="PCM_24")

    reconstruction = residual_audio + instrumental_audio * instrumental_gain
    reconstruction_error = raw_audio - reconstruction
    manifest = {
        "schema_version": 1,
        "audio_path": str(source_audio_path),
        "source_audio_sha256": _sha256_file(source_audio_path),
        "sample_rate": int(sample_rate),
        "export_subtype": "PCM_24",
        "preview_window_sec": {"start": float(start_sec), "end": None if end_sec is None else float(end_sec)},
        "pipeline": {
            "mode": "ir_residual_vocal_candidate",
            "candidate_mode": profile.candidate_mode,
            "instrumental_minus_path": str(instrumental_minus_path),
            "vocal_postprocess": "raw_minus_instrumental_resurrection_v1",
            "ground_truth_contract": {
                "verified": False,
                "reason": "IR residual is raw minus model instrumental; it is a proxy vocal candidate, not an oracle vocal stem.",
            },
        },
        "tuning": profile.tuning,
        "residual_contract": {
            "formula": "source_window - instrumental_minus_window * instrumental_gain",
            "instrumental_gain": instrumental_gain,
            "reconstruction_error_rms_dbfs": round(rms_dbfs(reconstruction_error), 6),
        },
        "stems": {
            "vocals": {
                "path": str(output_vocals_path),
                "instrumental_minus_path": str(instrumental_minus_path),
                **_audio_summary(residual_audio, int(sample_rate)),
            }
        },
    }
    manifest_path = output_dir / "stem_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _smooth_frame_activity(activity: np.ndarray, smoothing_frames: int) -> np.ndarray:
    if int(smoothing_frames) <= 1 or activity.size == 0:
        return np.clip(activity, 0.0, 1.0).astype(np.float32)
    kernel = np.ones(int(smoothing_frames), dtype=np.float32) / float(smoothing_frames)
    return np.clip(np.convolve(activity, kernel, mode="same"), 0.0, 1.0).astype(np.float32)


def _causal_tail_activity(activity: np.ndarray, *, sample_rate: int, hop_length: int, tail_memory_sec: float) -> np.ndarray:
    if activity.size == 0:
        return np.zeros(0, dtype=np.float32)
    if float(tail_memory_sec) <= 1e-6:
        return np.clip(activity, 0.0, 1.0).astype(np.float32)
    decay = math.exp(-float(hop_length) / max(float(sample_rate) * float(tail_memory_sec), 1e-6))
    tail = np.zeros_like(activity, dtype=np.float32)
    carried = 0.0
    for frame_index, value in enumerate(np.clip(activity, 0.0, 1.0)):
        carried = max(float(value), carried * decay)
        tail[frame_index] = carried
    return np.clip(tail, 0.0, 1.0).astype(np.float32)


def _anchor_activity_from_stft(
    anchor_magnitude: np.ndarray,
    *,
    floor_ratio: float,
    smoothing_frames: int,
) -> np.ndarray:
    frame_energy = np.mean(anchor_magnitude, axis=0).astype(np.float32)
    reference_energy = float(np.percentile(frame_energy, 95.0)) if frame_energy.size else 0.0
    activity = np.clip(
        frame_energy / max(reference_energy * float(floor_ratio), 1e-8),
        0.0,
        1.0,
    ).astype(np.float32)
    return _smooth_frame_activity(activity, smoothing_frames)


def apply_anchor_guided_leakage_suppression(
    *,
    source_vocals_path: Path,
    anchor_vocals_path: Path,
    output_vocals_path: Path,
    tuning: dict[str, float | int],
) -> dict[str, Any]:
    source_audio, source_sr = sf.read(str(source_vocals_path), always_2d=True, dtype="float32")
    anchor_audio, anchor_sr = sf.read(str(anchor_vocals_path), always_2d=True, dtype="float32")
    source_audio, anchor_audio, sample_rate = _match_audio(
        source_audio,
        int(source_sr),
        anchor_audio,
        int(anchor_sr),
    )

    n_fft = 2048
    hop_length = 512
    support_threshold = float(tuning["support_threshold"])
    mask_power = float(tuning["mask_power"])
    inactive_floor = float(tuning["inactive_residual_floor"])
    active_floor = float(tuning["active_residual_floor"])
    tail_floor = float(tuning["tail_residual_floor"])
    activity_floor_ratio = float(tuning["anchor_activity_floor_ratio"])
    activity_smoothing_frames = int(tuning["anchor_activity_smoothing_frames"])
    tail_memory_sec = float(tuning["tail_memory_sec"])
    high_band_start_hz = float(tuning["high_band_start_hz"])
    high_band_extra_power = float(tuning["high_band_extra_power"])
    high_band_floor_scale = float(tuning["high_band_floor_scale"])
    target_rms_ratio = float(tuning["target_rms_ratio"])
    max_gain = float(tuning["max_gain"])

    source_rms = _safe_rms(source_audio)
    cleaned_channels: list[np.ndarray] = []
    mask_means: list[float] = []
    activity_means: list[float] = []
    tail_activity_means: list[float] = []
    freq_bins = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft).astype(np.float32)
    high_band = freq_bins >= high_band_start_hz
    floor_scale = np.ones_like(freq_bins, dtype=np.float32)
    floor_scale[high_band] = float(high_band_floor_scale)

    for channel_index in range(source_audio.shape[1]):
        source_stft = librosa.stft(source_audio[:, channel_index], n_fft=n_fft, hop_length=hop_length)
        anchor_stft = librosa.stft(anchor_audio[:, channel_index], n_fft=n_fft, hop_length=hop_length)
        source_mag = np.abs(source_stft).astype(np.float32)
        anchor_mag = np.abs(anchor_stft).astype(np.float32)

        anchor_activity = _anchor_activity_from_stft(
            anchor_mag,
            floor_ratio=activity_floor_ratio,
            smoothing_frames=activity_smoothing_frames,
        )
        tail_activity = _causal_tail_activity(
            anchor_activity,
            sample_rate=sample_rate,
            hop_length=hop_length,
            tail_memory_sec=tail_memory_sec,
        )
        dynamic_floor = np.maximum(
            inactive_floor,
            np.maximum(active_floor * anchor_activity, tail_floor * tail_activity),
        ).astype(np.float32)
        support_ratio = anchor_mag / (source_mag + 1e-8)
        support_mask = np.clip(
            (support_ratio - support_threshold) / max(1e-6, 1.0 - support_threshold),
            0.0,
            1.0,
        ).astype(np.float32)
        if mask_power != 1.0:
            support_mask = np.power(support_mask, mask_power, dtype=np.float32)
        if high_band_extra_power != 1.0 and np.any(high_band):
            support_mask[high_band, :] = np.power(
                support_mask[high_band, :],
                high_band_extra_power,
                dtype=np.float32,
            )

        dynamic_floor_by_bin = (dynamic_floor[np.newaxis, :] * floor_scale[:, np.newaxis]).astype(np.float32)
        final_mask = np.maximum(support_mask, dynamic_floor_by_bin).astype(np.float32)
        cleaned_channel = librosa.istft(
            source_stft * final_mask,
            hop_length=hop_length,
            length=source_audio.shape[0],
        ).astype(np.float32)
        cleaned_channels.append(cleaned_channel)
        mask_means.append(float(np.mean(final_mask)) if final_mask.size else 0.0)
        activity_means.append(float(np.mean(anchor_activity)) if anchor_activity.size else 0.0)
        tail_activity_means.append(float(np.mean(tail_activity)) if tail_activity.size else 0.0)

    cleaned_audio = np.stack(cleaned_channels, axis=1).astype(np.float32)
    gain_applied = 1.0
    cleaned_rms = _safe_rms(cleaned_audio)
    if target_rms_ratio > 1e-6 and source_rms > 1e-8 and cleaned_rms > 1e-8:
        target_rms = source_rms * target_rms_ratio
        if target_rms > cleaned_rms:
            gain_applied = min(max_gain, target_rms / cleaned_rms)
            cleaned_audio = (cleaned_audio * gain_applied).astype(np.float32)
    cleaned_audio = _limit_peak(cleaned_audio)

    output_vocals_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_vocals_path), cleaned_audio, int(sample_rate), subtype="PCM_24")
    return {
        "sample_rate": int(sample_rate),
        "mask_mean": round(float(np.mean(mask_means)) if mask_means else 0.0, 6),
        "anchor_activity_mean": round(float(np.mean(activity_means)) if activity_means else 0.0, 6),
        "tail_activity_mean": round(float(np.mean(tail_activity_means)) if tail_activity_means else 0.0, 6),
        "source_vocals_rms": round(source_rms, 6),
        "cleaned_vocals_rms": round(_safe_rms(cleaned_audio), 6),
        "gain_applied": round(float(gain_applied), 6),
        **_audio_summary(cleaned_audio, int(sample_rate)),
    }


def _read_and_match_donor_audio(base_audio: np.ndarray, base_sr: int, donor_path: Path) -> np.ndarray:
    donor_audio, donor_sr = sf.read(str(donor_path), always_2d=True, dtype="float32")
    _, matched_donor, _ = _match_audio(base_audio, int(base_sr), donor_audio, int(donor_sr))
    return matched_donor.astype(np.float32)


def _donor_activity_from_consensus(
    consensus_magnitude: np.ndarray,
    frequency_mask: np.ndarray,
    *,
    floor_ratio: float,
    smoothing_frames: int,
) -> np.ndarray:
    if consensus_magnitude.size == 0 or not np.any(frequency_mask):
        return np.zeros(consensus_magnitude.shape[1], dtype=np.float32)
    frame_energy = np.mean(consensus_magnitude[frequency_mask, :], axis=0).astype(np.float32)
    reference_energy = float(np.percentile(frame_energy, 95.0)) if frame_energy.size else 0.0
    activity = np.clip(
        frame_energy / max(reference_energy * float(floor_ratio), 1e-8),
        0.0,
        1.0,
    ).astype(np.float32)
    return _smooth_frame_activity(activity, int(smoothing_frames))


def apply_consensus_vocal_body_repair(
    *,
    base_vocals_path: Path,
    donor_vocals_paths: tuple[Path, ...],
    output_vocals_path: Path,
    tuning: dict[str, float | int | str],
) -> dict[str, Any]:
    if not donor_vocals_paths:
        raise ValueError("Vocal body repair requires at least one donor vocal path")

    base_audio, base_sr = sf.read(str(base_vocals_path), always_2d=True, dtype="float32")
    donor_audios = [
        _read_and_match_donor_audio(base_audio, int(base_sr), donor_path)
        for donor_path in donor_vocals_paths
    ]
    frame_count = min([base_audio.shape[0], *(donor_audio.shape[0] for donor_audio in donor_audios)])
    base_audio = base_audio[:frame_count].astype(np.float32)
    donor_audios = [donor_audio[:frame_count].astype(np.float32) for donor_audio in donor_audios]

    n_fft = 2048
    hop_length = 512
    fill_low_hz = float(tuning["fill_low_hz"])
    fill_high_hz = float(tuning["fill_high_hz"])
    fill_threshold = float(tuning["fill_threshold"])
    fill_strength = float(tuning["fill_strength"])
    max_fill_ratio = float(tuning["max_fill_ratio"])
    donor_activity_floor_ratio = float(tuning["donor_activity_floor_ratio"])
    donor_activity_smoothing_frames = int(tuning["donor_activity_smoothing_frames"])
    target_rms_ratio = float(tuning["target_rms_ratio"])
    max_gain = float(tuning["max_gain"])
    phase_strategy = str(tuning.get("phase_strategy", "mixed")).strip() or "mixed"
    if phase_strategy not in {"mixed", "base"}:
        raise ValueError(f"Unsupported vocal body repair phase_strategy: {phase_strategy}")

    repaired_channels: list[np.ndarray] = []
    fill_mask_means: list[float] = []
    donor_activity_means: list[float] = []
    freq_bins = librosa.fft_frequencies(sr=int(base_sr), n_fft=n_fft).astype(np.float32)
    vocal_core_mask = (freq_bins >= fill_low_hz) & (freq_bins <= fill_high_hz)
    source_rms = _safe_rms(base_audio)

    for channel_index in range(base_audio.shape[1]):
        base_stft = librosa.stft(base_audio[:, channel_index], n_fft=n_fft, hop_length=hop_length)
        donor_stfts = [
            librosa.stft(donor_audio[:, channel_index], n_fft=n_fft, hop_length=hop_length)
            for donor_audio in donor_audios
        ]
        min_frames = min([base_stft.shape[1], *(donor_stft.shape[1] for donor_stft in donor_stfts)])
        base_stft = base_stft[:, :min_frames]
        donor_stfts = [donor_stft[:, :min_frames] for donor_stft in donor_stfts]
        base_magnitude = np.abs(base_stft).astype(np.float32)
        donor_magnitudes = np.stack([np.abs(donor_stft).astype(np.float32) for donor_stft in donor_stfts], axis=0)
        consensus_magnitude = np.median(donor_magnitudes, axis=0).astype(np.float32)
        donor_activity = _donor_activity_from_consensus(
            consensus_magnitude,
            vocal_core_mask,
            floor_ratio=donor_activity_floor_ratio,
            smoothing_frames=donor_activity_smoothing_frames,
        )

        deficit = np.clip(
            (consensus_magnitude - base_magnitude) / np.maximum(consensus_magnitude, 1e-8),
            0.0,
            1.0,
        ).astype(np.float32)
        fill_mask = np.clip(
            (deficit - fill_threshold) / max(1e-6, 1.0 - fill_threshold),
            0.0,
            1.0,
        ).astype(np.float32)
        fill_mask *= vocal_core_mask[:, np.newaxis].astype(np.float32)
        fill_mask *= donor_activity[np.newaxis, :].astype(np.float32)
        fill_mask = np.clip(fill_mask * fill_strength, 0.0, 1.0).astype(np.float32)

        fill_cap = np.maximum(base_magnitude, consensus_magnitude * max_fill_ratio).astype(np.float32)
        if phase_strategy == "base":
            target_magnitude = (
                base_magnitude + (consensus_magnitude - base_magnitude) * fill_mask
            ).astype(np.float32)
            capped_magnitude = np.minimum(target_magnitude, fill_cap).astype(np.float32)
            base_phase = np.exp(1j * np.angle(base_stft)).astype(np.complex64)
            repaired_stft = capped_magnitude * base_phase
        else:
            donor_complex = np.mean(np.stack(donor_stfts, axis=0), axis=0).astype(np.complex64)
            mixed_stft = base_stft * (1.0 - fill_mask) + donor_complex * fill_mask
            mixed_magnitude = np.abs(mixed_stft).astype(np.float32)
            capped_magnitude = np.minimum(mixed_magnitude, fill_cap).astype(np.float32)
            mixed_phase = np.exp(1j * np.angle(mixed_stft)).astype(np.complex64)
            repaired_stft = capped_magnitude * mixed_phase
        repaired_channel = librosa.istft(
            repaired_stft,
            hop_length=hop_length,
            length=base_audio.shape[0],
        ).astype(np.float32)
        repaired_channels.append(repaired_channel)
        fill_mask_means.append(float(np.mean(fill_mask)) if fill_mask.size else 0.0)
        donor_activity_means.append(float(np.mean(donor_activity)) if donor_activity.size else 0.0)

    repaired_audio = np.stack(repaired_channels, axis=1).astype(np.float32)
    repaired_rms = _safe_rms(repaired_audio)
    gain_applied = 1.0
    if target_rms_ratio > 1e-6 and source_rms > 1e-8 and repaired_rms > 1e-8:
        target_rms = source_rms * target_rms_ratio
        if target_rms > repaired_rms:
            gain_applied = min(max_gain, target_rms / repaired_rms)
            repaired_audio = (repaired_audio * gain_applied).astype(np.float32)
    repaired_audio = _limit_peak(repaired_audio)

    output_vocals_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_vocals_path), repaired_audio, int(base_sr), subtype="PCM_24")
    return {
        "sample_rate": int(base_sr),
        "fill_mask_mean": round(float(np.mean(fill_mask_means)) if fill_mask_means else 0.0, 6),
        "donor_activity_mean": round(float(np.mean(donor_activity_means)) if donor_activity_means else 0.0, 6),
        "source_vocals_rms": round(source_rms, 6),
        "repaired_vocals_rms": round(_safe_rms(repaired_audio), 6),
        "gain_applied": round(float(gain_applied), 6),
        "phase_strategy": phase_strategy,
        **_audio_summary(repaired_audio, int(base_sr)),
    }


def _read_and_match_candidate_to_reference(reference_audio: np.ndarray, reference_sr: int, candidate_path: Path) -> np.ndarray:
    candidate_audio, candidate_sr = sf.read(str(candidate_path), always_2d=True, dtype="float32")
    matched_candidate, _, _ = _match_audio(
        candidate_audio,
        int(candidate_sr),
        reference_audio,
        int(reference_sr),
    )
    if matched_candidate.shape[1] == reference_audio.shape[1]:
        return matched_candidate.astype(np.float32)
    if matched_candidate.shape[1] == 1 and reference_audio.shape[1] > 1:
        return np.repeat(matched_candidate, reference_audio.shape[1], axis=1).astype(np.float32)
    if matched_candidate.shape[1] > reference_audio.shape[1]:
        return matched_candidate[:, : reference_audio.shape[1]].astype(np.float32)
    repeats = int(math.ceil(reference_audio.shape[1] / float(matched_candidate.shape[1])))
    return np.tile(matched_candidate, (1, repeats))[:, : reference_audio.shape[1]].astype(np.float32)


def _frame_activity_from_band_energy(
    energy: np.ndarray,
    *,
    percentile: float,
    smoothing_frames: int,
) -> np.ndarray:
    if energy.size == 0:
        return np.zeros(0, dtype=np.float32)
    reference_energy = float(np.percentile(energy, float(percentile)))
    activity = np.clip(
        energy / max(reference_energy, 1e-8),
        0.0,
        1.0,
    ).astype(np.float32)
    return _smooth_frame_activity(activity, int(smoothing_frames))


def apply_segment_adaptive_vocal_router(
    *,
    source_audio_path: Path,
    start_sec: float,
    end_sec: float | None,
    clean_vocals_path: Path,
    strict_vocals_path: Path,
    repair_vocals_path: Path,
    preserve_vocals_path: Path,
    anchor_vocals_path: Path,
    output_vocals_path: Path,
    tuning: dict[str, float | int | str],
) -> dict[str, Any]:
    clean_audio, clean_sr = sf.read(str(clean_vocals_path), always_2d=True, dtype="float32")
    raw_audio, raw_sr = _read_source_window(source_audio_path, start_sec=start_sec, end_sec=end_sec)
    raw_audio, clean_audio, sample_rate = _match_audio(raw_audio, int(raw_sr), clean_audio, int(clean_sr))
    strict_audio = _read_and_match_candidate_to_reference(clean_audio, sample_rate, strict_vocals_path)
    repair_audio = _read_and_match_candidate_to_reference(clean_audio, sample_rate, repair_vocals_path)
    preserve_audio = _read_and_match_candidate_to_reference(clean_audio, sample_rate, preserve_vocals_path)
    anchor_audio = _read_and_match_candidate_to_reference(clean_audio, sample_rate, anchor_vocals_path)

    frame_count = min(
        raw_audio.shape[0],
        clean_audio.shape[0],
        strict_audio.shape[0],
        repair_audio.shape[0],
        preserve_audio.shape[0],
        anchor_audio.shape[0],
    )
    raw_audio = raw_audio[:frame_count].astype(np.float32)
    clean_audio = clean_audio[:frame_count].astype(np.float32)
    strict_audio = strict_audio[:frame_count].astype(np.float32)
    repair_audio = repair_audio[:frame_count].astype(np.float32)
    preserve_audio = preserve_audio[:frame_count].astype(np.float32)
    anchor_audio = anchor_audio[:frame_count].astype(np.float32)

    n_fft = 2048
    hop_length = 512
    body_low_hz = float(tuning["body_low_hz"])
    body_high_hz = float(tuning["body_high_hz"])
    core_debleed_low_hz = float(tuning.get("core_debleed_low_hz", body_low_hz))
    core_debleed_high_hz = float(tuning.get("core_debleed_high_hz", body_high_hz))
    high_band_start_hz = float(tuning["high_band_start_hz"])
    high_band_end_hz = float(tuning["high_band_end_hz"])
    body_deficit_threshold = float(tuning["body_deficit_threshold"])
    body_repair_strength = float(tuning["body_repair_strength"])
    preserve_tail_strength = float(tuning["preserve_tail_strength"])
    preserve_max_ratio = float(tuning["preserve_max_ratio"])
    max_body_fill_ratio = float(tuning["max_body_fill_ratio"])
    body_support_floor = float(tuning["body_support_floor"])
    body_support_power = float(tuning["body_support_power"])
    unsupported_busy_damping = float(tuning["unsupported_busy_damping"])
    core_unsupported_tolerance = float(tuning["core_unsupported_tolerance"])
    core_unsupported_damping = float(tuning["core_unsupported_damping"])
    core_debleed_strength = float(tuning.get("core_debleed_strength", 0.0))
    strict_high_tolerance = float(tuning["strict_high_tolerance"])
    strict_high_strength = float(tuning["strict_high_strength"])
    activity_floor_ratio = float(tuning["activity_floor_ratio"])
    activity_smoothing_frames = int(tuning["activity_smoothing_frames"])
    raw_busy_percentile = float(tuning["raw_busy_percentile"])
    raw_busy_boost = float(tuning["raw_busy_boost"])
    target_rms_ratio = float(tuning["target_rms_ratio"])
    max_gain = float(tuning["max_gain"])
    phase_strategy = str(tuning.get("phase_strategy", "clean_base")).strip() or "clean_base"
    if phase_strategy != "clean_base":
        raise ValueError(f"Unsupported segment router phase_strategy: {phase_strategy}")

    freq_bins = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft).astype(np.float32)
    vocal_core_mask = (freq_bins >= body_low_hz) & (freq_bins <= body_high_hz)
    core_debleed_mask = (freq_bins >= core_debleed_low_hz) & (freq_bins <= core_debleed_high_hz)
    high_band_mask = (freq_bins >= high_band_start_hz) & (freq_bins <= high_band_end_hz)
    raw_core_energy = _band_frame_energy(
        raw_audio,
        sample_rate,
        low_hz=body_low_hz,
        high_hz=body_high_hz,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    raw_high_energy = _band_frame_energy(
        raw_audio,
        sample_rate,
        low_hz=high_band_start_hz,
        high_hz=high_band_end_hz,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    raw_busy_activity = _frame_activity_from_band_energy(
        raw_core_energy,
        percentile=raw_busy_percentile,
        smoothing_frames=activity_smoothing_frames,
    )
    raw_high_activity = _frame_activity_from_band_energy(
        raw_high_energy,
        percentile=raw_busy_percentile,
        smoothing_frames=activity_smoothing_frames,
    )

    routed_channels: list[np.ndarray] = []
    body_route_means: list[float] = []
    preserve_route_means: list[float] = []
    strict_route_means: list[float] = []
    body_support_means: list[float] = []
    busy_damping_means: list[float] = []
    core_unsupported_gate_means: list[float] = []
    core_debleed_route_means: list[float] = []
    source_rms = _safe_rms(clean_audio)

    for channel_index in range(clean_audio.shape[1]):
        clean_stft = librosa.stft(clean_audio[:, channel_index], n_fft=n_fft, hop_length=hop_length)
        strict_stft = librosa.stft(strict_audio[:, channel_index], n_fft=n_fft, hop_length=hop_length)
        repair_stft = librosa.stft(repair_audio[:, channel_index], n_fft=n_fft, hop_length=hop_length)
        preserve_stft = librosa.stft(preserve_audio[:, channel_index], n_fft=n_fft, hop_length=hop_length)
        anchor_stft = librosa.stft(anchor_audio[:, channel_index], n_fft=n_fft, hop_length=hop_length)
        min_frames = min(
            clean_stft.shape[1],
            strict_stft.shape[1],
            repair_stft.shape[1],
            preserve_stft.shape[1],
            anchor_stft.shape[1],
            raw_busy_activity.size,
            raw_high_activity.size,
        )
        clean_stft = clean_stft[:, :min_frames]
        strict_stft = strict_stft[:, :min_frames]
        repair_stft = repair_stft[:, :min_frames]
        preserve_stft = preserve_stft[:, :min_frames]
        anchor_stft = anchor_stft[:, :min_frames]
        busy_activity = raw_busy_activity[:min_frames]
        high_activity = raw_high_activity[:min_frames]

        clean_magnitude = np.abs(clean_stft).astype(np.float32)
        strict_magnitude = np.abs(strict_stft).astype(np.float32)
        repair_magnitude = np.abs(repair_stft).astype(np.float32)
        preserve_magnitude = np.abs(preserve_stft).astype(np.float32)
        anchor_magnitude = np.abs(anchor_stft).astype(np.float32)
        strict_reference_magnitude = np.maximum(strict_magnitude, anchor_magnitude).astype(np.float32)
        capped_preserve = np.minimum(
            preserve_magnitude,
            np.maximum(clean_magnitude, repair_magnitude) * preserve_max_ratio,
        ).astype(np.float32)
        recovery_magnitude = np.maximum(repair_magnitude, capped_preserve).astype(np.float32)
        recovery_activity = _donor_activity_from_consensus(
            recovery_magnitude,
            vocal_core_mask,
            floor_ratio=activity_floor_ratio,
            smoothing_frames=activity_smoothing_frames,
        )[:min_frames]
        body_support_ratio = strict_reference_magnitude / np.maximum(recovery_magnitude, 1e-8)
        body_support_gate = np.clip(
            (body_support_ratio - body_support_floor) / max(1e-6, 1.0 - body_support_floor),
            0.0,
            1.0,
        ).astype(np.float32)
        if body_support_power != 1.0:
            body_support_gate = np.power(body_support_gate, body_support_power, dtype=np.float32)
        body_support_gate *= vocal_core_mask[:, np.newaxis].astype(np.float32)
        busy_damping = np.clip(
            1.0 - unsupported_busy_damping * high_activity[np.newaxis, :] * (1.0 - body_support_gate),
            0.0,
            1.0,
        ).astype(np.float32)
        unsupported_core = np.clip(
            (recovery_magnitude - strict_reference_magnitude * core_unsupported_tolerance)
            / np.maximum(recovery_magnitude, 1e-8),
            0.0,
            1.0,
        ).astype(np.float32)
        core_unsupported_gate = np.clip(
            1.0 - core_unsupported_damping * unsupported_core * vocal_core_mask[:, np.newaxis].astype(np.float32),
            0.0,
            1.0,
        ).astype(np.float32)

        body_deficit = np.clip(
            (recovery_magnitude - clean_magnitude) / np.maximum(recovery_magnitude, 1e-8),
            0.0,
            1.0,
        ).astype(np.float32)
        body_route = np.clip(
            (body_deficit - body_deficit_threshold) / max(1e-6, 1.0 - body_deficit_threshold),
            0.0,
            1.0,
        ).astype(np.float32)
        body_route *= vocal_core_mask[:, np.newaxis].astype(np.float32)
        body_route *= recovery_activity[np.newaxis, :].astype(np.float32)
        body_route *= body_support_gate
        body_route *= busy_damping
        body_route *= core_unsupported_gate
        body_route *= (1.0 + raw_busy_boost * busy_activity[np.newaxis, :]).astype(np.float32)
        body_route = np.clip(body_route * body_repair_strength, 0.0, 1.0).astype(np.float32)

        preserve_deficit = np.clip(
            (capped_preserve - repair_magnitude) / np.maximum(capped_preserve, 1e-8),
            0.0,
            1.0,
        ).astype(np.float32)
        preserve_route = np.clip(
            preserve_deficit * preserve_tail_strength,
            0.0,
            1.0,
        ).astype(np.float32)
        preserve_route *= vocal_core_mask[:, np.newaxis].astype(np.float32)
        preserve_route *= recovery_activity[np.newaxis, :].astype(np.float32)
        preserve_route *= body_support_gate
        preserve_route *= busy_damping
        preserve_route *= core_unsupported_gate

        body_target_magnitude = (
            clean_magnitude
            + (recovery_magnitude - clean_magnitude) * body_route
            + (capped_preserve - clean_magnitude) * preserve_route
        ).astype(np.float32)
        body_fill_cap = np.maximum(clean_magnitude, recovery_magnitude * max_body_fill_ratio).astype(np.float32)
        body_target_magnitude[vocal_core_mask, :] = np.minimum(
            body_target_magnitude[vocal_core_mask, :],
            body_fill_cap[vocal_core_mask, :],
        ).astype(np.float32)

        if core_debleed_strength > 0.0:
            core_debleed_activity = np.maximum(busy_activity, high_activity).astype(np.float32)
            core_debleed_route = unsupported_core * core_debleed_mask[:, np.newaxis].astype(np.float32)
            core_debleed_route *= (0.55 + 0.45 * core_debleed_activity[np.newaxis, :]).astype(np.float32)
            core_debleed_route = np.clip(core_debleed_route * core_debleed_strength, 0.0, 1.0).astype(np.float32)
            core_debleed_target = np.minimum(body_target_magnitude, strict_reference_magnitude).astype(np.float32)
            body_target_magnitude = (
                body_target_magnitude * (1.0 - core_debleed_route)
                + core_debleed_target * core_debleed_route
            ).astype(np.float32)
        else:
            core_debleed_route = np.zeros_like(body_target_magnitude, dtype=np.float32)

        unsupported_high = np.clip(
            (body_target_magnitude - strict_reference_magnitude * strict_high_tolerance)
            / np.maximum(body_target_magnitude, 1e-8),
            0.0,
            1.0,
        ).astype(np.float32)
        strict_route = unsupported_high * high_band_mask[:, np.newaxis].astype(np.float32)
        strict_route *= (0.72 + 0.28 * high_activity[np.newaxis, :]).astype(np.float32)
        strict_route = np.clip(strict_route * strict_high_strength, 0.0, 1.0).astype(np.float32)
        routed_magnitude = (
            body_target_magnitude * (1.0 - strict_route)
            + strict_reference_magnitude * strict_route
        ).astype(np.float32)

        clean_phase = np.exp(1j * np.angle(clean_stft)).astype(np.complex64)
        routed_stft = routed_magnitude * clean_phase
        routed_channel = librosa.istft(
            routed_stft,
            hop_length=hop_length,
            length=clean_audio.shape[0],
        ).astype(np.float32)
        routed_channels.append(routed_channel)
        body_route_means.append(float(np.mean(body_route)) if body_route.size else 0.0)
        preserve_route_means.append(float(np.mean(preserve_route)) if preserve_route.size else 0.0)
        strict_route_means.append(float(np.mean(strict_route)) if strict_route.size else 0.0)
        body_support_means.append(float(np.mean(body_support_gate[vocal_core_mask, :])) if np.any(vocal_core_mask) else 0.0)
        busy_damping_means.append(float(np.mean(busy_damping[vocal_core_mask, :])) if np.any(vocal_core_mask) else 0.0)
        core_unsupported_gate_means.append(
            float(np.mean(core_unsupported_gate[vocal_core_mask, :])) if np.any(vocal_core_mask) else 0.0
        )
        core_debleed_route_means.append(
            float(np.mean(core_debleed_route[core_debleed_mask, :])) if np.any(core_debleed_mask) else 0.0
        )

    routed_audio = np.stack(routed_channels, axis=1).astype(np.float32)
    routed_rms = _safe_rms(routed_audio)
    gain_applied = 1.0
    if target_rms_ratio > 1e-6 and source_rms > 1e-8 and routed_rms > 1e-8:
        target_rms = source_rms * target_rms_ratio
        if target_rms > routed_rms:
            gain_applied = min(max_gain, target_rms / routed_rms)
            routed_audio = (routed_audio * gain_applied).astype(np.float32)
    routed_audio = _limit_peak(routed_audio)

    output_vocals_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_vocals_path), routed_audio, int(sample_rate), subtype="PCM_24")
    return {
        "sample_rate": int(sample_rate),
        "body_route_mean": round(float(np.mean(body_route_means)) if body_route_means else 0.0, 6),
        "preserve_route_mean": round(float(np.mean(preserve_route_means)) if preserve_route_means else 0.0, 6),
        "strict_route_mean": round(float(np.mean(strict_route_means)) if strict_route_means else 0.0, 6),
        "body_support_mean": round(float(np.mean(body_support_means)) if body_support_means else 0.0, 6),
        "busy_damping_mean": round(float(np.mean(busy_damping_means)) if busy_damping_means else 0.0, 6),
        "core_unsupported_gate_mean": round(
            float(np.mean(core_unsupported_gate_means)) if core_unsupported_gate_means else 0.0,
            6,
        ),
        "core_debleed_route_mean": round(
            float(np.mean(core_debleed_route_means)) if core_debleed_route_means else 0.0,
            6,
        ),
        "raw_busy_activity_mean": round(float(np.mean(raw_busy_activity)) if raw_busy_activity.size else 0.0, 6),
        "raw_high_activity_mean": round(float(np.mean(raw_high_activity)) if raw_high_activity.size else 0.0, 6),
        "source_vocals_rms": round(source_rms, 6),
        "routed_vocals_rms": round(_safe_rms(routed_audio), 6),
        "gain_applied": round(float(gain_applied), 6),
        "phase_strategy": phase_strategy,
        **_audio_summary(routed_audio, int(sample_rate)),
    }


def _band_frame_energy(
    audio: np.ndarray,
    sample_rate: int,
    *,
    low_hz: float,
    high_hz: float,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    mono = _mono(audio)
    if mono.size == 0:
        return np.zeros(0, dtype=np.float32)
    stft = librosa.stft(mono, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft).astype(np.float32)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    mask = (freqs >= float(low_hz)) & (freqs < float(high_hz))
    if not np.any(mask):
        return np.zeros(magnitude.shape[1], dtype=np.float32)
    return np.mean(np.square(magnitude[mask], dtype=np.float32), axis=0).astype(np.float32)


def _band_ratio_db(audio: np.ndarray, sample_rate: int, *, low_hz: float, high_hz: float) -> float:
    mono = _mono(audio)
    if mono.size == 0:
        return -120.0
    stft = librosa.stft(mono, n_fft=4096, hop_length=1024)
    power = np.square(np.abs(stft).astype(np.float32), dtype=np.float32)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=4096)
    mask = (freqs >= float(low_hz)) & (freqs < float(high_hz))
    band_energy = float(np.sum(power[mask])) if np.any(mask) else 0.0
    total_energy = float(np.sum(power))
    return 10.0 * math.log10(max(band_energy, 1e-12) / max(total_energy, 1e-12))


def _spectral_flatness_band(audio: np.ndarray, sample_rate: int, *, low_hz: float, high_hz: float) -> float:
    mono = _mono(audio)
    if mono.size == 0:
        return 0.0
    stft = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512)).astype(np.float32) + 1e-8
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
    mask = (freqs >= float(low_hz)) & (freqs < float(high_hz))
    if not np.any(mask):
        return 0.0
    flatness = librosa.feature.spectral_flatness(S=stft[mask, :])
    return float(np.mean(flatness)) if flatness.size else 0.0


def _tonal_stability_band_metrics(
    audio: np.ndarray,
    sample_rate: int,
    *,
    low_hz: float,
    high_hz: float,
) -> dict[str, float]:
    mono = _mono(audio)
    if mono.size == 0:
        return {
            "flatness": 1.0,
            "flux": 1.0,
            "peak_ratio": 0.0,
        }
    magnitude = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512)).astype(np.float32) + 1e-8
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
    mask = (freqs >= float(low_hz)) & (freqs < float(high_hz))
    if not np.any(mask):
        return {
            "flatness": 1.0,
            "flux": 1.0,
            "peak_ratio": 0.0,
        }
    band_magnitude = magnitude[mask, :]
    frame_energy = np.sum(np.square(band_magnitude, dtype=np.float32), axis=0)
    if frame_energy.size == 0 or float(np.max(frame_energy)) <= 1e-10:
        return {
            "flatness": 1.0,
            "flux": 1.0,
            "peak_ratio": 0.0,
        }
    active_floor = max(float(np.percentile(frame_energy, 55.0)) * 0.20, float(np.max(frame_energy)) * 0.02)
    active_mask = frame_energy > active_floor
    if not np.any(active_mask):
        active_mask = np.ones_like(frame_energy, dtype=bool)
    active_magnitude = band_magnitude[:, active_mask]
    active_power = np.square(active_magnitude, dtype=np.float32)
    frame_power = np.sum(active_power, axis=0) + 1e-10
    flatness = librosa.feature.spectral_flatness(S=active_magnitude)
    normalized_magnitude = active_magnitude / (np.sum(active_magnitude, axis=0, keepdims=True) + 1e-8)
    if normalized_magnitude.shape[1] > 1:
        flux = float(np.mean(np.sqrt(np.sum(np.square(np.diff(normalized_magnitude, axis=1)), axis=0))))
    else:
        flux = 0.0
    peak_ratio = float(np.mean(np.max(active_power, axis=0) / frame_power))
    return {
        "flatness": float(np.mean(flatness)) if flatness.size else 1.0,
        "flux": flux,
        "peak_ratio": peak_ratio,
    }


def _stereo_side_mid_ratio_db(audio: np.ndarray) -> float:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 2:
        return -120.0
    mid = ((array[:, 0] + array[:, 1]) * 0.5).astype(np.float32)
    side = ((array[:, 0] - array[:, 1]) * 0.5).astype(np.float32)
    return 20.0 * math.log10(max(_safe_rms(side), 1e-10) / max(_safe_rms(mid), 1e-10))


def _backing_vocal_side_loss_db(raw_audio: np.ndarray, vocals_audio: np.ndarray) -> float:
    source_ratio_db = _stereo_side_mid_ratio_db(raw_audio)
    candidate_ratio_db = _stereo_side_mid_ratio_db(vocals_audio)
    if source_ratio_db <= -12.0:
        return 0.0
    return max(0.0, source_ratio_db - candidate_ratio_db)


def calculate_anchor_support_metrics(
    vocals_path: Path,
    anchor_vocals_path: Path,
    *,
    anchor_tolerance: float = 1.15,
    analysis_cache: AudioAnalysisCache | None = None,
) -> dict[str, float]:
    if analysis_cache is None:
        vocals_audio, vocals_sr = sf.read(str(vocals_path), always_2d=True)
        anchor_audio, anchor_sr = sf.read(str(anchor_vocals_path), always_2d=True)
    else:
        vocals_audio, vocals_sr = analysis_cache.read_audio(vocals_path)
        anchor_audio, anchor_sr = analysis_cache.read_audio(anchor_vocals_path)
    vocals_audio, anchor_audio, sample_rate = _match_audio(
        vocals_audio,
        int(vocals_sr),
        anchor_audio,
        int(anchor_sr),
    )
    return _calculate_anchor_support_metrics_from_audio(
        vocals_audio,
        anchor_audio,
        sample_rate,
        anchor_tolerance=anchor_tolerance,
    )


def _calculate_anchor_support_metrics_from_audio(
    vocals_audio: np.ndarray,
    anchor_audio: np.ndarray,
    sample_rate: int,
    *,
    anchor_tolerance: float = 1.15,
) -> dict[str, float]:
    vocals_mono = _mono(vocals_audio)
    anchor_mono = _mono(anchor_audio)
    vocals_stft = np.abs(librosa.stft(vocals_mono, n_fft=2048, hop_length=512)).astype(np.float32)
    anchor_stft = np.abs(librosa.stft(anchor_mono, n_fft=2048, hop_length=512)).astype(np.float32)
    bins = min(vocals_stft.shape[0], anchor_stft.shape[0])
    frames = min(vocals_stft.shape[1], anchor_stft.shape[1])
    vocals_power = np.square(vocals_stft[:bins, :frames], dtype=np.float32)
    unsupported = np.maximum(
        vocals_stft[:bins, :frames] - anchor_stft[:bins, :frames] * float(anchor_tolerance),
        0.0,
    ).astype(np.float32)
    unsupported_power = np.square(unsupported, dtype=np.float32)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)[:bins]
    high_mask = (freqs >= 4500.0) & (freqs < 16000.0)
    high_unsupported_power = float(np.sum(unsupported_power[high_mask])) if np.any(high_mask) else 0.0
    high_total_power = float(np.sum(vocals_power[high_mask])) if np.any(high_mask) else 0.0
    return {
        "unsupported_vs_anchor_ratio_db": round(
            10.0 * math.log10(max(float(np.sum(unsupported_power)), 1e-12) / max(float(np.sum(vocals_power)), 1e-12)),
            6,
        ),
        "high_unsupported_vs_anchor_ratio_db": round(
            10.0 * math.log10(max(high_unsupported_power, 1e-12) / max(high_total_power, 1e-12)),
            6,
        ),
        "high_noise_flatness_6000_16000": round(_spectral_flatness_band(vocals_audio, sample_rate, low_hz=6000.0, high_hz=16000.0), 6),
        "stereo_side_mid_ratio_db": round(_stereo_side_mid_ratio_db(vocals_audio), 6),
    }


def calculate_vocal_proxy_metrics(
    source_audio_path: Path,
    vocals_path: Path,
    *,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    analysis_cache: AudioAnalysisCache | None = None,
) -> dict[str, float]:
    if analysis_cache is None:
        raw_audio, raw_sr = _read_source_window(source_audio_path, start_sec=start_sec, end_sec=end_sec)
        vocals_audio, vocals_sr = sf.read(str(vocals_path), always_2d=True)
    else:
        raw_audio, raw_sr = analysis_cache.read_source_window(source_audio_path, start_sec=start_sec, end_sec=end_sec)
        vocals_audio, vocals_sr = analysis_cache.read_audio(vocals_path)
    raw_audio, vocals_audio, sample_rate = _match_audio(raw_audio, int(raw_sr), vocals_audio, int(vocals_sr))
    return _calculate_vocal_proxy_metrics_from_audio(raw_audio, vocals_audio, sample_rate)


def _calculate_vocal_proxy_metrics_from_audio(
    raw_audio: np.ndarray,
    vocals_audio: np.ndarray,
    sample_rate: int,
) -> dict[str, float]:
    vocal_band_raw = _band_frame_energy(raw_audio, sample_rate, low_hz=120.0, high_hz=8000.0)
    vocal_band_candidate = _band_frame_energy(vocals_audio, sample_rate, low_hz=120.0, high_hz=8000.0)
    frame_count = min(vocal_band_raw.size, vocal_band_candidate.size)
    vocal_band_raw = vocal_band_raw[:frame_count]
    vocal_band_candidate = vocal_band_candidate[:frame_count]

    if frame_count:
        raw_active_floor = float(np.percentile(vocal_band_raw, 55.0))
        raw_strong_floor = float(np.percentile(vocal_band_raw, 82.0))
        raw_active = vocal_band_raw > raw_active_floor
        raw_strong = vocal_band_raw > raw_strong_floor
        candidate_reference = max(float(np.percentile(vocal_band_candidate, 80.0)), 1e-10)
        candidate_active_floor = max(candidate_reference * 0.055, float(np.max(vocal_band_candidate)) * 0.012)
        candidate_missing = vocal_band_candidate < candidate_active_floor
        dropout_frame_ratio = float(np.mean(candidate_missing[raw_active])) if np.any(raw_active) else 0.0

        memory_frames = max(1, int(round(0.8 * sample_rate / 512.0)))
        recent_strong = np.convolve(raw_strong.astype(np.float32), np.ones(memory_frames, dtype=np.float32), mode="same") > 0
        tail_context = raw_active & recent_strong & (~raw_strong)
        tail_dropout_ratio = float(np.mean(candidate_missing[tail_context])) if np.any(tail_context) else dropout_frame_ratio
    else:
        dropout_frame_ratio = 1.0
        tail_dropout_ratio = 1.0

    raw_rms_db = rms_dbfs(raw_audio)
    vocals_rms_db = rms_dbfs(vocals_audio)
    source_stereo_side_mid_ratio_db = _stereo_side_mid_ratio_db(raw_audio)
    candidate_stereo_side_mid_ratio_db = _stereo_side_mid_ratio_db(vocals_audio)
    mid_tonal_metrics = _tonal_stability_band_metrics(
        vocals_audio,
        sample_rate,
        low_hz=700.0,
        high_hz=1800.0,
    )
    return {
        "source_rms_dbfs": round(raw_rms_db, 6),
        "candidate_rms_dbfs": round(vocals_rms_db, 6),
        "candidate_to_source_rms_db": round(vocals_rms_db - raw_rms_db, 6),
        "candidate_peak_dbfs": round(
            20.0 * math.log10(max(float(np.max(np.abs(vocals_audio))) if vocals_audio.size else 0.0, 1e-10)),
            6,
        ),
        "dropout_frame_ratio": round(dropout_frame_ratio, 6),
        "tail_dropout_ratio": round(tail_dropout_ratio, 6),
        "low_band_ratio_db_35_180": round(_band_ratio_db(vocals_audio, sample_rate, low_hz=35.0, high_hz=180.0), 6),
        "kick_punch_ratio_db_250_900": round(_band_ratio_db(vocals_audio, sample_rate, low_hz=250.0, high_hz=900.0), 6),
        "mid_tonal_band_ratio_db_700_1800": round(
            _band_ratio_db(vocals_audio, sample_rate, low_hz=700.0, high_hz=1800.0),
            6,
        ),
        "mid_tonal_flatness_700_1800": round(float(mid_tonal_metrics["flatness"]), 6),
        "mid_tonal_flux_700_1800": round(float(mid_tonal_metrics["flux"]), 6),
        "mid_tonal_peak_ratio_700_1800": round(float(mid_tonal_metrics["peak_ratio"]), 6),
        "presence_ratio_db_1800_4500": round(_band_ratio_db(vocals_audio, sample_rate, low_hz=1800.0, high_hz=4500.0), 6),
        "air_ratio_db_4500_9000": round(_band_ratio_db(vocals_audio, sample_rate, low_hz=4500.0, high_hz=9000.0), 6),
        "source_stereo_side_mid_ratio_db": round(source_stereo_side_mid_ratio_db, 6),
        "candidate_stereo_side_mid_ratio_db": round(candidate_stereo_side_mid_ratio_db, 6),
        "backing_vocal_side_loss_db": round(_backing_vocal_side_loss_db(raw_audio, vocals_audio), 6),
    }


def calculate_vocal_proxy_consistency_metrics(
    source_audio_path: Path,
    vocals_path: Path,
    *,
    start_sec: float,
    end_sec: float | None,
    segment_duration_sec: float,
    analysis_cache: AudioAnalysisCache | None = None,
) -> dict[str, float]:
    if float(segment_duration_sec) <= 0.0:
        raise ValueError("segment_duration_sec must be greater than zero")

    if analysis_cache is None:
        raw_audio, raw_sr = _read_source_window(source_audio_path, start_sec=start_sec, end_sec=end_sec)
        vocals_audio, vocals_sr = sf.read(str(vocals_path), always_2d=True)
    else:
        raw_audio, raw_sr = analysis_cache.read_source_window(source_audio_path, start_sec=start_sec, end_sec=end_sec)
        vocals_audio, vocals_sr = analysis_cache.read_audio(vocals_path)
    raw_audio, vocals_audio, sample_rate = _match_audio(raw_audio, int(raw_sr), vocals_audio, int(vocals_sr))
    frame_count = min(raw_audio.shape[0], vocals_audio.shape[0])
    if frame_count <= 0:
        return {
            "segment_consistency_window_sec": round(float(segment_duration_sec), 6),
            "segment_consistency_count": 0.0,
            "segment_consistency_worst_dropout_frame_ratio": 1.0,
            "segment_consistency_worst_tail_dropout_ratio": 1.0,
            "segment_consistency_dropout_std": 0.0,
            "segment_consistency_tail_dropout_std": 0.0,
            "segment_consistency_rms_ratio_min_db": -120.0,
            "segment_consistency_rms_ratio_max_db": -120.0,
            "segment_consistency_rms_ratio_range_db": 0.0,
            "segment_consistency_rms_ratio_std_db": 0.0,
        }

    segment_frames = max(1, int(round(float(segment_duration_sec) * sample_rate)))
    min_segment_frames = max(1, int(round(min(float(segment_duration_sec), 2.0) * sample_rate)))
    segment_metrics: list[dict[str, float]] = []
    for segment_start in range(0, frame_count, segment_frames):
        segment_end = min(frame_count, segment_start + segment_frames)
        if segment_end - segment_start < min_segment_frames and segment_metrics:
            break
        segment_metrics.append(
            _calculate_vocal_proxy_metrics_from_audio(
                raw_audio[segment_start:segment_end],
                vocals_audio[segment_start:segment_end],
                sample_rate,
            )
        )

    dropout_values = np.array([metrics["dropout_frame_ratio"] for metrics in segment_metrics], dtype=np.float32)
    tail_values = np.array([metrics["tail_dropout_ratio"] for metrics in segment_metrics], dtype=np.float32)
    rms_ratio_values = np.array([metrics["candidate_to_source_rms_db"] for metrics in segment_metrics], dtype=np.float32)
    return {
        "segment_consistency_window_sec": round(float(segment_duration_sec), 6),
        "segment_consistency_count": float(len(segment_metrics)),
        "segment_consistency_worst_dropout_frame_ratio": round(float(np.max(dropout_values)), 6),
        "segment_consistency_worst_tail_dropout_ratio": round(float(np.max(tail_values)), 6),
        "segment_consistency_dropout_std": round(float(np.std(dropout_values)), 6),
        "segment_consistency_tail_dropout_std": round(float(np.std(tail_values)), 6),
        "segment_consistency_rms_ratio_min_db": round(float(np.min(rms_ratio_values)), 6),
        "segment_consistency_rms_ratio_max_db": round(float(np.max(rms_ratio_values)), 6),
        "segment_consistency_rms_ratio_range_db": round(float(np.max(rms_ratio_values) - np.min(rms_ratio_values)), 6),
        "segment_consistency_rms_ratio_std_db": round(float(np.std(rms_ratio_values)), 6),
    }


def _add_segment_issue(
    *,
    labels: list[str],
    severity_parts: dict[str, float],
    label: str,
    severity: float,
) -> None:
    if severity <= 0.0:
        return
    labels.append(label)
    severity_parts[label] = round(float(severity), 6)


def _is_both_near_silent_segment(segment: dict[str, Any]) -> bool:
    metrics = segment.get("metrics") if isinstance(segment.get("metrics"), dict) else segment
    try:
        source_rms_dbfs = float(metrics.get("source_rms_dbfs", 0.0))
        candidate_rms_dbfs = float(metrics.get("candidate_rms_dbfs", 0.0))
    except (TypeError, ValueError):
        return False
    return (
        source_rms_dbfs <= SEGMENT_NEAR_SILENT_RMS_DBFS
        and candidate_rms_dbfs <= SEGMENT_NEAR_SILENT_RMS_DBFS
    )


def _issue_labels(segment: dict[str, Any]) -> list[str]:
    return [str(label) for label in segment.get("labels", []) if str(label) != "acceptable_proxy_window"]


def _actionable_issue_segments(segments: list[Any]) -> list[dict[str, Any]]:
    actionable_segments: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        if _is_both_near_silent_segment(segment):
            continue
        if float(segment.get("severity_score", 0.0)) <= 0.0:
            continue
        if not _issue_labels(segment):
            continue
        actionable_segments.append(segment)
    return actionable_segments


def classify_vocal_segment_issue(
    segment_metrics: dict[str, float],
    *,
    track_metrics: dict[str, float],
) -> dict[str, Any]:
    labels: list[str] = []
    severity_parts: dict[str, float] = {}

    dropout_ratio = float(segment_metrics["dropout_frame_ratio"])
    tail_dropout_ratio = float(segment_metrics["tail_dropout_ratio"])
    track_dropout_ratio = float(track_metrics.get("dropout_frame_ratio", 0.0))
    track_tail_dropout_ratio = float(track_metrics.get("tail_dropout_ratio", 0.0))
    rms_ratio_db = float(segment_metrics["candidate_to_source_rms_db"])
    vocals_rms_db = float(segment_metrics.get("candidate_rms_dbfs", -200.0))
    low_ratio_db = float(segment_metrics["low_band_ratio_db_35_180"])
    kick_ratio_db = float(segment_metrics["kick_punch_ratio_db_250_900"])
    presence_ratio_db = float(segment_metrics["presence_ratio_db_1800_4500"])
    air_ratio_db = float(segment_metrics["air_ratio_db_4500_9000"])
    unsupported_vs_anchor_db = segment_metrics.get("unsupported_vs_anchor_ratio_db")
    high_unsupported_vs_anchor_db = segment_metrics.get("high_unsupported_vs_anchor_ratio_db")
    high_noise_flatness = segment_metrics.get("high_noise_flatness_6000_16000")
    stereo_side_mid_ratio_db = segment_metrics.get("stereo_side_mid_ratio_db")
    source_stereo_side_mid_ratio_db = segment_metrics.get("source_stereo_side_mid_ratio_db")
    backing_vocal_side_loss_db = segment_metrics.get("backing_vocal_side_loss_db")
    mid_tonal_band_ratio_db = segment_metrics.get("mid_tonal_band_ratio_db_700_1800")
    mid_tonal_flatness = segment_metrics.get("mid_tonal_flatness_700_1800")
    mid_tonal_flux = segment_metrics.get("mid_tonal_flux_700_1800")
    mid_tonal_peak_ratio = segment_metrics.get("mid_tonal_peak_ratio_700_1800")
    if _is_both_near_silent_segment(segment_metrics):
        return {
            "labels": ["acceptable_proxy_window"],
            "severity_score": 0.0,
            "severity_parts": {},
        }

    dropout_threshold = max(track_dropout_ratio + 0.22, 0.50)
    _add_segment_issue(
        labels=labels,
        severity_parts=severity_parts,
        label="vocal_dropout",
        severity=(dropout_ratio - dropout_threshold) * 2.20,
    )

    tail_threshold = max(track_tail_dropout_ratio + 0.22, 0.50)
    _add_segment_issue(
        labels=labels,
        severity_parts=severity_parts,
        label="vocal_tail_loss",
        severity=(tail_dropout_ratio - tail_threshold) * 2.45,
    )

    _add_segment_issue(
        labels=labels,
        severity_parts=severity_parts,
        label="thin_or_overcleaned",
        severity=(-31.0 - rms_ratio_db) * 0.12,
    )
    _add_segment_issue(
        labels=labels,
        severity_parts=severity_parts,
        label="leaky_full_mix_risk",
        severity=(rms_ratio_db + 4.5) * 0.18,
    )
    _add_segment_issue(
        labels=labels,
        severity_parts=severity_parts,
        label="low_band_bleed",
        severity=(low_ratio_db + 18.5) * 0.08,
    )
    _add_segment_issue(
        labels=labels,
        severity_parts=severity_parts,
        label="kick_punch_bleed",
        severity=(kick_ratio_db + 11.5) * 0.07,
    )
    if unsupported_vs_anchor_db is not None:
        _add_segment_issue(
            labels=labels,
            severity_parts=severity_parts,
            label="unsupported_anchor_leakage",
            severity=(float(unsupported_vs_anchor_db) + 22.0) * 0.08,
        )
    if high_unsupported_vs_anchor_db is not None:
        _add_segment_issue(
            labels=labels,
            severity_parts=severity_parts,
            label="high_band_unsupported_leakage",
            severity=(float(high_unsupported_vs_anchor_db) + 20.0) * 0.06,
        )
    if high_noise_flatness is not None:
        _add_segment_issue(
            labels=labels,
            severity_parts=severity_parts,
            label="high_noise_or_robotic_artifact",
            severity=(float(high_noise_flatness) - 0.16) * 1.60,
        )
    if (
        mid_tonal_band_ratio_db is not None
        and mid_tonal_flatness is not None
        and mid_tonal_flux is not None
        and mid_tonal_peak_ratio is not None
    ):
        mid_tonal_band_ratio = float(mid_tonal_band_ratio_db)
        flatness_value = float(mid_tonal_flatness)
        flux_value = float(mid_tonal_flux)
        peak_ratio_value = float(mid_tonal_peak_ratio)
        has_audible_mid_tonal_content = (
            vocals_rms_db > -55.0
            and rms_ratio_db > -34.0
            and mid_tonal_band_ratio > -12.0
        )
        has_stable_narrowband_shape = (
            flatness_value < 0.095
            and peak_ratio_value > 0.18
            and flux_value < 0.21
        )
        if has_audible_mid_tonal_content and has_stable_narrowband_shape:
            _add_segment_issue(
                labels=labels,
                severity_parts=severity_parts,
                label="drop_melody_leakage",
                severity=(
                    max(0.0, 0.095 - flatness_value) * 2.20
                    + max(0.0, 0.21 - flux_value) * 0.75
                    + max(0.0, peak_ratio_value - 0.18) * 0.85
                    + max(0.0, mid_tonal_band_ratio + 14.0) * 0.026
                ),
            )
    if high_noise_flatness is not None and stereo_side_mid_ratio_db is not None:
        high_noise_value = float(high_noise_flatness)
        stereo_side_mid_value = float(stereo_side_mid_ratio_db)
        has_tail_context = (
            tail_dropout_ratio <= max(track_tail_dropout_ratio + 0.30, 0.42)
            and dropout_ratio <= max(track_dropout_ratio + 0.26, 0.42)
        )
        has_noisy_wide_tail = (
            high_noise_value > 0.20
            and stereo_side_mid_value > -6.0
            and air_ratio_db > -28.0
        )
        if has_tail_context and has_noisy_wide_tail:
            _add_segment_issue(
                labels=labels,
                severity_parts=severity_parts,
                label="excessive_reverb_tail_smear",
                severity=(
                    (high_noise_value - 0.20) * 1.20
                    + (stereo_side_mid_value + 6.0) * 0.07
                    + max(0.0, air_ratio_db + 28.0) * 0.015
                ),
            )
    anchor_supported = unsupported_vs_anchor_db is None or float(unsupported_vs_anchor_db) <= -18.0
    high_band_supported = high_unsupported_vs_anchor_db is None or float(high_unsupported_vs_anchor_db) <= -14.0
    high_noise_guard = high_noise_flatness is None or float(high_noise_flatness) <= 0.24
    body_rms_deficit = max(0.0, -21.0 - rms_ratio_db)
    local_body_activity_deficit = max(0.0, dropout_ratio - 0.24, tail_dropout_ratio - 0.32)
    if anchor_supported and high_band_supported and high_noise_guard and local_body_activity_deficit > 0.0:
        _add_segment_issue(
            labels=labels,
            severity_parts=severity_parts,
            label="local_vocal_body_dip",
            severity=body_rms_deficit * 0.08 + local_body_activity_deficit * 0.45,
        )
    if stereo_side_mid_ratio_db is not None:
        _add_segment_issue(
            labels=labels,
            severity_parts=severity_parts,
            label="wide_stereo_non_vocal_leakage",
            severity=(float(stereo_side_mid_ratio_db) + 8.0) * 0.06,
        )
    if (
        backing_vocal_side_loss_db is not None
        and source_stereo_side_mid_ratio_db is not None
        and float(source_stereo_side_mid_ratio_db) > -12.0
    ):
        _add_segment_issue(
            labels=labels,
            severity_parts=severity_parts,
            label="backing_vocal_loss_risk",
            severity=(float(backing_vocal_side_loss_db) - 6.0) * 0.075,
        )
    _add_segment_issue(
        labels=labels,
        severity_parts=severity_parts,
        label="presence_loss",
        severity=(-30.0 - presence_ratio_db) * 0.07,
    )
    _add_segment_issue(
        labels=labels,
        severity_parts=severity_parts,
        label="air_detail_loss",
        severity=(-40.0 - air_ratio_db) * 0.05,
    )

    severity_score = round(float(sum(severity_parts.values())), 6)
    return {
        "labels": labels if labels else ["acceptable_proxy_window"],
        "severity_score": severity_score,
        "severity_parts": severity_parts,
    }


def summarize_vocal_segment_issue_report(report: dict[str, Any]) -> dict[str, float]:
    summary = report.get("summary", {})
    segments = report.get("segments")
    reverb_tail_smear_scores: list[float] = []
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            labels = segment.get("labels", [])
            if not isinstance(labels, list) or "excessive_reverb_tail_smear" not in labels:
                continue
            severity_parts = segment.get("severity_parts")
            if isinstance(severity_parts, dict):
                score = float(severity_parts.get("excessive_reverb_tail_smear", 0.0))
            else:
                score = float(segment.get("severity_score", 0.0))
            if score > 0.0:
                reverb_tail_smear_scores.append(score)
    segment_count = float(len(segments)) if isinstance(segments, list) else float(summary.get("issue_segments_count", 0.0))
    return {
        "segment_issue_max_score": float(summary.get("max_issue_score", 0.0)),
        "segment_issue_mean_top3_score": float(summary.get("mean_top3_issue_score", 0.0)),
        "segment_issue_count": float(summary.get("issue_segments_count", 0.0)),
        "segment_issue_window_ratio": float(summary.get("issue_window_ratio", 0.0)),
        "segment_reverb_tail_smear_count": float(len(reverb_tail_smear_scores)),
        "segment_reverb_tail_smear_window_ratio": round(
            float(len(reverb_tail_smear_scores)) / max(segment_count, 1.0),
            6,
        ),
        "segment_reverb_tail_smear_max_score": round(
            float(max(reverb_tail_smear_scores)) if reverb_tail_smear_scores else 0.0,
            6,
        ),
    }


def calculate_vocal_segment_issue_report(
    source_audio_path: Path,
    vocals_path: Path,
    *,
    anchor_vocals_path: Path | None,
    start_sec: float,
    end_sec: float | None,
    segment_duration_sec: float,
    max_ranked_segments: int,
    analysis_cache: AudioAnalysisCache | None = None,
) -> dict[str, Any]:
    if float(segment_duration_sec) <= 0.0:
        raise ValueError("segment_duration_sec must be greater than zero")
    if int(max_ranked_segments) < 0:
        raise ValueError("max_ranked_segments must be zero or greater")

    if analysis_cache is None:
        raw_audio, raw_sr = _read_source_window(source_audio_path, start_sec=start_sec, end_sec=end_sec)
        vocals_audio, vocals_sr = sf.read(str(vocals_path), always_2d=True)
    else:
        raw_audio, raw_sr = analysis_cache.read_source_window(source_audio_path, start_sec=start_sec, end_sec=end_sec)
        vocals_audio, vocals_sr = analysis_cache.read_audio(vocals_path)
    raw_audio, vocals_audio, sample_rate = _match_audio(raw_audio, int(raw_sr), vocals_audio, int(vocals_sr))

    anchor_audio: np.ndarray | None = None
    if anchor_vocals_path is not None:
        if analysis_cache is None:
            anchor_audio_raw, anchor_sr = sf.read(str(anchor_vocals_path), always_2d=True)
        else:
            anchor_audio_raw, anchor_sr = analysis_cache.read_audio(anchor_vocals_path)
        anchor_audio, vocals_audio, sample_rate = _match_audio(
            anchor_audio_raw,
            int(anchor_sr),
            vocals_audio,
            sample_rate,
        )

    frame_count = min(raw_audio.shape[0], vocals_audio.shape[0])
    if anchor_audio is not None:
        frame_count = min(frame_count, anchor_audio.shape[0])
    raw_audio = raw_audio[:frame_count]
    vocals_audio = vocals_audio[:frame_count]
    if anchor_audio is not None:
        anchor_audio = anchor_audio[:frame_count]

    if frame_count <= 0:
        empty_summary = {
            "max_issue_score": 0.0,
            "mean_top3_issue_score": 0.0,
            "issue_segments_count": 0.0,
            "issue_window_ratio": 0.0,
            "dominant_issue_labels": [],
        }
        return {
            "schema_version": 1,
            "mode": "vocal_segment_issue_report",
            "segment_duration_sec": round(float(segment_duration_sec), 6),
            "max_ranked_segments": int(max_ranked_segments),
            "segments_analyzed": 0,
            "summary": empty_summary,
            "summary_metrics": summarize_vocal_segment_issue_report({"summary": empty_summary}),
            "segments": [],
            "top_segments": [],
        }

    track_metrics = _calculate_vocal_proxy_metrics_from_audio(raw_audio, vocals_audio, sample_rate)
    if anchor_audio is not None:
        track_metrics.update(_calculate_anchor_support_metrics_from_audio(vocals_audio, anchor_audio, sample_rate))

    segment_frames = max(1, int(round(float(segment_duration_sec) * sample_rate)))
    min_segment_frames = max(1, int(round(min(float(segment_duration_sec), 2.0) * sample_rate)))
    segments: list[dict[str, Any]] = []
    for segment_start in range(0, frame_count, segment_frames):
        segment_end = min(frame_count, segment_start + segment_frames)
        if segment_end - segment_start < min_segment_frames and segments:
            break
        metrics = _calculate_vocal_proxy_metrics_from_audio(
            raw_audio[segment_start:segment_end],
            vocals_audio[segment_start:segment_end],
            sample_rate,
        )
        if anchor_audio is not None:
            metrics.update(
                _calculate_anchor_support_metrics_from_audio(
                    vocals_audio[segment_start:segment_end],
                    anchor_audio[segment_start:segment_end],
                    sample_rate,
                )
            )
        issue = classify_vocal_segment_issue(metrics, track_metrics=track_metrics)
        relative_start_sec = float(segment_start) / float(sample_rate)
        relative_end_sec = float(segment_end) / float(sample_rate)
        segments.append(
            {
                "start_sec": round(float(start_sec) + relative_start_sec, 6),
                "end_sec": round(float(start_sec) + relative_end_sec, 6),
                "relative_start_sec": round(relative_start_sec, 6),
                "relative_end_sec": round(relative_end_sec, 6),
                "severity_score": issue["severity_score"],
                "labels": issue["labels"],
                "severity_parts": issue["severity_parts"],
                "metrics": metrics,
            }
        )

    issue_segments = [
        segment
        for segment in segments
        if segment["severity_score"] > 0.0
    ]
    ranked_segments = sorted(
        issue_segments,
        key=lambda segment: (float(segment["severity_score"]), float(segment["end_sec"]) - float(segment["start_sec"])),
        reverse=True,
    )
    label_counts: dict[str, int] = {}
    for segment in issue_segments:
        for label in segment["labels"]:
            if label == "acceptable_proxy_window":
                continue
            label_counts[label] = label_counts.get(label, 0) + 1
    top_scores = [float(segment["severity_score"]) for segment in ranked_segments[:3]]
    summary = {
        "max_issue_score": round(float(ranked_segments[0]["severity_score"]) if ranked_segments else 0.0, 6),
        "mean_top3_issue_score": round(float(np.mean(top_scores)) if top_scores else 0.0, 6),
        "issue_segments_count": float(len(issue_segments)),
        "issue_window_ratio": round(float(len(issue_segments)) / max(float(len(segments)), 1.0), 6),
        "dominant_issue_labels": [
            {"label": label, "count": count}
            for label, count in sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
    }
    report = {
        "schema_version": 1,
        "mode": "vocal_segment_issue_report",
        "segment_duration_sec": round(float(segment_duration_sec), 6),
        "max_ranked_segments": int(max_ranked_segments),
        "segments_analyzed": int(len(segments)),
        "summary": summary,
        "segments": segments,
        "top_segments": ranked_segments[: int(max_ranked_segments)] if int(max_ranked_segments) > 0 else [],
    }
    report["summary_metrics"] = summarize_vocal_segment_issue_report(report)
    return report


def write_anchor_suppressed_candidate_manifest(
    *,
    profile: AnchorSuppressionProfile,
    source_manifest: dict[str, Any],
    anchor_manifest: dict[str, Any],
    dependency_fingerprints: dict[str, dict[str, str]],
    output_dir: Path,
    output_vocals_path: Path,
    suppression_debug: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "audio_path": source_manifest.get("audio_path"),
        "source_audio_sha256": source_manifest.get("source_audio_sha256"),
        "sample_rate": int(suppression_debug["sample_rate"]),
        "export_subtype": "PCM_24",
        "preview_window_sec": source_manifest.get("preview_window_sec"),
        "pipeline": {
            "mode": "vocal_preview_derived_candidate",
            "candidate_mode": "anchor_suppressed_vocals",
            "source_profile": profile.source_profile_name,
            "anchor_profile": profile.anchor_profile_name,
            "source_candidate_mode": source_manifest.get("pipeline", {}).get("candidate_mode"),
            "anchor_candidate_mode": anchor_manifest.get("pipeline", {}).get("candidate_mode"),
            "dependency_fingerprints": dependency_fingerprints,
            "vocal_postprocess": "anchor_guided_leakage_suppression_v1",
        },
        "tuning": profile.tuning,
        "stems": {
            "vocals": {
                "path": str(output_vocals_path),
                "vocal_postprocess": "anchor_guided_leakage_suppression_v1",
                "source_profile": profile.source_profile_name,
                "anchor_profile": profile.anchor_profile_name,
                **suppression_debug,
            }
        },
    }
    manifest_path = output_dir / "stem_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_anchor_suppressed_candidate(
    *,
    profile: AnchorSuppressionProfile,
    source_record: VocalCandidateRecord,
    anchor_record: VocalCandidateRecord,
    output_dir: Path,
) -> dict[str, Any]:
    output_vocals_path = output_dir / "vocals.flac"
    suppression_debug = apply_anchor_guided_leakage_suppression(
        source_vocals_path=source_record.vocals_path,
        anchor_vocals_path=anchor_record.vocals_path,
        output_vocals_path=output_vocals_path,
        tuning=profile.tuning,
    )
    return write_anchor_suppressed_candidate_manifest(
        profile=profile,
        source_manifest=source_record.manifest,
        anchor_manifest=anchor_record.manifest,
        dependency_fingerprints={
            profile.source_profile_name: _dependency_fingerprint(source_record),
            profile.anchor_profile_name: _dependency_fingerprint(anchor_record),
        },
        output_dir=output_dir,
        output_vocals_path=output_vocals_path,
        suppression_debug=suppression_debug,
    )


def write_vocal_body_repair_candidate_manifest(
    *,
    profile: VocalBodyRepairProfile,
    base_manifest: dict[str, Any],
    donor_manifests: dict[str, dict[str, Any]],
    dependency_fingerprints: dict[str, dict[str, str]],
    output_dir: Path,
    output_vocals_path: Path,
    repair_debug: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "audio_path": base_manifest.get("audio_path"),
        "source_audio_sha256": base_manifest.get("source_audio_sha256"),
        "sample_rate": int(repair_debug["sample_rate"]),
        "export_subtype": "PCM_24",
        "preview_window_sec": base_manifest.get("preview_window_sec"),
        "pipeline": {
            "mode": "vocal_preview_derived_candidate",
            "candidate_mode": "vocal_body_repaired_vocals",
            "base_profile": profile.base_profile_name,
            "donor_profiles": list(profile.donor_profile_names),
            "base_candidate_mode": base_manifest.get("pipeline", {}).get("candidate_mode"),
            "donor_candidate_modes": {
                donor_name: donor_manifest.get("pipeline", {}).get("candidate_mode")
                for donor_name, donor_manifest in donor_manifests.items()
            },
            "dependency_fingerprints": dependency_fingerprints,
            "vocal_postprocess": "consensus_vocal_body_repair_v1",
        },
        "tuning": profile.tuning,
        "stems": {
            "vocals": {
                "path": str(output_vocals_path),
                "vocal_postprocess": "consensus_vocal_body_repair_v1",
                "base_profile": profile.base_profile_name,
                "donor_profiles": list(profile.donor_profile_names),
                **repair_debug,
            }
        },
    }
    manifest_path = output_dir / "stem_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_vocal_body_repair_candidate(
    *,
    profile: VocalBodyRepairProfile,
    base_record: VocalCandidateRecord,
    donor_records: dict[str, VocalCandidateRecord],
    output_dir: Path,
) -> dict[str, Any]:
    output_vocals_path = output_dir / "vocals.flac"
    repair_debug = apply_consensus_vocal_body_repair(
        base_vocals_path=base_record.vocals_path,
        donor_vocals_paths=tuple(donor_records[name].vocals_path for name in profile.donor_profile_names),
        output_vocals_path=output_vocals_path,
        tuning=profile.tuning,
    )
    return write_vocal_body_repair_candidate_manifest(
        profile=profile,
        base_manifest=base_record.manifest,
        donor_manifests={name: donor_records[name].manifest for name in profile.donor_profile_names},
        dependency_fingerprints={
            profile.base_profile_name: _dependency_fingerprint(base_record),
            **{
                donor_name: _dependency_fingerprint(donor_records[donor_name])
                for donor_name in profile.donor_profile_names
            },
        },
        output_dir=output_dir,
        output_vocals_path=output_vocals_path,
        repair_debug=repair_debug,
    )


def write_segment_router_candidate_manifest(
    *,
    profile: SegmentRouterProfile,
    clean_manifest: dict[str, Any],
    strict_manifest: dict[str, Any],
    repair_manifest: dict[str, Any],
    preserve_manifest: dict[str, Any],
    anchor_manifest: dict[str, Any],
    dependency_fingerprints: dict[str, dict[str, str]],
    output_dir: Path,
    output_vocals_path: Path,
    router_debug: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "audio_path": clean_manifest.get("audio_path"),
        "source_audio_sha256": clean_manifest.get("source_audio_sha256"),
        "sample_rate": int(router_debug["sample_rate"]),
        "export_subtype": "PCM_24",
        "preview_window_sec": clean_manifest.get("preview_window_sec"),
        "pipeline": {
            "mode": "vocal_preview_derived_candidate",
            "candidate_mode": "segment_adaptive_routed_vocals",
            "clean_profile": profile.clean_profile_name,
            "strict_profile": profile.strict_profile_name,
            "repair_profile": profile.repair_profile_name,
            "preserve_profile": profile.preserve_profile_name,
            "anchor_profile": profile.anchor_profile_name,
            "source_candidate_modes": {
                profile.clean_profile_name: clean_manifest.get("pipeline", {}).get("candidate_mode"),
                profile.strict_profile_name: strict_manifest.get("pipeline", {}).get("candidate_mode"),
                profile.repair_profile_name: repair_manifest.get("pipeline", {}).get("candidate_mode"),
                profile.preserve_profile_name: preserve_manifest.get("pipeline", {}).get("candidate_mode"),
                profile.anchor_profile_name: anchor_manifest.get("pipeline", {}).get("candidate_mode"),
            },
            "dependency_fingerprints": dependency_fingerprints,
            "vocal_postprocess": "segment_adaptive_vocal_router_v1",
        },
        "tuning": profile.tuning,
        "stems": {
            "vocals": {
                "path": str(output_vocals_path),
                "vocal_postprocess": "segment_adaptive_vocal_router_v1",
                "clean_profile": profile.clean_profile_name,
                "strict_profile": profile.strict_profile_name,
                "repair_profile": profile.repair_profile_name,
                "preserve_profile": profile.preserve_profile_name,
                "anchor_profile": profile.anchor_profile_name,
                **router_debug,
            }
        },
    }
    manifest_path = output_dir / "stem_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_segment_router_candidate(
    *,
    profile: SegmentRouterProfile,
    source_audio_path: Path,
    start_sec: float,
    end_sec: float | None,
    clean_record: VocalCandidateRecord,
    strict_record: VocalCandidateRecord,
    repair_record: VocalCandidateRecord,
    preserve_record: VocalCandidateRecord,
    anchor_record: VocalCandidateRecord,
    output_dir: Path,
) -> dict[str, Any]:
    output_vocals_path = output_dir / "vocals.flac"
    router_debug = apply_segment_adaptive_vocal_router(
        source_audio_path=source_audio_path,
        start_sec=start_sec,
        end_sec=end_sec,
        clean_vocals_path=clean_record.vocals_path,
        strict_vocals_path=strict_record.vocals_path,
        repair_vocals_path=repair_record.vocals_path,
        preserve_vocals_path=preserve_record.vocals_path,
        anchor_vocals_path=anchor_record.vocals_path,
        output_vocals_path=output_vocals_path,
        tuning=profile.tuning,
    )
    return write_segment_router_candidate_manifest(
        profile=profile,
        clean_manifest=clean_record.manifest,
        strict_manifest=strict_record.manifest,
        repair_manifest=repair_record.manifest,
        preserve_manifest=preserve_record.manifest,
        anchor_manifest=anchor_record.manifest,
        dependency_fingerprints={
            profile.clean_profile_name: _dependency_fingerprint(clean_record),
            profile.strict_profile_name: _dependency_fingerprint(strict_record),
            profile.repair_profile_name: _dependency_fingerprint(repair_record),
            profile.preserve_profile_name: _dependency_fingerprint(preserve_record),
            profile.anchor_profile_name: _dependency_fingerprint(anchor_record),
        },
        output_dir=output_dir,
        output_vocals_path=output_vocals_path,
        router_debug=router_debug,
    )


def _segment_issue_label_penalty(labels: list[str], tuning: dict[str, float | int | str]) -> float:
    penalty = 0.0
    if "leaky_full_mix_risk" in labels:
        penalty += float(tuning["leaky_full_mix_label_penalty"])
    if "unsupported_anchor_leakage" in labels:
        penalty += float(tuning["unsupported_anchor_label_penalty"])
    if "high_band_unsupported_leakage" in labels:
        penalty += float(tuning["high_band_unsupported_label_penalty"])
    if "backing_vocal_loss_risk" in labels:
        penalty += float(tuning["backing_vocal_loss_label_penalty"])
    if "thin_or_overcleaned" in labels:
        penalty += float(tuning["thin_or_overcleaned_label_penalty"])
    return penalty


def _adjusted_segment_issue_score(
    *,
    segment: dict[str, Any],
    candidate_score: VocalCandidateScore,
    base_score: VocalCandidateScore,
    tuning: dict[str, float | int | str],
) -> float:
    labels = [str(label) for label in segment.get("labels", [])]
    severity = float(segment.get("severity_score", 0.0))
    score_drop = max(0.0, float(base_score.score) - float(candidate_score.score))
    capped_score_drop = min(score_drop, float(tuning["max_global_score_drop"]))
    return (
        severity
        + _segment_issue_label_penalty(labels, tuning)
        + capped_score_drop * float(tuning["global_score_penalty_weight"])
    )


def _issue_aware_selection_metrics(manifest: dict[str, Any]) -> dict[str, float]:
    vocals = manifest.get("stems", {}).get("vocals", {})
    if not isinstance(vocals, dict):
        return {}
    window_decisions = vocals.get("window_decisions")
    if not isinstance(window_decisions, list):
        return {}
    improvements = [
        float(decision.get("adjusted_improvement", 0.0))
        for decision in window_decisions
        if isinstance(decision, dict)
    ]
    if not improvements:
        return {
            "issue_aware_replaced_window_count": 0.0,
            "issue_aware_adjusted_improvement_sum": 0.0,
            "issue_aware_adjusted_improvement_max": 0.0,
        }
    return {
        "issue_aware_replaced_window_count": float(len(improvements)),
        "issue_aware_adjusted_improvement_sum": round(float(sum(improvements)), 6),
        "issue_aware_adjusted_improvement_max": round(float(max(improvements)), 6),
        "issue_aware_base_segment_issue_max_score": float(vocals.get("base_segment_issue_max_score", 0.0)),
        "issue_aware_base_segment_issue_mean_top3_score": float(vocals.get("base_segment_issue_mean_top3_score", 0.0)),
    }


def _segments_share_window(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        _float_values_match(left.get("relative_start_sec"), right.get("relative_start_sec"), tolerance=0.02)
        and _float_values_match(left.get("relative_end_sec"), right.get("relative_end_sec"), tolerance=0.02)
    )


def _candidate_report_segment(
    report: dict[str, Any],
    segment_index: int,
    base_segment: dict[str, Any],
) -> dict[str, Any] | None:
    segments = report.get("segments")
    if not isinstance(segments, list) or segment_index >= len(segments):
        return None
    candidate_segment = segments[segment_index]
    if not isinstance(candidate_segment, dict):
        return None
    if not _segments_share_window(base_segment, candidate_segment):
        return None
    return candidate_segment


def _segment_for_shared_issue_scoring(
    segment: dict[str, Any],
    *,
    track_metrics: dict[str, float],
) -> tuple[dict[str, Any], str]:
    if not {
        "dropout_frame_ratio",
        "tail_dropout_ratio",
    }.issubset(track_metrics):
        return dict(segment), "reported_segment_score"

    segment_metrics = segment.get("metrics")
    if not isinstance(segment_metrics, dict):
        return dict(segment), "reported_segment_score"

    try:
        issue = classify_vocal_segment_issue(segment_metrics, track_metrics=track_metrics)
    except (KeyError, TypeError, ValueError):
        return dict(segment), "reported_segment_score"

    reclassified = dict(segment)
    reclassified["reported_severity_score"] = float(segment.get("severity_score", 0.0))
    reclassified["reported_labels"] = list(segment.get("labels", []))
    reclassified["severity_score"] = issue["severity_score"]
    reclassified["labels"] = issue["labels"]
    reclassified["severity_parts"] = issue["severity_parts"]
    return reclassified, "shared_track_metrics"


def _build_issue_aware_window_decisions(
    *,
    base_score: VocalCandidateScore,
    candidate_scores: list[VocalCandidateScore],
    candidate_issue_reports: dict[str, dict[str, Any]],
    tuning: dict[str, float | int | str],
) -> list[dict[str, Any]]:
    base_report = candidate_issue_reports.get(base_score.profile_name)
    if base_report is None:
        return []
    base_segments = base_report.get("segments")
    if not isinstance(base_segments, list):
        return []

    decisions: list[dict[str, Any]] = []
    min_segment_issue_score = float(tuning["min_segment_issue_score"])
    min_adjusted_improvement = float(tuning["min_adjusted_improvement"])
    shared_track_metrics = dict(base_score.metrics)
    for segment_index, base_segment in enumerate(base_segments):
        if not isinstance(base_segment, dict):
            continue
        base_segment_for_scoring, base_selection_basis = _segment_for_shared_issue_scoring(
            base_segment,
            track_metrics=shared_track_metrics,
        )
        base_adjusted_score = _adjusted_segment_issue_score(
            segment=base_segment_for_scoring,
            candidate_score=base_score,
            base_score=base_score,
            tuning=tuning,
        )
        if base_adjusted_score < min_segment_issue_score:
            continue

        best_profile_name = base_score.profile_name
        best_segment = base_segment_for_scoring
        best_adjusted_score = base_adjusted_score
        best_selection_basis = base_selection_basis
        for candidate_score in candidate_scores:
            candidate_report = candidate_issue_reports.get(candidate_score.profile_name)
            if candidate_report is None:
                continue
            candidate_segment = _candidate_report_segment(candidate_report, segment_index, base_segment)
            if candidate_segment is None:
                continue
            candidate_segment_for_scoring, candidate_selection_basis = _segment_for_shared_issue_scoring(
                candidate_segment,
                track_metrics=shared_track_metrics,
            )
            adjusted_score = _adjusted_segment_issue_score(
                segment=candidate_segment_for_scoring,
                candidate_score=candidate_score,
                base_score=base_score,
                tuning=tuning,
            )
            if adjusted_score < best_adjusted_score:
                best_profile_name = candidate_score.profile_name
                best_segment = candidate_segment_for_scoring
                best_adjusted_score = adjusted_score
                best_selection_basis = candidate_selection_basis

        improvement = base_adjusted_score - best_adjusted_score
        if best_profile_name == base_score.profile_name or improvement < min_adjusted_improvement:
            continue
        decisions.append(
            {
                "segment_index": int(segment_index),
                "start_sec": float(base_segment["start_sec"]),
                "end_sec": float(base_segment["end_sec"]),
                "relative_start_sec": float(base_segment["relative_start_sec"]),
                "relative_end_sec": float(base_segment["relative_end_sec"]),
                "base_profile": base_score.profile_name,
                "selected_profile": best_profile_name,
                "selection_basis": best_selection_basis,
                "base_severity_score": float(base_segment_for_scoring.get("severity_score", 0.0)),
                "selected_severity_score": float(best_segment.get("severity_score", 0.0)),
                "base_reported_severity_score": float(base_segment.get("severity_score", 0.0)),
                "selected_reported_severity_score": float(best_segment.get("reported_severity_score", best_segment.get("severity_score", 0.0))),
                "base_adjusted_issue_score": round(base_adjusted_score, 6),
                "selected_adjusted_issue_score": round(best_adjusted_score, 6),
                "adjusted_improvement": round(improvement, 6),
                "base_labels": list(base_segment_for_scoring.get("labels", [])),
                "selected_labels": list(best_segment.get("labels", [])),
                "base_reported_labels": list(base_segment.get("labels", [])),
                "selected_reported_labels": list(best_segment.get("reported_labels", best_segment.get("labels", []))),
            }
        )
    return decisions


def _blend_candidate_segment(
    *,
    output_audio: np.ndarray,
    candidate_audio: np.ndarray,
    sample_rate: int,
    relative_start_sec: float,
    relative_end_sec: float,
    crossfade_sec: float,
) -> None:
    start_frame = max(0, int(round(float(relative_start_sec) * int(sample_rate))))
    end_frame = min(
        output_audio.shape[0],
        candidate_audio.shape[0],
        int(round(float(relative_end_sec) * int(sample_rate))),
    )
    if end_frame <= start_frame:
        return
    segment_length = end_frame - start_frame
    weight = np.ones(segment_length, dtype=np.float32)
    crossfade_frames = min(int(round(float(crossfade_sec) * int(sample_rate))), segment_length // 2)
    if crossfade_frames > 0:
        weight[:crossfade_frames] = np.linspace(0.0, 1.0, crossfade_frames, endpoint=False, dtype=np.float32)
        weight[-crossfade_frames:] = np.linspace(1.0, 0.0, crossfade_frames, endpoint=False, dtype=np.float32)
    segment_weight = weight[:, np.newaxis]
    output_audio[start_frame:end_frame] = (
        output_audio[start_frame:end_frame] * (1.0 - segment_weight)
        + candidate_audio[start_frame:end_frame] * segment_weight
    ).astype(np.float32)


def write_issue_aware_segment_selection_manifest(
    *,
    profile: VocalCandidateProfile,
    source_manifest: dict[str, Any],
    dependency_fingerprints: dict[str, dict[str, str]],
    output_dir: Path,
    output_vocals_path: Path,
    selection_debug: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "audio_path": source_manifest.get("audio_path"),
        "source_audio_sha256": source_manifest.get("source_audio_sha256"),
        "sample_rate": int(selection_debug["sample_rate"]),
        "export_subtype": "PCM_24",
        "preview_window_sec": source_manifest.get("preview_window_sec"),
        "pipeline": {
            "mode": "vocal_preview_derived_candidate",
            "candidate_mode": profile.candidate_mode,
            "base_profile": selection_debug["base_profile"],
            "dependency_fingerprints": dependency_fingerprints,
            "vocal_postprocess": "issue_aware_segment_selection_v1",
        },
        "tuning": profile.tuning,
        "stems": {
            "vocals": {
                "path": str(output_vocals_path),
                "vocal_postprocess": "issue_aware_segment_selection_v1",
                **selection_debug,
            }
        },
    }
    manifest_path = output_dir / "stem_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_issue_aware_segment_selection_candidate(
    *,
    profile: VocalCandidateProfile,
    base_score: VocalCandidateScore,
    candidate_scores: list[VocalCandidateScore],
    candidate_records_by_name: dict[str, VocalCandidateRecord],
    candidate_issue_reports: dict[str, dict[str, Any]],
    dependency_fingerprints: dict[str, dict[str, str]],
    output_dir: Path,
) -> dict[str, Any] | None:
    window_decisions = _build_issue_aware_window_decisions(
        base_score=base_score,
        candidate_scores=candidate_scores,
        candidate_issue_reports=candidate_issue_reports,
        tuning=profile.tuning,
    )
    if not window_decisions:
        return None

    base_record = candidate_records_by_name[base_score.profile_name]
    base_audio, base_sr = sf.read(str(base_record.vocals_path), always_2d=True, dtype="float32")
    selected_audio = base_audio.astype(np.float32).copy()
    candidate_audio_cache: dict[str, np.ndarray] = {base_score.profile_name: selected_audio}
    for decision in window_decisions:
        selected_profile = str(decision["selected_profile"])
        if selected_profile not in candidate_audio_cache:
            candidate_audio_cache[selected_profile] = _read_and_match_candidate_to_reference(
                selected_audio,
                int(base_sr),
                candidate_records_by_name[selected_profile].vocals_path,
            )
        _blend_candidate_segment(
            output_audio=selected_audio,
            candidate_audio=candidate_audio_cache[selected_profile],
            sample_rate=int(base_sr),
            relative_start_sec=float(decision["relative_start_sec"]),
            relative_end_sec=float(decision["relative_end_sec"]),
            crossfade_sec=float(profile.tuning["crossfade_sec"]),
        )

    selected_audio = _limit_peak(selected_audio)
    output_vocals_path = output_dir / "vocals.flac"
    output_vocals_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_vocals_path), selected_audio, int(base_sr), subtype="PCM_24")
    selected_profiles = sorted({str(decision["selected_profile"]) for decision in window_decisions})
    base_issue_summary = candidate_issue_reports.get(base_score.profile_name, {}).get("summary", {})
    selection_debug = {
        "sample_rate": int(base_sr),
        "base_profile": base_score.profile_name,
        "base_segment_issue_max_score": float(base_issue_summary.get("max_issue_score", 0.0)),
        "base_segment_issue_mean_top3_score": float(base_issue_summary.get("mean_top3_issue_score", 0.0)),
        "selected_profiles": selected_profiles,
        "window_decisions": window_decisions,
        "replaced_window_count": int(len(window_decisions)),
        "crossfade_sec": float(profile.tuning["crossfade_sec"]),
        **_audio_summary(selected_audio, int(base_sr)),
    }
    return write_issue_aware_segment_selection_manifest(
        profile=profile,
        source_manifest=base_record.manifest,
        dependency_fingerprints=dependency_fingerprints,
        output_dir=output_dir,
        output_vocals_path=output_vocals_path,
        selection_debug=selection_debug,
    )


VOCAL_RECOVERY_ISSUE_LABELS = frozenset(
    {
        "vocal_dropout",
        "vocal_tail_loss",
        "thin_or_overcleaned",
        "local_vocal_body_dip",
        "presence_loss",
        "air_detail_loss",
    }
)
BLEED_CONTROL_ISSUE_LABELS = frozenset(
    {
        "leaky_full_mix_risk",
        "low_band_bleed",
        "kick_punch_bleed",
        "unsupported_anchor_leakage",
        "high_band_unsupported_leakage",
        "high_noise_or_robotic_artifact",
        "excessive_reverb_tail_smear",
        "wide_stereo_non_vocal_leakage",
        "drop_melody_leakage",
    }
)
CLUSTERED_ISSUE_REPAIR_LABELS = frozenset(
    {
        "leaky_full_mix_risk",
        "low_band_bleed",
        "kick_punch_bleed",
        "high_noise_or_robotic_artifact",
        "wide_stereo_non_vocal_leakage",
        "drop_melody_leakage",
    }
)
NO_VOCAL_LEAKAGE_CLEANUP_LABELS = VOCAL_RECOVERY_ISSUE_LABELS | BLEED_CONTROL_ISSUE_LABELS
VOCAL_LOSS_PROTECTION_LABELS = frozenset(
    {
        "vocal_dropout",
        "vocal_tail_loss",
        "local_vocal_body_dip",
        "backing_vocal_loss_risk",
    }
)
NO_VOCAL_CLEANUP_PROTECTION_LABELS = VOCAL_LOSS_PROTECTION_LABELS | frozenset(
    {
        "excessive_reverb_tail_smear",
    }
)
FRAGILE_VOCAL_RECOVERY_LABELS = frozenset(
    {
        "vocal_dropout",
        "vocal_tail_loss",
        "thin_or_overcleaned",
        "presence_loss",
        "air_detail_loss",
    }
)
ISSUE_ACTION_TARGETS: dict[str, tuple[str, ...]] = {
    "vocal_dropout": ("segment_issue_repair", "issue_aware_segment_selection"),
    "vocal_tail_loss": ("segment_issue_repair", "issue_aware_segment_selection"),
    "local_vocal_body_dip": ("segment_issue_repair", "issue_aware_segment_selection"),
    "thin_or_overcleaned": (
        "segment_issue_repair",
        "no_vocal_leakage_cleanup",
        "issue_aware_segment_selection",
    ),
    "presence_loss": (
        "segment_issue_repair",
        "no_vocal_leakage_cleanup",
        "issue_aware_segment_selection",
    ),
    "air_detail_loss": (
        "segment_issue_repair",
        "no_vocal_leakage_cleanup",
        "vocal_harshness_guard",
        "issue_aware_segment_selection",
    ),
    "leaky_full_mix_risk": ("segment_issue_repair", "issue_aware_segment_selection"),
    "low_band_bleed": ("segment_issue_repair", "issue_aware_segment_selection"),
    "kick_punch_bleed": ("segment_issue_repair", "issue_aware_segment_selection"),
    "unsupported_anchor_leakage": ("vocal_harshness_guard", "segment_issue_repair", "issue_aware_segment_selection"),
    "high_band_unsupported_leakage": ("vocal_harshness_guard", "segment_issue_repair", "issue_aware_segment_selection"),
    "high_noise_or_robotic_artifact": ("vocal_harshness_guard", "segment_issue_repair", "issue_aware_segment_selection"),
    "excessive_reverb_tail_smear": (
        "vocal_harshness_guard",
        "segment_issue_repair",
        "issue_aware_segment_selection",
    ),
    "wide_stereo_non_vocal_leakage": ("segment_issue_repair", "issue_aware_segment_selection"),
    "drop_melody_leakage": (
        "clustered_issue_window_ratio_repair",
        "segment_issue_repair",
        "issue_aware_segment_selection",
    ),
}
VOCAL_DEFECT_ACTION_FAMILIES: dict[str, str] = {
    "vocal_dropout": "vocal_presence_reconstruction",
    "vocal_tail_loss": "vocal_presence_reconstruction",
    "local_vocal_body_dip": "vocal_body_recovery",
    "thin_or_overcleaned": "vocal_body_recovery",
    "presence_loss": "vocal_body_recovery",
    "air_detail_loss": "vocal_air_detail_recovery",
    "leaky_full_mix_risk": "full_mix_leakage_rejection",
    "low_band_bleed": "low_band_bleed_rejection",
    "kick_punch_bleed": "percussive_bleed_rejection",
    "unsupported_anchor_leakage": "unsupported_leakage_rejection",
    "high_band_unsupported_leakage": "unsupported_leakage_rejection",
    "high_noise_or_robotic_artifact": "artifact_rejection",
    "excessive_reverb_tail_smear": "reverb_smear_rejection",
    "wide_stereo_non_vocal_leakage": "stereo_non_vocal_leakage_rejection",
    "drop_melody_leakage": "melody_leakage_rejection",
    "backing_vocal_loss_risk": "protect_vocal_content",
}
LIVE_GPU_VOCAL_LADDER_MODEL_IDS: tuple[str, ...] = (
    "audio_separator_bs_roformer_vocals",
    "audio_separator_melband_roformer_viperx_vocals",
    "mdx23c_vocals",
    "scnet_xl_ihf_musdb18",
    "melband_roformer_vocals",
    "htdemucs_ft",
    "htdemucs_6s",
    "hybrid_htdemucs_ft_bs_roformer_6s",
)
UNSTABLE_VOCAL_CONSISTENCY_BLOCKER = "selected_candidate_has_unstable_vocal_consistency"
ISSUE_PROTECTIVE_REJECTION_REASONS: dict[str, str] = {
    "backing_vocal_loss_risk": (
        "Backing vocals are vocal content; without isolated backing-vocal ground truth this label "
        "is diagnostic/protective and must not trigger suppression by itself."
    ),
}


def _segment_issue_part_score(segment: dict[str, Any], label_names: frozenset[str]) -> float:
    severity_parts = segment.get("severity_parts")
    if not isinstance(severity_parts, dict):
        return 0.0
    return float(
        sum(
            float(severity_parts.get(label_name, 0.0))
            for label_name in label_names
        )
    )


def _segment_issue_repair_metrics(manifest: dict[str, Any]) -> dict[str, float]:
    vocals = manifest.get("stems", {}).get("vocals", {})
    if not isinstance(vocals, dict):
        return {}
    window_decisions = vocals.get("window_decisions")
    if not isinstance(window_decisions, list):
        return {}
    improvements = [
        float(decision.get("recovery_improvement", 0.0))
        for decision in window_decisions
        if isinstance(decision, dict)
    ]
    issue_improvements = [
        float(decision.get("issue_improvement", decision.get("recovery_improvement", 0.0)))
        for decision in window_decisions
        if isinstance(decision, dict)
    ]
    return {
        "segment_issue_repair_window_count": float(len(improvements)),
        "segment_issue_repair_recovery_improvement_sum": round(float(sum(improvements)), 6),
        "segment_issue_repair_recovery_improvement_max": round(float(max(improvements)) if improvements else 0.0, 6),
        "segment_issue_repair_issue_improvement_sum": round(float(sum(issue_improvements)), 6),
        "segment_issue_repair_issue_improvement_max": round(
            float(max(issue_improvements)) if issue_improvements else 0.0,
            6,
        ),
        "segment_issue_repair_base_segment_issue_max_score": float(vocals.get("base_segment_issue_max_score", 0.0)),
        "segment_issue_repair_base_segment_issue_mean_top3_score": float(
            vocals.get("base_segment_issue_mean_top3_score", 0.0)
        ),
    }


def _segment_issue_repair_router_tuning(
    base_tuning: dict[str, float | int | str],
    labels: list[str],
) -> dict[str, float | int | str]:
    label_set = set(labels)
    tuning = dict(base_tuning)
    has_vocal_recovery_issue = bool(label_set & VOCAL_RECOVERY_ISSUE_LABELS)
    has_bleed_control_issue = bool(label_set & BLEED_CONTROL_ISSUE_LABELS)
    has_kick_or_low_bleed = bool({"low_band_bleed", "kick_punch_bleed"} & label_set)
    has_high_noise = bool(
        {
            "high_band_unsupported_leakage",
            "high_noise_or_robotic_artifact",
            "excessive_reverb_tail_smear",
            "wide_stereo_non_vocal_leakage",
        }
        & label_set
    )
    tuning["body_repair_strength"] = float(base_tuning["body_repair_strength"])
    tuning["preserve_tail_strength"] = float(base_tuning["preserve_tail_strength"])
    tuning["core_debleed_strength"] = 0.0
    if has_bleed_control_issue:
        if has_vocal_recovery_issue:
            tuning["core_debleed_strength"] = float(base_tuning["combined_core_debleed_strength"])
        else:
            tuning["body_repair_strength"] = 0.0
            tuning["preserve_tail_strength"] = 0.0
            tuning["core_debleed_strength"] = float(base_tuning["bleed_core_debleed_strength"])
    if has_vocal_recovery_issue:
        tuning["body_support_floor"] = float(base_tuning["dropout_body_support_floor"])
        tuning["unsupported_busy_damping"] = float(base_tuning["dropout_unsupported_busy_damping"])
        if {"vocal_dropout", "vocal_tail_loss"} & label_set and has_bleed_control_issue:
            tuning["core_debleed_strength"] = min(
                float(tuning["core_debleed_strength"]),
                float(base_tuning["dropout_core_debleed_strength"]),
            )
    if has_kick_or_low_bleed:
        tuning["core_unsupported_tolerance"] = float(base_tuning["kick_core_unsupported_tolerance"])
        tuning["core_unsupported_damping"] = max(float(base_tuning["core_unsupported_damping"]), 1.0)
        if not has_vocal_recovery_issue:
            tuning["unsupported_busy_damping"] = max(float(base_tuning["unsupported_busy_damping"]), 0.96)
    if has_high_noise:
        tuning["strict_high_strength"] = max(float(base_tuning["strict_high_strength"]), 1.0)
        tuning["strict_high_tolerance"] = min(float(base_tuning["strict_high_tolerance"]), 1.0)
        if not has_vocal_recovery_issue:
            tuning["core_unsupported_damping"] = max(float(tuning["core_unsupported_damping"]), 1.0)
    return tuning


def _segment_issue_repair_candidate_score(
    *,
    segment: dict[str, Any],
    candidate_score: VocalCandidateScore,
    base_score: VocalCandidateScore,
    labels: list[str],
    tuning: dict[str, float | int | str],
) -> float:
    label_set = set(labels)
    recovery_weight = 1.0 if label_set & VOCAL_RECOVERY_ISSUE_LABELS else 0.25
    bleed_weight = 0.25 if label_set & BLEED_CONTROL_ISSUE_LABELS else 0.0
    score_drop = max(0.0, float(base_score.score) - float(candidate_score.score))
    return (
        _segment_issue_part_score(segment, VOCAL_RECOVERY_ISSUE_LABELS) * recovery_weight
        + _segment_issue_part_score(segment, BLEED_CONTROL_ISSUE_LABELS) * bleed_weight
        + score_drop * float(tuning["candidate_score_penalty_weight"])
    )


def _build_segment_issue_repair_window_decisions(
    *,
    base_score: VocalCandidateScore,
    candidate_scores: list[VocalCandidateScore],
    candidate_issue_reports: dict[str, dict[str, Any]],
    tuning: dict[str, float | int | str],
) -> list[dict[str, Any]]:
    base_report = candidate_issue_reports.get(base_score.profile_name)
    if base_report is None:
        return []
    base_segments = base_report.get("segments")
    if not isinstance(base_segments, list):
        return []

    decisions: list[dict[str, Any]] = []
    min_segment_issue_score = float(tuning["min_segment_issue_score"])
    min_bleed_issue_score = float(tuning["min_bleed_issue_score"])
    min_recovery_improvement = float(tuning["min_recovery_improvement"])
    for segment_index, base_segment in enumerate(base_segments):
        if not isinstance(base_segment, dict):
            continue
        labels = [str(label) for label in base_segment.get("labels", [])]
        label_set = set(labels)
        if not (label_set & (VOCAL_RECOVERY_ISSUE_LABELS | BLEED_CONTROL_ISSUE_LABELS)):
            continue
        has_vocal_recovery_issue = bool(label_set & VOCAL_RECOVERY_ISSUE_LABELS)
        has_bleed_control_issue = bool(label_set & BLEED_CONTROL_ISSUE_LABELS)
        recovery_issue_score = _segment_issue_part_score(base_segment, VOCAL_RECOVERY_ISSUE_LABELS)
        bleed_issue_score = _segment_issue_part_score(base_segment, BLEED_CONTROL_ISSUE_LABELS)
        recovery_eligible = has_vocal_recovery_issue and float(base_segment.get("severity_score", 0.0)) >= min_segment_issue_score
        bleed_eligible = has_bleed_control_issue and bleed_issue_score >= min_bleed_issue_score
        if not (recovery_eligible or bleed_eligible):
            continue

        base_repair_score = _segment_issue_repair_candidate_score(
            segment=base_segment,
            candidate_score=base_score,
            base_score=base_score,
            labels=labels,
            tuning=tuning,
        )
        best_profile_name = base_score.profile_name
        best_segment = base_segment
        best_repair_score = base_repair_score
        for candidate_score in candidate_scores:
            if candidate_score.profile_name == base_score.profile_name:
                continue
            candidate_report = candidate_issue_reports.get(candidate_score.profile_name)
            if candidate_report is None:
                continue
            candidate_segment = _candidate_report_segment(candidate_report, segment_index, base_segment)
            if candidate_segment is None:
                continue
            candidate_repair_score = _segment_issue_repair_candidate_score(
                segment=candidate_segment,
                candidate_score=candidate_score,
                base_score=base_score,
                labels=labels,
                tuning=tuning,
            )
            if candidate_repair_score < best_repair_score:
                best_profile_name = candidate_score.profile_name
                best_segment = candidate_segment
                best_repair_score = candidate_repair_score

        recovery_improvement = max(0.0, base_repair_score - best_repair_score)
        if recovery_eligible and not bleed_eligible and recovery_improvement < min_recovery_improvement:
            continue
        if (
            recovery_eligible
            and bleed_eligible
            and (label_set & FRAGILE_VOCAL_RECOVERY_LABELS)
            and best_profile_name == base_score.profile_name
            and recovery_improvement < min_recovery_improvement
        ):
            continue
        issue_improvement = recovery_improvement
        if bleed_eligible:
            issue_improvement = max(issue_improvement, max(0.0, bleed_issue_score - _segment_issue_part_score(best_segment, BLEED_CONTROL_ISSUE_LABELS)))
        if recovery_eligible and bleed_eligible:
            resolution_kind = "combined_recovery_and_bleed_control"
        elif recovery_eligible:
            resolution_kind = "vocal_recovery"
        else:
            resolution_kind = "bleed_control"

        decisions.append(
            {
                "segment_index": int(segment_index),
                "start_sec": float(base_segment["start_sec"]),
                "end_sec": float(base_segment["end_sec"]),
                "relative_start_sec": float(base_segment["relative_start_sec"]),
                "relative_end_sec": float(base_segment["relative_end_sec"]),
                "base_profile": base_score.profile_name,
                "repair_profile": best_profile_name,
                "base_severity_score": float(base_segment.get("severity_score", 0.0)),
                "repair_severity_score": float(best_segment.get("severity_score", 0.0)),
                "base_repair_score": round(float(base_repair_score), 6),
                "repair_score": round(float(best_repair_score), 6),
                "recovery_improvement": round(float(recovery_improvement), 6),
                "issue_improvement": round(float(issue_improvement), 6),
                "recovery_issue_score": round(float(recovery_issue_score), 6),
                "bleed_issue_score": round(float(bleed_issue_score), 6),
                "resolution_kind": resolution_kind,
                "base_labels": labels,
                "repair_labels": list(best_segment.get("labels", [])),
                "router_tuning": _segment_issue_repair_router_tuning(tuning, labels),
            }
        )
    return decisions


def write_segment_issue_repair_candidate_manifest(
    *,
    profile: VocalCandidateProfile,
    source_manifest: dict[str, Any],
    dependency_fingerprints: dict[str, dict[str, str]],
    output_dir: Path,
    output_vocals_path: Path,
    repair_debug: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "audio_path": source_manifest.get("audio_path"),
        "source_audio_sha256": source_manifest.get("source_audio_sha256"),
        "sample_rate": int(repair_debug["sample_rate"]),
        "export_subtype": "PCM_24",
        "preview_window_sec": source_manifest.get("preview_window_sec"),
        "pipeline": {
            "mode": "vocal_preview_derived_candidate",
            "candidate_mode": profile.candidate_mode,
            "base_profile": repair_debug["base_profile"],
            "strict_profile": repair_debug["strict_profile"],
            "anchor_profile": repair_debug["anchor_profile"],
            "dependency_fingerprints": dependency_fingerprints,
            "vocal_postprocess": "segment_issue_repair_v1",
        },
        "tuning": profile.tuning,
        "stems": {
            "vocals": {
                "path": str(output_vocals_path),
                "vocal_postprocess": "segment_issue_repair_v1",
                **repair_debug,
            }
        },
    }
    manifest_path = output_dir / "stem_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _preferred_record(
    candidate_records_by_name: dict[str, VocalCandidateRecord],
    names: tuple[str, ...],
) -> VocalCandidateRecord | None:
    for name in names:
        record = candidate_records_by_name.get(name)
        if record is not None:
            return record
    return None


def _issue_summary_float(summary: dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(summary.get(key, default))
    except (TypeError, ValueError):
        return math.inf
    if not math.isfinite(value):
        return math.inf
    return value


def _issue_report_support_score(report: dict[str, Any] | None) -> float:
    if not isinstance(report, dict):
        return math.inf
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        return math.inf

    max_issue = _issue_summary_float(summary, "max_issue_score", math.inf)
    mean_top3 = _issue_summary_float(summary, "mean_top3_issue_score", max_issue)
    issue_window_ratio = _issue_summary_float(summary, "issue_window_ratio", 0.0)
    return max_issue + mean_top3 * 0.35 + issue_window_ratio * 0.10


def _candidate_defect_selection_key(
    candidate_score: VocalCandidateScore,
    candidate_issue_reports: dict[str, dict[str, Any]],
) -> tuple[int, float, float, float, float]:
    report = candidate_issue_reports.get(candidate_score.profile_name)
    score_tiebreaker = -float(candidate_score.score)
    if not isinstance(report, dict):
        return (1, math.inf, math.inf, math.inf, score_tiebreaker)
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        return (1, math.inf, math.inf, math.inf, score_tiebreaker)

    max_issue = _issue_summary_float(summary, "max_issue_score", math.inf)
    if not math.isfinite(max_issue):
        return (1, math.inf, math.inf, math.inf, score_tiebreaker)
    mean_top3 = _issue_summary_float(summary, "mean_top3_issue_score", max_issue)
    issue_window_ratio = _issue_summary_float(summary, "issue_window_ratio", 0.0)
    return (0, max_issue, mean_top3, issue_window_ratio, score_tiebreaker)


def select_defect_aware_candidate_score(
    candidate_scores: list[VocalCandidateScore],
    candidate_issue_reports: dict[str, dict[str, Any]],
) -> VocalCandidateScore:
    if not candidate_scores:
        raise ValueError("no candidate scores to select")
    return min(
        candidate_scores,
        key=lambda candidate_score: _candidate_defect_selection_key(candidate_score, candidate_issue_reports),
    )


def _select_segment_repair_support_record(
    *,
    candidate_records_by_name: dict[str, VocalCandidateRecord],
    candidate_issue_reports: dict[str, dict[str, Any]],
    base_profile_name: str,
    preferred_names: tuple[str, ...],
) -> tuple[VocalCandidateRecord | None, str]:
    preferred_record = _preferred_record(candidate_records_by_name, preferred_names)
    if preferred_record is not None:
        return preferred_record, "preferred_named_support"

    scored_records: list[tuple[float, str, VocalCandidateRecord]] = []
    for profile_name, record in candidate_records_by_name.items():
        if profile_name == base_profile_name:
            continue
        support_score = _issue_report_support_score(candidate_issue_reports.get(profile_name))
        if not math.isfinite(support_score):
            continue
        scored_records.append((support_score, profile_name, record))
    if not scored_records:
        return None, "missing_support"
    _, _, selected_record = min(scored_records, key=lambda item: (item[0], item[1]))
    return selected_record, "adaptive_low_issue_support"


def build_segment_issue_repair_candidate(
    *,
    profile: VocalCandidateProfile,
    source_audio_path: Path,
    start_sec: float,
    end_sec: float | None,
    base_score: VocalCandidateScore,
    candidate_scores: list[VocalCandidateScore],
    candidate_records_by_name: dict[str, VocalCandidateRecord],
    candidate_issue_reports: dict[str, dict[str, Any]],
    dependency_fingerprints: dict[str, dict[str, str]],
    output_dir: Path,
    progress: Callable[..., Any] | None = None,
) -> dict[str, Any] | None:
    window_decisions = _build_segment_issue_repair_window_decisions(
        base_score=base_score,
        candidate_scores=candidate_scores,
        candidate_issue_reports=candidate_issue_reports,
        tuning=profile.tuning,
    )
    if progress is not None:
        progress(
            status="decisions_ready",
            base_profile=base_score.profile_name,
            window_decision_count=len(window_decisions),
        )
    if not window_decisions:
        return None

    base_record = candidate_records_by_name[base_score.profile_name]
    strict_record, strict_record_source = _select_segment_repair_support_record(
        candidate_records_by_name=candidate_records_by_name,
        candidate_issue_reports=candidate_issue_reports,
        base_profile_name=base_score.profile_name,
        preferred_names=("ft_tail_safe_anchor_strict", "bs_artifact_clean"),
    )
    anchor_record, anchor_record_source = _select_segment_repair_support_record(
        candidate_records_by_name=candidate_records_by_name,
        candidate_issue_reports=candidate_issue_reports,
        base_profile_name=base_score.profile_name,
        preferred_names=("bs_artifact_clean", "ft_tail_safe_anchor_clean"),
    )
    if strict_record is None or anchor_record is None:
        return None

    base_audio, base_sr = sf.read(str(base_record.vocals_path), always_2d=True, dtype="float32")
    repaired_audio = base_audio.astype(np.float32).copy()
    intermediate_dir = output_dir / "intermediate_router_outputs"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    intermediate_outputs: list[dict[str, Any]] = []

    total_windows = len(window_decisions)
    for window_index, decision in enumerate(window_decisions, start=1):
        repair_profile_name = str(decision["repair_profile"])
        repair_record = candidate_records_by_name.get(repair_profile_name, base_record)
        intermediate_path = intermediate_dir / f"segment_{int(decision['segment_index']):03d}_{repair_profile_name}.flac"
        if progress is not None:
            progress(
                status="window_start",
                window_index=window_index,
                total_windows=total_windows,
                segment_index=int(decision["segment_index"]),
                repair_profile=repair_profile_name,
                start_sec=float(decision["start_sec"]),
                end_sec=float(decision["end_sec"]),
            )
        router_debug = apply_segment_adaptive_vocal_router(
            source_audio_path=source_audio_path,
            start_sec=start_sec,
            end_sec=end_sec,
            clean_vocals_path=base_record.vocals_path,
            strict_vocals_path=strict_record.vocals_path,
            repair_vocals_path=repair_record.vocals_path,
            preserve_vocals_path=repair_record.vocals_path,
            anchor_vocals_path=anchor_record.vocals_path,
            output_vocals_path=intermediate_path,
            tuning=dict(decision["router_tuning"]),
        )
        routed_audio = _read_and_match_candidate_to_reference(repaired_audio, int(base_sr), intermediate_path)
        _blend_candidate_segment(
            output_audio=repaired_audio,
            candidate_audio=routed_audio,
            sample_rate=int(base_sr),
            relative_start_sec=float(decision["relative_start_sec"]),
            relative_end_sec=float(decision["relative_end_sec"]),
            crossfade_sec=float(profile.tuning["crossfade_sec"]),
        )
        intermediate_outputs.append(
            {
                "segment_index": int(decision["segment_index"]),
                "repair_profile": repair_profile_name,
                "path": str(intermediate_path),
                "router_debug": router_debug,
            }
        )
        if progress is not None:
            progress(
                status="window_finish",
                window_index=window_index,
                total_windows=total_windows,
                segment_index=int(decision["segment_index"]),
                repair_profile=repair_profile_name,
                intermediate_path=str(intermediate_path),
            )

    repaired_audio = _limit_peak(repaired_audio)
    output_vocals_path = output_dir / "vocals.flac"
    output_vocals_path.parent.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress(status="write_output_start", output_vocals_path=str(output_vocals_path))
    sf.write(str(output_vocals_path), repaired_audio, int(base_sr), subtype="PCM_24")
    if progress is not None:
        progress(status="write_output_finish", output_vocals_path=str(output_vocals_path))
    base_issue_summary = candidate_issue_reports.get(base_score.profile_name, {}).get("summary", {})
    repair_debug = {
        "sample_rate": int(base_sr),
        "base_profile": base_score.profile_name,
        "strict_profile": strict_record.profile.name,
        "anchor_profile": anchor_record.profile.name,
        "base_segment_issue_max_score": float(base_issue_summary.get("max_issue_score", 0.0)),
        "base_segment_issue_mean_top3_score": float(base_issue_summary.get("mean_top3_issue_score", 0.0)),
        "window_decisions": window_decisions,
        "repaired_window_count": int(len(window_decisions)),
        "crossfade_sec": float(profile.tuning["crossfade_sec"]),
        "support_selection": {
            "strict_profile_source": strict_record_source,
            "anchor_profile_source": anchor_record_source,
        },
        "intermediate_outputs": intermediate_outputs,
        **_audio_summary(repaired_audio, int(base_sr)),
    }
    return write_segment_issue_repair_candidate_manifest(
        profile=profile,
        source_manifest=base_record.manifest,
        dependency_fingerprints=dependency_fingerprints,
        output_dir=output_dir,
        output_vocals_path=output_vocals_path,
        repair_debug=repair_debug,
    )


def _build_gate_blocker_best_segment_splice_window_decisions(
    *,
    base_score: VocalCandidateScore,
    candidate_scores: list[VocalCandidateScore],
    candidate_issue_reports: dict[str, dict[str, Any]],
    defect_effectiveness_gate: dict[str, Any],
    tuning: dict[str, float | int | str],
) -> list[dict[str, Any]]:
    base_report = candidate_issue_reports.get(base_score.profile_name)
    if base_report is None:
        return []
    base_segments = base_report.get("segments")
    if not isinstance(base_segments, list):
        return []
    gate_blockers = defect_effectiveness_gate.get("blockers")
    if not isinstance(gate_blockers, list):
        return []

    blocker_by_window = {
        _window_key(blocker): blocker
        for blocker in gate_blockers
        if isinstance(blocker, dict)
        and blocker.get("reason") == "selected_candidate_still_has_severe_issue"
        and "start_sec" in blocker
        and "end_sec" in blocker
    }
    if not blocker_by_window:
        return []

    decisions: list[dict[str, Any]] = []
    min_blocker_severity_score = float(tuning["min_blocker_severity_score"])
    min_donor_severity_improvement = float(tuning["min_donor_severity_improvement"])
    shared_track_metrics = dict(base_score.metrics)
    for segment_index, base_segment in enumerate(base_segments):
        if not isinstance(base_segment, dict):
            continue
        blocker = blocker_by_window.get(_window_key(base_segment))
        if blocker is None:
            continue
        base_segment_for_scoring, base_selection_basis = _segment_for_shared_issue_scoring(
            base_segment,
            track_metrics=shared_track_metrics,
        )
        base_severity = float(base_segment_for_scoring.get("severity_score", 0.0))
        if base_severity < min_blocker_severity_score:
            continue

        best_score: VocalCandidateScore | None = None
        best_segment: dict[str, Any] | None = None
        best_severity = base_severity
        best_selection_basis = base_selection_basis
        for candidate_score in candidate_scores:
            if candidate_score.profile_name == base_score.profile_name:
                continue
            candidate_report = candidate_issue_reports.get(candidate_score.profile_name)
            if candidate_report is None:
                continue
            candidate_segment = _candidate_report_segment(candidate_report, segment_index, base_segment)
            if candidate_segment is None:
                continue
            candidate_segment_for_scoring, candidate_selection_basis = _segment_for_shared_issue_scoring(
                candidate_segment,
                track_metrics=shared_track_metrics,
            )
            candidate_severity = float(candidate_segment_for_scoring.get("severity_score", 0.0))
            if candidate_severity < best_severity:
                best_score = candidate_score
                best_segment = candidate_segment_for_scoring
                best_severity = candidate_severity
                best_selection_basis = candidate_selection_basis

        if best_score is None or best_segment is None:
            continue
        severity_improvement = base_severity - best_severity
        if severity_improvement < min_donor_severity_improvement:
            continue

        decisions.append(
            {
                "segment_index": int(segment_index),
                "start_sec": float(base_segment["start_sec"]),
                "end_sec": float(base_segment["end_sec"]),
                "relative_start_sec": float(base_segment["relative_start_sec"]),
                "relative_end_sec": float(base_segment["relative_end_sec"]),
                "base_profile": base_score.profile_name,
                "donor_profile": best_score.profile_name,
                "base_severity_score": round(float(base_severity), 6),
                "donor_severity_score": round(float(best_severity), 6),
                "base_reported_severity_score": round(float(base_segment.get("severity_score", 0.0)), 6),
                "donor_reported_severity_score": round(
                    float(best_segment.get("reported_severity_score", best_segment.get("severity_score", 0.0))),
                    6,
                ),
                "severity_improvement": round(float(severity_improvement), 6),
                "blocker_reason": str(blocker.get("reason", "")),
                "base_selection_basis": base_selection_basis,
                "donor_selection_basis": best_selection_basis,
                "base_labels": [str(label) for label in base_segment_for_scoring.get("labels", [])],
                "donor_labels": [str(label) for label in best_segment.get("labels", [])],
                "base_reported_labels": [str(label) for label in base_segment.get("labels", [])],
                "donor_reported_labels": [
                    str(label)
                    for label in best_segment.get("reported_labels", best_segment.get("labels", []))
                ],
                "splice_policy": "direct_best_measured_donor_for_gate_blocker",
            }
        )
    return decisions


def write_gate_blocker_best_segment_splice_candidate_manifest(
    *,
    profile: VocalCandidateProfile,
    source_manifest: dict[str, Any],
    dependency_fingerprints: dict[str, dict[str, str]],
    output_dir: Path,
    output_vocals_path: Path,
    splice_debug: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "audio_path": source_manifest.get("audio_path"),
        "source_audio_sha256": source_manifest.get("source_audio_sha256"),
        "sample_rate": int(splice_debug["sample_rate"]),
        "export_subtype": "PCM_24",
        "preview_window_sec": source_manifest.get("preview_window_sec"),
        "pipeline": {
            "mode": "vocal_preview_derived_candidate",
            "candidate_mode": profile.candidate_mode,
            "base_profile": splice_debug["base_profile"],
            "dependency_fingerprints": dependency_fingerprints,
            "vocal_postprocess": "gate_blocker_best_segment_splice_v1",
            "target_reference_assisted": False,
            "target_reference_used_at_application": False,
            "application_reference_free": True,
            "fallback_used": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
        },
        "tuning": profile.tuning,
        "stems": {
            "vocals": {
                "path": str(output_vocals_path),
                "vocal_postprocess": "gate_blocker_best_segment_splice_v1",
                **splice_debug,
            }
        },
    }
    manifest_path = output_dir / "stem_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _gate_blocker_best_segment_splice_metrics(manifest: dict[str, Any]) -> dict[str, float]:
    vocals = manifest.get("stems", {}).get("vocals", {})
    if not isinstance(vocals, dict):
        return {}
    decisions = vocals.get("window_decisions")
    if not isinstance(decisions, list):
        return {}
    improvements = [
        float(decision.get("severity_improvement", 0.0))
        for decision in decisions
        if isinstance(decision, dict)
    ]
    return {
        "gate_blocker_best_segment_splice_window_count": float(len(decisions)),
        "gate_blocker_best_segment_splice_improvement_sum": round(float(sum(improvements)), 6),
        "gate_blocker_best_segment_splice_improvement_max": round(
            float(max(improvements)) if improvements else 0.0,
            6,
        ),
        "gate_blocker_best_segment_splice_base_segment_issue_max_score": float(
            vocals.get("base_segment_issue_max_score", 0.0),
        ),
        "gate_blocker_best_segment_splice_base_segment_issue_mean_top3_score": float(
            vocals.get("base_segment_issue_mean_top3_score", 0.0),
        ),
    }


def build_gate_blocker_best_segment_splice_candidate(
    *,
    profile: VocalCandidateProfile,
    base_score: VocalCandidateScore,
    candidate_scores: list[VocalCandidateScore],
    candidate_records_by_name: dict[str, VocalCandidateRecord],
    candidate_issue_reports: dict[str, dict[str, Any]],
    defect_effectiveness_gate: dict[str, Any],
    dependency_fingerprints: dict[str, dict[str, str]],
    output_dir: Path,
) -> dict[str, Any] | None:
    window_decisions = _build_gate_blocker_best_segment_splice_window_decisions(
        base_score=base_score,
        candidate_scores=candidate_scores,
        candidate_issue_reports=candidate_issue_reports,
        defect_effectiveness_gate=defect_effectiveness_gate,
        tuning=profile.tuning,
    )
    if not window_decisions:
        return None

    base_record = candidate_records_by_name.get(base_score.profile_name)
    if base_record is None:
        return None
    base_audio, base_sr = sf.read(str(base_record.vocals_path), always_2d=True, dtype="float32")
    spliced_audio = base_audio.astype(np.float32).copy()
    donor_audio_by_profile: dict[str, np.ndarray] = {}
    applied_decisions: list[dict[str, Any]] = []
    for decision in window_decisions:
        donor_profile_name = str(decision["donor_profile"])
        donor_record = candidate_records_by_name.get(donor_profile_name)
        if donor_record is None:
            continue
        donor_audio = donor_audio_by_profile.get(donor_profile_name)
        if donor_audio is None:
            donor_audio = _read_and_match_candidate_to_reference(spliced_audio, int(base_sr), donor_record.vocals_path)
            donor_audio_by_profile[donor_profile_name] = donor_audio
        _blend_candidate_segment(
            output_audio=spliced_audio,
            candidate_audio=donor_audio,
            sample_rate=int(base_sr),
            relative_start_sec=float(decision["relative_start_sec"]),
            relative_end_sec=float(decision["relative_end_sec"]),
            crossfade_sec=float(profile.tuning["crossfade_sec"]),
        )
        applied_decisions.append(decision)

    if not applied_decisions:
        return None

    spliced_audio = _limit_peak(spliced_audio)
    output_vocals_path = output_dir / "vocals.flac"
    output_vocals_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_vocals_path), spliced_audio, int(base_sr), subtype="PCM_24")
    base_issue_summary = candidate_issue_reports.get(base_score.profile_name, {}).get("summary", {})
    splice_debug = {
        "sample_rate": int(base_sr),
        "base_profile": base_score.profile_name,
        "base_segment_issue_max_score": float(base_issue_summary.get("max_issue_score", 0.0)),
        "base_segment_issue_mean_top3_score": float(base_issue_summary.get("mean_top3_issue_score", 0.0)),
        "window_decisions": applied_decisions,
        "spliced_window_count": int(len(applied_decisions)),
        "crossfade_sec": float(profile.tuning["crossfade_sec"]),
        "donor_profiles": sorted({str(decision["donor_profile"]) for decision in applied_decisions}),
        **_audio_summary(spliced_audio, int(base_sr)),
    }
    return write_gate_blocker_best_segment_splice_candidate_manifest(
        profile=profile,
        source_manifest=base_record.manifest,
        dependency_fingerprints=dependency_fingerprints,
        output_dir=output_dir,
        output_vocals_path=output_vocals_path,
        splice_debug=splice_debug,
    )


def _band_limited_stft_component(
    audio: np.ndarray,
    sample_rate: int,
    *,
    low_hz: float,
    high_hz: float,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, np.newaxis]
    if array.size == 0 or array.shape[0] == 0:
        return np.zeros_like(array, dtype=np.float32)
    frame_count = int(array.shape[0])
    effective_n_fft = min(max(64, int(n_fft)), frame_count)
    effective_hop = min(max(1, int(hop_length)), max(1, effective_n_fft // 2))
    freqs = librosa.fft_frequencies(sr=int(sample_rate), n_fft=effective_n_fft)
    band_mask = (freqs >= float(low_hz)) & (freqs < float(high_hz))
    if not np.any(band_mask):
        return np.zeros_like(array, dtype=np.float32)

    components: list[np.ndarray] = []
    for channel_index in range(array.shape[1]):
        channel_stft = librosa.stft(
            array[:, channel_index],
            n_fft=effective_n_fft,
            hop_length=effective_hop,
        )
        filtered_stft = np.zeros_like(channel_stft)
        filtered_stft[band_mask, :] = channel_stft[band_mask, :]
        component = librosa.istft(
            filtered_stft,
            hop_length=effective_hop,
            length=frame_count,
        )
        components.append(component.astype(np.float32))
    return np.stack(components, axis=1).astype(np.float32)


def _presence_band_recovery_audio(
    *,
    base_audio: np.ndarray,
    donor_audio: np.ndarray,
    sample_rate: int,
    tuning: dict[str, float | int | str],
) -> np.ndarray:
    frame_count = min(int(base_audio.shape[0]), int(donor_audio.shape[0]))
    channel_count = min(int(base_audio.shape[1]), int(donor_audio.shape[1]))
    base = base_audio[:frame_count, :channel_count].astype(np.float32)
    donor = donor_audio[:frame_count, :channel_count].astype(np.float32)
    n_fft = int(tuning["band_filter_n_fft"])
    hop_length = int(tuning["band_filter_hop_length"])
    donor_presence_delta = _band_limited_stft_component(
        donor - base,
        sample_rate,
        low_hz=float(tuning["presence_low_hz"]),
        high_hz=float(tuning["presence_high_hz"]),
        n_fft=n_fft,
        hop_length=hop_length,
    )
    base_bleed_band = _band_limited_stft_component(
        base,
        sample_rate,
        low_hz=float(tuning["bleed_low_hz"]),
        high_hz=float(tuning["bleed_high_hz"]),
        n_fft=n_fft,
        hop_length=hop_length,
    )
    recovered = (
        base
        + donor_presence_delta * float(tuning["presence_delta_gain"])
        + base_bleed_band * (float(tuning["bleed_band_gain"]) - 1.0)
    )
    return recovered.astype(np.float32)


def _segment_frame_bounds(
    segment: dict[str, Any],
    *,
    sample_rate: int,
    frame_limit: int,
) -> tuple[int, int]:
    start_frame = max(0, int(round(float(segment["relative_start_sec"]) * int(sample_rate))))
    end_frame = min(int(frame_limit), int(round(float(segment["relative_end_sec"]) * int(sample_rate))))
    return start_frame, end_frame


def _build_gate_blocker_presence_band_recovery_window_decisions(
    *,
    base_score: VocalCandidateScore,
    candidate_scores: list[VocalCandidateScore],
    candidate_records_by_name: dict[str, VocalCandidateRecord],
    candidate_issue_reports: dict[str, dict[str, Any]],
    defect_effectiveness_gate: dict[str, Any],
    source_audio: np.ndarray,
    base_audio: np.ndarray,
    recovered_audio_by_profile: dict[str, np.ndarray],
    sample_rate: int,
    tuning: dict[str, float | int | str],
) -> list[dict[str, Any]]:
    base_report = candidate_issue_reports.get(base_score.profile_name)
    if base_report is None:
        return []
    base_segments = base_report.get("segments")
    if not isinstance(base_segments, list):
        return []
    gate_blockers = defect_effectiveness_gate.get("blockers")
    if not isinstance(gate_blockers, list):
        return []

    blocker_by_window = {
        _window_key(blocker): blocker
        for blocker in gate_blockers
        if isinstance(blocker, dict)
        and blocker.get("reason") == "selected_candidate_still_has_severe_issue"
        and "start_sec" in blocker
        and "end_sec" in blocker
    }
    if not blocker_by_window:
        return []

    frame_limit = min(int(source_audio.shape[0]), int(base_audio.shape[0]))
    if frame_limit <= 0:
        return []
    source_for_metrics = source_audio[:frame_limit].astype(np.float32)
    base_for_metrics = base_audio[:frame_limit].astype(np.float32)
    track_metrics = _calculate_vocal_proxy_metrics_from_audio(source_for_metrics, base_for_metrics, int(sample_rate))
    shared_track_metrics = dict(base_score.metrics)
    shared_track_metrics.update(track_metrics)
    min_blocker_severity_score = float(tuning["min_blocker_severity_score"])
    min_predicted_segment_improvement = float(tuning["min_predicted_segment_improvement"])
    decisions: list[dict[str, Any]] = []

    for segment_index, base_segment in enumerate(base_segments):
        if not isinstance(base_segment, dict):
            continue
        blocker = blocker_by_window.get(_window_key(base_segment))
        if blocker is None:
            continue
        base_segment_for_scoring, base_selection_basis = _segment_for_shared_issue_scoring(
            base_segment,
            track_metrics=shared_track_metrics,
        )
        base_severity = float(base_segment_for_scoring.get("severity_score", 0.0))
        if base_severity < min_blocker_severity_score:
            continue
        start_frame, end_frame = _segment_frame_bounds(
            base_segment,
            sample_rate=int(sample_rate),
            frame_limit=frame_limit,
        )
        if end_frame <= start_frame:
            continue

        best_score: VocalCandidateScore | None = None
        best_issue: dict[str, Any] | None = None
        best_predicted_severity = base_severity
        for candidate_score in candidate_scores:
            if candidate_score.profile_name == base_score.profile_name:
                continue
            if candidate_score.profile_name not in candidate_records_by_name:
                continue
            candidate_report = candidate_issue_reports.get(candidate_score.profile_name)
            if candidate_report is None:
                continue
            candidate_segment = _candidate_report_segment(candidate_report, segment_index, base_segment)
            if candidate_segment is None:
                continue
            recovered_audio = recovered_audio_by_profile.get(candidate_score.profile_name)
            if recovered_audio is None:
                continue
            recovered_frame_limit = min(frame_limit, int(recovered_audio.shape[0]))
            local_end_frame = min(end_frame, recovered_frame_limit)
            if local_end_frame <= start_frame:
                continue
            predicted_metrics = _calculate_vocal_proxy_metrics_from_audio(
                source_for_metrics[start_frame:local_end_frame],
                recovered_audio[start_frame:local_end_frame],
                int(sample_rate),
            )
            predicted_issue = classify_vocal_segment_issue(predicted_metrics, track_metrics=track_metrics)
            predicted_severity = float(predicted_issue.get("severity_score", 0.0))
            if predicted_severity < best_predicted_severity:
                best_score = candidate_score
                best_issue = predicted_issue
                best_predicted_severity = predicted_severity

        if best_score is None or best_issue is None:
            continue
        predicted_improvement = base_severity - best_predicted_severity
        if predicted_improvement < min_predicted_segment_improvement:
            continue
        decisions.append(
            {
                "segment_index": int(segment_index),
                "start_sec": float(base_segment["start_sec"]),
                "end_sec": float(base_segment["end_sec"]),
                "relative_start_sec": float(base_segment["relative_start_sec"]),
                "relative_end_sec": float(base_segment["relative_end_sec"]),
                "base_profile": base_score.profile_name,
                "donor_profile": best_score.profile_name,
                "base_severity_score": round(float(base_severity), 6),
                "predicted_severity_score": round(float(best_predicted_severity), 6),
                "predicted_severity_improvement": round(float(predicted_improvement), 6),
                "blocker_reason": str(blocker.get("reason", "")),
                "base_selection_basis": base_selection_basis,
                "base_labels": [str(label) for label in base_segment_for_scoring.get("labels", [])],
                "predicted_labels": [str(label) for label in best_issue.get("labels", [])],
                "predicted_severity_parts": {
                    str(label): float(score)
                    for label, score in dict(best_issue.get("severity_parts", {})).items()
                },
                "presence_low_hz": float(tuning["presence_low_hz"]),
                "presence_high_hz": float(tuning["presence_high_hz"]),
                "bleed_low_hz": float(tuning["bleed_low_hz"]),
                "bleed_high_hz": float(tuning["bleed_high_hz"]),
                "presence_delta_gain": float(tuning["presence_delta_gain"]),
                "bleed_band_gain": float(tuning["bleed_band_gain"]),
                "recovery_policy": "measured_presence_band_delta_for_gate_blocker",
            }
        )
    return decisions


def write_gate_blocker_presence_band_recovery_candidate_manifest(
    *,
    profile: VocalCandidateProfile,
    source_manifest: dict[str, Any],
    dependency_fingerprints: dict[str, dict[str, str]],
    output_dir: Path,
    output_vocals_path: Path,
    recovery_debug: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "audio_path": source_manifest.get("audio_path"),
        "source_audio_sha256": source_manifest.get("source_audio_sha256"),
        "sample_rate": int(recovery_debug["sample_rate"]),
        "export_subtype": "PCM_24",
        "preview_window_sec": source_manifest.get("preview_window_sec"),
        "pipeline": {
            "mode": "vocal_preview_derived_candidate",
            "candidate_mode": profile.candidate_mode,
            "base_profile": recovery_debug["base_profile"],
            "dependency_fingerprints": dependency_fingerprints,
            "vocal_postprocess": "gate_blocker_presence_band_recovery_v1",
            "target_reference_assisted": False,
            "target_reference_used_at_application": False,
            "application_reference_free": True,
            "delivery_allowed_without_reference": True,
            "fallback_used": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
        },
        "tuning": profile.tuning,
        "stems": {
            "vocals": {
                "path": str(output_vocals_path),
                "vocal_postprocess": "gate_blocker_presence_band_recovery_v1",
                **recovery_debug,
            }
        },
    }
    manifest_path = output_dir / "stem_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _gate_blocker_presence_band_recovery_metrics(manifest: dict[str, Any]) -> dict[str, float]:
    vocals = manifest.get("stems", {}).get("vocals", {})
    if not isinstance(vocals, dict):
        return {}
    decisions = vocals.get("window_decisions")
    if not isinstance(decisions, list):
        return {}
    improvements = [
        float(decision.get("predicted_severity_improvement", 0.0))
        for decision in decisions
        if isinstance(decision, dict)
    ]
    return {
        "gate_blocker_presence_band_recovery_window_count": float(len(decisions)),
        "gate_blocker_presence_band_recovery_improvement_sum": round(float(sum(improvements)), 6),
        "gate_blocker_presence_band_recovery_improvement_max": round(
            float(max(improvements)) if improvements else 0.0,
            6,
        ),
        "gate_blocker_presence_band_recovery_base_segment_issue_max_score": float(
            vocals.get("base_segment_issue_max_score", 0.0),
        ),
        "gate_blocker_presence_band_recovery_base_segment_issue_mean_top3_score": float(
            vocals.get("base_segment_issue_mean_top3_score", 0.0),
        ),
    }


def build_gate_blocker_presence_band_recovery_candidate(
    *,
    profile: VocalCandidateProfile,
    base_score: VocalCandidateScore,
    candidate_scores: list[VocalCandidateScore],
    candidate_records_by_name: dict[str, VocalCandidateRecord],
    candidate_issue_reports: dict[str, dict[str, Any]],
    defect_effectiveness_gate: dict[str, Any],
    dependency_fingerprints: dict[str, dict[str, str]],
    output_dir: Path,
) -> dict[str, Any] | None:
    base_record = candidate_records_by_name.get(base_score.profile_name)
    if base_record is None:
        return None
    source_audio_path_value = base_record.manifest.get("audio_path")
    if not source_audio_path_value:
        return None
    source_audio_path = Path(str(source_audio_path_value))
    if not source_audio_path.exists():
        return None
    source_audio_raw, source_sr = sf.read(str(source_audio_path), always_2d=True, dtype="float32")
    base_audio_raw, base_sr = sf.read(str(base_record.vocals_path), always_2d=True, dtype="float32")
    source_audio, base_audio, sample_rate = _match_audio(source_audio_raw, int(source_sr), base_audio_raw, int(base_sr))
    recovered_audio_by_profile: dict[str, np.ndarray] = {}
    donor_audio_by_profile: dict[str, np.ndarray] = {}
    for candidate_score in candidate_scores:
        if candidate_score.profile_name == base_score.profile_name:
            continue
        donor_record = candidate_records_by_name.get(candidate_score.profile_name)
        if donor_record is None:
            continue
        donor_audio = donor_audio_by_profile.get(candidate_score.profile_name)
        if donor_audio is None:
            donor_audio = _read_and_match_candidate_to_reference(base_audio, int(sample_rate), donor_record.vocals_path)
            donor_audio_by_profile[candidate_score.profile_name] = donor_audio
        recovered_audio_by_profile[candidate_score.profile_name] = _presence_band_recovery_audio(
            base_audio=base_audio,
            donor_audio=donor_audio,
            sample_rate=int(sample_rate),
            tuning=profile.tuning,
        )

    if not recovered_audio_by_profile:
        return None
    window_decisions = _build_gate_blocker_presence_band_recovery_window_decisions(
        base_score=base_score,
        candidate_scores=candidate_scores,
        candidate_records_by_name=candidate_records_by_name,
        candidate_issue_reports=candidate_issue_reports,
        defect_effectiveness_gate=defect_effectiveness_gate,
        source_audio=source_audio,
        base_audio=base_audio,
        recovered_audio_by_profile=recovered_audio_by_profile,
        sample_rate=int(sample_rate),
        tuning=profile.tuning,
    )
    if not window_decisions:
        return None

    recovered_audio = base_audio.astype(np.float32).copy()
    applied_decisions: list[dict[str, Any]] = []
    for decision in window_decisions:
        donor_profile_name = str(decision["donor_profile"])
        donor_recovered_audio = recovered_audio_by_profile.get(donor_profile_name)
        if donor_recovered_audio is None:
            continue
        _blend_candidate_segment(
            output_audio=recovered_audio,
            candidate_audio=donor_recovered_audio,
            sample_rate=int(sample_rate),
            relative_start_sec=float(decision["relative_start_sec"]),
            relative_end_sec=float(decision["relative_end_sec"]),
            crossfade_sec=float(profile.tuning["crossfade_sec"]),
        )
        applied_decisions.append(decision)

    if not applied_decisions:
        return None
    recovered_audio = _limit_peak(recovered_audio)
    output_vocals_path = output_dir / "vocals.flac"
    output_vocals_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_vocals_path), recovered_audio, int(sample_rate), subtype="PCM_24")
    base_issue_summary = candidate_issue_reports.get(base_score.profile_name, {}).get("summary", {})
    recovery_debug = {
        "sample_rate": int(sample_rate),
        "base_profile": base_score.profile_name,
        "base_segment_issue_max_score": float(base_issue_summary.get("max_issue_score", 0.0)),
        "base_segment_issue_mean_top3_score": float(base_issue_summary.get("mean_top3_issue_score", 0.0)),
        "window_decisions": applied_decisions,
        "presence_band_recovered_window_count": int(len(applied_decisions)),
        "crossfade_sec": float(profile.tuning["crossfade_sec"]),
        "donor_profiles": sorted({str(decision["donor_profile"]) for decision in applied_decisions}),
        **_audio_summary(recovered_audio, int(sample_rate)),
    }
    return write_gate_blocker_presence_band_recovery_candidate_manifest(
        profile=profile,
        source_manifest=base_record.manifest,
        dependency_fingerprints=dependency_fingerprints,
        output_dir=output_dir,
        output_vocals_path=output_vocals_path,
        recovery_debug=recovery_debug,
    )


def _has_clustered_issue_ratio_blocker(defect_effectiveness_gate: dict[str, Any]) -> bool:
    blockers = defect_effectiveness_gate.get("blockers")
    if not isinstance(blockers, list):
        return False
    clustered_reasons = {
        "selected_candidate_still_has_clustered_issues",
        "selected_candidate_has_too_many_issue_windows",
    }
    for blocker in blockers:
        if not isinstance(blocker, dict):
            continue
        if blocker.get("reason") in clustered_reasons:
            return True
        if blocker.get("reason") != "selected_candidate_still_has_severe_issue":
            continue
        blocker_labels = {
            str(label)
            for label in blocker.get("labels", [])
            if str(label) != "acceptable_proxy_window"
        }
        if blocker_labels & CLUSTERED_ISSUE_REPAIR_LABELS:
            return True
    return False


def _clustered_issue_repair_variant_specs(
    labels: set[str],
    tuning: dict[str, float | int | str],
) -> list[dict[str, float | str]]:
    variants: list[dict[str, float | str]] = []
    if "kick_punch_bleed" in labels:
        for suffix, gain_key in (
            ("", "kick_band_gain"),
            ("_strong", "kick_band_gain_strong"),
            ("_aggressive", "kick_band_gain_aggressive"),
            ("_extreme", "kick_band_gain_extreme"),
        ):
            variants.append(
                {
                    "name": f"kick_punch_bleed_band_reduction{suffix}",
                    "low_hz": float(tuning["kick_low_hz"]),
                    "high_hz": float(tuning["kick_high_hz"]),
                    "gain": float(tuning[gain_key]),
                }
            )
    if "low_band_bleed" in labels:
        for suffix, gain_key in (("", "low_band_gain"), ("_strong", "low_band_gain_strong")):
            variants.append(
                {
                    "name": f"low_band_bleed_reduction{suffix}",
                    "low_hz": float(tuning["low_low_hz"]),
                    "high_hz": float(tuning["low_high_hz"]),
                    "gain": float(tuning[gain_key]),
                }
            )
    if "leaky_full_mix_risk" in labels:
        for suffix, gain_key in (("", "full_mix_band_gain"), ("_strong", "full_mix_band_gain_strong")):
            variants.append(
                {
                    "name": f"full_mix_low_mid_leakage_reduction{suffix}",
                    "low_hz": float(tuning["full_mix_low_hz"]),
                    "high_hz": float(tuning["full_mix_high_hz"]),
                    "gain": float(tuning[gain_key]),
                }
            )
    if "high_noise_or_robotic_artifact" in labels or "wide_stereo_non_vocal_leakage" in labels:
        for suffix, gain_key in (("", "high_band_gain"), ("_strong", "high_band_gain_strong")):
            variants.append(
                {
                    "name": f"high_band_noise_leakage_reduction{suffix}",
                    "low_hz": float(tuning["high_low_hz"]),
                    "high_hz": float(tuning["high_high_hz"]),
                    "gain": float(tuning[gain_key]),
                }
            )
    if "drop_melody_leakage" in labels:
        for suffix, gain_key in (
            ("", "full_mix_band_gain"),
            ("_strong", "full_mix_band_gain_strong"),
            ("_aggressive", "kick_band_gain_aggressive"),
            ("_extreme", "kick_band_gain_extreme"),
        ):
            variants.append(
                {
                    "name": f"drop_melody_mid_tonal_reduction{suffix}",
                    "low_hz": 700.0,
                    "high_hz": 1800.0,
                    "gain": float(tuning[gain_key]),
                }
            )
    if {"kick_punch_bleed", "drop_melody_leakage"} <= labels:
        variants.append(
            {
                "name": "kick_drop_melody_combo_reduction_extreme",
                "low_hz": float(tuning["kick_low_hz"]),
                "high_hz": 1800.0,
                "gain": float(tuning["kick_band_gain_extreme"]),
            }
        )
    return variants


def _band_limited_fft_component(
    audio: np.ndarray,
    sample_rate: int,
    *,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    frame_count = int(audio.shape[0])
    if frame_count <= 0:
        return np.zeros_like(audio, dtype=np.float32)
    nyquist_hz = float(sample_rate) * 0.5
    low_hz = max(0.0, min(float(low_hz), nyquist_hz))
    high_hz = max(low_hz, min(float(high_hz), nyquist_hz))
    frequencies = np.fft.rfftfreq(frame_count, d=1.0 / float(sample_rate))
    band_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    component = np.zeros_like(audio, dtype=np.float32)
    for channel_index in range(int(audio.shape[1])):
        spectrum = np.fft.rfft(audio[:, channel_index].astype(np.float32))
        spectrum[~band_mask] = 0.0
        component[:, channel_index] = np.fft.irfft(spectrum, n=frame_count).astype(np.float32)
    return component


def _clustered_issue_repair_variant_audio(
    base_audio: np.ndarray,
    sample_rate: int,
    *,
    variant: dict[str, float | str],
    tuning: dict[str, float | int | str],
) -> np.ndarray:
    if str(tuning.get("band_filter_method", "stft")) == "fft":
        band_component = _band_limited_fft_component(
            base_audio,
            sample_rate,
            low_hz=float(variant["low_hz"]),
            high_hz=float(variant["high_hz"]),
        )
    else:
        band_component = _band_limited_stft_component(
            base_audio,
            sample_rate,
            low_hz=float(variant["low_hz"]),
            high_hz=float(variant["high_hz"]),
            n_fft=int(tuning["band_filter_n_fft"]),
            hop_length=int(tuning["band_filter_hop_length"]),
        )
    return (base_audio + band_component * (float(variant["gain"]) - 1.0)).astype(np.float32)


def _build_clustered_issue_window_ratio_repair_decisions(
    *,
    base_score: VocalCandidateScore,
    candidate_issue_reports: dict[str, dict[str, Any]],
    defect_effectiveness_gate: dict[str, Any],
    source_audio: np.ndarray,
    base_audio: np.ndarray,
    variant_audio_by_name: dict[str, np.ndarray],
    sample_rate: int,
    tuning: dict[str, float | int | str],
) -> list[dict[str, Any]]:
    if not _has_clustered_issue_ratio_blocker(defect_effectiveness_gate):
        return []
    base_report = candidate_issue_reports.get(base_score.profile_name)
    if base_report is None:
        return []
    base_segments = base_report.get("segments")
    if not isinstance(base_segments, list):
        return []

    frame_limit = min(int(source_audio.shape[0]), int(base_audio.shape[0]))
    if frame_limit <= 0:
        return []
    source_for_metrics = source_audio[:frame_limit].astype(np.float32)
    base_for_metrics = base_audio[:frame_limit].astype(np.float32)
    track_metrics = _calculate_vocal_proxy_metrics_from_audio(source_for_metrics, base_for_metrics, int(sample_rate))
    shared_track_metrics = dict(base_score.metrics)
    shared_track_metrics.update(track_metrics)
    min_segment_issue_score = float(tuning["min_segment_issue_score"])
    min_predicted_segment_improvement = float(tuning["min_predicted_segment_improvement"])
    max_repaired_windows = int(tuning["max_repaired_windows"])
    decisions: list[dict[str, Any]] = []

    ranked_segments = [
        (segment_index, segment)
        for segment_index, segment in enumerate(base_segments)
        if isinstance(segment, dict)
    ]
    ranked_segments.sort(key=lambda item: float(item[1].get("severity_score", 0.0)), reverse=True)

    for segment_index, base_segment in ranked_segments:
        if len(decisions) >= max_repaired_windows:
            break
        base_segment_for_scoring, base_selection_basis = _segment_for_shared_issue_scoring(
            base_segment,
            track_metrics=shared_track_metrics,
        )
        reported_labels = {
            str(label)
            for label in base_segment_for_scoring.get("labels", [])
            if str(label) != "acceptable_proxy_window"
        }
        if not reported_labels & CLUSTERED_ISSUE_REPAIR_LABELS:
            continue
        start_frame, end_frame = _segment_frame_bounds(
            base_segment,
            sample_rate=int(sample_rate),
            frame_limit=frame_limit,
        )
        if end_frame <= start_frame:
            continue
        labels = set(reported_labels)
        if not isinstance(base_segment.get("metrics"), dict):
            base_metrics = _calculate_vocal_proxy_metrics_from_audio(
                source_for_metrics[start_frame:end_frame],
                base_for_metrics[start_frame:end_frame],
                int(sample_rate),
            )
            base_issue = classify_vocal_segment_issue(base_metrics, track_metrics=track_metrics)
            base_segment_for_scoring = dict(base_segment_for_scoring)
            base_segment_for_scoring["reported_severity_score"] = float(
                base_segment_for_scoring.get("severity_score", 0.0)
            )
            base_segment_for_scoring["reported_labels"] = sorted(reported_labels)
            base_segment_for_scoring["severity_score"] = base_issue["severity_score"]
            base_segment_for_scoring["labels"] = sorted(
                labels
                | {
                    str(label)
                    for label in base_issue.get("labels", [])
                    if str(label) != "acceptable_proxy_window"
                }
            )
            base_segment_for_scoring["severity_parts"] = base_issue["severity_parts"]
            base_selection_basis = "local_audio_reclassified_score"
            labels = {
                str(label)
                for label in base_segment_for_scoring.get("labels", [])
                if str(label) != "acceptable_proxy_window"
            }
        base_severity = float(base_segment_for_scoring.get("severity_score", 0.0))
        if base_severity < min_segment_issue_score:
            continue
        if not labels & CLUSTERED_ISSUE_REPAIR_LABELS:
            continue
        variants = _clustered_issue_repair_variant_specs(labels, tuning)
        if not variants:
            continue

        best_variant: dict[str, float | str] | None = None
        best_issue: dict[str, Any] | None = None
        best_predicted_severity = base_severity
        for variant in variants:
            variant_name = str(variant["name"])
            variant_audio = variant_audio_by_name.get(variant_name)
            if variant_audio is None:
                continue
            variant_frame_limit = min(frame_limit, int(variant_audio.shape[0]))
            local_end_frame = min(end_frame, variant_frame_limit)
            if local_end_frame <= start_frame:
                continue
            predicted_metrics = _calculate_vocal_proxy_metrics_from_audio(
                source_for_metrics[start_frame:local_end_frame],
                variant_audio[start_frame:local_end_frame],
                int(sample_rate),
            )
            predicted_issue = classify_vocal_segment_issue(predicted_metrics, track_metrics=track_metrics)
            predicted_labels = {
                str(label)
                for label in predicted_issue.get("labels", [])
                if str(label) != "acceptable_proxy_window"
            }
            introduced_vocal_loss = (predicted_labels - labels) & VOCAL_LOSS_PROTECTION_LABELS
            if introduced_vocal_loss:
                continue
            predicted_severity = float(predicted_issue.get("severity_score", 0.0))
            if predicted_severity < best_predicted_severity:
                best_variant = variant
                best_issue = predicted_issue
                best_predicted_severity = predicted_severity

        if best_variant is None or best_issue is None:
            continue
        predicted_improvement = base_severity - best_predicted_severity
        if predicted_improvement < min_predicted_segment_improvement:
            continue
        decisions.append(
            {
                "segment_index": int(segment_index),
                "start_sec": float(base_segment["start_sec"]),
                "end_sec": float(base_segment["end_sec"]),
                "relative_start_sec": float(base_segment["relative_start_sec"]),
                "relative_end_sec": float(base_segment["relative_end_sec"]),
                "base_profile": base_score.profile_name,
                "base_severity_score": round(float(base_severity), 6),
                "predicted_severity_score": round(float(best_predicted_severity), 6),
                "predicted_severity_improvement": round(float(predicted_improvement), 6),
                "base_selection_basis": base_selection_basis,
                "base_labels": sorted(labels),
                "predicted_labels": [
                    str(label)
                    for label in best_issue.get("labels", [])
                    if str(label) != "acceptable_proxy_window"
                ],
                "predicted_severity_parts": {
                    str(label): float(score)
                    for label, score in dict(best_issue.get("severity_parts", {})).items()
                },
                "variant_name": str(best_variant["name"]),
                "variant_low_hz": float(best_variant["low_hz"]),
                "variant_high_hz": float(best_variant["high_hz"]),
                "variant_gain": float(best_variant["gain"]),
                "repair_policy": "measured_clustered_issue_band_reduction",
            }
        )
    decisions.sort(key=lambda decision: int(decision["segment_index"]))
    return decisions


def write_clustered_issue_window_ratio_repair_candidate_manifest(
    *,
    profile: VocalCandidateProfile,
    source_manifest: dict[str, Any],
    dependency_fingerprints: dict[str, dict[str, str]],
    output_dir: Path,
    output_vocals_path: Path,
    repair_debug: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "audio_path": source_manifest.get("audio_path"),
        "source_audio_sha256": source_manifest.get("source_audio_sha256"),
        "sample_rate": int(repair_debug["sample_rate"]),
        "export_subtype": "PCM_24",
        "preview_window_sec": source_manifest.get("preview_window_sec"),
        "pipeline": {
            "mode": "vocal_preview_derived_candidate",
            "candidate_mode": profile.candidate_mode,
            "base_profile": repair_debug["base_profile"],
            "dependency_fingerprints": dependency_fingerprints,
            "vocal_postprocess": "clustered_issue_window_ratio_repair_v1",
            "target_reference_assisted": False,
            "target_reference_used_at_application": False,
            "application_reference_free": True,
            "delivery_allowed_without_reference": True,
            "fallback_used": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
        },
        "tuning": profile.tuning,
        "stems": {
            "vocals": {
                "path": str(output_vocals_path),
                "vocal_postprocess": "clustered_issue_window_ratio_repair_v1",
                **repair_debug,
            }
        },
    }
    manifest_path = output_dir / "stem_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _clustered_issue_window_ratio_repair_metrics(manifest: dict[str, Any]) -> dict[str, float]:
    vocals = manifest.get("stems", {}).get("vocals", {})
    if not isinstance(vocals, dict):
        return {}
    decisions = vocals.get("window_decisions")
    if not isinstance(decisions, list):
        return {}
    improvements = [
        float(decision.get("predicted_severity_improvement", 0.0))
        for decision in decisions
        if isinstance(decision, dict)
    ]
    return {
        "clustered_issue_window_ratio_repair_window_count": float(len(decisions)),
        "clustered_issue_window_ratio_repair_improvement_sum": round(float(sum(improvements)), 6),
        "clustered_issue_window_ratio_repair_improvement_max": round(
            float(max(improvements)) if improvements else 0.0,
            6,
        ),
        "clustered_issue_window_ratio_repair_base_segment_issue_max_score": float(
            vocals.get("base_segment_issue_max_score", 0.0),
        ),
        "clustered_issue_window_ratio_repair_base_segment_issue_mean_top3_score": float(
            vocals.get("base_segment_issue_mean_top3_score", 0.0),
        ),
        "clustered_issue_window_ratio_repair_base_issue_window_ratio": float(
            vocals.get("base_issue_window_ratio", 0.0),
        ),
    }


def build_clustered_issue_window_ratio_repair_candidate(
    *,
    profile: VocalCandidateProfile,
    base_score: VocalCandidateScore,
    candidate_records_by_name: dict[str, VocalCandidateRecord],
    candidate_issue_reports: dict[str, dict[str, Any]],
    defect_effectiveness_gate: dict[str, Any],
    dependency_fingerprints: dict[str, dict[str, str]],
    output_dir: Path,
) -> dict[str, Any] | None:
    base_record = candidate_records_by_name.get(base_score.profile_name)
    if base_record is None:
        return None
    source_audio_path_value = base_record.manifest.get("audio_path")
    if not source_audio_path_value:
        return None
    source_audio_path = Path(str(source_audio_path_value))
    if not source_audio_path.exists():
        return None
    source_audio_raw, source_sr = sf.read(str(source_audio_path), always_2d=True, dtype="float32")
    base_audio_raw, base_sr = sf.read(str(base_record.vocals_path), always_2d=True, dtype="float32")
    source_audio, base_audio, sample_rate = _match_audio(source_audio_raw, int(source_sr), base_audio_raw, int(base_sr))
    candidate_variant_specs_by_name: dict[str, dict[str, float | str]] = {}
    base_segments = candidate_issue_reports.get(base_score.profile_name, {}).get("segments", [])
    if isinstance(base_segments, list):
        for segment in base_segments:
            if not isinstance(segment, dict):
                continue
            labels = {str(label) for label in segment.get("labels", [])}
            for variant in _clustered_issue_repair_variant_specs(labels, profile.tuning):
                candidate_variant_specs_by_name[str(variant["name"])] = variant

    variant_audio_by_name: dict[str, np.ndarray] = {}
    for variant_name, variant in sorted(candidate_variant_specs_by_name.items()):
        variant_audio_by_name[variant_name] = _clustered_issue_repair_variant_audio(
            base_audio,
            int(sample_rate),
            variant=variant,
            tuning=profile.tuning,
        )

    if not variant_audio_by_name:
        return None
    window_decisions = _build_clustered_issue_window_ratio_repair_decisions(
        base_score=base_score,
        candidate_issue_reports=candidate_issue_reports,
        defect_effectiveness_gate=defect_effectiveness_gate,
        source_audio=source_audio,
        base_audio=base_audio,
        variant_audio_by_name=variant_audio_by_name,
        sample_rate=int(sample_rate),
        tuning=profile.tuning,
    )
    if not window_decisions:
        return None

    repaired_audio = base_audio.astype(np.float32).copy()
    applied_decisions: list[dict[str, Any]] = []
    for decision in window_decisions:
        variant_audio = variant_audio_by_name.get(str(decision["variant_name"]))
        if variant_audio is None:
            continue
        _blend_candidate_segment(
            output_audio=repaired_audio,
            candidate_audio=variant_audio,
            sample_rate=int(sample_rate),
            relative_start_sec=float(decision["relative_start_sec"]),
            relative_end_sec=float(decision["relative_end_sec"]),
            crossfade_sec=float(profile.tuning["crossfade_sec"]),
        )
        applied_decisions.append(decision)

    if not applied_decisions:
        return None
    repaired_audio = _limit_peak(repaired_audio)
    output_vocals_path = output_dir / "vocals.flac"
    output_vocals_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_vocals_path), repaired_audio, int(sample_rate), subtype="PCM_24")
    base_issue_summary = candidate_issue_reports.get(base_score.profile_name, {}).get("summary", {})
    repair_debug = {
        "sample_rate": int(sample_rate),
        "base_profile": base_score.profile_name,
        "base_segment_issue_max_score": float(base_issue_summary.get("max_issue_score", 0.0)),
        "base_segment_issue_mean_top3_score": float(base_issue_summary.get("mean_top3_issue_score", 0.0)),
        "base_issue_window_ratio": float(base_issue_summary.get("issue_window_ratio", 0.0)),
        "window_decisions": applied_decisions,
        "clustered_repaired_window_count": int(len(applied_decisions)),
        "crossfade_sec": float(profile.tuning["crossfade_sec"]),
        "variant_names": sorted({str(decision["variant_name"]) for decision in applied_decisions}),
        **_audio_summary(repaired_audio, int(sample_rate)),
    }
    return write_clustered_issue_window_ratio_repair_candidate_manifest(
        profile=profile,
        source_manifest=base_record.manifest,
        dependency_fingerprints=dependency_fingerprints,
        output_dir=output_dir,
        output_vocals_path=output_vocals_path,
        repair_debug=repair_debug,
    )


def _segment_labels(segment: dict[str, Any]) -> set[str]:
    return {str(label) for label in segment.get("labels", [])}


def _best_candidate_segment_for_window(
    candidate_scores: list[VocalCandidateScore],
    candidate_issue_reports: dict[str, dict[str, Any]],
    segment_index: int,
    base_segment: dict[str, Any],
) -> tuple[VocalCandidateScore | None, dict[str, Any] | None, float]:
    best_score: VocalCandidateScore | None = None
    best_segment: dict[str, Any] | None = None
    best_severity = math.inf
    for candidate_score in candidate_scores:
        candidate_report = candidate_issue_reports.get(candidate_score.profile_name)
        if candidate_report is None:
            continue
        candidate_segment = _candidate_report_segment(candidate_report, segment_index, base_segment)
        if candidate_segment is None:
            continue
        severity = float(candidate_segment.get("severity_score", 0.0))
        if severity < best_severity:
            best_score = candidate_score
            best_segment = candidate_segment
            best_severity = severity
    return best_score, best_segment, best_severity


def _no_vocal_cleanup_candidate_kind(
    *,
    base_segment: dict[str, Any],
    best_candidate_severity: float,
    tuning: dict[str, float | int | str],
) -> str | None:
    labels = _segment_labels(base_segment)
    severity = float(base_segment.get("severity_score", 0.0))
    if labels & NO_VOCAL_CLEANUP_PROTECTION_LABELS or not (labels & NO_VOCAL_LEAKAGE_CLEANUP_LABELS):
        if not _has_low_energy_drop_melody_only_residue(
            labels=labels,
            metrics=base_segment.get("metrics", {}),
            best_candidate_severity=float(best_candidate_severity),
            severity=severity,
            tuning=tuning,
        ):
            return None
        return "drop_melody_only_residue"
    metrics = base_segment.get("metrics", {})
    rms_ratio_db = float(metrics.get("candidate_to_source_rms_db", 0.0))
    kick_punch_ratio_db = float(metrics.get("kick_punch_ratio_db_250_900", 0.0))
    no_vocal_evidence_labels = {
        "thin_or_overcleaned",
        "presence_loss",
        "air_detail_loss",
    }
    has_low_kick_no_vocal_evidence = kick_punch_ratio_db <= -8.0
    has_low_energy_no_vocal_evidence = rms_ratio_db <= -28.0 and has_low_kick_no_vocal_evidence
    has_no_vocal_evidence = bool(labels & no_vocal_evidence_labels) and has_low_energy_no_vocal_evidence
    min_consensus_issue_score = float(tuning["min_consensus_issue_score"])
    if has_no_vocal_evidence and best_candidate_severity >= min_consensus_issue_score and (
        severity >= 1.50 or rms_ratio_db <= float(tuning["bridge_rms_ratio_db"])
    ):
        return "consensus_no_vocal"
    bridge_labels = {
        "thin_or_overcleaned",
        "high_noise_or_robotic_artifact",
        "unsupported_anchor_leakage",
        "high_band_unsupported_leakage",
        "presence_loss",
        "air_detail_loss",
    }
    has_bridge_no_vocal_evidence = (
        rms_ratio_db <= float(tuning["bridge_rms_ratio_db"]) and has_low_kick_no_vocal_evidence
    )
    if (
        best_candidate_severity >= float(tuning["bridge_candidate_issue_score"])
        and has_bridge_no_vocal_evidence
        and labels & bridge_labels
        and "local_vocal_body_dip" not in labels
    ):
        return "low_energy_no_vocal_bridge"
    return None


def _has_low_energy_drop_melody_only_residue(
    *,
    labels: set[str],
    metrics: Any,
    best_candidate_severity: float,
    severity: float,
    tuning: dict[str, float | int | str],
) -> bool:
    if "drop_melody_leakage" not in labels:
        return False
    if "backing_vocal_loss_risk" in labels:
        return False
    if not isinstance(metrics, dict):
        return False
    candidate_rms_dbfs = float(metrics.get("candidate_rms_dbfs", 0.0))
    mid_band_ratio_db = float(metrics.get("mid_tonal_band_ratio_db_700_1800", -200.0))
    mid_flatness = float(metrics.get("mid_tonal_flatness_700_1800", 1.0))
    mid_flux = float(metrics.get("mid_tonal_flux_700_1800", 1.0))
    mid_peak_ratio = float(metrics.get("mid_tonal_peak_ratio_700_1800", 0.0))
    presence_ratio_db = float(metrics.get("presence_ratio_db_1800_4500", 0.0))
    if best_candidate_severity < float(tuning["drop_melody_only_min_best_candidate_issue_score"]):
        return False
    if severity < float(tuning["bridge_candidate_issue_score"]):
        return False
    return (
        candidate_rms_dbfs <= float(tuning["drop_melody_only_max_candidate_rms_dbfs"])
        and mid_band_ratio_db >= float(tuning["drop_melody_only_min_mid_band_ratio_db"])
        and presence_ratio_db <= float(tuning["drop_melody_only_max_presence_ratio_db"])
        and mid_flatness <= float(tuning["drop_melody_only_max_flatness"])
        and mid_flux <= float(tuning["drop_melody_only_max_flux"])
        and mid_peak_ratio >= float(tuning["drop_melody_only_min_peak_ratio"])
    )


def _build_no_vocal_leakage_cleanup_window_decisions(
    *,
    base_score: VocalCandidateScore,
    candidate_scores: list[VocalCandidateScore],
    candidate_issue_reports: dict[str, dict[str, Any]],
    tuning: dict[str, float | int | str],
) -> list[dict[str, Any]]:
    base_report = candidate_issue_reports.get(base_score.profile_name)
    if base_report is None:
        return []
    base_segments = base_report.get("segments")
    if not isinstance(base_segments, list):
        return []

    raw_decisions: list[dict[str, Any]] = []
    for segment_index, base_segment in enumerate(base_segments):
        if not isinstance(base_segment, dict):
            continue
        best_score, best_segment, best_candidate_severity = _best_candidate_segment_for_window(
            candidate_scores,
            candidate_issue_reports,
            segment_index,
            base_segment,
        )
        if best_score is None or best_segment is None:
            continue
        cleanup_kind = _no_vocal_cleanup_candidate_kind(
            base_segment=base_segment,
            best_candidate_severity=best_candidate_severity,
            tuning=tuning,
        )
        if cleanup_kind is None:
            continue
        if cleanup_kind == "consensus_no_vocal":
            suppression_gain = float(tuning["suppression_gain"])
        elif cleanup_kind == "drop_melody_only_residue":
            suppression_gain = float(tuning["drop_melody_only_suppression_gain"])
        else:
            suppression_gain = float(tuning["bridge_suppression_gain"])
        raw_decisions.append(
            {
                "segment_index": int(segment_index),
                "start_sec": float(base_segment["start_sec"]),
                "end_sec": float(base_segment["end_sec"]),
                "relative_start_sec": float(base_segment["relative_start_sec"]),
                "relative_end_sec": float(base_segment["relative_end_sec"]),
                "base_profile": base_score.profile_name,
                "best_candidate_profile": best_score.profile_name,
                "cleanup_kind": cleanup_kind,
                "base_severity_score": float(base_segment.get("severity_score", 0.0)),
                "best_candidate_severity_score": round(float(best_candidate_severity), 6),
                "base_candidate_to_source_rms_db": round(
                    float(base_segment.get("metrics", {}).get("candidate_to_source_rms_db", 0.0)),
                    6,
                ),
                "base_kick_punch_ratio_db_250_900": round(
                    float(base_segment.get("metrics", {}).get("kick_punch_ratio_db_250_900", 0.0)),
                    6,
                ),
                "base_backing_vocal_side_loss_db": round(
                    float(base_segment.get("metrics", {}).get("backing_vocal_side_loss_db", 0.0)),
                    6,
                ),
                "suppression_gain": round(float(suppression_gain), 6),
                "base_labels": list(base_segment.get("labels", [])),
                "best_candidate_labels": list(best_segment.get("labels", [])),
            }
        )

    consensus_indexes = {
        int(decision["segment_index"])
        for decision in raw_decisions
        if decision["cleanup_kind"] == "consensus_no_vocal"
    }
    decisions = [
        decision
        for decision in raw_decisions
        if decision["cleanup_kind"] == "consensus_no_vocal"
        or decision["cleanup_kind"] == "drop_melody_only_residue"
        or int(decision["segment_index"]) - 1 in consensus_indexes
        or int(decision["segment_index"]) + 1 in consensus_indexes
    ]
    min_region_duration_sec = float(tuning["min_region_duration_sec"])
    return [
        decision
        for decision in decisions
        if float(decision["end_sec"]) - float(decision["start_sec"]) >= min_region_duration_sec
    ]


def _attenuate_candidate_segment(
    *,
    output_audio: np.ndarray,
    sample_rate: int,
    relative_start_sec: float,
    relative_end_sec: float,
    gain: float,
    crossfade_sec: float,
) -> None:
    start_frame = max(0, int(round(float(relative_start_sec) * int(sample_rate))))
    end_frame = min(output_audio.shape[0], int(round(float(relative_end_sec) * int(sample_rate))))
    if end_frame <= start_frame:
        return
    segment_length = end_frame - start_frame
    weight = np.ones(segment_length, dtype=np.float32)
    crossfade_frames = min(int(round(float(crossfade_sec) * int(sample_rate))), segment_length // 2)
    if crossfade_frames > 0:
        weight[:crossfade_frames] = np.linspace(0.0, 1.0, crossfade_frames, endpoint=False, dtype=np.float32)
        weight[-crossfade_frames:] = np.linspace(1.0, 0.0, crossfade_frames, endpoint=False, dtype=np.float32)
    gain_curve = 1.0 - weight[:, np.newaxis] * (1.0 - float(gain))
    output_audio[start_frame:end_frame] = (output_audio[start_frame:end_frame] * gain_curve).astype(np.float32)


def write_no_vocal_leakage_cleanup_candidate_manifest(
    *,
    profile: VocalCandidateProfile,
    source_manifest: dict[str, Any],
    dependency_fingerprints: dict[str, dict[str, str]],
    output_dir: Path,
    output_vocals_path: Path,
    cleanup_debug: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "audio_path": source_manifest.get("audio_path"),
        "source_audio_sha256": source_manifest.get("source_audio_sha256"),
        "sample_rate": int(cleanup_debug["sample_rate"]),
        "export_subtype": "PCM_24",
        "preview_window_sec": source_manifest.get("preview_window_sec"),
        "pipeline": {
            "mode": "vocal_preview_derived_candidate",
            "candidate_mode": profile.candidate_mode,
            "base_profile": cleanup_debug["base_profile"],
            "dependency_fingerprints": dependency_fingerprints,
            "vocal_postprocess": "no_vocal_leakage_cleanup_v1",
        },
        "tuning": profile.tuning,
        "stems": {
            "vocals": {
                "path": str(output_vocals_path),
                "vocal_postprocess": "no_vocal_leakage_cleanup_v1",
                **cleanup_debug,
            }
        },
    }
    manifest_path = output_dir / "stem_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _no_vocal_leakage_cleanup_metrics(manifest: dict[str, Any]) -> dict[str, float]:
    vocals = manifest.get("stems", {}).get("vocals", {})
    decisions = vocals.get("window_decisions", [])
    if not isinstance(decisions, list):
        decisions = []
    confidence_sum = sum(float(decision.get("best_candidate_severity_score", 0.0)) for decision in decisions)
    return {
        "no_vocal_cleanup_window_count": float(len(decisions)),
        "no_vocal_cleanup_confidence_sum": round(float(confidence_sum), 6),
        "no_vocal_cleanup_suppressed_duration_sec": round(
            float(
                sum(
                    max(0.0, float(decision.get("end_sec", 0.0)) - float(decision.get("start_sec", 0.0)))
                    for decision in decisions
                )
            ),
            6,
        ),
    }


def build_no_vocal_leakage_cleanup_candidate(
    *,
    profile: VocalCandidateProfile,
    base_score: VocalCandidateScore,
    candidate_scores: list[VocalCandidateScore],
    candidate_records_by_name: dict[str, VocalCandidateRecord],
    candidate_issue_reports: dict[str, dict[str, Any]],
    dependency_fingerprints: dict[str, dict[str, str]],
    output_dir: Path,
) -> dict[str, Any] | None:
    window_decisions = _build_no_vocal_leakage_cleanup_window_decisions(
        base_score=base_score,
        candidate_scores=candidate_scores,
        candidate_issue_reports=candidate_issue_reports,
        tuning=profile.tuning,
    )
    if not window_decisions:
        return None

    base_record = candidate_records_by_name[base_score.profile_name]
    base_audio, base_sr = sf.read(str(base_record.vocals_path), always_2d=True, dtype="float32")
    cleaned_audio = base_audio.astype(np.float32).copy()
    for decision in window_decisions:
        _attenuate_candidate_segment(
            output_audio=cleaned_audio,
            sample_rate=int(base_sr),
            relative_start_sec=float(decision["relative_start_sec"]),
            relative_end_sec=float(decision["relative_end_sec"]),
            gain=float(decision["suppression_gain"]),
            crossfade_sec=float(profile.tuning["crossfade_sec"]),
        )

    cleaned_audio = _limit_peak(cleaned_audio)
    output_vocals_path = output_dir / "vocals.flac"
    output_vocals_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_vocals_path), cleaned_audio, int(base_sr), subtype="PCM_24")
    cleanup_debug = {
        "sample_rate": int(base_sr),
        "base_profile": base_score.profile_name,
        "window_decisions": window_decisions,
        "cleaned_window_count": int(len(window_decisions)),
        "crossfade_sec": float(profile.tuning["crossfade_sec"]),
        **_audio_summary(cleaned_audio, int(base_sr)),
    }
    return write_no_vocal_leakage_cleanup_candidate_manifest(
        profile=profile,
        source_manifest=base_record.manifest,
        dependency_fingerprints=dependency_fingerprints,
        output_dir=output_dir,
        output_vocals_path=output_vocals_path,
        cleanup_debug=cleanup_debug,
    )


def _load_vocal_harshness_guard_module() -> Any:
    module_path = Path(__file__).with_name("apply_vocal_harshness_guard.py")
    spec = importlib.util.spec_from_file_location("apply_vocal_harshness_guard", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load vocal harshness guard module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_vocal_harshness_guard_candidate_manifest(
    *,
    profile: VocalCandidateProfile,
    source_manifest: dict[str, Any],
    anchor_manifest: dict[str, Any],
    dependency_fingerprints: dict[str, dict[str, str]],
    output_dir: Path,
    output_vocals_path: Path,
    guard_debug: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "audio_path": source_manifest.get("audio_path"),
        "source_audio_sha256": source_manifest.get("source_audio_sha256"),
        "sample_rate": int(guard_debug["sample_rate"]),
        "export_subtype": "PCM_24",
        "preview_window_sec": source_manifest.get("preview_window_sec"),
        "pipeline": {
            "mode": "vocal_preview_derived_candidate",
            "candidate_mode": profile.candidate_mode,
            "base_profile": guard_debug["base_profile"],
            "anchor_profile": guard_debug["anchor_profile"],
            "base_candidate_mode": source_manifest.get("pipeline", {}).get("candidate_mode"),
            "anchor_candidate_mode": anchor_manifest.get("pipeline", {}).get("candidate_mode"),
            "dependency_fingerprints": dependency_fingerprints,
            "vocal_postprocess": "vocal_harshness_guard_v1",
        },
        "tuning": profile.tuning,
        "stems": {
            "vocals": {
                "path": str(output_vocals_path),
                "vocal_postprocess": "vocal_harshness_guard_v1",
                **guard_debug,
            }
        },
    }
    manifest_path = output_dir / "stem_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _vocal_harshness_guard_metrics(manifest: dict[str, Any]) -> dict[str, float]:
    vocals = manifest.get("stems", {}).get("vocals", {})
    if not isinstance(vocals, dict):
        return {}
    decisions = vocals.get("window_decisions")
    debug = vocals.get("debug")
    if not isinstance(decisions, list) or not isinstance(debug, dict):
        return {}
    base_high = float(debug.get("base_high_band_rms", 0.0))
    guarded_high = float(debug.get("guarded_high_band_rms", 0.0))
    high_reduction = max(0.0, base_high - guarded_high)
    high_reduction_ratio = high_reduction / max(base_high, 1e-12)
    return {
        "vocal_harshness_guard_window_count": float(len(decisions)),
        "vocal_harshness_guard_high_band_reduction": round(float(high_reduction), 6),
        "vocal_harshness_guard_high_band_reduction_ratio": round(float(high_reduction_ratio), 6),
        "vocal_harshness_guard_route_mean": float(debug.get("route_mean", 0.0)),
    }


def build_vocal_harshness_guard_candidate(
    *,
    profile: VocalCandidateProfile,
    base_score: VocalCandidateScore,
    candidate_records_by_name: dict[str, VocalCandidateRecord],
    candidate_issue_reports: dict[str, dict[str, Any]],
    dependency_fingerprints: dict[str, dict[str, str]],
    output_dir: Path,
) -> dict[str, Any] | None:
    base_record = candidate_records_by_name.get(base_score.profile_name)
    if base_record is None:
        return None
    anchor_profile_name = str(profile.tuning["anchor_profile"])
    anchor_record = candidate_records_by_name.get(anchor_profile_name)
    if anchor_record is None:
        return None
    base_issue_report = candidate_issue_reports.get(base_score.profile_name)
    if base_issue_report is None:
        return None

    guard_module = _load_vocal_harshness_guard_module()
    tuning = guard_module.HarshnessGuardTuning(
        high_band_start_hz=float(profile.tuning["high_band_start_hz"]),
        high_band_end_hz=float(profile.tuning["high_band_end_hz"]),
        anchor_tolerance=float(profile.tuning["anchor_tolerance"]),
        min_high_band_floor=float(profile.tuning["min_high_band_floor"]),
        base_strength=float(profile.tuning["base_strength"]),
        max_strength=float(profile.tuning["max_strength"]),
        vocal_loss_strength_scale=float(profile.tuning["vocal_loss_strength_scale"]),
        min_segment_severity=float(profile.tuning["min_segment_severity"]),
        high_noise_flatness_threshold=float(profile.tuning["high_noise_flatness_threshold"]),
        fade_sec=float(profile.tuning["fade_sec"]),
        n_fft=int(profile.tuning["n_fft"]),
        hop_length=int(profile.tuning["hop_length"]),
    )
    window_decisions = guard_module.build_window_decisions(
        {"segment_issue_report": base_issue_report},
        tuning,
    )
    if not window_decisions:
        return None

    base_audio, base_sr = sf.read(str(base_record.vocals_path), always_2d=True, dtype="float32")
    anchor_audio, anchor_sr = sf.read(str(anchor_record.vocals_path), always_2d=True, dtype="float32")
    if int(base_sr) != int(anchor_sr):
        raise ValueError("Base and harshness guard anchor sample rates must match")
    guarded_audio, debug = guard_module.apply_harshness_guard(
        base_audio,
        anchor_audio,
        sample_rate=int(base_sr),
        decisions=window_decisions,
        tuning=tuning,
    )
    guarded_audio = _limit_peak(guarded_audio)
    output_vocals_path = output_dir / "vocals.flac"
    output_vocals_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_vocals_path), guarded_audio, int(base_sr), subtype="PCM_24")
    guard_debug = {
        "sample_rate": int(base_sr),
        "base_profile": base_score.profile_name,
        "anchor_profile": anchor_profile_name,
        "window_decisions": window_decisions,
        "guarded_window_count": int(len(window_decisions)),
        "debug": debug,
        **_audio_summary(guarded_audio, int(base_sr)),
    }
    return write_vocal_harshness_guard_candidate_manifest(
        profile=profile,
        source_manifest=base_record.manifest,
        anchor_manifest=anchor_record.manifest,
        dependency_fingerprints=dependency_fingerprints,
        output_dir=output_dir,
        output_vocals_path=output_vocals_path,
        guard_debug=guard_debug,
    )


def _candidate_skip_reason(skipped_candidates: dict[str, list[dict[str, str]]], candidate_name: str) -> str:
    for skipped_record in skipped_candidates.get(candidate_name, []):
        reason = skipped_record.get("reason")
        if reason:
            return str(reason)
    return "candidate_not_generated"


def _candidate_issue_dependency_fingerprints(
    *,
    candidate_scores: list[VocalCandidateScore],
    candidate_records_by_name: dict[str, VocalCandidateRecord],
    candidate_issue_reports: dict[str, dict[str, Any]],
) -> dict[str, dict[str, str]]:
    dependency_fingerprints: dict[str, dict[str, str]] = {}
    for candidate_score in candidate_scores:
        source_record = candidate_records_by_name[candidate_score.profile_name]
        fingerprint = _dependency_fingerprint(source_record)
        issue_summary = candidate_issue_reports.get(candidate_score.profile_name, {}).get("summary", {})
        fingerprint["score"] = f"{candidate_score.score:.6f}"
        fingerprint["issue_summary_sha256"] = hashlib.sha256(
            json.dumps(issue_summary, sort_keys=True).encode("utf-8"),
        ).hexdigest()
        dependency_fingerprints[candidate_score.profile_name] = fingerprint
    return dependency_fingerprints


def build_detected_issue_action_contract(
    *,
    issue_report: dict[str, Any] | None,
    available_candidate_names: set[str],
    skipped_candidates: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    if issue_report is None:
        return {
            "schema_version": 1,
            "mode": "detected_issue_action_contract",
            "status": "missing_issue_report",
            "segments": [],
            "summary": {
                "issue_segment_count": 0,
                "covered_label_count": 0,
                "rejected_label_count": 0,
                "uncovered_label_count": 0,
                "uncovered_labels": [],
            },
        }

    segments = issue_report.get("segments")
    if not isinstance(segments, list):
        segments = []

    covered_labels: set[str] = set()
    rejected_labels: set[str] = set()
    uncovered_labels: set[str] = set()
    segment_actions: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, dict) or float(segment.get("severity_score", 0.0)) <= 0.0:
            continue
        if "start_sec" not in segment or "end_sec" not in segment:
            continue
        labels = [str(label) for label in segment.get("labels", []) if str(label) != "acceptable_proxy_window"]
        actions: list[dict[str, Any]] = []
        rejections: list[dict[str, str]] = []
        uncovered_segment_labels: list[str] = []
        for label in labels:
            target_candidates = ISSUE_ACTION_TARGETS.get(label, ())
            available_targets = [target for target in target_candidates if target in available_candidate_names]
            if available_targets:
                covered_labels.add(label)
                actions.append(
                    {
                        "label": label,
                        "status": "action_available",
                        "action_family": VOCAL_DEFECT_ACTION_FAMILIES.get(label, "generic_issue_repair"),
                        "candidate_profiles": available_targets,
                    }
                )
                continue
            protective_reason = ISSUE_PROTECTIVE_REJECTION_REASONS.get(label)
            if protective_reason is not None:
                rejected_labels.add(label)
                rejections.append(
                    {
                        "label": label,
                        "status": "rejected_by_contract",
                        "action_family": VOCAL_DEFECT_ACTION_FAMILIES.get(label, "protective_rejection"),
                        "reason": protective_reason,
                    }
                )
                continue
            if target_candidates:
                rejected_labels.add(label)
                rejections.append(
                    {
                        "label": label,
                        "status": "candidate_unavailable",
                        "action_family": VOCAL_DEFECT_ACTION_FAMILIES.get(label, "generic_issue_repair"),
                        "candidate_profiles": ",".join(target_candidates),
                        "reason": _candidate_skip_reason(skipped_candidates, target_candidates[0]),
                    }
                )
                continue
            uncovered_labels.add(label)
            uncovered_segment_labels.append(label)
        segment_actions.append(
            {
                "start_sec": float(segment["start_sec"]),
                "end_sec": float(segment["end_sec"]),
                "severity_score": float(segment.get("severity_score", 0.0)),
                "labels": labels,
                "actions": actions,
                "rejections": rejections,
                "uncovered_labels": uncovered_segment_labels,
            }
        )

    return {
        "schema_version": 1,
        "mode": "detected_issue_action_contract",
        "status": "pass" if not uncovered_labels else "uncovered_labels",
        "segments": segment_actions,
        "summary": {
            "issue_segment_count": int(len(segment_actions)),
            "covered_label_count": int(len(covered_labels)),
            "rejected_label_count": int(len(rejected_labels)),
            "uncovered_label_count": int(len(uncovered_labels)),
            "covered_labels": sorted(covered_labels),
            "rejected_labels": sorted(rejected_labels),
            "uncovered_labels": sorted(uncovered_labels),
        },
    }


DEFECT_EFFECTIVENESS_SEVERE_SEGMENT_THRESHOLD = 1.0
DEFECT_EFFECTIVENESS_MEAN_TOP3_THRESHOLD = 0.75
DEFECT_EFFECTIVENESS_WINDOW_RATIO_THRESHOLD = 0.45
DEFECT_EFFECTIVENESS_MATERIAL_SEGMENT_THRESHOLD = 0.20
DEFECT_EFFECTIVENESS_MIN_ISSUE_IMPROVEMENT = 0.08
DEFECT_EFFECTIVENESS_GLOBAL_DROPOUT_THRESHOLD = 0.24
DEFECT_EFFECTIVENESS_GLOBAL_TAIL_DROPOUT_THRESHOLD = 0.24
DEFECT_EFFECTIVENESS_WORST_SEGMENT_DROPOUT_THRESHOLD = 0.50
DEFECT_EFFECTIVENESS_WORST_SEGMENT_TAIL_DROPOUT_THRESHOLD = 0.55
DEFECT_EFFECTIVENESS_RMS_RANGE_THRESHOLD_DB = 18.0


def _summary_float(summary: dict[str, Any], key: str) -> float:
    value = summary.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _metric_float(metrics: dict[str, float], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _material_issue_improvement(selected_metrics: dict[str, float]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for base_key, current_key, label in (
        (
            "issue_aware_base_segment_issue_max_score",
            "segment_issue_max_score",
            "issue_aware_segment_selection",
        ),
        (
            "segment_issue_repair_base_segment_issue_max_score",
            "segment_issue_max_score",
            "segment_issue_repair",
        ),
    ):
        if base_key not in selected_metrics or current_key not in selected_metrics:
            continue
        improvement = float(selected_metrics[base_key]) - float(selected_metrics[current_key])
        checks.append(
            {
                "label": label,
                "base_metric": base_key,
                "current_metric": current_key,
                "improvement": round(float(improvement), 6),
                "passed": bool(improvement >= DEFECT_EFFECTIVENESS_MIN_ISSUE_IMPROVEMENT),
            }
        )

    high_band_reduction = selected_metrics.get("vocal_harshness_guard_high_band_reduction_ratio")
    if high_band_reduction is not None:
        checks.append(
            {
                "label": "vocal_harshness_guard",
                "base_metric": "base_high_band_rms",
                "current_metric": "guarded_high_band_rms",
                "improvement": round(float(high_band_reduction), 6),
                "passed": bool(float(high_band_reduction) > 0.0),
            }
        )

    return {
        "checks": checks,
        "has_measured_action": bool(checks),
        "all_measured_actions_passed": all(check["passed"] for check in checks) if checks else None,
    }


def build_defect_effectiveness_gate(
    *,
    issue_report: dict[str, Any] | None,
    action_contract: dict[str, Any],
    selected_metrics: dict[str, float],
) -> dict[str, Any]:
    thresholds = {
        "severe_segment_score": DEFECT_EFFECTIVENESS_SEVERE_SEGMENT_THRESHOLD,
        "mean_top3_issue_score": DEFECT_EFFECTIVENESS_MEAN_TOP3_THRESHOLD,
        "issue_window_ratio": DEFECT_EFFECTIVENESS_WINDOW_RATIO_THRESHOLD,
        "material_segment_score": DEFECT_EFFECTIVENESS_MATERIAL_SEGMENT_THRESHOLD,
        "minimum_issue_improvement": DEFECT_EFFECTIVENESS_MIN_ISSUE_IMPROVEMENT,
        "global_dropout_frame_ratio": DEFECT_EFFECTIVENESS_GLOBAL_DROPOUT_THRESHOLD,
        "global_tail_dropout_ratio": DEFECT_EFFECTIVENESS_GLOBAL_TAIL_DROPOUT_THRESHOLD,
        "worst_segment_dropout_frame_ratio": DEFECT_EFFECTIVENESS_WORST_SEGMENT_DROPOUT_THRESHOLD,
        "worst_segment_tail_dropout_ratio": DEFECT_EFFECTIVENESS_WORST_SEGMENT_TAIL_DROPOUT_THRESHOLD,
        "segment_rms_ratio_range_db": DEFECT_EFFECTIVENESS_RMS_RANGE_THRESHOLD_DB,
    }
    blockers: list[dict[str, Any]] = []

    if action_contract.get("status") != "pass":
        blockers.append(
            {
                "reason": "action_contract_not_passed",
                "status": str(action_contract.get("status")),
            }
        )

    global_dropout_ratio = _metric_float(selected_metrics, "dropout_frame_ratio")
    global_tail_dropout_ratio = _metric_float(selected_metrics, "tail_dropout_ratio")
    worst_segment_dropout_ratio = _metric_float(selected_metrics, "segment_consistency_worst_dropout_frame_ratio")
    worst_segment_tail_dropout_ratio = _metric_float(selected_metrics, "segment_consistency_worst_tail_dropout_ratio")
    segment_rms_range_db = _metric_float(selected_metrics, "segment_consistency_rms_ratio_range_db")
    unstable_vocal_metrics: dict[str, float] = {}
    if global_dropout_ratio is not None and global_dropout_ratio >= DEFECT_EFFECTIVENESS_GLOBAL_DROPOUT_THRESHOLD:
        unstable_vocal_metrics["dropout_frame_ratio"] = float(global_dropout_ratio)
    if global_tail_dropout_ratio is not None and global_tail_dropout_ratio >= DEFECT_EFFECTIVENESS_GLOBAL_TAIL_DROPOUT_THRESHOLD:
        unstable_vocal_metrics["tail_dropout_ratio"] = float(global_tail_dropout_ratio)
    if (
        worst_segment_dropout_ratio is not None
        and worst_segment_dropout_ratio >= DEFECT_EFFECTIVENESS_WORST_SEGMENT_DROPOUT_THRESHOLD
    ):
        unstable_vocal_metrics["segment_consistency_worst_dropout_frame_ratio"] = float(worst_segment_dropout_ratio)
    if (
        worst_segment_tail_dropout_ratio is not None
        and worst_segment_tail_dropout_ratio >= DEFECT_EFFECTIVENESS_WORST_SEGMENT_TAIL_DROPOUT_THRESHOLD
    ):
        unstable_vocal_metrics["segment_consistency_worst_tail_dropout_ratio"] = float(worst_segment_tail_dropout_ratio)
    if segment_rms_range_db is not None and segment_rms_range_db >= DEFECT_EFFECTIVENESS_RMS_RANGE_THRESHOLD_DB:
        unstable_vocal_metrics["segment_consistency_rms_ratio_range_db"] = float(segment_rms_range_db)
    if unstable_vocal_metrics:
        blockers.append(
            {
                "reason": "selected_candidate_has_unstable_vocal_consistency",
                "metrics": {key: round(float(value), 6) for key, value in sorted(unstable_vocal_metrics.items())},
            }
        )

    if issue_report is None:
        blockers.append({"reason": "missing_selected_issue_report"})
        summary = {
            "max_issue_score": 0.0,
            "mean_top3_issue_score": 0.0,
            "issue_window_ratio": 0.0,
            "blocker_count": int(len(blockers)),
        }
        return {
            "schema_version": 1,
            "mode": "defect_effectiveness_gate",
            "status": "blocked_for_listening",
            "release_status": "diagnostic_only",
            "thresholds": thresholds,
            "summary": summary,
            "material_improvement": _material_issue_improvement(selected_metrics),
            "blockers": blockers,
        }

    report_summary = issue_report.get("summary")
    if not isinstance(report_summary, dict):
        report_summary = {}
    max_issue_score = _summary_float(report_summary, "max_issue_score")
    mean_top3_issue_score = _summary_float(report_summary, "mean_top3_issue_score")
    raw_issue_window_ratio = _summary_float(report_summary, "issue_window_ratio")
    issue_window_ratio = raw_issue_window_ratio

    segments = issue_report.get("segments")
    if not isinstance(segments, list):
        segments = []
    actionable_segments = _actionable_issue_segments(segments)
    if segments:
        actionable_scores = sorted(
            (float(segment.get("severity_score", 0.0)) for segment in actionable_segments),
            reverse=True,
        )
        max_issue_score = actionable_scores[0] if actionable_scores else 0.0
        mean_top3_issue_score = float(np.mean(actionable_scores[:3])) if actionable_scores else 0.0
        segment_count_base = int(issue_report.get("segments_analyzed") or len(segments))
        segment_count_base = max(segment_count_base, len(segments), 1)
        raw_issue_window_ratio = float(len(actionable_scores)) / float(segment_count_base)
        material_actionable_scores = [
            score
            for score in actionable_scores
            if score >= DEFECT_EFFECTIVENESS_MATERIAL_SEGMENT_THRESHOLD
        ]
        issue_window_ratio = float(len(material_actionable_scores)) / float(segment_count_base)
    severe_segments = [
        segment
        for segment in actionable_segments
        if float(segment.get("severity_score", 0.0)) >= DEFECT_EFFECTIVENESS_SEVERE_SEGMENT_THRESHOLD
    ]
    for segment in severe_segments:
        labels = _issue_labels(segment)
        blockers.append(
            {
                "reason": "selected_candidate_still_has_severe_issue",
                "start_sec": float(segment.get("start_sec", 0.0)),
                "end_sec": float(segment.get("end_sec", 0.0)),
                "severity_score": float(segment.get("severity_score", 0.0)),
                "labels": labels,
            }
        )

    if not severe_segments and mean_top3_issue_score >= DEFECT_EFFECTIVENESS_MEAN_TOP3_THRESHOLD:
        blockers.append(
            {
                "reason": "selected_candidate_still_has_clustered_issues",
                "mean_top3_issue_score": mean_top3_issue_score,
            }
        )
    if not severe_segments and issue_window_ratio >= DEFECT_EFFECTIVENESS_WINDOW_RATIO_THRESHOLD:
        blockers.append(
            {
                "reason": "selected_candidate_has_too_many_issue_windows",
                "issue_window_ratio": issue_window_ratio,
                "raw_issue_window_ratio": raw_issue_window_ratio,
                "material_segment_score": DEFECT_EFFECTIVENESS_MATERIAL_SEGMENT_THRESHOLD,
            }
        )

    material_improvement = _material_issue_improvement(selected_metrics)
    status = "blocked_for_listening" if blockers else "release_ready"
    release_status = "diagnostic_only" if blockers else "release_ready"
    return {
        "schema_version": 1,
        "mode": "defect_effectiveness_gate",
        "status": status,
        "release_status": release_status,
        "thresholds": thresholds,
        "summary": {
            "max_issue_score": round(float(max_issue_score), 6),
            "mean_top3_issue_score": round(float(mean_top3_issue_score), 6),
            "issue_window_ratio": round(float(issue_window_ratio), 6),
            "raw_issue_window_ratio": round(float(raw_issue_window_ratio), 6),
            "severe_segment_count": int(len(severe_segments)),
            "blocker_count": int(len(blockers)),
        },
        "material_improvement": material_improvement,
        "blockers": blockers,
    }


def _defect_gate_has_blocker(defect_effectiveness_gate: dict[str, Any], reason: str) -> bool:
    blockers = defect_effectiveness_gate.get("blockers")
    if not isinstance(blockers, list):
        return False
    return any(isinstance(blocker, dict) and blocker.get("reason") == reason for blocker in blockers)


def _action_contract_labels(action_contract: dict[str, Any]) -> set[str]:
    summary = action_contract.get("summary")
    if not isinstance(summary, dict):
        return set()
    labels: set[str] = set()
    for key in ("covered_labels", "uncovered_labels", "rejected_labels"):
        values = summary.get(key)
        if isinstance(values, list):
            labels.update(str(value) for value in values)
    return labels


def _live_gpu_ladder_id_from_path(path_value: Any) -> str | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    try:
        candidate_path = Path(path_value)
    except (TypeError, ValueError):
        return None
    path_parts = [part for part in candidate_path.parts if part]
    for model_id in LIVE_GPU_VOCAL_LADDER_MODEL_IDS:
        if model_id in path_parts:
            return model_id
    return None


def _live_gpu_ladder_id_from_manifest(manifest: dict[str, Any]) -> str | None:
    if not isinstance(manifest, dict):
        return None
    pipeline = manifest.get("pipeline")
    if isinstance(pipeline, dict):
        for key in (
            "model_id",
            "registered_model_id",
            "source_model_id",
            "route",
            "model_profile",
        ):
            value = pipeline.get(key)
            if isinstance(value, str) and value in LIVE_GPU_VOCAL_LADDER_MODEL_IDS:
                return value
        source_path_id = _live_gpu_ladder_id_from_path(pipeline.get("source_candidate_path"))
        if source_path_id is not None:
            return source_path_id
    stems = manifest.get("stems")
    if isinstance(stems, dict):
        vocals = stems.get("vocals")
        if isinstance(vocals, dict):
            source_path_id = _live_gpu_ladder_id_from_path(vocals.get("source_candidate_path"))
            if source_path_id is not None:
                return source_path_id
    return None


def collect_evaluated_live_gpu_vocal_model_ids(
    candidate_records: Iterable[VocalCandidateRecord],
) -> set[str]:
    evaluated: set[str] = set()
    for record in candidate_records:
        profile_name = str(record.profile.name)
        if profile_name in LIVE_GPU_VOCAL_LADDER_MODEL_IDS:
            evaluated.add(profile_name)
            continue
        manifest_model_id = _live_gpu_ladder_id_from_manifest(record.manifest)
        if manifest_model_id is not None:
            evaluated.add(manifest_model_id)
    return evaluated


def build_vocal_next_action_plan(
    *,
    defect_effectiveness_gate: dict[str, Any],
    action_contract: dict[str, Any],
    external_only: bool,
    evaluated_model_ids: Iterable[str] = (),
) -> dict[str, Any]:
    evaluated = {str(model_id) for model_id in evaluated_model_ids}
    recommended_model_ids = [
        model_id for model_id in LIVE_GPU_VOCAL_LADDER_MODEL_IDS if model_id not in evaluated
    ]
    gate_status = str(defect_effectiveness_gate.get("status", "unknown"))
    gate_release_status = str(defect_effectiveness_gate.get("release_status", "unknown"))
    blocker_reasons = [
        str(blocker.get("reason"))
        for blocker in defect_effectiveness_gate.get("blockers", [])
        if isinstance(blocker, dict) and blocker.get("reason")
    ]
    failed_labels = sorted(_action_contract_labels(action_contract))
    unstable_vocal_consistency = _defect_gate_has_blocker(
        defect_effectiveness_gate,
        UNSTABLE_VOCAL_CONSISTENCY_BLOCKER,
    )
    if unstable_vocal_consistency and external_only:
        failed_labels = sorted(set(failed_labels) | {"unstable_vocal_consistency", "vocal_dropout", "vocal_tail_loss"})
        if not recommended_model_ids:
            return {
                "schema_version": 1,
                "mode": "vocal_next_action_plan",
                "status": "candidate_space_exhausted",
                "next_action": "escalate_to_new_vocal_route_or_teacher_calibration",
                "reason": "unstable_vocal_consistency_after_live_gpu_ladder_candidates_exhausted",
                "gate_status": gate_status,
                "gate_release_status": gate_release_status,
                "blocker_reasons": blocker_reasons,
                "external_only_source_run": bool(external_only),
                "forbid_cpu_dsp_repair_loop": True,
                "required_acceleration_mode": "gpu_inference_or_teacher_calibration",
                "failed_labels": failed_labels,
                "evaluated_model_ids": sorted(evaluated),
                "recommended_model_ids": [],
                "required_next_step_options": [
                    "register_new_gpu_vocal_candidate",
                    "run_teacher_calibrated_vocal_selector",
                    "collect_reference_holdout_track",
                ],
            }
        return {
            "schema_version": 1,
            "mode": "vocal_next_action_plan",
            "status": "requires_new_model_candidates",
            "next_action": "run_live_gpu_vocal_candidate_ladder",
            "reason": "unstable_vocal_consistency_cannot_be_repaired_by_cpu_dsp_loop",
            "gate_status": gate_status,
            "gate_release_status": gate_release_status,
            "blocker_reasons": blocker_reasons,
            "external_only_source_run": bool(external_only),
            "forbid_cpu_dsp_repair_loop": True,
            "required_acceleration_mode": "gpu_inference",
            "failed_labels": failed_labels,
            "evaluated_model_ids": sorted(evaluated),
            "recommended_model_ids": recommended_model_ids,
        }
    if gate_release_status == "release_ready" and gate_status == "release_ready":
        return {
            "schema_version": 1,
            "mode": "vocal_next_action_plan",
            "status": "release_ready",
            "next_action": "publish_if_delivery_contract_accepts",
            "reason": "defect_effectiveness_gate_release_ready",
            "gate_status": gate_status,
            "gate_release_status": gate_release_status,
            "blocker_reasons": blocker_reasons,
            "external_only_source_run": bool(external_only),
            "forbid_cpu_dsp_repair_loop": False,
            "required_acceleration_mode": None,
            "failed_labels": failed_labels,
            "evaluated_model_ids": sorted(evaluated),
            "recommended_model_ids": [],
        }
    return {
        "schema_version": 1,
        "mode": "vocal_next_action_plan",
        "status": "requires_bounded_defect_specific_repair",
        "next_action": "run_bounded_defect_specific_repair_or_model_ladder",
        "reason": "defect_gate_blocked_without_global_unstable_vocal_consistency",
        "gate_status": gate_status,
        "gate_release_status": gate_release_status,
        "blocker_reasons": blocker_reasons,
        "external_only_source_run": bool(external_only),
        "forbid_cpu_dsp_repair_loop": False,
        "required_acceleration_mode": "cpu_dsp_scoring_or_gpu_inference",
        "failed_labels": failed_labels,
        "evaluated_model_ids": sorted(evaluated),
        "recommended_model_ids": recommended_model_ids,
    }


def _append_unique_strings(target: list[str], values: Any) -> None:
    if values is None:
        return
    if isinstance(values, (list, tuple, set)):
        iterable = values
    else:
        iterable = (values,)
    seen = set(target)
    for value in iterable:
        text = str(value).strip()
        if text and text not in seen:
            target.append(text)
            seen.add(text)


def build_adaptive_decision_contract(
    *,
    detected_issue_action_contract: dict[str, Any],
    defect_effectiveness_gate: dict[str, Any],
    vocal_next_action_plan: dict[str, Any],
    selected_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    issue_labels: list[str] = []
    failed_checks: list[str] = []
    selected_artifacts = selected_artifacts or {}

    gate_status = str(defect_effectiveness_gate.get("status") or "")
    gate_release_status = str(defect_effectiveness_gate.get("release_status") or "")
    plan_status = str(vocal_next_action_plan.get("status") or "")
    action_status = str(detected_issue_action_contract.get("status") or "")
    candidate_status = (
        gate_status
        or gate_release_status
        or plan_status
        or str(selected_artifacts.get("presentation_status") or "")
        or "unknown"
    )

    blockers = defect_effectiveness_gate.get("blockers")
    if not isinstance(blockers, list):
        blockers = []
    for blocker in blockers:
        if not isinstance(blocker, dict):
            continue
        _append_unique_strings(failed_checks, blocker.get("reason"))
        _append_unique_strings(failed_checks, blocker.get("status"))
        _append_unique_strings(issue_labels, blocker.get("labels"))

    summary = detected_issue_action_contract.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    if action_status and action_status != "pass":
        _append_unique_strings(failed_checks, action_status)
        _append_unique_strings(issue_labels, summary.get("uncovered_labels"))
        _append_unique_strings(issue_labels, summary.get("rejected_labels"))

    if gate_release_status != "release_ready" or gate_status != "release_ready":
        _append_unique_strings(issue_labels, vocal_next_action_plan.get("failed_labels"))

    if plan_status == "candidate_space_exhausted":
        _append_unique_strings(issue_labels, ("candidate_space_exhausted", "no_accepted_candidate"))
        _append_unique_strings(failed_checks, ("candidate_space_exhausted", "no_accepted_candidate"))
    elif plan_status and plan_status != "release_ready":
        _append_unique_strings(failed_checks, plan_status)

    decision = decision_from_evidence(
        issue_labels=issue_labels,
        failed_checks=failed_checks,
        candidate_status=candidate_status,
    )
    return {
        "schema_version": 1,
        "mode": "adaptive_decision_contract",
        "evidence": {
            "issue_labels": issue_labels,
            "failed_checks": failed_checks,
            "candidate_status": candidate_status,
            "source_contracts": {
                "detected_issue_action_contract_status": action_status or "unknown",
                "defect_effectiveness_gate_status": gate_status or "unknown",
                "defect_effectiveness_gate_release_status": gate_release_status or "unknown",
                "vocal_next_action_plan_status": plan_status or "unknown",
            },
        },
        **decision,
    }


def _window_key(segment: dict[str, Any]) -> tuple[float, float]:
    return (
        round(float(segment.get("start_sec", 0.0)), 3),
        round(float(segment.get("end_sec", 0.0)), 3),
    )


def _segments_by_window(issue_report: dict[str, Any] | None) -> dict[tuple[float, float], dict[str, Any]]:
    if not isinstance(issue_report, dict):
        return {}
    segments = issue_report.get("segments")
    if not isinstance(segments, list):
        return {}
    return {
        _window_key(segment): segment
        for segment in segments
        if isinstance(segment, dict)
        and "start_sec" in segment
        and "end_sec" in segment
    }


def _candidate_action_effectiveness(
    *,
    segment: dict[str, Any],
    actions: list[Any],
    candidate_issue_reports: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    if not candidate_issue_reports:
        return {
            "status": "not_measured",
            "reason": "missing_candidate_issue_reports",
            "candidate_profiles": [],
        }

    base_window_key = _window_key(segment)
    base_score = float(segment.get("severity_score", 0.0))
    base_labels = {
        str(label)
        for label in segment.get("labels", [])
        if str(label) != "acceptable_proxy_window"
    }
    candidate_profiles = sorted(
        {
            str(profile)
            for action in actions
            if isinstance(action, dict)
            for profile in action.get("candidate_profiles", [])
        }
    )
    candidate_deltas: list[dict[str, Any]] = []
    for profile_name in candidate_profiles:
        candidate_segments = _segments_by_window(candidate_issue_reports.get(profile_name))
        candidate_segment = candidate_segments.get(base_window_key)
        candidate_score = float(candidate_segment.get("severity_score", 0.0)) if candidate_segment else 0.0
        candidate_labels = {
            str(label)
            for label in (candidate_segment or {}).get("labels", [])
            if str(label) != "acceptable_proxy_window"
        }
        severity_delta = base_score - candidate_score
        resolved_labels = sorted(base_labels - candidate_labels)
        introduced_labels = sorted(candidate_labels - base_labels)
        candidate_deltas.append(
            {
                "candidate_profile": profile_name,
                "base_severity_score": round(base_score, 6),
                "candidate_severity_score": round(candidate_score, 6),
                "severity_delta": round(float(severity_delta), 6),
                "resolved_labels": resolved_labels,
                "introduced_labels": introduced_labels,
                "status": (
                    "improved"
                    if severity_delta >= DEFECT_EFFECTIVENESS_MIN_ISSUE_IMPROVEMENT
                    else "worsened"
                    if severity_delta < 0.0
                    else "unchanged"
                ),
            }
        )

    if not candidate_deltas:
        return {
            "status": "not_measured",
            "reason": "no_action_candidate_reports",
            "candidate_profiles": candidate_profiles,
        }

    best_delta = max(candidate_deltas, key=lambda item: float(item["severity_delta"]))
    return {
        "status": best_delta["status"],
        "best_candidate_profile": best_delta["candidate_profile"],
        "severity_delta": best_delta["severity_delta"],
        "resolved_labels": best_delta["resolved_labels"],
        "introduced_labels": best_delta["introduced_labels"],
        "candidate_profiles": candidate_profiles,
        "candidate_deltas": candidate_deltas,
    }


def build_issue_effectiveness_ledger(
    *,
    issue_report: dict[str, Any] | None,
    action_contract: dict[str, Any],
    defect_effectiveness_gate: dict[str, Any],
    candidate_issue_reports: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if issue_report is None:
        return {
            "schema_version": 1,
            "mode": "issue_effectiveness_ledger",
            "status": "blocked_for_listening",
            "summary": {
                "window_count": 0,
                "blocked_window_count": 0,
                "action_window_count": 0,
                "uncovered_window_count": 0,
            },
            "windows": [],
        }

    issue_segments = issue_report.get("segments")
    if not isinstance(issue_segments, list):
        issue_segments = []
    contract_segments = action_contract.get("segments")
    if not isinstance(contract_segments, list):
        contract_segments = []
    gate_blockers = defect_effectiveness_gate.get("blockers")
    if not isinstance(gate_blockers, list):
        gate_blockers = []

    contract_by_window = {
        _window_key(segment): segment
        for segment in contract_segments
        if isinstance(segment, dict)
    }
    blockers_by_window: dict[tuple[float, float], list[dict[str, Any]]] = {}
    global_blockers: list[dict[str, Any]] = []
    for blocker in gate_blockers:
        if not isinstance(blocker, dict):
            continue
        if "start_sec" in blocker and "end_sec" in blocker:
            blockers_by_window.setdefault(_window_key(blocker), []).append(blocker)
        else:
            global_blockers.append(blocker)

    windows: list[dict[str, Any]] = []
    for segment in issue_segments:
        if (
            not isinstance(segment, dict)
            or float(segment.get("severity_score", 0.0)) <= 0.0
            or _is_both_near_silent_segment(segment)
        ):
            continue
        key = _window_key(segment)
        contract_segment = contract_by_window.get(key, {})
        blockers = blockers_by_window.get(key, [])
        actions = contract_segment.get("actions", []) if isinstance(contract_segment, dict) else []
        rejections = contract_segment.get("rejections", []) if isinstance(contract_segment, dict) else []
        uncovered_labels = contract_segment.get("uncovered_labels", []) if isinstance(contract_segment, dict) else []
        action_effectiveness = _candidate_action_effectiveness(
            segment=segment,
            actions=actions if isinstance(actions, list) else [],
            candidate_issue_reports=candidate_issue_reports,
        )
        release_decision = "blocked" if blockers or global_blockers else "release_ready"
        windows.append(
            {
                "start_sec": key[0],
                "end_sec": key[1],
                "severity_score": float(segment.get("severity_score", 0.0)),
                "labels": [str(label) for label in segment.get("labels", [])],
                "severity_parts": segment.get("severity_parts", {}),
                "metrics": segment.get("metrics", {}),
                "actions": actions,
                "rejections": rejections,
                "uncovered_labels": uncovered_labels,
                "action_effectiveness": action_effectiveness,
                "blockers": blockers,
                "release_decision": release_decision,
            }
        )

    blocked_window_count = sum(1 for window in windows if window["release_decision"] == "blocked")
    action_improved_window_count = sum(
        1 for window in windows if window["action_effectiveness"].get("status") == "improved"
    )
    action_worsened_window_count = sum(
        1 for window in windows if window["action_effectiveness"].get("status") == "worsened"
    )
    action_unmeasured_window_count = sum(
        1 for window in windows if window["action_effectiveness"].get("status") == "not_measured"
    )
    return {
        "schema_version": 1,
        "mode": "issue_effectiveness_ledger",
        "status": defect_effectiveness_gate.get("status", "blocked_for_listening"),
        "summary": {
            "window_count": int(len(windows)),
            "blocked_window_count": int(blocked_window_count),
            "action_window_count": int(sum(1 for window in windows if window["actions"])),
            "uncovered_window_count": int(sum(1 for window in windows if window["uncovered_labels"])),
            "global_blocker_count": int(len(global_blockers)),
            "action_improved_window_count": int(action_improved_window_count),
            "action_worsened_window_count": int(action_worsened_window_count),
            "action_unmeasured_window_count": int(action_unmeasured_window_count),
        },
        "global_blockers": global_blockers,
        "windows": windows,
    }


def score_candidate(
    *,
    profile: VocalCandidateProfile | AnchorSuppressionProfile | VocalBodyRepairProfile | SegmentRouterProfile,
    metrics: dict[str, float],
    feedback_adjustments: dict[str, float],
    manifest_path: Path,
    output_dir: Path,
    vocals_path: Path,
) -> VocalCandidateScore:
    score = profile.prior_score + feedback_adjustments.get(profile.name, 0.0)
    reasons = [f"profile prior {profile.prior_score:+.3f}"]
    if profile.name in feedback_adjustments:
        reasons.append(f"feedback adjustment {feedback_adjustments[profile.name]:+.3f}")

    dropout_ratio = float(metrics["dropout_frame_ratio"])
    tail_dropout_ratio = float(metrics["tail_dropout_ratio"])
    rms_ratio_db = float(metrics["candidate_to_source_rms_db"])
    low_ratio_db = float(metrics["low_band_ratio_db_35_180"])
    kick_ratio_db = float(metrics["kick_punch_ratio_db_250_900"])
    presence_ratio_db = float(metrics["presence_ratio_db_1800_4500"])
    air_ratio_db = float(metrics["air_ratio_db_4500_9000"])
    peak_dbfs = float(metrics["candidate_peak_dbfs"])
    unsupported_vs_anchor_db = metrics.get("unsupported_vs_anchor_ratio_db")
    high_unsupported_vs_anchor_db = metrics.get("high_unsupported_vs_anchor_ratio_db")
    high_noise_flatness = metrics.get("high_noise_flatness_6000_16000")
    stereo_side_mid_ratio_db = metrics.get("stereo_side_mid_ratio_db")
    source_stereo_side_mid_ratio_db = metrics.get("source_stereo_side_mid_ratio_db")
    backing_vocal_side_loss_db = metrics.get("backing_vocal_side_loss_db")
    segment_worst_dropout_ratio = metrics.get("segment_consistency_worst_dropout_frame_ratio")
    segment_worst_tail_dropout_ratio = metrics.get("segment_consistency_worst_tail_dropout_ratio")
    segment_dropout_std = metrics.get("segment_consistency_dropout_std")
    segment_tail_dropout_std = metrics.get("segment_consistency_tail_dropout_std")
    segment_rms_range_db = metrics.get("segment_consistency_rms_ratio_range_db")
    segment_issue_max_score = metrics.get("segment_issue_max_score")
    segment_issue_mean_top3_score = metrics.get("segment_issue_mean_top3_score")
    segment_issue_window_ratio = metrics.get("segment_issue_window_ratio")
    segment_reverb_tail_smear_count = metrics.get("segment_reverb_tail_smear_count")
    segment_reverb_tail_smear_window_ratio = metrics.get("segment_reverb_tail_smear_window_ratio")
    segment_reverb_tail_smear_max_score = metrics.get("segment_reverb_tail_smear_max_score")
    issue_aware_replaced_window_count = metrics.get("issue_aware_replaced_window_count")
    issue_aware_adjusted_improvement_sum = metrics.get("issue_aware_adjusted_improvement_sum")
    issue_aware_base_segment_issue_max_score = metrics.get("issue_aware_base_segment_issue_max_score")
    issue_aware_base_segment_issue_mean_top3_score = metrics.get("issue_aware_base_segment_issue_mean_top3_score")
    segment_issue_repair_window_count = metrics.get("segment_issue_repair_window_count")
    segment_issue_repair_recovery_improvement_sum = metrics.get("segment_issue_repair_recovery_improvement_sum")
    segment_issue_repair_issue_improvement_sum = metrics.get("segment_issue_repair_issue_improvement_sum")
    segment_issue_repair_base_segment_issue_max_score = metrics.get("segment_issue_repair_base_segment_issue_max_score")
    segment_issue_repair_base_segment_issue_mean_top3_score = metrics.get(
        "segment_issue_repair_base_segment_issue_mean_top3_score"
    )
    no_vocal_cleanup_window_count = metrics.get("no_vocal_cleanup_window_count")
    no_vocal_cleanup_confidence_sum = metrics.get("no_vocal_cleanup_confidence_sum")
    no_vocal_cleanup_suppressed_duration_sec = metrics.get("no_vocal_cleanup_suppressed_duration_sec")
    vocal_harshness_guard_window_count = metrics.get("vocal_harshness_guard_window_count")
    vocal_harshness_guard_high_band_reduction_ratio = metrics.get("vocal_harshness_guard_high_band_reduction_ratio")

    coverage_bonus = (1.0 - dropout_ratio) * 1.35
    tail_bonus = (1.0 - tail_dropout_ratio) * 1.55
    score += coverage_bonus + tail_bonus
    reasons.append(f"coverage bonus {coverage_bonus:+.3f}")
    reasons.append(f"tail preservation bonus {tail_bonus:+.3f}")

    dropout_penalty = dropout_ratio * 1.15
    tail_penalty = tail_dropout_ratio * 1.55
    score -= dropout_penalty + tail_penalty
    reasons.append(f"dropout penalty -{dropout_penalty:.3f}")
    reasons.append(f"tail dropout penalty -{tail_penalty:.3f}")

    if rms_ratio_db < -31.0:
        penalty = (-31.0 - rms_ratio_db) * 0.07
        score -= penalty
        reasons.append(f"too-thin vocal rms penalty -{penalty:.3f}")
    elif rms_ratio_db > -4.5:
        penalty = (rms_ratio_db + 4.5) * 0.12
        score -= penalty
        reasons.append(f"possible full-mix bleed rms penalty -{penalty:.3f}")
    else:
        reasons.append("candidate/source rms is inside vocal isolation band")

    if low_ratio_db > -18.5:
        penalty = (low_ratio_db + 18.5) * 0.055
        score -= penalty
        reasons.append(f"low-band bleed penalty -{penalty:.3f}")
    if kick_ratio_db > -11.5:
        penalty = (kick_ratio_db + 11.5) * 0.045
        score -= penalty
        reasons.append(f"kick/punch bleed penalty -{penalty:.3f}")
    if unsupported_vs_anchor_db is not None and float(unsupported_vs_anchor_db) > -22.0:
        penalty = (float(unsupported_vs_anchor_db) + 22.0) * 0.055
        score -= penalty
        reasons.append(f"unsupported-vs-anchor leakage penalty -{penalty:.3f}")
    if high_unsupported_vs_anchor_db is not None and float(high_unsupported_vs_anchor_db) > -20.0:
        penalty = (float(high_unsupported_vs_anchor_db) + 20.0) * 0.035
        score -= penalty
        reasons.append(f"high-band unsupported leakage penalty -{penalty:.3f}")
    if high_noise_flatness is not None and float(high_noise_flatness) > 0.16:
        penalty = (float(high_noise_flatness) - 0.16) * 1.25
        score -= penalty
        reasons.append(f"high-band noise flatness penalty -{penalty:.3f}")
    if stereo_side_mid_ratio_db is not None and float(stereo_side_mid_ratio_db) > -8.0:
        penalty = (float(stereo_side_mid_ratio_db) + 8.0) * 0.04
        score -= penalty
        reasons.append(f"wide stereo non-vocal leakage penalty -{penalty:.3f}")
    if (
        backing_vocal_side_loss_db is not None
        and source_stereo_side_mid_ratio_db is not None
        and float(source_stereo_side_mid_ratio_db) > -12.0
        and float(backing_vocal_side_loss_db) > 9.0
    ):
        penalty = (float(backing_vocal_side_loss_db) - 9.0) * 0.025
        score -= penalty
        reasons.append(f"backing vocal side loss penalty -{penalty:.3f}")

    if segment_worst_dropout_ratio is not None:
        segment_dropout_threshold = max(dropout_ratio + 0.22, 0.50)
        if float(segment_worst_dropout_ratio) > segment_dropout_threshold:
            penalty = (float(segment_worst_dropout_ratio) - segment_dropout_threshold) * 0.35
            score -= penalty
            reasons.append(f"segment worst dropout penalty -{penalty:.3f}")
    if segment_worst_tail_dropout_ratio is not None:
        segment_tail_threshold = max(tail_dropout_ratio + 0.22, 0.50)
        if float(segment_worst_tail_dropout_ratio) > segment_tail_threshold:
            penalty = (float(segment_worst_tail_dropout_ratio) - segment_tail_threshold) * 0.50
            score -= penalty
            reasons.append(f"segment worst tail penalty -{penalty:.3f}")
    if segment_dropout_std is not None and float(segment_dropout_std) > 0.18:
        penalty = (float(segment_dropout_std) - 0.18) * 0.45
        score -= penalty
        reasons.append(f"segment dropout variance penalty -{penalty:.3f}")
    if segment_tail_dropout_std is not None and float(segment_tail_dropout_std) > 0.18:
        penalty = (float(segment_tail_dropout_std) - 0.18) * 0.55
        score -= penalty
        reasons.append(f"segment tail variance penalty -{penalty:.3f}")
    if segment_rms_range_db is not None and float(segment_rms_range_db) > 8.0:
        penalty = (float(segment_rms_range_db) - 8.0) * 0.035
        score -= penalty
        reasons.append(f"segment rms consistency penalty -{penalty:.3f}")
    if segment_issue_max_score is not None and float(segment_issue_max_score) > 0.85:
        penalty = (float(segment_issue_max_score) - 0.85) * 0.16
        score -= penalty
        reasons.append(f"segment issue hotspot penalty -{penalty:.3f}")
    if segment_issue_mean_top3_score is not None and float(segment_issue_mean_top3_score) > 0.65:
        penalty = (float(segment_issue_mean_top3_score) - 0.65) * 0.12
        score -= penalty
        reasons.append(f"segment issue cluster penalty -{penalty:.3f}")
    if segment_issue_window_ratio is not None and float(segment_issue_window_ratio) > 0.35:
        penalty = (float(segment_issue_window_ratio) - 0.35) * 0.18
        score -= penalty
        reasons.append(f"segment issue coverage penalty -{penalty:.3f}")
    if (
        segment_reverb_tail_smear_count is not None
        and segment_reverb_tail_smear_window_ratio is not None
        and segment_reverb_tail_smear_max_score is not None
    ):
        smear_hotspot_penalty = max(0.0, float(segment_reverb_tail_smear_max_score) - 0.55) * 0.22
        smear_coverage_penalty = max(0.0, float(segment_reverb_tail_smear_window_ratio) - 0.20) * 0.30
        smear_count_penalty = min(float(segment_reverb_tail_smear_count), 10.0) * 0.015
        penalty = smear_hotspot_penalty + smear_coverage_penalty + smear_count_penalty
        if penalty > 0.0:
            score -= penalty
            reasons.append(f"reverb tail smear penalty -{penalty:.3f}")
    if (
        issue_aware_replaced_window_count is not None
        and issue_aware_adjusted_improvement_sum is not None
        and issue_aware_base_segment_issue_max_score is not None
        and issue_aware_base_segment_issue_mean_top3_score is not None
        and segment_issue_max_score is not None
        and segment_issue_mean_top3_score is not None
        and float(issue_aware_replaced_window_count) > 0.0
        and float(issue_aware_adjusted_improvement_sum) > 0.0
    ):
        measured_issue_improvement = max(
            0.0,
            float(issue_aware_base_segment_issue_max_score) - float(segment_issue_max_score),
        )
        measured_top3_improvement = max(
            0.0,
            float(issue_aware_base_segment_issue_mean_top3_score) - float(segment_issue_mean_top3_score),
        )
        issue_aware_bonus = min(
            0.18,
            measured_issue_improvement * 0.55 + measured_top3_improvement * 0.65,
            float(issue_aware_replaced_window_count) * 0.035
            + float(issue_aware_adjusted_improvement_sum) * 0.16,
        )
        if issue_aware_bonus > 0.0:
            score += issue_aware_bonus
            reasons.append(f"issue-aware measured improvement bonus +{issue_aware_bonus:.3f}")
        else:
            reasons.append("issue-aware segment decisions did not improve recalculated issue metrics")
    if (
        segment_issue_repair_window_count is not None
        and segment_issue_repair_recovery_improvement_sum is not None
        and segment_issue_repair_base_segment_issue_max_score is not None
        and segment_issue_repair_base_segment_issue_mean_top3_score is not None
        and segment_issue_max_score is not None
        and segment_issue_mean_top3_score is not None
        and float(segment_issue_repair_window_count) > 0.0
    ):
        measured_issue_improvement = max(
            0.0,
            float(segment_issue_repair_base_segment_issue_max_score) - float(segment_issue_max_score),
        )
        measured_top3_improvement = max(
            0.0,
            float(segment_issue_repair_base_segment_issue_mean_top3_score) - float(segment_issue_mean_top3_score),
        )
        segment_issue_repair_bonus = min(
            0.16,
            measured_issue_improvement * 0.50 + measured_top3_improvement * 0.60,
            float(segment_issue_repair_window_count) * 0.025
            + max(
                0.0,
                float(
                    segment_issue_repair_issue_improvement_sum
                    if segment_issue_repair_issue_improvement_sum is not None
                    else segment_issue_repair_recovery_improvement_sum
                ),
            )
            * 0.10,
        )
        if segment_issue_repair_bonus > 0.0:
            score += segment_issue_repair_bonus
            reasons.append(f"segment issue repair measured improvement bonus +{segment_issue_repair_bonus:.3f}")
        else:
            reasons.append("segment issue repair did not improve recalculated issue metrics")
    if (
        no_vocal_cleanup_window_count is not None
        and no_vocal_cleanup_confidence_sum is not None
        and no_vocal_cleanup_suppressed_duration_sec is not None
        and float(no_vocal_cleanup_window_count) > 0.0
    ):
        cleanup_bonus = min(
            1.75,
            float(no_vocal_cleanup_window_count) * 0.080
            + float(no_vocal_cleanup_confidence_sum) * 0.050
            + float(no_vocal_cleanup_suppressed_duration_sec) * 0.012,
        )
        if cleanup_bonus > 0.0:
            score += cleanup_bonus
            reasons.append(f"no-vocal leakage cleanup bonus +{cleanup_bonus:.3f}")
    if (
        vocal_harshness_guard_window_count is not None
        and vocal_harshness_guard_high_band_reduction_ratio is not None
        and float(vocal_harshness_guard_window_count) > 0.0
        and float(vocal_harshness_guard_high_band_reduction_ratio) > 0.0
    ):
        harshness_bonus = min(
            0.16,
            float(vocal_harshness_guard_window_count) * 0.018
            + float(vocal_harshness_guard_high_band_reduction_ratio) * 0.18,
        )
        if harshness_bonus > 0.0:
            score += harshness_bonus
            reasons.append(f"vocal harshness guard high-band reduction bonus +{harshness_bonus:.3f}")

    if presence_ratio_db < -30.0:
        penalty = (-30.0 - presence_ratio_db) * 0.05
        score -= penalty
        reasons.append(f"presence loss penalty -{penalty:.3f}")
    if air_ratio_db < -40.0:
        penalty = (-40.0 - air_ratio_db) * 0.035
        score -= penalty
        reasons.append(f"air/detail loss penalty -{penalty:.3f}")
    if peak_dbfs > -0.05:
        penalty = (peak_dbfs + 0.05) * 0.30
        score -= penalty
        reasons.append(f"near-clipping penalty -{penalty:.3f}")

    return VocalCandidateScore(
        profile_name=profile.name,
        score=round(score, 6),
        reasons=reasons,
        metrics=metrics,
        manifest_path=str(manifest_path),
        output_dir=str(output_dir),
        vocals_path=str(vocals_path),
    )


def write_listen_pack(
    *,
    source_audio_path: Path,
    selected_vocals_path: Path,
    candidate_scores: list[VocalCandidateScore],
    listen_dir: Path,
    selected_label: str,
) -> dict[str, str]:
    listen_dir.mkdir(parents=True, exist_ok=True)
    for generated_path in [
        listen_dir / "00_raw_ref.flac",
        listen_dir / selected_label,
        listen_dir / "candidate_map.json",
        listen_dir / "00_README.txt",
    ]:
        generated_path.unlink(missing_ok=True)
    for stale_candidate_path in listen_dir.glob("candidate_*.flac"):
        stale_candidate_path.unlink(missing_ok=True)

    raw_audio, raw_sr = sf.read(str(source_audio_path), always_2d=True)
    raw_ref_path = listen_dir / "00_raw_ref.flac"
    sf.write(str(raw_ref_path), raw_audio, int(raw_sr), subtype="PCM_24")
    selected_path = listen_dir / selected_label
    shutil.copyfile(selected_vocals_path, selected_path)
    mapping = {
        "00_raw_ref.flac": str(source_audio_path),
        selected_label: str(selected_vocals_path),
    }
    for index, candidate in enumerate(
        sorted(candidate_scores, key=lambda item: item.score, reverse=True),
        start=1,
    ):
        candidate_label = f"candidate_{index:02d}_{candidate.profile_name}.flac"
        candidate_target = listen_dir / candidate_label
        shutil.copyfile(candidate.vocals_path, candidate_target)
        mapping[candidate_label] = candidate.vocals_path
    (listen_dir / "00_README.txt").write_text(
        "Listen only here.\n00 = raw full mix reference.\n01 = adaptive selected isolated vocals.\n",
        encoding="utf-8",
    )
    write_json(listen_dir / "candidate_map.json", mapping)
    return {"raw_ref_flac": str(raw_ref_path), "selected_vocals_flac": str(selected_path)}


def write_segment_issue_listen_clips(
    *,
    source_audio_path: Path,
    selected_vocals_path: Path,
    issue_report: dict[str, Any],
    listen_dir: Path,
    max_clips: int,
) -> dict[str, str]:
    if int(max_clips) <= 0:
        return {}

    issue_dir = listen_dir / "segment_issues"
    issue_dir.mkdir(parents=True, exist_ok=True)
    for generated_path in issue_dir.glob("issue_*.flac"):
        generated_path.unlink(missing_ok=True)
    manifest_path = issue_dir / "segment_issue_clips.json"
    manifest_path.unlink(missing_ok=True)

    clips: list[dict[str, Any]] = []
    for index, segment in enumerate(issue_report.get("top_segments", [])[: int(max_clips)], start=1):
        labels = [
            _validate_candidate_name(str(label))
            for label in segment.get("labels", [])
            if str(label) != "acceptable_proxy_window"
        ]
        label_slug = "_".join(labels[:2]) if labels else "watch"
        start_sec = float(segment["start_sec"])
        end_sec = float(segment["end_sec"])
        relative_start_sec = float(segment["relative_start_sec"])
        relative_end_sec = float(segment["relative_end_sec"])

        raw_audio, raw_sr = _read_source_window(source_audio_path, start_sec=start_sec, end_sec=end_sec)
        selected_audio, selected_sr = _read_source_window(
            selected_vocals_path,
            start_sec=relative_start_sec,
            end_sec=relative_end_sec,
        )
        raw_label = f"issue_{index:02d}_{label_slug}_raw_{start_sec:.2f}_{end_sec:.2f}.flac"
        selected_label = f"issue_{index:02d}_{label_slug}_selected_vocals_{start_sec:.2f}_{end_sec:.2f}.flac"
        raw_path = issue_dir / raw_label
        selected_path = issue_dir / selected_label
        sf.write(str(raw_path), raw_audio, int(raw_sr), subtype="PCM_24")
        sf.write(str(selected_path), selected_audio, int(selected_sr), subtype="PCM_24")
        clips.append(
            {
                "rank": index,
                "start_sec": round(start_sec, 6),
                "end_sec": round(end_sec, 6),
                "labels": segment.get("labels", []),
                "severity_score": segment.get("severity_score", 0.0),
                "raw_clip": str(raw_path),
                "selected_vocals_clip": str(selected_path),
            }
        )

    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "mode": "vocal_segment_issue_listen_clips",
            "source_audio_path": str(source_audio_path),
            "selected_vocals_path": str(selected_vocals_path),
            "clips": clips,
        },
    )
    return {"segment_issue_clips_manifest": str(manifest_path), "segment_issue_clips_dir": str(issue_dir)}


def run_adaptive_vocal_isolation(args: argparse.Namespace) -> dict[str, Any]:
    source_audio_path = Path(args.source_audio_path)
    output_dir = Path(args.output_dir)
    export_script = Path(args.export_script)
    listen_dir = Path(args.listen_dir) if args.listen_dir else None
    feedback_store_path = Path(args.feedback_store) if args.feedback_store else None
    start_sec = float(args.start_sec)

    if not source_audio_path.exists():
        raise FileNotFoundError(source_audio_path)
    if not export_script.exists():
        raise FileNotFoundError(export_script)
    end_sec = audio_duration_sec(source_audio_path) if args.end_sec is None else float(args.end_sec)
    if end_sec <= start_sec:
        raise ValueError(f"Invalid vocal extraction window: {start_sec}:{end_sec}")
    if float(args.segment_consistency_sec) < 0.0:
        raise ValueError("segment_consistency_sec must be zero or greater")
    if int(args.segment_issue_report_max_windows) < 0:
        raise ValueError("segment_issue_report_max_windows must be zero or greater")
    if int(args.segment_issue_listen_clip_count) < 0:
        raise ValueError("segment_issue_listen_clip_count must be zero or greater")

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_log_path = output_dir / "adaptive_vocal_progress.jsonl"
    stage_profiler = StageProfiler(
        emit_heartbeat=not bool(getattr(args, "disable_stage_heartbeat", False)),
        event_log_path=progress_log_path,
    )
    analysis_cache = AudioAnalysisCache()
    external_only = bool(getattr(args, "external_only", False))
    bounded_thin_repair_only = bool(getattr(args, "bounded_thin_repair_only", False))
    disable_anchor_suppression = bool(args.disable_anchor_suppression) or bounded_thin_repair_only
    disable_body_repair = bool(args.disable_body_repair)
    disable_segment_router = bool(args.disable_segment_router) or bounded_thin_repair_only
    disable_issue_aware_segment_selection = bool(args.disable_issue_aware_segment_selection) or bounded_thin_repair_only
    disable_segment_issue_repair = bool(args.disable_segment_issue_repair)
    disable_no_vocal_leakage_cleanup = bool(args.disable_no_vocal_leakage_cleanup) or bounded_thin_repair_only
    disable_vocal_harshness_guard = bool(args.disable_vocal_harshness_guard) or bounded_thin_repair_only
    ir_instrumental_minus_path = Path(args.ir_instrumental_minus_path) if args.ir_instrumental_minus_path else None
    external_candidate_specs = list(args.external_vocal_candidate or [])
    if external_only and ir_instrumental_minus_path is None and not external_candidate_specs:
        raise ValueError("--external-only requires --external-vocal-candidate or --ir-instrumental-minus-path")
    profiles = () if external_only else select_profiles(args.profile, default_candidate_profiles())
    needs_bs_roformer_runtime = profiles_require_bs_roformer_runtime(profiles)
    execution_contract = {
        "route_name": "adaptive_vocal_isolation",
        "docker_runtime_required": True,
        "source_audio_path": str(source_audio_path),
        "output_dir": str(output_dir),
        "output_manifest_path": str(output_dir / "adaptive_vocal_manifest.json"),
        "progress_log_path": str(progress_log_path),
        "device": str(args.device),
        "external_only": external_only,
        "bounded_thin_repair_only": bounded_thin_repair_only,
        "candidate_profile_count": len(profiles),
        "external_candidate_count": len(external_candidate_specs) + (1 if ir_instrumental_minus_path is not None else 0),
        "gpu_inference_skipped": external_only,
        "gpu_inference_skip_reason": "external_only_precomputed_candidates" if external_only else None,
        "expected_acceleration_mode": (
            "external_only_cpu_dsp_scoring" if external_only else "gpu_inference_then_cpu_dsp_scoring"
        ),
    }
    stage_profiler.mark("execution_contract", **execution_contract)
    stage_profiler.mark(
        "candidate_plan",
        built_in_candidate_count=len(profiles),
        external_candidate_count=execution_contract["external_candidate_count"],
        needs_bs_roformer_runtime=needs_bs_roformer_runtime,
        gpu_inference_skipped=external_only,
        expected_acceleration_mode=execution_contract["expected_acceleration_mode"],
    )
    with stage_profiler.measure(
        "ensure_runtime",
        external_only=external_only,
        needs_bs_roformer_runtime=needs_bs_roformer_runtime,
        expected_acceleration_mode=execution_contract["expected_acceleration_mode"],
    ):
        if needs_bs_roformer_runtime:
            ensure_bs_roformer_runtime()
    with stage_profiler.measure("source_hash_and_feedback"):
        source_audio_sha256 = _sha256_file(source_audio_path)
        feedback_adjustments = load_feedback_adjustments(
            feedback_store_path,
            source_audio_sha256=source_audio_sha256,
            start_sec=start_sec,
            end_sec=end_sec,
        )
    def collect_proxy_metrics(profile_name: str, vocals_path: Path) -> dict[str, float]:
        with stage_profiler.measure("proxy_metrics", candidate=profile_name):
            return calculate_vocal_proxy_metrics(
                source_audio_path,
                vocals_path,
                start_sec=start_sec,
                end_sec=end_sec,
                analysis_cache=analysis_cache,
            )

    def collect_consistency_metrics(profile_name: str, vocals_path: Path) -> dict[str, float]:
        with stage_profiler.measure("segment_consistency_metrics", candidate=profile_name):
            return calculate_vocal_proxy_consistency_metrics(
                source_audio_path,
                vocals_path,
                start_sec=start_sec,
                end_sec=end_sec,
                segment_duration_sec=float(args.segment_consistency_sec),
                analysis_cache=analysis_cache,
            )

    def collect_segment_issue_report(
        profile_name: str,
        vocals_path: Path,
        *,
        anchor_vocals_path: Path | None,
    ) -> dict[str, Any]:
        with stage_profiler.measure("segment_issue_report", candidate=profile_name):
            return calculate_vocal_segment_issue_report(
                source_audio_path,
                vocals_path,
                anchor_vocals_path=anchor_vocals_path,
                start_sec=start_sec,
                end_sec=end_sec,
                segment_duration_sec=float(args.segment_consistency_sec),
                max_ranked_segments=int(args.segment_issue_report_max_windows),
                analysis_cache=analysis_cache,
            )

    def collect_anchor_support_metrics(profile_name: str, vocals_path: Path, anchor_vocals_path: Path) -> dict[str, float]:
        with stage_profiler.measure("anchor_support_metrics", candidate=profile_name):
            return calculate_anchor_support_metrics(
                vocals_path,
                anchor_vocals_path,
                analysis_cache=analysis_cache,
            )

    candidate_records: list[VocalCandidateRecord] = []
    candidate_records_by_name: dict[str, VocalCandidateRecord] = {}
    candidate_root_dir = output_dir / "candidates"
    candidate_root_dir.mkdir(parents=True, exist_ok=True)
    reuse_rejected_profiles: list[dict[str, str]] = []
    rebuilt_profile_names: set[str] = set()

    for profile in profiles:
        candidate_output_dir = candidate_root_dir / profile.name
        manifest_path = candidate_output_dir / "stem_manifest.json"
        manifest = load_reusable_candidate_manifest(
            manifest_path=manifest_path,
            profile=profile,
            source_audio_path=source_audio_path,
            start_sec=start_sec,
            end_sec=end_sec,
            reuse_existing_candidates=bool(args.reuse_existing_candidates),
            reuse_rejected_profiles=reuse_rejected_profiles,
        )
        if manifest is None:
            command = build_vocal_export_command(
                python_executable=args.python_executable,
                export_script=export_script,
                source_audio_path=source_audio_path,
                output_dir=candidate_output_dir,
                start_sec=start_sec,
                end_sec=end_sec,
                device=str(args.device),
                profile=profile,
            )
            with stage_profiler.measure("candidate_inference", candidate=profile.name, reused=False):
                _run_command(command)
            manifest = load_json(manifest_path)
            rebuilt_profile_names.add(profile.name)

        vocals_path = resolve_vocals_path(manifest)
        if not vocals_path.exists():
            raise FileNotFoundError(vocals_path)
        metrics = collect_proxy_metrics(profile.name, vocals_path)
        record = VocalCandidateRecord(
            profile=profile,
            manifest_path=manifest_path,
            output_dir=candidate_output_dir,
            vocals_path=vocals_path,
            metrics=metrics,
            manifest=manifest,
        )
        candidate_records.append(record)
        candidate_records_by_name[profile.name] = record

    external_profiles: list[VocalCandidateProfile] = []
    if ir_instrumental_minus_path is not None:
        if not ir_instrumental_minus_path.exists():
            raise FileNotFoundError(ir_instrumental_minus_path)
        external_profiles.append(
            VocalCandidateProfile(
                name="ir_residual_vocals",
                candidate_mode="ir_residual_vocals",
                prior_score=0.18,
                intent=(
                    "Use raw minus Instrumental Resurrection instrumental as a proxy vocal candidate "
                    "for swallowed syllables and dense drop recovery."
                ),
                tuning={
                    "instrumental_minus_path": str(ir_instrumental_minus_path),
                    "instrumental_gain": 1.0,
                    "source_contract": "proxy_candidate_not_ground_truth",
                },
            )
        )
    for candidate_name, candidate_path in external_candidate_specs:
        if not candidate_path.exists():
            raise FileNotFoundError(candidate_path)
        external_profiles.append(
            _external_candidate_profile(
                name=candidate_name,
                candidate_mode="external_vocals",
                source_path=candidate_path,
                prior_score=0.12,
                intent="Use a precomputed external vocal candidate such as MDX, MelBand, VirtualDJ, or a model-zoo output.",
            )
        )

    external_profile_names = [profile.name for profile in external_profiles]
    duplicate_profile_names = sorted(
        name
        for name in set(external_profile_names)
        if external_profile_names.count(name) > 1
    )
    if duplicate_profile_names:
        raise ValueError(f"Duplicate external vocal candidate name: {', '.join(duplicate_profile_names)}")
    duplicate_external_names = sorted(
        name
        for name in {profile.name for profile in external_profiles}
        if name in candidate_records_by_name
    )
    if duplicate_external_names:
        raise ValueError(f"External vocal candidate name collides with existing profile: {', '.join(duplicate_external_names)}")

    for profile in external_profiles:
        candidate_output_dir = candidate_root_dir / profile.name
        manifest_path = candidate_output_dir / "stem_manifest.json"
        manifest = load_reusable_candidate_manifest(
            manifest_path=manifest_path,
            profile=profile,
            source_audio_path=source_audio_path,
            start_sec=start_sec,
            end_sec=end_sec,
            reuse_existing_candidates=bool(args.reuse_existing_candidates),
            reuse_rejected_profiles=reuse_rejected_profiles,
        )
        if manifest is None:
            candidate_output_dir.mkdir(parents=True, exist_ok=True)
            with stage_profiler.measure(
                "external_candidate_materialization",
                candidate=profile.name,
                candidate_mode=profile.candidate_mode,
                gpu_inference_skipped=True,
                acceleration_mode="cpu_dsp_scoring",
            ):
                if profile.name == "ir_residual_vocals":
                    manifest = build_ir_residual_vocal_candidate(
                        profile=profile,
                        source_audio_path=source_audio_path,
                        instrumental_minus_path=ir_instrumental_minus_path,
                        output_dir=candidate_output_dir,
                        start_sec=start_sec,
                        end_sec=end_sec,
                    )
                else:
                    source_path_value = profile.tuning.get("source_path")
                    if not isinstance(source_path_value, str):
                        raise ValueError(f"External candidate {profile.name} has no source_path")
                    manifest = build_external_vocal_candidate(
                        profile=profile,
                        source_audio_path=source_audio_path,
                        candidate_audio_path=Path(source_path_value),
                        output_dir=candidate_output_dir,
                        start_sec=start_sec,
                        end_sec=end_sec,
                    )
            rebuilt_profile_names.add(profile.name)
        vocals_path = resolve_vocals_path(manifest)
        if not vocals_path.exists():
            raise FileNotFoundError(vocals_path)
        metrics = collect_proxy_metrics(profile.name, vocals_path)
        record = VocalCandidateRecord(
            profile=profile,
            manifest_path=manifest_path,
            output_dir=candidate_output_dir,
            vocals_path=vocals_path,
            metrics=metrics,
            manifest=manifest,
        )
        candidate_records.append(record)
        candidate_records_by_name[profile.name] = record

    derived_profiles: tuple[AnchorSuppressionProfile, ...] = (
        () if disable_anchor_suppression else default_anchor_suppression_profiles()
    )
    skipped_derived_profiles: list[dict[str, str]] = []
    for profile in derived_profiles:
        source_record = candidate_records_by_name.get(profile.source_profile_name)
        anchor_record = candidate_records_by_name.get(profile.anchor_profile_name)
        if source_record is None or anchor_record is None:
            skipped_derived_profiles.append(
                {
                    "name": profile.name,
                    "reason": "missing_source_or_anchor_candidate",
                    "source_profile": profile.source_profile_name,
                    "anchor_profile": profile.anchor_profile_name,
                }
            )
            continue
        candidate_output_dir = candidate_root_dir / profile.name
        manifest_path = candidate_output_dir / "stem_manifest.json"
        dependency_names = {profile.source_profile_name, profile.anchor_profile_name}
        expected_dependency_fingerprints = {
            profile.source_profile_name: _dependency_fingerprint(source_record),
            profile.anchor_profile_name: _dependency_fingerprint(anchor_record),
        }
        upstream_rebuilt = bool(dependency_names & rebuilt_profile_names)
        if upstream_rebuilt and bool(args.reuse_existing_candidates) and manifest_path.exists():
            reuse_rejected_profiles.append(
                {
                    "name": profile.name,
                    "reason": "upstream_candidate_rebuilt",
                    "manifest_path": str(manifest_path),
                }
            )
        manifest = None if upstream_rebuilt else load_reusable_candidate_manifest(
            manifest_path=manifest_path,
            profile=profile,
            source_audio_path=source_audio_path,
            start_sec=start_sec,
            end_sec=end_sec,
            reuse_existing_candidates=bool(args.reuse_existing_candidates),
            reuse_rejected_profiles=reuse_rejected_profiles,
            expected_dependency_fingerprints=expected_dependency_fingerprints,
        )
        if manifest is None:
            candidate_output_dir.mkdir(parents=True, exist_ok=True)
            with stage_profiler.measure(
                "derived_candidate_build",
                candidate=profile.name,
                builder="build_anchor_suppressed_candidate",
                acceleration_mode="cpu_dsp_scoring",
            ):
                manifest = build_anchor_suppressed_candidate(
                    profile=profile,
                    source_record=source_record,
                    anchor_record=anchor_record,
                    output_dir=candidate_output_dir,
                )
            rebuilt_profile_names.add(profile.name)
        vocals_path = resolve_vocals_path(manifest)
        if not vocals_path.exists():
            raise FileNotFoundError(vocals_path)
        metrics = collect_proxy_metrics(profile.name, vocals_path)
        record = VocalCandidateRecord(
            profile=profile,
            manifest_path=manifest_path,
            output_dir=candidate_output_dir,
            vocals_path=vocals_path,
            metrics=metrics,
            manifest=manifest,
        )
        candidate_records.append(record)
        candidate_records_by_name[profile.name] = record

    if disable_body_repair:
        body_repair_profiles: tuple[VocalBodyRepairProfile, ...] = ()
    elif bounded_thin_repair_only:
        body_repair_profiles = default_bounded_thin_repair_body_profiles()
    else:
        body_repair_profiles = default_vocal_body_repair_profiles()
    skipped_body_repair_profiles: list[dict[str, str]] = []
    for profile in body_repair_profiles:
        base_record = candidate_records_by_name.get(profile.base_profile_name)
        missing_donor_profiles = [
            donor_name
            for donor_name in profile.donor_profile_names
            if donor_name not in candidate_records_by_name
        ]
        if base_record is None or missing_donor_profiles:
            skipped_body_repair_profiles.append(
                {
                    "name": profile.name,
                    "reason": "missing_base_or_donor_candidate",
                    "base_profile": profile.base_profile_name,
                    "missing_donor_profiles": ",".join(missing_donor_profiles),
                }
            )
            continue
        candidate_output_dir = candidate_root_dir / profile.name
        manifest_path = candidate_output_dir / "stem_manifest.json"
        dependency_names = {profile.base_profile_name, *profile.donor_profile_names}
        expected_dependency_fingerprints = {
            profile.base_profile_name: _dependency_fingerprint(base_record),
            **{
                donor_name: _dependency_fingerprint(candidate_records_by_name[donor_name])
                for donor_name in profile.donor_profile_names
            },
        }
        upstream_rebuilt = bool(dependency_names & rebuilt_profile_names)
        if upstream_rebuilt and bool(args.reuse_existing_candidates) and manifest_path.exists():
            reuse_rejected_profiles.append(
                {
                    "name": profile.name,
                    "reason": "upstream_candidate_rebuilt",
                    "manifest_path": str(manifest_path),
                }
            )
        manifest = None if upstream_rebuilt else load_reusable_candidate_manifest(
            manifest_path=manifest_path,
            profile=profile,
            source_audio_path=source_audio_path,
            start_sec=start_sec,
            end_sec=end_sec,
            reuse_existing_candidates=bool(args.reuse_existing_candidates),
            reuse_rejected_profiles=reuse_rejected_profiles,
            expected_dependency_fingerprints=expected_dependency_fingerprints,
        )
        if manifest is None:
            candidate_output_dir.mkdir(parents=True, exist_ok=True)
            with stage_profiler.measure(
                "derived_candidate_build",
                candidate=profile.name,
                builder="build_vocal_body_repair_candidate",
                acceleration_mode="cpu_dsp_scoring",
            ):
                manifest = build_vocal_body_repair_candidate(
                    profile=profile,
                    base_record=base_record,
                    donor_records={
                        donor_name: candidate_records_by_name[donor_name]
                        for donor_name in profile.donor_profile_names
                    },
                    output_dir=candidate_output_dir,
                )
            rebuilt_profile_names.add(profile.name)
        vocals_path = resolve_vocals_path(manifest)
        if not vocals_path.exists():
            raise FileNotFoundError(vocals_path)
        metrics = collect_proxy_metrics(profile.name, vocals_path)
        record = VocalCandidateRecord(
            profile=profile,
            manifest_path=manifest_path,
            output_dir=candidate_output_dir,
            vocals_path=vocals_path,
            metrics=metrics,
            manifest=manifest,
        )
        candidate_records.append(record)
        candidate_records_by_name[profile.name] = record

    segment_router_profiles: tuple[SegmentRouterProfile, ...] = (
        () if disable_segment_router else default_segment_router_profiles()
    )
    skipped_segment_router_profiles: list[dict[str, str]] = []
    for profile in segment_router_profiles:
        required_records = {
            "clean_profile": candidate_records_by_name.get(profile.clean_profile_name),
            "strict_profile": candidate_records_by_name.get(profile.strict_profile_name),
            "repair_profile": candidate_records_by_name.get(profile.repair_profile_name),
            "preserve_profile": candidate_records_by_name.get(profile.preserve_profile_name),
            "anchor_profile": candidate_records_by_name.get(profile.anchor_profile_name),
        }
        missing_dependencies = [
            key
            for key, record in required_records.items()
            if record is None
        ]
        if missing_dependencies:
            skipped_segment_router_profiles.append(
                {
                    "name": profile.name,
                    "reason": "missing_router_dependency",
                    "missing_dependencies": ",".join(missing_dependencies),
                    "clean_profile": profile.clean_profile_name,
                    "strict_profile": profile.strict_profile_name,
                    "repair_profile": profile.repair_profile_name,
                    "preserve_profile": profile.preserve_profile_name,
                    "anchor_profile": profile.anchor_profile_name,
                }
            )
            continue
        candidate_output_dir = candidate_root_dir / profile.name
        manifest_path = candidate_output_dir / "stem_manifest.json"
        dependency_names = {
            profile.clean_profile_name,
            profile.strict_profile_name,
            profile.repair_profile_name,
            profile.preserve_profile_name,
            profile.anchor_profile_name,
        }
        expected_dependency_fingerprints = {
            profile.clean_profile_name: _dependency_fingerprint(required_records["clean_profile"]),
            profile.strict_profile_name: _dependency_fingerprint(required_records["strict_profile"]),
            profile.repair_profile_name: _dependency_fingerprint(required_records["repair_profile"]),
            profile.preserve_profile_name: _dependency_fingerprint(required_records["preserve_profile"]),
            profile.anchor_profile_name: _dependency_fingerprint(required_records["anchor_profile"]),
        }
        upstream_rebuilt = bool(dependency_names & rebuilt_profile_names)
        if upstream_rebuilt and bool(args.reuse_existing_candidates) and manifest_path.exists():
            reuse_rejected_profiles.append(
                {
                    "name": profile.name,
                    "reason": "upstream_candidate_rebuilt",
                    "manifest_path": str(manifest_path),
                }
            )
        manifest = None if upstream_rebuilt else load_reusable_candidate_manifest(
            manifest_path=manifest_path,
            profile=profile,
            source_audio_path=source_audio_path,
            start_sec=start_sec,
            end_sec=end_sec,
            reuse_existing_candidates=bool(args.reuse_existing_candidates),
            reuse_rejected_profiles=reuse_rejected_profiles,
            expected_dependency_fingerprints=expected_dependency_fingerprints,
        )
        if manifest is None:
            candidate_output_dir.mkdir(parents=True, exist_ok=True)
            with stage_profiler.measure(
                "derived_candidate_build",
                candidate=profile.name,
                builder="build_segment_router_candidate",
                acceleration_mode="cpu_dsp_scoring",
            ):
                manifest = build_segment_router_candidate(
                    profile=profile,
                    source_audio_path=source_audio_path,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    clean_record=required_records["clean_profile"],
                    strict_record=required_records["strict_profile"],
                    repair_record=required_records["repair_profile"],
                    preserve_record=required_records["preserve_profile"],
                    anchor_record=required_records["anchor_profile"],
                    output_dir=candidate_output_dir,
                )
            rebuilt_profile_names.add(profile.name)
        vocals_path = resolve_vocals_path(manifest)
        if not vocals_path.exists():
            raise FileNotFoundError(vocals_path)
        metrics = collect_proxy_metrics(profile.name, vocals_path)
        record = VocalCandidateRecord(
            profile=profile,
            manifest_path=manifest_path,
            output_dir=candidate_output_dir,
            vocals_path=vocals_path,
            metrics=metrics,
            manifest=manifest,
        )
        candidate_records.append(record)
        candidate_records_by_name[profile.name] = record

    anchor_record = candidate_records_by_name.get("bs_artifact_clean")
    candidate_scores: list[VocalCandidateScore] = []
    candidate_issue_reports: dict[str, dict[str, Any]] = {}
    for record in candidate_records:
        metrics = dict(record.metrics)
        if float(args.segment_consistency_sec) > 0.0:
            metrics.update(collect_consistency_metrics(record.profile.name, record.vocals_path))
            issue_report = collect_segment_issue_report(
                record.profile.name,
                record.vocals_path,
                anchor_vocals_path=None if anchor_record is None else anchor_record.vocals_path,
            )
            metrics.update(issue_report["summary_metrics"])
            candidate_issue_reports[record.profile.name] = issue_report
        if anchor_record is not None:
            metrics.update(collect_anchor_support_metrics(record.profile.name, record.vocals_path, anchor_record.vocals_path))
        candidate_scores.append(
            score_candidate(
                profile=record.profile,
                metrics=metrics,
                feedback_adjustments=feedback_adjustments,
                manifest_path=record.manifest_path,
                output_dir=record.output_dir,
                vocals_path=record.vocals_path,
            )
        )

    issue_aware_profile_template = default_issue_aware_segment_selection_profile()
    issue_aware_profile = VocalCandidateProfile(
        name=issue_aware_profile_template.name,
        candidate_mode=issue_aware_profile_template.candidate_mode,
        prior_score=issue_aware_profile_template.prior_score,
        intent=issue_aware_profile_template.intent,
        tuning={
            **issue_aware_profile_template.tuning,
            "segment_consistency_sec": round(float(args.segment_consistency_sec), 6),
        },
    )
    skipped_issue_aware_segment_selection: list[dict[str, str]] = []
    if disable_issue_aware_segment_selection:
        skipped_issue_aware_segment_selection.append(
            {"name": issue_aware_profile.name, "reason": "disabled_by_flag"}
        )
    elif float(args.segment_consistency_sec) <= 0.0:
        skipped_issue_aware_segment_selection.append(
            {"name": issue_aware_profile.name, "reason": "segment_consistency_disabled"}
        )
    elif not candidate_issue_reports:
        skipped_issue_aware_segment_selection.append(
            {"name": issue_aware_profile.name, "reason": "missing_candidate_issue_reports"}
        )
    else:
        initial_selected_score = max(candidate_scores, key=lambda candidate: candidate.score)
        candidate_output_dir = candidate_root_dir / issue_aware_profile.name
        manifest_path = candidate_output_dir / "stem_manifest.json"
        expected_dependency_fingerprints: dict[str, dict[str, str]] = {}
        for candidate_score in candidate_scores:
            source_record = candidate_records_by_name[candidate_score.profile_name]
            fingerprint = _dependency_fingerprint(source_record)
            issue_summary = candidate_issue_reports.get(candidate_score.profile_name, {}).get("summary", {})
            fingerprint["score"] = f"{candidate_score.score:.6f}"
            fingerprint["issue_summary_sha256"] = hashlib.sha256(
                json.dumps(issue_summary, sort_keys=True).encode("utf-8"),
            ).hexdigest()
            expected_dependency_fingerprints[candidate_score.profile_name] = fingerprint

        manifest = load_reusable_candidate_manifest(
            manifest_path=manifest_path,
            profile=issue_aware_profile,
            source_audio_path=source_audio_path,
            start_sec=start_sec,
            end_sec=end_sec,
            reuse_existing_candidates=bool(args.reuse_existing_candidates),
            reuse_rejected_profiles=reuse_rejected_profiles,
            expected_dependency_fingerprints=expected_dependency_fingerprints,
        )
        if manifest is None:
            candidate_output_dir.mkdir(parents=True, exist_ok=True)
            with stage_profiler.measure(
                "derived_candidate_build",
                candidate=issue_aware_profile.name,
                builder="build_issue_aware_segment_selection_candidate",
                acceleration_mode="cpu_dsp_scoring",
            ):
                manifest = build_issue_aware_segment_selection_candidate(
                    profile=issue_aware_profile,
                    base_score=initial_selected_score,
                    candidate_scores=candidate_scores,
                    candidate_records_by_name=candidate_records_by_name,
                    candidate_issue_reports=candidate_issue_reports,
                    dependency_fingerprints=expected_dependency_fingerprints,
                    output_dir=candidate_output_dir,
                )
            if manifest is not None:
                rebuilt_profile_names.add(issue_aware_profile.name)

        if manifest is None:
            skipped_issue_aware_segment_selection.append(
                {"name": issue_aware_profile.name, "reason": "no_issue_window_improvement"}
            )
        else:
            vocals_path = resolve_vocals_path(manifest)
            if not vocals_path.exists():
                raise FileNotFoundError(vocals_path)
            metrics = collect_proxy_metrics(issue_aware_profile.name, vocals_path)
            metrics.update(_issue_aware_selection_metrics(manifest))
            if float(args.segment_consistency_sec) > 0.0:
                metrics.update(collect_consistency_metrics(issue_aware_profile.name, vocals_path))
                issue_report = collect_segment_issue_report(
                    issue_aware_profile.name,
                    vocals_path,
                    anchor_vocals_path=None if anchor_record is None else anchor_record.vocals_path,
                )
                metrics.update(issue_report["summary_metrics"])
                candidate_issue_reports[issue_aware_profile.name] = issue_report
            if anchor_record is not None:
                metrics.update(collect_anchor_support_metrics(issue_aware_profile.name, vocals_path, anchor_record.vocals_path))
            record = VocalCandidateRecord(
                profile=issue_aware_profile,
                manifest_path=manifest_path,
                output_dir=candidate_output_dir,
                vocals_path=vocals_path,
                metrics=metrics,
                manifest=manifest,
            )
            candidate_records.append(record)
            candidate_records_by_name[issue_aware_profile.name] = record
            candidate_scores.append(
                score_candidate(
                    profile=issue_aware_profile,
                    metrics=metrics,
                    feedback_adjustments=feedback_adjustments,
                    manifest_path=manifest_path,
                    output_dir=candidate_output_dir,
                    vocals_path=vocals_path,
                )
            )

    segment_issue_repair_profile_template = default_segment_issue_repair_profile()
    segment_issue_repair_profile = VocalCandidateProfile(
        name=segment_issue_repair_profile_template.name,
        candidate_mode=segment_issue_repair_profile_template.candidate_mode,
        prior_score=segment_issue_repair_profile_template.prior_score,
        intent=segment_issue_repair_profile_template.intent,
        tuning={
            **segment_issue_repair_profile_template.tuning,
            "segment_consistency_sec": round(float(args.segment_consistency_sec), 6),
        },
    )
    skipped_segment_issue_repair: list[dict[str, str]] = []
    if bounded_thin_repair_only:
        skipped_segment_issue_repair.append(
            {"name": segment_issue_repair_profile.name, "reason": "disabled_by_bounded_thin_repair_only"}
        )
    elif disable_segment_issue_repair:
        skipped_segment_issue_repair.append(
            {"name": segment_issue_repair_profile.name, "reason": "disabled_by_flag"}
        )
    elif float(args.segment_consistency_sec) <= 0.0:
        skipped_segment_issue_repair.append(
            {"name": segment_issue_repair_profile.name, "reason": "segment_consistency_disabled"}
        )
    elif not candidate_issue_reports:
        skipped_segment_issue_repair.append(
            {"name": segment_issue_repair_profile.name, "reason": "missing_candidate_issue_reports"}
        )
    else:
        repair_base_score = max(candidate_scores, key=lambda candidate: candidate.score)
        candidate_output_dir = candidate_root_dir / segment_issue_repair_profile.name
        manifest_path = candidate_output_dir / "stem_manifest.json"
        expected_dependency_fingerprints = {}
        for candidate_score in candidate_scores:
            source_record = candidate_records_by_name[candidate_score.profile_name]
            fingerprint = _dependency_fingerprint(source_record)
            issue_summary = candidate_issue_reports.get(candidate_score.profile_name, {}).get("summary", {})
            fingerprint["score"] = f"{candidate_score.score:.6f}"
            fingerprint["issue_summary_sha256"] = hashlib.sha256(
                json.dumps(issue_summary, sort_keys=True).encode("utf-8"),
            ).hexdigest()
            expected_dependency_fingerprints[candidate_score.profile_name] = fingerprint

        manifest = load_reusable_candidate_manifest(
            manifest_path=manifest_path,
            profile=segment_issue_repair_profile,
            source_audio_path=source_audio_path,
            start_sec=start_sec,
            end_sec=end_sec,
            reuse_existing_candidates=bool(args.reuse_existing_candidates),
            reuse_rejected_profiles=reuse_rejected_profiles,
            expected_dependency_fingerprints=expected_dependency_fingerprints,
        )
        if manifest is None:
            candidate_output_dir.mkdir(parents=True, exist_ok=True)
            with stage_profiler.measure(
                "derived_candidate_build",
                candidate=segment_issue_repair_profile.name,
                builder="build_segment_issue_repair_candidate",
                acceleration_mode="cpu_dsp_scoring",
            ):
                manifest = build_segment_issue_repair_candidate(
                    profile=segment_issue_repair_profile,
                    source_audio_path=source_audio_path,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    base_score=repair_base_score,
                    candidate_scores=candidate_scores,
                    candidate_records_by_name=candidate_records_by_name,
                    candidate_issue_reports=candidate_issue_reports,
                    dependency_fingerprints=expected_dependency_fingerprints,
                    output_dir=candidate_output_dir,
                    progress=lambda **metadata: stage_profiler.mark(
                        "segment_issue_repair_window",
                        candidate=segment_issue_repair_profile.name,
                        builder="build_segment_issue_repair_candidate",
                        **metadata,
                    ),
                )
            if manifest is not None:
                rebuilt_profile_names.add(segment_issue_repair_profile.name)

        if manifest is None:
            skipped_segment_issue_repair.append(
                {"name": segment_issue_repair_profile.name, "reason": "no_repairable_issue_window"}
            )
        else:
            vocals_path = resolve_vocals_path(manifest)
            if not vocals_path.exists():
                raise FileNotFoundError(vocals_path)
            metrics = collect_proxy_metrics(segment_issue_repair_profile.name, vocals_path)
            metrics.update(_segment_issue_repair_metrics(manifest))
            if float(args.segment_consistency_sec) > 0.0:
                metrics.update(collect_consistency_metrics(segment_issue_repair_profile.name, vocals_path))
                issue_report = collect_segment_issue_report(
                    segment_issue_repair_profile.name,
                    vocals_path,
                    anchor_vocals_path=None if anchor_record is None else anchor_record.vocals_path,
                )
                metrics.update(issue_report["summary_metrics"])
                candidate_issue_reports[segment_issue_repair_profile.name] = issue_report
            if anchor_record is not None:
                metrics.update(collect_anchor_support_metrics(segment_issue_repair_profile.name, vocals_path, anchor_record.vocals_path))
            record = VocalCandidateRecord(
                profile=segment_issue_repair_profile,
                manifest_path=manifest_path,
                output_dir=candidate_output_dir,
                vocals_path=vocals_path,
                metrics=metrics,
                manifest=manifest,
            )
            candidate_records.append(record)
            candidate_records_by_name[segment_issue_repair_profile.name] = record
            candidate_scores.append(
                score_candidate(
                    profile=segment_issue_repair_profile,
                    metrics=metrics,
                    feedback_adjustments=feedback_adjustments,
                    manifest_path=manifest_path,
                    output_dir=candidate_output_dir,
                    vocals_path=vocals_path,
                )
            )

    no_vocal_cleanup_profile_template = default_no_vocal_leakage_cleanup_profile()
    no_vocal_cleanup_profile = VocalCandidateProfile(
        name=no_vocal_cleanup_profile_template.name,
        candidate_mode=no_vocal_cleanup_profile_template.candidate_mode,
        prior_score=no_vocal_cleanup_profile_template.prior_score,
        intent=no_vocal_cleanup_profile_template.intent,
        tuning={
            **no_vocal_cleanup_profile_template.tuning,
            "segment_consistency_sec": round(float(args.segment_consistency_sec), 6),
        },
    )
    skipped_no_vocal_leakage_cleanup: list[dict[str, str]] = []
    if disable_no_vocal_leakage_cleanup:
        skipped_no_vocal_leakage_cleanup.append(
            {"name": no_vocal_cleanup_profile.name, "reason": "disabled_by_flag"}
        )
    elif float(args.segment_consistency_sec) <= 0.0:
        skipped_no_vocal_leakage_cleanup.append(
            {"name": no_vocal_cleanup_profile.name, "reason": "segment_consistency_disabled"}
        )
    elif not candidate_issue_reports:
        skipped_no_vocal_leakage_cleanup.append(
            {"name": no_vocal_cleanup_profile.name, "reason": "missing_candidate_issue_reports"}
        )
    else:
        cleanup_base_score = max(candidate_scores, key=lambda candidate: candidate.score)
        candidate_output_dir = candidate_root_dir / no_vocal_cleanup_profile.name
        manifest_path = candidate_output_dir / "stem_manifest.json"
        expected_dependency_fingerprints = {}
        for candidate_score in candidate_scores:
            source_record = candidate_records_by_name[candidate_score.profile_name]
            fingerprint = _dependency_fingerprint(source_record)
            issue_summary = candidate_issue_reports.get(candidate_score.profile_name, {}).get("summary", {})
            fingerprint["score"] = f"{candidate_score.score:.6f}"
            fingerprint["issue_summary_sha256"] = hashlib.sha256(
                json.dumps(issue_summary, sort_keys=True).encode("utf-8"),
            ).hexdigest()
            expected_dependency_fingerprints[candidate_score.profile_name] = fingerprint

        manifest = load_reusable_candidate_manifest(
            manifest_path=manifest_path,
            profile=no_vocal_cleanup_profile,
            source_audio_path=source_audio_path,
            start_sec=start_sec,
            end_sec=end_sec,
            reuse_existing_candidates=bool(args.reuse_existing_candidates),
            reuse_rejected_profiles=reuse_rejected_profiles,
            expected_dependency_fingerprints=expected_dependency_fingerprints,
        )
        if manifest is None:
            candidate_output_dir.mkdir(parents=True, exist_ok=True)
            with stage_profiler.measure(
                "derived_candidate_build",
                candidate=no_vocal_cleanup_profile.name,
                builder="build_no_vocal_leakage_cleanup_candidate",
                acceleration_mode="cpu_dsp_scoring",
            ):
                manifest = build_no_vocal_leakage_cleanup_candidate(
                    profile=no_vocal_cleanup_profile,
                    base_score=cleanup_base_score,
                    candidate_scores=candidate_scores,
                    candidate_records_by_name=candidate_records_by_name,
                    candidate_issue_reports=candidate_issue_reports,
                    dependency_fingerprints=expected_dependency_fingerprints,
                    output_dir=candidate_output_dir,
                )
            if manifest is not None:
                rebuilt_profile_names.add(no_vocal_cleanup_profile.name)

        if manifest is None:
            skipped_no_vocal_leakage_cleanup.append(
                {"name": no_vocal_cleanup_profile.name, "reason": "no_consensus_no_vocal_window"}
            )
        else:
            vocals_path = resolve_vocals_path(manifest)
            if not vocals_path.exists():
                raise FileNotFoundError(vocals_path)
            metrics = collect_proxy_metrics(no_vocal_cleanup_profile.name, vocals_path)
            metrics.update(_no_vocal_leakage_cleanup_metrics(manifest))
            if float(args.segment_consistency_sec) > 0.0:
                metrics.update(collect_consistency_metrics(no_vocal_cleanup_profile.name, vocals_path))
                issue_report = collect_segment_issue_report(
                    no_vocal_cleanup_profile.name,
                    vocals_path,
                    anchor_vocals_path=None if anchor_record is None else anchor_record.vocals_path,
                )
                metrics.update(issue_report["summary_metrics"])
                candidate_issue_reports[no_vocal_cleanup_profile.name] = issue_report
            if anchor_record is not None:
                metrics.update(collect_anchor_support_metrics(no_vocal_cleanup_profile.name, vocals_path, anchor_record.vocals_path))
            record = VocalCandidateRecord(
                profile=no_vocal_cleanup_profile,
                manifest_path=manifest_path,
                output_dir=candidate_output_dir,
                vocals_path=vocals_path,
                metrics=metrics,
                manifest=manifest,
            )
            candidate_records.append(record)
            candidate_records_by_name[no_vocal_cleanup_profile.name] = record
            candidate_scores.append(
                score_candidate(
                    profile=no_vocal_cleanup_profile,
                    metrics=metrics,
                    feedback_adjustments=feedback_adjustments,
                    manifest_path=manifest_path,
                    output_dir=candidate_output_dir,
                    vocals_path=vocals_path,
                )
            )

    harshness_guard_profile_template = default_vocal_harshness_guard_profile()
    harshness_guard_profile = VocalCandidateProfile(
        name=harshness_guard_profile_template.name,
        candidate_mode=harshness_guard_profile_template.candidate_mode,
        prior_score=harshness_guard_profile_template.prior_score,
        intent=harshness_guard_profile_template.intent,
        tuning={
            **harshness_guard_profile_template.tuning,
            "segment_consistency_sec": round(float(args.segment_consistency_sec), 6),
        },
    )
    skipped_vocal_harshness_guard: list[dict[str, str]] = []
    if disable_vocal_harshness_guard:
        skipped_vocal_harshness_guard.append(
            {"name": harshness_guard_profile.name, "reason": "disabled_by_flag"}
        )
    elif float(args.segment_consistency_sec) <= 0.0:
        skipped_vocal_harshness_guard.append(
            {"name": harshness_guard_profile.name, "reason": "segment_consistency_disabled"}
        )
    elif not candidate_issue_reports:
        skipped_vocal_harshness_guard.append(
            {"name": harshness_guard_profile.name, "reason": "missing_candidate_issue_reports"}
        )
    elif str(harshness_guard_profile.tuning["anchor_profile"]) not in candidate_records_by_name:
        skipped_vocal_harshness_guard.append(
            {"name": harshness_guard_profile.name, "reason": "missing_anchor_profile"}
        )
    else:
        harshness_base_score = max(candidate_scores, key=lambda candidate: candidate.score)
        candidate_output_dir = candidate_root_dir / harshness_guard_profile.name
        manifest_path = candidate_output_dir / "stem_manifest.json"
        expected_dependency_fingerprints = {}
        for dependency_name in {
            harshness_base_score.profile_name,
            str(harshness_guard_profile.tuning["anchor_profile"]),
        }:
            source_record = candidate_records_by_name[dependency_name]
            fingerprint = _dependency_fingerprint(source_record)
            issue_summary = candidate_issue_reports.get(dependency_name, {}).get("summary", {})
            fingerprint["score"] = next(
                (
                    f"{candidate_score.score:.6f}"
                    for candidate_score in candidate_scores
                    if candidate_score.profile_name == dependency_name
                ),
                "0.000000",
            )
            fingerprint["issue_summary_sha256"] = hashlib.sha256(
                json.dumps(issue_summary, sort_keys=True).encode("utf-8"),
            ).hexdigest()
            expected_dependency_fingerprints[dependency_name] = fingerprint

        manifest = load_reusable_candidate_manifest(
            manifest_path=manifest_path,
            profile=harshness_guard_profile,
            source_audio_path=source_audio_path,
            start_sec=start_sec,
            end_sec=end_sec,
            reuse_existing_candidates=bool(args.reuse_existing_candidates),
            reuse_rejected_profiles=reuse_rejected_profiles,
            expected_dependency_fingerprints=expected_dependency_fingerprints,
        )
        if manifest is None:
            candidate_output_dir.mkdir(parents=True, exist_ok=True)
            with stage_profiler.measure(
                "derived_candidate_build",
                candidate=harshness_guard_profile.name,
                builder="build_vocal_harshness_guard_candidate",
                acceleration_mode="cpu_dsp_scoring",
            ):
                manifest = build_vocal_harshness_guard_candidate(
                    profile=harshness_guard_profile,
                    base_score=harshness_base_score,
                    candidate_records_by_name=candidate_records_by_name,
                    candidate_issue_reports=candidate_issue_reports,
                    dependency_fingerprints=expected_dependency_fingerprints,
                    output_dir=candidate_output_dir,
                )
            if manifest is not None:
                rebuilt_profile_names.add(harshness_guard_profile.name)

        if manifest is None:
            skipped_vocal_harshness_guard.append(
                {"name": harshness_guard_profile.name, "reason": "no_harshness_guard_window"}
            )
        else:
            vocals_path = resolve_vocals_path(manifest)
            if not vocals_path.exists():
                raise FileNotFoundError(vocals_path)
            metrics = collect_proxy_metrics(harshness_guard_profile.name, vocals_path)
            metrics.update(_vocal_harshness_guard_metrics(manifest))
            if float(args.segment_consistency_sec) > 0.0:
                metrics.update(collect_consistency_metrics(harshness_guard_profile.name, vocals_path))
                issue_report = collect_segment_issue_report(
                    harshness_guard_profile.name,
                    vocals_path,
                    anchor_vocals_path=None if anchor_record is None else anchor_record.vocals_path,
                )
                metrics.update(issue_report["summary_metrics"])
                candidate_issue_reports[harshness_guard_profile.name] = issue_report
            if anchor_record is not None:
                metrics.update(collect_anchor_support_metrics(harshness_guard_profile.name, vocals_path, anchor_record.vocals_path))
            record = VocalCandidateRecord(
                profile=harshness_guard_profile,
                manifest_path=manifest_path,
                output_dir=candidate_output_dir,
                vocals_path=vocals_path,
                metrics=metrics,
                manifest=manifest,
            )
            candidate_records.append(record)
            candidate_records_by_name[harshness_guard_profile.name] = record
            candidate_scores.append(
                score_candidate(
                    profile=harshness_guard_profile,
                    metrics=metrics,
                    feedback_adjustments=feedback_adjustments,
                    manifest_path=manifest_path,
                    output_dir=candidate_output_dir,
                    vocals_path=vocals_path,
                )
            )

    skipped_candidates_by_name = {
        "issue_aware_segment_selection": skipped_issue_aware_segment_selection,
        "segment_issue_repair": skipped_segment_issue_repair,
        "no_vocal_leakage_cleanup": skipped_no_vocal_leakage_cleanup,
        "vocal_harshness_guard": skipped_vocal_harshness_guard,
    }
    post_gate_repair_profile_template = default_post_gate_residual_issue_repair_profile()
    post_gate_repair_profile = VocalCandidateProfile(
        name=post_gate_repair_profile_template.name,
        candidate_mode=post_gate_repair_profile_template.candidate_mode,
        prior_score=post_gate_repair_profile_template.prior_score,
        intent=post_gate_repair_profile_template.intent,
        tuning={
            **post_gate_repair_profile_template.tuning,
            "segment_consistency_sec": round(float(args.segment_consistency_sec), 6),
        },
    )
    gate_blocker_splice_profile_template = default_gate_blocker_best_segment_splice_profile()
    gate_blocker_splice_profile = VocalCandidateProfile(
        name=gate_blocker_splice_profile_template.name,
        candidate_mode=gate_blocker_splice_profile_template.candidate_mode,
        prior_score=gate_blocker_splice_profile_template.prior_score,
        intent=gate_blocker_splice_profile_template.intent,
        tuning={
            **gate_blocker_splice_profile_template.tuning,
            "segment_consistency_sec": round(float(args.segment_consistency_sec), 6),
        },
    )
    gate_blocker_presence_band_recovery_profile_template = default_gate_blocker_presence_band_recovery_profile()
    gate_blocker_presence_band_recovery_profile = VocalCandidateProfile(
        name=gate_blocker_presence_band_recovery_profile_template.name,
        candidate_mode=gate_blocker_presence_band_recovery_profile_template.candidate_mode,
        prior_score=gate_blocker_presence_band_recovery_profile_template.prior_score,
        intent=gate_blocker_presence_band_recovery_profile_template.intent,
        tuning={
            **gate_blocker_presence_band_recovery_profile_template.tuning,
            "segment_consistency_sec": round(float(args.segment_consistency_sec), 6),
        },
    )
    clustered_issue_window_ratio_repair_profile_template = default_clustered_issue_window_ratio_repair_profile()
    clustered_issue_window_ratio_repair_profile = VocalCandidateProfile(
        name=clustered_issue_window_ratio_repair_profile_template.name,
        candidate_mode=clustered_issue_window_ratio_repair_profile_template.candidate_mode,
        prior_score=clustered_issue_window_ratio_repair_profile_template.prior_score,
        intent=clustered_issue_window_ratio_repair_profile_template.intent,
        tuning={
            **clustered_issue_window_ratio_repair_profile_template.tuning,
            "segment_consistency_sec": round(float(args.segment_consistency_sec), 6),
        },
    )
    clustered_issue_window_ratio_repair_pass2_profile = VocalCandidateProfile(
        name=f"{clustered_issue_window_ratio_repair_profile.name}_pass2",
        candidate_mode=f"{clustered_issue_window_ratio_repair_profile.candidate_mode}_pass2",
        prior_score=clustered_issue_window_ratio_repair_profile.prior_score,
        intent=f"{clustered_issue_window_ratio_repair_profile.intent}; bounded_second_pass_for_persistent_clustered_issue_blocker",
        tuning={
            **clustered_issue_window_ratio_repair_profile.tuning,
            "repair_pass": 2,
        },
    )
    skipped_post_gate_residual_issue_repair: list[dict[str, str]] = []
    skipped_gate_blocker_best_segment_splice: list[dict[str, str]] = []
    skipped_gate_blocker_presence_band_recovery: list[dict[str, str]] = []
    skipped_clustered_issue_window_ratio_repair: list[dict[str, str]] = []
    skipped_clustered_issue_window_ratio_repair_pass2: list[dict[str, str]] = []

    selected_score = select_defect_aware_candidate_score(candidate_scores, candidate_issue_reports)
    selected_issue_report = candidate_issue_reports.get(selected_score.profile_name)
    detected_issue_action_contract = build_detected_issue_action_contract(
        issue_report=selected_issue_report,
        available_candidate_names={record.profile.name for record in candidate_records},
        skipped_candidates=skipped_candidates_by_name,
    )
    defect_effectiveness_gate = build_defect_effectiveness_gate(
        issue_report=selected_issue_report,
        action_contract=detected_issue_action_contract,
        selected_metrics=selected_score.metrics,
    )
    vocal_next_action_plan = build_vocal_next_action_plan(
        defect_effectiveness_gate=defect_effectiveness_gate,
        action_contract=detected_issue_action_contract,
        external_only=external_only,
        evaluated_model_ids=collect_evaluated_live_gpu_vocal_model_ids(candidate_records),
    )

    if defect_effectiveness_gate.get("release_status") == "diagnostic_only":
        if bool(vocal_next_action_plan.get("forbid_cpu_dsp_repair_loop")):
            skipped_post_gate_residual_issue_repair.append(
                {"name": post_gate_repair_profile.name, "reason": "live_gpu_vocal_ladder_required"}
            )
        elif bounded_thin_repair_only:
            skipped_post_gate_residual_issue_repair.append(
                {"name": post_gate_repair_profile.name, "reason": "disabled_by_bounded_thin_repair_only"}
            )
        elif disable_segment_issue_repair:
            skipped_post_gate_residual_issue_repair.append(
                {"name": post_gate_repair_profile.name, "reason": "disabled_by_segment_issue_repair_flag"}
            )
        elif float(args.segment_consistency_sec) <= 0.0:
            skipped_post_gate_residual_issue_repair.append(
                {"name": post_gate_repair_profile.name, "reason": "segment_consistency_disabled"}
            )
        elif not candidate_issue_reports:
            skipped_post_gate_residual_issue_repair.append(
                {"name": post_gate_repair_profile.name, "reason": "missing_candidate_issue_reports"}
            )
        else:
            candidate_output_dir = candidate_root_dir / post_gate_repair_profile.name
            manifest_path = candidate_output_dir / "stem_manifest.json"
            expected_dependency_fingerprints = _candidate_issue_dependency_fingerprints(
                candidate_scores=candidate_scores,
                candidate_records_by_name=candidate_records_by_name,
                candidate_issue_reports=candidate_issue_reports,
            )
            expected_dependency_fingerprints["__post_gate_context"] = {
                "selected_profile": selected_score.profile_name,
                "selected_score": f"{selected_score.score:.6f}",
                "gate_blockers_sha256": hashlib.sha256(
                    json.dumps(defect_effectiveness_gate.get("blockers", []), sort_keys=True).encode("utf-8"),
                ).hexdigest(),
            }
            manifest = load_reusable_candidate_manifest(
                manifest_path=manifest_path,
                profile=post_gate_repair_profile,
                source_audio_path=source_audio_path,
                start_sec=start_sec,
                end_sec=end_sec,
                reuse_existing_candidates=bool(args.reuse_existing_candidates),
                reuse_rejected_profiles=reuse_rejected_profiles,
                expected_dependency_fingerprints=expected_dependency_fingerprints,
            )
            if manifest is None:
                candidate_output_dir.mkdir(parents=True, exist_ok=True)
                with stage_profiler.measure(
                    "derived_candidate_build",
                    candidate=post_gate_repair_profile.name,
                    builder="build_segment_issue_repair_candidate",
                    acceleration_mode="cpu_dsp_scoring",
                ):
                    manifest = build_segment_issue_repair_candidate(
                        profile=post_gate_repair_profile,
                        source_audio_path=source_audio_path,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        base_score=selected_score,
                        candidate_scores=candidate_scores,
                        candidate_records_by_name=candidate_records_by_name,
                        candidate_issue_reports=candidate_issue_reports,
                        dependency_fingerprints=expected_dependency_fingerprints,
                        output_dir=candidate_output_dir,
                        progress=lambda **metadata: stage_profiler.mark(
                            "segment_issue_repair_window",
                            candidate=post_gate_repair_profile.name,
                            builder="build_segment_issue_repair_candidate",
                            **metadata,
                        ),
                    )
                if manifest is not None:
                    rebuilt_profile_names.add(post_gate_repair_profile.name)

            if manifest is None:
                skipped_post_gate_residual_issue_repair.append(
                    {"name": post_gate_repair_profile.name, "reason": "no_post_gate_repairable_issue_window"}
                )
            else:
                vocals_path = resolve_vocals_path(manifest)
                if not vocals_path.exists():
                    raise FileNotFoundError(vocals_path)
                metrics = collect_proxy_metrics(post_gate_repair_profile.name, vocals_path)
                metrics.update(_segment_issue_repair_metrics(manifest))
                if float(args.segment_consistency_sec) > 0.0:
                    metrics.update(collect_consistency_metrics(post_gate_repair_profile.name, vocals_path))
                    issue_report = collect_segment_issue_report(
                        post_gate_repair_profile.name,
                        vocals_path,
                        anchor_vocals_path=None if anchor_record is None else anchor_record.vocals_path,
                    )
                    metrics.update(issue_report["summary_metrics"])
                    candidate_issue_reports[post_gate_repair_profile.name] = issue_report
                if anchor_record is not None:
                    metrics.update(collect_anchor_support_metrics(post_gate_repair_profile.name, vocals_path, anchor_record.vocals_path))
                record = VocalCandidateRecord(
                    profile=post_gate_repair_profile,
                    manifest_path=manifest_path,
                    output_dir=candidate_output_dir,
                    vocals_path=vocals_path,
                    metrics=metrics,
                    manifest=manifest,
                )
                candidate_records.append(record)
                candidate_records_by_name[post_gate_repair_profile.name] = record
                candidate_scores.append(
                    score_candidate(
                        profile=post_gate_repair_profile,
                        metrics=metrics,
                        feedback_adjustments=feedback_adjustments,
                        manifest_path=manifest_path,
                        output_dir=candidate_output_dir,
                        vocals_path=vocals_path,
                    )
                )
    else:
        skipped_post_gate_residual_issue_repair.append(
            {"name": post_gate_repair_profile.name, "reason": "selected_candidate_release_ready"}
        )

    skipped_candidates_by_name["post_gate_residual_issue_repair"] = skipped_post_gate_residual_issue_repair

    selected_score = select_defect_aware_candidate_score(candidate_scores, candidate_issue_reports)
    selected_issue_report = candidate_issue_reports.get(selected_score.profile_name)
    detected_issue_action_contract = build_detected_issue_action_contract(
        issue_report=selected_issue_report,
        available_candidate_names={record.profile.name for record in candidate_records},
        skipped_candidates=skipped_candidates_by_name,
    )
    defect_effectiveness_gate = build_defect_effectiveness_gate(
        issue_report=selected_issue_report,
        action_contract=detected_issue_action_contract,
        selected_metrics=selected_score.metrics,
    )
    vocal_next_action_plan = build_vocal_next_action_plan(
        defect_effectiveness_gate=defect_effectiveness_gate,
        action_contract=detected_issue_action_contract,
        external_only=external_only,
        evaluated_model_ids=collect_evaluated_live_gpu_vocal_model_ids(candidate_records),
    )
    if defect_effectiveness_gate.get("release_status") == "diagnostic_only":
        if bool(vocal_next_action_plan.get("forbid_cpu_dsp_repair_loop")):
            skipped_gate_blocker_best_segment_splice.append(
                {"name": gate_blocker_splice_profile.name, "reason": "live_gpu_vocal_ladder_required"}
            )
        elif bounded_thin_repair_only:
            skipped_gate_blocker_best_segment_splice.append(
                {"name": gate_blocker_splice_profile.name, "reason": "disabled_by_bounded_thin_repair_only"}
            )
        elif disable_segment_issue_repair:
            skipped_gate_blocker_best_segment_splice.append(
                {"name": gate_blocker_splice_profile.name, "reason": "disabled_by_segment_issue_repair_flag"}
            )
        elif float(args.segment_consistency_sec) <= 0.0:
            skipped_gate_blocker_best_segment_splice.append(
                {"name": gate_blocker_splice_profile.name, "reason": "segment_consistency_disabled"}
            )
        elif not candidate_issue_reports:
            skipped_gate_blocker_best_segment_splice.append(
                {"name": gate_blocker_splice_profile.name, "reason": "missing_candidate_issue_reports"}
            )
        elif not defect_effectiveness_gate.get("blockers"):
            skipped_gate_blocker_best_segment_splice.append(
                {"name": gate_blocker_splice_profile.name, "reason": "missing_gate_blockers"}
            )
        else:
            candidate_output_dir = candidate_root_dir / gate_blocker_splice_profile.name
            manifest_path = candidate_output_dir / "stem_manifest.json"
            expected_dependency_fingerprints = _candidate_issue_dependency_fingerprints(
                candidate_scores=candidate_scores,
                candidate_records_by_name=candidate_records_by_name,
                candidate_issue_reports=candidate_issue_reports,
            )
            expected_dependency_fingerprints["__gate_blocker_splice_context"] = {
                "selected_profile": selected_score.profile_name,
                "selected_score": f"{selected_score.score:.6f}",
                "gate_blockers_sha256": hashlib.sha256(
                    json.dumps(defect_effectiveness_gate.get("blockers", []), sort_keys=True).encode("utf-8"),
                ).hexdigest(),
            }
            manifest = load_reusable_candidate_manifest(
                manifest_path=manifest_path,
                profile=gate_blocker_splice_profile,
                source_audio_path=source_audio_path,
                start_sec=start_sec,
                end_sec=end_sec,
                reuse_existing_candidates=bool(args.reuse_existing_candidates),
                reuse_rejected_profiles=reuse_rejected_profiles,
                expected_dependency_fingerprints=expected_dependency_fingerprints,
            )
            if manifest is None:
                candidate_output_dir.mkdir(parents=True, exist_ok=True)
                with stage_profiler.measure(
                    "derived_candidate_build",
                    candidate=gate_blocker_splice_profile.name,
                    builder="build_gate_blocker_best_segment_splice_candidate",
                    acceleration_mode="cpu_dsp_scoring",
                ):
                    manifest = build_gate_blocker_best_segment_splice_candidate(
                        profile=gate_blocker_splice_profile,
                        base_score=selected_score,
                        candidate_scores=candidate_scores,
                        candidate_records_by_name=candidate_records_by_name,
                        candidate_issue_reports=candidate_issue_reports,
                        defect_effectiveness_gate=defect_effectiveness_gate,
                        dependency_fingerprints=expected_dependency_fingerprints,
                        output_dir=candidate_output_dir,
                    )
                if manifest is not None:
                    rebuilt_profile_names.add(gate_blocker_splice_profile.name)

            if manifest is None:
                skipped_gate_blocker_best_segment_splice.append(
                    {"name": gate_blocker_splice_profile.name, "reason": "no_better_measured_gate_blocker_donor"}
                )
            else:
                vocals_path = resolve_vocals_path(manifest)
                if not vocals_path.exists():
                    raise FileNotFoundError(vocals_path)
                metrics = collect_proxy_metrics(gate_blocker_splice_profile.name, vocals_path)
                metrics.update(_gate_blocker_best_segment_splice_metrics(manifest))
                if float(args.segment_consistency_sec) > 0.0:
                    metrics.update(collect_consistency_metrics(gate_blocker_splice_profile.name, vocals_path))
                    issue_report = collect_segment_issue_report(
                        gate_blocker_splice_profile.name,
                        vocals_path,
                        anchor_vocals_path=None if anchor_record is None else anchor_record.vocals_path,
                    )
                    metrics.update(issue_report["summary_metrics"])
                    candidate_issue_reports[gate_blocker_splice_profile.name] = issue_report
                if anchor_record is not None:
                    metrics.update(collect_anchor_support_metrics(gate_blocker_splice_profile.name, vocals_path, anchor_record.vocals_path))
                record = VocalCandidateRecord(
                    profile=gate_blocker_splice_profile,
                    manifest_path=manifest_path,
                    output_dir=candidate_output_dir,
                    vocals_path=vocals_path,
                    metrics=metrics,
                    manifest=manifest,
                )
                candidate_records.append(record)
                candidate_records_by_name[gate_blocker_splice_profile.name] = record
                candidate_scores.append(
                    score_candidate(
                        profile=gate_blocker_splice_profile,
                        metrics=metrics,
                        feedback_adjustments=feedback_adjustments,
                        manifest_path=manifest_path,
                        output_dir=candidate_output_dir,
                        vocals_path=vocals_path,
                    )
                )
    else:
        skipped_gate_blocker_best_segment_splice.append(
            {"name": gate_blocker_splice_profile.name, "reason": "selected_candidate_release_ready"}
        )

    skipped_candidates_by_name["gate_blocker_best_segment_splice"] = skipped_gate_blocker_best_segment_splice

    selected_score = select_defect_aware_candidate_score(candidate_scores, candidate_issue_reports)
    selected_issue_report = candidate_issue_reports.get(selected_score.profile_name)
    detected_issue_action_contract = build_detected_issue_action_contract(
        issue_report=selected_issue_report,
        available_candidate_names={record.profile.name for record in candidate_records},
        skipped_candidates=skipped_candidates_by_name,
    )
    defect_effectiveness_gate = build_defect_effectiveness_gate(
        issue_report=selected_issue_report,
        action_contract=detected_issue_action_contract,
        selected_metrics=selected_score.metrics,
    )
    vocal_next_action_plan = build_vocal_next_action_plan(
        defect_effectiveness_gate=defect_effectiveness_gate,
        action_contract=detected_issue_action_contract,
        external_only=external_only,
        evaluated_model_ids=collect_evaluated_live_gpu_vocal_model_ids(candidate_records),
    )
    if defect_effectiveness_gate.get("release_status") == "diagnostic_only":
        if bool(vocal_next_action_plan.get("forbid_cpu_dsp_repair_loop")):
            skipped_gate_blocker_presence_band_recovery.append(
                {"name": gate_blocker_presence_band_recovery_profile.name, "reason": "live_gpu_vocal_ladder_required"}
            )
        elif disable_segment_issue_repair:
            skipped_gate_blocker_presence_band_recovery.append(
                {"name": gate_blocker_presence_band_recovery_profile.name, "reason": "disabled_by_segment_issue_repair_flag"}
            )
        elif float(args.segment_consistency_sec) <= 0.0:
            skipped_gate_blocker_presence_band_recovery.append(
                {"name": gate_blocker_presence_band_recovery_profile.name, "reason": "segment_consistency_disabled"}
            )
        elif not candidate_issue_reports:
            skipped_gate_blocker_presence_band_recovery.append(
                {"name": gate_blocker_presence_band_recovery_profile.name, "reason": "missing_candidate_issue_reports"}
            )
        elif not defect_effectiveness_gate.get("blockers"):
            skipped_gate_blocker_presence_band_recovery.append(
                {"name": gate_blocker_presence_band_recovery_profile.name, "reason": "missing_gate_blockers"}
            )
        else:
            candidate_output_dir = candidate_root_dir / gate_blocker_presence_band_recovery_profile.name
            manifest_path = candidate_output_dir / "stem_manifest.json"
            expected_dependency_fingerprints = _candidate_issue_dependency_fingerprints(
                candidate_scores=candidate_scores,
                candidate_records_by_name=candidate_records_by_name,
                candidate_issue_reports=candidate_issue_reports,
            )
            expected_dependency_fingerprints["__gate_blocker_presence_band_recovery_context"] = {
                "selected_profile": selected_score.profile_name,
                "selected_score": f"{selected_score.score:.6f}",
                "gate_blockers_sha256": hashlib.sha256(
                    json.dumps(defect_effectiveness_gate.get("blockers", []), sort_keys=True).encode("utf-8"),
                ).hexdigest(),
            }
            manifest = load_reusable_candidate_manifest(
                manifest_path=manifest_path,
                profile=gate_blocker_presence_band_recovery_profile,
                source_audio_path=source_audio_path,
                start_sec=start_sec,
                end_sec=end_sec,
                reuse_existing_candidates=bool(args.reuse_existing_candidates),
                reuse_rejected_profiles=reuse_rejected_profiles,
                expected_dependency_fingerprints=expected_dependency_fingerprints,
            )
            if manifest is None:
                candidate_output_dir.mkdir(parents=True, exist_ok=True)
                with stage_profiler.measure(
                    "derived_candidate_build",
                    candidate=gate_blocker_presence_band_recovery_profile.name,
                    builder="build_gate_blocker_presence_band_recovery_candidate",
                    acceleration_mode="cpu_dsp_scoring",
                ):
                    manifest = build_gate_blocker_presence_band_recovery_candidate(
                        profile=gate_blocker_presence_band_recovery_profile,
                        base_score=selected_score,
                        candidate_scores=candidate_scores,
                        candidate_records_by_name=candidate_records_by_name,
                        candidate_issue_reports=candidate_issue_reports,
                        defect_effectiveness_gate=defect_effectiveness_gate,
                        dependency_fingerprints=expected_dependency_fingerprints,
                        output_dir=candidate_output_dir,
                    )
                if manifest is not None:
                    rebuilt_profile_names.add(gate_blocker_presence_band_recovery_profile.name)

            if manifest is None:
                skipped_gate_blocker_presence_band_recovery.append(
                    {"name": gate_blocker_presence_band_recovery_profile.name, "reason": "no_better_measured_presence_band_recovery"}
                )
            else:
                vocals_path = resolve_vocals_path(manifest)
                if not vocals_path.exists():
                    raise FileNotFoundError(vocals_path)
                metrics = collect_proxy_metrics(gate_blocker_presence_band_recovery_profile.name, vocals_path)
                metrics.update(_gate_blocker_presence_band_recovery_metrics(manifest))
                if float(args.segment_consistency_sec) > 0.0:
                    metrics.update(collect_consistency_metrics(gate_blocker_presence_band_recovery_profile.name, vocals_path))
                    issue_report = collect_segment_issue_report(
                        gate_blocker_presence_band_recovery_profile.name,
                        vocals_path,
                        anchor_vocals_path=None if anchor_record is None else anchor_record.vocals_path,
                    )
                    metrics.update(issue_report["summary_metrics"])
                    candidate_issue_reports[gate_blocker_presence_band_recovery_profile.name] = issue_report
                if anchor_record is not None:
                    metrics.update(
                        collect_anchor_support_metrics(
                            gate_blocker_presence_band_recovery_profile.name,
                            vocals_path,
                            anchor_record.vocals_path,
                        )
                    )
                record = VocalCandidateRecord(
                    profile=gate_blocker_presence_band_recovery_profile,
                    manifest_path=manifest_path,
                    output_dir=candidate_output_dir,
                    vocals_path=vocals_path,
                    metrics=metrics,
                    manifest=manifest,
                )
                candidate_records.append(record)
                candidate_records_by_name[gate_blocker_presence_band_recovery_profile.name] = record
                candidate_scores.append(
                    score_candidate(
                        profile=gate_blocker_presence_band_recovery_profile,
                        metrics=metrics,
                        feedback_adjustments=feedback_adjustments,
                        manifest_path=manifest_path,
                        output_dir=candidate_output_dir,
                        vocals_path=vocals_path,
                    )
                )
    else:
        skipped_gate_blocker_presence_band_recovery.append(
            {"name": gate_blocker_presence_band_recovery_profile.name, "reason": "selected_candidate_release_ready"}
        )

    skipped_candidates_by_name[
        "gate_blocker_presence_band_recovery"
    ] = skipped_gate_blocker_presence_band_recovery

    selected_score = select_defect_aware_candidate_score(candidate_scores, candidate_issue_reports)
    selected_issue_report = candidate_issue_reports.get(selected_score.profile_name)
    detected_issue_action_contract = build_detected_issue_action_contract(
        issue_report=selected_issue_report,
        available_candidate_names={record.profile.name for record in candidate_records},
        skipped_candidates=skipped_candidates_by_name,
    )
    defect_effectiveness_gate = build_defect_effectiveness_gate(
        issue_report=selected_issue_report,
        action_contract=detected_issue_action_contract,
        selected_metrics=selected_score.metrics,
    )
    vocal_next_action_plan = build_vocal_next_action_plan(
        defect_effectiveness_gate=defect_effectiveness_gate,
        action_contract=detected_issue_action_contract,
        external_only=external_only,
        evaluated_model_ids=collect_evaluated_live_gpu_vocal_model_ids(candidate_records),
    )
    if defect_effectiveness_gate.get("release_status") == "diagnostic_only":
        if bool(vocal_next_action_plan.get("forbid_cpu_dsp_repair_loop")):
            skipped_clustered_issue_window_ratio_repair.append(
                {"name": clustered_issue_window_ratio_repair_profile.name, "reason": "live_gpu_vocal_ladder_required"}
            )
        elif bounded_thin_repair_only:
            skipped_clustered_issue_window_ratio_repair.append(
                {"name": clustered_issue_window_ratio_repair_profile.name, "reason": "disabled_by_bounded_thin_repair_only"}
            )
        elif disable_segment_issue_repair:
            skipped_clustered_issue_window_ratio_repair.append(
                {"name": clustered_issue_window_ratio_repair_profile.name, "reason": "disabled_by_segment_issue_repair_flag"}
            )
        elif float(args.segment_consistency_sec) <= 0.0:
            skipped_clustered_issue_window_ratio_repair.append(
                {"name": clustered_issue_window_ratio_repair_profile.name, "reason": "segment_consistency_disabled"}
            )
        elif not candidate_issue_reports:
            skipped_clustered_issue_window_ratio_repair.append(
                {"name": clustered_issue_window_ratio_repair_profile.name, "reason": "missing_candidate_issue_reports"}
            )
        elif not _has_clustered_issue_ratio_blocker(defect_effectiveness_gate):
            skipped_clustered_issue_window_ratio_repair.append(
                {"name": clustered_issue_window_ratio_repair_profile.name, "reason": "missing_clustered_issue_ratio_blocker"}
            )
        else:
            candidate_output_dir = candidate_root_dir / clustered_issue_window_ratio_repair_profile.name
            manifest_path = candidate_output_dir / "stem_manifest.json"
            expected_dependency_fingerprints = _candidate_issue_dependency_fingerprints(
                candidate_scores=candidate_scores,
                candidate_records_by_name=candidate_records_by_name,
                candidate_issue_reports=candidate_issue_reports,
            )
            expected_dependency_fingerprints["__clustered_issue_window_ratio_repair_context"] = {
                "selected_profile": selected_score.profile_name,
                "selected_score": f"{selected_score.score:.6f}",
                "gate_blockers_sha256": hashlib.sha256(
                    json.dumps(defect_effectiveness_gate.get("blockers", []), sort_keys=True).encode("utf-8"),
                ).hexdigest(),
            }
            manifest = load_reusable_candidate_manifest(
                manifest_path=manifest_path,
                profile=clustered_issue_window_ratio_repair_profile,
                source_audio_path=source_audio_path,
                start_sec=start_sec,
                end_sec=end_sec,
                reuse_existing_candidates=bool(args.reuse_existing_candidates),
                reuse_rejected_profiles=reuse_rejected_profiles,
                expected_dependency_fingerprints=expected_dependency_fingerprints,
            )
            if manifest is None:
                candidate_output_dir.mkdir(parents=True, exist_ok=True)
                with stage_profiler.measure(
                    "derived_candidate_build",
                    candidate=clustered_issue_window_ratio_repair_profile.name,
                    builder="build_clustered_issue_window_ratio_repair_candidate",
                    acceleration_mode="cpu_dsp_scoring",
                ):
                    manifest = build_clustered_issue_window_ratio_repair_candidate(
                        profile=clustered_issue_window_ratio_repair_profile,
                        base_score=selected_score,
                        candidate_records_by_name=candidate_records_by_name,
                        candidate_issue_reports=candidate_issue_reports,
                        defect_effectiveness_gate=defect_effectiveness_gate,
                        dependency_fingerprints=expected_dependency_fingerprints,
                        output_dir=candidate_output_dir,
                    )
                if manifest is not None:
                    rebuilt_profile_names.add(clustered_issue_window_ratio_repair_profile.name)

            if manifest is None:
                skipped_clustered_issue_window_ratio_repair.append(
                    {"name": clustered_issue_window_ratio_repair_profile.name, "reason": "no_better_measured_clustered_issue_window_repair"}
                )
            else:
                vocals_path = resolve_vocals_path(manifest)
                if not vocals_path.exists():
                    raise FileNotFoundError(vocals_path)
                metrics = collect_proxy_metrics(clustered_issue_window_ratio_repair_profile.name, vocals_path)
                metrics.update(_clustered_issue_window_ratio_repair_metrics(manifest))
                if float(args.segment_consistency_sec) > 0.0:
                    metrics.update(collect_consistency_metrics(clustered_issue_window_ratio_repair_profile.name, vocals_path))
                    issue_report = collect_segment_issue_report(
                        clustered_issue_window_ratio_repair_profile.name,
                        vocals_path,
                        anchor_vocals_path=None if anchor_record is None else anchor_record.vocals_path,
                    )
                    metrics.update(issue_report["summary_metrics"])
                    candidate_issue_reports[clustered_issue_window_ratio_repair_profile.name] = issue_report
                if anchor_record is not None:
                    metrics.update(
                        collect_anchor_support_metrics(
                            clustered_issue_window_ratio_repair_profile.name,
                            vocals_path,
                            anchor_record.vocals_path,
                        )
                    )
                record = VocalCandidateRecord(
                    profile=clustered_issue_window_ratio_repair_profile,
                    manifest_path=manifest_path,
                    output_dir=candidate_output_dir,
                    vocals_path=vocals_path,
                    metrics=metrics,
                    manifest=manifest,
                )
                candidate_records.append(record)
                candidate_records_by_name[clustered_issue_window_ratio_repair_profile.name] = record
                candidate_scores.append(
                    score_candidate(
                        profile=clustered_issue_window_ratio_repair_profile,
                        metrics=metrics,
                        feedback_adjustments=feedback_adjustments,
                        manifest_path=manifest_path,
                        output_dir=candidate_output_dir,
                        vocals_path=vocals_path,
                    )
                )
    else:
        skipped_clustered_issue_window_ratio_repair.append(
            {"name": clustered_issue_window_ratio_repair_profile.name, "reason": "selected_candidate_release_ready"}
        )

    skipped_candidates_by_name[
        "clustered_issue_window_ratio_repair"
    ] = skipped_clustered_issue_window_ratio_repair

    selected_score = select_defect_aware_candidate_score(candidate_scores, candidate_issue_reports)
    selected_issue_report = candidate_issue_reports.get(selected_score.profile_name)
    detected_issue_action_contract = build_detected_issue_action_contract(
        issue_report=selected_issue_report,
        available_candidate_names={record.profile.name for record in candidate_records},
        skipped_candidates=skipped_candidates_by_name,
    )
    defect_effectiveness_gate = build_defect_effectiveness_gate(
        issue_report=selected_issue_report,
        action_contract=detected_issue_action_contract,
        selected_metrics=selected_score.metrics,
    )
    vocal_next_action_plan = build_vocal_next_action_plan(
        defect_effectiveness_gate=defect_effectiveness_gate,
        action_contract=detected_issue_action_contract,
        external_only=external_only,
        evaluated_model_ids=collect_evaluated_live_gpu_vocal_model_ids(candidate_records),
    )
    if defect_effectiveness_gate.get("release_status") == "diagnostic_only":
        if bool(vocal_next_action_plan.get("forbid_cpu_dsp_repair_loop")):
            skipped_clustered_issue_window_ratio_repair_pass2.append(
                {
                    "name": clustered_issue_window_ratio_repair_pass2_profile.name,
                    "reason": "live_gpu_vocal_ladder_required",
                }
            )
        elif bounded_thin_repair_only:
            skipped_clustered_issue_window_ratio_repair_pass2.append(
                {
                    "name": clustered_issue_window_ratio_repair_pass2_profile.name,
                    "reason": "disabled_by_bounded_thin_repair_only",
                }
            )
        elif disable_segment_issue_repair:
            skipped_clustered_issue_window_ratio_repair_pass2.append(
                {
                    "name": clustered_issue_window_ratio_repair_pass2_profile.name,
                    "reason": "disabled_by_segment_issue_repair_flag",
                }
            )
        elif float(args.segment_consistency_sec) <= 0.0:
            skipped_clustered_issue_window_ratio_repair_pass2.append(
                {"name": clustered_issue_window_ratio_repair_pass2_profile.name, "reason": "segment_consistency_disabled"}
            )
        elif not candidate_issue_reports:
            skipped_clustered_issue_window_ratio_repair_pass2.append(
                {
                    "name": clustered_issue_window_ratio_repair_pass2_profile.name,
                    "reason": "missing_candidate_issue_reports",
                }
            )
        elif clustered_issue_window_ratio_repair_profile.name not in candidate_records_by_name:
            skipped_clustered_issue_window_ratio_repair_pass2.append(
                {"name": clustered_issue_window_ratio_repair_pass2_profile.name, "reason": "missing_first_pass_candidate"}
            )
        elif not _has_clustered_issue_ratio_blocker(defect_effectiveness_gate):
            skipped_clustered_issue_window_ratio_repair_pass2.append(
                {
                    "name": clustered_issue_window_ratio_repair_pass2_profile.name,
                    "reason": "missing_clustered_issue_ratio_blocker",
                }
            )
        else:
            candidate_output_dir = candidate_root_dir / clustered_issue_window_ratio_repair_pass2_profile.name
            manifest_path = candidate_output_dir / "stem_manifest.json"
            expected_dependency_fingerprints = _candidate_issue_dependency_fingerprints(
                candidate_scores=candidate_scores,
                candidate_records_by_name=candidate_records_by_name,
                candidate_issue_reports=candidate_issue_reports,
            )
            expected_dependency_fingerprints["__clustered_issue_window_ratio_repair_pass2_context"] = {
                "selected_profile": selected_score.profile_name,
                "selected_score": f"{selected_score.score:.6f}",
                "gate_blockers_sha256": hashlib.sha256(
                    json.dumps(defect_effectiveness_gate.get("blockers", []), sort_keys=True).encode("utf-8"),
                ).hexdigest(),
            }
            manifest = load_reusable_candidate_manifest(
                manifest_path=manifest_path,
                profile=clustered_issue_window_ratio_repair_pass2_profile,
                source_audio_path=source_audio_path,
                start_sec=start_sec,
                end_sec=end_sec,
                reuse_existing_candidates=bool(args.reuse_existing_candidates),
                reuse_rejected_profiles=reuse_rejected_profiles,
                expected_dependency_fingerprints=expected_dependency_fingerprints,
            )
            if manifest is None:
                candidate_output_dir.mkdir(parents=True, exist_ok=True)
                with stage_profiler.measure(
                    "derived_candidate_build",
                    candidate=clustered_issue_window_ratio_repair_pass2_profile.name,
                    builder="build_clustered_issue_window_ratio_repair_candidate",
                    acceleration_mode="cpu_dsp_scoring",
                ):
                    manifest = build_clustered_issue_window_ratio_repair_candidate(
                        profile=clustered_issue_window_ratio_repair_pass2_profile,
                        base_score=selected_score,
                        candidate_records_by_name=candidate_records_by_name,
                        candidate_issue_reports=candidate_issue_reports,
                        defect_effectiveness_gate=defect_effectiveness_gate,
                        dependency_fingerprints=expected_dependency_fingerprints,
                        output_dir=candidate_output_dir,
                    )
                if manifest is not None:
                    rebuilt_profile_names.add(clustered_issue_window_ratio_repair_pass2_profile.name)

            if manifest is None:
                skipped_clustered_issue_window_ratio_repair_pass2.append(
                    {
                        "name": clustered_issue_window_ratio_repair_pass2_profile.name,
                        "reason": "no_better_measured_clustered_issue_window_repair",
                    }
                )
            else:
                vocals_path = resolve_vocals_path(manifest)
                if not vocals_path.exists():
                    raise FileNotFoundError(vocals_path)
                metrics = collect_proxy_metrics(clustered_issue_window_ratio_repair_pass2_profile.name, vocals_path)
                metrics.update(_clustered_issue_window_ratio_repair_metrics(manifest))
                if float(args.segment_consistency_sec) > 0.0:
                    metrics.update(collect_consistency_metrics(clustered_issue_window_ratio_repair_pass2_profile.name, vocals_path))
                    issue_report = collect_segment_issue_report(
                        clustered_issue_window_ratio_repair_pass2_profile.name,
                        vocals_path,
                        anchor_vocals_path=None if anchor_record is None else anchor_record.vocals_path,
                    )
                    metrics.update(issue_report["summary_metrics"])
                    candidate_issue_reports[clustered_issue_window_ratio_repair_pass2_profile.name] = issue_report
                if anchor_record is not None:
                    metrics.update(
                        collect_anchor_support_metrics(
                            clustered_issue_window_ratio_repair_pass2_profile.name,
                            vocals_path,
                            anchor_record.vocals_path,
                        )
                    )
                record = VocalCandidateRecord(
                    profile=clustered_issue_window_ratio_repair_pass2_profile,
                    manifest_path=manifest_path,
                    output_dir=candidate_output_dir,
                    vocals_path=vocals_path,
                    metrics=metrics,
                    manifest=manifest,
                )
                candidate_records.append(record)
                candidate_records_by_name[clustered_issue_window_ratio_repair_pass2_profile.name] = record
                candidate_scores.append(
                    score_candidate(
                        profile=clustered_issue_window_ratio_repair_pass2_profile,
                        metrics=metrics,
                        feedback_adjustments=feedback_adjustments,
                        manifest_path=manifest_path,
                        output_dir=candidate_output_dir,
                        vocals_path=vocals_path,
                    )
                )
    else:
        skipped_clustered_issue_window_ratio_repair_pass2.append(
            {"name": clustered_issue_window_ratio_repair_pass2_profile.name, "reason": "selected_candidate_release_ready"}
        )

    skipped_candidates_by_name[
        "clustered_issue_window_ratio_repair_pass2"
    ] = skipped_clustered_issue_window_ratio_repair_pass2

    selected_score = select_defect_aware_candidate_score(candidate_scores, candidate_issue_reports)
    selected_vocals_path = Path(selected_score.vocals_path)
    selected_output_path = output_dir / "01_adaptive_selected_vocals.flac"
    shutil.copyfile(selected_vocals_path, selected_output_path)
    selected_issue_report = candidate_issue_reports.get(selected_score.profile_name)
    detected_issue_action_contract = build_detected_issue_action_contract(
        issue_report=selected_issue_report,
        available_candidate_names={record.profile.name for record in candidate_records},
        skipped_candidates=skipped_candidates_by_name,
    )
    defect_effectiveness_gate = build_defect_effectiveness_gate(
        issue_report=selected_issue_report,
        action_contract=detected_issue_action_contract,
        selected_metrics=selected_score.metrics,
    )
    issue_effectiveness_ledger = build_issue_effectiveness_ledger(
        issue_report=selected_issue_report,
        action_contract=detected_issue_action_contract,
        defect_effectiveness_gate=defect_effectiveness_gate,
        candidate_issue_reports=candidate_issue_reports,
    )

    listen_artifacts: dict[str, str] | None = None
    if listen_dir is not None:
        listen_artifacts = {}
        if defect_effectiveness_gate.get("status") == "release_ready":
            with stage_profiler.measure("listen_pack"):
                listen_artifacts.update(
                    write_listen_pack(
                        source_audio_path=source_audio_path,
                        selected_vocals_path=selected_vocals_path,
                        candidate_scores=candidate_scores,
                        listen_dir=listen_dir,
                        selected_label="01_adaptive_selected_vocals.flac",
                    )
                )
        else:
            listen_artifacts["listen_pack_skipped_reason"] = "blocked_for_listening"
        if selected_issue_report is not None and int(args.segment_issue_listen_clip_count) > 0:
            with stage_profiler.measure("segment_issue_listen_clips"):
                listen_artifacts.update(
                    write_segment_issue_listen_clips(
                        source_audio_path=source_audio_path,
                        selected_vocals_path=selected_vocals_path,
                        issue_report=selected_issue_report,
                        listen_dir=listen_dir,
                        max_clips=int(args.segment_issue_listen_clip_count),
                    )
                )

    delivery_allowed = defect_effectiveness_gate.get("status") == "release_ready"
    presentation_status = "accepted_for_listen_pack" if delivery_allowed else "blocked_from_listen_pack"
    listen_selected_vocals = None if listen_artifacts is None else listen_artifacts.get("selected_vocals_flac")
    selected_delivery_gate_manifest_path = output_dir / "selected_delivery_gate_manifest.json"
    selected_artifacts = {
        "selected_vocals_flac": str(selected_output_path),
        "listen_selected_vocals_flac": listen_selected_vocals if delivery_allowed else None,
        "listen_segment_issue_clips_manifest": None if listen_artifacts is None else listen_artifacts.get("segment_issue_clips_manifest"),
        "release_status": defect_effectiveness_gate["release_status"],
        "delivery_allowed": bool(delivery_allowed),
        "presentation_status": presentation_status,
        "deliverable_vocals_flac": str(selected_output_path) if delivery_allowed else None,
        "deliverable_listen_vocals_flac": listen_selected_vocals if delivery_allowed else None,
        "diagnostic_vocals_flac": None if delivery_allowed else str(selected_output_path),
        "diagnostic_listen_vocals_flac": None,
        "listen_pack_skipped_reason": None if listen_artifacts is None else listen_artifacts.get("listen_pack_skipped_reason"),
        "selected_delivery_gate_manifest": str(selected_delivery_gate_manifest_path),
    }
    adaptive_decision_contract = build_adaptive_decision_contract(
        detected_issue_action_contract=detected_issue_action_contract,
        defect_effectiveness_gate=defect_effectiveness_gate,
        vocal_next_action_plan=vocal_next_action_plan,
        selected_artifacts=selected_artifacts,
    )
    selected_delivery_gate_manifest = {
        "schema_version": 1,
        "mode": "adaptive_vocal_selected_delivery_gate",
        "source_audio_path": str(source_audio_path),
        "source_audio_sha256": source_audio_sha256,
        "output_dir": str(output_dir),
        "adaptive_manifest_path": str(output_dir / "adaptive_vocal_manifest.json"),
        "selected_profile": selected_score.profile_name,
        "selected_score": selected_score.score,
        "selected_manifest_path": selected_score.manifest_path,
        "selected_vocals_flac": str(selected_output_path),
        "release_status": defect_effectiveness_gate["release_status"],
        "delivery_allowed": bool(delivery_allowed),
        "presentation_status": presentation_status,
        "deliverable_vocals_flac": selected_artifacts["deliverable_vocals_flac"],
        "deliverable_listen_vocals_flac": selected_artifacts["deliverable_listen_vocals_flac"],
        "diagnostic_vocals_flac": selected_artifacts["diagnostic_vocals_flac"],
        "diagnostic_listen_vocals_flac": selected_artifacts["diagnostic_listen_vocals_flac"],
        "listen_pack_skipped_reason": selected_artifacts["listen_pack_skipped_reason"],
        "defect_effectiveness_gate": defect_effectiveness_gate,
        "vocal_next_action_plan": vocal_next_action_plan,
        "decision_contract": adaptive_decision_contract,
    }
    stage_profiler.mark(
        "selected_delivery_gate_manifest",
        path=str(selected_delivery_gate_manifest_path),
        delivery_allowed=bool(delivery_allowed),
        presentation_status=presentation_status,
    )

    adaptive_manifest = {
        "schema_version": 1,
        "mode": "adaptive_vocal_isolation_controller",
        "source_audio_path": str(source_audio_path),
        "source_audio_sha256": source_audio_sha256,
        "output_dir": str(output_dir),
        "listen_dir": str(listen_dir) if listen_dir else None,
        "export_script": str(export_script),
        "window_sec": {"start": start_sec, "end": end_sec},
        "feedback_store_path": str(feedback_store_path) if feedback_store_path else None,
        "feedback_adjustments": feedback_adjustments,
        "external_only": external_only,
        "bounded_thin_repair_only": bounded_thin_repair_only,
        "bounded_thin_repair_contract": {
            "enabled": bounded_thin_repair_only,
            "allowed_actions": ["direct_body_repair", "gate_blocker_presence_band_recovery"],
            "forbidden_actions": [
                "stronger_debleed",
                "stronger_denoise",
                "hard_vad_gate",
                "switch_backbone",
                "gate_blocker_best_segment_splice",
                "post_gate_residual_issue_repair",
                "clustered_issue_window_ratio_repair",
            ],
            "anchor_suppression_forced_disabled": bool(bounded_thin_repair_only),
        },
        "execution_contract": execution_contract,
        "candidate_profiles": [
            {
                "name": profile.name,
                "candidate_mode": profile.candidate_mode,
                "intent": profile.intent,
                "prior_score": profile.prior_score,
                "tuning": profile.tuning,
            }
            for profile in profiles
        ],
        "external_candidate_profiles": [
            {
                "name": profile.name,
                "candidate_mode": profile.candidate_mode,
                "intent": profile.intent,
                "prior_score": profile.prior_score,
                "tuning": profile.tuning,
            }
            for profile in external_profiles
        ],
        "derived_candidate_profiles": [
            {
                "name": profile.name,
                "candidate_mode": "anchor_suppressed_vocals",
                "source_profile": profile.source_profile_name,
                "anchor_profile": profile.anchor_profile_name,
                "intent": profile.intent,
                "prior_score": profile.prior_score,
                "tuning": profile.tuning,
            }
            for profile in derived_profiles
        ],
        "skipped_derived_profiles": skipped_derived_profiles,
        "body_repair_candidate_profiles": [
            {
                "name": profile.name,
                "candidate_mode": "vocal_body_repaired_vocals",
                "base_profile": profile.base_profile_name,
                "donor_profiles": list(profile.donor_profile_names),
                "intent": profile.intent,
                "prior_score": profile.prior_score,
                "tuning": profile.tuning,
            }
            for profile in body_repair_profiles
        ],
        "skipped_body_repair_profiles": skipped_body_repair_profiles,
        "segment_router_candidate_profiles": [
            {
                "name": profile.name,
                "candidate_mode": "segment_adaptive_routed_vocals",
                "clean_profile": profile.clean_profile_name,
                "strict_profile": profile.strict_profile_name,
                "repair_profile": profile.repair_profile_name,
                "preserve_profile": profile.preserve_profile_name,
                "anchor_profile": profile.anchor_profile_name,
                "intent": profile.intent,
                "prior_score": profile.prior_score,
                "tuning": profile.tuning,
            }
            for profile in segment_router_profiles
        ],
        "skipped_segment_router_profiles": skipped_segment_router_profiles,
        "issue_aware_segment_selection_profile": {
            "name": issue_aware_profile.name,
            "candidate_mode": issue_aware_profile.candidate_mode,
            "intent": issue_aware_profile.intent,
            "prior_score": issue_aware_profile.prior_score,
            "tuning": issue_aware_profile.tuning,
        },
        "skipped_issue_aware_segment_selection": skipped_issue_aware_segment_selection,
        "segment_issue_repair_profile": {
            "name": segment_issue_repair_profile.name,
            "candidate_mode": segment_issue_repair_profile.candidate_mode,
            "intent": segment_issue_repair_profile.intent,
            "prior_score": segment_issue_repair_profile.prior_score,
            "tuning": segment_issue_repair_profile.tuning,
        },
        "skipped_segment_issue_repair": skipped_segment_issue_repair,
        "post_gate_residual_issue_repair_profile": {
            "name": post_gate_repair_profile.name,
            "candidate_mode": post_gate_repair_profile.candidate_mode,
            "intent": post_gate_repair_profile.intent,
            "prior_score": post_gate_repair_profile.prior_score,
            "tuning": post_gate_repair_profile.tuning,
        },
        "skipped_post_gate_residual_issue_repair": skipped_post_gate_residual_issue_repair,
        "gate_blocker_best_segment_splice_profile": {
            "name": gate_blocker_splice_profile.name,
            "candidate_mode": gate_blocker_splice_profile.candidate_mode,
            "intent": gate_blocker_splice_profile.intent,
            "prior_score": gate_blocker_splice_profile.prior_score,
            "tuning": gate_blocker_splice_profile.tuning,
        },
        "skipped_gate_blocker_best_segment_splice": skipped_gate_blocker_best_segment_splice,
        "gate_blocker_presence_band_recovery_profile": {
            "name": gate_blocker_presence_band_recovery_profile.name,
            "candidate_mode": gate_blocker_presence_band_recovery_profile.candidate_mode,
            "intent": gate_blocker_presence_band_recovery_profile.intent,
            "prior_score": gate_blocker_presence_band_recovery_profile.prior_score,
            "tuning": gate_blocker_presence_band_recovery_profile.tuning,
        },
        "skipped_gate_blocker_presence_band_recovery": skipped_gate_blocker_presence_band_recovery,
        "clustered_issue_window_ratio_repair_profile": {
            "name": clustered_issue_window_ratio_repair_profile.name,
            "candidate_mode": clustered_issue_window_ratio_repair_profile.candidate_mode,
            "intent": clustered_issue_window_ratio_repair_profile.intent,
            "prior_score": clustered_issue_window_ratio_repair_profile.prior_score,
            "tuning": clustered_issue_window_ratio_repair_profile.tuning,
        },
        "skipped_clustered_issue_window_ratio_repair": skipped_clustered_issue_window_ratio_repair,
        "clustered_issue_window_ratio_repair_pass2_profile": {
            "name": clustered_issue_window_ratio_repair_pass2_profile.name,
            "candidate_mode": clustered_issue_window_ratio_repair_pass2_profile.candidate_mode,
            "intent": clustered_issue_window_ratio_repair_pass2_profile.intent,
            "prior_score": clustered_issue_window_ratio_repair_pass2_profile.prior_score,
            "tuning": clustered_issue_window_ratio_repair_pass2_profile.tuning,
        },
        "skipped_clustered_issue_window_ratio_repair_pass2": skipped_clustered_issue_window_ratio_repair_pass2,
        "no_vocal_leakage_cleanup_profile": {
            "name": no_vocal_cleanup_profile.name,
            "candidate_mode": no_vocal_cleanup_profile.candidate_mode,
            "intent": no_vocal_cleanup_profile.intent,
            "prior_score": no_vocal_cleanup_profile.prior_score,
            "tuning": no_vocal_cleanup_profile.tuning,
        },
        "skipped_no_vocal_leakage_cleanup": skipped_no_vocal_leakage_cleanup,
        "vocal_harshness_guard_profile": {
            "name": harshness_guard_profile.name,
            "candidate_mode": harshness_guard_profile.candidate_mode,
            "intent": harshness_guard_profile.intent,
            "prior_score": harshness_guard_profile.prior_score,
            "tuning": harshness_guard_profile.tuning,
        },
        "skipped_vocal_harshness_guard": skipped_vocal_harshness_guard,
        "reuse_rejected_profiles": reuse_rejected_profiles,
        "candidates": [
            {
                "profile_name": candidate.profile_name,
                "score": candidate.score,
                "reasons": candidate.reasons,
                "metrics": candidate.metrics,
                "segment_issue_report": candidate_issue_reports.get(candidate.profile_name),
                "manifest_path": candidate.manifest_path,
                "output_dir": candidate.output_dir,
                "vocals_path": candidate.vocals_path,
            }
            for candidate in sorted(candidate_scores, key=lambda candidate: candidate.score, reverse=True)
        ],
        "selected_profile": selected_score.profile_name,
        "selected_score": selected_score.score,
        "selected_manifest_path": selected_score.manifest_path,
        "candidate_selection_policy": {
            "mode": "defect_aware_issue_summary_then_proxy",
            "primary_order": [
                "max_issue_score_ascending",
                "mean_top3_issue_score_ascending",
                "issue_window_ratio_ascending",
            ],
            "proxy_score_role": "tie_breaker_descending",
            "fallback_without_issue_reports": "proxy_score_descending",
            "delivery_gate_unchanged": True,
        },
        "selected_segment_issue_report": selected_issue_report,
        "detected_issue_action_contract": detected_issue_action_contract,
        "decision_contract": adaptive_decision_contract,
        "defect_effectiveness_gate": defect_effectiveness_gate,
        "vocal_next_action_plan": vocal_next_action_plan,
        "issue_effectiveness_ledger": issue_effectiveness_ledger,
        "stage_profiler": stage_profiler.to_manifest(),
        "analysis_cache": analysis_cache.stats(),
        "selected_artifacts": selected_artifacts,
        "learning_contract": {
            "type": "feedback_weighted_vocal_profile_selection",
            "feedback_record_schema": {
                "profile": "selected profile name",
                "rating": "accept|reject|good|bad|better|worse",
                "score": "optional numeric value from -1.0 to 1.0",
                "listener_verified": "optional boolean; requires source_audio_sha256 and window_sec",
                "human_verified": "optional boolean; requires source_audio_sha256 and window_sec",
                "confidence": "optional numeric value from 0.0 to 1.0 for verified listener feedback",
                "source_audio_sha256": "optional source file SHA-256; required for verified listener feedback",
                "window_sec": "optional object with start and end seconds; required for verified listener feedback",
                "feedback_split": "optional train|holdout. holdout records are reserved for validation and do not adjust routing priors",
                "swallowed_endings": "optional boolean",
                "backing_bleed": "optional boolean",
                "backing_vocal_loss": "optional boolean",
                "watery_artifacts": "optional boolean",
                "non_vocal_bleed": "optional boolean",
                "uplifter_bleed": "optional boolean",
                "synth_bleed": "optional boolean",
                "robotic_drop_bleed": "optional boolean",
            },
            "limits": [
                "This controller adapts profile selection from persisted feedback.",
                "It can build a segment-adaptive routed candidate from already generated vocal candidates.",
                "It can build an issue-aware segment selection candidate from diagnosed windows and existing candidates.",
                "It can build a segment issue repair candidate from unresolved detector labels without fixed timestamps.",
                "It can build a no-vocal leakage cleanup candidate from consensus no-vocal windows without fixed timestamps.",
                "It can include IR residual and external vocal candidates as proxy inputs, not ground truth.",
                "It rejects stale reused candidate manifests when source path, window, mode, or tuning no longer match.",
                "It does not train or fine-tune neural model weights.",
                "It uses proxy metrics when isolated ground-truth vocals are not available.",
                "Holdout feedback records are never used to adjust training priors.",
            ],
        },
    }
    write_json(selected_delivery_gate_manifest_path, selected_delivery_gate_manifest)
    write_json(output_dir / "adaptive_vocal_manifest.json", adaptive_manifest)
    return adaptive_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adaptive candidate search for isolated vocals with tail-preservation scoring.",
    )
    parser.add_argument("source_audio_path", help="Full song audio path.")
    parser.add_argument("output_dir", help="Directory for adaptive vocal artifacts.")
    parser.add_argument("--listen-dir", default=None, help="Optional listen pack directory.")
    parser.add_argument(
        "--profile",
        action="append",
        choices=[profile.name for profile in default_candidate_profiles()],
        help="Candidate profile to run. Repeat to limit the search space.",
    )
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument(
        "--segment-consistency-sec",
        type=float,
        default=5.0,
        help=(
            "Window length for per-segment proxy consistency scoring. "
            "Use 0 to disable this scoring layer."
        ),
    )
    parser.add_argument(
        "--segment-issue-report-max-windows",
        type=int,
        default=8,
        help="Maximum ranked problematic windows saved per candidate in adaptive_vocal_manifest.json.",
    )
    parser.add_argument(
        "--segment-issue-listen-clip-count",
        type=int,
        default=3,
        help="Maximum selected-candidate issue windows exported as raw/vocal clips into listen_dir/segment_issues.",
    )
    parser.add_argument(
        "--disable-stage-heartbeat",
        action="store_true",
        help="Disable JSON heartbeat events on stderr while keeping stage_profiler in the manifest.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--feedback-store",
        default=None,
        help="Optional JSONL feedback store used to adjust future profile selection.",
    )
    parser.add_argument(
        "--export-script",
        default="/app/tools/export_hq_vocal_preview.py",
        help="Path to export_hq_vocal_preview.py inside the runtime.",
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--reuse-existing-candidates",
        action="store_true",
        help="Reuse candidate manifests already present under output_dir/candidates.",
    )
    parser.add_argument(
        "--ir-instrumental-minus-path",
        default=None,
        help=(
            "Optional Instrumental Resurrection minus path. When set, the controller builds "
            "ir_residual_vocals = source - instrumental_minus as a proxy vocal candidate."
        ),
    )
    parser.add_argument(
        "--external-vocal-candidate",
        action="append",
        type=parse_external_vocal_candidate,
        default=[],
        metavar="NAME=PATH",
        help=(
            "Optional precomputed vocal candidate such as MDX/MelBand/model-zoo output. "
            "Repeat as needed; NAME must be unique."
        ),
    )
    parser.add_argument(
        "--external-only",
        action="store_true",
        help=(
            "Evaluate only precomputed external vocal candidates and skip built-in candidate inference. "
            "Requires --external-vocal-candidate or --ir-instrumental-minus-path."
        ),
    )
    parser.add_argument(
        "--bounded-thin-repair-only",
        action="store_true",
        help=(
            "Run the thin/over-cleaned bounded repair contract: direct body repair plus gate-blocker "
            "presence recovery only, while disabling stronger cleanup/debleed-style derived routes."
        ),
    )
    parser.add_argument(
        "--disable-anchor-suppression",
        action="store_true",
        help="Disable derived candidates that clean a tail-safe vocal with a stricter vocal anchor.",
    )
    parser.add_argument(
        "--disable-body-repair",
        action="store_true",
        help="Disable derived candidates that repair vocal-body dips from consensus donors.",
    )
    parser.add_argument(
        "--disable-segment-router",
        action="store_true",
        help="Disable the derived candidate that routes clean/strict/body-preserve behavior by local time-frequency evidence.",
    )
    parser.add_argument(
        "--disable-issue-aware-segment-selection",
        action="store_true",
        help="Disable the derived candidate that uses segment issue diagnostics to choose less problematic windows.",
    )
    parser.add_argument(
        "--disable-segment-issue-repair",
        action="store_true",
        help="Disable the derived candidate that applies label-driven repair only inside unresolved issue windows.",
    )
    parser.add_argument(
        "--disable-no-vocal-leakage-cleanup",
        action="store_true",
        help="Disable the derived candidate that suppresses consensus no-vocal leakage windows.",
    )
    parser.add_argument(
        "--disable-vocal-harshness-guard",
        action="store_true",
        help="Disable the derived candidate that cleans detector-marked high-band harshness windows.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="adaptive_vocal_isolation.py",
        entrypoint_path="/app/tools/adaptive_vocal_isolation.py",
    )
    args = parse_args(argv)
    manifest = run_adaptive_vocal_isolation(args)
    print(
        json.dumps(
            {
                "manifest_path": str(Path(args.output_dir) / "adaptive_vocal_manifest.json"),
                "selected_profile": manifest["selected_profile"],
                "selected_score": manifest["selected_score"],
                "selected_artifacts": manifest["selected_artifacts"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
