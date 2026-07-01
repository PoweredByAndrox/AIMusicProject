#!/usr/bin/env python3
"""Train a Mac/MPS refiner from a MelBand pitch-guard failure.

This is pseudo-teacher training. It may produce a useful diagnostic checkpoint,
but it is not verified ground truth and cannot publish listen output.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.apply_mac_stem_refiner_to_mel_guard import (
    _build_region_mask_and_source,
    _fit_audio,
    _load_json,
    _resolve_audio_path,
    _rms,
)
from tools.mac_melband_pitch_guard import (
    pitch_shift_resample_same_length,
    score_best_value,
    score_pitch_shifted_vocal_leak,
    score_region_count,
    score_target_issue_improved,
)
from tools.stem_region_repair_common import ISSUE_VOCAL_LEAK, read_audio_stereo, write_audio, write_json
from tools.stem_region_repair_model import (
    StemRegionRepairTrainConfig,
    apply_region_repair_model,
    resolve_region_repair_training_device,
    save_region_repair_checkpoint,
    train_region_repair_model_from_arrays,
    write_training_history,
)


def _append_progress(path: Path, phase: str, status: str, **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_unix": time.time(), "phase": phase, "status": status, **extra}, sort_keys=True) + "\n")
        handle.flush()


def _projection_coeff(bad_stem: np.ndarray, source: np.ndarray, mask: np.ndarray) -> float:
    weights = mask[:, None].astype(np.float64)
    src = source.astype(np.float64) * weights
    bad = bad_stem.astype(np.float64) * weights
    denom = float(np.sum(np.square(src)) + 1.0e-10)
    return float(np.sum(bad * src) / denom)


def _merge_training_ranges(mask: np.ndarray, *, sample_rate: int, context_sec: float) -> list[tuple[int, int]]:
    active = np.flatnonzero(np.asarray(mask, dtype=np.float32) > 1.0e-4)
    if active.size <= 0:
        return []
    context = max(1, int(round(float(context_sec) * float(sample_rate))))
    frame_count = int(mask.shape[0])
    raw_ranges: list[tuple[int, int]] = []
    start = int(active[0])
    end = int(active[0])
    for frame in active[1:]:
        frame = int(frame)
        if frame <= end + context:
            end = frame
            continue
        raw_ranges.append((max(0, start - context), min(frame_count, end + context + 1)))
        start = frame
        end = frame
    raw_ranges.append((max(0, start - context), min(frame_count, end + context + 1)))

    merged: list[tuple[int, int]] = []
    for start_frame, end_frame in raw_ranges:
        if not merged or start_frame > merged[-1][1]:
            merged.append((start_frame, end_frame))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end_frame))
    return merged


def _concat_ranges(audio: np.ndarray, ranges: list[tuple[int, int]]) -> np.ndarray:
    if not ranges:
        return audio
    return np.concatenate([audio[start:end] for start, end in ranges], axis=0).astype(np.float32)


def _build_training_crop(
    *,
    raw_audio: np.ndarray,
    instrumental_audio: np.ndarray,
    pseudo_target: np.ndarray,
    source_condition: np.ndarray,
    region_mask: np.ndarray,
    sample_rate: int,
    context_sec: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    ranges = _merge_training_ranges(region_mask, sample_rate=sample_rate, context_sec=context_sec)
    if not ranges:
        return raw_audio, instrumental_audio, pseudo_target, source_condition, region_mask, {
            "enabled": False,
            "reason": "no_active_region",
            "original_frames": int(region_mask.shape[0]),
            "training_frames": int(region_mask.shape[0]),
            "ranges": [],
        }
    cropped_raw = _concat_ranges(raw_audio, ranges)
    cropped_bad = _concat_ranges(instrumental_audio, ranges)
    cropped_target = _concat_ranges(pseudo_target, ranges)
    cropped_source = _concat_ranges(source_condition, ranges)
    cropped_mask = np.concatenate([region_mask[start:end] for start, end in ranges], axis=0).astype(np.float32)
    return cropped_raw, cropped_bad, cropped_target, cropped_source, cropped_mask, {
        "enabled": True,
        "context_sec": float(context_sec),
        "original_frames": int(region_mask.shape[0]),
        "training_frames": int(cropped_mask.shape[0]),
        "active_frames": int(np.sum(cropped_mask > 1.0e-4)),
        "ranges": [
            {
                "start_frame": int(start),
                "end_frame": int(end),
                "start_sec": round(float(start) / float(sample_rate), 6),
                "end_sec": round(float(end) / float(sample_rate), 6),
            }
            for start, end in ranges
        ],
    }


def _build_training_region_constraint_masks(
    *,
    regions: Sequence[dict[str, Any]],
    sample_rate: int,
    original_frame_count: int,
    training_crop: dict[str, Any],
    pad_sec: float,
) -> list[np.ndarray]:
    crop_ranges = list(training_crop.get("ranges") or [])
    if not crop_ranges:
        crop_ranges = [{"start_frame": 0, "end_frame": int(original_frame_count)}]
    constraint_masks: list[np.ndarray] = []
    for region in regions:
        start = max(0, int(round((float(region.get("start", 0.0)) - float(pad_sec)) * float(sample_rate))))
        end = min(
            int(original_frame_count),
            int(round((float(region.get("end", 0.0)) + float(pad_sec)) * float(sample_rate))),
        )
        if end <= start:
            continue
        original_mask = np.zeros(int(original_frame_count), dtype=np.float32)
        original_mask[start:end] = 1.0
        cropped_parts: list[np.ndarray] = []
        for crop_range in crop_ranges:
            crop_start = int(crop_range.get("start_frame", 0))
            crop_end = int(crop_range.get("end_frame", original_frame_count))
            cropped_parts.append(original_mask[crop_start:crop_end])
        if not cropped_parts:
            continue
        cropped_mask = np.concatenate(cropped_parts, axis=0).astype(np.float32)
        if float(np.max(cropped_mask)) > 0.0:
            constraint_masks.append(cropped_mask)
    return constraint_masks


def _normalize_probe_semitones(values: Sequence[float] | None) -> list[float]:
    if not values:
        return []
    result: list[float] = []
    for value in values:
        semitones = float(value)
        if semitones not in result:
            result.append(semitones)
    if len(result) > 16:
        raise ValueError("too many probe semitones")
    return result


def _build_training_source_condition(
    *,
    vocals_audio: np.ndarray,
    sample_rate: int,
    regions: list[dict[str, Any]],
    pad_sec: float,
    fade_sec: float,
    probe_semitones: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], str, list[float]]:
    probe_values = _normalize_probe_semitones(probe_semitones)
    if not probe_values:
        region_mask, source_condition, applied_regions = _build_region_mask_and_source(
            vocals_audio=vocals_audio,
            sample_rate=sample_rate,
            regions=regions,
            pad_sec=pad_sec,
            fade_sec=fade_sec,
        )
        return region_mask, source_condition, applied_regions, "region_probe", []

    frame_count = int(vocals_audio.shape[0])
    region_mask = np.zeros(frame_count, dtype=np.float32)
    source_condition = np.zeros_like(vocals_audio, dtype=np.float32)
    probe_cache = {float(semitones): pitch_shift_resample_same_length(vocals_audio, semitones) for semitones in probe_values}
    applied_regions: list[dict[str, Any]] = []
    for region in regions:
        local_mask, _, local_applied = _build_region_mask_and_source(
            vocals_audio=vocals_audio,
            sample_rate=sample_rate,
            regions=[region],
            pad_sec=pad_sec,
            fade_sec=fade_sec,
        )
        if not local_applied:
            continue
        region_semitones = float(region.get("semitones", 0.0))
        if np.isfinite(region_semitones):
            selected_probe_values = [region_semitones]
            if region_semitones not in probe_cache:
                probe_cache[region_semitones] = pitch_shift_resample_same_length(vocals_audio, region_semitones)
        else:
            selected_probe_values = list(probe_values)
        combined_probe = np.zeros_like(vocals_audio, dtype=np.float32)
        for semitones in selected_probe_values:
            combined_probe += probe_cache[float(semitones)]
        probe_scale = float(np.sqrt(max(1, len(selected_probe_values))))
        combined_probe = (combined_probe / probe_scale).astype(np.float32)
        source_condition += combined_probe * local_mask[:, None]
        region_mask = np.maximum(region_mask, local_mask)
        applied = dict(local_applied[0])
        applied["probe_semitones"] = [float(value) for value in selected_probe_values]
        applied["source_condition_mode"] = "multi_probe"
        applied_regions.append(applied)
    return region_mask.astype(np.float32), source_condition.astype(np.float32), applied_regions, "multi_probe", probe_values


def _training_config_for_profile(*, objective_profile: str, epochs: int, device: str) -> StemRegionRepairTrainConfig:
    if str(objective_profile) == "pitch_region_detector_damage_margin_v3":
        return StemRegionRepairTrainConfig(
            hidden_channels=56,
            kernel_size=41,
            dilated_layers=2,
            epochs=int(epochs),
            learning_rate=0.0020,
            region_loss_weight=0.10,
            outside_loss_weight=1.05,
            forensic_source_conditioning=True,
            forensic_source_decorrelation_loss_weight=1.0,
            forensic_source_envelope_loss_weight=1.05,
            forensic_source_active_energy_loss_weight=0.10,
            forensic_source_target_loss_weight=0.12,
            forensic_source_correlation_loss_weight=4.0,
            forensic_source_projection_margin_loss_weight=18.0,
            forensic_source_projection_margin=0.016,
            forensic_source_hard_region_loss_weight=6.0,
            instrumental_region_delta_ratio_loss_weight=28.0,
            instrumental_region_delta_ratio_limit=0.38,
            target_delta_active_loss_weight=0.06,
            source_skip_init_from_target=False,
            max_delta=0.55,
            device=str(device),
        )
    if str(objective_profile) == "pitch_region_detector_damage_aware_v2":
        return StemRegionRepairTrainConfig(
            hidden_channels=56,
            kernel_size=41,
            dilated_layers=2,
            epochs=int(epochs),
            learning_rate=0.0020,
            region_loss_weight=0.10,
            outside_loss_weight=0.95,
            forensic_source_conditioning=True,
            forensic_source_decorrelation_loss_weight=1.0,
            forensic_source_envelope_loss_weight=1.05,
            forensic_source_active_energy_loss_weight=0.10,
            forensic_source_target_loss_weight=0.12,
            forensic_source_correlation_loss_weight=4.0,
            forensic_source_projection_margin_loss_weight=18.0,
            forensic_source_projection_margin=0.016,
            forensic_source_hard_region_loss_weight=6.0,
            instrumental_region_delta_ratio_loss_weight=12.0,
            instrumental_region_delta_ratio_limit=0.42,
            target_delta_active_loss_weight=0.07,
            source_skip_init_from_target=False,
            max_delta=0.55,
            device=str(device),
        )
    if str(objective_profile) == "pitch_region_detector_damage_guarded_v1":
        return StemRegionRepairTrainConfig(
            hidden_channels=48,
            kernel_size=41,
            dilated_layers=2,
            epochs=int(epochs),
            learning_rate=0.0018,
            region_loss_weight=0.08,
            outside_loss_weight=1.20,
            forensic_source_conditioning=True,
            forensic_source_decorrelation_loss_weight=0.75,
            forensic_source_envelope_loss_weight=0.70,
            forensic_source_active_energy_loss_weight=0.08,
            forensic_source_target_loss_weight=0.08,
            forensic_source_correlation_loss_weight=3.0,
            forensic_source_projection_margin_loss_weight=14.0,
            forensic_source_projection_margin=0.014,
            forensic_source_hard_region_loss_weight=3.0,
            target_delta_active_loss_weight=0.04,
            source_skip_init_from_target=False,
            max_delta=0.42,
            device=str(device),
        )
    if str(objective_profile) == "pitch_region_detector_objective_v3":
        return StemRegionRepairTrainConfig(
            hidden_channels=56,
            kernel_size=41,
            dilated_layers=2,
            epochs=int(epochs),
            learning_rate=0.0022,
            region_loss_weight=0.12,
            outside_loss_weight=0.85,
            forensic_source_conditioning=True,
            forensic_source_decorrelation_loss_weight=1.0,
            forensic_source_envelope_loss_weight=1.15,
            forensic_source_active_energy_loss_weight=0.12,
            forensic_source_target_loss_weight=0.15,
            forensic_source_correlation_loss_weight=4.0,
            forensic_source_projection_margin_loss_weight=18.0,
            forensic_source_projection_margin=0.018,
            forensic_source_hard_region_loss_weight=6.0,
            target_delta_active_loss_weight=0.10,
            source_skip_init_from_target=False,
            max_delta=0.65,
            device=str(device),
        )
    if str(objective_profile) == "pitch_region_count_attack_v2":
        return StemRegionRepairTrainConfig(
            hidden_channels=48,
            kernel_size=41,
            dilated_layers=2,
            epochs=int(epochs),
            learning_rate=0.0025,
            region_loss_weight=0.8,
            outside_loss_weight=0.75,
            forensic_source_conditioning=True,
            forensic_source_decorrelation_loss_weight=1.25,
            forensic_source_envelope_loss_weight=0.45,
            forensic_source_active_energy_loss_weight=0.35,
            forensic_source_target_loss_weight=0.70,
            target_delta_active_loss_weight=0.45,
            source_skip_init_from_target=True,
            max_delta=1.0,
            device=str(device),
        )
    if str(objective_profile) == "multi_probe_anti_vocal":
        return StemRegionRepairTrainConfig(
            hidden_channels=32,
            kernel_size=31,
            epochs=int(epochs),
            learning_rate=0.003,
            region_loss_weight=1.0,
            outside_loss_weight=0.55,
            forensic_source_conditioning=True,
            forensic_source_decorrelation_loss_weight=0.45,
            forensic_source_envelope_loss_weight=0.24,
            forensic_source_active_energy_loss_weight=0.18,
            forensic_source_target_loss_weight=0.35,
            target_delta_active_loss_weight=0.18,
            source_skip_init_from_target=True,
            max_delta=0.95,
            device=str(device),
        )
    return StemRegionRepairTrainConfig(
        hidden_channels=24,
        kernel_size=31,
        epochs=int(epochs),
        learning_rate=0.003,
        region_loss_weight=1.0,
        outside_loss_weight=0.45,
        forensic_source_conditioning=True,
        forensic_source_decorrelation_loss_weight=0.2,
        forensic_source_envelope_loss_weight=0.08,
        max_delta=0.8,
        device=str(device),
    )


def _score_has_guard_regions(score: Any) -> bool:
    return isinstance(score, dict) and isinstance(score.get("regions"), list) and "config" in score


def _requires_region_count_reduction(objective_profile: str) -> bool:
    return str(objective_profile) in {
        "pitch_region_count_attack_v2",
        "pitch_region_detector_objective_v3",
        "pitch_region_detector_damage_guarded_v1",
        "pitch_region_detector_damage_aware_v2",
        "pitch_region_detector_damage_margin_v3",
    }


def run_training(
    *,
    mel_guard_dir: Path,
    output_dir: Path,
    target_gain: float = 1.0,
    epochs: int = 160,
    require_mps: bool = True,
    max_regions: int = 8,
    probe_semitones: Sequence[float] | None = None,
    objective_profile: str = "projection_teacher",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    _append_progress(progress_path, "preflight", "start", require_mps=bool(require_mps))
    device = resolve_region_repair_training_device("mps")
    if bool(require_mps) and device.type != "mps":
        manifest = {
            "schema": "mac_melband_refiner_training.v1",
            "status": "blocked_mps_unavailable",
            "delivery_allowed": False,
            "presentation_status": "training_diagnostic_only",
            "resolved_device": str(device),
        }
        write_json(output_dir / "manifest.json", manifest)
        _append_progress(progress_path, "preflight", "blocked_mps_unavailable", resolved_device=str(device))
        return manifest

    guard_manifest = _load_json(Path(mel_guard_dir) / "manifest.json")
    audio_map = dict(guard_manifest.get("audio") or {})
    vocals_path = _resolve_audio_path(str(audio_map.get("vocals")), guard_dir=Path(mel_guard_dir))
    instrumental_path = _resolve_audio_path(str(audio_map.get("instrumental")), guard_dir=Path(mel_guard_dir))
    raw_path = _resolve_audio_path(str(audio_map.get("raw_ref")), guard_dir=Path(mel_guard_dir))
    if _score_has_guard_regions(guard_manifest.get("score")):
        before_score = dict(guard_manifest["score"])
        write_json(output_dir / "score" / "before" / "pitch.json", before_score)
        _append_progress(
            progress_path,
            "before_score",
            "reused_guard_manifest",
            region_count=score_region_count(before_score),
            selected_region_count=len(before_score.get("regions") or []),
            verdict=before_score.get("verdict"),
        )
    else:
        _append_progress(progress_path, "before_score", "scan_start")
        before_score = score_pitch_shifted_vocal_leak(
            vocals_path=vocals_path,
            instrumental_path=instrumental_path,
            output_dir=output_dir / "score" / "before",
            config=(guard_manifest.get("score") or {}).get("config"),
            progress_callback=lambda event: _append_progress(progress_path, "before_score", "progress", **event),
        )
        _append_progress(
            progress_path,
            "before_score",
            "scan_end",
            region_count=score_region_count(before_score),
            selected_region_count=len(before_score.get("regions") or []),
            verdict=before_score.get("verdict"),
        )
    regions = list(before_score.get("regions") or [])[: int(max_regions)]
    if not regions:
        manifest = {
            "schema": "mac_melband_refiner_training.v1",
            "status": "skipped_no_guard_regions",
            "delivery_allowed": False,
            "presentation_status": "training_diagnostic_only",
            "before_score": before_score,
        }
        write_json(output_dir / "manifest.json", manifest)
        return manifest

    vocals_audio, vocals_rate = read_audio_stereo(vocals_path)
    instrumental_audio, instrumental_rate = read_audio_stereo(instrumental_path)
    raw_audio, raw_rate = read_audio_stereo(raw_path)
    if len({vocals_rate, instrumental_rate, raw_rate}) != 1:
        raise ValueError("mac_refiner_training_sample_rate_mismatch")
    vocals_audio, instrumental_audio, raw_audio = _fit_audio(vocals_audio, instrumental_audio, raw_audio)
    region_mask, source_condition, applied_regions, source_condition_mode, normalized_probe_semitones = _build_training_source_condition(
        vocals_audio=vocals_audio,
        sample_rate=vocals_rate,
        regions=regions,
        pad_sec=0.08,
        fade_sec=0.03,
        probe_semitones=probe_semitones,
    )
    coeff = _projection_coeff(instrumental_audio, source_condition, region_mask)
    coeff = max(-2.5, min(2.5, coeff * float(target_gain)))
    pseudo_target = np.clip(instrumental_audio - coeff * source_condition * region_mask[:, None], -1.0, 1.0).astype(np.float32)
    audio_dir = output_dir / "audio"
    write_audio(audio_dir / "bad.flac", instrumental_audio, vocals_rate)
    write_audio(audio_dir / "target.flac", pseudo_target, vocals_rate)
    write_audio(audio_dir / "src.flac", source_condition, vocals_rate)
    write_audio(audio_dir / "raw.flac", raw_audio, vocals_rate)

    (
        training_raw,
        training_bad,
        training_target,
        training_source,
        training_mask,
        training_crop,
    ) = _build_training_crop(
        raw_audio=raw_audio,
        instrumental_audio=instrumental_audio,
        pseudo_target=pseudo_target,
        source_condition=source_condition,
        region_mask=region_mask,
        sample_rate=vocals_rate,
    )
    training_constraint_masks = _build_training_region_constraint_masks(
        regions=regions,
        sample_rate=vocals_rate,
        original_frame_count=int(region_mask.shape[0]),
        training_crop=training_crop,
        pad_sec=0.08,
    )
    _append_progress(progress_path, "train_crop", "end", **training_crop)
    _append_progress(
        progress_path,
        "train",
        "start",
        device=str(device),
        epochs=int(epochs),
        projection_coeff=coeff,
        training_frames=int(training_mask.shape[0]),
        original_frames=int(region_mask.shape[0]),
        constraint_region_count=len(training_constraint_masks),
    )

    def _training_progress(event: dict[str, Any]) -> None:
        _append_progress(progress_path, "train", "progress", **event)

    model, history = train_region_repair_model_from_arrays(
        mix_audio=training_raw,
        bad_stem_audio=training_bad,
        target_stem_audio=training_target,
        forensic_source_audio=training_source,
        region_mask=training_mask,
        region_constraint_masks=training_constraint_masks,
        sample_rate=vocals_rate,
        issue_type=ISSUE_VOCAL_LEAK,
        config=_training_config_for_profile(objective_profile=str(objective_profile), epochs=int(epochs), device=str(device)),
        progress_callback=_training_progress,
    )
    checkpoint_path = save_region_repair_checkpoint(model=model, path=output_dir / "model.pt", training_history=history)
    write_training_history(output_dir / "history.json", history)
    _append_progress(progress_path, "train", "end", final_loss=history.get("final_loss"), resolved_device=history.get("resolved_device"))

    repaired_instrumental, apply_diag = apply_region_repair_model(
        model=model,
        mix_audio=raw_audio,
        bad_stem_audio=instrumental_audio,
        source_condition_audio=source_condition,
        region_mask=region_mask,
        sample_rate=vocals_rate,
        issue_type=ISSUE_VOCAL_LEAK,
        device=str(device),
    )
    delta = repaired_instrumental - instrumental_audio
    repaired_vocals = np.clip(vocals_audio - delta, -1.0, 1.0).astype(np.float32)
    repaired_instrumental = np.clip(repaired_instrumental, -1.0, 1.0).astype(np.float32)
    vocals_out = write_audio(audio_dir / "v.flac", repaired_vocals, vocals_rate)
    instrumental_out = write_audio(audio_dir / "i.flac", repaired_instrumental, vocals_rate)
    _append_progress(progress_path, "after_score", "scan_start")
    after_score = score_pitch_shifted_vocal_leak(
        vocals_path=vocals_out,
        instrumental_path=instrumental_out,
        output_dir=output_dir / "score" / "after",
        config=before_score.get("config"),
        progress_callback=lambda event: _append_progress(progress_path, "after_score", "progress", **event),
    )
    _append_progress(
        progress_path,
        "after_score",
        "scan_end",
        region_count=score_region_count(after_score),
        selected_region_count=len(after_score.get("regions") or []),
        verdict=after_score.get("verdict"),
    )
    before_region_count = score_region_count(before_score)
    after_region_count = score_region_count(after_score)
    before_selected_region_count = len(before_score.get("regions") or [])
    after_selected_region_count = len(after_score.get("regions") or [])
    region_count_reduced = int(after_region_count) < int(before_region_count)
    before_best = score_best_value(before_score)
    after_best = score_best_value(after_score)
    requires_region_count_reduction = _requires_region_count_reduction(str(objective_profile))
    target_issue_improved = score_target_issue_improved(before_score=before_score, after_score=after_score)
    if not target_issue_improved and after_score.get("verdict") != "pass":
        target_issue_improved = False
        status = "trained_repair_model_gap"
        root_cause = "model_gap"
        next_training_action = "change_training_objective_or_separator"
    else:
        status = "trained_repair_passed_pitch_guard" if after_score.get("verdict") == "pass" else "trained_repair_still_blocked"
        root_cause = "resolved" if after_score.get("verdict") == "pass" else "repair_gap"
        next_training_action = "validate_or_replay" if after_score.get("verdict") == "pass" else "continue_targeted_training"
    manifest = {
        "schema": "mac_melband_refiner_training.v1",
        "status": status,
        "source_guard_dir": str(mel_guard_dir),
        "checkpoint": str(checkpoint_path),
        "history_json": str(output_dir / "history.json"),
        "resolved_device": str(device),
        "label_type": "pseudo_teacher",
        "is_ground_truth": False,
        "usable_for_promotion": False,
        "target_gain": float(target_gain),
        "projection_coeff": coeff,
        "objective_profile": str(objective_profile),
        "root_cause": root_cause,
        "next_training_action": next_training_action,
        "source_condition_mode": source_condition_mode,
        "probe_semitones": normalized_probe_semitones,
        "applied_regions": applied_regions,
        "audio": {
            "vocals": str(vocals_out),
            "instrumental": str(instrumental_out),
            "raw_ref": str(audio_dir / "raw.flac"),
            "pseudo_target": str(audio_dir / "target.flac"),
        },
        "before_score": before_score,
        "after_score": after_score,
        "metrics": {
            "before_region_count": before_region_count,
            "after_region_count": after_region_count,
            "before_selected_region_count": before_selected_region_count,
            "after_selected_region_count": after_selected_region_count,
            "region_count_reduced": bool(region_count_reduced),
            "before_best_score": before_best,
            "after_best_score": after_best,
            "target_issue_improved": bool(target_issue_improved),
            "reconstruction_rms": _rms(repaired_vocals + repaired_instrumental - raw_audio),
            "instrumental_delta_rms": _rms(delta),
            "training_constraint_region_count": len(training_constraint_masks),
            "apply_diagnostics": apply_diag,
        },
        "training": history,
        "training_crop": training_crop,
        "contract": {
            "pseudo_teacher_only": True,
            "diagnostic_only": True,
            "no_listen_delivery": True,
            "requires_region_count_reduction": bool(requires_region_count_reduction),
            "delivery_allowed": False,
            "usable_for_promotion": False,
        },
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "training_diagnostic_only",
    }
    write_json(output_dir / "manifest.json", manifest)
    _append_progress(
        progress_path,
        "manifest",
        "end",
        run_status=status,
        before_region_count=before_region_count,
        after_region_count=after_region_count,
        target_issue_improved=target_issue_improved,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mel-guard-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-gain", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--max-regions", type=int, default=8)
    parser.add_argument("--probe-semitone", action="append", type=float, dest="probe_semitones")
    parser.add_argument(
        "--objective-profile",
        default="projection_teacher",
        choices=[
            "projection_teacher",
            "multi_probe_anti_vocal",
            "pitch_region_count_attack_v2",
            "pitch_region_detector_objective_v3",
            "pitch_region_detector_damage_guarded_v1",
            "pitch_region_detector_damage_aware_v2",
            "pitch_region_detector_damage_margin_v3",
        ],
    )
    parser.add_argument("--require-mps", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_training(
        mel_guard_dir=args.mel_guard_dir,
        output_dir=args.out,
        target_gain=float(args.target_gain),
        epochs=int(args.epochs),
        require_mps=bool(args.require_mps),
        max_regions=int(args.max_regions),
        probe_semitones=args.probe_semitones,
        objective_profile=str(args.objective_profile),
    )
    print(json.dumps({"manifest": str(args.out / "manifest.json"), "status": manifest["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
