from __future__ import annotations

import argparse
import json
import math
import stat
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from adaptive_vocal_isolation import (  # noqa: E402
    _audio_summary,
    _limit_peak,
    _match_audio,
    _safe_rms,
    calculate_anchor_support_metrics,
    calculate_vocal_proxy_metrics,
    write_json,
)
from export_melband_vocal_preview import (  # noqa: E402
    _apply_light_debleed_presence,
    _apply_peak_note_debleed,
)


@dataclass(frozen=True)
class CandidateArtifact:
    label: str
    path: Path
    strategy: str
    tuning: dict[str, float | int | str]


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    return audio.astype(np.float32, copy=False), int(sample_rate)


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), _limit_peak(audio), int(sample_rate), subtype="PCM_24")


def _match_support_audio(primary_audio: np.ndarray, primary_sr: int, support_path: Path) -> np.ndarray:
    support_audio, support_sr = _read_audio(support_path)
    _, matched_support, _ = _match_audio(primary_audio, primary_sr, support_audio, support_sr)
    return matched_support.astype(np.float32, copy=False)


def _smooth_activity(activity: np.ndarray, frames: int) -> np.ndarray:
    if frames <= 1 or activity.size == 0:
        return np.clip(activity, 0.0, 1.0).astype(np.float32)
    window_size = min(int(frames), int(activity.size))
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    return np.clip(np.convolve(activity, kernel, mode="same"), 0.0, 1.0).astype(np.float32)


def _causal_tail(activity: np.ndarray, *, sample_rate: int, hop_length: int, tail_sec: float) -> np.ndarray:
    if activity.size == 0:
        return np.zeros(0, dtype=np.float32)
    decay = math.exp(-float(hop_length) / max(float(sample_rate) * float(tail_sec), 1e-6))
    carried = 0.0
    tail = np.zeros_like(activity, dtype=np.float32)
    for index, value in enumerate(np.clip(activity, 0.0, 1.0)):
        carried = max(float(value), carried * decay)
        tail[index] = carried
    return np.clip(tail, 0.0, 1.0).astype(np.float32)


def _focus_frame_envelope(
    *,
    frame_count: int,
    sample_rate: int,
    hop_length: int,
    start_sec: float,
    end_sec: float,
    fade_sec: float,
) -> np.ndarray:
    if end_sec <= start_sec:
        raise ValueError("Focus end must be greater than focus start")
    if frame_count <= 0:
        return np.zeros(0, dtype=np.float32)
    frame_times = np.arange(frame_count, dtype=np.float32) * float(hop_length) / float(sample_rate)
    fade = max(float(fade_sec), 1e-6)
    fade_in = np.clip((frame_times - float(start_sec)) / fade, 0.0, 1.0)
    fade_out = np.clip((float(end_sec) - frame_times) / fade, 0.0, 1.0)
    return np.minimum(fade_in, fade_out).astype(np.float32)


def _soft_band_weight(freqs: np.ndarray, *, low_hz: float, high_hz: float, fade_hz: float) -> np.ndarray:
    fade = max(float(fade_hz), 1e-6)
    low_ramp = np.clip((freqs - float(low_hz)) / fade, 0.0, 1.0)
    high_ramp = np.clip((float(high_hz) - freqs) / fade, 0.0, 1.0)
    return np.minimum(low_ramp, high_ramp).astype(np.float32)


def apply_multimodel_support_gate(
    *,
    primary_vocals_path: Path,
    support_vocals_paths: list[Path],
    output_vocals_path: Path,
    support_mode: str,
    support_threshold: float,
    mask_power: float,
    inactive_floor: float,
    active_floor: float,
    vocal_core_floor: float,
    high_band_floor_scale: float,
    high_band_start_hz: float,
    activity_floor_ratio: float,
    activity_smoothing_frames: int,
    tail_memory_sec: float,
    target_rms_ratio: float,
    max_gain: float,
) -> dict[str, Any]:
    primary_audio, primary_sr = _read_audio(primary_vocals_path)
    support_audios = [
        _match_support_audio(primary_audio, primary_sr, support_path)
        for support_path in support_vocals_paths
    ]
    if not support_audios:
        raise ValueError("At least one support vocal path is required")

    frame_count = min([primary_audio.shape[0], *(audio.shape[0] for audio in support_audios)])
    channel_count = min([primary_audio.shape[1], *(audio.shape[1] for audio in support_audios)])
    primary_audio = primary_audio[:frame_count, :channel_count]
    support_audios = [audio[:frame_count, :channel_count] for audio in support_audios]

    n_fft = 2048
    hop_length = 512
    freqs = librosa.fft_frequencies(sr=primary_sr, n_fft=n_fft).astype(np.float32)
    high_band = freqs >= float(high_band_start_hz)
    vocal_core = (freqs >= 140.0) & (freqs <= 4300.0)

    cleaned_channels: list[np.ndarray] = []
    gate_means: list[float] = []
    activity_means: list[float] = []

    for channel_index in range(channel_count):
        primary_channel = primary_audio[:, channel_index]
        primary_stft = librosa.stft(primary_channel, n_fft=n_fft, hop_length=hop_length)
        primary_mag = np.abs(primary_stft).astype(np.float32)
        support_magnitudes = np.stack(
            [
                np.abs(librosa.stft(support[:, channel_index], n_fft=n_fft, hop_length=hop_length)).astype(np.float32)
                for support in support_audios
            ],
            axis=0,
        )
        if support_mode == "max":
            support_mag = np.max(support_magnitudes, axis=0).astype(np.float32)
        elif support_mode == "mean":
            support_mag = np.mean(support_magnitudes, axis=0).astype(np.float32)
        elif support_mode == "median":
            support_mag = np.median(support_magnitudes, axis=0).astype(np.float32)
        else:
            raise ValueError(f"Unsupported support mode: {support_mode}")

        support_energy = np.mean(support_mag[vocal_core, :], axis=0) if np.any(vocal_core) else np.mean(support_mag, axis=0)
        reference_energy = float(np.percentile(support_energy, 95.0)) if support_energy.size else 0.0
        activity = np.clip(
            support_energy / max(reference_energy * float(activity_floor_ratio), 1e-8),
            0.0,
            1.0,
        ).astype(np.float32)
        activity = _smooth_activity(activity, int(activity_smoothing_frames))
        tail_activity = _causal_tail(activity, sample_rate=primary_sr, hop_length=hop_length, tail_sec=float(tail_memory_sec))

        support_ratio = support_mag / (primary_mag + 1e-8)
        support_mask = np.clip(
            (support_ratio - float(support_threshold)) / max(1e-6, 1.0 - float(support_threshold)),
            0.0,
            1.0,
        ).astype(np.float32)
        support_mask = np.power(support_mask, float(mask_power), dtype=np.float32)

        dynamic_floor = np.maximum(
            float(inactive_floor),
            np.maximum(float(active_floor) * activity, float(vocal_core_floor) * tail_activity),
        ).astype(np.float32)
        dynamic_floor_by_bin = dynamic_floor[np.newaxis, :]
        high_floor = dynamic_floor_by_bin * float(high_band_floor_scale)
        final_mask = np.maximum(support_mask, dynamic_floor_by_bin)
        final_mask[vocal_core, :] = np.maximum(final_mask[vocal_core, :], float(vocal_core_floor) * tail_activity[np.newaxis, :])
        final_mask[high_band, :] = np.minimum(final_mask[high_band, :], np.maximum(support_mask[high_band, :], high_floor))

        cleaned_stft = primary_stft * np.clip(final_mask, 0.0, 1.0)
        cleaned = librosa.istft(cleaned_stft, hop_length=hop_length, length=primary_channel.shape[0]).astype(np.float32)
        cleaned_channels.append(cleaned)
        gate_means.append(float(np.mean(final_mask)) if final_mask.size else 0.0)
        activity_means.append(float(np.mean(activity)) if activity.size else 0.0)

    cleaned_audio = np.stack(cleaned_channels, axis=1).astype(np.float32, copy=False)
    source_rms = _safe_rms(primary_audio)
    cleaned_rms = _safe_rms(cleaned_audio)
    if source_rms > 0.0 and cleaned_rms > 0.0 and float(target_rms_ratio) > 0.0:
        gain = min(float(max_gain), (source_rms * float(target_rms_ratio)) / cleaned_rms)
        cleaned_audio = (cleaned_audio * gain).astype(np.float32)
    else:
        gain = 1.0

    _write_audio(output_vocals_path, cleaned_audio, primary_sr)
    return {
        "primary_vocals_path": str(primary_vocals_path),
        "support_vocals_paths": [str(path) for path in support_vocals_paths],
        "support_mode": support_mode,
        "gate_mean": round(float(np.mean(gate_means)) if gate_means else 0.0, 6),
        "activity_mean": round(float(np.mean(activity_means)) if activity_means else 0.0, 6),
        "applied_gain": round(float(gain), 6),
        "audio": _audio_summary(cleaned_audio, primary_sr),
    }


def build_focused_detail_recovery_candidate(
    *,
    base_vocals_path: Path,
    donor_vocals_paths: list[Path],
    output_vocals_path: Path,
    focus_start_sec: float,
    focus_end_sec: float,
    focus_fade_sec: float,
    body_max_gain: float,
    body_blend: float,
    air_max_gain: float,
    air_blend: float,
) -> dict[str, Any]:
    base_audio, base_sr = _read_audio(base_vocals_path)
    donor_audios = [
        _match_support_audio(base_audio, base_sr, donor_path)
        for donor_path in donor_vocals_paths
    ]
    if not donor_audios:
        raise ValueError("At least one donor vocal path is required")

    frame_count = min([base_audio.shape[0], *(audio.shape[0] for audio in donor_audios)])
    channel_count = min([base_audio.shape[1], *(audio.shape[1] for audio in donor_audios)])
    base_audio = base_audio[:frame_count, :channel_count]
    donor_audios = [audio[:frame_count, :channel_count] for audio in donor_audios]

    n_fft = 2048
    hop_length = 512
    freqs = librosa.fft_frequencies(sr=base_sr, n_fft=n_fft).astype(np.float32)
    body_weight = _soft_band_weight(freqs, low_hz=160.0, high_hz=4300.0, fade_hz=160.0)[:, np.newaxis]
    air_weight = _soft_band_weight(freqs, low_hz=4600.0, high_hz=9000.0, fade_hz=500.0)[:, np.newaxis]

    recovered_channels: list[np.ndarray] = []
    gain_means: list[float] = []
    focus_envelope_mean = 0.0

    for channel_index in range(channel_count):
        base_channel = base_audio[:, channel_index]
        base_stft = librosa.stft(base_channel, n_fft=n_fft, hop_length=hop_length)
        base_mag = np.abs(base_stft).astype(np.float32)
        donor_magnitudes = np.stack(
            [
                np.abs(librosa.stft(donor[:, channel_index], n_fft=n_fft, hop_length=hop_length)).astype(np.float32)
                for donor in donor_audios
            ],
            axis=0,
        )
        donor_mag = np.median(donor_magnitudes, axis=0).astype(np.float32)
        focus_envelope = _focus_frame_envelope(
            frame_count=base_mag.shape[1],
            sample_rate=base_sr,
            hop_length=hop_length,
            start_sec=focus_start_sec,
            end_sec=focus_end_sec,
            fade_sec=focus_fade_sec,
        )[np.newaxis, :]
        focus_envelope_mean = float(np.mean(focus_envelope)) if focus_envelope.size else 0.0

        donor_ratio = donor_mag / (base_mag + 1e-8)
        body_gain = 1.0 + np.clip(donor_ratio - 1.0, 0.0, max(float(body_max_gain) - 1.0, 0.0)) * float(body_blend)
        air_gain = 1.0 + np.clip(donor_ratio - 1.0, 0.0, max(float(air_max_gain) - 1.0, 0.0)) * float(air_blend)
        body_bins = body_weight[:, 0] > 0.05
        air_bins = air_weight[:, 0] > 0.05
        if np.any(body_bins):
            base_body_energy = np.mean(base_mag[body_bins, :], axis=0)
            donor_body_energy = np.mean(donor_mag[body_bins, :], axis=0)
            body_frame_ratio = donor_body_energy / (base_body_energy + 1e-8)
            body_frame_gain = 1.0 + np.clip(body_frame_ratio - 1.0, 0.0, max(float(body_max_gain) - 1.0, 0.0)) * float(body_blend)
            body_gain = np.maximum(body_gain, body_frame_gain[np.newaxis, :])
        if np.any(air_bins):
            base_air_energy = np.mean(base_mag[air_bins, :], axis=0)
            donor_air_energy = np.mean(donor_mag[air_bins, :], axis=0)
            air_frame_ratio = donor_air_energy / (base_air_energy + 1e-8)
            air_frame_gain = 1.0 + np.clip(air_frame_ratio - 1.0, 0.0, max(float(air_max_gain) - 1.0, 0.0)) * float(air_blend)
            air_gain = np.maximum(air_gain, air_frame_gain[np.newaxis, :])
        final_gain = 1.0 + ((body_gain - 1.0) * body_weight + (air_gain - 1.0) * air_weight) * focus_envelope
        final_gain = np.clip(final_gain, 1.0, max(float(body_max_gain), float(air_max_gain))).astype(np.float32)

        recovered_stft = base_stft * final_gain
        recovered = librosa.istft(recovered_stft, hop_length=hop_length, length=base_channel.shape[0]).astype(np.float32)
        recovered_channels.append(recovered)
        gain_means.append(float(np.mean(final_gain)) if final_gain.size else 1.0)

    recovered_audio = np.stack(recovered_channels, axis=1).astype(np.float32, copy=False)
    _write_audio(output_vocals_path, recovered_audio, base_sr)
    return {
        "base_vocals_path": str(base_vocals_path),
        "donor_vocals_paths": [str(path) for path in donor_vocals_paths],
        "focus_start_sec": focus_start_sec,
        "focus_end_sec": focus_end_sec,
        "focus_fade_sec": focus_fade_sec,
        "body_max_gain": body_max_gain,
        "body_blend": body_blend,
        "air_max_gain": air_max_gain,
        "air_blend": air_blend,
        "gain_mean": round(float(np.mean(gain_means)) if gain_means else 1.0, 6),
        "focus_envelope_mean": round(float(focus_envelope_mean), 6),
        "audio": _audio_summary(recovered_audio, base_sr),
    }


def build_melband_debleed_candidate(
    *,
    melband_vocals_path: Path,
    melband_instrumental_path: Path,
    output_vocals_path: Path,
) -> dict[str, Any]:
    vocals_audio, vocals_sr = _read_audio(melband_vocals_path)
    instrumental_audio, instrumental_sr = _read_audio(melband_instrumental_path)
    vocals_audio, instrumental_audio, sample_rate = _match_audio(
        vocals_audio,
        vocals_sr,
        instrumental_audio,
        instrumental_sr,
    )
    cleaned = _apply_light_debleed_presence(
        vocals_audio,
        instrumental_audio,
        sample_rate=sample_rate,
        debleed_strength=0.30,
        debleed_start_hz=2200.0,
        debleed_margin=1.04,
        debleed_power=1.25,
        presence_gain=1.025,
        presence_start_hz=1500.0,
    )
    cleaned = _apply_peak_note_debleed(
        cleaned,
        instrumental_audio,
        sample_rate=sample_rate,
        peak_strength=0.22,
        peak_start_hz=1800.0,
        peak_dominance_ratio=1.28,
        peak_local_ratio=1.55,
        peak_spread_bins=1,
        presence_gain=1.015,
        presence_start_hz=1500.0,
    )
    _write_audio(output_vocals_path, cleaned, sample_rate)
    return {
        "melband_vocals_path": str(melband_vocals_path),
        "melband_instrumental_path": str(melband_instrumental_path),
        "postprocess": "light_debleed_plus_peak_note_suppressor",
        "audio": _audio_summary(cleaned, sample_rate),
    }


def score_candidate(
    source_audio_path: Path,
    candidate: CandidateArtifact,
    anchor_paths: list[tuple[str, Path]],
) -> dict[str, Any]:
    metrics = calculate_vocal_proxy_metrics(
        source_audio_path,
        candidate.path,
        start_sec=0.0,
        end_sec=None,
    )
    anchor_metrics = {}
    for anchor_label, anchor_path in anchor_paths:
        if anchor_path.exists() and anchor_path != candidate.path:
            anchor_metrics[anchor_label] = calculate_anchor_support_metrics(candidate.path, anchor_path)
    return {
        "label": candidate.label,
        "path": str(candidate.path),
        "strategy": candidate.strategy,
        "tuning": candidate.tuning,
        "metrics": metrics,
        "anchor_metrics": anchor_metrics,
    }


def calculate_delta_metrics(candidates: list[CandidateArtifact]) -> list[dict[str, Any]]:
    if not candidates:
        return []

    baseline = candidates[0]
    baseline_audio, baseline_sr = _read_audio(baseline.path)
    deltas: list[dict[str, Any]] = []
    for candidate in candidates[1:]:
        candidate_audio, candidate_sr = _read_audio(candidate.path)
        matched_baseline, matched_candidate, _ = _match_audio(
            baseline_audio,
            baseline_sr,
            candidate_audio,
            candidate_sr,
        )
        delta = matched_candidate - matched_baseline
        deltas.append(
            {
                "baseline_label": baseline.label,
                "candidate_label": candidate.label,
                "candidate_rms": round(_safe_rms(matched_candidate), 6),
                "delta_vs_baseline_rms": round(_safe_rms(delta), 6),
                "delta_vs_baseline_peak": round(float(np.max(np.abs(delta))) if delta.size else 0.0, 6),
            }
        )
    return deltas


def _unlink_generated_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        path.unlink()
        return
    except PermissionError:
        path.chmod(path.stat().st_mode | stat.S_IWRITE)
        path.unlink()


def write_listen_pack(
    *,
    listen_dir: Path,
    raw_ref_path: Path,
    candidates: list[CandidateArtifact],
    metrics: list[dict[str, Any]],
    delta_metrics: list[dict[str, Any]],
) -> None:
    listen_dir.mkdir(parents=True, exist_ok=True)
    for generated_path in [
        listen_dir / "candidate_map.json",
        listen_dir / "model_zoo_metrics.json",
        listen_dir / "model_zoo_delta_metrics.json",
        listen_dir / "00_README.txt",
    ]:
        _unlink_generated_file(generated_path)
    for stale_audio_path in listen_dir.glob("[0-9][0-9]_*.flac"):
        _unlink_generated_file(stale_audio_path)

    mapping: dict[str, str] = {}
    raw_target = listen_dir / "00_raw_ref.flac"
    shutil.copyfile(raw_ref_path, raw_target)
    mapping[raw_target.name] = str(raw_ref_path)
    for candidate in candidates:
        target = listen_dir / candidate.label
        shutil.copyfile(candidate.path, target)
        mapping[candidate.label] = str(candidate.path)
    write_json(listen_dir / "candidate_map.json", mapping)
    write_json(listen_dir / "model_zoo_metrics.json", {"candidates": metrics})
    write_json(listen_dir / "model_zoo_delta_metrics.json", {"deltas": delta_metrics})
    (listen_dir / "00_README.txt").write_text(
        "Listen only here.\n00 = raw first30 reference.\n01+ = model-zoo isolated vocal candidates.\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model-zoo vocal candidates from existing first30 model outputs.")
    parser.add_argument("--source-audio-path", required=True)
    parser.add_argument("--raw-ref-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--listen-dir", required=True)
    parser.add_argument("--bs-vocals-path", required=True)
    parser.add_argument("--mdx-vocals-path", required=True)
    parser.add_argument("--melband-vocals-path", required=True)
    parser.add_argument("--melband-instrumental-path", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_vocal_model_zoo_candidates.py",
        entrypoint_path="/app/tools/build_vocal_model_zoo_candidates.py",
    )
    args = parse_args(argv)
    source_audio_path = Path(args.source_audio_path)
    raw_ref_path = Path(args.raw_ref_path)
    output_dir = Path(args.output_dir)
    listen_dir = Path(args.listen_dir)
    bs_vocals_path = Path(args.bs_vocals_path)
    mdx_vocals_path = Path(args.mdx_vocals_path)
    melband_vocals_path = Path(args.melband_vocals_path)
    melband_instrumental_path = Path(args.melband_instrumental_path)

    for required_path in [
        source_audio_path,
        raw_ref_path,
        bs_vocals_path,
        mdx_vocals_path,
        melband_vocals_path,
        melband_instrumental_path,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[CandidateArtifact] = [
        CandidateArtifact(
            label="01_old_bs_artifact_clean_first30.flac",
            path=bs_vocals_path,
            strategy="reference_existing_bs_clean_candidate",
            tuning={},
        ),
        CandidateArtifact(
            label="02_mdx23c_first30.flac",
            path=mdx_vocals_path,
            strategy="external_mdx23c_vocals",
            tuning={},
        ),
        CandidateArtifact(
            label="03_melband_plain_first30.flac",
            path=melband_vocals_path,
            strategy="external_melband_roformer_vocals",
            tuning={},
        ),
    ]

    melband_debleed_path = output_dir / "04_melband_debleed_first30.flac"
    melband_debleed_debug = build_melband_debleed_candidate(
        melband_vocals_path=melband_vocals_path,
        melband_instrumental_path=melband_instrumental_path,
        output_vocals_path=melband_debleed_path,
    )
    artifacts.append(
        CandidateArtifact(
            label="04_melband_debleed_first30.flac",
            path=melband_debleed_path,
            strategy="melband_instrumental_guided_debleed",
            tuning={"profile": "light_debleed_plus_peak_note_suppressor"},
        )
    )

    bs_mdx_gate_path = output_dir / "05_bs_supported_by_mdx_first30.flac"
    bs_mdx_debug = apply_multimodel_support_gate(
        primary_vocals_path=bs_vocals_path,
        support_vocals_paths=[mdx_vocals_path],
        output_vocals_path=bs_mdx_gate_path,
        support_mode="max",
        support_threshold=0.16,
        mask_power=1.25,
        inactive_floor=0.10,
        active_floor=0.48,
        vocal_core_floor=0.58,
        high_band_floor_scale=0.52,
        high_band_start_hz=5200.0,
        activity_floor_ratio=0.050,
        activity_smoothing_frames=13,
        tail_memory_sec=0.34,
        target_rms_ratio=0.98,
        max_gain=1.08,
    )
    artifacts.append(
        CandidateArtifact(
            label="05_bs_supported_by_mdx_first30.flac",
            path=bs_mdx_gate_path,
            strategy="bs_primary_mdx_support_gate",
            tuning={"support_threshold": 0.16, "mask_power": 1.25},
        )
    )

    bs_consensus_gate_path = output_dir / "06_bs_supported_by_mdx_melband_first30.flac"
    bs_consensus_debug = apply_multimodel_support_gate(
        primary_vocals_path=bs_vocals_path,
        support_vocals_paths=[mdx_vocals_path, melband_debleed_path],
        output_vocals_path=bs_consensus_gate_path,
        support_mode="max",
        support_threshold=0.20,
        mask_power=1.35,
        inactive_floor=0.08,
        active_floor=0.44,
        vocal_core_floor=0.56,
        high_band_floor_scale=0.46,
        high_band_start_hz=5000.0,
        activity_floor_ratio=0.052,
        activity_smoothing_frames=13,
        tail_memory_sec=0.36,
        target_rms_ratio=0.98,
        max_gain=1.08,
    )
    artifacts.append(
        CandidateArtifact(
            label="06_bs_supported_by_mdx_melband_first30.flac",
            path=bs_consensus_gate_path,
            strategy="bs_primary_external_model_support_gate",
            tuning={"support_threshold": 0.20, "mask_power": 1.35},
        )
    )

    melband_consensus_gate_path = output_dir / "07_melband_supported_by_bs_mdx_first30.flac"
    melband_consensus_debug = apply_multimodel_support_gate(
        primary_vocals_path=melband_debleed_path,
        support_vocals_paths=[bs_vocals_path, mdx_vocals_path],
        output_vocals_path=melband_consensus_gate_path,
        support_mode="max",
        support_threshold=0.18,
        mask_power=1.30,
        inactive_floor=0.08,
        active_floor=0.46,
        vocal_core_floor=0.58,
        high_band_floor_scale=0.48,
        high_band_start_hz=5200.0,
        activity_floor_ratio=0.050,
        activity_smoothing_frames=13,
        tail_memory_sec=0.36,
        target_rms_ratio=0.98,
        max_gain=1.06,
    )
    artifacts.append(
        CandidateArtifact(
            label="07_melband_supported_by_bs_mdx_first30.flac",
            path=melband_consensus_gate_path,
            strategy="melband_primary_external_model_support_gate",
            tuning={"support_threshold": 0.18, "mask_power": 1.30},
        )
    )

    melband_drop_safe_path = output_dir / "08_melband_drop_safe_first30.flac"
    melband_drop_safe_debug = apply_multimodel_support_gate(
        primary_vocals_path=melband_debleed_path,
        support_vocals_paths=[bs_vocals_path, mdx_vocals_path],
        output_vocals_path=melband_drop_safe_path,
        support_mode="max",
        support_threshold=0.14,
        mask_power=1.08,
        inactive_floor=0.10,
        active_floor=0.58,
        vocal_core_floor=0.72,
        high_band_floor_scale=0.58,
        high_band_start_hz=5400.0,
        activity_floor_ratio=0.045,
        activity_smoothing_frames=17,
        tail_memory_sec=0.46,
        target_rms_ratio=1.02,
        max_gain=1.10,
    )
    artifacts.append(
        CandidateArtifact(
            label="08_melband_drop_safe_first30.flac",
            path=melband_drop_safe_path,
            strategy="melband_primary_drop_safe_support_gate",
            tuning={
                "support_threshold": 0.14,
                "mask_power": 1.08,
                "active_floor": 0.58,
                "vocal_core_floor": 0.72,
                "high_band_floor_scale": 0.58,
            },
        )
    )

    melband_drop_body_path = output_dir / "09_melband_drop_body_first30.flac"
    melband_drop_body_debug = apply_multimodel_support_gate(
        primary_vocals_path=melband_debleed_path,
        support_vocals_paths=[bs_vocals_path, mdx_vocals_path],
        output_vocals_path=melband_drop_body_path,
        support_mode="max",
        support_threshold=0.12,
        mask_power=0.98,
        inactive_floor=0.12,
        active_floor=0.64,
        vocal_core_floor=0.80,
        high_band_floor_scale=0.56,
        high_band_start_hz=5600.0,
        activity_floor_ratio=0.040,
        activity_smoothing_frames=19,
        tail_memory_sec=0.52,
        target_rms_ratio=1.04,
        max_gain=1.12,
    )
    artifacts.append(
        CandidateArtifact(
            label="09_melband_drop_body_first30.flac",
            path=melband_drop_body_path,
            strategy="melband_primary_drop_vocal_body_preserve_gate",
            tuning={
                "support_threshold": 0.12,
                "mask_power": 0.98,
                "active_floor": 0.64,
                "vocal_core_floor": 0.80,
                "high_band_floor_scale": 0.56,
            },
        )
    )

    melband_drop_air_path = output_dir / "10_melband_drop_air_guard_first30.flac"
    melband_drop_air_debug = apply_multimodel_support_gate(
        primary_vocals_path=melband_debleed_path,
        support_vocals_paths=[bs_vocals_path, mdx_vocals_path],
        output_vocals_path=melband_drop_air_path,
        support_mode="max",
        support_threshold=0.13,
        mask_power=1.02,
        inactive_floor=0.10,
        active_floor=0.62,
        vocal_core_floor=0.78,
        high_band_floor_scale=0.42,
        high_band_start_hz=4800.0,
        activity_floor_ratio=0.042,
        activity_smoothing_frames=17,
        tail_memory_sec=0.50,
        target_rms_ratio=1.03,
        max_gain=1.11,
    )
    artifacts.append(
        CandidateArtifact(
            label="10_melband_drop_air_guard_first30.flac",
            path=melband_drop_air_path,
            strategy="melband_primary_drop_safe_with_air_guard",
            tuning={
                "support_threshold": 0.13,
                "mask_power": 1.02,
                "active_floor": 0.62,
                "vocal_core_floor": 0.78,
                "high_band_floor_scale": 0.42,
            },
        )
    )

    focus_detail_path = output_dir / "11_10_focus_25_30_vocal_air_detail.flac"
    focus_detail_debug = build_focused_detail_recovery_candidate(
        base_vocals_path=melband_drop_air_path,
        donor_vocals_paths=[bs_vocals_path, mdx_vocals_path],
        output_vocals_path=focus_detail_path,
        focus_start_sec=25.0,
        focus_end_sec=30.0,
        focus_fade_sec=0.35,
        body_max_gain=1.58,
        body_blend=0.78,
        air_max_gain=2.35,
        air_blend=0.34,
    )
    artifacts.append(
        CandidateArtifact(
            label="11_10_focus_25_30_vocal_air_detail.flac",
            path=focus_detail_path,
            strategy="melband_drop_air_guard_focused_25_30_vocal_air_recovery",
            tuning={
                "focus_start_sec": 25.0,
                "focus_end_sec": 30.0,
                "body_max_gain": 1.58,
                "body_blend": 0.78,
                "air_max_gain": 2.35,
                "air_blend": 0.34,
            },
        )
    )

    focus_detail_plus_path = output_dir / "12_10_focus_25_30_vocal_air_plus.flac"
    focus_detail_plus_debug = build_focused_detail_recovery_candidate(
        base_vocals_path=melband_drop_air_path,
        donor_vocals_paths=[bs_vocals_path, mdx_vocals_path],
        output_vocals_path=focus_detail_plus_path,
        focus_start_sec=25.0,
        focus_end_sec=30.0,
        focus_fade_sec=0.32,
        body_max_gain=1.82,
        body_blend=0.88,
        air_max_gain=3.00,
        air_blend=0.42,
    )
    artifacts.append(
        CandidateArtifact(
            label="12_10_focus_25_30_vocal_air_plus.flac",
            path=focus_detail_plus_path,
            strategy="melband_drop_air_guard_focused_25_30_vocal_air_plus",
            tuning={
                "focus_start_sec": 25.0,
                "focus_end_sec": 30.0,
                "body_max_gain": 1.82,
                "body_blend": 0.88,
                "air_max_gain": 3.00,
                "air_blend": 0.42,
            },
        )
    )

    focus_body_plus_path = output_dir / "13_11_focus_25_30_vocal_body_plus.flac"
    focus_body_plus_debug = build_focused_detail_recovery_candidate(
        base_vocals_path=focus_detail_path,
        donor_vocals_paths=[bs_vocals_path, mdx_vocals_path],
        output_vocals_path=focus_body_plus_path,
        focus_start_sec=25.0,
        focus_end_sec=30.0,
        focus_fade_sec=0.28,
        body_max_gain=1.46,
        body_blend=0.72,
        air_max_gain=1.55,
        air_blend=0.16,
    )
    artifacts.append(
        CandidateArtifact(
            label="13_11_focus_25_30_vocal_body_plus.flac",
            path=focus_body_plus_path,
            strategy="melband_focus_detail_body_plus_air_guard",
            tuning={
                "focus_start_sec": 25.0,
                "focus_end_sec": 30.0,
                "body_max_gain": 1.46,
                "body_blend": 0.72,
                "air_max_gain": 1.55,
                "air_blend": 0.16,
            },
        )
    )

    anchor_paths = [
        ("old_bs_artifact_clean", bs_vocals_path),
        ("mdx23c", mdx_vocals_path),
        ("melband_plain", melband_vocals_path),
        ("melband_debleed", melband_debleed_path),
    ]
    metrics = [score_candidate(source_audio_path, artifact, anchor_paths) for artifact in artifacts]
    delta_metrics = calculate_delta_metrics(artifacts)
    manifest = {
        "schema_version": 1,
        "mode": "vocal_model_zoo_first30_candidates",
        "source_audio_path": str(source_audio_path),
        "raw_ref_path": str(raw_ref_path),
        "output_dir": str(output_dir),
        "listen_dir": str(listen_dir),
        "inputs": {
            "bs_vocals_path": str(bs_vocals_path),
            "mdx_vocals_path": str(mdx_vocals_path),
            "melband_vocals_path": str(melband_vocals_path),
            "melband_instrumental_path": str(melband_instrumental_path),
        },
        "debug": {
            "melband_debleed": melband_debleed_debug,
            "bs_supported_by_mdx": bs_mdx_debug,
            "bs_supported_by_mdx_melband": bs_consensus_debug,
            "melband_supported_by_bs_mdx": melband_consensus_debug,
            "melband_drop_safe": melband_drop_safe_debug,
            "melband_drop_body": melband_drop_body_debug,
            "melband_drop_air_guard": melband_drop_air_debug,
            "focus_25_30_vocal_air_detail": focus_detail_debug,
            "focus_25_30_vocal_air_plus": focus_detail_plus_debug,
            "focus_25_30_vocal_body_plus": focus_body_plus_debug,
        },
        "candidates": metrics,
        "delta_metrics": delta_metrics,
    }
    write_json(output_dir / "model_zoo_candidate_manifest.json", manifest)
    write_json(output_dir / "model_zoo_delta_metrics.json", {"deltas": delta_metrics})
    write_listen_pack(
        listen_dir=listen_dir,
        raw_ref_path=raw_ref_path,
        candidates=artifacts,
        metrics=metrics,
        delta_metrics=delta_metrics,
    )
    print(json.dumps({"manifest_path": str(output_dir / "model_zoo_candidate_manifest.json"), "listen_dir": str(listen_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
