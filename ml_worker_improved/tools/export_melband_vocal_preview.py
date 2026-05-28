from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402

import librosa
import numpy as np
import soundfile as sf
import torchaudio
import yaml


DEFAULT_REPO_PATH = Path("/app/vendor/mel_band_roformer_vocal_model")
DEFAULT_MODEL_PATH = Path("/app/vendor/mel_band_roformer_vocal_model_weights/MelBandRoformer.ckpt")
DEFAULT_CONFIG_PATH = DEFAULT_REPO_PATH / "configs" / "config_vocals_mel_band_roformer.yaml"


def _emit(message: str) -> None:
    print(f"[melband_vocal_preview] {message}", flush=True)


def _save_preview_wav(
    source_audio_path: Path,
    output_wav_path: Path,
    *,
    start_sec: float,
    end_sec: float | None,
) -> tuple[int, int]:
    waveform, sample_rate = torchaudio.load(str(source_audio_path))
    start_frame = max(0, int(round(start_sec * sample_rate)))
    end_frame = waveform.shape[1] if end_sec is None else max(start_frame, int(round(end_sec * sample_rate)))
    preview = waveform[:, start_frame:end_frame]
    torchaudio.save(str(output_wav_path), preview, sample_rate)
    return int(sample_rate), int(preview.shape[1])


def _copy_listen_file(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)


def _write_config_with_overlap(
    config_path: Path,
    target_path: Path,
    *,
    num_overlap: int,
) -> None:
    config = yaml.load(config_path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    config.setdefault("inference", {})
    if "chunk_size" not in config["inference"]:
        audio_chunk_size = config.get("audio", {}).get("chunk_size")
        if audio_chunk_size is not None:
            config["inference"]["chunk_size"] = int(audio_chunk_size)
    config["inference"]["num_overlap"] = int(num_overlap)
    target_path.write_text(yaml.dump(config, sort_keys=False), encoding="utf-8")


def _apply_light_debleed_presence(
    vocals_audio: np.ndarray,
    instrumental_audio: np.ndarray,
    *,
    sample_rate: int,
    debleed_strength: float,
    debleed_start_hz: float,
    debleed_margin: float,
    debleed_power: float,
    presence_gain: float,
    presence_start_hz: float,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    vocals = np.asarray(vocals_audio, dtype=np.float32)
    instrumental = np.asarray(instrumental_audio, dtype=np.float32)
    if vocals.ndim == 1:
        vocals = vocals[:, np.newaxis]
    if instrumental.ndim == 1:
        instrumental = instrumental[:, np.newaxis]

    if debleed_strength <= 0.0 and presence_gain <= 1.0:
        return vocals.astype(np.float32, copy=True)

    cleaned_channels: list[np.ndarray] = []
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    debleed_mask = (freqs >= float(debleed_start_hz)).astype(np.float32)[:, np.newaxis]
    presence_mask = (freqs >= float(presence_start_hz)).astype(np.float32)[:, np.newaxis]

    for channel_index in range(vocals.shape[1]):
        vocal_channel = vocals[:, channel_index]
        instrumental_channel = instrumental[:, min(channel_index, instrumental.shape[1] - 1)]
        vocal_stft = librosa.stft(vocal_channel, n_fft=n_fft, hop_length=hop_length)
        instrumental_stft = librosa.stft(instrumental_channel, n_fft=n_fft, hop_length=hop_length)

        vocal_mag = np.abs(vocal_stft).astype(np.float32)
        instrumental_mag = np.abs(instrumental_stft).astype(np.float32)
        epsilon = 1e-8

        if debleed_strength > 0.0:
            dominance = np.clip(
                (instrumental_mag - float(debleed_margin) * vocal_mag) / (instrumental_mag + epsilon),
                0.0,
                1.0,
            )
            suppress = float(debleed_strength) * np.power(dominance, float(debleed_power)).astype(np.float32, copy=False)
            vocal_stft = vocal_stft * (1.0 - suppress * debleed_mask)
            vocal_mag = np.abs(vocal_stft).astype(np.float32)

        if presence_gain > 1.0:
            vocal_ratio = vocal_mag / (vocal_mag + instrumental_mag + epsilon)
            vocal_presence_dom = np.clip((vocal_ratio - 0.58) / 0.42, 0.0, 1.0)
            restore = (float(presence_gain) - 1.0) * vocal_presence_dom * presence_mask
            vocal_stft = vocal_stft * (1.0 + restore)

        cleaned = librosa.istft(vocal_stft, hop_length=hop_length, length=vocal_channel.shape[0]).astype(np.float32)
        cleaned_channels.append(cleaned)

    return np.stack(cleaned_channels, axis=1).astype(np.float32, copy=False)


def _apply_peak_note_debleed(
    vocals_audio: np.ndarray,
    instrumental_audio: np.ndarray,
    *,
    sample_rate: int,
    peak_strength: float,
    peak_start_hz: float,
    peak_dominance_ratio: float,
    peak_local_ratio: float,
    peak_spread_bins: int,
    presence_gain: float,
    presence_start_hz: float,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    vocals = np.asarray(vocals_audio, dtype=np.float32)
    instrumental = np.asarray(instrumental_audio, dtype=np.float32)
    if vocals.ndim == 1:
        vocals = vocals[:, np.newaxis]
    if instrumental.ndim == 1:
        instrumental = instrumental[:, np.newaxis]

    if peak_strength <= 0.0 and presence_gain <= 1.0:
        return vocals.astype(np.float32, copy=True)

    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft).astype(np.float32)
    high_band = (freqs >= float(peak_start_hz)).astype(np.float32)[:, np.newaxis]
    presence_mask = (freqs >= float(presence_start_hz)).astype(np.float32)[:, np.newaxis]
    cleaned_channels: list[np.ndarray] = []
    epsilon = 1e-8

    for channel_index in range(vocals.shape[1]):
        vocal_channel = vocals[:, channel_index]
        instrumental_channel = instrumental[:, min(channel_index, instrumental.shape[1] - 1)]
        vocal_stft = librosa.stft(vocal_channel, n_fft=n_fft, hop_length=hop_length)
        instrumental_stft = librosa.stft(instrumental_channel, n_fft=n_fft, hop_length=hop_length)

        vocal_mag = np.abs(vocal_stft).astype(np.float32)
        instrumental_mag = np.abs(instrumental_stft).astype(np.float32)

        if peak_strength > 0.0:
            left_1 = np.pad(instrumental_mag[:-1], ((1, 0), (0, 0)), mode="edge")
            right_1 = np.pad(instrumental_mag[1:], ((0, 1), (0, 0)), mode="edge")
            left_2 = np.pad(instrumental_mag[:-2], ((2, 0), (0, 0)), mode="edge")
            right_2 = np.pad(instrumental_mag[2:], ((0, 2), (0, 0)), mode="edge")
            neighbor_floor = (left_1 + right_1 + left_2 + right_2) / 4.0

            is_peak = (instrumental_mag >= left_1) & (instrumental_mag >= right_1)
            local_ratio = instrumental_mag / (neighbor_floor + epsilon)
            dominance_ratio = instrumental_mag / (vocal_mag + epsilon)

            peak_mask = (
                is_peak
                & (local_ratio >= float(peak_local_ratio))
                & (dominance_ratio >= float(peak_dominance_ratio))
                & (high_band > 0.0)
            )
            dilated_mask = peak_mask.astype(np.float32, copy=False)
            for offset in range(1, max(0, int(peak_spread_bins)) + 1):
                shifted_up = np.pad(peak_mask[:-offset], ((offset, 0), (0, 0)), mode="constant")
                shifted_down = np.pad(peak_mask[offset:], ((0, offset), (0, 0)), mode="constant")
                dilated_mask = np.maximum(dilated_mask, shifted_up.astype(np.float32, copy=False))
                dilated_mask = np.maximum(dilated_mask, shifted_down.astype(np.float32, copy=False))

            dominance_excess = np.clip(
                (dominance_ratio - float(peak_dominance_ratio)) / max(0.05, float(peak_dominance_ratio)),
                0.0,
                1.0,
            )
            local_excess = np.clip(
                (local_ratio - float(peak_local_ratio)) / max(0.05, float(peak_local_ratio)),
                0.0,
                1.0,
            )
            suppress = float(peak_strength) * np.maximum(dominance_excess, local_excess) * dilated_mask
            vocal_stft = vocal_stft * (1.0 - np.clip(suppress, 0.0, 0.95))
            vocal_mag = np.abs(vocal_stft).astype(np.float32)

        if presence_gain > 1.0:
            vocal_ratio = vocal_mag / (vocal_mag + instrumental_mag + epsilon)
            vocal_presence_dom = np.clip((vocal_ratio - 0.62) / 0.38, 0.0, 1.0)
            restore = (float(presence_gain) - 1.0) * vocal_presence_dom * presence_mask
            vocal_stft = vocal_stft * (1.0 + restore)

        cleaned = librosa.istft(vocal_stft, hop_length=hop_length, length=vocal_channel.shape[0]).astype(np.float32)
        cleaned_channels.append(cleaned)

    return np.stack(cleaned_channels, axis=1).astype(np.float32, copy=False)


def _apply_windowed_note_notch(
    vocals_audio: np.ndarray,
    instrumental_audio: np.ndarray,
    *,
    sample_rate: int,
    focus_start_sec: float,
    focus_end_sec: float,
    search_pre_roll_sec: float,
    notch_strength: float,
    notch_freq_min_hz: float,
    notch_freq_max_hz: float,
    notch_peak_count: int,
    notch_bin_radius: int,
    notch_min_peak_separation_bins: int,
    notch_presence_gain: float,
    n_fft: int = 4096,
    hop_length: int = 256,
) -> np.ndarray:
    vocals = np.asarray(vocals_audio, dtype=np.float32)
    instrumental = np.asarray(instrumental_audio, dtype=np.float32)
    if vocals.ndim == 1:
        vocals = vocals[:, np.newaxis]
    if instrumental.ndim == 1:
        instrumental = instrumental[:, np.newaxis]

    if notch_strength <= 0.0:
        return vocals.astype(np.float32, copy=True)

    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft).astype(np.float32)
    freq_band = (freqs >= float(notch_freq_min_hz)) & (freqs <= float(notch_freq_max_hz))
    if not np.any(freq_band):
        return vocals.astype(np.float32, copy=True)

    start_frame = librosa.time_to_frames(float(focus_start_sec), sr=sample_rate, hop_length=hop_length)
    end_frame = librosa.time_to_frames(float(focus_end_sec), sr=sample_rate, hop_length=hop_length)
    search_start_sec = max(0.0, float(focus_start_sec) - float(search_pre_roll_sec))
    search_start_frame = librosa.time_to_frames(search_start_sec, sr=sample_rate, hop_length=hop_length)

    cleaned_channels: list[np.ndarray] = []
    epsilon = 1e-8

    for channel_index in range(vocals.shape[1]):
        vocal_channel = vocals[:, channel_index]
        instrumental_channel = instrumental[:, min(channel_index, instrumental.shape[1] - 1)]
        vocal_stft = librosa.stft(vocal_channel, n_fft=n_fft, hop_length=hop_length)
        instrumental_stft = librosa.stft(instrumental_channel, n_fft=n_fft, hop_length=hop_length)

        vocal_mag = np.abs(vocal_stft).astype(np.float32)
        instrumental_mag = np.abs(instrumental_stft).astype(np.float32)
        total_frames = vocal_mag.shape[1]
        focus_end_frame = min(total_frames, max(start_frame + 1, end_frame))
        focus_start_frame = min(max(0, start_frame), focus_end_frame - 1)
        search_start_frame_clamped = min(max(0, search_start_frame), focus_start_frame)

        inst_focus = instrumental_mag[:, search_start_frame_clamped:focus_end_frame]
        voc_focus = vocal_mag[:, search_start_frame_clamped:focus_end_frame]
        dominance = inst_focus / (voc_focus + epsilon)
        score = np.where(freq_band, inst_focus.mean(axis=1) * np.clip(dominance.mean(axis=1) - 1.0, 0.0, None), 0.0)

        ranked_bins = np.argsort(score)[::-1]
        peak_bins: list[int] = []
        min_sep = max(1, int(notch_min_peak_separation_bins))
        for peak_bin in ranked_bins:
            if score[peak_bin] <= 0.0:
                break
            if any(abs(int(peak_bin) - existing) < min_sep for existing in peak_bins):
                continue
            peak_bins.append(int(peak_bin))
            if len(peak_bins) >= max(1, int(notch_peak_count)):
                break
        notch_mask = np.zeros_like(vocal_mag, dtype=np.float32)
        for peak_bin in peak_bins:
            lo = max(0, int(peak_bin) - int(notch_bin_radius))
            hi = min(notch_mask.shape[0], int(peak_bin) + int(notch_bin_radius) + 1)
            notch_mask[lo:hi, focus_start_frame:focus_end_frame] = 1.0

        frame_inst = instrumental_mag[:, focus_start_frame:focus_end_frame]
        frame_voc = vocal_mag[:, focus_start_frame:focus_end_frame]
        frame_dom = np.clip((frame_inst / (frame_voc + epsilon)) - 1.0, 0.0, 1.0)
        notch_slice = notch_mask[:, focus_start_frame:focus_end_frame]
        vocal_stft[:, focus_start_frame:focus_end_frame] *= (1.0 - np.clip(float(notch_strength) * frame_dom * notch_slice, 0.0, 0.92))

        if notch_presence_gain > 1.0:
            updated_mag = np.abs(vocal_stft).astype(np.float32)
            focus_mag = updated_mag[:, focus_start_frame:focus_end_frame]
            focus_inst = instrumental_mag[:, focus_start_frame:focus_end_frame]
            vocal_ratio = focus_mag / (focus_mag + focus_inst + epsilon)
            restore = (float(notch_presence_gain) - 1.0) * np.clip((vocal_ratio - 0.55) / 0.45, 0.0, 1.0)
            vocal_stft[:, focus_start_frame:focus_end_frame] *= (1.0 + restore * notch_slice)

        cleaned = librosa.istft(vocal_stft, hop_length=hop_length, length=vocal_channel.shape[0]).astype(np.float32)
        cleaned_channels.append(cleaned)

    return np.stack(cleaned_channels, axis=1).astype(np.float32, copy=False)


def _apply_surgical_notches(
    vocals_audio: np.ndarray,
    *,
    sample_rate: int,
    focus_start_sec: float,
    focus_end_sec: float,
    fade_sec: float,
    centers_hz: list[float],
    half_width_hz: float,
    notch_strength: float,
    preserve_gain: float,
    n_fft: int = 4096,
    hop_length: int = 256,
) -> np.ndarray:
    vocals = np.asarray(vocals_audio, dtype=np.float32)
    if vocals.ndim == 1:
        vocals = vocals[:, np.newaxis]

    if notch_strength <= 0.0 or not centers_hz:
        return vocals.astype(np.float32, copy=True)

    frame_times = librosa.frames_to_time(np.arange(0, 1 + int(math.ceil(vocals.shape[0] / hop_length))), sr=sample_rate, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft).astype(np.float32)
    half_width = max(1.0, float(half_width_hz))
    fade = max(0.001, float(fade_sec))

    # Build a smooth time envelope that is active mainly on the offending region.
    def _time_gain(t: float) -> float:
        if t < focus_start_sec - fade or t > focus_end_sec + fade:
            return 0.0
        if focus_start_sec - fade <= t < focus_start_sec:
            return float((t - (focus_start_sec - fade)) / fade)
        if focus_start_sec <= t <= focus_end_sec:
            return 1.0
        if focus_end_sec < t <= focus_end_sec + fade:
            return float(((focus_end_sec + fade) - t) / fade)
        return 0.0

    time_env = np.asarray([_time_gain(float(t)) for t in frame_times], dtype=np.float32)

    # Build a combined frequency mask around explicit problem frequencies.
    freq_env = np.zeros_like(freqs, dtype=np.float32)
    for center in centers_hz:
        distance = np.abs(freqs - float(center))
        freq_env = np.maximum(freq_env, np.clip(1.0 - (distance / half_width), 0.0, 1.0))

    cleaned_channels: list[np.ndarray] = []
    for channel_index in range(vocals.shape[1]):
        vocal_channel = vocals[:, channel_index]
        vocal_stft = librosa.stft(vocal_channel, n_fft=n_fft, hop_length=hop_length)
        freq_mask = freq_env[:, np.newaxis]
        time_mask = time_env[np.newaxis, : vocal_stft.shape[1]]
        suppress_mask = np.clip(float(notch_strength) * freq_mask * time_mask, 0.0, 0.98)
        vocal_stft = vocal_stft * (1.0 - suppress_mask)

        if preserve_gain > 1.0:
            complement = (1.0 - freq_mask) * time_mask
            vocal_stft = vocal_stft * (1.0 + (float(preserve_gain) - 1.0) * complement)

        cleaned = librosa.istft(vocal_stft, hop_length=hop_length, length=vocal_channel.shape[0]).astype(np.float32)
        cleaned_channels.append(cleaned)

    return np.stack(cleaned_channels, axis=1).astype(np.float32, copy=False)


def _apply_windowed_band_subtract(
    vocals_audio: np.ndarray,
    instrumental_audio: np.ndarray,
    *,
    sample_rate: int,
    focus_start_sec: float,
    focus_end_sec: float,
    fade_sec: float,
    centers_hz: list[float],
    half_width_hz: float,
    subtract_strength: float,
    residual_floor: float,
    preserve_gain: float,
    n_fft: int = 4096,
    hop_length: int = 256,
) -> np.ndarray:
    vocals = np.asarray(vocals_audio, dtype=np.float32)
    instrumental = np.asarray(instrumental_audio, dtype=np.float32)
    if vocals.ndim == 1:
        vocals = vocals[:, np.newaxis]
    if instrumental.ndim == 1:
        instrumental = instrumental[:, np.newaxis]

    if subtract_strength <= 0.0 or not centers_hz:
        return vocals.astype(np.float32, copy=True)

    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft).astype(np.float32)
    half_width = max(1.0, float(half_width_hz))
    fade = max(0.001, float(fade_sec))
    epsilon = 1e-8

    def _time_gain(t: float) -> float:
        if t < focus_start_sec - fade or t > focus_end_sec + fade:
            return 0.0
        if focus_start_sec - fade <= t < focus_start_sec:
            return float((t - (focus_start_sec - fade)) / fade)
        if focus_start_sec <= t <= focus_end_sec:
            return 1.0
        if focus_end_sec < t <= focus_end_sec + fade:
            return float(((focus_end_sec + fade) - t) / fade)
        return 0.0

    frame_times = librosa.frames_to_time(
        np.arange(0, 1 + int(math.ceil(vocals.shape[0] / hop_length))),
        sr=sample_rate,
        hop_length=hop_length,
    )
    time_env = np.asarray([_time_gain(float(t)) for t in frame_times], dtype=np.float32)

    freq_env = np.zeros_like(freqs, dtype=np.float32)
    for center in centers_hz:
        distance = np.abs(freqs - float(center))
        freq_env = np.maximum(freq_env, np.clip(1.0 - (distance / half_width), 0.0, 1.0))

    cleaned_channels: list[np.ndarray] = []
    for channel_index in range(vocals.shape[1]):
        vocal_channel = vocals[:, channel_index]
        instrumental_channel = instrumental[:, min(channel_index, instrumental.shape[1] - 1)]
        vocal_stft = librosa.stft(vocal_channel, n_fft=n_fft, hop_length=hop_length)
        instrumental_stft = librosa.stft(instrumental_channel, n_fft=n_fft, hop_length=hop_length)

        vocal_mag = np.abs(vocal_stft).astype(np.float32)
        instrumental_mag = np.abs(instrumental_stft).astype(np.float32)
        freq_mask = freq_env[:, np.newaxis]
        time_mask = time_env[np.newaxis, : vocal_stft.shape[1]]
        dominance = instrumental_mag / (vocal_mag + epsilon)
        suppress = np.clip(float(subtract_strength) * dominance * freq_mask * time_mask, 0.0, 1.0)
        final_mask = np.clip(1.0 - suppress, float(residual_floor), 1.0)
        vocal_stft = vocal_stft * final_mask

        if preserve_gain > 1.0:
            complement = (1.0 - freq_mask) * time_mask
            vocal_stft = vocal_stft * (1.0 + (float(preserve_gain) - 1.0) * complement)

        cleaned = librosa.istft(vocal_stft, hop_length=hop_length, length=vocal_channel.shape[0]).astype(np.float32)
        cleaned_channels.append(cleaned)

    return np.stack(cleaned_channels, axis=1).astype(np.float32, copy=False)


save_audio_window_as_wav = _save_preview_wav
copy_listen_file = _copy_listen_file
write_config_with_overlap = _write_config_with_overlap
apply_light_debleed_presence = _apply_light_debleed_presence
apply_peak_note_debleed = _apply_peak_note_debleed
apply_windowed_note_notch = _apply_windowed_note_notch
apply_surgical_notches = _apply_surgical_notches
apply_windowed_band_subtract = _apply_windowed_band_subtract


def main() -> None:
    require_docker_runtime(
        tool_name="export_melband_vocal_preview.py",
        entrypoint_path="/app/tools/export_melband_vocal_preview.py",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("source_audio_path")
    parser.add_argument("output_dir")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, default=40.0)
    parser.add_argument("--repo-path", default=str(DEFAULT_REPO_PATH))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--listen-dir")
    parser.add_argument("--listen-label", default="01_melband_roformer_vocals.flac")
    parser.add_argument("--num-overlap", type=int, default=2)
    parser.add_argument("--light-debleed-strength", type=float, default=0.0)
    parser.add_argument("--light-debleed-start-hz", type=float, default=2600.0)
    parser.add_argument("--light-debleed-margin", type=float, default=1.08)
    parser.add_argument("--light-debleed-power", type=float, default=1.4)
    parser.add_argument("--presence-gain", type=float, default=1.0)
    parser.add_argument("--presence-start-hz", type=float, default=1800.0)
    parser.add_argument("--note-peak-strength", type=float, default=0.0)
    parser.add_argument("--note-peak-start-hz", type=float, default=2400.0)
    parser.add_argument("--note-peak-dominance-ratio", type=float, default=1.45)
    parser.add_argument("--note-peak-local-ratio", type=float, default=1.7)
    parser.add_argument("--note-peak-spread-bins", type=int, default=1)
    parser.add_argument("--note-peak-presence-gain", type=float, default=1.0)
    parser.add_argument("--note-peak-presence-start-hz", type=float, default=1500.0)
    parser.add_argument("--window-notch-strength", type=float, default=0.0)
    parser.add_argument("--window-notch-focus-start-sec", type=float, default=8.0)
    parser.add_argument("--window-notch-focus-end-sec", type=float, default=10.0)
    parser.add_argument("--window-notch-search-pre-roll-sec", type=float, default=0.35)
    parser.add_argument("--window-notch-freq-min-hz", type=float, default=1700.0)
    parser.add_argument("--window-notch-freq-max-hz", type=float, default=3600.0)
    parser.add_argument("--window-notch-peak-count", type=int, default=3)
    parser.add_argument("--window-notch-bin-radius", type=int, default=2)
    parser.add_argument("--window-notch-min-peak-separation-bins", type=int, default=10)
    parser.add_argument("--window-notch-presence-gain", type=float, default=1.02)
    parser.add_argument("--surgical-notch-strength", type=float, default=0.0)
    parser.add_argument("--surgical-notch-focus-start-sec", type=float, default=8.0)
    parser.add_argument("--surgical-notch-focus-end-sec", type=float, default=10.0)
    parser.add_argument("--surgical-notch-fade-sec", type=float, default=0.25)
    parser.add_argument("--surgical-notch-centers-hz", default="")
    parser.add_argument("--surgical-notch-half-width-hz", type=float, default=65.0)
    parser.add_argument("--surgical-notch-preserve-gain", type=float, default=1.02)
    parser.add_argument("--window-band-subtract-strength", type=float, default=0.0)
    parser.add_argument("--window-band-subtract-focus-start-sec", type=float, default=8.0)
    parser.add_argument("--window-band-subtract-focus-end-sec", type=float, default=10.0)
    parser.add_argument("--window-band-subtract-fade-sec", type=float, default=0.25)
    parser.add_argument("--window-band-subtract-centers-hz", default="")
    parser.add_argument("--window-band-subtract-half-width-hz", type=float, default=130.0)
    parser.add_argument("--window-band-subtract-residual-floor", type=float, default=0.16)
    parser.add_argument("--window-band-subtract-preserve-gain", type=float, default=1.02)
    args = parser.parse_args()

    source_audio_path = Path(args.source_audio_path)
    output_dir = Path(args.output_dir)
    repo_path = Path(args.repo_path)
    model_path = Path(args.model_path)
    config_path = Path(args.config_path)
    listen_dir = Path(args.listen_dir) if args.listen_dir else None

    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_audio_path.exists():
        raise FileNotFoundError(f"Source audio is missing: {source_audio_path}")
    if not repo_path.exists():
        raise FileNotFoundError(f"MelBand repo is missing: {repo_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"MelBand checkpoint is missing: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"MelBand config is missing: {config_path}")

    start_wall = time.time()
    with tempfile.TemporaryDirectory(prefix="melband_preview_input_") as input_dir_str, tempfile.TemporaryDirectory(
        prefix="melband_preview_output_"
    ) as model_output_dir_str, tempfile.TemporaryDirectory(prefix="melband_preview_config_") as config_dir_str:
        input_dir = Path(input_dir_str)
        model_output_dir = Path(model_output_dir_str)
        config_dir = Path(config_dir_str)
        preview_input_path = input_dir / "preview.wav"
        runtime_config_path = config_dir / "config_runtime.yaml"

        _emit(
            f"ENTER source_audio={source_audio_path} output_dir={output_dir} start_sec={args.start_sec} end_sec={args.end_sec}"
        )
        sample_rate, frames = _save_preview_wav(
            source_audio_path,
            preview_input_path,
            start_sec=float(args.start_sec),
            end_sec=float(args.end_sec) if args.end_sec is not None else None,
        )
        _emit(f"prepared_preview_wav sample_rate={sample_rate} frames={frames}")
        _write_config_with_overlap(config_path, runtime_config_path, num_overlap=int(args.num_overlap))

        command = [
            "python",
            "inference.py",
            "--model_type",
            "mel_band_roformer",
            "--config_path",
            str(runtime_config_path),
            "--model_path",
            str(model_path),
            "--input_folder",
            str(input_dir),
            "--store_dir",
            str(model_output_dir),
            "--device_ids",
            str(args.device_id),
        ]
        completed = subprocess.run(
            command,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=True,
        )
        if completed.stdout.strip():
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
        if completed.stderr.strip():
            print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", flush=True)

        vocals_wav = model_output_dir / "preview_vocals.wav"
        instrumental_wav = model_output_dir / "preview_instrumental.wav"
        if not vocals_wav.exists():
            raise FileNotFoundError(f"MelBand vocals output is missing: {vocals_wav}")
        if not instrumental_wav.exists():
            raise FileNotFoundError(f"MelBand instrumental output is missing: {instrumental_wav}")

        vocals_audio, vocals_sr = sf.read(str(vocals_wav), always_2d=True)
        instrumental_audio, instrumental_sr = sf.read(str(instrumental_wav), always_2d=True)
        raw_audio, raw_sr = sf.read(str(preview_input_path), always_2d=True)
        tuned_vocals_audio = _apply_light_debleed_presence(
            vocals_audio,
            instrumental_audio,
            sample_rate=int(vocals_sr),
            debleed_strength=float(args.light_debleed_strength),
            debleed_start_hz=float(args.light_debleed_start_hz),
            debleed_margin=float(args.light_debleed_margin),
            debleed_power=float(args.light_debleed_power),
            presence_gain=float(args.presence_gain),
            presence_start_hz=float(args.presence_start_hz),
        )
        tuned_vocals_audio = _apply_peak_note_debleed(
            tuned_vocals_audio,
            instrumental_audio,
            sample_rate=int(vocals_sr),
            peak_strength=float(args.note_peak_strength),
            peak_start_hz=float(args.note_peak_start_hz),
            peak_dominance_ratio=float(args.note_peak_dominance_ratio),
            peak_local_ratio=float(args.note_peak_local_ratio),
            peak_spread_bins=int(args.note_peak_spread_bins),
            presence_gain=float(args.note_peak_presence_gain),
            presence_start_hz=float(args.note_peak_presence_start_hz),
        )
        tuned_vocals_audio = _apply_windowed_note_notch(
            tuned_vocals_audio,
            instrumental_audio,
            sample_rate=int(vocals_sr),
            focus_start_sec=float(args.window_notch_focus_start_sec),
            focus_end_sec=float(args.window_notch_focus_end_sec),
            search_pre_roll_sec=float(args.window_notch_search_pre_roll_sec),
            notch_strength=float(args.window_notch_strength),
            notch_freq_min_hz=float(args.window_notch_freq_min_hz),
            notch_freq_max_hz=float(args.window_notch_freq_max_hz),
            notch_peak_count=int(args.window_notch_peak_count),
            notch_bin_radius=int(args.window_notch_bin_radius),
            notch_min_peak_separation_bins=int(args.window_notch_min_peak_separation_bins),
            notch_presence_gain=float(args.window_notch_presence_gain),
        )
        surgical_centers = [float(item.strip()) for item in str(args.surgical_notch_centers_hz).split(",") if item.strip()]
        tuned_vocals_audio = _apply_surgical_notches(
            tuned_vocals_audio,
            sample_rate=int(vocals_sr),
            focus_start_sec=float(args.surgical_notch_focus_start_sec),
            focus_end_sec=float(args.surgical_notch_focus_end_sec),
            fade_sec=float(args.surgical_notch_fade_sec),
            centers_hz=surgical_centers,
            half_width_hz=float(args.surgical_notch_half_width_hz),
            notch_strength=float(args.surgical_notch_strength),
            preserve_gain=float(args.surgical_notch_preserve_gain),
        )
        window_band_subtract_centers = [
            float(item.strip()) for item in str(args.window_band_subtract_centers_hz).split(",") if item.strip()
        ]
        tuned_vocals_audio = _apply_windowed_band_subtract(
            tuned_vocals_audio,
            instrumental_audio,
            sample_rate=int(vocals_sr),
            focus_start_sec=float(args.window_band_subtract_focus_start_sec),
            focus_end_sec=float(args.window_band_subtract_focus_end_sec),
            fade_sec=float(args.window_band_subtract_fade_sec),
            centers_hz=window_band_subtract_centers,
            half_width_hz=float(args.window_band_subtract_half_width_hz),
            subtract_strength=float(args.window_band_subtract_strength),
            residual_floor=float(args.window_band_subtract_residual_floor),
            preserve_gain=float(args.window_band_subtract_preserve_gain),
        )

        vocals_flac_path = output_dir / "vocals.flac"
        instrumental_flac_path = output_dir / "instrumental.flac"
        raw_ref_flac_path = output_dir / "raw_ref.flac"
        sf.write(str(vocals_flac_path), tuned_vocals_audio, vocals_sr, subtype="PCM_24")
        sf.write(str(instrumental_flac_path), instrumental_audio, instrumental_sr, subtype="PCM_24")
        sf.write(str(raw_ref_flac_path), raw_audio, raw_sr, subtype="PCM_24")

        manifest = {
            "source_audio_path": str(source_audio_path),
            "preview_window_sec": {
                "start": float(args.start_sec),
                "end": None if args.end_sec is None else float(args.end_sec),
            },
            "model": {
                "name": "mel_band_roformer_vocal_model",
                "repo_path": str(repo_path),
                "config_path": str(runtime_config_path),
                "config_template_path": str(config_path),
                "model_path": str(model_path),
                "device_id": int(args.device_id),
                "num_overlap": int(args.num_overlap),
            },
            "artifacts": {
                "raw_ref_flac": str(raw_ref_flac_path),
                "vocals_flac": str(vocals_flac_path),
                "instrumental_flac": str(instrumental_flac_path),
            },
            "postprocess": {
                "name": (
                    "window_band_subtract_v1"
                    if float(args.window_band_subtract_strength) > 0.0
                    else
                    "surgical_notch_v1"
                    if float(args.surgical_notch_strength) > 0.0
                    else
                    "windowed_note_notch_v1"
                    if float(args.window_notch_strength) > 0.0
                    else
                    "peak_note_melody_bleed_suppressor_v1"
                    if float(args.note_peak_strength) > 0.0
                    else "light_high_band_debleed_presence_v1"
                    if float(args.light_debleed_strength) > 0.0 or float(args.presence_gain) > 1.0
                    else None
                ),
                "light_debleed_strength": float(args.light_debleed_strength),
                "light_debleed_start_hz": float(args.light_debleed_start_hz),
                "light_debleed_margin": float(args.light_debleed_margin),
                "light_debleed_power": float(args.light_debleed_power),
                "presence_gain": float(args.presence_gain),
                "presence_start_hz": float(args.presence_start_hz),
                "note_peak_strength": float(args.note_peak_strength),
                "note_peak_start_hz": float(args.note_peak_start_hz),
                "note_peak_dominance_ratio": float(args.note_peak_dominance_ratio),
                "note_peak_local_ratio": float(args.note_peak_local_ratio),
                "note_peak_spread_bins": int(args.note_peak_spread_bins),
                "note_peak_presence_gain": float(args.note_peak_presence_gain),
                "note_peak_presence_start_hz": float(args.note_peak_presence_start_hz),
                "window_notch_strength": float(args.window_notch_strength),
                "window_notch_focus_start_sec": float(args.window_notch_focus_start_sec),
                "window_notch_focus_end_sec": float(args.window_notch_focus_end_sec),
                "window_notch_search_pre_roll_sec": float(args.window_notch_search_pre_roll_sec),
                "window_notch_freq_min_hz": float(args.window_notch_freq_min_hz),
                "window_notch_freq_max_hz": float(args.window_notch_freq_max_hz),
                "window_notch_peak_count": int(args.window_notch_peak_count),
                "window_notch_bin_radius": int(args.window_notch_bin_radius),
                "window_notch_min_peak_separation_bins": int(args.window_notch_min_peak_separation_bins),
                "window_notch_presence_gain": float(args.window_notch_presence_gain),
                "surgical_notch_strength": float(args.surgical_notch_strength),
                "surgical_notch_focus_start_sec": float(args.surgical_notch_focus_start_sec),
                "surgical_notch_focus_end_sec": float(args.surgical_notch_focus_end_sec),
                "surgical_notch_fade_sec": float(args.surgical_notch_fade_sec),
                "surgical_notch_centers_hz": surgical_centers,
                "surgical_notch_half_width_hz": float(args.surgical_notch_half_width_hz),
                "surgical_notch_preserve_gain": float(args.surgical_notch_preserve_gain),
                "window_band_subtract_strength": float(args.window_band_subtract_strength),
                "window_band_subtract_focus_start_sec": float(args.window_band_subtract_focus_start_sec),
                "window_band_subtract_focus_end_sec": float(args.window_band_subtract_focus_end_sec),
                "window_band_subtract_fade_sec": float(args.window_band_subtract_fade_sec),
                "window_band_subtract_centers_hz": window_band_subtract_centers,
                "window_band_subtract_half_width_hz": float(args.window_band_subtract_half_width_hz),
                "window_band_subtract_residual_floor": float(args.window_band_subtract_residual_floor),
                "window_band_subtract_preserve_gain": float(args.window_band_subtract_preserve_gain),
            },
            "audio": {
                "sample_rate": int(raw_sr),
                "frames": int(raw_audio.shape[0]),
                "duration_sec": round(float(raw_audio.shape[0]) / float(raw_sr), 6) if raw_sr else 0.0,
            },
            "elapsed_sec": round(time.time() - start_wall, 3),
        }
        manifest_path = output_dir / "stem_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        if listen_dir is not None:
            listen_dir.mkdir(parents=True, exist_ok=True)
            _copy_listen_file(raw_ref_flac_path, listen_dir / "00_raw_ref.flac")
            _copy_listen_file(vocals_flac_path, listen_dir / args.listen_label)
            (listen_dir / "00_README.txt").write_text(
                "Listen only here.\n00 = raw baseline vocals preview.\nOther files = Mel-Band RoFormer variants.\n",
                encoding="utf-8",
            )
            candidate_map_path = listen_dir / "candidate_map.json"
            candidate_map: dict[str, str] = {}
            if candidate_map_path.exists():
                candidate_map = json.loads(candidate_map_path.read_text(encoding="utf-8"))
            candidate_map["00_raw_ref.flac"] = str(raw_ref_flac_path)
            candidate_map[args.listen_label] = str(vocals_flac_path)
            candidate_map_path.write_text(json.dumps(candidate_map, indent=2, ensure_ascii=False), encoding="utf-8")

        _emit(f"EXIT output_dir={output_dir} elapsed_sec={manifest['elapsed_sec']}")
        print(json.dumps({"output_dir": str(output_dir), "manifest_path": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
