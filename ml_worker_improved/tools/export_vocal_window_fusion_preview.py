from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402

import librosa
import numpy as np
import soundfile as sf


def _emit(message: str) -> None:
    print(f"[vocal_window_fusion_preview] {message}", flush=True)


def _copy_listen_file(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)


def _safe_rms(audio: np.ndarray) -> float:
    data = np.asarray(audio, dtype=np.float32)
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data, dtype=np.float32))))


def _build_time_env(
    frame_count: int,
    *,
    sample_rate: int,
    hop_length: int,
    focus_start_sec: float,
    focus_end_sec: float,
    fade_sec: float,
) -> np.ndarray:
    fade = max(0.001, float(fade_sec))
    frame_times = librosa.frames_to_time(np.arange(frame_count), sr=sample_rate, hop_length=hop_length)

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

    return np.asarray([_time_gain(float(t)) for t in frame_times], dtype=np.float32)


def _build_freq_env(
    freqs: np.ndarray,
    *,
    centers_hz: list[float],
    half_width_hz: float,
) -> np.ndarray:
    half_width = max(1.0, float(half_width_hz))
    freq_env = np.zeros_like(freqs, dtype=np.float32)
    for center in centers_hz:
        distance = np.abs(freqs - float(center))
        freq_env = np.maximum(freq_env, np.clip(1.0 - (distance / half_width), 0.0, 1.0))
    return freq_env.astype(np.float32, copy=False)


def _apply_windowed_secondary_clamp(
    primary_audio: np.ndarray,
    secondary_audio: np.ndarray,
    *,
    sample_rate: int,
    focus_start_sec: float,
    focus_end_sec: float,
    fade_sec: float,
    centers_hz: list[float],
    half_width_hz: float,
    secondary_ceiling_scale: float,
    residual_floor: float,
    n_fft: int = 4096,
    hop_length: int = 256,
) -> np.ndarray:
    primary = np.asarray(primary_audio, dtype=np.float32)
    secondary = np.asarray(secondary_audio, dtype=np.float32)
    if primary.ndim == 1:
        primary = primary[:, np.newaxis]
    if secondary.ndim == 1:
        secondary = secondary[:, np.newaxis]
    if primary.shape != secondary.shape:
        raise ValueError(f"Shape mismatch: primary={primary.shape} secondary={secondary.shape}")
    if not centers_hz:
        return primary.astype(np.float32, copy=True)

    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft).astype(np.float32)
    freq_env = _build_freq_env(freqs, centers_hz=centers_hz, half_width_hz=float(half_width_hz))[:, np.newaxis]

    cleaned_channels: list[np.ndarray] = []
    epsilon = 1e-8
    for channel_index in range(primary.shape[1]):
        primary_channel = primary[:, channel_index]
        secondary_channel = secondary[:, min(channel_index, secondary.shape[1] - 1)]

        primary_stft = librosa.stft(primary_channel, n_fft=n_fft, hop_length=hop_length)
        secondary_stft = librosa.stft(secondary_channel, n_fft=n_fft, hop_length=hop_length)
        primary_mag = np.abs(primary_stft).astype(np.float32)
        secondary_mag = np.abs(secondary_stft).astype(np.float32)
        time_env = _build_time_env(
            primary_stft.shape[1],
            sample_rate=sample_rate,
            hop_length=hop_length,
            focus_start_sec=float(focus_start_sec),
            focus_end_sec=float(focus_end_sec),
            fade_sec=float(fade_sec),
        )[np.newaxis, :]

        target_mag = np.minimum(primary_mag, secondary_mag * float(secondary_ceiling_scale)).astype(np.float32)
        ratio = target_mag / (primary_mag + epsilon)
        ratio = np.clip(ratio, float(residual_floor), 1.0).astype(np.float32)
        final_mask = 1.0 - (1.0 - ratio) * (freq_env * time_env)
        fused_stft = primary_stft * final_mask
        fused = librosa.istft(fused_stft, hop_length=hop_length, length=primary_channel.shape[0]).astype(np.float32)
        cleaned_channels.append(fused)

    return np.stack(cleaned_channels, axis=1).astype(np.float32, copy=False)


def _apply_windowed_secondary_replace(
    primary_audio: np.ndarray,
    secondary_audio: np.ndarray,
    *,
    sample_rate: int,
    focus_start_sec: float,
    focus_end_sec: float,
    fade_sec: float,
    centers_hz: list[float],
    half_width_hz: float,
    replace_mix: float,
    n_fft: int = 4096,
    hop_length: int = 256,
) -> np.ndarray:
    primary = np.asarray(primary_audio, dtype=np.float32)
    secondary = np.asarray(secondary_audio, dtype=np.float32)
    if primary.ndim == 1:
        primary = primary[:, np.newaxis]
    if secondary.ndim == 1:
        secondary = secondary[:, np.newaxis]
    if primary.shape != secondary.shape:
        raise ValueError(f"Shape mismatch: primary={primary.shape} secondary={secondary.shape}")
    if not centers_hz or replace_mix <= 0.0:
        return primary.astype(np.float32, copy=True)

    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft).astype(np.float32)
    freq_env = _build_freq_env(freqs, centers_hz=centers_hz, half_width_hz=float(half_width_hz))[:, np.newaxis]

    cleaned_channels: list[np.ndarray] = []
    for channel_index in range(primary.shape[1]):
        primary_channel = primary[:, channel_index]
        secondary_channel = secondary[:, min(channel_index, secondary.shape[1] - 1)]

        primary_stft = librosa.stft(primary_channel, n_fft=n_fft, hop_length=hop_length)
        secondary_stft = librosa.stft(secondary_channel, n_fft=n_fft, hop_length=hop_length)
        time_env = _build_time_env(
            primary_stft.shape[1],
            sample_rate=sample_rate,
            hop_length=hop_length,
            focus_start_sec=float(focus_start_sec),
            focus_end_sec=float(focus_end_sec),
            fade_sec=float(fade_sec),
        )[np.newaxis, :]

        replace_mask = np.clip(float(replace_mix) * freq_env * time_env, 0.0, 1.0).astype(np.float32)
        fused_stft = primary_stft * (1.0 - replace_mask) + secondary_stft * replace_mask
        fused = librosa.istft(fused_stft, hop_length=hop_length, length=primary_channel.shape[0]).astype(np.float32)
        cleaned_channels.append(fused)

    return np.stack(cleaned_channels, axis=1).astype(np.float32, copy=False)


def main() -> None:
    require_docker_runtime(
        tool_name="export_vocal_window_fusion_preview.py",
        entrypoint_path="/app/tools/export_vocal_window_fusion_preview.py",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("primary_vocals_path")
    parser.add_argument("secondary_vocals_path")
    parser.add_argument("raw_ref_path")
    parser.add_argument("output_dir")
    parser.add_argument("--listen-dir")
    parser.add_argument("--listen-label", default="02_fusion_window_clamp.flac")
    parser.add_argument("--focus-start-sec", type=float, default=7.85)
    parser.add_argument("--focus-end-sec", type=float, default=10.15)
    parser.add_argument("--fade-sec", type=float, default=0.18)
    parser.add_argument("--centers-hz", default="1852,3150")
    parser.add_argument("--half-width-hz", type=float, default=180.0)
    parser.add_argument("--secondary-ceiling-scale", type=float, default=1.0)
    parser.add_argument("--residual-floor", type=float, default=0.08)
    parser.add_argument("--replace-mix", type=float, default=0.0)
    args = parser.parse_args()

    primary_vocals_path = Path(args.primary_vocals_path)
    secondary_vocals_path = Path(args.secondary_vocals_path)
    raw_ref_path = Path(args.raw_ref_path)
    output_dir = Path(args.output_dir)
    listen_dir = Path(args.listen_dir) if args.listen_dir else None

    start_wall = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_audio, primary_sr = sf.read(str(primary_vocals_path), always_2d=True)
    secondary_audio, secondary_sr = sf.read(str(secondary_vocals_path), always_2d=True)
    raw_audio, raw_sr = sf.read(str(raw_ref_path), always_2d=True)
    if primary_sr != secondary_sr:
        raise ValueError(f"Sample rate mismatch: primary={primary_sr} secondary={secondary_sr}")
    if primary_audio.shape != secondary_audio.shape:
        raise ValueError(f"Audio shape mismatch: primary={primary_audio.shape} secondary={secondary_audio.shape}")

    centers_hz = [float(item.strip()) for item in str(args.centers_hz).split(",") if item.strip()]
    fused_audio = _apply_windowed_secondary_clamp(
        primary_audio,
        secondary_audio,
        sample_rate=int(primary_sr),
        focus_start_sec=float(args.focus_start_sec),
        focus_end_sec=float(args.focus_end_sec),
        fade_sec=float(args.fade_sec),
        centers_hz=centers_hz,
        half_width_hz=float(args.half_width_hz),
        secondary_ceiling_scale=float(args.secondary_ceiling_scale),
        residual_floor=float(args.residual_floor),
    )
    fused_audio = _apply_windowed_secondary_replace(
        fused_audio,
        secondary_audio,
        sample_rate=int(primary_sr),
        focus_start_sec=float(args.focus_start_sec),
        focus_end_sec=float(args.focus_end_sec),
        fade_sec=float(args.fade_sec),
        centers_hz=centers_hz,
        half_width_hz=float(args.half_width_hz),
        replace_mix=float(args.replace_mix),
    )

    vocals_flac_path = output_dir / "vocals.flac"
    raw_ref_flac_path = output_dir / "raw_ref.flac"
    secondary_copy_path = output_dir / "secondary_vocals_ref.flac"
    sf.write(str(vocals_flac_path), fused_audio, primary_sr, subtype="PCM_24")
    sf.write(str(raw_ref_flac_path), raw_audio, raw_sr, subtype="PCM_24")
    sf.write(str(secondary_copy_path), secondary_audio, secondary_sr, subtype="PCM_24")

    manifest = {
        "primary_vocals_path": str(primary_vocals_path),
        "secondary_vocals_path": str(secondary_vocals_path),
        "raw_ref_path": str(raw_ref_path),
        "postprocess": {
            "name": (
                "windowed_secondary_replace_fusion_v1"
                if float(args.replace_mix) > 0.0
                else "windowed_secondary_clamp_fusion_v1"
            ),
            "focus_start_sec": float(args.focus_start_sec),
            "focus_end_sec": float(args.focus_end_sec),
            "fade_sec": float(args.fade_sec),
            "centers_hz": centers_hz,
            "half_width_hz": float(args.half_width_hz),
            "secondary_ceiling_scale": float(args.secondary_ceiling_scale),
            "residual_floor": float(args.residual_floor),
            "replace_mix": float(args.replace_mix),
        },
        "audio": {
            "sample_rate": int(primary_sr),
            "frames": int(primary_audio.shape[0]),
            "duration_sec": round(float(primary_audio.shape[0]) / float(primary_sr), 6) if primary_sr else 0.0,
            "primary_rms": round(_safe_rms(primary_audio), 6),
            "secondary_rms": round(_safe_rms(secondary_audio), 6),
            "fused_rms": round(_safe_rms(fused_audio), 6),
        },
        "artifacts": {
            "vocals_flac": str(vocals_flac_path),
            "raw_ref_flac": str(raw_ref_flac_path),
            "secondary_vocals_ref_flac": str(secondary_copy_path),
        },
        "elapsed_sec": round(time.time() - start_wall, 3),
    }
    manifest_path = output_dir / "stem_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if listen_dir is not None:
        listen_dir.mkdir(parents=True, exist_ok=True)
        _copy_listen_file(raw_ref_flac_path, listen_dir / "00_raw_ref.flac")
        _copy_listen_file(primary_vocals_path, listen_dir / "01_primary_melband_plain.flac")
        _copy_listen_file(vocals_flac_path, listen_dir / args.listen_label)
        (listen_dir / "00_README.txt").write_text(
            "Listen only here.\n00 = raw baseline vocals preview.\n01 = primary Mel-Band baseline.\n02 = fusion candidate.\n",
            encoding="utf-8",
        )
        candidate_map = {
            "00_raw_ref.flac": str(raw_ref_flac_path),
            "01_primary_melband_plain.flac": str(primary_vocals_path),
            args.listen_label: str(vocals_flac_path),
        }
        (listen_dir / "candidate_map.json").write_text(json.dumps(candidate_map, indent=2, ensure_ascii=False), encoding="utf-8")

    _emit(f"EXIT output_dir={output_dir} elapsed_sec={manifest['elapsed_sec']}")
    print(json.dumps({"output_dir": str(output_dir), "manifest_path": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
