from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402

def _emit(message: str) -> None:
    print(f"[windowed_drum_body_restore] {message}", flush=True)


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _copy_file(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)


def _write_flac_copy(source_audio_path: Path, target_flac_path: Path) -> None:
    if source_audio_path.suffix.lower() == ".flac":
        shutil.copyfile(source_audio_path, target_flac_path)
        return
    audio, sample_rate = sf.read(str(source_audio_path), always_2d=True)
    sf.write(str(target_flac_path), audio, sample_rate, subtype="PCM_24")


def _safe_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    return math.sqrt(float(np.mean(np.square(mono, dtype=np.float32))))


def _build_freq_weight(freqs: np.ndarray, *, low_end_hz: float, body_end_hz: float, low_blend: float, body_blend: float) -> np.ndarray:
    weight = np.zeros_like(freqs, dtype=np.float32)
    low_idx = freqs <= float(low_end_hz)
    weight[low_idx] = float(low_blend)
    body_idx = (freqs > float(low_end_hz)) & (freqs <= float(body_end_hz))
    if np.any(body_idx):
        span = max(float(body_end_hz - low_end_hz), 1.0)
        pos = (freqs[body_idx] - float(low_end_hz)) / span
        weight[body_idx] = np.clip(float(body_blend) * (1.0 - pos), 0.0, 1.0)
    return weight.astype(np.float32)


def _build_time_mask(length: int, sample_rate: int, *, start_sec: float, end_sec: float, xfade_sec: float) -> np.ndarray:
    positions = np.arange(length, dtype=np.float32) / float(sample_rate)
    mask = np.zeros(length, dtype=np.float32)
    fade_in_start = max(0.0, float(start_sec) - float(xfade_sec))
    fade_out_end = float(end_sec) + float(xfade_sec)
    for idx, t_sec in enumerate(positions):
        if t_sec < fade_in_start or t_sec > fade_out_end:
            continue
        if t_sec < float(start_sec):
            mask[idx] = (t_sec - fade_in_start) / max(float(xfade_sec), 1e-6)
        elif t_sec <= float(end_sec):
            mask[idx] = 1.0
        else:
            mask[idx] = 1.0 - ((t_sec - float(end_sec)) / max(float(xfade_sec), 1e-6))
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def _clip(audio: np.ndarray, max_peak: float = 0.9999) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= max_peak:
        return audio.astype(np.float32, copy=False)
    return (audio / peak * float(max_peak)).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore low/body drum energy from a base stem only inside a target window.")
    parser.add_argument("base_drums_path")
    parser.add_argument("cleaned_drums_path")
    parser.add_argument("output_dir")
    parser.add_argument("--raw-ref-path")
    parser.add_argument("--start-sec", type=float, required=True)
    parser.add_argument("--end-sec", type=float, required=True)
    parser.add_argument("--xfade-sec", type=float, default=0.2)
    parser.add_argument("--low-end-hz", type=float, default=220.0)
    parser.add_argument("--body-end-hz", type=float, default=900.0)
    parser.add_argument("--low-blend", type=float, default=0.85)
    parser.add_argument("--body-blend", type=float, default=0.35)
    parser.add_argument("--listen-dir")
    parser.add_argument("--listen-label", default="03_drums_windowed_body_restore.flac")
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="apply_windowed_drum_body_restore.py",
        entrypoint_path="/app/tools/apply_windowed_drum_body_restore.py",
    )
    args = parse_args()
    base_path = Path(args.base_drums_path).resolve()
    cleaned_path = Path(args.cleaned_drums_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    raw_ref_path = Path(args.raw_ref_path).resolve() if args.raw_ref_path else None
    listen_dir = Path(args.listen_dir).resolve() if args.listen_dir else None

    base_audio, sample_rate = _read_audio(base_path)
    cleaned_audio, cleaned_sample_rate = _read_audio(cleaned_path)
    if sample_rate != cleaned_sample_rate:
        raise ValueError(f"Sample rate mismatch: {sample_rate} vs {cleaned_sample_rate}")
    if base_audio.shape != cleaned_audio.shape:
        raise ValueError(f"Shape mismatch: {base_audio.shape} vs {cleaned_audio.shape}")

    length = int(base_audio.shape[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    time_mask = _build_time_mask(
        length,
        sample_rate,
        start_sec=float(args.start_sec),
        end_sec=float(args.end_sec),
        xfade_sec=float(args.xfade_sec),
    )

    restored = cleaned_audio.copy()
    for channel_index in range(base_audio.shape[1]):
        base_channel = base_audio[:, channel_index]
        cleaned_channel = cleaned_audio[:, channel_index]
        base_stft = librosa.stft(base_channel, n_fft=2048, hop_length=512)
        cleaned_stft = librosa.stft(cleaned_channel, n_fft=2048, hop_length=512)
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
        freq_weight = _build_freq_weight(
            freqs,
            low_end_hz=float(args.low_end_hz),
            body_end_hz=float(args.body_end_hz),
            low_blend=float(args.low_blend),
            body_blend=float(args.body_blend),
        )[:, np.newaxis]

        frame_times = librosa.frames_to_time(np.arange(cleaned_stft.shape[1]), sr=sample_rate, hop_length=512, n_fft=2048)
        frame_mask = np.interp(
            frame_times.astype(np.float32),
            np.arange(length, dtype=np.float32) / float(sample_rate),
            time_mask,
            left=0.0,
            right=0.0,
        ).astype(np.float32)[np.newaxis, :]

        blend = np.clip(freq_weight * frame_mask, 0.0, 1.0)
        mixed_stft = cleaned_stft * (1.0 - blend) + base_stft * blend
        restored[:, channel_index] = librosa.istft(mixed_stft, hop_length=512, length=length).astype(np.float32)

    restored = _clip(restored)
    output_path = output_dir / "drums_windowed_body_restore.flac"
    sf.write(str(output_path), restored, sample_rate, subtype="PCM_24")

    start_frame = int(float(args.start_sec) * sample_rate)
    end_frame = int(float(args.end_sec) * sample_rate)
    base_win = base_audio[start_frame:end_frame]
    cleaned_win = cleaned_audio[start_frame:end_frame]
    restored_win = restored[start_frame:end_frame]

    manifest = {
        "base_drums_path": str(base_path),
        "cleaned_drums_path": str(cleaned_path),
        "output_path": str(output_path),
        "sample_rate": sample_rate,
        "pipeline": {
            "mode": "windowed_drum_body_restore_v1",
            "strategy": "aggressive_clean_global_plus_base_low_body_local_restore",
        },
        "window_sec": {
            "start": float(args.start_sec),
            "end": float(args.end_sec),
            "xfade": float(args.xfade_sec),
        },
        "tuning": {
            "low_end_hz": float(args.low_end_hz),
            "body_end_hz": float(args.body_end_hz),
            "low_blend": float(args.low_blend),
            "body_blend": float(args.body_blend),
        },
        "metrics": {
            "base_window_rms": round(_safe_rms(base_win), 6),
            "cleaned_window_rms": round(_safe_rms(cleaned_win), 6),
            "restored_window_rms": round(_safe_rms(restored_win), 6),
            "full_rms": round(_safe_rms(restored), 6),
        },
    }
    manifest_path = output_dir / "stem_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if listen_dir is not None:
        listen_dir.mkdir(parents=True, exist_ok=True)
        if raw_ref_path is not None:
            _write_flac_copy(raw_ref_path, listen_dir / "00_raw_ref.flac")
        _copy_file(base_path, listen_dir / "01_drums_base.flac")
        _copy_file(cleaned_path, listen_dir / "02_drums_hybridbs_aggressive.flac")
        _copy_file(output_path, listen_dir / args.listen_label)
        (listen_dir / "00_README.txt").write_text(
            "Listen only here.\n00 = raw reference.\n01 = current base drums.\n02 = aggressive clean drums.\n03 = windowed low/body restored drums.\nFocus first: 0:29-0:35.\n",
            encoding="utf-8",
        )
        (listen_dir / "candidate_map.json").write_text(
            json.dumps(
                {
                    "00_raw_ref.flac": str(listen_dir / "00_raw_ref.flac"),
                    "01_drums_base.flac": str(base_path),
                    "02_drums_hybridbs_aggressive.flac": str(cleaned_path),
                    args.listen_label: str(output_path),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    _emit(f"EXIT output_path={output_path}")
    print(json.dumps({"manifest_path": str(manifest_path), "output_path": str(output_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
