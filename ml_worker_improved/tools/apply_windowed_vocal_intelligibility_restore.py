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


def _copy_listen_file(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)


def _safe_rms(audio: np.ndarray) -> float:
    data = np.asarray(audio, dtype=np.float32)
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data, dtype=np.float32))))


def _build_time_env(frame_count: int, *, sample_rate: int, hop_length: int, focus_start_sec: float, focus_end_sec: float, fade_sec: float) -> np.ndarray:
    fade = max(0.001, float(fade_sec))
    frame_times = librosa.frames_to_time(np.arange(frame_count), sr=sample_rate, hop_length=hop_length)

    def _gain(t: float) -> float:
        if t < focus_start_sec - fade or t > focus_end_sec + fade:
            return 0.0
        if focus_start_sec - fade <= t < focus_start_sec:
            return float((t - (focus_start_sec - fade)) / fade)
        if focus_start_sec <= t <= focus_end_sec:
            return 1.0
        if focus_end_sec < t <= focus_end_sec + fade:
            return float(((focus_end_sec + fade) - t) / fade)
        return 0.0

    return np.asarray([_gain(float(t)) for t in frame_times], dtype=np.float32)


def _activity_env(vocal_mag: np.ndarray) -> np.ndarray:
    frame_energy = vocal_mag.mean(axis=0).astype(np.float32)
    if frame_energy.size == 0:
        return frame_energy
    floor = float(np.percentile(frame_energy, 35))
    ceil = float(np.percentile(frame_energy, 95))
    denom = max(1e-6, ceil - floor)
    activity = np.clip((frame_energy - floor) / denom, 0.0, 1.0).astype(np.float32)
    activity = np.sqrt(activity).astype(np.float32)
    return activity


def _apply_windowed_intelligibility_restore(
    vocals_audio: np.ndarray,
    *,
    sample_rate: int,
    focus_start_sec: float,
    focus_end_sec: float,
    fade_sec: float,
    lowmid_attenuation: float,
    lowmid_start_hz: float,
    lowmid_end_hz: float,
    presence_gain: float,
    presence_start_hz: float,
    presence_end_hz: float,
    air_gain: float,
    air_start_hz: float,
    air_end_hz: float,
    n_fft: int = 4096,
    hop_length: int = 512,
) -> np.ndarray:
    vocals = np.asarray(vocals_audio, dtype=np.float32)
    if vocals.ndim == 1:
        vocals = vocals[:, np.newaxis]

    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft).astype(np.float32)
    lowmid_mask = ((freqs >= float(lowmid_start_hz)) & (freqs <= float(lowmid_end_hz))).astype(np.float32)[:, np.newaxis]
    presence_mask = ((freqs >= float(presence_start_hz)) & (freqs <= float(presence_end_hz))).astype(np.float32)[:, np.newaxis]
    air_mask = ((freqs >= float(air_start_hz)) & (freqs <= float(air_end_hz))).astype(np.float32)[:, np.newaxis]

    enhanced_channels: list[np.ndarray] = []
    for channel_index in range(vocals.shape[1]):
        vocal_channel = vocals[:, channel_index]
        vocal_stft = librosa.stft(vocal_channel, n_fft=n_fft, hop_length=hop_length)
        vocal_mag = np.abs(vocal_stft).astype(np.float32)
        time_env = _build_time_env(
            vocal_stft.shape[1],
            sample_rate=sample_rate,
            hop_length=hop_length,
            focus_start_sec=float(focus_start_sec),
            focus_end_sec=float(focus_end_sec),
            fade_sec=float(fade_sec),
        )[np.newaxis, :]
        activity = _activity_env(vocal_mag)[np.newaxis, :]
        env = activity * time_env

        eq = np.ones_like(vocal_mag, dtype=np.float32)
        if float(lowmid_attenuation) > 0.0:
            eq -= float(lowmid_attenuation) * lowmid_mask * env
        if float(presence_gain) > 0.0:
            eq += float(presence_gain) * presence_mask * env
        if float(air_gain) > 0.0:
            eq += float(air_gain) * air_mask * env
        eq = np.clip(eq, 0.55, 2.0).astype(np.float32)

        enhanced_stft = vocal_stft * eq
        enhanced = librosa.istft(enhanced_stft, hop_length=hop_length, length=vocal_channel.shape[0]).astype(np.float32)
        enhanced_channels.append(enhanced)

    return np.stack(enhanced_channels, axis=1).astype(np.float32, copy=False)


def main() -> None:
    require_docker_runtime(
        tool_name="apply_windowed_vocal_intelligibility_restore.py",
        entrypoint_path="/app/tools/apply_windowed_vocal_intelligibility_restore.py",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("vocals_path")
    parser.add_argument("raw_ref_path")
    parser.add_argument("output_dir")
    parser.add_argument("--listen-dir")
    parser.add_argument("--listen-label", default="02_mdx23c_intelligibility_26_60.flac")
    parser.add_argument("--focus-start-sec", type=float, default=26.0)
    parser.add_argument("--focus-end-sec", type=float, default=60.0)
    parser.add_argument("--fade-sec", type=float, default=0.35)
    parser.add_argument("--lowmid-attenuation", type=float, default=0.12)
    parser.add_argument("--lowmid-start-hz", type=float, default=250.0)
    parser.add_argument("--lowmid-end-hz", type=float, default=1200.0)
    parser.add_argument("--presence-gain", type=float, default=0.38)
    parser.add_argument("--presence-start-hz", type=float, default=1800.0)
    parser.add_argument("--presence-end-hz", type=float, default=4500.0)
    parser.add_argument("--air-gain", type=float, default=0.18)
    parser.add_argument("--air-start-hz", type=float, default=4500.0)
    parser.add_argument("--air-end-hz", type=float, default=9000.0)
    args = parser.parse_args()

    vocals_path = Path(args.vocals_path)
    raw_ref_path = Path(args.raw_ref_path)
    output_dir = Path(args.output_dir)
    listen_dir = Path(args.listen_dir) if args.listen_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)

    start_wall = time.time()
    vocals_audio, vocals_sr = sf.read(str(vocals_path), always_2d=True)
    raw_audio, raw_sr = sf.read(str(raw_ref_path), always_2d=True)

    enhanced_audio = _apply_windowed_intelligibility_restore(
        vocals_audio,
        sample_rate=int(vocals_sr),
        focus_start_sec=float(args.focus_start_sec),
        focus_end_sec=float(args.focus_end_sec),
        fade_sec=float(args.fade_sec),
        lowmid_attenuation=float(args.lowmid_attenuation),
        lowmid_start_hz=float(args.lowmid_start_hz),
        lowmid_end_hz=float(args.lowmid_end_hz),
        presence_gain=float(args.presence_gain),
        presence_start_hz=float(args.presence_start_hz),
        presence_end_hz=float(args.presence_end_hz),
        air_gain=float(args.air_gain),
        air_start_hz=float(args.air_start_hz),
        air_end_hz=float(args.air_end_hz),
    )

    vocals_flac_path = output_dir / "vocals.flac"
    raw_ref_flac_path = output_dir / "raw_ref.flac"
    sf.write(str(vocals_flac_path), enhanced_audio, vocals_sr, subtype="PCM_24")
    sf.write(str(raw_ref_flac_path), raw_audio, raw_sr, subtype="PCM_24")

    manifest = {
        "source_vocals_path": str(vocals_path),
        "source_raw_ref_path": str(raw_ref_path),
        "postprocess": {
            "name": "windowed_intelligibility_restore_v1",
            "focus_start_sec": float(args.focus_start_sec),
            "focus_end_sec": float(args.focus_end_sec),
            "fade_sec": float(args.fade_sec),
            "lowmid_attenuation": float(args.lowmid_attenuation),
            "lowmid_start_hz": float(args.lowmid_start_hz),
            "lowmid_end_hz": float(args.lowmid_end_hz),
            "presence_gain": float(args.presence_gain),
            "presence_start_hz": float(args.presence_start_hz),
            "presence_end_hz": float(args.presence_end_hz),
            "air_gain": float(args.air_gain),
            "air_start_hz": float(args.air_start_hz),
            "air_end_hz": float(args.air_end_hz),
        },
        "audio": {
            "sample_rate": int(vocals_sr),
            "frames": int(vocals_audio.shape[0]),
            "duration_sec": round(float(vocals_audio.shape[0]) / float(vocals_sr), 6) if vocals_sr else 0.0,
            "source_vocals_rms": round(_safe_rms(vocals_audio), 6),
            "enhanced_vocals_rms": round(_safe_rms(enhanced_audio), 6),
        },
        "artifacts": {
            "vocals_flac": str(vocals_flac_path),
            "raw_ref_flac": str(raw_ref_flac_path),
        },
        "elapsed_sec": round(time.time() - start_wall, 3),
    }
    manifest_path = output_dir / "stem_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if listen_dir is not None:
        listen_dir.mkdir(parents=True, exist_ok=True)
        _copy_listen_file(raw_ref_flac_path, listen_dir / "00_raw_ref.flac")
        _copy_listen_file(vocals_path, listen_dir / "01_mdx23c_base.flac")
        _copy_listen_file(vocals_flac_path, listen_dir / args.listen_label)
        (listen_dir / "00_README.txt").write_text(
            "Listen only here.\n00 = raw baseline.\n01 = mdx23c base.\n02 = mdx23c intelligibility restore.\n",
            encoding="utf-8",
        )
        (listen_dir / "candidate_map.json").write_text(
            json.dumps(
                {
                    "00_raw_ref.flac": str(raw_ref_flac_path),
                    "01_mdx23c_base.flac": str(vocals_path),
                    args.listen_label: str(vocals_flac_path),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print(json.dumps({"output_dir": str(output_dir), "manifest_path": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
