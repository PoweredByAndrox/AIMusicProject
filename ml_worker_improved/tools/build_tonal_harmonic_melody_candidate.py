from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

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

ROUTE_NAME = "tonal_harmonic_melody_v1"
OUTPUT_SUBTYPE = "PCM_24"


def _ensure_stereo(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, np.newaxis]
    if array.ndim != 2:
        raise ValueError(f"expected mono/stereo audio array, got shape={array.shape}")
    if array.shape[1] == 1:
        array = np.repeat(array, 2, axis=1)
    if array.shape[1] != 2:
        raise ValueError(f"expected one or two channels, got channels={array.shape[1]}")
    return array.astype(np.float32, copy=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audio_summary(audio: np.ndarray) -> dict[str, Any]:
    array = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(array)) + 1e-12)) if array.size else 0.0
    return {
        "rms": round(rms, 9),
        "rms_dbfs": round(20.0 * np.log10(max(rms, 1e-12)), 6),
        "peak": round(peak, 9),
        "peak_dbfs": round(20.0 * np.log10(max(peak, 1e-12)), 6),
        "frames": int(array.shape[0]),
        "channels": int(array.shape[1]) if array.ndim == 2 else 1,
    }


def _limit_peak(audio: np.ndarray, peak_limit: float = 0.999) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= peak_limit or peak <= 1e-8:
        return audio.astype(np.float32, copy=False)
    return (audio * (peak_limit / peak)).astype(np.float32)


def _frequency_weight(
    *,
    sample_rate: int,
    n_fft: int,
    highpass_hz: float,
    lowpass_hz: float,
) -> np.ndarray:
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    high = 1.0 / (1.0 + np.exp(-(frequencies - float(highpass_hz)) / max(float(highpass_hz) * 0.08, 1.0)))
    low = 1.0 / (1.0 + np.exp((frequencies - float(lowpass_hz)) / max(float(lowpass_hz) * 0.06, 1.0)))
    return np.clip(high * low, 0.0, 1.0).astype(np.float32)[:, np.newaxis]


def _channel_tonal_candidate(
    channel_audio: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    harmonic_margin: float,
    percussive_margin: float,
    percussive_suppression: float,
    highpass_hz: float,
    lowpass_hz: float,
) -> np.ndarray:
    stft = librosa.stft(
        np.asarray(channel_audio, dtype=np.float32),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        center=True,
    )
    magnitude = np.abs(stft).astype(np.float32)
    harmonic_mag, percussive_mag = librosa.decompose.hpss(
        magnitude,
        margin=(float(harmonic_margin), float(percussive_margin)),
    )
    total_mag = harmonic_mag + percussive_mag + 1e-8
    harmonic_ratio = harmonic_mag / total_mag
    percussive_ratio = percussive_mag / total_mag
    percussive_gate = 1.0 - np.clip(percussive_ratio * float(percussive_suppression), 0.0, 0.95)
    mask = harmonic_ratio * percussive_gate
    mask *= _frequency_weight(
        sample_rate=int(sample_rate),
        n_fft=int(n_fft),
        highpass_hz=float(highpass_hz),
        lowpass_hz=float(lowpass_hz),
    )
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    candidate = librosa.istft(
        stft * mask,
        hop_length=int(hop_length),
        length=int(channel_audio.shape[0]),
    )
    return candidate.astype(np.float32)


def build_tonal_harmonic_melody_candidate_audio(
    source_audio: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int = 4096,
    hop_length: int = 512,
    harmonic_margin: float = 4.0,
    percussive_margin: float = 8.0,
    percussive_suppression: float = 0.72,
    highpass_hz: float = 110.0,
    lowpass_hz: float = 12000.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = _ensure_stereo(source_audio)
    channels = [
        _channel_tonal_candidate(
            source[:, channel_index],
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            harmonic_margin=harmonic_margin,
            percussive_margin=percussive_margin,
            percussive_suppression=percussive_suppression,
            highpass_hz=highpass_hz,
            lowpass_hz=lowpass_hz,
        )
        for channel_index in range(source.shape[1])
    ]
    candidate = np.stack(channels, axis=1).astype(np.float32)
    candidate = _limit_peak(candidate)
    diagnostics = {
        "route_name": ROUTE_NAME,
        "n_fft": int(n_fft),
        "hop_length": int(hop_length),
        "harmonic_margin": float(harmonic_margin),
        "percussive_margin": float(percussive_margin),
        "percussive_suppression": float(percussive_suppression),
        "highpass_hz": float(highpass_hz),
        "lowpass_hz": float(lowpass_hz),
        "source": _audio_summary(source),
        "candidate": _audio_summary(candidate),
    }
    return candidate, diagnostics


def _no_reference_contract() -> dict[str, Any]:
    return {
        "full_track": True,
        "benchmark_compatible": True,
        "target_reference_assisted": False,
        "target_reference_used": False,
        "target_reference_used_at_application": False,
        "application_reference_free": True,
        "delivery_allowed_without_reference": True,
        "registered_model_candidate": True,
        "fallback_used": False,
        "hidden_inference": False,
        "per_track_hardcode_allowed": False,
        "not_timestamp_hardcoded": True,
        "no_reference_production_claim": True,
        "route_name": ROUTE_NAME,
    }


def build_tonal_harmonic_melody_candidate(
    *,
    source_audio_path: str | Path,
    output_dir: str | Path,
    listen_dir: str | Path | None = None,
    n_fft: int = 4096,
    hop_length: int = 512,
    harmonic_margin: float = 4.0,
    percussive_margin: float = 8.0,
    percussive_suppression: float = 0.72,
    highpass_hz: float = 110.0,
    lowpass_hz: float = 12000.0,
) -> dict[str, Any]:
    source_path = Path(source_audio_path).resolve()
    resolved_output_dir = Path(output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    if not source_path.exists():
        raise FileNotFoundError(f"source audio is missing: {source_path}")

    source_audio, sample_rate = sf.read(str(source_path), always_2d=True)
    candidate, diagnostics = build_tonal_harmonic_melody_candidate_audio(
        source_audio.astype(np.float32),
        sample_rate=int(sample_rate),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        harmonic_margin=float(harmonic_margin),
        percussive_margin=float(percussive_margin),
        percussive_suppression=float(percussive_suppression),
        highpass_hz=float(highpass_hz),
        lowpass_hz=float(lowpass_hz),
    )

    melody_path = resolved_output_dir / "melody.flac"
    sf.write(str(melody_path), candidate, int(sample_rate), subtype=OUTPUT_SUBTYPE)
    manifest = {
        "audio_path": str(source_path),
        "audio_sha256": _sha256_file(source_path),
        "model_name": ROUTE_NAME,
        "sample_rate": int(sample_rate),
        "export_subtype": OUTPUT_SUBTYPE,
        "melody_stem_included": True,
        "contract": _no_reference_contract(),
        "pipeline": {
            "mode": ROUTE_NAME,
            "source": "authoritative_full_mix_only",
            "target_reference_read": False,
            "reference_sum_mix_read": False,
            "postprocess": "stft_hpss_tonal_mask_percussive_suppression_v1",
            "diagnostics": diagnostics,
        },
        "delivery": {
            "status": "requires_objective_quality_gate",
            "reason": "no_reference_candidate_requires_ground_truth_benchmark_before_delivery",
        },
        "stems": {
            "melody": {
                "path": str(melody_path),
                **_audio_summary(candidate),
            },
        },
    }
    if listen_dir is not None:
        resolved_listen_dir = Path(listen_dir).resolve()
        resolved_listen_dir.mkdir(parents=True, exist_ok=True)
        source_listen_path = resolved_listen_dir / "00_source_mix.flac"
        candidate_listen_path = resolved_listen_dir / "01_tonal_harmonic_melody_v1.flac"
        shutil.copyfile(source_path, source_listen_path)
        shutil.copyfile(melody_path, candidate_listen_path)
        (resolved_listen_dir / "00_README.txt").write_text(
            "Diagnostic listen pack only.\n00 = authoritative source mix.\n01 = no-reference tonal harmonic melody candidate.\n",
            encoding="utf-8",
        )
        manifest["listen"] = {
            "listen_dir": str(resolved_listen_dir),
            "source_mix": str(source_listen_path),
            "candidate": str(candidate_listen_path),
            "status": "diagnostic_requires_objective_gate",
        }

    manifest_path = resolved_output_dir / "stem_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a no-reference tonal harmonic melody candidate from source mix.")
    parser.add_argument("source_audio_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--n-fft", type=int, default=4096)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--harmonic-margin", type=float, default=4.0)
    parser.add_argument("--percussive-margin", type=float, default=8.0)
    parser.add_argument("--percussive-suppression", type=float, default=0.72)
    parser.add_argument("--highpass-hz", type=float, default=110.0)
    parser.add_argument("--lowpass-hz", type=float, default=12000.0)
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="build_tonal_harmonic_melody_candidate.py",
        entrypoint_path="/app/tools/build_tonal_harmonic_melody_candidate.py",
    )
    args = parse_args()
    manifest = build_tonal_harmonic_melody_candidate(
        source_audio_path=args.source_audio_path,
        output_dir=args.output_dir,
        listen_dir=args.listen_dir,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        harmonic_margin=args.harmonic_margin,
        percussive_margin=args.percussive_margin,
        percussive_suppression=args.percussive_suppression,
        highpass_hz=args.highpass_hz,
        lowpass_hz=args.lowpass_hz,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
