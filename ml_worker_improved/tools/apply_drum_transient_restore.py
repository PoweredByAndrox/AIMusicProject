from __future__ import annotations

import argparse
import hashlib
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
from worker.high_quality_stem_export import _split_harmonic_percussive


def _emit(message: str) -> None:
    print(f"[drum_transient_restore] {message}", flush=True)


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _write_flac_copy(source_audio_path: Path, target_flac_path: Path) -> None:
    if source_audio_path.suffix.lower() == ".flac":
        shutil.copyfile(source_audio_path, target_flac_path)
        return
    audio, sample_rate = sf.read(str(source_audio_path), always_2d=True)
    sf.write(str(target_flac_path), audio, sample_rate, subtype="PCM_24")


def _copy_file(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)


def _safe_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    return math.sqrt(float(np.mean(np.square(mono, dtype=np.float32))))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _highpass(signal: np.ndarray, *, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    if signal.size == 0:
        return signal.astype(np.float32, copy=False)
    out = np.zeros_like(signal, dtype=np.float32)
    for channel_index in range(signal.shape[1]):
        stft = librosa.stft(signal[:, channel_index], n_fft=2048, hop_length=512)
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
        mask = (freqs >= float(cutoff_hz)).astype(np.float32)[:, np.newaxis]
        filtered = stft * mask
        out[:, channel_index] = librosa.istft(filtered, hop_length=512, length=signal.shape[0]).astype(np.float32)
    return out


def _clip_to_peak(audio: np.ndarray, *, max_peak: float = 0.9999) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= max_peak:
        return audio.astype(np.float32, copy=False)
    return (audio / peak * float(max_peak)).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore drum transient/air detail from a base drum stem into a cleaned drum stem."
    )
    parser.add_argument("base_drums_path")
    parser.add_argument("cleaned_drums_path")
    parser.add_argument("output_dir")
    parser.add_argument("--raw-ref-path")
    parser.add_argument("--perc-restore", type=float, default=0.9)
    parser.add_argument("--air-restore", type=float, default=0.45)
    parser.add_argument("--air-cutoff-hz", type=float, default=4200.0)
    parser.add_argument("--cleaned-harmonic-weight", type=float, default=1.0)
    parser.add_argument("--cleaned-percussive-weight", type=float, default=0.25)
    parser.add_argument("--base-harmonic-weight", type=float, default=0.0)
    parser.add_argument("--dsp-device", default="cuda")
    parser.add_argument("--listen-dir")
    parser.add_argument("--listen-label", default="03_drums_transient_restored.flac")
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="apply_drum_transient_restore.py",
        entrypoint_path="/app/tools/apply_drum_transient_restore.py",
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

    output_dir.mkdir(parents=True, exist_ok=True)
    _emit("splitting_base_hpss")
    base_harmonic, base_percussive = _split_harmonic_percussive(
        base_audio,
        harmonic_margin=1.0,
        percussive_margin=4.0,
        dsp_device=str(args.dsp_device),
    )
    _emit("splitting_cleaned_hpss")
    cleaned_harmonic, cleaned_percussive = _split_harmonic_percussive(
        cleaned_audio,
        harmonic_margin=1.0,
        percussive_margin=4.0,
        dsp_device=str(args.dsp_device),
    )

    percussive_restore = np.clip(
        base_percussive - cleaned_percussive,
        a_min=0.0,
        a_max=None,
    ) - np.clip(
        cleaned_percussive - base_percussive,
        a_min=0.0,
        a_max=None,
    )
    air_restore = _highpass(base_harmonic - cleaned_harmonic, sample_rate=sample_rate, cutoff_hz=float(args.air_cutoff_hz))

    restored = (
        cleaned_harmonic * float(args.cleaned_harmonic_weight)
        + cleaned_percussive * float(args.cleaned_percussive_weight)
        + base_percussive * float(args.perc_restore)
        + air_restore * float(args.air_restore)
        + base_harmonic * float(args.base_harmonic_weight)
    ).astype(np.float32)
    restored = _clip_to_peak(restored)

    output_path = output_dir / "drums_transient_restored.flac"
    sf.write(str(output_path), restored, sample_rate, subtype="PCM_24")

    base_harmonic_rms = _safe_rms(base_harmonic)
    base_percussive_rms = _safe_rms(base_percussive)
    cleaned_harmonic_rms = _safe_rms(cleaned_harmonic)
    cleaned_percussive_rms = _safe_rms(cleaned_percussive)
    restored_harmonic, restored_percussive = _split_harmonic_percussive(
        restored,
        harmonic_margin=1.0,
        percussive_margin=4.0,
        dsp_device=str(args.dsp_device),
    )
    restored_harmonic_rms = _safe_rms(restored_harmonic)
    restored_percussive_rms = _safe_rms(restored_percussive)

    diagnostic_manifest = {
        "base_drums_path": str(base_path),
        "cleaned_drums_path": str(cleaned_path),
        "output_path": str(output_path),
        "sample_rate": sample_rate,
        "pipeline": {
            "mode": "drum_transient_restore_v1",
            "strategy": "cleaned_harmonic_plus_base_percussive_plus_base_air",
        },
        "tuning": {
            "perc_restore": float(args.perc_restore),
            "air_restore": float(args.air_restore),
            "air_cutoff_hz": float(args.air_cutoff_hz),
            "cleaned_harmonic_weight": float(args.cleaned_harmonic_weight),
            "cleaned_percussive_weight": float(args.cleaned_percussive_weight),
            "base_harmonic_weight": float(args.base_harmonic_weight),
            "dsp_device": str(args.dsp_device),
        },
        "metrics": {
            "base_harmonic_rms": round(base_harmonic_rms, 6),
            "base_percussive_rms": round(base_percussive_rms, 6),
            "cleaned_harmonic_rms": round(cleaned_harmonic_rms, 6),
            "cleaned_percussive_rms": round(cleaned_percussive_rms, 6),
            "restored_harmonic_rms": round(restored_harmonic_rms, 6),
            "restored_percussive_rms": round(restored_percussive_rms, 6),
            "restored_ratio": round(restored_percussive_rms / max(restored_harmonic_rms, 1e-9), 6),
            "air_restore_rms": round(_safe_rms(air_restore), 6),
            "percussive_restore_rms": round(_safe_rms(percussive_restore), 6),
        },
    }
    diagnostic_manifest_path = output_dir / "drum_transient_restore_manifest.json"
    diagnostic_manifest_path.write_text(json.dumps(diagnostic_manifest, indent=2), encoding="utf-8")
    candidate_manifest = {
        "schema_version": 1,
        "model_id": "drum_transient_restore",
        "model_name": "drum_transient_restore",
        "model": {
            "model_id": "drum_transient_restore",
            "family": "no_reference_drum_transient_repair",
        },
        "contract": {
            "full_track": True,
            "preview": False,
            "benchmark_compatible": True,
            "fallback_used": False,
            "hidden_inference": False,
            "registered_model_candidate": False,
            "target_reference_assisted": False,
            "target_reference_used_at_application": False,
            "application_reference_free": True,
            "delivery_allowed_without_reference": True,
            "delivery_allowed_without_quality_gate": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
            "quality_gate_required": True,
        },
        "route": {
            "mode": "drum_transient_restore_v1",
            "diagnostic_manifest_path": str(diagnostic_manifest_path),
        },
        "inputs": {
            "base_drums_path": str(base_path),
            "cleaned_drums_path": str(cleaned_path),
        },
        "stems": {
            "drums": {
                "path": str(output_path),
                "sha256": _sha256(output_path),
                "role": "drum_transient_restored_candidate",
            }
        },
    }
    manifest_path = output_dir / "stem_manifest.json"
    manifest_path.write_text(json.dumps(candidate_manifest, indent=2), encoding="utf-8")

    if listen_dir is not None:
        listen_dir.mkdir(parents=True, exist_ok=True)
        if raw_ref_path is not None:
            _write_flac_copy(raw_ref_path, listen_dir / "00_raw_ref.flac")
        _copy_file(base_path, listen_dir / "01_drums_base.flac")
        _copy_file(cleaned_path, listen_dir / "02_drums_hybridbs_aggressive.flac")
        _copy_file(output_path, listen_dir / args.listen_label)
        (listen_dir / "00_README.txt").write_text(
            "Listen only here.\n00 = raw reference.\n01 = current base drums.\n02 = aggressive clean drums.\n03 = transient-restored hybrid drums.\n",
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
