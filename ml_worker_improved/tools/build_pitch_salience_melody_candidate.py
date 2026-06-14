from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from build_model_stack_melody_candidate import (  # noqa: E402
    _audio_summary,
    _base_manifest_source_sha256,
    _build_base_manifest_from_source,
    _fit_audio_shape,
    _limit_peak,
    _load_stems_from_manifest,
    _remove_positive_projection,
    _resample_stereo_audio,
    sha256_file,
)

ROUTE_NAME = "pitch_salience_melody_v1"
OUTPUT_SUBTYPE = "PCM_24"
PROCESS_START_MONOTONIC = time.monotonic()


def _emit_progress(event: str, *, route_name: str = ROUTE_NAME, **metadata: Any) -> None:
    record = {
        "event": event,
        "route_name": route_name,
        "elapsed_sec": round(time.monotonic() - PROCESS_START_MONOTONIC, 6),
        **metadata,
    }
    print(json.dumps(record, sort_keys=True), flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_stereo(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, np.newaxis]
    if array.shape[1] == 1:
        array = np.repeat(array, 2, axis=1)
    return array.astype(np.float32, copy=False)


def _rms(audio: np.ndarray) -> float:
    array = np.asarray(audio, dtype=np.float32)
    if array.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(array.astype(np.float64))) + 1e-12))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _analysis_signal(mono: np.ndarray, sample_rate: int, *, max_samples: int = 262_144) -> tuple[np.ndarray, float]:
    if mono.size <= max_samples:
        return mono.astype(np.float32), float(sample_rate)
    step = int(np.ceil(mono.size / max_samples))
    return mono[::step].astype(np.float32), float(sample_rate) / float(step)


def _pitch_confidence_like_semantic_judge(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    stereo = _ensure_stereo(audio)
    mono = np.mean(stereo, axis=1).astype(np.float32)
    signal, effective_sr = _analysis_signal(mono, int(sample_rate))
    if signal.size < 64 or effective_sr <= 0:
        return {"dominant_frequency_hz": 0.0, "pitch_confidence": 0.0}
    signal = signal - float(np.mean(signal))
    spectrum = np.fft.rfft(signal * np.hanning(signal.size).astype(np.float32))
    power = np.abs(spectrum).astype(np.float64) ** 2
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / effective_sr)
    if power.size <= 1 or float(np.sum(power[1:])) <= 0.0:
        return {"dominant_frequency_hz": 0.0, "pitch_confidence": 0.0}
    power[0] = 0.0
    total = float(np.sum(power))
    peak_index = int(np.argmax(power))
    return {
        "dominant_frequency_hz": round(float(freqs[peak_index]), 6),
        "pitch_confidence": round(_clip01((float(power[peak_index]) / total) * 60.0), 6),
    }


def _dominant_pitch_frequency(
    audio: np.ndarray,
    sample_rate: int,
    *,
    min_frequency_hz: float,
    max_frequency_hz: float,
) -> float:
    stereo = _ensure_stereo(audio)
    mono = np.mean(stereo, axis=1).astype(np.float32)
    signal, effective_sr = _analysis_signal(mono, int(sample_rate))
    if signal.size < 64 or effective_sr <= 0:
        return 0.0
    signal = signal - float(np.mean(signal))
    spectrum = np.fft.rfft(signal * np.hanning(signal.size).astype(np.float32))
    power = np.abs(spectrum).astype(np.float64) ** 2
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / effective_sr)
    mask = (freqs >= float(min_frequency_hz)) & (freqs <= float(max_frequency_hz))
    if not np.any(mask):
        return 0.0
    masked_power = np.where(mask, power, 0.0)
    peak_index = int(np.argmax(masked_power))
    if float(masked_power[peak_index]) <= 0.0:
        return 0.0
    return float(freqs[peak_index])


def _harmonic_salience_filter(
    audio: np.ndarray,
    *,
    sample_rate: int,
    fundamental_hz: float,
    bandwidth_ratio: float,
    max_harmonic_hz: float,
) -> np.ndarray:
    source = _ensure_stereo(audio)
    if fundamental_hz <= 0.0:
        return np.zeros_like(source, dtype=np.float32)
    channels: list[np.ndarray] = []
    freqs = np.fft.rfftfreq(source.shape[0], d=1.0 / float(sample_rate))
    for channel_index in range(source.shape[1]):
        spectrum = np.fft.rfft(source[:, channel_index])
        mask = np.zeros_like(freqs, dtype=np.float64)
        harmonic = 1
        while fundamental_hz * harmonic <= float(max_harmonic_hz):
            center = fundamental_hz * harmonic
            width = max(2.5, center * float(bandwidth_ratio))
            mask += np.exp(-0.5 * np.square((freqs - center) / width)) / float(harmonic)
            harmonic += 1
        mask = np.clip(mask, 0.0, 1.0)
        filtered = np.fft.irfft(spectrum * mask, n=source.shape[0])
        channels.append(filtered.astype(np.float32))
    return np.stack(channels, axis=1).astype(np.float32)


def _normalize_to_rms(audio: np.ndarray, target_rms: float) -> np.ndarray:
    current = _rms(audio)
    if current <= 1e-8 or target_rms <= 0.0:
        return np.zeros_like(audio, dtype=np.float32)
    return (np.asarray(audio, dtype=np.float32) * (float(target_rms) / current)).astype(np.float32)


def _execution_contract_from_base(base_manifest: dict[str, Any], *, device: str, route_name: str) -> dict[str, Any]:
    base_execution_contract = base_manifest.get("execution_contract")
    if not isinstance(base_execution_contract, dict):
        base_execution_contract = {}
    base_cuda_preflight = base_execution_contract.get("cuda_preflight")
    if not isinstance(base_cuda_preflight, dict):
        base_cuda_preflight = {}
    return {
        "route_name": route_name,
        "docker_runtime_required": True,
        "expected_acceleration_mode": str(
            base_execution_contract.get("expected_acceleration_mode")
            or "gpu_inference_then_cpu_dsp_scoring"
        ),
        "gpu_inference_required": True,
        "gpu_inference_skipped": base_execution_contract.get("gpu_inference_skipped") is True,
        "cuda_preflight": {
            "cuda_available": base_cuda_preflight.get("cuda_available") is True,
            "requested_device": str(base_cuda_preflight.get("requested_device") or device or ""),
            "resolved_device": str(base_cuda_preflight.get("resolved_device") or ""),
            "device_count": int(base_cuda_preflight.get("device_count") or 0),
            "device_name": str(base_cuda_preflight.get("device_name") or ""),
        },
        "base_route_name": str(base_execution_contract.get("route_name") or ""),
        "base_model_name": str(base_manifest.get("model_name") or ""),
        "base_execution_contract_inherited": bool(base_execution_contract),
    }


def _no_reference_contract(route_name: str) -> dict[str, Any]:
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
        "route_name": route_name,
    }


def build_pitch_salience_melody_candidate_audio(
    stems: dict[str, np.ndarray],
    *,
    source_audio: np.ndarray | None,
    sample_rate: int,
    min_frequency_hz: float = 80.0,
    max_frequency_hz: float = 1200.0,
    max_harmonic_hz: float = 4000.0,
    bandwidth_ratio: float = 0.035,
    base_weight: float = 0.70,
    salience_weight: float = 0.72,
    salience_rms_gain: float = 1.45,
    vocal_suppression: float = 0.22,
    non_target_suppression: float = 0.18,
    route_name: str = ROUTE_NAME,
) -> tuple[np.ndarray, dict[str, Any]]:
    base_melody = stems.get("melody")
    if base_melody is None:
        raise ValueError("base manifest does not contain melody stem")
    base = _ensure_stereo(base_melody)
    frames = int(base.shape[0])
    channels = int(base.shape[1])
    fitted_stems = {
        stem_name: _fit_audio_shape(stem_audio, frames=frames, channels=channels)
        for stem_name, stem_audio in stems.items()
    }
    source = (
        _fit_audio_shape(source_audio, frames=frames, channels=channels)
        if source_audio is not None
        else base
    )
    _emit_progress(
        "pitch_salience_audio_start",
        route_name=route_name,
        frame_count=frames,
        channel_count=channels,
        sample_rate=int(sample_rate),
        stem_count=len(fitted_stems),
    )
    base_pitch = _pitch_confidence_like_semantic_judge(base, int(sample_rate))
    dominant_hz = _dominant_pitch_frequency(
        base,
        int(sample_rate),
        min_frequency_hz=float(min_frequency_hz),
        max_frequency_hz=float(max_frequency_hz),
    )
    _emit_progress(
        "pitch_salience_detected",
        route_name=route_name,
        dominant_frequency_hz=round(float(dominant_hz), 6),
        base_pitch_confidence=base_pitch["pitch_confidence"],
    )
    salience_from_base = _harmonic_salience_filter(
        base,
        sample_rate=int(sample_rate),
        fundamental_hz=float(dominant_hz),
        bandwidth_ratio=float(bandwidth_ratio),
        max_harmonic_hz=float(max_harmonic_hz),
    )
    salience_from_source = _harmonic_salience_filter(
        source,
        sample_rate=int(sample_rate),
        fundamental_hz=float(dominant_hz),
        bandwidth_ratio=float(bandwidth_ratio),
        max_harmonic_hz=float(max_harmonic_hz),
    )
    salience = salience_from_base * 0.78 + salience_from_source * 0.22
    salience = _normalize_to_rms(salience, _rms(base) * float(salience_rms_gain))
    candidate = base * float(base_weight) + salience * float(salience_weight)
    removals: dict[str, Any] = {}
    for stem_name, strength in (
        ("vocals", float(vocal_suppression)),
        ("drums", float(non_target_suppression)),
        ("bass", float(non_target_suppression)),
    ):
        suppressor = fitted_stems.get(stem_name)
        if suppressor is None:
            continue
        candidate, removal = _remove_positive_projection(candidate, suppressor, strength=float(strength))
        removals[stem_name] = removal
    candidate = _limit_peak(candidate.astype(np.float32))
    candidate_pitch = _pitch_confidence_like_semantic_judge(candidate, int(sample_rate))
    _emit_progress(
        "pitch_salience_audio_end",
        route_name=route_name,
        candidate_pitch_confidence=candidate_pitch["pitch_confidence"],
        candidate_dominant_frequency_hz=candidate_pitch["dominant_frequency_hz"],
    )
    diagnostics = {
        "route_name": route_name,
        "sample_rate": int(sample_rate),
        "dominant_frequency_hz": round(float(dominant_hz), 6),
        "pitch_confidence_before": base_pitch,
        "pitch_confidence_after": candidate_pitch,
        "parameters": {
            "min_frequency_hz": float(min_frequency_hz),
            "max_frequency_hz": float(max_frequency_hz),
            "max_harmonic_hz": float(max_harmonic_hz),
            "bandwidth_ratio": float(bandwidth_ratio),
            "base_weight": float(base_weight),
            "salience_weight": float(salience_weight),
            "salience_rms_gain": float(salience_rms_gain),
            "vocal_suppression": float(vocal_suppression),
            "non_target_suppression": float(non_target_suppression),
        },
        "projection_debleed": removals,
        "source_audio_used": source_audio is not None,
        "base": _audio_summary(base),
        "salience": _audio_summary(salience),
        "candidate": _audio_summary(candidate),
    }
    return candidate.astype(np.float32), diagnostics


def build_pitch_salience_melody_candidate(
    *,
    source_audio_path: str | Path,
    output_dir: str | Path,
    base_manifest_path: str | Path | None = None,
    listen_dir: str | Path | None = None,
    model_name: str = "hybrid_htdemucs_ft_bs_roformer_6s",
    device: str = "cuda",
    min_frequency_hz: float = 80.0,
    max_frequency_hz: float = 1200.0,
    max_harmonic_hz: float = 4000.0,
    bandwidth_ratio: float = 0.035,
    base_weight: float = 0.70,
    salience_weight: float = 0.72,
    salience_rms_gain: float = 1.45,
    route_name: str = ROUTE_NAME,
) -> dict[str, Any]:
    _emit_progress(
        "candidate_build_start",
        route_name=route_name,
        source_audio_path=str(source_audio_path),
        output_dir=str(output_dir),
        model_name=str(model_name),
        device=str(device),
    )
    source_path = Path(source_audio_path).resolve()
    resolved_output_dir = Path(output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    if not source_path.exists():
        raise FileNotFoundError(f"source audio is missing: {source_path}")
    resolved_base_manifest_path = (
        Path(base_manifest_path).resolve()
        if base_manifest_path is not None
        else _build_base_manifest_from_source(
            source_audio_path=source_path,
            output_dir=resolved_output_dir,
            model_name=str(model_name),
            device=str(device),
        )
    )
    stems, stem_sample_rate, base_manifest, contract_check = _load_stems_from_manifest(resolved_base_manifest_path)
    _emit_progress(
        "base_manifest_loaded",
        route_name=route_name,
        base_manifest_path=str(resolved_base_manifest_path),
        stem_sample_rate=int(stem_sample_rate),
        stem_count=len(stems),
        contract_status=str(contract_check.get("status") or ""),
    )
    source_audio, source_sample_rate = sf.read(str(source_path), always_2d=True)
    source_audio, source_resampled = _resample_stereo_audio(
        source_audio,
        source_sample_rate=int(source_sample_rate),
        target_sample_rate=int(stem_sample_rate),
    )
    source_sha256 = sha256_file(source_path)
    base_source_sha256 = _base_manifest_source_sha256(base_manifest, resolved_base_manifest_path)
    if str(base_source_sha256) != source_sha256:
        raise ValueError(
            "base manifest source hash does not match source audio: "
            + json.dumps(
                {"source": source_sha256, "base_manifest": str(base_source_sha256)},
                ensure_ascii=False,
            )
        )
    candidate, diagnostics = build_pitch_salience_melody_candidate_audio(
        stems,
        source_audio=source_audio.astype(np.float32),
        sample_rate=int(stem_sample_rate),
        min_frequency_hz=float(min_frequency_hz),
        max_frequency_hz=float(max_frequency_hz),
        max_harmonic_hz=float(max_harmonic_hz),
        bandwidth_ratio=float(bandwidth_ratio),
        base_weight=float(base_weight),
        salience_weight=float(salience_weight),
        salience_rms_gain=float(salience_rms_gain),
        route_name=route_name,
    )
    melody_path = resolved_output_dir / "melody.flac"
    _emit_progress("write_stem_start", route_name=route_name, stem_name="melody", path=str(melody_path))
    sf.write(str(melody_path), candidate, int(stem_sample_rate), subtype=OUTPUT_SUBTYPE)
    _emit_progress("write_stem_end", route_name=route_name, stem_name="melody", path=str(melody_path))
    manifest = {
        "audio_path": str(source_path),
        "audio_sha256": _sha256_file(source_path),
        "model_name": route_name,
        "sample_rate": int(stem_sample_rate),
        "export_subtype": OUTPUT_SUBTYPE,
        "melody_stem_included": True,
        "target_stem": "melody",
        "target_stem_artifact": {
            "target_stem": "melody",
            "path": str(melody_path),
            "source": "pitch_salience_melody_candidate",
            "source_stems": ["melody"],
        },
        "delivery_allowed": False,
        "presentation_status": "diagnostic_candidate_not_for_listen_pack",
        "release_status": "not_release_ready",
        "release_ready": False,
        "contract": _no_reference_contract(route_name),
        "execution_contract": _execution_contract_from_base(base_manifest, device=str(device), route_name=route_name),
        "pipeline": {
            "mode": route_name,
            "source": "authoritative_full_mix_plus_no_reference_base_melody_pitch_salience",
            "base_manifest_path": str(resolved_base_manifest_path),
            "base_manifest_model_name": base_manifest.get("model_name"),
            "base_manifest_contract_checked": contract_check["status"] == "accepted",
            "base_manifest_contract": contract_check,
            "source_sample_rate": int(source_sample_rate),
            "source_resampled_to_sample_rate": int(stem_sample_rate),
            "source_resampled": bool(source_resampled),
            "target_reference_read": False,
            "reference_sum_mix_read": False,
            "postprocess": "adaptive_harmonic_pitch_salience_reinforcement_v1",
            "diagnostics": diagnostics,
        },
        "delivery": {
            "status": "requires_objective_quality_gate",
            "reason": "pitch_salience_melody_candidate_requires_stem_semantic_gate_before_delivery",
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
        _emit_progress("write_listen_diagnostic_start", route_name=route_name, listen_dir=str(resolved_listen_dir))
        source_listen_path = resolved_listen_dir / "00_source_mix.flac"
        candidate_listen_path = resolved_listen_dir / f"01_{route_name}.flac"
        sf.write(str(source_listen_path), source_audio.astype(np.float32), int(stem_sample_rate), subtype=OUTPUT_SUBTYPE)
        shutil.copyfile(melody_path, candidate_listen_path)
        (resolved_listen_dir / "00_README.txt").write_text(
            "Diagnostic listen pack only.\n"
            "00 = authoritative source mix.\n"
            "01 = no-reference pitch-salience melody candidate.\n",
            encoding="utf-8",
        )
        manifest["listen"] = {
            "listen_dir": str(resolved_listen_dir),
            "source_mix": str(source_listen_path),
            "candidate": str(candidate_listen_path),
            "status": "diagnostic_requires_objective_gate",
        }
        _emit_progress("write_listen_diagnostic_end", route_name=route_name, listen_dir=str(resolved_listen_dir))
    manifest_path = resolved_output_dir / "stem_manifest.json"
    _emit_progress("write_manifest_start", route_name=route_name, manifest_path=str(manifest_path))
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    _emit_progress("write_manifest_end", route_name=route_name, manifest_path=str(manifest_path))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build no-reference pitch-salience melody candidate.")
    parser.add_argument("source_audio_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--model-name", default="hybrid_htdemucs_ft_bs_roformer_6s")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-frequency-hz", type=float, default=80.0)
    parser.add_argument("--max-frequency-hz", type=float, default=1200.0)
    parser.add_argument("--max-harmonic-hz", type=float, default=4000.0)
    parser.add_argument("--bandwidth-ratio", type=float, default=0.035)
    parser.add_argument("--base-weight", type=float, default=0.70)
    parser.add_argument("--salience-weight", type=float, default=0.72)
    parser.add_argument("--salience-rms-gain", type=float, default=1.45)
    parser.add_argument("--route-name", default=ROUTE_NAME)
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="build_pitch_salience_melody_candidate.py",
        entrypoint_path="/app/tools/build_pitch_salience_melody_candidate.py",
    )
    args = parse_args()
    manifest = build_pitch_salience_melody_candidate(
        source_audio_path=args.source_audio_path,
        output_dir=args.output_dir,
        base_manifest_path=args.base_manifest,
        listen_dir=args.listen_dir,
        model_name=args.model_name,
        device=args.device,
        min_frequency_hz=args.min_frequency_hz,
        max_frequency_hz=args.max_frequency_hz,
        max_harmonic_hz=args.max_harmonic_hz,
        bandwidth_ratio=args.bandwidth_ratio,
        base_weight=args.base_weight,
        salience_weight=args.salience_weight,
        salience_rms_gain=args.salience_rms_gain,
        route_name=str(args.route_name),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
