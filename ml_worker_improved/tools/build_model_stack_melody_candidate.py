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
from build_tonal_harmonic_melody_candidate import (  # noqa: E402
    build_tonal_harmonic_melody_candidate_audio,
)

ROUTE_NAME = "model_stack_melody_v1"
OUTPUT_SUBTYPE = "PCM_24"
TARGET_SOURCE_WEIGHTS = {
    "melody": 1.0,
    "other": 0.35,
    "guitar": 0.50,
    "piano": 0.35,
}
REQUIRED_NO_REFERENCE_FLAGS = {
    "target_reference_assisted": False,
    "target_reference_used": False,
    "target_reference_used_at_application": False,
    "application_reference_free": True,
    "delivery_allowed_without_reference": True,
    "fallback_used": False,
    "hidden_inference": False,
    "per_track_hardcode_allowed": False,
    "not_timestamp_hardcoded": True,
}


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


def _fit_audio_shape(audio: np.ndarray, *, frames: int, channels: int) -> np.ndarray:
    array = _ensure_stereo(audio)
    if array.shape[1] != channels:
        if array.shape[1] == 1 and channels == 2:
            array = np.repeat(array, 2, axis=1)
        else:
            raise ValueError(f"channel mismatch: expected={channels}, got={array.shape[1]}")
    if array.shape[0] == frames:
        return array.astype(np.float32, copy=False)
    fitted = np.zeros((frames, channels), dtype=np.float32)
    copy_frames = min(frames, array.shape[0])
    fitted[:copy_frames] = array[:copy_frames]
    return fitted


def _resample_stereo_audio(audio: np.ndarray, *, source_sample_rate: int, target_sample_rate: int) -> tuple[np.ndarray, bool]:
    array = _ensure_stereo(audio)
    if int(source_sample_rate) == int(target_sample_rate):
        return array.astype(np.float32, copy=False), False
    resampled = librosa.resample(
        array.T,
        orig_sr=int(source_sample_rate),
        target_sr=int(target_sample_rate),
        res_type="kaiser_best",
    ).T
    return _ensure_stereo(resampled), True


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    resolved_path = Path(path)
    with resolved_path.open("rb") as handle:
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


def _channel_harmonic_component(
    channel_audio: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    harmonic_margin: float,
    percussive_margin: float,
) -> np.ndarray:
    mono = np.asarray(channel_audio, dtype=np.float32)
    stft = librosa.stft(mono, n_fft=int(n_fft), hop_length=int(hop_length), center=True)
    magnitude = np.abs(stft).astype(np.float32)
    harmonic_mag, percussive_mag = librosa.decompose.hpss(
        magnitude,
        margin=(float(harmonic_margin), float(percussive_margin)),
    )
    harmonic_mask = harmonic_mag / (harmonic_mag + percussive_mag + 1e-8)
    harmonic = librosa.istft(
        stft * np.clip(harmonic_mask, 0.0, 1.0),
        hop_length=int(hop_length),
        length=int(mono.shape[0]),
    )
    return harmonic.astype(np.float32)


def _harmonic_component(
    audio: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    harmonic_margin: float,
    percussive_margin: float,
) -> np.ndarray:
    source = _ensure_stereo(audio)
    channels = [
        _channel_harmonic_component(
            source[:, channel_index],
            n_fft=n_fft,
            hop_length=hop_length,
            harmonic_margin=harmonic_margin,
            percussive_margin=percussive_margin,
        )
        for channel_index in range(source.shape[1])
    ]
    return np.stack(channels, axis=1).astype(np.float32)


def _remove_positive_projection(
    candidate: np.ndarray,
    suppressor: np.ndarray,
    *,
    strength: float,
    max_gain: float = 4.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    target = np.asarray(candidate, dtype=np.float32)
    reference = _fit_audio_shape(suppressor, frames=target.shape[0], channels=target.shape[1])
    denominator = float(np.sum(reference * reference))
    if denominator <= 1e-12:
        return target, {
            "gain": 0.0,
            "strength": float(strength),
            "removed_rms": 0.0,
            "skipped": "silent_suppressor",
        }
    raw_gain = float(np.sum(target * reference) / (denominator + 1e-12))
    gain = float(np.clip(max(raw_gain, 0.0), 0.0, float(max_gain)))
    removed = reference * gain * float(strength)
    cleaned = target - removed
    return cleaned.astype(np.float32), {
        "gain": round(gain, 9),
        "raw_gain": round(raw_gain, 9),
        "strength": float(strength),
        "removed_rms": _audio_summary(removed)["rms"],
    }


def build_model_stack_melody_candidate_audio(
    stems: dict[str, np.ndarray],
    *,
    source_audio: np.ndarray | None = None,
    sample_rate: int,
    harmonic_n_fft: int = 2048,
    harmonic_hop_length: int = 512,
    harmonic_margin: float = 2.8,
    percussive_margin: float = 1.2,
    source_tonal_n_fft: int = 4096,
    source_tonal_hop_length: int = 512,
    source_harmonic_margin: float = 5.0,
    source_percussive_margin: float = 10.0,
    source_percussive_suppression: float = 0.84,
    source_highpass_hz: float = 180.0,
    source_lowpass_hz: float = 8000.0,
    source_tonal_weight: float = 1.0,
    model_stack_weight: float = 0.0,
    vocal_suppression: float = 0.95,
    drum_suppression: float = 0.0,
    bass_suppression: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not stems:
        raise ValueError("no model-stack stems provided")
    reference_audio = _ensure_stereo(next(iter(stems.values())))
    frames, channels = reference_audio.shape
    fitted_stems = {
        stem_name: _fit_audio_shape(stem_audio, frames=frames, channels=channels)
        for stem_name, stem_audio in stems.items()
    }
    candidate = np.zeros((frames, channels), dtype=np.float32)
    source_contributions: dict[str, dict[str, Any]] = {}
    if source_audio is not None and float(source_tonal_weight) > 0.0:
        fitted_source = _fit_audio_shape(source_audio, frames=frames, channels=channels)
        source_harmonic, source_prior_diagnostics = build_tonal_harmonic_melody_candidate_audio(
            fitted_source,
            sample_rate=int(sample_rate),
            n_fft=int(source_tonal_n_fft),
            hop_length=int(source_tonal_hop_length),
            harmonic_margin=float(source_harmonic_margin),
            percussive_margin=float(source_percussive_margin),
            percussive_suppression=float(source_percussive_suppression),
            highpass_hz=float(source_highpass_hz),
            lowpass_hz=float(source_lowpass_hz),
        )
        weighted_source = source_harmonic * float(source_tonal_weight)
        candidate += weighted_source
        source_contributions["source_tonal_prior"] = {
            "weight": float(source_tonal_weight),
            "prior_route": source_prior_diagnostics["route_name"],
            "prior_parameters": {
                "n_fft": int(source_tonal_n_fft),
                "hop_length": int(source_tonal_hop_length),
                "harmonic_margin": float(source_harmonic_margin),
                "percussive_margin": float(source_percussive_margin),
                "percussive_suppression": float(source_percussive_suppression),
                "highpass_hz": float(source_highpass_hz),
                "lowpass_hz": float(source_lowpass_hz),
            },
            "input": _audio_summary(fitted_source),
            "harmonic": _audio_summary(source_harmonic),
            "weighted": _audio_summary(weighted_source),
        }

    stack_weight_scale = float(model_stack_weight) if source_audio is not None else 1.0
    for stem_name, weight in TARGET_SOURCE_WEIGHTS.items():
        stem_audio = fitted_stems.get(stem_name)
        if stem_audio is None:
            continue
        harmonic = _harmonic_component(
            stem_audio,
            n_fft=int(harmonic_n_fft),
            hop_length=int(harmonic_hop_length),
            harmonic_margin=float(harmonic_margin),
            percussive_margin=float(percussive_margin),
        )
        weighted = harmonic * float(weight) * stack_weight_scale
        candidate += weighted
        source_contributions[stem_name] = {
            "weight": float(weight) * stack_weight_scale,
            "base_weight": float(weight),
            "stack_weight_scale": stack_weight_scale,
            "input": _audio_summary(stem_audio),
            "harmonic": _audio_summary(harmonic),
            "weighted": _audio_summary(weighted),
        }

    projection_removal: dict[str, dict[str, Any]] = {}
    suppressor_plan = (
        ("vocals", float(vocal_suppression)),
        ("drums", float(drum_suppression)),
        ("bass", float(bass_suppression)),
    )
    for suppressor_name, suppression_strength in suppressor_plan:
        suppressor = fitted_stems.get(suppressor_name)
        if suppressor is None:
            continue
        candidate, removal_summary = _remove_positive_projection(
            candidate,
            suppressor,
            strength=suppression_strength,
        )
        projection_removal[suppressor_name] = removal_summary

    if source_audio is None:
        candidate = _harmonic_component(
            candidate,
            n_fft=int(harmonic_n_fft),
            hop_length=int(harmonic_hop_length),
            harmonic_margin=max(float(harmonic_margin), 2.0),
            percussive_margin=max(float(percussive_margin), 1.0),
        )
    candidate = _limit_peak(candidate)
    diagnostics = {
        "route_name": ROUTE_NAME,
        "sample_rate": int(sample_rate),
        "harmonic_n_fft": int(harmonic_n_fft),
        "harmonic_hop_length": int(harmonic_hop_length),
        "harmonic_margin": float(harmonic_margin),
        "percussive_margin": float(percussive_margin),
        "source_tonal_n_fft": int(source_tonal_n_fft),
        "source_tonal_hop_length": int(source_tonal_hop_length),
        "source_harmonic_margin": float(source_harmonic_margin),
        "source_percussive_margin": float(source_percussive_margin),
        "source_percussive_suppression": float(source_percussive_suppression),
        "source_highpass_hz": float(source_highpass_hz),
        "source_lowpass_hz": float(source_lowpass_hz),
        "source_tonal_weight": float(source_tonal_weight),
        "model_stack_weight": stack_weight_scale,
        "source_contributions": source_contributions,
        "projection_removal": projection_removal,
        "candidate": _audio_summary(candidate),
    }
    return candidate.astype(np.float32), diagnostics


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


def _execution_contract_from_base(base_manifest: dict[str, Any], *, device: str) -> dict[str, Any]:
    base_execution_contract = base_manifest.get("execution_contract")
    if not isinstance(base_execution_contract, dict):
        base_execution_contract = {}
    base_cuda_preflight = base_execution_contract.get("cuda_preflight")
    if not isinstance(base_cuda_preflight, dict):
        base_cuda_preflight = {}
    requested_device = str(base_cuda_preflight.get("requested_device") or device or "")
    resolved_device = str(base_cuda_preflight.get("resolved_device") or "")
    return {
        "route_name": ROUTE_NAME,
        "docker_runtime_required": True,
        "expected_acceleration_mode": str(
            base_execution_contract.get("expected_acceleration_mode")
            or "gpu_inference_then_cpu_dsp_scoring"
        ),
        "gpu_inference_required": True,
        "gpu_inference_skipped": base_execution_contract.get("gpu_inference_skipped") is True,
        "cuda_preflight": {
            "cuda_available": base_cuda_preflight.get("cuda_available") is True,
            "requested_device": requested_device,
            "resolved_device": resolved_device,
            "device_count": int(base_cuda_preflight.get("device_count") or 0),
            "device_name": str(base_cuda_preflight.get("device_name") or ""),
        },
        "base_route_name": str(base_execution_contract.get("route_name") or ""),
        "base_model_name": str(base_manifest.get("model_name") or ""),
        "base_execution_contract_inherited": bool(base_execution_contract),
    }


def _validate_no_reference_base_manifest(base_manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    contract = base_manifest.get("contract", {})
    failures: list[dict[str, Any]] = []
    for key, expected_value in REQUIRED_NO_REFERENCE_FLAGS.items():
        observed_value = contract.get(key)
        if observed_value is not expected_value:
            failures.append({"key": key, "expected": expected_value, "observed": observed_value})
    if failures:
        raise ValueError(
            "base manifest is not no-reference safe: "
            + json.dumps({"manifest_path": str(manifest_path), "failures": failures}, ensure_ascii=False)
        )
    return {
        "status": "accepted",
        "manifest_path": str(manifest_path),
        "checked_flags": sorted(REQUIRED_NO_REFERENCE_FLAGS.keys()),
    }


def _resolve_manifest_stem_path(base_dir: Path, stem_entry: dict[str, Any]) -> Path:
    raw_path = Path(str(stem_entry["path"]))
    return raw_path if raw_path.is_absolute() else base_dir / raw_path


def _load_stems_from_manifest(base_manifest_path: Path) -> tuple[dict[str, np.ndarray], int, dict[str, Any], dict[str, Any]]:
    manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    contract_check = _validate_no_reference_base_manifest(manifest, base_manifest_path)
    stems: dict[str, np.ndarray] = {}
    sample_rate: int | None = None
    base_dir = base_manifest_path.parent
    for stem_name, stem_entry in manifest.get("stems", {}).items():
        if stem_name not in {"melody", "other", "guitar", "piano", "vocals", "drums", "bass"}:
            continue
        stem_path = _resolve_manifest_stem_path(base_dir, stem_entry)
        if not stem_path.exists():
            raise FileNotFoundError(f"base stem is missing: {stem_path}")
        stem_audio, stem_sample_rate = sf.read(str(stem_path), always_2d=True)
        if sample_rate is None:
            sample_rate = int(stem_sample_rate)
        elif sample_rate != int(stem_sample_rate):
            raise ValueError(f"sample rate mismatch in base manifest stem={stem_name}: {stem_sample_rate}")
        stems[stem_name] = stem_audio.astype(np.float32)
    if sample_rate is None or not stems:
        raise ValueError(f"base manifest contains no supported stems: {base_manifest_path}")
    return stems, sample_rate, manifest, contract_check


def _base_manifest_source_sha256(base_manifest: dict[str, Any], base_manifest_path: Path) -> str:
    declared_sha256 = base_manifest.get("audio_sha256")
    if declared_sha256:
        return str(declared_sha256)
    base_audio_path_value = base_manifest.get("audio_path") or base_manifest.get("source_audio_path")
    if not base_audio_path_value:
        raise ValueError(f"base manifest source audio is not hashable: {base_manifest_path}")
    base_audio_path = Path(str(base_audio_path_value))
    if not base_audio_path.is_absolute():
        base_audio_path = base_manifest_path.parent / base_audio_path
    if not base_audio_path.exists():
        raise ValueError(
            "base manifest source audio is not hashable: "
            + json.dumps(
                {
                    "manifest_path": str(base_manifest_path),
                    "audio_path": str(base_audio_path),
                },
                ensure_ascii=False,
            )
        )
    return sha256_file(base_audio_path)


def _ensure_base_model_runtime(model_name: str) -> None:
    if not str(model_name).startswith("hybrid_"):
        return
    from export_hq_stems_full import _ensure_hybrid_bs_runtime

    _ensure_hybrid_bs_runtime()


def _export_base_stems_aggressive_melody_first(
    source_audio_path: Path,
    base_output_dir: Path,
    *,
    model_name: str,
    device: str,
    sharpen_bleed: bool,
) -> dict[str, Any]:
    from worker.high_quality_stem_export import export_track_stems_aggressive_melody_first

    return export_track_stems_aggressive_melody_first(
        source_audio_path,
        base_output_dir,
        model_name=model_name,
        device=device,
        sharpen_bleed=sharpen_bleed,
    )


def _build_base_manifest_from_source(
    *,
    source_audio_path: Path,
    output_dir: Path,
    model_name: str,
    device: str,
) -> Path:
    base_output_dir = output_dir / "_base_model_stack"
    _ensure_base_model_runtime(model_name)
    _export_base_stems_aggressive_melody_first(
        source_audio_path,
        base_output_dir,
        model_name=model_name,
        device=device,
        sharpen_bleed=True,
    )
    return base_output_dir / "stem_manifest.json"


def build_model_stack_melody_candidate(
    *,
    source_audio_path: str | Path,
    output_dir: str | Path,
    base_manifest_path: str | Path | None = None,
    listen_dir: str | Path | None = None,
    model_name: str = "htdemucs_6s",
    device: str = "cuda",
    harmonic_n_fft: int = 2048,
    harmonic_hop_length: int = 512,
    harmonic_margin: float = 2.8,
    percussive_margin: float = 1.2,
    source_tonal_n_fft: int = 4096,
    source_tonal_hop_length: int = 512,
    source_harmonic_margin: float = 5.0,
    source_percussive_margin: float = 10.0,
    source_percussive_suppression: float = 0.84,
    source_highpass_hz: float = 180.0,
    source_lowpass_hz: float = 8000.0,
    source_tonal_weight: float = 1.0,
    model_stack_weight: float = 0.0,
    vocal_suppression: float = 0.95,
    drum_suppression: float = 0.0,
    bass_suppression: float = 0.0,
) -> dict[str, Any]:
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

    candidate, diagnostics = build_model_stack_melody_candidate_audio(
        stems,
        source_audio=source_audio.astype(np.float32),
        sample_rate=int(stem_sample_rate),
        harmonic_n_fft=int(harmonic_n_fft),
        harmonic_hop_length=int(harmonic_hop_length),
        harmonic_margin=float(harmonic_margin),
        percussive_margin=float(percussive_margin),
        source_tonal_n_fft=int(source_tonal_n_fft),
        source_tonal_hop_length=int(source_tonal_hop_length),
        source_harmonic_margin=float(source_harmonic_margin),
        source_percussive_margin=float(source_percussive_margin),
        source_percussive_suppression=float(source_percussive_suppression),
        source_highpass_hz=float(source_highpass_hz),
        source_lowpass_hz=float(source_lowpass_hz),
        source_tonal_weight=float(source_tonal_weight),
        model_stack_weight=float(model_stack_weight),
        vocal_suppression=float(vocal_suppression),
        drum_suppression=float(drum_suppression),
        bass_suppression=float(bass_suppression),
    )

    melody_path = resolved_output_dir / "melody.flac"
    sf.write(str(melody_path), candidate, int(stem_sample_rate), subtype=OUTPUT_SUBTYPE)
    manifest = {
        "audio_path": str(source_path),
        "audio_sha256": source_sha256,
        "model_name": ROUTE_NAME,
        "sample_rate": int(stem_sample_rate),
        "export_subtype": OUTPUT_SUBTYPE,
        "melody_stem_included": True,
        "target_stem": "melody",
        "target_stem_artifact": {
            "target_stem": "melody",
            "path": str(melody_path),
            "source": "model_stack_melody_candidate",
            "source_stems": ["melody"],
        },
        "delivery_allowed": False,
        "presentation_status": "diagnostic_candidate_not_for_listen_pack",
        "release_status": "not_release_ready",
        "release_ready": False,
        "contract": _no_reference_contract(),
        "execution_contract": _execution_contract_from_base(base_manifest, device=str(device)),
        "pipeline": {
            "mode": ROUTE_NAME,
            "source": "authoritative_full_mix_plus_no_reference_model_stack",
            "base_manifest_path": str(resolved_base_manifest_path),
            "base_manifest_model_name": base_manifest.get("model_name"),
            "base_manifest_contract_checked": contract_check["status"] == "accepted",
            "base_manifest_contract": contract_check,
            "source_sample_rate": int(source_sample_rate),
            "source_resampled_to_sample_rate": int(stem_sample_rate),
            "source_resampled": bool(source_resampled),
            "target_reference_read": False,
            "reference_sum_mix_read": False,
            "postprocess": "model_stack_harmonic_candidate_with_projection_debleed_v1",
            "diagnostics": diagnostics,
            "source_audio": _audio_summary(source_audio.astype(np.float32)),
        },
        "delivery": {
            "status": "requires_objective_quality_gate",
            "reason": "no_reference_model_stack_candidate_requires_ground_truth_benchmark_before_delivery",
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
        candidate_listen_path = resolved_listen_dir / "01_model_stack_melody_v1.flac"
        shutil.copyfile(source_path, source_listen_path)
        shutil.copyfile(melody_path, candidate_listen_path)
        (resolved_listen_dir / "00_README.txt").write_text(
            "Diagnostic listen pack only.\n"
            "00 = authoritative source mix.\n"
            "01 = no-reference model-stack melody candidate; must pass objective gate before delivery.\n",
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
    parser = argparse.ArgumentParser(description="Build a no-reference model-stack melody candidate.")
    parser.add_argument("source_audio_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--model-name", default="htdemucs_6s")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--harmonic-n-fft", type=int, default=2048)
    parser.add_argument("--harmonic-hop-length", type=int, default=512)
    parser.add_argument("--harmonic-margin", type=float, default=2.8)
    parser.add_argument("--percussive-margin", type=float, default=1.2)
    parser.add_argument("--source-tonal-n-fft", type=int, default=4096)
    parser.add_argument("--source-tonal-hop-length", type=int, default=512)
    parser.add_argument("--source-harmonic-margin", type=float, default=5.0)
    parser.add_argument("--source-percussive-margin", type=float, default=10.0)
    parser.add_argument("--source-percussive-suppression", type=float, default=0.84)
    parser.add_argument("--source-highpass-hz", type=float, default=180.0)
    parser.add_argument("--source-lowpass-hz", type=float, default=8000.0)
    parser.add_argument("--source-tonal-weight", type=float, default=1.0)
    parser.add_argument("--model-stack-weight", type=float, default=0.0)
    parser.add_argument("--vocal-suppression", type=float, default=0.95)
    parser.add_argument("--drum-suppression", type=float, default=0.0)
    parser.add_argument("--bass-suppression", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="build_model_stack_melody_candidate.py",
        entrypoint_path="/app/tools/build_model_stack_melody_candidate.py",
    )
    args = parse_args()
    manifest = build_model_stack_melody_candidate(
        source_audio_path=args.source_audio_path,
        output_dir=args.output_dir,
        base_manifest_path=args.base_manifest,
        listen_dir=args.listen_dir,
        model_name=args.model_name,
        device=args.device,
        harmonic_n_fft=args.harmonic_n_fft,
        harmonic_hop_length=args.harmonic_hop_length,
        harmonic_margin=args.harmonic_margin,
        percussive_margin=args.percussive_margin,
        source_tonal_n_fft=args.source_tonal_n_fft,
        source_tonal_hop_length=args.source_tonal_hop_length,
        source_harmonic_margin=args.source_harmonic_margin,
        source_percussive_margin=args.source_percussive_margin,
        source_percussive_suppression=args.source_percussive_suppression,
        source_highpass_hz=args.source_highpass_hz,
        source_lowpass_hz=args.source_lowpass_hz,
        source_tonal_weight=args.source_tonal_weight,
        model_stack_weight=args.model_stack_weight,
        vocal_suppression=args.vocal_suppression,
        drum_suppression=args.drum_suppression,
        bass_suppression=args.bass_suppression,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
