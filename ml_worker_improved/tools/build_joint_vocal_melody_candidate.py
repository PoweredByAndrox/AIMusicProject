from __future__ import annotations

import argparse
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
from build_model_stack_melody_candidate import (  # noqa: E402
    _audio_summary,
    _base_manifest_source_sha256,
    _build_base_manifest_from_source,
    _fit_audio_shape,
    _limit_peak,
    _load_stems_from_manifest,
    _remove_positive_projection,
    _resample_stereo_audio,
    build_model_stack_melody_candidate_audio,
    sha256_file,
)

ROUTE_NAME = "joint_vocal_melody_v1"
OUTPUT_SUBTYPE = "PCM_24"
MELODY_RESCUE_PROJECTION_RATIO_FLOOR = 0.45
MELODY_RESCUE_WEIGHT = 0.4


def _spectral_dominance_filter(
    primary: np.ndarray,
    secondary: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    dominance_power: float = 5.0,
    mask_floor: float = 0.005,
) -> np.ndarray:
    primary_audio = np.asarray(primary, dtype=np.float32)
    secondary_audio = _fit_audio_shape(
        secondary,
        frames=int(primary_audio.shape[0]),
        channels=int(primary_audio.shape[1]) if primary_audio.ndim == 2 else 2,
    )
    channels: list[np.ndarray] = []
    for channel_index in range(primary_audio.shape[1]):
        primary_stft = librosa.stft(
            primary_audio[:, channel_index],
            n_fft=int(n_fft),
            hop_length=int(hop_length),
            center=True,
        )
        secondary_stft = librosa.stft(
            secondary_audio[:, channel_index],
            n_fft=int(n_fft),
            hop_length=int(hop_length),
            center=True,
        )
        primary_mag = np.abs(primary_stft).astype(np.float32)
        secondary_mag = np.abs(secondary_stft).astype(np.float32)
        primary_power = np.power(primary_mag, float(dominance_power))
        secondary_power = np.power(secondary_mag, float(dominance_power))
        mask = primary_power / (primary_power + secondary_power + 1e-8)
        mask = np.clip(mask, float(mask_floor), 1.0).astype(np.float32)
        filtered = librosa.istft(
            primary_stft * mask,
            hop_length=int(hop_length),
            length=int(primary_audio.shape[0]),
        )
        channels.append(filtered.astype(np.float32))
    return np.stack(channels, axis=1).astype(np.float32)


def _projection_rms_ratio(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_audio = np.asarray(candidate, dtype=np.float32)
    reference_audio = np.asarray(reference, dtype=np.float32)
    frame_count = min(int(candidate_audio.shape[0]), int(reference_audio.shape[0]))
    if frame_count <= 0:
        return 0.0
    fitted_candidate = candidate_audio[:frame_count]
    fitted_reference = reference_audio[:frame_count]
    reference_energy = float(np.sum(fitted_reference * fitted_reference))
    if reference_energy <= 1e-12:
        return 0.0
    projection_gain = float(np.sum(fitted_candidate * fitted_reference) / (reference_energy + 1e-12))
    projected = fitted_reference * projection_gain
    projected_rms = float(np.sqrt(np.mean(projected * projected) + 1e-12))
    reference_rms = float(np.sqrt(np.mean(fitted_reference * fitted_reference) + 1e-12))
    return projected_rms / (reference_rms + 1e-12)


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
            "requested_device": str(base_cuda_preflight.get("requested_device") or device or ""),
            "resolved_device": str(base_cuda_preflight.get("resolved_device") or ""),
            "device_count": int(base_cuda_preflight.get("device_count") or 0),
            "device_name": str(base_cuda_preflight.get("device_name") or ""),
        },
        "base_route_name": str(base_execution_contract.get("route_name") or ""),
        "base_model_name": str(base_manifest.get("model_name") or ""),
        "base_execution_contract_inherited": bool(base_execution_contract),
    }


def build_joint_vocal_melody_candidate_audio(
    stems: dict[str, np.ndarray],
    *,
    source_audio: np.ndarray | None,
    sample_rate: int,
    harmonic_n_fft: int = 2048,
    harmonic_hop_length: int = 512,
    source_tonal_n_fft: int = 4096,
    source_tonal_hop_length: int = 512,
    vocal_from_melody_suppression: float = 0.88,
    melody_from_vocal_suppression: float = 0.95,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not stems:
        raise ValueError("no model-stack stems provided")
    reference_audio = next(iter(stems.values()))
    frames = int(reference_audio.shape[0])
    channels = int(reference_audio.shape[1]) if reference_audio.ndim == 2 else 2
    fitted_stems = {
        stem_name: _fit_audio_shape(stem_audio, frames=frames, channels=channels)
        for stem_name, stem_audio in stems.items()
    }
    if source_audio is not None:
        fitted_source = _fit_audio_shape(source_audio, frames=frames, channels=channels)
    else:
        fitted_source = None

    raw_vocals = fitted_stems.get("vocals")
    if raw_vocals is None:
        raw_vocals = np.zeros((frames, channels), dtype=np.float32)

    raw_melody = fitted_stems.get("melody")
    if raw_melody is None:
        raw_melody = np.zeros((frames, channels), dtype=np.float32)

    vocal_dominant = _spectral_dominance_filter(
        raw_vocals,
        raw_melody,
        n_fft=int(harmonic_n_fft),
        hop_length=int(harmonic_hop_length),
    )
    melody_dominant = _spectral_dominance_filter(
        raw_melody,
        raw_vocals,
        n_fft=int(harmonic_n_fft),
        hop_length=int(harmonic_hop_length),
    )

    source_melody_candidate, melody_diagnostics = build_model_stack_melody_candidate_audio(
        fitted_stems,
        source_audio=fitted_source,
        sample_rate=int(sample_rate),
        harmonic_n_fft=int(harmonic_n_fft),
        harmonic_hop_length=int(harmonic_hop_length),
        source_tonal_n_fft=int(source_tonal_n_fft),
        source_tonal_hop_length=int(source_tonal_hop_length),
        vocal_suppression=float(melody_from_vocal_suppression),
        drum_suppression=0.0,
        bass_suppression=0.0,
    )
    source_to_raw_melody_projection_ratio = _projection_rms_ratio(source_melody_candidate, raw_melody)
    dominance_rescue_weight = (
        MELODY_RESCUE_WEIGHT
        if source_to_raw_melody_projection_ratio < MELODY_RESCUE_PROJECTION_RATIO_FLOOR
        else 0.0
    )
    melody_candidate = _limit_peak(
        source_melody_candidate * (1.0 - dominance_rescue_weight)
        + melody_dominant * dominance_rescue_weight
    )

    vocals_candidate, vocal_melody_removal = _remove_positive_projection(
        vocal_dominant,
        melody_candidate,
        strength=float(vocal_from_melody_suppression),
    )
    vocals_candidate, vocal_model_melody_removal = _remove_positive_projection(
        vocals_candidate,
        raw_melody,
        strength=float(vocal_from_melody_suppression) * 0.35,
    )
    melody_candidate, melody_vocal_removal = _remove_positive_projection(
        melody_candidate,
        vocals_candidate,
        strength=float(melody_from_vocal_suppression) * 0.1,
    )

    vocals_candidate = _limit_peak(vocals_candidate)
    melody_candidate = _limit_peak(melody_candidate)
    diagnostics = {
        "route_name": ROUTE_NAME,
        "sample_rate": int(sample_rate),
        "melody_builder": melody_diagnostics,
        "melody_selection": {
            "selected_source": "model_stack_melody_candidate_with_vocal_suppression",
            "source_to_raw_melody_projection_ratio": round(float(source_to_raw_melody_projection_ratio), 6),
            "raw_melody_dominance_mix_weight": round(float(dominance_rescue_weight), 6),
            "post_vocal_projection_strength": round(float(melody_from_vocal_suppression) * 0.1, 6),
        },
        "spectral_dominance": {
            "vocals": _audio_summary(vocal_dominant),
            "melody": _audio_summary(melody_dominant),
        },
        "cross_debleed": {
            "vocals_from_melody": vocal_melody_removal,
            "vocals_from_model_melody_prior": vocal_model_melody_removal,
            "melody_from_vocals": melody_vocal_removal,
        },
        "raw": {
            "vocals": _audio_summary(raw_vocals),
            "melody": _audio_summary(raw_melody),
        },
        "candidate": {
            "vocals": _audio_summary(vocals_candidate),
            "melody": _audio_summary(melody_candidate),
        },
    }
    return {
        "vocals": vocals_candidate.astype(np.float32),
        "melody": melody_candidate.astype(np.float32),
    }, diagnostics


def build_joint_vocal_melody_candidate(
    *,
    source_audio_path: str | Path,
    output_dir: str | Path,
    base_manifest_path: str | Path | None = None,
    listen_dir: str | Path | None = None,
    model_name: str = "hybrid_htdemucs_ft_bs_roformer_6s",
    device: str = "cuda",
    harmonic_n_fft: int = 2048,
    harmonic_hop_length: int = 512,
    source_tonal_n_fft: int = 4096,
    source_tonal_hop_length: int = 512,
    vocal_from_melody_suppression: float = 0.88,
    melody_from_vocal_suppression: float = 0.95,
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

    outputs, diagnostics = build_joint_vocal_melody_candidate_audio(
        stems,
        source_audio=source_audio.astype(np.float32),
        sample_rate=int(stem_sample_rate),
        harmonic_n_fft=int(harmonic_n_fft),
        harmonic_hop_length=int(harmonic_hop_length),
        source_tonal_n_fft=int(source_tonal_n_fft),
        source_tonal_hop_length=int(source_tonal_hop_length),
        vocal_from_melody_suppression=float(vocal_from_melody_suppression),
        melody_from_vocal_suppression=float(melody_from_vocal_suppression),
    )

    stem_paths: dict[str, Path] = {}
    for stem_name, stem_audio in outputs.items():
        stem_path = resolved_output_dir / f"{stem_name}.flac"
        sf.write(str(stem_path), stem_audio, int(stem_sample_rate), subtype=OUTPUT_SUBTYPE)
        stem_paths[stem_name] = stem_path

    manifest = {
        "audio_path": str(source_path),
        "audio_sha256": source_sha256,
        "model_name": ROUTE_NAME,
        "sample_rate": int(stem_sample_rate),
        "export_subtype": OUTPUT_SUBTYPE,
        "melody_stem_included": True,
        "vocals_stem_included": True,
        "contract": _no_reference_contract(),
        "execution_contract": _execution_contract_from_base(base_manifest, device=str(device)),
        "pipeline": {
            "mode": ROUTE_NAME,
            "source": "authoritative_full_mix_plus_no_reference_joint_vocal_melody_stack",
            "base_manifest_path": str(resolved_base_manifest_path),
            "base_manifest_model_name": base_manifest.get("model_name"),
            "base_manifest_contract_checked": contract_check["status"] == "accepted",
            "base_manifest_contract": contract_check,
            "source_sample_rate": int(source_sample_rate),
            "source_resampled_to_sample_rate": int(stem_sample_rate),
            "source_resampled": bool(source_resampled),
            "target_reference_read": False,
            "reference_sum_mix_read": False,
            "postprocess": "joint_vocal_melody_cross_projection_debleed_v1",
            "diagnostics": diagnostics,
            "source_audio": _audio_summary(source_audio.astype(np.float32)),
        },
        "delivery": {
            "status": "requires_objective_quality_gate",
            "reason": "joint_no_reference_vocal_melody_candidate_requires_ground_truth_benchmark_before_delivery",
        },
        "stems": {
            stem_name: {
                "path": str(stem_path),
                **_audio_summary(outputs[stem_name]),
            }
            for stem_name, stem_path in stem_paths.items()
        },
    }

    if listen_dir is not None:
        resolved_listen_dir = Path(listen_dir).resolve()
        resolved_listen_dir.mkdir(parents=True, exist_ok=True)
        source_listen_path = resolved_listen_dir / "00_source_mix.flac"
        vocals_listen_path = resolved_listen_dir / "01_joint_vocals.flac"
        melody_listen_path = resolved_listen_dir / "02_joint_melody.flac"
        shutil.copyfile(source_path, source_listen_path)
        shutil.copyfile(stem_paths["vocals"], vocals_listen_path)
        shutil.copyfile(stem_paths["melody"], melody_listen_path)
        (resolved_listen_dir / "00_README.txt").write_text(
            "Diagnostic listen pack only.\n"
            "00 = authoritative source mix.\n"
            "01 = no-reference joint vocals candidate.\n"
            "02 = no-reference joint melody candidate.\n"
            "Both stems must pass objective gate before delivery.\n",
            encoding="utf-8",
        )
        manifest["listen"] = {
            "listen_dir": str(resolved_listen_dir),
            "source_mix": str(source_listen_path),
            "vocals": str(vocals_listen_path),
            "melody": str(melody_listen_path),
            "status": "diagnostic_requires_objective_gate",
        }

    manifest_path = resolved_output_dir / "stem_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build no-reference joint vocals+melody candidates.")
    parser.add_argument("source_audio_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--model-name", default="hybrid_htdemucs_ft_bs_roformer_6s")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--harmonic-n-fft", type=int, default=2048)
    parser.add_argument("--harmonic-hop-length", type=int, default=512)
    parser.add_argument("--source-tonal-n-fft", type=int, default=4096)
    parser.add_argument("--source-tonal-hop-length", type=int, default=512)
    parser.add_argument("--vocal-from-melody-suppression", type=float, default=0.88)
    parser.add_argument("--melody-from-vocal-suppression", type=float, default=0.95)
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="build_joint_vocal_melody_candidate.py",
        entrypoint_path="/app/tools/build_joint_vocal_melody_candidate.py",
    )
    args = parse_args()
    manifest = build_joint_vocal_melody_candidate(
        source_audio_path=args.source_audio_path,
        output_dir=args.output_dir,
        base_manifest_path=args.base_manifest,
        listen_dir=args.listen_dir,
        model_name=args.model_name,
        device=args.device,
        harmonic_n_fft=args.harmonic_n_fft,
        harmonic_hop_length=args.harmonic_hop_length,
        source_tonal_n_fft=args.source_tonal_n_fft,
        source_tonal_hop_length=args.source_tonal_hop_length,
        vocal_from_melody_suppression=args.vocal_from_melody_suppression,
        melody_from_vocal_suppression=args.melody_from_vocal_suppression,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
