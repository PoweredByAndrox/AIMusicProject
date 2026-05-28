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
from build_joint_vocal_melody_candidate import (  # noqa: E402
    _projection_rms_ratio,
    _spectral_dominance_filter,
)
from build_model_stack_melody_candidate import (  # noqa: E402
    _audio_summary,
    _base_manifest_source_sha256,
    _build_base_manifest_from_source,
    _fit_audio_shape,
    _limit_peak,
    _load_stems_from_manifest,
    _remove_positive_projection,
    build_model_stack_melody_candidate_audio,
    sha256_file,
)

ROUTE_NAME = "joint_vocal_melody_v2"
OUTPUT_SUBTYPE = "PCM_24"


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


def _silent_like(reference_audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(reference_audio, dtype=np.float32)
    channels = int(audio.shape[1]) if audio.ndim == 2 else 2
    return np.zeros((int(audio.shape[0]), channels), dtype=np.float32)


def _source_pool_without_non_targets(
    source_audio: np.ndarray,
    stems: dict[str, np.ndarray],
    *,
    non_target_suppression: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    pool = np.asarray(source_audio, dtype=np.float32)
    removals: dict[str, Any] = {}
    for stem_name in ("drums", "bass"):
        suppressor = stems.get(stem_name)
        if suppressor is None:
            continue
        pool, removal = _remove_positive_projection(
            pool,
            suppressor,
            strength=float(non_target_suppression),
        )
        removals[stem_name] = removal
    return pool.astype(np.float32), {
        "enabled": True,
        "non_target_suppression": float(non_target_suppression),
        "removals": removals,
        "source_pool": _audio_summary(pool),
    }


def _spectral_source_partition(
    source_audio: np.ndarray,
    primary_prior: np.ndarray,
    secondary_prior: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    mask_power: float = 2.0,
    mask_floor: float = 0.0,
) -> np.ndarray:
    source = np.asarray(source_audio, dtype=np.float32)
    primary = _fit_audio_shape(primary_prior, frames=source.shape[0], channels=source.shape[1])
    secondary = _fit_audio_shape(secondary_prior, frames=source.shape[0], channels=source.shape[1])
    channels: list[np.ndarray] = []
    for channel_index in range(source.shape[1]):
        source_stft = librosa.stft(
            source[:, channel_index],
            n_fft=int(n_fft),
            hop_length=int(hop_length),
            center=True,
        )
        primary_stft = librosa.stft(
            primary[:, channel_index],
            n_fft=int(n_fft),
            hop_length=int(hop_length),
            center=True,
        )
        secondary_stft = librosa.stft(
            secondary[:, channel_index],
            n_fft=int(n_fft),
            hop_length=int(hop_length),
            center=True,
        )
        primary_power = np.power(np.abs(primary_stft).astype(np.float32), float(mask_power))
        secondary_power = np.power(np.abs(secondary_stft).astype(np.float32), float(mask_power))
        mask = primary_power / (primary_power + secondary_power + 1e-8)
        mask = np.clip(mask, float(mask_floor), 1.0).astype(np.float32)
        partition = librosa.istft(
            source_stft * mask,
            hop_length=int(hop_length),
            length=int(source.shape[0]),
        )
        channels.append(partition.astype(np.float32))
    return np.stack(channels, axis=1).astype(np.float32)


def _remove_non_target_projection(
    candidate: np.ndarray,
    stems: dict[str, np.ndarray],
    *,
    strength: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    cleaned = np.asarray(candidate, dtype=np.float32)
    removals: dict[str, Any] = {}
    for stem_name in ("drums", "bass"):
        suppressor = stems.get(stem_name)
        if suppressor is None:
            continue
        cleaned, removal = _remove_positive_projection(
            cleaned,
            suppressor,
            strength=float(strength),
        )
        removals[stem_name] = removal
    return cleaned.astype(np.float32), removals


def build_joint_vocal_melody_v2_candidate_audio(
    stems: dict[str, np.ndarray],
    *,
    source_audio: np.ndarray | None,
    sample_rate: int,
    harmonic_n_fft: int = 2048,
    harmonic_hop_length: int = 512,
    source_tonal_n_fft: int = 4096,
    source_tonal_hop_length: int = 512,
    vocal_from_melody_suppression: float = 0.92,
    melody_from_vocal_suppression: float = 0.94,
    non_target_suppression: float = 0.85,
    vocal_pool_weight: float = 0.0,
    melody_pool_weight: float = 0.30,
    vocal_non_target_suppression: float = 0.0,
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
    fitted_source = (
        _fit_audio_shape(source_audio, frames=frames, channels=channels)
        if source_audio is not None
        else None
    )
    raw_vocals = fitted_stems.get("vocals")
    if raw_vocals is None:
        raw_vocals = _silent_like(reference_audio)
    raw_melody = fitted_stems.get("melody")
    if raw_melody is None:
        raw_melody = _silent_like(reference_audio)

    source_melody_candidate, melody_builder_diagnostics = build_model_stack_melody_candidate_audio(
        fitted_stems,
        source_audio=fitted_source,
        sample_rate=int(sample_rate),
        harmonic_n_fft=int(harmonic_n_fft),
        harmonic_hop_length=int(harmonic_hop_length),
        source_tonal_n_fft=int(source_tonal_n_fft),
        source_tonal_hop_length=int(source_tonal_hop_length),
        vocal_suppression=float(melody_from_vocal_suppression),
        drum_suppression=float(non_target_suppression) * 0.35,
        bass_suppression=float(non_target_suppression) * 0.35,
    )

    vocal_secondary = raw_melody + source_melody_candidate
    vocal_prior = _spectral_dominance_filter(
        raw_vocals,
        vocal_secondary,
        n_fft=int(harmonic_n_fft),
        hop_length=int(harmonic_hop_length),
        dominance_power=5.0,
        mask_floor=0.002,
    )
    vocal_prior, vocal_model_melody_removal = _remove_positive_projection(
        vocal_prior,
        source_melody_candidate,
        strength=float(vocal_from_melody_suppression),
    )
    vocal_prior, vocal_raw_melody_removal = _remove_positive_projection(
        vocal_prior,
        raw_melody,
        strength=float(vocal_from_melody_suppression) * 0.35,
    )

    melody_seed = source_melody_candidate * 0.75 + raw_melody * 0.25
    melody_prior = _spectral_dominance_filter(
        melody_seed,
        vocal_prior,
        n_fft=int(harmonic_n_fft),
        hop_length=int(harmonic_hop_length),
        dominance_power=4.0,
        mask_floor=0.0,
    )
    melody_prior, melody_vocal_removal = _remove_positive_projection(
        melody_prior,
        vocal_prior,
        strength=float(melody_from_vocal_suppression),
    )
    melody_prior, melody_non_target_removals = _remove_non_target_projection(
        melody_prior,
        fitted_stems,
        strength=float(non_target_suppression),
    )

    if fitted_source is not None:
        source_pool, source_pool_diagnostics = _source_pool_without_non_targets(
            fitted_source,
            fitted_stems,
            non_target_suppression=float(non_target_suppression),
        )
        vocal_partition = _spectral_source_partition(
            source_pool,
            vocal_prior,
            melody_prior,
            n_fft=int(harmonic_n_fft),
            hop_length=int(harmonic_hop_length),
            mask_power=2.0,
        )
        melody_partition = _spectral_source_partition(
            source_pool,
            melody_prior,
            vocal_prior,
            n_fft=int(harmonic_n_fft),
            hop_length=int(harmonic_hop_length),
            mask_power=2.0,
        )
        spectral_partition = {
            "enabled": True,
            "vocal_pool_weight": float(vocal_pool_weight),
            "melody_pool_weight": float(melody_pool_weight),
            "vocal_non_target_suppression": float(vocal_non_target_suppression),
            "vocals": _audio_summary(vocal_partition),
            "melody": _audio_summary(melody_partition),
        }
        vocals_candidate = vocal_prior * (1.0 - float(vocal_pool_weight)) + vocal_partition * float(vocal_pool_weight)
        melody_candidate = melody_prior * (1.0 - float(melody_pool_weight)) + melody_partition * float(melody_pool_weight)
    else:
        source_pool_diagnostics = {"enabled": False, "reason": "source_audio_missing"}
        spectral_partition = {"enabled": False, "reason": "source_audio_missing"}
        vocals_candidate = vocal_prior
        melody_candidate = melody_prior

    vocals_candidate, final_vocal_melody_removal = _remove_positive_projection(
        vocals_candidate,
        melody_candidate,
        strength=float(vocal_from_melody_suppression),
    )
    melody_candidate, final_melody_vocal_removal = _remove_positive_projection(
        melody_candidate,
        vocals_candidate,
        strength=float(melody_from_vocal_suppression),
    )
    vocals_candidate, vocal_non_target_removals = _remove_non_target_projection(
        vocals_candidate,
        fitted_stems,
        strength=float(vocal_non_target_suppression),
    )
    melody_candidate, final_melody_non_target_removals = _remove_non_target_projection(
        melody_candidate,
        fitted_stems,
        strength=float(non_target_suppression),
    )

    vocals_candidate = _limit_peak(vocals_candidate)
    melody_candidate = _limit_peak(melody_candidate)
    diagnostics = {
        "route_name": ROUTE_NAME,
        "sample_rate": int(sample_rate),
        "melody_builder": melody_builder_diagnostics,
        "source_pool_suppression": source_pool_diagnostics,
        "spectral_partition": spectral_partition,
        "projection_ratios": {
            "source_melody_to_raw_melody": round(
                float(_projection_rms_ratio(source_melody_candidate, raw_melody)),
                6,
            ),
            "vocal_prior_to_raw_vocals": round(float(_projection_rms_ratio(vocal_prior, raw_vocals)), 6),
            "melody_prior_to_raw_melody": round(float(_projection_rms_ratio(melody_prior, raw_melody)), 6),
        },
        "projection_debleed": {
            "vocals_from_model_melody": vocal_model_melody_removal,
            "vocals_from_raw_melody": vocal_raw_melody_removal,
            "melody_from_vocals": melody_vocal_removal,
            "melody_from_non_targets": melody_non_target_removals,
            "final_vocals_from_melody": final_vocal_melody_removal,
            "final_melody_from_vocals": final_melody_vocal_removal,
            "final_vocals_from_non_targets": vocal_non_target_removals,
            "final_melody_from_non_targets": final_melody_non_target_removals,
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


def build_joint_vocal_melody_v2_candidate(
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
    vocal_from_melody_suppression: float = 0.92,
    melody_from_vocal_suppression: float = 0.94,
    non_target_suppression: float = 0.85,
    vocal_pool_weight: float = 0.0,
    melody_pool_weight: float = 0.30,
    vocal_non_target_suppression: float = 0.0,
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
    if int(source_sample_rate) != int(stem_sample_rate):
        raise ValueError(
            f"source/base sample-rate mismatch: source={source_sample_rate}, base={stem_sample_rate}"
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

    outputs, diagnostics = build_joint_vocal_melody_v2_candidate_audio(
        stems,
        source_audio=source_audio.astype(np.float32),
        sample_rate=int(stem_sample_rate),
        harmonic_n_fft=int(harmonic_n_fft),
        harmonic_hop_length=int(harmonic_hop_length),
        source_tonal_n_fft=int(source_tonal_n_fft),
        source_tonal_hop_length=int(source_tonal_hop_length),
        vocal_from_melody_suppression=float(vocal_from_melody_suppression),
        melody_from_vocal_suppression=float(melody_from_vocal_suppression),
        non_target_suppression=float(non_target_suppression),
        vocal_pool_weight=float(vocal_pool_weight),
        melody_pool_weight=float(melody_pool_weight),
        vocal_non_target_suppression=float(vocal_non_target_suppression),
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
        "pipeline": {
            "mode": ROUTE_NAME,
            "source": "authoritative_full_mix_plus_no_reference_source_pool_joint_vocal_melody_stack",
            "base_manifest_path": str(resolved_base_manifest_path),
            "base_manifest_model_name": base_manifest.get("model_name"),
            "base_manifest_contract_checked": contract_check["status"] == "accepted",
            "base_manifest_contract": contract_check,
            "target_reference_read": False,
            "reference_sum_mix_read": False,
            "postprocess": "joint_vocal_melody_source_pool_partition_debleed_v2",
            "parameters": {
                "harmonic_n_fft": int(harmonic_n_fft),
                "harmonic_hop_length": int(harmonic_hop_length),
                "source_tonal_n_fft": int(source_tonal_n_fft),
                "source_tonal_hop_length": int(source_tonal_hop_length),
                "vocal_from_melody_suppression": float(vocal_from_melody_suppression),
                "melody_from_vocal_suppression": float(melody_from_vocal_suppression),
                "non_target_suppression": float(non_target_suppression),
                "vocal_pool_weight": float(vocal_pool_weight),
                "melody_pool_weight": float(melody_pool_weight),
                "vocal_non_target_suppression": float(vocal_non_target_suppression),
            },
            "diagnostics": diagnostics,
            "source_audio": _audio_summary(source_audio.astype(np.float32)),
        },
        "delivery": {
            "status": "requires_objective_quality_gate",
            "reason": "joint_no_reference_vocal_melody_v2_candidate_requires_ground_truth_benchmark_before_delivery",
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
        vocals_listen_path = resolved_listen_dir / "01_joint_v2_vocals.flac"
        melody_listen_path = resolved_listen_dir / "02_joint_v2_melody.flac"
        shutil.copyfile(source_path, source_listen_path)
        shutil.copyfile(stem_paths["vocals"], vocals_listen_path)
        shutil.copyfile(stem_paths["melody"], melody_listen_path)
        (resolved_listen_dir / "00_README.txt").write_text(
            "Diagnostic listen pack only.\n"
            "00 = authoritative source mix.\n"
            "01 = no-reference joint v2 vocals candidate.\n"
            "02 = no-reference joint v2 melody candidate.\n"
            "Both stems must pass objective gate and exact-reference audit before delivery.\n",
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
    parser = argparse.ArgumentParser(description="Build no-reference joint vocals+melody v2 candidates.")
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
    parser.add_argument("--vocal-from-melody-suppression", type=float, default=0.92)
    parser.add_argument("--melody-from-vocal-suppression", type=float, default=0.94)
    parser.add_argument("--non-target-suppression", type=float, default=0.85)
    parser.add_argument("--vocal-pool-weight", type=float, default=0.0)
    parser.add_argument("--melody-pool-weight", type=float, default=0.30)
    parser.add_argument("--vocal-non-target-suppression", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="build_joint_vocal_melody_candidate_v2.py",
        entrypoint_path="/app/tools/build_joint_vocal_melody_candidate_v2.py",
    )
    args = parse_args()
    manifest = build_joint_vocal_melody_v2_candidate(
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
        non_target_suppression=args.non_target_suppression,
        vocal_pool_weight=args.vocal_pool_weight,
        melody_pool_weight=args.melody_pool_weight,
        vocal_non_target_suppression=args.vocal_non_target_suppression,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
