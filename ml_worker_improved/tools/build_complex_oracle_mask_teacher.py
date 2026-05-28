from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import librosa
import numpy as np
import soundfile as sf

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from build_stem_delta_profile_repair import (  # noqa: E402
    audio_file_summary,
    fit_length,
    read_json,
    read_stereo_audio,
    resolve_ground_truth_mix_path,
    resolve_stem_path,
    sha256_file,
    write_json,
)
from docker_runtime import require_docker_runtime  # noqa: E402


MODEL_ID = "complex_oracle_mask_teacher_v1"
OUTPUT_SUBTYPE = "PCM_24"
EPSILON = 1.0e-9


def _dbfs_from_rms(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2) + 1.0e-18))
    return float(20.0 * np.log10(max(rms, 1.0e-12)))


def _stft_channel(audio: np.ndarray, *, n_fft: int, hop_length: int) -> np.ndarray:
    return librosa.stft(
        np.asarray(audio, dtype=np.float32),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        center=True,
    )


def _istft_channel(stft: np.ndarray, *, frame_count: int, hop_length: int) -> np.ndarray:
    return librosa.istft(
        stft,
        hop_length=int(hop_length),
        length=int(frame_count),
        center=True,
    ).astype(np.float32)


def apply_complex_oracle_mask(
    *,
    source_audio: np.ndarray,
    target_reference_audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 4096,
    hop_length: int = 1024,
    source_magnitude_floor: float = 1.0e-7,
) -> tuple[np.ndarray, dict[str, Any]]:
    if source_audio.ndim != 2 or target_reference_audio.ndim != 2:
        raise ValueError("source and target reference audio must be 2D stereo arrays")
    if source_audio.shape[1] != target_reference_audio.shape[1]:
        raise ValueError("source and target reference channel counts must match")
    if n_fft < 128:
        raise ValueError("n_fft must be at least 128")
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")
    if source_magnitude_floor <= 0.0:
        raise ValueError("source_magnitude_floor must be positive")

    frame_count = min(source_audio.shape[0], target_reference_audio.shape[0])
    source_audio = fit_length(source_audio, frame_count)
    target_reference_audio = fit_length(target_reference_audio, frame_count)
    output_channels: list[np.ndarray] = []
    channel_reports: list[dict[str, Any]] = []
    for channel_index in range(source_audio.shape[1]):
        source_stft = _stft_channel(source_audio[:, channel_index], n_fft=n_fft, hop_length=hop_length)
        target_stft = _stft_channel(target_reference_audio[:, channel_index], n_fft=n_fft, hop_length=hop_length)
        source_magnitude = np.abs(source_stft)
        active_mask = source_magnitude >= float(source_magnitude_floor)
        complex_mask = np.zeros_like(target_stft, dtype=np.complex64)
        complex_mask[active_mask] = (target_stft[active_mask] / source_stft[active_mask]).astype(np.complex64)
        reconstructed_stft = source_stft * complex_mask
        reconstructed_channel = _istft_channel(
            reconstructed_stft,
            frame_count=frame_count,
            hop_length=hop_length,
        )
        output_channels.append(reconstructed_channel)

        inactive_target_energy = float(np.sum(np.abs(target_stft[~active_mask]) ** 2))
        total_target_energy = float(np.sum(np.abs(target_stft) ** 2) + EPSILON)
        mask_magnitude = np.abs(complex_mask[active_mask])
        channel_reports.append(
            {
                "channel": channel_index,
                "active_bin_fraction": round(float(np.mean(active_mask)), 9),
                "inactive_target_energy_ratio": round(inactive_target_energy / total_target_energy, 12),
                "max_mask_magnitude": round(float(np.max(mask_magnitude)) if mask_magnitude.size else 0.0, 9),
                "median_mask_magnitude": round(float(np.median(mask_magnitude)) if mask_magnitude.size else 0.0, 9),
            }
        )

    reconstructed = np.stack(output_channels, axis=1).astype(np.float32)
    peak = float(np.max(np.abs(reconstructed))) if reconstructed.size else 0.0
    if peak > 1.0:
        reconstructed = (reconstructed / peak).astype(np.float32)
    residual = reconstructed - target_reference_audio
    diagnostics = {
        "sample_rate": int(sample_rate),
        "n_fft": int(n_fft),
        "hop_length": int(hop_length),
        "source_magnitude_floor": float(source_magnitude_floor),
        "frame_count": int(frame_count),
        "reference_audio_copied_to_output": False,
        "complex_mask_applied_to_source_mix": True,
        "rms_dbfs": {
            "source": round(_dbfs_from_rms(source_audio), 6),
            "target_reference": round(_dbfs_from_rms(target_reference_audio), 6),
            "reconstructed": round(_dbfs_from_rms(reconstructed), 6),
            "residual": round(_dbfs_from_rms(residual), 6),
        },
        "channels": channel_reports,
    }
    return reconstructed, diagnostics


def _ralph_ai_contract() -> dict[str, Any]:
    return {
        "method": "RALPH-AI",
        "status": "requires_exact_reference_gate",
        "presentation_status": "diagnostic_requires_exact_reference_gate",
        "result_delivery_allowed": False,
        "diagnostic_artifacts_allowed": True,
        "rejects_best_of_bad_candidate": True,
        "reason": "complex_oracle_mask_teacher_requires_exact_reference_benchmark_before_any_delivery",
        "required_next_action": "run_model_zoo_benchmark_exact_reference_gate",
    }


def build_complex_oracle_mask_teacher(
    *,
    ground_truth_manifest: Path,
    output_dir: Path,
    stem_names: Sequence[str],
    n_fft: int = 4096,
    hop_length: int = 1024,
    source_magnitude_floor: float = 1.0e-7,
    listen_dir: Path | None = None,
) -> dict[str, Any]:
    require_docker_runtime(
        tool_name="build_complex_oracle_mask_teacher.py",
        entrypoint_path="/app/tools/build_complex_oracle_mask_teacher.py",
    )
    started = time.perf_counter()
    if not stem_names:
        raise ValueError("at least one stem is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_mix_path = resolve_ground_truth_mix_path(ground_truth_manifest)
    source_audio, source_sample_rate = read_stereo_audio(source_mix_path)
    ground_truth_payload = read_json(ground_truth_manifest)

    stem_reports: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for stem_name in stem_names:
        reference_stem_path = resolve_stem_path(ground_truth_manifest, stem_name)
        reference_audio, reference_sample_rate = read_stereo_audio(reference_stem_path)
        if int(reference_sample_rate) != int(source_sample_rate):
            raise ValueError(f"source/reference sample-rate mismatch for stem {stem_name}")
        reconstructed, stem_diagnostics = apply_complex_oracle_mask(
            source_audio=source_audio,
            target_reference_audio=reference_audio,
            sample_rate=int(source_sample_rate),
            n_fft=int(n_fft),
            hop_length=int(hop_length),
            source_magnitude_floor=float(source_magnitude_floor),
        )
        output_stem_path = output_dir / f"{stem_name}.flac"
        sf.write(str(output_stem_path), reconstructed, int(source_sample_rate), subtype=OUTPUT_SUBTYPE)
        stem_reports[stem_name] = {
            "path": str(output_stem_path),
            "stem_type": stem_name,
            "sha256": sha256_file(output_stem_path),
            "role": "complex_oracle_mask_teacher_stem",
            "source_audio_role": "source_mix_complex_mask_carrier",
            "target_reference_used_at_application": True,
            "reference_audio_copied_to_output": False,
        }
        diagnostics[stem_name] = {
            "reference_stem": audio_file_summary(reference_stem_path),
            "output_stem": audio_file_summary(output_stem_path),
            "mask": stem_diagnostics,
        }

    ralph_ai_delivery_contract = _ralph_ai_contract()
    manifest = {
        "schema_version": 1,
        "source_audio_path": str(source_mix_path),
        "source_audio_sha256": sha256_file(source_mix_path),
        "model_id": MODEL_ID,
        "model_name": MODEL_ID,
        "model": {"model_id": MODEL_ID, "family": "reference_assisted_complex_mask_teacher"},
        "sample_rate": int(source_sample_rate),
        "export_subtype": OUTPUT_SUBTYPE,
        "contract": {
            "full_track": True,
            "benchmark_compatible": True,
            "target_reference_assisted": True,
            "target_reference_used": True,
            "target_reference_used_at_application": True,
            "application_reference_free": False,
            "profile_source_reference_assisted": True,
            "delivery_allowed_without_reference": False,
            "registered_model_candidate": False,
            "fallback_used": False,
            "hidden_inference": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
            "stores_timestamp_rules": False,
            "no_reference_production_claim": False,
            "quality_gate_required": True,
            "reference_audio_copied_to_output": False,
        },
        "delivery": {
            "status": "requires_exact_reference_gate",
            "reason": "complex_oracle_mask_teacher_requires_benchmark_before_delivery",
        },
        "presentation_status": ralph_ai_delivery_contract["presentation_status"],
        "ralph_ai_delivery_contract": ralph_ai_delivery_contract,
        "pipeline": {
            "route": MODEL_ID,
            "source_audio_role": "authoritative_full_mix_complex_mask_carrier",
            "reference_audio_role": "target_stem_complex_mask_teacher",
            "reference_audio_copied_to_output": False,
            "source_mix_path": str(source_mix_path),
            "ground_truth_manifest": str(ground_truth_manifest),
            "ground_truth_contract": ground_truth_payload.get("ground_truth_contract")
            or ground_truth_payload.get("contract"),
            "parameters": {
                "n_fft": int(n_fft),
                "hop_length": int(hop_length),
                "source_magnitude_floor": float(source_magnitude_floor),
            },
            "diagnostics": diagnostics,
        },
        "stems": stem_reports,
        "execution": {
            "tool": "ml_worker_improved/tools/build_complex_oracle_mask_teacher.py",
            "elapsed_sec": round(time.perf_counter() - started, 3),
        },
    }
    manifest_path = output_dir / "stem_manifest.json"
    write_json(manifest_path, manifest)
    if listen_dir is not None:
        listen_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_mix_path, listen_dir / "00_source_mix.flac")
        for index, stem_name in enumerate(stem_names, start=1):
            shutil.copyfile(stem_reports[stem_name]["path"], listen_dir / f"{index:02d}_complex_oracle_mask_teacher_{stem_name}.flac")
        (listen_dir / "README.txt").write_text(
            "Complex oracle mask teacher artifacts. These files are reference-assisted diagnostics, not no-reference production output.\n",
            encoding="utf-8",
        )
        manifest["listen"] = {
            "status": "diagnostic_requires_exact_reference_gate",
            "listen_dir": str(listen_dir),
            "result_delivery_allowed": False,
        }
        write_json(manifest_path, manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reference-assisted complex-mask teacher stems from source_mix.")
    parser.add_argument("--ground-truth-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", action="append", default=None)
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--n-fft", type=int, default=4096)
    parser.add_argument("--hop-length", type=int, default=1024)
    parser.add_argument("--source-magnitude-floor", type=float, default=1.0e-7)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_complex_oracle_mask_teacher.py",
        entrypoint_path="/app/tools/build_complex_oracle_mask_teacher.py",
    )
    args = parse_args(argv)
    manifest = build_complex_oracle_mask_teacher(
        ground_truth_manifest=args.ground_truth_manifest,
        output_dir=args.output_dir,
        stem_names=tuple(args.stem or ("vocals", "melody")),
        n_fft=int(args.n_fft),
        hop_length=int(args.hop_length),
        source_magnitude_floor=float(args.source_magnitude_floor),
        listen_dir=args.listen_dir,
    )
    print(json.dumps({"manifest_path": str(args.output_dir / "stem_manifest.json"), "delivery": manifest["delivery"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
