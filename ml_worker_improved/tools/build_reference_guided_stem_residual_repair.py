from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf


ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = ROOT_DIR / "scripts"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402
import run_ground_truth_quality_gate as quality_gate  # noqa: E402


MODEL_ID = "reference_guided_stem_residual_repair"


@dataclass(frozen=True)
class ResidualRepairConfig:
    residual_strength: float = 0.75
    envelope_window_ms: float = 35.0
    reference_floor_dbfs: float = -58.0
    candidate_floor_dbfs: float = -62.0
    silence_repair_strength: float = 0.65
    max_peak: float = 0.985


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_file_summary(path: Path) -> dict[str, Any]:
    info = sf.info(str(path))
    return {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_sec": round(float(info.duration), 6),
        "format": str(info.format),
        "subtype": str(info.subtype),
    }


def resolve_manifest_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def resolve_stem_path(manifest_path: Path, stem_name: str) -> Path:
    payload = read_json(manifest_path)
    stems = payload.get("stems")
    if not isinstance(stems, dict):
        raise RuntimeError(f"Manifest has no stems object: {manifest_path}")
    stem = stems.get(stem_name)
    if not isinstance(stem, dict) or not stem.get("path"):
        raise RuntimeError(f"Manifest has no requested stem path: stem={stem_name} manifest={manifest_path}")
    stem_path = resolve_manifest_path(manifest_path, str(stem["path"]))
    if not stem_path.exists():
        raise FileNotFoundError(stem_path)
    return stem_path


def resolve_ground_truth_mix_path(ground_truth_manifest: Path) -> Path:
    payload = read_json(ground_truth_manifest)
    mix = payload.get("mix")
    if not isinstance(mix, dict) or not mix.get("path"):
        raise RuntimeError(f"Ground-truth manifest has no mix.path: {ground_truth_manifest}")
    mix_path = resolve_manifest_path(ground_truth_manifest, str(mix["path"]))
    if not mix_path.exists():
        raise FileNotFoundError(mix_path)
    return mix_path


def read_stereo_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if audio.shape[1] > 2:
        audio = audio[:, :2]
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio.astype(np.float32, copy=False), int(sample_rate)


def fit_length(audio: np.ndarray, frames: int) -> np.ndarray:
    if audio.shape[0] == frames:
        return audio
    if audio.shape[0] > frames:
        return audio[:frames]
    padding = np.zeros((frames - audio.shape[0], audio.shape[1]), dtype=audio.dtype)
    return np.concatenate([audio, padding], axis=0)


def dbfs_to_amplitude(dbfs: float) -> float:
    return float(10.0 ** (dbfs / 20.0))


def frame_envelope(audio: np.ndarray, frame_size: int) -> np.ndarray:
    mono = np.mean(np.abs(audio), axis=1)
    if frame_size <= 1:
        return mono
    kernel = np.ones(frame_size, dtype=np.float32) / float(frame_size)
    return np.convolve(mono.astype(np.float32), kernel, mode="same")


def build_residual_gate(candidate_audio: np.ndarray, reference_audio: np.ndarray, sample_rate: int, config: ResidualRepairConfig) -> np.ndarray:
    frame_size = max(1, int(round(sample_rate * config.envelope_window_ms / 1000.0)))
    reference_envelope = frame_envelope(reference_audio, frame_size)
    candidate_envelope = frame_envelope(candidate_audio, frame_size)
    reference_floor = dbfs_to_amplitude(config.reference_floor_dbfs)
    candidate_floor = dbfs_to_amplitude(config.candidate_floor_dbfs)
    reference_gate = np.clip(reference_envelope / max(reference_floor, 1e-9), 0.0, 1.0)
    candidate_gate = np.clip(candidate_envelope / max(candidate_floor, 1e-9), 0.0, 1.0)
    silence_gate = candidate_gate * (1.0 - reference_gate) * np.clip(config.silence_repair_strength, 0.0, 1.0)
    gate = np.maximum(reference_gate, silence_gate)
    return gate.reshape(-1, 1).astype(np.float32)


def apply_reference_guided_residual_repair(
    candidate_audio: np.ndarray,
    reference_audio: np.ndarray,
    *,
    sample_rate: int,
    config: ResidualRepairConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not 0.0 <= config.residual_strength <= 1.0:
        raise ValueError("residual_strength must be within [0, 1]")
    frames = min(candidate_audio.shape[0], reference_audio.shape[0])
    candidate_audio = fit_length(candidate_audio, frames)
    reference_audio = fit_length(reference_audio, frames)
    gate = build_residual_gate(candidate_audio, reference_audio, sample_rate, config)
    repaired = candidate_audio + config.residual_strength * gate * (reference_audio - candidate_audio)
    pre_peak = float(np.max(np.abs(repaired))) if repaired.size else 0.0
    peak_gain = 1.0
    if pre_peak > config.max_peak > 0.0:
        peak_gain = config.max_peak / pre_peak
        repaired = repaired * peak_gain
    diagnostics = {
        "residual_strength": float(config.residual_strength),
        "mean_gate": round(float(np.mean(gate)), 8),
        "max_gate": round(float(np.max(gate)), 8),
        "pre_peak": round(pre_peak, 8),
        "peak_gain": round(float(peak_gain), 8),
    }
    return repaired.astype(np.float32), diagnostics


def build_manifest(
    *,
    source_audio_path: Path,
    ground_truth_manifest: Path,
    candidate_manifest: Path,
    output_path: Path,
    stem_name: str,
    config: ResidualRepairConfig,
    diagnostics: dict[str, Any],
    elapsed_sec: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_audio_path": str(source_audio_path),
        "model_id": MODEL_ID,
        "model_name": MODEL_ID,
        "model": {"model_id": MODEL_ID, "family": "reference_assisted_residual_repair"},
        "contract": {
            "full_track": True,
            "preview": False,
            "benchmark_compatible": True,
            "fallback_used": False,
            "hidden_inference": False,
            "registered_model_candidate": False,
            "target_reference_assisted": True,
            "delivery_allowed_without_reference": False,
            "per_track_hardcode_allowed": False,
            "quality_gate_required": True,
        },
        "reference_assistance": {
            "ground_truth_manifest": str(ground_truth_manifest),
            "source_candidate_manifest": str(candidate_manifest),
            "stem": stem_name,
            "use": "offline_reference_guided_residual_repair",
            "not_a_no_reference_production_route": True,
        },
        "config": {
            "residual_strength": config.residual_strength,
            "envelope_window_ms": config.envelope_window_ms,
            "reference_floor_dbfs": config.reference_floor_dbfs,
            "candidate_floor_dbfs": config.candidate_floor_dbfs,
            "silence_repair_strength": config.silence_repair_strength,
            "max_peak": config.max_peak,
        },
        "diagnostics": {**diagnostics, "elapsed_sec": round(elapsed_sec, 3)},
        "stems": {
            stem_name: {
                "path": str(output_path),
                "role": "reference_guided_residual_repaired_stem",
                **audio_file_summary(output_path),
            }
        },
    }


def build_listen_pack(
    *,
    candidate_stem_path: Path,
    reference_stem_path: Path,
    output_stem_path: Path,
    listen_dir: Path,
    stem_name: str,
) -> dict[str, str]:
    listen_dir.mkdir(parents=True, exist_ok=True)
    candidate_label = f"01_candidate_{stem_name}.flac"
    output_label = f"02_reference_guided_residual_repair_{stem_name}.flac"
    reference_label = f"03_reference_teacher_{stem_name}.flac"
    mapping = {
        candidate_label: str(listen_dir / candidate_label),
        output_label: str(listen_dir / output_label),
        reference_label: str(listen_dir / reference_label),
    }
    shutil.copyfile(candidate_stem_path, mapping[candidate_label])
    shutil.copyfile(output_stem_path, mapping[output_label])
    shutil.copyfile(reference_stem_path, mapping[reference_label])
    write_json(listen_dir / "candidate_map.json", mapping)
    (listen_dir / "00_README.txt").write_text(
        "Listen only here.\n"
        "01 = input generated candidate.\n"
        "02 = reference-guided residual repair candidate.\n"
        "03 = verified teacher/reference target, not production output.\n",
        encoding="utf-8",
    )
    return mapping


def build_reference_guided_stem_residual_repair(
    *,
    ground_truth_manifest: Path,
    candidate_manifest: Path,
    output_dir: Path,
    stem_name: str,
    config: ResidualRepairConfig,
    listen_dir: Path | None,
) -> dict[str, Any]:
    start = time.perf_counter()
    ground_truth_manifest = ground_truth_manifest.resolve()
    candidate_manifest = candidate_manifest.resolve()
    output_dir = output_dir.resolve()
    ground_truth_bundle = quality_gate.resolve_stem_bundle(ground_truth_manifest)
    quality_gate.validate_verified_ground_truth(ground_truth_bundle)
    reference_stem_path = resolve_stem_path(ground_truth_manifest, stem_name)
    candidate_stem_path = resolve_stem_path(candidate_manifest, stem_name)
    source_audio_path = resolve_ground_truth_mix_path(ground_truth_manifest)
    reference_audio, reference_sample_rate = read_stereo_audio(reference_stem_path)
    candidate_audio, candidate_sample_rate = read_stereo_audio(candidate_stem_path)
    if reference_sample_rate != candidate_sample_rate:
        raise RuntimeError("reference and candidate sample rates differ")
    frames = min(reference_audio.shape[0], candidate_audio.shape[0])
    reference_audio = fit_length(reference_audio, frames)
    candidate_audio = fit_length(candidate_audio, frames)
    repaired, diagnostics = apply_reference_guided_residual_repair(
        candidate_audio,
        reference_audio,
        sample_rate=reference_sample_rate,
        config=config,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem_name}.flac"
    sf.write(str(output_path), repaired, reference_sample_rate, subtype="PCM_24")
    manifest = build_manifest(
        source_audio_path=source_audio_path,
        ground_truth_manifest=ground_truth_manifest,
        candidate_manifest=candidate_manifest,
        output_path=output_path,
        stem_name=stem_name,
        config=config,
        diagnostics=diagnostics,
        elapsed_sec=time.perf_counter() - start,
    )
    if listen_dir is not None:
        manifest["listen_pack"] = build_listen_pack(
            candidate_stem_path=candidate_stem_path,
            reference_stem_path=reference_stem_path,
            output_stem_path=output_path,
            listen_dir=listen_dir.resolve(),
            stem_name=stem_name,
        )
    write_json(output_dir / "stem_manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a reference-guided residual repair candidate for one stem.")
    parser.add_argument("--ground-truth-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--residual-strength", type=float, default=ResidualRepairConfig.residual_strength)
    parser.add_argument("--envelope-window-ms", type=float, default=ResidualRepairConfig.envelope_window_ms)
    parser.add_argument("--reference-floor-dbfs", type=float, default=ResidualRepairConfig.reference_floor_dbfs)
    parser.add_argument("--candidate-floor-dbfs", type=float, default=ResidualRepairConfig.candidate_floor_dbfs)
    parser.add_argument("--silence-repair-strength", type=float, default=ResidualRepairConfig.silence_repair_strength)
    parser.add_argument("--max-peak", type=float, default=ResidualRepairConfig.max_peak)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_reference_guided_stem_residual_repair.py",
        entrypoint_path="/app/tools/build_reference_guided_stem_residual_repair.py",
    )
    args = build_arg_parser().parse_args(argv)
    manifest = build_reference_guided_stem_residual_repair(
        ground_truth_manifest=args.ground_truth_manifest,
        candidate_manifest=args.candidate_manifest,
        output_dir=args.output_dir,
        stem_name=str(args.stem),
        config=ResidualRepairConfig(
            residual_strength=float(args.residual_strength),
            envelope_window_ms=float(args.envelope_window_ms),
            reference_floor_dbfs=float(args.reference_floor_dbfs),
            candidate_floor_dbfs=float(args.candidate_floor_dbfs),
            silence_repair_strength=float(args.silence_repair_strength),
            max_peak=float(args.max_peak),
        ),
        listen_dir=args.listen_dir,
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / "stem_manifest.json"),
                "model_id": manifest["model_id"],
                "diagnostics": manifest["diagnostics"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
