#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402


TARGET_STEM = "drums_plus_percussive_fx_aux"
BASE_REFERENCE_STEM = "drums"
MODEL_ID = "reference_aux_overlay_candidate"
EPSILON = 1.0e-12


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_summary(path: Path) -> dict[str, Any]:
    info = sf.info(str(path))
    return {
        "path": str(path),
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


def resolve_stem_path(manifest_path: Path, manifest: dict[str, Any], stem_name: str) -> Path:
    stem_entry = manifest.get("stems", {}).get(stem_name)
    if not isinstance(stem_entry, dict):
        raise ValueError(f"manifest has no stem: {stem_name}")
    raw_path = stem_entry.get("path")
    if not raw_path:
        raise ValueError(f"stem has no path: {stem_name}")
    stem_path = resolve_manifest_path(manifest_path, str(raw_path))
    if not stem_path.exists():
        raise FileNotFoundError(stem_path)
    return stem_path


def validate_no_reference_base_manifest(manifest: dict[str, Any]) -> None:
    contract = manifest.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("candidate manifest has no contract")
    unsafe_true_keys = (
        "target_reference_assisted",
        "target_reference_used_at_application",
        "target_reference_used",
        "fallback_used",
        "per_track_hardcode_allowed",
    )
    for key in unsafe_true_keys:
        if bool(contract.get(key)):
            raise ValueError(f"candidate manifest has unsafe contract value: {key}")
    if contract.get("application_reference_free") is False:
        raise ValueError("candidate manifest is not application-reference-free")
    if contract.get("not_timestamp_hardcoded") is False:
        raise ValueError("candidate manifest may contain timestamp hardcodes")


def read_stereo_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if audio.shape[1] > 2:
        audio = audio[:, :2]
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio.astype(np.float32, copy=False), int(sample_rate)


def fit_audio(audio: np.ndarray, frames: int, channels: int) -> np.ndarray:
    if audio.shape[1] < channels:
        audio = np.repeat(audio[:, :1], channels, axis=1)
    if audio.shape[1] > channels:
        audio = audio[:, :channels]
    if audio.shape[0] > frames:
        return audio[:frames].astype(np.float32, copy=False)
    if audio.shape[0] < frames:
        padding = np.zeros((frames - audio.shape[0], channels), dtype=np.float32)
        return np.vstack([audio, padding]).astype(np.float32, copy=False)
    return audio.astype(np.float32, copy=False)


def add_aux_overlay_preserving_base(
    candidate_audio: np.ndarray,
    aux_audio: np.ndarray,
    *,
    aux_gain: float = 1.0,
    max_peak: float = 0.999,
) -> tuple[np.ndarray, dict[str, Any]]:
    if aux_gain < 0.0:
        raise ValueError("aux_gain must be >= 0")
    if not 0.0 < max_peak <= 1.0:
        raise ValueError("max_peak must be in (0, 1]")
    frames = min(candidate_audio.shape[0], aux_audio.shape[0])
    channels = min(candidate_audio.shape[1], aux_audio.shape[1])
    candidate = fit_audio(candidate_audio, frames, channels)
    aux = fit_audio(aux_audio, frames, channels) * float(aux_gain)

    raw_output = candidate + aux
    adjusted_aux = aux.copy()
    over_peak = np.abs(raw_output) > max_peak
    adjustable = over_peak & (np.abs(aux) > EPSILON)
    if np.any(adjustable):
        target = np.sign(raw_output[adjustable]) * max_peak
        scale = (target - candidate[adjustable]) / aux[adjustable]
        adjusted_aux[adjustable] = aux[adjustable] * np.clip(scale, 0.0, 1.0)
    output = candidate + adjusted_aux
    diagnostics = {
        "base_audio_modified": False,
        "aux_gain": float(aux_gain),
        "peak_limited_by_aux_attenuation": bool(np.any(adjustable)),
        "max_peak": float(max_peak),
        "raw_peak_before_aux_attenuation": round(float(np.max(np.abs(raw_output))) if raw_output.size else 0.0, 9),
        "output_peak": round(float(np.max(np.abs(output))) if output.size else 0.0, 9),
        "attenuated_sample_count": int(np.count_nonzero(adjustable)),
    }
    return output.astype(np.float32, copy=False), diagnostics


def build_reference_aux_overlay_candidate(
    *,
    candidate_manifest_path: Path,
    ground_truth_manifest_path: Path,
    output_dir: Path,
    target_stem: str = TARGET_STEM,
    base_reference_stem: str = BASE_REFERENCE_STEM,
    route_name: str = MODEL_ID,
    aux_gain: float = 1.0,
    max_peak: float = 0.999,
    listen_dir: Path | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    candidate_manifest_path = candidate_manifest_path.resolve()
    ground_truth_manifest_path = ground_truth_manifest_path.resolve()
    output_dir = output_dir.resolve()

    candidate_manifest = read_json(candidate_manifest_path)
    ground_truth_manifest = read_json(ground_truth_manifest_path)
    validate_no_reference_base_manifest(candidate_manifest)

    candidate_path = resolve_stem_path(candidate_manifest_path, candidate_manifest, target_stem)
    reference_target_path = resolve_stem_path(ground_truth_manifest_path, ground_truth_manifest, target_stem)
    reference_base_path = resolve_stem_path(ground_truth_manifest_path, ground_truth_manifest, base_reference_stem)

    candidate_audio, candidate_sample_rate = read_stereo_audio(candidate_path)
    reference_target_audio, target_sample_rate = read_stereo_audio(reference_target_path)
    reference_base_audio, base_sample_rate = read_stereo_audio(reference_base_path)
    if len({candidate_sample_rate, target_sample_rate, base_sample_rate}) != 1:
        raise ValueError("sample-rate mismatch between candidate/reference files")

    frames = min(candidate_audio.shape[0], reference_target_audio.shape[0], reference_base_audio.shape[0])
    channels = min(candidate_audio.shape[1], reference_target_audio.shape[1], reference_base_audio.shape[1])
    candidate_audio = fit_audio(candidate_audio, frames, channels)
    reference_target_audio = fit_audio(reference_target_audio, frames, channels)
    reference_base_audio = fit_audio(reference_base_audio, frames, channels)
    reference_aux_delta = reference_target_audio - reference_base_audio
    rendered, overlay_diagnostics = add_aux_overlay_preserving_base(
        candidate_audio,
        reference_aux_delta,
        aux_gain=aux_gain,
        max_peak=max_peak,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{target_stem}.flac"
    aux_path = output_dir / "reference_aux_delta.flac"
    sf.write(str(output_path), rendered, candidate_sample_rate, format="FLAC", subtype="PCM_24")
    sf.write(str(aux_path), reference_aux_delta, candidate_sample_rate, format="FLAC", subtype="PCM_24")

    manifest_path = output_dir / "stem_manifest.json"
    manifest = {
        "schema_version": 1,
        "source_audio_path": candidate_manifest.get("source_audio_path") or candidate_manifest.get("audio_path"),
        "model_id": route_name,
        "model_name": route_name,
        "contract": {
            "full_track": True,
            "preview": False,
            "benchmark_compatible": True,
            "target_reference_assisted": True,
            "target_reference_used_at_application": True,
            "application_reference_free": False,
            "delivery_allowed_without_reference": False,
            "no_reference_production_claim": False,
            "fallback_used": False,
            "hidden_inference": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
            "quality_gate_required": True,
        },
        "pipeline": {
            "route": route_name,
            "family": "reference_aux_overlay",
            "base_candidate_manifest": str(candidate_manifest_path),
            "ground_truth_manifest": str(ground_truth_manifest_path),
            "target_stem": target_stem,
            "base_reference_stem": base_reference_stem,
            "reference_aux_formula": f"{target_stem} - {base_reference_stem}",
            "base_preserving_addition": True,
            "base_audio_unmodified": True,
            "clip_protection": "attenuate_aux_only",
            "aux_gain": float(aux_gain),
            "reference_audio_used": True,
            "track_specific_state_used": False,
            "not_a_no_reference_production_route": True,
            "overlay_diagnostics": overlay_diagnostics,
            "elapsed_sec": round(time.perf_counter() - started_at, 3),
        },
        "inputs": {
            "candidate": audio_summary(candidate_path),
            "reference_target": audio_summary(reference_target_path),
            "reference_base": audio_summary(reference_base_path),
            "reference_aux_delta": audio_summary(aux_path),
        },
        "stems": {
            target_stem: {
                "path": str(output_path),
                "role": "reference_aux_overlay_candidate",
                **audio_summary(output_path),
            }
        },
    }
    if listen_dir is not None:
        manifest["listen"] = build_listen_pack(
            candidate_path=candidate_path,
            reference_target_path=reference_target_path,
            output_path=output_path,
            listen_dir=listen_dir.resolve(),
            target_stem=target_stem,
        )
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_listen_pack(
    *,
    candidate_path: Path,
    reference_target_path: Path,
    output_path: Path,
    listen_dir: Path,
    target_stem: str,
) -> dict[str, Any]:
    listen_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "reference_target": listen_dir / f"01_reference_{target_stem}.flac",
        "base_candidate": listen_dir / f"02_base_candidate_{target_stem}.flac",
        "reference_aux_overlay": listen_dir / f"03_reference_aux_overlay_{target_stem}.flac",
    }
    shutil.copy2(reference_target_path, mapping["reference_target"])
    shutil.copy2(candidate_path, mapping["base_candidate"])
    shutil.copy2(output_path, mapping["reference_aux_overlay"])
    readme = (
        "Reference-assisted aux overlay listen pack.\n\n"
        "01 = verified FL Studio target/reference, not generated by separator.\n"
        "02 = original base candidate preserved from no-reference route.\n"
        "03 = base candidate + reference aux delta. This uses reference at application time and is a teacher/diagnostic artifact, not no-reference delivery.\n"
    )
    (listen_dir / "00_README.txt").write_text(readme, encoding="utf-8")
    serializable = {name: str(path) for name, path in mapping.items()}
    write_json(listen_dir / "candidate_map.json", serializable)
    return {"listen_dir": str(listen_dir), "files": serializable}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a base-preserving reference aux overlay candidate.")
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--ground-truth-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-stem", default=TARGET_STEM)
    parser.add_argument("--base-reference-stem", default=BASE_REFERENCE_STEM)
    parser.add_argument("--route-name", default=MODEL_ID)
    parser.add_argument("--aux-gain", type=float, default=1.0)
    parser.add_argument("--max-peak", type=float, default=0.999)
    parser.add_argument("--listen-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="build_reference_aux_overlay_candidate.py",
        entrypoint_path="/app/tools/build_reference_aux_overlay_candidate.py",
    )
    args = parse_args()
    manifest = build_reference_aux_overlay_candidate(
        candidate_manifest_path=args.candidate_manifest,
        ground_truth_manifest_path=args.ground_truth_manifest,
        output_dir=args.output_dir,
        target_stem=str(args.target_stem),
        base_reference_stem=str(args.base_reference_stem),
        route_name=str(args.route_name),
        aux_gain=float(args.aux_gain),
        max_peak=float(args.max_peak),
        listen_dir=args.listen_dir,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
