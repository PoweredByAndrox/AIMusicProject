#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
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
from provenance_contract import assert_no_reference_lineage  # noqa: E402


TARGET_STEM = "drums_plus_percussive_fx_aux"
BASE_STEM = "drums"
DEFAULT_ROUTE_NAME = "percussive_fx_aux_estimator_v1"
EPSILON = 1.0e-12
REQUIRED_NO_REFERENCE_CONTRACT: dict[str, bool] = {
    "target_reference_assisted": False,
    "target_reference_used_at_application": False,
    "application_reference_free": True,
    "fallback_used": False,
    "per_track_hardcode_allowed": False,
    "not_timestamp_hardcoded": True,
}


@dataclass(frozen=True)
class PercussiveAuxConfig:
    frame_size: int = 2048
    hop_size: int = 512
    min_flatness: float = 0.12
    flux_percentile: float = 72.0
    activity_floor_ratio: float = 0.025
    aux_gain: float = 0.75
    max_aux_rms_ratio: float = 0.45
    limiter_peak: float = 0.98


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    return payload


def _manifest_model_id(manifest: dict[str, Any]) -> str:
    model_id = manifest.get("model_id") or manifest.get("model_name")
    if not model_id and isinstance(manifest.get("model"), dict):
        model_id = manifest["model"].get("model_id") or manifest["model"].get("base_model")
    return str(model_id or "unknown_model")


def _resolve_manifest_relative_path(path_value: str, manifest_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return manifest_dir / path


def _validate_no_reference_manifest(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    role: str,
) -> dict[str, Any]:
    return assert_no_reference_lineage(
        manifest,
        manifest_path=manifest_path,
        role=role,
        required_contract=REQUIRED_NO_REFERENCE_CONTRACT,
    )


def _stem_path_from_manifest(manifest: dict[str, Any], manifest_path: Path, stem_name: str) -> Path | None:
    stem_entry = manifest.get("stems", {}).get(stem_name)
    if not isinstance(stem_entry, dict):
        return None
    path_value = stem_entry.get("path")
    if not path_value:
        raise ValueError(f"stem has no path: stem={stem_name} manifest={manifest_path}")
    stem_path = _resolve_manifest_relative_path(str(path_value), manifest_path.parent)
    if not stem_path.exists():
        raise FileNotFoundError(f"stem file is missing: stem={stem_name} path={stem_path}")
    return stem_path


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    return audio.astype(np.float32, copy=False), int(sample_rate)


def _fit_audio(audio: np.ndarray, frames: int, channels: int) -> np.ndarray:
    if audio.shape[1] < channels:
        audio = np.repeat(audio[:, :1], channels, axis=1)
    elif audio.shape[1] > channels:
        audio = audio[:, :channels]
    if audio.shape[0] > frames:
        return audio[:frames]
    if audio.shape[0] < frames:
        return np.pad(audio, ((0, frames - audio.shape[0]), (0, 0)))
    return audio


def _write_flac(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, format="FLAC", subtype="PCM_24")


def _frame_features(mono: np.ndarray, config: PercussiveAuxConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_size = int(config.frame_size)
    hop_size = int(config.hop_size)
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_size and hop_size must be positive")
    if mono.shape[0] < frame_size:
        mono = np.pad(mono, (0, frame_size - mono.shape[0]))
    frame_count = 1 + int(np.ceil(max(0, mono.shape[0] - frame_size) / float(hop_size)))
    window = np.hanning(frame_size).astype(np.float32)
    rms_values: list[float] = []
    flatness_values: list[float] = []
    flux_values: list[float] = []
    previous_magnitude: np.ndarray | None = None
    for frame_index in range(frame_count):
        start = frame_index * hop_size
        frame = mono[start : start + frame_size]
        if frame.shape[0] < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.shape[0]))
        windowed = frame * window
        magnitude = np.abs(np.fft.rfft(windowed)).astype(np.float64) + EPSILON
        rms_values.append(float(np.sqrt(np.mean(frame * frame) + EPSILON)))
        flatness_values.append(float(np.exp(np.mean(np.log(magnitude))) / (np.mean(magnitude) + EPSILON)))
        if previous_magnitude is None:
            flux_values.append(0.0)
        else:
            positive_delta = np.maximum(magnitude - previous_magnitude, 0.0)
            flux_values.append(float(np.sum(positive_delta) / (np.sum(magnitude) + EPSILON)))
        previous_magnitude = magnitude
    return (
        np.asarray(rms_values, dtype=np.float32),
        np.asarray(flatness_values, dtype=np.float32),
        np.asarray(flux_values, dtype=np.float32),
    )


def _sample_gate_from_frame_gate(frame_gate: np.ndarray, frame_count: int, config: PercussiveAuxConfig) -> np.ndarray:
    smoothed = np.convolve(frame_gate.astype(np.float32), np.asarray([0.25, 0.5, 0.25], dtype=np.float32), mode="same")
    sample_gate = np.zeros(frame_count, dtype=np.float32)
    for frame_index, gate_value in enumerate(smoothed):
        start = frame_index * config.hop_size
        end = min(start + config.frame_size, frame_count)
        if start >= frame_count:
            break
        sample_gate[start:end] = np.maximum(sample_gate[start:end], float(gate_value))
    return sample_gate


def estimate_percussive_aux(audio: np.ndarray, *, base_audio: np.ndarray, config: PercussiveAuxConfig) -> tuple[np.ndarray, dict[str, Any]]:
    mono = np.mean(audio, axis=1).astype(np.float32)
    rms_values, flatness_values, flux_values = _frame_features(mono, config)
    active_floor = max(float(np.max(rms_values)) * float(config.activity_floor_ratio), 1.0e-5)
    flux_threshold = float(np.percentile(flux_values, float(config.flux_percentile)))
    activity_gate = rms_values >= active_floor
    transient_gate = (flatness_values >= float(config.min_flatness)) | (flux_values >= flux_threshold)
    frame_gate = activity_gate & transient_gate
    sample_gate = _sample_gate_from_frame_gate(frame_gate, audio.shape[0], config)
    aux_estimate = audio * sample_gate[:, None] * float(config.aux_gain)

    base_mono = np.mean(base_audio, axis=1).astype(np.float32)
    base_rms_values, _, _ = _frame_features(base_mono, config)
    base_anchor_rms = max(float(np.percentile(base_rms_values, 95.0)), float(np.sqrt(np.mean(base_audio * base_audio) + EPSILON)))
    aux_rms = float(np.sqrt(np.mean(aux_estimate * aux_estimate) + EPSILON))
    max_aux_rms = base_anchor_rms * float(config.max_aux_rms_ratio)
    if aux_rms > max_aux_rms > 0.0:
        aux_estimate = aux_estimate * float(max_aux_rms / aux_rms)
        aux_rms = float(np.sqrt(np.mean(aux_estimate * aux_estimate) + EPSILON))

    metrics = {
        "input_rms_dbfs": _rms_dbfs(audio),
        "estimated_aux_rms_dbfs": _rms_dbfs(aux_estimate),
        "active_frame_count": int(np.sum(frame_gate)),
        "total_frame_count": int(frame_gate.size),
        "mean_flatness": round(float(np.mean(flatness_values)), 6),
        "flux_threshold": round(flux_threshold, 6),
        "active_floor": round(active_floor, 9),
        "aux_gain": float(config.aux_gain),
        "max_aux_rms": round(max_aux_rms, 9),
        "final_aux_rms": round(aux_rms, 9),
    }
    return aux_estimate.astype(np.float32, copy=False), metrics


def _rms_dbfs(audio: np.ndarray) -> float:
    return round(float(20.0 * np.log10(np.sqrt(np.mean(audio * audio) + EPSILON) + EPSILON)), 6)


def _peak_dbfs(audio: np.ndarray) -> float:
    return round(float(20.0 * np.log10(np.max(np.abs(audio)) + EPSILON)), 6)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_candidate_stem(
    *,
    manifest_path: Path,
    stem_name: str,
    target_frames: int,
    target_channels: int,
    expected_sample_rate: int,
    role: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest = _load_json(manifest_path)
    lineage_report = _validate_no_reference_manifest(manifest, manifest_path=manifest_path, role=role)
    stem_path = _stem_path_from_manifest(manifest, manifest_path, stem_name)
    if stem_path is None:
        raise ValueError(f"manifest does not contain required stem: stem={stem_name} manifest={manifest_path}")
    audio, sample_rate = _read_audio(stem_path)
    if sample_rate != expected_sample_rate:
        raise ValueError(f"sample-rate mismatch: {stem_path} has {sample_rate}, expected {expected_sample_rate}")
    return _fit_audio(audio, target_frames, target_channels), {
        "manifest_path": str(manifest_path),
        "model_id": _manifest_model_id(manifest),
        "stem_name": stem_name,
        "stem_path": str(stem_path),
        "sha256": _sha256_file(stem_path),
        "provenance_contract": lineage_report,
    }


def build_percussive_fx_aux_candidate(
    *,
    base_manifest_path: Path,
    aux_manifest_paths: Sequence[Path],
    output_dir: Path,
    route_name: str = DEFAULT_ROUTE_NAME,
    source_stems: Sequence[str] = ("other", "fx"),
    config: PercussiveAuxConfig = PercussiveAuxConfig(),
) -> dict[str, Any]:
    started_at = time.perf_counter()
    base_manifest_path = base_manifest_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_manifest = _load_json(base_manifest_path)
    base_lineage_report = _validate_no_reference_manifest(
        base_manifest,
        manifest_path=base_manifest_path,
        role="base",
    )
    base_audio, sample_rate, base_info = _read_base_audio(base_manifest, base_manifest_path)
    target_frames, target_channels = base_audio.shape

    combined_aux = np.zeros_like(base_audio, dtype=np.float32)
    source_records: list[dict[str, Any]] = []
    source_stems_used: list[str] = []
    for aux_manifest_path in aux_manifest_paths:
        aux_manifest_path = aux_manifest_path.resolve()
        aux_manifest = _load_json(aux_manifest_path)
        aux_lineage_report = _validate_no_reference_manifest(
            aux_manifest,
            manifest_path=aux_manifest_path,
            role="aux",
        )
        for stem_name in source_stems:
            stem_path = _stem_path_from_manifest(aux_manifest, aux_manifest_path, stem_name)
            if stem_path is None:
                continue
            aux_audio, aux_sample_rate = _read_audio(stem_path)
            if aux_sample_rate != sample_rate:
                raise ValueError(f"sample-rate mismatch: {stem_path} has {aux_sample_rate}, expected {sample_rate}")
            aux_audio = _fit_audio(aux_audio, target_frames, target_channels)
            aux_estimate, metrics = estimate_percussive_aux(aux_audio, base_audio=base_audio, config=config)
            combined_aux += aux_estimate
            source_stems_used.append(stem_name)
            source_records.append(
                {
                    "manifest_path": str(aux_manifest_path),
                    "model_id": _manifest_model_id(aux_manifest),
                    "stem_name": stem_name,
                    "stem_path": str(stem_path),
                    "sha256": _sha256_file(stem_path),
                    "provenance_contract": aux_lineage_report,
                    "metrics": metrics,
                }
            )

    if not source_records:
        requested = ", ".join(source_stems)
        raise ValueError(f"no requested aux stems were found: {requested}")

    rendered = base_audio + combined_aux
    peak = float(np.max(np.abs(rendered)))
    limiter_gain = 1.0
    if peak > float(config.limiter_peak) > 0.0:
        limiter_gain = float(config.limiter_peak / peak)
        rendered *= limiter_gain
        combined_aux *= limiter_gain

    target_path = output_dir / f"{TARGET_STEM}.flac"
    aux_path = output_dir / "estimated_percussive_fx_aux.flac"
    _write_flac(target_path, rendered, sample_rate)
    _write_flac(aux_path, combined_aux, sample_rate)
    manifest_path = output_dir / "stem_manifest.json"
    source_model_ids = sorted({record["model_id"] for record in source_records})
    source_stems_used_unique = sorted(set(source_stems_used))
    manifest = {
        "schema_version": 1,
        "source_audio_path": base_manifest.get("source_audio_path") or base_manifest.get("audio_path"),
        "model_id": route_name,
        "model_name": route_name,
        "contract": {
            "benchmark_compatible": True,
            "target_reference_assisted": False,
            "target_reference_used_at_application": False,
            "application_reference_free": True,
            "delivery_allowed_without_reference": True,
            "reference_teacher_pack": False,
            "registered_model_candidate": False,
            "fallback_used": False,
            "hidden_inference": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
            "no_reference_production_claim": True,
            "quality_gate_required": True,
        },
        "pipeline": {
            "route": route_name,
            "base_manifest": str(base_manifest_path),
            "base_model_id": base_info["model_id"],
            "aux_manifest_paths": [str(path.resolve()) for path in aux_manifest_paths],
            "target_definition": TARGET_STEM,
            "reference_audio_used": False,
            "track_specific_state_used": False,
            "aux_estimator": {
                "family": "no_reference_percussive_fx_aux_estimator",
                "config": asdict(config),
                "base_provenance_contract": base_lineage_report,
                "source_model_ids": source_model_ids,
                "source_stems_requested": list(source_stems),
                "source_stems_used": source_stems_used_unique,
                "source_records": source_records,
                "limiter_gain": round(limiter_gain, 9),
                "estimated_aux_path": str(aux_path),
            },
        },
        "stems": {
            "percussive_fx_aux": {
                "path": str(aux_path),
                "target_definition": "percussive_fx_aux",
                "source_model_stems": source_stems_used_unique,
                "source_model_ids": source_model_ids,
                "derived": False,
                "sha256": _sha256_file(aux_path),
                "rms_dbfs": _rms_dbfs(combined_aux),
                "peak_dbfs": _peak_dbfs(combined_aux),
            },
            TARGET_STEM: {
                "path": str(target_path),
                "target_definition": TARGET_STEM,
                "source_model_stem": BASE_STEM,
                "source_candidate_model_id": base_info["model_id"],
                "estimated_aux_path": str(aux_path),
                "derived": False,
                "sha256": _sha256_file(target_path),
                "rms_dbfs": _rms_dbfs(rendered),
                "peak_dbfs": _peak_dbfs(rendered),
            }
        },
        "execution": {
            "tool": "ml_worker_improved/tools/build_percussive_fx_aux_candidate.py",
            "mode": "manifest_only_no_inference",
            "elapsed_sec": round(time.perf_counter() - started_at, 3),
        },
    }
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _read_base_audio(base_manifest: dict[str, Any], base_manifest_path: Path) -> tuple[np.ndarray, int, dict[str, Any]]:
    stem_path = _stem_path_from_manifest(base_manifest, base_manifest_path, BASE_STEM)
    if stem_path is None:
        raise ValueError(f"base manifest does not contain required stem: {BASE_STEM}")
    audio, sample_rate = _read_audio(stem_path)
    return audio, sample_rate, {
        "model_id": _manifest_model_id(base_manifest),
        "stem_path": str(stem_path),
        "sha256": _sha256_file(stem_path),
    }


def _safe_listen_path(listen_dir: Path, listen_label: str) -> Path:
    label_path = Path(listen_label)
    if not listen_label or label_path.is_absolute() or ".." in label_path.parts:
        raise ValueError("listen label must stay inside listen_dir")
    listen_path = (listen_dir / label_path).resolve()
    try:
        listen_path.relative_to(listen_dir)
    except ValueError as exc:
        raise ValueError("listen label must stay inside listen_dir") from exc
    return listen_path


def _copy_listen_file(
    *,
    target_path: Path,
    listen_dir: Path,
    listen_label: str,
    aux_path: Path | None = None,
    aux_label: str = "02_percussive_fx_aux.flac",
) -> dict[str, Any]:
    listen_dir = listen_dir.resolve()
    listen_path = _safe_listen_path(listen_dir, listen_label)
    listen_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target_path, listen_path)
    files = {TARGET_STEM: str(listen_path)}
    if aux_path is not None:
        aux_listen_path = _safe_listen_path(listen_dir, aux_label)
        aux_listen_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(aux_path, aux_listen_path)
        files["percussive_fx_aux"] = str(aux_listen_path)
    return {"listen_dir": str(listen_dir), "files": files}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a no-reference drums+percussive-FX-aux candidate from safe candidate manifests."
    )
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--aux-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route-name", default=DEFAULT_ROUTE_NAME)
    parser.add_argument("--source-stem", action="append", dest="source_stems")
    parser.add_argument("--frame-size", type=int, default=PercussiveAuxConfig.frame_size)
    parser.add_argument("--hop-size", type=int, default=PercussiveAuxConfig.hop_size)
    parser.add_argument("--min-flatness", type=float, default=PercussiveAuxConfig.min_flatness)
    parser.add_argument("--flux-percentile", type=float, default=PercussiveAuxConfig.flux_percentile)
    parser.add_argument("--aux-gain", type=float, default=PercussiveAuxConfig.aux_gain)
    parser.add_argument("--max-aux-rms-ratio", type=float, default=PercussiveAuxConfig.max_aux_rms_ratio)
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--listen-label", default="01_drums_plus_percussive_fx_aux.flac")
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="build_percussive_fx_aux_candidate.py",
        entrypoint_path="/app/tools/build_percussive_fx_aux_candidate.py",
    )
    args = parse_args()
    config = PercussiveAuxConfig(
        frame_size=int(args.frame_size),
        hop_size=int(args.hop_size),
        min_flatness=float(args.min_flatness),
        flux_percentile=float(args.flux_percentile),
        aux_gain=float(args.aux_gain),
        max_aux_rms_ratio=float(args.max_aux_rms_ratio),
    )
    manifest = build_percussive_fx_aux_candidate(
        base_manifest_path=args.base_manifest,
        aux_manifest_paths=list(args.aux_manifest),
        output_dir=args.output_dir,
        route_name=str(args.route_name),
        source_stems=tuple(args.source_stems or ("other", "fx")),
        config=config,
    )
    if args.listen_dir is not None:
        listen = _copy_listen_file(
            target_path=Path(manifest["stems"][TARGET_STEM]["path"]),
            listen_dir=args.listen_dir.resolve(),
            listen_label=str(args.listen_label),
            aux_path=Path(manifest["stems"]["percussive_fx_aux"]["path"]),
        )
        manifest["listen"] = listen
        _write_json(Path(manifest["manifest_path"]), manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
