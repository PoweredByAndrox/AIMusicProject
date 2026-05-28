#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from build_percussive_fx_aux_candidate import (  # noqa: E402
    PercussiveAuxConfig,
    _fit_audio,
    _load_json,
    _peak_dbfs,
    _read_audio,
    _rms_dbfs,
    _safe_listen_path,
    _sha256_file,
    _stem_path_from_manifest,
    _validate_no_reference_manifest,
    _write_flac,
    _write_json,
    estimate_percussive_aux,
)


TARGET_STEM = "drums_plus_percussive_fx_aux"
AUX_STEM = "percussive_fx_aux"
DEFAULT_ROUTE_NAME = "fullmix_residual_percussive_aux"
ResidualPercussiveAuxConfig = PercussiveAuxConfig


def read_json(path: Path) -> dict[str, Any]:
    return _load_json(path)


def _resolve_base_candidate_audio(base_manifest_path: Path, base_manifest: dict[str, Any]) -> Path:
    stem_path = _stem_path_from_manifest(base_manifest, base_manifest_path, TARGET_STEM)
    if stem_path is None:
        raise ValueError(f"base manifest does not contain required stem: {TARGET_STEM}")
    return stem_path


def _attenuate_aux_to_preserve_base_peak(
    base_audio: np.ndarray,
    aux_audio: np.ndarray,
    *,
    max_peak: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not 0.0 < max_peak <= 1.0:
        raise ValueError("max_peak must be in (0, 1]")
    raw_output = base_audio + aux_audio
    adjusted_aux = aux_audio.copy()
    over_peak = np.abs(raw_output) > max_peak
    adjustable = over_peak & (np.abs(aux_audio) > 1.0e-12)
    if np.any(adjustable):
        target = np.sign(raw_output[adjustable]) * max_peak
        scale = (target - base_audio[adjustable]) / aux_audio[adjustable]
        adjusted_aux[adjustable] = aux_audio[adjustable] * np.clip(scale, 0.0, 1.0)
    rendered = base_audio + adjusted_aux
    diagnostics = {
        "base_audio_modified": False,
        "peak_limited_by_aux_attenuation": bool(np.any(adjustable)),
        "attenuated_sample_count": int(np.count_nonzero(adjustable)),
        "raw_peak_before_aux_attenuation": round(float(np.max(np.abs(raw_output))) if raw_output.size else 0.0, 9),
        "output_peak": round(float(np.max(np.abs(rendered))) if rendered.size else 0.0, 9),
        "max_peak": float(max_peak),
    }
    return adjusted_aux.astype(np.float32, copy=False), diagnostics


def _copy_listen_file(*, target_path: Path, listen_dir: Path, listen_label: str) -> dict[str, Any]:
    listen_dir = listen_dir.resolve()
    listen_path = _safe_listen_path(listen_dir, listen_label)
    listen_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target_path, listen_path)
    aux_listen_path = _safe_listen_path(listen_dir, "02_percussive_fx_aux.flac")
    aux_listen_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target_path.parent / "estimated_fullmix_residual_percussive_aux.flac", aux_listen_path)
    return {
        "listen_dir": str(listen_dir),
        "files": {
            TARGET_STEM: str(listen_path),
            AUX_STEM: str(aux_listen_path),
        },
    }


def build_fullmix_residual_percussive_aux_candidate(
    *,
    base_manifest_path: Path,
    source_audio_path: Path,
    output_dir: Path,
    route_name: str = DEFAULT_ROUTE_NAME,
    config: ResidualPercussiveAuxConfig = ResidualPercussiveAuxConfig(),
) -> dict[str, Any]:
    started_at = time.perf_counter()
    base_manifest_path = base_manifest_path.resolve()
    source_audio_path = source_audio_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_manifest = _load_json(base_manifest_path)
    base_lineage_report = _validate_no_reference_manifest(
        base_manifest,
        manifest_path=base_manifest_path,
        role="base",
    )
    base_candidate_path = _resolve_base_candidate_audio(base_manifest_path, base_manifest)
    base_audio, sample_rate = _read_audio(base_candidate_path)
    source_audio, source_sample_rate = _read_audio(source_audio_path)
    if source_sample_rate != sample_rate:
        raise ValueError(f"sample-rate mismatch: source={source_sample_rate}, base={sample_rate}")
    source_audio = _fit_audio(source_audio, base_audio.shape[0], base_audio.shape[1])
    residual_audio = source_audio - base_audio
    estimated_aux, estimator_metrics = estimate_percussive_aux(residual_audio, base_audio=base_audio, config=config)
    adjusted_aux, peak_diagnostics = _attenuate_aux_to_preserve_base_peak(
        base_audio,
        estimated_aux,
        max_peak=float(config.limiter_peak),
    )
    rendered = base_audio + adjusted_aux

    target_path = output_dir / f"{TARGET_STEM}.flac"
    aux_path = output_dir / "estimated_fullmix_residual_percussive_aux.flac"
    residual_path = output_dir / "fullmix_minus_base_residual.flac"
    _write_flac(target_path, rendered, sample_rate)
    _write_flac(aux_path, adjusted_aux, sample_rate)
    _write_flac(residual_path, residual_audio, sample_rate)
    manifest_path = output_dir / "stem_manifest.json"
    manifest = {
        "schema_version": 1,
        "source_audio_path": str(source_audio_path),
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
            "family": "fullmix_residual_percussive_aux",
            "base_manifest": str(base_manifest_path),
            "base_candidate_path": str(base_candidate_path),
            "source_audio_path": str(source_audio_path),
            "source_audio_role": "full_mix",
            "target_definition": TARGET_STEM,
            "reference_audio_used": False,
            "track_specific_state_used": False,
            "base_preserving_addition": True,
            "base_audio_unmodified": True,
            "deterministic": True,
            "config": asdict(config),
            "base_provenance_contract": base_lineage_report,
            "estimator_metrics": estimator_metrics,
            "peak_diagnostics": peak_diagnostics,
            "estimated_aux_path": str(aux_path),
            "residual_path": str(residual_path),
        },
        "stems": {
            AUX_STEM: {
                "path": str(aux_path),
                "target_definition": AUX_STEM,
                "source_audio_role": "full_mix_minus_base_residual",
                "derived": False,
                "sha256": _sha256_file(aux_path),
                "rms_dbfs": _rms_dbfs(adjusted_aux),
                "peak_dbfs": _peak_dbfs(adjusted_aux),
            },
            TARGET_STEM: {
                "path": str(target_path),
                "target_definition": TARGET_STEM,
                "source_model_stem": TARGET_STEM,
                "estimated_aux_path": str(aux_path),
                "derived": False,
                "sha256": _sha256_file(target_path),
                "rms_dbfs": _rms_dbfs(rendered),
                "peak_dbfs": _peak_dbfs(rendered),
            }
        },
        "execution": {
            "tool": "ml_worker_improved/tools/build_fullmix_residual_percussive_aux_candidate.py",
            "mode": "deterministic_fullmix_residual_no_reference",
            "elapsed_sec": round(time.perf_counter() - started_at, 3),
        },
    }
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a repeatable no-reference percussive aux candidate from full-mix residual.")
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--source-audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route-name", default=DEFAULT_ROUTE_NAME)
    parser.add_argument("--frame-size", type=int, default=ResidualPercussiveAuxConfig.frame_size)
    parser.add_argument("--hop-size", type=int, default=ResidualPercussiveAuxConfig.hop_size)
    parser.add_argument("--min-flatness", type=float, default=ResidualPercussiveAuxConfig.min_flatness)
    parser.add_argument("--flux-percentile", type=float, default=ResidualPercussiveAuxConfig.flux_percentile)
    parser.add_argument("--activity-floor-ratio", type=float, default=ResidualPercussiveAuxConfig.activity_floor_ratio)
    parser.add_argument("--aux-gain", type=float, default=ResidualPercussiveAuxConfig.aux_gain)
    parser.add_argument("--max-aux-rms-ratio", type=float, default=ResidualPercussiveAuxConfig.max_aux_rms_ratio)
    parser.add_argument("--limiter-peak", type=float, default=ResidualPercussiveAuxConfig.limiter_peak)
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--listen-label", default="01_fullmix_residual_percussive_aux.flac")
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="build_fullmix_residual_percussive_aux_candidate.py",
        entrypoint_path="/app/tools/build_fullmix_residual_percussive_aux_candidate.py",
    )
    args = parse_args()
    config = ResidualPercussiveAuxConfig(
        frame_size=int(args.frame_size),
        hop_size=int(args.hop_size),
        min_flatness=float(args.min_flatness),
        flux_percentile=float(args.flux_percentile),
        activity_floor_ratio=float(args.activity_floor_ratio),
        aux_gain=float(args.aux_gain),
        max_aux_rms_ratio=float(args.max_aux_rms_ratio),
        limiter_peak=float(args.limiter_peak),
    )
    manifest = build_fullmix_residual_percussive_aux_candidate(
        base_manifest_path=args.base_manifest,
        source_audio_path=args.source_audio,
        output_dir=args.output_dir,
        route_name=str(args.route_name),
        config=config,
    )
    if args.listen_dir is not None:
        manifest["listen"] = _copy_listen_file(
            target_path=Path(manifest["stems"][TARGET_STEM]["path"]),
            listen_dir=args.listen_dir.resolve(),
            listen_label=str(args.listen_label),
        )
        _write_json(Path(manifest["manifest_path"]), manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
