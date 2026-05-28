from __future__ import annotations

import argparse
import json
import shutil
import sys
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
    _ensure_stereo,
    _fit_audio_shape,
    _harmonic_component,
    _limit_peak,
    _load_stems_from_manifest,
    _no_reference_contract as _melody_no_reference_contract,
    _remove_positive_projection,
    sha256_file,
)
from build_tonal_harmonic_melody_candidate import build_tonal_harmonic_melody_candidate_audio  # noqa: E402


ROUTE_NAME = "model_stack_strings_orchestra_v1"
OUTPUT_SUBTYPE = "PCM_24"
TARGET_SOURCE_WEIGHTS = {
    "other": 1.0,
    "melody": 0.18,
    "piano": 0.18,
    "guitar": 0.24,
}


def _no_reference_contract(route_name: str = ROUTE_NAME) -> dict[str, Any]:
    contract = _melody_no_reference_contract()
    contract["route_name"] = str(route_name)
    return contract


def build_model_stack_strings_orchestra_candidate_audio(
    stems: dict[str, np.ndarray],
    *,
    source_audio: np.ndarray | None = None,
    sample_rate: int,
    harmonic_n_fft: int = 2048,
    harmonic_hop_length: int = 512,
    harmonic_margin: float = 3.4,
    percussive_margin: float = 1.1,
    source_tonal_n_fft: int = 4096,
    source_tonal_hop_length: int = 512,
    source_harmonic_margin: float = 7.0,
    source_percussive_margin: float = 12.0,
    source_percussive_suppression: float = 0.90,
    source_highpass_hz: float = 260.0,
    source_lowpass_hz: float = 9500.0,
    source_tonal_weight: float = 0.45,
    model_stack_weight: float = 0.80,
    vocal_suppression: float = 0.95,
    drum_suppression: float = 0.88,
    bass_suppression: float = 0.95,
    melody_suppression: float = 0.45,
    piano_suppression: float = 0.25,
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
        ("melody", float(melody_suppression)),
        ("piano", float(piano_suppression)),
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

    candidate = _harmonic_component(
        candidate,
        n_fft=int(harmonic_n_fft),
        hop_length=int(harmonic_hop_length),
        harmonic_margin=max(float(harmonic_margin), 3.0),
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


def build_model_stack_strings_orchestra_candidate(
    *,
    source_audio_path: str | Path,
    output_dir: str | Path,
    base_manifest_path: str | Path | None = None,
    listen_dir: str | Path | None = None,
    model_name: str = "htdemucs_6s",
    device: str = "cuda",
    harmonic_n_fft: int = 2048,
    harmonic_hop_length: int = 512,
    harmonic_margin: float = 3.4,
    percussive_margin: float = 1.1,
    source_tonal_n_fft: int = 4096,
    source_tonal_hop_length: int = 512,
    source_harmonic_margin: float = 7.0,
    source_percussive_margin: float = 12.0,
    source_percussive_suppression: float = 0.90,
    source_highpass_hz: float = 260.0,
    source_lowpass_hz: float = 9500.0,
    source_tonal_weight: float = 0.45,
    model_stack_weight: float = 0.80,
    vocal_suppression: float = 0.95,
    drum_suppression: float = 0.88,
    bass_suppression: float = 0.95,
    melody_suppression: float = 0.45,
    piano_suppression: float = 0.25,
    route_name: str = ROUTE_NAME,
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
        raise ValueError(f"source/base sample-rate mismatch: source={source_sample_rate}, base={stem_sample_rate}")
    source_sha256 = sha256_file(source_path)
    base_source_sha256 = _base_manifest_source_sha256(base_manifest, resolved_base_manifest_path)
    if str(base_source_sha256) != source_sha256:
        raise ValueError(
            "base manifest source hash does not match source audio: "
            + json.dumps({"source": source_sha256, "base_manifest": str(base_source_sha256)}, ensure_ascii=False)
        )

    candidate, diagnostics = build_model_stack_strings_orchestra_candidate_audio(
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
        melody_suppression=float(melody_suppression),
        piano_suppression=float(piano_suppression),
    )
    effective_route_name = str(route_name)
    diagnostics = {**diagnostics, "route_name": effective_route_name, "base_algorithm_route_name": ROUTE_NAME}

    strings_path = resolved_output_dir / "strings_orchestra.flac"
    sf.write(str(strings_path), candidate, int(stem_sample_rate), subtype=OUTPUT_SUBTYPE)
    manifest = {
        "audio_path": str(source_path),
        "audio_sha256": source_sha256,
        "model_id": effective_route_name,
        "model_name": effective_route_name,
        "model": {"model_id": effective_route_name, "family": "hybrid"},
        "sample_rate": int(stem_sample_rate),
        "export_subtype": OUTPUT_SUBTYPE,
        "strings_orchestra_stem_included": True,
        "contract": _no_reference_contract(effective_route_name),
        "pipeline": {
            "mode": effective_route_name,
            "source": "authoritative_full_mix_plus_no_reference_model_stack",
            "base_manifest_path": str(resolved_base_manifest_path),
            "base_manifest_model_name": base_manifest.get("model_name"),
            "base_manifest_contract_checked": contract_check["status"] == "accepted",
            "base_manifest_contract": contract_check,
            "target_reference_read": False,
            "reference_sum_mix_read": False,
            "postprocess": "model_stack_harmonic_strings_candidate_with_projection_debleed_v1",
            "diagnostics": diagnostics,
            "source_audio": _audio_summary(source_audio.astype(np.float32)),
        },
        "delivery": {
            "status": "requires_objective_quality_gate",
            "reason": "no_reference_model_stack_candidate_requires_ground_truth_benchmark_before_delivery",
        },
        "presentation": {
            "status": "diagnostic_requires_objective_gate",
            "result_delivery_allowed": False,
            "reason": "candidate_must_pass_quality_gate_before_listen_delivery",
        },
        "stems": {
            "strings_orchestra": {
                "path": str(strings_path),
                **_audio_summary(candidate),
            },
        },
    }
    if listen_dir is not None:
        resolved_listen_dir = Path(listen_dir).resolve()
        resolved_listen_dir.mkdir(parents=True, exist_ok=True)
        source_listen_path = resolved_listen_dir / "00_source_mix.flac"
        candidate_listen_path = resolved_listen_dir / "01_model_stack_strings_orchestra_v1.flac"
        shutil.copyfile(source_path, source_listen_path)
        shutil.copyfile(strings_path, candidate_listen_path)
        (resolved_listen_dir / "00_README.txt").write_text(
            "Diagnostic listen pack only.\n"
            "00 = authoritative source mix.\n"
            "01 = no-reference model-stack strings_orchestra candidate; must pass objective gate before delivery.\n",
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
    parser = argparse.ArgumentParser(description="Build a no-reference model-stack strings/orchestra candidate.")
    parser.add_argument("source_audio_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--model-name", default="htdemucs_6s")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--harmonic-n-fft", type=int, default=2048)
    parser.add_argument("--harmonic-hop-length", type=int, default=512)
    parser.add_argument("--harmonic-margin", type=float, default=3.4)
    parser.add_argument("--percussive-margin", type=float, default=1.1)
    parser.add_argument("--source-tonal-n-fft", type=int, default=4096)
    parser.add_argument("--source-tonal-hop-length", type=int, default=512)
    parser.add_argument("--source-harmonic-margin", type=float, default=7.0)
    parser.add_argument("--source-percussive-margin", type=float, default=12.0)
    parser.add_argument("--source-percussive-suppression", type=float, default=0.90)
    parser.add_argument("--source-highpass-hz", type=float, default=260.0)
    parser.add_argument("--source-lowpass-hz", type=float, default=9500.0)
    parser.add_argument("--source-tonal-weight", type=float, default=0.45)
    parser.add_argument("--model-stack-weight", type=float, default=0.80)
    parser.add_argument("--vocal-suppression", type=float, default=0.95)
    parser.add_argument("--drum-suppression", type=float, default=0.88)
    parser.add_argument("--bass-suppression", type=float, default=0.95)
    parser.add_argument("--melody-suppression", type=float, default=0.45)
    parser.add_argument("--piano-suppression", type=float, default=0.25)
    parser.add_argument("--route-name", default=ROUTE_NAME)
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="build_model_stack_strings_orchestra_candidate.py",
        entrypoint_path="/app/tools/build_model_stack_strings_orchestra_candidate.py",
    )
    args = parse_args()
    manifest = build_model_stack_strings_orchestra_candidate(
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
        melody_suppression=args.melody_suppression,
        piano_suppression=args.piano_suppression,
        route_name=args.route_name,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
