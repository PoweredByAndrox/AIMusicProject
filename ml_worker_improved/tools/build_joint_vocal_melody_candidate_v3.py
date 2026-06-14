from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf
from scipy import signal

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
    sha256_file,
)

ROUTE_NAME = "joint_vocal_melody_v3"
OUTPUT_SUBTYPE = "PCM_24"
DEFAULT_MODEL_NAMES = (
    "hybrid_htdemucs_ft_bs_roformer_6s",
    "htdemucs_6s",
    "htdemucs_ft",
)
PROCESS_START_MONOTONIC = time.monotonic()


def _emit_progress(event: str, **metadata: Any) -> None:
    record = {
        "event": event,
        "route_name": ROUTE_NAME,
        "elapsed_sec": round(time.monotonic() - PROCESS_START_MONOTONIC, 6),
        **metadata,
    }
    print(json.dumps(record, sort_keys=True), flush=True)


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


def _execution_contract_from_base_summaries(base_summaries: Sequence[dict[str, Any]], *, device: str) -> dict[str, Any]:
    base_execution_contracts = [
        summary.get("execution_contract")
        for summary in base_summaries
        if isinstance(summary.get("execution_contract"), dict)
    ]
    selected_contract: dict[str, Any] = {}
    for contract in base_execution_contracts:
        cuda_preflight = contract.get("cuda_preflight")
        if isinstance(cuda_preflight, dict) and cuda_preflight.get("cuda_available") is True:
            selected_contract = contract
            break
    if not selected_contract and base_execution_contracts:
        selected_contract = base_execution_contracts[0]

    cuda_preflight = selected_contract.get("cuda_preflight") if isinstance(selected_contract, dict) else {}
    if not isinstance(cuda_preflight, dict):
        cuda_preflight = {}
    return {
        "route_name": ROUTE_NAME,
        "docker_runtime_required": True,
        "expected_acceleration_mode": str(
            selected_contract.get("expected_acceleration_mode")
            or "gpu_inference_then_cpu_dsp_scoring"
        ),
        "gpu_inference_required": True,
        "gpu_inference_skipped": selected_contract.get("gpu_inference_skipped") is True,
        "cuda_preflight": {
            "cuda_available": cuda_preflight.get("cuda_available") is True,
            "requested_device": str(cuda_preflight.get("requested_device") or device or ""),
            "resolved_device": str(cuda_preflight.get("resolved_device") or ""),
            "device_count": int(cuda_preflight.get("device_count") or 0),
            "device_name": str(cuda_preflight.get("device_name") or ""),
        },
        "base_models": [str(summary.get("model_name") or "") for summary in base_summaries],
        "base_execution_contract_count": len(base_execution_contracts),
    }


def _safe_label(value: str) -> str:
    label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return label.strip("._-") or "candidate"


def _rms(audio: np.ndarray) -> float:
    array = np.asarray(audio, dtype=np.float32)
    if array.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(array * array) + 1e-12))


def _projection_rms(audio: np.ndarray, reference: np.ndarray) -> float:
    candidate = np.asarray(audio, dtype=np.float32)
    suppressor = _fit_audio_shape(reference, frames=candidate.shape[0], channels=candidate.shape[1])
    denominator = float(np.sum(suppressor * suppressor))
    if denominator <= 1e-12:
        return 0.0
    gain = float(np.sum(candidate * suppressor) / (denominator + 1e-12))
    projected = suppressor * gain
    return _rms(projected)


def _fit_candidate_sets(
    candidate_sets: dict[str, dict[str, np.ndarray]],
    *,
    source_audio: np.ndarray | None,
) -> tuple[dict[str, dict[str, np.ndarray]], int, int]:
    if not candidate_sets:
        raise ValueError("no candidate sets provided")
    if source_audio is not None:
        source = np.asarray(source_audio, dtype=np.float32)
        frames = int(source.shape[0])
        channels = int(source.shape[1]) if source.ndim == 2 else 2
    else:
        first_stem = next(iter(next(iter(candidate_sets.values())).values()))
        frames = int(first_stem.shape[0])
        channels = int(first_stem.shape[1]) if first_stem.ndim == 2 else 2
    fitted: dict[str, dict[str, np.ndarray]] = {}
    for label, stems in candidate_sets.items():
        fitted[label] = {
            stem_name: _fit_audio_shape(stem_audio, frames=frames, channels=channels)
            for stem_name, stem_audio in stems.items()
        }
    return fitted, frames, channels


def _resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if int(source_rate) == int(target_rate):
        return np.asarray(audio, dtype=np.float32)
    gcd = math.gcd(int(source_rate), int(target_rate))
    up = int(target_rate) // gcd
    down = int(source_rate) // gcd
    return signal.resample_poly(audio, up, down, axis=0).astype(np.float32)


def _aggregate_suppressor(
    candidate_sets: dict[str, dict[str, np.ndarray]],
    stem_name: str,
    *,
    frames: int,
    channels: int,
) -> np.ndarray | None:
    available = [
        _fit_audio_shape(stems[stem_name], frames=frames, channels=channels)
        for stems in candidate_sets.values()
        if stem_name in stems
    ]
    if not available:
        return None
    return np.mean(np.stack(available, axis=0), axis=0).astype(np.float32)


def _stem_proxy_score(
    candidate: np.ndarray,
    suppressors: Sequence[np.ndarray],
    *,
    leak_weight: float,
) -> dict[str, Any]:
    candidate_rms = _rms(candidate)
    leakage_terms = [
        _projection_rms(candidate, suppressor) / (candidate_rms + 1e-12)
        for suppressor in suppressors
        if _rms(suppressor) > 1e-8
    ]
    leakage_ratio = float(sum(leakage_terms))
    silence_penalty = 0.4 if candidate_rms < 1e-5 else 0.0
    score = float(np.log1p(candidate_rms * 20.0) - float(leak_weight) * leakage_ratio - silence_penalty)
    return {
        "score": round(score, 9),
        "rms": round(candidate_rms, 9),
        "leakage_ratio": round(leakage_ratio, 9),
        "leakage_terms": [round(float(term), 9) for term in leakage_terms],
        "silence_penalty": round(float(silence_penalty), 9),
    }


def _score_and_select(
    candidate_sets: dict[str, dict[str, np.ndarray]],
    stem_name: str,
    suppressor_names: Sequence[str],
    *,
    frames: int,
    channels: int,
    leak_weight: float,
) -> tuple[str, np.ndarray, dict[str, Any]]:
    suppressors = [
        suppressor
        for suppressor_name in suppressor_names
        if (
            suppressor := _aggregate_suppressor(
                candidate_sets,
                suppressor_name,
                frames=frames,
                channels=channels,
            )
        )
        is not None
    ]
    scored: dict[str, Any] = {}
    selected_label: str | None = None
    selected_score: float | None = None
    selected_audio: np.ndarray | None = None
    for label, stems in candidate_sets.items():
        stem_audio = stems.get(stem_name)
        if stem_audio is None:
            continue
        score = _stem_proxy_score(stem_audio, suppressors, leak_weight=float(leak_weight))
        scored[label] = score
        if selected_score is None or float(score["score"]) > selected_score:
            selected_label = label
            selected_score = float(score["score"])
            selected_audio = stem_audio
    if selected_label is None or selected_audio is None:
        raise ValueError(f"no candidate contains required stem: {stem_name}")
    return selected_label, selected_audio.astype(np.float32), scored


def _remove_available_projections(
    candidate: np.ndarray,
    suppressors: Sequence[np.ndarray],
    *,
    strength: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    cleaned = np.asarray(candidate, dtype=np.float32)
    removals: list[dict[str, Any]] = []
    for suppressor in suppressors:
        cleaned, removal = _remove_positive_projection(
            cleaned,
            suppressor,
            strength=float(strength),
        )
        removals.append(removal)
    return cleaned.astype(np.float32), removals


def build_joint_vocal_melody_v3_candidate_audio(
    candidate_sets: dict[str, dict[str, np.ndarray]],
    *,
    source_audio: np.ndarray | None,
    sample_rate: int,
    cross_stem_suppression: float = 0.92,
    non_target_suppression: float = 0.85,
    vocal_leak_weight: float = 2.2,
    melody_leak_weight: float = 2.4,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    fitted_sets, frames, channels = _fit_candidate_sets(candidate_sets, source_audio=source_audio)
    _emit_progress(
        "candidate_audio_start",
        candidate_set_count=len(candidate_sets),
        frame_count=frames,
        channel_count=channels,
        sample_rate=int(sample_rate),
    )
    vocals_label, vocals_candidate, vocal_scores = _score_and_select(
        fitted_sets,
        "vocals",
        ("melody", "drums", "bass"),
        frames=frames,
        channels=channels,
        leak_weight=float(vocal_leak_weight),
    )
    _emit_progress("candidate_audio_vocals_selected", selected_label=str(vocals_label))
    melody_label, melody_candidate, melody_scores = _score_and_select(
        fitted_sets,
        "melody",
        ("vocals", "drums", "bass"),
        frames=frames,
        channels=channels,
        leak_weight=float(melody_leak_weight),
    )
    _emit_progress("candidate_audio_melody_selected", selected_label=str(melody_label))
    drum_suppressor = _aggregate_suppressor(fitted_sets, "drums", frames=frames, channels=channels)
    bass_suppressor = _aggregate_suppressor(fitted_sets, "bass", frames=frames, channels=channels)
    non_target_suppressors = [suppressor for suppressor in (drum_suppressor, bass_suppressor) if suppressor is not None]

    vocals_candidate, vocal_cross_removals = _remove_available_projections(
        vocals_candidate,
        [melody_candidate],
        strength=float(cross_stem_suppression),
    )
    melody_candidate, melody_cross_removals = _remove_available_projections(
        melody_candidate,
        [vocals_candidate],
        strength=float(cross_stem_suppression),
    )
    vocals_candidate, vocal_non_target_removals = _remove_available_projections(
        vocals_candidate,
        non_target_suppressors,
        strength=float(non_target_suppression) * 0.35,
    )
    melody_candidate, melody_non_target_removals = _remove_available_projections(
        melody_candidate,
        non_target_suppressors,
        strength=float(non_target_suppression),
    )

    vocals_candidate = _limit_peak(vocals_candidate)
    melody_candidate = _limit_peak(melody_candidate)
    _emit_progress("candidate_audio_end", frame_count=int(melody_candidate.shape[0]))
    diagnostics = {
        "route_name": ROUTE_NAME,
        "sample_rate": int(sample_rate),
        "selection_strategy": "stem_specific_proxy_score_v1",
        "selected_candidates": {
            "vocals": vocals_label,
            "melody": melody_label,
        },
        "candidate_scores": {
            "vocals": vocal_scores,
            "melody": melody_scores,
        },
        "projection_debleed": {
            "vocals_from_melody": vocal_cross_removals,
            "melody_from_vocals": melody_cross_removals,
            "vocals_from_non_targets": vocal_non_target_removals,
            "melody_from_non_targets": melody_non_target_removals,
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


def _build_default_base_manifest_paths(
    *,
    source_audio_path: Path,
    output_dir: Path,
    model_names: Sequence[str],
    device: str,
) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for model_name in model_names:
        if model_name in seen:
            continue
        seen.add(model_name)
        base_root = output_dir / f"_base_{_safe_label(model_name)}"
        paths.append(
            _build_base_manifest_from_source(
                source_audio_path=source_audio_path,
                output_dir=base_root,
                model_name=str(model_name),
                device=str(device),
            )
        )
    return paths


def _load_candidate_sets(
    base_manifest_paths: Sequence[str | Path],
    *,
    source_audio_path: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], int, list[dict[str, Any]]]:
    source_sha256 = sha256_file(source_audio_path)
    candidate_sets: dict[str, dict[str, np.ndarray]] = {}
    base_summaries: list[dict[str, Any]] = []
    sample_rate: int | None = None
    for index, manifest_path_value in enumerate(base_manifest_paths, start=1):
        manifest_path = Path(manifest_path_value).resolve()
        stems, stem_sample_rate, base_manifest, contract_check = _load_stems_from_manifest(manifest_path)
        if sample_rate is None:
            sample_rate = int(stem_sample_rate)
        elif sample_rate != int(stem_sample_rate):
            raise ValueError(f"sample-rate mismatch in base manifest: {manifest_path}")
        base_source_sha256 = _base_manifest_source_sha256(base_manifest, manifest_path)
        if str(base_source_sha256) != source_sha256:
            raise ValueError(
                "base manifest source hash does not match source audio: "
                + json.dumps(
                    {"source": source_sha256, "base_manifest": str(base_source_sha256), "path": str(manifest_path)},
                    ensure_ascii=False,
                )
            )
        model_name = str(base_manifest.get("model_name") or f"candidate_{index}")
        label = _safe_label(model_name)
        if label in candidate_sets:
            label = f"{label}_{index}"
        candidate_sets[label] = stems
        base_summaries.append(
            {
                "label": label,
                "manifest_path": str(manifest_path),
                "model_name": model_name,
                "contract_check": contract_check,
                "execution_contract": base_manifest.get("execution_contract")
                if isinstance(base_manifest.get("execution_contract"), dict)
                else {},
            }
        )
    if sample_rate is None:
        raise ValueError("no base manifests loaded")
    return candidate_sets, sample_rate, base_summaries


def build_joint_vocal_melody_v3_candidate(
    *,
    source_audio_path: str | Path,
    output_dir: str | Path,
    base_manifest_paths: Sequence[str | Path] | None = None,
    listen_dir: str | Path | None = None,
    model_names: Sequence[str] = DEFAULT_MODEL_NAMES,
    device: str = "cuda",
    cross_stem_suppression: float = 0.92,
    non_target_suppression: float = 0.85,
) -> dict[str, Any]:
    _emit_progress(
        "candidate_build_start",
        source_audio_path=str(source_audio_path),
        output_dir=str(output_dir),
        device=str(device),
        model_names=[str(model_name) for model_name in model_names],
    )
    source_path = Path(source_audio_path).resolve()
    resolved_output_dir = Path(output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    if not source_path.exists():
        raise FileNotFoundError(f"source audio is missing: {source_path}")

    resolved_base_manifest_paths = (
        [Path(path).resolve() for path in base_manifest_paths]
        if base_manifest_paths
        else _build_default_base_manifest_paths(
            source_audio_path=source_path,
            output_dir=resolved_output_dir,
            model_names=tuple(model_names),
            device=str(device),
        )
    )
    candidate_sets, stem_sample_rate, base_summaries = _load_candidate_sets(
        resolved_base_manifest_paths,
        source_audio_path=source_path,
    )
    _emit_progress(
        "base_manifests_loaded",
        base_manifest_count=len(base_summaries),
        stem_sample_rate=int(stem_sample_rate),
        candidate_set_count=len(candidate_sets),
    )
    source_audio, source_sample_rate = sf.read(str(source_path), always_2d=True)
    source_audio_for_candidate = _resample_audio(
        np.asarray(source_audio, dtype=np.float32),
        int(source_sample_rate),
        int(stem_sample_rate),
    )
    source_audio_resampled_for_shape = int(source_sample_rate) != int(stem_sample_rate)
    _emit_progress(
        "source_audio_ready",
        source_sample_rate=int(source_sample_rate),
        target_sample_rate=int(stem_sample_rate),
        source_resampled=bool(source_audio_resampled_for_shape),
        frame_count=int(source_audio_for_candidate.shape[0]),
    )

    _emit_progress("build_candidate_audio_call_start")
    outputs, diagnostics = build_joint_vocal_melody_v3_candidate_audio(
        candidate_sets,
        source_audio=source_audio_for_candidate,
        sample_rate=int(stem_sample_rate),
        cross_stem_suppression=float(cross_stem_suppression),
        non_target_suppression=float(non_target_suppression),
    )
    _emit_progress("build_candidate_audio_call_end", output_stems=sorted(outputs))

    stem_paths: dict[str, Path] = {}
    for stem_name, stem_audio in outputs.items():
        stem_path = resolved_output_dir / f"{stem_name}.flac"
        _emit_progress("write_stem_start", stem_name=str(stem_name), path=str(stem_path))
        sf.write(str(stem_path), stem_audio, int(stem_sample_rate), subtype=OUTPUT_SUBTYPE)
        stem_paths[stem_name] = stem_path
        _emit_progress("write_stem_end", stem_name=str(stem_name), path=str(stem_path))

    manifest = {
        "audio_path": str(source_path),
        "audio_sha256": sha256_file(source_path),
        "model_name": ROUTE_NAME,
        "sample_rate": int(stem_sample_rate),
        "export_subtype": OUTPUT_SUBTYPE,
        "melody_stem_included": True,
        "vocals_stem_included": True,
        "contract": _no_reference_contract(),
        "execution_contract": _execution_contract_from_base_summaries(base_summaries, device=str(device)),
        "pipeline": {
            "mode": ROUTE_NAME,
            "source": "authoritative_full_mix_plus_multiple_no_reference_model_stack_candidates",
            "base_manifests": base_summaries,
            "target_reference_read": False,
            "reference_sum_mix_read": False,
            "postprocess": "multi_model_stem_specific_proxy_router_v3",
            "parameters": {
                "cross_stem_suppression": float(cross_stem_suppression),
                "non_target_suppression": float(non_target_suppression),
                "source_sample_rate": int(source_sample_rate),
                "stem_sample_rate": int(stem_sample_rate),
                "source_audio_resampled_for_shape": bool(source_audio_resampled_for_shape),
            },
            "diagnostics": diagnostics,
            "source_audio": _audio_summary(source_audio_for_candidate),
        },
        "delivery": {
            "status": "requires_objective_quality_gate",
            "reason": "joint_no_reference_vocal_melody_v3_candidate_requires_ground_truth_benchmark_before_delivery",
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
        _emit_progress("write_listen_diagnostic_start", listen_dir=str(resolved_listen_dir))
        source_listen_path = resolved_listen_dir / "00_source_mix.flac"
        vocals_listen_path = resolved_listen_dir / "01_joint_v3_vocals.flac"
        melody_listen_path = resolved_listen_dir / "02_joint_v3_melody.flac"
        shutil.copyfile(source_path, source_listen_path)
        shutil.copyfile(stem_paths["vocals"], vocals_listen_path)
        shutil.copyfile(stem_paths["melody"], melody_listen_path)
        (resolved_listen_dir / "00_README.txt").write_text(
            "Diagnostic listen pack only.\n"
            "00 = authoritative source mix.\n"
            "01 = no-reference joint v3 vocals candidate.\n"
            "02 = no-reference joint v3 melody candidate.\n"
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
        _emit_progress("write_listen_diagnostic_end", listen_dir=str(resolved_listen_dir))

    manifest_path = resolved_output_dir / "stem_manifest.json"
    _emit_progress("write_manifest_start", manifest_path=str(manifest_path))
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    _emit_progress("write_manifest_end", manifest_path=str(manifest_path))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build no-reference joint vocals+melody v3 multi-candidate route.")
    parser.add_argument("source_audio_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--base-manifest", type=Path, action="append", default=[])
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--model-name", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cross-stem-suppression", type=float, default=0.92)
    parser.add_argument("--non-target-suppression", type=float, default=0.85)
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="build_joint_vocal_melody_candidate_v3.py",
        entrypoint_path="/app/tools/build_joint_vocal_melody_candidate_v3.py",
    )
    args = parse_args()
    model_names = tuple(args.model_name) if args.model_name else DEFAULT_MODEL_NAMES
    manifest = build_joint_vocal_melody_v3_candidate(
        source_audio_path=args.source_audio_path,
        output_dir=args.output_dir,
        base_manifest_paths=args.base_manifest,
        listen_dir=args.listen_dir,
        model_names=model_names,
        device=args.device,
        cross_stem_suppression=args.cross_stem_suppression,
        non_target_suppression=args.non_target_suppression,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
