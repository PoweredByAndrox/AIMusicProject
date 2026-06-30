from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from export_melband_vocal_preview import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_REPO_PATH,
    apply_light_debleed_presence,
    apply_peak_note_debleed,
    apply_surgical_notches,
    apply_windowed_band_subtract,
    apply_windowed_note_notch,
    copy_listen_file,
    save_audio_window_as_wav,
    write_config_with_overlap,
)


MODEL_ID = "melband_roformer_vocals"
MODEL_DISPLAY_NAME = "MelBand RoFormer vocal checkpoint"
MODEL_INPUT_STEM = "full_track"


def _emit(message: str) -> None:
    print(f"[melband_vocal_full] {message}", flush=True)


def _native_macos_diagnostic_allowed() -> bool:
    return (
        os.environ.get("AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC") == "1"
        and platform.system() == "Darwin"
    )


def verify_cuda_inference_device(device_id: int) -> dict[str, Any]:
    if int(device_id) < 0:
        raise RuntimeError(f"CUDA inference required for MelBand exporter but device_id is invalid: {device_id}")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on runtime image packaging
        raise RuntimeError("CUDA inference required for MelBand exporter but torch could not be imported") from exc

    cuda = getattr(torch, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if (
            _native_macos_diagnostic_allowed()
            and mps_backend is not None
            and bool(mps_backend.is_available())
        ):
            return {
                "cuda_available": False,
                "mps_built": bool(mps_backend.is_built()) if hasattr(mps_backend, "is_built") else True,
                "mps_available": True,
                "requested_device": "mps",
                "resolved_device": "mps",
                "device_id": int(device_id),
                "device_count": 1,
                "device_name": "Apple Metal Performance Shaders",
                "acceleration_mode": "native_macos_mps_diagnostic",
            }
        raise RuntimeError("CUDA inference required for MelBand exporter but torch.cuda.is_available() is false")
    device_count = int(cuda.device_count())
    if int(device_id) >= device_count:
        raise RuntimeError(
            "CUDA inference required for MelBand exporter but requested device_id "
            f"{device_id} is outside available CUDA device count {device_count}"
        )
    return {
        "cuda_available": True,
        "device_id": int(device_id),
        "device_count": device_count,
        "device_name": str(cuda.get_device_name(int(device_id))),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audio_file_summary(path: Path) -> dict[str, Any]:
    info = sf.info(str(path))
    return {
        "sample_rate": int(info.samplerate),
        "frames": int(info.frames),
        "channels": int(info.channels),
        "duration_sec": round(float(info.frames) / float(info.samplerate), 6) if info.samplerate else 0.0,
        "sha256": _sha256_file(path),
    }


def build_full_track_stem_manifest(
    *,
    source_audio_path: Path,
    vocals_flac_path: Path,
    instrumental_flac_path: Path,
    raw_ref_flac_path: Path,
    repo_path: Path,
    model_path: Path,
    config_template_path: Path,
    runtime_config_path: Path,
    device_id: int,
    num_overlap: int,
    sample_rate: int,
    frames: int,
    elapsed_sec: float,
    postprocess: dict[str, Any],
    model_id: str = MODEL_ID,
    model_display_name: str = MODEL_DISPLAY_NAME,
    stem_provenance: dict[str, dict[str, Any]] | None = None,
    residual_companion_materialized: bool = False,
    cuda_preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stem_provenance = stem_provenance or {}
    vocals_summary = {
        "path": str(vocals_flac_path),
        "role": "target",
        **_audio_file_summary(vocals_flac_path),
    }
    vocals_summary.update(stem_provenance.get("vocals", {}))
    instrumental_summary = {
        "path": str(instrumental_flac_path),
        "role": "companion",
        **_audio_file_summary(instrumental_flac_path),
    }
    instrumental_summary.update(stem_provenance.get("instrumental", {}))
    return {
        "schema_version": 1,
        "source_audio_path": str(source_audio_path),
        "model_id": model_id,
        "model_name": model_id,
        "model": {
            "model_id": model_id,
            "name": model_display_name,
            "family": "roformer",
            "repo_path": str(repo_path),
            "config_path": str(runtime_config_path),
            "config_template_path": str(config_template_path),
            "model_path": str(model_path),
            "device_id": int(device_id),
            "num_overlap": int(num_overlap),
        },
        "contract": {
            "full_track": True,
            "preview": False,
            "benchmark_compatible": True,
            "fallback_used": False,
            "hidden_inference": False,
            "registered_model_candidate": True,
            "per_track_hardcode_allowed": False,
            "residual_companion_materialized": bool(residual_companion_materialized),
        },
        "execution_contract": {
            "docker_runtime_required": True,
            "expected_acceleration_mode": "gpu_inference_then_cpu_dsp_scoring",
            "gpu_inference_required": True,
            "gpu_inference_skipped": False,
            "cuda_preflight": dict(cuda_preflight or {}),
        },
        "stems": {
            "vocals": vocals_summary,
            "instrumental": instrumental_summary,
        },
        "artifacts": {
            "raw_ref_flac": str(raw_ref_flac_path),
            "vocals_flac": str(vocals_flac_path),
            "instrumental_flac": str(instrumental_flac_path),
        },
        "postprocess": postprocess,
        "audio": {
            "sample_rate": int(sample_rate),
            "frames": int(frames),
            "duration_sec": round(float(frames) / float(sample_rate), 6) if sample_rate else 0.0,
        },
        "elapsed_sec": round(float(elapsed_sec), 3),
    }


def write_full_track_stem_manifest(
    *,
    output_dir: Path,
    source_audio_path: Path,
    vocals_flac_path: Path,
    instrumental_flac_path: Path,
    raw_ref_flac_path: Path,
    repo_path: Path,
    model_path: Path,
    config_template_path: Path,
    runtime_config_path: Path,
    device_id: int,
    num_overlap: int,
    sample_rate: int,
    frames: int,
    elapsed_sec: float,
    postprocess: dict[str, Any],
    model_id: str = MODEL_ID,
    model_display_name: str = MODEL_DISPLAY_NAME,
    stem_provenance: dict[str, dict[str, Any]] | None = None,
    residual_companion_materialized: bool = False,
    cuda_preflight: dict[str, Any] | None = None,
) -> Path:
    manifest = build_full_track_stem_manifest(
        source_audio_path=source_audio_path,
        vocals_flac_path=vocals_flac_path,
        instrumental_flac_path=instrumental_flac_path,
        raw_ref_flac_path=raw_ref_flac_path,
        repo_path=repo_path,
        model_path=model_path,
        config_template_path=config_template_path,
        runtime_config_path=runtime_config_path,
        device_id=device_id,
        num_overlap=num_overlap,
        sample_rate=sample_rate,
        frames=frames,
        elapsed_sec=elapsed_sec,
        postprocess=postprocess,
        model_id=model_id,
        model_display_name=model_display_name,
        stem_provenance=stem_provenance,
        residual_companion_materialized=residual_companion_materialized,
        cuda_preflight=cuda_preflight,
    )
    manifest_path = output_dir / "stem_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def resolve_model_output_path(model_output_dir: Path, stem_name: str) -> Path:
    exact_path = model_output_dir / f"{MODEL_INPUT_STEM}_{stem_name}.wav"
    if exact_path.exists():
        return exact_path

    target_name = exact_path.name.lower()
    for candidate_path in model_output_dir.glob(f"{MODEL_INPUT_STEM}_*.wav"):
        if candidate_path.name.lower() == target_name:
            return candidate_path

    raise FileNotFoundError(f"MelBand {stem_name} output is missing: {exact_path}")


def try_resolve_model_output_path(model_output_dir: Path, stem_name: str) -> Path | None:
    try:
        return resolve_model_output_path(model_output_dir, stem_name)
    except FileNotFoundError:
        return None


def normalize_model_stem_name(raw_value: Any) -> str:
    value = str(raw_value or "").strip().lower()
    if value in {"vocal", "vocals", "voice"}:
        return "vocals"
    if value in {"instrumental", "instrumentals", "minus"}:
        return "instrumental"
    return value


def load_melband_output_contract(config_path: Path) -> dict[str, Any]:
    payload = yaml.load(config_path.read_text(encoding="utf-8"), Loader=yaml.FullLoader) or {}
    model = payload.get("model") or {}
    training = payload.get("training") or {}
    num_stems = int(model.get("num_stems") or 0)
    target_instrument = training.get("target_instrument")
    normalized_target = normalize_model_stem_name(target_instrument)
    if num_stems == 1 and normalized_target in {"vocals", "instrumental"}:
        return {
            "num_stems": num_stems,
            "target_instrument": target_instrument,
            "expected_model_stems": (normalized_target,),
            "allow_residual_companion_materialization": True,
        }
    return {
        "num_stems": num_stems,
        "target_instrument": target_instrument,
        "expected_model_stems": ("vocals", "instrumental"),
        "allow_residual_companion_materialization": False,
    }


def _require_matching_audio_layout(
    *,
    raw_audio: np.ndarray,
    raw_sample_rate: int,
    stem_audio: np.ndarray,
    stem_sample_rate: int | None,
    stem_name: str,
) -> None:
    if stem_sample_rate is None:
        raise RuntimeError(f"{stem_name} sample rate is required for residual companion materialization")
    if int(stem_sample_rate) != int(raw_sample_rate):
        raise RuntimeError(
            f"{stem_name} sample rate mismatch for residual companion materialization: "
            f"{stem_sample_rate} != {raw_sample_rate}"
        )
    if tuple(stem_audio.shape) != tuple(raw_audio.shape):
        raise RuntimeError(
            f"{stem_name} audio shape mismatch for residual companion materialization: "
            f"{tuple(stem_audio.shape)} != {tuple(raw_audio.shape)}"
        )


def materialize_missing_companion_stems(
    *,
    output_contract: dict[str, Any],
    raw_audio: np.ndarray,
    raw_sample_rate: int,
    vocals_audio: np.ndarray | None,
    vocals_sample_rate: int | None,
    instrumental_audio: np.ndarray | None,
    instrumental_sample_rate: int | None,
) -> dict[str, Any]:
    if vocals_audio is None and instrumental_audio is None:
        raise FileNotFoundError("MelBand output is missing both vocals and instrumental stems")

    residual_companion_materialized = False
    stem_provenance: dict[str, dict[str, Any]] = {}
    expected_model_stems = {
        normalize_model_stem_name(stem_name)
        for stem_name in tuple(output_contract.get("expected_model_stems") or ())
    }
    allow_residual = bool(output_contract.get("allow_residual_companion_materialization"))

    if expected_model_stems == {"instrumental"} and instrumental_audio is None:
        raise RuntimeError("MelBand output contract expected instrumental output but it is missing")
    if expected_model_stems == {"vocals"} and vocals_audio is None:
        raise RuntimeError("MelBand output contract expected vocals output but it is missing")
    if expected_model_stems != {"instrumental"} and vocals_audio is None and not allow_residual:
        raise FileNotFoundError("MelBand vocals output is missing")
    if expected_model_stems != {"vocals"} and instrumental_audio is None and not allow_residual:
        raise FileNotFoundError("MelBand instrumental output is missing")

    if vocals_audio is None:
        assert instrumental_audio is not None
        _require_matching_audio_layout(
            raw_audio=raw_audio,
            raw_sample_rate=raw_sample_rate,
            stem_audio=instrumental_audio,
            stem_sample_rate=instrumental_sample_rate,
            stem_name="instrumental",
        )
        vocals_audio = raw_audio - instrumental_audio
        vocals_sample_rate = int(raw_sample_rate)
        residual_companion_materialized = True
        stem_provenance["vocals"] = {
            "source": "residual_from_raw_minus_instrumental",
            "model_output": False,
            "residual_companion": True,
        }
    else:
        stem_provenance["vocals"] = {"source": "model_output", "model_output": True}

    if instrumental_audio is None:
        assert vocals_audio is not None
        _require_matching_audio_layout(
            raw_audio=raw_audio,
            raw_sample_rate=raw_sample_rate,
            stem_audio=vocals_audio,
            stem_sample_rate=vocals_sample_rate,
            stem_name="vocals",
        )
        instrumental_audio = raw_audio - vocals_audio
        instrumental_sample_rate = int(raw_sample_rate)
        residual_companion_materialized = True
        stem_provenance["instrumental"] = {
            "source": "residual_from_raw_minus_vocals",
            "model_output": False,
            "residual_companion": True,
        }
    else:
        stem_provenance["instrumental"] = {"source": "model_output", "model_output": True}

    return {
        "vocals_audio": vocals_audio,
        "vocals_sample_rate": int(vocals_sample_rate or raw_sample_rate),
        "instrumental_audio": instrumental_audio,
        "instrumental_sample_rate": int(instrumental_sample_rate or raw_sample_rate),
        "stem_provenance": stem_provenance,
        "residual_companion_materialized": residual_companion_materialized,
    }


def parse_frequency_list(raw_value: str) -> list[float]:
    return [float(item.strip()) for item in str(raw_value).split(",") if item.strip()]


def build_postprocess_manifest(args: argparse.Namespace, surgical_centers: list[float], band_subtract_centers: list[float]) -> dict[str, Any]:
    postprocess_name = (
        "window_band_subtract_v1"
        if float(args.window_band_subtract_strength) > 0.0
        else "surgical_notch_v1"
        if float(args.surgical_notch_strength) > 0.0
        else "windowed_note_notch_v1"
        if float(args.window_notch_strength) > 0.0
        else "peak_note_melody_bleed_suppressor_v1"
        if float(args.note_peak_strength) > 0.0
        else "light_high_band_debleed_presence_v1"
        if float(args.light_debleed_strength) > 0.0 or float(args.presence_gain) > 1.0
        else None
    )
    return {
        "name": postprocess_name,
        "light_debleed_strength": float(args.light_debleed_strength),
        "light_debleed_start_hz": float(args.light_debleed_start_hz),
        "light_debleed_margin": float(args.light_debleed_margin),
        "light_debleed_power": float(args.light_debleed_power),
        "presence_gain": float(args.presence_gain),
        "presence_start_hz": float(args.presence_start_hz),
        "note_peak_strength": float(args.note_peak_strength),
        "note_peak_start_hz": float(args.note_peak_start_hz),
        "note_peak_dominance_ratio": float(args.note_peak_dominance_ratio),
        "note_peak_local_ratio": float(args.note_peak_local_ratio),
        "note_peak_spread_bins": int(args.note_peak_spread_bins),
        "note_peak_presence_gain": float(args.note_peak_presence_gain),
        "note_peak_presence_start_hz": float(args.note_peak_presence_start_hz),
        "window_notch_strength": float(args.window_notch_strength),
        "window_notch_focus_start_sec": float(args.window_notch_focus_start_sec),
        "window_notch_focus_end_sec": float(args.window_notch_focus_end_sec),
        "window_notch_search_pre_roll_sec": float(args.window_notch_search_pre_roll_sec),
        "window_notch_freq_min_hz": float(args.window_notch_freq_min_hz),
        "window_notch_freq_max_hz": float(args.window_notch_freq_max_hz),
        "window_notch_peak_count": int(args.window_notch_peak_count),
        "window_notch_bin_radius": int(args.window_notch_bin_radius),
        "window_notch_min_peak_separation_bins": int(args.window_notch_min_peak_separation_bins),
        "window_notch_presence_gain": float(args.window_notch_presence_gain),
        "surgical_notch_strength": float(args.surgical_notch_strength),
        "surgical_notch_focus_start_sec": float(args.surgical_notch_focus_start_sec),
        "surgical_notch_focus_end_sec": float(args.surgical_notch_focus_end_sec),
        "surgical_notch_fade_sec": float(args.surgical_notch_fade_sec),
        "surgical_notch_centers_hz": surgical_centers,
        "surgical_notch_half_width_hz": float(args.surgical_notch_half_width_hz),
        "surgical_notch_preserve_gain": float(args.surgical_notch_preserve_gain),
        "window_band_subtract_strength": float(args.window_band_subtract_strength),
        "window_band_subtract_focus_start_sec": float(args.window_band_subtract_focus_start_sec),
        "window_band_subtract_focus_end_sec": float(args.window_band_subtract_focus_end_sec),
        "window_band_subtract_fade_sec": float(args.window_band_subtract_fade_sec),
        "window_band_subtract_centers_hz": band_subtract_centers,
        "window_band_subtract_half_width_hz": float(args.window_band_subtract_half_width_hz),
        "window_band_subtract_residual_floor": float(args.window_band_subtract_residual_floor),
        "window_band_subtract_preserve_gain": float(args.window_band_subtract_preserve_gain),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export full-track MelBand RoFormer vocals/instrumental stems.")
    parser.add_argument("source_audio_path")
    parser.add_argument("output_dir")
    parser.add_argument("--repo-path", default=str(DEFAULT_REPO_PATH))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--manifest-model-id", default=MODEL_ID)
    parser.add_argument("--manifest-model-display-name", default=MODEL_DISPLAY_NAME)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--listen-dir")
    parser.add_argument("--listen-label", default="01_melband_roformer_vocals_full.flac")
    parser.add_argument("--num-overlap", type=int, default=2)
    parser.add_argument("--light-debleed-strength", type=float, default=0.0)
    parser.add_argument("--light-debleed-start-hz", type=float, default=2600.0)
    parser.add_argument("--light-debleed-margin", type=float, default=1.08)
    parser.add_argument("--light-debleed-power", type=float, default=1.4)
    parser.add_argument("--presence-gain", type=float, default=1.0)
    parser.add_argument("--presence-start-hz", type=float, default=1800.0)
    parser.add_argument("--note-peak-strength", type=float, default=0.0)
    parser.add_argument("--note-peak-start-hz", type=float, default=2400.0)
    parser.add_argument("--note-peak-dominance-ratio", type=float, default=1.45)
    parser.add_argument("--note-peak-local-ratio", type=float, default=1.7)
    parser.add_argument("--note-peak-spread-bins", type=int, default=1)
    parser.add_argument("--note-peak-presence-gain", type=float, default=1.0)
    parser.add_argument("--note-peak-presence-start-hz", type=float, default=1500.0)
    parser.add_argument("--window-notch-strength", type=float, default=0.0)
    parser.add_argument("--window-notch-focus-start-sec", type=float, default=8.0)
    parser.add_argument("--window-notch-focus-end-sec", type=float, default=10.0)
    parser.add_argument("--window-notch-search-pre-roll-sec", type=float, default=0.35)
    parser.add_argument("--window-notch-freq-min-hz", type=float, default=1700.0)
    parser.add_argument("--window-notch-freq-max-hz", type=float, default=3600.0)
    parser.add_argument("--window-notch-peak-count", type=int, default=3)
    parser.add_argument("--window-notch-bin-radius", type=int, default=2)
    parser.add_argument("--window-notch-min-peak-separation-bins", type=int, default=10)
    parser.add_argument("--window-notch-presence-gain", type=float, default=1.02)
    parser.add_argument("--surgical-notch-strength", type=float, default=0.0)
    parser.add_argument("--surgical-notch-focus-start-sec", type=float, default=8.0)
    parser.add_argument("--surgical-notch-focus-end-sec", type=float, default=10.0)
    parser.add_argument("--surgical-notch-fade-sec", type=float, default=0.25)
    parser.add_argument("--surgical-notch-centers-hz", default="")
    parser.add_argument("--surgical-notch-half-width-hz", type=float, default=65.0)
    parser.add_argument("--surgical-notch-preserve-gain", type=float, default=1.02)
    parser.add_argument("--window-band-subtract-strength", type=float, default=0.0)
    parser.add_argument("--window-band-subtract-focus-start-sec", type=float, default=8.0)
    parser.add_argument("--window-band-subtract-focus-end-sec", type=float, default=10.0)
    parser.add_argument("--window-band-subtract-fade-sec", type=float, default=0.25)
    parser.add_argument("--window-band-subtract-centers-hz", default="")
    parser.add_argument("--window-band-subtract-half-width-hz", type=float, default=130.0)
    parser.add_argument("--window-band-subtract-residual-floor", type=float, default=0.16)
    parser.add_argument("--window-band-subtract-preserve-gain", type=float, default=1.02)
    return parser


def main() -> None:
    require_docker_runtime(
        tool_name="export_melband_vocal_full.py",
        entrypoint_path="/app/tools/export_melband_vocal_full.py",
    )
    args = build_arg_parser().parse_args()
    source_audio_path = Path(args.source_audio_path)
    output_dir = Path(args.output_dir)
    repo_path = Path(args.repo_path)
    model_path = Path(args.model_path)
    config_path = Path(args.config_path)
    listen_dir = Path(args.listen_dir) if args.listen_dir else None

    output_dir.mkdir(parents=True, exist_ok=True)
    if not source_audio_path.exists():
        raise FileNotFoundError(f"Source audio is missing: {source_audio_path}")
    if not repo_path.exists():
        raise FileNotFoundError(f"MelBand repo is missing: {repo_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"MelBand checkpoint is missing: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"MelBand config is missing: {config_path}")

    cuda_preflight = verify_cuda_inference_device(int(args.device_id))
    _emit(
        "cuda_preflight "
        f"device_id={cuda_preflight['device_id']} "
        f"device_count={cuda_preflight['device_count']} "
        f"device_name={cuda_preflight['device_name']}"
    )

    start_wall = time.time()
    with tempfile.TemporaryDirectory(prefix="melband_full_input_") as input_dir_str, tempfile.TemporaryDirectory(
        prefix="melband_full_output_"
    ) as model_output_dir_str, tempfile.TemporaryDirectory(prefix="melband_full_config_") as config_dir_str:
        input_dir = Path(input_dir_str)
        model_output_dir = Path(model_output_dir_str)
        config_dir = Path(config_dir_str)
        full_input_path = input_dir / f"{MODEL_INPUT_STEM}.wav"
        runtime_config_path = config_dir / "config_runtime.yaml"

        _emit(f"ENTER source_audio={source_audio_path} output_dir={output_dir} full_track=true")
        sample_rate, frames = save_audio_window_as_wav(
            source_audio_path,
            full_input_path,
            start_sec=0.0,
            end_sec=None,
        )
        _emit(f"prepared_full_track_wav sample_rate={sample_rate} frames={frames}")
        write_config_with_overlap(config_path, runtime_config_path, num_overlap=int(args.num_overlap))
        output_contract = load_melband_output_contract(runtime_config_path)

        command = [
            sys.executable,
            "inference.py",
            "--model_type",
            "mel_band_roformer",
            "--config_path",
            str(runtime_config_path),
            "--model_path",
            str(model_path),
            "--input_folder",
            str(input_dir),
            "--store_dir",
            str(model_output_dir),
            "--device_ids",
            str(args.device_id),
        ]
        completed = subprocess.run(
            command,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=True,
        )
        if completed.stdout.strip():
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
        if completed.stderr.strip():
            print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", flush=True)

        vocals_wav = try_resolve_model_output_path(model_output_dir, "vocals")
        instrumental_wav = try_resolve_model_output_path(model_output_dir, "instrumental")

        vocals_audio: np.ndarray | None
        instrumental_audio: np.ndarray | None
        vocals_sr: int | None
        instrumental_sr: int | None
        vocals_audio, vocals_sr = (
            sf.read(str(vocals_wav), always_2d=True) if vocals_wav is not None else (None, None)
        )
        instrumental_audio, instrumental_sr = (
            sf.read(str(instrumental_wav), always_2d=True) if instrumental_wav is not None else (None, None)
        )
        raw_audio, raw_sr = sf.read(str(full_input_path), always_2d=True)
        resolved_stems = materialize_missing_companion_stems(
            output_contract=output_contract,
            raw_audio=raw_audio,
            raw_sample_rate=int(raw_sr),
            vocals_audio=vocals_audio,
            vocals_sample_rate=vocals_sr,
            instrumental_audio=instrumental_audio,
            instrumental_sample_rate=instrumental_sr,
        )
        vocals_audio = resolved_stems["vocals_audio"]
        vocals_sr = int(resolved_stems["vocals_sample_rate"])
        instrumental_audio = resolved_stems["instrumental_audio"]
        instrumental_sr = int(resolved_stems["instrumental_sample_rate"])
        tuned_vocals_audio = apply_light_debleed_presence(
            vocals_audio,
            instrumental_audio,
            sample_rate=int(vocals_sr),
            debleed_strength=float(args.light_debleed_strength),
            debleed_start_hz=float(args.light_debleed_start_hz),
            debleed_margin=float(args.light_debleed_margin),
            debleed_power=float(args.light_debleed_power),
            presence_gain=float(args.presence_gain),
            presence_start_hz=float(args.presence_start_hz),
        )
        tuned_vocals_audio = apply_peak_note_debleed(
            tuned_vocals_audio,
            instrumental_audio,
            sample_rate=int(vocals_sr),
            peak_strength=float(args.note_peak_strength),
            peak_start_hz=float(args.note_peak_start_hz),
            peak_dominance_ratio=float(args.note_peak_dominance_ratio),
            peak_local_ratio=float(args.note_peak_local_ratio),
            peak_spread_bins=int(args.note_peak_spread_bins),
            presence_gain=float(args.note_peak_presence_gain),
            presence_start_hz=float(args.note_peak_presence_start_hz),
        )
        tuned_vocals_audio = apply_windowed_note_notch(
            tuned_vocals_audio,
            instrumental_audio,
            sample_rate=int(vocals_sr),
            focus_start_sec=float(args.window_notch_focus_start_sec),
            focus_end_sec=float(args.window_notch_focus_end_sec),
            search_pre_roll_sec=float(args.window_notch_search_pre_roll_sec),
            notch_strength=float(args.window_notch_strength),
            notch_freq_min_hz=float(args.window_notch_freq_min_hz),
            notch_freq_max_hz=float(args.window_notch_freq_max_hz),
            notch_peak_count=int(args.window_notch_peak_count),
            notch_bin_radius=int(args.window_notch_bin_radius),
            notch_min_peak_separation_bins=int(args.window_notch_min_peak_separation_bins),
            notch_presence_gain=float(args.window_notch_presence_gain),
        )
        surgical_centers = parse_frequency_list(str(args.surgical_notch_centers_hz))
        tuned_vocals_audio = apply_surgical_notches(
            tuned_vocals_audio,
            sample_rate=int(vocals_sr),
            focus_start_sec=float(args.surgical_notch_focus_start_sec),
            focus_end_sec=float(args.surgical_notch_focus_end_sec),
            fade_sec=float(args.surgical_notch_fade_sec),
            centers_hz=surgical_centers,
            half_width_hz=float(args.surgical_notch_half_width_hz),
            notch_strength=float(args.surgical_notch_strength),
            preserve_gain=float(args.surgical_notch_preserve_gain),
        )
        band_subtract_centers = parse_frequency_list(str(args.window_band_subtract_centers_hz))
        tuned_vocals_audio = apply_windowed_band_subtract(
            tuned_vocals_audio,
            instrumental_audio,
            sample_rate=int(vocals_sr),
            focus_start_sec=float(args.window_band_subtract_focus_start_sec),
            focus_end_sec=float(args.window_band_subtract_focus_end_sec),
            fade_sec=float(args.window_band_subtract_fade_sec),
            centers_hz=band_subtract_centers,
            half_width_hz=float(args.window_band_subtract_half_width_hz),
            subtract_strength=float(args.window_band_subtract_strength),
            residual_floor=float(args.window_band_subtract_residual_floor),
            preserve_gain=float(args.window_band_subtract_preserve_gain),
        )

        vocals_flac_path = output_dir / "vocals.flac"
        instrumental_flac_path = output_dir / "instrumental.flac"
        raw_ref_flac_path = output_dir / "raw_ref.flac"
        sf.write(str(vocals_flac_path), tuned_vocals_audio, vocals_sr, subtype="PCM_24")
        sf.write(str(instrumental_flac_path), instrumental_audio, instrumental_sr, subtype="PCM_24")
        sf.write(str(raw_ref_flac_path), raw_audio, raw_sr, subtype="PCM_24")
        persisted_runtime_config_path = output_dir / "config_runtime.yaml"
        shutil.copy2(runtime_config_path, persisted_runtime_config_path)

        manifest_path = write_full_track_stem_manifest(
            output_dir=output_dir,
            source_audio_path=source_audio_path,
            vocals_flac_path=vocals_flac_path,
            instrumental_flac_path=instrumental_flac_path,
            raw_ref_flac_path=raw_ref_flac_path,
            repo_path=repo_path,
            model_path=model_path,
            config_template_path=config_path,
            runtime_config_path=persisted_runtime_config_path,
            device_id=int(args.device_id),
            num_overlap=int(args.num_overlap),
            sample_rate=int(raw_sr),
            frames=int(raw_audio.shape[0]),
            elapsed_sec=time.time() - start_wall,
            postprocess=build_postprocess_manifest(args, surgical_centers, band_subtract_centers),
            model_id=str(args.manifest_model_id),
            model_display_name=str(args.manifest_model_display_name),
            stem_provenance=resolved_stems["stem_provenance"],
            residual_companion_materialized=bool(resolved_stems["residual_companion_materialized"]),
            cuda_preflight=cuda_preflight,
        )

        if listen_dir is not None:
            listen_dir.mkdir(parents=True, exist_ok=True)
            copy_listen_file(raw_ref_flac_path, listen_dir / "00_raw_ref.flac")
            copy_listen_file(vocals_flac_path, listen_dir / args.listen_label)
            (listen_dir / "00_README.txt").write_text(
                "Listen only here.\n00 = full raw source reference.\nOther files = full-track Mel-Band RoFormer candidates.\n",
                encoding="utf-8",
            )
            candidate_map_path = listen_dir / "candidate_map.json"
            candidate_map: dict[str, str] = {}
            if candidate_map_path.exists():
                candidate_map = json.loads(candidate_map_path.read_text(encoding="utf-8"))
            candidate_map["00_raw_ref.flac"] = str(raw_ref_flac_path)
            candidate_map[args.listen_label] = str(vocals_flac_path)
            candidate_map_path.write_text(json.dumps(candidate_map, indent=2, ensure_ascii=False), encoding="utf-8")

        elapsed_sec = round(time.time() - start_wall, 3)
        _emit(f"EXIT output_dir={output_dir} elapsed_sec={elapsed_sec}")
        print(json.dumps({"output_dir": str(output_dir), "manifest_path": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
