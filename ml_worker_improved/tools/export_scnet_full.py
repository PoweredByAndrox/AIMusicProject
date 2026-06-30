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
from typing import Any, Sequence

import soundfile as sf

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from export_dereverb_vocal_preview import (  # noqa: E402
    copy_listen_file,
    save_audio_window_as_wav,
    write_config_with_overlap,
)

MODEL_ID = "scnet_xl_ihf_musdb18"
MODEL_DISPLAY_NAME = "SCNet XL IHF MUSDB18 checkpoint"
MODEL_TYPE = "scnet"
MODEL_INPUT_STEM = "full_track"
SCNET_STEMS = ("drums", "bass", "other", "vocals")

DEFAULT_REPO_PATH = Path("/app/vendor/music_source_separation_training")
DEFAULT_CONFIG_PATH = Path("/app/vendor/scnet_xl_ihf/config_musdb18_scnet_xl_more_wide_v5.yaml")
DEFAULT_MODEL_PATH = Path("/app/vendor/scnet_xl_ihf/model_scnet_ep_36_sdr_10.0891.ckpt")
DEFAULT_SCNET_TIMEOUT_SEC = 180


def _emit(message: str) -> None:
    print(f"[scnet_full] {message}", flush=True)


def _native_macos_diagnostic_allowed() -> bool:
    return (
        os.environ.get("AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC") == "1"
        and platform.system() == "Darwin"
    )


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


def verify_cuda_inference_device(device_id: int) -> dict[str, Any]:
    if int(device_id) < 0:
        raise RuntimeError(f"CUDA inference required for SCNet exporter but device_id is invalid: {device_id}")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on runtime image packaging
        raise RuntimeError("CUDA inference required for SCNet exporter but torch could not be imported") from exc

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
        raise RuntimeError("CUDA inference required for SCNet exporter but torch.cuda.is_available() is false")
    device_count = int(cuda.device_count())
    if int(device_id) >= device_count:
        raise RuntimeError(
            "CUDA inference required for SCNet exporter but requested device_id "
            f"{device_id} is outside available CUDA device count {device_count}"
        )
    return {
        "cuda_available": True,
        "device_id": int(device_id),
        "device_count": device_count,
        "device_name": str(cuda.get_device_name(int(device_id))),
    }


def _build_no_reference_contract() -> dict[str, Any]:
    return {
        "full_track": True,
        "preview": False,
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
        "route_name": "scnet_xl_ihf_full_track",
    }


def build_full_track_stem_manifest(
    *,
    source_audio_path: Path,
    stem_flac_paths: dict[str, Path],
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
    cuda_preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_audio_path": str(source_audio_path),
        "model_id": MODEL_ID,
        "model_name": MODEL_ID,
        "model": {
            "model_id": MODEL_ID,
            "name": MODEL_DISPLAY_NAME,
            "family": "scnet",
            "model_type": MODEL_TYPE,
            "repo_path": str(repo_path),
            "config_path": str(runtime_config_path),
            "config_template_path": str(config_template_path),
            "model_path": str(model_path),
            "device_id": int(device_id),
            "num_overlap": int(num_overlap),
            "target_output": "multi_stem",
            "output_stems": list(SCNET_STEMS),
        },
        "contract": _build_no_reference_contract(),
        "execution_contract": {
            "docker_runtime_required": True,
            "expected_acceleration_mode": "gpu_inference_then_cpu_dsp_scoring",
            "gpu_inference_required": True,
            "gpu_inference_skipped": False,
            "cuda_preflight": dict(cuda_preflight or {}),
        },
        "stems": {
            stem_name: {
                "path": str(stem_path),
                "role": "target",
                **_audio_file_summary(stem_path),
            }
            for stem_name, stem_path in stem_flac_paths.items()
        },
        "raw_reference": {
            "path": str(raw_ref_flac_path),
            **_audio_file_summary(raw_ref_flac_path),
        },
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
    stem_flac_paths: dict[str, Path],
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
    cuda_preflight: dict[str, Any] | None = None,
) -> Path:
    manifest = build_full_track_stem_manifest(
        source_audio_path=source_audio_path,
        stem_flac_paths=stem_flac_paths,
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
        cuda_preflight=cuda_preflight,
    )
    manifest_path = output_dir / "stem_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export full-track SCNet multistem separation.")
    parser.add_argument("source_audio_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--repo-path", type=Path, default=DEFAULT_REPO_PATH)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-overlap", type=int, default=4)
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_SCNET_TIMEOUT_SEC)
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--listen-stem", default="drums")
    parser.add_argument("--listen-label", default="01_scnet_drums.flac")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="export_scnet_full.py",
        entrypoint_path="/app/tools/export_scnet_full.py",
    )
    args = build_arg_parser().parse_args(argv)
    source_audio_path = Path(args.source_audio_path)
    output_dir = Path(args.output_dir)
    repo_path = Path(args.repo_path)
    config_path = Path(args.config_path)
    model_path = Path(args.model_path)
    listen_dir = Path(args.listen_dir) if args.listen_dir else None

    output_dir.mkdir(parents=True, exist_ok=True)
    if not source_audio_path.exists():
        raise FileNotFoundError(f"Source audio is missing: {source_audio_path}")
    if not repo_path.exists():
        raise FileNotFoundError(f"MSST repo is missing: {repo_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"SCNet config is missing: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"SCNet model is missing: {model_path}")

    cuda_preflight = verify_cuda_inference_device(int(args.device_id))
    _emit(
        "cuda_preflight "
        f"device_id={cuda_preflight['device_id']} "
        f"device_count={cuda_preflight['device_count']} "
        f"device_name={cuda_preflight['device_name']}"
    )

    started_at = time.time()
    with tempfile.TemporaryDirectory(prefix="scnet_full_input_") as input_dir_str, tempfile.TemporaryDirectory(
        prefix="scnet_full_output_"
    ) as model_output_dir_str, tempfile.TemporaryDirectory(prefix="scnet_full_config_") as config_dir_str:
        input_dir = Path(input_dir_str)
        model_output_dir = Path(model_output_dir_str)
        config_dir = Path(config_dir_str)
        full_input_path = input_dir / f"{MODEL_INPUT_STEM}.wav"
        runtime_config_path = config_dir / "config_runtime.yaml"

        if int(args.timeout_sec) <= 0:
            raise ValueError(f"SCNet timeout must be positive: {args.timeout_sec}")
        _emit(
            f"ENTER source_audio={source_audio_path} output_dir={output_dir} "
            f"full_track=true timeout_sec={int(args.timeout_sec)}"
        )
        sample_rate, frames = save_audio_window_as_wav(
            source_audio_path,
            full_input_path,
            start_sec=0.0,
            end_sec=None,
        )
        _emit(f"prepared_full_track_wav sample_rate={sample_rate} frames={frames}")
        write_config_with_overlap(config_path, runtime_config_path, num_overlap=int(args.num_overlap))

        command = [
            sys.executable,
            "inference.py",
            "--model_type",
            MODEL_TYPE,
            "--config_path",
            str(runtime_config_path),
            "--start_check_point",
            str(model_path),
            "--input_folder",
            str(input_dir),
            "--store_dir",
            str(model_output_dir),
            "--device_ids",
            str(args.device_id),
            "--disable_detailed_pbar",
            "--filename_template",
            "{file_name}/{instr}",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                check=False,
                timeout=int(args.timeout_sec),
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.output or ""
            stderr = exc.stderr or ""
            if stdout:
                print(stdout, end="" if str(stdout).endswith("\n") else "\n", flush=True)
            if stderr:
                print(stderr, end="" if str(stderr).endswith("\n") else "\n", flush=True)
            raise RuntimeError(f"scnet_subprocess_timeout timeout_sec={int(args.timeout_sec)}") from exc
        if completed.stdout.strip():
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
        if completed.stderr.strip():
            print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", flush=True)
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                command,
                output=completed.stdout,
                stderr=completed.stderr,
            )

        model_stem_dir = model_output_dir / MODEL_INPUT_STEM
        stem_flac_paths: dict[str, Path] = {}
        for stem_name in SCNET_STEMS:
            source_wav_path = model_stem_dir / f"{stem_name}.wav"
            if not source_wav_path.exists():
                raise FileNotFoundError(f"SCNet did not produce required stem: {source_wav_path}")
            stem_audio, stem_sample_rate = sf.read(str(source_wav_path), always_2d=True)
            stem_flac_path = output_dir / f"{stem_name}.flac"
            sf.write(str(stem_flac_path), stem_audio, stem_sample_rate, subtype="PCM_24")
            stem_flac_paths[stem_name] = stem_flac_path

        raw_ref_flac_path = output_dir / "00_raw_ref.flac"
        shutil.copyfile(full_input_path, raw_ref_flac_path)
        elapsed_sec = time.time() - started_at
        manifest_path = write_full_track_stem_manifest(
            output_dir=output_dir,
            source_audio_path=source_audio_path,
            stem_flac_paths=stem_flac_paths,
            raw_ref_flac_path=raw_ref_flac_path,
            repo_path=repo_path,
            model_path=model_path,
            config_template_path=config_path,
            runtime_config_path=runtime_config_path,
            device_id=int(args.device_id),
            num_overlap=int(args.num_overlap),
            sample_rate=int(sample_rate),
            frames=int(frames),
            elapsed_sec=elapsed_sec,
            cuda_preflight=cuda_preflight,
        )

        listen_summary: dict[str, str] | None = None
        if listen_dir is not None:
            listen_stem = str(args.listen_stem)
            if listen_stem not in stem_flac_paths:
                raise KeyError(f"Requested listen stem is missing from SCNet output: {listen_stem}")
            listen_dir.mkdir(parents=True, exist_ok=True)
            listen_raw_path = listen_dir / "00_raw_ref.flac"
            listen_stem_path = listen_dir / str(args.listen_label)
            copy_listen_file(raw_ref_flac_path, listen_raw_path)
            copy_listen_file(stem_flac_paths[listen_stem], listen_stem_path)
            (listen_dir / "candidate_map.json").write_text(
                json.dumps(
                    {
                        "00_raw_ref.flac": str(raw_ref_flac_path),
                        str(args.listen_label): str(stem_flac_paths[listen_stem]),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            listen_summary = {
                "listen_dir": str(listen_dir),
                "listen_stem": listen_stem,
                "listen_label": str(args.listen_label),
            }

    _emit(f"EXIT manifest_path={manifest_path}")
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "output_dir": str(output_dir),
                "listen": listen_summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
