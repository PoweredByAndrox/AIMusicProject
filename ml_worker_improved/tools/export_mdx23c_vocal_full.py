from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

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
from export_mdx23c_vocal_preview import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_REPO_PATH,
)


MODEL_ID = "mdx23c_vocals"
MODEL_DISPLAY_NAME = "MDX23C vocal checkpoint"
MODEL_INPUT_STEM = "full_track"
MODEL_TYPE = "mdx23c"


def _emit(message: str) -> None:
    print(f"[mdx23c_vocal_full] {message}", flush=True)


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
        raise RuntimeError(f"CUDA inference required for MDX23C exporter but device_id is invalid: {device_id}")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on runtime image packaging
        raise RuntimeError("CUDA inference required for MDX23C exporter but torch could not be imported") from exc

    cuda = getattr(torch, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        raise RuntimeError("CUDA inference required for MDX23C exporter but torch.cuda.is_available() is false")
    device_count = int(cuda.device_count())
    if int(device_id) >= device_count:
        raise RuntimeError(
            "CUDA inference required for MDX23C exporter but requested device_id "
            f"{device_id} is outside available CUDA device count {device_count}"
        )
    return {
        "cuda_available": True,
        "device_id": int(device_id),
        "device_count": device_count,
        "device_name": str(cuda.get_device_name(int(device_id))),
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
            "family": "mdx",
            "model_type": MODEL_TYPE,
            "repo_path": str(repo_path),
            "config_path": str(runtime_config_path),
            "config_template_path": str(config_template_path),
            "model_path": str(model_path),
            "device_id": int(device_id),
            "num_overlap": int(num_overlap),
            "target_output": "vocals",
        },
        "contract": {
            "full_track": True,
            "preview": False,
            "benchmark_compatible": True,
            "fallback_used": False,
            "hidden_inference": False,
            "registered_model_candidate": True,
            "per_track_hardcode_allowed": False,
        },
        "execution_contract": {
            "docker_runtime_required": True,
            "expected_acceleration_mode": "gpu_inference_then_cpu_dsp_scoring",
            "gpu_inference_required": True,
            "gpu_inference_skipped": False,
            "cuda_preflight": dict(cuda_preflight or {}),
        },
        "stems": {
            "vocals": {
                "path": str(vocals_flac_path),
                "role": "target",
                **_audio_file_summary(vocals_flac_path),
            },
            "instrumental": {
                "path": str(instrumental_flac_path),
                "role": "companion",
                **_audio_file_summary(instrumental_flac_path),
            },
        },
        "artifacts": {
            "raw_ref_flac": str(raw_ref_flac_path),
            "vocals_flac": str(vocals_flac_path),
            "instrumental_flac": str(instrumental_flac_path),
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
        cuda_preflight=cuda_preflight,
    )
    manifest_path = output_dir / "stem_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export full-track MDX23C vocals/instrumental stems.")
    parser.add_argument("source_audio_path")
    parser.add_argument("output_dir")
    parser.add_argument("--repo-path", default=str(DEFAULT_REPO_PATH))
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-overlap", type=int, default=4)
    parser.add_argument("--listen-dir")
    parser.add_argument("--listen-label", default="01_mdx23c_vocals_full.flac")
    return parser


def main() -> None:
    require_docker_runtime(
        tool_name="export_mdx23c_vocal_full.py",
        entrypoint_path="/app/tools/export_mdx23c_vocal_full.py",
    )
    args = build_arg_parser().parse_args()
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
        raise FileNotFoundError(f"MDX23C config is missing: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"MDX23C model is missing: {model_path}")

    cuda_preflight = verify_cuda_inference_device(int(args.device_id))
    _emit(
        "cuda_preflight "
        f"device_id={cuda_preflight['device_id']} "
        f"device_count={cuda_preflight['device_count']} "
        f"device_name={cuda_preflight['device_name']}"
    )

    start_wall = time.time()
    with tempfile.TemporaryDirectory(prefix="mdx23c_full_input_") as input_dir_str, tempfile.TemporaryDirectory(
        prefix="mdx23c_full_output_"
    ) as model_output_dir_str, tempfile.TemporaryDirectory(prefix="mdx23c_full_config_") as config_dir_str:
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
            "--extract_instrumental",
        ]
        completed = subprocess.run(
            command,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=False,
        )
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
        vocals_wav = model_stem_dir / "vocals.wav"
        instrumental_wav = model_stem_dir / "instrumental.wav"
        if not vocals_wav.exists():
            raise FileNotFoundError(f"MDX23C vocals output is missing: {vocals_wav}")
        if not instrumental_wav.exists():
            raise FileNotFoundError(f"MDX23C instrumental output is missing: {instrumental_wav}")

        vocals_audio, vocals_sr = sf.read(str(vocals_wav), always_2d=True)
        instrumental_audio, instrumental_sr = sf.read(str(instrumental_wav), always_2d=True)
        raw_audio, raw_sr = sf.read(str(full_input_path), always_2d=True)

        vocals_flac_path = output_dir / "vocals.flac"
        instrumental_flac_path = output_dir / "instrumental.flac"
        raw_ref_flac_path = output_dir / "raw_ref.flac"
        sf.write(str(vocals_flac_path), vocals_audio, vocals_sr, subtype="PCM_24")
        sf.write(str(instrumental_flac_path), instrumental_audio, instrumental_sr, subtype="PCM_24")
        sf.write(str(raw_ref_flac_path), raw_audio, raw_sr, subtype="PCM_24")

        manifest_path = write_full_track_stem_manifest(
            output_dir=output_dir,
            source_audio_path=source_audio_path,
            vocals_flac_path=vocals_flac_path,
            instrumental_flac_path=instrumental_flac_path,
            raw_ref_flac_path=raw_ref_flac_path,
            repo_path=repo_path,
            model_path=model_path,
            config_template_path=config_path,
            runtime_config_path=runtime_config_path,
            device_id=int(args.device_id),
            num_overlap=int(args.num_overlap),
            sample_rate=int(raw_sr),
            frames=int(raw_audio.shape[0]),
            elapsed_sec=time.time() - start_wall,
            cuda_preflight=cuda_preflight,
        )

        if listen_dir is not None:
            listen_dir.mkdir(parents=True, exist_ok=True)
            copy_listen_file(raw_ref_flac_path, listen_dir / "00_raw_ref.flac")
            copy_listen_file(vocals_flac_path, listen_dir / args.listen_label)
            (listen_dir / "00_README.txt").write_text(
                "Listen only here.\n00 = full raw source reference.\nOther files = full-track MDX23C vocal candidates.\n",
                encoding="utf-8",
            )
            (listen_dir / "candidate_map.json").write_text(
                json.dumps(
                    {
                        "00_raw_ref.flac": str(raw_ref_flac_path),
                        args.listen_label: str(vocals_flac_path),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        elapsed_sec = round(time.time() - start_wall, 3)
        _emit(f"EXIT output_dir={output_dir} elapsed_sec={elapsed_sec}")
        print(json.dumps({"output_dir": str(output_dir), "manifest_path": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
