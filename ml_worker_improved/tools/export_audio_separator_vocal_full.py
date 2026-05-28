from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from export_dereverb_vocal_preview import (  # noqa: E402
    copy_listen_file,
    save_audio_window_as_wav,
)


DEFAULT_MODEL_FILENAME = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
DEFAULT_MODEL_FILE_DIR = Path("/app/models/audio_separator")
DEFAULT_AUDIO_SEPARATOR_PYTHON = Path("/opt/audio-separator-venv/bin/python")
DEFAULT_MODEL_ID = "audio_separator_bs_roformer_vocals"
DEFAULT_MODEL_DISPLAY_NAME = "Audio Separator BS-RoFormer vocal checkpoint"
ROUTE_NAME = "audio_separator_uvr_full_track_vocals"


def _emit(message: str) -> None:
    print(f"[audio_separator_vocal_full] {message}", flush=True)


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


def _resolve_named_output(output_files: Iterable[str | Path], output_dir: Path, stem_name: str) -> Path:
    normalized_stem = stem_name.lower()
    equivalent_stems = {
        "instrumental": ("instrumental", "other", "no_vocals", "accompaniment"),
    }.get(normalized_stem, (normalized_stem,))
    candidates = [Path(path) for path in output_files]
    candidates.extend(output_dir.glob("*"))
    exact_names = set()
    for candidate_stem in equivalent_stems:
        exact_names.update(
            {
                f"{candidate_stem}.flac",
                f"{candidate_stem}.wav",
            }
        )
    for candidate in candidates:
        if candidate.name in exact_names and candidate.exists():
            return candidate
    for candidate in candidates:
        candidate_stem = candidate.stem.lower()
        if any(alias in candidate_stem for alias in equivalent_stems) and candidate.exists():
            return candidate
    raise FileNotFoundError(f"Audio Separator {stem_name} output is missing in {output_dir}")


def _copy_audio_as_flac(source_path: Path, target_path: Path) -> None:
    audio, sample_rate = sf.read(str(source_path), always_2d=True)
    sf.write(str(target_path), audio, sample_rate, subtype="PCM_24")


def model_sidecar_yaml_path(model_file_dir: Path, model_filename: str) -> Path:
    return model_file_dir / f"{Path(model_filename).stem}.yaml"


def build_runtime_contract_status(
    *,
    audio_separator_python: Path,
    audio_separator_python_exists: bool,
    model_file: Path,
    model_file_exists: bool,
    model_sidecar: Path,
    model_sidecar_exists: bool,
    onnxruntime_version: str | None,
    available_providers: Iterable[str],
    require_cuda_provider: bool,
    probe_error: str | None,
) -> dict[str, Any]:
    providers = list(available_providers)
    blockers: list[str] = []
    if not audio_separator_python_exists:
        blockers.append("audio_separator_python_missing")
    if not model_file_exists:
        blockers.append("audio_separator_model_file_missing")
    if not model_sidecar_exists:
        blockers.append("audio_separator_model_sidecar_missing")
    if probe_error:
        blockers.append("audio_separator_runtime_probe_failed")
    cuda_provider_available = "CUDAExecutionProvider" in providers
    if require_cuda_provider and not cuda_provider_available:
        blockers.append("cuda_execution_provider_missing")

    return {
        "status": "ready" if not blockers else "blocked",
        "audio_separator_python": str(audio_separator_python),
        "audio_separator_python_exists": bool(audio_separator_python_exists),
        "model_file": str(model_file),
        "model_file_exists": bool(model_file_exists),
        "model_sidecar": str(model_sidecar),
        "model_sidecar_exists": bool(model_sidecar_exists),
        "onnxruntime_version": onnxruntime_version,
        "available_providers": providers,
        "cuda_provider_available": cuda_provider_available,
        "require_cuda_provider": bool(require_cuda_provider),
        "probe_error": probe_error,
        "blockers": blockers,
    }


def probe_audio_separator_runtime(
    *,
    audio_separator_python: Path,
    model_file_dir: Path,
    model_filename: str,
    require_cuda_provider: bool,
) -> dict[str, Any]:
    model_file = model_file_dir / model_filename
    model_sidecar = model_sidecar_yaml_path(model_file_dir, model_filename)
    audio_separator_python_exists = audio_separator_python.exists()
    model_file_exists = model_file.exists()
    model_sidecar_exists = model_sidecar.exists()
    onnxruntime_version: str | None = None
    available_providers: list[str] = []
    probe_error: str | None = None

    if audio_separator_python_exists:
        runner_code = r"""
from __future__ import annotations
import json
try:
    import onnxruntime as ort
    from audio_separator.separator import Separator
    print(json.dumps({
        "onnxruntime_version": ort.__version__,
        "available_providers": list(ort.get_available_providers()),
        "audio_separator_import": Separator.__name__,
    }, ensure_ascii=False))
except Exception as exc:
    print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
    raise
"""
        completed = subprocess.run(
            [str(audio_separator_python), "-c", runner_code],
            capture_output=True,
            text=True,
            check=False,
        )
        last_json_line = next(
            (line for line in reversed(completed.stdout.splitlines()) if line.strip().startswith("{")),
            "",
        )
        if last_json_line:
            try:
                payload = json.loads(last_json_line)
                onnxruntime_version = payload.get("onnxruntime_version")
                available_providers = list(payload.get("available_providers") or [])
                probe_error = payload.get("error")
            except json.JSONDecodeError as exc:
                probe_error = f"JSONDecodeError: {exc}"
        if completed.returncode != 0 and probe_error is None:
            probe_error = completed.stderr.strip() or f"probe_returncode={completed.returncode}"

    return build_runtime_contract_status(
        audio_separator_python=audio_separator_python,
        audio_separator_python_exists=audio_separator_python_exists,
        model_file=model_file,
        model_file_exists=model_file_exists,
        model_sidecar=model_sidecar,
        model_sidecar_exists=model_sidecar_exists,
        onnxruntime_version=onnxruntime_version,
        available_providers=available_providers,
        require_cuda_provider=require_cuda_provider,
        probe_error=probe_error,
    )


def build_full_track_stem_manifest(
    *,
    source_audio_path: Path,
    vocals_flac_path: Path,
    instrumental_flac_path: Path,
    raw_ref_flac_path: Path,
    model_filename: str,
    model_file_dir: Path,
    model_id: str,
    model_display_name: str,
    output_format: str,
    normalization_threshold: float,
    mdx_overlap: float,
    mdx_batch_size: int,
    enable_denoise: bool,
    sample_rate: int,
    frames: int,
    elapsed_sec: float,
    runtime_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_audio_path": str(source_audio_path),
        "model_id": model_id,
        "model_name": model_id,
        "model": {
            "model_id": model_id,
            "name": model_display_name,
            "family": "audio_separator_uvr",
            "model_type": "bs_roformer",
            "model_filename": model_filename,
            "model_file_dir": str(model_file_dir),
            "target_output": "vocals",
            "output_format": output_format,
            "normalization_threshold": float(normalization_threshold),
            "mdx_overlap": float(mdx_overlap),
            "mdx_batch_size": int(mdx_batch_size),
            "enable_denoise": bool(enable_denoise),
        },
        "runtime": {
            "audio_separator": runtime_contract or {},
        },
        "contract": {
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
            "route_name": ROUTE_NAME,
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
    model_filename: str,
    model_file_dir: Path,
    model_id: str,
    model_display_name: str,
    output_format: str,
    normalization_threshold: float,
    mdx_overlap: float,
    mdx_batch_size: int,
    enable_denoise: bool,
    sample_rate: int,
    frames: int,
    elapsed_sec: float,
    runtime_contract: dict[str, Any] | None = None,
) -> Path:
    manifest = build_full_track_stem_manifest(
        source_audio_path=source_audio_path,
        vocals_flac_path=vocals_flac_path,
        instrumental_flac_path=instrumental_flac_path,
        raw_ref_flac_path=raw_ref_flac_path,
        model_filename=model_filename,
        model_file_dir=model_file_dir,
        model_id=model_id,
        model_display_name=model_display_name,
        output_format=output_format,
        normalization_threshold=normalization_threshold,
        mdx_overlap=mdx_overlap,
        mdx_batch_size=mdx_batch_size,
        enable_denoise=enable_denoise,
        sample_rate=sample_rate,
        frames=frames,
        elapsed_sec=elapsed_sec,
        runtime_contract=runtime_contract,
    )
    manifest_path = output_dir / "stem_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def run_audio_separator(
    *,
    source_audio_path: Path,
    model_output_dir: Path,
    audio_separator_python: Path,
    model_filename: str,
    model_file_dir: Path,
    output_format: str,
    normalization_threshold: float,
    mdx_overlap: float,
    mdx_batch_size: int,
    enable_denoise: bool,
) -> list[str]:
    if not audio_separator_python.exists():
        raise FileNotFoundError(f"Audio Separator Python runtime is missing: {audio_separator_python}")

    payload = {
        "source_audio_path": str(source_audio_path),
        "model_output_dir": str(model_output_dir),
        "model_filename": model_filename,
        "model_file_dir": str(model_file_dir),
        "output_format": output_format,
        "normalization_threshold": float(normalization_threshold),
        "mdx_overlap": float(mdx_overlap),
        "mdx_batch_size": int(mdx_batch_size),
        "enable_denoise": bool(enable_denoise),
    }
    runner_code = r"""
from __future__ import annotations
import json
import sys
from audio_separator.separator import Separator

payload = json.loads(open(sys.argv[1], "r", encoding="utf-8").read())
separator = Separator(
    model_file_dir=payload["model_file_dir"],
    output_dir=payload["model_output_dir"],
    output_format=payload["output_format"],
    normalization_threshold=float(payload["normalization_threshold"]),
    amplification_threshold=0.0,
    use_soundfile=False,
    mdx_params={
        "overlap": float(payload["mdx_overlap"]),
        "batch_size": int(payload["mdx_batch_size"]),
        "enable_denoise": bool(payload["enable_denoise"]),
    },
)
separator.load_model(model_filename=payload["model_filename"])
output_files = separator.separate(
    payload["source_audio_path"],
    {
        "Vocals": "vocals",
        "Instrumental": "instrumental",
    },
)
print(json.dumps({"output_files": list(output_files)}, ensure_ascii=False))
"""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as payload_file:
        payload_file.write(json.dumps(payload, ensure_ascii=False))
        payload_path = Path(payload_file.name)
    try:
        completed = subprocess.run(
            [str(audio_separator_python), "-c", runner_code, str(payload_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        payload_path.unlink(missing_ok=True)
    if completed.stdout.strip():
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
    if completed.stderr.strip():
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", flush=True)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            [str(audio_separator_python), "-c", "<audio_separator_runner>", str(payload_path)],
            output=completed.stdout,
            stderr=completed.stderr,
        )
    last_json_line = next(
        (line for line in reversed(completed.stdout.splitlines()) if line.strip().startswith("{")),
        "",
    )
    if not last_json_line:
        raise RuntimeError("Audio Separator runner did not emit output JSON")
    return list(json.loads(last_json_line)["output_files"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export full-track Audio Separator UVR vocals/instrumental stems.")
    parser.add_argument("source_audio_path")
    parser.add_argument("output_dir")
    parser.add_argument("--model-filename", default=DEFAULT_MODEL_FILENAME)
    parser.add_argument("--model-file-dir", default=str(DEFAULT_MODEL_FILE_DIR))
    parser.add_argument("--audio-separator-python", default=str(DEFAULT_AUDIO_SEPARATOR_PYTHON))
    parser.add_argument("--manifest-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--manifest-model-display-name", default=DEFAULT_MODEL_DISPLAY_NAME)
    parser.add_argument("--output-format", default="FLAC")
    parser.add_argument("--normalization-threshold", type=float, default=0.9)
    parser.add_argument("--mdx-overlap", type=float, default=0.25)
    parser.add_argument("--mdx-batch-size", type=int, default=1)
    parser.add_argument("--enable-denoise", action="store_true")
    parser.add_argument("--require-cuda-provider", action="store_true")
    parser.add_argument("--listen-dir")
    parser.add_argument("--listen-label", default="01_audio_separator_bs_roformer_vocals.flac")
    return parser


def main() -> None:
    require_docker_runtime(
        tool_name="export_audio_separator_vocal_full.py",
        entrypoint_path="/app/tools/export_audio_separator_vocal_full.py",
    )
    args = build_arg_parser().parse_args()
    source_audio_path = Path(args.source_audio_path)
    output_dir = Path(args.output_dir)
    model_file_dir = Path(args.model_file_dir)
    audio_separator_python = Path(args.audio_separator_python)
    listen_dir = Path(args.listen_dir) if args.listen_dir else None

    output_dir.mkdir(parents=True, exist_ok=True)
    if not source_audio_path.exists():
        raise FileNotFoundError(f"Source audio is missing: {source_audio_path}")
    runtime_contract = probe_audio_separator_runtime(
        audio_separator_python=audio_separator_python,
        model_file_dir=model_file_dir,
        model_filename=str(args.model_filename),
        require_cuda_provider=bool(args.require_cuda_provider),
    )
    if runtime_contract["status"] != "ready":
        print(json.dumps({"runtime_contract": runtime_contract}, indent=2, ensure_ascii=False), flush=True)
        raise RuntimeError(
            "Audio Separator runtime contract blocked: "
            + ", ".join(runtime_contract["blockers"])
        )

    start_wall = time.time()
    with tempfile.TemporaryDirectory(prefix="audio_separator_full_output_") as model_output_dir_str, tempfile.TemporaryDirectory(
        prefix="audio_separator_full_input_"
    ) as input_dir_str:
        model_output_dir = Path(model_output_dir_str)
        input_dir = Path(input_dir_str)
        raw_ref_wav_path = input_dir / "full_track.wav"

        _emit(
            "ENTER "
            f"source_audio={source_audio_path} output_dir={output_dir} "
            f"model_filename={args.model_filename} full_track=true"
        )
        sample_rate, frames = save_audio_window_as_wav(
            source_audio_path,
            raw_ref_wav_path,
            start_sec=0.0,
            end_sec=None,
        )

        output_files = run_audio_separator(
            source_audio_path=source_audio_path,
            model_output_dir=model_output_dir,
            audio_separator_python=audio_separator_python,
            model_filename=str(args.model_filename),
            model_file_dir=model_file_dir,
            output_format=str(args.output_format),
            normalization_threshold=float(args.normalization_threshold),
            mdx_overlap=float(args.mdx_overlap),
            mdx_batch_size=int(args.mdx_batch_size),
            enable_denoise=bool(args.enable_denoise),
        )
        _emit(f"audio_separator_outputs={output_files}")

        model_vocals_path = _resolve_named_output(output_files, model_output_dir, "vocals")
        model_instrumental_path = _resolve_named_output(output_files, model_output_dir, "instrumental")

        vocals_flac_path = output_dir / "vocals.flac"
        instrumental_flac_path = output_dir / "instrumental.flac"
        raw_ref_flac_path = output_dir / "raw_ref.flac"
        _copy_audio_as_flac(model_vocals_path, vocals_flac_path)
        _copy_audio_as_flac(model_instrumental_path, instrumental_flac_path)
        _copy_audio_as_flac(raw_ref_wav_path, raw_ref_flac_path)

        manifest_path = write_full_track_stem_manifest(
            output_dir=output_dir,
            source_audio_path=source_audio_path,
            vocals_flac_path=vocals_flac_path,
            instrumental_flac_path=instrumental_flac_path,
            raw_ref_flac_path=raw_ref_flac_path,
            model_filename=str(args.model_filename),
            model_file_dir=model_file_dir,
            model_id=str(args.manifest_model_id),
            model_display_name=str(args.manifest_model_display_name),
            output_format=str(args.output_format),
            normalization_threshold=float(args.normalization_threshold),
            mdx_overlap=float(args.mdx_overlap),
            mdx_batch_size=int(args.mdx_batch_size),
            enable_denoise=bool(args.enable_denoise),
            sample_rate=int(sample_rate),
            frames=int(frames),
            elapsed_sec=time.time() - start_wall,
            runtime_contract=runtime_contract,
        )

        if listen_dir is not None:
            listen_dir.mkdir(parents=True, exist_ok=True)
            copy_listen_file(raw_ref_flac_path, listen_dir / "00_raw_ref.flac")
            copy_listen_file(vocals_flac_path, listen_dir / args.listen_label)
            (listen_dir / "00_README.txt").write_text(
                "Diagnostic only. Run benchmark/effectiveness gate before delivery.\n"
                "00 = full raw source reference.\n"
                "Other files = full-track Audio Separator UVR vocal candidates.\n",
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
