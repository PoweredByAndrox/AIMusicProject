from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

import yaml

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import cli_runtime_contract, require_docker_runtime  # noqa: E402
from audio_sample_rate_contract import (  # noqa: E402
    load_configured_model_sample_rate,
    resample_audio_to_layout,
)
from mac_mps_worker_client import send_request  # noqa: E402

import soundfile as sf


DEFAULT_MODEL_TYPE = "bs_roformer"
DEFAULT_TARGET_INSTRUMENT = "other"
DEFAULT_CONFIG_PATH = Path(
    "/app/vendor/bs_roformer_instrumental_resurrection_manual/config_bs_roformer_instrumental_resurrection_unwa.yaml"
)
DEFAULT_MODEL_PATH = Path(
    "/app/vendor/bs_roformer_instrumental_resurrection_manual/bs_roformer_instrumental_resurrection_unwa.ckpt"
)
DEFAULT_RUNTIME_REPO = Path("/app/vendor/music_source_separation_training")
DEFAULT_MODEL_NAME = "bs_roformer_instrumental_resurrection_unwa"
BS_ROFORMER_RUNTIME_RELATIVE_PATH = Path("models/bs_roformer/bs_roformer.py")
BS_ROFORMER_COMPLEX_ZERO_DC_ORIGINAL = """        if self.zero_dc:
            # whether to dc filter
            stft_repr = stft_repr.index_fill(1, tensor(0, device = device), 0.)
"""
BS_ROFORMER_COMPLEX_ZERO_DC_MPS_SAFE = """        if self.zero_dc:
            # whether to dc filter
            dc_index = tensor(0, device = device)
            if stft_repr.is_complex() and stft_repr.device.type == 'mps':
                stft_repr = torch.complex(
                    stft_repr.real.index_fill(1, dc_index, 0.),
                    stft_repr.imag.index_fill(1, dc_index, 0.),
                )
            else:
                stft_repr = stft_repr.index_fill(1, dc_index, 0.)
"""
BS_ROFORMER_ISTFT_CPU_FALLBACK_ORIGINAL = """        try:
            recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=stft_window, return_complex=False, length=raw_audio.shape[-1])
        except:
            recon_audio = torch.istft(
                stft_repr.cpu() if x_is_mps else stft_repr,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=False,
                length=raw_audio.shape[-1]
            ).to(device)
"""
BS_ROFORMER_ISTFT_NO_FALLBACK = """        recon_audio = torch.istft(
            stft_repr,
            **self.stft_kwargs,
            window=stft_window,
            return_complex=False,
            length=raw_audio.shape[-1],
        )
"""


def _emit(message: str) -> None:
    print(f"[instrumental_resurrection_minus] {message}", flush=True)


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model_source_metadata(path: Path | None) -> dict:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Model source metadata must be an object: {path}")
    return payload


def _verify_model_source_files(metadata: dict, paths: list[Path]) -> dict[str, str]:
    records = {
        str(item.get("file") or ""): item
        for item in metadata.get("files") or []
        if isinstance(item, dict)
    }
    verified: dict[str, str] = {}
    for path in paths:
        actual = _sha256(path)
        verified[path.name] = actual
        if not metadata:
            continue
        record = records.get(path.name)
        if record is None:
            raise ValueError(f"Model source metadata is missing file: {path.name}")
        expected = str(record.get("sha256") or record.get("expected_sha256") or "").lower()
        if not expected:
            raise ValueError(f"Model source metadata is missing sha256: {path.name}")
        if actual.lower() != expected:
            raise ValueError(f"Model source sha256 mismatch: {path.name}")
    return verified


def _patch_standard_bs_roformer_mps_runtime(
    runtime_repo: Path,
    *,
    device: str,
    model_type: str,
    custom_implementation: bool,
) -> list[dict]:
    if str(device).strip().lower() not in {"mps", "metal"} or custom_implementation:
        return []
    if str(model_type).strip().lower() != "bs_roformer":
        return []
    path = runtime_repo / BS_ROFORMER_RUNTIME_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"BS-RoFormer MPS runtime implementation is missing: {path}")
    source = path.read_text(encoding="utf-8")
    pre_patch_sha256 = _sha256(path)
    changed = False
    if BS_ROFORMER_COMPLEX_ZERO_DC_MPS_SAFE not in source:
        if BS_ROFORMER_COMPLEX_ZERO_DC_ORIGINAL not in source:
            raise RuntimeError("BS-RoFormer runtime has divergent complex zero-DC block")
        source = source.replace(
            BS_ROFORMER_COMPLEX_ZERO_DC_ORIGINAL,
            BS_ROFORMER_COMPLEX_ZERO_DC_MPS_SAFE,
            1,
        )
        changed = True
    if BS_ROFORMER_ISTFT_NO_FALLBACK not in source:
        if BS_ROFORMER_ISTFT_CPU_FALLBACK_ORIGINAL not in source:
            raise RuntimeError("BS-RoFormer runtime has divergent ISTFT fallback block")
        source = source.replace(
            BS_ROFORMER_ISTFT_CPU_FALLBACK_ORIGINAL,
            BS_ROFORMER_ISTFT_NO_FALLBACK,
            1,
        )
        changed = True
    if changed:
        path.write_text(source, encoding="utf-8")
    post_patch_sha256 = _sha256(path)
    return [
        {
            "patch_id": "bs_roformer_mps_complex_zero_dc_no_cpu_istft",
            "file": str(path),
            "status": "applied" if changed else "already_present",
            "pre_patch_sha256": pre_patch_sha256,
            "post_patch_sha256": post_patch_sha256,
            "sha256": post_patch_sha256,
            "cpu_fallback_allowed": False,
            "patch_origin": "tracked_exporter_runtime_patch" if changed else "validated_exact_runtime_patch",
        }
    ]


def _materialize_runtime_config(
    source_path: Path,
    output_dir: Path,
    *,
    chunk_size: int | None,
    batch_size: int | None,
    num_overlap: int | None,
) -> tuple[Path, dict[str, int]]:
    overrides = {
        key: int(value)
        for key, value in {
            "chunk_size": chunk_size,
            "batch_size": batch_size,
            "num_overlap": num_overlap,
        }.items()
        if value is not None
    }
    if not overrides:
        return source_path, {}
    if any(value <= 0 for value in overrides.values()):
        raise ValueError("Inference config overrides must be positive integers")
    payload = yaml.load(source_path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    if not isinstance(payload, dict):
        raise ValueError(f"Model config must be a mapping: {source_path}")
    audio = payload.setdefault("audio", {})
    inference = payload.setdefault("inference", {})
    if "chunk_size" in overrides:
        audio["chunk_size"] = overrides["chunk_size"]
    if "batch_size" in overrides:
        inference["batch_size"] = overrides["batch_size"]
    if "num_overlap" in overrides:
        inference["num_overlap"] = overrides["num_overlap"]
    runtime_path = output_dir / "cfg.yaml"
    runtime_path.write_text(yaml.dump(payload, Dumper=yaml.Dumper, sort_keys=False), encoding="utf-8")
    return runtime_path, overrides


def _build_precision_contract(
    runtime_config_path: Path,
    *,
    device: str,
    model_implementation_path: Path | None,
) -> dict:
    payload = yaml.load(runtime_config_path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    if not isinstance(payload, dict):
        raise ValueError(f"Model config must be a mapping: {runtime_config_path}")
    training = payload.get("training") or {}
    if not isinstance(training, dict):
        raise ValueError(f"Model training config must be a mapping: {runtime_config_path}")
    use_amp = bool(training.get("use_amp", True))
    is_mps = str(device).strip().lower() in {"mps", "metal"}
    compatibility_requested = bool(is_mps and model_implementation_path is not None)
    if compatibility_requested and use_amp:
        requested_mode = "mps_autocast_float16"
    elif is_mps:
        requested_mode = "mps_float32"
    elif use_amp:
        requested_mode = "runtime_autocast"
    else:
        requested_mode = "float32"
    return {
        "mode": "pending_adapter_initialization" if compatibility_requested else requested_mode,
        "requested_mode": requested_mode,
        "config_use_amp": use_amp,
        "requested_autocast_device_type": "mps" if compatibility_requested else None,
        "requested_autocast_dtype": "float16" if compatibility_requested and use_amp else None,
        "autocast_device_type": None,
        "autocast_dtype": None,
        "mps_cuda_autocast_compatibility_requested": compatibility_requested,
        "mps_cuda_autocast_compatibility_installed": False,
        "cpu_fallback_allowed": False if is_mps else None,
        "requires_quality_equivalence_gate": bool(compatibility_requested and use_amp),
    }


def _mark_precision_contract_installed(contract: dict) -> None:
    if not contract.get("mps_cuda_autocast_compatibility_requested"):
        return
    contract["mode"] = contract["requested_mode"]
    contract["autocast_device_type"] = contract["requested_autocast_device_type"]
    contract["autocast_dtype"] = contract["requested_autocast_dtype"]
    contract["mps_cuda_autocast_compatibility_installed"] = True


def _observed_runtime_contract(device: str) -> dict[str, object]:
    policy_contract = dict(cli_runtime_contract(allow_native_runtime=False))
    normalized_device = str(device).strip().lower()
    runtime_environment = str(policy_contract["runtime_environment"])
    if runtime_environment == "native_mac_cpu_dsp" and normalized_device in {"mps", "metal"}:
        runtime_environment = "native_mac_mps_diagnostic"
    is_native = runtime_environment.startswith("native_mac_")
    return {
        "runtime_environment": runtime_environment,
        "execution_surface": "native_macos" if is_native else "container",
        "container_runtime": None if is_native else policy_contract["container_runtime"],
        "policy_container_runtime": policy_contract["container_runtime"],
        "native_runtime_allowed": policy_contract["native_runtime_allowed"],
        "accelerator": "mps" if normalized_device in {"mps", "metal"} else normalized_device,
    }


def _mark_progress(
    event_log_path: Path,
    *,
    run_started_at: float,
    stage: str,
    status: str = "progress",
    stage_started_at: float | None = None,
    **metadata,
) -> float:
    now = time.perf_counter()
    event = {
        "type": "instrumental_resurrection_stage",
        "stage": str(stage),
        "status": str(status),
        "elapsed_sec": round(max(0.0, now - stage_started_at), 6) if stage_started_at is not None else 0.0,
        "since_start_sec": round(max(0.0, now - run_started_at), 6),
        "metadata": {
            str(key): _json_safe(value)
            for key, value in metadata.items()
            if value is not None
        },
    }
    with event_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True)
    return now


def _resolve_bs_roformer_device(device: str) -> tuple[bool, list[str]]:
    requested = str(device).strip().lower()
    if requested in {"", "auto", "cuda"}:
        return False, ["0"]
    if requested in {"cpu", "cpu:0"}:
        return True, []
    if requested in {"mps", "metal"}:
        return False, []
    normalized = requested.replace("cuda:", "").replace("cuda", "")
    normalized = normalized.replace(" ", ",")
    device_ids: list[str] = [
        token.strip()
        for token in normalized.split(",")
        if token.strip()
    ]
    if not device_ids:
        return False, ["0"]
    if any(not token.isdigit() for token in device_ids):
        raise ValueError(f"Unsupported BS Roformer device value: {device}")
    return False, device_ids


def _expected_acceleration_mode(device: str, *, force_cpu: bool) -> str:
    requested = str(device).strip().lower()
    if force_cpu:
        return "cpu_diagnostic_inference"
    if requested in {"mps", "metal"}:
        return "mps_inference"
    return "gpu_inference"


def _resolve_output_by_stem(store_dir: Path, stem_name: str) -> Path:
    matches = sorted(
        path
        for path in store_dir.rglob(f"{stem_name}.*")
        if path.is_file()
    )
    if not matches:
        raise FileNotFoundError(f"BS Roformer {stem_name} output is missing in: {store_dir}")
    return matches[0]


def _run_command(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_tail: deque[str] = deque(maxlen=200)
    assert process.stdout is not None
    for line in process.stdout:
        output_tail.append(line)
        print(line, end="" if line.endswith("\n") else "\n", flush=True)
    returncode = process.wait()
    output = "".join(output_tail)
    completed = subprocess.CompletedProcess(command, returncode, stdout=output, stderr="")
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command, output=output, stderr="")
    return completed


def _run_custom_architecture_in_process(
    *,
    runtime_repo: Path,
    model_implementation_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    input_dir: Path,
    output_dir: Path,
    use_tta: bool,
) -> None:
    from run_bs_roformer_custom_impl import run_custom_architecture

    custom_args = argparse.Namespace(
        runtime_repo=runtime_repo,
        model_implementation_path=model_implementation_path,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        input_dir=input_dir,
        output_dir=output_dir,
        filename_template="{instr}",
        disable_detailed_pbar=False,
        use_tta=use_tta,
    )
    try:
        run_custom_architecture(custom_args)
    except Exception as exc:
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["in_process_custom_bs_roformer_adapter"],
            output=f"{type(exc).__name__}: {exc}",
        ) from exc


def _run_resident_custom_architecture(
    *,
    socket_path: Path,
    source_audio_path: Path,
    output_dir: Path,
    model_id: str,
    runtime_config_path: Path,
    checkpoint_path: Path,
    implementation_path: Path,
    use_tta: bool,
    timeout_sec: float,
) -> dict:
    if timeout_sec <= 0.0:
        raise ValueError("resident worker timeout must be positive")
    if not socket_path.is_socket():
        raise subprocess.CalledProcessError(
            returncode=69,
            cmd=["resident_large_inst_worker"],
            output=f"resident worker socket is unavailable: {socket_path}",
        )
    request_id = "li" + hashlib.sha256(str(output_dir).encode("utf-8")).hexdigest()[:10]
    expected_fingerprint = {
        "model_id": str(model_id),
        "config_sha256": _sha256(runtime_config_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "implementation_sha256": _sha256(implementation_path),
    }
    try:
        response = send_request(
            socket_path,
            {
                "schema": "mac_mps_large_inst_request.v1",
                "action": "separate",
                "request_id": request_id,
                "source_audio_path": str(source_audio_path),
                "source_scope": "disposable_scratch",
                "output_dir": str(output_dir),
                "output_scope": "disposable_scratch",
                "use_tta": bool(use_tta),
                "expected_model_fingerprint": expected_fingerprint,
            },
            timeout_sec,
        )
    except Exception as exc:
        raise subprocess.CalledProcessError(
            returncode=70,
            cmd=["resident_large_inst_worker"],
            output=f"{type(exc).__name__}: {exc}",
        ) from exc
    if response.get("status") != "ok":
        raise subprocess.CalledProcessError(
            returncode=71,
            cmd=["resident_large_inst_worker"],
            output=json.dumps(response, ensure_ascii=False, sort_keys=True),
        )
    required_contract = {
        "request_id": request_id,
        "model_id": str(model_id),
        "device": "mps",
        "cpu_fallback_allowed": False,
        "delivery_allowed": False,
        "model_load_count": 1,
    }
    for key, expected in required_contract.items():
        if response.get(key) != expected:
            raise subprocess.CalledProcessError(
                returncode=72,
                cmd=["resident_large_inst_worker"],
                output=f"resident worker contract mismatch: {key}={response.get(key)!r}",
            )
    if response.get("model_fingerprint") != expected_fingerprint:
        raise subprocess.CalledProcessError(
            returncode=72,
            cmd=["resident_large_inst_worker"],
            output="resident worker model fingerprint mismatch",
        )
    output_contract = response.get("output_contract")
    if (
        not isinstance(output_contract, dict)
        or output_contract.get("output_scope") != "disposable_scratch"
        or output_contract.get("source_scope") != "disposable_scratch"
    ):
        raise subprocess.CalledProcessError(
            returncode=73,
            cmd=["resident_large_inst_worker"],
            output="resident worker did not declare disposable scratch output",
        )
    worker_output_path = Path(str(response.get("output_path") or "")).resolve()
    try:
        worker_output_path.relative_to(output_dir.resolve())
    except ValueError as exc:
        raise subprocess.CalledProcessError(
            returncode=74,
            cmd=["resident_large_inst_worker"],
            output=f"resident worker output escaped scratch root: {worker_output_path}",
        ) from exc
    if not worker_output_path.is_file():
        raise subprocess.CalledProcessError(
            returncode=75,
            cmd=["resident_large_inst_worker"],
            output=f"resident worker output is missing: {worker_output_path}",
        )
    actual_output_sha256 = _sha256(worker_output_path)
    if response.get("output_sha256") != actual_output_sha256:
        raise subprocess.CalledProcessError(
            returncode=76,
            cmd=["resident_large_inst_worker"],
            output=(
                "resident worker output SHA-256 mismatch: "
                f"reported={response.get('output_sha256')!r} actual={actual_output_sha256}"
            ),
        )
    return response


def _transcode_to_wav(
    source_audio_path: Path,
    target_wav_path: Path,
    *,
    model_sample_rate: int,
) -> tuple[int, int, int]:
    audio, sample_rate = sf.read(str(source_audio_path), always_2d=True, dtype="float32")
    source_frames = int(audio.shape[0])
    model_frames = max(1, int(round(source_frames * int(model_sample_rate) / int(sample_rate))))
    model_audio = resample_audio_to_layout(
        audio,
        source_sample_rate=int(sample_rate),
        target_sample_rate=int(model_sample_rate),
        target_frames=model_frames,
    )
    sf.write(str(target_wav_path), model_audio, int(model_sample_rate), format="WAV", subtype="FLOAT")
    return int(sample_rate), source_frames, model_frames


def _write_flac_copy(source_audio_path: Path, target_flac_path: Path) -> tuple[int, int]:
    audio, sample_rate = sf.read(str(source_audio_path), always_2d=True, dtype="float32")
    sf.write(str(target_flac_path), audio, sample_rate, format="FLAC", subtype="PCM_24")
    return int(sample_rate), int(audio.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a full-track Instrumental Resurrection minus from the accepted manual model recovery."
    )
    parser.add_argument("source_audio_path", help="Source full-track audio to process.")
    parser.add_argument("output_dir", help="Target directory for exported artifacts.")
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--runtime-repo", default=str(DEFAULT_RUNTIME_REPO))
    parser.add_argument("--model-type", default=DEFAULT_MODEL_TYPE)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-implementation-path")
    parser.add_argument("--model-source-metadata-path")
    parser.add_argument("--resident-worker-socket")
    parser.add_argument("--resident-worker-timeout-sec", type=float, default=600.0)
    parser.add_argument("--registry-recovery", default="manual_huggingface_mirror")
    parser.add_argument("--target-instrument", default=DEFAULT_TARGET_INSTRUMENT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-chunk-size", type=int)
    parser.add_argument("--inference-batch-size", type=int)
    parser.add_argument("--inference-num-overlap", type=int)
    parser.add_argument("--use-tta", action="store_true")
    parser.add_argument("--listen-dir")
    parser.add_argument("--listen-label", default="01_instrumental_resurrection_minus.flac")
    parser.add_argument("--minus-output-name", default="instrumental_resurrection_minus.flac")
    parser.add_argument("--progress-name", default="instrumental_resurrection_progress.jsonl")
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="export_instrumental_resurrection_minus.py",
        entrypoint_path="/app/tools/export_instrumental_resurrection_minus.py",
    )
    args = parse_args()
    source_audio_path = Path(args.source_audio_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    listen_dir = Path(args.listen_dir).resolve() if args.listen_dir else None
    runtime_repo = Path(args.runtime_repo).resolve()
    config_path = Path(args.config_path).resolve()
    model_path = Path(args.model_path).resolve()
    model_implementation_path = Path(args.model_implementation_path).resolve() if args.model_implementation_path else None
    model_source_metadata_path = Path(args.model_source_metadata_path).resolve() if args.model_source_metadata_path else None
    resident_worker_socket = Path(args.resident_worker_socket).expanduser() if args.resident_worker_socket else None

    if not source_audio_path.exists():
        raise FileNotFoundError(f"Source audio is missing: {source_audio_path}")
    if not runtime_repo.exists():
        raise FileNotFoundError(f"BS Roformer runtime repo is missing: {runtime_repo}")
    if not (runtime_repo / "inference.py").exists():
        raise FileNotFoundError(f"BS Roformer runtime missing inference.py: {runtime_repo / 'inference.py'}")
    for required_path in (config_path, model_path, model_implementation_path, model_source_metadata_path):
        if required_path is not None and not required_path.exists():
            raise FileNotFoundError(required_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_source = _load_model_source_metadata(model_source_metadata_path)
    provenance_paths = [config_path, model_path]
    if model_implementation_path is not None:
        provenance_paths.append(model_implementation_path)
    verified_sha256 = _verify_model_source_files(model_source, provenance_paths)
    runtime_config_path, inference_overrides = _materialize_runtime_config(
        config_path,
        output_dir,
        chunk_size=args.inference_chunk_size,
        batch_size=args.inference_batch_size,
        num_overlap=args.inference_num_overlap,
    )
    model_sample_rate = load_configured_model_sample_rate(runtime_config_path)
    precision_contract = _build_precision_contract(
        runtime_config_path,
        device=args.device,
        model_implementation_path=model_implementation_path,
    )
    runtime_patches = _patch_standard_bs_roformer_mps_runtime(
        runtime_repo,
        device=args.device,
        model_type=args.model_type,
        custom_implementation=model_implementation_path is not None,
    )
    started_at = time.perf_counter()
    progress_log_path = output_dir / str(args.progress_name)
    progress_log_path.write_text("", encoding="utf-8")
    force_cpu, device_ids = _resolve_bs_roformer_device(args.device)
    observed_runtime = _observed_runtime_contract(args.device)
    execution_contract = {
        "route_name": "export_instrumental_resurrection_minus",
        "project_approved_runtime_required": True,
        "docker_runtime_required": observed_runtime["runtime_environment"] == "docker",
        "observed_runtime": observed_runtime,
        "source_audio_path": str(source_audio_path),
        "output_dir": str(output_dir),
        "output_manifest_path": str(output_dir / "stem_manifest.json"),
        "progress_log_path": str(progress_log_path),
        "device": str(args.device),
        "target_instrument": str(args.target_instrument),
        "strict_checkpoint_load": True,
        "cpu_fallback_allowed": False if str(args.device).lower() in {"mps", "metal"} else None,
        "model_implementation_path": str(model_implementation_path) if model_implementation_path else None,
        "runtime_config_path": str(runtime_config_path),
        "inference_overrides": inference_overrides,
        "sample_rate_contract": {
            "model_sample_rate": model_sample_rate,
            "resampler": "soxr_hq",
            "target_output_restored_to_source_rate": True,
        },
        "gpu_inference_skipped": bool(force_cpu),
        "gpu_inference_skip_reason": "cpu_requested" if force_cpu else None,
        "expected_acceleration_mode": _expected_acceleration_mode(args.device, force_cpu=force_cpu),
        "precision_contract": precision_contract,
        "vendor_runtime_patches": runtime_patches,
        "vendor_runtime_patches_applied": bool(runtime_patches),
        "custom_adapter_execution_mode": (
            "resident_single_model_worker"
            if resident_worker_socket is not None
            else "in_process_isolated_exporter"
            if model_implementation_path is not None
            else None
        ),
        "resident_worker_requested": resident_worker_socket is not None,
        "resident_worker_socket": str(resident_worker_socket) if resident_worker_socket is not None else None,
        "resident_worker_fallback_allowed": False,
    }
    _mark_progress(
        progress_log_path,
        run_started_at=started_at,
        stage="execution_contract",
        **execution_contract,
    )

    resident_worker_response = None
    resident_scratch_paths: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="irs_input_") as input_dir_str, tempfile.TemporaryDirectory(
        prefix="irs_output_"
    ) as store_dir_str:
        input_dir = Path(input_dir_str)
        store_dir = Path(store_dir_str)
        resident_scratch_paths = [input_dir, store_dir]
        input_wav_path = input_dir / f"{source_audio_path.stem}.wav"
        stage_started_at = _mark_progress(
            progress_log_path,
            run_started_at=started_at,
            stage="transcode_input",
            status="start",
            source_audio_path=str(source_audio_path),
            input_wav_path=str(input_wav_path),
        )
        sample_rate, frames, model_input_frames = _transcode_to_wav(
            source_audio_path,
            input_wav_path,
            model_sample_rate=model_sample_rate,
        )
        _mark_progress(
            progress_log_path,
            run_started_at=started_at,
            stage="transcode_input",
            status="finish",
            stage_started_at=stage_started_at,
            sample_rate=sample_rate,
            frames=frames,
            model_sample_rate=model_sample_rate,
            model_input_frames=model_input_frames,
            resampled=model_sample_rate != sample_rate,
            input_wav_path=str(input_wav_path),
        )
        _emit(
            "prepared_input_wav "
            f"source_sample_rate={sample_rate} source_frames={frames} "
            f"model_sample_rate={model_sample_rate} model_frames={model_input_frames} "
            f"resampled={model_sample_rate != sample_rate}"
        )

        custom_adapter_kwargs = None
        if model_implementation_path is not None:
            if str(args.device).strip().lower() not in {"mps", "metal"}:
                raise ValueError("Custom BS-RoFormer implementation route currently requires MPS")
            custom_adapter_kwargs = {
                "runtime_repo": runtime_repo,
                "model_implementation_path": model_implementation_path,
                "config_path": runtime_config_path,
                "checkpoint_path": model_path,
                "input_dir": input_dir,
                "output_dir": store_dir,
                "use_tta": bool(args.use_tta),
            }
            command = [
                "resident_large_inst_worker"
                if resident_worker_socket is not None
                else "in_process_custom_bs_roformer_adapter"
            ]
        else:
            command = [
                sys.executable,
                "inference.py",
                "--model_type",
                str(args.model_type),
                "--config_path",
                str(runtime_config_path),
                "--start_check_point",
                str(model_path),
                "--input_folder",
                str(input_dir),
                "--store_dir",
                str(store_dir),
                "--filename_template",
                "{instr}",
                "--flac_file",
                "--pcm_type",
                "PCM_24",
                "--disable_detailed_pbar",
            ]
            if force_cpu:
                command.append("--force_cpu")
            elif device_ids:
                command.extend(["--device_ids", *device_ids])
            if args.use_tta:
                command.append("--use_tta")
        stage_started_at = _mark_progress(
            progress_log_path,
            run_started_at=started_at,
            stage="model_inference",
            status="start",
            runtime_repo=str(runtime_repo),
            model_type=str(args.model_type),
            target_instrument=str(args.target_instrument),
            device=str(args.device),
            expected_acceleration_mode=execution_contract["expected_acceleration_mode"],
            precision_mode=precision_contract["mode"],
            strict_checkpoint_load=True,
            model_implementation_path=model_implementation_path,
        )
        try:
            if resident_worker_socket is not None:
                if custom_adapter_kwargs is None:
                    raise ValueError("Resident Large-Inst worker requires the exact custom architecture route")
                resident_worker_response = _run_resident_custom_architecture(
                    socket_path=resident_worker_socket,
                    source_audio_path=input_wav_path,
                    output_dir=store_dir,
                    model_id=str(args.model_name),
                    runtime_config_path=runtime_config_path,
                    checkpoint_path=model_path,
                    implementation_path=model_implementation_path,
                    use_tta=bool(args.use_tta),
                    timeout_sec=float(args.resident_worker_timeout_sec),
                )
                execution_contract["resident_worker_evidence"] = {
                    "schema": resident_worker_response.get("schema"),
                    "request_id": resident_worker_response.get("request_id"),
                    "device": resident_worker_response.get("device"),
                    "model_load_count": resident_worker_response.get("model_load_count"),
                    "model_load_elapsed_sec": resident_worker_response.get("model_load_elapsed_sec"),
                    "request_count": resident_worker_response.get("request_count"),
                    "worker_elapsed_sec": resident_worker_response.get("elapsed_sec"),
                    "cpu_fallback_allowed": resident_worker_response.get("cpu_fallback_allowed"),
                    "output_sha256": resident_worker_response.get("output_sha256"),
                    "model_fingerprint": resident_worker_response.get("model_fingerprint"),
                    "output_scope": (resident_worker_response.get("output_contract") or {}).get("output_scope"),
                    "scratch_cleanup_verified": False,
                    "requires_quality_equivalence_gate": True,
                }
                _mark_precision_contract_installed(precision_contract)
            elif custom_adapter_kwargs is not None:
                _run_custom_architecture_in_process(**custom_adapter_kwargs)
                _mark_precision_contract_installed(precision_contract)
            else:
                _run_command(command, cwd=runtime_repo)
        except subprocess.CalledProcessError as exc:
            error_text = str(exc.stderr or exc.output or exc)
            _mark_progress(
                progress_log_path,
                run_started_at=started_at,
                stage="model_inference",
                status="failed",
                stage_started_at=stage_started_at,
                returncode=exc.returncode,
                error=error_text,
                expected_acceleration_mode=execution_contract["expected_acceleration_mode"],
            )
            blocked_manifest = {
                "schema_version": 1,
                "mode": "export_instrumental_resurrection_minus",
                "source_audio_path": str(source_audio_path),
                "target_stem": "minus",
                "target_audio_path": None,
                "delivery_allowed": False,
                "delivery_status": "blocked_model_inference_failed",
                "presentation_status": "diagnostic_blocked_not_for_listen_pack",
                "release_status": "not_release_ready",
                "failed_checks": ["instrumental_resurrection_model_inference_failed"],
                "error": error_text,
                "returncode": exc.returncode,
                "model": {
                    "name": str(args.model_name),
                    "model_type": str(args.model_type),
                    "config_path": str(config_path),
                    "runtime_config_path": str(runtime_config_path),
                    "checkpoint_path": str(model_path),
                    "model_implementation_path": str(model_implementation_path) if model_implementation_path else None,
                    "target_instrument": str(args.target_instrument),
                    "device": str(args.device),
                    "registry_recovery": str(args.registry_recovery),
                    "strict_checkpoint_load": True,
                    "verified_sha256": verified_sha256,
                    "precision_contract": precision_contract,
                },
                "model_source": model_source,
                "execution_contract": execution_contract,
                "artifacts": {
                    "progress_log_jsonl": str(progress_log_path),
                },
                "elapsed_sec": round(time.perf_counter() - started_at, 3),
            }
            (output_dir / "stem_manifest.json").write_text(
                json.dumps(blocked_manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            raise
        _mark_progress(
            progress_log_path,
            run_started_at=started_at,
            stage="model_inference",
            status="finish",
            stage_started_at=stage_started_at,
            store_dir=str(store_dir),
            target_instrument=str(args.target_instrument),
            precision_contract=precision_contract,
            resident_worker_evidence=execution_contract.get("resident_worker_evidence"),
        )

        target_output_path = _resolve_output_by_stem(store_dir, args.target_instrument)
        if not target_output_path.exists():
            raise FileNotFoundError(f"Instrumental Resurrection target stem is missing: {target_output_path}")

        stage_started_at = _mark_progress(
            progress_log_path,
            run_started_at=started_at,
            stage="write_artifacts",
            status="start",
            target_output_path=str(target_output_path),
        )
        minus_audio, minus_sample_rate = sf.read(str(target_output_path), always_2d=True)
        minus_audio = resample_audio_to_layout(
            minus_audio,
            source_sample_rate=int(minus_sample_rate),
            target_sample_rate=int(sample_rate),
            target_frames=int(frames),
        )
        minus_sample_rate = int(sample_rate)
        raw_ref_flac_path = output_dir / "raw_ref.flac"
        minus_flac_path = output_dir / str(args.minus_output_name)

        _write_flac_copy(source_audio_path, raw_ref_flac_path)
        sf.write(str(minus_flac_path), minus_audio, minus_sample_rate, subtype="PCM_24")
        _mark_progress(
            progress_log_path,
            run_started_at=started_at,
            stage="write_artifacts",
            status="finish",
            stage_started_at=stage_started_at,
            raw_ref_flac=str(raw_ref_flac_path),
            instrumental_resurrection_minus_flac=str(minus_flac_path),
        )

        elapsed_sec = round(time.perf_counter() - started_at, 3)
        target_stem_artifact = {
            "target_stem": "minus",
            "path": str(minus_flac_path),
            "source": "instrumental_resurrection_model_output",
            "source_stems": [str(args.target_instrument)],
        }
        manifest = {
            "schema_version": 1,
            "mode": "export_instrumental_resurrection_minus",
            "source_audio_path": str(source_audio_path),
            "target_stem": "minus",
            "target_audio_path": str(minus_flac_path),
            "target_stem_artifact": target_stem_artifact,
            "stems": {
                "minus": target_stem_artifact,
            },
            "instrumental_resurrection_output_path": str(minus_flac_path),
            "instrumental_resurrection_model_output_name": target_output_path.name,
            "artifacts": {
                "raw_ref_flac": str(raw_ref_flac_path),
                "instrumental_resurrection_minus_flac": str(minus_flac_path),
                "progress_log_jsonl": str(progress_log_path),
            },
            "model": {
                "name": str(args.model_name),
                "model_type": str(args.model_type),
                "config_path": str(config_path),
                "runtime_config_path": str(runtime_config_path),
                "runtime_config_sha256": _sha256(runtime_config_path),
                "checkpoint_path": str(model_path),
                "model_implementation_path": str(model_implementation_path) if model_implementation_path else None,
                "target_instrument": str(args.target_instrument),
                "device": str(args.device),
                "registry_recovery": str(args.registry_recovery),
                "strict_checkpoint_load": True,
                "verified_sha256": verified_sha256,
                "inference_overrides": inference_overrides,
                "precision_contract": precision_contract,
                "configured_sample_rate": model_sample_rate,
            },
            "model_source": model_source,
            "model_production_allowed": model_source.get("production_allowed") is True if model_source else None,
            "delivery_allowed": False,
            "delivery_status": "candidate_only",
            "presentation_status": "blocked_until_judged",
            "release_status": "not_release_ready",
            "requires_post_judge": True,
            "requested_listen_dir": str(listen_dir) if listen_dir is not None else None,
            "listen_pack_status": (
                "skipped_candidate_not_accepted" if listen_dir is not None else "not_requested"
            ),
            "audio": {
                "sample_rate": int(sample_rate),
                "frames": int(frames),
                "duration_sec": round(float(frames) / float(sample_rate), 6) if sample_rate else 0.0,
                "model_input_sample_rate": int(model_sample_rate),
                "model_input_frames": int(model_input_frames),
                "sample_rate_conversion": {
                    "performed": int(model_sample_rate) != int(sample_rate),
                    "source_sample_rate": int(sample_rate),
                    "model_sample_rate": int(model_sample_rate),
                    "resampler": "soxr_hq" if int(model_sample_rate) != int(sample_rate) else "none",
                    "target_output_restored_to_source_rate": True,
                },
            },
            "execution_contract": execution_contract,
            "elapsed_sec": elapsed_sec,
        }
        manifest_path = output_dir / "stem_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        if listen_dir is not None:
            _mark_progress(
                progress_log_path,
                run_started_at=started_at,
                stage="listen_pack",
                status="skipped",
                listen_dir=str(listen_dir),
                reason="candidate_not_accepted",
            )

        _mark_progress(
            progress_log_path,
            run_started_at=started_at,
            stage="finish",
            status="finish",
            output_dir=str(output_dir),
            manifest_path=str(manifest_path),
            elapsed_sec=elapsed_sec,
        )
        _emit(f"EXIT output_dir={output_dir} elapsed_sec={elapsed_sec}")
        print(json.dumps({"output_dir": str(output_dir), "manifest_path": str(manifest_path)}, indent=2))
    if resident_worker_response is not None:
        scratch_cleanup_verified = all(not path.exists() for path in resident_scratch_paths)
        manifest_payload = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
        resident_evidence = manifest_payload.get("execution_contract", {}).get("resident_worker_evidence")
        if not isinstance(resident_evidence, dict):
            raise RuntimeError("Resident worker evidence is missing from final manifest")
        resident_evidence["scratch_cleanup_verified"] = scratch_cleanup_verified
        if not scratch_cleanup_verified:
            manifest_payload["delivery_allowed"] = False
            manifest_payload["delivery_status"] = "blocked_resident_scratch_cleanup_failed"
            manifest_payload["presentation_status"] = "diagnostic_blocked_not_for_listen_pack"
        (output_dir / "stem_manifest.json").write_text(
            json.dumps(manifest_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _mark_progress(
            progress_log_path,
            run_started_at=started_at,
            stage="resident_scratch_cleanup",
            status="finish" if scratch_cleanup_verified else "failed",
            scratch_cleanup_verified=scratch_cleanup_verified,
        )
        if not scratch_cleanup_verified:
            raise RuntimeError("Resident worker scratch cleanup failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
