"""In-process resident separator executor POC for persistent audio routes.

This module is deliberately fail-closed: resident execution can produce route
artifacts for the persistent separator server, but it cannot accept delivery.
"""

from __future__ import annotations

import json
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Sequence


RESIDENT_FIXTURE_ROUTE_ID = "resident_fixture_copy"
AUDIO_SEPARATOR_RESIDENT_ROUTE_IDS = {
    "audio_separator_bs_roformer_vocals",
    "audio_separator_bs_roformer_vocals:instrumental",
    "audio_separator_bs_roformer_viperx_1297_vendor_config_vocals",
    "audio_separator_melband_roformer_viperx_vocals",
}
AUDIO_SEPARATOR_EXPORTER = "/app/tools/export_audio_separator_vocal_full.py"
AUDIO_SEPARATOR_EXPORTER_SUFFIX = "tools/export_audio_separator_vocal_full.py"
AUDIO_SEPARATOR_WORKER_RESPONSE_PREFIX = "__AUDIO_SEPARATOR_WORKER_JSON__"
MELBAND_RESIDENT_ROUTE_IDS = {
    "melband_roformer_vocals",
    "melband_roformer_vocals:instrumental",
    "becruily_instrumental_vocals",
    "becruily_instrumental_vocals:instrumental",
}
MELBAND_EXPORTER = "/app/tools/export_melband_vocal_full.py"
MELBAND_EXPORTER_SUFFIX = "tools/export_melband_vocal_full.py"
MELBAND_WORKER_RESPONSE_PREFIX = "__MELBAND_WORKER_JSON__"


AUDIO_SEPARATOR_WORKER_CODE = r"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

PREFIX = "__AUDIO_SEPARATOR_WORKER_JSON__"
separator = None
model_load_count = 0
worker_request_count = 0


def respond(payload):
    sys.stdout.write(PREFIX + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def runtime_contract(payload, providers, probe_error=None):
    model_file_dir = payload["model_file_dir"]
    model_filename = payload["model_filename"]
    model_file = os.path.join(model_file_dir, model_filename)
    model_sidecar = os.path.join(model_file_dir, os.path.splitext(model_filename)[0] + ".yaml")
    blockers = []
    if not os.path.isfile(model_file):
        blockers.append("audio_separator_model_file_missing")
    if not os.path.isfile(model_sidecar):
        blockers.append("audio_separator_model_sidecar_missing")
    if probe_error:
        blockers.append("audio_separator_runtime_probe_failed")
    cuda_provider_available = "CUDAExecutionProvider" in providers
    if payload.get("require_cuda_provider") and not cuda_provider_available:
        blockers.append("cuda_execution_provider_missing")
    return {
        "status": "ready" if not blockers else "blocked",
        "model_file": model_file,
        "model_file_exists": os.path.isfile(model_file),
        "model_sidecar": model_sidecar,
        "model_sidecar_exists": os.path.isfile(model_sidecar),
        "available_providers": providers,
        "cuda_provider_available": cuda_provider_available,
        "require_cuda_provider": bool(payload.get("require_cuda_provider")),
        "probe_error": probe_error,
        "blockers": blockers,
    }


try:
    import onnxruntime as ort
    from audio_separator.separator import Separator
    providers = list(ort.get_available_providers())
    import_error = None
except Exception as exc:
    providers = []
    import_error = f"{type(exc).__name__}: {exc}"
    Separator = None


for raw_line in sys.stdin:
    raw_line = raw_line.strip()
    if not raw_line:
        continue
    try:
        payload = json.loads(raw_line)
        request_id = payload.get("request_id", "")
        if payload.get("action") == "shutdown":
            respond({"request_id": request_id, "status": "ok", "shutdown": True})
            break
        worker_request_count += 1
        contract = runtime_contract(payload, providers, import_error)
        if contract["status"] != "ready" or Separator is None:
            respond({
                "request_id": request_id,
                "status": "blocked",
                "reason": "audio_separator_runtime_contract_blocked",
                "runtime_contract": contract,
                "available_providers": providers,
                "model_load_count": model_load_count,
                "worker_request_count": worker_request_count,
                "persistent_worker_process_used": True,
                "external_runtime_subprocess_used": True,
                "per_request_external_runtime_subprocess_used": False,
            })
            continue
        started = time.monotonic()
        if separator is None:
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
            model_load_count += 1
        else:
            separator.output_dir = payload["model_output_dir"]
            if getattr(separator, "model_instance", None) is not None:
                separator.model_instance.output_dir = payload["model_output_dir"]
        output_files = separator.separate(
            payload["source_audio_path"],
            {
                "Vocals": "vocals",
                "Instrumental": "instrumental",
            },
        )
        respond({
            "request_id": request_id,
            "status": "ok",
            "output_files": list(output_files),
            "runtime_contract": contract,
            "available_providers": providers,
            "model_load_count": model_load_count,
            "worker_request_count": worker_request_count,
            "worker_elapsed_sec": round(time.monotonic() - started, 6),
            "persistent_worker_process_used": True,
            "external_runtime_subprocess_used": True,
            "per_request_external_runtime_subprocess_used": False,
        })
    except Exception as exc:
        respond({
            "request_id": locals().get("request_id", ""),
            "status": "blocked",
            "reason": "audio_separator_worker_exception",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback_tail": traceback.format_exc()[-4000:],
            },
            "model_load_count": model_load_count,
            "worker_request_count": worker_request_count,
            "persistent_worker_process_used": True,
            "external_runtime_subprocess_used": True,
            "per_request_external_runtime_subprocess_used": False,
        })
"""


MELBAND_WORKER_CODE = r"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

PREFIX = "__MELBAND_WORKER_JSON__"
model = None
config = None
device = None
loaded_key = None
model_load_count = 0
worker_request_count = 0


def respond(payload):
    sys.stdout.write(PREFIX + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def runtime_contract(payload, import_error=None):
    blockers = []
    if import_error:
        blockers.append("melband_runtime_import_failed")
    for key, reason in (
        ("repo_path", "melband_repo_missing"),
        ("model_path", "melband_model_missing"),
        ("config_path", "melband_config_missing"),
        ("input_wav_path", "melband_input_wav_missing"),
    ):
        path = str(payload.get(key) or "")
        if not path or not os.path.exists(path):
            blockers.append(reason)
    cuda_available = False
    if "torch" in globals() and torch is not None:
        cuda_available = bool(torch.cuda.is_available())
    if payload.get("require_gpu") and not cuda_available:
        blockers.append("cuda_unavailable")
    return {
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "cuda_available": cuda_available,
        "require_gpu": bool(payload.get("require_gpu")),
        "repo_path": str(payload.get("repo_path") or ""),
        "model_path": str(payload.get("model_path") or ""),
        "config_path": str(payload.get("config_path") or ""),
        "import_error": import_error,
    }


try:
    import numpy as np
    import soundfile as sf
    import torch
    import torch.nn as nn
    import yaml
    from ml_collections import ConfigDict
    import_error = None
except Exception as exc:
    np = None
    sf = None
    torch = None
    nn = None
    yaml = None
    ConfigDict = None
    import_error = f"{type(exc).__name__}: {exc}"


def ensure_loaded(payload):
    global model, config, device, loaded_key, model_load_count
    repo_path = os.path.abspath(str(payload["repo_path"]))
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    from utils import demix_track, get_model_from_config

    key = (
        str(payload["model_type"]),
        os.path.abspath(str(payload.get("model_cache_config_path") or payload["config_path"])),
        os.path.abspath(str(payload["model_path"])),
        str(payload.get("device_id", 0)),
        str(payload.get("num_overlap", "")),
    )
    if loaded_key == key and model is not None and config is not None and device is not None:
        return demix_track
    torch.backends.cudnn.benchmark = True
    with open(str(payload["config_path"]), "r", encoding="utf-8") as handle:
        config = ConfigDict(yaml.load(handle, Loader=yaml.FullLoader))
    model = get_model_from_config(str(payload["model_type"]), config)
    model_path = str(payload["model_path"])
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    if bool(torch.cuda.is_available()):
        device_id = int(payload.get("device_id", 0))
        device = torch.device(f"cuda:{device_id}")
        model = model.to(device)
    else:
        device = torch.device("cpu")
        model = model.to(device)
    model.eval()
    loaded_key = key
    model_load_count += 1
    return demix_track


def run_one(payload):
    demix_track = ensure_loaded(payload)
    input_wav_path = str(payload["input_wav_path"])
    output_dir = str(payload["model_output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    mix, sr = sf.read(input_wav_path)
    original_mono = False
    if len(mix.shape) == 1:
        original_mono = True
        mix = np.stack([mix, mix], axis=-1)
    mixture = torch.tensor(mix.T, dtype=torch.float32)
    with torch.inference_mode():
        res, _first_chunk_time = demix_track(config, model, mixture, device, None)
    instruments = config.training.instruments
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]
    stem_name = os.path.splitext(os.path.basename(input_wav_path))[0]
    output_files = []
    first_output = None
    for instr in instruments:
        stem_audio = res[instr].T
        if original_mono:
            stem_audio = stem_audio[:, 0]
        out_path = os.path.join(output_dir, f"{stem_name}_{instr}.wav")
        sf.write(out_path, stem_audio, sr, subtype="FLOAT")
        output_files.append(out_path)
        if first_output is None:
            first_output = stem_audio
    if first_output is not None:
        original_mix, _ = sf.read(input_wav_path)
        instrumental = original_mix - first_output
        instrumental_path = os.path.join(output_dir, f"{stem_name}_instrumental.wav")
        sf.write(instrumental_path, instrumental, sr, subtype="FLOAT")
        output_files.append(instrumental_path)
    return output_files


for raw_line in sys.stdin:
    raw_line = raw_line.strip()
    if not raw_line:
        continue
    try:
        payload = json.loads(raw_line)
        request_id = payload.get("request_id", "")
        if payload.get("action") == "shutdown":
            respond({"request_id": request_id, "status": "ok", "shutdown": True})
            break
        worker_request_count += 1
        contract = runtime_contract(payload, import_error)
        if contract["status"] != "ready" or import_error:
            respond({
                "request_id": request_id,
                "status": "blocked",
                "reason": "melband_runtime_contract_blocked",
                "runtime_contract": contract,
                "model_load_count": model_load_count,
                "worker_request_count": worker_request_count,
                "persistent_worker_process_used": True,
                "per_request_external_runtime_subprocess_used": False,
            })
            continue
        started = time.monotonic()
        output_files = run_one(payload)
        respond({
            "request_id": request_id,
            "status": "ok",
            "output_files": output_files,
            "runtime_contract": contract,
            "model_load_count": model_load_count,
            "worker_request_count": worker_request_count,
            "persistent_worker_process_used": True,
            "per_request_external_runtime_subprocess_used": False,
            "elapsed_sec": round(time.monotonic() - started, 6),
        })
    except Exception as exc:
        respond({
            "request_id": locals().get("request_id", ""),
            "status": "blocked",
            "reason": "melband_worker_exception",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "model_load_count": model_load_count,
            "worker_request_count": worker_request_count,
            "persistent_worker_process_used": True,
            "per_request_external_runtime_subprocess_used": False,
        })
"""


class ResidentSeparatorExecutor:
    def __init__(self, executor_id: str) -> None:
        self.executor_id = str(executor_id)
        self.adapter_kind = "fixture_copy"
        self._loaded = False
        self.model_load_count = 0
        self.worker_request_count = 0
        self.last_warm_elapsed_sec = 0.0
        self.last_used_at = 0.0

    def warm(self) -> None:
        started = time.monotonic()
        if not self._loaded:
            self.model_load_count += 1
            self._loaded = True
        self.last_warm_elapsed_sec = round(time.monotonic() - started, 6)
        self.last_used_at = time.time()

    def snapshot(self) -> dict[str, Any]:
        return {
            "executor_id": self.executor_id,
            "adapter_kind": self.adapter_kind,
            "resident_capable": True,
            "loaded": self._loaded,
            "model_load_count": self.model_load_count,
            "worker_request_count": self.worker_request_count,
            "last_warm_elapsed_sec": self.last_warm_elapsed_sec,
            "last_used_at": self.last_used_at,
            "can_accept_delivery": False,
        }

    def separate(
        self,
        *,
        source_audio_path: str | Path,
        output_dir: str | Path,
        stems: Sequence[str],
        device: str,
        require_gpu: bool,
        command: Sequence[str] | None = None,
        timeout_sec: int | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        source = Path(source_audio_path)
        target_dir = Path(output_dir) / "stems"
        target_dir.mkdir(parents=True, exist_ok=True)
        if require_gpu:
            return self._blocked(
                "resident_fixture_gpu_inference_unavailable",
                output_dir=output_dir,
                stems_dir=target_dir,
                missing_stems=[str(stem) for stem in stems],
                elapsed_sec=round(time.monotonic() - started, 6),
                extra={
                    "device": str(device),
                    "require_gpu": True,
                    "gpu_inference_skipped": True,
                    "accelerator_mode": "resident_fixture_cpu_copy",
                },
            )
        invalid_stems = [stem for stem in stems if not _is_safe_stem_name(str(stem))]
        if invalid_stems:
            return self._blocked(
                "invalid_stem_name",
                output_dir=output_dir,
                stems_dir=target_dir,
                missing_stems=[str(stem) for stem in stems],
                elapsed_sec=round(time.monotonic() - started, 6),
                extra={"invalid_stems": invalid_stems},
            )
        if not source.is_file():
            return self._blocked(
                "missing_source_audio",
                output_dir=output_dir,
                stems_dir=target_dir,
                missing_stems=[str(stem) for stem in stems],
                elapsed_sec=round(time.monotonic() - started, 6),
            )
        self.warm()
        for stem in stems:
            shutil.copy2(source, target_dir / f"{stem}.flac")
        return {
            "status": "ok",
            "resident_backend_used": True,
            "resident_capable": True,
            "resident_executor": self.snapshot(),
            "stems_dir": str(target_dir),
            "missing_stems": [],
            "device": str(device),
            "require_gpu": bool(require_gpu),
            "gpu_inference_skipped": True,
            "accelerator_mode": "resident_fixture_cpu_copy",
            "elapsed_sec": round(time.monotonic() - started, 6),
            "delivery_allowed": False,
            "can_accept_delivery": False,
        }

    def _blocked(
        self,
        reason: str,
        *,
        output_dir: str | Path,
        stems_dir: str | Path,
        missing_stems: list[str],
        elapsed_sec: float,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = {
            "status": "blocked",
            "reason": reason,
            "resident_backend_used": True,
            "resident_capable": True,
            "resident_executor": self.snapshot(),
            "output_dir": str(output_dir),
            "stems_dir": str(stems_dir),
            "missing_stems": missing_stems,
            "elapsed_sec": elapsed_sec,
            "delivery_allowed": False,
            "can_accept_delivery": False,
        }
        response.update(extra or {})
        return response


class AudioSeparatorWorkerProcess:
    def __init__(self, *, audio_separator_python: Path, worker_key: str) -> None:
        self.audio_separator_python = audio_separator_python
        self.worker_key = worker_key
        self.process: subprocess.Popen[str] | None = None
        self.lines: queue.Queue[str] = queue.Queue()
        self.logs: list[str] = []
        self.lock = threading.Lock()
        self.request_counter = 0

    def request(self, payload: dict[str, Any], *, timeout_sec: int) -> dict[str, Any]:
        with self.lock:
            self._ensure_started()
            self.request_counter += 1
            request_id = f"r{self.request_counter:06d}"
            request_payload = dict(payload)
            request_payload["request_id"] = request_id
            assert self.process is not None
            assert self.process.stdin is not None
            self.process.stdin.write(json.dumps(request_payload, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            deadline = time.monotonic() + max(1.0, float(timeout_sec))
            ignored_protocol_messages: list[dict[str, Any]] = []
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    return {
                        "request_id": request_id,
                        "status": "blocked",
                        "reason": "audio_separator_worker_process_exited",
                        "returncode": self.process.returncode,
                        "worker_logs_tail": self.logs[-40:],
                        "persistent_worker_process_used": True,
                    }
                timeout = max(0.05, min(0.5, deadline - time.monotonic()))
                try:
                    line = self.lines.get(timeout=timeout)
                except queue.Empty:
                    continue
                if not line.startswith(AUDIO_SEPARATOR_WORKER_RESPONSE_PREFIX):
                    self._append_log(line)
                    continue
                try:
                    response = json.loads(line[len(AUDIO_SEPARATOR_WORKER_RESPONSE_PREFIX) :])
                except json.JSONDecodeError:
                    self._append_log(line)
                    continue
                if response.get("request_id") == request_id:
                    response.setdefault("worker_logs_tail", self.logs[-40:])
                    return response
                ignored_protocol_messages.append(response)
            worker_logs_tail = self.logs[-40:]
            self.stop()
            return {
                "request_id": request_id,
                "status": "blocked",
                "reason": "audio_separator_worker_timeout",
                "timeout_sec": int(timeout_sec),
                "ignored_protocol_messages": ignored_protocol_messages[-5:],
                "worker_logs_tail": worker_logs_tail,
                "persistent_worker_process_used": True,
            }

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            try:
                assert process.stdin is not None
                process.stdin.write(json.dumps({"action": "shutdown", "request_id": "shutdown"}) + "\n")
                process.stdin.flush()
            except OSError:
                pass
        deadline = time.monotonic() + 2.0
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if process.poll() is None:
            process.kill()
        self.process = None

    def _ensure_started(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        if not self.audio_separator_python.is_file():
            raise FileNotFoundError(f"Audio Separator Python runtime is missing: {self.audio_separator_python}")
        self.process = subprocess.Popen(
            [str(self.audio_separator_python), "-u", "-c", AUDIO_SEPARATOR_WORKER_CODE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.logs.clear()
        threading.Thread(target=self._read_stdout, daemon=True).start()

    def _read_stdout(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        while True:
            line = process.stdout.readline()
            if line == "":
                break
            self.lines.put(line)

    def _append_log(self, line: str) -> None:
        self.logs.append(line.rstrip())
        if len(self.logs) > 200:
            del self.logs[:100]


class MelBandWorkerProcess(AudioSeparatorWorkerProcess):
    def __init__(self, *, worker_key: str) -> None:
        super().__init__(audio_separator_python=Path(sys.executable), worker_key=worker_key)

    def request(self, payload: dict[str, Any], *, timeout_sec: int) -> dict[str, Any]:
        with self.lock:
            self._ensure_started()
            self.request_counter += 1
            request_id = f"r{self.request_counter:06d}"
            request_payload = dict(payload)
            request_payload["request_id"] = request_id
            assert self.process is not None
            assert self.process.stdin is not None
            self.process.stdin.write(json.dumps(request_payload, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            deadline = time.monotonic() + max(1.0, float(timeout_sec))
            ignored_protocol_messages: list[dict[str, Any]] = []
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    return {
                        "request_id": request_id,
                        "status": "blocked",
                        "reason": "melband_worker_process_exited",
                        "returncode": self.process.returncode,
                        "worker_logs_tail": self.logs[-40:],
                        "persistent_worker_process_used": True,
                    }
                timeout = max(0.05, min(0.5, deadline - time.monotonic()))
                try:
                    line = self.lines.get(timeout=timeout)
                except queue.Empty:
                    continue
                if not line.startswith(MELBAND_WORKER_RESPONSE_PREFIX):
                    self._append_log(line)
                    continue
                try:
                    response = json.loads(line[len(MELBAND_WORKER_RESPONSE_PREFIX) :])
                except json.JSONDecodeError:
                    self._append_log(line)
                    continue
                if response.get("request_id") == request_id:
                    response.setdefault("worker_logs_tail", self.logs[-40:])
                    return response
                ignored_protocol_messages.append(response)
            worker_logs_tail = self.logs[-40:]
            self.stop()
            return {
                "request_id": request_id,
                "status": "blocked",
                "reason": "melband_worker_timeout",
                "timeout_sec": int(timeout_sec),
                "ignored_protocol_messages": ignored_protocol_messages[-5:],
                "worker_logs_tail": worker_logs_tail,
                "persistent_worker_process_used": True,
            }

    def _ensure_started(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.process = subprocess.Popen(
            [sys.executable, "-u", "-c", MELBAND_WORKER_CODE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.logs.clear()
        threading.Thread(target=self._read_stdout, daemon=True).start()


class AudioSeparatorResidentExecutor(ResidentSeparatorExecutor):
    def __init__(self, executor_id: str) -> None:
        super().__init__(executor_id)
        self.adapter_kind = "audio_separator_worker"

    def separate(
        self,
        *,
        source_audio_path: str | Path,
        output_dir: str | Path,
        stems: Sequence[str],
        device: str,
        require_gpu: bool,
        command: Sequence[str] | None = None,
        timeout_sec: int | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        target_dir = Path(output_dir) / "stems"
        target_dir.mkdir(parents=True, exist_ok=True)
        if require_gpu and str(device).lower() != "cuda":
            return self._blocked(
                "cpu_fallback_forbidden",
                output_dir=output_dir,
                stems_dir=target_dir,
                missing_stems=[str(stem) for stem in stems],
                elapsed_sec=round(time.monotonic() - started, 6),
                extra={"device": str(device), "require_gpu": True},
            )
        if not require_gpu:
            return self._blocked(
                "resident_audio_separator_requires_gpu",
                output_dir=output_dir,
                stems_dir=target_dir,
                missing_stems=[str(stem) for stem in stems],
                elapsed_sec=round(time.monotonic() - started, 6),
                extra={"device": str(device), "require_gpu": False},
            )
        source = Path(source_audio_path)
        if not source.is_file():
            return self._blocked(
                "missing_source_audio",
                output_dir=output_dir,
                stems_dir=target_dir,
                missing_stems=[str(stem) for stem in stems],
                elapsed_sec=round(time.monotonic() - started, 6),
            )
        try:
            request = self._build_worker_request(
                command=command or (),
                source_audio_path=source,
                output_dir=Path(output_dir),
                stems=stems,
                timeout_sec=timeout_sec,
            )
        except ValueError as exc:
            return self._blocked(
                str(exc),
                output_dir=output_dir,
                stems_dir=target_dir,
                missing_stems=[str(stem) for stem in stems],
                elapsed_sec=round(time.monotonic() - started, 6),
            )
        self.warm()
        self.worker_request_count += 1
        try:
            worker_response = self._run_worker_request(request)
        except Exception as exc:
            return self._blocked(
                "resident_audio_separator_worker_failed",
                output_dir=output_dir,
                stems_dir=target_dir,
                missing_stems=[str(stem) for stem in stems],
                elapsed_sec=round(time.monotonic() - started, 6),
                extra={
                    "worker_error": {"type": type(exc).__name__, "message": str(exc)},
                    "gpu_inference_skipped": False,
                    "accelerator_mode": "audio_separator_resident_worker",
                },
            )
        status = str(worker_response.get("status") or "blocked")
        missing = [stem for stem in stems if not (target_dir / f"{stem}.flac").is_file()]
        if status != "ok" or missing:
            return self._blocked(
                str(worker_response.get("reason") or "resident_audio_separator_missing_required_stems"),
                output_dir=output_dir,
                stems_dir=target_dir,
                missing_stems=missing or [str(stem) for stem in stems],
                elapsed_sec=round(time.monotonic() - started, 6),
                extra={
                    "worker_response": worker_response,
                    "gpu_inference_skipped": False,
                    "accelerator_mode": "audio_separator_resident_worker",
                },
            )
        return {
            "status": "ok",
            "resident_backend_used": True,
            "resident_capable": True,
            "resident_executor": self.snapshot(),
            "stems_dir": str(target_dir),
            "missing_stems": [],
            "worker_response": worker_response,
            "resident_benchmark": {
                "adapter_elapsed_sec": round(time.monotonic() - started, 6),
                "model_load_count": int(worker_response.get("model_load_count") or self.model_load_count),
                "worker_request_count": int(worker_response.get("worker_request_count") or self.worker_request_count),
                "persistent_worker_process_used": bool(worker_response.get("persistent_worker_process_used", False)),
                "per_request_external_runtime_subprocess_used": bool(
                    worker_response.get("per_request_external_runtime_subprocess_used", False)
                ),
            },
            "gpu_inference_skipped": False,
            "accelerator_mode": "audio_separator_resident_worker",
            "elapsed_sec": round(time.monotonic() - started, 6),
            "delivery_allowed": False,
            "can_accept_delivery": False,
        }

    def _build_worker_request(
        self,
        *,
        command: Sequence[str],
        source_audio_path: Path,
        output_dir: Path,
        stems: Sequence[str],
        timeout_sec: int | None = None,
    ) -> dict[str, Any]:
        parts = [str(part) for part in command]
        if not any(_is_audio_separator_exporter_command_part(part) for part in parts):
            raise ValueError("resident_audio_separator_exporter_mismatch")
        return {
            "schema": "audio_separator_resident_request.v1",
            "route_id": self.executor_id,
            "source_audio_path": str(source_audio_path),
            "output_dir": str(output_dir),
            "stems_dir": str(output_dir / "stems"),
            "stems": [str(stem) for stem in stems],
            "model_filename": _command_value(parts, "--model-filename", required=True),
            "model_file_dir": _command_value(parts, "--model-file-dir", required=True),
            "audio_separator_python": _command_value(parts, "--audio-separator-python", required=True),
            "require_cuda_provider": "--require-cuda-provider" in parts,
            "output_format": _command_value(parts, "--output-format", default="FLAC"),
            "normalization_threshold": float(_command_value(parts, "--normalization-threshold", default="0.9")),
            "mdx_overlap": float(_command_value(parts, "--mdx-overlap", default="0.25")),
            "mdx_batch_size": int(_command_value(parts, "--mdx-batch-size", default="1")),
            "enable_denoise": "--enable-denoise" in parts,
            "timeout_sec": int(timeout_sec or _command_value(parts, "--timeout-sec", default="180")),
        }

    def _run_worker_request(self, request: dict[str, Any]) -> dict[str, Any]:
        from tools import export_audio_separator_vocal_full as exporter

        output_dir = Path(str(request["output_dir"]))
        stems_dir = Path(str(request["stems_dir"]))
        source_audio_path = Path(str(request["source_audio_path"]))
        model_file_dir = Path(str(request["model_file_dir"]))
        audio_separator_python = Path(str(request["audio_separator_python"]))
        model_filename = str(request["model_filename"])
        output_dir.mkdir(parents=True, exist_ok=True)
        stems_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="resident_as_out_") as model_output_dir_str, tempfile.TemporaryDirectory(
            prefix="resident_as_in_"
        ) as input_dir_str:
            model_output_dir = Path(model_output_dir_str)
            input_dir = Path(input_dir_str)
            raw_ref_wav_path = input_dir / "full_track.wav"
            sample_rate, frames = exporter.save_audio_window_as_wav(
                source_audio_path,
                raw_ref_wav_path,
                start_sec=0.0,
                end_sec=None,
            )
            worker = _get_audio_separator_worker(
                audio_separator_python=audio_separator_python,
                request=request,
            )
            worker_payload = {
                **request,
                "model_output_dir": str(model_output_dir),
            }
            worker_response = worker.request(
                worker_payload,
                timeout_sec=int(request["timeout_sec"]),
            )
            if str(worker_response.get("status") or "") != "ok":
                return {
                    **worker_response,
                    "status": "blocked",
                    "reason": str(worker_response.get("reason") or "audio_separator_worker_blocked"),
                    "runtime_contract": worker_response.get("runtime_contract") or {},
                }
            output_files = list(worker_response.get("output_files") or [])
            vocals_source = exporter._resolve_named_output(output_files, model_output_dir, "vocals")
            instrumental_source = exporter._resolve_named_output(output_files, model_output_dir, "instrumental")
            vocals_path = stems_dir / "vocals.flac"
            instrumental_path = stems_dir / "instrumental.flac"
            raw_ref_path = output_dir / "raw_ref.flac"
            exporter._copy_audio_as_flac(vocals_source, vocals_path)
            exporter._copy_audio_as_flac(instrumental_source, instrumental_path)
            exporter._copy_audio_as_flac(raw_ref_wav_path, raw_ref_path)
            manifest_path = exporter.write_full_track_stem_manifest(
                output_dir=output_dir,
                source_audio_path=source_audio_path,
                vocals_flac_path=vocals_path,
                instrumental_flac_path=instrumental_path,
                raw_ref_flac_path=raw_ref_path,
                model_filename=model_filename,
                model_file_dir=model_file_dir,
                model_id=self.executor_id,
                model_display_name=self.executor_id,
                output_format=str(request["output_format"]),
                normalization_threshold=float(request["normalization_threshold"]),
                mdx_overlap=float(request["mdx_overlap"]),
                mdx_batch_size=int(request["mdx_batch_size"]),
                enable_denoise=bool(request["enable_denoise"]),
                sample_rate=int(sample_rate),
                frames=int(frames),
                elapsed_sec=time.monotonic() - started,
                runtime_contract=worker_response.get("runtime_contract") or {},
            )
        return {
            "status": "ok",
            "stems_dir": str(stems_dir),
            "output_files": [str(vocals_path), str(instrumental_path)],
            "manifest_path": str(manifest_path),
            "runtime_contract": worker_response.get("runtime_contract") or {},
            "external_runtime_subprocess_used": True,
            "per_request_external_runtime_subprocess_used": False,
            "persistent_worker_process_used": True,
            "model_load_count": int(worker_response.get("model_load_count") or 0),
            "worker_request_count": int(worker_response.get("worker_request_count") or 0),
            "worker_response": worker_response,
            "worker_elapsed_sec": round(time.monotonic() - started, 6),
        }


class MelBandResidentExecutor(ResidentSeparatorExecutor):
    def __init__(self, executor_id: str) -> None:
        super().__init__(executor_id)
        self.adapter_kind = "melband_roformer_worker"

    def separate(
        self,
        *,
        source_audio_path: str | Path,
        output_dir: str | Path,
        stems: Sequence[str],
        device: str,
        require_gpu: bool,
        command: Sequence[str] | None = None,
        timeout_sec: int | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        target_dir = Path(output_dir) / "stems"
        target_dir.mkdir(parents=True, exist_ok=True)
        if require_gpu and str(device).lower() != "cuda":
            return self._blocked(
                "cpu_fallback_forbidden",
                output_dir=output_dir,
                stems_dir=target_dir,
                missing_stems=[str(stem) for stem in stems],
                elapsed_sec=round(time.monotonic() - started, 6),
                extra={"device": str(device), "require_gpu": True},
            )
        if not require_gpu:
            return self._blocked(
                "resident_melband_requires_gpu",
                output_dir=output_dir,
                stems_dir=target_dir,
                missing_stems=[str(stem) for stem in stems],
                elapsed_sec=round(time.monotonic() - started, 6),
                extra={"device": str(device), "require_gpu": False},
            )
        source = Path(source_audio_path)
        if not source.is_file():
            return self._blocked(
                "missing_source_audio",
                output_dir=output_dir,
                stems_dir=target_dir,
                missing_stems=[str(stem) for stem in stems],
                elapsed_sec=round(time.monotonic() - started, 6),
            )
        try:
            request = self._build_worker_request(
                command=command or (),
                source_audio_path=source,
                output_dir=Path(output_dir),
                stems=stems,
                timeout_sec=timeout_sec,
                require_gpu=require_gpu,
            )
        except ValueError as exc:
            return self._blocked(
                str(exc),
                output_dir=output_dir,
                stems_dir=target_dir,
                missing_stems=[str(stem) for stem in stems],
                elapsed_sec=round(time.monotonic() - started, 6),
            )
        self.warm()
        self.worker_request_count += 1
        try:
            worker_response = self._run_worker_request(request)
        except Exception as exc:
            return self._blocked(
                "resident_melband_worker_failed",
                output_dir=output_dir,
                stems_dir=target_dir,
                missing_stems=[str(stem) for stem in stems],
                elapsed_sec=round(time.monotonic() - started, 6),
                extra={
                    "worker_error": {"type": type(exc).__name__, "message": str(exc)},
                    "gpu_inference_skipped": False,
                    "accelerator_mode": "melband_resident_worker",
                },
            )
        status = str(worker_response.get("status") or "blocked")
        missing = [stem for stem in stems if not (target_dir / f"{stem}.flac").is_file()]
        if status != "ok" or missing:
            return self._blocked(
                str(worker_response.get("reason") or "resident_melband_missing_required_stems"),
                output_dir=output_dir,
                stems_dir=target_dir,
                missing_stems=missing or [str(stem) for stem in stems],
                elapsed_sec=round(time.monotonic() - started, 6),
                extra={
                    "worker_response": worker_response,
                    "gpu_inference_skipped": False,
                    "accelerator_mode": "melband_resident_worker",
                },
            )
        return {
            "status": "ok",
            "resident_backend_used": True,
            "resident_capable": True,
            "resident_executor": self.snapshot(),
            "stems_dir": str(target_dir),
            "missing_stems": [],
            "worker_response": worker_response,
            "resident_benchmark": {
                "adapter_elapsed_sec": round(time.monotonic() - started, 6),
                "model_load_count": int(worker_response.get("model_load_count") or self.model_load_count),
                "worker_request_count": int(worker_response.get("worker_request_count") or self.worker_request_count),
                "persistent_worker_process_used": bool(worker_response.get("persistent_worker_process_used", False)),
                "per_request_external_runtime_subprocess_used": bool(
                    worker_response.get("per_request_external_runtime_subprocess_used", False)
                ),
            },
            "gpu_inference_skipped": False,
            "accelerator_mode": "melband_resident_worker",
            "elapsed_sec": round(time.monotonic() - started, 6),
            "delivery_allowed": False,
            "can_accept_delivery": False,
        }

    def _build_worker_request(
        self,
        *,
        command: Sequence[str],
        source_audio_path: Path,
        output_dir: Path,
        stems: Sequence[str],
        timeout_sec: int | None = None,
        require_gpu: bool = True,
    ) -> dict[str, Any]:
        from tools import export_melband_vocal_full as exporter

        parts = [str(part) for part in command]
        if not any(_is_melband_exporter_command_part(part) for part in parts):
            raise ValueError("resident_melband_exporter_mismatch")
        exporter_args = _exporter_args_after(parts, MELBAND_EXPORTER_SUFFIX)
        parsed = exporter.build_arg_parser().parse_args(exporter_args)
        return {
            "schema": "melband_resident_request.v1",
            "route_id": self.executor_id,
            "source_audio_path": str(source_audio_path),
            "output_dir": str(output_dir),
            "stems_dir": str(output_dir / "stems"),
            "stems": [str(stem) for stem in stems],
            "exporter_args": exporter_args,
            "repo_path": str(Path(parsed.repo_path)),
            "model_path": str(Path(parsed.model_path)),
            "config_path": str(Path(parsed.config_path)),
            "model_type": "mel_band_roformer",
            "device_id": int(parsed.device_id),
            "num_overlap": int(parsed.num_overlap),
            "manifest_model_id": str(parsed.manifest_model_id),
            "manifest_model_display_name": str(parsed.manifest_model_display_name),
            "require_gpu": bool(require_gpu),
            "timeout_sec": int(timeout_sec or 180),
        }

    def _run_worker_request(self, request: dict[str, Any]) -> dict[str, Any]:
        from tools import export_melband_vocal_full as exporter

        args = exporter.build_arg_parser().parse_args(list(request["exporter_args"]))
        output_dir = Path(str(request["output_dir"]))
        stems_dir = Path(str(request["stems_dir"]))
        source_audio_path = Path(str(request["source_audio_path"]))
        repo_path = Path(str(request["repo_path"]))
        model_path = Path(str(request["model_path"]))
        config_path = Path(str(request["config_path"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        stems_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="resident_mb_in_") as input_dir_str, tempfile.TemporaryDirectory(
            prefix="resident_mb_out_"
        ) as model_output_dir_str, tempfile.TemporaryDirectory(prefix="resident_mb_cfg_") as config_dir_str:
            input_dir = Path(input_dir_str)
            model_output_dir = Path(model_output_dir_str)
            config_dir = Path(config_dir_str)
            input_wav_path = input_dir / f"{exporter.MODEL_INPUT_STEM}.wav"
            runtime_config_path = config_dir / "config_runtime.yaml"
            sample_rate, frames = exporter.save_audio_window_as_wav(
                source_audio_path,
                input_wav_path,
                start_sec=0.0,
                end_sec=None,
            )
            exporter.write_config_with_overlap(config_path, runtime_config_path, num_overlap=int(args.num_overlap))
            output_contract = exporter.load_melband_output_contract(runtime_config_path)
            worker = _get_melband_worker(request=request)
            worker_payload = {
                **request,
                "input_wav_path": str(input_wav_path),
                "model_output_dir": str(model_output_dir),
                "model_cache_config_path": str(config_path),
                "config_path": str(runtime_config_path),
            }
            worker_response = worker.request(
                worker_payload,
                timeout_sec=int(request["timeout_sec"]),
            )
            if str(worker_response.get("status") or "") != "ok":
                return {
                    **worker_response,
                    "status": "blocked",
                    "reason": str(worker_response.get("reason") or "melband_worker_blocked"),
                    "runtime_contract": worker_response.get("runtime_contract") or {},
                }
            vocals_wav = exporter.try_resolve_model_output_path(model_output_dir, "vocals")
            instrumental_wav = exporter.try_resolve_model_output_path(model_output_dir, "instrumental")
            vocals_audio, vocals_sr = (
                exporter.sf.read(str(vocals_wav), always_2d=True) if vocals_wav is not None else (None, None)
            )
            instrumental_audio, instrumental_sr = (
                exporter.sf.read(str(instrumental_wav), always_2d=True) if instrumental_wav is not None else (None, None)
            )
            raw_audio, raw_sr = exporter.sf.read(str(input_wav_path), always_2d=True)
            resolved_stems = exporter.materialize_missing_companion_stems(
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
            tuned_vocals_audio = exporter.apply_light_debleed_presence(
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
            tuned_vocals_audio = exporter.apply_peak_note_debleed(
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
            tuned_vocals_audio = exporter.apply_windowed_note_notch(
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
            surgical_centers = exporter.parse_frequency_list(str(args.surgical_notch_centers_hz))
            tuned_vocals_audio = exporter.apply_surgical_notches(
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
            band_subtract_centers = exporter.parse_frequency_list(str(args.window_band_subtract_centers_hz))
            tuned_vocals_audio = exporter.apply_windowed_band_subtract(
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
            vocals_path = stems_dir / "vocals.flac"
            instrumental_path = stems_dir / "instrumental.flac"
            raw_ref_path = output_dir / "raw_ref.flac"
            exporter.sf.write(str(vocals_path), tuned_vocals_audio, vocals_sr, subtype="PCM_24")
            exporter.sf.write(str(instrumental_path), instrumental_audio, instrumental_sr, subtype="PCM_24")
            exporter.sf.write(str(raw_ref_path), raw_audio, raw_sr, subtype="PCM_24")
            persisted_runtime_config_path = output_dir / "config_runtime.yaml"
            shutil.copy2(runtime_config_path, persisted_runtime_config_path)
            manifest_path = exporter.write_full_track_stem_manifest(
                output_dir=output_dir,
                source_audio_path=source_audio_path,
                vocals_flac_path=vocals_path,
                instrumental_flac_path=instrumental_path,
                raw_ref_flac_path=raw_ref_path,
                repo_path=repo_path,
                model_path=model_path,
                config_template_path=config_path,
                runtime_config_path=persisted_runtime_config_path,
                device_id=int(args.device_id),
                num_overlap=int(args.num_overlap),
                sample_rate=int(raw_sr),
                frames=int(raw_audio.shape[0]),
                elapsed_sec=time.monotonic() - started,
                postprocess=exporter.build_postprocess_manifest(args, surgical_centers, band_subtract_centers),
                model_id=str(args.manifest_model_id),
                model_display_name=str(args.manifest_model_display_name),
                stem_provenance=resolved_stems["stem_provenance"],
                residual_companion_materialized=bool(resolved_stems["residual_companion_materialized"]),
                cuda_preflight=worker_response.get("runtime_contract") or {},
            )
        return {
            "status": "ok",
            "stems_dir": str(stems_dir),
            "output_files": [str(vocals_path), str(instrumental_path)],
            "manifest_path": str(manifest_path),
            "runtime_contract": worker_response.get("runtime_contract") or {},
            "external_runtime_subprocess_used": False,
            "per_request_external_runtime_subprocess_used": False,
            "persistent_worker_process_used": True,
            "model_load_count": int(worker_response.get("model_load_count") or 0),
            "worker_request_count": int(worker_response.get("worker_request_count") or 0),
            "worker_response": worker_response,
            "worker_elapsed_sec": round(time.monotonic() - started, 6),
        }


_RESIDENT_EXECUTORS: dict[str, ResidentSeparatorExecutor] = {}
_AUDIO_SEPARATOR_WORKERS: dict[str, AudioSeparatorWorkerProcess] = {}
_MELBAND_WORKERS: dict[str, MelBandWorkerProcess] = {}


def _is_safe_stem_name(stem: str) -> bool:
    return bool(stem) and stem == Path(stem).name and "/" not in stem and "\\" not in stem


def _command_value(parts: Sequence[str], flag: str, *, required: bool = False, default: str = "") -> str:
    try:
        index = list(parts).index(flag)
    except ValueError:
        if required:
            raise ValueError(f"missing_{flag.lstrip('-').replace('-', '_')}")
        return default
    value_index = index + 1
    if value_index >= len(parts) or str(parts[value_index]).startswith("--"):
        if required:
            raise ValueError(f"missing_{flag.lstrip('-').replace('-', '_')}")
        return default
    return str(parts[value_index])


def _is_audio_separator_exporter_command_part(part: str) -> bool:
    normalized = str(part or "").replace("\\", "/")
    return normalized == AUDIO_SEPARATOR_EXPORTER or normalized == AUDIO_SEPARATOR_EXPORTER_SUFFIX


def _is_melband_exporter_command_part(part: str) -> bool:
    normalized = str(part or "").replace("\\", "/")
    return normalized == MELBAND_EXPORTER or normalized == MELBAND_EXPORTER_SUFFIX


def _exporter_args_after(parts: Sequence[str], exporter_suffix: str) -> list[str]:
    normalized_suffix = exporter_suffix.replace("\\", "/")
    for index, part in enumerate(parts):
        normalized = str(part or "").replace("\\", "/")
        if normalized == normalized_suffix or normalized.endswith("/" + normalized_suffix):
            return [str(value) for value in parts[index + 1 :]]
    raise ValueError("resident_exporter_command_missing")


def _audio_separator_worker_key(*, audio_separator_python: Path, request: dict[str, Any]) -> str:
    payload = {
        "audio_separator_python": str(audio_separator_python),
        "model_filename": str(request["model_filename"]),
        "model_file_dir": str(request["model_file_dir"]),
        "output_format": str(request["output_format"]),
        "normalization_threshold": float(request["normalization_threshold"]),
        "mdx_overlap": float(request["mdx_overlap"]),
        "mdx_batch_size": int(request["mdx_batch_size"]),
        "enable_denoise": bool(request["enable_denoise"]),
        "require_cuda_provider": bool(request["require_cuda_provider"]),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _get_audio_separator_worker(*, audio_separator_python: Path, request: dict[str, Any]) -> AudioSeparatorWorkerProcess:
    worker_key = _audio_separator_worker_key(audio_separator_python=audio_separator_python, request=request)
    worker = _AUDIO_SEPARATOR_WORKERS.get(worker_key)
    if worker is None:
        worker = AudioSeparatorWorkerProcess(audio_separator_python=audio_separator_python, worker_key=worker_key)
        _AUDIO_SEPARATOR_WORKERS[worker_key] = worker
    return worker


def _melband_worker_key(*, request: dict[str, Any]) -> str:
    payload = {
        "route_id": str(request["route_id"]),
        "repo_path": str(request["repo_path"]),
        "model_path": str(request["model_path"]),
        "config_path": str(request["config_path"]),
        "device_id": int(request["device_id"]),
        "num_overlap": int(request["num_overlap"]),
        "model_type": str(request["model_type"]),
        "require_gpu": bool(request["require_gpu"]),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _get_melband_worker(*, request: dict[str, Any]) -> MelBandWorkerProcess:
    worker_key = _melband_worker_key(request=request)
    worker = _MELBAND_WORKERS.get(worker_key)
    if worker is None:
        worker = MelBandWorkerProcess(worker_key=worker_key)
        _MELBAND_WORKERS[worker_key] = worker
    return worker


def get_resident_separator_executor(route_id: str, *, prefer_resident: bool = False) -> ResidentSeparatorExecutor | None:
    normalized_route = str(route_id or "").strip()
    if normalized_route == RESIDENT_FIXTURE_ROUTE_ID:
        executor_cls: type[ResidentSeparatorExecutor] = ResidentSeparatorExecutor
    elif prefer_resident and normalized_route in AUDIO_SEPARATOR_RESIDENT_ROUTE_IDS:
        executor_cls = AudioSeparatorResidentExecutor
    elif prefer_resident and normalized_route in MELBAND_RESIDENT_ROUTE_IDS:
        executor_cls = MelBandResidentExecutor
    else:
        return None
    executor = _RESIDENT_EXECUTORS.get(normalized_route)
    if executor is None:
        executor = executor_cls(normalized_route)
        _RESIDENT_EXECUTORS[normalized_route] = executor
    return executor


def resident_executor_health() -> dict[str, Any]:
    return {
        "schema": "resident_separator_executor_health.v1",
        "supported_routes": [
            RESIDENT_FIXTURE_ROUTE_ID,
            *sorted(AUDIO_SEPARATOR_RESIDENT_ROUTE_IDS),
            *sorted(MELBAND_RESIDENT_ROUTE_IDS),
        ],
        "loaded_routes": {route_id: executor.snapshot() for route_id, executor in _RESIDENT_EXECUTORS.items()},
        "can_accept_delivery": False,
    }


def reset_resident_separator_executors_for_tests() -> None:
    for worker in list(_AUDIO_SEPARATOR_WORKERS.values()):
        worker.stop()
    _AUDIO_SEPARATOR_WORKERS.clear()
    for worker in list(_MELBAND_WORKERS.values()):
        worker.stop()
    _MELBAND_WORKERS.clear()
    _RESIDENT_EXECUTORS.clear()
