"""Persistent single-model Mac MPS worker for the exact Large-Inst checkpoint.

The worker owns one model instance and serializes requests through a Unix
socket. It never falls back to CPU and never marks generated audio as accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import re
import socket
import socketserver
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from mac_windows_mount_preflight import preflight_mac_windows_project_mount  # noqa: E402


SCHEMA = "mac_mps_large_inst_worker.v1"
REQUEST_SCHEMA = "mac_mps_large_inst_request.v1"
RESPONSE_SCHEMA = "mac_mps_large_inst_response.v1"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
DEFAULT_SOCKET_READ_TIMEOUT_SEC = 30.0


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def require_mps_environment(environment: Mapping[str, str] | None = None) -> None:
    env = os.environ if environment is None else environment
    if env.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK=0 is required")
    if env.get("AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC") != "1":
        raise RuntimeError("AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC=1 is required")


@dataclass(frozen=True)
class LargeInstWorkerConfig:
    socket_path: Path
    project_root: Path
    artifacts_root: Path
    state_dir: Path
    windows_project_root: str
    runtime_repo: Path
    model_implementation_path: Path
    config_path: Path
    checkpoint_path: Path
    model_id: str = "bs_roformer_large_inst"

    def validated(self) -> "LargeInstWorkerConfig":
        project_root = self.project_root.expanduser().resolve()
        artifacts_root = self.artifacts_root.expanduser().resolve()
        state_dir = self.state_dir.expanduser().resolve()
        if not _is_relative_to(artifacts_root, project_root):
            raise ValueError("artifacts_root must be inside project_root")
        if not _is_relative_to(state_dir, artifacts_root):
            raise ValueError("state_dir must be inside artifacts_root")
        if len(self.socket_path.name) > 64:
            raise ValueError("socket filename exceeds 64 characters")
        for path in (
            self.runtime_repo,
            self.model_implementation_path,
            self.config_path,
            self.checkpoint_path,
        ):
            if not path.expanduser().resolve().exists():
                raise FileNotFoundError(path)
        return LargeInstWorkerConfig(
            socket_path=self.socket_path.expanduser(),
            project_root=project_root,
            artifacts_root=artifacts_root,
            state_dir=state_dir,
            windows_project_root=self.windows_project_root,
            runtime_repo=self.runtime_repo.expanduser().resolve(),
            model_implementation_path=self.model_implementation_path.expanduser().resolve(),
            config_path=self.config_path.expanduser().resolve(),
            checkpoint_path=self.checkpoint_path.expanduser().resolve(),
            model_id=self.model_id,
        )


def validate_separate_request(
    payload: Mapping[str, Any],
    *,
    config: LargeInstWorkerConfig,
) -> tuple[str, Path, Path, dict[str, Any]]:
    request_id = str(payload.get("request_id") or "")
    if not REQUEST_ID_PATTERN.fullmatch(request_id):
        raise ValueError("request_id must be a 1-32 character Windows-safe alias")
    source_raw = payload.get("source_audio_path")
    output_raw = payload.get("output_dir")
    if not isinstance(source_raw, str) or not source_raw.strip():
        raise ValueError("source_audio_path is required")
    if not isinstance(output_raw, str) or not output_raw.strip():
        raise ValueError("output_dir is required")
    source_path = Path(source_raw).expanduser().resolve()
    output_dir = Path(output_raw).expanduser().resolve()
    source_scope = str(payload.get("source_scope") or "mounted_project")
    output_scope = str(payload.get("output_scope") or "mounted_artifact")
    temporary_root = Path(tempfile.gettempdir()).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_scope == "disposable_scratch":
        if not _is_relative_to(source_path, temporary_root):
            raise ValueError("disposable scratch source must be inside the system temporary root")
    elif source_scope != "mounted_project":
        raise ValueError(f"unsupported source_scope: {source_scope}")
    elif not _is_relative_to(source_path, config.project_root):
        raise ValueError("source_audio_path must be inside project_root")
    if output_scope == "disposable_scratch":
        if not _is_relative_to(output_dir, temporary_root):
            raise ValueError("disposable scratch output must be inside the system temporary root")
        relative_output = output_dir.relative_to(temporary_root)
    elif output_scope == "mounted_artifact":
        if not _is_relative_to(output_dir, config.artifacts_root):
            raise ValueError("output_dir must be inside artifacts_root")
        relative_output = output_dir.relative_to(config.artifacts_root)
    else:
        raise ValueError(f"unsupported output_scope: {output_scope}")
    if any(len(part) > 64 for part in relative_output.parts):
        raise ValueError("output_dir contains a path segment longer than 64 characters")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output_dir is not empty: {output_dir}")
    if output_scope == "disposable_scratch":
        output_contract = {
            "schema": "mac_mps_worker_output_scope.v1",
            "preflight_status": "passed",
            "output_scope": "disposable_scratch",
            "output_path": str(output_dir),
            "can_write_route_artifacts": False,
            "cleanup_required": True,
            "delivery_allowed": False,
        }
    else:
        output_contract = preflight_mac_windows_project_mount(
            mac_windows_project_mount=config.project_root,
            output_path=output_dir,
            windows_project_root=config.windows_project_root,
        )
        if output_contract.get("preflight_status") != "passed":
            raise RuntimeError(
                str(output_contract.get("blocked_reason") or "mac_windows_project_mount_preflight_failed")
            )
        output_contract["output_scope"] = "mounted_artifact"
    output_contract["source_scope"] = source_scope
    return request_id, source_path, output_dir, output_contract


class LargeInstModelRuntime:
    def __init__(self, config: LargeInstWorkerConfig) -> None:
        self.config = config.validated()
        self.model: Any = None
        self.model_config: Any = None
        self.device: Any = None
        self.run_folder: Any = None
        self.model_type = "bs_roformer"
        self.target_instrument = "instrument"
        self.model_load_count = 0
        self.request_count = 0
        self.success_count = 0
        self.model_load_elapsed_sec = 0.0
        self.poisoned_reason: str | None = None
        self._torch: Any = None
        self._original_cuda_autocast: Any = None
        self._original_stft: Any = None
        self._original_istft: Any = None
        self._precision_contract: dict[str, Any] = {}
        self._fft_guard_state: dict[str, int] = {}
        self._started_at = time.time()
        self._progress_path = self.config.state_dir / "progress.jsonl"
        self._state_path = self.config.state_dir / "worker.json"
        self.model_fingerprint = {
            "model_id": self.config.model_id,
            "config_sha256": _sha256(self.config.config_path),
            "checkpoint_sha256": _sha256(self.config.checkpoint_path),
            "implementation_sha256": _sha256(self.config.model_implementation_path),
        }

    def _memory(self) -> dict[str, int | None]:
        if self._torch is None:
            return {
                "current_allocated_bytes": None,
                "driver_allocated_bytes": None,
                "recommended_max_bytes": None,
            }
        mps = self._torch.mps
        values: dict[str, int | None] = {}
        for output_key, function_name in (
            ("current_allocated_bytes", "current_allocated_memory"),
            ("driver_allocated_bytes", "driver_allocated_memory"),
            ("recommended_max_bytes", "recommended_max_memory"),
        ):
            function = getattr(mps, function_name, None)
            values[output_key] = int(function()) if callable(function) else None
        return values

    def _state_manifest(self, status: str) -> dict[str, Any]:
        preflight = preflight_mac_windows_project_mount(
            mac_windows_project_mount=self.config.project_root,
            output_path=self.config.state_dir,
            windows_project_root=self.config.windows_project_root,
        )
        return {
            "schema": SCHEMA,
            "status": status,
            "model_id": self.config.model_id,
            "device": "mps",
            "cpu_fallback_allowed": False,
            "model_load_count": self.model_load_count,
            "model_load_elapsed_sec": round(self.model_load_elapsed_sec, 6),
            "request_count": self.request_count,
            "success_count": self.success_count,
            "poisoned_reason": self.poisoned_reason,
            "socket_path": str(self.config.socket_path),
            "state_dir": str(self.config.state_dir),
            "model": {
                "runtime_repo": str(self.config.runtime_repo),
                "implementation_path": str(self.config.model_implementation_path),
                "config_path": str(self.config.config_path),
                "checkpoint_path": str(self.config.checkpoint_path),
                "fingerprint": self.model_fingerprint,
                "strict_checkpoint_load": True,
            },
            "precision_contract": self._precision_contract,
            "fft_guard_state": self._fft_guard_state,
            "mps_memory": self._memory(),
            "mount_preflight": preflight,
            "delivery_allowed": False,
            "presentation_status": "diagnostic_resident_worker_state_only",
        }

    def _write_state(self, status: str) -> None:
        _write_json(self._state_path, self._state_manifest(status))

    def load(self) -> None:
        require_mps_environment()
        if platform.system() != "Darwin":
            raise RuntimeError("Large-Inst MPS worker requires macOS")
        mount_preflight = preflight_mac_windows_project_mount(
            mac_windows_project_mount=self.config.project_root,
            output_path=self.config.state_dir,
            windows_project_root=self.config.windows_project_root,
        )
        if mount_preflight.get("preflight_status") != "passed":
            raise RuntimeError(str(mount_preflight.get("blocked_reason") or "mount_preflight_failed"))
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self._progress_path.write_text("", encoding="utf-8")
        _append_jsonl(
            self._progress_path,
            {
                "phase": "model_load",
                "status": "start",
                "time_unix": time.time(),
                "expected_acceleration_mode": "native_macos_mps_resident_inference",
                "cpu_fallback_allowed": False,
            },
        )
        started = time.perf_counter()
        import torch

        from run_bs_roformer_custom_impl import (
            config_uses_amp,
            install_bs_roformer_implementation,
            install_mps_autocast_compatibility,
            install_mps_fft_cpu_retry_guard,
            require_mps,
        )

        self._torch = torch
        self._original_cuda_autocast = torch.cuda.amp.autocast
        self._original_stft = torch.stft
        self._original_istft = torch.istft
        require_mps()
        self._precision_contract = install_mps_autocast_compatibility()
        self._precision_contract["enabled_by_config"] = config_uses_amp(self.config.config_path)
        self._fft_guard_state = install_mps_fft_cpu_retry_guard()
        implementation_module = install_bs_roformer_implementation(
            self.config.runtime_repo,
            self.config.model_implementation_path,
        )
        inference = importlib.import_module("inference")
        from utils.model_utils import load_start_checkpoint, prefer_target_instrument
        from utils.settings import get_model_from_config

        model, model_config = get_model_from_config(self.model_type, str(self.config.config_path))
        if "model_type" in model_config.training:
            self.model_type = str(model_config.training.model_type)
        checkpoint = torch.load(self.config.checkpoint_path, weights_only=False, map_location="cpu")
        checkpoint_args = SimpleNamespace(
            start_check_point=str(self.config.checkpoint_path),
            model_type=self.model_type,
            lora_checkpoint_loralib=None,
        )
        load_start_checkpoint(checkpoint_args, model, checkpoint, type_="inference")
        self.device = torch.device("mps")
        self.model = model.to(self.device)
        self.model.eval()
        self.model_config = model_config
        self.run_folder = inference.run_folder
        instruments = list(prefer_target_instrument(model_config))
        if len(instruments) != 1:
            raise RuntimeError(f"Expected one Large-Inst target instrument, got {instruments}")
        self.target_instrument = str(instruments[0])
        self.model_load_count = 1
        self.model_load_elapsed_sec = time.perf_counter() - started
        self._precision_contract.update(
            {
                "implementation_module": implementation_module,
                "mode": "mps_autocast_float16"
                if self._precision_contract.get("enabled_by_config")
                else "mps_float32",
                "mps_fft_cpu_retry_guard": True,
            }
        )
        self._write_state("ready")
        _append_jsonl(
            self._progress_path,
            {
                "phase": "model_load",
                "status": "end",
                "time_unix": time.time(),
                "elapsed_sec": round(self.model_load_elapsed_sec, 6),
                "model_load_count": self.model_load_count,
                "target_instrument": self.target_instrument,
                "mps_memory": self._memory(),
            },
        )

    def health(self) -> dict[str, Any]:
        status = "poisoned" if self.poisoned_reason else "ready"
        return {
            "schema": RESPONSE_SCHEMA,
            "status": status,
            "model_id": self.config.model_id,
            "device": "mps",
            "model_load_count": self.model_load_count,
            "model_load_elapsed_sec": round(self.model_load_elapsed_sec, 6),
            "request_count": self.request_count,
            "success_count": self.success_count,
            "poisoned_reason": self.poisoned_reason,
            "mps_memory": self._memory(),
            "model_fingerprint": self.model_fingerprint,
            "delivery_allowed": False,
        }

    def release_cache(self) -> dict[str, Any]:
        if self._torch is None or self.model is None:
            raise RuntimeError("worker_model_not_loaded")
        if self.poisoned_reason:
            response = self._poisoned_response({})
            response["action"] = "release_cache"
            return response
        before = self._memory()
        self._torch.mps.synchronize()
        self._torch.mps.empty_cache()
        after = self._memory()
        _append_jsonl(
            self._progress_path,
            {
                "phase": "release_mps_cache",
                "status": "end",
                "time_unix": time.time(),
                "memory_before": before,
                "memory_after": after,
                "model_load_count": self.model_load_count,
            },
        )
        self._write_state("ready")
        return {
            "schema": RESPONSE_SCHEMA,
            "status": "ready",
            "action": "release_cache",
            "device": "mps",
            "model_load_count": self.model_load_count,
            "request_count": self.request_count,
            "mps_memory_before": before,
            "mps_memory_after": after,
            "delivery_allowed": False,
        }

    def _request_args(self, *, input_dir: Path, output_dir: Path, use_tta: bool) -> SimpleNamespace:
        return SimpleNamespace(
            input_folder=str(input_dir),
            store_dir=str(output_dir),
            model_type=self.model_type,
            start_check_point=str(self.config.checkpoint_path),
            device_ids=[0],
            force_cpu=False,
            disable_detailed_pbar=True,
            filename_template="{instr}",
            use_tta=use_tta,
            extract_instrumental=False,
            pcm_type="PCM_24",
            draw_spectro=0,
        )

    def _poisoned_response(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema": RESPONSE_SCHEMA,
            "status": "failed",
            "request_id": str(payload.get("request_id") or ""),
            "reason": f"worker_poisoned_restart_required: {self.poisoned_reason}",
            "restart_required": True,
            "model_load_count": self.model_load_count,
            "request_count": self.request_count,
            "delivery_allowed": False,
            "presentation_status": "diagnostic_failed_resident_request_only",
        }

    def separate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self.model is None or self.run_folder is None:
            raise RuntimeError("worker_model_not_loaded")
        if self.poisoned_reason:
            return self._poisoned_response(payload)
        expected_fingerprint = payload.get("expected_model_fingerprint")
        if not isinstance(expected_fingerprint, Mapping):
            raise ValueError("expected_model_fingerprint is required")
        for key, actual_value in self.model_fingerprint.items():
            if expected_fingerprint.get(key) != actual_value:
                raise ValueError(
                    f"model_fingerprint_mismatch:{key}:"
                    f"expected={expected_fingerprint.get(key)!r}:actual={actual_value!r}"
                )
        request_id, source_path, output_dir, output_contract = validate_separate_request(
            payload,
            config=self.config,
        )
        use_tta = bool(payload.get("use_tta", False))
        output_dir.mkdir(parents=True, exist_ok=True)
        progress_path = output_dir / "progress.jsonl"
        progress_path.write_text("", encoding="utf-8")
        self.request_count += 1
        started = time.perf_counter()
        memory_before = self._memory()
        _append_jsonl(
            progress_path,
            {
                "phase": "resident_inference",
                "status": "start",
                "time_unix": time.time(),
                "request_id": request_id,
                "request_count": self.request_count,
                "model_load_count": self.model_load_count,
                "device": "mps",
                "cpu_fallback_allowed": False,
            },
        )
        try:
            with tempfile.TemporaryDirectory(prefix="large_inst_input_") as input_name:
                input_dir = Path(input_name)
                source_link = input_dir / f"in{source_path.suffix.lower()}"
                source_link.symlink_to(source_path)
                self.run_folder(
                    self.model,
                    self._request_args(input_dir=input_dir, output_dir=output_dir, use_tta=use_tta),
                    self.model_config,
                    self.device,
                    verbose=True,
                )
            matches = sorted(
                path
                for path in output_dir.rglob(f"{self.target_instrument}.*")
                if path.is_file() and path.suffix.lower() in {".wav", ".flac"}
            )
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one {self.target_instrument} output, found {len(matches)} in {output_dir}"
                )
        except Exception as exc:
            self.poisoned_reason = f"{type(exc).__name__}: {exc}"
            elapsed_sec = time.perf_counter() - started
            failed = {
                "schema": RESPONSE_SCHEMA,
                "status": "failed",
                "request_id": request_id,
                "reason": self.poisoned_reason,
                "restart_required": True,
                "elapsed_sec": round(elapsed_sec, 6),
                "model_load_count": self.model_load_count,
                "request_count": self.request_count,
                "delivery_allowed": False,
                "presentation_status": "diagnostic_failed_resident_request_only",
            }
            _write_json(output_dir / "worker.json", failed)
            _append_jsonl(
                progress_path,
                {
                    "phase": "resident_inference",
                    "status": "failed",
                    "time_unix": time.time(),
                    "reason": self.poisoned_reason,
                    "restart_required": True,
                },
            )
            self._write_state("poisoned")
            return failed

        output_path = matches[0]
        elapsed_sec = time.perf_counter() - started
        self.success_count += 1
        response = {
            "schema": RESPONSE_SCHEMA,
            "status": "ok",
            "request_id": request_id,
            "model_id": self.config.model_id,
            "model_fingerprint": self.model_fingerprint,
            "device": "mps",
            "cpu_fallback_allowed": False,
            "source_audio_path": str(source_path),
            "source_audio_sha256": _sha256(source_path),
            "output_path": str(output_path),
            "output_sha256": _sha256(output_path),
            "target_instrument": self.target_instrument,
            "elapsed_sec": round(elapsed_sec, 6),
            "model_load_count": self.model_load_count,
            "model_load_elapsed_sec": round(self.model_load_elapsed_sec, 6),
            "request_count": self.request_count,
            "success_count": self.success_count,
            "mps_memory_before": memory_before,
            "mps_memory_after": self._memory(),
            "output_contract": output_contract,
            "mount_preflight": (
                output_contract if output_contract.get("output_scope") == "mounted_artifact" else None
            ),
            "delivery_allowed": False,
            "delivery_status": "candidate_only",
            "presentation_status": "blocked_until_quality_audit",
            "requires_post_judge": True,
        }
        _write_json(output_dir / "worker.json", response)
        _append_jsonl(
            progress_path,
            {
                "phase": "resident_inference",
                "status": "end",
                "time_unix": time.time(),
                "elapsed_sec": round(elapsed_sec, 6),
                "output_path": str(output_path),
                "output_sha256": response["output_sha256"],
                "model_load_count": self.model_load_count,
                "request_count": self.request_count,
                "mps_memory": response["mps_memory_after"],
            },
        )
        self._write_state("ready")
        return response

    def close(self) -> None:
        if self._torch is not None:
            if self._original_cuda_autocast is not None:
                self._torch.cuda.amp.autocast = self._original_cuda_autocast
            if self._original_stft is not None:
                self._torch.stft = self._original_stft
            if self._original_istft is not None:
                self._torch.istft = self._original_istft
        self._write_state("stopped")


class LargeInstWorkerRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        self.request.settimeout(float(self.server.read_timeout_sec))  # type: ignore[attr-defined]
        try:
            raw = self.rfile.readline(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                response = {"schema": RESPONSE_SCHEMA, "status": "blocked", "reason": "request_too_large"}
            else:
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("request must be a JSON object")
                response = self.server.dispatch(payload)  # type: ignore[attr-defined]
        except socket.timeout:
            response = {
                "schema": RESPONSE_SCHEMA,
                "status": "failed",
                "reason": "request_read_timeout",
                "restart_required": False,
                "delivery_allowed": False,
            }
        except Exception as exc:
            response = {
                "schema": RESPONSE_SCHEMA,
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "delivery_allowed": False,
            }
        try:
            self.wfile.write((json.dumps(response, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            return


class LargeInstWorkerServer(socketserver.UnixStreamServer):
    def __init__(
        self,
        socket_path: Path,
        runtime: LargeInstModelRuntime,
        *,
        read_timeout_sec: float = DEFAULT_SOCKET_READ_TIMEOUT_SEC,
    ) -> None:
        if float(read_timeout_sec) <= 0:
            raise ValueError("read_timeout_sec must be positive")
        self.socket_path = socket_path
        self.runtime = runtime
        self.read_timeout_sec = float(read_timeout_sec)
        self._prepare_socket_path()
        super().__init__(str(socket_path), LargeInstWorkerRequestHandler)
        self.socket_path.chmod(0o600)

    def _prepare_socket_path(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.socket_path.exists():
            return
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.2)
        try:
            probe.connect(str(self.socket_path))
        except OSError:
            self.socket_path.unlink()
        else:
            raise RuntimeError(f"worker socket is already active: {self.socket_path}")
        finally:
            probe.close()

    def dispatch(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action") or "")
        if action == "health":
            return self.runtime.health()
        if action == "separate":
            return self.runtime.separate(payload)
        if action == "release_cache":
            return self.runtime.release_cache()
        if action == "shutdown":
            threading.Thread(target=self.shutdown, daemon=True).start()
            return {
                "schema": RESPONSE_SCHEMA,
                "status": "stopping",
                "delivery_allowed": False,
            }
        return {
            "schema": RESPONSE_SCHEMA,
            "status": "blocked",
            "reason": "unsupported_action",
            "delivery_allowed": False,
        }

    def server_close(self) -> None:
        super().server_close()
        if self.socket_path.exists():
            self.socket_path.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket-path", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--windows-project-root", default=r"D:\AIMusicProject")
    parser.add_argument("--runtime-repo", type=Path, required=True)
    parser.add_argument("--model-implementation-path", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--model-id", default="bs_roformer_large_inst")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = LargeInstWorkerConfig(
        socket_path=args.socket_path,
        project_root=args.project_root,
        artifacts_root=args.artifacts_root,
        state_dir=args.state_dir,
        windows_project_root=str(args.windows_project_root),
        runtime_repo=args.runtime_repo,
        model_implementation_path=args.model_implementation_path,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        model_id=str(args.model_id),
    ).validated()
    runtime = LargeInstModelRuntime(config)
    runtime.load()
    server = LargeInstWorkerServer(config.socket_path, runtime)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
