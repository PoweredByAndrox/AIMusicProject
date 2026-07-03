"""Minimal native macOS MPS separator API for diagnostic Demucs jobs.

This server is intentionally diagnostic-only. It can start Mac MPS Demucs
exports and write artifacts into the configured shared artifact root, but it
cannot mark delivery/listen output as accepted.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from mac_windows_mount_preflight import (  # noqa: E402
    CANONICAL_MAC_ARTIFACT_ROOT,
    CANONICAL_MAC_WINDOWS_PROJECT_MOUNT,
    DEFAULT_WINDOWS_PROJECT_ROOT,
    preflight_mac_windows_project_mount,
)


DEFAULT_MODEL_NAME = "htdemucs_6s"
DEMUX_ROUTE_IDS = {"htdemucs_6s", "demucs_htdemucs_6s"}
MDX23C_ROUTE_ID = "mdx23c_vocals"
MELBAND_ROUTE_ID = "melband_roformer_vocals"
BECRUILY_INSTRUMENTAL_ROUTE_ID = "becruily_instrumental_vocals"
SCNET_ROUTE_ID = "scnet_xl_ihf_musdb18"
SUPPORTED_ROUTE_IDS = DEMUX_ROUTE_IDS | {
    MDX23C_ROUTE_ID,
    MELBAND_ROUTE_ID,
    BECRUILY_INSTRUMENTAL_ROUTE_ID,
    SCNET_ROUTE_ID,
}
SCNET_MPS_TIMEOUT_SEC = "900"


@dataclass(frozen=True)
class MacMpsApiConfig:
    project_root: Path
    artifacts_root: Path
    python_bin: Path
    mac_windows_project_mount: Optional[Path] = None
    windows_project_root: str = DEFAULT_WINDOWS_PROJECT_ROOT


def build_config(
    *,
    project_root: Path,
    artifacts_root: Path,
    python_bin: Path,
    mac_windows_project_mount: Optional[Path] = None,
    windows_project_root: str = DEFAULT_WINDOWS_PROJECT_ROOT,
) -> MacMpsApiConfig:
    resolved_mount = Path(mac_windows_project_mount or CANONICAL_MAC_WINDOWS_PROJECT_MOUNT).expanduser().resolve()
    resolved_artifacts_root = Path(artifacts_root).expanduser().resolve()
    mount_preflight = preflight_mac_windows_project_mount(
        mac_windows_project_mount=resolved_mount,
        output_path=resolved_artifacts_root,
        windows_project_root=windows_project_root,
    )
    if mount_preflight.get("preflight_status") != "passed":
        reason = str(mount_preflight.get("blocked_reason") or "mac_windows_project_mount_preflight_failed")
        raise ValueError(f"mac_mps_api_artifacts_root_preflight_failed:{reason}")
    try:
        relative_artifacts_root = resolved_artifacts_root.relative_to(resolved_mount)
    except ValueError as exc:
        raise ValueError("mac_mps_api_artifacts_root_outside_windows_project_mount") from exc
    if not relative_artifacts_root.parts or relative_artifacts_root.parts[0] != "artifacts":
        raise ValueError("mac_mps_api_artifacts_root_must_be_under_windows_artifacts")
    return MacMpsApiConfig(
        project_root=Path(project_root).expanduser().resolve(),
        artifacts_root=resolved_artifacts_root,
        python_bin=Path(python_bin).expanduser(),
        mac_windows_project_mount=resolved_mount,
        windows_project_root=str(windows_project_root),
    )


def make_job_id(seed: Optional[str] = None) -> str:
    material = f"{seed or ''}:{time.time_ns()}".encode("utf-8", errors="ignore")
    return "m" + hashlib.sha256(material).hexdigest()[:10]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_short_output_segments(path: Path, root: Path) -> None:
    relative = path.relative_to(root)
    for part in relative.parts:
        if part in {"", ".", ".."}:
            raise ValueError(f"unsafe output path segment: {part!r}")
        if len(part) > 64:
            raise ValueError(f"output path segment is too long: {part!r}")


def resolve_job_output_dir(config: MacMpsApiConfig, requested_output: Any) -> Path:
    requested = Path(requested_output)
    root = config.artifacts_root.resolve()
    if requested.is_absolute():
        output_dir = requested.resolve()
    else:
        output_dir = (root / requested).resolve()
    if not _is_relative_to(output_dir, root):
        raise ValueError(f"output directory is outside artifacts root: {output_dir}")
    _validate_short_output_segments(output_dir, root)
    return output_dir


def build_demucs_export_command(
    *,
    config: MacMpsApiConfig,
    source_audio_path: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[list[str], dict[str, str]]:
    tool_path = config.project_root / "ml_worker_improved" / "tools" / "export_hq_stems_full.py"
    command = [
        str(config.python_bin),
        str(tool_path),
        str(source_audio_path),
        str(output_dir),
        "--model-name",
        str(model_name),
        "--device",
        "mps",
        "--no-melody-stem",
    ]
    env = dict(os.environ)
    env["AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC"] = "1"
    return command, env


def _diagnostic_env() -> dict[str, str]:
    env = dict(os.environ)
    env["AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC"] = "1"
    return env


def _vendor_path(config: MacMpsApiConfig, *parts: str) -> Path:
    return config.project_root / "ml_worker_improved" / "vendor" / Path(*parts)


def build_separator_export_command(
    *,
    config: MacMpsApiConfig,
    route_id: str,
    source_audio_path: Path,
    output_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    normalized_route = str(route_id or "").strip()
    if normalized_route not in SUPPORTED_ROUTE_IDS:
        raise ValueError(f"unsupported Mac MPS route: {normalized_route}")
    if normalized_route in DEMUX_ROUTE_IDS:
        return build_demucs_export_command(
            config=config,
            source_audio_path=source_audio_path,
            output_dir=output_dir,
            model_name=DEFAULT_MODEL_NAME,
        )

    tools_dir = config.project_root / "ml_worker_improved" / "tools"
    repo_path = _vendor_path(config, "music_source_separation_training")
    command = [str(config.python_bin)]
    if normalized_route == MDX23C_ROUTE_ID:
        command.extend(
            [
                str(tools_dir / "export_mdx23c_vocal_full.py"),
                str(source_audio_path),
                str(output_dir),
                "--repo-path",
                str(repo_path),
                "--config-path",
                str(_vendor_path(config, "mdx23c_vocals_official", "config_vocals_mdx23c.yaml")),
                "--model-path",
                str(_vendor_path(config, "mdx23c_vocals_official", "model_vocals_mdx23c_sdr_10.17.ckpt")),
                "--device-id",
                "0",
            ]
        )
    elif normalized_route == MELBAND_ROUTE_ID:
        command.extend(
            [
                str(tools_dir / "export_melband_vocal_full.py"),
                str(source_audio_path),
                str(output_dir),
                "--repo-path",
                str(_vendor_path(config, "mel_band_roformer_vocal_model")),
                "--config-path",
                str(
                    _vendor_path(
                        config,
                        "mel_band_roformer_vocal_model",
                        "configs",
                        "config_vocals_mel_band_roformer.yaml",
                    )
                ),
                "--model-path",
                str(_vendor_path(config, "mel_band_roformer_vocal_model_weights", "MelBandRoformer.ckpt")),
                "--device-id",
                "0",
            ]
        )
    elif normalized_route == BECRUILY_INSTRUMENTAL_ROUTE_ID:
        command.extend(
            [
                str(tools_dir / "export_melband_vocal_full.py"),
                str(source_audio_path),
                str(output_dir),
                "--repo-path",
                str(_vendor_path(config, "mel_band_roformer_vocal_model")),
                "--config-path",
                str(_vendor_path(config, "becruily_instrumental", "config_instrumental_becruily.yaml")),
                "--model-path",
                str(_vendor_path(config, "becruily_instrumental", "mel_band_roformer_instrumental_becruily.ckpt")),
                "--manifest-model-id",
                BECRUILY_INSTRUMENTAL_ROUTE_ID,
                "--manifest-model-display-name",
                "Becruily MelBand RoFormer instrumental checkpoint",
                "--device-id",
                "0",
            ]
        )
    elif normalized_route == SCNET_ROUTE_ID:
        command.extend(
            [
                str(tools_dir / "export_scnet_full.py"),
                str(source_audio_path),
                str(output_dir),
                "--repo-path",
                str(repo_path),
                "--config-path",
                str(_vendor_path(config, "scnet_xl_ihf", "config_musdb18_scnet_xl_more_wide_v5.yaml")),
                "--model-path",
                str(_vendor_path(config, "scnet_xl_ihf", "model_scnet_ep_36_sdr_10.0891.ckpt")),
                "--device-id",
                "0",
                "--timeout-sec",
                SCNET_MPS_TIMEOUT_SEC,
            ]
        )
    return command, _diagnostic_env()


def build_job_manifest(
    *,
    job_id: str,
    status: str,
    source_audio_path: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    route_id: str = DEFAULT_MODEL_NAME,
    returncode: Optional[int] = None,
    error: Optional[str] = None,
    mount_preflight: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    manifest = {
        "schema": "mac_mps_separator_api_job.v1",
        "job_id": str(job_id),
        "status": str(status),
        "model_name": str(model_name),
        "route_id": str(route_id),
        "source_audio_path": str(source_audio_path),
        "output_dir": str(output_dir),
        "run_dir": str(output_dir / "run"),
        "returncode": returncode,
        "error": error,
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "diagnostic_mps_api_only",
        "created_at_unix": time.time(),
    }
    if mount_preflight is not None:
        manifest["mount_preflight"] = mount_preflight
        manifest["mac_windows_project_mount"] = str(mount_preflight.get("mac_windows_project_mount") or "")
        manifest["windows_output_path"] = str(mount_preflight.get("windows_output_path") or "")
    return manifest


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_progress(path: Path, phase: str, status: str, **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time_unix": time.time(), "phase": phase, "status": status, **extra}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _summarize_stems(run_dir: Path) -> dict[str, Any]:
    stem_manifest_path = run_dir / "stem_manifest.json"
    if not stem_manifest_path.exists():
        return {"stem_manifest_exists": False, "stems": {}}
    try:
        manifest = json.loads(stem_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"stem_manifest_exists": True, "stem_manifest_error": str(exc), "stems": {}}
    stems: dict[str, Any] = {}
    for name, info in dict(manifest.get("stems") or {}).items():
        stem_path = Path(str(info.get("path") or ""))
        stems[str(name)] = {
            "path": str(stem_path),
            "exists": stem_path.exists(),
            "bytes": stem_path.stat().st_size if stem_path.exists() else 0,
            "duration_sec": info.get("duration_sec"),
            "rms": info.get("rms"),
            "peak": info.get("peak"),
            "role": info.get("role"),
            "source": info.get("source"),
            "model_output": info.get("model_output"),
            "residual_companion": info.get("residual_companion"),
        }
    return {"stem_manifest_exists": True, "stems": stems}


def _persistent_separator_response(
    *,
    output_dir: Path,
    route_id: str,
    stems: list[str],
    status: str,
    returncode: int,
    reason: str,
    elapsed_sec: float,
    requested_device: str,
    require_gpu: bool,
    mount_preflight: Optional[dict[str, Any]] = None,
    missing_stems: Optional[list[str]] = None,
    copied_stems: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    response = {
        "schema": "persistent_separator_response.v1",
        "status": str(status),
        "cache_hit": False,
        "cache_key": "",
        "route_id": str(route_id),
        "output_dir": str(output_dir),
        "stems_dir": str(output_dir / "stems"),
        "returncode": int(returncode),
        "reason": str(reason or ""),
        "missing_stems": list(missing_stems or []),
        "copied_stems": dict(copied_stems or {}),
        "requested_stems": list(stems),
        "requested_device": str(requested_device or ""),
        "actual_device": "mps",
        "require_gpu": bool(require_gpu),
        "elapsed_sec": float(elapsed_sec),
        "mac_mps_backend_used": True,
        "resident_backend_used": False,
        "resident_capable": False,
        "delivery_allowed": False,
        "can_accept_delivery": False,
        "presentation_status": "diagnostic_mps_api_only",
    }
    if mount_preflight is not None:
        response["mount_preflight"] = mount_preflight
        response["mac_windows_project_mount"] = str(mount_preflight.get("mac_windows_project_mount") or "")
        response["windows_output_path"] = str(mount_preflight.get("windows_output_path") or "")
    return response


def _copy_stems_for_persistent_response(run_dir: Path, output_dir: Path, stems: list[str]) -> tuple[dict[str, str], list[str]]:
    summary = _summarize_stems(run_dir)
    available = summary.get("stems") if isinstance(summary.get("stems"), dict) else {}
    stems_dir = output_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    missing: list[str] = []
    for stem in stems:
        info = available.get(stem) if isinstance(available, dict) else None
        source_path = Path(str(info.get("path") or "")) if isinstance(info, dict) else run_dir / "audio" / f"{stem}.flac"
        if not source_path.is_file():
            missing.append(stem)
            continue
        target_path = stems_dir / f"{stem}.flac"
        shutil.copy2(source_path, target_path)
        copied[stem] = str(target_path)
    return copied, missing


class MacMpsJobRunner:
    def __init__(self, config: MacMpsApiConfig):
        self.config = config
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        source_audio_path = Path(str(payload["source_audio_path"])).expanduser().resolve()
        route_id = str(payload.get("route_id") or payload.get("model_name") or DEFAULT_MODEL_NAME)
        model_name = str(payload.get("model_name") or route_id)
        job_id = make_job_id(str(source_audio_path))
        output_subdir = str(payload.get("output_subdir") or f"mac_api/{job_id}")
        output_dir = resolve_job_output_dir(self.config, output_subdir)
        mount_preflight = preflight_mac_windows_project_mount(
            mac_windows_project_mount=self.config.mac_windows_project_mount,
            output_path=output_dir,
            windows_project_root=self.config.windows_project_root,
        )
        if mount_preflight.get("preflight_status") != "passed":
            manifest = build_job_manifest(
                job_id=job_id,
                status="blocked",
                source_audio_path=source_audio_path,
                output_dir=output_dir,
                model_name=model_name,
                route_id=route_id,
                returncode=78,
                error=str(mount_preflight.get("blocked_reason") or "mac_windows_project_mount_preflight_failed"),
                mount_preflight=mount_preflight,
            )
            manifest["blocked_reason"] = str(mount_preflight.get("blocked_reason") or "")
            with self._lock:
                self._jobs[job_id] = manifest
            return manifest
        manifest = build_job_manifest(
            job_id=job_id,
            status="queued",
            source_audio_path=source_audio_path,
            output_dir=output_dir,
            model_name=model_name,
            route_id=route_id,
            mount_preflight=mount_preflight,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "manifest.json", manifest)
        with self._lock:
            self._jobs[job_id] = manifest
        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, source_audio_path, output_dir, model_name, route_id, mount_preflight),
            daemon=True,
        )
        thread.start()
        return manifest

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            manifest = self._jobs.get(job_id)
        if manifest is not None:
            return manifest
        for manifest_path in self.config.artifacts_root.glob("mac_api/**/manifest.json"):
            try:
                candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if str(candidate.get("job_id") or "") == str(job_id):
                return candidate
        return None

    def separate(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        started_at = time.perf_counter()
        route_id = str(payload.get("route_id") or payload.get("model_name") or DEFAULT_MODEL_NAME)
        source_audio_path = Path(str(payload.get("source_audio_path") or "")).expanduser().resolve()
        output_dir_raw = str(payload.get("output_dir") or "").strip()
        output_dir = Path(output_dir_raw).expanduser().resolve() if output_dir_raw else self.config.artifacts_root / "mac_api" / "missing_output"
        stems = [str(stem) for stem in payload.get("stems") or []]
        requested_device = str(payload.get("device") or "").strip().lower()
        require_gpu = bool(payload.get("require_gpu", True))
        timeout_sec = int(payload.get("timeout_sec") or 0)
        if not output_dir_raw:
            return 400, _persistent_separator_response(
                output_dir=output_dir,
                route_id=route_id,
                stems=stems,
                status="blocked",
                returncode=64,
                reason="missing_output_dir",
                elapsed_sec=round(time.perf_counter() - started_at, 6),
                requested_device=requested_device,
                require_gpu=require_gpu,
            )
        mount_preflight = preflight_mac_windows_project_mount(
            mac_windows_project_mount=self.config.mac_windows_project_mount,
            output_path=output_dir,
            windows_project_root=self.config.windows_project_root,
        )
        if mount_preflight.get("preflight_status") != "passed":
            response = _persistent_separator_response(
                output_dir=output_dir,
                route_id=route_id,
                stems=stems,
                status="blocked",
                returncode=78,
                reason=str(mount_preflight.get("blocked_reason") or "mac_windows_project_mount_preflight_failed"),
                elapsed_sec=round(time.perf_counter() - started_at, 6),
                requested_device=requested_device,
                require_gpu=require_gpu,
                mount_preflight=mount_preflight,
            )
            _write_json(output_dir / "persistent_separator.json", response)
            return 400, response
        output_dir.mkdir(parents=True, exist_ok=True)
        progress_path = output_dir / "progress.jsonl"
        _append_progress(
            progress_path,
            "persistent_separate",
            "start",
            route_id=route_id,
            requested_device=requested_device,
            actual_device="mps",
            require_gpu=require_gpu,
            stems=stems,
        )
        if not source_audio_path.exists():
            response = _persistent_separator_response(
                output_dir=output_dir,
                route_id=route_id,
                stems=stems,
                status="blocked",
                returncode=20,
                reason=f"source audio is missing: {source_audio_path}",
                elapsed_sec=round(time.perf_counter() - started_at, 6),
                requested_device=requested_device,
                require_gpu=require_gpu,
                mount_preflight=mount_preflight,
            )
            _write_json(output_dir / "persistent_separator.json", response)
            _append_progress(progress_path, "persistent_separate", "missing_source", reason=response["reason"])
            return 400, response
        run_dir = output_dir / "run"
        try:
            command, env = build_separator_export_command(
                config=self.config,
                route_id=route_id,
                source_audio_path=source_audio_path,
                output_dir=run_dir,
            )
        except ValueError as exc:
            response = _persistent_separator_response(
                output_dir=output_dir,
                route_id=route_id,
                stems=stems,
                status="blocked",
                returncode=65,
                reason=str(exc),
                elapsed_sec=round(time.perf_counter() - started_at, 6),
                requested_device=requested_device,
                require_gpu=require_gpu,
                mount_preflight=mount_preflight,
            )
            _write_json(output_dir / "persistent_separator.json", response)
            _append_progress(progress_path, "persistent_separate", "unsupported_route", reason=response["reason"])
            return 400, response
        run_log_path = output_dir / "run.log"
        _append_progress(progress_path, "mps_export", "start", command=command, timeout_sec=timeout_sec)
        with run_log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command,
                cwd=str(self.config.project_root / "ml_worker_improved"),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=timeout_sec if timeout_sec > 0 else None,
            )
        copied, missing = _copy_stems_for_persistent_response(run_dir, output_dir, stems)
        status = "ok" if completed.returncode == 0 and not missing else "blocked"
        reason = "" if status == "ok" else ("missing_required_stems" if completed.returncode == 0 else f"export failed rc={completed.returncode}")
        response = _persistent_separator_response(
            output_dir=output_dir,
            route_id=route_id,
            stems=stems,
            status=status,
            returncode=int(completed.returncode),
            reason=reason,
            elapsed_sec=round(time.perf_counter() - started_at, 6),
            requested_device=requested_device,
            require_gpu=require_gpu,
            mount_preflight=mount_preflight,
            missing_stems=missing,
            copied_stems=copied,
        )
        _write_json(output_dir / "persistent_separator.json", response)
        _append_progress(
            progress_path,
            "persistent_separate",
            f"end_{status}",
            returncode=int(completed.returncode),
            missing_stems=missing,
            elapsed_sec=response["elapsed_sec"],
        )
        return (200 if status == "ok" else 500), response

    def _run_job(
        self,
        job_id: str,
        source_audio_path: Path,
        output_dir: Path,
        model_name: str,
        route_id: str,
        mount_preflight: dict[str, Any],
    ) -> None:
        progress_path = output_dir / "progress.jsonl"
        run_dir = output_dir / "run"
        run_log_path = output_dir / "run.log"
        manifest = build_job_manifest(
            job_id=job_id,
            status="running",
            source_audio_path=source_audio_path,
            output_dir=output_dir,
            model_name=model_name,
            route_id=route_id,
            mount_preflight=mount_preflight,
        )
        _write_json(output_dir / "manifest.json", manifest)
        with self._lock:
            self._jobs[job_id] = manifest
        _append_progress(progress_path, "preflight", "start")
        if not source_audio_path.exists():
            error = f"source audio is missing: {source_audio_path}"
            self._finish(
                job_id,
                source_audio_path,
                output_dir,
                model_name,
                route_id,
                status="failed",
                returncode=20,
                error=error,
                mount_preflight=mount_preflight,
            )
            _append_progress(progress_path, "preflight", "missing_source", error=error)
            return
        _append_progress(progress_path, "preflight", "end")
        command, env = build_separator_export_command(
            config=self.config,
            route_id=route_id,
            source_audio_path=source_audio_path,
            output_dir=run_dir,
        )
        _append_progress(progress_path, "mps_export", "start", command=command)
        started_at = time.perf_counter()
        with run_log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command,
                cwd=str(self.config.project_root / "ml_worker_improved"),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        elapsed_sec = round(time.perf_counter() - started_at, 3)
        _append_progress(
            progress_path,
            "mps_export",
            f"end_rc_{completed.returncode}",
            elapsed_sec=elapsed_sec,
        )
        status = "ok" if completed.returncode == 0 else "failed"
        self._finish(
            job_id,
            source_audio_path,
            output_dir,
            model_name,
            route_id,
            status=status,
            returncode=int(completed.returncode),
            error=None if completed.returncode == 0 else f"export failed rc={completed.returncode}",
            mount_preflight=mount_preflight,
        )
        _append_progress(progress_path, "summarize", "end")

    def _finish(
        self,
        job_id: str,
        source_audio_path: Path,
        output_dir: Path,
        model_name: str,
        route_id: str,
        *,
        status: str,
        returncode: int,
        error: Optional[str],
        mount_preflight: Optional[dict[str, Any]] = None,
    ) -> None:
        manifest = build_job_manifest(
            job_id=job_id,
            status=status,
            source_audio_path=source_audio_path,
            output_dir=output_dir,
            model_name=model_name,
            route_id=route_id,
            returncode=returncode,
            error=error,
            mount_preflight=mount_preflight,
        )
        manifest.update(_summarize_stems(output_dir / "run"))
        _write_json(output_dir / "manifest.json", manifest)
        with self._lock:
            self._jobs[job_id] = manifest


class MacMpsApiHandler(BaseHTTPRequestHandler):
    runner: MacMpsJobRunner

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib HTTP handler API
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(200, {"status": "ok", "schema": "mac_mps_separator_api_health.v1"})
            return
        if parsed.path.startswith("/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            manifest = self.runner.get_job(job_id)
            if manifest is None:
                self._send_json(404, {"error": "job not found", "job_id": job_id})
                return
            self._send_json(200, manifest)
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib HTTP handler API
        parsed = urlparse(self.path)
        if parsed.path not in {"/jobs", "/separate"}:
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if parsed.path == "/separate":
                status_code, manifest = self.runner.separate(payload)
                self._send_json(status_code, manifest)
                return
            manifest = self.runner.start_job(payload)
        except Exception as exc:  # pragma: no cover - exercised through integration/manual API
            self._send_json(400, {"error": str(exc)})
            return
        self._send_json(202, manifest)


def serve(config: MacMpsApiConfig, host: str, port: int) -> None:
    runner = MacMpsJobRunner(config)

    class BoundHandler(MacMpsApiHandler):
        pass

    BoundHandler.runner = runner
    server = ThreadingHTTPServer((host, int(port)), BoundHandler)
    print(f"[mac_mps_separator_api] listening host={host} port={port}", flush=True)
    server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run native macOS MPS diagnostic separator API.")
    parser.add_argument("--project-root", type=Path, default=Path(CANONICAL_MAC_WINDOWS_PROJECT_MOUNT))
    parser.add_argument("--artifacts-root", type=Path, default=Path(CANONICAL_MAC_ARTIFACT_ROOT))
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--mac-windows-project-mount", type=Path, default=Path(CANONICAL_MAC_WINDOWS_PROJECT_MOUNT))
    parser.add_argument("--windows-project-root", default=DEFAULT_WINDOWS_PROJECT_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    serve(
        build_config(
            project_root=args.project_root,
            artifacts_root=args.artifacts_root,
            python_bin=args.python_bin,
            mac_windows_project_mount=args.mac_windows_project_mount,
            windows_project_root=args.windows_project_root,
        ),
        host=str(args.host),
        port=int(args.port),
    )


if __name__ == "__main__":
    main()
