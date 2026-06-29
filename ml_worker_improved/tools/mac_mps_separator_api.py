"""Minimal native macOS MPS separator API for diagnostic Demucs jobs.

This server is intentionally diagnostic-only. It can start Mac MPS Demucs
exports and write artifacts into the configured shared artifact root, but it
cannot mark delivery/listen output as accepted.
"""

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse


DEFAULT_MODEL_NAME = "htdemucs_6s"


@dataclass(frozen=True)
class MacMpsApiConfig:
    project_root: Path
    artifacts_root: Path
    python_bin: Path


def build_config(*, project_root: Path, artifacts_root: Path, python_bin: Path) -> MacMpsApiConfig:
    return MacMpsApiConfig(
        project_root=Path(project_root).expanduser().resolve(),
        artifacts_root=Path(artifacts_root).expanduser().resolve(),
        python_bin=Path(python_bin).expanduser(),
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


def build_job_manifest(
    *,
    job_id: str,
    status: str,
    source_audio_path: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    returncode: Optional[int] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "schema": "mac_mps_separator_api_job.v1",
        "job_id": str(job_id),
        "status": str(status),
        "model_name": str(model_name),
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
        }
    return {"stem_manifest_exists": True, "stems": stems}


class MacMpsJobRunner:
    def __init__(self, config: MacMpsApiConfig):
        self.config = config
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        source_audio_path = Path(str(payload["source_audio_path"])).expanduser().resolve()
        model_name = str(payload.get("model_name") or DEFAULT_MODEL_NAME)
        job_id = make_job_id(str(source_audio_path))
        output_subdir = str(payload.get("output_subdir") or f"mac_api/{job_id}")
        output_dir = resolve_job_output_dir(self.config, output_subdir)
        manifest = build_job_manifest(
            job_id=job_id,
            status="queued",
            source_audio_path=source_audio_path,
            output_dir=output_dir,
            model_name=model_name,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "manifest.json", manifest)
        with self._lock:
            self._jobs[job_id] = manifest
        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, source_audio_path, output_dir, model_name),
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

    def _run_job(self, job_id: str, source_audio_path: Path, output_dir: Path, model_name: str) -> None:
        progress_path = output_dir / "progress.jsonl"
        run_dir = output_dir / "run"
        run_log_path = output_dir / "run.log"
        manifest = build_job_manifest(
            job_id=job_id,
            status="running",
            source_audio_path=source_audio_path,
            output_dir=output_dir,
            model_name=model_name,
        )
        _write_json(output_dir / "manifest.json", manifest)
        with self._lock:
            self._jobs[job_id] = manifest
        _append_progress(progress_path, "preflight", "start")
        if not source_audio_path.exists():
            error = f"source audio is missing: {source_audio_path}"
            self._finish(job_id, source_audio_path, output_dir, model_name, status="failed", returncode=20, error=error)
            _append_progress(progress_path, "preflight", "missing_source", error=error)
            return
        _append_progress(progress_path, "preflight", "end")
        command, env = build_demucs_export_command(
            config=self.config,
            source_audio_path=source_audio_path,
            output_dir=run_dir,
            model_name=model_name,
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
            status=status,
            returncode=int(completed.returncode),
            error=None if completed.returncode == 0 else f"export failed rc={completed.returncode}",
        )
        _append_progress(progress_path, "summarize", "end")

    def _finish(
        self,
        job_id: str,
        source_audio_path: Path,
        output_dir: Path,
        model_name: str,
        *,
        status: str,
        returncode: int,
        error: Optional[str],
    ) -> None:
        manifest = build_job_manifest(
            job_id=job_id,
            status=status,
            source_audio_path=source_audio_path,
            output_dir=output_dir,
            model_name=model_name,
            returncode=returncode,
            error=error,
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
        if parsed.path != "/jobs":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
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
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, required=True)
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
        ),
        host=str(args.host),
        port=int(args.port),
    )


if __name__ == "__main__":
    main()
