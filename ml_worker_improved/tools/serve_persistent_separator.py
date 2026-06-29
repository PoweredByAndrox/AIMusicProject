"""Serve existing separator route commands behind a persistent HTTP process."""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.resident_separator_executor import get_resident_separator_executor, resident_executor_health


_ROUTE_CACHE: dict[str, dict[str, Any]] = {}
GLOBAL_PROGRESS_PATH = Path("/var/log/separator/progress.jsonl")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def _emit_progress(output_dir: Path | None, payload: dict[str, Any]) -> None:
    event = dict(payload)
    event.setdefault("schema", "persistent_separator_progress.v1")
    if output_dir is not None:
        _append_jsonl(output_dir / "server_progress.jsonl", event)
    _append_jsonl(GLOBAL_PROGRESS_PATH, event)


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        process.terminate()
    deadline = time.monotonic() + 5.0
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.2)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except OSError:
            process.kill()


def _copy_stems(source_dir: Path, output_dir: Path, stems: list[str]) -> list[str]:
    missing: list[str] = []
    target_stems_dir = output_dir / "stems"
    target_stems_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        source = source_dir / f"{stem}.flac"
        if not source.is_file():
            missing.append(stem)
            continue
        shutil.copy2(source, target_stems_dir / source.name)
    return missing


def _cache_blocked_route(cache_key: str, response: dict[str, Any]) -> None:
    if not cache_key:
        return
    _ROUTE_CACHE[cache_key] = {
        "status": "blocked",
        "route_id": str(response.get("route_id") or ""),
        "reason": str(response.get("reason") or "separator_command_failed"),
        "returncode": response.get("returncode"),
        "resident_backend_used": bool(response.get("resident_backend_used", False)),
        "resident_capable": bool(response.get("resident_capable", False)),
        "resident_executor": response.get("resident_executor"),
    }


def _handle_separate(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    started = time.monotonic()
    route_id = str(payload.get("route_id") or "")
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    output_dir = Path(output_dir_raw)
    stems = [str(stem) for stem in payload.get("stems") or []]
    cache_key = str(payload.get("cache_key") or "")
    command = [str(part) for part in payload.get("command") or []]
    prefer_resident = bool(payload.get("prefer_resident"))
    resident_executor = get_resident_separator_executor(route_id, prefer_resident=prefer_resident)
    require_gpu = bool(payload.get("require_gpu"))
    device = str(payload.get("device") or "").lower()
    timeout_sec = int(payload.get("timeout_sec") or 300)
    if require_gpu and device != "cuda":
        return 400, {
            "status": "blocked",
            "reason": "cpu_fallback_forbidden",
            "delivery_allowed": False,
        }
    if not output_dir_raw:
        return 400, {"status": "blocked", "reason": "missing_output_dir", "delivery_allowed": False}
    if not command and resident_executor is None:
        return 400, {"status": "blocked", "reason": "missing_command", "delivery_allowed": False}
    _emit_progress(
        output_dir,
        {
            "event": "request_start",
            "route_id": route_id,
            "batch_id": str(payload.get("batch_id") or ""),
            "cache_key": cache_key,
            "stems": stems,
            "device": device,
            "require_gpu": require_gpu,
            "prefer_resident": prefer_resident,
            "output_dir": str(output_dir),
        },
    )
    if cache_key and cache_key in _ROUTE_CACHE:
        cached_entry = _ROUTE_CACHE[cache_key]
        if cached_entry.get("status") == "blocked":
            manifest = {
                "schema": "persistent_separator_response.v1",
                "status": "blocked",
                "cache_hit": True,
                "cache_key": cache_key,
                "route_id": str(payload.get("route_id") or ""),
                "output_dir": str(output_dir),
                "stems_dir": str(output_dir / "stems"),
                "returncode": cached_entry.get("returncode"),
                "reason": f"cached_{cached_entry.get('reason') or 'separator_command_failed'}",
                "elapsed_sec": round(time.monotonic() - started, 6),
                "resident_backend_used": bool(cached_entry.get("resident_backend_used", False)),
                "resident_capable": bool(cached_entry.get("resident_capable", False)),
                "resident_executor": cached_entry.get("resident_executor"),
                "delivery_allowed": False,
            }
            _write_json(output_dir / "persistent_separator.json", manifest)
            _emit_progress(
                output_dir,
                {
                    "event": "cache_failure_hit",
                    "route_id": route_id,
                    "batch_id": str(payload.get("batch_id") or ""),
                    "cache_key": cache_key,
                    "reason": manifest["reason"],
                    "elapsed_sec": manifest["elapsed_sec"],
                },
            )
            return 500, manifest
        cached_stems_dir = Path(str(cached_entry.get("stems_dir") or ""))
        missing = _copy_stems(cached_stems_dir, output_dir, stems)
        if not missing:
            manifest = {
                "schema": "persistent_separator_response.v1",
                "status": "cached",
                "cache_hit": True,
                "cache_key": cache_key,
                "route_id": str(payload.get("route_id") or ""),
                "output_dir": str(output_dir),
                "stems_dir": str(output_dir / "stems"),
                "elapsed_sec": round(time.monotonic() - started, 6),
                "resident_backend_used": bool(cached_entry.get("resident_backend_used", False)),
                "resident_capable": bool(cached_entry.get("resident_capable", False)),
                "resident_executor": cached_entry.get("resident_executor"),
                "delivery_allowed": False,
            }
            _write_json(output_dir / "persistent_separator.json", manifest)
            _emit_progress(
                output_dir,
                {
                    "event": "cache_hit",
                    "route_id": route_id,
                    "batch_id": str(payload.get("batch_id") or ""),
                    "cache_key": cache_key,
                    "elapsed_sec": manifest["elapsed_sec"],
                },
            )
            return 200, manifest
    if resident_executor is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _emit_progress(
            output_dir,
            {
                "event": "resident_executor_start",
                "route_id": route_id,
                "batch_id": str(payload.get("batch_id") or ""),
                "cache_key": cache_key,
                "executor_id": resident_executor.executor_id,
                "stems": stems,
                "device": device,
                "require_gpu": require_gpu,
            },
        )
        resident_response = resident_executor.separate(
            source_audio_path=str(payload.get("source_audio_path") or ""),
            output_dir=output_dir,
            stems=stems,
            device=device,
            require_gpu=require_gpu,
            command=command,
            timeout_sec=timeout_sec,
        )
        status = str(resident_response.get("status") or "blocked")
        resident_response.update(
            {
                "schema": "persistent_separator_response.v1",
                "status": status,
                "cache_hit": False,
                "cache_key": cache_key,
                "route_id": route_id,
                "output_dir": str(output_dir),
                "elapsed_sec": round(time.monotonic() - started, 6),
                "delivery_allowed": False,
                "can_accept_delivery": False,
            }
        )
        _write_json(output_dir / "persistent_separator.json", resident_response)
        _emit_progress(
            output_dir,
            {
                "event": "resident_executor_end",
                "route_id": route_id,
                "batch_id": str(payload.get("batch_id") or ""),
                "cache_key": cache_key,
                "executor_id": resident_executor.executor_id,
                "status": status,
                "reason": resident_response.get("reason"),
                "elapsed_sec": resident_response["elapsed_sec"],
            },
        )
        if status == "ok" and cache_key:
            _ROUTE_CACHE[cache_key] = {
                "status": "ok",
                "route_id": route_id,
                "output_dir": str(output_dir),
                "stems_dir": str(output_dir / "stems"),
                "resident_backend_used": True,
                "resident_capable": True,
                "resident_executor": resident_response.get("resident_executor"),
            }
            return 200, resident_response
        _cache_blocked_route(cache_key, resident_response)
        return 500, resident_response
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if require_gpu:
        env["REQUIRE_CUDA"] = "1"
        env.setdefault("ML_TORCH_DEVICE", "cuda")
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    _emit_progress(
        output_dir,
        {
            "event": "command_start",
            "route_id": route_id,
            "batch_id": str(payload.get("batch_id") or ""),
            "cache_key": cache_key,
            "command": command,
            "timeout_sec": timeout_sec,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        },
    )
    timed_out = False
    with stdout_path.open("ab") as stdout_handle, stderr_path.open("ab") as stderr_handle:
        process = subprocess.Popen(
            command,
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=env,
            cwd="/app",
            start_new_session=True,
        )
        deadline = time.monotonic() + float(timeout_sec)
        next_heartbeat = time.monotonic() + 5.0
        while process.poll() is None:
            now = time.monotonic()
            if now >= deadline:
                timed_out = True
                _emit_progress(
                    output_dir,
                    {
                        "event": "command_timeout",
                        "route_id": route_id,
                        "batch_id": str(payload.get("batch_id") or ""),
                        "cache_key": cache_key,
                        "elapsed_sec": round(now - started, 6),
                    },
                )
                _terminate_process_group(process)
                break
            if now >= next_heartbeat:
                _emit_progress(
                    output_dir,
                    {
                        "event": "command_heartbeat",
                        "route_id": route_id,
                        "batch_id": str(payload.get("batch_id") or ""),
                        "cache_key": cache_key,
                        "elapsed_sec": round(now - started, 6),
                    },
                )
                next_heartbeat = now + 5.0
            time.sleep(0.5)
        returncode = process.poll()
    if timed_out:
        response = {
            "schema": "persistent_separator_response.v1",
            "status": "blocked",
            "cache_hit": False,
            "cache_key": cache_key,
            "route_id": str(payload.get("route_id") or ""),
            "output_dir": str(output_dir),
            "stems_dir": str(output_dir / "stems"),
            "returncode": returncode,
            "reason": "separator_command_timeout",
            "elapsed_sec": round(time.monotonic() - started, 6),
            "resident_backend_used": False,
            "resident_capable": False,
            "delivery_allowed": False,
        }
        _write_json(output_dir / "persistent_separator.json", response)
        _cache_blocked_route(cache_key, response)
        return 500, response
    route_stems_dir = output_dir / "stems"
    missing = [stem for stem in stems if not (route_stems_dir / f"{stem}.flac").is_file()]
    status = "ok" if returncode == 0 and not missing else "blocked"
    response = {
        "schema": "persistent_separator_response.v1",
        "status": status,
        "cache_hit": False,
        "cache_key": cache_key,
        "route_id": str(payload.get("route_id") or ""),
        "output_dir": str(output_dir),
        "stems_dir": str(route_stems_dir),
        "returncode": returncode,
        "missing_stems": missing,
        "elapsed_sec": round(time.monotonic() - started, 6),
        "resident_backend_used": False,
        "resident_capable": False,
        "delivery_allowed": False,
    }
    _write_json(output_dir / "persistent_separator.json", response)
    _emit_progress(
        output_dir,
        {
            "event": "command_end",
            "route_id": route_id,
            "batch_id": str(payload.get("batch_id") or ""),
            "cache_key": cache_key,
            "returncode": returncode,
            "status": status,
            "missing_stems": missing,
            "elapsed_sec": response["elapsed_sec"],
        },
    )
    if status == "ok" and cache_key:
        _ROUTE_CACHE[cache_key] = {
            "status": "ok",
            "route_id": str(payload.get("route_id") or ""),
            "output_dir": str(output_dir),
            "stems_dir": str(route_stems_dir),
            "resident_backend_used": False,
            "resident_capable": False,
        }
        return 200, response
    response["reason"] = "separator_command_failed" if returncode else "missing_required_stems"
    _cache_blocked_route(cache_key, response)
    return 500, response


class Handler(BaseHTTPRequestHandler):
    server_version = "PersistentSeparator/0.1"

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("[separator-server] " + fmt % args + "\n")

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
        if self.path.rstrip("/") == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "schema": "persistent_separator_health.v1",
                    "cache_entries": len(_ROUTE_CACHE),
                    "gpu_required_default": os.environ.get("REQUIRE_CUDA") == "1",
                    "resident_executors": resident_executor_health(),
                },
            )
            return
        self._send_json(404, {"status": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
        if self.path.rstrip("/") != "/separate":
            self._send_json(404, {"status": "not_found"})
            return
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("payload_must_be_object")
            status, response = _handle_separate(payload)
            self._send_json(status, response)
        except Exception as exc:  # fail closed and surface infra evidence
            self._send_json(
                500,
                {
                    "status": "blocked",
                    "reason": "persistent_separator_server_error",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "delivery_allowed": False,
                },
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8791)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"event": "persistent_separator_start", "host": args.host, "port": args.port}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
