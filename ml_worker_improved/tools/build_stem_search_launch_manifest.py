"""Build a reproducible launch manifest for a filtered stem-search recipe."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.run_fully_automatic_stem_search import (
    DEFAULT_MIN_GPU_FREE_MIB,
    MODEL_SPLIT_V2_PROFILE,
    _append_jsonl,
    _collect_cuda_preflight,
    _parse_bool,
    _write_json,
)
from tools.stem_contamination_common import validate_no_artifact_rag_output


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _in_container_command(
    *,
    mix_path: Path,
    recipe_path: Path,
    run_output_dir: Path,
    max_candidates: int,
    require_gpu: bool,
    no_artifact_rag: bool,
    profile: str,
    min_gpu_free_mib: int,
    wait_for_gpu_free_sec: float,
    gpu_preflight_poll_sec: float,
) -> list[str]:
    command = [
        "python",
        "tools/run_fully_automatic_stem_search.py",
        "--mix",
        str(mix_path),
        "--out",
        str(run_output_dir),
        "--max-candidates",
        str(int(max_candidates)),
        "--require-gpu",
        "true" if require_gpu else "false",
        "--recipe",
        str(recipe_path),
        "--profile",
        profile,
        "--min-gpu-free-mib",
        str(int(min_gpu_free_mib)),
        "--wait-for-gpu-free-sec",
        str(float(wait_for_gpu_free_sec)),
        "--gpu-preflight-poll-sec",
        str(float(gpu_preflight_poll_sec)),
    ]
    if no_artifact_rag:
        command.append("--no-artifact-rag")
    return command


def _docker_compose_command_host(in_container_command: list[str]) -> list[str]:
    return [
        "docker",
        "compose",
        "run",
        "--rm",
        "--no-deps",
        "-T",
        "--volume",
        "D:/AIMusicProject:/repo",
        "-v",
        "/var/run/docker.sock:/var/run/docker.sock",
        "--workdir",
        "/repo/ml_worker_improved",
        "ml-worker",
        *in_container_command,
    ]


def _gpu_preflight_summary(preflight: Mapping[str, Any] | None) -> dict[str, Any]:
    source = dict(preflight or {})
    summary: dict[str, Any] = {
        "status": str(source.get("status") or "unknown"),
        "reason": str(source.get("reason") or ""),
        "require_gpu": bool(source.get("require_gpu", False)),
        "min_free_mib": int(source.get("min_free_mib") or 0),
    }
    for key in ("free_mib", "used_mib", "total_mib", "utilization_pct"):
        if source.get(key) is not None:
            summary[key] = int(source.get(key) or 0)
    if source.get("device_name"):
        summary["device_name"] = str(source["device_name"])
    return summary


def _run_text_command(command: list[str], *, timeout_sec: float = 10.0) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "reason": f"{command[0]}_cli_unavailable", "stdout": "", "stderr": "", "returncode": None}
    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": f"{command[0]}_failed", "stdout": "", "stderr": str(exc)[:500], "returncode": None}
    return {
        "available": completed.returncode == 0,
        "reason": "" if completed.returncode == 0 else f"{command[0]}_failed",
        "stdout": completed.stdout,
        "stderr": completed.stderr[:500],
        "returncode": completed.returncode,
    }


def _parse_gpu_process_rows(stdout: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) < 3:
            continue
        if parts[1] == "[Not Found]":
            continue
        rows.append({"pid": parts[0], "process_name": parts[1], "used_memory_mib": parts[2]})
    return rows


def _parse_docker_stats_rows(stdout: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 4 or not parts[0]:
            continue
        rows.append(
            {
                "name": parts[0],
                "cpu_percent": parts[1],
                "mem_usage": parts[2],
                "mem_percent": parts[3],
            }
        )
    return rows[:10]


class _UnixSocketHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str, *, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self.socket_path)
        self.sock = sock


def _docker_socket_api_get(path: str, *, socket_path: str = "/var/run/docker.sock", timeout_sec: float = 3.0) -> Any:
    if not Path(socket_path).exists():
        raise FileNotFoundError(socket_path)
    connection = _UnixSocketHTTPConnection(socket_path, timeout=timeout_sec)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        body = response.read().decode("utf-8", errors="replace")
        if response.status >= 400:
            raise RuntimeError(f"docker socket API {path} returned HTTP {response.status}: {body[:200]}")
        return json.loads(body) if body else {}
    finally:
        connection.close()


def _bytes_to_gib(value: int) -> str:
    return f"{float(value) / float(1024 ** 3):.3f}GiB"


def _container_name(container: Mapping[str, Any]) -> str:
    names = container.get("Names")
    if isinstance(names, list) and names:
        return str(names[0]).lstrip("/")
    return str(container.get("Name") or container.get("Id") or "").lstrip("/")


def _collect_docker_socket_stats(
    *,
    api_get: Any | None = None,
    socket_path: str = "/var/run/docker.sock",
    timeout_sec: float = 3.0,
) -> dict[str, Any]:
    getter = api_get or (lambda path: _docker_socket_api_get(path, socket_path=socket_path, timeout_sec=timeout_sec))
    try:
        containers = getter("/containers/json?all=0")
    except Exception as exc:
        return {
            "available": False,
            "reason": "docker_socket_unavailable",
            "source": "docker_engine_socket",
            "error": str(exc)[:500],
            "top_container_memory": [],
            "container_states": [],
        }
    if not isinstance(containers, list):
        return {
            "available": False,
            "reason": "docker_socket_unexpected_response",
            "source": "docker_engine_socket",
            "top_container_memory": [],
            "container_states": [],
        }

    memory_rows: list[dict[str, Any]] = []
    states: list[dict[str, str]] = []
    for container in containers[:30]:
        if not isinstance(container, Mapping):
            continue
        container_id = str(container.get("Id") or "")
        name = _container_name(container)
        states.append(
            {
                "name": name,
                "image": str(container.get("Image") or ""),
                "status": str(container.get("Status") or ""),
            }
        )
        if not container_id:
            continue
        try:
            stats = getter(f"/containers/{container_id}/stats?stream=false")
        except Exception:
            continue
        if not isinstance(stats, Mapping):
            continue
        memory_stats = stats.get("memory_stats") or {}
        if not isinstance(memory_stats, Mapping):
            continue
        usage = int(memory_stats.get("usage") or 0)
        limit = int(memory_stats.get("limit") or 0)
        mem_percent = (float(usage) / float(limit) * 100.0) if limit > 0 else 0.0
        memory_rows.append(
            {
                "name": name,
                "cpu_percent": "",
                "mem_usage": f"{_bytes_to_gib(usage)} / {_bytes_to_gib(limit)}" if limit > 0 else _bytes_to_gib(usage),
                "mem_percent": f"{mem_percent:.2f}%",
            }
        )

    memory_rows.sort(key=lambda row: float(str(row.get("mem_percent") or "0").rstrip("%") or 0.0), reverse=True)
    return {
        "available": True,
        "reason": "",
        "source": "docker_engine_socket",
        "top_container_memory": memory_rows[:10],
        "container_states": states[:50],
    }


def _collect_resource_blocker_snapshot() -> dict[str, Any]:
    gpu_apps = _run_text_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    docker_stats = _run_text_command(
        [
            "docker",
            "stats",
            "--no-stream",
            "--format",
            "{{.Name}}|{{.CPUPerc}}|{{.MemUsage}}|{{.MemPerc}}",
        ]
    )
    socket_stats = _collect_docker_socket_stats() if not docker_stats["available"] else {}
    docker_available = bool(docker_stats["available"] or socket_stats.get("available"))
    docker_reason = "" if docker_available else str(docker_stats["reason"] or socket_stats.get("reason") or "")
    top_container_memory = (
        _parse_docker_stats_rows(str(docker_stats["stdout"]))
        if docker_stats["available"]
        else list(socket_stats.get("top_container_memory") or [])
    )
    snapshot = {
        "schema": "stem_search_resource_blocker_snapshot.v1",
        "source": "best_effort_local_tools",
        "gpu_processes_available": bool(gpu_apps["available"]),
        "gpu_processes_reason": str(gpu_apps["reason"]),
        "gpu_processes": _parse_gpu_process_rows(str(gpu_apps["stdout"])) if gpu_apps["available"] else [],
        "docker_container_stats_available": docker_available,
        "docker_container_stats_reason": docker_reason,
        "docker_container_stats_source": "docker_cli" if docker_stats["available"] else str(socket_stats.get("source") or ""),
        "top_container_memory": top_container_memory,
    }
    if socket_stats.get("container_states"):
        snapshot["container_states"] = list(socket_stats.get("container_states") or [])
    if gpu_apps.get("stderr"):
        snapshot["gpu_processes_stderr"] = str(gpu_apps["stderr"])
    if docker_stats.get("stderr"):
        snapshot["docker_container_stats_stderr"] = str(docker_stats["stderr"])
    if socket_stats.get("error"):
        snapshot["docker_socket_error"] = str(socket_stats["error"])
    return snapshot


def _resource_blocker_summary(snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    source = dict(snapshot or {})
    containers = [dict(row) for row in source.get("top_container_memory") or [] if isinstance(row, Mapping)]
    gpu_processes = [dict(row) for row in source.get("gpu_processes") or [] if isinstance(row, Mapping)]
    return {
        "docker_container_stats_available": bool(source.get("docker_container_stats_available", False)),
        "docker_container_stats_reason": str(source.get("docker_container_stats_reason") or ""),
        "top_container_memory_names": [str(row.get("name") or "") for row in containers[:5] if row.get("name")],
        "gpu_process_count": len(gpu_processes),
        "gpu_process_names": [str(row.get("process_name") or "") for row in gpu_processes[:5] if row.get("process_name")],
    }


def _status_summary(
    *,
    gpu_preflight: Mapping[str, Any] | None,
    result_status: str,
    run_command_started: bool,
    launch_status: str = "",
    execution_status: str = "",
) -> dict[str, Any]:
    gpu_summary = _gpu_preflight_summary(gpu_preflight)
    blocked_reason = str(gpu_summary.get("reason") or execution_status or launch_status or "")
    return {
        "launch_status": launch_status,
        "execution_status": execution_status,
        "gpu_status": gpu_summary["status"],
        "blocked_reason": blocked_reason,
        "result_status": result_status,
        "run_command_started": bool(run_command_started),
    }


def _write_status_summary_md(output_dir: Path, manifest: Mapping[str, Any]) -> Path:
    gpu = dict(manifest.get("gpu_preflight_summary") or _gpu_preflight_summary(manifest.get("gpu_preflight")))
    status = dict(manifest.get("status_summary") or {})
    resource = dict(manifest.get("resource_blocker_summary") or _resource_blocker_summary(manifest.get("resource_blocker_snapshot")))
    result_label = str(status.get("result_status") or "unknown").replace("_", " ")
    top_names = list(resource.get("top_container_memory_names") or [])
    if resource.get("docker_container_stats_available"):
        docker_line = f"Top container memory: {', '.join(str(name) for name in top_names) if top_names else 'none'}"
    else:
        docker_line = f"Docker stats: unavailable ({resource.get('docker_container_stats_reason') or 'unknown'})"
    lines = [
        "# Stem Search Launch Status",
        "",
        f"GPU preflight: {gpu.get('status', 'unknown')}",
        (
            "GPU memory: "
            f"free_mib={gpu.get('free_mib', 'unknown')} "
            f"used_mib={gpu.get('used_mib', 'unknown')} "
            f"total_mib={gpu.get('total_mib', 'unknown')} "
            f"min_free_mib={gpu.get('min_free_mib', 'unknown')}"
        ),
        f"GPU reason: {gpu.get('reason', '')}",
        f"GPU processes observed: {resource.get('gpu_process_count', 0)}",
        docker_line,
        f"Command started: {str(bool(status.get('run_command_started'))).lower()}",
        f"Result: {result_label}",
        f"Delivery allowed: {str(bool(manifest.get('delivery_allowed'))).lower()}",
        f"Presentation status: {manifest.get('presentation_status', '')}",
    ]
    path = output_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_stem_search_launch_manifest(
    *,
    output_dir: Path,
    mix_path: Path,
    recipe_path: Path,
    run_output_dir: Path,
    max_candidates: int = 10,
    require_gpu: bool = True,
    min_gpu_free_mib: int = DEFAULT_MIN_GPU_FREE_MIB,
    wait_for_gpu_free_sec: float = 600.0,
    gpu_preflight_poll_sec: float = 10.0,
    profile: str = MODEL_SPLIT_V2_PROFILE,
    no_artifact_rag: bool = True,
    gpu_preflight: Mapping[str, Any] | None = None,
    resource_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if no_artifact_rag:
        validate_no_artifact_rag_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    started = time.monotonic()
    _append_jsonl(
        progress_path,
        {
            "event": "start",
            "elapsed_sec": 0.0,
            "recipe_path": str(recipe_path),
            "run_output_dir": str(run_output_dir),
            "gpu_inference_skipped": True,
            "reason": "diagnostic_launch_manifest_only",
        },
    )
    preflight = dict(
        gpu_preflight
        if gpu_preflight is not None
        else _collect_cuda_preflight(require_gpu=require_gpu, min_free_mib=min_gpu_free_mib)
    )
    resource_snapshot_source = "provided" if resource_snapshot is not None else "collector"
    _append_jsonl(
        progress_path,
        {
            "event": "resource_snapshot_start",
            "elapsed_sec": round(time.monotonic() - started, 6),
            "resource_snapshot_source": resource_snapshot_source,
            "gpu_inference_skipped": True,
        },
    )
    blocker_snapshot = dict(resource_snapshot if resource_snapshot is not None else _collect_resource_blocker_snapshot())
    _append_jsonl(
        progress_path,
        {
            "event": "resource_snapshot_end",
            "elapsed_sec": round(time.monotonic() - started, 6),
            "resource_snapshot_source": resource_snapshot_source,
            "resource_blocker_summary": _resource_blocker_summary(blocker_snapshot),
            "gpu_inference_skipped": True,
        },
    )
    gpu_ready = (not require_gpu) or preflight.get("status") in {"passed", "not_required", "not_checked"}
    in_container_command = _in_container_command(
        mix_path=mix_path,
        recipe_path=recipe_path,
        run_output_dir=run_output_dir,
        max_candidates=max_candidates,
        require_gpu=require_gpu,
        no_artifact_rag=no_artifact_rag,
        profile=profile,
        min_gpu_free_mib=min_gpu_free_mib,
        wait_for_gpu_free_sec=wait_for_gpu_free_sec,
        gpu_preflight_poll_sec=gpu_preflight_poll_sec,
    )
    manifest = {
        "schema": "stem_search_launch_manifest.v1",
        "output_dir": str(output_dir),
        "mix_path": str(mix_path),
        "recipe_path": str(recipe_path),
        "recipe_sha256": _sha256_file(recipe_path),
        "run_output_dir": str(run_output_dir),
        "profile": profile,
        "max_candidates": int(max_candidates),
        "require_gpu": bool(require_gpu),
        "min_gpu_free_mib": int(max(0, min_gpu_free_mib)),
        "wait_for_gpu_free_sec": float(max(0.0, wait_for_gpu_free_sec)),
        "gpu_preflight_poll_sec": float(max(0.1, gpu_preflight_poll_sec)),
        "gpu_preflight": preflight,
        "gpu_preflight_summary": _gpu_preflight_summary(preflight),
        "resource_blocker_snapshot": blocker_snapshot,
        "resource_blocker_summary": _resource_blocker_summary(blocker_snapshot),
        "gpu_ready": bool(gpu_ready),
        "launch_status": "ready_to_run" if gpu_ready else "blocked_gpu_preflight",
        "in_container_command": in_container_command,
        "docker_compose_command_host": _docker_compose_command_host(in_container_command),
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "diagnostic_launch_only",
        "listen_pack": {},
        "gpu_inference_skipped": True,
        "artifact_rag_write_allowed": not no_artifact_rag,
    }
    manifest["status_summary"] = _status_summary(
        gpu_preflight=preflight,
        result_status="not_started",
        run_command_started=False,
        launch_status=str(manifest["launch_status"]),
    )
    manifest["summary_md"] = str(output_dir / "summary.md")
    _write_json(output_dir / "manifest.json", manifest)
    _write_status_summary_md(output_dir, manifest)
    _append_jsonl(
        progress_path,
        {
            "event": "end",
            "elapsed_sec": round(time.monotonic() - started, 6),
            "launch_status": manifest["launch_status"],
            "gpu_ready": manifest["gpu_ready"],
            "gpu_inference_skipped": True,
            "gpu_preflight_summary": manifest["gpu_preflight_summary"],
            "status_summary": manifest["status_summary"],
            "resource_blocker_summary": manifest["resource_blocker_summary"],
        },
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mix", required=True, type=Path)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--run-out", required=True, type=Path)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--require-gpu", default="true")
    parser.add_argument("--profile", choices=("", MODEL_SPLIT_V2_PROFILE), default=MODEL_SPLIT_V2_PROFILE)
    parser.add_argument("--min-gpu-free-mib", type=int, default=DEFAULT_MIN_GPU_FREE_MIB)
    parser.add_argument("--wait-for-gpu-free-sec", type=float, default=600.0)
    parser.add_argument("--gpu-preflight-poll-sec", type=float, default=10.0)
    parser.add_argument("--no-artifact-rag", action="store_true")
    parser.add_argument(
        "--echo-progress",
        dest="echo_progress",
        action="store_true",
        default=True,
        help="Mirror progress.jsonl events to stderr during CLI runs",
    )
    parser.add_argument(
        "--no-echo-progress",
        dest="echo_progress",
        action="store_false",
        help="Write progress only to progress.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["STEM_SEARCH_ECHO_PROGRESS"] = "1" if args.echo_progress else "0"
    manifest = build_stem_search_launch_manifest(
        output_dir=args.out,
        mix_path=args.mix,
        recipe_path=args.recipe,
        run_output_dir=args.run_out,
        max_candidates=args.max_candidates,
        require_gpu=_parse_bool(args.require_gpu),
        min_gpu_free_mib=args.min_gpu_free_mib,
        wait_for_gpu_free_sec=args.wait_for_gpu_free_sec,
        gpu_preflight_poll_sec=args.gpu_preflight_poll_sec,
        profile=args.profile,
        no_artifact_rag=args.no_artifact_rag,
    )
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "manifest_path": str(Path(str(manifest["output_dir"])) / "manifest.json"),
                "launch_status": manifest["launch_status"],
                "gpu_ready": manifest["gpu_ready"],
                "gpu_preflight": manifest["gpu_preflight"],
                "run_output_dir": manifest["run_output_dir"],
                "delivery_allowed": manifest["delivery_allowed"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
