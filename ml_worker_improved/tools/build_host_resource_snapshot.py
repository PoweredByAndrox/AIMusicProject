"""Build a diagnostic host-side resource snapshot for blocked GPU launches."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

CommandRunner = Callable[[list[str], float], Mapping[str, Any]]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
    if os.environ.get("STEM_SEARCH_ECHO_PROGRESS", "0").lower() not in {"", "0", "false", "no"}:
        print(line, file=sys.stderr, flush=True)


def _validate_no_artifact_rag_output(output_dir: Path) -> None:
    resolved_parts = {part.lower() for part in output_dir.resolve().parts}
    if "artifact_rag" in resolved_parts:
        raise ValueError(f"host resource snapshot cannot write artifact_rag output: {output_dir}")


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
        if len(parts) < 3 or parts[1] == "[Not Found]":
            continue
        rows.append({"pid": parts[0], "process_name": parts[1], "used_memory_mib": parts[2]})
    return rows


def _parse_docker_stats_rows(stdout: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 4 or not parts[0]:
            continue
        rows.append({"name": parts[0], "cpu_percent": parts[1], "mem_usage": parts[2], "mem_percent": parts[3]})
    return rows[:10]


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


def _parse_gpu_summary(stdout: str, *, min_free_mib: int) -> dict[str, Any]:
    first_line = next((line for line in stdout.splitlines() if line.strip()), "")
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) < 5:
        return {
            "schema": "stem_search_cuda_preflight.v1",
            "status": "unknown",
            "reason": "gpu_query_unparseable",
            "require_gpu": True,
            "min_free_mib": int(min_free_mib),
        }
    name, total_mib, used_mib, free_mib, utilization_pct = parts[:5]
    free = int(float(free_mib))
    return {
        "schema": "stem_search_cuda_preflight.v1",
        "status": "passed" if free >= min_free_mib else "blocked",
        "reason": "" if free >= min_free_mib else "gpu_free_memory_too_low",
        "require_gpu": True,
        "device_name": name,
        "total_mib": int(float(total_mib)),
        "used_mib": int(float(used_mib)),
        "free_mib": free,
        "utilization_pct": int(float(utilization_pct)),
        "min_free_mib": int(min_free_mib),
    }


def _parse_container_states(stdout: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split("|", 2)]
        if len(parts) != 3 or not parts[0]:
            continue
        rows.append({"name": parts[0], "image": parts[1], "status": parts[2]})
    return rows[:50]


def _mem_percent_value(row: Mapping[str, Any]) -> float:
    raw = str(row.get("mem_percent") or "0").strip().rstrip("%")
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _run(command_runner: CommandRunner | None, command: list[str], timeout_sec: float) -> dict[str, Any]:
    if command_runner is not None:
        return dict(command_runner(command, timeout_sec))
    return dict(_run_text_command(command, timeout_sec=timeout_sec))


def build_host_resource_snapshot(
    *,
    output_dir: Path,
    no_artifact_rag: bool = True,
    min_gpu_free_mib: int = 4096,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    if no_artifact_rag:
        _validate_no_artifact_rag_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    started = time.monotonic()
    _append_jsonl(
        progress_path,
        {
            "event": "start",
            "elapsed_sec": 0.0,
            "gpu_inference_skipped": True,
            "reason": "diagnostic_host_resource_snapshot_only",
        },
    )

    gpu_query = _run(
        command_runner,
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        10.0,
    )
    gpu_apps = _run(
        command_runner,
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        10.0,
    )
    docker_stats = _run(
        command_runner,
        [
            "docker",
            "stats",
            "--no-stream",
            "--format",
            "{{.Name}}|{{.CPUPerc}}|{{.MemUsage}}|{{.MemPerc}}",
        ],
        20.0,
    )
    docker_ps = _run(
        command_runner,
        ["docker", "ps", "--format", "{{.Names}}|{{.Image}}|{{.Status}}"],
        10.0,
    )

    top_container_memory = (
        sorted(_parse_docker_stats_rows(str(docker_stats.get("stdout") or "")), key=_mem_percent_value, reverse=True)
        if docker_stats.get("available")
        else []
    )
    resource_snapshot = {
        "schema": "stem_search_resource_blocker_snapshot.v1",
        "source": "host_observability_only",
        "gpu_processes_available": bool(gpu_apps.get("available")),
        "gpu_processes_reason": str(gpu_apps.get("reason") or ""),
        "gpu_processes": _parse_gpu_process_rows(str(gpu_apps.get("stdout") or "")) if gpu_apps.get("available") else [],
        "docker_container_stats_available": bool(docker_stats.get("available")),
        "docker_container_stats_reason": str(docker_stats.get("reason") or ""),
        "top_container_memory": top_container_memory,
        "docker_ps_available": bool(docker_ps.get("available")),
        "docker_ps_reason": str(docker_ps.get("reason") or ""),
        "container_states": _parse_container_states(str(docker_ps.get("stdout") or "")) if docker_ps.get("available") else [],
    }
    gpu_summary = (
        _parse_gpu_summary(str(gpu_query.get("stdout") or ""), min_free_mib=min_gpu_free_mib)
        if gpu_query.get("available")
        else {
            "schema": "stem_search_cuda_preflight.v1",
            "status": "unknown",
            "reason": str(gpu_query.get("reason") or "nvidia_smi_unavailable"),
            "require_gpu": True,
            "min_free_mib": int(min_gpu_free_mib),
        }
    )
    if gpu_query.get("stderr"):
        resource_snapshot["gpu_query_stderr"] = str(gpu_query.get("stderr"))[:500]
    if gpu_apps.get("stderr"):
        resource_snapshot["gpu_processes_stderr"] = str(gpu_apps.get("stderr"))[:500]
    if docker_stats.get("stderr"):
        resource_snapshot["docker_container_stats_stderr"] = str(docker_stats.get("stderr"))[:500]
    if docker_ps.get("stderr"):
        resource_snapshot["docker_ps_stderr"] = str(docker_ps.get("stderr"))[:500]

    resource_summary = _resource_blocker_summary(resource_snapshot)
    manifest = {
        "schema": "host_resource_snapshot.v1",
        "output_dir": str(output_dir),
        "diagnostic_only": True,
        "execution_mode": "host_observability_only",
        "gpu_preflight_summary": gpu_summary,
        "resource_blocker_snapshot": resource_snapshot,
        "resource_blocker_summary": resource_summary,
        "resource_json": str(output_dir / "resource.json"),
        "summary_md": str(output_dir / "summary.md"),
        "progress_jsonl": str(progress_path),
        "destructive_action_started": False,
        "audio_replay_started": False,
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "diagnostic_host_resource_snapshot_only",
        "listen_pack": {},
        "artifact_rag_write_allowed": not no_artifact_rag,
    }
    _write_json(output_dir / "resource.json", resource_snapshot)
    _write_json(output_dir / "manifest.json", manifest)
    top_names = ", ".join(resource_summary["top_container_memory_names"]) or "none"
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Host Resource Snapshot",
                "",
                "Execution mode: `host_observability_only`",
                f"GPU status: `{gpu_summary.get('status')}`",
                f"GPU free MiB: `{gpu_summary.get('free_mib', '')}`",
                f"GPU used MiB: `{gpu_summary.get('used_mib', '')}`",
                f"GPU utilization pct: `{gpu_summary.get('utilization_pct', '')}`",
                f"Top container memory: {top_names}",
                f"Docker stats available: `{resource_summary.get('docker_container_stats_available')}`",
                "Destructive action started: `False`",
                "Audio replay started: `False`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _append_jsonl(
        progress_path,
        {
            "event": "end",
            "elapsed_sec": round(time.monotonic() - started, 6),
            "gpu_inference_skipped": True,
            "gpu_preflight_summary": gpu_summary,
            "resource_blocker_summary": resource_summary,
        },
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-gpu-free-mib", type=int, default=4096)
    parser.add_argument("--no-artifact-rag", action="store_true")
    parser.add_argument("--echo-progress", dest="echo_progress", action="store_true", default=True)
    parser.add_argument("--no-echo-progress", dest="echo_progress", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["STEM_SEARCH_ECHO_PROGRESS"] = "1" if args.echo_progress else "0"
    manifest = build_host_resource_snapshot(
        output_dir=args.out,
        min_gpu_free_mib=args.min_gpu_free_mib,
        no_artifact_rag=args.no_artifact_rag,
    )
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "manifest_path": str(Path(str(manifest["output_dir"])) / "manifest.json"),
                "gpu_status": manifest["gpu_preflight_summary"].get("status"),
                "top_container_memory_names": manifest["resource_blocker_summary"].get("top_container_memory_names"),
                "delivery_allowed": manifest["delivery_allowed"],
                "presentation_status": manifest["presentation_status"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
