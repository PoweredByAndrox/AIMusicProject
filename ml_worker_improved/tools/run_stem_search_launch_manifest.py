"""Execute a stem-search launch manifest fail-closed."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_stem_search_launch_manifest import (
    _collect_resource_blocker_snapshot,
    _gpu_preflight_summary,
    _resource_blocker_summary,
    _sha256_file,
    _status_summary,
    _write_status_summary_md,
)
from tools.run_fully_automatic_stem_search import _append_jsonl, _collect_cuda_preflight, _parse_bool, _write_json
from tools.stem_contamination_common import validate_no_artifact_rag_output


LaunchCommandRunner = Callable[[list[str], int], None]
GpuPreflightCollector = Callable[[bool, int], Mapping[str, Any]]
ResourceSnapshotCollector = Callable[[], Mapping[str, Any]]
SleepFn = Callable[[float], None]


def _default_command_runner(command: list[str], timeout_sec: int) -> None:
    completed = subprocess.run(command, check=False, timeout=timeout_sec)
    if completed.returncode != 0:
        raise RuntimeError(f"launch command failed with exit code {completed.returncode}")


def _status_manifest(
    *,
    launch_manifest_path: Path,
    output_dir: Path,
    launch_manifest: Mapping[str, Any],
    execution_status: str,
    recipe_sha256_verified: bool,
    run_command_started: bool,
    gpu_preflight: Mapping[str, Any] | None = None,
    resource_snapshot: Mapping[str, Any] | None = None,
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": "stem_search_launch_execution.v1",
        "launch_manifest_path": str(launch_manifest_path),
        "output_dir": str(output_dir),
        "source_launch_schema": str(launch_manifest.get("schema") or ""),
        "recipe_path": str(launch_manifest.get("recipe_path") or ""),
        "recipe_sha256": str(launch_manifest.get("recipe_sha256") or ""),
        "recipe_sha256_verified": bool(recipe_sha256_verified),
        "run_output_dir": str(launch_manifest.get("run_output_dir") or ""),
        "in_container_command": list(launch_manifest.get("in_container_command") or []),
        "execution_status": execution_status,
        "run_command_started": bool(run_command_started),
        "gpu_preflight": dict(gpu_preflight or {}),
        "gpu_preflight_summary": _gpu_preflight_summary(gpu_preflight),
        "resource_blocker_snapshot": dict(resource_snapshot or {}),
        "resource_blocker_summary": _resource_blocker_summary(resource_snapshot),
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "diagnostic_launch_execution_only",
        "listen_pack": {},
    }
    manifest["status_summary"] = _status_summary(
        gpu_preflight=gpu_preflight,
        result_status=("command_completed" if execution_status == "command_completed" else "not_started" if not run_command_started else "failed"),
        run_command_started=run_command_started,
        execution_status=execution_status,
    )
    manifest["summary_md"] = str(output_dir / "summary.md")
    if error:
        manifest["error"] = dict(error)
    return manifest


def run_stem_search_launch_manifest(
    *,
    launch_manifest_path: Path,
    output_dir: Path,
    command_runner: LaunchCommandRunner | None = None,
    gpu_preflight_collector: GpuPreflightCollector | None = None,
    resource_snapshot_collector: ResourceSnapshotCollector | None = None,
    sleep_fn: SleepFn | None = None,
    command_timeout_sec: int = 24 * 60 * 60,
    wait_for_gpu_free_sec_override: float | None = None,
    gpu_preflight_poll_sec_override: float | None = None,
    no_artifact_rag: bool = True,
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
            "launch_manifest_path": str(launch_manifest_path),
            "gpu_inference_skipped": False,
        },
    )
    launch_manifest = json.loads(launch_manifest_path.read_text(encoding="utf-8-sig"))
    recipe_path = Path(str(launch_manifest.get("recipe_path") or ""))
    expected_sha = str(launch_manifest.get("recipe_sha256") or "")
    actual_sha = _sha256_file(recipe_path) if recipe_path.is_file() else ""
    recipe_ok = bool(expected_sha and actual_sha == expected_sha)

    def _persist_status_manifest(manifest: Mapping[str, Any]) -> None:
        _write_json(output_dir / "manifest.json", manifest)
        _write_status_summary_md(output_dir, manifest)

    if not recipe_ok:
        manifest = _status_manifest(
            launch_manifest_path=launch_manifest_path,
            output_dir=output_dir,
            launch_manifest=launch_manifest,
            execution_status="blocked_recipe_sha_mismatch",
            recipe_sha256_verified=False,
            run_command_started=False,
            error={
                "type": "recipe_sha_mismatch",
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
            },
        )
        _persist_status_manifest(manifest)
        _append_jsonl(
            progress_path,
            {
                "event": "end",
                "elapsed_sec": round(time.monotonic() - started, 6),
                "execution_status": manifest["execution_status"],
                "run_command_started": False,
                "gpu_preflight_summary": manifest["gpu_preflight_summary"],
                "status_summary": manifest["status_summary"],
            },
        )
        return manifest

    require_gpu = bool(launch_manifest.get("require_gpu", True))
    min_gpu_free_mib = int(launch_manifest.get("min_gpu_free_mib") or 0)
    wait_for_gpu_free_sec = float(
        max(
            0.0,
            float(
                wait_for_gpu_free_sec_override
                if wait_for_gpu_free_sec_override is not None
                else launch_manifest.get("wait_for_gpu_free_sec") or 0.0
            ),
        )
    )
    gpu_preflight_poll_sec = float(
        max(
            0.1,
            float(
                gpu_preflight_poll_sec_override
                if gpu_preflight_poll_sec_override is not None
                else launch_manifest.get("gpu_preflight_poll_sec") or 5.0
            ),
        )
    )
    collector = gpu_preflight_collector or (lambda require_gpu, min_free_mib: _collect_cuda_preflight(require_gpu=require_gpu, min_free_mib=min_free_mib))
    resource_collector = resource_snapshot_collector or _collect_resource_blocker_snapshot
    sleeper = sleep_fn or time.sleep
    gpu_preflight = dict(collector(require_gpu, min_gpu_free_mib))
    gpu_ready = (not require_gpu) or gpu_preflight.get("status") in {"passed", "not_required", "not_checked"}
    wait_deadline = time.monotonic() + wait_for_gpu_free_sec
    while require_gpu and not gpu_ready and wait_for_gpu_free_sec > 0.0 and time.monotonic() < wait_deadline:
        remaining_sec = max(0.0, wait_deadline - time.monotonic())
        sleep_sec = min(gpu_preflight_poll_sec, remaining_sec)
        _append_jsonl(
            progress_path,
            {
                "event": "gpu_preflight_wait",
                "elapsed_sec": round(time.monotonic() - started, 6),
                "poll_sec": sleep_sec,
                "deadline_remaining_sec": round(remaining_sec, 6),
                **gpu_preflight,
            },
        )
        sleeper(sleep_sec)
        gpu_preflight = dict(collector(require_gpu, min_gpu_free_mib))
        gpu_ready = gpu_preflight.get("status") in {"passed", "not_required", "not_checked"}
    _append_jsonl(
        progress_path,
        {
            "event": "gpu_preflight",
            "elapsed_sec": round(time.monotonic() - started, 6),
            **gpu_preflight,
        },
    )
    _append_jsonl(
        progress_path,
        {
            "event": "resource_snapshot_start",
            "elapsed_sec": round(time.monotonic() - started, 6),
            "gpu_inference_skipped": True,
        },
    )
    resource_snapshot = dict(resource_collector())
    _append_jsonl(
        progress_path,
        {
            "event": "resource_blocker_snapshot",
            "elapsed_sec": round(time.monotonic() - started, 6),
            "resource_blocker_summary": _resource_blocker_summary(resource_snapshot),
        },
    )
    if not gpu_ready:
        manifest = _status_manifest(
            launch_manifest_path=launch_manifest_path,
            output_dir=output_dir,
            launch_manifest=launch_manifest,
            execution_status="blocked_gpu_preflight",
            recipe_sha256_verified=True,
            run_command_started=False,
            gpu_preflight=gpu_preflight,
            resource_snapshot=resource_snapshot,
        )
        _persist_status_manifest(manifest)
        _append_jsonl(
            progress_path,
            {
                "event": "end",
                "elapsed_sec": round(time.monotonic() - started, 6),
                "execution_status": manifest["execution_status"],
                "run_command_started": False,
                "gpu_preflight_summary": manifest["gpu_preflight_summary"],
                "status_summary": manifest["status_summary"],
                "resource_blocker_summary": manifest["resource_blocker_summary"],
            },
        )
        return manifest

    command = [str(part) for part in launch_manifest.get("in_container_command") or []]
    if not command:
        manifest = _status_manifest(
            launch_manifest_path=launch_manifest_path,
            output_dir=output_dir,
            launch_manifest=launch_manifest,
            execution_status="blocked_missing_command",
            recipe_sha256_verified=True,
            run_command_started=False,
            gpu_preflight=gpu_preflight,
            resource_snapshot=resource_snapshot,
        )
        _persist_status_manifest(manifest)
        _append_jsonl(
            progress_path,
            {
                "event": "end",
                "elapsed_sec": round(time.monotonic() - started, 6),
                "execution_status": manifest["execution_status"],
                "run_command_started": False,
                "gpu_preflight_summary": manifest["gpu_preflight_summary"],
                "status_summary": manifest["status_summary"],
                "resource_blocker_summary": manifest["resource_blocker_summary"],
            },
        )
        return manifest

    _append_jsonl(
        progress_path,
        {
            "event": "command_start",
            "elapsed_sec": round(time.monotonic() - started, 6),
            "command": command,
        },
    )
    runner = command_runner or _default_command_runner
    try:
        runner(command, int(command_timeout_sec))
    except Exception as exc:
        manifest = _status_manifest(
            launch_manifest_path=launch_manifest_path,
            output_dir=output_dir,
            launch_manifest=launch_manifest,
            execution_status="failed_command",
            recipe_sha256_verified=True,
            run_command_started=True,
            gpu_preflight=gpu_preflight,
            resource_snapshot=resource_snapshot,
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        _persist_status_manifest(manifest)
        _append_jsonl(
            progress_path,
            {
                "event": "end",
                "elapsed_sec": round(time.monotonic() - started, 6),
                "execution_status": manifest["execution_status"],
                "run_command_started": True,
                "gpu_preflight_summary": manifest["gpu_preflight_summary"],
                "status_summary": manifest["status_summary"],
                "resource_blocker_summary": manifest["resource_blocker_summary"],
            },
        )
        return manifest

    manifest = _status_manifest(
        launch_manifest_path=launch_manifest_path,
        output_dir=output_dir,
        launch_manifest=launch_manifest,
        execution_status="command_completed",
        recipe_sha256_verified=True,
        run_command_started=True,
        gpu_preflight=gpu_preflight,
        resource_snapshot=resource_snapshot,
    )
    _persist_status_manifest(manifest)
    _append_jsonl(
        progress_path,
        {
            "event": "end",
            "elapsed_sec": round(time.monotonic() - started, 6),
            "execution_status": manifest["execution_status"],
            "run_command_started": True,
            "gpu_preflight_summary": manifest["gpu_preflight_summary"],
            "status_summary": manifest["status_summary"],
            "resource_blocker_summary": manifest["resource_blocker_summary"],
        },
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--command-timeout-sec", type=int, default=24 * 60 * 60)
    parser.add_argument("--wait-for-gpu-free-sec", type=float)
    parser.add_argument("--gpu-preflight-poll-sec", type=float)
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
    manifest = run_stem_search_launch_manifest(
        launch_manifest_path=args.launch_manifest,
        output_dir=args.out,
        command_timeout_sec=args.command_timeout_sec,
        wait_for_gpu_free_sec_override=args.wait_for_gpu_free_sec,
        gpu_preflight_poll_sec_override=args.gpu_preflight_poll_sec,
        no_artifact_rag=args.no_artifact_rag,
    )
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "manifest_path": str(Path(str(manifest["output_dir"])) / "manifest.json"),
                "execution_status": manifest["execution_status"],
                "run_command_started": manifest["run_command_started"],
                "delivery_allowed": manifest["delivery_allowed"],
                "gpu_preflight": manifest.get("gpu_preflight") or {},
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
