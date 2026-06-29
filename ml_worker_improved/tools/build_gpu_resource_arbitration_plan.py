"""Build a non-destructive GPU resource arbitration plan."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_stem_search_launch_manifest import _collect_resource_blocker_snapshot, _resource_blocker_summary
from tools.run_fully_automatic_stem_search import _append_jsonl, _parse_bool, _write_json
from tools.stem_contamination_common import validate_no_artifact_rag_output


FORBIDDEN_ACTIONS = [
    "stop_container_without_permission",
    "kill_process_without_permission",
    "cpu_fallback_as_success",
    "threshold_relaxation",
    "deliver_blocked_output",
]


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def _gpu_summary(execution: Mapping[str, Any]) -> dict[str, Any]:
    return dict(execution.get("gpu_preflight_summary") or execution.get("gpu_preflight") or {})


def _arbitration_actions(
    *,
    gpu_summary: Mapping[str, Any],
    resource_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    gpu_status = str(gpu_summary.get("status") or "")
    free_mib = int(gpu_summary.get("free_mib") or 0)
    min_free_mib = int(gpu_summary.get("min_free_mib") or 0)
    top_containers = list(resource_summary.get("top_container_memory_names") or [])
    if gpu_status == "passed":
        return [
            {
                "action_id": "rerun_saved_launch_manifest",
                "action_type": "safe_next_step",
                "requires_user_permission": False,
                "destructive": False,
                "reason": "GPU preflight is already passed",
            }
        ]
    return [
        {
            "action_id": "wait_for_gpu_free_memory",
            "action_type": "safe_wait",
            "requires_user_permission": False,
            "destructive": False,
            "free_mib": free_mib,
            "min_free_mib": min_free_mib,
            "reason": "GPU free memory is below required threshold",
        },
        {
            "action_id": "request_permission_to_pause_external_gpu_services",
            "action_type": "permission_required",
            "requires_user_permission": True,
            "destructive": False,
            "top_container_memory_names": top_containers,
            "reason": "External GPU/RAM services may need to be paused outside the audio pipeline",
        },
    ]


def _recommended_action(gpu_summary: Mapping[str, Any]) -> str:
    return "rerun_saved_launch_manifest" if str(gpu_summary.get("status") or "") == "passed" else "request_external_gpu_release_or_wait"


def build_gpu_resource_arbitration_plan(
    *,
    output_dir: Path,
    iteration_manifest_path: Path | None = None,
    launch_execution_manifest_path: Path | None = None,
    resource_snapshot: Mapping[str, Any] | None = None,
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
            "gpu_inference_skipped": True,
            "reason": "diagnostic_gpu_resource_arbitration_only",
        },
    )

    iteration = _read_json(iteration_manifest_path)
    execution = _read_json(launch_execution_manifest_path)
    snapshot = dict(
        resource_snapshot
        if resource_snapshot is not None
        else execution.get("resource_blocker_snapshot") or _collect_resource_blocker_snapshot()
    )
    resource_summary = _resource_blocker_summary(snapshot)
    gpu_summary = _gpu_summary(execution)
    selected_action = str(iteration.get("selected_top_action") or "")
    recommended_action = _recommended_action(gpu_summary)
    actions = _arbitration_actions(gpu_summary=gpu_summary, resource_summary=resource_summary)
    arbitration_status = (
        "ready_to_rerun_launch"
        if recommended_action == "rerun_saved_launch_manifest"
        else "blocked_requires_external_gpu_release"
    )
    manifest = {
        "schema": "gpu_resource_arbitration_plan.v1",
        "output_dir": str(output_dir),
        "source_of_truth": {
            "iteration_manifest": str(iteration_manifest_path or ""),
            "launch_execution_manifest": str(launch_execution_manifest_path or ""),
        },
        "selected_top_action": selected_action,
        "arbitration_status": arbitration_status,
        "recommended_action": recommended_action,
        "gpu_preflight_summary": gpu_summary,
        "resource_blocker_snapshot": snapshot,
        "resource_blocker_summary": resource_summary,
        "actions_jsonl": str(output_dir / "actions.jsonl"),
        "summary_md": str(output_dir / "summary.md"),
        "forbidden_actions": FORBIDDEN_ACTIONS,
        "destructive_action_started": False,
        "audio_replay_started": False,
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "diagnostic_gpu_resource_arbitration_only",
        "listen_pack": {},
        "artifact_rag_write_allowed": not no_artifact_rag,
    }
    _write_jsonl(output_dir / "actions.jsonl", actions)
    _write_json(output_dir / "manifest.json", manifest)
    top_names = ", ".join(str(name) for name in resource_summary.get("top_container_memory_names") or []) or "none"
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# GPU Resource Arbitration",
                "",
                f"Recommended action: `{recommended_action}`",
                f"Arbitration status: `{arbitration_status}`",
                f"GPU free MiB: `{gpu_summary.get('free_mib', '')}`",
                f"GPU required free MiB: `{gpu_summary.get('min_free_mib', '')}`",
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
            "arbitration_status": arbitration_status,
            "recommended_action": recommended_action,
            "gpu_inference_skipped": True,
        },
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--iteration-manifest", type=Path)
    parser.add_argument("--launch-execution-manifest", type=Path)
    parser.add_argument("--resource-snapshot-json", type=Path)
    parser.add_argument("--no-artifact-rag", action="store_true")
    parser.add_argument("--echo-progress", dest="echo_progress", action="store_true", default=True)
    parser.add_argument("--no-echo-progress", dest="echo_progress", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["STEM_SEARCH_ECHO_PROGRESS"] = "1" if args.echo_progress else "0"
    resource_snapshot = _read_json(args.resource_snapshot_json) if args.resource_snapshot_json else None
    manifest = build_gpu_resource_arbitration_plan(
        output_dir=args.out,
        iteration_manifest_path=args.iteration_manifest,
        launch_execution_manifest_path=args.launch_execution_manifest,
        resource_snapshot=resource_snapshot,
        no_artifact_rag=args.no_artifact_rag,
    )
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "manifest_path": str(Path(str(manifest["output_dir"])) / "manifest.json"),
                "arbitration_status": manifest["arbitration_status"],
                "recommended_action": manifest["recommended_action"],
                "delivery_allowed": manifest["delivery_allowed"],
                "presentation_status": manifest["presentation_status"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
