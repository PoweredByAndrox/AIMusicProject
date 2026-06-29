"""Build a strict source-of-truth iteration analysis and idea funnel artifact."""

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

from tools.run_fully_automatic_stem_search import _append_jsonl, _write_json
from tools.stem_contamination_common import validate_no_artifact_rag_output


REJECTED_IDEAS = ["threshold_relaxation", "human_review", "deliver_blocked_output"]


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


def _normalized_task(current_task: str) -> str:
    return str(current_task or "").replace("НАПИШИ_ЗАДАЧУ_ЗДЕСЬ:", "").replace("НАПИШИ_ЗАДАЧУ_ЗДЕСЬ", "").strip()


def _ideas(selected_action: str) -> list[dict[str, Any]]:
    def status_for(idea_id: str, default: str) -> str:
        return "selected" if idea_id == selected_action else default

    return [
        {
            "idea_id": "wait_capable_launch_execution",
            "status": status_for("wait_capable_launch_execution", "completed_current"),
            "impact": 8,
            "feasibility": 9,
            "risk": 2,
            "reason": "uses filtered recipe and launches only after GPU preflight passes",
        },
        {
            "idea_id": "gpu_resource_arbitration",
            "status": status_for("gpu_resource_arbitration", "keep_next"),
            "impact": 8,
            "feasibility": 5,
            "risk": 5,
            "reason": "would free scheduling from unrelated GPU services, but may need explicit external-state permission",
        },
        {
            "idea_id": "retrain_failed_repair_checkpoints",
            "status": "keep_after_replay",
            "impact": 7,
            "feasibility": 6,
            "risk": 4,
            "reason": "needs replay/hardcase evidence after filtered launch to avoid training on stale blockers",
        },
        {
            "idea_id": "threshold_relaxation",
            "status": "rejected",
            "impact": 0,
            "feasibility": 8,
            "risk": 10,
            "reason": "forbidden fallback-as-success path",
        },
        {
            "idea_id": "human_review",
            "status": "rejected",
            "impact": 0,
            "feasibility": 2,
            "risk": 10,
            "reason": "explicitly forbidden; pipeline must stay automatic",
        },
        {
            "idea_id": "deliver_blocked_output",
            "status": "rejected",
            "impact": 0,
            "feasibility": 10,
            "risk": 10,
            "reason": "violates delivery contract",
        },
    ]


def build_audio_iteration_analysis(
    *,
    output_dir: Path,
    current_task: str,
    policy_manifest_path: Path | None = None,
    launch_manifest_path: Path | None = None,
    launch_execution_manifest_path: Path | None = None,
    no_artifact_rag: bool = True,
) -> dict[str, Any]:
    if no_artifact_rag:
        validate_no_artifact_rag_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    progress_path = output_dir / "progress.jsonl"
    task = _normalized_task(current_task)
    _append_jsonl(
        progress_path,
        {
            "event": "start",
            "elapsed_sec": 0.0,
            "current_task": task,
            "gpu_inference_skipped": True,
            "reason": "strict_iteration_analysis_only",
        },
    )

    policy = _read_json(policy_manifest_path)
    launch = _read_json(launch_manifest_path)
    execution = _read_json(launch_execution_manifest_path)
    confirmed = {
        "policy_manifest_path": str(policy_manifest_path or ""),
        "launch_manifest_path": str(launch_manifest_path or ""),
        "launch_execution_manifest_path": str(launch_execution_manifest_path or ""),
        "policy_delivery_allowed": bool(policy.get("delivery_allowed")) if policy else False,
        "launch_delivery_allowed": bool(launch.get("delivery_allowed")) if launch else False,
        "execution_delivery_allowed": bool(execution.get("delivery_allowed")) if execution else False,
        "active_blocked_repair_option_count": int(policy.get("active_blocked_repair_option_count") or 0),
        "skipped_repair_option_count": int((policy.get("root_cause_route_policy") or {}).get("skipped_repair_option_count") or 0),
        "launch_status": str(launch.get("launch_status") or ""),
        "execution_status": str(execution.get("execution_status") or ""),
        "run_command_started": bool(execution.get("run_command_started")),
        "recipe_sha256_verified": bool(execution.get("recipe_sha256_verified")),
        "gpu_preflight": execution.get("gpu_preflight") or launch.get("gpu_preflight") or {},
        "gpu_preflight_summary": execution.get("gpu_preflight_summary") or launch.get("gpu_preflight_summary") or {},
        "resource_blocker_summary": execution.get("resource_blocker_summary") or launch.get("resource_blocker_summary") or {},
    }
    selected_top_action = (
        "gpu_resource_arbitration"
        if confirmed["execution_status"] == "blocked_gpu_preflight" and confirmed["run_command_started"] is False
        else "wait_capable_launch_execution"
    )
    ideas = _ideas(selected_top_action)
    manifest = {
        "schema": "audio_iteration_analysis.v1",
        "output_dir": str(output_dir),
        "current_task": task,
        "source_of_truth": {
            "policy_manifest": str(policy_manifest_path or ""),
            "launch_manifest": str(launch_manifest_path or ""),
            "launch_execution_manifest": str(launch_execution_manifest_path or ""),
        },
        "confirmed": confirmed,
        "unconfirmed": {
            "accepted_full_track_listen_artifact": False,
            "gpu_available_for_full_replay": str((confirmed.get("gpu_preflight") or {}).get("status") or "") == "passed",
        },
        "idea_count": len(ideas),
        "selected_top_action": selected_top_action,
        "rejected_idea_ids": REJECTED_IDEAS,
        "ideas_jsonl": str(output_dir / "ideas.jsonl"),
        "summary_md": str(output_dir / "summary.md"),
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "diagnostic_iteration_analysis_only",
        "listen_pack": {},
        "artifact_rag_write_allowed": not no_artifact_rag,
    }
    _write_jsonl(output_dir / "ideas.jsonl", ideas)
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Audio Iteration Analysis",
                "",
                f"Task: {task}",
                "",
                f"Selected top action: `{manifest['selected_top_action']}`",
                f"Execution status: `{confirmed['execution_status']}`",
                f"Run command started: `{confirmed['run_command_started']}`",
                f"GPU free MiB: `{(confirmed.get('gpu_preflight_summary') or confirmed.get('gpu_preflight') or {}).get('free_mib', '')}`",
                f"Resource blocker: `{(confirmed.get('resource_blocker_summary') or {}).get('docker_container_stats_reason', '')}`",
                f"Delivery allowed: `{manifest['delivery_allowed']}`",
                "",
                "Rejected ideas: " + ", ".join(REJECTED_IDEAS),
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
            "selected_top_action": manifest["selected_top_action"],
            "gpu_inference_skipped": True,
        },
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--current-task", required=True)
    parser.add_argument("--policy-manifest", type=Path)
    parser.add_argument("--launch-manifest", type=Path)
    parser.add_argument("--launch-execution-manifest", type=Path)
    parser.add_argument("--no-artifact-rag", action="store_true")
    parser.add_argument("--echo-progress", dest="echo_progress", action="store_true", default=True)
    parser.add_argument("--no-echo-progress", dest="echo_progress", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["STEM_SEARCH_ECHO_PROGRESS"] = "1" if args.echo_progress else "0"
    manifest = build_audio_iteration_analysis(
        output_dir=args.out,
        current_task=args.current_task,
        policy_manifest_path=args.policy_manifest,
        launch_manifest_path=args.launch_manifest,
        launch_execution_manifest_path=args.launch_execution_manifest,
        no_artifact_rag=args.no_artifact_rag,
    )
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "manifest_path": str(Path(str(manifest["output_dir"])) / "manifest.json"),
                "selected_top_action": manifest["selected_top_action"],
                "delivery_allowed": manifest["delivery_allowed"],
                "presentation_status": manifest["presentation_status"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
