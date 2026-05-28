from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402


AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "track"


def audio_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def blocker_reasons(blockers: list[Any]) -> list[str]:
    reasons: set[str] = set()
    for blocker in blockers:
        if isinstance(blocker, dict) and blocker.get("reason") is not None:
            reasons.add(str(blocker["reason"]))
    return sorted(reasons)


def blocker_labels(blockers: list[Any]) -> list[str]:
    labels: set[str] = set()
    for blocker in blockers:
        if not isinstance(blocker, dict):
            continue
        if blocker.get("label") is not None:
            labels.add(str(blocker["label"]))
        for label in as_list(blocker.get("labels")):
            labels.add(str(label))
    return sorted(labels)


def worst_windows(ledger: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    windows = [window for window in as_list(ledger.get("windows")) if isinstance(window, dict)]
    windows.sort(key=lambda window: float(window.get("severity_score", 0.0)), reverse=True)
    return [
        {
            "start_sec": window.get("start_sec"),
            "end_sec": window.get("end_sec"),
            "severity_score": window.get("severity_score"),
            "labels": as_list(window.get("labels")),
            "release_decision": window.get("release_decision"),
        }
        for window in windows[:limit]
    ]


def contract_summary_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    contract = as_dict(manifest.get("detected_issue_action_contract"))
    contract_summary = as_dict(contract.get("summary"))
    selected_artifacts = as_dict(manifest.get("selected_artifacts"))
    defect_gate = as_dict(manifest.get("defect_effectiveness_gate"))
    issue_ledger = as_dict(manifest.get("issue_effectiveness_ledger"))
    ledger_summary = as_dict(issue_ledger.get("summary"))
    blockers = as_list(defect_gate.get("blockers"))
    artifact_release_status = selected_artifacts.get("release_status")
    release_status = (
        defect_gate["release_status"]
        if isinstance(defect_gate.get("release_status"), str)
        else "diagnostic_only"
    )
    release_status_drift = bool(
        artifact_release_status is not None and artifact_release_status != release_status
    )
    delivery_status = "deliverable" if release_status == "release_ready" else "diagnostic_only"
    selected_vocals_path = selected_artifacts.get("selected_vocals_flac")
    listen_selected_vocals_path = selected_artifacts.get("listen_selected_vocals_flac")
    return {
        "selected_profile": manifest.get("selected_profile"),
        "selected_score": manifest.get("selected_score"),
        "contract_status": contract.get("status", "missing_contract"),
        "gate_status": defect_gate.get("status", "missing_release_gate"),
        "release_status": release_status,
        "artifact_release_status": artifact_release_status,
        "release_status_drift": release_status_drift,
        "delivery_status": delivery_status,
        "uncovered_labels": contract_summary.get("uncovered_labels", []),
        "covered_labels": contract_summary.get("covered_labels", []),
        "rejected_labels": contract_summary.get("rejected_labels", []),
        "issue_segment_count": contract_summary.get("issue_segment_count", 0),
        "blocked_window_count": ledger_summary.get("blocked_window_count", 0),
        "blocker_reasons": blocker_reasons(blockers),
        "blocker_labels": blocker_labels(blockers),
        "worst_windows": worst_windows(issue_ledger),
        "selected_vocals_flac": selected_vocals_path,
        "listen_selected_vocals_flac": listen_selected_vocals_path,
        "deliverable_vocals_flac": selected_vocals_path if delivery_status == "deliverable" else None,
        "deliverable_listen_vocals_flac": (
            listen_selected_vocals_path if delivery_status == "deliverable" else None
        ),
        "diagnostic_vocals_flac": selected_vocals_path if delivery_status != "deliverable" else None,
        "diagnostic_listen_vocals_flac": (
            listen_selected_vocals_path if delivery_status != "deliverable" else None
        ),
        "listen_segment_issue_clips_manifest": selected_artifacts.get("listen_segment_issue_clips_manifest"),
    }


def write_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def run_track(
    *,
    source_path: Path,
    output_dir: Path,
    listen_dir: Path,
    adaptive_script: Path,
    python_executable: str,
    device: str,
    feedback_store: Path | None,
    reuse_existing_candidates: bool,
    segment_issue_report_max_windows: int,
    segment_issue_listen_clip_count: int,
) -> dict[str, Any]:
    command = [
        python_executable,
        str(adaptive_script),
        str(source_path),
        str(output_dir),
        "--listen-dir",
        str(listen_dir),
        "--device",
        device,
        "--segment-issue-report-max-windows",
        str(segment_issue_report_max_windows),
        "--segment-issue-listen-clip-count",
        str(segment_issue_listen_clip_count),
    ]
    if feedback_store is not None:
        command.extend(["--feedback-store", str(feedback_store)])
    if reuse_existing_candidates:
        command.append("--reuse-existing-candidates")

    started_at = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed_sec = round(time.perf_counter() - started_at, 3)
    manifest_path = output_dir / "adaptive_vocal_manifest.json"
    record: dict[str, Any] = {
        "source_path": str(source_path),
        "output_dir": str(output_dir),
        "listen_dir": str(listen_dir),
        "manifest_path": str(manifest_path),
        "exit_code": completed.returncode,
        "elapsed_sec": elapsed_sec,
    }
    if completed.stdout.strip():
        record["stdout_tail"] = completed.stdout.strip().splitlines()[-20:]
    if completed.stderr.strip():
        record["stderr_tail"] = completed.stderr.strip().splitlines()[-40:]
    if completed.returncode == 0 and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record.update(contract_summary_from_manifest(manifest))
        record["candidate_count"] = len(manifest.get("candidates", []))
    else:
        record["contract_status"] = "route_failed"
        record["uncovered_labels"] = []
    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adaptive vocal isolation on a directory and summarize issue contracts.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--listen-root", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--feedback-store", type=Path, default=None)
    parser.add_argument("--adaptive-script", type=Path, default=Path("/app/tools/adaptive_vocal_isolation.py"))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reuse-existing-candidates", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--fail-on-uncovered", action="store_true")
    parser.add_argument("--segment-issue-report-max-windows", type=int, default=8)
    parser.add_argument("--segment-issue-listen-clip-count", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_adaptive_vocal_contract_batch.py",
        entrypoint_path="/app/tools/run_adaptive_vocal_contract_batch.py",
    )
    args = parse_args(argv)
    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    tracks = audio_files(input_dir)
    if args.limit is not None:
        tracks = tracks[: max(0, int(args.limit))]

    summary: dict[str, Any] = {
        "schema_version": 1,
        "mode": "adaptive_vocal_contract_batch",
        "input_dir": str(input_dir),
        "output_root": str(args.output_root),
        "listen_root": str(args.listen_root),
        "feedback_store": None if args.feedback_store is None else str(args.feedback_store),
        "track_count": len(tracks),
        "release_ready_count": 0,
        "diagnostic_only_count": 0,
        "release_blocked_count": 0,
        "tracks": [],
    }
    write_summary(args.summary_path, summary)

    for index, source_path in enumerate(tracks, start=1):
        track_slug = slugify(source_path.stem)
        output_dir = args.output_root / f"{index:02d}_{track_slug}"
        listen_dir = args.listen_root / f"{index:02d}_{track_slug}"
        print(json.dumps({"event": "track_start", "index": index, "track_count": len(tracks), "source_path": str(source_path)}), flush=True)
        record = run_track(
            source_path=source_path,
            output_dir=output_dir,
            listen_dir=listen_dir,
            adaptive_script=args.adaptive_script,
            python_executable=str(args.python_executable),
            device=str(args.device),
            feedback_store=args.feedback_store,
            reuse_existing_candidates=bool(args.reuse_existing_candidates),
            segment_issue_report_max_windows=int(args.segment_issue_report_max_windows),
            segment_issue_listen_clip_count=int(args.segment_issue_listen_clip_count),
        )
        summary["tracks"].append(record)
        summary["completed_count"] = len(summary["tracks"])
        summary["failed_count"] = sum(1 for item in summary["tracks"] if item.get("exit_code") != 0)
        summary["uncovered_count"] = sum(1 for item in summary["tracks"] if item.get("contract_status") == "uncovered_labels")
        summary["release_ready_count"] = sum(
            1 for item in summary["tracks"] if item.get("release_status") == "release_ready"
        )
        summary["diagnostic_only_count"] = sum(
            1 for item in summary["tracks"] if item.get("release_status") == "diagnostic_only"
        )
        summary["release_blocked_count"] = sum(
            1 for item in summary["tracks"] if item.get("delivery_status") == "diagnostic_only"
        )
        write_summary(args.summary_path, summary)
        print(
            json.dumps(
                {
                    "event": "track_done",
                    "index": index,
                    "source_path": str(source_path),
                    "exit_code": record["exit_code"],
                    "contract_status": record.get("contract_status"),
                    "release_status": record.get("release_status"),
                    "delivery_status": record.get("delivery_status"),
                    "selected_profile": record.get("selected_profile"),
                }
            ),
            flush=True,
        )
        if bool(args.fail_fast) and record["exit_code"] != 0:
            return 1

    if bool(args.fail_on_uncovered) and int(summary.get("uncovered_count", 0)) > 0:
        return 2
    if int(summary.get("failed_count", 0)) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
