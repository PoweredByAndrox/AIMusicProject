from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from build_testing_stuff_music_corpus_job_matrix import minus_baseline_availability  # noqa: E402
from worker.model_registry import ModelRegistry, default_model_registry  # noqa: E402


CommandRunner = Callable[[list[str]], Any]
EXECUTION_RESULTS_NAME = "testing_stuff_music_corpus_job_execution_results.jsonl"
EXECUTION_MANIFEST_NAME = "testing_stuff_music_corpus_job_execution_manifest.json"
EXECUTION_SUMMARY_NAME = "testing_stuff_music_corpus_job_execution_summary.md"
EXECUTION_PROGRESS_NAME = "testing_stuff_music_corpus_job_execution_progress.jsonl"
EXPECTED_ACCELERATION_MODE = "gpu_inference_then_cpu_dsp_scoring"
REGISTERED_MODEL_ENTRYPOINT = "/app/tools/run_registered_model_candidate.py"
DIRECT_MINUS_ENTRYPOINT = "/app/tools/export_instrumental_resurrection_minus.py"
MINUS_PHASE_REPAIR_ENTRYPOINT = "/app/tools/run_no_reference_minus_phase_repair.py"
REGISTERED_ROUTE_CONTRACT = "registered_model_candidate_manifest_required"
DIRECT_MINUS_ROUTE_CONTRACT = "documented_clean_minus_baseline"
MINUS_PHASE_REPAIR_ROUTE_CONTRACT = "no_reference_minus_phase_repair_candidate"


def stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def stable_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(stable_json(row) + "\n")


def safe_job_key(row: dict[str, Any], index: int) -> str:
    raw = "__".join(
        [
            str(row.get("track_index") or index),
            str(row.get("track_id") or "track"),
            str(row.get("target_stem") or "stem"),
            str(row.get("model_id") or row.get("route_name") or "route"),
        ]
    )
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in raw)[:180] or f"job_{index}"


class ProgressWriter:
    def __init__(self, path: Path, started_at: float) -> None:
        self.path = path
        self.started_at = started_at
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8", newline="\n")

    def append(self, stage: str, *, status: str = "progress", **metadata: Any) -> None:
        event = {
            "type": "testing_stuff_music_corpus_job_execution_progress",
            "stage": stage,
            "status": status,
            "since_start_sec": round(max(0.0, time.perf_counter() - self.started_at), 6),
            "metadata": metadata,
        }
        self._handle.write(stable_json(event) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def is_execution_ready(row: dict[str, Any]) -> bool:
    return bool(row.get("execution_ready")) and str(row.get("job_status") or "") == "queued"


def matches_filters(
    row: dict[str, Any],
    *,
    stems: Sequence[str] | None,
    track_ids: Sequence[str] | None,
    model_ids: Sequence[str] | None,
) -> bool:
    if stems and str(row.get("target_stem") or "") not in set(stems):
        return False
    if track_ids and str(row.get("track_id") or "") not in set(track_ids):
        return False
    if model_ids and str(row.get("model_id") or "") not in set(model_ids):
        return False
    return True


def select_jobs(
    rows: list[dict[str, Any]],
    *,
    max_jobs: int,
    stems: Sequence[str] | None = None,
    track_ids: Sequence[str] | None = None,
    model_ids: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    if max_jobs <= 0:
        raise ValueError("--max-jobs must be a positive integer")
    ready_candidates: list[tuple[int, dict[str, Any]]] = []
    skipped: list[dict[str, Any]] = []
    filtered_out = 0
    for row_index, row in enumerate(rows):
        if not matches_filters(row, stems=stems, track_ids=track_ids, model_ids=model_ids):
            filtered_out += 1
            continue
        if not is_execution_ready(row):
            skipped.append({**row, "skip_reason": "job_not_execution_ready"})
            continue
        ready_candidates.append((row_index, row))

    selected_pairs: list[tuple[int, dict[str, Any]]] = []
    selected_indexes: set[int] = set()
    if stems:
        for requested_stem in stable_unique(str(stem) for stem in stems):
            if len(selected_pairs) >= max_jobs:
                break
            for row_index, row in ready_candidates:
                if row_index in selected_indexes:
                    continue
                if str(row.get("target_stem") or "") != requested_stem:
                    continue
                selected_pairs.append((row_index, row))
                selected_indexes.add(row_index)
                break
    for row_index, row in ready_candidates:
        if len(selected_pairs) >= max_jobs:
            break
        if row_index in selected_indexes:
            continue
        selected_pairs.append((row_index, row))
        selected_indexes.add(row_index)
    selected = [row for _row_index, row in selected_pairs]
    filtered_out += len(ready_candidates) - len(selected)
    return selected, skipped, filtered_out


def default_command_runner(command: list[str], *, stdout_path: Path, stderr_path: Path) -> subprocess.CompletedProcess[str]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_handle:
        with stderr_path.open("w", encoding="utf-8") as stderr_handle:
            return subprocess.run(command, stdout=stdout_handle, stderr=stderr_handle, text=True, check=False)


def candidate_manifest_state(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    return {
        "path": str(manifest_path),
        "exists": manifest_path.exists(),
    }


def merge_target_manifest_fields(state: dict[str, Any], candidate_manifest: dict[str, Any]) -> dict[str, Any]:
    target_artifact = candidate_manifest.get("target_stem_artifact")
    if isinstance(target_artifact, dict):
        if target_artifact.get("target_stem"):
            state["target_stem"] = str(target_artifact["target_stem"])
        if target_artifact.get("path"):
            state["target_audio_path"] = str(target_artifact["path"])
        if target_artifact.get("source"):
            state["target_source"] = str(target_artifact["source"])
        if isinstance(target_artifact.get("source_stems"), list):
            state["target_source_stems"] = list(target_artifact["source_stems"])
    if candidate_manifest.get("target_stem"):
        state["target_stem"] = str(candidate_manifest["target_stem"])
    return state


def enriched_candidate_manifest_state(row: dict[str, Any]) -> dict[str, Any]:
    state = candidate_manifest_state(str(row.get("expected_candidate_manifest") or ""))
    execution_manifest_path = Path(str(row.get("execution_manifest_path") or ""))
    if execution_manifest_path.exists():
        try:
            execution_manifest = json.loads(execution_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            execution_manifest = {}
        execution_candidate = execution_manifest.get("candidate_manifest") if isinstance(execution_manifest, dict) else None
        if isinstance(execution_candidate, dict):
            for key in ("target_stem", "target_audio_path", "target_source", "target_source_stems"):
                if key in execution_candidate:
                    state[key] = execution_candidate[key]
    candidate_manifest_path = Path(str(state.get("path") or ""))
    if candidate_manifest_path.exists():
        try:
            candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            candidate_manifest = {}
        if isinstance(candidate_manifest, dict):
            merge_target_manifest_fields(state, candidate_manifest)
    return state


def path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def command_arg(command: Sequence[str], flag: str) -> str | None:
    try:
        index = list(command).index(flag)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return str(command[index + 1])


def model_availability_dict(registry: ModelRegistry, model_id: str) -> dict[str, Any]:
    entry = registry.get(model_id)
    availability = registry.availability(entry)
    return {
        "status": availability.status,
        "missing_paths": list(availability.missing_paths),
        "missing_python_modules": list(availability.missing_python_modules),
        "runtime_blockers": list(getattr(availability, "runtime_blockers", ())),
    }


def cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def preflight_job(row: dict[str, Any], *, registry: ModelRegistry, enforce_accelerator: bool = True) -> dict[str, Any]:
    command = [str(part) for part in row.get("command") or []]
    reasons: list[str] = []
    availability: dict[str, Any] = {"status": "available", "missing_paths": [], "missing_python_modules": []}
    route_contract = str(row.get("route_contract") or "")
    source_audio_path = Path(str(row.get("source_audio_path") or ""))
    output_dir = Path(str(row.get("output_dir") or ""))
    candidate_manifest = Path(str(row.get("expected_candidate_manifest") or ""))

    if not source_audio_path.exists():
        reasons.append("source_audio_missing_at_execution")
    if not command or command[0] != "python":
        reasons.append("unexpected_command_launcher")
    if not output_dir:
        reasons.append("missing_output_dir")
    if not candidate_manifest or not path_is_relative_to(candidate_manifest, output_dir):
        reasons.append("candidate_manifest_outside_output_dir")
    if enforce_accelerator and bool(row.get("gpu_inference_required")) and not cuda_available():
        reasons.append("cuda_unavailable")

    entrypoint = command[1] if len(command) > 1 else ""
    if route_contract == REGISTERED_ROUTE_CONTRACT:
        if entrypoint != REGISTERED_MODEL_ENTRYPOINT:
            reasons.append("unexpected_command_entrypoint")
        if len(command) < 4 or command[2] != str(source_audio_path) or command[3] != str(output_dir):
            reasons.append("command_input_output_mismatch")
        command_model_id = command_arg(command, "--model-id")
        row_model_id = str(row.get("model_id") or "")
        if command_model_id != row_model_id:
            reasons.append("command_model_id_mismatch")
        try:
            availability = model_availability_dict(registry, row_model_id)
        except Exception as exc:
            availability = {
                "status": "unavailable",
                "missing_paths": [],
                "missing_python_modules": [],
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            reasons.append("registered_model_unknown")
        if availability.get("status") != "available":
            reasons.append("registered_model_unavailable")
    elif route_contract == DIRECT_MINUS_ROUTE_CONTRACT:
        if entrypoint != DIRECT_MINUS_ENTRYPOINT:
            reasons.append("unexpected_command_entrypoint")
        if len(command) < 4 or command[2] != str(source_audio_path) or command[3] != str(output_dir):
            reasons.append("command_input_output_mismatch")
        availability = minus_baseline_availability()
        if availability.get("status") != "available":
            reasons.append("minus_baseline_unavailable")
    elif route_contract == MINUS_PHASE_REPAIR_ROUTE_CONTRACT:
        if entrypoint != MINUS_PHASE_REPAIR_ENTRYPOINT:
            reasons.append("unexpected_command_entrypoint")
        if command_arg(command, "--mix-path") != str(source_audio_path):
            reasons.append("command_mix_path_mismatch")
        if command_arg(command, "--output-dir") != str(output_dir):
            reasons.append("command_output_dir_mismatch")
        for flag, reason in (("--vocal-path", "vocal_audio_missing_at_execution"), ("--minus-path", "minus_audio_missing_at_execution")):
            value = command_arg(command, flag)
            if not value or not Path(value).exists():
                reasons.append(reason)
    else:
        reasons.append("unsupported_route_contract")

    if not reasons:
        return {
            "status": "passed",
            "reasons": [],
            "availability": availability,
            "route_contract": route_contract,
        }

    status = (
        "blocked_unavailable_at_execution"
        if any(reason.endswith("_unavailable") or reason in {"registered_model_unknown"} for reason in reasons)
        else "blocked_preflight_contract"
    )
    return {
        "status": "blocked",
        "execution_status": status,
        "reasons": stable_unique(reasons),
        "availability": availability,
        "route_contract": route_contract,
    }


def base_execution_row(row: dict[str, Any], *, index: int, output_root: Path) -> dict[str, Any]:
    job_key = safe_job_key(row, index)
    log_dir = output_root / "logs" / job_key
    return {
        **row,
        "execution_schema": "testing_stuff_music_corpus_job_execution_row.v1",
        "execution_job_key": job_key,
        "execution_started": False,
        "execution_status": "pending",
        "execution_elapsed_sec": 0.0,
        "dry_run": False,
        "logs": {
            "stdout_path": str(log_dir / "stdout.log"),
            "stderr_path": str(log_dir / "stderr.log"),
        },
        "candidate_manifest": candidate_manifest_state(str(row.get("expected_candidate_manifest") or "")),
        "preflight": {},
        "delivery_allowed": False,
        "presentation_status": "diagnostic_execution_not_for_listen_pack",
        "release_status": "not_release_ready",
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "downstream_gates_required_before_delivery": list(row.get("required_downstream_gates") or []),
    }


def normalize_runner_result(result: Any) -> int:
    if isinstance(result, int):
        return result
    return int(getattr(result, "returncode", 0))


def execute_one_job(
    row: dict[str, Any],
    *,
    index: int,
    output_root: Path,
    dry_run: bool,
    command_runner: Callable[..., Any],
    progress: ProgressWriter,
    registry: ModelRegistry,
    enforce_accelerator: bool = True,
) -> dict[str, Any]:
    execution_row = base_execution_row(row, index=index, output_root=output_root)
    command = list(row.get("command") or [])
    stdout_path = Path(execution_row["logs"]["stdout_path"])
    stderr_path = Path(execution_row["logs"]["stderr_path"])
    started_at = time.perf_counter()
    progress.append(
        "job_start",
        track_id=row.get("track_id"),
        target_stem=row.get("target_stem"),
        model_id=row.get("model_id"),
        dry_run=dry_run,
        gpu_inference_required=bool(row.get("gpu_inference_required")),
        expected_acceleration_mode=row.get("expected_acceleration_mode") or EXPECTED_ACCELERATION_MODE,
    )
    preflight = preflight_job(row, registry=registry, enforce_accelerator=enforce_accelerator)
    execution_row["preflight"] = preflight
    if preflight["status"] != "passed":
        execution_row.update(
            {
                "execution_status": preflight["execution_status"],
                "execution_started": False,
                "execution_elapsed_sec": round(time.perf_counter() - started_at, 6),
                "candidate_manifest": candidate_manifest_state(str(row.get("expected_candidate_manifest") or "")),
            }
        )
        progress.append(
            "job_end",
            status=preflight["execution_status"],
            execution_job_key=execution_row["execution_job_key"],
            preflight_reasons=preflight["reasons"],
        )
        return execution_row
    if dry_run:
        execution_row.update(
            {
                "execution_status": "dry_run",
                "execution_started": False,
                "dry_run": True,
                "gpu_inference_skipped": True,
                "gpu_inference_skip_reason": "dry_run",
                "execution_elapsed_sec": round(time.perf_counter() - started_at, 6),
                "candidate_manifest": candidate_manifest_state(str(row.get("expected_candidate_manifest") or "")),
            }
        )
        progress.append("job_end", status="dry_run", execution_job_key=execution_row["execution_job_key"])
        return execution_row

    execution_row["execution_started"] = True
    execution_row["dry_run"] = False
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return_code = normalize_runner_result(command_runner(command, stdout_path=stdout_path, stderr_path=stderr_path))
        if return_code != 0:
            raise RuntimeError(f"command returned non-zero exit code: {return_code}")
        candidate_state = enriched_candidate_manifest_state(row)
        if not candidate_state["exists"]:
            raise RuntimeError(f"expected candidate manifest was not created: {candidate_state['path']}")
    except Exception as exc:
        execution_row.update(
            {
                "execution_status": "failed",
                "execution_elapsed_sec": round(time.perf_counter() - started_at, 6),
                "candidate_manifest": candidate_manifest_state(str(row.get("expected_candidate_manifest") or "")),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        )
        progress.append("job_end", status="failed", execution_job_key=execution_row["execution_job_key"], error=execution_row["error"])
        return execution_row

    execution_row.update(
        {
            "execution_status": "completed",
            "execution_elapsed_sec": round(time.perf_counter() - started_at, 6),
            "candidate_manifest": candidate_state,
        }
    )
    progress.append("job_end", status="completed", execution_job_key=execution_row["execution_job_key"])
    return execution_row


def build_summary(
    *,
    source_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    execution_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
    filtered_out_jobs: int,
    max_jobs: int,
    dry_run: bool,
) -> dict[str, Any]:
    execution_counts = Counter(str(row.get("execution_status") or "unknown") for row in execution_rows)
    attempted_jobs = sum(1 for row in execution_rows if bool(row.get("execution_started")))
    acceleration_modes = sorted(
        {str(row.get("expected_acceleration_mode") or "") for row in selected_rows if row.get("expected_acceleration_mode")}
    )
    return {
        "schema": "testing_stuff_music_corpus_job_execution_summary.v1",
        "total_input_jobs": len(source_rows),
        "selected_jobs": len(selected_rows),
        "filtered_out_jobs": filtered_out_jobs,
        "skipped_blocked_jobs": len(skipped_rows),
        "max_jobs": int(max_jobs),
        "dry_run": bool(dry_run),
        "dry_run_jobs": int(execution_counts.get("dry_run", 0)),
        "completed_jobs": int(execution_counts.get("completed", 0)),
        "failed_jobs": int(execution_counts.get("failed", 0)),
        "attempted_jobs": attempted_jobs,
        "not_run_fail_fast_jobs": int(execution_counts.get("not_run_fail_fast", 0)),
        "blocked_preflight_jobs": int(execution_counts.get("blocked_preflight_contract", 0))
        + int(execution_counts.get("blocked_unavailable_at_execution", 0)),
        "execution_status_counts": dict(sorted(execution_counts.items())),
        "execution_started": any(bool(row.get("execution_started")) for row in execution_rows),
        "target_stems": sorted({str(row.get("target_stem") or "") for row in selected_rows if row.get("target_stem")}),
        "track_ids": sorted({str(row.get("track_id") or "") for row in selected_rows if row.get("track_id")}),
        "expected_acceleration_mode": acceleration_modes[0] if len(acceleration_modes) == 1 else "mixed",
        "expected_acceleration_modes": acceleration_modes,
        "gpu_inference_required": any(bool(row.get("gpu_inference_required")) for row in selected_rows),
        "delivery_allowed": False,
        "release_ready": False,
    }


def build_manifest(output_root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    if summary["dry_run"] and summary["selected_jobs"] and summary["dry_run_jobs"] == summary["selected_jobs"]:
        execution_status = "dry_run"
    elif summary["failed_jobs"]:
        execution_status = "failed"
    elif summary["blocked_preflight_jobs"] and not summary["attempted_jobs"]:
        execution_status = "blocked_preflight"
    elif summary["selected_jobs"] == 0:
        execution_status = "blocked_no_selected_jobs"
    elif summary["completed_jobs"] == summary["selected_jobs"]:
        execution_status = "completed"
    else:
        execution_status = "partial"
    return {
        "schema": "testing_stuff_music_corpus_job_execution_manifest.v1",
        "manifest_type": "testing_stuff_music_corpus_job_execution",
        "output_root": str(output_root),
        "execution_started": bool(summary["execution_started"]),
        "execution_status": execution_status,
        "delivery_allowed": False,
        "presentation_status": "diagnostic_execution_not_for_listen_pack",
        "release_status": "not_release_ready",
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "cleanup_routes_added": False,
        "thresholds_relaxed": False,
        "summary": summary,
        "output_files": {
            "results_jsonl": str(output_root / EXECUTION_RESULTS_NAME),
            "manifest_json": str(output_root / EXECUTION_MANIFEST_NAME),
            "summary_md": str(output_root / EXECUTION_SUMMARY_NAME),
            "progress_jsonl": str(output_root / EXECUTION_PROGRESS_NAME),
        },
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Testing Stuff Music Corpus Job Execution",
        "",
        f"- total_input_jobs: {summary['total_input_jobs']}",
        f"- selected_jobs: {summary['selected_jobs']}",
        f"- skipped_blocked_jobs: {summary['skipped_blocked_jobs']}",
        f"- completed_jobs: {summary['completed_jobs']}",
        f"- failed_jobs: {summary['failed_jobs']}",
        f"- dry_run: {str(summary['dry_run']).lower()}",
        f"- execution_started: {str(summary['execution_started']).lower()}",
        f"- delivery_allowed: {str(summary['delivery_allowed']).lower()}",
        f"- release_ready: {str(summary['release_ready']).lower()}",
        f"- expected_acceleration_mode: {summary['expected_acceleration_mode']}",
        "",
        "Downstream gates remain required before any listen-pack acceptance.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(output_root: Path, result: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / EXECUTION_RESULTS_NAME, result["rows"])
    write_json(output_root / EXECUTION_MANIFEST_NAME, result["manifest"])
    write_summary_md(output_root / EXECUTION_SUMMARY_NAME, result["summary"])
    if result["skipped_rows"]:
        write_jsonl(output_root / "testing_stuff_music_corpus_job_execution_skipped.jsonl", result["skipped_rows"])


def execute_job_matrix(
    *,
    job_matrix_jsonl: Path,
    output_root: Path,
    max_jobs: int,
    stems: Sequence[str] | None = None,
    track_ids: Sequence[str] | None = None,
    model_ids: Sequence[str] | None = None,
    dry_run: bool = False,
    command_runner: Callable[..., Any] | None = None,
    fail_fast: bool = False,
    registry: ModelRegistry | None = None,
    runtime_root: Path = Path("/app"),
) -> dict[str, Any]:
    started_at = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=True)
    progress = ProgressWriter(output_root / EXECUTION_PROGRESS_NAME, started_at=started_at)
    runner = command_runner or default_command_runner
    enforce_accelerator = command_runner is None and not dry_run
    registry = registry or default_model_registry(runtime_root=runtime_root)
    source_rows = list(iter_jsonl(job_matrix_jsonl))
    selected_rows, skipped_rows, filtered_out_jobs = select_jobs(
        source_rows,
        max_jobs=max_jobs,
        stems=stems,
        track_ids=track_ids,
        model_ids=model_ids,
    )
    progress.append(
        "start",
        total_input_jobs=len(source_rows),
        selected_jobs=len(selected_rows),
        skipped_blocked_jobs=len(skipped_rows),
        max_jobs=max_jobs,
        dry_run=dry_run,
        gpu_inference_required=any(bool(row.get("gpu_inference_required")) for row in selected_rows),
        expected_acceleration_mode=EXPECTED_ACCELERATION_MODE,
    )
    execution_rows: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(selected_rows, start=1):
            execution_row = execute_one_job(
                row,
                index=index,
                output_root=output_root,
                dry_run=dry_run,
                command_runner=runner,
                progress=progress,
                registry=registry,
                enforce_accelerator=enforce_accelerator,
            )
            execution_rows.append(execution_row)
            if fail_fast and execution_row["execution_status"] in {
                "failed",
                "blocked_preflight_contract",
                "blocked_unavailable_at_execution",
            }:
                for remaining_index, remaining_row in enumerate(selected_rows[index:], start=index + 1):
                    not_run_row = base_execution_row(remaining_row, index=remaining_index, output_root=output_root)
                    not_run_row.update(
                        {
                            "execution_status": "not_run_fail_fast",
                            "execution_started": False,
                            "skip_reason": "fail_fast_after_previous_failure",
                        }
                    )
                    execution_rows.append(not_run_row)
                break
        summary = build_summary(
            source_rows=source_rows,
            selected_rows=selected_rows,
            execution_rows=execution_rows,
            skipped_rows=skipped_rows,
            filtered_out_jobs=filtered_out_jobs,
            max_jobs=max_jobs,
            dry_run=dry_run,
        )
        manifest = build_manifest(output_root, summary)
        result = {
            "rows": execution_rows,
            "skipped_rows": skipped_rows,
            "summary": summary,
            "manifest": manifest,
        }
        write_outputs(output_root, result)
        progress.append("write_outputs", status="completed", output_root=str(output_root))
        return result
    finally:
        progress.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a bounded subset of testing_stuff_music corpus candidate-generation jobs.")
    parser.add_argument("--job-matrix-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-jobs", type=int, required=True)
    parser.add_argument("--stem", action="append", default=[])
    parser.add_argument("--track-id", action="append", default=[])
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--runtime-root", type=Path, default=Path("/app"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_testing_stuff_music_corpus_job_matrix.py",
        entrypoint_path="/app/tools/run_testing_stuff_music_corpus_job_matrix.py",
    )
    args = build_arg_parser().parse_args(argv)
    result = execute_job_matrix(
        job_matrix_jsonl=args.job_matrix_jsonl,
        output_root=args.output_root,
        max_jobs=int(args.max_jobs),
        stems=args.stem or None,
        track_ids=args.track_id or None,
        model_ids=args.model_id or None,
        dry_run=bool(args.dry_run),
        fail_fast=bool(args.fail_fast),
        runtime_root=args.runtime_root,
    )
    print(stable_json(result["summary"]))
    return 1 if result["summary"]["failed_jobs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
