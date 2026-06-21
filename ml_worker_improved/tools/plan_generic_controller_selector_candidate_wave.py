#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402
from tools.train_generic_finetuning_controller_model import stable_hash  # noqa: E402
from worker.model_registry import ModelRegistry, default_model_registry  # noqa: E402


CANDIDATE_WAVE_JOBS_NAME = "jobs.jsonl"
CANDIDATE_WAVE_MANIFEST_NAME = "plan.json"
CANDIDATE_WAVE_PROGRESS_NAME = "progress.jsonl"

REGISTERED_MODEL_ENTRYPOINT = "/app/tools/run_registered_model_candidate.py"
DIRECT_INSTRUMENTAL_RESURRECTION_ENTRYPOINT = "/app/tools/export_instrumental_resurrection_minus.py"
REGISTERED_ROUTE_CONTRACT = "registered_model_candidate_manifest_required"
DIRECT_MINUS_ROUTE_CONTRACT = "documented_clean_minus_baseline"

PLANNED_ACTIONS = {
    "generate_new_candidate_wave",
    "collect_controller_prior_evidence",
}

SAFETY_FLAGS = {
    "metadata_only": True,
    "audio_inference_started": False,
    "audio_render_started": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "delivery_allowed": False,
    "release_ready": False,
    "not_release_artifact": True,
    "thresholds_relaxed": False,
}

START_TIME = time.monotonic()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "gpu_inference_required": False,
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "candidate_wave_planner_writes_metadata_jobs_only",
        "expected_acceleration_mode": "external_only_cpu_metadata_candidate_wave_planning",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def safe_slug(value: str, *, fallback: str) -> str:
    slug = "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value))
    return slug[:180] or fallback


def normalize_target_stem(stem_type: str) -> str:
    normalized = str(stem_type or "").strip().lower()
    if normalized in {"minus", "instrumental", "instrumentals"}:
        return "minus"
    if normalized in {"vocal", "vocals", "voice"}:
        return "vocal"
    return normalized or "unknown"


def route_gap_rows(route_gap_plan_manifest_json: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(route_gap_plan_manifest_json)
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs.get("route_gaps_jsonl"):
        raise ValueError("route gap plan manifest is missing outputs.route_gaps_jsonl")
    rows = list(iter_jsonl(Path(str(outputs["route_gaps_jsonl"]))))
    return manifest, rows


def shadow_batch_manifest_path(route_gap_manifest: dict[str, Any]) -> Path:
    summary = route_gap_manifest.get("summary")
    if isinstance(summary, dict) and summary.get("shadow_batch_manifest_json"):
        return Path(str(summary["shadow_batch_manifest_json"]))
    if route_gap_manifest.get("shadow_batch_manifest_json"):
        return Path(str(route_gap_manifest["shadow_batch_manifest_json"]))
    raise ValueError("route gap plan manifest is missing summary.shadow_batch_manifest_json")


def selector_decisions_path(single_manifest_json: Path) -> Path:
    manifest = read_json(single_manifest_json)
    outputs = manifest.get("outputs")
    if isinstance(outputs, dict) and outputs.get("selector_decisions_jsonl"):
        return Path(str(outputs["selector_decisions_jsonl"]))
    return single_manifest_json.parent / "generic_finetuning_controller_selector_decisions.jsonl"


def source_path_from_selector_decision(decision: dict[str, Any]) -> str:
    selected = decision.get("selected_candidate")
    if isinstance(selected, dict):
        for key in ("mix_path", "source_audio_path"):
            if selected.get(key):
                return str(selected[key])
    for key in ("mix_path", "source_audio_path"):
        if decision.get(key):
            return str(decision[key])
    return ""


def source_refs_from_shadow_batch(shadow_batch_manifest_json: Path) -> list[dict[str, Any]]:
    manifest = read_json(shadow_batch_manifest_json)
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs.get("items_jsonl"):
        raise ValueError("shadow batch manifest is missing outputs.items_jsonl")

    refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item_index, item in enumerate(iter_jsonl(Path(str(outputs["items_jsonl"]))), start=1):
        single_manifest_raw = item.get("single_manifest_json")
        if not single_manifest_raw:
            continue
        single_manifest_json = Path(str(single_manifest_raw))
        decisions_path = selector_decisions_path(single_manifest_json)
        for decision_index, decision in enumerate(iter_jsonl(decisions_path), start=1):
            source_audio_path = source_path_from_selector_decision(decision)
            if not source_audio_path or source_audio_path in seen:
                continue
            seen.add(source_audio_path)
            source_path = Path(source_audio_path)
            refs.append(
                {
                    "source_ref_index": len(refs) + 1,
                    "item_index": item_index,
                    "decision_index": decision_index,
                    "track_id": str(decision.get("track_id") or item.get("track_id") or f"item_{item_index}"),
                    "source_audio_path": source_audio_path,
                    "source_audio_exists": source_path.exists(),
                    "single_manifest_json": str(single_manifest_json),
                    "selector_decisions_jsonl": str(decisions_path),
                }
            )
    return refs


def registered_entry_or_none(registry: ModelRegistry, route_id: str) -> Any | None:
    try:
        return registry.get(route_id)
    except KeyError:
        return None


def job_output_dir(*, output_dir: Path, index: int, route_id: str, track_id: str) -> Path:
    _ = (route_id, track_id)
    return output_dir / "candidate_jobs" / f"j{index:03d}"


def expected_manifest_for_job(*, output_dir: Path) -> Path:
    return output_dir / "stem_manifest.json"


def build_registered_command(
    *,
    source_audio_path: str,
    output_dir: Path,
    route_id: str,
    device: str,
    target_stem: str,
) -> list[str]:
    return [
        "python",
        REGISTERED_MODEL_ENTRYPOINT,
        source_audio_path,
        str(output_dir),
        "--model-id",
        route_id,
        "--device",
        device,
        "--target-stem",
        target_stem,
    ]


def build_direct_resurrection_command(
    *,
    source_audio_path: str,
    output_dir: Path,
    resurrection_device: str,
) -> list[str]:
    return [
        "python",
        DIRECT_INSTRUMENTAL_RESURRECTION_ENTRYPOINT,
        source_audio_path,
        str(output_dir),
        "--device",
        resurrection_device,
    ]


def execution_backend_for_route(*, registry: ModelRegistry, route_id: str) -> tuple[str, str, Any | None]:
    if route_id == "instrumental_resurrection_minus":
        return ("direct_instrumental_resurrection_export", DIRECT_MINUS_ROUTE_CONTRACT, None)
    entry = registered_entry_or_none(registry, route_id)
    if entry is not None:
        return ("registered_model_candidate", REGISTERED_ROUTE_CONTRACT, entry)
    return ("unsupported_route", "unsupported_route_contract", None)


def job_status_and_blockers(*, source_ref: dict[str, Any], execution_backend: str) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if not bool(source_ref.get("source_audio_exists")):
        blockers.append("source_audio_missing")
    if execution_backend == "unsupported_route":
        blockers.append("unsupported_route")
    if blockers:
        return ("blocked", blockers)
    return ("planned", [])


def build_candidate_wave_job(
    *,
    route_gap: dict[str, Any],
    source_ref: dict[str, Any],
    output_dir: Path,
    index: int,
    device: str,
    resurrection_device: str,
    registry: ModelRegistry,
) -> dict[str, Any]:
    route_id = str(route_gap.get("route_id") or "")
    stem_type = str(route_gap.get("stem_type") or "unknown")
    target_stem = normalize_target_stem(stem_type)
    backend, route_contract, entry = execution_backend_for_route(registry=registry, route_id=route_id)
    row_output_dir = job_output_dir(
        output_dir=output_dir,
        index=index,
        route_id=route_id,
        track_id=str(source_ref.get("track_id") or "track"),
    )
    if backend == "direct_instrumental_resurrection_export":
        command = build_direct_resurrection_command(
            source_audio_path=str(source_ref["source_audio_path"]),
            output_dir=row_output_dir,
            resurrection_device=resurrection_device,
        )
        model_id = route_id
        model_family = "bs_roformer"
        route_runtime = "direct_docker_gpu_tool"
    elif backend == "registered_model_candidate":
        command = build_registered_command(
            source_audio_path=str(source_ref["source_audio_path"]),
            output_dir=row_output_dir,
            route_id=route_id,
            device=device,
            target_stem=target_stem,
        )
        model_id = route_id
        model_family = str(getattr(entry, "family", "unknown"))
        route_runtime = "registered_model_registry"
    else:
        command = []
        model_id = route_id
        model_family = "unknown"
        route_runtime = "unsupported"

    status, blockers = job_status_and_blockers(source_ref=source_ref, execution_backend=backend)
    expected_manifest = expected_manifest_for_job(output_dir=row_output_dir)
    job_alias = row_output_dir.name
    job = {
        "schema": "generic_controller_selector_candidate_wave_job.v1",
        "job_id": stable_hash(["controller_selector_candidate_wave", index, route_gap, source_ref])[:16],
        "job_alias": job_alias,
        "job_index": index,
        "job_status": status,
        "blockers": blockers,
        "wave_intent": str(route_gap.get("recommended_action") or ""),
        "priority": str(route_gap.get("priority") or ""),
        "next_wave_required": bool(route_gap.get("next_wave_required")),
        "stem_type": stem_type,
        "target_stem": target_stem,
        "route_id": route_id,
        "model_id": model_id,
        "model_family": model_family,
        "execution_backend": backend,
        "route_contract": route_contract,
        "route_runtime": route_runtime,
        "source_audio_path": str(source_ref["source_audio_path"]),
        "source_audio_exists": bool(source_ref.get("source_audio_exists")),
        "track_id": str(source_ref.get("track_id") or ""),
        "source_ref": dict(source_ref),
        "output_dir": str(row_output_dir),
        "expected_candidate_manifest": str(expected_manifest),
        "command": command,
        "execution_ready": status == "planned",
        "gpu_inference_required": status == "planned",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "candidate_wave_planner_only_does_not_execute_jobs",
        "expected_acceleration_mode": "gpu_inference_candidate_generation",
        "required_downstream_gates": [
            "candidate_execution_preflight",
            "mode_specific_no_reference_score_matrix",
            "controller_selector_shadow_replay",
            "artifact_status_contract",
        ],
        "delivery_allowed": False,
        "presentation_status": "candidate_wave_plan_not_for_listen_pack",
        "release_status": "planning_artifact_only",
        "release_ready": False,
    }
    return job


def planned_route_gap_filter(row: dict[str, Any]) -> bool:
    return str(row.get("recommended_action") or "") in PLANNED_ACTIONS


def select_source_refs(source_refs: Sequence[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("max_sources_per_route must be > 0")
    return list(source_refs[:limit])


def promotion_status(summary: dict[str, Any]) -> tuple[str, str]:
    if int(summary.get("planned_job_count") or 0) <= 0 and int(summary.get("blocked_job_count") or 0) <= 0:
        return ("blocked_no_candidate_wave_jobs", "route gap plan did not produce candidate-wave jobs")
    if int(summary.get("blocked_job_count") or 0) > 0:
        return ("blocked_candidate_wave_plan_has_blockers", "candidate-wave plan has blocked jobs")
    return ("candidate_wave_plan_ready", "candidate-wave jobs are ready for a separate execution runner")


def plan_candidate_wave(
    *,
    route_gap_plan_manifest_json: Path,
    output_dir: Path,
    device: str = "cuda",
    resurrection_device: str = "cuda:0",
    max_sources_per_route: int = 1,
    registry: ModelRegistry | None = None,
) -> dict[str, Any]:
    if max_sources_per_route <= 0:
        raise ValueError("max_sources_per_route must be > 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / CANDIDATE_WAVE_PROGRESS_NAME
    progress_path.write_text("", encoding="utf-8")
    progress(
        progress_path,
        "start",
        route_gap_plan_manifest_json=str(route_gap_plan_manifest_json),
        device=str(device),
        resurrection_device=str(resurrection_device),
        max_sources_per_route=max_sources_per_route,
    )

    registry = registry or default_model_registry()
    route_gap_manifest, route_rows = route_gap_rows(route_gap_plan_manifest_json)
    source_refs = source_refs_from_shadow_batch(shadow_batch_manifest_path(route_gap_manifest))
    selected_route_rows = [row for row in route_rows if planned_route_gap_filter(row)]

    jobs: list[dict[str, Any]] = []
    for route_gap in selected_route_rows:
        for source_ref in select_source_refs(source_refs, limit=max_sources_per_route):
            jobs.append(
                build_candidate_wave_job(
                    route_gap=route_gap,
                    source_ref=source_ref,
                    output_dir=output_dir,
                    index=len(jobs) + 1,
                    device=str(device),
                    resurrection_device=str(resurrection_device),
                    registry=registry,
                )
            )

    summary = {
        "route_gap_plan_manifest_json": str(route_gap_plan_manifest_json),
        "shadow_batch_manifest_json": str(shadow_batch_manifest_path(route_gap_manifest)),
        "route_gap_count": len(route_rows),
        "planned_route_gap_count": len(selected_route_rows),
        "source_ref_count": len(source_refs),
        "max_sources_per_route": int(max_sources_per_route),
        "job_count": len(jobs),
        "planned_job_count": sum(1 for row in jobs if row["job_status"] == "planned"),
        "blocked_job_count": sum(1 for row in jobs if row["job_status"] != "planned"),
        "registered_model_job_count": sum(1 for row in jobs if row["execution_backend"] == "registered_model_candidate"),
        "direct_instrumental_resurrection_job_count": sum(
            1 for row in jobs if row["execution_backend"] == "direct_instrumental_resurrection_export"
        ),
        "unsupported_route_job_count": sum(1 for row in jobs if row["execution_backend"] == "unsupported_route"),
        "expected_acceleration_mode": "gpu_inference_candidate_generation",
        "audio_inference_started": False,
        "training_started": False,
        "release_ready": False,
    }
    status, reason = promotion_status(summary)
    fingerprint = stable_hash(
        {
            "route_gap_plan_manifest_json": str(route_gap_plan_manifest_json),
            "device": str(device),
            "resurrection_device": str(resurrection_device),
            "max_sources_per_route": int(max_sources_per_route),
            "jobs": [
                {
                    "route_id": row["route_id"],
                    "source_audio_path": row["source_audio_path"],
                    "command": row["command"],
                    "job_status": row["job_status"],
                }
                for row in jobs
            ],
        }
    )
    manifest = {
        "schema": "generic_controller_selector_candidate_wave_manifest.v1",
        "manifest_type": "generic_controller_selector_candidate_wave_plan",
        "route_gap_plan_manifest_json": str(route_gap_plan_manifest_json),
        "output_dir": str(output_dir),
        "fingerprint": fingerprint,
        "summary": summary,
        "promotion": {
            "status": status,
            "reason": reason,
            "can_execute_audio_inference": False,
            "can_start_training": False,
            "can_edit_model_registry": False,
            "next_required_step": "run_candidate_wave_execution_preflight_then_execute_bounded_jobs",
        },
        "training_started": False,
        "fine_tune_started": False,
        "audio_inference_started": False,
        "audio_render_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "candidate_wave_plan_not_for_listen_pack",
        "release_status": "planning_artifact_only",
        "release_ready": False,
        "safety": dict(SAFETY_FLAGS),
        "outputs": {
            "jobs_jsonl": str(output_dir / CANDIDATE_WAVE_JOBS_NAME),
            "manifest_json": str(output_dir / CANDIDATE_WAVE_MANIFEST_NAME),
            "progress_jsonl": str(progress_path),
        },
    }

    progress(progress_path, "write_outputs", **summary, status=status)
    write_jsonl(output_dir / CANDIDATE_WAVE_JOBS_NAME, jobs)
    write_json(output_dir / CANDIDATE_WAVE_MANIFEST_NAME, manifest)
    progress(progress_path, "finish", manifest_path=str(output_dir / CANDIDATE_WAVE_MANIFEST_NAME), fingerprint=fingerprint)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan controller/selector candidate-wave jobs from route gaps.")
    parser.add_argument("--route-gap-plan-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resurrection-device", default="cuda:0")
    parser.add_argument("--max-sources-per-route", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="plan_generic_controller_selector_candidate_wave.py",
        entrypoint_path="/app/tools/plan_generic_controller_selector_candidate_wave.py",
    )
    args = parse_args(argv)
    manifest = plan_candidate_wave(
        route_gap_plan_manifest_json=args.route_gap_plan_manifest_json,
        output_dir=args.output_dir,
        device=str(args.device),
        resurrection_device=str(args.resurrection_device),
        max_sources_per_route=int(args.max_sources_per_route),
    )
    print(json.dumps({"summary": manifest["summary"], "promotion": manifest["promotion"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
