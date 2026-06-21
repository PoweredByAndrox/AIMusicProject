from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_no_reference_vocal_minus_master_manifest import (
    build_manifest as build_master_manifest,
    build_summary as build_master_summary,
)
from tools.build_no_reference_full_stem_pack_manifest import (
    REQUIRED_NONVOCAL_STEMS,
    build_manifest as build_full_stem_pack_manifest,
    build_summary as build_full_stem_pack_summary,
)
from tools.docker_runtime import require_docker_runtime
from tools.run_no_reference_minus_delivery_pipeline import (
    SELF_JUDGE_MANIFEST_NAME,
    discover_self_judge_manifests,
    run_minus_delivery_pipeline,
)
from tools.run_no_reference_vocal_delivery_pipeline import run_vocal_delivery_pipeline
from tools.run_no_reference_stem_semantic_batch import run_batch as run_stem_semantic_batch
from tools.run_testing_stuff_music_corpus_job_matrix import EXECUTION_MANIFEST_NAME, execute_job_matrix

PROGRESS_JSONL_NAME = "testing_stuff_music_corpus_vocal_minus_master_execution_progress.jsonl"
STEPS_JSONL_NAME = "testing_stuff_music_corpus_vocal_minus_master_execution_steps.jsonl"
SUMMARY_JSON_NAME = "testing_stuff_music_corpus_vocal_minus_master_execution_summary.json"
MANIFEST_JSON_NAME = "testing_stuff_music_corpus_vocal_minus_master_execution_manifest.json"

VOCAL_JUDGE_NAME = "no_reference_vocal_judge.jsonl"
VOCAL_JUDGE_NAMES = (VOCAL_JUDGE_NAME, "no_reference_windowed_judge_results.jsonl")
VOCAL_SCORE_MATRIX_NAME = "no_reference_vocal_candidate_score_matrix.json"
MINUS_SCORE_MATRIX_NAME = "no_reference_minus_candidate_score_matrix.json"
STEM_SCORE_MATRIX_NAME = "no_reference_stem_candidate_score_matrix.json"
PAIR_COMPATIBILITY_NAMES = (
    "no_reference_pair_compatibility.jsonl",
    "no_reference_vocal_minus_pair_compatibility.jsonl",
)
DEFAULT_GENERATION_STEMS = ("vocal", "minus", "melody", "drums", "bass", "other")
SUPPORTED_GATES = {
    "run_corpus_job_matrix_for_track",
    "run_vocal_delivery_pipeline",
    "run_minus_delivery_pipeline",
    "run_pair_compatibility_and_master_manifest",
    "complete_nonvocal_stem_score_matrices",
    "build_full_stem_pack_manifest",
}
SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "release_ready": False,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


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


def source_ref(path: Path | None) -> dict[str, str]:
    return {"path": str(path)} if path is not None else {"path": ""}


def path_matches_track(path: Path, track_id: str) -> bool:
    return track_id in [part for part in path.parts]


def json_file_track_ids(path: Path) -> set[str]:
    try:
        if path.suffix.lower() == ".jsonl":
            return {str(row.get("track_id") or "") for row in iter_jsonl(path) if row.get("track_id")}
        payload = read_json(path)
    except Exception:
        return set()
    ids = set()
    if isinstance(payload, dict):
        if payload.get("track_id"):
            ids.add(str(payload.get("track_id")))
        rows = payload.get("rows")
        if isinstance(rows, list):
            ids.update(str(row.get("track_id")) for row in rows if isinstance(row, dict) and row.get("track_id"))
    return ids


def discover_named_files(roots: Sequence[Path], filenames: Sequence[str]) -> list[Path]:
    names = set(filenames)
    discovered: list[Path] = []
    for root in roots:
        if root.is_file() and root.name in names:
            discovered.append(root)
        elif root.is_dir():
            for name in sorted(names):
                discovered.extend(sorted(root.rglob(name)))
    return sorted({path.resolve(): path for path in discovered}.values(), key=lambda path: str(path))


def select_file_for_track(roots: Sequence[Path], filenames: Sequence[str], track_id: str) -> Path | None:
    candidates = discover_named_files(roots, filenames)
    for path in candidates:
        if path_matches_track(path, track_id):
            return path
    for path in candidates:
        if track_id in json_file_track_ids(path):
            return path
    return candidates[0] if len(candidates) == 1 else None


def select_self_judge_manifests_for_track(roots: Sequence[Path], track_id: str) -> list[Path]:
    candidates = discover_self_judge_manifests(roots)
    path_matches = [path for path in candidates if path_matches_track(path, track_id)]
    if path_matches:
        return path_matches
    content_matches = [path for path in candidates if track_id in json_file_track_ids(path)]
    return content_matches


def selected_plan_rows(plan: dict[str, Any], max_steps: int) -> list[dict[str, Any]]:
    if max_steps <= 0:
        raise ValueError("--max-steps must be a positive integer")
    rows = plan.get("tracks")
    if not isinstance(rows, list):
        return []
    actionable = [
        row
        for row in rows
        if isinstance(row, dict) and str(row.get("next_gate") or "") in SUPPORTED_GATES
    ]
    return actionable[:max_steps]


def base_step(row: dict[str, Any], *, index: int, output_root: Path) -> dict[str, Any]:
    track_id = str(row.get("track_id") or "")
    next_gate = str(row.get("next_gate") or "")
    output_dir = output_root / next_gate / track_id
    if next_gate == "run_corpus_job_matrix_for_track":
        mode = "gpu_inference_then_cpu_dsp_scoring"
    elif next_gate == "complete_nonvocal_stem_score_matrices":
        mode = "external_only_cpu_dsp_scoring"
    else:
        mode = "external_only_cpu_scoring"
    return {
        "schema": "testing_stuff_music_corpus_vocal_minus_master_execution_step.v1",
        "step_index": index,
        "track_id": track_id,
        "source_audio_path": str(row.get("source_audio_path") or ""),
        "next_gate": next_gate,
        "output_dir": str(output_dir),
        "step_status": "planned",
        "execution_started": False,
        "expected_acceleration_mode": mode,
        "gpu_inference_required": next_gate == "run_corpus_job_matrix_for_track",
        "gpu_inference_skipped": next_gate != "run_corpus_job_matrix_for_track",
        "gpu_inference_skip_reason": "external_delivery_or_contract_gate" if next_gate != "run_corpus_job_matrix_for_track" else "",
        "delivery_allowed": False,
        "release_ready": False,
    }


def output_manifest_path(step: dict[str, Any]) -> Path:
    output_dir = Path(str(step["output_dir"]))
    gate = str(step["next_gate"])
    if gate == "run_vocal_delivery_pipeline":
        return output_dir / "no_reference_vocal_listen_preview_manifest.json"
    if gate == "run_minus_delivery_pipeline":
        return output_dir / "no_reference_minus_listen_preview_manifest.json"
    if gate == "run_pair_compatibility_and_master_manifest":
        return output_dir / "no_reference_vocal_minus_master_manifest.json"
    if gate == "build_full_stem_pack_manifest":
        return output_dir / "no_reference_full_stem_pack_manifest.json"
    if gate == "complete_nonvocal_stem_score_matrices":
        return output_dir / str(step["track_id"]) / "stem_score_matrix" / STEM_SCORE_MATRIX_NAME
    return output_dir / EXECUTION_MANIFEST_NAME


def blocked_step(step: dict[str, Any], status: str, reason: str, **extra: Any) -> dict[str, Any]:
    step.update(
        {
            "step_status": status,
            "dominant_failure_reason": reason,
            "execution_started": False,
            "delivery_allowed": False,
            "release_ready": False,
            **extra,
        }
    )
    return step


def run_generation_step(
    *,
    step: dict[str, Any],
    job_matrix_jsonl: Path | None,
    max_jobs_per_track: int,
    generation_stems: Sequence[str],
    job_matrix_executor: Callable[..., dict[str, Any]],
    runtime_root: Path,
) -> dict[str, Any]:
    if job_matrix_jsonl is None:
        return blocked_step(step, "blocked_missing_job_matrix", "missing_job_matrix_jsonl")
    generation = job_matrix_executor(
        job_matrix_jsonl=job_matrix_jsonl,
        output_root=Path(str(step["output_dir"])),
        max_jobs=max_jobs_per_track,
        stems=list(generation_stems),
        track_ids=[str(step["track_id"])],
        dry_run=False,
        runtime_root=runtime_root,
    )
    summary = generation.get("summary") if isinstance(generation.get("summary"), dict) else {}
    status = "completed" if int(summary.get("completed_jobs") or 0) > 0 else "blocked_generation_no_completed_jobs"
    step.update(
        {
            "step_status": status,
            "execution_started": bool(summary.get("execution_started")),
            "generation_summary": summary,
            "output_manifest_json": str(Path(str(step["output_dir"])) / EXECUTION_MANIFEST_NAME),
        }
    )
    return step


def run_vocal_step(
    *,
    step: dict[str, Any],
    vocal_judge_roots: Sequence[Path],
    vocal_score_matrix_roots: Sequence[Path],
    vocal_delivery_runner: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    track_id = str(step["track_id"])
    vocal_judge_jsonl = select_file_for_track(vocal_judge_roots, VOCAL_JUDGE_NAMES, track_id)
    if vocal_judge_jsonl is None:
        return blocked_step(step, "blocked_missing_vocal_judge", "missing_vocal_judge_jsonl")
    matrix_json = select_file_for_track(vocal_score_matrix_roots, (VOCAL_SCORE_MATRIX_NAME,), track_id)
    if matrix_json is None:
        return blocked_step(
            step,
            "blocked_missing_vocal_score_matrix",
            "missing_vocal_candidate_score_matrix",
            vocal_judge_jsonl=str(vocal_judge_jsonl),
        )
    output_dir = Path(str(step["output_dir"]))
    manifest = vocal_delivery_runner(
        vocal_judge_jsonl=vocal_judge_jsonl,
        output_dir=output_dir,
        base_candidate_id=None,
        candidate_score_matrix_json=matrix_json,
    )
    manifest_path = output_manifest_path(step)
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    step.update(
        {
            "step_status": "completed" if manifest.get("delivery_allowed") is True else "blocked_vocal_delivery_not_accepted",
            "execution_started": True,
            "vocal_judge_jsonl": str(vocal_judge_jsonl),
            "candidate_score_matrix_json": str(matrix_json),
            "output_manifest_json": str(manifest_path),
            "delivery_status": manifest.get("delivery_status"),
            "presentation_status": manifest.get("presentation_status"),
            "release_status": manifest.get("release_status"),
            "delivery_allowed": False,
            "release_ready": False,
        }
    )
    return step


def run_minus_step(
    *,
    step: dict[str, Any],
    minus_self_judge_roots: Sequence[Path],
    minus_score_matrix_roots: Sequence[Path],
    minus_delivery_runner: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    track_id = str(step["track_id"])
    self_judges = select_self_judge_manifests_for_track(minus_self_judge_roots, track_id)
    if not self_judges:
        return blocked_step(step, "blocked_missing_minus_self_judge", "missing_minus_self_judge_manifest")
    matrix_json = select_file_for_track(minus_score_matrix_roots, (MINUS_SCORE_MATRIX_NAME,), track_id)
    if matrix_json is None:
        return blocked_step(
            step,
            "blocked_missing_minus_score_matrix",
            "missing_minus_candidate_score_matrix",
            self_judge_manifest_count=len(self_judges),
        )
    source_audio = Path(str(step["source_audio_path"])) if step.get("source_audio_path") else None
    output_dir = Path(str(step["output_dir"]))
    manifest = minus_delivery_runner(
        self_judge_manifest_paths=self_judges,
        output_dir=output_dir,
        base_candidate_id=None,
        candidate_score_matrix_json=matrix_json,
        source_audio_path=source_audio,
    )
    manifest_path = output_manifest_path(step)
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    step.update(
        {
            "step_status": "completed" if manifest.get("delivery_allowed") is True else "blocked_minus_delivery_not_accepted",
            "execution_started": True,
            "self_judge_manifest_paths": [str(path) for path in self_judges],
            "candidate_score_matrix_json": str(matrix_json),
            "output_manifest_json": str(manifest_path),
            "delivery_status": manifest.get("delivery_status"),
            "presentation_status": manifest.get("presentation_status"),
            "release_status": manifest.get("release_status"),
            "delivery_allowed": False,
            "release_ready": False,
        }
    )
    return step


def run_master_step(
    *,
    step: dict[str, Any],
    plan_row: dict[str, Any],
    pair_compatibility_roots: Sequence[Path],
    master_manifest_builder: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    vocal_manifest = Path(str(plan_row.get("vocal_manifest_path") or ""))
    minus_manifest = Path(str(plan_row.get("minus_manifest_path") or ""))
    if not vocal_manifest.exists():
        return blocked_step(step, "blocked_missing_vocal_manifest", "missing_vocal_manifest_for_master")
    if not minus_manifest.exists():
        return blocked_step(step, "blocked_missing_minus_manifest", "missing_minus_manifest_for_master")
    pair_jsonl = select_file_for_track(pair_compatibility_roots, PAIR_COMPATIBILITY_NAMES, str(step["track_id"]))
    if pair_jsonl is None:
        return blocked_step(step, "blocked_missing_pair_compatibility", "missing_pair_compatibility_jsonl")
    output_dir = Path(str(step["output_dir"]))
    manifest_path = output_manifest_path(step)
    summary_md = output_dir / "no_reference_vocal_minus_master_summary.md"
    manifest = master_manifest_builder(
        vocal_manifest_json=vocal_manifest,
        minus_manifest_json=minus_manifest,
        pair_compatibility_jsonl=pair_jsonl,
    )
    write_json(manifest_path, manifest)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text(build_master_summary(manifest), encoding="utf-8")
    step.update(
        {
            "step_status": "completed" if manifest.get("delivery_allowed") is True else "blocked_master_manifest_not_accepted",
            "execution_started": True,
            "vocal_manifest_json": str(vocal_manifest),
            "minus_manifest_json": str(minus_manifest),
            "pair_compatibility_jsonl": str(pair_jsonl),
            "output_manifest_json": str(manifest_path),
            "summary_md": str(summary_md),
            "delivery_status": manifest.get("delivery_status"),
            "presentation_status": manifest.get("presentation_status"),
            "release_status": manifest.get("release_status"),
            "delivery_allowed": False,
            "release_ready": False,
        }
    )
    return step


def run_full_stem_pack_step(
    *,
    step: dict[str, Any],
    plan_row: dict[str, Any],
    stem_score_matrix_roots: Sequence[Path],
    full_stem_pack_builder: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    master_manifest = Path(str(plan_row.get("master_manifest_path") or ""))
    if not master_manifest.exists():
        return blocked_step(step, "blocked_missing_master_manifest", "missing_master_manifest_for_full_stem_pack")
    matrix_json = select_file_for_track(stem_score_matrix_roots, (STEM_SCORE_MATRIX_NAME,), str(step["track_id"]))
    if matrix_json is None:
        return blocked_step(
            step,
            "blocked_missing_stem_score_matrix",
            "missing_stem_candidate_score_matrix",
            master_manifest_json=str(master_manifest),
        )
    output_dir = Path(str(step["output_dir"]))
    manifest_path = output_manifest_path(step)
    summary_md = output_dir / "no_reference_full_stem_pack_summary.md"
    manifest = full_stem_pack_builder(
        master_manifest_json=master_manifest,
        stem_score_matrix_json=matrix_json,
    )
    write_json(manifest_path, manifest)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text(build_full_stem_pack_summary(manifest), encoding="utf-8")
    step.update(
        {
            "step_status": "completed" if manifest.get("delivery_allowed") is True else "blocked_full_stem_pack_not_accepted",
            "execution_started": True,
            "master_manifest_json": str(master_manifest),
            "stem_score_matrix_json": str(matrix_json),
            "output_manifest_json": str(manifest_path),
            "summary_md": str(summary_md),
            "delivery_status": manifest.get("delivery_status"),
            "presentation_status": manifest.get("presentation_status"),
            "release_status": manifest.get("release_status"),
            "accepted_stem_count": int(manifest.get("accepted_stem_count") or 0),
            "delivery_allowed": False,
            "release_ready": False,
        }
    )
    return step


def matrix_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    rows = matrix.get("ranked_candidates")
    if not isinstance(rows, list):
        rows = matrix.get("rows")
    return [row for row in rows or [] if isinstance(row, dict)]


def row_matches_track(row: dict[str, Any], track_id: str) -> bool:
    row_track_id = str(row.get("track_id") or "")
    return not row_track_id or row_track_id == track_id


def row_is_accepted_nonvocal(row: dict[str, Any]) -> bool:
    blockers = row.get("stem_semantic_blockers")
    if blockers is None:
        blockers = []
    if not isinstance(blockers, list):
        blockers = [blockers]
    return (
        row.get("delivery_allowed") is True
        and row.get("semantic_quality_verified") is True
        and str(row.get("selected_audio_path") or "")
        and row.get("selected_audio_exists") is not False
        and not [str(item) for item in blockers if str(item)]
        and row.get("release_ready") is not True
        and row.get("can_mark_release_ready") is not True
        and row.get("can_publish_as_release") is not True
    )


def accepted_nonvocal_stems_from_matrix(matrix_json: Path, track_id: str) -> tuple[list[str], list[str]]:
    if not matrix_json.exists():
        return [], [f"missing_matrix:{stem}" for stem in REQUIRED_NONVOCAL_STEMS]
    try:
        matrix = read_json(matrix_json)
    except (OSError, ValueError, json.JSONDecodeError):
        return [], [f"invalid_matrix:{stem}" for stem in REQUIRED_NONVOCAL_STEMS]
    rows = matrix_rows(matrix)
    accepted: list[str] = []
    missing: list[str] = []
    for stem in REQUIRED_NONVOCAL_STEMS:
        stem_rows = [
            row
            for row in rows
            if str(row.get("stem_type") or "") == stem and row_matches_track(row, track_id)
        ]
        if any(row_is_accepted_nonvocal(row) for row in stem_rows):
            accepted.append(stem)
        else:
            missing.append(stem)
    return accepted, missing


def track_summary_from_semantic_batch(summary: dict[str, Any], track_id: str) -> dict[str, Any]:
    tracks = summary.get("tracks")
    if not isinstance(tracks, list):
        return {}
    for track in tracks:
        if isinstance(track, dict) and str(track.get("track_id") or "") == track_id:
            return track
    return {}


def run_nonvocal_score_matrix_step(
    *,
    step: dict[str, Any],
    stem_listen_manifest_roots: Sequence[Path],
    stem_semantic_batch_runner: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    track_id = str(step["track_id"])
    if not stem_listen_manifest_roots:
        return blocked_step(step, "blocked_missing_stem_listen_manifest_root", "missing_stem_listen_manifest_root")
    output_dir = Path(str(step["output_dir"]))
    semantic_summary = stem_semantic_batch_runner(
        listen_manifest_roots=list(stem_listen_manifest_roots),
        output_root=output_dir,
        track_ids=[track_id],
    )
    track_summary = track_summary_from_semantic_batch(semantic_summary, track_id)
    if not track_summary:
        return blocked_step(
            step,
            "blocked_missing_stem_listen_manifests",
            "semantic_batch_found_no_track_manifests",
            execution_started=True,
            semantic_batch_summary=semantic_summary,
        )
    matrix_json_text = str(track_summary.get("matrix_json") or "")
    if not matrix_json_text:
        return blocked_step(
            step,
            "blocked_missing_stem_score_matrix",
            "semantic_batch_missing_matrix_json",
            execution_started=True,
            semantic_batch_summary=semantic_summary,
        )
    matrix_json = Path(matrix_json_text)
    accepted_stems, missing_stems = accepted_nonvocal_stems_from_matrix(matrix_json, track_id)
    if missing_stems:
        return blocked_step(
            step,
            "blocked_nonvocal_stem_score_matrix_not_accepted",
            "missing_or_unaccepted_required_nonvocal_stems:" + ",".join(missing_stems),
            execution_started=True,
            semantic_batch_summary=semantic_summary,
            stem_score_matrix_json=str(matrix_json),
            output_manifest_json=str(matrix_json),
            accepted_nonvocal_stems=accepted_stems,
            missing_nonvocal_stems=missing_stems,
        )
    step.update(
        {
            "step_status": "completed",
            "execution_started": True,
            "semantic_batch_summary": semantic_summary,
            "stem_score_matrix_json": str(matrix_json),
            "output_manifest_json": str(matrix_json),
            "accepted_nonvocal_stems": accepted_stems,
            "missing_nonvocal_stems": [],
            "delivery_allowed": False,
            "release_ready": False,
        }
    )
    return step


def build_summary(
    *,
    plan_json: Path,
    output_root: Path,
    selected_rows: Sequence[dict[str, Any]],
    steps: Sequence[dict[str, Any]],
    execute: bool,
    started: float,
) -> dict[str, Any]:
    counts = Counter(str(step.get("step_status") or "unknown") for step in steps)
    return {
        "schema": "testing_stuff_music_corpus_vocal_minus_master_execution_summary.v1",
        "source_plan_json": source_ref(plan_json),
        "output_root": str(output_root),
        "selected_step_count": len(selected_rows),
        "executed_step_count": sum(1 for step in steps if bool(step.get("execution_started"))),
        "execution_started": any(bool(step.get("execution_started")) for step in steps),
        "execute": bool(execute),
        "step_status_counts": dict(sorted(counts.items())),
        "selected_track_ids": [str(row.get("track_id") or "") for row in selected_rows],
        "selected_next_gates": [str(row.get("next_gate") or "") for row in selected_rows],
        "expected_acceleration_modes": sorted(
            {str(step.get("expected_acceleration_mode") or "") for step in steps if step.get("expected_acceleration_mode")}
        ),
        "gpu_inference_required": any(bool(step.get("gpu_inference_required")) for step in steps),
        "delivery_allowed": False,
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "safety": dict(SAFETY_FLAGS),
        "steps": list(steps),
        "elapsed_sec": round(time.time() - started, 6),
    }


def persist_outputs(output_root: Path, summary: dict[str, Any], steps: Sequence[dict[str, Any]]) -> dict[str, Any]:
    write_jsonl(output_root / STEPS_JSONL_NAME, steps)
    write_json(output_root / SUMMARY_JSON_NAME, summary)
    manifest = {
        "schema": "testing_stuff_music_corpus_vocal_minus_master_execution_manifest.v1",
        "manifest_type": "testing_stuff_music_corpus_vocal_minus_master_execution",
        "execution_status": "completed" if steps else "no_selected_steps",
        "delivery_allowed": False,
        "presentation_status": "diagnostic_corpus_execution_not_for_listen_pack",
        "release_status": "not_release_ready",
        "release_ready": False,
        "summary": {
            key: summary.get(key)
            for key in (
                "selected_step_count",
                "executed_step_count",
                "execution_started",
                "step_status_counts",
                "selected_track_ids",
                "selected_next_gates",
                "expected_acceleration_modes",
                "gpu_inference_required",
            )
        },
        "summary_json": str(output_root / SUMMARY_JSON_NAME),
        "steps_jsonl": str(output_root / STEPS_JSONL_NAME),
        "progress_jsonl": str(output_root / PROGRESS_JSONL_NAME),
        "safety": dict(SAFETY_FLAGS),
    }
    write_json(output_root / MANIFEST_JSON_NAME, manifest)
    return manifest


def run_plan(
    *,
    plan_json: Path,
    output_root: Path,
    max_steps: int,
    execute: bool = False,
    job_matrix_jsonl: Path | None = None,
    max_jobs_per_track: int = 6,
    generation_stems: Sequence[str] = DEFAULT_GENERATION_STEMS,
    vocal_judge_roots: Sequence[Path] = (),
    vocal_score_matrix_roots: Sequence[Path] = (),
    minus_self_judge_roots: Sequence[Path] = (),
    minus_score_matrix_roots: Sequence[Path] = (),
    pair_compatibility_roots: Sequence[Path] = (),
    stem_listen_manifest_roots: Sequence[Path] = (),
    stem_score_matrix_roots: Sequence[Path] = (),
    fail_fast: bool = False,
    runtime_root: Path = Path("/app"),
    job_matrix_executor: Callable[..., dict[str, Any]] = execute_job_matrix,
    vocal_delivery_runner: Callable[..., dict[str, Any]] = run_vocal_delivery_pipeline,
    minus_delivery_runner: Callable[..., dict[str, Any]] = run_minus_delivery_pipeline,
    master_manifest_builder: Callable[..., dict[str, Any]] = build_master_manifest,
    full_stem_pack_builder: Callable[..., dict[str, Any]] = build_full_stem_pack_manifest,
    stem_semantic_batch_runner: Callable[..., dict[str, Any]] = run_stem_semantic_batch,
) -> dict[str, Any]:
    started = time.time()
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / PROGRESS_JSONL_NAME
    plan = read_json(plan_json)
    selected_rows = selected_plan_rows(plan, max_steps)
    append_jsonl(
        progress_path,
        {
            "event": "corpus_vocal_minus_master_execution_start",
            "plan_json": str(plan_json),
            "selected_step_count": len(selected_rows),
            "execute": execute,
            "max_steps": max_steps,
            "max_jobs_per_track": max_jobs_per_track,
        },
    )
    steps: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows, start=1):
        step = base_step(row, index=index, output_root=output_root)
        append_jsonl(
            progress_path,
            {
                "event": "step_start",
                "track_id": step["track_id"],
                "next_gate": step["next_gate"],
                "execute": execute,
                "expected_acceleration_mode": step["expected_acceleration_mode"],
                "gpu_inference_required": step["gpu_inference_required"],
                "gpu_inference_skipped": step["gpu_inference_skipped"],
            },
        )
        if not execute:
            step["step_status"] = "dry_run"
        elif step["next_gate"] == "run_corpus_job_matrix_for_track":
            step = run_generation_step(
                step=step,
                job_matrix_jsonl=job_matrix_jsonl,
                max_jobs_per_track=max_jobs_per_track,
                generation_stems=generation_stems,
                job_matrix_executor=job_matrix_executor,
                runtime_root=runtime_root,
            )
        elif step["next_gate"] == "run_vocal_delivery_pipeline":
            step = run_vocal_step(
                step=step,
                vocal_judge_roots=vocal_judge_roots,
                vocal_score_matrix_roots=vocal_score_matrix_roots,
                vocal_delivery_runner=vocal_delivery_runner,
            )
        elif step["next_gate"] == "run_minus_delivery_pipeline":
            step = run_minus_step(
                step=step,
                minus_self_judge_roots=minus_self_judge_roots,
                minus_score_matrix_roots=minus_score_matrix_roots,
                minus_delivery_runner=minus_delivery_runner,
            )
        elif step["next_gate"] == "run_pair_compatibility_and_master_manifest":
            step = run_master_step(
                step=step,
                plan_row=row,
                pair_compatibility_roots=pair_compatibility_roots,
                master_manifest_builder=master_manifest_builder,
            )
        elif step["next_gate"] == "complete_nonvocal_stem_score_matrices":
            step = run_nonvocal_score_matrix_step(
                step=step,
                stem_listen_manifest_roots=stem_listen_manifest_roots,
                stem_semantic_batch_runner=stem_semantic_batch_runner,
            )
        elif step["next_gate"] == "build_full_stem_pack_manifest":
            step = run_full_stem_pack_step(
                step=step,
                plan_row=row,
                stem_score_matrix_roots=stem_score_matrix_roots,
                full_stem_pack_builder=full_stem_pack_builder,
            )
        steps.append(step)
        append_jsonl(
            progress_path,
            {
                "event": "step_end",
                "track_id": step["track_id"],
                "next_gate": step["next_gate"],
                "step_status": step["step_status"],
                "execution_started": step["execution_started"],
                "output_manifest_json": step.get("output_manifest_json", ""),
            },
        )
        if fail_fast and execute and step["step_status"] != "completed":
            break
    summary = build_summary(
        plan_json=plan_json,
        output_root=output_root,
        selected_rows=selected_rows[: len(steps)],
        steps=steps,
        execute=execute,
        started=started,
    )
    manifest = persist_outputs(output_root, summary, steps)
    summary["manifest_json"] = str(output_root / MANIFEST_JSON_NAME)
    summary["manifest"] = manifest
    append_jsonl(
        progress_path,
        {
            "event": "corpus_vocal_minus_master_execution_end",
            "selected_step_count": summary["selected_step_count"],
            "executed_step_count": summary["executed_step_count"],
            "step_status_counts": summary["step_status_counts"],
            "release_ready": False,
            "elapsed_sec": summary["elapsed_sec"],
        },
    )
    write_json(output_root / SUMMARY_JSON_NAME, summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bounded next gates from a testing_stuff_music vocal/minus/master corpus plan."
    )
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--job-matrix-jsonl", type=Path)
    parser.add_argument("--max-jobs-per-track", type=int, default=6)
    parser.add_argument("--generation-stem", action="append", default=[])
    parser.add_argument("--vocal-judge-root", type=Path, action="append", default=[])
    parser.add_argument("--vocal-score-matrix-root", type=Path, action="append", default=[])
    parser.add_argument("--minus-self-judge-root", type=Path, action="append", default=[])
    parser.add_argument("--minus-score-matrix-root", type=Path, action="append", default=[])
    parser.add_argument("--pair-compatibility-root", type=Path, action="append", default=[])
    parser.add_argument("--stem-listen-manifest-root", type=Path, action="append", default=[])
    parser.add_argument("--stem-score-matrix-root", type=Path, action="append", default=[])
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--runtime-root", type=Path, default=Path("/app"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_testing_stuff_music_corpus_vocal_minus_master_plan.py",
        entrypoint_path="/app/tools/run_testing_stuff_music_corpus_vocal_minus_master_plan.py",
    )
    args = parse_args(argv)
    summary = run_plan(
        plan_json=args.plan_json,
        output_root=args.output_root,
        max_steps=args.max_steps,
        execute=bool(args.execute),
        job_matrix_jsonl=args.job_matrix_jsonl,
        max_jobs_per_track=int(args.max_jobs_per_track),
        generation_stems=tuple(args.generation_stem or DEFAULT_GENERATION_STEMS),
        vocal_judge_roots=args.vocal_judge_root,
        vocal_score_matrix_roots=args.vocal_score_matrix_root,
        minus_self_judge_roots=args.minus_self_judge_root,
        minus_score_matrix_roots=args.minus_score_matrix_root,
        pair_compatibility_roots=args.pair_compatibility_root,
        stem_listen_manifest_roots=args.stem_listen_manifest_root,
        stem_score_matrix_roots=args.stem_score_matrix_root,
        fail_fast=bool(args.fail_fast),
        runtime_root=args.runtime_root,
    )
    print(
        json.dumps(
            {
                "selected_step_count": summary["selected_step_count"],
                "executed_step_count": summary["executed_step_count"],
                "step_status_counts": summary["step_status_counts"],
                "selected_track_ids": summary["selected_track_ids"],
                "selected_next_gates": summary["selected_next_gates"],
                "release_ready": summary["release_ready"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not any(str(status).startswith("blocked_") for status in summary["step_status_counts"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
