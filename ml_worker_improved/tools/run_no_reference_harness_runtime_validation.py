from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_no_reference_ml_dataset import stable_hash, stable_unique, write_jsonl  # noqa: E402
from tools.optimize_no_reference_harness import merge_missing_baseline_defaults  # noqa: E402
from tools.replay_no_reference_harness_candidate import (  # noqa: E402
    default_harness_candidate,
    validate_harness_candidate,
)


ALLOWED_DECISIONS = {
    "keep_baseline",
    "candidate_quality_improved",
    "candidate_efficiency_only",
    "candidate_no_op",
    "candidate_specialist_only",
    "candidate_rejected_runtime",
    "candidate_rejected_safety",
    "candidate_inconclusive",
}

ALLOWED_RUNTIME_STATUSES = {
    "accepted_pair_ready",
    "manifest_preview_ready",
    "audio_preview_ready",
    "preview_render_skipped",
    "preview_render_failed",
    "preview_validation_failed",
    "preview_blocked_safety_contract",
    "preview_ready",
    "accepted_for_listen_pack",
    "blocked_hard_case",
    "needs_more_candidates",
    "needs_pair_reconciliation",
    "segment_blend_needed",
    "failed_safety_contract",
}

RUNTIME_ARTIFACTS = {
    "run_manifest": "run_manifest.json",
    "runtime_summary": "runtime_summary.md",
    "candidate_pool": "candidate_pool.jsonl",
    "candidate_wave_summary": "candidates/candidate_wave_summary.json",
    "candidate_wave_summary_md": "candidates/candidate_wave_summary.md",
    "candidate_wave_jobs": "candidates/candidate_wave_jobs.jsonl",
    "candidate_wave_failures": "candidates/candidate_wave_failures.jsonl",
    "vocal_judge": "vocal_windowed_judge.jsonl",
    "minus_judge": "minus_windowed_judge.jsonl",
    "learned_critic_scores": "learned_critic_scores.jsonl",
    "selected_candidates": "selected_candidates.jsonl",
    "pair_compatibility": "pair_compatibility.jsonl",
    "preview_manifest": "preview_manifest.json",
    "rendered_preview_audio": "rendered_preview_audio.jsonl",
    "preview_render_contract": "preview_render_contract.json",
    "preview_render_summary": "preview_render_summary.md",
    "post_render_judge": "post_render_judge.jsonl",
    "learning_outcome": "learning_outcome.jsonl",
    "music_intelligence_input_manifest": "music_intelligence_input_manifest.json",
}

SAFETY_FLAG_KEYS = (
    "release_ready",
    "reference_required",
    "human_csv_required",
    "training_started",
    "separator_training_started",
    "model_downloaded",
    "model_registry_edited",
    "thresholds_relaxed",
    "cleanup_routes_added",
)

BLOCKER_FIELDS = (
    "blockers",
    "script_blockers",
    "minus_blockers",
    "vocal_blockers",
    "pair_blockers",
    "safety_blockers",
    "failure_reasons",
    "quality_blockers",
)

QUALITY_POSITIVE_OBJECTIVES = (
    "audio_preview_ready_count",
    "rendered_preview_audio_count",
    "accepted_minus_count",
    "pair_compatible_count",
    "analysis_ready_stems_count",
)
QUALITY_NEGATIVE_OBJECTIVES = (
    "severe_phase_collapse_count",
    "severe_vocal_bleed_in_minus_count",
    "blocked_stems_count",
)
EFFICIENCY_OBJECTIVES = (
    "runtime_sec",
    "estimated_runtime_sec",
)


RuntimeRunner = Callable[[list[str], Path, dict[str, Any]], int]


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8").strip()
        return json.loads(raw) if raw else {}
    except (json.JSONDecodeError, OSError):
        return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def safe_slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    collapsed = "_".join(part for part in cleaned.split("_") if part)
    return collapsed[:80] or "track"


def candidate_from_row(row: dict[str, Any]) -> dict[str, Any]:
    config = row.get("candidate_config")
    if isinstance(config, dict):
        candidate = dict(config)
    else:
        candidate = dict(row)
    return merge_missing_baseline_defaults(candidate)


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def load_baseline_candidate(pareto_frontier_jsonl: Path) -> dict[str, Any] | None:
    eval_jsonl = pareto_frontier_jsonl.with_name("harness_candidate_eval.jsonl")
    for row in load_jsonl_rows(eval_jsonl):
        candidate = candidate_from_row(row)
        if candidate.get("mutation_type") == "baseline_keep" or candidate.get("harness_candidate_id") == "baseline_safe_harness":
            return candidate
    return None


def load_pareto_candidates(pareto_frontier_jsonl: Path, max_pareto_candidates: int) -> list[dict[str, Any]]:
    rows = load_jsonl_rows(pareto_frontier_jsonl)
    candidates: list[dict[str, Any]] = []
    for row in rows:
        candidate = candidate_from_row(row)
        if candidate.get("mutation_type") == "baseline_keep":
            continue
        candidates.append(candidate)
        if len(candidates) >= max_pareto_candidates:
            break
    return candidates


def should_enable_hard_case_route(candidate: dict[str, Any]) -> bool:
    stage_graph = candidate.get("stage_graph") if isinstance(candidate.get("stage_graph"), dict) else {}
    optional = {str(item) for item in stage_graph.get("optional_mutations") or []}
    return "enable_optional_hard_case_route" in optional and int(candidate.get("max_hard_case_extra_jobs") or 0) > 0


def bounded_runtime_seconds(candidate: dict[str, Any], max_runtime_sec_per_track: int) -> int:
    policy = candidate.get("timeout_budget_policy") if isinstance(candidate.get("timeout_budget_policy"), dict) else {}
    configured = int(policy.get("max_runtime_sec") or max_runtime_sec_per_track)
    return max(1, min(configured, int(max_runtime_sec_per_track)))


def map_candidate_to_runtime_args(
    *,
    candidate: dict[str, Any],
    input_audio: Path,
    output_dir: Path,
    model_dir: Path,
    max_runtime_sec_per_track: int,
    require_audio_preview_ready: bool = False,
) -> list[str]:
    candidate_profile = str(candidate.get("candidate_profile") or "safe")
    diversity_profile = str(candidate.get("candidate_diversity_profile") or candidate_profile)
    runtime_profile = "balanced" if candidate_profile in {"balanced", "exploration"} else "safe"
    max_candidates = max(1, int(candidate.get("max_candidates") or 1))
    max_hard_case_extra_jobs = max(0, int(candidate.get("max_hard_case_extra_jobs") or 0))
    hard_case_enabled = should_enable_hard_case_route(candidate)
    runtime_seconds = bounded_runtime_seconds(candidate, max_runtime_sec_per_track)
    return [
        sys.executable,
        "tools/run_autonomous_no_reference_pipeline.py",
        "--input-audio",
        str(input_audio),
        "--output-dir",
        str(output_dir),
        "--model-dir",
        str(model_dir),
        "--profile",
        runtime_profile,
        "--generate-candidates",
        "true",
        "--candidate-profile",
        candidate_profile,
        "--candidate-execution-mode",
        "auto",
        "--allow-shape-normalization",
        "true",
        "--candidate-diversity-profile",
        diversity_profile,
        "--max-candidates",
        str(max_candidates),
        "--max-candidate-runtime-sec",
        str(runtime_seconds),
        "--continue-on-candidate-failure",
        "true",
        "--enable-minus-hard-case-routing",
        "true" if hard_case_enabled else "false",
        "--max-hard-case-extra-jobs",
        str(max_hard_case_extra_jobs),
        "--hard-case-runtime-sec",
        str(runtime_seconds),
        "--enable-audio-preview-render",
        "true",
        "--require-preview-audio-for-preview-ready",
        "true" if require_audio_preview_ready else "false",
    ]


def default_runtime_runner(command: list[str], run_dir: Path, context: dict[str, Any]) -> int:
    del run_dir, context
    result = subprocess.run(command, cwd=APP_ROOT, check=False)
    return int(result.returncode)


def values_as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, tuple):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        return [value] if value else []
    return [str(value)]


def rendered_path_for(row: dict[str, Any]) -> str:
    return str(row.get("rendered_path") or row.get("preview_audio_path") or row.get("audio_path") or "")


def path_exists_for_preview(path: str) -> bool:
    if not path:
        return False
    return Path(path).exists()


def preview_status_for(manifest: dict[str, Any], preview: dict[str, Any]) -> str:
    return str(preview.get("preview_status") or preview.get("runtime_status") or manifest.get("output_status") or "")


def collect_blockers(rows: Iterable[dict[str, Any]], manifest: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for key in BLOCKER_FIELDS:
        blockers.extend(values_as_list(manifest.get(key)))
    for row in rows:
        for key in BLOCKER_FIELDS:
            blockers.extend(values_as_list(row.get(key)))
    return stable_unique(blockers)


def count_blocker(rows: Iterable[dict[str, Any]], manifest: dict[str, Any], blocker: str) -> int:
    count = 0
    for key in BLOCKER_FIELDS:
        if blocker in values_as_list(manifest.get(key)):
            count += 1
    for row in rows:
        for key in BLOCKER_FIELDS:
            if blocker in values_as_list(row.get(key)):
                count += 1
    return count


def accepted_count(rows: Iterable[dict[str, Any]], mode: str) -> int:
    total = 0
    for row in rows:
        if str(row.get("score_mode") or "") != mode:
            continue
        final_decision = str(row.get("final_guarded_decision") or row.get("script_decision") or "")
        if final_decision == "accept_candidate":
            total += 1
    return total


def pair_compatible_count(pair_rows: Iterable[dict[str, Any]]) -> int:
    total = 0
    for row in pair_rows:
        decision = str(row.get("pair_decision") or row.get("runtime_status") or "")
        if decision in {"pair_compatible", "preview_ready", "accepted_for_listen_pack"}:
            total += 1
    return total


def truthy_release_ready(value: Any) -> int:
    if value is True:
        return 1
    if isinstance(value, str) and value.strip().lower() == "true":
        return 1
    return 0


def release_ready_count(*values: Any) -> int:
    total = 0
    for value in values:
        if isinstance(value, list):
            total += sum(release_ready_count(item) for item in value)
        elif isinstance(value, dict):
            total += truthy_release_ready(value.get("release_ready"))
            total += int(value.get("release_ready_count") or 0)
            if "stems" in value and isinstance(value["stems"], list):
                total += release_ready_count(value["stems"])
        else:
            total += truthy_release_ready(value)
    return total


def bool_flag_count(*manifests: dict[str, Any]) -> dict[str, int]:
    counts = {key: 0 for key in SAFETY_FLAG_KEYS}
    for manifest in manifests:
        for key in counts:
            counts[key] += 1 if manifest.get(key) is True else 0
    return counts


def artifact_paths(run_dir: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    for name, relative in RUNTIME_ARTIFACTS.items():
        path = run_dir / relative
        paths[name] = str(path) if path.exists() else ""
    return paths


def collect_runtime_measurement(
    *,
    candidate: dict[str, Any],
    track: Path,
    run_dir: Path,
    exit_code: int,
    elapsed_sec: float,
) -> dict[str, Any]:
    manifest = read_json(run_dir / "run_manifest.json")
    preview = read_json(run_dir / "preview_manifest.json")
    preview_contract = read_json(run_dir / "preview_render_contract.json")
    rendered_rows = read_jsonl(run_dir / "rendered_preview_audio.jsonl")
    music_manifest = read_json(run_dir / "music_intelligence_input_manifest.json")
    candidate_pool = read_jsonl(run_dir / "candidate_pool.jsonl")
    learned_rows = read_jsonl(run_dir / "learned_critic_scores.jsonl")
    vocal_rows = read_jsonl(run_dir / "vocal_windowed_judge.jsonl")
    minus_rows = read_jsonl(run_dir / "minus_windowed_judge.jsonl")
    pair_rows = read_jsonl(run_dir / "pair_compatibility.jsonl")
    post_rows = read_jsonl(run_dir / "post_render_judge.jsonl")
    learning_rows = read_jsonl(run_dir / "learning_outcome.jsonl")
    candidate_wave_summary = manifest.get("candidate_wave_summary") if isinstance(manifest.get("candidate_wave_summary"), dict) else {}
    candidate_wave_summary_file = read_json(run_dir / "candidates" / "candidate_wave_summary.json")
    if not candidate_wave_summary and candidate_wave_summary_file:
        candidate_wave_summary = candidate_wave_summary_file
    wave_jobs = read_jsonl(run_dir / "candidates" / "candidate_wave_jobs.jsonl")
    wave_failures = read_jsonl(run_dir / "candidates" / "candidate_wave_failures.jsonl")
    output_status = str(manifest.get("output_status") or ("runtime_failed" if exit_code else "unknown"))
    preview_status = preview_status_for(manifest, preview)
    rendered_manifest_paths = values_as_list(preview.get("rendered_preview_audio"))
    rendered_row_paths = [rendered_path_for(row) for row in rendered_rows if str(row.get("render_status") or "") == "rendered"]
    rendered_valid_paths = [
        rendered_path_for(row)
        for row in rendered_rows
        if str(row.get("render_status") or "") == "rendered" and str(row.get("validation_status") or "") == "validated"
    ]
    rendered_preview_audio = stable_unique(rendered_manifest_paths + rendered_row_paths)
    rendered_existing = [path for path in rendered_preview_audio if path_exists_for_preview(path)]
    rendered_preview_audio_count = max(
        len(rendered_valid_paths),
        int(preview.get("rendered_preview_audio_count") or preview_contract.get("rendered_preview_audio_count") or 0),
        len(rendered_manifest_paths),
    )
    accepted_pair_ready = (
        preview.get("accepted_pair_ready") is True
        or output_status in {"accepted_pair_ready", "audio_preview_ready", "manifest_preview_ready", "preview_render_skipped", "preview_render_failed"}
    )
    manifest_preview_ready = (
        preview.get("manifest_preview_ready") is True
        or output_status in {"preview_ready", "accepted_for_listen_pack", "manifest_preview_ready", "audio_preview_ready"}
        or preview_status in {"preview_ready", "accepted_for_listen_pack", "manifest_preview_ready", "audio_preview_ready"}
    )
    audio_preview_ready = (
        (preview.get("audio_preview_ready") is True or output_status == "audio_preview_ready" or preview_status == "audio_preview_ready")
        and rendered_preview_audio_count > 0
        and (not rendered_preview_audio or len(rendered_existing) == len(rendered_preview_audio))
    )
    preview_render_failed = (
        preview.get("preview_render_failed") is True
        or preview_status in {"preview_render_failed", "preview_blocked_safety_contract"}
        or any(str(row.get("render_status") or "") == "failed" for row in rendered_rows)
    )
    preview_render_skipped = (
        preview.get("preview_render_skipped") is True
        or preview_status == "preview_render_skipped"
        or any(str(row.get("render_status") or "") == "skipped" for row in rendered_rows)
    )
    validation_status = str(preview.get("validation_status") or preview_contract.get("validation_status") or "")
    preview_validation_failed = (
        validation_status in {"failed", "blocked", "invalid"}
        or preview_status == "preview_validation_failed"
        or (preview_status == "audio_preview_ready" and not audio_preview_ready)
        or bool(set(collect_blockers(rendered_rows, preview)) & {"rendered_audio_not_validated"})
    )
    all_rows_for_blockers = learned_rows + vocal_rows + minus_rows + pair_rows + post_rows + learning_rows
    safety_flags = bool_flag_count(manifest, preview, music_manifest)
    release_count = release_ready_count(manifest, preview, music_manifest, pair_rows, post_rows, learning_rows)
    complete_pairs = int(candidate_wave_summary.get("complete_pairs") or sum(1 for row in candidate_pool if row.get("is_complete_pair") is True))
    vocal_usable = int(candidate_wave_summary.get("vocal_usable_candidates") or sum(1 for row in candidate_pool if row.get("can_use_for_vocal") is True))
    minus_usable = int(candidate_wave_summary.get("minus_usable_candidates") or sum(1 for row in candidate_pool if row.get("can_use_for_minus") is True))
    pair_usable = int(candidate_wave_summary.get("pair_usable_candidates") or sum(1 for row in candidate_pool if row.get("can_use_for_pair") is True))
    accepted_vocal = accepted_count(learned_rows, "vocal")
    accepted_minus = accepted_count(learned_rows, "minus")
    pair_compatible = pair_compatible_count(pair_rows)
    music_section = manifest.get("music_intelligence") if isinstance(manifest.get("music_intelligence"), dict) else {}
    analysis_ready = int(music_section.get("analysis_ready_stems_count") or music_manifest.get("analysis_ready_stems_count") or 0)
    blocked_stems = int(music_section.get("blocked_stems_count") or music_manifest.get("blocked_stems_count") or 0)
    failed_status = 1 if output_status == "failed_safety_contract" or exit_code != 0 else 0
    return {
        "schema_version": 1,
        "harness_candidate_id": str(candidate.get("harness_candidate_id") or ""),
        "parent_id": str(candidate.get("parent_id") or ""),
        "mutation_type": str(candidate.get("mutation_type") or ""),
        "track_id": track.stem,
        "input_track_path": str(track),
        "runtime_run_dir": str(run_dir),
        "runtime_exit_code": int(exit_code),
        "runtime_elapsed_sec": round(float(elapsed_sec), 3),
        "final_runtime_status": output_status if output_status in ALLOWED_RUNTIME_STATUSES else "failed_safety_contract",
        "jobs_attempted": int(candidate_wave_summary.get("jobs_attempted") or len(wave_jobs)),
        "jobs_generated": int(candidate_wave_summary.get("jobs_generated") or candidate_wave_summary.get("candidates_generated") or 0),
        "jobs_failed": int(candidate_wave_summary.get("jobs_failed") or len(wave_failures)),
        "jobs_skipped": int(candidate_wave_summary.get("jobs_skipped") or 0),
        "duplicate_jobs_avoided": int(candidate_wave_summary.get("duplicate_jobs_avoided") or 0),
        "complete_pairs": complete_pairs,
        "vocal_usable_candidates": vocal_usable,
        "minus_usable_candidates": minus_usable,
        "pair_usable_candidates": pair_usable,
        "accepted_vocal_count": accepted_vocal,
        "accepted_minus_count": accepted_minus,
        "pair_compatible_count": pair_compatible,
        "accepted_pair_ready": bool(accepted_pair_ready),
        "accepted_pair_ready_count": 1 if accepted_pair_ready else 0,
        "severe_phase_collapse_count": count_blocker(all_rows_for_blockers, manifest, "severe_phase_collapse"),
        "severe_vocal_bleed_in_minus_count": count_blocker(all_rows_for_blockers, manifest, "severe_vocal_bleed_in_minus"),
        "preview_manifest_created": (run_dir / "preview_manifest.json").exists(),
        "rendered_preview_audio_count": rendered_preview_audio_count,
        "audio_preview_ready": bool(audio_preview_ready),
        "manifest_preview_ready": bool(manifest_preview_ready),
        "preview_render_failed_count": 1 if preview_render_failed else 0,
        "preview_render_skipped_count": 1 if preview_render_skipped else 0,
        "preview_validation_failed_count": 1 if preview_validation_failed else 0,
        "analysis_ready_stems_count": analysis_ready,
        "blocked_stems_count": blocked_stems,
        "release_ready_count": release_count,
        "release_ready": release_count > 0,
        "melody_transcription_ran": bool(music_section.get("melody_transcription_ran") or music_manifest.get("melody_transcription_ran")),
        "recommendation_index_updated": bool(
            music_section.get("recommendation_index_updated") or music_manifest.get("recommendation_index_updated")
        ),
        "failed_safety_contract_count": failed_status,
        "safety_flag_counts": safety_flags,
        "safety_blockers": collect_blockers(all_rows_for_blockers, manifest),
        "artifact_paths": artifact_paths(run_dir),
    }


def numeric(row: dict[str, Any], key: str) -> int:
    try:
        return int(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def aggregate_track_rows(candidate: dict[str, Any], track_rows: list[dict[str, Any]]) -> dict[str, Any]:
    release_count = sum(numeric(row, "release_ready_count") for row in track_rows)
    flag_totals = {key: sum(int((row.get("safety_flag_counts") or {}).get(key) or 0) for row in track_rows) for key in SAFETY_FLAG_KEYS}
    return {
        "schema_version": 1,
        "harness_candidate_id": str(candidate.get("harness_candidate_id") or ""),
        "parent_id": str(candidate.get("parent_id") or ""),
        "mutation_type": str(candidate.get("mutation_type") or ""),
        "tracks_tested": len(track_rows),
        "runtime_sec": round(sum(numeric(row, "runtime_elapsed_sec") for row in track_rows), 3),
        "jobs_attempted": sum(numeric(row, "jobs_attempted") for row in track_rows),
        "jobs_generated": sum(numeric(row, "jobs_generated") for row in track_rows),
        "jobs_failed": sum(numeric(row, "jobs_failed") for row in track_rows),
        "jobs_skipped": sum(numeric(row, "jobs_skipped") for row in track_rows),
        "duplicate_jobs_avoided": sum(numeric(row, "duplicate_jobs_avoided") for row in track_rows),
        "complete_pairs": sum(numeric(row, "complete_pairs") for row in track_rows),
        "vocal_usable_candidates": sum(numeric(row, "vocal_usable_candidates") for row in track_rows),
        "minus_usable_candidates": sum(numeric(row, "minus_usable_candidates") for row in track_rows),
        "pair_usable_candidates": sum(numeric(row, "pair_usable_candidates") for row in track_rows),
        "accepted_vocal_count": sum(numeric(row, "accepted_vocal_count") for row in track_rows),
        "accepted_minus_count": sum(numeric(row, "accepted_minus_count") for row in track_rows),
        "pair_compatible_count": sum(numeric(row, "pair_compatible_count") for row in track_rows),
        "accepted_pair_ready_count": sum(numeric(row, "accepted_pair_ready_count") for row in track_rows),
        "severe_phase_collapse_count": sum(numeric(row, "severe_phase_collapse_count") for row in track_rows),
        "severe_vocal_bleed_in_minus_count": sum(numeric(row, "severe_vocal_bleed_in_minus_count") for row in track_rows),
        "audio_preview_ready_count": sum(1 for row in track_rows if row.get("audio_preview_ready") is True),
        "manifest_preview_ready_count": sum(1 for row in track_rows if row.get("manifest_preview_ready") is True),
        "rendered_preview_audio_count": sum(numeric(row, "rendered_preview_audio_count") for row in track_rows),
        "preview_render_failed_count": sum(numeric(row, "preview_render_failed_count") for row in track_rows),
        "preview_render_skipped_count": sum(numeric(row, "preview_render_skipped_count") for row in track_rows),
        "preview_validation_failed_count": sum(numeric(row, "preview_validation_failed_count") for row in track_rows),
        "analysis_ready_stems_count": sum(numeric(row, "analysis_ready_stems_count") for row in track_rows),
        "blocked_stems_count": sum(numeric(row, "blocked_stems_count") for row in track_rows),
        "release_ready_count": release_count,
        "release_ready": release_count > 0,
        "melody_transcription_ran": any(row.get("melody_transcription_ran") is True for row in track_rows),
        "recommendation_index_updated": any(row.get("recommendation_index_updated") is True for row in track_rows),
        "failed_safety_contract_count": sum(numeric(row, "failed_safety_contract_count") for row in track_rows),
        "safety_flag_counts": flag_totals,
    }


def track_improvement(candidate_row: dict[str, Any], baseline_row: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if candidate_row.get("audio_preview_ready") is True and baseline_row.get("audio_preview_ready") is not True:
        reasons.append("audio_preview_ready_added")
    if numeric(candidate_row, "rendered_preview_audio_count") > numeric(baseline_row, "rendered_preview_audio_count"):
        reasons.append("rendered_preview_audio_count_increased")
    if numeric(candidate_row, "accepted_minus_count") > numeric(baseline_row, "accepted_minus_count"):
        reasons.append("accepted_minus_count_increased")
    if numeric(candidate_row, "severe_phase_collapse_count") < numeric(baseline_row, "severe_phase_collapse_count"):
        reasons.append("severe_phase_collapse_count_decreased")
    if numeric(candidate_row, "severe_vocal_bleed_in_minus_count") < numeric(baseline_row, "severe_vocal_bleed_in_minus_count"):
        reasons.append("severe_vocal_bleed_in_minus_count_decreased")
    if numeric(candidate_row, "pair_compatible_count") > numeric(baseline_row, "pair_compatible_count"):
        reasons.append("pair_compatible_count_increased")
    if numeric(candidate_row, "analysis_ready_stems_count") > numeric(baseline_row, "analysis_ready_stems_count"):
        reasons.append("analysis_ready_stems_count_increased")
    if numeric(candidate_row, "blocked_stems_count") < numeric(baseline_row, "blocked_stems_count"):
        reasons.append("blocked_stems_count_decreased")
    return bool(reasons), reasons


def aggregate_quality_improvement_reasons(candidate: dict[str, Any], baseline: dict[str, Any] | None) -> list[str]:
    if baseline is None:
        return []
    reasons: list[str] = []
    for key in QUALITY_POSITIVE_OBJECTIVES:
        if numeric(candidate, key) > numeric(baseline, key):
            reasons.append(f"{key}_increased")
    for key in QUALITY_NEGATIVE_OBJECTIVES:
        if numeric(candidate, key) < numeric(baseline, key):
            reasons.append(f"{key}_decreased")
    return reasons


def aggregate_quality_regression_reasons(candidate: dict[str, Any], baseline: dict[str, Any] | None) -> list[str]:
    if baseline is None:
        return []
    reasons: list[str] = []
    for key in QUALITY_POSITIVE_OBJECTIVES:
        if numeric(candidate, key) < numeric(baseline, key):
            reasons.append(f"{key}_decreased")
    for key in QUALITY_NEGATIVE_OBJECTIVES:
        if numeric(candidate, key) > numeric(baseline, key):
            reasons.append(f"{key}_increased")
    return reasons


def efficiency_improvement_detected(candidate: dict[str, Any], baseline: dict[str, Any] | None) -> bool:
    if baseline is None:
        return False
    return any(
        numeric(candidate, key) > 0.0 and numeric(baseline, key) > 0.0 and numeric(candidate, key) < numeric(baseline, key)
        for key in EFFICIENCY_OBJECTIVES
    )


def has_safety_regression(candidate: dict[str, Any], baseline: dict[str, Any] | None = None) -> bool:
    if candidate.get("release_ready") is True or numeric(candidate, "release_ready_count") > 0:
        return True
    if candidate.get("melody_transcription_ran") is True or candidate.get("recommendation_index_updated") is True:
        return True
    if any(int((candidate.get("safety_flag_counts") or {}).get(key) or 0) > 0 for key in SAFETY_FLAG_KEYS):
        return True
    if baseline is None and numeric(candidate, "preview_render_failed_count") > 0:
        return True
    if baseline is not None and numeric(candidate, "preview_render_failed_count") > numeric(baseline, "preview_render_failed_count"):
        return True
    if baseline is not None and numeric(candidate, "failed_safety_contract_count") > numeric(baseline, "failed_safety_contract_count"):
        return True
    if baseline is not None and numeric(candidate, "preview_validation_failed_count") > numeric(baseline, "preview_validation_failed_count"):
        return True
    return False


def improvement_score(candidate: dict[str, Any], baseline: dict[str, Any]) -> int:
    return (
        (numeric(candidate, "audio_preview_ready_count") - numeric(baseline, "audio_preview_ready_count")) * 6
        + (numeric(candidate, "rendered_preview_audio_count") - numeric(baseline, "rendered_preview_audio_count")) * 2
        + numeric(candidate, "accepted_minus_count")
        - numeric(baseline, "accepted_minus_count")
        + numeric(candidate, "pair_compatible_count")
        - numeric(baseline, "pair_compatible_count")
        + numeric(baseline, "severe_phase_collapse_count")
        - numeric(candidate, "severe_phase_collapse_count")
        + numeric(baseline, "severe_vocal_bleed_in_minus_count")
        - numeric(candidate, "severe_vocal_bleed_in_minus_count")
        - (numeric(candidate, "preview_render_failed_count") - numeric(baseline, "preview_render_failed_count")) * 6
        - (numeric(candidate, "preview_validation_failed_count") - numeric(baseline, "preview_validation_failed_count")) * 4
    )


def decide_candidate(
    *,
    candidate_result: dict[str, Any],
    candidate_track_rows: list[dict[str, Any]],
    baseline_result: dict[str, Any] | None,
    baseline_by_track: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if candidate_result.get("mutation_type") == "baseline_keep":
        return {
            **candidate_result,
            "decision": "keep_baseline",
            "decision_class": "keep_baseline",
            "improved_tracks": [],
            "improvement_reasons": {},
            "quality_improvement_detected": False,
            "efficiency_improvement_detected": False,
            "promotion_eligible": False,
            "promotion_block_reason": "baseline",
        }
    if baseline_result is None:
        return {
            **candidate_result,
            "decision": "candidate_inconclusive",
            "decision_class": "candidate_inconclusive",
            "improved_tracks": [],
            "improvement_reasons": {},
            "quality_improvement_detected": False,
            "efficiency_improvement_detected": False,
            "promotion_eligible": False,
            "promotion_block_reason": "missing_baseline",
        }
    improved_tracks: list[str] = []
    improvement_reasons: dict[str, list[str]] = {}
    for row in candidate_track_rows:
        baseline_row = baseline_by_track.get(str(row.get("track_id") or ""))
        if baseline_row is None:
            continue
        improved, reasons = track_improvement(row, baseline_row)
        if improved:
            track_id = str(row.get("track_id") or "")
            improved_tracks.append(track_id)
            improvement_reasons[track_id] = reasons
    aggregate_quality_reasons = aggregate_quality_improvement_reasons(candidate_result, baseline_result)
    aggregate_quality_regressions = aggregate_quality_regression_reasons(candidate_result, baseline_result)
    quality_improved = bool(improved_tracks or aggregate_quality_reasons)
    efficiency_improved = efficiency_improvement_detected(candidate_result, baseline_result)
    if has_safety_regression(candidate_result, baseline_result):
        decision = "candidate_rejected_safety"
        promotion_eligible = False
        promotion_block_reason = "safety_regression"
    elif not candidate_track_rows:
        decision = "candidate_inconclusive"
        promotion_eligible = False
        promotion_block_reason = "no_track_measurements"
    elif improved_tracks and len(improved_tracks) >= 2:
        decision = "candidate_quality_improved"
        promotion_eligible = True
        promotion_block_reason = ""
    elif improved_tracks or aggregate_quality_reasons:
        decision = "candidate_specialist_only"
        promotion_eligible = False
        promotion_block_reason = "specialist_only_not_global"
    elif aggregate_quality_regressions:
        decision = "candidate_rejected_runtime"
        promotion_eligible = False
        promotion_block_reason = "quality_regression"
    elif efficiency_improved:
        decision = "candidate_efficiency_only"
        promotion_eligible = False
        promotion_block_reason = "efficiency_only_no_quality_improvement"
    else:
        decision = "candidate_no_op"
        promotion_eligible = False
        promotion_block_reason = "no_quality_or_efficiency_improvement"
    return {
        **candidate_result,
        "decision": decision,
        "decision_class": decision,
        "improved_tracks": improved_tracks,
        "improvement_reasons": improvement_reasons,
        "aggregate_quality_improvement_reasons": aggregate_quality_reasons,
        "aggregate_quality_regression_reasons": aggregate_quality_regressions,
        "quality_improvement_detected": quality_improved,
        "efficiency_improvement_detected": efficiency_improved,
        "promotion_eligible": promotion_eligible,
        "promotion_block_reason": promotion_block_reason,
        "accepted_minus_delta": numeric(candidate_result, "accepted_minus_count") - numeric(baseline_result, "accepted_minus_count"),
        "phase_collapse_delta": numeric(candidate_result, "severe_phase_collapse_count") - numeric(baseline_result, "severe_phase_collapse_count"),
        "vocal_bleed_delta": numeric(candidate_result, "severe_vocal_bleed_in_minus_count")
        - numeric(baseline_result, "severe_vocal_bleed_in_minus_count"),
        "pair_compatibility_delta": numeric(candidate_result, "pair_compatible_count") - numeric(baseline_result, "pair_compatible_count"),
        "audio_preview_delta": numeric(candidate_result, "audio_preview_ready_count") - numeric(baseline_result, "audio_preview_ready_count"),
        "rendered_preview_audio_delta": numeric(candidate_result, "rendered_preview_audio_count") - numeric(baseline_result, "rendered_preview_audio_count"),
        "preview_render_failed_delta": numeric(candidate_result, "preview_render_failed_count") - numeric(baseline_result, "preview_render_failed_count"),
        "preview_validation_failed_delta": numeric(candidate_result, "preview_validation_failed_count") - numeric(baseline_result, "preview_validation_failed_count"),
        "runtime_improvement_score": improvement_score(candidate_result, baseline_result),
    }


def rejected_safety_result(candidate: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "harness_candidate_id": str(candidate.get("harness_candidate_id") or ""),
        "parent_id": str(candidate.get("parent_id") or ""),
        "mutation_type": str(candidate.get("mutation_type") or ""),
        "tracks_tested": 0,
        "decision": "candidate_rejected_safety",
        "decision_class": "candidate_rejected_safety",
        "validation_valid": False,
        "validation_rejection_reasons": validation.get("rejection_reasons") or [],
        "quality_improvement_detected": False,
        "efficiency_improvement_detected": False,
        "promotion_eligible": False,
        "promotion_block_reason": "safety_regression",
        "release_ready": False,
        "release_ready_count": 0,
        "melody_transcription_ran": False,
        "recommendation_index_updated": False,
    }


def run_candidate_on_tracks(
    *,
    candidate: dict[str, Any],
    tracks: Sequence[Path],
    model_dir: Path,
    output_dir: Path,
    max_runtime_sec_per_track: int,
    runtime_runner: RuntimeRunner,
    progress_jsonl: Path,
    require_audio_preview_ready: bool = False,
) -> list[dict[str, Any]]:
    candidate_id = str(candidate.get("harness_candidate_id") or stable_hash(candidate)[:12])
    rows: list[dict[str, Any]] = []
    for index, track in enumerate(tracks):
        run_dir = output_dir / "runs" / safe_slug(candidate_id) / f"{index:02d}_{safe_slug(track.stem)}"
        command = map_candidate_to_runtime_args(
            candidate=candidate,
            input_audio=track,
            output_dir=run_dir,
            model_dir=model_dir,
            max_runtime_sec_per_track=max_runtime_sec_per_track,
            require_audio_preview_ready=require_audio_preview_ready,
        )
        start = time.monotonic()
        append_jsonl(
            progress_jsonl,
            {
                "event": "runtime_track_start",
                "harness_candidate_id": candidate_id,
                "track_id": track.stem,
                "runtime_run_dir": str(run_dir),
                "gpu_inference_expected": True,
                "command": command,
            },
        )
        exit_code = runtime_runner(command, run_dir, {"harness_candidate_id": candidate_id, "track": str(track), "candidate": candidate})
        elapsed = time.monotonic() - start
        row = collect_runtime_measurement(candidate=candidate, track=track, run_dir=run_dir, exit_code=exit_code, elapsed_sec=elapsed)
        rows.append(row)
        append_jsonl(
            progress_jsonl,
            {
                "event": "runtime_track_end",
                "harness_candidate_id": candidate_id,
                "track_id": track.stem,
                "runtime_run_dir": str(run_dir),
                "runtime_exit_code": int(exit_code),
                "elapsed_sec": round(float(elapsed), 3),
                "final_runtime_status": row["final_runtime_status"],
                "release_ready": row["release_ready"],
            },
        )
    return rows


def build_summary_md(manifest: dict[str, Any], candidate_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# No-Reference Harness Runtime Validation",
        "",
        f"- candidates_tested: {manifest['candidates_tested']}",
        f"- tracks_tested: {manifest['tracks_tested']}",
        f"- candidate_quality_improved_count: {manifest['candidate_quality_improved_count']}",
        f"- candidate_efficiency_only_count: {manifest['candidate_efficiency_only_count']}",
        f"- candidate_no_op_count: {manifest['candidate_no_op_count']}",
        f"- candidate_runtime_improved_count: {manifest['candidate_runtime_improved_count']}",
        f"- candidate_specialist_only_count: {manifest['candidate_specialist_only_count']}",
        f"- candidate_rejected_runtime_count: {manifest['candidate_rejected_runtime_count']}",
        f"- candidate_rejected_safety_count: {manifest['candidate_rejected_safety_count']}",
        f"- best_global_candidate: {manifest['best_global_candidate']}",
        f"- best_track002_specialist: {manifest['best_track002_specialist']}",
        f"- accepted_minus_improvement: {manifest['accepted_minus_improvement']}",
        f"- phase_collapse_delta: {manifest['phase_collapse_delta']}",
        f"- vocal_bleed_delta: {manifest['vocal_bleed_delta']}",
        f"- pair_compatibility_delta: {manifest['pair_compatibility_delta']}",
        f"- audio_preview_improvement: {manifest['audio_preview_improvement']}",
        f"- accepted_pair_ready_count: {manifest['accepted_pair_ready_count']}",
        f"- audio_preview_ready_count: {manifest['audio_preview_ready_count']}",
        f"- rendered_preview_audio_count: {manifest['rendered_preview_audio_count']}",
        f"- manifest_preview_ready_count: {manifest['manifest_preview_ready_count']}",
        f"- preview_render_failed_count: {manifest['preview_render_failed_count']}",
        f"- preview_render_skipped_count: {manifest['preview_render_skipped_count']}",
        f"- preview_validation_failed_count: {manifest['preview_validation_failed_count']}",
        f"- analysis_ready_stems_count: {manifest['analysis_ready_stems_count']}",
        f"- blocked_stems_count: {manifest['blocked_stems_count']}",
        f"- release_ready_count: {manifest['release_ready_count']}",
        f"- melody_transcription_ran: {str(manifest['melody_transcription_ran']).lower()}",
        f"- recommendation_index_updated: {str(manifest['recommendation_index_updated']).lower()}",
        f"- measurement_only: {str(manifest['measurement_only']).lower()}",
        "",
        "## Candidate Decisions",
    ]
    for row in candidate_rows:
        lines.append(f"- {row.get('harness_candidate_id')}: {row.get('decision')}")
    return "\n".join(lines) + "\n"


def run_harness_runtime_validation(
    *,
    pareto_frontier_jsonl: Path,
    tracks: Sequence[Path],
    model_dir: Path,
    output_dir: Path,
    max_pareto_candidates: int,
    max_runtime_sec_per_track: int,
    measurement_only: bool,
    runtime_runner: RuntimeRunner | None = None,
    require_audio_preview_ready: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_runner = runtime_runner or default_runtime_runner
    progress_jsonl = output_dir / "harness_runtime_progress.jsonl"
    progress_jsonl.write_text("", encoding="utf-8")
    track_paths = [Path(track) for track in tracks]
    baseline = load_baseline_candidate(pareto_frontier_jsonl)
    candidates = load_pareto_candidates(pareto_frontier_jsonl, max_pareto_candidates)
    runnable: list[dict[str, Any]] = []
    if baseline is not None:
        runnable.append(baseline)
    runnable.extend(candidates)
    candidate_result_rows: list[dict[str, Any]] = []
    track_result_rows: list[dict[str, Any]] = []
    baseline_result: dict[str, Any] | None = None
    baseline_by_track: dict[str, dict[str, Any]] = {}
    candidate_track_rows_by_id: dict[str, list[dict[str, Any]]] = {}

    for candidate in runnable:
        validation = validate_harness_candidate(candidate)
        candidate_id = str(candidate.get("harness_candidate_id") or "")
        if not validation["valid"]:
            candidate_result_rows.append(rejected_safety_result(candidate, validation))
            continue
        rows = run_candidate_on_tracks(
            candidate=candidate,
            tracks=track_paths,
            model_dir=model_dir,
            output_dir=output_dir,
            max_runtime_sec_per_track=max_runtime_sec_per_track,
            runtime_runner=runtime_runner,
            progress_jsonl=progress_jsonl,
            require_audio_preview_ready=require_audio_preview_ready,
        )
        candidate_track_rows_by_id[candidate_id] = rows
        track_result_rows.extend(rows)
        aggregate = aggregate_track_rows(candidate, rows)
        aggregate.update({"validation_valid": True, "validation_rejection_reasons": []})
        if candidate.get("mutation_type") == "baseline_keep":
            baseline_result = {**aggregate, "decision": "keep_baseline", "improved_tracks": [], "improvement_reasons": {}}
            baseline_by_track = {str(row.get("track_id") or ""): row for row in rows}
            candidate_result_rows.append(baseline_result)
        else:
            candidate_result_rows.append(aggregate)

    decided_rows: list[dict[str, Any]] = []
    for row in candidate_result_rows:
        if row.get("decision") in {"candidate_rejected_safety", "keep_baseline"}:
            decided_rows.append(row)
            continue
        candidate_id = str(row.get("harness_candidate_id") or "")
        decided_rows.append(
            decide_candidate(
                candidate_result=row,
                candidate_track_rows=candidate_track_rows_by_id.get(candidate_id, []),
                baseline_result=baseline_result,
                baseline_by_track=baseline_by_track,
            )
        )
    candidate_result_rows = decided_rows

    candidate_rows_without_baseline = [row for row in candidate_result_rows if row.get("decision") != "keep_baseline"]
    improved_rows = [row for row in candidate_rows_without_baseline if row.get("decision") == "candidate_quality_improved"]
    efficiency_only_rows = [row for row in candidate_rows_without_baseline if row.get("decision") == "candidate_efficiency_only"]
    no_op_rows = [row for row in candidate_rows_without_baseline if row.get("decision") == "candidate_no_op"]
    specialist_rows = [row for row in candidate_rows_without_baseline if row.get("decision") == "candidate_specialist_only"]
    best_global = max(improved_rows, key=lambda row: numeric(row, "runtime_improvement_score"), default={})
    track002_specialists = [
        row
        for row in specialist_rows
        if any("sub_zero" in safe_slug(track) or "track002" in safe_slug(track) for track in row.get("improved_tracks") or [])
    ]
    best_track002 = max(track002_specialists, key=lambda row: numeric(row, "runtime_improvement_score"), default={})
    reference = best_global or (best_track002 if not best_global else {})
    baseline_ref = baseline_result or {}
    manifest = {
        "schema_version": 1,
        "pareto_frontier_jsonl": str(pareto_frontier_jsonl),
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "measurement_only": bool(measurement_only),
        "require_audio_preview_ready": bool(require_audio_preview_ready),
        "candidates_tested": len([row for row in candidate_result_rows if row.get("decision") != "candidate_rejected_safety"]),
        "tracks_tested": len(track_paths),
        "candidate_quality_improved_count": sum(1 for row in candidate_result_rows if row.get("decision") == "candidate_quality_improved"),
        "candidate_runtime_improved_count": sum(1 for row in candidate_result_rows if row.get("decision") == "candidate_quality_improved"),
        "candidate_efficiency_only_count": len(efficiency_only_rows),
        "candidate_no_op_count": len(no_op_rows),
        "candidate_specialist_only_count": sum(1 for row in candidate_result_rows if row.get("decision") == "candidate_specialist_only"),
        "candidate_rejected_runtime_count": sum(1 for row in candidate_result_rows if row.get("decision") == "candidate_rejected_runtime"),
        "candidate_rejected_safety_count": sum(1 for row in candidate_result_rows if row.get("decision") == "candidate_rejected_safety"),
        "candidate_inconclusive_count": sum(1 for row in candidate_result_rows if row.get("decision") == "candidate_inconclusive"),
        "best_global_candidate": str(best_global.get("harness_candidate_id") or ""),
        "best_track002_specialist": str(best_track002.get("harness_candidate_id") or ""),
        "accepted_minus_improvement": numeric(reference, "accepted_minus_count") - numeric(baseline_ref, "accepted_minus_count") if reference else 0,
        "phase_collapse_delta": numeric(reference, "severe_phase_collapse_count") - numeric(baseline_ref, "severe_phase_collapse_count") if reference else 0,
        "vocal_bleed_delta": numeric(reference, "severe_vocal_bleed_in_minus_count") - numeric(baseline_ref, "severe_vocal_bleed_in_minus_count") if reference else 0,
        "pair_compatibility_delta": numeric(reference, "pair_compatible_count") - numeric(baseline_ref, "pair_compatible_count") if reference else 0,
        "audio_preview_improvement": numeric(reference, "audio_preview_ready_count") - numeric(baseline_ref, "audio_preview_ready_count") if reference else 0,
        "rendered_preview_audio_delta": numeric(reference, "rendered_preview_audio_count") - numeric(baseline_ref, "rendered_preview_audio_count") if reference else 0,
        "audio_preview_ready_count": sum(1 for row in track_result_rows if row.get("audio_preview_ready") is True),
        "accepted_pair_ready_count": sum(numeric(row, "accepted_pair_ready_count") for row in track_result_rows),
        "manifest_preview_ready_count": sum(1 for row in track_result_rows if row.get("manifest_preview_ready") is True),
        "rendered_preview_audio_count": sum(numeric(row, "rendered_preview_audio_count") for row in track_result_rows),
        "preview_render_failed_count": sum(numeric(row, "preview_render_failed_count") for row in track_result_rows),
        "preview_render_skipped_count": sum(numeric(row, "preview_render_skipped_count") for row in track_result_rows),
        "preview_validation_failed_count": sum(numeric(row, "preview_validation_failed_count") for row in track_result_rows),
        "analysis_ready_stems_count": sum(numeric(row, "analysis_ready_stems_count") for row in track_result_rows),
        "blocked_stems_count": sum(numeric(row, "blocked_stems_count") for row in track_result_rows),
        "release_ready_count": sum(numeric(row, "release_ready_count") for row in track_result_rows),
        "melody_transcription_ran": any(row.get("melody_transcription_ran") is True for row in track_result_rows),
        "recommendation_index_updated": any(row.get("recommendation_index_updated") is True for row in track_result_rows),
        "release_ready": False,
        "reference_required": False,
        "human_csv_required": False,
        "training_started": False,
        "separator_training_started": False,
        "model_downloaded": False,
        "model_registry_edited": False,
        "thresholds_relaxed": False,
        "cleanup_routes_added": False,
        "artifacts": {
            "candidate_results": str(output_dir / "harness_runtime_candidate_results.jsonl"),
            "track_results": str(output_dir / "harness_runtime_track_results.jsonl"),
            "progress": str(progress_jsonl),
            "summary_md": str(output_dir / "runtime_validation_summary.md"),
        },
    }
    if manifest["release_ready_count"] > 0:
        manifest["release_ready"] = False
    write_jsonl(output_dir / "harness_runtime_track_results.jsonl", track_result_rows)
    write_jsonl(output_dir / "harness_runtime_candidate_results.jsonl", candidate_result_rows)
    write_json(output_dir / "runtime_validation_manifest.json", manifest)
    (output_dir / "runtime_validation_summary.md").write_text(build_summary_md(manifest, candidate_result_rows), encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate no-reference harness Pareto candidates on real autonomous runtime.")
    parser.add_argument("--pareto-frontier-jsonl", type=Path, required=True)
    parser.add_argument("--tracks", nargs="+", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pareto-candidates", type=int, default=3)
    parser.add_argument("--max-runtime-sec-per-track", type=int, default=1800)
    parser.add_argument("--measurement-only", type=parse_bool, default=True)
    parser.add_argument("--require-audio-preview-ready", type=parse_bool, default=False)
    args = parser.parse_args(argv)
    manifest = run_harness_runtime_validation(
        pareto_frontier_jsonl=args.pareto_frontier_jsonl,
        tracks=args.tracks,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        max_pareto_candidates=args.max_pareto_candidates,
        max_runtime_sec_per_track=args.max_runtime_sec_per_track,
        measurement_only=args.measurement_only,
        require_audio_preview_ready=args.require_audio_preview_ready,
    )
    print(json.dumps({"runtime_validation_manifest": str(args.output_dir / "runtime_validation_manifest.json"), **manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
