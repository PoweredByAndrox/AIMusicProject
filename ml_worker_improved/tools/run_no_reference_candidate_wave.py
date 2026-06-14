from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import soundfile as sf

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_no_reference_ml_dataset import stable_hash, write_jsonl  # noqa: E402
from tools.no_reference_candidate_job_registry import (  # noqa: E402
    CandidateJobConfig,
    build_candidate_job_registry,
)
from tools.no_reference_candidate_job_scheduler import (  # noqa: E402
    build_candidate_job_plan,
    write_candidate_job_plan,
)
from tools.run_local_separator_candidate_job import run_local_separator_candidate_job  # noqa: E402


ALLOWED_PROFILES = {"safe", "balanced", "exploration"}
ALLOWED_EXECUTION_MODES = {"collect_existing", "run_jobs", "auto"}
ALLOWED_TARGET_MODES = {"vocal", "minus", "both"}
DETERMINISTIC_CREATED_AT = "1970-01-01T00:00:00Z"

SAFETY_FLAGS = {
    "codex_required_at_runtime": False,
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "separator_training_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "release_ready": False,
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def sanitize(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value.strip())
    return cleaned.strip("._-")[:96] or "candidate"


def stable_id(prefix: str, payload: dict[str, Any]) -> str:
    return sanitize(f"{prefix}_{stable_hash(payload)[:16]}")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {value}")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_input_audio(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "reason": "input_audio_missing", "path": str(path)}
    if path.stat().st_size <= 0:
        return {"ok": False, "reason": "input_audio_empty", "path": str(path)}
    try:
        info = sf.info(str(path))
    except Exception as exc:  # pragma: no cover - defensive corrupt input path.
        return {"ok": False, "reason": f"input_audio_unreadable:{type(exc).__name__}", "path": str(path)}
    return {
        "ok": True,
        "path": str(path),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_sec": round(float(info.frames) / float(info.samplerate), 6) if info.samplerate else 0.0,
    }


def default_source_paths(output_dir: Path) -> list[Path]:
    return [
        output_dir / "candidate_wave_sources.jsonl",
        output_dir / "mock_separator_outputs.jsonl",
        output_dir / "separator_outputs.jsonl",
    ]


def load_source_rows(output_dir: Path, source_jsonl: Path | None) -> tuple[list[dict[str, Any]], list[Path]]:
    paths = [source_jsonl] if source_jsonl is not None else default_source_paths(output_dir)
    existing = [path for path in paths if path.exists()]
    rows: list[dict[str, Any]] = []
    for path in existing:
        for row in iter_jsonl(path):
            rows.append({**row, "source_jsonl": str(path)})
    rows.sort(key=lambda row: (str(row.get("candidate_config_id") or ""), str(row.get("target_mode") or ""), str(row.get("vocal_path") or ""), str(row.get("minus_path") or row.get("instrumental_path") or "")))
    return rows, existing


def source_hash(row: dict[str, Any]) -> str:
    stable_payload = {
        key: row.get(key)
        for key in sorted(row)
        if key not in {"created_at", "source_jsonl"}
    }
    return stable_hash(stable_payload)


def target_mode(row: dict[str, Any]) -> str:
    text = str(row.get("target_mode") or row.get("mode") or "").strip().lower()
    if text in {"vocals"}:
        return "vocal"
    if text in {"instrumental"}:
        return "minus"
    return text


def audio_path(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def failure_row(
    *,
    run_id: str,
    track_id: str,
    input_audio: Path,
    row: dict[str, Any],
    skip_reason: str,
    profile: str,
) -> dict[str, Any]:
    mode = target_mode(row)
    config_id = str(row.get("candidate_config_id") or row.get("config_id") or "unknown_config")
    family = str(row.get("candidate_family") or row.get("family") or mode or "unknown_family")
    return {
        "run_id": run_id,
        "track_id": track_id,
        "input_audio_path": str(input_audio),
        "mix_path": str(input_audio),
        "candidate_id": stable_id("failed", {"run_id": run_id, "config": config_id, "mode": mode, "reason": skip_reason}),
        "candidate_family": family,
        "candidate_config_id": config_id,
        "target_mode": mode or "unknown",
        "vocal_path": audio_path(row, "vocal_path"),
        "minus_path": audio_path(row, "minus_path", "instrumental_path"),
        "instrumental_path": audio_path(row, "minus_path", "instrumental_path"),
        "raw_vocal_path": audio_path(row, "raw_vocal_path"),
        "raw_minus_path": audio_path(row, "raw_minus_path"),
        "normalized_vocal_path": audio_path(row, "normalized_vocal_path"),
        "normalized_minus_path": audio_path(row, "normalized_minus_path"),
        "shape_normalization_applied": bool(row.get("shape_normalization_applied") is True),
        "shape_normalization_reason": str(row.get("shape_normalization_reason") or ""),
        "can_use_for_vocal": False,
        "can_use_for_minus": False,
        "can_use_for_pair": False,
        "generation_profile": profile,
        "generation_status": "skipped",
        "skip_reason": skip_reason,
        "source_tool": str(row.get("source_tool") or "existing_local_separator_output"),
        "source_model_id": str(row.get("source_model_id") or ""),
        "source_config_hash": source_hash(row),
        "job_manifest_path": str(row.get("job_manifest_path") or ""),
        "job_progress_path": str(row.get("job_progress_path") or ""),
        "created_at": DETERMINISTIC_CREATED_AT,
        "audio_rendered": False,
        "release_ready": False,
        **SAFETY_FLAGS,
    }


def copy_stem(source: Path, output_dir: Path, candidate_id: str, stem: str) -> str:
    suffix = source.suffix if source.suffix else ".wav"
    target = output_dir / candidate_id / f"{stem}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        raise ValueError(f"candidate_output_would_overwrite_source:{source}")
    shutil.copy2(source, target)
    return str(target)


def materialize_asset(
    *,
    run_id: str,
    track_id: str,
    input_audio: Path,
    output_dir: Path,
    row: dict[str, Any],
    profile: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    mode = target_mode(row)
    if mode not in ALLOWED_TARGET_MODES:
        return None, failure_row(run_id=run_id, track_id=track_id, input_audio=input_audio, row=row, skip_reason="unsupported_target_mode", profile=profile)
    vocal_source = audio_path(row, "vocal_path")
    minus_source = audio_path(row, "minus_path", "instrumental_path")
    raw_vocal_source = audio_path(row, "raw_vocal_path") or vocal_source
    raw_minus_source = audio_path(row, "raw_minus_path") or minus_source
    normalized_vocal_source = audio_path(row, "normalized_vocal_path") or vocal_source
    normalized_minus_source = audio_path(row, "normalized_minus_path") or minus_source
    missing: list[str] = []
    if mode in {"vocal", "both"} and (not vocal_source or not Path(vocal_source).exists()):
        missing.append("vocal_path")
    if mode in {"minus", "both"} and (not minus_source or not Path(minus_source).exists()):
        missing.append("minus_path")
    if missing:
        return None, failure_row(run_id=run_id, track_id=track_id, input_audio=input_audio, row=row, skip_reason="missing_" + "_and_".join(missing), profile=profile)
    config_id = str(row.get("candidate_config_id") or row.get("config_id") or stable_hash(row)[:12])
    family = str(row.get("candidate_family") or row.get("family") or f"{mode}_focused")
    candidate_id = stable_id(
        mode,
        {
            "input_audio": str(input_audio),
            "profile": profile,
            "candidate_config_id": config_id,
            "target_mode": mode,
            "source_config_hash": source_hash(row),
        },
    )
    try:
        copied_vocal = copy_stem(Path(vocal_source), output_dir, candidate_id, "vocals") if vocal_source and mode in {"vocal", "both"} else ""
        copied_minus = copy_stem(Path(minus_source), output_dir, candidate_id, "minus") if minus_source and mode in {"minus", "both"} else ""
    except Exception as exc:
        return None, failure_row(run_id=run_id, track_id=track_id, input_audio=input_audio, row=row, skip_reason=f"copy_failed:{type(exc).__name__}", profile=profile)
    return {
        "run_id": run_id,
        "track_id": track_id,
        "input_audio_path": str(input_audio),
        "mix_path": str(input_audio),
        "candidate_id": candidate_id,
        "candidate_family": family,
        "candidate_config_id": config_id,
        "target_mode": mode,
        "vocal_path": copied_vocal,
        "minus_path": copied_minus,
        "instrumental_path": copied_minus,
        "raw_vocal_path": raw_vocal_source if copied_vocal else "",
        "raw_minus_path": raw_minus_source if copied_minus else "",
        "normalized_vocal_path": normalized_vocal_source if copied_vocal else "",
        "normalized_minus_path": normalized_minus_source if copied_minus else "",
        "shape_normalization_applied": bool(row.get("shape_normalization_applied") is True),
        "shape_normalization_reason": str(row.get("shape_normalization_reason") or ""),
        "can_use_for_vocal": bool(copied_vocal),
        "can_use_for_minus": bool(copied_minus),
        "can_use_for_pair": bool(copied_vocal and copied_minus),
        "generation_profile": profile,
        "generation_status": "generated",
        "skip_reason": "",
        "source_tool": str(row.get("source_tool") or "existing_local_separator_output"),
        "source_model_id": str(row.get("source_model_id") or ""),
        "source_config_hash": source_hash(row),
        "job_manifest_path": str(row.get("job_manifest_path") or ""),
        "job_progress_path": str(row.get("job_progress_path") or ""),
        "created_at": DETERMINISTIC_CREATED_AT,
        "audio_rendered": False,
        "release_ready": False,
        **SAFETY_FLAGS,
    }, None


def pair_candidate(vocal: dict[str, Any], minus: dict[str, Any], *, max_index: int) -> dict[str, Any]:
    candidate_id = stable_id(
        "pair",
        {
            "vocal_candidate_id": vocal["candidate_id"],
            "minus_candidate_id": minus["candidate_id"],
            "vocal_config": vocal["candidate_config_id"],
            "minus_config": minus["candidate_config_id"],
            "index": max_index,
        },
    )
    return {
        "run_id": vocal["run_id"],
        "track_id": vocal["track_id"],
        "input_audio_path": vocal["input_audio_path"],
        "mix_path": vocal["mix_path"],
        "candidate_id": candidate_id,
        "candidate_family": f"{vocal['candidate_family']}+{minus['candidate_family']}",
        "candidate_config_id": f"{vocal['candidate_config_id']}+{minus['candidate_config_id']}",
        "target_mode": "both",
        "vocal_path": vocal["vocal_path"],
        "minus_path": minus["minus_path"],
        "instrumental_path": minus["minus_path"],
        "raw_vocal_path": str(vocal.get("raw_vocal_path") or vocal["vocal_path"]),
        "raw_minus_path": str(minus.get("raw_minus_path") or minus["minus_path"]),
        "normalized_vocal_path": str(vocal.get("normalized_vocal_path") or vocal["vocal_path"]),
        "normalized_minus_path": str(minus.get("normalized_minus_path") or minus["minus_path"]),
        "shape_normalization_applied": bool(vocal.get("shape_normalization_applied") is True or minus.get("shape_normalization_applied") is True),
        "shape_normalization_reason": "+".join(sorted({value for value in (str(vocal.get("shape_normalization_reason") or ""), str(minus.get("shape_normalization_reason") or "")) if value})),
        "can_use_for_vocal": True,
        "can_use_for_minus": True,
        "can_use_for_pair": True,
        "generation_profile": vocal["generation_profile"],
        "generation_status": "generated_with_shape_normalization" if bool(vocal.get("shape_normalization_applied") is True or minus.get("shape_normalization_applied") is True) else "generated",
        "skip_reason": "",
        "source_tool": "+".join(sorted({vocal["source_tool"], minus["source_tool"]})),
        "source_model_id": "+".join(sorted({value for value in (vocal["source_model_id"], minus["source_model_id"]) if value})),
        "source_config_hash": stable_hash({"vocal": vocal["source_config_hash"], "minus": minus["source_config_hash"]}),
        "job_manifest_path": "+".join(sorted({value for value in (str(vocal.get("job_manifest_path") or ""), str(minus.get("job_manifest_path") or "")) if value})),
        "job_progress_path": "+".join(sorted({value for value in (str(vocal.get("job_progress_path") or ""), str(minus.get("job_progress_path") or "")) if value})),
        "created_at": DETERMINISTIC_CREATED_AT,
        "audio_rendered": bool(vocal.get("audio_rendered") or minus.get("audio_rendered")),
        "release_ready": False,
        "is_complete_pair": True,
        **SAFETY_FLAGS,
    }


def build_pairs(assets: list[dict[str, Any]], max_candidates: int) -> list[dict[str, Any]]:
    direct_pairs = [row for row in assets if row["can_use_for_pair"] is True]
    vocals = [row for row in assets if row["can_use_for_vocal"] is True]
    minuses = [row for row in assets if row["can_use_for_minus"] is True]
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in sorted(direct_pairs, key=lambda item: (item["candidate_config_id"], item["candidate_id"])):
        key = (row["vocal_path"], row["minus_path"])
        if key in seen:
            continue
        seen.add(key)
        rows.append({**row, "is_complete_pair": True})
    pair_index = 0
    for vocal in sorted(vocals, key=lambda item: (item["candidate_config_id"], item["candidate_id"])):
        for minus in sorted(minuses, key=lambda item: (item["candidate_config_id"], item["candidate_id"])):
            key = (vocal["vocal_path"], minus["minus_path"])
            if key in seen:
                continue
            seen.add(key)
            pair_index += 1
            rows.append(pair_candidate(vocal, minus, max_index=pair_index))
    rows.sort(key=lambda row: (row["candidate_config_id"], row["candidate_id"], row["vocal_path"], row["minus_path"]))
    limit = max(0, int(max_candidates))
    return rows[:limit] if limit else rows


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# No-Reference Candidate Wave Summary",
        "",
        f"- candidate_execution_mode: {summary.get('candidate_execution_mode')}",
        f"- allow_shape_normalization: {str(summary.get('allow_shape_normalization') is True).lower()}",
        f"- candidate_diversity_profile: {summary.get('candidate_diversity_profile')}",
        f"- registry_configs_total: {summary.get('registry_configs_total', 0)}",
        f"- registry_configs_available: {summary.get('registry_configs_available', 0)}",
        f"- registry_configs_unavailable: {summary.get('registry_configs_unavailable', 0)}",
        f"- shared_jobs_planned: {summary.get('shared_jobs_planned', 0)}",
        f"- duplicate_jobs_avoided: {summary.get('duplicate_jobs_avoided', 0)}",
        f"- scheduler_skipped_jobs: {summary.get('scheduler_skipped_jobs', 0)}",
        f"- jobs_attempted: {summary.get('jobs_attempted', 0)}",
        f"- jobs_generated: {summary.get('jobs_generated', 0)}",
        f"- jobs_generated_raw: {summary.get('jobs_generated_raw', 0)}",
        f"- jobs_generated_with_shape_normalization: {summary.get('jobs_generated_with_shape_normalization', 0)}",
        f"- jobs_generated_with_late_timeout: {summary.get('jobs_generated_with_late_timeout', 0)}",
        f"- jobs_failed_timeout: {summary.get('jobs_failed_timeout', 0)}",
        f"- partial_outputs_recovered: {summary.get('partial_outputs_recovered', 0)}",
        f"- jobs_failed_shape_unrecoverable: {summary.get('jobs_failed_shape_unrecoverable', 0)}",
        f"- jobs_failed: {summary.get('jobs_failed', 0)}",
        f"- jobs_skipped: {summary.get('jobs_skipped', 0)}",
        f"- candidates_generated: {summary.get('candidates_generated', 0)}",
        f"- materialized_candidate_rows: {summary.get('materialized_candidate_rows', 0)}",
        f"- complete_pairs: {summary.get('complete_pairs', 0)}",
        f"- vocal_usable_candidates: {summary.get('vocal_usable_candidates', 0)}",
        f"- minus_usable_candidates: {summary.get('minus_usable_candidates', 0)}",
        f"- pair_usable_candidates: {summary.get('pair_usable_candidates', 0)}",
        f"- candidate_families_generated: {summary.get('candidate_families_generated', [])}",
        f"- target_modes_generated: {summary.get('target_modes_generated', [])}",
        f"- release_ready_count: {summary.get('release_ready_count', 0)}",
        f"- codex_required_at_runtime: {str(summary.get('codex_required_at_runtime') is True).lower()}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def wave_progress_row(phase: str, status: str, **extra: Any) -> dict[str, Any]:
    return {"phase": phase, "status": status, "created_at": DETERMINISTIC_CREATED_AT, **extra}


def job_manifest_to_asset(
    *,
    run_id: str,
    track_id: str,
    input_audio: Path,
    profile: str,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    mode = str(manifest.get("target_mode") or "").strip().lower()
    if mode not in ALLOWED_TARGET_MODES:
        row = {
            "candidate_config_id": str(manifest.get("candidate_config_id") or "unknown_config"),
            "candidate_family": str(manifest.get("candidate_family") or "unknown_family"),
            "target_mode": mode or "unknown",
            "source_tool": str(manifest.get("source_tool") or ""),
            "source_model_id": str(manifest.get("source_model_id") or ""),
            "source_config_hash": str(manifest.get("source_config_hash") or ""),
            "job_manifest_path": str(manifest.get("job_manifest_path") or ""),
            "job_progress_path": str(manifest.get("job_progress_path") or ""),
        }
        return None, failure_row(run_id=run_id, track_id=track_id, input_audio=input_audio, row=row, skip_reason="unsupported_target_mode", profile=profile)
    output_paths = manifest.get("output_paths") if isinstance(manifest.get("output_paths"), dict) else {}
    status = str(manifest.get("generation_status") or "")
    successful_generation = status in {"generated", "generated_with_shape_normalization", "generated_with_late_timeout"}
    row = {
        "candidate_config_id": str(manifest.get("candidate_config_id") or "unknown_config"),
        "candidate_family": str(manifest.get("candidate_family") or f"{mode}_focused"),
        "target_mode": mode,
        "vocal_path": str(output_paths.get("vocal_path") or ""),
        "minus_path": str(output_paths.get("minus_path") or ""),
        "instrumental_path": str(output_paths.get("minus_path") or ""),
        "raw_vocal_path": str(manifest.get("raw_vocal_path") or output_paths.get("vocal_path") or ""),
        "raw_minus_path": str(manifest.get("raw_minus_path") or output_paths.get("minus_path") or ""),
        "normalized_vocal_path": str(manifest.get("normalized_vocal_path") or output_paths.get("vocal_path") or ""),
        "normalized_minus_path": str(manifest.get("normalized_minus_path") or output_paths.get("minus_path") or ""),
        "shape_normalization_applied": bool(manifest.get("shape_normalization_applied") is True),
        "shape_normalization_reason": str(manifest.get("shape_normalization_reason") or ""),
        "source_tool": str(manifest.get("source_tool") or "local_separator_candidate_job"),
        "source_model_id": str(manifest.get("source_model_id") or ""),
        "source_config_hash": str(manifest.get("source_config_hash") or source_hash(manifest)),
        "job_manifest_path": str(manifest.get("job_manifest_path") or ""),
        "job_progress_path": str(manifest.get("job_progress_path") or ""),
    }
    if not successful_generation:
        failure = failure_row(
            run_id=run_id,
            track_id=track_id,
            input_audio=input_audio,
            row=row,
            skip_reason=str(manifest.get("skip_reason") or status or "job_not_generated"),
            profile=profile,
        )
        failure["generation_status"] = status or "failed_runtime_error"
        failure["audio_rendered"] = False
        return None, failure
    candidate_id = stable_id(
        mode,
        {
            "input_audio": str(input_audio),
            "profile": profile,
            "candidate_config_id": row["candidate_config_id"],
            "target_mode": mode,
            "source_config_hash": row["source_config_hash"],
            "job_manifest_path": row["job_manifest_path"],
        },
    )
    return {
        "run_id": run_id,
        "track_id": track_id,
        "input_audio_path": str(input_audio),
        "mix_path": str(input_audio),
        "candidate_id": candidate_id,
        "candidate_family": row["candidate_family"],
        "candidate_config_id": row["candidate_config_id"],
        "target_mode": mode,
        "vocal_path": row["vocal_path"],
        "minus_path": row["minus_path"],
        "instrumental_path": row["minus_path"],
        "raw_vocal_path": row["raw_vocal_path"],
        "raw_minus_path": row["raw_minus_path"],
        "normalized_vocal_path": row["normalized_vocal_path"],
        "normalized_minus_path": row["normalized_minus_path"],
        "shape_normalization_applied": row["shape_normalization_applied"],
        "shape_normalization_reason": row["shape_normalization_reason"],
        "can_use_for_vocal": bool(row["vocal_path"]) and mode in {"vocal", "both"},
        "can_use_for_minus": bool(row["minus_path"]) and mode in {"minus", "both"},
        "can_use_for_pair": bool(mode == "both" and row["vocal_path"] and row["minus_path"]),
        "generation_profile": profile,
        "generation_status": status,
        "skip_reason": "",
        "source_tool": row["source_tool"],
        "source_model_id": row["source_model_id"],
        "source_config_hash": row["source_config_hash"],
        "job_manifest_path": row["job_manifest_path"],
        "job_progress_path": row["job_progress_path"],
        "created_at": DETERMINISTIC_CREATED_AT,
        "audio_rendered": bool(manifest.get("audio_rendered")),
        "release_ready": False,
        **SAFETY_FLAGS,
    }, None


def runnable_configs(candidate_diversity_profile: str, max_candidates: int) -> list[CandidateJobConfig]:
    configs = sorted(build_candidate_job_registry(profile=candidate_diversity_profile), key=lambda config: config.candidate_config_id)
    limit = max(0, int(max_candidates))
    return configs[:limit] if limit else configs


def manifest_for_materialized_mode(manifest: dict[str, Any], mode: str) -> dict[str, Any]:
    output_paths = manifest.get("output_paths") if isinstance(manifest.get("output_paths"), dict) else {}
    raw_vocal_path = str(manifest.get("raw_vocal_path") or output_paths.get("vocal_path") or "")
    raw_minus_path = str(manifest.get("raw_minus_path") or output_paths.get("minus_path") or "")
    normalized_vocal_path = str(manifest.get("normalized_vocal_path") or output_paths.get("vocal_path") or "")
    normalized_minus_path = str(manifest.get("normalized_minus_path") or output_paths.get("minus_path") or "")
    if mode == "vocal":
        materialized_paths = {"vocal_path": str(output_paths.get("vocal_path") or ""), "minus_path": ""}
        raw_minus_path = ""
        normalized_minus_path = ""
    elif mode == "minus":
        materialized_paths = {"vocal_path": "", "minus_path": str(output_paths.get("minus_path") or "")}
        raw_vocal_path = ""
        normalized_vocal_path = ""
    else:
        materialized_paths = {"vocal_path": str(output_paths.get("vocal_path") or ""), "minus_path": str(output_paths.get("minus_path") or "")}
    return {
        **manifest,
        "target_mode": mode,
        "output_paths": materialized_paths,
        "raw_vocal_path": raw_vocal_path,
        "raw_minus_path": raw_minus_path,
        "normalized_vocal_path": normalized_vocal_path,
        "normalized_minus_path": normalized_minus_path,
    }


def job_manifest_to_assets(
    *,
    run_id: str,
    track_id: str,
    input_audio: Path,
    profile: str,
    manifest: dict[str, Any],
    materialized_modes: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    assets: list[dict[str, Any]] = []
    first_failure: dict[str, Any] | None = None
    for mode in materialized_modes:
        asset, failure = job_manifest_to_asset(
            run_id=run_id,
            track_id=track_id,
            input_audio=input_audio,
            profile=profile,
            manifest=manifest_for_materialized_mode(manifest, mode),
        )
        if asset is not None:
            assets.append(asset)
        if failure is not None and first_failure is None:
            first_failure = failure
    return assets, first_failure


def run_job_wave(
    *,
    run_id: str,
    track_id: str,
    input_audio: Path,
    output_dir: Path,
    profile: str,
    max_candidates: int,
    max_runtime_sec: int,
    continue_on_failure: bool,
    allow_shape_normalization: bool,
    candidate_diversity_profile: str,
    progress_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    configs = sorted(build_candidate_job_registry(profile=candidate_diversity_profile), key=lambda config: config.candidate_config_id)
    config_by_id = {config.candidate_config_id: config for config in configs}
    plan_rows = build_candidate_job_plan(
        input_audio_path=input_audio,
        registry_configs=configs,
        candidate_profile=profile,
        candidate_diversity_profile=candidate_diversity_profile,
        max_candidates=max_candidates,
        max_runtime_sec=max_runtime_sec,
    )
    write_candidate_job_plan(output_dir / "candidate_job_plan.jsonl", plan_rows)
    stats = {
        "registry_configs_total": len(configs),
        "registry_configs_available": sum(1 for config in configs if config.entrypoint_type != "unavailable"),
        "registry_configs_unavailable": sum(1 for config in configs if config.entrypoint_type == "unavailable"),
        "shared_jobs_planned": sum(1 for row in plan_rows if row.get("can_run") is True),
        "duplicate_jobs_avoided": sum(1 for row in plan_rows if row.get("reason_skipped") == "duplicate_shared_execution_group"),
        "scheduler_skipped_jobs": sum(1 for row in plan_rows if row.get("can_run") is not True),
        "jobs_attempted": 0,
        "jobs_generated": 0,
        "jobs_generated_raw": 0,
        "jobs_generated_with_shape_normalization": 0,
        "jobs_generated_with_late_timeout": 0,
        "jobs_failed_shape_unrecoverable": 0,
        "jobs_failed_timeout": 0,
        "partial_outputs_recovered": 0,
        "jobs_failed": 0,
        "jobs_skipped": 0,
    }
    assets: list[dict[str, Any]] = []
    materialized_assets: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    job_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for plan_row in plan_rows:
        if plan_row.get("can_run") is not True:
            continue
        config = config_by_id[str(plan_row["candidate_config_id"])]
        elapsed = time.monotonic() - started
        if max_runtime_sec > 0 and elapsed >= max_runtime_sec:
            progress_rows.append(wave_progress_row("run_jobs", "stopped_max_runtime_sec", candidate_config_id=config.candidate_config_id))
            break
        remaining_runtime_sec = max(1, int(max_runtime_sec - elapsed)) if max_runtime_sec > 0 else int(config.timeout_sec)
        stats["jobs_attempted"] += 1
        candidate_output_dir = output_dir / "jobs" / config.candidate_config_id
        manifest = run_local_separator_candidate_job(
            input_audio=input_audio,
            candidate_config_id=config.candidate_config_id,
            output_dir=candidate_output_dir,
            job_manifest_json=candidate_output_dir / "job_manifest.json",
            job_progress_jsonl=candidate_output_dir / "job_progress.jsonl",
            registry_configs=configs,
            profile=profile,
            timeout_sec=min(int(plan_row["timeout_sec"]), remaining_runtime_sec),
            allow_shape_normalization=allow_shape_normalization,
        )
        manifest["job_plan_id"] = str(plan_row["job_plan_id"])
        manifest["shared_execution_group_id"] = str(plan_row["shared_execution_group_id"])
        manifest["materialized_candidate_modes"] = list(plan_row.get("target_modes_produced") or [])
        write_json(candidate_output_dir / "job_manifest.json", manifest)
        job_rows.append(manifest)
        status = str(manifest.get("generation_status") or "")
        if status in {"generated", "generated_with_shape_normalization", "generated_with_late_timeout"}:
            stats["jobs_generated"] += 1
            if status == "generated_with_shape_normalization":
                stats["jobs_generated_with_shape_normalization"] += 1
            elif status == "generated_with_late_timeout":
                stats["jobs_generated_with_late_timeout"] += 1
                stats["partial_outputs_recovered"] += 1
            else:
                stats["jobs_generated_raw"] += 1
        elif status == "skipped_unavailable_entrypoint":
            stats["jobs_skipped"] += 1
        else:
            stats["jobs_failed"] += 1
            if status == "failed_invalid_audio_shape_unrecoverable":
                stats["jobs_failed_shape_unrecoverable"] += 1
            if "timeout" in status:
                stats["jobs_failed_timeout"] += 1
        job_assets, failure = job_manifest_to_assets(
            run_id=run_id,
            track_id=track_id,
            input_audio=input_audio,
            profile=profile,
            manifest=manifest,
            materialized_modes=tuple(str(mode) for mode in plan_row.get("target_modes_produced") or (config.target_mode,)),
        )
        if job_assets:
            assets.extend(job_assets)
            materialized_assets.extend(job_assets)
        if failure is not None:
            failures.append(failure)
        if failure is not None and not continue_on_failure:
            progress_rows.append(wave_progress_row("run_jobs", "stopped_after_failure", candidate_config_id=config.candidate_config_id, generation_status=status))
            break
    return assets, materialized_assets, failures, job_rows, plan_rows, stats


def build_summary(
    *,
    source_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    source_paths: list[Path],
    input_validation: dict[str, Any],
    execution_mode: str,
    allow_shape_normalization: bool,
    candidate_diversity_profile: str,
    job_rows: list[dict[str, Any]],
    plan_rows: list[dict[str, Any]],
    materialized_assets: list[dict[str, Any]],
    job_stats: dict[str, int],
) -> dict[str, Any]:
    return {
        "candidate_wave_executed": True,
        "candidate_execution_mode": execution_mode,
        "allow_shape_normalization": bool(allow_shape_normalization),
        "candidate_diversity_profile": candidate_diversity_profile,
        "source_jsonl_paths": [str(path) for path in source_paths],
        "input_validation": input_validation,
        "candidates_attempted": len(source_rows) + len(job_rows),
        "candidates_generated": len(rows),
        "materialized_candidate_rows": len(materialized_assets),
        "candidate_job_plan_rows": len(plan_rows),
        "complete_pairs": sum(1 for row in rows if row.get("is_complete_pair") is True),
        "vocal_usable_candidates": sum(1 for row in rows if row.get("can_use_for_vocal") is True),
        "minus_usable_candidates": sum(1 for row in rows if row.get("can_use_for_minus") is True),
        "pair_usable_candidates": sum(1 for row in rows if row.get("can_use_for_pair") is True),
        "failed_candidates": len(failures),
        **job_stats,
        "candidate_families_generated": sorted({str(row.get("candidate_family") or "") for row in rows if str(row.get("candidate_family") or "")}),
        "target_modes_generated": sorted({str(row.get("target_mode") or "") for row in rows if str(row.get("target_mode") or "")}),
        "skip_reasons": dict(sorted({reason: sum(1 for row in failures if row.get("skip_reason") == reason) for reason in {str(row.get("skip_reason") or "") for row in failures}}.items())),
        "release_ready_count": sum(1 for row in rows + failures if row.get("release_ready") is True),
        **SAFETY_FLAGS,
    }


def run_candidate_wave(
    *,
    input_audio: Path,
    output_dir: Path,
    candidate_pool_jsonl: Path,
    profile: str = "safe",
    source_jsonl: Path | None = None,
    max_candidates: int = 6,
    execution_mode: str = "collect_existing",
    max_runtime_sec: int = 600,
    continue_on_failure: bool = True,
    allow_shape_normalization: bool = True,
    candidate_diversity_profile: str = "safe",
) -> dict[str, Any]:
    if profile not in ALLOWED_PROFILES:
        raise ValueError(f"Unsupported candidate profile: {profile}")
    if candidate_diversity_profile not in ALLOWED_PROFILES:
        raise ValueError(f"Unsupported candidate diversity profile: {candidate_diversity_profile}")
    if execution_mode not in ALLOWED_EXECUTION_MODES:
        raise ValueError(f"Unsupported candidate execution mode: {execution_mode}")
    output_dir.mkdir(parents=True, exist_ok=True)
    input_validation = validate_input_audio(input_audio)
    source_rows, source_paths = load_source_rows(output_dir, source_jsonl)
    track_id = sanitize(input_audio.stem)
    run_id = stable_id(
        "candidate_wave",
        {
            "input_audio_path": str(input_audio),
            "input_audio_sha256": file_digest(input_audio) if input_validation["ok"] else "",
        "profile": profile,
        "candidate_diversity_profile": candidate_diversity_profile,
        "execution_mode": execution_mode,
        "allow_shape_normalization": bool(allow_shape_normalization),
        "source_rows": [{key: row.get(key) for key in sorted(row)} for row in source_rows],
    },
    )
    assets: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    job_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    materialized_assets: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = [
        wave_progress_row("start", "started", candidate_execution_mode=execution_mode, profile=profile)
    ]
    job_stats = {
        "registry_configs_total": 0,
        "registry_configs_available": 0,
        "registry_configs_unavailable": 0,
        "shared_jobs_planned": 0,
        "duplicate_jobs_avoided": 0,
        "scheduler_skipped_jobs": 0,
        "jobs_attempted": 0,
        "jobs_generated": 0,
        "jobs_generated_raw": 0,
        "jobs_generated_with_shape_normalization": 0,
        "jobs_generated_with_late_timeout": 0,
        "jobs_failed_shape_unrecoverable": 0,
        "jobs_failed_timeout": 0,
        "partial_outputs_recovered": 0,
        "jobs_failed": 0,
        "jobs_skipped": 0,
    }
    if not input_validation["ok"]:
        failures.append(
            failure_row(
                run_id=run_id,
                track_id=track_id,
                input_audio=input_audio,
                row={"candidate_config_id": "input_audio", "candidate_family": "input_validation", "target_mode": "both"},
                skip_reason=str(input_validation["reason"]),
                profile=profile,
            )
        )
    else:
        if execution_mode in {"collect_existing", "auto"}:
            progress_rows.append(wave_progress_row("collect_existing", "started", source_rows=len(source_rows)))
            for row in source_rows:
                asset, failure = materialize_asset(run_id=run_id, track_id=str(row.get("track_id") or track_id), input_audio=input_audio, output_dir=output_dir, row=row, profile=profile)
                if asset is not None:
                    assets.append(asset)
                if failure is not None:
                    failures.append(failure)
            progress_rows.append(wave_progress_row("collect_existing", "completed", assets=len(assets), failures=len(failures)))
        existing_rows = build_pairs(assets, max_candidates=max_candidates)
        should_run_jobs = execution_mode == "run_jobs" or (execution_mode == "auto" and len(existing_rows) < int(max_candidates))
        if should_run_jobs:
            progress_rows.append(wave_progress_row("run_jobs", "started", existing_pairs=len(existing_rows)))
            job_assets, job_materialized_assets, job_failures, job_rows, plan_rows, job_stats = run_job_wave(
                run_id=run_id,
                track_id=track_id,
                input_audio=input_audio,
                output_dir=output_dir,
                profile=profile,
                max_candidates=max_candidates,
                max_runtime_sec=max_runtime_sec,
                continue_on_failure=continue_on_failure,
                allow_shape_normalization=allow_shape_normalization,
                candidate_diversity_profile=candidate_diversity_profile,
                progress_rows=progress_rows,
            )
            assets.extend(job_assets)
            materialized_assets.extend(job_materialized_assets)
            failures.extend(job_failures)
            progress_rows.append(wave_progress_row("run_jobs", "completed", **job_stats))
        else:
            registry_configs = build_candidate_job_registry(profile=candidate_diversity_profile)
            job_stats = {
                "registry_configs_total": len(registry_configs),
                "registry_configs_available": sum(1 for config in registry_configs if config.entrypoint_type != "unavailable"),
                "registry_configs_unavailable": sum(1 for config in registry_configs if config.entrypoint_type == "unavailable"),
                "shared_jobs_planned": 0,
                "duplicate_jobs_avoided": 0,
                "scheduler_skipped_jobs": 0,
                "jobs_attempted": 0,
                "jobs_generated": 0,
                "jobs_generated_raw": 0,
                "jobs_generated_with_shape_normalization": 0,
                "jobs_generated_with_late_timeout": 0,
                "jobs_failed_shape_unrecoverable": 0,
                "jobs_failed_timeout": 0,
                "partial_outputs_recovered": 0,
                "jobs_failed": 0,
                "jobs_skipped": 0,
            }
    rows = build_pairs(assets, max_candidates=max_candidates)
    summary = build_summary(
        source_rows=source_rows,
        rows=rows,
        failures=failures,
        source_paths=source_paths,
        input_validation=input_validation,
        execution_mode=execution_mode,
        allow_shape_normalization=allow_shape_normalization,
        candidate_diversity_profile=candidate_diversity_profile,
        job_rows=job_rows,
        plan_rows=plan_rows,
        materialized_assets=materialized_assets,
        job_stats=job_stats,
    )
    progress_rows.append(wave_progress_row("write_outputs", "completed", candidates_generated=len(rows), failures=len(failures)))
    write_jsonl(candidate_pool_jsonl, rows)
    write_jsonl(output_dir / "candidate_wave_failures.jsonl", failures)
    write_jsonl(output_dir / "candidate_wave_jobs.jsonl", job_rows)
    write_jsonl(output_dir / "candidate_wave_materialized_assets.jsonl", materialized_assets)
    write_jsonl(output_dir / "candidate_wave_progress.jsonl", progress_rows)
    write_json(output_dir / "candidate_wave_summary.json", summary)
    write_summary_markdown(output_dir / "candidate_wave_summary.md", summary)
    write_json(
        output_dir / "candidate_wave_manifest.json",
        {
            "run_id": run_id,
            "input_audio_path": str(input_audio),
            "candidate_pool_jsonl": str(candidate_pool_jsonl),
            "profile": profile,
            "candidate_execution_mode": execution_mode,
            "allow_shape_normalization": bool(allow_shape_normalization),
            "candidate_diversity_profile": candidate_diversity_profile,
            "max_candidates": int(max_candidates),
            "max_runtime_sec": int(max_runtime_sec),
            "continue_on_failure": bool(continue_on_failure),
            "summary": summary,
            **SAFETY_FLAGS,
        },
    )
    return {"run_id": run_id, "rows": rows, "materialized_assets": materialized_assets, "failures": failures, "jobs": job_rows, "plan": plan_rows, "summary": summary}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an autonomous no-reference candidate generation/collection wave.")
    parser.add_argument("--input-audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-pool-jsonl", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(sorted(ALLOWED_PROFILES)), default="safe")
    parser.add_argument("--source-jsonl", type=Path)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--execution-mode", choices=tuple(sorted(ALLOWED_EXECUTION_MODES)), default="collect_existing")
    parser.add_argument("--max-runtime-sec", type=int, default=600)
    parser.add_argument("--continue-on-failure", type=parse_bool, default=True)
    parser.add_argument("--allow-shape-normalization", type=parse_bool, default=True)
    parser.add_argument("--candidate-diversity-profile", choices=tuple(sorted(ALLOWED_PROFILES)), default="safe")
    args = parser.parse_args(argv)
    result = run_candidate_wave(
        input_audio=args.input_audio,
        output_dir=args.output_dir,
        candidate_pool_jsonl=args.candidate_pool_jsonl,
        profile=args.profile,
        source_jsonl=args.source_jsonl,
        max_candidates=args.max_candidates,
        execution_mode=args.execution_mode,
        max_runtime_sec=args.max_runtime_sec,
        continue_on_failure=args.continue_on_failure,
        allow_shape_normalization=args.allow_shape_normalization,
        candidate_diversity_profile=args.candidate_diversity_profile,
    )
    print(json.dumps({"candidate_pool_jsonl": str(args.candidate_pool_jsonl), **result["summary"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
