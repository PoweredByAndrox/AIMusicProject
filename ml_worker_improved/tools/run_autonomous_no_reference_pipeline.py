from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import soundfile as sf

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_no_reference_ml_dataset import iter_jsonl, stable_unique, write_jsonl  # noqa: E402
from tools.no_reference_score_matrix import pair_compatibility_score  # noqa: E402
from tools.judge_no_reference_segment_blend_preview import build_preview_window_judge  # noqa: E402
from tools.run_no_reference_optimizer import optimize_tracks_mode_specific  # noqa: E402
from tools.run_no_reference_learned_critic import load_model_dir  # noqa: E402
from tools.run_no_reference_shadow_reranker import build_shadow_rerank  # noqa: E402
from tools.run_no_reference_windowed_judge import judge_candidate_rows_windowed  # noqa: E402
from tools.plan_no_reference_segment_blends import build_segment_blend_plan  # noqa: E402
from tools.prepare_no_reference_segment_blend_dry_run import build_dry_run_contract  # noqa: E402
from tools.render_no_reference_segment_blend_preview import build_preview_result  # noqa: E402
from tools.run_no_reference_candidate_wave import run_candidate_wave  # noqa: E402
from tools.no_reference_minus_hard_case_router import run_minus_hard_case_router  # noqa: E402
from tools.render_no_reference_audio_preview import render_audio_preview_contract  # noqa: E402
from tools.build_music_intelligence_input_manifest import (  # noqa: E402
    music_intelligence_run_section,
    write_music_intelligence_input_manifest,
)


ALLOWED_RUNTIME_STATUSES = {
    "accepted_pair_ready",
    "manifest_preview_ready",
    "audio_preview_ready",
    "preview_render_skipped",
    "preview_render_failed",
    "preview_blocked_safety_contract",
    "accepted_for_listen_pack",
    "blocked_hard_case",
    "needs_more_candidates",
    "needs_pair_reconciliation",
    "segment_blend_needed",
    "failed_safety_contract",
}

REQUIRED_OUTPUTS = {
    "run_manifest": "run_manifest.json",
    "runtime_summary": "runtime_summary.md",
    "selected_candidates": "selected_candidates.jsonl",
    "learned_critic_scores": "learned_critic_scores.jsonl",
    "pair_compatibility": "pair_compatibility.jsonl",
    "preview_manifest": "preview_manifest.json",
    "rendered_preview_audio": "rendered_preview_audio.jsonl",
    "preview_render_contract": "preview_render_contract.json",
    "preview_render_summary": "preview_render_summary.md",
    "post_render_judge": "post_render_judge.jsonl",
    "learning_outcome": "learning_outcome.jsonl",
    "music_intelligence_input_manifest": "music_intelligence_input_manifest.json",
}

SAFETY_FLAGS = {
    "codex_required_at_runtime": False,
    "manual_phase_execution_required": False,
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "separator_training_started": False,
    "models_changed": False,
    "model_downloaded": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "release_ready": False,
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_empty_jsonl_if_missing(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    write_jsonl(path, [])


def ensure_output_files(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in REQUIRED_OUTPUTS.items():
        path = output_dir / filename
        if path.exists():
            continue
        if filename.endswith(".jsonl"):
            path.write_text("", encoding="utf-8")
        elif filename.endswith(".md"):
            path.write_text("# Autonomous No-Reference Runtime Summary\n\n", encoding="utf-8")
        else:
            write_json(path, {"artifact": key, "status": "pending", **SAFETY_FLAGS})


def stage(name: str, status: str, **payload: Any) -> dict[str, Any]:
    return {"stage": name, "status": status, **payload}


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {value}")


def docker_visible_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def validate_input_audio(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "reason": "input_audio_missing", "path": str(path)}
    if path.stat().st_size <= 0:
        return {"ok": False, "reason": "input_audio_empty", "path": str(path)}
    try:
        info = sf.info(str(path))
    except Exception as exc:  # pragma: no cover - exercised by corrupt user input.
        return {"ok": False, "reason": f"input_audio_unreadable:{type(exc).__name__}", "path": str(path)}
    return {
        "ok": True,
        "path": str(path),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_sec": round(float(info.frames) / float(info.samplerate), 6) if info.samplerate else 0.0,
    }


def load_candidate_rows(output_dir: Path, candidate_pool_jsonl: Path | None) -> tuple[list[dict[str, Any]], Path]:
    source = candidate_pool_jsonl or (output_dir / "candidate_pool.jsonl")
    if not source.exists():
        return [], source
    return list(iter_jsonl(source)), source


def validate_candidate_pool_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    usable: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for row in rows:
        minus_path = str(row.get("minus_path") or row.get("instrumental_path") or "")
        reasons: list[str] = []
        if row.get("release_ready") is True:
            reasons.append("release_ready_candidate_forbidden")
        if row.get("is_complete_pair") is False:
            reasons.append("incomplete_pair")
        if not str(row.get("mix_path") or "") or not Path(str(row.get("mix_path") or "")).exists():
            reasons.append("missing_mix_path")
        if not str(row.get("vocal_path") or "") or not Path(str(row.get("vocal_path") or "")).exists():
            reasons.append("missing_vocal_path")
        if not minus_path or not Path(minus_path).exists():
            reasons.append("missing_minus_path")
        if reasons:
            invalid.append({**row, "is_complete_pair": False, "skip_reason": "_and_".join(reasons), "release_ready": False})
            continue
        normalized = {**row, "minus_path": minus_path, "instrumental_path": minus_path, "is_complete_pair": True, "release_ready": False}
        usable.append(normalized)
    usable.sort(key=lambda item: (str(item.get("track_id") or ""), str(item.get("candidate_id") or ""), str(item.get("vocal_path") or ""), str(item.get("minus_path") or "")))
    invalid.sort(key=lambda item: (str(item.get("track_id") or ""), str(item.get("candidate_id") or ""), str(item.get("skip_reason") or "")))
    return usable, invalid


def write_windowed_result(result: dict[str, Any], *, output_jsonl: Path, summary_json: Path) -> None:
    write_jsonl(output_jsonl, result["rows"])
    write_json(summary_json, result["summary"])


def is_accept(decision: Any) -> bool:
    return str(decision or "") in {"accept_candidate", "pair_compatible"}


def select_best_candidate(rows: Iterable[dict[str, Any]], mode: str) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if str(row.get("score_mode") or "") == mode
        and is_accept(row.get("script_decision"))
        and is_accept(row.get("final_guarded_decision"))
        and not row.get("script_blockers")
        and row.get("can_publish_as_result") is not True
        and row.get("release_ready") is not True
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (
            -float(row.get("hybrid_score") or 0.0),
            -float(row.get("script_score") or 0.0),
            str(row.get("track_id") or ""),
            str(row.get("candidate_id") or ""),
        ),
    )[0]


def minus_hard_case_blockers(rows: Iterable[dict[str, Any]]) -> list[str]:
    blockers: set[str] = set()
    for row in rows:
        for key in ("minus_blockers", "blockers", "failure_reasons"):
            value = row.get(key)
            if isinstance(value, list):
                blockers.update(str(item) for item in value)
            elif isinstance(value, dict):
                blockers.update(str(item) for item in value)
            elif value:
                blockers.add(str(value))
        for key in ("minus_dominant_defect", "dominant_defect", "failure_reason", "judge_reason"):
            if row.get(key):
                blockers.add(str(row[key]))
    return sorted(
        blocker
        for blocker in blockers
        if blocker in {"severe_vocal_bleed_in_minus", "severe_phase_collapse", "severe_phase_comb_filtering"}
    )


def evaluate_pair_compatibility(vocal_row: dict[str, Any], minus_row: dict[str, Any]) -> dict[str, Any]:
    pair = pair_compatibility_score(vocal_row, minus_row)
    compatible = pair["pair_decision"] == "pair_compatible"
    return {
        "track_id": str(vocal_row.get("track_id") or minus_row.get("track_id") or ""),
        "selected_vocal_candidate_id": str(vocal_row.get("candidate_id") or ""),
        "selected_minus_candidate_id": str(minus_row.get("candidate_id") or ""),
        **pair,
        "runtime_status": "accepted_pair_ready" if compatible else "needs_pair_reconciliation",
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "release_ready": False,
    }


def model_version(model_dir: Path) -> dict[str, Any]:
    manifest_path = model_dir / "model_manifest.json"
    if not manifest_path.exists():
        return {"model_dir": str(model_dir), "model_manifest": "", "model_version": "unknown"}
    manifest = read_json(manifest_path)
    return {
        "model_dir": str(model_dir),
        "model_manifest": str(manifest_path),
        "model_family": str(manifest.get("model_family") or ""),
        "model_version": str(manifest.get("model_version") or manifest.get("training_data_hash") or "unknown"),
        "training_data_hash": str(manifest.get("training_data_hash") or ""),
        "safety": manifest.get("safety") if isinstance(manifest.get("safety"), dict) else {},
    }


def preview_manifest_for(status: str, selected_rows: list[dict[str, Any]], pair_row: dict[str, Any] | None) -> dict[str, Any]:
    pair_row = pair_row or {}
    selected = selected_rows[0] if selected_rows else {}
    accepted_pair_ready = pair_row.get("pair_decision") == "pair_compatible"
    preview_status = status if status in {
        "accepted_pair_ready",
        "manifest_preview_ready",
        "audio_preview_ready",
        "preview_render_skipped",
        "preview_render_failed",
        "preview_blocked_safety_contract",
    } else ("preview_blocked_safety_contract" if status in {"failed_safety_contract", "needs_more_candidates", "blocked_hard_case"} else status)
    return {
        "run_id": "",
        "track_id": str(selected.get("track_id") or pair_row.get("track_id") or ""),
        "preview_status": preview_status,
        "runtime_status": status,
        "accepted_pair_ready": accepted_pair_ready,
        "manifest_preview_ready": bool(selected_rows and accepted_pair_ready),
        "audio_preview_ready": False,
        "preview_render_attempted": False,
        "preview_render_skipped": False,
        "preview_render_failed": preview_status == "preview_blocked_safety_contract",
        "presentation_status": "accepted_for_listen_pack" if status in {"preview_ready", "audio_preview_ready"} else status,
        "release_status": "listen_preview_only",
        "delivery_allowed": False,
        "selected_vocal_candidate_id": str(selected.get("selected_vocal_candidate_id") or pair_row.get("selected_vocal_candidate_id") or ""),
        "selected_minus_candidate_id": str(selected.get("selected_minus_candidate_id") or pair_row.get("selected_minus_candidate_id") or ""),
        "selected_vocal_path": str(selected.get("selected_vocal_path") or ""),
        "selected_minus_path": str(selected.get("selected_minus_path") or ""),
        "rendered_vocal_preview_path": "",
        "rendered_minus_preview_path": "",
        "rendered_pair_preview_path": "",
        "duration_sec": 0.0,
        "sample_rate": 0,
        "channels": 0,
        "validation_status": "blocked" if preview_status == "preview_blocked_safety_contract" else "pending",
        "failure_reason": "",
        "skip_reason": "",
        "selected_candidates": selected_rows,
        "pair_compatibility": pair_row,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "release_ready": False,
        "rendered_preview_audio": [],
        "rows": [],
        "summary": {},
        **SAFETY_FLAGS,
    }


def rendered_preview_paths(rows: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    for row in rows:
        for key in ("rendered_vocal_preview_path", "rendered_minus_preview_path", "rendered_pair_preview_path", "rendered_path"):
            value = str(row.get(key) or "")
            if value:
                paths.append(value)
    return stable_unique(paths)


def preview_run_section(preview_manifest: dict[str, Any]) -> dict[str, Any]:
    rendered = preview_manifest.get("rendered_preview_audio")
    rendered_count = len(rendered) if isinstance(rendered, list) else int(preview_manifest.get("rendered_preview_audio_count") or 0)
    return {
        "preview_status": str(preview_manifest.get("preview_status") or ""),
        "accepted_pair_ready": preview_manifest.get("accepted_pair_ready") is True,
        "manifest_preview_ready": preview_manifest.get("manifest_preview_ready") is True,
        "audio_preview_ready": preview_manifest.get("audio_preview_ready") is True,
        "preview_render_attempted": preview_manifest.get("preview_render_attempted") is True,
        "preview_render_skipped": preview_manifest.get("preview_render_skipped") is True,
        "preview_render_failed": preview_manifest.get("preview_render_failed") is True,
        "rendered_preview_audio_count": int(rendered_count),
        "rendered_vocal_preview_path": str(preview_manifest.get("rendered_vocal_preview_path") or ""),
        "rendered_minus_preview_path": str(preview_manifest.get("rendered_minus_preview_path") or ""),
        "validation_status": str(preview_manifest.get("validation_status") or ""),
        "failure_reason": str(preview_manifest.get("failure_reason") or ""),
        "skip_reason": str(preview_manifest.get("skip_reason") or ""),
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "release_ready": False,
    }


def learning_outcome_row(
    *,
    status: str,
    input_audio: Path,
    selected_vocal: dict[str, Any] | None,
    selected_minus: dict[str, Any] | None,
    model_info: dict[str, Any],
    blockers: list[str],
    music_intelligence: dict[str, Any] | None = None,
    preview: dict[str, Any] | None = None,
) -> dict[str, Any]:
    music_intelligence = music_intelligence or {}
    preview = preview or {}
    return {
        "track_id": Path(input_audio).stem,
        "input_audio": str(input_audio),
        "output_status": status,
        "selected_vocal_candidate_id": str((selected_vocal or {}).get("candidate_id") or ""),
        "selected_minus_candidate_id": str((selected_minus or {}).get("candidate_id") or ""),
        "learned_model_version": model_info.get("model_version", "unknown"),
        "safety_blockers": blockers,
        "verified_outcome": False,
        "raw_prediction_only": True,
        "can_train_from_row": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "preview_status": str(preview.get("preview_status") or ""),
        "accepted_pair_ready": preview.get("accepted_pair_ready") is True,
        "manifest_preview_ready": preview.get("manifest_preview_ready") is True,
        "audio_preview_ready": preview.get("audio_preview_ready") is True,
        "preview_render_attempted": preview.get("preview_render_attempted") is True,
        "preview_render_skipped": preview.get("preview_render_skipped") is True,
        "preview_render_failed": preview.get("preview_render_failed") is True,
        "rendered_preview_audio_count": int(preview.get("rendered_preview_audio_count") or 0),
        "music_intelligence_manifest_written": music_intelligence.get("manifest_written") is True,
        "music_intelligence_analysis_ready_stems_count": int(music_intelligence.get("analysis_ready_stems_count") or 0),
        "music_intelligence_blocked_stems_count": int(music_intelligence.get("blocked_stems_count") or 0),
        "melody_transcription_ran": False,
        "recommendation_index_updated": False,
    }


def write_music_intelligence_contract(output_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    manifest_path = output_dir / REQUIRED_OUTPUTS["music_intelligence_input_manifest"]
    write_music_intelligence_input_manifest(
        run_dir=output_dir,
        output_path=manifest_path,
        runtime_manifest={**manifest, "release_ready": False},
    )
    payload = read_json(manifest_path)
    return music_intelligence_run_section(payload, manifest_path)


def write_runtime_summary(path: Path, manifest: dict[str, Any]) -> None:
    music = manifest.get("music_intelligence") if isinstance(manifest.get("music_intelligence"), dict) else {}
    preview = manifest.get("preview") if isinstance(manifest.get("preview"), dict) else {}
    lines = [
        "# Autonomous No-Reference Runtime Summary",
        "",
        f"- one_command_runtime_available: true",
        f"- codex_required_at_runtime: false",
        f"- output_status: {manifest['output_status']}",
        f"- selected_vocal_candidate: {manifest.get('selected_vocal_candidate') or ''}",
        f"- selected_minus_candidate: {manifest.get('selected_minus_candidate') or ''}",
        f"- learned_model_version: {manifest.get('learned_model', {}).get('model_version', 'unknown')}",
        f"- candidate_wave_executed: {str(manifest.get('candidate_wave_executed') is True).lower()}",
        f"- candidate_execution_mode: {manifest.get('candidate_execution_mode') or ''}",
        f"- allow_shape_normalization: {str(manifest.get('allow_shape_normalization') is True).lower()}",
        f"- candidate_diversity_profile: {manifest.get('candidate_diversity_profile') or ''}",
        f"- escalation_executed: {str(manifest.get('escalation_executed') is True).lower()}",
        f"- escalated_profile: {manifest.get('escalated_profile') or ''}",
        f"- escalation_reason: {manifest.get('escalation_reason') or ''}",
        f"- candidate_pool_jsonl: {manifest.get('candidate_pool_jsonl') or ''}",
        f"- release_ready: false",
        f"- preview_status: {preview.get('preview_status') or ''}",
        f"- accepted_pair_ready: {str(preview.get('accepted_pair_ready') is True).lower()}",
        f"- manifest_preview_ready: {str(preview.get('manifest_preview_ready') is True).lower()}",
        f"- audio_preview_ready: {str(preview.get('audio_preview_ready') is True).lower()}",
        f"- preview_render_attempted: {str(preview.get('preview_render_attempted') is True).lower()}",
        f"- preview_render_skipped: {str(preview.get('preview_render_skipped') is True).lower()}",
        f"- preview_render_failed: {str(preview.get('preview_render_failed') is True).lower()}",
        f"- rendered_preview_audio_count: {int(preview.get('rendered_preview_audio_count') or 0)}",
        f"- rendered_vocal_preview_path: {preview.get('rendered_vocal_preview_path') or ''}",
        f"- rendered_minus_preview_path: {preview.get('rendered_minus_preview_path') or ''}",
        f"- preview_validation_status: {preview.get('validation_status') or ''}",
        f"- preview_failure_reason: {preview.get('failure_reason') or ''}",
        f"- preview_skip_reason: {preview.get('skip_reason') or ''}",
        f"- music_intelligence_manifest_written: {str(music.get('manifest_written') is True).lower()}",
        f"- music_intelligence_analysis_ready_stems_count: {int(music.get('analysis_ready_stems_count') or 0)}",
        f"- music_intelligence_blocked_stems_count: {int(music.get('blocked_stems_count') or 0)}",
        f"- future_analysis_supported: false",
        f"- melody_transcription_ran: false",
        f"- recommendation_index_updated: false",
        "",
        "## Candidate Wave",
        "",
    ]
    wave = manifest.get("candidate_wave_summary") if isinstance(manifest.get("candidate_wave_summary"), dict) else {}
    for key in (
        "candidate_execution_mode",
        "registry_configs_total",
        "registry_configs_available",
        "registry_configs_unavailable",
        "jobs_attempted",
        "jobs_generated",
        "jobs_failed",
        "jobs_skipped",
        "candidates_attempted",
        "candidates_generated",
        "complete_pairs",
        "vocal_usable_candidates",
        "minus_usable_candidates",
        "pair_usable_candidates",
        "failed_candidates",
        "release_ready_count",
    ):
        if key in wave:
            lines.append(f"- {key}: {wave[key]}")
    lines.extend(
        [
            "",
            "## Stages",
            "",
        ]
    )
    for row in manifest.get("stages", []):
        lines.append(f"- {row['stage']}: {row['status']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_failure_manifest(
    *,
    output_dir: Path,
    input_audio: Path,
    output_status: str,
    reason: str,
    stages: list[dict[str, Any]],
    model_info: dict[str, Any],
    candidate_wave_executed: bool = False,
    candidate_wave_summary: dict[str, Any] | None = None,
    candidate_pool_jsonl: Path | None = None,
    candidate_execution_mode: str = "",
    profile: str = "safe",
    allow_shape_normalization: bool = True,
    candidate_diversity_profile: str = "safe",
    escalation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blockers = stable_unique([reason])
    write_empty_jsonl_if_missing(output_dir / REQUIRED_OUTPUTS["selected_candidates"])
    write_empty_jsonl_if_missing(output_dir / REQUIRED_OUTPUTS["learned_critic_scores"])
    write_empty_jsonl_if_missing(output_dir / REQUIRED_OUTPUTS["pair_compatibility"])
    preview_payload = preview_manifest_for(output_status, [], None)
    preview_payload["failure_reason"] = reason
    write_json(output_dir / REQUIRED_OUTPUTS["preview_manifest"], preview_payload)
    write_empty_jsonl_if_missing(output_dir / REQUIRED_OUTPUTS["rendered_preview_audio"])
    write_json(
        output_dir / REQUIRED_OUTPUTS["preview_render_contract"],
        {
            "contract_version": "no_reference_audio_preview_render_v1",
            "preview_status": preview_payload["preview_status"],
            "rendered_preview_audio_count": 0,
            "failure_reason": reason,
            **SAFETY_FLAGS,
        },
    )
    (output_dir / REQUIRED_OUTPUTS["preview_render_summary"]).write_text(
        "\n".join(
            [
                "# No-Reference Audio Preview Render Contract",
                "",
                f"- preview_status: {preview_payload['preview_status']}",
                "- audio_preview_ready: false",
                "- rendered_preview_audio_count: 0",
                f"- failure_reason: {reason}",
                "- can_publish_as_result: false",
                "- release_ready: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_empty_jsonl_if_missing(output_dir / REQUIRED_OUTPUTS["post_render_judge"])
    preview_section = preview_run_section(preview_payload)
    manifest = {
        "input_audio": str(input_audio),
        "output_dir": str(output_dir),
        "output_status": output_status,
        "failure_reason": reason,
        "stages": stages,
        "selected_vocal_candidate": "",
        "selected_minus_candidate": "",
        "safety_blockers": blockers,
        "learned_model": model_info,
        "candidate_wave_executed": bool(candidate_wave_executed),
        "candidate_wave_summary": candidate_wave_summary or {},
        "candidate_execution_mode": candidate_execution_mode,
        "profile": profile,
        "allow_shape_normalization": bool(allow_shape_normalization),
        "candidate_diversity_profile": candidate_diversity_profile,
        "candidate_pool_jsonl": str(candidate_pool_jsonl or output_dir / "candidate_pool.jsonl"),
        "preview": preview_section,
        **(escalation or {}),
        **SAFETY_FLAGS,
    }
    manifest["artifacts"] = {name: str(output_dir / filename) for name, filename in REQUIRED_OUTPUTS.items()}
    manifest["music_intelligence"] = write_music_intelligence_contract(output_dir, manifest)
    stages.append(stage("write_music_intelligence_input_manifest", "completed", **manifest["music_intelligence"]))
    write_jsonl(
        output_dir / REQUIRED_OUTPUTS["learning_outcome"],
        [
            learning_outcome_row(
                status=output_status,
                input_audio=input_audio,
                selected_vocal=None,
                selected_minus=None,
                model_info=model_info,
                blockers=blockers,
                music_intelligence=manifest["music_intelligence"],
                preview=manifest["preview"],
            )
        ],
    )
    write_json(output_dir / REQUIRED_OUTPUTS["run_manifest"], manifest)
    write_runtime_summary(output_dir / REQUIRED_OUTPUTS["runtime_summary"], manifest)
    return manifest


def run_pipeline(
    *,
    input_audio: Path,
    output_dir: Path,
    model_dir: Path,
    profile: str = "safe",
    candidate_pool_jsonl: Path | None = None,
    generate_candidates: bool = True,
    candidate_profile: str = "safe",
    max_candidates: int = 6,
    candidate_wave_source_jsonl: Path | None = None,
    candidate_execution_mode: str = "collect_existing",
    max_candidate_runtime_sec: int = 600,
    continue_on_candidate_failure: bool = True,
    allow_shape_normalization: bool = True,
    candidate_diversity_profile: str = "safe",
    auto_escalate_profile: bool = False,
    enable_minus_hard_case_routing: bool = False,
    max_hard_case_extra_jobs: int = 4,
    hard_case_runtime_sec: int = 1800,
    window_seconds: float = 12.0,
    max_windows_per_candidate: int = 5,
    window_step_seconds: float = 0.0,
    enable_audio_preview_render: bool = True,
    preview_duration_sec: float = 30.0,
    preview_start_sec: float | None = None,
    preview_render_mode: str = "bounded",
    require_preview_audio_for_preview_ready: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_output_files(output_dir)
    stages: list[dict[str, Any]] = []
    model_info = model_version(model_dir)
    candidate_source = candidate_pool_jsonl or (output_dir / "candidate_pool.jsonl")
    candidate_wave_executed = False
    candidate_wave_summary: dict[str, Any] = {}
    escalation = {
        "initial_profile": candidate_diversity_profile,
        "escalated_profile": "",
        "escalation_reason": "",
        "escalation_executed": False,
        "additional_jobs_attempted": 0,
        "additional_candidates_generated": 0,
    }
    minus_hard_case = {
        "minus_hard_case_routing_enabled": bool(enable_minus_hard_case_routing),
        "minus_hard_case_route_executed": False,
        "minus_hard_case_recommended_route": "",
        "minus_hard_case_available_route_configs": [],
        "minus_hard_case_extra_jobs_attempted": 0,
        "minus_hard_case_extra_jobs_generated": 0,
        "minus_hard_case_extra_candidates_generated": 0,
        "minus_hard_case_initial_blockers": [],
        "minus_hard_case_final_blockers": [],
    }

    validation = validate_input_audio(input_audio)
    stages.append(stage("validate_input_audio", "completed" if validation["ok"] else "failed", **validation))
    if not validation["ok"]:
        return build_failure_manifest(
            output_dir=output_dir,
            input_audio=input_audio,
            output_status="failed_safety_contract",
            reason=str(validation["reason"]),
            stages=stages,
            model_info=model_info,
            candidate_wave_executed=candidate_wave_executed,
            candidate_wave_summary=candidate_wave_summary,
            candidate_pool_jsonl=candidate_source,
            candidate_execution_mode=candidate_execution_mode,
            profile=profile,
            allow_shape_normalization=allow_shape_normalization,
            candidate_diversity_profile=candidate_diversity_profile,
            escalation=escalation,
        )

    if generate_candidates:
        source_jsonl = candidate_wave_source_jsonl
        if source_jsonl is None and (output_dir / "candidate_wave_sources.jsonl").exists():
            source_jsonl = output_dir / "candidate_wave_sources.jsonl"
        wave_result = run_candidate_wave(
            input_audio=input_audio,
            output_dir=output_dir / "candidates",
            candidate_pool_jsonl=candidate_source,
            profile=candidate_profile,
            source_jsonl=source_jsonl,
            max_candidates=max_candidates,
            execution_mode=candidate_execution_mode,
            max_runtime_sec=max_candidate_runtime_sec,
            continue_on_failure=continue_on_candidate_failure,
            allow_shape_normalization=allow_shape_normalization,
            candidate_diversity_profile=candidate_diversity_profile,
        )
        candidate_wave_executed = True
        candidate_wave_summary = wave_result["summary"]
        stages.append(stage("run_candidate_wave", "completed", candidate_pool_jsonl=str(candidate_source), **candidate_wave_summary))
        should_escalate = (
            auto_escalate_profile
            and candidate_diversity_profile == "safe"
            and (
                int(candidate_wave_summary.get("complete_pairs") or 0) <= 1
                or int(candidate_wave_summary.get("pair_usable_candidates") or 0) <= 1
                or int(candidate_wave_summary.get("jobs_failed_timeout") or 0) > 0
            )
        )
        if should_escalate:
            escalation_reason = "safe_profile_insufficient_candidate_diversity"
            if int(candidate_wave_summary.get("complete_pairs") or 0) == 0:
                escalation_reason = "safe_profile_zero_complete_pairs"
            elif int(candidate_wave_summary.get("jobs_failed_timeout") or 0) > 0:
                escalation_reason = "safe_profile_timeout_or_partial_jobs"
            balanced_result = run_candidate_wave(
                input_audio=input_audio,
                output_dir=output_dir / "candidates_balanced",
                candidate_pool_jsonl=candidate_source,
                profile="balanced",
                source_jsonl=source_jsonl,
                max_candidates=max_candidates,
                execution_mode=candidate_execution_mode,
                max_runtime_sec=max_candidate_runtime_sec,
                continue_on_failure=continue_on_candidate_failure,
                allow_shape_normalization=allow_shape_normalization,
                candidate_diversity_profile="balanced",
            )
            candidate_wave_summary = balanced_result["summary"]
            escalation.update(
                {
                    "escalated_profile": "balanced",
                    "escalation_reason": escalation_reason,
                    "escalation_executed": True,
                    "additional_jobs_attempted": int(candidate_wave_summary.get("jobs_attempted") or 0),
                    "additional_candidates_generated": int(candidate_wave_summary.get("candidates_generated") or 0),
                }
            )
            stages.append(stage("run_candidate_wave_escalated", "completed", candidate_pool_jsonl=str(candidate_source), **candidate_wave_summary, **escalation))
    else:
        stages.append(stage("run_candidate_wave", "skipped_generate_candidates_false", candidate_pool_jsonl=str(candidate_source)))

    candidate_rows, candidate_source = load_candidate_rows(output_dir, candidate_source)
    usable_candidate_rows, invalid_candidate_rows = validate_candidate_pool_rows(candidate_rows)
    write_jsonl(output_dir / "candidate_pool_invalid_rows.jsonl", invalid_candidate_rows)
    stages.append(
        stage(
            "validate_candidate_pool",
            "completed" if usable_candidate_rows else "failed",
            candidate_pool_jsonl=str(candidate_source),
            raw_rows=len(candidate_rows),
            usable_rows=len(usable_candidate_rows),
            invalid_rows=len(invalid_candidate_rows),
        )
    )
    candidate_rows = usable_candidate_rows
    stages.append(stage("generate_or_collect_candidate_pairs", "completed" if candidate_rows else "failed", candidate_pool_jsonl=str(candidate_source), rows=len(candidate_rows)))
    if not candidate_rows:
        reason = "candidate_pool_missing_or_empty" if not candidate_rows and not invalid_candidate_rows else "candidate_pool_has_no_usable_complete_pairs"
        return build_failure_manifest(
            output_dir=output_dir,
            input_audio=input_audio,
            output_status="needs_more_candidates",
            reason=reason,
            stages=stages,
            model_info=model_info,
            candidate_wave_executed=candidate_wave_executed,
            candidate_wave_summary=candidate_wave_summary,
            candidate_pool_jsonl=candidate_source,
            candidate_execution_mode=candidate_execution_mode,
            profile=profile,
            allow_shape_normalization=allow_shape_normalization,
            candidate_diversity_profile=candidate_diversity_profile,
            escalation=escalation,
        )

    vocal_judge = judge_candidate_rows_windowed(
        candidate_rows,
        cache_jsonl=output_dir / "vocal_windowed_metric_cache.jsonl",
        window_seconds=float(window_seconds),
        max_windows_per_candidate=int(max_windows_per_candidate),
        window_step_seconds=float(window_step_seconds),
        score_mode="vocal",
    )
    write_windowed_result(vocal_judge, output_jsonl=output_dir / "vocal_windowed_judge.jsonl", summary_json=output_dir / "vocal_windowed_judge_summary.json")
    stages.append(stage("run_vocal_mode_judge", "completed", rows=len(vocal_judge["rows"])))

    minus_judge = judge_candidate_rows_windowed(
        candidate_rows,
        cache_jsonl=output_dir / "minus_windowed_metric_cache.jsonl",
        window_seconds=float(window_seconds),
        max_windows_per_candidate=int(max_windows_per_candidate),
        window_step_seconds=float(window_step_seconds),
        score_mode="minus",
    )
    write_windowed_result(minus_judge, output_jsonl=output_dir / "minus_windowed_judge.jsonl", summary_json=output_dir / "minus_windowed_judge_summary.json")
    stages.append(stage("run_minus_mode_judge", "completed", rows=len(minus_judge["rows"])))

    optimizer_result = optimize_tracks_mode_specific(vocal_judge["rows"], minus_judge["rows"])
    write_jsonl(output_dir / "optimizer_mode_specific_result.jsonl", optimizer_result["rows"])
    stages.append(stage("run_optimizer", "completed", rows=len(optimizer_result["rows"])))

    shadow_result = build_shadow_rerank(
        candidate_pool_rows=candidate_rows,
        windowed_judge_rows=vocal_judge["rows"] + minus_judge["rows"],
        optimizer_result_rows=optimizer_result["rows"],
        bundles=load_model_dir(model_dir),
    )
    write_jsonl(output_dir / REQUIRED_OUTPUTS["learned_critic_scores"], shadow_result["rows"])
    write_jsonl(output_dir / "learned_critic_disagreements.jsonl", shadow_result["disagreements"])
    stages.append(stage("run_learned_critic_shadow_hybrid_scoring", "completed", rows=len(shadow_result["rows"])))

    selected_vocal = select_best_candidate(shadow_result["rows"], "vocal")
    stages.append(stage("select_best_vocal_candidate", "completed" if selected_vocal else "failed", candidate_id=str((selected_vocal or {}).get("candidate_id") or "")))
    selected_minus = select_best_candidate(shadow_result["rows"], "minus")
    stages.append(stage("select_best_minus_candidate", "completed" if selected_minus else "failed", candidate_id=str((selected_minus or {}).get("candidate_id") or "")))
    if selected_vocal is None or selected_minus is None:
        initial_minus_blockers = minus_hard_case_blockers(minus_judge["rows"])
        minus_hard_case["minus_hard_case_initial_blockers"] = initial_minus_blockers
        if (
            enable_minus_hard_case_routing
            and selected_minus is None
            and initial_minus_blockers
        ):
            router_summary = run_minus_hard_case_router(
                run_dir=output_dir,
                output_jsonl=output_dir / "minus_hard_case_router.jsonl",
                summary_md=output_dir / "minus_hard_case_router_summary.md",
            )
            recommended_route = str(router_summary.get("recommended_route") or "")
            available_route_configs = list(router_summary.get("available_route_configs") or [])
            minus_hard_case.update(
                {
                    "minus_hard_case_route_executed": True,
                    "minus_hard_case_recommended_route": recommended_route,
                    "minus_hard_case_available_route_configs": available_route_configs,
                }
            )
            stages.append(
                stage(
                    "minus_hard_case_router",
                    "completed",
                    recommended_route=recommended_route,
                    available_route_configs=available_route_configs,
                    can_run_without_new_models=router_summary.get("can_run_without_new_models") is True,
                )
            )
            if router_summary.get("can_run_without_new_models") is True and recommended_route:
                hard_case_wave = run_candidate_wave(
                    input_audio=input_audio,
                    output_dir=output_dir / "candidates_minus_hard_case",
                    candidate_pool_jsonl=candidate_source,
                    profile="balanced",
                    source_jsonl=candidate_source,
                    max_candidates=max_candidates,
                    max_job_count=int(max_hard_case_extra_jobs),
                    execution_mode="auto",
                    max_runtime_sec=int(hard_case_runtime_sec),
                    continue_on_failure=continue_on_candidate_failure,
                    allow_shape_normalization=allow_shape_normalization,
                    candidate_diversity_profile="balanced",
                    candidate_route_filter=recommended_route,
                )
                candidate_wave_summary = hard_case_wave["summary"]
                minus_hard_case.update(
                    {
                        "minus_hard_case_extra_jobs_attempted": int(candidate_wave_summary.get("jobs_attempted") or 0),
                        "minus_hard_case_extra_jobs_generated": int(candidate_wave_summary.get("jobs_generated") or 0),
                        "minus_hard_case_extra_candidates_generated": int(candidate_wave_summary.get("candidates_generated") or 0),
                    }
                )
                stages.append(stage("run_minus_hard_case_candidate_wave", "completed", candidate_pool_jsonl=str(candidate_source), **candidate_wave_summary))
                candidate_rows, candidate_source = load_candidate_rows(output_dir, candidate_source)
                usable_candidate_rows, invalid_candidate_rows = validate_candidate_pool_rows(candidate_rows)
                write_jsonl(output_dir / "candidate_pool_invalid_rows.jsonl", invalid_candidate_rows)
                candidate_rows = usable_candidate_rows
                vocal_judge = judge_candidate_rows_windowed(
                    candidate_rows,
                    cache_jsonl=output_dir / "vocal_windowed_metric_cache.jsonl",
                    window_seconds=float(window_seconds),
                    max_windows_per_candidate=int(max_windows_per_candidate),
                    window_step_seconds=float(window_step_seconds),
                    score_mode="vocal",
                )
                write_windowed_result(vocal_judge, output_jsonl=output_dir / "vocal_windowed_judge.jsonl", summary_json=output_dir / "vocal_windowed_judge_summary.json")
                minus_judge = judge_candidate_rows_windowed(
                    candidate_rows,
                    cache_jsonl=output_dir / "minus_windowed_metric_cache.jsonl",
                    window_seconds=float(window_seconds),
                    max_windows_per_candidate=int(max_windows_per_candidate),
                    window_step_seconds=float(window_step_seconds),
                    score_mode="minus",
                )
                write_windowed_result(minus_judge, output_jsonl=output_dir / "minus_windowed_judge.jsonl", summary_json=output_dir / "minus_windowed_judge_summary.json")
                minus_hard_case["minus_hard_case_final_blockers"] = minus_hard_case_blockers(minus_judge["rows"])
                optimizer_result = optimize_tracks_mode_specific(vocal_judge["rows"], minus_judge["rows"])
                write_jsonl(output_dir / "optimizer_mode_specific_result.jsonl", optimizer_result["rows"])
                shadow_result = build_shadow_rerank(
                    candidate_pool_rows=candidate_rows,
                    windowed_judge_rows=vocal_judge["rows"] + minus_judge["rows"],
                    optimizer_result_rows=optimizer_result["rows"],
                    bundles=load_model_dir(model_dir),
                )
                write_jsonl(output_dir / REQUIRED_OUTPUTS["learned_critic_scores"], shadow_result["rows"])
                write_jsonl(output_dir / "learned_critic_disagreements.jsonl", shadow_result["disagreements"])
                selected_vocal = select_best_candidate(shadow_result["rows"], "vocal")
                selected_minus = select_best_candidate(shadow_result["rows"], "minus")
                stages.append(stage("rerun_hard_case_vocal_selection", "completed" if selected_vocal else "failed", candidate_id=str((selected_vocal or {}).get("candidate_id") or "")))
                stages.append(stage("rerun_hard_case_minus_selection", "completed" if selected_minus else "failed", candidate_id=str((selected_minus or {}).get("candidate_id") or "")))
                if selected_vocal is not None and selected_minus is not None:
                    pass
                else:
                    return build_failure_manifest(
                        output_dir=output_dir,
                        input_audio=input_audio,
                        output_status="needs_more_candidates",
                        reason="minus_hard_case_route_exhausted",
                        stages=stages,
                        model_info=model_info,
                        candidate_wave_executed=candidate_wave_executed,
                        candidate_wave_summary=candidate_wave_summary,
                        candidate_pool_jsonl=candidate_source,
                        candidate_execution_mode=candidate_execution_mode,
                        profile=profile,
                        allow_shape_normalization=allow_shape_normalization,
                        candidate_diversity_profile=candidate_diversity_profile,
                        escalation={**escalation, **minus_hard_case},
                    )
            else:
                return build_failure_manifest(
                    output_dir=output_dir,
                    input_audio=input_audio,
                    output_status="blocked_hard_case",
                    reason="minus_hard_case_route_unavailable",
                    stages=stages,
                    model_info=model_info,
                    candidate_wave_executed=candidate_wave_executed,
                    candidate_wave_summary=candidate_wave_summary,
                    candidate_pool_jsonl=candidate_source,
                    candidate_execution_mode=candidate_execution_mode,
                    profile=profile,
                    allow_shape_normalization=allow_shape_normalization,
                    candidate_diversity_profile=candidate_diversity_profile,
                    escalation={**escalation, **minus_hard_case},
                )
        if selected_vocal is not None and selected_minus is not None:
            pass
        else:
            return build_failure_manifest(
                output_dir=output_dir,
                input_audio=input_audio,
                output_status="needs_more_candidates",
                reason="missing_safe_mode_candidate",
                stages=stages,
                model_info=model_info,
                candidate_wave_executed=candidate_wave_executed,
                candidate_wave_summary=candidate_wave_summary,
                candidate_pool_jsonl=candidate_source,
                candidate_execution_mode=candidate_execution_mode,
                profile=profile,
                allow_shape_normalization=allow_shape_normalization,
                candidate_diversity_profile=candidate_diversity_profile,
                escalation={**escalation, **minus_hard_case},
            )

    selected_rows = [
        {
            "track_id": str(selected_vocal.get("track_id") or selected_minus.get("track_id") or ""),
            "selected_vocal_candidate_id": str(selected_vocal.get("candidate_id") or ""),
            "selected_minus_candidate_id": str(selected_minus.get("candidate_id") or ""),
            "selected_vocal_path": str(selected_vocal.get("vocal_path") or ""),
            "selected_minus_path": str(selected_minus.get("instrumental_path") or selected_minus.get("minus_path") or ""),
            "can_publish_as_result": False,
            "can_mark_release_ready": False,
            "release_ready": False,
        }
    ]
    write_jsonl(output_dir / REQUIRED_OUTPUTS["selected_candidates"], selected_rows)

    pair_row = evaluate_pair_compatibility(selected_vocal, selected_minus)
    write_jsonl(output_dir / REQUIRED_OUTPUTS["pair_compatibility"], [pair_row])
    stages.append(stage("run_pair_compatibility", "completed", pair_decision=pair_row["pair_decision"]))
    extra_blockers: list[str] = []

    if pair_row["pair_decision"] == "pair_compatible":
        output_status = "accepted_pair_ready"
        stages.append(stage("create_preview_listen_candidate", "completed", accepted_pair_ready=True))
        stages.append(stage("segment_blend_planner", "skipped_pair_compatible"))
        stages.append(stage("dry_run_render_contract", "skipped_pair_compatible"))
        write_json(output_dir / REQUIRED_OUTPUTS["preview_manifest"], preview_manifest_for(output_status, selected_rows, pair_row))
        preview_payload = render_audio_preview_contract(
            run_dir=output_dir,
            preview_duration_sec=float(preview_duration_sec),
            preview_start_sec=preview_start_sec,
            preview_render_mode=preview_render_mode,
            enable_audio_preview_render=bool(enable_audio_preview_render),
            require_preview_audio_for_preview_ready=bool(require_preview_audio_for_preview_ready),
        )
        output_status = str(preview_payload.get("preview_status") or "manifest_preview_ready")
        if output_status in {"preview_render_failed", "preview_blocked_safety_contract"}:
            extra_blockers.append(str(preview_payload.get("failure_reason") or output_status))
        stages.append(
            stage(
                "render_audio_preview_contract",
                "completed" if preview_payload.get("audio_preview_ready") is True else str(preview_payload.get("preview_status") or "completed"),
                preview_status=output_status,
                rendered_preview_audio_count=int(preview_payload.get("rendered_preview_audio_count") or 0),
            )
        )
        post_render_rows = [
            {
                "track_id": selected_rows[0]["track_id"],
                "post_render_status": output_status,
                "requires_post_render_judge": preview_payload.get("audio_preview_ready") is True,
                "rendered_preview_audio_count": int(preview_payload.get("rendered_preview_audio_count") or 0),
                "validation_status": str(preview_payload.get("validation_status") or ""),
                "failure_reason": str(preview_payload.get("failure_reason") or ""),
                "skip_reason": str(preview_payload.get("skip_reason") or ""),
                "can_publish_as_result": False,
                "can_mark_release_ready": False,
                "release_ready": False,
            }
        ]
        write_jsonl(output_dir / REQUIRED_OUTPUTS["post_render_judge"], post_render_rows)
        stages.append(stage("targeted_post_render_judge", "completed", rows=len(post_render_rows)))
    else:
        segment_plan = build_segment_blend_plan(candidate_rows, vocal_judge["rows"] + minus_judge["rows"], optimizer_result["rows"])
        write_jsonl(output_dir / "segment_blend_plan.jsonl", segment_plan["rows"])
        stages.append(stage("segment_blend_planner", "completed", rows=len(segment_plan["rows"])))
        dry_run = build_dry_run_contract(
            segment_plan["rows"],
            windowed_judge_rows=vocal_judge["rows"] + minus_judge["rows"],
            candidate_pool_rows=candidate_rows,
            optimizer_rows=optimizer_result["rows"],
        )
        write_jsonl(output_dir / "segment_blend_dry_run.jsonl", dry_run["rows"])
        write_jsonl(output_dir / "segment_blend_render_contract.jsonl", dry_run["render_contract_rows"])
        stages.append(stage("dry_run_render_contract", "completed", rows=len(dry_run["render_contract_rows"])))
        output_status = "segment_blend_needed" if segment_plan["rows"] else "needs_pair_reconciliation"
        preview_payload = preview_manifest_for(output_status, selected_rows, pair_row)
        try:
            preview_result = build_preview_result(
                dry_run["rows"],
                dry_run["render_contract_rows"],
                output_dir / "rendered_previews",
                max_render_count=1,
            )
        except Exception as exc:  # pragma: no cover - defensive user-artifact safety path.
            output_status = "failed_safety_contract"
            preview_payload = preview_manifest_for(output_status, selected_rows, pair_row)
            preview_payload["render_failure_reason"] = f"{type(exc).__name__}:{exc}"
            extra_blockers.append(str(preview_payload["render_failure_reason"]))
            stages.append(stage("render_bounded_preview", "failed", reason=preview_payload["render_failure_reason"]))
            write_json(output_dir / REQUIRED_OUTPUTS["preview_manifest"], preview_payload)
            write_jsonl(output_dir / REQUIRED_OUTPUTS["post_render_judge"], [])
            stages.append(stage("targeted_post_render_judge", "skipped_render_failed"))
        else:
            preview_rows = [row for row in preview_result["rows"] if isinstance(row, dict)]
            preview_payload.update(
                {
                    "manifest_version": preview_result.get("manifest_version"),
                    "rows": preview_rows,
                    "summary": preview_result.get("summary") if isinstance(preview_result.get("summary"), dict) else {},
                    "rendered_preview_audio": rendered_preview_paths(preview_rows),
                }
            )
            write_json(output_dir / REQUIRED_OUTPUTS["preview_manifest"], preview_payload)
            stages.append(stage("render_bounded_preview", "completed" if preview_rows else "skipped_no_render_ready_contract", rows=len(preview_rows)))
            if preview_rows:
                post_judge = build_preview_window_judge(
                    preview_payload,
                    dry_run["render_contract_rows"],
                    dry_run["rows"],
                    vocal_judge["rows"] + minus_judge["rows"],
                )
                write_jsonl(output_dir / REQUIRED_OUTPUTS["post_render_judge"], post_judge["rows"])
                write_json(output_dir / "post_render_judge_summary.json", post_judge["summary"])
                stages.append(stage("targeted_post_render_judge", "completed", rows=len(post_judge["rows"])))
            else:
                write_jsonl(output_dir / REQUIRED_OUTPUTS["post_render_judge"], [])
                stages.append(stage("targeted_post_render_judge", "skipped_no_preview_render"))

    blockers = stable_unique(list(pair_row.get("pair_blockers") or []) + extra_blockers)
    preview_payload = read_json(output_dir / REQUIRED_OUTPUTS["preview_manifest"])
    preview_section = preview_run_section(preview_payload)
    manifest = {
        "input_audio": str(input_audio),
        "output_dir": str(output_dir),
        "profile": profile,
        "output_status": output_status,
        "stages": stages,
        "selected_vocal_candidate": str(selected_vocal.get("candidate_id") or ""),
        "selected_minus_candidate": str(selected_minus.get("candidate_id") or ""),
        "safety_blockers": blockers,
        "learned_model": model_info,
        "candidate_wave_executed": bool(candidate_wave_executed),
        "candidate_wave_summary": candidate_wave_summary,
        "candidate_execution_mode": candidate_execution_mode,
        "allow_shape_normalization": bool(allow_shape_normalization),
        "candidate_diversity_profile": candidate_diversity_profile,
        "candidate_pool_jsonl": str(candidate_source),
        "artifacts": {name: str(output_dir / filename) for name, filename in REQUIRED_OUTPUTS.items()},
        "preview": preview_section,
        **escalation,
        **minus_hard_case,
        **SAFETY_FLAGS,
    }
    if output_status not in ALLOWED_RUNTIME_STATUSES:
        manifest["output_status"] = "failed_safety_contract"
        manifest["safety_blockers"] = stable_unique(blockers + ["invalid_runtime_status"])
    manifest["music_intelligence"] = write_music_intelligence_contract(output_dir, manifest)
    stages.append(stage("write_music_intelligence_input_manifest", "completed", **manifest["music_intelligence"]))
    write_jsonl(
        output_dir / REQUIRED_OUTPUTS["learning_outcome"],
        [
            learning_outcome_row(
                status=manifest["output_status"],
                input_audio=input_audio,
                selected_vocal=selected_vocal,
                selected_minus=selected_minus,
                model_info=model_info,
                blockers=manifest["safety_blockers"],
                music_intelligence=manifest["music_intelligence"],
                preview=manifest["preview"],
            )
        ],
    )
    stages.append(stage("write_learning_outcome_row", "completed", rows=1))
    write_json(output_dir / REQUIRED_OUTPUTS["run_manifest"], manifest)
    write_runtime_summary(output_dir / REQUIRED_OUTPUTS["runtime_summary"], manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local autonomous no-reference vocal/minus runtime pipeline.")
    parser.add_argument("--input-audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("safe", "balanced"), default="safe")
    parser.add_argument("--candidate-pool-jsonl", type=Path)
    parser.add_argument("--generate-candidates", type=parse_bool, default=True)
    parser.add_argument("--candidate-profile", choices=("safe", "balanced", "exploration"), default="safe")
    parser.add_argument("--candidate-wave-source-jsonl", type=Path)
    parser.add_argument("--candidate-execution-mode", choices=("collect_existing", "run_jobs", "auto"), default="collect_existing")
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--max-candidate-runtime-sec", type=int, default=600)
    parser.add_argument("--continue-on-candidate-failure", type=parse_bool, default=True)
    parser.add_argument("--allow-shape-normalization", type=parse_bool, default=True)
    parser.add_argument("--candidate-diversity-profile", choices=("safe", "balanced", "exploration"), default="safe")
    parser.add_argument("--auto-escalate-profile", type=parse_bool, default=False)
    parser.add_argument("--enable-minus-hard-case-routing", type=parse_bool, default=False)
    parser.add_argument("--max-hard-case-extra-jobs", type=int, default=4)
    parser.add_argument("--hard-case-runtime-sec", type=int, default=1800)
    parser.add_argument("--window-seconds", type=float, default=12.0)
    parser.add_argument("--max-windows-per-candidate", type=int, default=5)
    parser.add_argument("--window-step-seconds", type=float, default=0.0)
    parser.add_argument("--enable-audio-preview-render", type=parse_bool, default=True)
    parser.add_argument("--preview-duration-sec", type=float, default=30.0)
    parser.add_argument("--preview-start-sec", type=float)
    parser.add_argument("--preview-render-mode", choices=("bounded", "full"), default="bounded")
    parser.add_argument("--require-preview-audio-for-preview-ready", type=parse_bool, default=True)
    args = parser.parse_args(argv)
    manifest = run_pipeline(
        input_audio=args.input_audio,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        profile=args.profile,
        candidate_pool_jsonl=args.candidate_pool_jsonl,
        generate_candidates=args.generate_candidates,
        candidate_profile=args.candidate_profile,
        max_candidates=args.max_candidates,
        candidate_wave_source_jsonl=args.candidate_wave_source_jsonl,
        candidate_execution_mode=args.candidate_execution_mode,
        max_candidate_runtime_sec=args.max_candidate_runtime_sec,
        continue_on_candidate_failure=args.continue_on_candidate_failure,
        allow_shape_normalization=args.allow_shape_normalization,
        candidate_diversity_profile=args.candidate_diversity_profile,
        auto_escalate_profile=args.auto_escalate_profile,
        enable_minus_hard_case_routing=args.enable_minus_hard_case_routing,
        max_hard_case_extra_jobs=args.max_hard_case_extra_jobs,
        hard_case_runtime_sec=args.hard_case_runtime_sec,
        window_seconds=args.window_seconds,
        max_windows_per_candidate=args.max_windows_per_candidate,
        window_step_seconds=args.window_step_seconds,
        enable_audio_preview_render=args.enable_audio_preview_render,
        preview_duration_sec=args.preview_duration_sec,
        preview_start_sec=args.preview_start_sec,
        preview_render_mode=args.preview_render_mode,
        require_preview_audio_for_preview_ready=args.require_preview_audio_for_preview_ready,
    )
    print(json.dumps({"output_status": manifest["output_status"], "output_dir": str(args.output_dir), "release_ready": False}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
