from __future__ import annotations

import json
from pathlib import Path

from tools.replay_no_reference_harness_candidate import (
    default_harness_candidate,
    replay_harness_candidate,
    validate_harness_candidate,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def make_replay_bundle(tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "bundle"
    write_json(
        bundle_dir / "replay_bundle_manifest.json",
        {
            "bundle_id": "bundle-a",
            "run_count": 1,
            "replay_complete_runs": 1,
            "replay_incomplete_runs": 0,
            "release_ready_count": 0,
        },
    )
    write_jsonl(
        bundle_dir / "replay_bundle_events.jsonl",
        [
            {
                "event_type": "run_summary",
                "track_id": "track-a",
                "run_root": "/runs/track-a",
                "replay_complete": True,
                "output_status": "needs_more_candidates",
                "accepted_pair_count": 0,
                "preview_ready_count": 0,
                "final_safe_candidate_count": 0,
                "pair_compatible_count": 0,
                "minus_accepted_count": 0,
                "vocal_accepted_count": 1,
                "severe_vocal_bleed_in_minus_count": 2,
                "severe_phase_collapse_count": 1,
                "failed_safety_contract_count": 0,
                "preview_contract_missing_count": 0,
                "release_ready_count": 0,
                "estimated_gpu_jobs": 2,
                "estimated_runtime_sec": 120,
                "duplicate_jobs_avoided": 1,
                "candidate_count": 3,
                "failed_run_count": 0,
                "timeout_count": 0,
                "incomplete_replay_count": 0,
            }
        ],
    )
    write_jsonl(bundle_dir / "replay_bundle_artifacts_index.jsonl", [])
    return bundle_dir


def test_harness_candidate_cannot_mutate_forbidden_safety_fields() -> None:
    candidate = default_harness_candidate()
    candidate["mutations"] = {"release_ready_rules": "enable_release"}

    result = validate_harness_candidate(candidate)

    assert result["valid"] is False
    assert "forbidden_mutation_field:release_ready_rules" in result["rejection_reasons"]


def test_harness_candidate_schema_requires_module_groups() -> None:
    candidate = default_harness_candidate()
    candidate["module_config"].pop("memory")

    result = validate_harness_candidate(candidate)

    assert result["valid"] is False
    assert "missing_module_group:memory" in result["rejection_reasons"]


def test_release_ready_threshold_cleanup_and_download_mutations_are_rejected() -> None:
    candidate = default_harness_candidate()
    candidate.update(
        {
            "release_ready": True,
            "thresholds_relaxed": True,
            "cleanup_routes_added": True,
            "model_downloaded": True,
            "mutations": {"cleanup_debleed_denoise": "on"},
        }
    )

    result = validate_harness_candidate(candidate)

    assert result["valid"] is False
    assert "release_ready_forbidden" in result["rejection_reasons"]
    assert "threshold_relaxation_forbidden" in result["rejection_reasons"]
    assert "cleanup_debleed_denoise_forbidden" in result["rejection_reasons"]
    assert "model_download_forbidden" in result["rejection_reasons"]


def test_code_mutation_candidate_is_rejected() -> None:
    candidate = default_harness_candidate()
    candidate["mutations"] = {"source_code_generation": "edit_optimizer_py"}

    result = validate_harness_candidate(candidate)

    assert result["valid"] is False
    assert "forbidden_mutation_field:source_code_generation" in result["rejection_reasons"]


def test_stage_graph_mutation_cannot_skip_safety_pair_or_post_render() -> None:
    candidate = default_harness_candidate()
    candidate["stage_graph"] = {
        "nodes": ["candidate_wave", "windowed_judge_vocal", "windowed_judge_minus", "learned_critic", "selector", "preview_render"],
        "edges": [["candidate_wave", "selector"], ["selector", "preview_render"]],
        "optional_mutations": ["skip_pair_compatibility", "remove_safety_gate"],
    }

    result = validate_harness_candidate(candidate)

    assert result["valid"] is False
    assert "stage_graph_missing_required_node:pair_compatibility" in result["rejection_reasons"]
    assert "stage_graph_missing_required_node:post_render_judge" in result["rejection_reasons"]
    assert "stage_graph_forbidden_mutation:skip_pair_compatibility" in result["rejection_reasons"]
    assert "stage_graph_forbidden_mutation:remove_safety_gate" in result["rejection_reasons"]


def test_stage_graph_edges_must_preserve_required_safety_path() -> None:
    candidate = default_harness_candidate()
    candidate["stage_graph"] = {
        "nodes": [
            "candidate_wave",
            "windowed_judge_vocal",
            "windowed_judge_minus",
            "learned_critic",
            "selector",
            "pair_compatibility",
            "dry_run",
            "preview_render",
            "post_render_judge",
        ],
        "edges": [
            ["candidate_wave", "windowed_judge_vocal"],
            ["windowed_judge_vocal", "learned_critic"],
            ["learned_critic", "selector"],
            ["selector", "preview_render"],
            ["preview_render", "selector"],
        ],
        "optional_mutations": [],
    }

    result = validate_harness_candidate(candidate)

    assert result["valid"] is False
    assert "stage_graph_unsafe_edge_to_preview_render:selector" in result["rejection_reasons"]
    assert "stage_graph_required_path_missing:selector->pair_compatibility" in result["rejection_reasons"]
    assert "stage_graph_cycle_detected" in result["rejection_reasons"]


def test_invalid_hybrid_weight_is_rejected_without_exception() -> None:
    candidate = default_harness_candidate()
    candidate["learned_script_hybrid_weight"] = "not-a-number"

    result = validate_harness_candidate(candidate)

    assert result["valid"] is False
    assert "learned_script_hybrid_weight_out_of_range" in result["rejection_reasons"]


def test_valid_harness_candidate_replays_existing_bundle_without_release_ready(tmp_path: Path) -> None:
    bundle_dir = make_replay_bundle(tmp_path)
    candidate = default_harness_candidate()

    result = replay_harness_candidate(candidate=candidate, replay_bundle_dir=bundle_dir, output_jsonl=tmp_path / "eval.jsonl")

    assert result["safety_valid"] is True
    assert result["release_ready_count"] == 0
    assert result["failed_safety_contract_count"] == 0
    assert result["estimated_gpu_jobs"] == 2
    assert (tmp_path / "eval.jsonl").exists()


def test_manifest_preview_ready_is_informational_in_replay_score(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    write_json(
        bundle_dir / "replay_bundle_manifest.json",
        {"bundle_id": "bundle-a", "run_count": 2, "replay_complete_runs": 2, "replay_incomplete_runs": 0, "release_ready_count": 0},
    )
    write_jsonl(
        bundle_dir / "replay_bundle_events.jsonl",
        [
            {
                "event_type": "run_summary",
                "track_id": "manifest-only",
                "replay_complete": True,
                "accepted_pair_ready_count": 1,
                "manifest_preview_ready_count": 1,
                "audio_preview_ready_count": 0,
                "rendered_preview_audio_count": 0,
                "preview_render_failed_count": 0,
                "preview_render_skipped_count": 0,
                "preview_validation_failed_count": 0,
                "preview_contract_missing_count": 1,
                "preview_ready_count": 0,
                "minus_accepted_count": 1,
                "pair_compatible_count": 1,
                "vocal_accepted_count": 1,
                "severe_vocal_bleed_in_minus_count": 0,
                "severe_phase_collapse_count": 0,
                "failed_safety_contract_count": 0,
                "release_ready_count": 0,
            },
            {
                "event_type": "run_summary",
                "track_id": "audio-ready",
                "replay_complete": True,
                "accepted_pair_ready_count": 1,
                "manifest_preview_ready_count": 1,
                "audio_preview_ready_count": 1,
                "rendered_preview_audio_count": 2,
                "preview_render_failed_count": 0,
                "preview_render_skipped_count": 0,
                "preview_validation_failed_count": 0,
                "preview_contract_missing_count": 0,
                "preview_ready_count": 1,
                "minus_accepted_count": 1,
                "pair_compatible_count": 1,
                "vocal_accepted_count": 1,
                "severe_vocal_bleed_in_minus_count": 0,
                "severe_phase_collapse_count": 0,
                "failed_safety_contract_count": 0,
                "release_ready_count": 0,
            },
        ],
    )
    write_jsonl(bundle_dir / "replay_bundle_artifacts_index.jsonl", [])

    result = replay_harness_candidate(candidate=default_harness_candidate(), replay_bundle_dir=bundle_dir)

    assert result["accepted_pair_ready_count"] == 2
    assert result["manifest_preview_ready_count"] == 2
    assert result["audio_preview_ready_count"] == 1
    assert result["rendered_preview_audio_count"] == 2
    assert result["preview_render_failed_count"] == 0
    assert result["preview_validation_failed_count"] == 0
    assert result["preview_contract_missing_count"] == 1
    assert result["preview_ready_count"] == 1
    assert result["per_track_scores"]["audio-ready"] > result["per_track_scores"]["manifest-only"]


def test_preview_contract_missing_penalizes_replay_score(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    write_json(
        bundle_dir / "replay_bundle_manifest.json",
        {"bundle_id": "bundle-a", "run_count": 2, "replay_complete_runs": 2, "replay_incomplete_runs": 0, "release_ready_count": 0},
    )
    base_row = {
        "event_type": "run_summary",
        "replay_complete": True,
        "accepted_pair_ready_count": 1,
        "manifest_preview_ready_count": 1,
        "audio_preview_ready_count": 0,
        "rendered_preview_audio_count": 0,
        "preview_render_failed_count": 0,
        "preview_render_skipped_count": 0,
        "preview_validation_failed_count": 0,
        "minus_accepted_count": 1,
        "pair_compatible_count": 1,
        "vocal_accepted_count": 1,
        "severe_vocal_bleed_in_minus_count": 0,
        "severe_phase_collapse_count": 0,
        "failed_safety_contract_count": 0,
        "release_ready_count": 0,
    }
    write_jsonl(
        bundle_dir / "replay_bundle_events.jsonl",
        [
            {**base_row, "track_id": "contract-present", "preview_contract_missing_count": 0},
            {**base_row, "track_id": "contract-missing", "preview_contract_missing_count": 1},
        ],
    )
    write_jsonl(bundle_dir / "replay_bundle_artifacts_index.jsonl", [])

    result = replay_harness_candidate(candidate=default_harness_candidate(), replay_bundle_dir=bundle_dir)

    assert result["preview_contract_missing_count"] == 1
    assert result["per_track_scores"]["contract-present"] > result["per_track_scores"]["contract-missing"]
