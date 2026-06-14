from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.optimize_no_reference_harness import mutate_candidate
from tools.replay_no_reference_harness_candidate import default_harness_candidate
from tools.run_no_reference_harness_runtime_validation import (
    map_candidate_to_runtime_args,
    run_harness_runtime_validation,
)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def frontier_row(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "harness_candidate_id": candidate["harness_candidate_id"],
        "parent_id": candidate.get("parent_id", ""),
        "mutation_type": candidate["mutation_type"],
        "selection_decision": "candidate_pareto_selected",
        "safety_valid": True,
        "candidate_config": candidate,
        "release_ready": False,
        "thresholds_relaxed": False,
        "cleanup_routes_added": False,
        "model_downloaded": False,
        "model_registry_edited": False,
        "training_started": False,
        "separator_training_started": False,
    }


def make_runtime_artifacts(
    run_dir: Path,
    *,
    stem_quality: str,
    preview_audio: bool = True,
    output_status: str = "preview_ready",
    preview_status: str | None = None,
    preview_validation_status: str = "validated",
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    phase_count = 3 if stem_quality == "baseline" else 1
    bleed_count = 4 if stem_quality == "baseline" else 2
    accepted_minus = 1 if stem_quality == "baseline" else 3
    pair_compatible = 1 if stem_quality == "baseline" else 2
    status = output_status
    preview_status = preview_status or output_status
    preview_paths: list[str] = []
    if preview_audio:
        preview = run_dir / "preview.wav"
        preview.write_bytes(b"RIFFunit-test")
        preview_paths.append(str(preview))
    manifest = {
        "output_status": status,
        "release_ready": False,
        "reference_required": False,
        "human_csv_required": False,
        "training_started": False,
        "separator_training_started": False,
        "model_downloaded": False,
        "model_registry_edited": False,
        "thresholds_relaxed": False,
        "cleanup_routes_added": False,
        "candidate_wave_summary": {
            "jobs_attempted": 4,
            "jobs_generated": 4,
            "duplicate_jobs_avoided": 1,
            "complete_pairs": 2,
            "vocal_usable_candidates": 2,
            "minus_usable_candidates": accepted_minus,
            "pair_usable_candidates": pair_compatible,
            "release_ready_count": 0,
        },
        "music_intelligence": {
            "manifest_written": True,
            "analysis_ready_stems_count": 2,
            "blocked_stems_count": 0,
            "future_analysis_supported": False,
            "melody_transcription_ran": False,
            "recommendation_index_updated": False,
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    (run_dir / "runtime_summary.md").write_text("- music_intelligence_manifest_written: true\n", encoding="utf-8")
    write_jsonl(
        run_dir / "candidate_pool.jsonl",
        [
            {"candidate_id": "pair-a", "is_complete_pair": True, "can_use_for_vocal": True, "can_use_for_minus": True, "can_use_for_pair": True},
            {"candidate_id": "pair-b", "is_complete_pair": True, "can_use_for_vocal": True, "can_use_for_minus": accepted_minus > 1, "can_use_for_pair": pair_compatible > 1},
        ],
    )
    write_jsonl(
        run_dir / "learned_critic_scores.jsonl",
        [
            {"candidate_id": "pair-a", "score_mode": "vocal", "final_guarded_decision": "accept_candidate", "script_decision": "accept_candidate", "script_blockers": []},
            {"candidate_id": "pair-a", "score_mode": "minus", "final_guarded_decision": "accept_candidate", "script_decision": "accept_candidate", "script_blockers": []},
            {"candidate_id": "pair-b", "score_mode": "minus", "final_guarded_decision": "accept_candidate" if accepted_minus > 1 else "blocked_hard_case", "script_decision": "accept_candidate" if accepted_minus > 1 else "blocked_hard_case", "script_blockers": [] if accepted_minus > 1 else ["severe_phase_collapse"]},
            {"candidate_id": "pair-c", "score_mode": "minus", "final_guarded_decision": "accept_candidate" if accepted_minus > 2 else "blocked_hard_case", "script_decision": "accept_candidate" if accepted_minus > 2 else "blocked_hard_case", "script_blockers": [] if accepted_minus > 2 else ["severe_vocal_bleed_in_minus"]},
        ],
    )
    write_jsonl(run_dir / "selected_candidates.jsonl", [{"candidate_id": "pair-a", "stem_mode": "both"}])
    write_jsonl(
        run_dir / "pair_compatibility.jsonl",
        [{"pair_decision": "pair_compatible", "release_ready": False} for _ in range(pair_compatible)],
    )
    (run_dir / "preview_manifest.json").write_text(
        json.dumps(
            {
                "preview_status": preview_status,
                "validation_status": preview_validation_status,
                "rendered_preview_audio": preview_paths,
                "can_publish_as_result": False,
                "can_mark_release_ready": False,
                "release_ready": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_jsonl(
        run_dir / "rendered_preview_audio.jsonl",
        [
            {
                "preview_audio_path": path,
                "render_status": "rendered",
                "validation_status": preview_validation_status,
                "release_ready": False,
            }
            for path in preview_paths
        ],
    )
    (run_dir / "preview_render_contract.json").write_text(
        json.dumps(
            {
                "preview_status": preview_status,
                "validation_status": preview_validation_status,
                "rendered_preview_audio_count": len(preview_paths),
                "release_ready": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_jsonl(run_dir / "post_render_judge.jsonl", [{"post_render_decision": "preview_only", "release_ready": False}])
    write_jsonl(run_dir / "vocal_windowed_judge.jsonl", [])
    write_jsonl(
        run_dir / "minus_windowed_judge.jsonl",
        [{"minus_blockers": ["severe_phase_collapse"]} for _ in range(phase_count)]
        + [{"minus_blockers": ["severe_vocal_bleed_in_minus"]} for _ in range(bleed_count)],
    )
    (run_dir / "music_intelligence_input_manifest.json").write_text(
        json.dumps(
            {
                "analysis_ready_stems_count": 2,
                "blocked_stems_count": 0,
                "release_ready": False,
                "melody_transcription_ran": False,
                "recommendation_index_updated": False,
                "stems": [
                    {"stem_id": "vocal-a", "analysis_allowed": True, "analysis_block_reason": "", "release_ready": False},
                    {"stem_id": "minus-a", "analysis_allowed": True, "analysis_block_reason": "", "release_ready": False},
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_jsonl(
        run_dir / "learning_outcome.jsonl",
        [
            {
                "release_ready": False,
                "melody_transcription_ran": False,
                "recommendation_index_updated": False,
                "analysis_ready_stems_count": 2,
                "blocked_stems_count": 0,
            }
        ],
    )


def test_validation_runs_baseline_and_pareto_candidate_with_runtime_comparison(tmp_path: Path) -> None:
    baseline = default_harness_candidate()
    candidate = mutate_candidate(baseline, 1)
    candidate["candidate_profile"] = "balanced"
    candidate["candidate_diversity_profile"] = "balanced"
    candidate["max_candidates"] = 8
    frontier = tmp_path / "harness_pareto_frontier.jsonl"
    eval_jsonl = tmp_path / "harness_candidate_eval.jsonl"
    write_jsonl(frontier, [frontier_row(candidate)])
    write_jsonl(eval_jsonl, [frontier_row(baseline)])
    tracks = [tmp_path / "track_a.wav", tmp_path / "track_b.wav"]
    for track in tracks:
        track.write_bytes(b"audio")
    calls: list[dict[str, Any]] = []

    def fake_runner(command: list[str], run_dir: Path, context: dict[str, Any]) -> int:
        calls.append({"command": command, "context": context})
        quality = "candidate" if context["harness_candidate_id"] == candidate["harness_candidate_id"] else "baseline"
        make_runtime_artifacts(run_dir, stem_quality=quality)
        return 0

    result = run_harness_runtime_validation(
        pareto_frontier_jsonl=frontier,
        tracks=tracks,
        model_dir=tmp_path / "models",
        output_dir=tmp_path / "out",
        max_pareto_candidates=1,
        max_runtime_sec_per_track=1800,
        measurement_only=True,
        runtime_runner=fake_runner,
    )

    assert len(calls) == 4
    assert result["candidate_runtime_improved_count"] == 1
    assert result["candidate_specialist_only_count"] == 0
    assert result["release_ready_count"] == 0
    assert result["melody_transcription_ran"] is False
    assert result["recommendation_index_updated"] is False
    candidate_rows = read_jsonl(tmp_path / "out" / "harness_runtime_candidate_results.jsonl")
    track_rows = read_jsonl(tmp_path / "out" / "harness_runtime_track_results.jsonl")
    baseline_row = next(row for row in candidate_rows if row["harness_candidate_id"] == "baseline_safe_harness")
    assert {row["decision"] for row in candidate_rows} == {"keep_baseline", "candidate_runtime_improved"}
    assert baseline_row["severe_phase_collapse_count"] == 8
    assert baseline_row["severe_vocal_bleed_in_minus_count"] == 10
    assert all(row["release_ready"] is False for row in candidate_rows)
    assert all(row["release_ready"] is False for row in track_rows)
    assert (tmp_path / "out" / "runtime_validation_manifest.json").exists()
    assert (tmp_path / "out" / "runtime_validation_summary.md").exists()


def test_forbidden_harness_candidate_is_rejected_before_runtime(tmp_path: Path) -> None:
    bad = default_harness_candidate()
    bad["harness_candidate_id"] = "bad-threshold"
    bad["parent_id"] = "baseline_safe_harness"
    bad["mutation_type"] = "balanced_candidate_budget"
    bad["mutations"] = {"threshold_relaxation": "loosen"}
    frontier = tmp_path / "harness_pareto_frontier.jsonl"
    write_jsonl(frontier, [frontier_row(bad)])
    track = tmp_path / "track.wav"
    track.write_bytes(b"audio")

    def should_not_run(command: list[str], run_dir: Path, context: dict[str, Any]) -> int:
        raise AssertionError("runtime must not run for rejected safety candidate")

    result = run_harness_runtime_validation(
        pareto_frontier_jsonl=frontier,
        tracks=[track],
        model_dir=tmp_path / "models",
        output_dir=tmp_path / "out",
        max_pareto_candidates=1,
        max_runtime_sec_per_track=1800,
        measurement_only=True,
        runtime_runner=should_not_run,
    )

    rows = read_jsonl(tmp_path / "out" / "harness_runtime_candidate_results.jsonl")
    assert result["candidate_rejected_safety_count"] == 1
    assert rows[0]["decision"] == "candidate_rejected_safety"
    assert "forbidden_mutation_field:threshold_relaxation" in rows[0]["validation_rejection_reasons"]
    assert rows[0]["release_ready"] is False


def test_candidate_config_maps_to_autonomous_runtime_cli_args(tmp_path: Path) -> None:
    candidate = default_harness_candidate()
    candidate.update(
        {
            "harness_candidate_id": "balanced-route",
            "candidate_profile": "balanced",
            "candidate_diversity_profile": "balanced",
            "max_candidates": 9,
            "max_hard_case_extra_jobs": 5,
            "timeout_budget_policy": {"mode": "bounded", "max_runtime_sec": 900},
        }
    )

    command = map_candidate_to_runtime_args(
        candidate=candidate,
        input_audio=tmp_path / "song.wav",
        output_dir=tmp_path / "run",
        model_dir=tmp_path / "models",
        max_runtime_sec_per_track=1800,
    )

    assert "--profile" in command and command[command.index("--profile") + 1] == "balanced"
    assert "--candidate-profile" in command and command[command.index("--candidate-profile") + 1] == "balanced"
    assert "--candidate-diversity-profile" in command and command[command.index("--candidate-diversity-profile") + 1] == "balanced"
    assert "--max-candidates" in command and command[command.index("--max-candidates") + 1] == "9"
    assert "--max-candidate-runtime-sec" in command and command[command.index("--max-candidate-runtime-sec") + 1] == "900"
    assert "--enable-minus-hard-case-routing" in command and command[command.index("--enable-minus-hard-case-routing") + 1] == "true"
    assert "--max-hard-case-extra-jobs" in command and command[command.index("--max-hard-case-extra-jobs") + 1] == "5"
    assert "--enable-audio-preview-render" in command and command[command.index("--enable-audio-preview-render") + 1] == "true"
    assert "--require-preview-audio-for-preview-ready" in command and command[command.index("--require-preview-audio-for-preview-ready") + 1] == "false"


def test_manifest_preview_ready_is_not_counted_as_audio_preview_ready_without_audio(tmp_path: Path) -> None:
    baseline = default_harness_candidate()
    frontier = tmp_path / "harness_pareto_frontier.jsonl"
    eval_jsonl = tmp_path / "harness_candidate_eval.jsonl"
    write_jsonl(frontier, [])
    write_jsonl(eval_jsonl, [frontier_row(baseline)])
    track = tmp_path / "track.wav"
    track.write_bytes(b"audio")

    def fake_runner(command: list[str], run_dir: Path, context: dict[str, Any]) -> int:
        make_runtime_artifacts(run_dir, stem_quality="baseline", preview_audio=False)
        return 0

    result = run_harness_runtime_validation(
        pareto_frontier_jsonl=frontier,
        tracks=[track],
        model_dir=tmp_path / "models",
        output_dir=tmp_path / "out",
        max_pareto_candidates=0,
        max_runtime_sec_per_track=1800,
        measurement_only=True,
        runtime_runner=fake_runner,
    )

    rows = read_jsonl(tmp_path / "out" / "harness_runtime_track_results.jsonl")
    assert rows[0]["manifest_preview_ready"] is True
    assert rows[0]["audio_preview_ready"] is False
    assert result["manifest_preview_ready_count"] == 1
    assert result["audio_preview_ready_count"] == 0
    assert result["release_ready_count"] == 0


def test_audio_preview_ready_requires_rendered_audio_and_validation(tmp_path: Path) -> None:
    baseline = default_harness_candidate()
    frontier = tmp_path / "harness_pareto_frontier.jsonl"
    eval_jsonl = tmp_path / "harness_candidate_eval.jsonl"
    write_jsonl(frontier, [])
    write_jsonl(eval_jsonl, [frontier_row(baseline)])
    track = tmp_path / "track.wav"
    track.write_bytes(b"audio")

    def fake_runner(command: list[str], run_dir: Path, context: dict[str, Any]) -> int:
        make_runtime_artifacts(
            run_dir,
            stem_quality="baseline",
            preview_audio=False,
            output_status="audio_preview_ready",
            preview_status="audio_preview_ready",
            preview_validation_status="validated",
        )
        return 0

    result = run_harness_runtime_validation(
        pareto_frontier_jsonl=frontier,
        tracks=[track],
        model_dir=tmp_path / "models",
        output_dir=tmp_path / "out",
        max_pareto_candidates=0,
        max_runtime_sec_per_track=1800,
        measurement_only=True,
        runtime_runner=fake_runner,
        require_audio_preview_ready=True,
    )

    rows = read_jsonl(tmp_path / "out" / "harness_runtime_track_results.jsonl")
    assert rows[0]["final_runtime_status"] == "audio_preview_ready"
    assert rows[0]["rendered_preview_audio_count"] == 0
    assert rows[0]["audio_preview_ready"] is False
    assert rows[0]["preview_validation_failed_count"] == 1
    assert result["audio_preview_ready_count"] == 0
    assert result["rendered_preview_audio_count"] == 0
    assert result["preview_validation_failed_count"] == 1


def test_preview_render_failed_penalizes_candidate(tmp_path: Path) -> None:
    baseline = default_harness_candidate()
    candidate = mutate_candidate(baseline, 1)
    candidate["candidate_profile"] = "balanced"
    frontier = tmp_path / "harness_pareto_frontier.jsonl"
    eval_jsonl = tmp_path / "harness_candidate_eval.jsonl"
    write_jsonl(frontier, [frontier_row(candidate)])
    write_jsonl(eval_jsonl, [frontier_row(baseline)])
    tracks = [tmp_path / "track_a.wav", tmp_path / "track_b.wav"]
    for track in tracks:
        track.write_bytes(b"audio")

    def fake_runner(command: list[str], run_dir: Path, context: dict[str, Any]) -> int:
        if context["harness_candidate_id"] == candidate["harness_candidate_id"]:
            make_runtime_artifacts(
                run_dir,
                stem_quality="candidate",
                preview_audio=False,
                output_status="preview_render_failed",
                preview_status="preview_render_failed",
                preview_validation_status="failed",
            )
        else:
            make_runtime_artifacts(run_dir, stem_quality="baseline", output_status="audio_preview_ready", preview_status="audio_preview_ready")
        return 0

    result = run_harness_runtime_validation(
        pareto_frontier_jsonl=frontier,
        tracks=tracks,
        model_dir=tmp_path / "models",
        output_dir=tmp_path / "out",
        max_pareto_candidates=1,
        max_runtime_sec_per_track=1800,
        measurement_only=True,
        runtime_runner=fake_runner,
        require_audio_preview_ready=True,
    )

    rows = read_jsonl(tmp_path / "out" / "harness_runtime_candidate_results.jsonl")
    candidate_row = next(row for row in rows if row["harness_candidate_id"] == candidate["harness_candidate_id"])
    assert candidate_row["preview_render_failed_count"] == 2
    assert candidate_row["decision"] != "candidate_runtime_improved"
    assert result["preview_render_failed_count"] == 2
    assert result["best_global_candidate"] == ""
