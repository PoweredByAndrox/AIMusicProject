from __future__ import annotations

import json
from pathlib import Path

from tools.build_no_reference_replay_bundle import build_replay_bundle


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def make_complete_run(tmp_path: Path, name: str = "run-a") -> Path:
    run_dir = tmp_path / name
    audio_path = tmp_path / "audio" / "minus.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"fake-audio")
    write_json(
        run_dir / "run_manifest.json",
        {
            "input_audio": str(tmp_path / "audio" / "mix.wav"),
            "output_status": "needs_more_candidates",
            "failure_reason": "missing_safe_mode_candidate",
            "release_ready": False,
            "reference_required": False,
            "human_csv_required": False,
            "training_started": False,
            "separator_training_started": False,
            "model_downloaded": False,
            "model_registry_edited": False,
            "thresholds_relaxed": False,
            "cleanup_routes_added": False,
        },
    )
    (run_dir / "runtime_summary.md").write_text("# summary\n", encoding="utf-8")
    write_jsonl(
        run_dir / "candidate_pool.jsonl",
        [
            {
                "track_id": "track-a",
                "candidate_id": "pair-a",
                "is_complete_pair": True,
                "minus_path": str(audio_path),
                "raw_minus_path": str(audio_path),
                "normalized_minus_path": str(audio_path),
                "release_ready": False,
            }
        ],
    )
    write_json(run_dir / "candidates" / "candidate_wave_summary.json", {"jobs_attempted": 1, "jobs_generated": 1, "complete_pairs": 1, "duplicate_jobs_avoided": 0})
    write_jsonl(run_dir / "candidates" / "candidate_wave_jobs.jsonl", [{"candidate_config_id": "job-a", "generation_status": "generated"}])
    write_jsonl(run_dir / "candidates" / "candidate_wave_failures.jsonl", [])
    write_jsonl(run_dir / "vocal_windowed_judge.jsonl", [{"candidate_id": "pair-a", "final_guarded_decision": "accept_candidate"}])
    write_jsonl(run_dir / "minus_windowed_judge.jsonl", [{"candidate_id": "pair-a", "minus_blockers": ["severe_vocal_bleed_in_minus"], "final_guarded_decision": "blocked_hard_case"}])
    write_jsonl(run_dir / "learned_critic_scores.jsonl", [{"candidate_id": "pair-a", "score_mode": "minus", "final_guarded_decision": "blocked_hard_case"}])
    write_jsonl(run_dir / "selected_candidates.jsonl", [])
    write_jsonl(run_dir / "pair_compatibility.jsonl", [])
    write_jsonl(run_dir / "minus_hard_case_router.jsonl", [{"recommended_route": "generate_phase_safe_variants", "release_ready": False}])
    write_json(run_dir / "preview_manifest.json", {"runtime_status": "needs_more_candidates", "release_ready": False, "can_publish_as_result": False})
    write_jsonl(run_dir / "post_render_judge.jsonl", [])
    write_jsonl(run_dir / "learning_outcome.jsonl", [{"runtime_status": "needs_more_candidates"}])
    return run_dir


def test_replay_bundle_does_not_copy_audio_and_indexes_metadata_only(tmp_path: Path) -> None:
    run_dir = make_complete_run(tmp_path)
    output_dir = tmp_path / "bundle"

    result = build_replay_bundle(run_roots=[run_dir], output_dir=output_dir)

    manifest = json.loads((output_dir / "replay_bundle_manifest.json").read_text(encoding="utf-8"))
    artifacts = [json.loads(line) for line in (output_dir / "replay_bundle_artifacts_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert result["replay_complete_runs"] == 1
    assert manifest["audio_files_copied"] is False
    assert not list(output_dir.rglob("*.wav"))
    assert any(row["artifact_role"] == "audio_reference" and row["copy_status"] == "metadata_only" for row in artifacts)
    assert all(row.get("release_ready") is not True for row in artifacts)


def test_replay_bundle_detects_missing_artifacts_without_faking(tmp_path: Path) -> None:
    run_dir = tmp_path / "incomplete-run"
    write_json(run_dir / "run_manifest.json", {"output_status": "needs_more_candidates", "release_ready": False})

    build_replay_bundle(run_roots=[run_dir], output_dir=tmp_path / "bundle")

    manifest = json.loads((tmp_path / "bundle" / "replay_bundle_manifest.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (tmp_path / "bundle" / "replay_bundle_events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert manifest["replay_complete_runs"] == 0
    assert manifest["replay_incomplete_runs"] == 1
    assert any(row["event_type"] == "replay_incomplete" and "candidate_pool.jsonl" in row["missing_artifacts"] for row in events)


def test_replay_bundle_counts_source_release_ready_without_republishing(tmp_path: Path) -> None:
    run_dir = make_complete_run(tmp_path)
    write_jsonl(
        run_dir / "candidate_pool.jsonl",
        [
            {
                "track_id": "track-a",
                "candidate_id": "unsafe-source",
                "is_complete_pair": True,
                "minus_path": str(tmp_path / "audio" / "minus.wav"),
                "release_ready": True,
            }
        ],
    )

    build_replay_bundle(run_roots=[run_dir], output_dir=tmp_path / "bundle")

    manifest = json.loads((tmp_path / "bundle" / "replay_bundle_manifest.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (tmp_path / "bundle" / "replay_bundle_events.jsonl").read_text(encoding="utf-8").splitlines()]
    candidate_pool_events = [row for row in events if row["event_type"] == "candidate_pool"]
    assert manifest["release_ready"] is False
    assert manifest["release_ready_count"] == 1
    assert candidate_pool_events[0]["payload"]["source_release_ready"] is True
    assert candidate_pool_events[0]["payload"]["release_ready"] is False
