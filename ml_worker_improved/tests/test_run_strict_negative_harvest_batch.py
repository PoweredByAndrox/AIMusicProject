import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_negative_harvest_batch_plans_runtime_jobs_without_training(tmp_path: Path) -> None:
    from tools.run_strict_negative_harvest_batch import run_strict_negative_harvest_batch

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "Song A.wav").write_bytes(b"audio-a")
    (input_dir / "Song B.flac").write_bytes(b"audio-b")
    (input_dir / "notes.txt").write_text("ignore", encoding="utf-8")

    manifest = run_strict_negative_harvest_batch(
        input_audio_dir=input_dir,
        output_dir=tmp_path / "harvest",
        model_dir=tmp_path / "models",
        max_tracks=1,
        profile="balanced",
        candidate_diversity_profile="balanced",
        enable_conditional_hard_case_routes=True,
        enable_audio_preview_render=True,
        require_preview_audio_for_preview_ready=True,
        continue_on_failure=True,
        allow_non_testing_stuff_input=True,
        execute=False,
    )

    rows = read_jsonl(tmp_path / "harvest" / "negative_harvest_runs.jsonl")
    assert manifest["tracks_planned"] == 1
    assert manifest["training_started"] is False
    assert manifest["separator_training_started"] is False
    assert manifest["release_ready"] is False
    assert rows[0]["execution_status"] == "planned_only"
    assert rows[0]["candidate_artifact_requirements"]["candidate_id"] is True
    assert rows[0]["candidate_artifact_requirements"]["route_id"] is True
    assert "--enable-conditional-hard-case-routes" in rows[0]["command"]
    assert "--require-preview-audio-for-preview-ready" in rows[0]["command"]


def test_negative_harvest_batch_writes_no_release_ready_when_input_empty(tmp_path: Path) -> None:
    from tools.run_strict_negative_harvest_batch import run_strict_negative_harvest_batch

    input_dir = tmp_path / "input"
    input_dir.mkdir()

    manifest = run_strict_negative_harvest_batch(
        input_audio_dir=input_dir,
        output_dir=tmp_path / "harvest",
        model_dir=tmp_path / "models",
        max_tracks=20,
        profile="balanced",
        candidate_diversity_profile="balanced",
        enable_conditional_hard_case_routes=True,
        enable_audio_preview_render=True,
        require_preview_audio_for_preview_ready=True,
        continue_on_failure=True,
        allow_non_testing_stuff_input=True,
        execute=False,
    )

    assert manifest["tracks_planned"] == 0
    assert manifest["release_ready"] is False
    assert (tmp_path / "harvest" / "negative_harvest_manifest.json").exists()
    assert (tmp_path / "harvest" / "negative_harvest_summary.md").exists()


def test_harvest_counts_strict_positive_and_negative_windows(tmp_path: Path) -> None:
    from tools.run_strict_negative_harvest_batch import harvest_example_counts, write_json, write_jsonl

    run_dir = tmp_path / "harvest" / "run_001_track"
    write_json(run_dir / "run_manifest.json", {"output_status": "needs_more_candidates", "release_ready": False})
    write_jsonl(run_dir / "candidate_pool.jsonl", [{"candidate_id": "cand-a", "route_id": "route-a"}])
    write_jsonl(run_dir / "pair_compatibility.jsonl", [{"candidate_id": "cand-a", "pair_decision": "pair_incompatible"}])
    write_jsonl(
        run_dir / "minus_windowed_judge.jsonl",
        [
            {
                "candidate_id": "cand-a",
                "window_metrics": [
                    {"start_sec": 0.0, "duration_sec": 1.0, "minus_decision": "accept_candidate", "minus_blockers": []},
                    {
                        "start_sec": 1.0,
                        "duration_sec": 1.0,
                        "minus_decision": "blocked_hard_case",
                        "minus_blockers": ["severe_vocal_bleed_in_minus"],
                    },
                ],
            }
        ],
    )

    counts = harvest_example_counts(tmp_path / "harvest")

    assert counts["candidate_pool_rows"] == 1
    assert counts["pair_rows"] == 1
    assert counts["minus_judge_rows"] == 1
    assert counts["strict_positive_windows"] == 1
    assert counts["strict_negative_windows"] == 1
    assert counts["blocker_counts"]["severe_vocal_bleed_in_minus"] == 1
    assert counts["final_status_counts"]["needs_more_candidates"] == 1


def test_negative_harvest_batch_rejects_flstudio_corpus(tmp_path: Path) -> None:
    from tools.run_strict_negative_harvest_batch import run_strict_negative_harvest_batch

    input_dir = tmp_path / "testing_flstudio_music"
    input_dir.mkdir()

    try:
        run_strict_negative_harvest_batch(
            input_audio_dir=input_dir,
            output_dir=tmp_path / "harvest",
            model_dir=tmp_path / "models",
            max_tracks=1,
            profile="balanced",
            candidate_diversity_profile="balanced",
            enable_conditional_hard_case_routes=True,
            enable_audio_preview_render=True,
            require_preview_audio_for_preview_ready=True,
            continue_on_failure=True,
            allow_non_testing_stuff_input=True,
            execute=False,
        )
    except ValueError as exc:
        assert str(exc) == "forbidden_input_audio_dir:testing_flstudio_music"
    else:
        raise AssertionError("expected forbidden_input_audio_dir")


def test_negative_harvest_batch_requires_testing_stuff_by_default(tmp_path: Path) -> None:
    from tools.run_strict_negative_harvest_batch import run_strict_negative_harvest_batch

    input_dir = tmp_path / "input"
    input_dir.mkdir()

    try:
        run_strict_negative_harvest_batch(
            input_audio_dir=input_dir,
            output_dir=tmp_path / "harvest",
            model_dir=tmp_path / "models",
            max_tracks=1,
            profile="balanced",
            candidate_diversity_profile="balanced",
            enable_conditional_hard_case_routes=True,
            enable_audio_preview_render=True,
            require_preview_audio_for_preview_ready=True,
            continue_on_failure=True,
            execute=False,
        )
    except ValueError as exc:
        assert str(exc) == "input_audio_dir_must_be_testing_stuff_music"
    else:
        raise AssertionError("expected input_audio_dir_must_be_testing_stuff_music")


def test_negative_harvest_batch_rejects_stale_run_dirs(tmp_path: Path) -> None:
    from tools.run_strict_negative_harvest_batch import run_strict_negative_harvest_batch

    input_dir = tmp_path / "testing_stuff_music"
    input_dir.mkdir()
    output_dir = tmp_path / "harvest"
    (output_dir / "run_001_old").mkdir(parents=True)

    try:
        run_strict_negative_harvest_batch(
            input_audio_dir=input_dir,
            output_dir=output_dir,
            model_dir=tmp_path / "models",
            max_tracks=1,
            profile="balanced",
            candidate_diversity_profile="balanced",
            enable_conditional_hard_case_routes=True,
            enable_audio_preview_render=True,
            require_preview_audio_for_preview_ready=True,
            continue_on_failure=True,
            execute=False,
        )
    except ValueError as exc:
        assert str(exc) == "stale_run_dirs_present:1"
    else:
        raise AssertionError("expected stale_run_dirs_present")
