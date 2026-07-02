from __future__ import annotations

import json
from pathlib import Path

import tools.build_testing_stuff_music_corpus_job_matrix as job_matrix
from worker.model_registry import ModelRegistry, ModelRegistryEntry


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def demo_registry(tmp_path: Path) -> ModelRegistry:
    entries = []
    for model_id, stem in (
        ("demo_vocal_model", "vocals"),
        ("demo_minus_registry_model", "instrumental"),
        ("demo_minus_registry_model_2", "instrumental"),
        ("demo_minus_registry_model_3", "instrumental"),
        ("demo_minus_registry_model_4", "instrumental"),
        ("demo_minus_registry_model_5", "instrumental"),
        ("demo_minus_registry_model_6", "instrumental"),
        ("demo_melody_model", "melody"),
        ("demo_drums_model", "drums"),
        ("demo_bass_model", "bass"),
        ("demo_other_model", "other"),
    ):
        entries.append(
            ModelRegistryEntry(
                model_id=model_id,
                display_name=model_id,
                family="test",
                target_stems=(stem,),
                output_stems=(stem,),
                command_template=("python", "unused_direct_command.py"),
                expected_manifest="{output_dir}/stem_manifest.json",
            )
        )
    return ModelRegistry(entries=tuple(entries), runtime_root=tmp_path)


def corpus_row(source_audio: Path) -> dict:
    return {
        "schema": "testing_stuff_music_corpus_plan_row.v1",
        "index": 1,
        "track_id": "song_one",
        "relative_path": "Song One.mp3",
        "source_audio_path": str(source_audio),
        "docker_audio_path": str(source_audio),
        "host_audio_path_hint": "D:/AIMusicProject/testing_stuff_music/Song One.mp3",
        "source_exists": True,
        "target_stems": ["vocal", "minus", "melody", "drums", "bass", "other"],
        "training_allowed": False,
        "fine_tune_allowed": False,
        "can_start_training_now": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
    }


def rows_by_stem(result: dict) -> dict[str, dict]:
    return {row["target_stem"]: row for row in result["rows"]}


def route_prior_row(*, route_id: str, stem_type: str = "minus", recommendation: str = "blocked_as_minus_baseline") -> dict:
    return {
        "schema_version": 1,
        "mode": "no_reference_corpus_route_prior",
        "stem_type": stem_type,
        "route_id": route_id,
        "candidate_count": 6,
        "blocked_count": 6,
        "accepted_count": 0,
        "recommendation": recommendation,
        "requires_new_source_candidate": recommendation == "blocked_as_minus_baseline",
        "top_blockers": {"severe_vocal_bleed_in_minus": 6},
        "top_failure_reasons": {"vocal_leakage_in_minus": 6},
    }


def test_builds_all_stem_candidate_generation_jobs_without_execution_or_release(tmp_path: Path) -> None:
    source_audio = tmp_path / "Song One.mp3"
    source_audio.write_bytes(b"audio-placeholder")
    plan_jsonl = write_jsonl(tmp_path / "testing_stuff_music_corpus_plan.jsonl", [corpus_row(source_audio)])

    result = job_matrix.build_job_matrix(
        corpus_plan_jsonl=plan_jsonl,
        output_root=tmp_path / "jobs",
        registry=demo_registry(tmp_path),
        max_models_per_stem=1,
    )

    assert result["summary"]["total_tracks"] == 1
    assert result["summary"]["total_jobs"] == 6
    assert result["summary"]["target_stems"] == ["vocal", "minus", "melody", "drums", "bass", "other"]
    assert result["summary"]["execution_started"] is False
    assert result["manifest"]["release_ready"] is False
    assert result["manifest"]["training_started"] is False
    assert result["manifest"]["fine_tune_started"] is False

    rows = rows_by_stem(result)
    assert set(rows) == {"vocal", "minus", "melody", "drums", "bass", "other"}
    assert "run_registered_model_candidate.py" in " ".join(rows["vocal"]["command"])
    assert rows["vocal"]["model_id"] == "demo_vocal_model"
    assert "export_instrumental_resurrection_minus.py" in " ".join(rows["minus"]["command"])
    assert "build_vocal_guided_minus.py" not in json.dumps(rows["minus"])
    assert rows["minus"]["route_contract"] == "documented_clean_minus_baseline"
    assert all(row["planned_only"] is True for row in result["rows"])
    assert all(row["execution_started"] is False for row in result["rows"])
    assert all(row["can_mark_release_ready"] is False for row in result["rows"])
    assert all(row["expected_acceleration_mode"] == "gpu_inference_then_cpu_dsp_scoring" for row in result["rows"])
    assert all("mode_specific_score_matrix" in row["required_downstream_gates"] for row in result["rows"])


def test_route_priors_block_repeated_bad_minus_baseline_before_execution(tmp_path: Path) -> None:
    source_audio = tmp_path / "Song One.mp3"
    source_audio.write_bytes(b"audio-placeholder")
    plan_jsonl = write_jsonl(tmp_path / "testing_stuff_music_corpus_plan.jsonl", [corpus_row(source_audio)])
    route_priors_jsonl = write_jsonl(
        tmp_path / "route_priors.jsonl",
        [route_prior_row(route_id="export_instrumental_resurrection_minus")],
    )

    result = job_matrix.build_job_matrix(
        corpus_plan_jsonl=plan_jsonl,
        output_root=tmp_path / "jobs",
        registry=demo_registry(tmp_path),
        target_stems=["minus"],
        max_models_per_stem=1,
        route_priors_jsonl=route_priors_jsonl,
    )

    assert result["summary"]["total_jobs"] == 2
    assert result["summary"]["queued"] == 1
    assert result["summary"]["blocked_route_prior"] == 1
    assert result["summary"]["status_counts"]["blocked_route_prior"] == 1
    direct_row = next(row for row in result["rows"] if row["model_id"] == "instrumental_resurrection_minus")
    assert direct_row["target_stem"] == "minus"
    assert direct_row["job_status"] == "blocked_route_prior"
    assert direct_row["execution_ready"] is False
    assert direct_row["route_prior"]["recommendation"] == "blocked_as_minus_baseline"
    assert direct_row["route_prior"]["requires_new_source_candidate"] is True
    assert "severe_vocal_bleed_in_minus" in direct_row["route_prior"]["top_blockers"]

    alternate_row = next(row for row in result["rows"] if row["model_id"] == "demo_minus_registry_model")
    assert alternate_row["target_stem"] == "minus"
    assert alternate_row["job_status"] == "queued"
    assert alternate_row["execution_ready"] is True
    assert "run_registered_model_candidate.py" in " ".join(alternate_row["command"])
    assert "--target-stem" in alternate_row["command"]
    assert alternate_row["command"][alternate_row["command"].index("--target-stem") + 1] == "minus"
    assert alternate_row["route_contract"] == "registered_model_candidate_manifest_required"
    assert result["manifest"]["delivery_allowed"] is False
    assert result["manifest"]["release_ready"] is False


def test_blocked_minus_prior_carries_source_pool_evidence_into_matrix_row(tmp_path: Path) -> None:
    source_audio = tmp_path / "Song One.mp3"
    source_audio.write_bytes(b"audio-placeholder")
    vocal_audio = tmp_path / "vocal.flac"
    minus_audio = tmp_path / "minus.flac"
    vocal_audio.write_bytes(b"vocal-placeholder")
    minus_audio.write_bytes(b"minus-placeholder")
    plan_jsonl = write_jsonl(tmp_path / "testing_stuff_music_corpus_plan.jsonl", [corpus_row(source_audio)])
    prior = route_prior_row(route_id="export_instrumental_resurrection_minus")
    prior.update(
        {
            "source_pool_eligible": True,
            "source_pool_status": "minus_source_candidate_supported_for_repair",
            "best_source_track_id": "song_one",
            "best_source_candidate_id": "source-minus-a",
            "best_source_mix_path": str(source_audio),
            "best_source_vocal_path": str(vocal_audio),
            "best_source_audio_path": str(minus_audio),
            "best_source_score": 0.832581,
            "best_source_vocal_leakage_in_minus_score": 0.284465,
            "source_pool_reasons": ["low_vocal_leakage_minus_source_available"],
        }
    )
    route_priors_jsonl = write_jsonl(tmp_path / "route_priors.jsonl", [prior])

    result = job_matrix.build_job_matrix(
        corpus_plan_jsonl=plan_jsonl,
        output_root=tmp_path / "jobs",
        registry=demo_registry(tmp_path),
        target_stems=["minus"],
        max_models_per_stem=1,
        route_priors_jsonl=route_priors_jsonl,
    )

    direct_row = next(row for row in result["rows"] if row["model_id"] == "instrumental_resurrection_minus")
    assert direct_row["job_status"] == "blocked_route_prior"
    assert direct_row["execution_ready"] is False
    assert direct_row["route_prior"]["source_pool_eligible"] is True
    assert direct_row["route_prior"]["source_pool_status"] == "minus_source_candidate_supported_for_repair"
    assert direct_row["route_prior"]["best_source_track_id"] == "song_one"
    assert direct_row["route_prior"]["best_source_candidate_id"] == "source-minus-a"
    assert direct_row["route_prior"]["best_source_score"] == 0.832581
    assert direct_row["route_prior"]["best_source_vocal_leakage_in_minus_score"] == 0.284465
    assert direct_row["delivery_allowed"] is False
    assert direct_row["release_ready"] is False


def test_matrix_emits_executable_source_pool_phase_repair_rows_from_priors(tmp_path: Path) -> None:
    source_audio = tmp_path / "Song One.mp3"
    source_audio.write_bytes(b"audio-placeholder")
    vocal_audio = tmp_path / "vocal.flac"
    minus_audio = tmp_path / "minus.flac"
    vocal_audio.write_bytes(b"vocal-placeholder")
    minus_audio.write_bytes(b"minus-placeholder")
    plan_jsonl = write_jsonl(tmp_path / "testing_stuff_music_corpus_plan.jsonl", [corpus_row(source_audio)])
    prior = route_prior_row(route_id="becruily_instrumental_vocals")
    prior.update(
        {
            "source_pool_eligible": True,
            "source_pool_status": "minus_source_candidate_supported_for_repair",
            "best_source_track_id": "song_one",
            "best_source_candidate_id": "becruily_source_pair",
            "best_source_mix_path": str(source_audio),
            "best_source_vocal_path": str(vocal_audio),
            "best_source_audio_path": str(minus_audio),
            "best_source_score": 0.832581,
            "best_source_vocal_leakage_in_minus_score": 0.284465,
            "best_source_mixture_consistency_score": 0.929946,
            "source_pool_reasons": ["low_vocal_leakage_minus_source_available"],
        }
    )
    route_priors_jsonl = write_jsonl(tmp_path / "route_priors.jsonl", [prior])

    result = job_matrix.build_job_matrix(
        corpus_plan_jsonl=plan_jsonl,
        output_root=tmp_path / "jobs",
        registry=demo_registry(tmp_path),
        target_stems=["minus"],
        max_models_per_stem=1,
        route_priors_jsonl=route_priors_jsonl,
    )

    source_pool_rows = [row for row in result["rows"] if row["phase"] == "source_pool_phase_repair_candidate"]
    assert len(source_pool_rows) == 1
    row = source_pool_rows[0]
    assert row["target_stem"] == "minus"
    assert row["track_id"] == "song_one"
    assert row["model_id"] == "becruily_instrumental_vocals"
    assert row["job_status"] == "queued"
    assert row["execution_ready"] is True
    assert row["planned_only"] is True
    assert row["route_contract"] == "no_reference_minus_phase_repair_candidate"
    assert "/app/tools/run_no_reference_minus_phase_repair.py" in row["command"]
    assert "--minus-path" in row["command"]
    assert row["expected_acceleration_mode"] == "cpu_dsp_candidate_generation"
    assert row["gpu_inference_required"] is False
    assert row["source_pool"]["best_source_candidate_id"] == "becruily_source_pair"
    assert row["source_pool"]["best_source_score"] == 0.832581
    assert row["delivery_allowed"] is False
    assert row["release_ready"] is False
    assert result["summary"]["source_pool_repair_candidates"] == 1


def test_minus_route_prior_does_not_block_vocal_job(tmp_path: Path) -> None:
    source_audio = tmp_path / "Song One.mp3"
    source_audio.write_bytes(b"audio-placeholder")
    plan_jsonl = write_jsonl(tmp_path / "testing_stuff_music_corpus_plan.jsonl", [corpus_row(source_audio)])
    route_priors_jsonl = write_jsonl(
        tmp_path / "route_priors.jsonl",
        [route_prior_row(route_id="demo_vocal_model")],
    )

    result = job_matrix.build_job_matrix(
        corpus_plan_jsonl=plan_jsonl,
        output_root=tmp_path / "jobs",
        registry=demo_registry(tmp_path),
        target_stems=["vocal"],
        max_models_per_stem=1,
        route_priors_jsonl=route_priors_jsonl,
    )

    assert result["summary"]["total_jobs"] == 1
    assert result["rows"][0]["target_stem"] == "vocal"
    assert result["rows"][0]["model_id"] == "demo_vocal_model"
    assert result["rows"][0]["job_status"] == "queued"
    assert result["rows"][0]["execution_ready"] is True


def test_vocal_and_pair_route_priors_do_not_block_minus_job_selection(tmp_path: Path) -> None:
    source_audio = tmp_path / "Song One.mp3"
    source_audio.write_bytes(b"audio-placeholder")
    plan_jsonl = write_jsonl(tmp_path / "testing_stuff_music_corpus_plan.jsonl", [corpus_row(source_audio)])
    route_priors_jsonl = write_jsonl(
        tmp_path / "route_priors.jsonl",
        [
            route_prior_row(route_id="instrumental_resurrection_minus", stem_type="vocal"),
            route_prior_row(route_id="instrumental_resurrection_minus", stem_type="pair"),
        ],
    )

    result = job_matrix.build_job_matrix(
        corpus_plan_jsonl=plan_jsonl,
        output_root=tmp_path / "jobs",
        registry=demo_registry(tmp_path),
        target_stems=["minus"],
        max_models_per_stem=1,
        route_priors_jsonl=route_priors_jsonl,
    )

    assert result["summary"]["blocked_route_prior"] == 0
    assert result["summary"]["queued"] == 1
    assert result["summary"]["total_jobs"] == 1
    row = result["rows"][0]
    assert row["target_stem"] == "minus"
    assert row["model_id"] == "instrumental_resurrection_minus"
    assert row["job_status"] == "queued"
    assert row["execution_ready"] is True


def test_dsp_registry_job_declares_cpu_dsp_acceleration_contract(tmp_path: Path) -> None:
    source_audio = tmp_path / "Song One.mp3"
    source_audio.write_bytes(b"audio-placeholder")
    plan_jsonl = write_jsonl(tmp_path / "testing_stuff_music_corpus_plan.jsonl", [corpus_row(source_audio)])
    registry = ModelRegistry(
        entries=(
            ModelRegistryEntry(
                model_id="demo_tonal_dsp_melody",
                display_name="Demo tonal DSP melody",
                family="dsp",
                target_stems=("melody",),
                output_stems=("melody",),
                command_template=("python", "unused_direct_command.py"),
                expected_manifest="{output_dir}/stem_manifest.json",
            ),
        ),
        runtime_root=tmp_path,
    )

    result = job_matrix.build_job_matrix(
        corpus_plan_jsonl=plan_jsonl,
        output_root=tmp_path / "jobs",
        registry=registry,
        target_stems=["melody"],
        max_models_per_stem=1,
    )

    row = result["rows"][0]
    assert row["model_id"] == "demo_tonal_dsp_melody"
    assert row["route_family"] == "dsp"
    assert row["expected_acceleration_mode"] == "cpu_dsp_candidate_generation"
    assert row["gpu_inference_required"] is False
    assert row["gpu_inference_skipped"] is True
    assert row["gpu_inference_skip_reason"] == "cpu_only_dsp_route"


def test_blocked_minus_registry_prior_selects_next_unblocked_registry_candidate(tmp_path: Path) -> None:
    source_audio = tmp_path / "Song One.mp3"
    source_audio.write_bytes(b"audio-placeholder")
    plan_jsonl = write_jsonl(tmp_path / "testing_stuff_music_corpus_plan.jsonl", [corpus_row(source_audio)])
    route_priors_jsonl = write_jsonl(
        tmp_path / "route_priors.jsonl",
        [
            route_prior_row(route_id="export_instrumental_resurrection_minus"),
            route_prior_row(route_id="demo_minus_registry_model"),
        ],
    )

    result = job_matrix.build_job_matrix(
        corpus_plan_jsonl=plan_jsonl,
        output_root=tmp_path / "jobs",
        registry=demo_registry(tmp_path),
        target_stems=["minus"],
        max_models_per_stem=1,
        route_priors_jsonl=route_priors_jsonl,
    )

    assert result["summary"]["total_jobs"] == 2
    assert result["summary"]["queued"] == 1
    assert result["summary"]["blocked_route_prior"] == 1
    assert [row["model_id"] for row in result["rows"] if row["job_status"] == "queued"] == [
        "demo_minus_registry_model_2"
    ]
    assert "demo_minus_registry_model" not in {
        row["model_id"] for row in result["rows"] if row["target_stem"] == "minus"
    }


def test_minus_matrix_rejects_four_stem_models_that_would_materialize_minus(tmp_path: Path) -> None:
    source_audio = tmp_path / "Song One.mp3"
    source_audio.write_bytes(b"audio-placeholder")
    plan_jsonl = write_jsonl(tmp_path / "testing_stuff_music_corpus_plan.jsonl", [corpus_row(source_audio)])
    route_priors_jsonl = write_jsonl(
        tmp_path / "route_priors.jsonl",
        [route_prior_row(route_id="export_instrumental_resurrection_minus")],
    )
    registry = ModelRegistry(
        entries=(
            ModelRegistryEntry(
                model_id="demo_four_stem_model",
                display_name="Demo four-stem model",
                family="test",
                target_stems=("vocals", "drums", "bass", "other"),
                output_stems=("vocals", "drums", "bass", "other"),
                command_template=("python", "unused_direct_command.py"),
                expected_manifest="{output_dir}/stem_manifest.json",
            ),
        ),
        runtime_root=tmp_path,
    )

    result = job_matrix.build_job_matrix(
        corpus_plan_jsonl=plan_jsonl,
        output_root=tmp_path / "jobs",
        registry=registry,
        target_stems=["minus"],
        route_priors_jsonl=route_priors_jsonl,
    )

    assert result["summary"]["total_jobs"] == 1
    assert result["summary"]["queued"] == 0
    assert result["summary"]["blocked_route_prior"] == 1
    assert all(row["model_id"] != "demo_four_stem_model" for row in result["rows"])


def test_family_level_minus_prior_is_treated_as_evaluated_before_selection(tmp_path: Path) -> None:
    source_audio = tmp_path / "Song One.mp3"
    source_audio.write_bytes(b"audio-placeholder")
    plan_jsonl = write_jsonl(tmp_path / "testing_stuff_music_corpus_plan.jsonl", [corpus_row(source_audio)])
    route_priors_jsonl = write_jsonl(
        tmp_path / "route_priors.jsonl",
        [
            route_prior_row(route_id="export_instrumental_resurrection_minus"),
            route_prior_row(route_id="blocked_family"),
        ],
    )
    registry = ModelRegistry(
        entries=(
            ModelRegistryEntry(
                model_id="blocked_family_model",
                display_name="Blocked family model",
                family="blocked_family",
                target_stems=("instrumental",),
                output_stems=("instrumental",),
                command_template=("python", "unused_direct_command.py"),
                expected_manifest="{output_dir}/stem_manifest.json",
            ),
            ModelRegistryEntry(
                model_id="unblocked_family_model",
                display_name="Unblocked family model",
                family="unblocked_family",
                target_stems=("instrumental",),
                output_stems=("instrumental",),
                command_template=("python", "unused_direct_command.py"),
                expected_manifest="{output_dir}/stem_manifest.json",
            ),
        ),
        runtime_root=tmp_path,
    )

    result = job_matrix.build_job_matrix(
        corpus_plan_jsonl=plan_jsonl,
        output_root=tmp_path / "jobs",
        registry=registry,
        target_stems=["minus"],
        route_priors_jsonl=route_priors_jsonl,
    )

    minus_model_ids = {row["model_id"] for row in result["rows"] if row["target_stem"] == "minus"}
    assert "blocked_family_model" not in minus_model_ids
    assert "unblocked_family_model" in minus_model_ids
    assert all(
        row["job_status"] != "blocked_route_prior"
        for row in result["rows"]
        if row["model_id"] == "unblocked_family_model"
    )


def test_default_route_prior_blocked_minus_plans_broad_registry_pool(tmp_path: Path) -> None:
    source_audio = tmp_path / "Song One.mp3"
    source_audio.write_bytes(b"audio-placeholder")
    plan_jsonl = write_jsonl(tmp_path / "testing_stuff_music_corpus_plan.jsonl", [corpus_row(source_audio)])
    route_priors_jsonl = write_jsonl(
        tmp_path / "route_priors.jsonl",
        [route_prior_row(route_id="export_instrumental_resurrection_minus")],
    )

    result = job_matrix.build_job_matrix(
        corpus_plan_jsonl=plan_jsonl,
        output_root=tmp_path / "jobs",
        registry=demo_registry(tmp_path),
        target_stems=["minus"],
        route_priors_jsonl=route_priors_jsonl,
    )

    queued_minus_models = [
        row["model_id"]
        for row in result["rows"]
        if row["target_stem"] == "minus" and row["job_status"] == "queued"
    ]
    assert queued_minus_models == [
        "demo_minus_registry_model",
        "demo_minus_registry_model_2",
        "demo_minus_registry_model_3",
        "demo_minus_registry_model_4",
        "demo_minus_registry_model_5",
        "demo_minus_registry_model_6",
    ]
    assert result["summary"]["queued"] == 6
    assert result["summary"]["blocked_route_prior"] == 1
    assert result["summary"]["total_jobs"] == 7


def test_blocks_jobs_when_docker_audio_path_is_missing(tmp_path: Path) -> None:
    missing_audio = tmp_path / "missing.mp3"
    plan_jsonl = write_jsonl(tmp_path / "testing_stuff_music_corpus_plan.jsonl", [corpus_row(missing_audio)])

    result = job_matrix.build_job_matrix(
        corpus_plan_jsonl=plan_jsonl,
        output_root=tmp_path / "jobs",
        registry=demo_registry(tmp_path),
        max_models_per_stem=1,
    )

    assert result["summary"]["blocked_missing_source_audio"] == 6
    assert {row["job_status"] for row in result["rows"]} == {"blocked_missing_source_audio"}
    assert all(row["execution_ready"] is False for row in result["rows"])
    assert result["manifest"]["delivery_allowed"] is False


def test_write_outputs_creates_job_manifest_progress_and_no_audio(tmp_path: Path) -> None:
    source_audio = tmp_path / "Song One.mp3"
    source_audio.write_bytes(b"audio-placeholder")
    plan_jsonl = write_jsonl(tmp_path / "testing_stuff_music_corpus_plan.jsonl", [corpus_row(source_audio)])
    output_root = tmp_path / "jobs"
    result = job_matrix.build_job_matrix(
        corpus_plan_jsonl=plan_jsonl,
        output_root=output_root,
        registry=demo_registry(tmp_path),
        max_models_per_stem=1,
    )

    job_matrix.write_outputs(result, output_root)

    assert (output_root / "testing_stuff_music_corpus_job_matrix.jsonl").exists()
    assert (output_root / "testing_stuff_music_corpus_job_matrix_manifest.json").exists()
    assert (output_root / "testing_stuff_music_corpus_job_matrix_summary.md").exists()
    assert (output_root / "testing_stuff_music_corpus_job_matrix_progress.jsonl").exists()
    assert not list(output_root.rglob("*.mp3"))
    assert not list(output_root.rglob("*.wav"))
    assert not list(output_root.rglob("*.flac"))


def test_main_checks_docker_runtime_before_parsing_args(monkeypatch) -> None:
    calls = []

    def reject_non_docker_runtime(**kwargs):
        calls.append(("require_docker_runtime", kwargs))
        raise RuntimeError("Docker-only test sentinel")

    def fail_if_args_are_parsed(self, argv=None):
        raise AssertionError("parse_args must not run before Docker runtime validation")

    monkeypatch.setattr(job_matrix, "require_docker_runtime", reject_non_docker_runtime)
    monkeypatch.setattr(job_matrix.argparse.ArgumentParser, "parse_args", fail_if_args_are_parsed)

    try:
        job_matrix.main(["--corpus-plan-jsonl", "/tmp/plan.jsonl", "--output-root", "/tmp/out"])
    except RuntimeError as exc:
        assert str(exc) == "Docker-only test sentinel"
    else:
        raise AssertionError("main must fail through Docker runtime guard")

    assert calls == [
        (
            "require_docker_runtime",
            {
                "tool_name": "build_testing_stuff_music_corpus_job_matrix.py",
                "entrypoint_path": "/app/tools/build_testing_stuff_music_corpus_job_matrix.py",
            },
        )
    ]
