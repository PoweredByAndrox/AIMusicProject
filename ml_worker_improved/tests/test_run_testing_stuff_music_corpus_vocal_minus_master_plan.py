from __future__ import annotations

import json
from pathlib import Path

import tools.run_testing_stuff_music_corpus_vocal_minus_master_plan as runner
from tools.run_testing_stuff_music_corpus_job_matrix import EXECUTION_MANIFEST_NAME


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def plan_json(tmp_path: Path, *tracks: dict) -> Path:
    return write_json(
        tmp_path / "plan.json",
        {
            "schema": "testing_stuff_music_corpus_vocal_minus_master_plan.v1",
            "manifest_type": "testing_stuff_music_corpus_vocal_minus_master_plan",
            "delivery_allowed": False,
            "release_ready": False,
            "tracks": list(tracks),
        },
    )


def track_row(track_id: str, next_gate: str, **extra: object) -> dict:
    return {
        "track_id": track_id,
        "source_audio_path": f"/app/audio/{track_id}.wav",
        "next_gate": next_gate,
        "missing_gates": ["vocal_delivery"],
        "delivery_allowed": False,
        "release_ready": False,
        **extra,
    }


def accepted_manifest(track_id: str, stem_type: str) -> dict:
    return {
        "track_id": track_id,
        "stem_type": stem_type,
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "release_ready": False,
    }


def source_ready_full_stem_pack_manifest(track_id: str) -> dict:
    return {
        "track_id": track_id,
        "stem_type": "full_stem_pack",
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "aggregate_machine_gate_required_not_for_listen_pack",
        "release_status": "not_release_ready",
        "dominant_failure_reason": "requires_staged_aggregate_validation_and_machine_gate",
        "source_pack_ready": True,
        "source_pack_delivery_status": "accepted",
        "source_pack_presentation_status": "accepted_for_listen_pack",
        "source_pack_release_status": "listen_preview_only",
        "requires_staged_aggregate_machine_gate": True,
        "final_delivery_blockers": ["requires_staged_aggregate_validation_and_machine_gate"],
        "release_ready": False,
        "accepted_stem_count": 6,
    }


def test_dry_run_records_next_gate_without_executing_delivery_runner(tmp_path: Path) -> None:
    plan_path = plan_json(tmp_path, track_row("track_a", "run_vocal_delivery_pipeline"))
    calls: list[dict] = []

    def reject_vocal_runner(**kwargs):
        calls.append(kwargs)
        raise AssertionError("dry-run must not execute vocal delivery")

    summary = runner.run_plan(
        plan_json=plan_path,
        output_root=tmp_path / "out",
        max_steps=1,
        execute=False,
        vocal_delivery_runner=reject_vocal_runner,
    )

    assert calls == []
    assert summary["execution_started"] is False
    assert summary["selected_step_count"] == 1
    assert summary["step_status_counts"] == {"dry_run": 1}
    assert summary["delivery_allowed"] is False
    assert summary["release_ready"] is False
    step = summary["steps"][0]
    assert step["step_status"] == "dry_run"
    assert step["expected_acceleration_mode"] == "external_only_cpu_scoring"
    assert step["gpu_inference_skipped"] is True
    assert (tmp_path / "out" / "testing_stuff_music_corpus_vocal_minus_master_execution_progress.jsonl").exists()


def test_vocal_delivery_uses_mode_specific_score_matrix_before_runner(tmp_path: Path) -> None:
    plan_path = plan_json(tmp_path, track_row("track_a", "run_vocal_delivery_pipeline"))
    vocal_judge = write_jsonl(tmp_path / "vocal_judge" / "track_a" / "no_reference_vocal_judge.jsonl", [{"track_id": "track_a"}])
    matrix = write_json(
        tmp_path / "vocal_matrix" / "track_a" / "no_reference_vocal_candidate_score_matrix.json",
        {"rows": [{"track_id": "track_a", "canonical_candidate_id": "candidate-a"}]},
    )
    calls: list[dict] = []

    def fake_vocal_runner(**kwargs):
        calls.append(kwargs)
        assert kwargs["vocal_judge_jsonl"] == vocal_judge
        assert kwargs["candidate_score_matrix_json"] == matrix
        assert kwargs["base_candidate_id"] is None
        write_json(kwargs["output_dir"] / "no_reference_vocal_listen_preview_manifest.json", accepted_manifest("track_a", "vocal"))
        return accepted_manifest("track_a", "vocal")

    summary = runner.run_plan(
        plan_json=plan_path,
        output_root=tmp_path / "out",
        max_steps=1,
        execute=True,
        vocal_judge_roots=[tmp_path / "vocal_judge"],
        vocal_score_matrix_roots=[tmp_path / "vocal_matrix"],
        vocal_delivery_runner=fake_vocal_runner,
    )

    assert len(calls) == 1
    assert summary["execution_started"] is True
    assert summary["step_status_counts"] == {"completed": 1}
    step = summary["steps"][0]
    assert step["step_status"] == "completed"
    assert step["candidate_score_matrix_json"] == str(matrix)
    assert step["output_manifest_json"].endswith("no_reference_vocal_listen_preview_manifest.json")
    assert summary["delivery_allowed"] is False
    assert summary["release_ready"] is False


def test_vocal_delivery_accepts_windowed_judge_results_filename(tmp_path: Path) -> None:
    plan_path = plan_json(tmp_path, track_row("track_a", "run_vocal_delivery_pipeline"))
    vocal_judge = write_jsonl(
        tmp_path / "vocal_judge" / "track_a" / "no_reference_windowed_judge_results.jsonl",
        [{"track_id": "track_a"}],
    )
    matrix = write_json(
        tmp_path / "vocal_matrix" / "track_a" / "no_reference_vocal_candidate_score_matrix.json",
        {"rows": [{"track_id": "track_a", "canonical_candidate_id": "candidate-a"}]},
    )
    calls: list[dict] = []

    def fake_vocal_runner(**kwargs):
        calls.append(kwargs)
        assert kwargs["vocal_judge_jsonl"] == vocal_judge
        assert kwargs["candidate_score_matrix_json"] == matrix
        return accepted_manifest("track_a", "vocal")

    summary = runner.run_plan(
        plan_json=plan_path,
        output_root=tmp_path / "out",
        max_steps=1,
        execute=True,
        vocal_judge_roots=[tmp_path / "vocal_judge"],
        vocal_score_matrix_roots=[tmp_path / "vocal_matrix"],
        vocal_delivery_runner=fake_vocal_runner,
    )

    assert len(calls) == 1
    assert summary["step_status_counts"] == {"completed": 1}
    assert summary["steps"][0]["vocal_judge_jsonl"] == str(vocal_judge)


def test_blocks_vocal_delivery_without_mode_specific_score_matrix(tmp_path: Path) -> None:
    plan_path = plan_json(tmp_path, track_row("track_a", "run_vocal_delivery_pipeline"))
    write_jsonl(tmp_path / "vocal_judge" / "track_a" / "no_reference_vocal_judge.jsonl", [{"track_id": "track_a"}])
    calls: list[dict] = []

    def reject_vocal_runner(**kwargs):
        calls.append(kwargs)
        raise AssertionError("runner must not execute without mode-specific matrix")

    summary = runner.run_plan(
        plan_json=plan_path,
        output_root=tmp_path / "out",
        max_steps=1,
        execute=True,
        vocal_judge_roots=[tmp_path / "vocal_judge"],
        vocal_delivery_runner=reject_vocal_runner,
    )

    assert calls == []
    assert summary["step_status_counts"] == {"blocked_missing_vocal_score_matrix": 1}
    assert summary["steps"][0]["step_status"] == "blocked_missing_vocal_score_matrix"
    assert summary["execution_started"] is False
    assert summary["release_ready"] is False


def test_minus_delivery_uses_self_judge_root_and_score_matrix(tmp_path: Path) -> None:
    plan_path = plan_json(tmp_path, track_row("track_b", "run_minus_delivery_pipeline"))
    self_judge = write_json(
        tmp_path / "self_judge" / "track_b" / "candidate-a" / "instrumental_self_judge_manifest.json",
        {"track_id": "track_b", "candidate_label": "candidate-a"},
    )
    matrix = write_json(
        tmp_path / "minus_matrix" / "track_b" / "no_reference_minus_candidate_score_matrix.json",
        {"rows": [{"track_id": "track_b", "canonical_candidate_id": "candidate-a"}]},
    )
    calls: list[dict] = []

    def fake_minus_runner(**kwargs):
        calls.append(kwargs)
        assert kwargs["self_judge_manifest_paths"] == [self_judge]
        assert kwargs["candidate_score_matrix_json"] == matrix
        assert kwargs["source_audio_path"] == Path("/app/audio/track_b.wav")
        write_json(kwargs["output_dir"] / "no_reference_minus_listen_preview_manifest.json", accepted_manifest("track_b", "minus"))
        return accepted_manifest("track_b", "minus")

    summary = runner.run_plan(
        plan_json=plan_path,
        output_root=tmp_path / "out",
        max_steps=1,
        execute=True,
        minus_self_judge_roots=[tmp_path / "self_judge"],
        minus_score_matrix_roots=[tmp_path / "minus_matrix"],
        minus_delivery_runner=fake_minus_runner,
    )

    assert len(calls) == 1
    assert summary["step_status_counts"] == {"completed": 1}
    assert summary["steps"][0]["candidate_score_matrix_json"] == str(matrix)


def test_master_step_builds_from_existing_vocal_minus_and_pair_compatibility(tmp_path: Path) -> None:
    vocal_manifest = write_json(tmp_path / "vocal" / "no_reference_vocal_listen_preview_manifest.json", accepted_manifest("track_c", "vocal"))
    minus_manifest = write_json(tmp_path / "minus" / "no_reference_minus_listen_preview_manifest.json", accepted_manifest("track_c", "minus"))
    pair_jsonl = write_jsonl(tmp_path / "pairs" / "no_reference_pair_compatibility.jsonl", [{"track_id": "track_c", "decision": "accepted"}])
    plan_path = plan_json(
        tmp_path,
        track_row(
            "track_c",
            "run_pair_compatibility_and_master_manifest",
            vocal_manifest_path=str(vocal_manifest),
            minus_manifest_path=str(minus_manifest),
        ),
    )
    calls: list[dict] = []

    def fake_master_builder(**kwargs):
        calls.append(kwargs)
        assert kwargs["vocal_manifest_json"] == vocal_manifest
        assert kwargs["minus_manifest_json"] == minus_manifest
        assert kwargs["pair_compatibility_jsonl"] == pair_jsonl
        return accepted_manifest("track_c", "both")

    summary = runner.run_plan(
        plan_json=plan_path,
        output_root=tmp_path / "out",
        max_steps=1,
        execute=True,
        pair_compatibility_roots=[tmp_path / "pairs"],
        master_manifest_builder=fake_master_builder,
    )

    assert len(calls) == 1
    assert summary["step_status_counts"] == {"completed": 1}
    assert summary["steps"][0]["output_manifest_json"].endswith("no_reference_vocal_minus_master_manifest.json")
    assert summary["release_ready"] is False


def test_full_stem_pack_step_builds_from_master_and_stem_score_matrix(tmp_path: Path) -> None:
    master_manifest = write_json(
        tmp_path / "master" / "no_reference_vocal_minus_master_manifest.json",
        accepted_manifest("track_d", "both"),
    )
    stem_matrix = write_json(
        tmp_path / "stem_matrix" / "track_d" / "no_reference_stem_candidate_score_matrix.json",
        {"rows": [{"track_id": "track_d", "stem_type": "melody"}]},
    )
    plan_path = plan_json(
        tmp_path,
        track_row(
            "track_d",
            "build_full_stem_pack_manifest",
            master_manifest_path=str(master_manifest),
        ),
    )
    calls: list[dict] = []

    def fake_full_stem_pack_builder(**kwargs):
        calls.append(kwargs)
        assert kwargs["master_manifest_json"] == master_manifest
        assert kwargs["stem_score_matrix_json"] == stem_matrix
        return source_ready_full_stem_pack_manifest("track_d")

    summary = runner.run_plan(
        plan_json=plan_path,
        output_root=tmp_path / "out",
        max_steps=1,
        execute=True,
        stem_score_matrix_roots=[tmp_path / "stem_matrix"],
        full_stem_pack_builder=fake_full_stem_pack_builder,
    )

    assert len(calls) == 1
    assert summary["step_status_counts"] == {"completed_source_ready_requires_aggregate_machine_gate": 1}
    step = summary["steps"][0]
    assert step["next_gate"] == "build_full_stem_pack_manifest"
    assert step["stem_score_matrix_json"] == str(stem_matrix)
    assert step["output_manifest_json"].endswith("no_reference_full_stem_pack_manifest.json")
    assert step["delivery_status"] == "blocked"
    assert step["source_pack_ready"] is True
    assert step["requires_staged_aggregate_machine_gate"] is True
    assert summary["delivery_allowed"] is False
    assert summary["release_ready"] is False


def test_fail_fast_continues_after_full_stem_source_ready_step(tmp_path: Path) -> None:
    master_manifest = write_json(
        tmp_path / "master" / "track_d" / "no_reference_vocal_minus_master_manifest.json",
        accepted_manifest("track_d", "both"),
    )
    stem_matrix = write_json(
        tmp_path / "stem_matrix" / "track_d" / "no_reference_stem_candidate_score_matrix.json",
        {"rows": [{"track_id": "track_d", "stem_type": "melody"}]},
    )
    plan_path = plan_json(
        tmp_path,
        track_row(
            "track_d",
            "build_full_stem_pack_manifest",
            master_manifest_path=str(master_manifest),
        ),
        track_row("track_h", "run_corpus_job_matrix_for_track"),
    )
    full_pack_calls: list[dict] = []
    generation_calls: list[dict] = []

    def fake_full_stem_pack_builder(**kwargs):
        full_pack_calls.append(kwargs)
        assert kwargs["stem_score_matrix_json"] == stem_matrix
        return source_ready_full_stem_pack_manifest("track_d")

    def fake_job_matrix_executor(**kwargs):
        generation_calls.append(kwargs)
        output_root = Path(kwargs["output_root"])
        return {
            "summary": {
                "completed_jobs": 1,
                "execution_started": True,
                "release_ready": False,
            },
            "manifest": {
                "output_files": {"manifest_json": str(output_root / EXECUTION_MANIFEST_NAME)},
                "delivery_allowed": False,
                "release_ready": False,
            },
            "rows": [],
        }

    summary = runner.run_plan(
        plan_json=plan_path,
        output_root=tmp_path / "out",
        max_steps=2,
        execute=True,
        fail_fast=True,
        stem_score_matrix_roots=[tmp_path / "stem_matrix"],
        job_matrix_jsonl=tmp_path / "jobs.jsonl",
        full_stem_pack_builder=fake_full_stem_pack_builder,
        job_matrix_executor=fake_job_matrix_executor,
    )

    assert len(full_pack_calls) == 1
    assert len(generation_calls) == 1
    assert summary["step_status_counts"] == {
        "completed": 1,
        "completed_source_ready_requires_aggregate_machine_gate": 1,
    }
    assert summary["executed_step_count"] == 2


def test_nonvocal_score_matrix_step_runs_semantic_batch_for_track(tmp_path: Path) -> None:
    plan_path = plan_json(tmp_path, track_row("track_e", "complete_nonvocal_stem_score_matrices"))
    listen_root = tmp_path / "stem_listen"
    calls: list[dict] = []

    def fake_semantic_batch_runner(**kwargs):
        calls.append(kwargs)
        assert kwargs["listen_manifest_roots"] == [listen_root]
        assert kwargs["track_ids"] == ["track_e"]
        output_root = kwargs["output_root"]
        matrix = {
            "matrix_type": "no_reference_stem_candidate_score_matrix",
            "score_mode": "stem",
            "ranked_candidates": [
                {
                    "track_id": "track_e",
                    "stem_type": stem,
                    "selected_audio_path": f"/app/audio/track_e_{stem}.flac",
                    "delivery_status": "accepted",
                    "delivery_allowed": True,
                    "presentation_status": "accepted_for_listen_pack",
                    "release_status": "listen_preview_only",
                    "release_ready": False,
                    "semantic_quality_verified": True,
                    "stem_semantic_blockers": [],
                    "selected_audio_exists": True,
                }
                for stem in ("bass", "drums", "melody", "other")
            ],
            "summary": {
                "accepted_listen_preview_count": 4,
                "semantic_quality_verified": True,
                "semantic_quality_verified_count": 4,
                "release_ready": False,
            },
            "release_ready": False,
        }
        matrix_json = output_root / "track_e" / "stem_score_matrix" / "no_reference_stem_candidate_score_matrix.json"
        write_json(matrix_json, matrix)
        return {
            "batch_status": "completed",
            "track_count": 1,
            "candidate_count": 4,
            "accepted_candidates": 4,
            "semantic_quality_verified_count": 4,
            "tracks": [
                {
                    "track_id": "track_e",
                    "candidate_count": 4,
                    "accepted_candidates": 4,
                    "semantic_quality_verified_count": 4,
                    "matrix_json": str(matrix_json),
                }
            ],
            "release_ready": False,
        }

    summary = runner.run_plan(
        plan_json=plan_path,
        output_root=tmp_path / "out",
        max_steps=1,
        execute=True,
        stem_listen_manifest_roots=[listen_root],
        stem_semantic_batch_runner=fake_semantic_batch_runner,
    )

    assert len(calls) == 1
    assert summary["step_status_counts"] == {"completed": 1}
    assert summary["expected_acceleration_modes"] == ["external_only_cpu_dsp_scoring"]
    step = summary["steps"][0]
    assert step["next_gate"] == "complete_nonvocal_stem_score_matrices"
    assert step["stem_score_matrix_json"].endswith("no_reference_stem_candidate_score_matrix.json")
    assert step["output_manifest_json"] == step["stem_score_matrix_json"]
    assert step["accepted_nonvocal_stems"] == ["bass", "drums", "melody", "other"]
    assert step["delivery_allowed"] is False
    assert step["release_ready"] is False
    assert summary["delivery_allowed"] is False
    assert summary["release_ready"] is False


def test_nonvocal_score_matrix_step_blocks_without_stem_listen_roots(tmp_path: Path) -> None:
    plan_path = plan_json(tmp_path, track_row("track_f", "complete_nonvocal_stem_score_matrices"))
    calls: list[dict] = []

    def reject_semantic_batch_runner(**kwargs):
        calls.append(kwargs)
        raise AssertionError("semantic batch must not run without stem listen roots")

    summary = runner.run_plan(
        plan_json=plan_path,
        output_root=tmp_path / "out",
        max_steps=1,
        execute=True,
        stem_semantic_batch_runner=reject_semantic_batch_runner,
    )

    assert calls == []
    assert summary["step_status_counts"] == {"blocked_missing_stem_listen_manifest_root": 1}
    assert summary["steps"][0]["dominant_failure_reason"] == "missing_stem_listen_manifest_root"
    assert summary["steps"][0]["execution_started"] is False
    assert summary["release_ready"] is False


def test_nonvocal_score_matrix_step_blocks_when_semantic_batch_omits_matrix(tmp_path: Path) -> None:
    plan_path = plan_json(tmp_path, track_row("track_g", "complete_nonvocal_stem_score_matrices"))

    def fake_semantic_batch_runner(**kwargs):
        return {
            "batch_status": "completed",
            "track_count": 1,
            "candidate_count": 4,
            "accepted_candidates": 4,
            "semantic_quality_verified_count": 4,
            "tracks": [{"track_id": "track_g"}],
            "release_ready": False,
        }

    summary = runner.run_plan(
        plan_json=plan_path,
        output_root=tmp_path / "out",
        max_steps=1,
        execute=True,
        stem_listen_manifest_roots=[tmp_path / "listen"],
        stem_semantic_batch_runner=fake_semantic_batch_runner,
    )

    assert summary["step_status_counts"] == {"blocked_missing_stem_score_matrix": 1}
    assert summary["steps"][0]["dominant_failure_reason"] == "semantic_batch_missing_matrix_json"
    assert summary["steps"][0]["execution_started"] is True
    assert summary["release_ready"] is False


def test_generation_step_uses_short_job_execution_manifest_name(tmp_path: Path) -> None:
    plan_path = plan_json(tmp_path, track_row("track_h", "run_corpus_job_matrix_for_track"))
    calls: list[dict] = []

    def fake_job_matrix_executor(**kwargs):
        calls.append(kwargs)
        output_root = Path(kwargs["output_root"])
        return {
            "summary": {
                "completed_jobs": 1,
                "execution_started": True,
                "release_ready": False,
            },
            "manifest": {
                "output_files": {
                    "manifest_json": str(output_root / EXECUTION_MANIFEST_NAME),
                },
                "delivery_allowed": False,
                "release_ready": False,
            },
            "rows": [],
        }

    summary = runner.run_plan(
        plan_json=plan_path,
        output_root=tmp_path / "out",
        max_steps=1,
        execute=True,
        job_matrix_jsonl=tmp_path / "jobs.jsonl",
        job_matrix_executor=fake_job_matrix_executor,
    )

    assert len(calls) == 1
    step = summary["steps"][0]
    assert step["step_status"] == "completed"
    assert Path(step["output_manifest_json"]).name == EXECUTION_MANIFEST_NAME
    assert len(Path(step["output_manifest_json"]).name) <= 32
