from __future__ import annotations

import json
from pathlib import Path

from tools.optimize_no_reference_harness import main, optimize_harness, select_pareto_frontier
from tools.replay_no_reference_harness_candidate import default_harness_candidate


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def make_bundle(tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "bundle"
    write_json(bundle_dir / "replay_bundle_manifest.json", {"bundle_id": "bundle-a", "run_count": 2, "replay_complete_runs": 2, "replay_incomplete_runs": 0, "release_ready_count": 0})
    write_jsonl(
        bundle_dir / "replay_bundle_events.jsonl",
        [
            {
                "event_type": "run_summary",
                "track_id": "track-a",
                "replay_complete": True,
                "preview_ready_count": 1,
                "accepted_pair_ready_count": 1,
                "manifest_preview_ready_count": 1,
                "audio_preview_ready_count": 1,
                "rendered_preview_audio_count": 2,
                "preview_render_failed_count": 0,
                "preview_render_skipped_count": 0,
                "preview_validation_failed_count": 0,
                "preview_contract_missing_count": 0,
                "minus_accepted_count": 1,
                "pair_compatible_count": 1,
                "severe_vocal_bleed_in_minus_count": 0,
                "severe_phase_collapse_count": 0,
                "failed_run_count": 0,
                "timeout_count": 0,
                "incomplete_replay_count": 0,
                "failed_safety_contract_count": 0,
                "release_ready_count": 0,
                "estimated_runtime_sec": 100,
                "estimated_gpu_jobs": 1,
                "duplicate_jobs_avoided": 1,
                "candidate_count": 2,
            },
            {
                "event_type": "run_summary",
                "track_id": "track-b",
                "replay_complete": True,
                "preview_ready_count": 0,
                "accepted_pair_ready_count": 0,
                "manifest_preview_ready_count": 0,
                "audio_preview_ready_count": 0,
                "rendered_preview_audio_count": 0,
                "preview_render_failed_count": 0,
                "preview_render_skipped_count": 0,
                "preview_validation_failed_count": 0,
                "preview_contract_missing_count": 0,
                "minus_accepted_count": 0,
                "pair_compatible_count": 0,
                "severe_vocal_bleed_in_minus_count": 2,
                "severe_phase_collapse_count": 1,
                "failed_run_count": 0,
                "timeout_count": 0,
                "incomplete_replay_count": 0,
                "failed_safety_contract_count": 0,
                "release_ready_count": 0,
                "estimated_runtime_sec": 150,
                "estimated_gpu_jobs": 2,
                "duplicate_jobs_avoided": 0,
                "candidate_count": 4,
            },
        ],
    )
    write_jsonl(bundle_dir / "replay_bundle_artifacts_index.jsonl", [])
    return bundle_dir


def candidate_eval(
    candidate_id: str,
    *,
    complexity: int,
    per_track_scores: dict[str, float],
    manifest_preview_ready_count: int = 1,
    audio_preview_ready_count: int = 1,
    rendered_preview_audio_count: int = 2,
    preview_render_failed_count: int = 0,
    preview_validation_failed_count: int = 0,
    preview_contract_missing_count: int = 0,
) -> dict:
    return {
        "harness_candidate_id": candidate_id,
        "safety_valid": True,
        "preview_ready_count": 1,
        "accepted_pair_ready_count": 1,
        "manifest_preview_ready_count": manifest_preview_ready_count,
        "audio_preview_ready_count": audio_preview_ready_count,
        "rendered_preview_audio_count": rendered_preview_audio_count,
        "preview_render_failed_count": preview_render_failed_count,
        "preview_render_skipped_count": 0,
        "preview_validation_failed_count": preview_validation_failed_count,
        "preview_contract_missing_count": preview_contract_missing_count,
        "minus_accepted_count": 1,
        "pair_compatible_count": 1,
        "severe_vocal_bleed_in_minus_count": 0,
        "severe_phase_collapse_count": 0,
        "failed_run_count": 0,
        "release_ready_count": 0,
        "estimated_runtime_sec": 100,
        "complexity_score": complexity,
        "per_track_scores": per_track_scores,
    }


def test_pareto_selector_preserves_lower_complexity_with_same_score() -> None:
    rows = [
        candidate_eval("complex", complexity=5, per_track_scores={"a": 1.0, "b": 1.0}),
        candidate_eval("simple", complexity=1, per_track_scores={"a": 1.0, "b": 1.0}),
    ]

    selected = select_pareto_frontier(rows)

    assert [row["harness_candidate_id"] for row in selected["frontier_rows"]] == ["simple"]
    rejected = {row["harness_candidate_id"]: row["selection_decision"] for row in selected["rejected_rows"]}
    assert rejected["complex"] == "candidate_rejected_complexity"


def test_candidate_good_on_one_track_bad_on_another_becomes_specialist() -> None:
    baseline = candidate_eval("baseline", complexity=1, per_track_scores={"a": 1.0, "b": 1.0})
    specialist = candidate_eval("specialist", complexity=2, per_track_scores={"a": 2.0, "b": 0.0})

    selected = select_pareto_frontier([baseline, specialist], baseline_eval=baseline)

    specialist_rows = [row for row in selected["specialist_rows"] if row["harness_candidate_id"] == "specialist"]
    assert specialist_rows
    assert specialist_rows[0]["selection_decision"] == "candidate_specialist_route_only"
    assert all(row["harness_candidate_id"] != "specialist" for row in selected["frontier_rows"])


def test_pareto_prioritizes_audio_preview_over_manifest_only_preview() -> None:
    manifest_only = candidate_eval(
        "manifest-only",
        complexity=1,
        per_track_scores={"a": 1.0},
        manifest_preview_ready_count=3,
        audio_preview_ready_count=0,
        rendered_preview_audio_count=0,
    )
    audio_ready = candidate_eval(
        "audio-ready",
        complexity=1,
        per_track_scores={"a": 1.0},
        manifest_preview_ready_count=1,
        audio_preview_ready_count=1,
        rendered_preview_audio_count=2,
    )

    selected = select_pareto_frontier([manifest_only, audio_ready])

    assert [row["harness_candidate_id"] for row in selected["frontier_rows"]] == ["audio-ready"]
    rejected = {row["harness_candidate_id"]: row["selection_decision"] for row in selected["rejected_rows"]}
    assert rejected["manifest-only"] == "candidate_rejected_pareto_dominated"


def test_preview_render_failed_is_negative_pareto_metric() -> None:
    failed = candidate_eval(
        "render-failed",
        complexity=1,
        per_track_scores={"a": 1.0},
        preview_render_failed_count=1,
    )
    clean = candidate_eval("clean", complexity=1, per_track_scores={"a": 1.0})

    selected = select_pareto_frontier([failed, clean])

    assert [row["harness_candidate_id"] for row in selected["frontier_rows"]] == ["clean"]
    rejected = {row["harness_candidate_id"]: row["selection_decision"] for row in selected["rejected_rows"]}
    assert rejected["render-failed"] == "candidate_rejected_pareto_dominated"


def test_preview_contract_missing_is_negative_pareto_metric() -> None:
    missing_contract = candidate_eval(
        "contract-missing",
        complexity=1,
        per_track_scores={"a": 1.0},
        preview_contract_missing_count=1,
    )
    clean = candidate_eval("clean", complexity=1, per_track_scores={"a": 1.0})

    selected = select_pareto_frontier([missing_contract, clean])

    assert [row["harness_candidate_id"] for row in selected["frontier_rows"]] == ["clean"]
    rejected = {row["harness_candidate_id"]: row["selection_decision"] for row in selected["rejected_rows"]}
    assert rejected["contract-missing"] == "candidate_rejected_preview_contract_missing"


def test_preview_contract_missing_candidate_cannot_be_promoted_globally() -> None:
    selected = select_pareto_frontier(
        [
            candidate_eval(
                "only-missing-contract",
                complexity=1,
                per_track_scores={"a": 1.0},
                preview_contract_missing_count=1,
            )
        ]
    )

    assert selected["frontier_rows"] == []
    assert selected["rejected_rows"][0]["selection_decision"] == "candidate_rejected_preview_contract_missing"


def test_deterministic_optimizer_works_without_llm(tmp_path: Path) -> None:
    bundle_dir = make_bundle(tmp_path)
    baseline = default_harness_candidate()
    baseline_json = tmp_path / "baseline_harness.json"
    write_json(baseline_json, baseline)

    result = optimize_harness(
        replay_bundle_dir=bundle_dir,
        baseline_harness_json=baseline_json,
        output_dir=tmp_path / "optimizer",
        generations=2,
        population_size=4,
        use_llm_proposer=False,
    )

    assert result["llm_proposer_used"] is False
    assert result["release_ready_count"] == 0
    assert result["failure_reflections_generated"] >= 0
    assert result["evaluated_candidates"] == 4
    assert (tmp_path / "optimizer" / "harness_candidate_eval.jsonl").exists()
    assert (tmp_path / "optimizer" / "harness_pareto_frontier.jsonl").exists()
    assert (tmp_path / "optimizer" / "failure_reflections" / "failure_reflections.jsonl").exists()
    assert (tmp_path / "optimizer" / "harness_optimizer_summary.md").exists()


def test_optimizer_cli_uses_default_baseline_when_not_provided(tmp_path: Path) -> None:
    bundle_dir = make_bundle(tmp_path)
    output_dir = tmp_path / "optimizer"

    exit_code = main(
        [
            "--replay-bundle-dir",
            str(bundle_dir),
            "--output-dir",
            str(output_dir),
            "--generations",
            "1",
            "--population-size",
            "3",
            "--use-llm-proposer",
            "false",
        ]
    )

    summary = json.loads((output_dir / "harness_optimizer_summary.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert summary["llm_proposer_used"] is False
    assert Path(summary["baseline_harness_json"]).name == "baseline_harness.json"
    assert Path(summary["baseline_harness_json"]).exists()


def test_optimizer_upgrades_existing_baseline_with_module_schema(tmp_path: Path) -> None:
    bundle_dir = make_bundle(tmp_path)
    output_dir = tmp_path / "optimizer"
    baseline_json = tmp_path / "baseline_harness.json"
    old_baseline = default_harness_candidate()
    old_baseline.pop("module_config")
    old_baseline.pop("stage_graph")
    write_json(baseline_json, old_baseline)

    result = optimize_harness(
        replay_bundle_dir=bundle_dir,
        baseline_harness_json=baseline_json,
        output_dir=output_dir,
        generations=1,
        population_size=3,
        use_llm_proposer=False,
    )

    rewritten = json.loads(baseline_json.read_text(encoding="utf-8"))
    assert result["rejected_safety_candidates"] == 0
    assert "module_config" in rewritten
    assert "stage_graph" in rewritten
