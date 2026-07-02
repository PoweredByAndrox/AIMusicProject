from __future__ import annotations

import json
from pathlib import Path

from tools.stage_generic_finetuning_controller_with_no_reference_selector import (
    INTEGRATION_DECISIONS_NAME,
    MANIFEST_NAME,
    PROGRESS_NAME,
    SELECTOR_DECISIONS_NAME,
    main,
    stage_controller_with_selector,
)
from tools.train_generic_finetuning_controller_model import train_controller_model


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def controller_row(row_id: str, stem: str, route: str, accepted: bool) -> dict:
    return {
        "id": row_id,
        "track_id": f"track-{row_id}",
        "stem_type": stem,
        "input": {
            "manifest_path": f"/app/lab_artifacts/controller/{row_id}/manifest.json",
            "model_id": route,
            "delivery_status": "accepted" if accepted else "blocked",
            "failure_labels": [] if accepted else ["leakage"],
        },
        "target": {
            "action": "reuse_route_candidate" if accepted else "avoid_route_and_generate_alternative",
            "chosen_model": route,
        },
        "result": {"accepted": accepted},
        "trainable": True,
    }


def judge_row(candidate_id: str, *, track_id: str, score: float = 0.86, decision: str = "accept_candidate") -> dict:
    return {
        "track_id": track_id,
        "candidate_id": candidate_id,
        "mix_path": f"/tmp/{track_id}/mix.wav",
        "instrumental_path": f"/tmp/{track_id}/{candidate_id}_minus.wav",
        "judge_mode": "no_reference",
        "score_mode": "minus",
        "overall_no_reference_score": score,
        "minus_score": score,
        "minus_decision": decision,
        "minus_blockers": [],
        "confidence": 0.88,
        "decision": decision,
        "reasons": [],
        "forbidden_actions": ["do_not_train"],
    }


def trained_controller(tmp_path: Path) -> dict:
    examples = write_jsonl(
        tmp_path / "controller.jsonl",
        [
            controller_row("good1", "minus", "good-route", True),
            controller_row("good2", "minus", "good-route", True),
            controller_row("bad1", "minus", "bad-route", False),
            controller_row("bad2", "minus", "bad-route", False),
        ],
    )
    return train_controller_model(
        controller_examples_jsonl=examples,
        output_dir=tmp_path / "controller_model",
        min_examples_per_route=1,
    )


def hard_case_policy(tmp_path: Path, **fields: object) -> Path:
    payload = {
        "schema": "no_reference_hard_case_route_policy.v1",
        "run_id": "policy-test",
        "track_id": "track-policy",
        "hard_case_detected": True,
        "trigger_reasons": [],
        "training_objective_required": False,
        "artifact_safe_route_required": False,
        "artifact_safe_candidate_route_ids": [],
        "selected_conditional_route_ids": [],
        "route_allowed": False,
        "route_reason": "",
        "route_block_reason": "",
        "release_ready": False,
    }
    payload.update(fields)
    return write_json(tmp_path / "hard_case_route_policy.json", payload)


def test_stages_controller_over_existing_selector_without_registry_edit(tmp_path: Path) -> None:
    training_manifest = trained_controller(tmp_path)
    judge_rows = write_jsonl(
        tmp_path / "judge.jsonl",
        [
            judge_row("good-route", track_id="track-good"),
            judge_row("bad-route", track_id="track-bad"),
        ],
    )

    manifest = stage_controller_with_selector(
        controller_model_json=Path(training_manifest["outputs"]["model_json"]),
        judge_jsonl=judge_rows,
        output_dir=tmp_path / "stage",
        score_mode="minus",
        min_selector_usable=2,
    )

    assert manifest["selector_summary"]["usable_without_stems"] == 2
    assert manifest["integration_summary"]["controller_missing_routes"] == 0
    assert manifest["integration_summary"]["controller_allowed_candidates"] == 1
    assert manifest["integration_summary"]["controller_blocked_candidates"] == 1
    assert manifest["promotion"]["status"] == "controller_selector_integration_staged"
    assert manifest["promotion"]["can_stage_controller_candidate"] is True
    assert manifest["promotion"]["can_edit_model_registry"] is False
    assert manifest["release_ready"] is False

    decisions = read_jsonl(tmp_path / "stage" / INTEGRATION_DECISIONS_NAME)
    actions = {row["selected_candidate_id"]: row["controller_action"] for row in decisions}
    assert actions == {
        "good-route": "allow_selector_candidate",
        "bad-route": "controller_block_selector_candidate",
    }


def test_blocks_staging_when_selector_route_is_missing_from_controller(tmp_path: Path) -> None:
    training_manifest = trained_controller(tmp_path)
    judge_rows = write_jsonl(tmp_path / "judge.jsonl", [judge_row("unknown-route", track_id="track-missing")])

    manifest = stage_controller_with_selector(
        controller_model_json=Path(training_manifest["outputs"]["model_json"]),
        judge_jsonl=judge_rows,
        output_dir=tmp_path / "stage",
        score_mode="minus",
        min_selector_usable=1,
    )

    assert manifest["integration_summary"]["controller_missing_routes"] == 1
    assert manifest["promotion"]["status"] == "blocked_missing_controller_routes"
    assert manifest["promotion"]["can_stage_controller_candidate"] is False


def test_uses_provenance_visible_alias_prior_for_selector_base_route(tmp_path: Path) -> None:
    examples = write_jsonl(
        tmp_path / "controller.jsonl",
        [
            controller_row("alias-bad1", "minus", "prefix_a__base-bad-route", False),
            controller_row("alias-bad2", "minus", "prefix_b__base-bad-route", False),
        ],
    )
    training_manifest = train_controller_model(
        controller_examples_jsonl=examples,
        output_dir=tmp_path / "controller_model",
        min_examples_per_route=1,
    )
    judge_rows = write_jsonl(tmp_path / "judge.jsonl", [judge_row("base-bad-route", track_id="track-alias")])

    manifest = stage_controller_with_selector(
        controller_model_json=Path(training_manifest["outputs"]["model_json"]),
        judge_jsonl=judge_rows,
        output_dir=tmp_path / "stage",
        score_mode="minus",
        min_selector_usable=1,
    )

    decisions = read_jsonl(tmp_path / "stage" / INTEGRATION_DECISIONS_NAME)
    assert decisions[0]["controller_route_match_kind"] == "alias_aggregate"
    assert decisions[0]["selected_candidate_id"] == "base-bad-route"
    assert decisions[0]["controller_matched_route_id"] == "alias:base-bad-route"
    assert set(decisions[0]["controller_source_route_ids"]) == {
        "prefix_a__base-bad-route",
        "prefix_b__base-bad-route",
    }
    assert decisions[0]["controller_action"] == "controller_block_selector_candidate"
    assert manifest["integration_summary"]["controller_missing_routes"] == 0
    assert manifest["promotion"]["status"] == "controller_selector_integration_staged"


def test_controller_aware_rerank_selects_next_allowed_candidate(tmp_path: Path) -> None:
    training_manifest = trained_controller(tmp_path)
    judge_rows = write_jsonl(
        tmp_path / "judge.jsonl",
        [
            judge_row("bad-route", track_id="track-rerank", score=0.95),
            judge_row("good-route", track_id="track-rerank", score=0.90),
        ],
    )

    manifest = stage_controller_with_selector(
        controller_model_json=Path(training_manifest["outputs"]["model_json"]),
        judge_jsonl=judge_rows,
        output_dir=tmp_path / "stage",
        score_mode="minus",
        min_selector_usable=1,
    )

    selector_rows = read_jsonl(tmp_path / "stage" / SELECTOR_DECISIONS_NAME)
    assert selector_rows[0]["selected_candidate_id"] == "bad-route"

    decisions = read_jsonl(tmp_path / "stage" / INTEGRATION_DECISIONS_NAME)
    assert decisions[0]["selector_original_selected_candidate_id"] == "bad-route"
    assert decisions[0]["selected_candidate_id"] == "good-route"
    assert decisions[0]["controller_action"] == "allow_controller_reranked_candidate"
    assert decisions[0]["controller_rerank_applied"] is True
    assert decisions[0]["controller_rerank_from_candidate_id"] == "bad-route"
    assert [row["candidate_id"] for row in decisions[0]["controller_rerank_evaluated_candidates"]] == [
        "bad-route",
        "good-route",
    ]
    assert manifest["integration_summary"]["controller_allowed_candidates"] == 1
    assert manifest["integration_summary"]["controller_blocked_candidates"] == 0
    assert manifest["integration_summary"]["controller_reranked_candidates"] == 1
    assert manifest["promotion"]["status"] == "controller_selector_integration_staged"


def test_controller_aware_rerank_does_not_allow_all_blocked_candidates(tmp_path: Path) -> None:
    training_manifest = trained_controller(tmp_path)
    judge_rows = write_jsonl(
        tmp_path / "judge.jsonl",
        [
            judge_row("bad-route", track_id="track-blocked", score=0.95),
            judge_row("bad-route-2", track_id="track-blocked", score=0.90),
        ],
    )

    manifest = stage_controller_with_selector(
        controller_model_json=Path(training_manifest["outputs"]["model_json"]),
        judge_jsonl=judge_rows,
        output_dir=tmp_path / "stage",
        score_mode="minus",
        min_selector_usable=1,
    )

    decisions = read_jsonl(tmp_path / "stage" / INTEGRATION_DECISIONS_NAME)
    assert decisions[0]["selected_candidate_id"] == "bad-route"
    assert decisions[0]["controller_action"] == "controller_block_selector_candidate"
    assert decisions[0]["controller_rerank_applied"] is False
    assert manifest["integration_summary"]["controller_allowed_candidates"] == 0
    assert manifest["integration_summary"]["controller_blocked_candidates"] == 1


def test_controller_aware_rerank_uses_candidate_stem_type_in_both_mode(tmp_path: Path) -> None:
    training_manifest = trained_controller(tmp_path)
    bad = judge_row("bad-route", track_id="track-both-rerank", score=0.95)
    bad.update({"stem_type": "minus", "pair_compatibility_score": 0.95, "pair_decision": "pair_compatible"})
    good = judge_row("good-route", track_id="track-both-rerank", score=0.90)
    good.update({"stem_type": "minus", "pair_compatibility_score": 0.90, "pair_decision": "pair_compatible"})
    judge_rows = write_jsonl(tmp_path / "judge.jsonl", [bad, good])

    stage_controller_with_selector(
        controller_model_json=Path(training_manifest["outputs"]["model_json"]),
        judge_jsonl=judge_rows,
        output_dir=tmp_path / "stage",
        score_mode="both",
        min_selector_usable=1,
    )

    decisions = read_jsonl(tmp_path / "stage" / INTEGRATION_DECISIONS_NAME)
    assert decisions[0]["stem_type"] == "minus"
    assert decisions[0]["selected_candidate_id"] == "good-route"
    assert decisions[0]["controller_action"] == "allow_controller_reranked_candidate"
    assert decisions[0]["controller_route_match_kind"] == "exact"


def test_cli_writes_staging_manifest_and_decisions(tmp_path: Path) -> None:
    training_manifest = trained_controller(tmp_path)
    judge_rows = write_jsonl(tmp_path / "judge.jsonl", [judge_row("good-route", track_id="track-good")])

    assert main(
        [
            "--controller-model-json",
            str(Path(training_manifest["outputs"]["model_json"])),
            "--judge-jsonl",
            str(judge_rows),
            "--output-dir",
            str(tmp_path / "stage"),
            "--score-mode",
            "minus",
            "--min-selector-usable",
            "1",
        ]
    ) == 0

    manifest = read_json(tmp_path / "stage" / MANIFEST_NAME)
    assert manifest["promotion"]["status"] == "controller_selector_integration_staged"
    assert (tmp_path / "stage" / SELECTOR_DECISIONS_NAME).exists()
    assert (tmp_path / "stage" / INTEGRATION_DECISIONS_NAME).exists()


def test_repeat_run_replaces_progress_log_instead_of_appending(tmp_path: Path) -> None:
    training_manifest = trained_controller(tmp_path)
    judge_rows = write_jsonl(tmp_path / "judge.jsonl", [judge_row("good-route", track_id="track-good")])
    output_dir = tmp_path / "stage"
    kwargs = {
        "controller_model_json": Path(training_manifest["outputs"]["model_json"]),
        "judge_jsonl": judge_rows,
        "output_dir": output_dir,
        "score_mode": "minus",
        "min_selector_usable": 1,
    }

    stage_controller_with_selector(**kwargs)
    stage_controller_with_selector(**kwargs)

    progress_rows = read_jsonl(output_dir / PROGRESS_NAME)
    assert [row["event"] for row in progress_rows] == ["start", "write_outputs", "finish"]


def test_training_objective_policy_blocks_selector_staging(tmp_path: Path) -> None:
    training_manifest = trained_controller(tmp_path)
    judge_rows = write_jsonl(tmp_path / "judge.jsonl", [judge_row("good-route", track_id="track-training-required")])
    policy_json = hard_case_policy(
        tmp_path,
        trigger_reasons=["model_gap_requires_separator_training_objective"],
        training_objective_required=True,
        route_block_reason="hard_case_requires_separator_training_objective",
        route_reason="train_separator_attribution_objective_required",
    )

    manifest = stage_controller_with_selector(
        controller_model_json=Path(training_manifest["outputs"]["model_json"]),
        judge_jsonl=judge_rows,
        output_dir=tmp_path / "stage",
        score_mode="minus",
        min_selector_usable=1,
        hard_case_route_policy_json=policy_json,
    )

    assert manifest["promotion"]["status"] == "blocked_hard_case_requires_separator_training_objective"
    assert manifest["promotion"]["can_stage_controller_candidate"] is False
    assert manifest["integration_summary"]["hard_case_training_objective_required"] is True
    assert manifest["integration_summary"]["controller_allowed_candidates"] == 0

    decisions = read_jsonl(tmp_path / "stage" / INTEGRATION_DECISIONS_NAME)
    assert decisions[0]["controller_action"] == "require_separator_training_objective"
    assert decisions[0]["hard_case_route_policy_applied"] is True
    assert decisions[0]["hard_case_policy_block_reason"] == "hard_case_requires_separator_training_objective"


def test_artifact_safe_policy_reranks_selector_to_allowed_route(tmp_path: Path) -> None:
    examples = write_jsonl(
        tmp_path / "controller.jsonl",
        [
            controller_row("good1", "minus", "good-route", True),
            controller_row("good2", "minus", "good-route", True),
            controller_row("safe1", "minus", "safe-route", True),
            controller_row("safe2", "minus", "safe-route", True),
        ],
    )
    training_manifest = train_controller_model(
        controller_examples_jsonl=examples,
        output_dir=tmp_path / "controller_model",
        min_examples_per_route=1,
    )
    judge_rows = write_jsonl(
        tmp_path / "judge.jsonl",
        [
            judge_row("good-route", track_id="track-safe-required", score=0.95),
            judge_row("safe-route", track_id="track-safe-required", score=0.90),
        ],
    )
    policy_json = hard_case_policy(
        tmp_path,
        trigger_reasons=["pitch_guard_damage_regression"],
        artifact_safe_route_required=True,
        artifact_safe_candidate_route_ids=["safe-route"],
        selected_conditional_route_ids=["safe-route"],
        route_allowed=True,
        route_reason="artifact_safe_route_for_damage_regression_blocker",
    )

    manifest = stage_controller_with_selector(
        controller_model_json=Path(training_manifest["outputs"]["model_json"]),
        judge_jsonl=judge_rows,
        output_dir=tmp_path / "stage",
        score_mode="minus",
        min_selector_usable=1,
        hard_case_route_policy_json=policy_json,
    )

    decisions = read_jsonl(tmp_path / "stage" / INTEGRATION_DECISIONS_NAME)
    assert decisions[0]["selector_original_selected_candidate_id"] == "good-route"
    assert decisions[0]["selected_candidate_id"] == "safe-route"
    assert decisions[0]["controller_action"] == "allow_controller_reranked_candidate"
    assert decisions[0]["hard_case_artifact_safe_route_required"] is True
    assert decisions[0]["hard_case_allowed_route_ids"] == ["safe-route"]
    assert manifest["integration_summary"]["hard_case_artifact_safe_route_required"] is True
    assert manifest["integration_summary"]["controller_reranked_candidates"] == 1
