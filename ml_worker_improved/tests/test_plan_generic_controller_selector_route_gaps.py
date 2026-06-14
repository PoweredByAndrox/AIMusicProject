from __future__ import annotations

import json
from pathlib import Path

from tools.plan_generic_controller_selector_route_gaps import (
    GAP_MANIFEST_NAME,
    GAP_PROGRESS_NAME,
    ROUTE_GAPS_NAME,
    plan_route_gaps,
)


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def single_case(tmp_path: Path, name: str, decisions: list[dict]) -> dict:
    case_dir = tmp_path / "shadow" / "single_cases" / name
    manifest_path = write_json(
        case_dir / "generic_finetuning_controller_selector_integration_manifest.json",
        {
            "outputs": {
                "integration_decisions_jsonl": str(
                    case_dir / "generic_finetuning_controller_selector_integration_decisions.jsonl"
                )
            }
        },
    )
    write_jsonl(case_dir / "generic_finetuning_controller_selector_integration_decisions.jsonl", decisions)
    return {
        "judge_jsonl": f"/app/fake/{name}.jsonl",
        "single_manifest_json": str(manifest_path),
        "single_promotion_status": "controller_selector_integration_staged",
        "controller_checked_candidates": len(decisions),
        "controller_allowed_candidates": sum(1 for row in decisions if row["controller_action"].startswith("allow")),
        "controller_blocked_candidates": sum(1 for row in decisions if row["controller_action"] == "controller_block_selector_candidate"),
        "controller_missing_routes": sum(1 for row in decisions if row["controller_missing_route"]),
        "controller_reranked_candidates": sum(1 for row in decisions if row.get("controller_rerank_applied")),
    }


def decision(
    route_id: str,
    action: str,
    *,
    stem_type: str = "minus",
    posterior: float | None = 0.1,
    match_kind: str = "exact",
    evaluated: list[dict] | None = None,
) -> dict:
    return {
        "track_id": f"track-{route_id}",
        "stem_type": stem_type,
        "selected_candidate_id": route_id,
        "selector_original_selected_candidate_id": route_id,
        "controller_action": action,
        "controller_missing_route": action == "manual_review_missing_controller_route",
        "controller_route_match_kind": match_kind,
        "posterior_accept_rate": posterior,
        "controller_rerank_applied": False,
        "controller_rerank_evaluated_candidates": evaluated or [
            {
                "candidate_id": route_id,
                "stem_type": stem_type,
                "controller_action": action,
                "controller_route_match_kind": match_kind,
                "posterior_accept_rate": posterior,
                "selection_score": 0.9,
                "confidence": 0.8,
            }
        ],
    }


def test_plans_next_wave_for_repeated_blocked_and_missing_routes(tmp_path: Path) -> None:
    items = [
        single_case(tmp_path, "case-a", [decision("bad-route", "controller_block_selector_candidate")]),
        single_case(tmp_path, "case-b", [decision("bad-route", "controller_block_selector_candidate")]),
        single_case(
            tmp_path,
            "case-c",
            [decision("unknown-route", "manual_review_missing_controller_route", posterior=None, match_kind="missing")],
        ),
        single_case(tmp_path, "case-d", [decision("good-route", "allow_selector_candidate", posterior=0.75)]),
    ]
    items_jsonl = write_jsonl(tmp_path / "shadow" / "generic_controller_selector_shadow_batch_items.jsonl", items)
    shadow_manifest = write_json(
        tmp_path / "shadow" / "generic_controller_selector_shadow_batch_manifest.json",
        {
            "outputs": {"items_jsonl": str(items_jsonl)},
            "batch_summary": {
                "input_count": 4,
                "controller_allowed_candidates": 1,
                "controller_blocked_candidates": 2,
                "controller_missing_routes": 1,
            },
        },
    )

    manifest = plan_route_gaps(
        shadow_batch_manifest_json=shadow_manifest,
        output_dir=tmp_path / "gap_plan",
        min_repeated_blocked=2,
    )

    gaps = read_jsonl(tmp_path / "gap_plan" / ROUTE_GAPS_NAME)
    by_route = {row["route_id"]: row for row in gaps}

    assert by_route["bad-route"]["recommended_action"] == "generate_new_candidate_wave"
    assert by_route["bad-route"]["priority"] == "P0"
    assert by_route["bad-route"]["blocked_count"] == 2
    assert by_route["bad-route"]["next_wave_required"] is True

    assert by_route["unknown-route"]["recommended_action"] == "collect_controller_prior_evidence"
    assert by_route["unknown-route"]["missing_count"] == 1
    assert by_route["unknown-route"]["next_wave_required"] is False

    assert by_route["good-route"]["recommended_action"] == "keep_candidate_route_available"
    assert by_route["good-route"]["allowed_count"] == 1

    assert manifest["summary"]["route_count"] == 3
    assert manifest["summary"]["repeated_blocked_route_count"] == 1
    assert manifest["summary"]["missing_route_count"] == 1
    assert manifest["promotion"]["status"] == "candidate_wave_required"
    assert manifest["promotion"]["can_start_training"] is False
    assert manifest["release_ready"] is False
    assert read_json(tmp_path / "gap_plan" / GAP_MANIFEST_NAME)["fingerprint"] == manifest["fingerprint"]
    assert [row["event"] for row in read_jsonl(tmp_path / "gap_plan" / GAP_PROGRESS_NAME)] == [
        "start",
        "write_outputs",
        "finish",
    ]
