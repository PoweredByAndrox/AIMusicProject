from __future__ import annotations

import json
from pathlib import Path

from tools.run_generic_controller_selector_shadow_batch import (
    BATCH_ITEMS_NAME,
    BATCH_MANIFEST_NAME,
    BATCH_PROGRESS_NAME,
    discover_judge_jsonls,
    run_shadow_batch,
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


def judge_row(candidate_id: str, *, track_id: str, score: float = 0.86) -> dict:
    return {
        "track_id": track_id,
        "candidate_id": candidate_id,
        "mix_path": f"/tmp/{track_id}/mix.wav",
        "instrumental_path": f"/tmp/{track_id}/{candidate_id}_minus.wav",
        "judge_mode": "no_reference",
        "score_mode": "minus",
        "overall_no_reference_score": score,
        "minus_score": score,
        "minus_decision": "accept_candidate",
        "minus_blockers": [],
        "confidence": 0.88,
        "decision": "accept_candidate",
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


def test_shadow_batch_aggregates_allow_block_and_missing_routes(tmp_path: Path) -> None:
    training_manifest = trained_controller(tmp_path)
    good_judge = write_jsonl(tmp_path / "judges" / "good" / "no_reference_minus_windowed_judge_results.jsonl", [judge_row("good-route", track_id="track-good")])
    rerank_judge = write_jsonl(
        tmp_path / "judges" / "rerank" / "no_reference_minus_windowed_judge_results.jsonl",
        [
            judge_row("bad-route", track_id="track-rerank", score=0.95),
            judge_row("good-route", track_id="track-rerank", score=0.90),
        ],
    )
    bad_judge = write_jsonl(tmp_path / "judges" / "bad" / "no_reference_minus_windowed_judge_results.jsonl", [judge_row("bad-route", track_id="track-bad")])
    missing_judge = write_jsonl(tmp_path / "judges" / "missing" / "no_reference_minus_windowed_judge_results.jsonl", [judge_row("unknown-route", track_id="track-missing")])

    manifest = run_shadow_batch(
        controller_model_json=Path(training_manifest["outputs"]["model_json"]),
        judge_jsonls=[good_judge, rerank_judge, bad_judge, missing_judge],
        output_dir=tmp_path / "shadow_batch",
        score_mode="minus",
        min_selector_usable_per_input=1,
        min_batch_checked_candidates=4,
    )

    assert manifest["batch_summary"]["input_count"] == 4
    assert manifest["batch_summary"]["controller_allowed_candidates"] == 2
    assert manifest["batch_summary"]["controller_reranked_candidates"] == 1
    assert manifest["batch_summary"]["controller_blocked_candidates"] == 1
    assert manifest["batch_summary"]["controller_missing_routes"] == 1
    assert manifest["promotion"]["status"] == "blocked_missing_controller_routes"
    assert manifest["promotion"]["can_stage_controller_candidate"] is False
    assert manifest["promotion"]["can_edit_model_registry"] is False
    assert manifest["release_ready"] is False

    items = read_jsonl(tmp_path / "shadow_batch" / BATCH_ITEMS_NAME)
    assert len(items) == 4
    assert sum(item["controller_reranked_candidates"] for item in items) == 1
    assert {item["single_promotion_status"] for item in items} == {
        "controller_selector_integration_staged",
        "blocked_missing_controller_routes",
    }
    assert all(Path(item["single_manifest_json"]).exists() for item in items)

    progress_rows = read_jsonl(tmp_path / "shadow_batch" / BATCH_PROGRESS_NAME)
    assert [row["event"] for row in progress_rows] == [
        "start",
        "input_start",
        "input_finish",
        "input_start",
        "input_finish",
        "input_start",
        "input_finish",
        "input_start",
        "input_finish",
        "write_outputs",
        "finish",
    ]

    disk_manifest = read_json(tmp_path / "shadow_batch" / BATCH_MANIFEST_NAME)
    assert disk_manifest["fingerprint"] == manifest["fingerprint"]


def test_discovers_judge_jsonls_under_root_with_glob(tmp_path: Path) -> None:
    first = write_jsonl(tmp_path / "root" / "a" / "no_reference_minus_windowed_judge_results.jsonl", [judge_row("good-route", track_id="track-a")])
    second = write_jsonl(tmp_path / "root" / "b" / "no_reference_vocal_windowed_judge_results.jsonl", [judge_row("good-route", track_id="track-b")])
    write_jsonl(tmp_path / "root" / "ignored" / "other.jsonl", [judge_row("good-route", track_id="track-ignored")])

    assert discover_judge_jsonls(tmp_path / "root", "**/no_reference_*_windowed_judge_results.jsonl") == [
        first,
        second,
    ]
