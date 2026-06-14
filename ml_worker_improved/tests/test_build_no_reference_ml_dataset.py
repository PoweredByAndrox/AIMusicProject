from __future__ import annotations

import json
from pathlib import Path

from tools.build_no_reference_ml_dataset import build_dataset, main


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def judge_row(track_id: str, candidate_id: str, *, mode: str | None, decision: str, score: float) -> dict:
    row = {
        "track_id": track_id,
        "candidate_id": candidate_id,
        "mix_path": f"/missing/{track_id}/mix.wav",
        "vocal_path": f"/missing/{track_id}/{candidate_id}_vocals.wav",
        "instrumental_path": f"/missing/{track_id}/{candidate_id}_minus.wav",
        "judge_mode": "no_reference_windowed",
        "decision": decision,
        "overall_no_reference_score": score,
        "vocal_score": score if mode != "minus" else 0.51,
        "minus_score": score if mode != "vocal" else 0.49,
        "pair_compatibility_score": score,
        "confidence": score,
        "mixture_consistency_score": score,
        "phase_consistency_score": score,
        "vocal_leakage_in_minus_score": 1.0 - score,
        "instrumental_leakage_in_vocal_score": 1.0 - score,
        "silence_hallucination_score": 1.0 - score,
        "robotic_artifact_score": 1.0 - score,
        "thin_vocal_score": 1.0 - score,
        "transient_damage_score": 1.0 - score,
        "vocal_activity_confidence": score,
        "window_count": 1,
        "window_metrics": [
            {
                "kind": "start",
                "start_sec": 0.0,
                "duration_sec": 10.0,
                "window_fitness": score,
                "vocal_score": score,
                "minus_score": score,
                "pair_compatibility_score": score,
                "decision": decision,
            }
        ],
        "forbidden_actions": ["do_not_train_separator"],
    }
    if mode is not None:
        row["score_mode"] = mode
    return row


def test_dataset_build_is_metadata_only_and_splits_by_track(tmp_path: Path) -> None:
    candidate_pool = tmp_path / "candidate_pool.jsonl"
    windowed = tmp_path / "windowed.jsonl"
    optimizer = tmp_path / "optimizer.jsonl"
    segment_plan = tmp_path / "segment_plan.jsonl"
    preview = tmp_path / "preview.jsonl"
    output_dir = tmp_path / "dataset"

    write_jsonl(
        candidate_pool,
        [
            {"track_id": "track-a", "candidate_id": "cand-vocal", "is_complete_pair": True},
            {"track_id": "track-b", "candidate_id": "cand-minus", "is_complete_pair": True},
            {"track_id": "track-c", "candidate_id": "cand-both", "is_complete_pair": True},
            {"track_id": "track-d", "candidate_id": "legacy", "is_complete_pair": True},
        ],
    )
    write_jsonl(
        windowed,
        [
            judge_row("track-a", "cand-vocal", mode="vocal", decision="accept_candidate", score=0.91),
            judge_row("track-b", "cand-minus", mode="minus", decision="blocked_hard_case", score=0.22),
            judge_row("track-c", "cand-both", mode="both", decision="accept_candidate", score=0.88),
            judge_row("track-d", "legacy", mode=None, decision="blocked_hard_case", score=0.31),
            {**judge_row("track-e", "ambiguous", mode="vocal", decision="manual_review", score=0.61)},
        ],
    )
    write_jsonl(
        optimizer,
        [
            {
                "track_id": "track-a",
                "input_candidate_id": "cand-vocal",
                "selected_candidate_id": "cand-vocal",
                "final_decision": "original_candidate_kept",
                "fitness_before": 0.8,
                "fitness_after": 0.8,
                "improvement_delta": 0.0,
            }
        ],
    )
    write_jsonl(
        segment_plan,
        [
            {
                "track_id": "track-b",
                "source_stem": "minus",
                "base_candidate_id": "base",
                "alternate_candidate_id": "alt",
                "base_window_fitness": 0.35,
                "alternate_window_fitness": 0.73,
                "improvement_delta": 0.38,
                "segment_blend_action": "use_alt_minus_window",
                "blocker_check_passed": True,
                "can_execute_blend_now": False,
            }
        ],
    )
    write_jsonl(
        preview,
        [
            {
                "track_id": "track-c",
                "source_stem": "both",
                "base_candidate_id": "base",
                "alternate_candidate_id": "alt",
                "base_window_score": 0.44,
                "preview_window_score": 0.77,
                "score_delta": 0.33,
                "blocker_delta": 0,
                "preview_decision": "preview_improved_window",
                "can_publish_as_result": False,
                "can_mark_release_ready": False,
            }
        ],
    )

    exit_code = main(
        [
            "--candidate-pool-jsonl",
            str(candidate_pool),
            "--windowed-judge-jsonl",
            str(windowed),
            "--optimizer-result-jsonl",
            str(optimizer),
            "--segment-blend-plan-jsonl",
            str(segment_plan),
            "--segment-blend-preview-window-judge-jsonl",
            str(preview),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    examples_path = output_dir / "no_reference_ml_examples.jsonl"
    pairwise_path = output_dir / "no_reference_pairwise_examples.jsonl"
    manifest_path = output_dir / "no_reference_ml_dataset_manifest.json"
    assert examples_path.exists()
    assert pairwise_path.exists()
    assert manifest_path.exists()

    examples = [json.loads(line) for line in examples_path.read_text(encoding="utf-8").splitlines()]
    pairwise = [json.loads(line) for line in pairwise_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert {row["mode"] for row in examples} >= {"vocal", "minus", "both", "window"}
    assert any(row["mode"] == "vocal" and row["label"] == 1 for row in examples)
    assert any(row["mode"] == "minus" and row["label"] == 0 for row in examples)
    assert all(row["release_ready"] is False for row in examples)
    assert all(row["training_started"] is False for row in examples)
    assert all(row["separator_training_started"] is False for row in examples)
    assert manifest["skipped_ambiguous_examples"] >= 1
    assert manifest["audio_read_performed"] is False
    assert manifest["dataset_fingerprint"]
    assert pairwise

    track_splits: dict[str, set[str]] = {}
    for row in examples:
        track_splits.setdefault(row["track_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in track_splits.values())


def test_build_dataset_marks_legacy_missing_score_mode_as_weak() -> None:
    result = build_dataset(
        candidate_pool_rows=[],
        windowed_judge_rows=[judge_row("legacy-track", "legacy-candidate", mode=None, decision="accept_candidate", score=0.82)],
        optimizer_result_rows=[],
        segment_blend_plan_rows=[],
        preview_window_judge_rows=[],
    )

    legacy_rows = [
        row
        for row in result["examples"]
        if row["track_id"] == "legacy-track" and row["source_artifact_type"] == "windowed_judge"
    ]
    assert legacy_rows
    assert any(row["mode"] == "window" for row in legacy_rows)
    assert any(row["mode"] in {"vocal", "minus"} and row["strict_mode_label"] is False for row in legacy_rows)
    assert result["manifest"]["legacy_missing_score_mode_rows"] == 1
