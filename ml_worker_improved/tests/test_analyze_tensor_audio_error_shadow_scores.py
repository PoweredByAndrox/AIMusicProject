import json
from pathlib import Path

import pytest


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_row(index: int, *, label: int, probability: float, decision: str, stem: str = "minus", route: str = "route-a") -> dict:
    return {
        "row_uid": f"row-{index}",
        "canonical_row_key": f"track-{index}|cand-{index}|{stem}",
        "track_id": f"track-{index}",
        "run_id": "run-1",
        "candidate_id": f"cand-{index}",
        "route_id": route,
        "candidate_family": "family-a",
        "source_model_family": "model-a",
        "stem_type": stem,
        "split": "test",
        "true_label": label,
        "window_start_sec": float(index * 12),
        "window_end_sec": float(index * 12 + 12),
        "predicted_accept_probability": probability,
        "calibrated_threshold": 0.5,
        "threshold_scope": f"stem:{stem}",
        "calibration_status": "calibrated",
        "shadow_decision": decision,
        "would_accept_at_calibrated_threshold": decision == "shadow_accept_candidate",
        "runtime_gate_decision_applied": False,
        "model_kind": "test_model",
        "training_started": False,
        "critic_training_started": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "thresholds_relaxed": False,
        "runtime_gates_changed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "shadow_only": True,
    }


def write_scores_dir(tmp_path: Path, rows: list[dict], *, release_ready: bool = False) -> Path:
    scores_dir = tmp_path / "scores"
    write_jsonl(scores_dir / "tensor_audio_error_shadow_scores.jsonl", rows)
    write_json(
        scores_dir / "tensor_audio_error_shadow_score_summary.json",
        {
            "schema": "tensor_audio_error_shadow_scores.v1",
            "rows_scored": len(rows),
            "calibration_status": "calibrated",
            "model_kind": "test_model",
            "runtime_gates_changed": False,
            "release_ready": release_ready,
            "shadow_only": True,
            "separator_training_started": False,
            "repair_adapter_training_started": False,
        },
    )
    return scores_dir


def test_analyzer_writes_hard_negative_report(tmp_path: Path) -> None:
    from tools.analyze_tensor_audio_error_shadow_scores import analyze_tensor_audio_error_shadow_scores

    scores_dir = write_scores_dir(
        tmp_path,
        [
            score_row(0, label=0, probability=0.91, decision="shadow_accept_candidate", stem="minus"),
            score_row(1, label=1, probability=0.12, decision="shadow_block_candidate", stem="vocal", route="route-b"),
            score_row(2, label=1, probability=0.88, decision="shadow_accept_candidate", stem="minus"),
            score_row(3, label=0, probability=0.20, decision="shadow_block_candidate", stem="vocal"),
        ],
    )

    summary = analyze_tensor_audio_error_shadow_scores(scores_dir=scores_dir, output_dir=tmp_path / "analysis", max_hard_negative_rows=5)

    hard_negatives = read_jsonl(tmp_path / "analysis" / "tensor_audio_error_hard_negative_candidates.jsonl")
    report = (tmp_path / "analysis" / "tensor_audio_error_model_improvement_report.md").read_text(encoding="utf-8")
    persisted_summary = read_json(tmp_path / "analysis" / "tensor_audio_error_shadow_analysis_summary.json")
    assert summary["rows_analyzed"] == 4
    assert summary["hard_negative_count"] == 1
    assert summary["false_reject_count"] == 1
    assert summary["release_ready"] is False
    assert summary["training_started"] is False
    assert summary["runtime_gates_changed"] is False
    assert persisted_summary == summary
    assert hard_negatives[0]["candidate_id"] == "cand-0"
    assert hard_negatives[0]["mining_label"] == "hard_negative_false_accept"
    assert "Hard negative candidates: 1" in report


def test_analyzer_rejects_forbidden_runtime_or_release_flags(tmp_path: Path) -> None:
    from tools.analyze_tensor_audio_error_shadow_scores import analyze_tensor_audio_error_shadow_scores

    bad_row = score_row(0, label=0, probability=0.91, decision="shadow_accept_candidate")
    bad_row["runtime_gate_decision_applied"] = True
    scores_dir = write_scores_dir(tmp_path, [bad_row], release_ready=True)

    with pytest.raises(ValueError, match="unsafe_shadow_scores"):
        analyze_tensor_audio_error_shadow_scores(scores_dir=scores_dir, output_dir=tmp_path / "analysis")


def test_analyzer_rejects_summary_row_count_mismatch(tmp_path: Path) -> None:
    from tools.analyze_tensor_audio_error_shadow_scores import analyze_tensor_audio_error_shadow_scores

    scores_dir = write_scores_dir(tmp_path, [score_row(0, label=0, probability=0.91, decision="shadow_accept_candidate")])
    summary_path = scores_dir / "tensor_audio_error_shadow_score_summary.json"
    summary = read_json(summary_path)
    summary["rows_scored"] = 2
    write_json(summary_path, summary)

    with pytest.raises(ValueError, match="summary_rows_scored_mismatch"):
        analyze_tensor_audio_error_shadow_scores(scores_dir=scores_dir, output_dir=tmp_path / "analysis")


def test_analyzer_rejects_negative_hard_negative_limit(tmp_path: Path) -> None:
    from tools.analyze_tensor_audio_error_shadow_scores import analyze_tensor_audio_error_shadow_scores

    scores_dir = write_scores_dir(tmp_path, [score_row(0, label=0, probability=0.91, decision="shadow_accept_candidate")])

    with pytest.raises(ValueError, match="max_hard_negative_rows_must_be_non_negative"):
        analyze_tensor_audio_error_shadow_scores(scores_dir=scores_dir, output_dir=tmp_path / "analysis", max_hard_negative_rows=-1)
