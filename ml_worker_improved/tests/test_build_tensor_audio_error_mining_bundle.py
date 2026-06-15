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


def candidate_row(index: int, *, mining_label: str) -> dict:
    true_label = 0 if mining_label == "hard_negative_false_accept" else 1
    return {
        "mining_label": mining_label,
        "row_uid": f"row-{index}",
        "canonical_row_key": f"track-{index}|cand-{index}|minus",
        "track_id": f"track-{index}",
        "run_id": "run-1",
        "candidate_id": f"cand-{index}",
        "route_id": "route-a",
        "candidate_family": "family-a",
        "source_model_family": "model-a",
        "stem_type": "minus",
        "split": "test",
        "true_label": true_label,
        "window_start_sec": float(index * 12),
        "window_end_sec": float(index * 12 + 12),
        "predicted_accept_probability": 0.9 if true_label == 0 else 0.1,
        "calibrated_threshold": 0.5,
        "threshold_scope": "stem:minus",
        "shadow_decision": "shadow_accept_candidate" if true_label == 0 else "shadow_block_candidate",
        "runtime_gate_decision_applied": False,
        "release_ready": False,
        "shadow_only": True,
    }


def write_analysis_dir(
    tmp_path: Path,
    *,
    release_ready: bool = False,
    summary_overrides: dict | None = None,
    hard_negative_overrides: dict | None = None,
) -> Path:
    analysis_dir = tmp_path / "analysis"
    hard_negative = candidate_row(0, mining_label="hard_negative_false_accept")
    hard_negative.update(hard_negative_overrides or {})
    hard_negatives = [hard_negative]
    false_rejects = [candidate_row(1, mining_label="false_reject_positive")]
    write_jsonl(analysis_dir / "tensor_audio_error_hard_negative_candidates.jsonl", hard_negatives)
    write_jsonl(analysis_dir / "tensor_audio_error_false_reject_candidates.jsonl", false_rejects)
    summary = {
        "schema": "tensor_audio_error_shadow_analysis.v1",
        "scores_dir": str(tmp_path / "scores"),
        "rows_analyzed": 4,
        "hard_negative_count": len(hard_negatives),
        "hard_negative_rows_written": len(hard_negatives),
        "false_reject_count": len(false_rejects),
        "calibration_status": "calibrated",
        "model_kind": "test_model",
        "release_ready": release_ready,
        "runtime_gates_changed": False,
        "training_started": False,
        "critic_training_started": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
        "shadow_only": True,
    }
    summary.update(summary_overrides or {})
    write_json(analysis_dir / "tensor_audio_error_shadow_analysis_summary.json", summary)
    return analysis_dir


def test_build_mining_bundle_writes_candidate_only_dataset(tmp_path: Path) -> None:
    from tools.build_tensor_audio_error_mining_bundle import build_tensor_audio_error_mining_bundle

    analysis_dir = write_analysis_dir(tmp_path)

    manifest = build_tensor_audio_error_mining_bundle(analysis_dir=analysis_dir, output_dir=tmp_path / "bundle")

    rows = read_jsonl(tmp_path / "bundle" / "tensor_audio_error_training_candidates.jsonl")
    persisted = read_json(tmp_path / "bundle" / "tensor_audio_error_mining_bundle_manifest.json")
    report = (tmp_path / "bundle" / "tensor_audio_error_mining_bundle_summary.md").read_text(encoding="utf-8")
    assert manifest == persisted
    assert manifest["schema"] == "tensor_audio_error_mining_bundle.v1"
    assert manifest["bundle_status"] == "candidate_only"
    assert manifest["training_started"] is False
    assert manifest["release_ready"] is False
    assert manifest["runtime_gates_changed"] is False
    assert manifest["candidate_rows_total"] == 2
    assert manifest["hard_negative_rows"] == 1
    assert manifest["false_reject_rows"] == 1
    assert {row["recommended_training_label"] for row in rows} == {0, 1}
    assert {row["training_started"] for row in rows} == {False}
    assert {row["release_ready"] for row in rows} == {False}
    assert {row["runtime_gate_decision_applied"] for row in rows} == {False}
    assert "Candidate rows: 2" in report


def test_build_mining_bundle_rejects_unsafe_analysis(tmp_path: Path) -> None:
    from tools.build_tensor_audio_error_mining_bundle import build_tensor_audio_error_mining_bundle

    analysis_dir = write_analysis_dir(tmp_path, release_ready=True)

    with pytest.raises(ValueError, match="unsafe_shadow_analysis"):
        build_tensor_audio_error_mining_bundle(analysis_dir=analysis_dir, output_dir=tmp_path / "bundle")


def test_build_mining_bundle_rejects_critic_training_started(tmp_path: Path) -> None:
    from tools.build_tensor_audio_error_mining_bundle import build_tensor_audio_error_mining_bundle

    analysis_dir = write_analysis_dir(tmp_path, summary_overrides={"critic_training_started": True})

    with pytest.raises(ValueError, match="analysis_forbidden_true_flag"):
        build_tensor_audio_error_mining_bundle(analysis_dir=analysis_dir, output_dir=tmp_path / "bundle")


def test_build_mining_bundle_rejects_hard_negative_semantic_mismatch(tmp_path: Path) -> None:
    from tools.build_tensor_audio_error_mining_bundle import build_tensor_audio_error_mining_bundle

    analysis_dir = write_analysis_dir(tmp_path, hard_negative_overrides={"true_label": 1})

    with pytest.raises(ValueError, match="hard_negative_row_invalid_true_label"):
        build_tensor_audio_error_mining_bundle(analysis_dir=analysis_dir, output_dir=tmp_path / "bundle")
