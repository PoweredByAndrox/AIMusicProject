import json
from pathlib import Path


VIEW_FILES = {
    "minus_critic": "minus_critic_training_view.jsonl",
    "vocal_critic": "vocal_critic_training_view.jsonl",
    "pair_critic": "pair_critic_training_view.jsonl",
    "overall_accept": "overall_accept_training_view.jsonl",
}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def base_view_row(index: int, *, label: int = 1, view_name: str = "minus_critic", release_ready: bool = False) -> dict:
    return {
        "view_name": view_name,
        "target_name": view_name.replace("_critic", ""),
        "row_id": f"row-{index}",
        "track_id": f"track-{index // 10}",
        "run_id": f"run-{index}",
        "candidate_id": f"cand-{index}",
        "route_id": "route-a",
        "stem_type": "minus" if view_name == "minus_critic" else "vocal",
        "window_start_sec": float(index),
        "window_end_sec": float(index + 1),
        "candidate_audio_path": f"/tmp/candidate-{index}.wav",
        "mix_path": f"/tmp/mix-{index}.wav",
        "error_map_path": f"/tmp/map-{index}.npz",
        "feature_values": {"phase_collapse_score": 0.1, "vocal_bleed_proxy_score": 0.2},
        "label": label,
        "label_source": "minus_windowed_judge.jsonl",
        "blocker_labels": [] if label == 1 else ["severe_vocal_bleed_in_minus"],
        "provenance_source_files": ["minus_windowed_judge.jsonl"],
        "safe_for_this_view": True,
        "unsafe_reason_for_this_view": "",
        "release_ready": release_ready,
    }


def write_views(views_dir: Path, rows_by_view: dict[str, list[dict]]) -> None:
    summary = {"views": {}, "release_ready": False}
    for view_name, filename in VIEW_FILES.items():
        rows = rows_by_view.get(view_name, [])
        write_jsonl(views_dir / filename, rows)
        safe = [row for row in rows if row.get("safe_for_this_view") is True]
        summary["views"][view_name] = {
            "rows_total": len(rows),
            "safe_rows": len(safe),
            "positive_labels": sum(1 for row in safe if row.get("label") == 1),
            "negative_labels": sum(1 for row in safe if row.get("label") == 0),
        }
    write_json(views_dir / "audio_error_training_views_manifest.json", summary)


def test_validator_detects_missing_negative_class(tmp_path: Path) -> None:
    from tools.validate_audio_error_training_views import validate_audio_error_training_views

    rows = [base_view_row(index, label=1) for index in range(100)]
    write_views(tmp_path / "views", {"minus_critic": rows})

    result = validate_audio_error_training_views(tmp_path / "views")

    assert result["ok"] is True
    assert result["views"]["minus_critic"]["negative_labels"] == 0
    assert result["can_train_minus_critic"] is False
    assert "missing_negative_class" in result["views"]["minus_critic"]["readiness_blockers"]


def test_validator_marks_minus_trainable_only_when_both_classes_exist(tmp_path: Path) -> None:
    from tools.validate_audio_error_training_views import validate_audio_error_training_views

    rows = [base_view_row(index, label=1) for index in range(80)]
    rows.extend(base_view_row(index + 80, label=0) for index in range(20))
    write_views(tmp_path / "views", {"minus_critic": rows})

    result = validate_audio_error_training_views(tmp_path / "views")

    assert result["ok"] is True
    assert result["can_train_minus_critic"] is True
    assert result["views"]["minus_critic"]["positive_labels"] == 80
    assert result["views"]["minus_critic"]["negative_labels"] == 20


def test_validator_rejects_release_ready_true(tmp_path: Path) -> None:
    from tools.validate_audio_error_training_views import validate_audio_error_training_views

    rows = [base_view_row(1, label=1, release_ready=True)]
    write_views(tmp_path / "views", {"minus_critic": rows})

    result = validate_audio_error_training_views(tmp_path / "views")

    assert result["ok"] is False
    assert any("release_ready_must_be_false" in error for error in result["errors"])


def test_validator_rejects_safe_row_with_missing_provenance(tmp_path: Path) -> None:
    from tools.validate_audio_error_training_views import validate_audio_error_training_views

    row = base_view_row(1, label=1)
    row["candidate_id"] = ""
    write_views(tmp_path / "views", {"minus_critic": [row]})

    result = validate_audio_error_training_views(tmp_path / "views")

    assert result["ok"] is False
    assert any("safe_row_missing_candidate_id" in error for error in result["errors"])


def test_validator_rejects_safe_row_without_acoustic_feature_values(tmp_path: Path) -> None:
    from tools.validate_audio_error_training_views import validate_audio_error_training_views

    row = base_view_row(2, label=0)
    row["error_map_path"] = ""
    row["feature_values"] = {
        "window_start_sec": 2.0,
        "window_end_sec": 3.0,
        "route_id": "route-a",
        "candidate_family": "test",
        "stem_type": "minus",
    }
    write_views(tmp_path / "views", {"minus_critic": [row]})

    result = validate_audio_error_training_views(tmp_path / "views")

    assert result["ok"] is False
    assert any("safe_row_missing_acoustic_feature_values" in error for error in result["errors"])
