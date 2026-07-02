import json
from pathlib import Path


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


def target_view_row(
    tmp_path: Path,
    target_name: str,
    index: int,
    *,
    label: int = 1,
    safe: bool = True,
    candidate_id: str | None = None,
    route_id: str | None = None,
    label_source: str | None = None,
    release_ready: bool = False,
    blockers: list[str] | None = None,
) -> dict:
    stem_type = "minus" if target_name == "minus_critic" else "vocal"
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = audio_dir / f"{target_name}_{index}_candidate.wav"
    mix_path = audio_dir / f"{target_name}_{index}_mix.wav"
    candidate_path.write_bytes(b"candidate")
    mix_path.write_bytes(b"mix")
    return {
        "view_name": target_name,
        "target_name": stem_type,
        "row_id": f"{target_name}-row-{index}",
        "track_id": f"track-{index % 8}",
        "run_id": f"run-{index % 3}",
        "candidate_id": candidate_id if candidate_id is not None else f"cand-{target_name}-{index}",
        "route_id": route_id if route_id is not None else f"route-{index % 4}",
        "stem_type": stem_type,
        "window_start_sec": float(index),
        "window_end_sec": float(index + 1),
        "candidate_audio_path": str(candidate_path),
        "mix_path": str(mix_path),
        "error_map_path": "",
        "feature_values": {
            "residual_energy": 0.01 + index * 0.001,
            "mixture_consistency_error": 0.02 + index * 0.001,
            "phase_collapse_score": 0.1 if label == 1 else 0.8,
            "stereo_width_score": 0.3,
            "vocal_bleed_proxy_score": 0.1 if label == 1 else 0.9,
            "instrumental_bleed_proxy_score": 0.2,
            "missing_detail_score": 0.1 if label == 1 else 0.7,
            "transient_damage_score": 0.05,
            "artifact_noise_score": 0.07,
            "window_start_sec": float(index),
            "window_end_sec": float(index + 1),
            "route_id": route_id if route_id is not None else f"route-{index % 4}",
            "candidate_family": "test_family",
            "stem_type": stem_type,
        },
        "label": label if safe else None,
        "label_source": label_source or f"{stem_type}_windowed_judge.jsonl",
        "blocker_labels": blockers if blockers is not None else ([] if label == 1 else [f"severe_{stem_type}_artifact"]),
        "provenance_source_files": [f"{stem_type}_windowed_judge.jsonl"],
        "safe_for_this_view": safe,
        "unsafe_reason_for_this_view": "" if safe else "manual_unsafe",
        "release_ready": release_ready,
    }


def write_views(views_dir: Path, minus_rows: list[dict], vocal_rows: list[dict], *, pair_rows: list[dict] | None = None, overall_rows: list[dict] | None = None) -> None:
    pair_rows = pair_rows or []
    overall_rows = overall_rows or []
    files = {
        "minus_critic_training_view.jsonl": minus_rows,
        "vocal_critic_training_view.jsonl": vocal_rows,
        "pair_critic_training_view.jsonl": pair_rows,
        "overall_accept_training_view.jsonl": overall_rows,
    }
    for filename, rows in files.items():
        write_jsonl(views_dir / filename, rows)
    write_json(
        views_dir / "audio_error_training_views_manifest.json",
        {
            "schema": "audio_error_training_views.v1",
            "views": {
                "minus_critic": {"rows_total": len(minus_rows), "safe_rows": sum(1 for row in minus_rows if row.get("safe_for_this_view") is True)},
                "vocal_critic": {"rows_total": len(vocal_rows), "safe_rows": sum(1 for row in vocal_rows if row.get("safe_for_this_view") is True)},
                "pair_critic": {"rows_total": len(pair_rows), "safe_rows": sum(1 for row in pair_rows if row.get("safe_for_this_view") is True)},
                "overall_accept": {"rows_total": len(overall_rows), "safe_rows": sum(1 for row in overall_rows if row.get("safe_for_this_view") is True)},
            },
            "release_ready": False,
            "training_started": False,
            "separator_training_started": False,
            "melody_transcription_ran": False,
            "recommendation_index_updated": False,
        },
    )
    (views_dir / "audio_error_training_views_summary.md").write_text("# test views\n", encoding="utf-8")


def trainable_rows(tmp_path: Path, target_name: str) -> list[dict]:
    rows: list[dict] = []
    for index in range(8):
        rows.append(target_view_row(tmp_path, target_name, index, label=1 if index % 2 == 0 else 0))
    return rows


def test_trainer_uses_minus_and_vocal_views_without_pair_or_overall(tmp_path: Path) -> None:
    from tools.train_target_audio_error_critics import train_target_audio_error_critics

    views_dir = tmp_path / "views"
    poison_pair = [target_view_row(tmp_path, "minus_critic", 90, label=0)]
    poison_overall = [target_view_row(tmp_path, "vocal_critic", 91, label=0)]
    write_views(views_dir, trainable_rows(tmp_path, "minus_critic"), trainable_rows(tmp_path, "vocal_critic"), pair_rows=poison_pair, overall_rows=poison_overall)
    (views_dir / "pair_critic_training_view.jsonl").write_text("{not-json}\n", encoding="utf-8")
    (views_dir / "overall_accept_training_view.jsonl").write_text("{not-json}\n", encoding="utf-8")

    manifest = train_target_audio_error_critics(
        views_dir=views_dir,
        output_dir=tmp_path / "model",
        targets=("minus", "vocal"),
        model_kind="random_forest",
        group_split="track_or_candidate",
        shadow_only=True,
        min_safe_rows=4,
        min_positive=1,
        min_negative=1,
    )

    assert manifest["minus_rows_used"] == 8
    assert manifest["vocal_rows_used"] == 8
    assert manifest["pair_rows_used"] is False
    assert manifest["overall_rows_used"] is False
    assert manifest["unsafe_rows_used"] is False
    assert manifest["shadow_only"] is True
    assert manifest["minus_critic_training_completed"] is True
    assert manifest["vocal_critic_training_completed"] is True
    assert (tmp_path / "model" / "minus_audio_error_critic.joblib").exists()
    assert (tmp_path / "model" / "vocal_audio_error_critic.joblib").exists()


def test_trainer_rejects_unsafe_unknown_route_and_synthetic_rows(tmp_path: Path) -> None:
    from tools.train_target_audio_error_critics import train_target_audio_error_critics

    rows = trainable_rows(tmp_path, "minus_critic")
    rows.extend(
        [
            target_view_row(tmp_path, "minus_critic", 20, safe=False),
            target_view_row(tmp_path, "minus_critic", 21, route_id="unknown"),
            target_view_row(tmp_path, "minus_critic", 22, candidate_id="synthetic_candidate_001"),
        ]
    )
    views_dir = tmp_path / "views"
    write_views(views_dir, rows, trainable_rows(tmp_path, "vocal_critic"))

    manifest = train_target_audio_error_critics(
        views_dir=views_dir,
        output_dir=tmp_path / "model",
        targets=("minus", "vocal"),
        model_kind="random_forest",
        group_split="track_or_candidate",
        shadow_only=True,
        min_safe_rows=4,
        min_positive=1,
        min_negative=1,
    )

    assert manifest["minus_rows_available"] == 11
    assert manifest["minus_rows_used"] == 8
    assert manifest["excluded_reason_counts"]["minus_critic"]["safe_for_this_view_false"] == 1
    assert manifest["excluded_reason_counts"]["minus_critic"]["route_id_unknown"] == 1
    assert manifest["excluded_reason_counts"]["minus_critic"]["synthetic_candidate_id"] == 1


def test_trainer_rejects_rows_without_acoustic_feature_values(tmp_path: Path) -> None:
    from tools.train_target_audio_error_critics import train_target_audio_error_critics

    rows = trainable_rows(tmp_path, "minus_critic")
    zero_feature_row = target_view_row(tmp_path, "minus_critic", 30, label=0)
    zero_feature_row["feature_values"] = {
        "window_start_sec": 30.0,
        "window_end_sec": 31.0,
        "route_id": "route-a",
        "candidate_family": "test",
        "stem_type": "minus",
    }
    rows.append(zero_feature_row)
    views_dir = tmp_path / "views"
    write_views(views_dir, rows, trainable_rows(tmp_path, "vocal_critic"))

    manifest = train_target_audio_error_critics(
        views_dir=views_dir,
        output_dir=tmp_path / "model",
        targets=("minus", "vocal"),
        model_kind="random_forest",
        group_split="track_or_candidate",
        shadow_only=True,
        min_safe_rows=4,
        min_positive=1,
        min_negative=1,
    )

    assert manifest["minus_rows_available"] == 9
    assert manifest["minus_rows_used"] == 8
    assert manifest["excluded_reason_counts"]["minus_critic"]["missing_acoustic_feature_values"] == 1


def test_trainer_skips_target_with_missing_negative_class(tmp_path: Path) -> None:
    from tools.train_target_audio_error_critics import train_target_audio_error_critics

    minus_rows = [target_view_row(tmp_path, "minus_critic", index, label=1) for index in range(5)]
    views_dir = tmp_path / "views"
    write_views(views_dir, minus_rows, trainable_rows(tmp_path, "vocal_critic"))

    manifest = train_target_audio_error_critics(
        views_dir=views_dir,
        output_dir=tmp_path / "model",
        targets=("minus", "vocal"),
        model_kind="random_forest",
        group_split="track_or_candidate",
        shadow_only=True,
        min_safe_rows=4,
        min_positive=1,
        min_negative=1,
    )

    assert manifest["minus_critic_training_completed"] is False
    assert manifest["reason_if_minus_not_usable"] == "missing_positive_or_negative_class"
    assert manifest["vocal_critic_training_completed"] is True
    assert not (tmp_path / "model" / "minus_audio_error_critic.joblib").exists()


def test_trainer_writes_false_accept_rate_for_trained_targets(tmp_path: Path) -> None:
    from tools.train_target_audio_error_critics import train_target_audio_error_critics

    views_dir = tmp_path / "views"
    write_views(views_dir, trainable_rows(tmp_path, "minus_critic"), trainable_rows(tmp_path, "vocal_critic"))

    manifest = train_target_audio_error_critics(
        views_dir=views_dir,
        output_dir=tmp_path / "model",
        targets=("minus", "vocal"),
        model_kind="random_forest",
        group_split="track_or_candidate",
        shadow_only=True,
        min_safe_rows=4,
        min_positive=1,
        min_negative=1,
    )

    minus_metrics = read_json(tmp_path / "model" / "minus_audio_error_critic_metrics.json")
    vocal_metrics = read_json(tmp_path / "model" / "vocal_audio_error_critic_metrics.json")
    assert "false_accept_rate" in minus_metrics
    assert "false_reject_rate" in minus_metrics
    assert "false_accept_rate" in vocal_metrics
    assert manifest["model_types"]["minus_accept_block_classifier"] == "sklearn_random_forest"


def test_split_audit_detects_candidate_id_overlap() -> None:
    from tools.train_target_audio_error_critics import split_audit

    rows = [
        {"candidate_id": "same-candidate", "track_id": "track-a", "route_id": "route-a", "label": 1, "split": "train"},
        {"candidate_id": "same-candidate", "track_id": "track-b", "route_id": "route-b", "label": 0, "split": "test"},
    ]

    audit = split_audit(rows, split_strategy="candidate_id")

    assert audit["candidate_id_overlap_train_test"] is True
    assert audit["track_id_overlap_train_test"] is False
    assert audit["test_negative_rows"] == 1


def test_threshold_selection_prioritizes_false_accept_over_recall() -> None:
    from tools.train_target_audio_error_critics import select_thresholds

    rows = [
        {"label": 1},
        {"label": 1},
        {"label": 0},
        {"label": 0},
    ]
    predictions = [
        {"predicted_accept_probability": 0.91},
        {"predicted_accept_probability": 0.52},
        {"predicted_accept_probability": 0.86},
        {"predicted_accept_probability": 0.30},
    ]

    result = select_thresholds(
        rows,
        predictions,
        max_false_accept_rate=0.0,
        threshold_objective="false_accept_first",
        enable_abstain_zone=True,
        abstain_margin=0.10,
    )

    assert result["false_accept_rate"] == 0.0
    assert result["accept_threshold"] > 0.86
    assert result["accepted_positive_rows"] == 1


def test_abstain_is_not_counted_as_accept() -> None:
    from tools.train_target_audio_error_critics import thresholded_metrics

    rows = [{"label": 0}, {"label": 1}]
    predictions = [
        {"predicted_accept_probability": 0.55},
        {"predicted_accept_probability": 0.95},
    ]

    metrics = thresholded_metrics(rows, predictions, accept_threshold=0.90, block_threshold=0.60)

    assert metrics["false_accept_rate"] == 0.0
    assert metrics["abstain_rows"] == 1
    assert metrics["accept_count"] == 1


def test_hard_negative_weighting_uses_only_real_negative_rows(tmp_path: Path) -> None:
    from tools.train_target_audio_error_critics import load_target_rows, sample_weights_for_rows

    views_dir = tmp_path / "views"
    rows = [
        target_view_row(tmp_path, "minus_critic", 1, label=1),
        target_view_row(tmp_path, "minus_critic", 2, label=0, blockers=["severe_phase_collapse"]),
        target_view_row(tmp_path, "minus_critic", 3, label=0, blockers=[]),
        target_view_row(tmp_path, "minus_critic", 4, label=1, blockers=["severe_phase_collapse"]),
    ]
    write_views(views_dir, rows, trainable_rows(tmp_path, "vocal_critic"))
    data = load_target_rows(views_dir, "minus")

    weights, count = sample_weights_for_rows(
        data["usable_rows"],
        negative_class_weight=3.0,
        hard_negative_blockers=("severe_phase_collapse",),
    )

    assert count == 1
    assert max(weights) > 3.0
