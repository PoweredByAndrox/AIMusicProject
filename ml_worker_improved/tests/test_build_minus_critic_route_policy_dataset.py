from __future__ import annotations

import json
from pathlib import Path

from tools.build_minus_critic_route_policy_dataset import build_minus_critic_route_policy_dataset
from tools.stem_route_policy_common import ACTION_BLOCK_UNKNOWN, ACTION_QUARANTINE_ROUTE, ISSUE_VOCAL_LEAK


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _scored_row(*, true_label: int, row_id: str = "raw_track_title__both_abc__minus__0.000-12.000__hash") -> dict:
    return {
        "abstained": False,
        "candidate_id": "both_0ea68e5599d8ad66",
        "confidence": 0.575,
        "critic_decision": "blocked_hard_case" if true_label == 0 else "accept_candidate",
        "label_source": "minus_windowed_judge.jsonl",
        "predicted_accept_probability": 0.2125 if true_label == 0 else 0.91,
        "predicted_block_probability": 0.7875 if true_label == 0 else 0.09,
        "predicted_blockers": ["severe_vocal_bleed_in_minus"] if true_label == 0 else [],
        "release_ready": False,
        "route_id": "local_registered__audio_separator_bs_roformer_viperx_1297_vendor_config_vocals__both__minus_direct",
        "row_id": row_id,
        "separator_training_started": False,
        "shadow_only": True,
        "target_name": "minus_critic",
        "track_id": "raw_track_title_should_not_become_generated_id",
        "true_label": true_label,
        "view_name": "minus_critic",
        "window_end_sec": 12.0,
        "window_start_sec": 0.0,
    }


def _read_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_failed_minus_label_maps_to_quarantine_route(tmp_path: Path) -> None:
    scores = _write_jsonl(tmp_path / "scores.jsonl", [_scored_row(true_label=0), _scored_row(true_label=1, row_id="good_row")])

    manifest = build_minus_critic_route_policy_dataset(scores_jsonl_path=scores, output_dir=tmp_path / "policy")
    rows = _read_rows(manifest["rows_jsonl"])

    failed = rows[0]
    good = rows[1]
    assert failed["oracle_action"] == ACTION_QUARANTINE_ROUTE
    assert failed["oracle_label_source"] == "target_audio_error_view_label"
    assert failed["issue_type"] == ISSUE_VOCAL_LEAK
    assert failed["affected_stem"] == "minus"
    assert failed["usable_for_promotion"] is False
    assert failed["delivery_allowed"] is False
    assert good["oracle_action"] == ACTION_BLOCK_UNKNOWN
    assert manifest["delivery_allowed"] is False
    assert manifest["runtime_gates_changed"] is False


def test_dedupes_scored_rows_and_uses_safe_generated_ids(tmp_path: Path) -> None:
    raw_track = "raw_track_title_should_not_become_generated_id"
    row = _scored_row(true_label=0)
    scores = _write_jsonl(tmp_path / "scores.jsonl", [row, dict(row)])

    manifest = build_minus_critic_route_policy_dataset(scores_jsonl_path=scores, output_dir=tmp_path / "policy")
    rows_text = Path(manifest["rows_jsonl"]).read_text(encoding="utf-8")
    rows = _read_rows(manifest["rows_jsonl"])

    assert manifest["input_scored_row_count"] == 2
    assert manifest["duplicate_rows_skipped"] == 1
    assert manifest["row_count"] == 1
    assert rows[0]["row_id"] == "p000001"
    assert rows[0]["candidate_id"].startswith("c")
    assert rows[0]["source_target_critic_row_id"].startswith("tc")
    assert raw_track not in rows_text


def test_non_minus_rows_are_skipped(tmp_path: Path) -> None:
    vocal = _scored_row(true_label=0)
    vocal["target_name"] = "vocal_critic"
    scores = _write_jsonl(tmp_path / "scores.jsonl", [vocal])

    manifest = build_minus_critic_route_policy_dataset(scores_jsonl_path=scores, output_dir=tmp_path / "policy")

    assert manifest["row_count"] == 0
    assert manifest["skipped_reason_counts"] == {"non_minus_critic_row": 1}
    assert manifest["delivery_allowed"] is False


def test_missing_target_name_is_skipped_fail_closed(tmp_path: Path) -> None:
    row = _scored_row(true_label=0)
    row.pop("target_name")
    row.pop("view_name")
    scores = _write_jsonl(tmp_path / "scores.jsonl", [row])

    manifest = build_minus_critic_route_policy_dataset(scores_jsonl_path=scores, output_dir=tmp_path / "policy")

    assert manifest["row_count"] == 0
    assert manifest["skipped_reason_counts"] == {"non_minus_critic_row": 1}
    assert manifest["delivery_allowed"] is False


def test_required_mount_preflight_blocks_dataset_build(tmp_path: Path) -> None:
    scores = _write_jsonl(tmp_path / "scores.jsonl", [_scored_row(true_label=0)])
    preflight = _write_json(
        tmp_path / "mount.json",
        {
            "schema": "mac_windows_project_mount_preflight.v1",
            "preflight_status": "blocked",
            "probe_status": "failed",
            "can_write_route_artifacts": False,
        },
    )

    manifest = build_minus_critic_route_policy_dataset(
        scores_jsonl_path=scores,
        output_dir=tmp_path / "policy",
        mount_preflight_json_path=preflight,
        require_mount_preflight=True,
    )

    assert manifest["output_status"] == "blocked_mount_preflight"
    assert manifest["row_count"] == 0
    assert manifest["delivery_allowed"] is False
