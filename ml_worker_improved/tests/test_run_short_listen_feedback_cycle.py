import json
from pathlib import Path

from tools.run_short_listen_feedback_cycle import run_short_listen_feedback_cycle


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _pack(root: Path) -> Path:
    pack = root / "pack.json"
    _write_json(
        pack,
        {
            "schema": "aggregate_short_listen_pack_manifest.v1",
            "pack_id": "lp_test",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_ready": False,
            "can_publish_as_result": False,
            "failed_checks": [],
            "coverage_json": str(root / "coverage.json"),
            "items": [
                {
                    "audio_path": "audio/vrep/sub/v001.flac",
                    "source_alias": "vrep",
                    "stem_type": "vocal",
                    "track_alias": "vrep",
                    "release_ready": False,
                    "can_publish_as_result": False,
                },
                {
                    "audio_path": "audio/mpr/min/m001.flac",
                    "source_alias": "mpr",
                    "stem_type": "minus",
                    "track_alias": "mpr",
                    "release_ready": False,
                    "can_publish_as_result": False,
                },
            ],
            "artifact_name_policy": {
                "schema": "short_windows_safe_artifact_names.v1",
                "raw_track_or_stem_ids_in_path": False,
                "provenance_in_manifest": True,
            },
        },
    )
    _write_json(
        root / "coverage.json",
        {
            "schema": "aggregate_short_listen_pack_coverage.v1",
            "pack_id": "lp_test",
            "release_ready": False,
            "can_publish_as_result": False,
            "failed_checks": [],
            "stem_groups": [
                {
                    "group_alias": "six",
                    "item_count": 7,
                    "stem_types": ["vocal", "minus", "drums", "bass", "other", "guitar", "piano"],
                    "semantic_quality_verified": False,
                }
            ],
        },
    )
    _write_json(
        root / "validation" / "validation.json",
        {
            "schema": "aggregate_short_listen_pack_validation.v1",
            "validation_status": "passed",
            "failed_checks": [],
            "release_ready": False,
            "can_publish_as_result": False,
        },
    )
    return pack


def test_run_short_listen_feedback_cycle_builds_repair_queue_for_rejection(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack")
    feedback_rows = tmp_path / "feedback_items.jsonl"
    _write_jsonl(
        feedback_rows,
        [
            {
                "audio_path": "audio/mpr/min/m001.flac",
                "verdict": "rejected",
                "issue_tags": ["vocal_bleed_in_minus", "minus_muffled"],
                "note": "residual vocal and muffled middle",
            }
        ],
    )

    manifest = run_short_listen_feedback_cycle(
        pack_manifest_json=pack,
        aggregate_validation_json=pack.parent / "validation" / "validation.json",
        feedback_jsonl=feedback_rows,
        output_dir=tmp_path / "cycle",
    )

    assert manifest["schema"] == "short_listen_feedback_cycle_manifest.v1"
    assert manifest["cycle_status"] == "repair_queue_ready"
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert manifest["failed_checks"] == []
    assert manifest["stages"]["feedback_manifest"]["overall_listen_verdict"] == "rejected"
    assert manifest["stages"]["goal_audit"]["goal_status"] == "human_listen_rejected"
    assert manifest["stages"]["repair_queue"]["queue_status"] == "ready_for_repair_cycle"
    rows = [json.loads(line) for line in (tmp_path / "cycle" / "repair_queue" / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["audio_path"] == "audio/mpr/min/m001.flac"
    assert rows[0]["route_intent"] == "minus_repair"
    assert (tmp_path / "cycle" / "cycle.json").exists()
    assert all(len(path.name) <= 32 for path in (tmp_path / "cycle").rglob("*") if path.is_file() or path.is_dir())


def test_run_short_listen_feedback_cycle_fails_closed_on_invalid_feedback(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack")
    feedback_rows = tmp_path / "feedback_items.jsonl"
    _write_jsonl(feedback_rows, [{"audio_path": "audio/not/in/pack.flac", "verdict": "accepted", "issue_tags": []}])

    manifest = run_short_listen_feedback_cycle(
        pack_manifest_json=pack,
        aggregate_validation_json=pack.parent / "validation" / "validation.json",
        feedback_jsonl=feedback_rows,
        output_dir=tmp_path / "cycle",
    )

    assert manifest["cycle_status"] == "failed_safety_contract"
    assert "feedback_manifest:feedback_audio_path_not_in_pack:audio/not/in/pack.flac" in manifest["failed_checks"]
    assert "repair_queue" not in manifest["stages"]


def test_run_short_listen_feedback_cycle_accepts_utf8_bom_jsonl(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack")
    feedback_rows = tmp_path / "feedback_items.jsonl"
    feedback_rows.write_text(
        '\ufeff{"audio_path":"audio/mpr/min/m001.flac","verdict":"rejected","issue_tags":["minus_muffled"]}\n',
        encoding="utf-8",
    )

    manifest = run_short_listen_feedback_cycle(
        pack_manifest_json=pack,
        aggregate_validation_json=pack.parent / "validation" / "validation.json",
        feedback_jsonl=feedback_rows,
        output_dir=tmp_path / "cycle",
    )

    assert manifest["cycle_status"] == "repair_queue_ready"
    assert manifest["failed_checks"] == []
