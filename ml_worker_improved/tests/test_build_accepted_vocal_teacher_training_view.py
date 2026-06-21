from __future__ import annotations

import json
from pathlib import Path

from tools.build_accepted_vocal_teacher_training_view import build_accepted_vocal_teacher_training_view


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_builds_human_verified_teacher_rows_without_ground_truth_claim(tmp_path: Path) -> None:
    target = tmp_path / "accepted_vocal.flac"
    second_target = tmp_path / "accepted_vocal_alt.flac"
    target.write_bytes(b"accepted vocal")
    second_target.write_bytes(b"accepted vocal alt")
    source = tmp_path / "source_mix.mp3"
    source.write_bytes(b"source mix")
    inventory = write_jsonl(
        tmp_path / "inventory.jsonl",
        [
            {
                "schema": "generic_finetuning_lab_artifact_inventory_row.v1",
                "accepted_contract": True,
                "stem_type": "vocal",
                "track_id": "song_one",
                "manifest_path": "/app/lab/song/no_reference_vocal_listen_preview_manifest.json",
                "model_id": "accepted_route",
                "selected_audio_exists": True,
                "selected_audio_path": str(target),
                "delivery_allowed": True,
                "presentation_status": "accepted_for_listen_pack",
                "release_status": "listen_preview_only",
                "reference_verified": False,
            },
            {
                "schema": "generic_finetuning_lab_artifact_inventory_row.v1",
                "accepted_contract": True,
                "stem_type": "vocal",
                "track_id": "song_one",
                "manifest_path": "/app/lab/song/alt_no_reference_vocal_listen_preview_manifest.json",
                "model_id": "accepted_route_alt",
                "selected_audio_exists": True,
                "selected_audio_path": str(second_target),
                "delivery_allowed": True,
                "presentation_status": "accepted_for_listen_pack",
                "release_status": "listen_preview_only",
                "reference_verified": False,
            }
        ],
    )
    source_plan = write_jsonl(
        tmp_path / "testing_stuff_music_corpus_plan.jsonl",
        [
            {
                "schema": "testing_stuff_music_corpus_plan_row.v1",
                "track_id": "song_one",
                "docker_audio_path": str(source),
                "reference_available": False,
                "training_allowed": False,
            }
        ],
    )

    manifest = build_accepted_vocal_teacher_training_view(
        inventory_jsonl=inventory,
        source_plan_jsonl=source_plan,
        output_dir=tmp_path / "out",
        verification_source="user_thread_assertion",
        permission_status="project_owned",
    )

    rows = read_jsonl(tmp_path / "out" / "accepted_vocal_teacher_training_examples.jsonl")
    pseudo_rows = read_jsonl(tmp_path / "out" / "accepted_vocal_pseudo_teacher_training_examples.jsonl")
    assert manifest["teacher_example_count"] == 2
    assert manifest["pseudo_teacher_example_count"] == 2
    assert manifest["blocked_example_count"] == 0
    assert rows[0]["schema"] == "generic_finetuning_audio_training_example.v1"
    assert rows[0]["track_id"] == "song_one"
    assert rows[0]["original_track_id"] == "song_one"
    assert rows[0]["source_plan_track_id"] == "song_one"
    assert rows[0]["id"].startswith("accepted_vocal_teacher__")
    assert rows[0]["id"] != rows[1]["id"]
    assert rows[0]["source_audio_path"] == str(source)
    assert rows[0]["target_audio_path"] == str(target)
    assert rows[0]["source_reconciliation_method"] == "exact_track_id"
    assert rows[0]["human_verified"] is True
    assert rows[0]["reference_verified"] is False
    assert rows[0]["source_teacher_is_ground_truth"] is False
    assert rows[0]["safety"]["accepted_no_reference_teacher_not_ground_truth"] is True
    assert rows[0]["safety"]["canonical_track_id_from_source_plan"] is True
    assert rows[0]["delivery_allowed"] is False
    assert rows[0]["release_ready"] is False
    assert pseudo_rows[0]["schema"] == "generic_finetuning_pseudo_teacher_audio_training_example.v1"
    assert pseudo_rows[0]["track_id"] == "song_one"
    assert pseudo_rows[0]["source_plan_track_id"] == "song_one"
    assert pseudo_rows[0]["pseudo_teacher"] is True
    assert pseudo_rows[0]["pseudo_teacher_targets_are_ground_truth"] is False
    assert pseudo_rows[0]["accepted_no_reference_teacher_human_audited"] is True
    assert pseudo_rows[0]["human_verified"] is False
    assert pseudo_rows[0]["reference_verified"] is False
    assert pseudo_rows[0]["delivery_allowed"] is False
    assert pseudo_rows[0]["release_ready"] is False


def test_reconciles_unique_normalized_source_alias_without_fallback(tmp_path: Path) -> None:
    target = tmp_path / "accepted_vocal.flac"
    target.write_bytes(b"accepted vocal")
    source = tmp_path / "baby__me_-_here_comes_the_hotstepper_(z3.fm).mp3"
    source.write_bytes(b"source mix")
    inventory = write_jsonl(
        tmp_path / "inventory.jsonl",
        [
            {
                "accepted_contract": True,
                "stem_type": "vocal",
                "track_id": "baby__me_-_here_comes_the_hotstepper__z3.fm",
                "manifest_path": "/app/lab/baby/no_reference_vocal_listen_preview_manifest.json",
                "selected_audio_path": str(target),
                "delivery_allowed": True,
                "presentation_status": "accepted_for_listen_pack",
                "reference_verified": False,
            }
        ],
    )
    source_plan = write_jsonl(
        tmp_path / "testing_stuff_music_corpus_plan.jsonl",
        [
            {
                "schema": "testing_stuff_music_corpus_plan_row.v1",
                "track_id": "baby_me_here_comes_the_hotstepper",
                "relative_path": source.name,
                "docker_audio_path": str(source),
                "reference_available": False,
                "training_allowed": False,
            }
        ],
    )

    manifest = build_accepted_vocal_teacher_training_view(
        inventory_jsonl=inventory,
        source_plan_jsonl=source_plan,
        output_dir=tmp_path / "out",
        verification_source="user_thread_assertion",
        permission_status="project_owned",
    )

    rows = read_jsonl(tmp_path / "out" / "accepted_vocal_teacher_training_examples.jsonl")
    pseudo_rows = read_jsonl(tmp_path / "out" / "accepted_vocal_pseudo_teacher_training_examples.jsonl")
    assert manifest["teacher_example_count"] == 1
    assert manifest["pseudo_teacher_example_count"] == 1
    assert manifest["blocked_example_count"] == 0
    assert rows[0]["source_audio_path"] == str(source)
    assert rows[0]["track_id"] == "baby_me_here_comes_the_hotstepper"
    assert rows[0]["original_track_id"] == "baby__me_-_here_comes_the_hotstepper__z3.fm"
    assert rows[0]["source_plan_track_id"] == "baby_me_here_comes_the_hotstepper"
    assert rows[0]["source_reconciliation_method"] == "normalized_unique_source_alias"
    assert rows[0]["safety"]["fallback_used"] is False
    assert manifest["track_counts"] == {"baby_me_here_comes_the_hotstepper": 1}
    assert manifest["original_track_counts"] == {"baby__me_-_here_comes_the_hotstepper__z3.fm": 1}
    assert pseudo_rows[0]["track_id"] == "baby_me_here_comes_the_hotstepper"
    assert pseudo_rows[0]["original_track_id"] == "baby__me_-_here_comes_the_hotstepper__z3.fm"
    assert pseudo_rows[0]["source_reconciliation_method"] == "normalized_unique_source_alias"
    assert pseudo_rows[0]["safety"]["fallback_used"] is False


def test_blocks_accepted_vocal_when_source_mix_cannot_be_reconciled(tmp_path: Path) -> None:
    target = tmp_path / "accepted_vocal.flac"
    target.write_bytes(b"accepted vocal")
    inventory = write_jsonl(
        tmp_path / "inventory.jsonl",
        [
            {
                "accepted_contract": True,
                "stem_type": "vocal",
                "track_id": "missing_source",
                "manifest_path": "/app/lab/missing/no_reference_vocal_listen_preview_manifest.json",
                "selected_audio_exists": True,
                "selected_audio_path": str(target),
                "delivery_allowed": True,
                "presentation_status": "accepted_for_listen_pack",
                "release_status": "listen_preview_only",
                "reference_verified": False,
            }
        ],
    )
    source_plan = write_jsonl(tmp_path / "testing_stuff_music_corpus_plan.jsonl", [])

    manifest = build_accepted_vocal_teacher_training_view(
        inventory_jsonl=inventory,
        source_plan_jsonl=source_plan,
        output_dir=tmp_path / "out",
        verification_source="user_thread_assertion",
        permission_status="project_owned",
    )

    rows = read_jsonl(tmp_path / "out" / "accepted_vocal_teacher_training_examples.jsonl")
    blocked = read_jsonl(tmp_path / "out" / "accepted_vocal_teacher_blocked_examples.jsonl")
    assert rows == []
    assert manifest["teacher_example_count"] == 0
    assert manifest["blocked_example_count"] == 1
    assert blocked[0]["blocked_reasons"] == ["source_mix_not_reconciled"]
    assert blocked[0]["fallback_used"] is False
    assert blocked[0]["release_ready"] is False
