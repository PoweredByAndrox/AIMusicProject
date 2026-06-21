from __future__ import annotations

import json
from pathlib import Path

from tools.build_pseudo_teacher_audio_training_examples import build_pseudo_teacher_audio_training_examples


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _audio(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return str(path)


def test_builds_pseudo_teacher_rows_from_accepted_selected_vocals(tmp_path: Path) -> None:
    source_audio = _audio(tmp_path / "mix.flac", b"mix")
    vocal_audio = _audio(tmp_path / "vocals.flac", b"vocal")
    run_root = tmp_path / "run"
    selected_path = _write_jsonl(
        run_root / "selected_candidates.jsonl",
        [
            {
                "track_id": "track-a",
                "selected_vocal_candidate_id": "both_a",
                "selected_vocal_path": vocal_audio,
                "can_publish_as_result": False,
                "can_mark_release_ready": False,
                "release_ready": False,
            }
        ],
    )
    run_manifest_path = _write_json(run_root / "run_manifest.json", {"input_audio": source_audio, "release_ready": False})
    music_manifest_path = _write_json(
        run_root / "music_intelligence_input_manifest.json",
        {
            "stems": [
                {
                    "track_id": "track-a",
                    "stem_id": "track-a__vocal",
                    "stem_type": "vocal",
                    "quality_status": "accepted",
                    "analysis_allowed": True,
                    "analysis_block_reason": "",
                    "blockers": [],
                    "audio_path": vocal_audio,
                    "normalized_audio_path": vocal_audio,
                    "source_separator_config": "local_registered__htdemucs_ft__both__shared",
                    "pair_compatibility_status": "pair_compatible",
                    "release_ready": False,
                }
            ]
        },
    )

    manifest = build_pseudo_teacher_audio_training_examples(
        run_roots=[run_root],
        output_dir=tmp_path / "out",
        stem="vocal",
        scope="pseudo_teacher_test",
    )

    assert manifest["status"] == "passed"
    assert manifest["pseudo_teacher_example_count"] == 1
    assert manifest["blocked_candidate_count"] == 0
    assert manifest["release_ready"] is False
    assert manifest["safety"]["dirty_stems_never_ground_truth"] is True

    rows = _read_jsonl(Path(manifest["pseudo_teacher_examples_jsonl"]))
    assert rows == [
        {
            "schema": "generic_finetuning_pseudo_teacher_audio_training_example.v1",
            "id": rows[0]["id"],
            "track_id": "track-a",
            "stem_type": "vocal",
            "source_audio_path": source_audio,
            "target_audio_path": vocal_audio,
            "target_audio_sha256": rows[0]["target_audio_sha256"],
            "split": "train",
            "trainable": True,
            "pseudo_teacher": True,
            "pseudo_teacher_source": "no_reference_runtime_selected_candidate",
            "pseudo_teacher_targets_are_ground_truth": False,
            "human_verified": False,
            "reference_verified": False,
            "permission_status": "pseudo_teacher_runtime_artifact",
            "release_ready": False,
            "delivery_allowed": False,
            "can_publish_as_result": False,
            "can_mark_release_ready": False,
            "source_run_root": str(run_root),
            "run_manifest_json": str(run_manifest_path),
            "selected_candidates_jsonl": str(selected_path),
            "music_intelligence_manifest_json": str(music_manifest_path),
            "selected_candidate_id": "both_a",
            "source_separator_config": "local_registered__htdemucs_ft__both__shared",
            "pair_compatibility_status": "pair_compatible",
        }
    ]


def test_blocks_release_ready_or_missing_audio_pseudo_teacher_rows(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_jsonl(
        run_root / "selected_candidates.jsonl",
        [
            {
                "track_id": "track-a",
                "selected_vocal_candidate_id": "both_a",
                "selected_vocal_path": str(tmp_path / "missing_vocal.flac"),
                "can_publish_as_result": True,
                "can_mark_release_ready": False,
                "release_ready": False,
            }
        ],
    )
    _write_json(run_root / "run_manifest.json", {"input_audio": str(tmp_path / "missing_mix.flac"), "release_ready": False})
    _write_json(
        run_root / "music_intelligence_input_manifest.json",
        {
            "stems": [
                {
                    "track_id": "track-a",
                    "stem_id": "track-a__vocal",
                    "stem_type": "vocal",
                    "quality_status": "accepted",
                    "analysis_allowed": True,
                    "blockers": [],
                    "audio_path": str(tmp_path / "missing_vocal.flac"),
                    "normalized_audio_path": str(tmp_path / "missing_vocal.flac"),
                    "release_ready": False,
                }
            ]
        },
    )

    manifest = build_pseudo_teacher_audio_training_examples(
        run_roots=[run_root],
        output_dir=tmp_path / "out",
        stem="vocal",
        scope="pseudo_teacher_test",
    )

    assert manifest["status"] == "failed"
    assert manifest["pseudo_teacher_example_count"] == 0
    assert manifest["blocked_candidate_count"] == 1
    blocked = _read_jsonl(Path(manifest["blocked_candidates_jsonl"]))
    assert "selected_candidate_publishable" in blocked[0]["blocked_reasons"]
    assert "source_audio_not_found" in blocked[0]["blocked_reasons"]
    assert "target_audio_not_found" in blocked[0]["blocked_reasons"]
