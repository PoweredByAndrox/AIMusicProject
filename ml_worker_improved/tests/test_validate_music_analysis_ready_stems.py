from __future__ import annotations

import json
from pathlib import Path

from tools.validate_music_analysis_ready_stems import (
    future_schema_examples,
    validate_future_schema_examples,
    validate_music_intelligence_manifest,
)


def write_manifest(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def valid_manifest() -> dict:
    return {
        "schema_version": 1,
        "run_id": "run-a",
        "track_id": "track-a",
        "release_ready": False,
        "future_analysis_supported": False,
        "melody_transcription_ran": False,
        "recommendation_index_updated": False,
        "analysis_ready_stems_count": 1,
        "blocked_stems_count": 1,
        "stems": [
            {
                "track_id": "track-a",
                "run_id": "run-a",
                "candidate_id": "candidate-a",
                "stem_id": "run-a__candidate-a__vocal",
                "stem_type": "vocal",
                "instrument_hint": "vocal",
                "audio_path": "/app/stems/vocal.wav",
                "normalized_audio_path": "/app/stems/vocal.wav",
                "sample_rate": 44100,
                "channels": 2,
                "duration_sec": 30.0,
                "quality_status": "accepted",
                "blockers": [],
                "analysis_allowed": True,
                "analysis_block_reason": "",
                "source_separator_config": "htdemucs_ft",
                "source_candidate_family": "local_registered",
                "pair_compatibility_status": "pair_compatible",
                "release_ready": False,
            },
            {
                "track_id": "track-a",
                "run_id": "run-a",
                "candidate_id": "candidate-b",
                "stem_id": "run-a__candidate-b__minus",
                "stem_type": "minus",
                "instrument_hint": "minus",
                "audio_path": "/app/stems/minus.wav",
                "normalized_audio_path": "/app/stems/minus.wav",
                "sample_rate": 44100,
                "channels": 2,
                "duration_sec": 30.0,
                "quality_status": "blocked",
                "blockers": ["severe_vocal_bleed_in_minus"],
                "analysis_allowed": False,
                "analysis_block_reason": "severe_vocal_bleed_in_minus",
                "source_separator_config": "htdemucs_ft",
                "source_candidate_family": "local_registered",
                "pair_compatibility_status": "not_evaluated",
                "release_ready": False,
            },
        ],
    }


def test_validator_accepts_safe_manifest_and_counts_stems(tmp_path: Path) -> None:
    path = write_manifest(tmp_path / "music_intelligence_input_manifest.json", valid_manifest())

    report = validate_music_intelligence_manifest(path)

    assert report["ok"] is True
    assert report["analysis_ready_stems_count"] == 1
    assert report["blocked_stems_count"] == 1
    assert report["release_ready"] is False
    assert report["melody_transcription_ran"] is False
    assert report["recommendation_index_updated"] is False


def test_validator_rejects_release_ready_and_transcription_side_effects(tmp_path: Path) -> None:
    payload = valid_manifest()
    payload["release_ready"] = True
    payload["melody_transcription_ran"] = True
    payload["recommendation_index_updated"] = True
    payload["stems"][0]["release_ready"] = True
    path = write_manifest(tmp_path / "music_intelligence_input_manifest.json", payload)

    report = validate_music_intelligence_manifest(path)

    assert report["ok"] is False
    assert "release_ready_forbidden" in report["errors"]
    assert "melody_transcription_forbidden" in report["errors"]
    assert "recommendation_index_update_forbidden" in report["errors"]
    assert "stem_release_ready_forbidden:run-a__candidate-a__vocal" in report["errors"]


def test_validator_rejects_analysis_allowed_for_blocked_or_dirty_stem(tmp_path: Path) -> None:
    payload = valid_manifest()
    payload["stems"][1]["analysis_allowed"] = True
    path = write_manifest(tmp_path / "music_intelligence_input_manifest.json", payload)

    report = validate_music_intelligence_manifest(path)

    assert report["ok"] is False
    assert "analysis_allowed_requires_accepted_quality:run-a__candidate-b__minus" in report["errors"]
    assert "analysis_allowed_requires_no_blockers:run-a__candidate-b__minus" in report["errors"]


def test_future_schema_examples_validate() -> None:
    examples = future_schema_examples()

    assert {
        "stem_note_events.jsonl",
        "stem_pitch_contours.jsonl",
        "stem_melody_motifs.jsonl",
        "stem_chord_events.jsonl",
        "stem_rhythm_features.jsonl",
        "stem_instrument_roles.jsonl",
        "track_music_fingerprint.json",
        "track_recommendation_features.json",
    } == set(examples)
    report = validate_future_schema_examples(examples)

    assert report["ok"] is True
    assert report["validated_examples_count"] == 8
