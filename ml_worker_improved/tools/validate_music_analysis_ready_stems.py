from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


ALLOWED_STEM_TYPES = {"vocal", "minus", "instrumental", "piano", "bass", "drums", "other", "unknown"}
ALLOWED_QUALITY_STATUSES = {"accepted", "blocked"}

REQUIRED_STEM_FIELDS = {
    "track_id",
    "run_id",
    "candidate_id",
    "stem_id",
    "stem_type",
    "instrument_hint",
    "audio_path",
    "normalized_audio_path",
    "sample_rate",
    "channels",
    "duration_sec",
    "quality_status",
    "blockers",
    "analysis_allowed",
    "analysis_block_reason",
    "source_separator_config",
    "source_candidate_family",
    "pair_compatibility_status",
    "release_ready",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def validate_music_intelligence_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    errors: list[str] = []
    stems = manifest.get("stems")
    if not isinstance(stems, list):
        stems = []
        errors.append("stems_must_be_list")
    if manifest.get("release_ready") is True:
        errors.append("release_ready_forbidden")
    if manifest.get("melody_transcription_ran") is True:
        errors.append("melody_transcription_forbidden")
    if manifest.get("recommendation_index_updated") is True:
        errors.append("recommendation_index_update_forbidden")
    if manifest.get("future_analysis_supported") is True:
        errors.append("future_analysis_supported_must_remain_false")

    ready_count = 0
    blocked_count = 0
    for row in stems:
        if not isinstance(row, dict):
            errors.append("stem_row_must_be_object")
            continue
        stem_id = str(row.get("stem_id") or "unknown_stem")
        missing = sorted(REQUIRED_STEM_FIELDS - set(row))
        errors.extend(f"missing_stem_field:{stem_id}:{field}" for field in missing)
        if row.get("release_ready") is True:
            errors.append(f"stem_release_ready_forbidden:{stem_id}")
        if row.get("stem_type") not in ALLOWED_STEM_TYPES:
            errors.append(f"invalid_stem_type:{stem_id}:{row.get('stem_type')}")
        if row.get("quality_status") not in ALLOWED_QUALITY_STATUSES:
            errors.append(f"invalid_quality_status:{stem_id}:{row.get('quality_status')}")
        blockers = row.get("blockers")
        if not isinstance(blockers, list):
            errors.append(f"blockers_must_be_list:{stem_id}")
            blockers = []
        if row.get("analysis_allowed") is True:
            ready_count += 1
            if row.get("quality_status") != "accepted":
                errors.append(f"analysis_allowed_requires_accepted_quality:{stem_id}")
            if blockers:
                errors.append(f"analysis_allowed_requires_no_blockers:{stem_id}")
            if row.get("analysis_block_reason"):
                errors.append(f"analysis_allowed_requires_empty_block_reason:{stem_id}")
        else:
            blocked_count += 1
            if row.get("quality_status") == "accepted" and not row.get("analysis_block_reason"):
                errors.append(f"accepted_stem_blocked_without_reason:{stem_id}")

    if int(manifest.get("analysis_ready_stems_count") or 0) != ready_count:
        errors.append("analysis_ready_stems_count_mismatch")
    if int(manifest.get("blocked_stems_count") or 0) != blocked_count:
        errors.append("blocked_stems_count_mismatch")
    return {
        "ok": not errors,
        "errors": errors,
        "manifest_path": str(path),
        "stem_count": len(stems),
        "analysis_ready_stems_count": ready_count,
        "blocked_stems_count": blocked_count,
        "melody_transcription_ran": False,
        "recommendation_index_updated": False,
        "release_ready": False,
    }


def future_schema_examples() -> dict[str, Any]:
    return {
        "stem_note_events.jsonl": {
            "track_id": "track-a",
            "stem_id": "run-a__candidate-a__vocal",
            "instrument_role": "vocal",
            "note_name": "A4",
            "midi_pitch": 69,
            "start_sec": 12.0,
            "end_sec": 12.5,
            "duration_sec": 0.5,
            "velocity_proxy": 0.72,
            "confidence": 0.91,
            "source_transcription_route": "future_disabled_stub",
        },
        "stem_pitch_contours.jsonl": {
            "track_id": "track-a",
            "stem_id": "run-a__candidate-a__vocal",
            "instrument_role": "vocal",
            "start_sec": 12.0,
            "end_sec": 13.0,
            "hop_sec": 0.01,
            "pitch_hz": [440.0, 441.0, 439.5],
            "voicing_confidence": [0.9, 0.91, 0.88],
            "source_transcription_route": "future_disabled_stub",
        },
        "stem_melody_motifs.jsonl": {
            "motif_id": "motif-a",
            "track_id": "track-a",
            "stem_id": "run-a__candidate-a__vocal",
            "instrument_role": "vocal",
            "interval_pattern": [0, 2, -1],
            "rhythm_pattern": [0.5, 0.5, 1.0],
            "scale_degree_pattern": [1, 2, 1],
            "contour": "up_down",
            "start_sec": 12.0,
            "end_sec": 14.0,
            "repeat_count": 2,
            "confidence": 0.83,
        },
        "stem_chord_events.jsonl": {
            "track_id": "track-a",
            "stem_id": "run-a__candidate-a__minus",
            "instrument_role": "harmony",
            "chord_name": "Am",
            "root_midi_pitch": 69,
            "start_sec": 0.0,
            "end_sec": 2.0,
            "confidence": 0.78,
            "source_transcription_route": "future_disabled_stub",
        },
        "stem_rhythm_features.jsonl": {
            "track_id": "track-a",
            "stem_id": "run-a__candidate-a__drums",
            "instrument_role": "rhythm",
            "tempo_bpm": 128.0,
            "meter": "4/4",
            "onset_times_sec": [0.0, 0.5, 1.0],
            "syncopation_score": 0.21,
            "confidence": 0.86,
        },
        "stem_instrument_roles.jsonl": {
            "track_id": "track-a",
            "stem_id": "run-a__candidate-a__bass",
            "instrument_role": "bassline",
            "instrument_hint": "bass",
            "role_confidence": 0.84,
            "active_segments": [{"start_sec": 0.0, "end_sec": 8.0}],
        },
        "track_music_fingerprint.json": {
            "track_id": "track-a",
            "primary_melody_stem_id": "run-a__candidate-a__vocal",
            "phrase_structure": [{"label": "A", "start_sec": 0.0, "end_sec": 16.0}],
            "motif_ids": ["motif-a"],
            "analysis_quality_score": 0.0,
            "release_ready": False,
        },
        "track_recommendation_features.json": {
            "track_id": "track-a",
            "global_melody_embedding": [],
            "vocal_melody_embedding": [],
            "piano_melody_embedding": [],
            "orchestral_melody_embedding": [],
            "bassline_embedding": [],
            "chord_progression_embedding": [],
            "rhythm_embedding": [],
            "motif_embedding_ids": ["motif-a"],
            "analysis_quality_score": 0.0,
            "release_ready": False,
        },
    }


def validate_future_schema_examples(examples: dict[str, Any] | None = None) -> dict[str, Any]:
    examples = examples or future_schema_examples()
    required = {
        "stem_note_events.jsonl": {
            "track_id",
            "stem_id",
            "instrument_role",
            "note_name",
            "midi_pitch",
            "start_sec",
            "end_sec",
            "duration_sec",
            "velocity_proxy",
            "confidence",
            "source_transcription_route",
        },
        "stem_melody_motifs.jsonl": {
            "motif_id",
            "track_id",
            "stem_id",
            "instrument_role",
            "interval_pattern",
            "rhythm_pattern",
            "scale_degree_pattern",
            "contour",
            "start_sec",
            "end_sec",
            "repeat_count",
            "confidence",
        },
        "track_recommendation_features.json": {
            "track_id",
            "global_melody_embedding",
            "vocal_melody_embedding",
            "piano_melody_embedding",
            "orchestral_melody_embedding",
            "bassline_embedding",
            "chord_progression_embedding",
            "rhythm_embedding",
            "motif_embedding_ids",
            "analysis_quality_score",
        },
    }
    errors: list[str] = []
    for filename, fields in required.items():
        payload = examples.get(filename)
        if not isinstance(payload, dict):
            errors.append(f"missing_example:{filename}")
            continue
        missing = sorted(fields - set(payload))
        errors.extend(f"missing_example_field:{filename}:{field}" for field in missing)
    if set(examples) != {
        "stem_note_events.jsonl",
        "stem_pitch_contours.jsonl",
        "stem_melody_motifs.jsonl",
        "stem_chord_events.jsonl",
        "stem_rhythm_features.jsonl",
        "stem_instrument_roles.jsonl",
        "track_music_fingerprint.json",
        "track_recommendation_features.json",
    }:
        errors.append("future_schema_example_set_mismatch")
    return {"ok": not errors, "errors": errors, "validated_examples_count": len(examples)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validate-schema-examples", default="true", choices=("true", "false"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = validate_music_intelligence_manifest(Path(args.manifest))
    if args.validate_schema_examples == "true":
        schema_report = validate_future_schema_examples()
        report["future_schema_examples"] = schema_report
        if not schema_report["ok"]:
            report["ok"] = False
            report["errors"].extend(schema_report["errors"])
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
