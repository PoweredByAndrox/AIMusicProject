from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.artifact_status_contract import can_promote_to_listen_result
from tools.build_no_reference_segment_preview_pair_listen_manifest import build_manifest, main


SAMPLE_RATE = 8000


def write_wav(path: Path, value: float = 0.1) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.full((SAMPLE_RATE, 2), value, dtype=np.float32)
    sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
    return str(path)


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def segment_preview_manifest(tmp_path: Path) -> dict:
    mix_path = write_wav(tmp_path / "mix.wav", 0.5)
    vocal_path = write_wav(tmp_path / "preview_vocal.flac", 0.2)
    minus_path = write_wav(tmp_path / "preview_minus.flac", 0.3)
    return {
        "manifest_type": "no_reference_segment_preview_listen",
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "stem_type": "vocal",
        "track_id": "track-a",
        "candidate_id": "base__alt__segment_preview",
        "mix_path": mix_path,
        "source_audio_path": mix_path,
        "selected_audio_path": vocal_path,
        "selected_vocal_path": vocal_path,
        "selected_minus_path": minus_path,
        "accepted_count": 1,
        "accepted_listen_preview_rows": [
            {
                "track_id": "track-a",
                "base_candidate_id": "base",
                "alternate_candidate_id": "alt",
                "source_stem": "vocal",
                "mix_path": mix_path,
                "source_audio_path": mix_path,
                "window_start_sec": 12.0,
                "window_duration_sec": 6.0,
                "selected_vocal_path": vocal_path,
                "selected_minus_path": minus_path,
                "preview_vocal_path": vocal_path,
                "preview_minus_path": minus_path,
                "base_window_score": 0.82,
                "preview_window_score": 0.84,
                "score_delta": 0.02,
                "preview_decision": "preview_improved_window",
                "allowed_next_step": "expand_preview_policy",
            }
        ],
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "release_ready": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": {"training_started": False, "models_changed": False, "thresholds_relaxed": False},
    }


def accepted_pair_row(source_manifest: dict) -> dict:
    return {
        "track_id": source_manifest["track_id"],
        "candidate_id": "segment-preview-pair-a",
        "mix_path": source_manifest["mix_path"],
        "source_audio_path": source_manifest["source_audio_path"],
        "vocal_path": source_manifest["selected_vocal_path"],
        "instrumental_path": source_manifest["selected_minus_path"],
        "minus_path": source_manifest["selected_minus_path"],
        "score_mode": "both",
        "decision": "accept_candidate",
        "pair_decision": "pair_compatible",
        "pair_compatibility_score": 0.88,
        "pair_blockers": [],
        "window_count": 5,
        "window_seconds": 6.0,
    }


def test_pair_listen_manifest_accepts_exact_compatible_segment_preview_pair(tmp_path: Path) -> None:
    source = segment_preview_manifest(tmp_path)
    source_json = write_json(tmp_path / "segment_preview_listen_manifest.json", source)
    pair_jsonl = write_jsonl(tmp_path / "pair_compatibility.jsonl", [accepted_pair_row(source)])

    manifest = build_manifest(segment_preview_manifest_json=source_json, pair_compatibility_jsonl=pair_jsonl)

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["presentation_status"] == "accepted_for_listen_pack"
    assert manifest["release_status"] == "listen_preview_only"
    assert manifest["stem_type"] == "both"
    assert manifest["accepted_stem_count"] == 2
    assert manifest["pair_decision"] == "pair_compatible"
    assert manifest["pair_compatibility_score"] == 0.88
    assert manifest["selected_vocal_path"] == source["selected_vocal_path"]
    assert manifest["selected_minus_path"] == source["selected_minus_path"]
    assert manifest["release_ready"] is False
    assert manifest["can_mark_release_ready"] is False

    output_manifest = tmp_path / "pair_listen_manifest.json"
    write_json(output_manifest, manifest)
    assert can_promote_to_listen_result(output_manifest) is True


def test_pair_listen_manifest_blocks_path_mismatch(tmp_path: Path) -> None:
    source = segment_preview_manifest(tmp_path)
    pair_row = accepted_pair_row(source)
    pair_row["minus_path"] = write_wav(tmp_path / "other_minus.flac", 0.4)
    pair_row["instrumental_path"] = pair_row["minus_path"]

    manifest = build_manifest(
        segment_preview_manifest_json=write_json(tmp_path / "segment_preview_listen_manifest.json", source),
        pair_compatibility_jsonl=write_jsonl(tmp_path / "pair_compatibility.jsonl", [pair_row]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "pair_compatibility_no_exact_match"


def test_cli_writes_promotable_segment_preview_pair_manifest(tmp_path: Path) -> None:
    source = segment_preview_manifest(tmp_path)
    output_manifest = tmp_path / "pair_listen_manifest.json"
    summary_md = tmp_path / "pair_listen_summary.md"

    exit_code = main(
        [
            "--segment-preview-manifest-json",
            str(write_json(tmp_path / "segment_preview_listen_manifest.json", source)),
            "--pair-compatibility-jsonl",
            str(write_jsonl(tmp_path / "pair_compatibility.jsonl", [accepted_pair_row(source)])),
            "--output-manifest-json",
            str(output_manifest),
            "--summary-md",
            str(summary_md),
        ]
    )

    assert exit_code == 0
    assert can_promote_to_listen_result(output_manifest) is True
    assert "delivery_status: accepted" in summary_md.read_text(encoding="utf-8")
