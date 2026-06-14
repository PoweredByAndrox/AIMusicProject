from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.build_no_reference_segment_preview_pair_candidate import build_pair_candidate_manifest, main


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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def accepted_segment_preview_manifest(tmp_path: Path, *, include_mix_path: bool = True) -> dict:
    mix_path = write_wav(tmp_path / "mix.wav", 0.5)
    vocal_path = write_wav(tmp_path / "preview_vocal.flac", 0.2)
    minus_path = write_wav(tmp_path / "preview_minus.flac", 0.3)
    row = {
        "track_id": "track-a",
        "base_candidate_id": "base",
        "alternate_candidate_id": "alt",
        "source_stem": "vocal",
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
    if include_mix_path:
        row["mix_path"] = mix_path
        row["source_audio_path"] = mix_path
    return {
        "manifest_type": "no_reference_segment_preview_listen",
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "stem_type": "vocal",
        "track_id": "track-a",
        "candidate_id": "base__alt__segment_preview",
        "selected_audio_path": vocal_path,
        "selected_vocal_path": vocal_path,
        "selected_minus_path": minus_path,
        "accepted_count": 1,
        "accepted_listen_preview_rows": [row],
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "release_ready": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": {
            "training_started": False,
            "fine_tune_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "cleanup_routes_added": False,
            "thresholds_relaxed": False,
            "release_ready": False,
        },
    }


def test_pair_candidate_accepts_segment_preview_with_exact_mix_provenance(tmp_path: Path) -> None:
    source_manifest = write_json(tmp_path / "segment_preview_listen_manifest.json", accepted_segment_preview_manifest(tmp_path))

    manifest = build_pair_candidate_manifest(segment_preview_manifest_json=source_manifest)

    assert manifest["delivery_status"] == "ready_for_pair_judge"
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "candidate_rows_ready_for_pair_judge"
    assert manifest["release_status"] == "listen_preview_only"
    assert manifest["stem_type"] == "both"
    assert manifest["candidate_count"] == 1
    row = manifest["candidate_rows"][0]
    source_row = json.loads(source_manifest.read_text(encoding="utf-8"))["accepted_listen_preview_rows"][0]
    assert row["mix_path"] == source_row["mix_path"]
    assert row["source_audio_path"] == source_row["source_audio_path"]
    assert row["vocal_path"] == source_row["selected_vocal_path"]
    assert row["instrumental_path"] == source_row["selected_minus_path"]
    assert row["minus_path"] == source_row["selected_minus_path"]
    assert row["requires_pair_compatibility_judge"] is True
    assert row["can_mark_release_ready"] is False
    assert row["audio_exported"] is False


def test_pair_candidate_blocks_segment_preview_without_mix_path(tmp_path: Path) -> None:
    source_manifest = write_json(
        tmp_path / "segment_preview_listen_manifest.json",
        accepted_segment_preview_manifest(tmp_path, include_mix_path=False),
    )

    manifest = build_pair_candidate_manifest(segment_preview_manifest_json=source_manifest)

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "no_pair_judge_ready_segment_preview_rows"
    assert manifest["candidate_rows"] == []
    assert manifest["skipped_rows"][0]["skip_reason"] == "missing_source_mix_path"


def test_cli_writes_segment_preview_pair_candidate_jsonl(tmp_path: Path) -> None:
    output_jsonl = tmp_path / "segment_preview_pair_candidate_rows.jsonl"
    summary_json = tmp_path / "segment_preview_pair_candidate_manifest.json"
    summary_md = tmp_path / "segment_preview_pair_candidate_summary.md"

    exit_code = main(
        [
            "--segment-preview-manifest-json",
            str(write_json(tmp_path / "segment_preview_listen_manifest.json", accepted_segment_preview_manifest(tmp_path))),
            "--output-jsonl",
            str(output_jsonl),
            "--summary-json",
            str(summary_json),
            "--summary-md",
            str(summary_md),
        ]
    )

    assert exit_code == 0
    assert len(read_jsonl(output_jsonl)) == 1
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["delivery_status"] == "ready_for_pair_judge"
    assert "delivery_status: ready_for_pair_judge" in summary_md.read_text(encoding="utf-8")
