from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.artifact_status_contract import can_promote_to_listen_result
from tools.build_no_reference_segment_preview_listen_manifest import build_manifest, main


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
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def preview_row(tmp_path: Path, *, source_stem: str = "vocal") -> dict:
    return {
        "track_id": "track-a",
        "base_candidate_id": "base",
        "alternate_candidate_id": "alt",
        "source_stem": source_stem,
        "window_start_sec": 12.0,
        "window_duration_sec": 6.0,
        "rendered_vocal_preview_path": write_wav(tmp_path / "preview_vocal.flac", 0.2),
        "rendered_minus_preview_path": write_wav(tmp_path / "preview_minus.flac", 0.3),
        "render_status": "render_ready",
        "requires_post_render_judge": True,
        "sample_rate_preserved": True,
        "channels_preserved": True,
        "duration_preserved": True,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
    }


def contract_row(row: dict) -> dict:
    return {
        "track_id": row["track_id"],
        "base_candidate_id": row["base_candidate_id"],
        "alternate_candidate_id": row["alternate_candidate_id"],
        "source_stem": row["source_stem"],
        "window_start_sec": row["window_start_sec"],
        "window_duration_sec": row["window_duration_sec"],
        "render_status": "render_ready",
        "can_render_preview_later": True,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "base_vocal_path": "/tmp/base_vocal.flac",
        "base_minus_path": "/tmp/base_minus.flac",
        "alt_vocal_path": "/tmp/alt_vocal.flac",
        "alt_minus_path": "/tmp/alt_minus.flac",
    }


def judge_row(row: dict, *, decision: str = "preview_improved_window") -> dict:
    return {
        "track_id": row["track_id"],
        "base_candidate_id": row["base_candidate_id"],
        "alternate_candidate_id": row["alternate_candidate_id"],
        "source_stem": row["source_stem"],
        "mix_path": "/tmp/mix.wav",
        "source_audio_path": "/tmp/mix.wav",
        "base_vocal_path": "/tmp/base_vocal.flac",
        "base_minus_path": "/tmp/base_minus.flac",
        "alt_vocal_path": "/tmp/alt_vocal.flac",
        "alt_minus_path": "/tmp/alt_minus.flac",
        "window_start_sec": row["window_start_sec"],
        "window_duration_sec": row["window_duration_sec"],
        "preview_vocal_path": row["rendered_vocal_preview_path"],
        "preview_minus_path": row["rendered_minus_preview_path"],
        "preview_decision": decision,
        "allowed_next_step": "expand_preview_policy" if decision == "preview_improved_window" else "discard_preview_and_fix_planner",
        "base_window_score": 0.82,
        "preview_window_score": 0.84 if decision == "preview_improved_window" else 0.80,
        "score_delta": 0.02 if decision == "preview_improved_window" else -0.02,
        "blocker_delta": 0,
        "preview_differs_from_base": True,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
    }


def test_accepts_only_post_render_improved_preview_as_listen_manifest(tmp_path: Path) -> None:
    row = preview_row(tmp_path)
    manifest = build_manifest(
        preview_manifest_json=write_json(tmp_path / "preview_manifest.json", {"rows": [row]}),
        preview_window_judge_jsonl=write_jsonl(tmp_path / "preview_judge.jsonl", [judge_row(row)]),
        render_contract_jsonl=write_jsonl(tmp_path / "contract.jsonl", [contract_row(row)]),
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["presentation_status"] == "accepted_for_listen_pack"
    assert manifest["release_status"] == "listen_preview_only"
    assert manifest["stem_type"] == "vocal"
    assert manifest["accepted_count"] == 1
    assert manifest["can_mark_release_ready"] is False
    assert manifest["can_publish_as_release"] is False
    assert manifest["mix_path"] == "/tmp/mix.wav"
    assert manifest["source_audio_path"] == "/tmp/mix.wav"
    assert manifest["selected_vocal_path"] == row["rendered_vocal_preview_path"]
    accepted = manifest["accepted_listen_preview_rows"][0]
    assert accepted["mix_path"] == "/tmp/mix.wav"
    assert accepted["source_audio_path"] == "/tmp/mix.wav"
    assert accepted["base_vocal_path"] == "/tmp/base_vocal.flac"
    assert accepted["alt_vocal_path"] == "/tmp/alt_vocal.flac"

    output_manifest = tmp_path / "listen_manifest.json"
    write_json(output_manifest, manifest)
    assert can_promote_to_listen_result(output_manifest) is True


def test_blocks_regressed_preview_rows(tmp_path: Path) -> None:
    row = preview_row(tmp_path)
    manifest = build_manifest(
        preview_manifest_json=write_json(tmp_path / "preview_manifest.json", {"rows": [row]}),
        preview_window_judge_jsonl=write_jsonl(tmp_path / "preview_judge.jsonl", [judge_row(row, decision="preview_regressed_window")]),
        render_contract_jsonl=write_jsonl(tmp_path / "contract.jsonl", [contract_row(row)]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "no_post_render_improved_preview_rows"
    assert manifest["skipped_rows"][0]["skip_reason"] == "preview_judge_not_improved"


def test_blocks_when_judge_paths_do_not_match_preview_manifest(tmp_path: Path) -> None:
    row = preview_row(tmp_path)
    mismatched_judge = judge_row(row)
    mismatched_judge["preview_vocal_path"] = write_wav(tmp_path / "other_preview.flac", 0.4)

    manifest = build_manifest(
        preview_manifest_json=write_json(tmp_path / "preview_manifest.json", {"rows": [row]}),
        preview_window_judge_jsonl=write_jsonl(tmp_path / "preview_judge.jsonl", [mismatched_judge]),
        render_contract_jsonl=write_jsonl(tmp_path / "contract.jsonl", [contract_row(row)]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["skipped_rows"][0]["skip_reason"] == "preview_judge_missing"


def test_blocks_missing_preview_audio(tmp_path: Path) -> None:
    row = preview_row(tmp_path)
    Path(row["rendered_vocal_preview_path"]).unlink()

    manifest = build_manifest(
        preview_manifest_json=write_json(tmp_path / "preview_manifest.json", {"rows": [row]}),
        preview_window_judge_jsonl=write_jsonl(tmp_path / "preview_judge.jsonl", [judge_row(row)]),
        render_contract_jsonl=write_jsonl(tmp_path / "contract.jsonl", [contract_row(row)]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["skipped_rows"][0]["skip_reason"] == "missing_preview_audio"


def test_cli_writes_promotable_manifest_and_summary(tmp_path: Path) -> None:
    row = preview_row(tmp_path)
    output_manifest = tmp_path / "segment_preview_listen_manifest.json"
    summary_md = tmp_path / "summary.md"

    exit_code = main(
        [
            "--preview-manifest-json",
            str(write_json(tmp_path / "preview_manifest.json", {"rows": [row]})),
            "--preview-window-judge-jsonl",
            str(write_jsonl(tmp_path / "preview_judge.jsonl", [judge_row(row)])),
            "--render-contract-jsonl",
            str(write_jsonl(tmp_path / "contract.jsonl", [contract_row(row)])),
            "--output-manifest-json",
            str(output_manifest),
            "--summary-md",
            str(summary_md),
        ]
    )

    assert exit_code == 0
    assert can_promote_to_listen_result(output_manifest) is True
    assert "delivery_status: accepted" in summary_md.read_text(encoding="utf-8")
