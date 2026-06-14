from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.judge_no_reference_segment_blend_preview import (
    build_preview_window_judge,
    classify_preview_decision,
    main,
    score,
    write_outputs,
)


def write_wav(path: Path, audio: np.ndarray, *, sample_rate: int = 16_000) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), sample_rate)
    return str(path)


def sine(seconds: float = 2.0, *, sample_rate: int = 16_000, amplitude: float = 0.2) -> np.ndarray:
    samples = int(seconds * sample_rate)
    t = np.arange(samples, dtype=np.float32) / float(sample_rate)
    mono = amplitude * np.sin(2.0 * np.pi * 220.0 * t)
    return np.stack([mono, mono], axis=1).astype(np.float32)


def fixture_rows(tmp_path: Path, *, same_paths: bool = False, noop_preview: bool = False) -> tuple[dict, dict, dict]:
    base_vocal_audio = sine(amplitude=0.2)
    base_minus_audio = sine(amplitude=0.1)
    alt_vocal_audio = base_vocal_audio if same_paths else sine(amplitude=0.25)
    alt_minus_audio = base_minus_audio if same_paths else sine(amplitude=0.12)
    preview_vocal_audio = base_vocal_audio if noop_preview else alt_vocal_audio
    preview_minus_audio = base_minus_audio if noop_preview else alt_minus_audio
    mix_audio = base_vocal_audio + base_minus_audio

    base_vocal = write_wav(tmp_path / "base_vocal.wav", base_vocal_audio)
    base_minus = write_wav(tmp_path / "base_minus.wav", base_minus_audio)
    alt_vocal = base_vocal if same_paths else write_wav(tmp_path / "alt_vocal.wav", alt_vocal_audio)
    alt_minus = base_minus if same_paths else write_wav(tmp_path / "alt_minus.wav", alt_minus_audio)
    preview_vocal = write_wav(tmp_path / "segment_blend_previews" / "preview_vocal.flac", preview_vocal_audio)
    preview_minus = write_wav(tmp_path / "segment_blend_previews" / "preview_minus.flac", preview_minus_audio)
    mix = write_wav(tmp_path / "mix.wav", mix_audio)

    preview_manifest = {
        "rows": [
            {
                "track_id": "track-a",
                "base_candidate_id": "same-id",
                "alternate_candidate_id": "same-id",
                "rendered_vocal_preview_path": preview_vocal,
                "rendered_minus_preview_path": preview_minus,
                "source_stem": "both",
                "window_start_sec": 0.25,
                "window_duration_sec": 1.0,
                "crossfade_ms": 80,
                "can_publish_as_result": False,
                "can_mark_release_ready": False,
            }
        ]
    }
    contract_row = {
        "track_id": "track-a",
        "base_candidate_id": "same-id",
        "alternate_candidate_id": "same-id",
        "base_vocal_path": base_vocal,
        "base_minus_path": base_minus,
        "alt_vocal_path": alt_vocal,
        "alt_minus_path": alt_minus,
        "source_stem": "both",
        "window_start_sec": 0.25,
        "window_duration_sec": 1.0,
        "render_status": "render_ready",
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
    }
    windowed_row = {
        "track_id": "track-a",
        "candidate_id": "same-id",
        "mix_path": mix,
        "vocal_path": base_vocal,
        "instrumental_path": base_minus,
    }
    return preview_manifest, contract_row, windowed_row


def test_detects_same_candidate_id(tmp_path: Path) -> None:
    preview_manifest, contract_row, windowed_row = fixture_rows(tmp_path)

    result = build_preview_window_judge(preview_manifest, [contract_row], [contract_row], [windowed_row])

    assert result["rows"][0]["base_alt_same_candidate_id"] is True


def test_detects_identical_base_alt_paths(tmp_path: Path) -> None:
    preview_manifest, contract_row, windowed_row = fixture_rows(tmp_path, same_paths=True)

    result = build_preview_window_judge(preview_manifest, [contract_row], [contract_row], [windowed_row])
    row = result["rows"][0]

    assert row["base_alt_same_vocal_path"] is True
    assert row["base_alt_same_minus_path"] is True
    assert row["preview_decision"] == "preview_noop_self_blend"
    assert row["allowed_next_step"] == "fix_candidate_identity/path-distinctness"


def test_detects_noop_preview(tmp_path: Path) -> None:
    preview_manifest, contract_row, windowed_row = fixture_rows(tmp_path, noop_preview=True)

    result = build_preview_window_judge(preview_manifest, [contract_row], [contract_row], [windowed_row])
    row = result["rows"][0]

    assert row["preview_differs_from_base"] is False
    assert row["preview_decision"] == "preview_noop_self_blend"


def test_improved_preview_gets_preview_improved_window() -> None:
    assert (
        classify_preview_decision(
            base_alt_paths_identical=False,
            preview_differs_from_base=True,
            score_delta=0.025,
            blocker_delta=0,
        )[0]
        == "preview_improved_window"
    )


def test_regressed_preview_gets_preview_regressed_window() -> None:
    assert (
        classify_preview_decision(
            base_alt_paths_identical=False,
            preview_differs_from_base=True,
            score_delta=-0.025,
            blocker_delta=0,
        )[0]
        == "preview_regressed_window"
    )


def test_score_uses_source_stem_specific_mode() -> None:
    metrics = {
        "mixture_consistency_score": 0.92,
        "vocal_leakage_in_minus_score": 0.88,
        "instrumental_leakage_in_vocal_score": 0.05,
        "vocal_activity_confidence": 0.86,
        "silence_hallucination_score": 0.04,
        "robotic_artifact_score": 0.04,
        "thin_vocal_score": 0.08,
        "transient_damage_score": 0.06,
        "phase_consistency_score": 0.92,
        "confidence": 0.82,
        "reasons": ["no_reference_proxy_passed"],
    }

    assert score(metrics, "vocal") > 0.70
    assert score(metrics, "minus") < 0.70


def test_publish_release_false_and_no_stems_required(tmp_path: Path) -> None:
    preview_manifest, contract_row, windowed_row = fixture_rows(tmp_path)

    result = build_preview_window_judge(preview_manifest, [contract_row], [contract_row], [windowed_row])
    row = result["rows"][0]

    assert row["can_publish_as_result"] is False
    assert row["can_mark_release_ready"] is False
    assert result["summary"]["safety"]["reference_required"] is False
    assert result["summary"]["safety"]["human_csv_required"] is False


def test_output_is_deterministic(tmp_path: Path) -> None:
    preview_manifest, contract_row, windowed_row = fixture_rows(tmp_path)

    first = build_preview_window_judge(preview_manifest, [contract_row], [contract_row], [windowed_row])
    second = build_preview_window_judge(preview_manifest, [contract_row], [contract_row], [windowed_row])

    assert first == second


def test_cli_writes_summary_counts_matching_jsonl(tmp_path: Path) -> None:
    preview_manifest, contract_row, windowed_row = fixture_rows(tmp_path, noop_preview=True)
    preview_manifest_json = tmp_path / "preview_manifest.json"
    contract_jsonl = tmp_path / "contract.jsonl"
    dry_run_jsonl = tmp_path / "dry.jsonl"
    windowed_jsonl = tmp_path / "windowed.jsonl"
    output_jsonl = tmp_path / "out.jsonl"
    summary_md = tmp_path / "summary.md"
    preview_manifest_json.write_text(json.dumps(preview_manifest, sort_keys=True) + "\n", encoding="utf-8")
    contract_jsonl.write_text(json.dumps(contract_row, sort_keys=True) + "\n", encoding="utf-8")
    dry_run_jsonl.write_text(json.dumps(contract_row, sort_keys=True) + "\n", encoding="utf-8")
    windowed_jsonl.write_text(json.dumps(windowed_row, sort_keys=True) + "\n", encoding="utf-8")

    exit_code = main(
        [
            "--preview-manifest-json",
            str(preview_manifest_json),
            "--render-contract-jsonl",
            str(contract_jsonl),
            "--dry-run-jsonl",
            str(dry_run_jsonl),
            "--windowed-judge-jsonl",
            str(windowed_jsonl),
            "--output-jsonl",
            str(output_jsonl),
            "--summary-md",
            str(summary_md),
        ]
    )

    assert exit_code == 0
    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert "- preview rows judged: 1" in summary_md.read_text(encoding="utf-8")
