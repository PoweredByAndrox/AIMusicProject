from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.render_no_reference_segment_blend_preview import (
    build_preview_result,
    main,
    write_outputs,
)


def write_wav(path: Path, value: float, *, sample_rate: int = 16_000, seconds: float = 3.0, channels: int = 2) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(sample_rate * seconds)
    audio = np.full((frames, channels), value, dtype=np.float32)
    sf.write(path, audio, sample_rate)
    return str(path)


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dry_run_row(
    tmp_path: Path,
    *,
    track_id: str = "track-a",
    source_stem: str = "both",
    render_status: str = "render_ready",
    start: float = 0.75,
    duration: float = 1.0,
    alt_sample_rate: int = 16_000,
    alternate_weight: float | None = None,
) -> dict:
    base_vocal = write_wav(tmp_path / track_id / "base_vocal.wav", 0.1)
    base_minus = write_wav(tmp_path / track_id / "base_minus.wav", 0.2)
    alt_vocal = write_wav(tmp_path / track_id / "alt_vocal.wav", 0.7, sample_rate=alt_sample_rate)
    alt_minus = write_wav(tmp_path / track_id / "alt_minus.wav", 0.8, sample_rate=alt_sample_rate)
    row = {
        "track_id": track_id,
        "base_candidate_id": "base",
        "alternate_candidate_id": "alt",
        "source_stem": source_stem,
        "window_start_sec": start,
        "window_duration_sec": duration,
        "crossfade_ms": 80,
        "base_vocal_path": base_vocal,
        "base_minus_path": base_minus,
        "alt_vocal_path": alt_vocal,
        "alt_minus_path": alt_minus,
        "render_status": render_status,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
    }
    if alternate_weight is not None:
        row["alternate_weight"] = alternate_weight
    return row


def test_renders_only_render_ready_row_and_skips_blocked(tmp_path: Path) -> None:
    ready = dry_run_row(tmp_path, track_id="ready")
    blocked = dry_run_row(tmp_path, track_id="blocked", render_status="blocked_overlap_conflict")

    result = build_preview_result([blocked, ready], [blocked, ready], tmp_path / "previews", max_render_count=1)

    assert result["summary"]["render_ready_rows"] == 1
    assert result["summary"]["previews_rendered"] == 1
    assert result["summary"]["previews_skipped"] == 1
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["track_id"] == "ready"
    assert row["render_status"] == "render_ready"
    assert Path(row["rendered_vocal_preview_path"]).exists()
    assert Path(row["rendered_minus_preview_path"]).exists()


def test_output_path_is_preview_folder_only(tmp_path: Path) -> None:
    output_dir = tmp_path / "segment_blend_previews"
    result = build_preview_result([dry_run_row(tmp_path)], [dry_run_row(tmp_path)], output_dir, max_render_count=1)
    row = result["rows"][0]

    assert Path(row["rendered_vocal_preview_path"]).parent == output_dir
    assert Path(row["rendered_minus_preview_path"]).parent == output_dir


def test_original_files_are_not_overwritten_and_shapes_preserved(tmp_path: Path) -> None:
    row = dry_run_row(tmp_path)
    before = {
        row["base_vocal_path"]: sha256(row["base_vocal_path"]),
        row["base_minus_path"]: sha256(row["base_minus_path"]),
        row["alt_vocal_path"]: sha256(row["alt_vocal_path"]),
        row["alt_minus_path"]: sha256(row["alt_minus_path"]),
    }

    result = build_preview_result([row], [row], tmp_path / "previews", max_render_count=1)

    for path, digest in before.items():
        assert sha256(path) == digest
    manifest_row = result["rows"][0]
    base_info = sf.info(row["base_vocal_path"])
    preview_info = sf.info(manifest_row["rendered_vocal_preview_path"])
    assert preview_info.samplerate == base_info.samplerate
    assert preview_info.channels == base_info.channels
    assert preview_info.frames == base_info.frames
    assert manifest_row["sample_rate_preserved"] is True
    assert manifest_row["channels_preserved"] is True
    assert manifest_row["duration_preserved"] is True


def test_resamples_alternate_preview_to_base_sample_rate(tmp_path: Path) -> None:
    row = dry_run_row(tmp_path, source_stem="vocal", alt_sample_rate=22_050)

    result = build_preview_result([row], [row], tmp_path / "previews", max_render_count=1)

    manifest_row = result["rows"][0]
    base_info = sf.info(row["base_vocal_path"])
    preview_info = sf.info(manifest_row["rendered_vocal_preview_path"])
    assert preview_info.samplerate == base_info.samplerate
    assert preview_info.channels == base_info.channels
    assert preview_info.frames == base_info.frames
    assert manifest_row["sample_rate_resample_applied"] is True
    assert manifest_row["sample_rate_preserved"] is True
    assert manifest_row["duration_preserved"] is True


def test_partial_alternate_weight_blends_fractional_segment(tmp_path: Path) -> None:
    row = dry_run_row(
        tmp_path,
        source_stem="minus",
        start=1.0,
        duration=1.0,
        alternate_weight=0.25,
    )
    row["crossfade_ms"] = 0

    result = build_preview_result([row], [row], tmp_path / "previews", max_render_count=1)
    manifest_row = result["rows"][0]
    minus, sample_rate = sf.read(manifest_row["rendered_minus_preview_path"], always_2d=True)

    assert sample_rate == 16_000
    assert manifest_row["alternate_weight"] == 0.25
    assert np.allclose(minus[:16_000], 0.2, atol=1e-4)
    assert np.allclose(minus[16_000:32_000], 0.35, atol=1e-4)
    assert np.allclose(minus[32_000:], 0.2, atol=1e-4)


def test_release_flags_false_and_post_render_judge_required(tmp_path: Path) -> None:
    result = build_preview_result([dry_run_row(tmp_path)], [dry_run_row(tmp_path)], tmp_path / "previews", max_render_count=1)
    row = result["rows"][0]

    assert row["can_publish_as_result"] is False
    assert row["can_mark_release_ready"] is False
    assert row["requires_post_render_judge"] is True
    assert result["summary"]["can_publish_as_result_count"] == 0
    assert result["summary"]["can_mark_release_ready_count"] == 0
    assert result["summary"]["requires_post_render_judge_count"] == 1


def test_manifest_is_deterministic_and_max_render_count_respected(tmp_path: Path) -> None:
    rows = [
        dry_run_row(tmp_path, track_id="b"),
        dry_run_row(tmp_path, track_id="a"),
    ]

    first = build_preview_result(list(reversed(rows)), list(reversed(rows)), tmp_path / "previews", max_render_count=1)
    second = build_preview_result(rows, rows, tmp_path / "previews", max_render_count=1)

    assert first == second
    assert first["summary"]["render_ready_rows"] == 2
    assert first["summary"]["previews_rendered"] == 1
    assert first["summary"]["previews_skipped"] == 1
    assert first["rows"][0]["track_id"] == "a"


def test_long_candidate_ids_do_not_create_overlong_preview_filenames(tmp_path: Path) -> None:
    row = dry_run_row(tmp_path)
    row["base_candidate_id"] = "base_" + ("very_long_candidate_name_" * 8)
    row["alternate_candidate_id"] = "alt_" + ("another_very_long_candidate_name_" * 8)

    result = build_preview_result([row], [row], tmp_path / "previews", max_render_count=1)

    manifest_row = result["rows"][0]
    assert len(Path(manifest_row["rendered_vocal_preview_path"]).name) < 160
    assert len(Path(manifest_row["rendered_minus_preview_path"]).name) < 160
    assert Path(manifest_row["rendered_vocal_preview_path"]).exists()
    assert Path(manifest_row["rendered_minus_preview_path"]).exists()


def test_cli_writes_manifest_and_summary(tmp_path: Path) -> None:
    row = dry_run_row(tmp_path)
    dry_run_jsonl = tmp_path / "dry.jsonl"
    contract_jsonl = tmp_path / "contract.jsonl"
    dry_run_jsonl.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    contract_jsonl.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    output_dir = tmp_path / "segment_blend_previews"
    manifest_json = tmp_path / "manifest.json"
    summary_md = tmp_path / "summary.md"

    exit_code = main(
        [
            "--dry-run-jsonl",
            str(dry_run_jsonl),
            "--render-contract-jsonl",
            str(contract_jsonl),
            "--output-dir",
            str(output_dir),
            "--manifest-json",
            str(manifest_json),
            "--summary-md",
            str(summary_md),
            "--max-render-count",
            "1",
        ]
    )

    assert exit_code == 0
    payload = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 1
    assert "- previews_rendered: 1" in summary_md.read_text(encoding="utf-8")
