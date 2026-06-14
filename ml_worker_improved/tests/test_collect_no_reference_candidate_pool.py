from __future__ import annotations

import json
from pathlib import Path

from tools.collect_no_reference_candidate_pool import collect_candidate_pool, write_outputs


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def touch_audio(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-audio-but-existing-artifact")
    return str(path)


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def vocal_manifest(root: Path, track: str, name: str, *, vocal_path: str | None = None) -> Path:
    mix_path = touch_audio(root / track / "source.mp3")
    vocal = vocal_path or touch_audio(root / track / name / "vocals.flac")
    return write_json(
        root / track / name / "stem_manifest.json",
        {
            "audio_path": mix_path,
            "pipeline": {"candidate_mode": name},
            "stems": {"vocals": {"path": vocal}},
        },
    )


def minus_manifest(root: Path, track: str, name: str, *, minus_path: str | None = None) -> Path:
    mix_path = touch_audio(root / track / "source.mp3")
    minus = minus_path or touch_audio(root / track / name / "minus.flac")
    return write_json(
        root / track / name / "stem_manifest.json",
        {
            "raw_audio_path": mix_path,
            "selected_profile": name,
            "artifacts": {"repaired_minus_flac": {"path": minus}},
        },
    )


def test_collects_multiple_candidates_for_one_track(tmp_path: Path) -> None:
    vocal_manifest(tmp_path, "track-a", "vocal-a")
    vocal_manifest(tmp_path, "track-a", "vocal-b")
    minus_manifest(tmp_path, "track-a", "minus-a")

    result = collect_candidate_pool(tmp_path)

    assert result["summary"]["complete_candidate_pairs"] == 2
    assert result["summary"]["tracks_with_2plus_candidates"] == 1
    assert {row["track_id"] for row in result["rows"]} == {"track-a"}
    assert all(row["is_complete_pair"] is True for row in result["rows"])
    assert all(row["mix_path"] and row["vocal_path"] and row["minus_path"] for row in result["rows"])


def test_skips_incomplete_candidate_with_reason(tmp_path: Path) -> None:
    vocal_manifest(tmp_path, "track-vocal-only", "vocal-a")

    result = collect_candidate_pool(tmp_path)

    assert result["rows"] == []
    assert result["summary"]["skipped_incomplete"] == 1
    assert result["skipped"][0]["track_id"] == "track-vocal-only"
    assert result["skipped"][0]["skip_reason"] == "missing_minus_path"


def test_does_not_classify_minus_path_as_vocal_from_vocal_named_key(tmp_path: Path) -> None:
    mix_path = touch_audio(tmp_path / "track-a" / "source.mp3")
    minus_path = touch_audio(tmp_path / "track-a" / "instrumental_resurrection_minus.flac")
    write_json(
        tmp_path / "track-a" / "manifest.json",
        {
            "source_audio_path": mix_path,
            "candidate_profiles": [
                {
                    "name": "vocal_harshness_guard",
                    "instrumental_path": minus_path,
                }
            ],
        },
    )

    result = collect_candidate_pool(tmp_path)

    assert result["rows"] == []
    assert result["skipped"][0]["skip_reason"] == "missing_vocal_path"


def test_deduplicates_identical_pairs(tmp_path: Path) -> None:
    shared_vocal = touch_audio(tmp_path / "track-a" / "shared" / "vocals.flac")
    shared_minus = touch_audio(tmp_path / "track-a" / "shared" / "minus.flac")
    vocal_manifest(tmp_path, "track-a", "vocal-a", vocal_path=shared_vocal)
    vocal_manifest(tmp_path, "track-a", "vocal-duplicate", vocal_path=shared_vocal)
    minus_manifest(tmp_path, "track-a", "minus-a", minus_path=shared_minus)
    minus_manifest(tmp_path, "track-a", "minus-duplicate", minus_path=shared_minus)

    result = collect_candidate_pool(tmp_path)

    assert result["summary"]["complete_candidate_pairs"] == 1
    assert result["summary"]["deduplicated_pairs"] == 3


def test_does_not_require_reference_stems(tmp_path: Path) -> None:
    vocal_manifest(tmp_path, "track-a", "vocal-a")
    minus_manifest(tmp_path, "track-a", "minus-a")

    result = collect_candidate_pool(tmp_path)
    row = result["rows"][0]

    assert "reference_vocal_path" not in row
    assert "reference_minus_path" not in row
    assert result["summary"]["safety"]["reference_required"] is False
    assert result["summary"]["safety"]["human_csv_required"] is False
    assert result["summary"]["safety"]["training_started"] is False
    assert result["summary"]["safety"]["models_changed"] is False
    assert result["summary"]["safety"]["cleanup_routes_added"] is False
    assert result["summary"]["safety"]["audio_exported"] is False


def test_collects_existing_preview_candidate_jsonl_pair(tmp_path: Path) -> None:
    mix_path = touch_audio(tmp_path / "track-a" / "source.mp3")
    vocal_path = touch_audio(tmp_path / "track-a" / "segment_blend_previews" / "vocal_preview.flac")
    minus_path = touch_audio(tmp_path / "track-a" / "segment_blend_previews" / "minus_preview.flac")
    preview_jsonl = tmp_path / "track-a" / "mode_specific_v19_preview_candidate_rejudge" / "preview_vs_base_candidates.jsonl"
    preview_jsonl.parent.mkdir(parents=True, exist_ok=True)
    preview_jsonl.write_text(
        json.dumps(
            {
                "track_id": "track-a",
                "candidate_id": "preview_segment_blend_window_0001",
                "mix_path": mix_path,
                "vocal_path": vocal_path,
                "instrumental_path": minus_path,
                "can_publish_as_result": False,
                "can_mark_release_ready": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = collect_candidate_pool(tmp_path)

    assert result["summary"]["complete_candidate_pairs"] == 1
    row = result["rows"][0]
    assert row["candidate_id"] == "preview_segment_blend_window_0001"
    assert row["candidate_origin"] == "existing_segment_blend_preview_pair"
    assert row["candidate_family"] == "preview_candidate_pair"
    assert row["is_complete_pair"] is True
    assert row["skip_reason"] == ""
    assert row["score_hints"]["can_publish_as_result"] is False
    assert row["score_hints"]["can_mark_release_ready"] is False


def test_collects_segment_blend_preview_judge_input_pair(tmp_path: Path) -> None:
    mix_path = touch_audio(tmp_path / "track-a" / "source.mp3")
    vocal_path = touch_audio(tmp_path / "track-a" / "segment_blend_previews" / "vocal_preview.flac")
    minus_path = touch_audio(tmp_path / "track-a" / "segment_blend_previews" / "minus_preview.flac")
    preview_input = tmp_path / "track-a" / "no_reference_segment_blend_preview_judge_input.jsonl"
    preview_input.write_text(
        json.dumps(
            {
                "track_id": "track-a",
                "candidate_id": "nr7_segment_blend_preview",
                "mix_path": mix_path,
                "vocal_path": vocal_path,
                "instrumental_path": minus_path,
                "can_publish_as_result": False,
                "can_mark_release_ready": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = collect_candidate_pool(tmp_path)

    assert result["summary"]["complete_candidate_pairs"] == 1
    row = result["rows"][0]
    assert row["candidate_id"] == "nr7_segment_blend_preview"
    assert row["candidate_origin"] == "existing_segment_blend_preview_pair"
    assert row["candidate_family"] == "preview_candidate_pair"
    assert row["is_complete_pair"] is True


def test_preview_word_in_candidate_id_does_not_force_segment_blend_origin(tmp_path: Path) -> None:
    mix_path = touch_audio(tmp_path / "track-a" / "source.mp3")
    vocal_path = touch_audio(tmp_path / "track-a" / "ordinary" / "vocal_preview_named.flac")
    minus_path = touch_audio(tmp_path / "track-a" / "ordinary" / "minus.flac")
    preview_jsonl = tmp_path / "track-a" / "preview_vs_base_candidates.jsonl"
    preview_jsonl.write_text(
        json.dumps(
            {
                "track_id": "track-a",
                "candidate_id": "ordinary_preview_named_candidate",
                "mix_path": mix_path,
                "vocal_path": vocal_path,
                "instrumental_path": minus_path,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = collect_candidate_pool(tmp_path)

    assert result["rows"][0]["candidate_origin"] == "existing_no_reference_candidate_jsonl_pair"


def test_collects_only_improved_preview_window_judge_rows(tmp_path: Path) -> None:
    mix_path = touch_audio(tmp_path / "track-a" / "source.mp3")
    improved_vocal = touch_audio(tmp_path / "track-a" / "segment_blend_previews" / "improved_vocal.flac")
    improved_minus = touch_audio(tmp_path / "track-a" / "segment_blend_previews" / "improved_minus.flac")
    regressed_vocal = touch_audio(tmp_path / "track-a" / "segment_blend_previews" / "regressed_vocal.flac")
    regressed_minus = touch_audio(tmp_path / "track-a" / "segment_blend_previews" / "regressed_minus.flac")
    judge_jsonl = tmp_path / "track-a" / "no_reference_segment_blend_preview_window_judge_resolved.jsonl"
    rows = [
        {
            "track_id": "track-a",
            "mix_path": mix_path,
            "preview_vocal_path": improved_vocal,
            "preview_minus_path": improved_minus,
            "preview_decision": "preview_improved_window",
            "allowed_next_step": "expand_preview_policy",
            "score_delta": 0.011,
            "base_window_score": 0.82,
            "preview_window_score": 0.831,
            "source_stem": "vocal",
            "window_start_sec": 10.0,
            "window_duration_sec": 5.0,
            "base_candidate_id": "base",
            "alternate_candidate_id": "alt",
            "can_publish_as_result": False,
            "can_mark_release_ready": False,
        },
        {
            "track_id": "track-a",
            "mix_path": mix_path,
            "preview_vocal_path": regressed_vocal,
            "preview_minus_path": regressed_minus,
            "preview_decision": "preview_regressed_window",
            "allowed_next_step": "discard_preview_and_fix_planner",
            "score_delta": -0.01,
            "source_stem": "vocal",
            "window_start_sec": 20.0,
            "window_duration_sec": 5.0,
            "base_candidate_id": "base",
            "alternate_candidate_id": "bad-alt",
            "can_publish_as_result": False,
            "can_mark_release_ready": False,
        },
    ]
    judge_jsonl.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")

    result = collect_candidate_pool(tmp_path)

    assert result["summary"]["complete_candidate_pairs"] == 1
    assert result["summary"]["skipped_incomplete"] == 1
    row = result["rows"][0]
    assert row["candidate_origin"] == "existing_segment_blend_preview_improved_window"
    assert row["candidate_family"] == "preview_candidate_pair"
    assert row["vocal_path"] == improved_vocal
    assert row["minus_path"] == improved_minus
    assert row["score_hints"]["preview_decision"] == "preview_improved_window"
    assert result["skipped"][0]["skip_reason"] == "not_preview_improved_window"


def test_unsafe_preview_window_judge_row_is_not_collected(tmp_path: Path) -> None:
    mix_path = touch_audio(tmp_path / "track-a" / "source.mp3")
    vocal_path = touch_audio(tmp_path / "track-a" / "segment_blend_previews" / "vocal.flac")
    minus_path = touch_audio(tmp_path / "track-a" / "segment_blend_previews" / "minus.flac")
    judge_jsonl = tmp_path / "track-a" / "no_reference_segment_blend_preview_window_judge.jsonl"
    judge_jsonl.write_text(
        json.dumps(
            {
                "track_id": "track-a",
                "mix_path": mix_path,
                "preview_vocal_path": vocal_path,
                "preview_minus_path": minus_path,
                "preview_decision": "preview_improved_window",
                "source_stem": "vocal",
                "window_start_sec": 10.0,
                "window_duration_sec": 5.0,
                "base_candidate_id": "base",
                "alternate_candidate_id": "alt",
                "can_publish_as_result": True,
                "can_mark_release_ready": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = collect_candidate_pool(tmp_path)

    assert result["rows"] == []
    assert result["summary"]["skipped_incomplete"] == 1
    assert result["skipped"][0]["skip_reason"] == "unsafe_preview_promotion_flags"


def test_noop_preview_window_judge_row_is_not_collected(tmp_path: Path) -> None:
    mix_path = touch_audio(tmp_path / "track-a" / "source.mp3")
    vocal_path = touch_audio(tmp_path / "track-a" / "segment_blend_previews" / "noop_vocal.flac")
    minus_path = touch_audio(tmp_path / "track-a" / "segment_blend_previews" / "noop_minus.flac")
    judge_jsonl = tmp_path / "track-a" / "no_reference_segment_blend_preview_window_judge.jsonl"
    judge_jsonl.write_text(
        json.dumps(
            {
                "track_id": "track-a",
                "mix_path": mix_path,
                "preview_vocal_path": vocal_path,
                "preview_minus_path": minus_path,
                "preview_decision": "preview_noop_self_blend",
                "allowed_next_step": "fix_candidate_identity/path-distinctness",
                "source_stem": "minus",
                "window_start_sec": 12.0,
                "window_duration_sec": 5.0,
                "base_candidate_id": "base",
                "alternate_candidate_id": "same",
                "can_publish_as_result": False,
                "can_mark_release_ready": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = collect_candidate_pool(tmp_path)

    assert result["rows"] == []
    assert result["summary"]["skipped_incomplete"] == 1
    assert result["skipped"][0]["skip_reason"] == "noop_self_blend_preview_window"
    assert result["skipped"][0]["score_hints"]["preview_decision"] == "preview_noop_self_blend"


def test_inconsistent_improved_noop_preview_window_judge_row_is_not_collected(tmp_path: Path) -> None:
    mix_path = touch_audio(tmp_path / "track-a" / "source.mp3")
    vocal_path = touch_audio(tmp_path / "track-a" / "segment_blend_previews" / "noop_vocal.flac")
    minus_path = touch_audio(tmp_path / "track-a" / "segment_blend_previews" / "noop_minus.flac")
    judge_jsonl = tmp_path / "track-a" / "no_reference_segment_blend_preview_window_judge.jsonl"
    judge_jsonl.write_text(
        json.dumps(
            {
                "track_id": "track-a",
                "mix_path": mix_path,
                "preview_vocal_path": vocal_path,
                "preview_minus_path": minus_path,
                "preview_decision": "preview_improved_window",
                "preview_differs_from_base": False,
                "allowed_next_step": "fix_candidate_identity/path-distinctness",
                "source_stem": "minus",
                "window_start_sec": 12.0,
                "window_duration_sec": 5.0,
                "base_candidate_id": "base",
                "alternate_candidate_id": "same",
                "can_publish_as_result": False,
                "can_mark_release_ready": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = collect_candidate_pool(tmp_path)

    assert result["rows"] == []
    assert result["summary"]["skipped_incomplete"] == 1
    assert result["skipped"][0]["skip_reason"] == "noop_self_blend_preview_window"


def test_explicit_candidate_pool_seed_merges_base_and_preview_rows(tmp_path: Path) -> None:
    mix_path = touch_audio(tmp_path / "track-a" / "source.mp3")
    base_vocal = touch_audio(tmp_path / "track-a" / "base" / "vocals.flac")
    base_minus = touch_audio(tmp_path / "track-a" / "base" / "minus.flac")
    preview_vocal = touch_audio(tmp_path / "track-a" / "segment_blend_previews" / "vocal_preview.flac")
    preview_minus = touch_audio(tmp_path / "track-a" / "segment_blend_previews" / "minus_preview.flac")
    seed_pool = write_jsonl(
        tmp_path / "seed_pool.jsonl",
        [
            {
                "track_id": "track-a",
                "candidate_id": "base_pair",
                "mix_path": mix_path,
                "vocal_path": base_vocal,
                "instrumental_path": base_minus,
                "source_manifest": "base_manifest.json",
                "candidate_family": "vocal+minus",
                "candidate_origin": "existing_generated_artifact_pair",
                "score_hints": {"source": "seed"},
                "is_complete_pair": True,
            }
        ],
    )
    judge_jsonl = write_jsonl(
        tmp_path / "track-a" / "no_reference_segment_blend_preview_window_judge.jsonl",
        [
            {
                "track_id": "track-a",
                "mix_path": mix_path,
                "preview_vocal_path": preview_vocal,
                "preview_minus_path": preview_minus,
                "preview_decision": "preview_improved_window",
                "allowed_next_step": "expand_preview_policy",
                "source_stem": "vocal",
                "window_start_sec": 42.0,
                "window_duration_sec": 6.0,
                "base_candidate_id": "base_pair",
                "alternate_candidate_id": "alt_pair",
                "can_publish_as_result": False,
                "can_mark_release_ready": False,
            }
        ],
    )
    assert judge_jsonl.exists()

    result = collect_candidate_pool(
        tmp_path,
        seed_candidate_pool_jsonls=[seed_pool],
        max_pairs_per_track=0,
        max_total_pairs=0,
    )

    assert result["summary"]["seed_candidate_pairs"] == 1
    assert result["summary"]["complete_candidate_pairs"] == 2
    assert result["summary"]["tracks_with_2plus_candidates"] == 1
    assert {row["candidate_id"] for row in result["rows"]} == {
        "base_pair",
        "preview_improved_window__vocal__42.0__base_pair__alt_pair",
    }
    seed_row = next(row for row in result["rows"] if row["candidate_id"] == "base_pair")
    assert seed_row["candidate_origin"] == "existing_generated_artifact_pair"
    assert seed_row["minus_path"] == base_minus
    assert seed_row["instrumental_path"] == base_minus
    assert result["summary"]["safety"]["audio_read"] is False
    assert result["summary"]["safety"]["audio_exported"] is False


def test_skips_incomplete_preview_candidate_jsonl_with_reason(tmp_path: Path) -> None:
    mix_path = touch_audio(tmp_path / "track-a" / "source.mp3")
    vocal_path = touch_audio(tmp_path / "track-a" / "segment_blend_previews" / "vocal_preview.flac")
    preview_jsonl = tmp_path / "track-a" / "preview_vs_base_candidates.jsonl"
    preview_jsonl.write_text(
        json.dumps(
            {
                "track_id": "track-a",
                "candidate_id": "preview_segment_blend_window_0001",
                "mix_path": mix_path,
                "vocal_path": vocal_path,
                "instrumental_path": "",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = collect_candidate_pool(tmp_path)

    assert result["rows"] == []
    assert result["summary"]["skipped_incomplete"] == 1
    assert result["skipped"][0]["candidate_id"] == "preview_segment_blend_window_0001"
    assert result["skipped"][0]["skip_reason"] == "missing_minus_path"


def test_output_is_deterministic_and_summary_matches_jsonl(tmp_path: Path) -> None:
    vocal_manifest(tmp_path, "track-a", "vocal-b")
    vocal_manifest(tmp_path, "track-a", "vocal-a")
    minus_manifest(tmp_path, "track-a", "minus-a")
    vocal_manifest(tmp_path, "track-b", "vocal-a")
    minus_manifest(tmp_path, "track-b", "minus-a")

    first = collect_candidate_pool(tmp_path)
    second = collect_candidate_pool(tmp_path)
    assert first == second

    output_jsonl = tmp_path / "out" / "pool.jsonl"
    summary_md = tmp_path / "out" / "summary.md"
    skipped_jsonl = tmp_path / "out" / "skipped.jsonl"
    write_outputs(first, output_jsonl, summary_md, skipped_jsonl)
    output_rows = [
        json.loads(line)
        for line in output_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    skipped_rows = [
        json.loads(line)
        for line in skipped_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(output_rows) == first["summary"]["complete_candidate_pairs"]
    assert len(skipped_rows) == first["summary"]["skipped_incomplete"]
    assert not list((tmp_path / "out").rglob("*.wav"))
    assert not list((tmp_path / "out").rglob("*.flac"))
