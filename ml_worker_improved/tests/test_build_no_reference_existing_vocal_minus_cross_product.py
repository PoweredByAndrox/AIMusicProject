from __future__ import annotations

import json
from pathlib import Path

from tools.build_no_reference_existing_vocal_minus_cross_product import build_cross_product_pairs, main


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def candidate_row(
    candidate_id: str,
    *,
    track_id: str = "track-a",
    vocal_path: str = "/vocals/a.flac",
    minus_path: str = "/minus/a.flac",
    mix_path: str = "/mix/a.mp3",
) -> dict:
    return {
        "track_id": track_id,
        "candidate_id": candidate_id,
        "mix_path": mix_path,
        "vocal_path": vocal_path,
        "instrumental_path": minus_path,
        "minus_path": minus_path,
        "candidate_family": "vocal+minus",
        "candidate_origin": "existing_generated_artifact_pair",
        "is_complete_pair": True,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
    }


def test_builds_cross_product_from_candidate_pool_rows_without_phase_selection() -> None:
    vocal_rows = [
        candidate_row("vocal-a", vocal_path="/vocals/a.flac", minus_path="/old/minus-a.flac"),
        candidate_row("vocal-b", vocal_path="/vocals/b.flac", minus_path="/old/minus-b.flac"),
    ]
    minus_rows = [
        candidate_row("minus-a", vocal_path="/old/vocal-a.flac", minus_path="/minus/a.flac"),
        candidate_row("minus-b", vocal_path="/old/vocal-b.flac", minus_path="/minus/b.flac"),
    ]

    result = build_cross_product_pairs(
        vocal_rows,
        minus_rows,
        vocal_source_manifest="/pool/vocals.jsonl",
        minus_source_manifest="/pool/minuses.jsonl",
    )

    rows = result["rows"]
    assert len(rows) == 4
    assert {row["vocal_path"] for row in rows} == {"/vocals/a.flac", "/vocals/b.flac"}
    assert {row["minus_path"] for row in rows} == {"/minus/a.flac", "/minus/b.flac"}
    assert all(row["candidate_family"] == "existing_vocal_minus_cross_product" for row in rows)
    assert all(row["candidate_origin"] == "no_reference_existing_vocal_minus_cross_product" for row in rows)
    assert all(row["requires_dense_post_cross_product_judge"] is True for row in rows)
    assert all(row["can_publish_as_result"] is False for row in rows)
    assert all(row["can_mark_release_ready"] is False for row in rows)
    assert all(row["audio_copied"] is False for row in rows)
    assert all(row["audio_exported"] is False for row in rows)
    assert result["summary"]["cross_product_pair_count"] == 4
    assert result["summary"]["safety"]["thresholds_relaxed"] is False


def test_cross_product_deduplicates_and_skips_mismatched_mix_paths() -> None:
    vocal_rows = [
        candidate_row("vocal-a", vocal_path="/vocals/a.flac"),
        candidate_row("vocal-a-copy", vocal_path="/vocals/a.flac"),
    ]
    minus_rows = [
        candidate_row("minus-a", minus_path="/minus/a.flac"),
        candidate_row("minus-mismatch", minus_path="/minus/b.flac", mix_path="/mix/other.mp3"),
    ]

    result = build_cross_product_pairs(
        vocal_rows,
        minus_rows,
        vocal_source_manifest="/pool/vocals.jsonl",
        minus_source_manifest="/pool/minuses.jsonl",
    )

    assert len(result["rows"]) == 1
    assert result["summary"]["deduplicated_pair_count"] == 1
    assert {row["skip_reason"] for row in result["skipped_rows"]} == {"mix_path_mismatch"}


def test_cross_product_matches_shared_mix_path_when_track_ids_differ() -> None:
    vocal_rows = [
        candidate_row(
            "vocal-a",
            track_id="track-from-source-filename",
            vocal_path="/vocals/a.flac",
            mix_path="/mix/shared.mp3",
        )
    ]
    minus_rows = [
        candidate_row(
            "minus-a",
            track_id="track-from-artifact-folder",
            minus_path="/minus/a.flac",
            mix_path="/mix/shared.mp3",
        )
    ]

    result = build_cross_product_pairs(
        vocal_rows,
        minus_rows,
        vocal_source_manifest="/pool/vocals.jsonl",
        minus_source_manifest="/pool/minuses.jsonl",
    )

    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["track_id"] == "track-from-source-filename"
    assert row["track_identity_match_method"] == "shared_mix_path"
    assert row["cross_product_provenance"]["source_vocal_track_id"] == "track-from-source-filename"
    assert row["cross_product_provenance"]["source_minus_track_id"] == "track-from-artifact-folder"


def test_cross_product_cli_writes_outputs(tmp_path: Path) -> None:
    vocal_jsonl = write_jsonl(
        tmp_path / "vocals.jsonl",
        [candidate_row("vocal-a", vocal_path="/vocals/a.flac", minus_path="/old/minus.flac")],
    )
    minus_jsonl = write_jsonl(
        tmp_path / "minuses.jsonl",
        [candidate_row("minus-a", vocal_path="/old/vocal.flac", minus_path="/minus/a.flac")],
    )
    output_jsonl = tmp_path / "cross.jsonl"
    skipped_jsonl = tmp_path / "cross_skipped.jsonl"
    summary_md = tmp_path / "cross_summary.md"

    exit_code = main(
        [
            "--vocal-candidates-jsonl",
            str(vocal_jsonl),
            "--minus-candidates-jsonl",
            str(minus_jsonl),
            "--output-jsonl",
            str(output_jsonl),
            "--skipped-jsonl",
            str(skipped_jsonl),
            "--summary-md",
            str(summary_md),
        ]
    )

    assert exit_code == 0
    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["requires_dense_post_cross_product_judge"] is True
    assert skipped_jsonl.exists()
    assert summary_md.exists()
