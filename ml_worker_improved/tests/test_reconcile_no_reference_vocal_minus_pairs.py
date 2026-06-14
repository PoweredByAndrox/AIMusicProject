from __future__ import annotations

import json
from pathlib import Path

from tools.reconcile_no_reference_vocal_minus_pairs import build_reconciled_pairs, main


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def selected_vocal_row(candidate_id: str = "phase-vocal") -> dict:
    return {
        "track_id": "track-a",
        "candidate_id": f"{candidate_id}__window_local",
        "input_candidate_id": candidate_id,
        "final_decision": "phase_corrected_candidate_selected",
        "mix_path": "/mix/a.mp3",
        "selected_vocal_path": "/phase/a_vocal.wav",
        "selected_minus_path": "/phase/a_compensated_minus.wav",
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
    }


def minus_row(candidate_id: str = "minus-a", *, track_id: str = "track-a", minus_path: str = "/minus/a.flac") -> dict:
    return {
        "track_id": track_id,
        "candidate_id": candidate_id,
        "mix_path": "/mix/a.mp3",
        "vocal_path": "/old/vocal.flac",
        "instrumental_path": minus_path,
        "minus_path": minus_path,
        "candidate_family": "vocal+minus",
        "candidate_origin": "existing_generated_artifact_pair",
        "is_complete_pair": True,
        "source_manifest": "/source/minus.jsonl",
    }


def test_reconciles_selected_phase_vocal_with_existing_minus_candidates() -> None:
    result = build_reconciled_pairs(
        [selected_vocal_row()],
        [minus_row("minus-a"), minus_row("minus-b", minus_path="/minus/b.flac")],
        vocal_source_manifest="/phase/result.jsonl",
        minus_source_manifest="/pool.jsonl",
    )

    rows = result["rows"]
    assert len(rows) == 2
    assert rows[0]["track_id"] == "track-a"
    assert rows[0]["vocal_path"] == "/phase/a_vocal.wav"
    assert rows[0]["minus_path"] == "/minus/a.flac"
    assert rows[0]["instrumental_path"] == "/minus/a.flac"
    assert rows[0]["candidate_family"] == "reconciled_vocal_existing_minus"
    assert rows[0]["candidate_origin"] == "no_reference_phase_vocal_existing_minus_reconciliation"
    assert rows[0]["can_publish_as_result"] is False
    assert rows[0]["can_mark_release_ready"] is False
    assert rows[0]["requires_dense_post_reconciliation_judge"] is True
    assert rows[0]["audio_copied"] is False
    assert rows[0]["audio_exported"] is False
    assert rows[0]["pair_reconciliation_provenance"]["vocal_source_manifest"] == "/phase/result.jsonl"
    assert rows[0]["pair_reconciliation_provenance"]["minus_source_manifest"] == "/pool.jsonl"
    assert result["summary"]["reconciled_pair_count"] == 2
    assert result["summary"]["safety"]["models_changed"] is False


def test_reconciliation_skips_unselected_vocals_and_incomplete_minus() -> None:
    skipped_vocal = dict(selected_vocal_row("not-selected"))
    skipped_vocal["final_decision"] = "original_candidate_kept"
    incomplete_minus = minus_row("minus-incomplete")
    incomplete_minus["instrumental_path"] = ""
    incomplete_minus["minus_path"] = ""

    result = build_reconciled_pairs(
        [skipped_vocal, selected_vocal_row()],
        [incomplete_minus, minus_row("minus-good")],
        vocal_source_manifest="/phase/result.jsonl",
        minus_source_manifest="/pool.jsonl",
    )

    assert len(result["rows"]) == 1
    skip_reasons = {row["skip_reason"] for row in result["skipped_rows"]}
    assert "vocal_not_phase_corrected_candidate" in skip_reasons
    assert "minus_missing_path" in skip_reasons


def test_reconciliation_deduplicates_identical_vocal_minus_pairs() -> None:
    result = build_reconciled_pairs(
        [selected_vocal_row()],
        [minus_row("minus-a"), minus_row("minus-a-copy")],
        vocal_source_manifest="/phase/result.jsonl",
        minus_source_manifest="/pool.jsonl",
    )

    assert len(result["rows"]) == 1
    assert result["summary"]["deduplicated_pair_count"] == 1


def test_reconciliation_output_is_deterministic() -> None:
    vocal_rows = [selected_vocal_row("b"), selected_vocal_row("a")]
    minus_rows = [minus_row("minus-b", minus_path="/minus/b.flac"), minus_row("minus-a")]

    first = build_reconciled_pairs(
        list(reversed(vocal_rows)),
        list(reversed(minus_rows)),
        vocal_source_manifest="/phase/result.jsonl",
        minus_source_manifest="/pool.jsonl",
    )
    second = build_reconciled_pairs(
        vocal_rows,
        minus_rows,
        vocal_source_manifest="/phase/result.jsonl",
        minus_source_manifest="/pool.jsonl",
    )

    assert first == second


def test_reconciliation_cli_writes_outputs(tmp_path: Path) -> None:
    vocal_jsonl = tmp_path / "phase.jsonl"
    minus_jsonl = tmp_path / "pool.jsonl"
    output_jsonl = tmp_path / "reconciled.jsonl"
    skipped_jsonl = tmp_path / "skipped.jsonl"
    summary_md = tmp_path / "summary.md"
    write_jsonl(vocal_jsonl, [selected_vocal_row()])
    write_jsonl(minus_jsonl, [minus_row("minus-a")])

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
    assert output_jsonl.exists()
    assert skipped_jsonl.exists()
    assert summary_md.exists()
    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["requires_dense_post_reconciliation_judge"] is True
