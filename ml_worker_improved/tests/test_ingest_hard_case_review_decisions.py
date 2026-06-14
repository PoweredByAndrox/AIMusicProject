from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools.ingest_hard_case_review_decisions import IngestValidationError, ingest_review_decisions, write_outputs


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark_case_id",
        "track_hash",
        "stem_type",
        "dominant_failure_reason",
        "failure_labels",
        "severity_score",
        "recommended_review_type",
        "review_priority",
        "expected_audio_problem",
        "human_rating",
        "human_notes",
        "reviewer_decision",
        "allowed_next_step",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def review_row(case_id: str, *, allowed_next_step: str = "manual_ab_only", track_hash: str | None = None) -> dict:
    return {
        "benchmark_case_id": case_id,
        "track_hash": track_hash or f"track-{case_id}",
        "stem_type": "vocal",
        "dominant_failure_reason": "selected_candidate_still_has_severe_issue",
        "failure_labels": ["selected_candidate_still_has_severe_issue"],
        "severity_score": 95,
        "recommended_review_type": "manual_ab_hard_case_review",
        "review_priority": "P0",
        "expected_audio_problem": "selected candidate still has severe audible issue windows",
        "human_rating": None,
        "human_notes": None,
        "reviewer_decision": None,
        "allowed_next_step": allowed_next_step,
    }


def sheet_row(pack_row: dict, decision: str, *, rating: str = "2", notes: str = "audible issue") -> dict:
    row = dict(pack_row)
    row["failure_labels"] = ";".join(pack_row["failure_labels"])
    row["human_rating"] = rating
    row["human_notes"] = notes
    row["reviewer_decision"] = decision
    return row


def make_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    pack_rows = [
        review_row("case-finetune", allowed_next_step="finetune_candidate"),
        review_row("case-policy", allowed_next_step="policy_fix_candidate"),
        review_row("case-reference", allowed_next_step="manual_ab_only"),
        review_row("case-proxy", allowed_next_step="manual_ab_only"),
        review_row("case-holdout", allowed_next_step="holdout_only"),
        review_row("case-manual", allowed_next_step="manual_ab_only"),
        review_row("case-artifact", allowed_next_step="manual_ab_only"),
        review_row("case-unreviewed", allowed_next_step="manual_ab_only"),
    ]
    pack = write_jsonl(tmp_path / "hard_case_review_pack.jsonl", pack_rows)
    sheet = write_csv(
        tmp_path / "hard_case_review_sheet.csv",
        [
            sheet_row(pack_rows[0], "needs_finetune"),
            sheet_row(pack_rows[1], "needs_policy_fix"),
            sheet_row(pack_rows[2], "needs_reference_fix"),
            sheet_row(pack_rows[3], "proxy_false_positive"),
            sheet_row(pack_rows[4], "needs_manual_ab"),
            sheet_row(pack_rows[5], "true_failure"),
            sheet_row(pack_rows[6], "acceptable_artifact"),
            sheet_row(pack_rows[7], "", rating="", notes=""),
        ],
    )
    summary = tmp_path / "hard_case_review_summary.md"
    summary.write_text("# summary\n", encoding="utf-8")
    golden = write_jsonl(tmp_path / "golden_corpus.jsonl", [{"track_hash": "golden-track"}])
    return pack, sheet, summary, golden


def test_ingest_routes_valid_decisions_and_keeps_human_evidence(tmp_path: Path) -> None:
    pack, sheet, summary, golden = make_inputs(tmp_path)

    result = ingest_review_decisions(
        review_pack_jsonl=pack,
        review_sheet_csv=sheet,
        review_summary_md=summary,
        golden_jsonl=golden,
    )
    repeat = ingest_review_decisions(
        review_pack_jsonl=pack,
        review_sheet_csv=sheet,
        review_summary_md=summary,
        golden_jsonl=golden,
    )

    assert result == repeat
    assert result["summary"]["reviewed_rows"] == 7
    assert result["summary"]["unreviewed_rows"] == 1
    assert result["summary"]["verified_finetune_candidates"] == 1
    assert result["summary"]["policy_fixes"] == 1
    assert result["summary"]["reference_fixes"] == 1
    assert result["summary"]["proxy_false_positives"] == 1
    assert result["summary"]["holdout_protected_count"] == 1
    assert len(result["queues"]["manual_ab_followup"]) == 2

    fine_tune = result["queues"]["finetune_candidate_verified"][0]
    assert fine_tune["benchmark_case_id"] == "case-finetune"
    assert fine_tune["human_rating"] == "2"
    assert fine_tune["human_notes"] == "audible issue"
    assert fine_tune["reviewer_decision"] == "needs_finetune"
    assert "audio_path" not in fine_tune


def test_invalid_decision_is_rejected(tmp_path: Path) -> None:
    pack, sheet, summary, golden = make_inputs(tmp_path)
    rows = list(csv.DictReader(sheet.open(encoding="utf-8", newline="")))
    rows[0]["reviewer_decision"] = "invented_decision"
    bad_sheet = write_csv(tmp_path / "bad.csv", rows)

    with pytest.raises(IngestValidationError, match="invalid_reviewer_decision"):
        ingest_review_decisions(
            review_pack_jsonl=pack,
            review_sheet_csv=bad_sheet,
            review_summary_md=summary,
            golden_jsonl=golden,
        )


def test_unknown_case_id_is_rejected(tmp_path: Path) -> None:
    pack, sheet, summary, golden = make_inputs(tmp_path)
    rows = list(csv.DictReader(sheet.open(encoding="utf-8", newline="")))
    rows[0]["benchmark_case_id"] = "unknown-case"
    bad_sheet = write_csv(tmp_path / "unknown.csv", rows)

    with pytest.raises(IngestValidationError, match="unknown_benchmark_case_id"):
        ingest_review_decisions(
            review_pack_jsonl=pack,
            review_sheet_csv=bad_sheet,
            review_summary_md=summary,
            golden_jsonl=golden,
        )


def test_holdout_cannot_enter_finetune_queue(tmp_path: Path) -> None:
    pack, sheet, summary, golden = make_inputs(tmp_path)
    rows = list(csv.DictReader(sheet.open(encoding="utf-8", newline="")))
    for row in rows:
        if row["benchmark_case_id"] == "case-holdout":
            row["reviewer_decision"] = "needs_finetune"
    bad_sheet = write_csv(tmp_path / "holdout_finetune.csv", rows)

    with pytest.raises(IngestValidationError, match="holdout_cannot_enter_finetune_queue"):
        ingest_review_decisions(
            review_pack_jsonl=pack,
            review_sheet_csv=bad_sheet,
            review_summary_md=summary,
            golden_jsonl=golden,
        )


def test_outputs_are_written_without_audio_files(tmp_path: Path) -> None:
    pack, sheet, summary, golden = make_inputs(tmp_path)
    result = ingest_review_decisions(
        review_pack_jsonl=pack,
        review_sheet_csv=sheet,
        review_summary_md=summary,
        golden_jsonl=golden,
    )
    out = tmp_path / "out"
    write_outputs(result, out)

    expected = {
        "validated_review_decisions.jsonl",
        "policy_fix_queue.jsonl",
        "finetune_candidate_verified.jsonl",
        "reference_fix_queue.jsonl",
        "proxy_false_positive_queue.jsonl",
        "manual_ab_followup_queue.jsonl",
        "review_ingest_summary.md",
    }
    assert expected == {path.name for path in out.iterdir()}
    assert not list(out.glob("*.flac"))
