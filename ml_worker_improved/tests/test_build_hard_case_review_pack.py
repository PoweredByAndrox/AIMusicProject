from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.build_hard_case_review_pack import build_review_pack, write_outputs


REVIEW_ROW_KEYS = {
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
}


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def benchmark_row(
    idx: int,
    *,
    track_hash: str | None = None,
    stem_type: str = "vocal",
    reason: str = "selected_candidate_still_has_severe_issue",
    labels: list[str] | None = None,
    split: str = "train",
) -> dict:
    failure_labels = labels if labels is not None else [reason]
    return {
        "benchmark_case_id": f"bench-{idx:03d}",
        "source_hard_case_id": f"hard-{idx:03d}",
        "track_hash": track_hash or f"track-{idx:03d}",
        "split": split,
        "stem_type": stem_type,
        "dominant_failure_reason": reason,
        "failure_labels": failure_labels,
        "failed_checks": [],
        "severity_score": 95 if reason == "selected_candidate_still_has_severe_issue" else 80,
        "window_start_sec": float(idx),
        "window_end_sec": float(idx + 5),
        "source_manifest": f"/manifests/{idx:03d}.json",
        "recommended_review_type": "manual_ab_hard_case_review",
        "audio_path": f"/must/not/copy/{idx:03d}.flac",
    }


def make_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    benchmark_rows = [
        benchmark_row(1, reason="no_accepted_candidate", labels=["no_accepted_candidate"]),
        benchmark_row(2, reason="candidate_space_exhausted", labels=["candidate_space_exhausted"]),
        benchmark_row(3, reason="selected_candidate_still_has_severe_issue"),
        benchmark_row(4, reason="selected_candidate_has_unstable_vocal_consistency"),
        benchmark_row(5, reason="unknown_blocked_reason", labels=["unknown_blocked_reason"]),
        benchmark_row(6, stem_type="minus", reason="thin_or_overcleaned", split="dev"),
        benchmark_row(7, reason="robotic_or_noise_artifact", split="holdout"),
    ]
    benchmark_jsonl = write_jsonl(tmp_path / "hard_case_benchmark.jsonl", benchmark_rows)
    benchmark_manifest = write_json(
        tmp_path / "hard_case_benchmark_manifest.json",
        {
            "schema": "hard_case_benchmark_manifest.v1",
            "row_count": len(benchmark_rows),
            "golden_usage": {
                "usage": "protected_holdout_exclusion_only",
                "training_allowed": False,
                "threshold_tuning_allowed": False,
            },
        },
    )
    benchmark_summary = tmp_path / "hard_case_benchmark_summary.md"
    benchmark_summary.write_text("# summary\n", encoding="utf-8")
    golden_jsonl = write_jsonl(
        tmp_path / "golden_corpus.jsonl",
        [{"track_hash": "not-in-benchmark", "training_allowed": False, "threshold_tuning_allowed": False}],
    )
    dashboard = write_json(
        tmp_path / "control_plane_dashboard.json",
        {"schema": "control_plane_dashboard.v1", "summary": {"hard_case_rows": 7}},
    )
    return benchmark_jsonl, benchmark_manifest, benchmark_summary, golden_jsonl, dashboard


def test_every_non_golden_benchmark_row_becomes_review_task_with_policy_fields(tmp_path: Path) -> None:
    benchmark_jsonl, benchmark_manifest, benchmark_summary, golden_jsonl, dashboard = make_inputs(tmp_path)

    first = build_review_pack(
        benchmark_jsonl=benchmark_jsonl,
        benchmark_manifest=benchmark_manifest,
        benchmark_summary_md=benchmark_summary,
        golden_jsonl=golden_jsonl,
        dashboard_json=dashboard,
    )
    second = build_review_pack(
        benchmark_jsonl=benchmark_jsonl,
        benchmark_manifest=benchmark_manifest,
        benchmark_summary_md=benchmark_summary,
        golden_jsonl=golden_jsonl,
        dashboard_json=dashboard,
    )

    assert first["rows"] == second["rows"]
    rows = first["rows"]
    assert len(rows) == 7
    assert all(set(row) == REVIEW_ROW_KEYS for row in rows)
    assert all("audio_path" not in row for row in rows)
    assert all(row["human_rating"] is None for row in rows)
    assert all(row["human_notes"] is None for row in rows)
    assert all(row["reviewer_decision"] is None for row in rows)

    by_id = {row["benchmark_case_id"]: row for row in rows}
    assert by_id["bench-001"]["review_priority"] == "P0"
    assert by_id["bench-002"]["review_priority"] == "P0"
    assert by_id["bench-003"]["review_priority"] == "P0"
    assert by_id["bench-005"]["review_priority"] == "P1"
    assert by_id["bench-005"]["allowed_next_step"] in {"manual_ab_only", "taxonomy_relabel"}
    assert by_id["bench-006"]["review_priority"] == "P2"
    assert by_id["bench-007"]["allowed_next_step"] == "holdout_only"

    assert first["summary"]["priority_counts"] == {"P0": 3, "P1": 2, "P2": 2}
    assert first["summary"]["golden_usage"]["training_allowed"] is False
    assert first["summary"]["golden_usage"]["threshold_tuning_allowed"] is False


def test_golden_overlap_is_excluded_and_outputs_are_csv_jsonl_md_only(tmp_path: Path) -> None:
    benchmark_jsonl, benchmark_manifest, benchmark_summary, _golden_jsonl, dashboard = make_inputs(tmp_path)
    golden_jsonl = write_jsonl(
        tmp_path / "golden_overlap.jsonl",
        [{"track_hash": "track-005", "training_allowed": False, "threshold_tuning_allowed": False}],
    )
    pack = build_review_pack(
        benchmark_jsonl=benchmark_jsonl,
        benchmark_manifest=benchmark_manifest,
        benchmark_summary_md=benchmark_summary,
        golden_jsonl=golden_jsonl,
        dashboard_json=dashboard,
    )

    assert len(pack["rows"]) == 6
    assert all(row["track_hash"] != "track-005" for row in pack["rows"])
    assert pack["summary"]["skipped_golden_overlap"] == 1

    output_jsonl = tmp_path / "out" / "hard_case_review_pack.jsonl"
    output_csv = tmp_path / "out" / "hard_case_review_sheet.csv"
    output_summary = tmp_path / "out" / "hard_case_review_summary.md"
    write_outputs(pack, output_jsonl, output_csv, output_summary)

    assert output_jsonl.exists()
    assert output_csv.exists()
    assert output_summary.exists()
    assert not list((tmp_path / "out").glob("*.flac"))
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert set(rows[0]) == REVIEW_ROW_KEYS
