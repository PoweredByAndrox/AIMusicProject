from __future__ import annotations

import json
from pathlib import Path

from tools.artifact_status_contract import stable_track_split
from tools.build_hard_case_benchmark import build_benchmark, write_outputs


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


def hard_row(
    idx: int,
    *,
    stem_type: str = "vocal",
    reason: str = "selected_candidate_still_has_severe_issue",
    labels: list[str] | None = None,
    track_hash: str | None = None,
) -> dict:
    track = track_hash or f"track-{idx:03d}"
    return {
        "hard_case_id": f"hard-{idx:03d}",
        "track_hash": track,
        "split": "wrong-split",
        "stem_type": stem_type,
        "source_manifest": f"/artifacts/manifest-{idx:03d}.json",
        "manifest_sha256": f"sha-{idx:03d}",
        "delivery_status": reason,
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "release_status": "blocked_from_listen_pack",
        "dominant_failure_reason": reason,
        "failure_labels": labels if labels is not None else [reason],
        "window_start_sec": float(idx),
        "window_end_sec": float(idx + 5),
        "audio_path": f"/do/not/copy/{idx:03d}.flac",
        "model_id": "model",
        "candidate_id": f"candidate-{idx:03d}",
    }


def make_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    rows: list[dict] = []
    rows.extend(hard_row(i, stem_type="vocal") for i in range(35))
    rows.extend(
        hard_row(i, stem_type="minus", reason="robotic_or_noise_artifact")
        for i in range(35, 50)
    )
    rows.extend(
        hard_row(i, stem_type="vocal", reason="unknown_blocked_reason", labels=["unknown_blocked_reason"])
        for i in range(50, 65)
    )
    rows.extend(
        hard_row(i, stem_type="vocal", reason="candidate_space_exhausted", labels=["candidate_space_exhausted"])
        for i in range(65, 70)
    )
    rows.extend(
        hard_row(i, stem_type="minus", reason="no_accepted_candidate", labels=["no_accepted_candidate"])
        for i in range(70, 75)
    )
    rows.append(
        hard_row(
            99,
            stem_type="vocal",
            reason="no_accepted_candidate",
            labels=["no_accepted_candidate"],
            track_hash="golden-track",
        )
    )

    hard_jsonl = write_jsonl(tmp_path / "hard_case_corpus.jsonl", rows)
    hard_manifest = write_json(
        tmp_path / "hard_case_corpus_manifest.json",
        {"schema": "hard_case_corpus_manifest.v1", "row_count": len(rows)},
    )
    dashboard = write_json(
        tmp_path / "control_plane_dashboard.json",
        {
            "schema": "control_plane_dashboard.v1",
            "summary": {
                "hard_case_rows": len(rows),
                "candidate_space_exhausted": 5,
                "no_accepted_candidate": 6,
                "unknown_reason": 15,
            },
        },
    )
    golden_jsonl = write_jsonl(
        tmp_path / "golden_corpus.jsonl",
        [
            {
                "case_id": "golden-1",
                "track_hash": "golden-track",
                "training_allowed": False,
                "threshold_tuning_allowed": False,
                "golden_role": "protected_regression_holdout",
            }
        ],
    )
    return hard_jsonl, hard_manifest, dashboard, golden_jsonl


def test_benchmark_sampling_is_deterministic_and_meets_required_coverage(tmp_path: Path) -> None:
    hard_jsonl, hard_manifest, dashboard, golden_jsonl = make_inputs(tmp_path)

    first = build_benchmark(
        hard_case_jsonl=hard_jsonl,
        hard_case_manifest=hard_manifest,
        dashboard_json=dashboard,
        golden_jsonl=golden_jsonl,
        target_size=60,
    )
    second = build_benchmark(
        hard_case_jsonl=hard_jsonl,
        hard_case_manifest=hard_manifest,
        dashboard_json=dashboard,
        golden_jsonl=golden_jsonl,
        target_size=60,
    )

    assert first["rows"] == second["rows"]
    rows = first["rows"]
    assert 50 <= len(rows) <= 200
    assert sum(row["stem_type"] == "vocal" for row in rows) >= 20
    assert sum(row["stem_type"] == "minus" for row in rows) >= 10
    assert sum(row["dominant_failure_reason"] == "unknown_blocked_reason" for row in rows) >= 10
    assert any("candidate_space_exhausted" in row["failure_labels"] for row in rows)
    assert any("no_accepted_candidate" in row["failure_labels"] for row in rows)
    assert all(row["track_hash"] != "golden-track" for row in rows)

    for row in rows:
        assert set(row) == {
            "benchmark_case_id",
            "source_hard_case_id",
            "track_hash",
            "split",
            "stem_type",
            "dominant_failure_reason",
            "failure_labels",
            "failed_checks",
            "severity_score",
            "window_start_sec",
            "window_end_sec",
            "source_manifest",
            "recommended_review_type",
        }
        assert row["split"] == stable_track_split(row["track_hash"])
        assert isinstance(row["failed_checks"], list)
        assert row["severity_score"] >= 0

    high_reasons = {"candidate_space_exhausted", "no_accepted_candidate", "selected_candidate_still_has_severe_issue"}
    assert all(
        row["severity_score"] >= 90
        for row in rows
        if row["dominant_failure_reason"] in high_reasons
    )
    assert all(
        row["severity_score"] >= 80
        for row in rows
        if row["dominant_failure_reason"] == "unknown_blocked_reason"
    )
    assert any(
        row["severity_score"] >= 90
        for row in rows
        if row["dominant_failure_reason"] == "robotic_or_noise_artifact"
    )


def test_outputs_include_manifest_summary_and_no_audio_files(tmp_path: Path) -> None:
    hard_jsonl, hard_manifest, dashboard, golden_jsonl = make_inputs(tmp_path)
    benchmark = build_benchmark(
        hard_case_jsonl=hard_jsonl,
        hard_case_manifest=hard_manifest,
        dashboard_json=dashboard,
        golden_jsonl=golden_jsonl,
        target_size=60,
    )

    output_jsonl = tmp_path / "out" / "hard_case_benchmark.jsonl"
    output_manifest = tmp_path / "out" / "hard_case_benchmark_manifest.json"
    output_summary = tmp_path / "out" / "hard_case_benchmark_summary.md"
    write_outputs(benchmark, output_jsonl, output_manifest, output_summary)

    assert output_jsonl.exists()
    assert output_manifest.exists()
    assert output_summary.exists()
    assert not list((tmp_path / "out").glob("*.flac"))
    manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert manifest["schema"] == "hard_case_benchmark_manifest.v1"
    assert manifest["golden_usage"]["training_allowed"] is False
    assert manifest["golden_usage"]["threshold_tuning_allowed"] is False
    assert manifest["golden_usage"]["usage"] == "protected_holdout_exclusion_only"
