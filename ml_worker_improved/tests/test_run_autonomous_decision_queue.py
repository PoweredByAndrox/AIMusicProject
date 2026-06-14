from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.run_autonomous_decision_queue import build_decision_queue, write_outputs


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


def manifest(tmp_path: Path, name: str, payload: dict) -> Path:
    return write_json(tmp_path / "manifests" / f"{name}.json", payload)


def benchmark_row(idx: int, *, source_manifest: Path | None, reason: str, split: str = "train", labels: list[str] | None = None) -> dict:
    return {
        "benchmark_case_id": f"bench-{idx:03d}",
        "source_hard_case_id": f"hard-{idx:03d}",
        "track_hash": f"track-{idx:03d}",
        "split": split,
        "stem_type": "vocal",
        "dominant_failure_reason": reason,
        "failure_labels": labels if labels is not None else [reason],
        "failed_checks": labels if labels is not None else [reason],
        "severity_score": 95,
        "window_start_sec": float(idx),
        "window_end_sec": float(idx + 5),
        "source_manifest": str(source_manifest) if source_manifest else "",
        "recommended_review_type": "autonomous_control_plane_review",
    }


def review_row(row: dict, *, allowed_next_step: str = "manual_ab_only") -> dict:
    return {
        "benchmark_case_id": row["benchmark_case_id"],
        "track_hash": row["track_hash"],
        "stem_type": row["stem_type"],
        "dominant_failure_reason": row["dominant_failure_reason"],
        "failure_labels": row["failure_labels"],
        "severity_score": row["severity_score"],
        "recommended_review_type": row["recommended_review_type"],
        "review_priority": "P0",
        "expected_audio_problem": "test problem",
        "human_rating": None,
        "human_notes": None,
        "reviewer_decision": None,
        "allowed_next_step": allowed_next_step,
    }


def make_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    accepted_manifest = manifest(
        tmp_path,
        "accepted",
        {
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "metrics": {"sdr_db": 20.0, "si_sdr_db": 18.0, "reference_likeness_percent": 96.0},
            "audio_path": str(tmp_path / "do_not_copy.flac"),
        },
    )
    blocked_manifest = manifest(
        tmp_path,
        "blocked",
        {
            "delivery_status": "accepted",
            "delivery_allowed": False,
            "presentation_status": "accepted_for_listen_pack",
        },
    )
    rows = [
        benchmark_row(1, source_manifest=accepted_manifest, reason="accepted", labels=[]),
        benchmark_row(2, source_manifest=blocked_manifest, reason="accepted", labels=[]),
        benchmark_row(3, source_manifest=None, reason="candidate_space_exhausted", labels=["candidate_space_exhausted"]),
        benchmark_row(4, source_manifest=None, reason="thin_or_overcleaned", labels=["thin_or_overcleaned"]),
        benchmark_row(5, source_manifest=None, reason="thin_or_overcleaned", split="holdout", labels=["thin_or_overcleaned"]),
    ]
    reviews = [
        review_row(rows[0], allowed_next_step="listen_pack"),
        review_row(rows[1], allowed_next_step="listen_pack"),
        review_row(rows[2], allowed_next_step="manual_ab_only"),
        review_row(rows[3], allowed_next_step="finetune_candidate"),
        review_row(rows[4], allowed_next_step="finetune_candidate"),
    ]
    benchmark_jsonl = write_jsonl(tmp_path / "hard_case_benchmark.jsonl", rows)
    review_jsonl = write_jsonl(tmp_path / "hard_case_review_pack.jsonl", reviews)
    golden_jsonl = write_jsonl(
        tmp_path / "golden_corpus.jsonl",
        [{"track_hash": "golden-track", "training_allowed": False, "threshold_tuning_allowed": False}],
    )
    dashboard = write_json(tmp_path / "control_plane_dashboard.json", {"schema": "control_plane_dashboard.v1", "summary": {}})
    return benchmark_jsonl, review_jsonl, golden_jsonl, dashboard


def test_decision_queue_is_deterministic_and_writes_no_audio(tmp_path: Path) -> None:
    benchmark_jsonl, review_jsonl, golden_jsonl, dashboard = make_inputs(tmp_path)

    first = build_decision_queue(
        hard_case_benchmark_jsonl=benchmark_jsonl,
        review_pack_jsonl=review_jsonl,
        golden_jsonl=golden_jsonl,
        dashboard_json=dashboard,
    )
    second = build_decision_queue(
        hard_case_benchmark_jsonl=benchmark_jsonl,
        review_pack_jsonl=review_jsonl,
        golden_jsonl=golden_jsonl,
        dashboard_json=dashboard,
    )

    assert first["rows"] == second["rows"]
    assert len(first["rows"]) == 5
    assert all(row["human_review_required"] is False for row in first["rows"])
    assert {row["decision"] for row in first["rows"]} >= {
        "accepted_for_listen_pack",
        "blocked_from_listen_pack",
        "hard_case",
        "finetune_candidate",
    }
    assert not any(row["split"] == "holdout" and row["decision"] == "finetune_candidate" for row in first["rows"])

    output_dir = tmp_path / "out"
    write_outputs(first, output_dir)
    assert (output_dir / "autonomous_decision_queue.jsonl").exists()
    assert (output_dir / "autonomous_decision_summary.md").exists()
    manifest_payload = json.loads((output_dir / "autonomous_judge_manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["safety"]["audio_copied"] is False
    assert manifest_payload["safety"]["training_started"] is False
    assert manifest_payload["safety"]["policy_changed"] is False
    assert manifest_payload["safety"]["models_changed"] is False
    assert manifest_payload["safety"]["weights_added"] is False
    assert manifest_payload["safety"]["cleanup_routes_added"] is False
    assert manifest_payload["safety"]["thresholds_changed"] is False
    assert not list(output_dir.glob("*.flac"))


def test_low_confidence_rows_are_not_promoted(tmp_path: Path) -> None:
    benchmark_jsonl = write_jsonl(
        tmp_path / "hard_case_benchmark.jsonl",
        [benchmark_row(1, source_manifest=None, reason="unknown_blocked_reason", labels=[])],
    )
    review_jsonl = write_jsonl(tmp_path / "hard_case_review_pack.jsonl", [])
    golden_jsonl = write_jsonl(tmp_path / "golden_corpus.jsonl", [])
    dashboard = write_json(tmp_path / "control_plane_dashboard.json", {"schema": "control_plane_dashboard.v1", "summary": {}})

    result = build_decision_queue(
        hard_case_benchmark_jsonl=benchmark_jsonl,
        review_pack_jsonl=review_jsonl,
        golden_jsonl=golden_jsonl,
        dashboard_json=dashboard,
    )

    row = result["rows"][0]
    assert row["confidence_band"] == "low"
    assert row["decision"] in {"blocked_from_listen_pack", "hard_case"}
    assert row["decision"] not in {"accepted_for_listen_pack", "release_ready"}


def test_direct_script_invocation_writes_outputs(tmp_path: Path) -> None:
    benchmark_jsonl, review_jsonl, golden_jsonl, dashboard = make_inputs(tmp_path)
    output_dir = tmp_path / "direct_out"

    result = subprocess.run(
        [
            sys.executable,
            "tools/run_autonomous_decision_queue.py",
            "--hard-case-benchmark-jsonl",
            str(benchmark_jsonl),
            "--review-pack-jsonl",
            str(review_jsonl),
            "--golden-jsonl",
            str(golden_jsonl),
            "--dashboard-json",
            str(dashboard),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "autonomous_decision_queue.jsonl").exists()
