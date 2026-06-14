from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.plan_autonomous_root_causes import plan_root_causes, write_outputs


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


def decision_row(
    idx: int,
    *,
    stem_type: str = "vocal",
    split: str = "train",
    decision: str = "hard_case",
    reason: str = "selected_candidate_still_has_severe_issue",
    labels: list[str] | None = None,
    judge_mode: str = "no_reference_proxy",
    confidence: float = 0.9,
    track_hash: str | None = None,
) -> dict:
    failure_labels = labels if labels is not None else [reason]
    return {
        "case_id": f"bench-{idx:03d}",
        "source_manifest": f"/manifests/{idx:03d}.json",
        "track_hash": track_hash or f"track-{idx:03d}",
        "stem_type": stem_type,
        "split": split,
        "dominant_failure_reason": reason,
        "failure_labels": failure_labels,
        "failed_checks": failure_labels,
        "available_evidence": {
            "benchmark_row": True,
            "reference_metrics": {},
            "proxy_signals": failure_labels,
            "golden_track_protected": False,
        },
        "judge_mode": judge_mode,
        "decision": decision,
        "confidence": confidence,
        "confidence_band": "high" if confidence >= 0.85 else "medium",
        "reasons": failure_labels,
        "forbidden_actions": ["no_audio_export", "no_cleanup_debleed_denoise_escalation"],
        "allowed_next_step": "hard_case_queue",
        "human_review_required": False,
        "optional_human_spotcheck": False,
    }


def make_inputs(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path, Path, Path, Path]:
    decisions = write_jsonl(tmp_path / "autonomous_decision_queue.jsonl", rows)
    manifest = write_json(
        tmp_path / "autonomous_judge_manifest.json",
        {
            "schema": "autonomous_judge_manifest.v1",
            "row_count": len(rows),
            "decision_counts": {},
            "safety": {
                "audio_copied": False,
                "training_started": False,
                "policy_changed": False,
                "models_changed": False,
                "weights_added": False,
                "cleanup_routes_added": False,
                "thresholds_changed": False,
            },
        },
    )
    benchmark = write_jsonl(
        tmp_path / "hard_case_benchmark.jsonl",
        [
            {
                "benchmark_case_id": row["case_id"],
                "track_hash": row["track_hash"],
                "split": row["split"],
                "stem_type": row["stem_type"],
                "dominant_failure_reason": row["dominant_failure_reason"],
                "failure_labels": row["failure_labels"],
                "failed_checks": row["failed_checks"],
                "source_manifest": row["source_manifest"],
            }
            for row in rows
        ],
    )
    dashboard = write_json(tmp_path / "control_plane_dashboard.json", {"schema": "control_plane_dashboard.v1", "summary": {}})
    golden = write_jsonl(
        tmp_path / "golden_corpus.jsonl",
        [{"track_hash": "golden-track", "training_allowed": False, "threshold_tuning_allowed": False}],
    )
    return decisions, manifest, benchmark, dashboard, golden


def by_case(rows: list[dict]) -> dict[str, dict]:
    return {row["case_id"]: row for row in rows}


def test_classifies_required_root_causes_and_safety_flags(tmp_path: Path) -> None:
    rows = [
        decision_row(1, reason="candidate_space_exhausted", labels=["candidate_space_exhausted"]),
        decision_row(2, reason="no_accepted_candidate", labels=["no_accepted_candidate"]),
        decision_row(3, reason="unknown_blocked_reason", labels=["unknown_blocked_reason"]),
        decision_row(4, split="holdout", reason="thin_or_overcleaned", labels=["thin_or_overcleaned"]),
        decision_row(5, track_hash="golden-track", reason="thin_or_overcleaned", labels=["thin_or_overcleaned"]),
        decision_row(6, stem_type="minus", reason="selected_candidate_still_has_severe_issue"),
        decision_row(7, reason="selected_candidate_has_unstable_vocal_consistency", labels=["selected_candidate_has_unstable_vocal_consistency"]),
        decision_row(8, reason="selected_candidate_still_has_severe_issue", labels=["selected_candidate_still_has_severe_issue"]),
        decision_row(
            9,
            reason="unknown_blocked_reason",
            labels=["reference_likeness_below_threshold"],
            judge_mode="reference_aware",
        ),
    ]
    decisions, manifest, benchmark, dashboard, golden = make_inputs(tmp_path, rows)

    first = plan_root_causes(
        autonomous_decisions_jsonl=decisions,
        autonomous_manifest_json=manifest,
        hard_case_benchmark_jsonl=benchmark,
        dashboard_json=dashboard,
        golden_jsonl=golden,
    )
    second = plan_root_causes(
        autonomous_decisions_jsonl=decisions,
        autonomous_manifest_json=manifest,
        hard_case_benchmark_jsonl=benchmark,
        dashboard_json=dashboard,
        golden_jsonl=golden,
    )

    assert first["rows"] == second["rows"]
    planned = by_case(first["rows"])
    assert planned["bench-001"]["root_cause_bucket"] == "candidate_space_exhausted"
    assert planned["bench-002"]["root_cause_bucket"] == "no_accepted_candidate"
    assert planned["bench-003"]["root_cause_bucket"] == "unknown_needs_more_evidence"
    assert planned["bench-003"]["can_enter_finetune_pool"] is False
    assert planned["bench-004"]["root_cause_bucket"] == "protected_holdout_only"
    assert planned["bench-004"]["can_enter_finetune_pool"] is False
    assert planned["bench-005"]["can_enter_finetune_pool"] is False
    assert planned["bench-006"]["root_cause_bucket"] == "minus_specific_failure"
    assert planned["bench-006"]["root_cause_bucket"] != "vocal_specific_failure"
    assert planned["bench-007"]["root_cause_bucket"] in {"policy_classification_problem", "vocal_specific_failure"}
    assert planned["bench-009"]["root_cause_bucket"] == "reference_or_alignment_problem"

    assert all(row["can_execute_automatically"] is False for row in first["rows"])
    assert all(row["human_review_required"] is False for row in first["rows"])
    assert not any(row["can_request_new_model"] for row in first["rows"] if row["root_cause_bucket"] != "current_model_insufficient")
    assert first["manifest"]["row_count"] == len(first["rows"])
    assert first["manifest"]["root_cause_bucket_counts"] == first["summary"]["root_cause_bucket_counts"]


def test_outputs_are_jsonl_md_manifest_and_no_audio(tmp_path: Path) -> None:
    rows = [
        decision_row(1, reason="candidate_space_exhausted", labels=["candidate_space_exhausted"]),
        decision_row(2, reason="no_accepted_candidate", labels=["no_accepted_candidate"]),
    ]
    decisions, manifest, benchmark, dashboard, golden = make_inputs(tmp_path, rows)
    result = plan_root_causes(
        autonomous_decisions_jsonl=decisions,
        autonomous_manifest_json=manifest,
        hard_case_benchmark_jsonl=benchmark,
        dashboard_json=dashboard,
        golden_jsonl=golden,
    )

    output_dir = tmp_path / "out"
    write_outputs(result, output_dir)

    assert (output_dir / "root_cause_plan.jsonl").exists()
    assert (output_dir / "root_cause_plan_summary.md").exists()
    assert (output_dir / "root_cause_plan_manifest.json").exists()
    assert not list(output_dir.glob("*.flac"))
    payload = json.loads((output_dir / "root_cause_plan_manifest.json").read_text(encoding="utf-8"))
    assert payload["safety"]["audio_copied"] is False
    assert payload["safety"]["training_started"] is False
    assert payload["safety"]["policy_changed"] is False
    assert payload["safety"]["models_changed"] is False
    assert payload["safety"]["weights_added"] is False
    assert payload["safety"]["cleanup_routes_added"] is False
    assert payload["safety"]["thresholds_changed"] is False


def test_direct_script_invocation(tmp_path: Path) -> None:
    rows = [decision_row(1, reason="candidate_space_exhausted", labels=["candidate_space_exhausted"])]
    decisions, manifest, benchmark, dashboard, golden = make_inputs(tmp_path, rows)
    output_dir = tmp_path / "direct"

    result = subprocess.run(
        [
            sys.executable,
            "tools/plan_autonomous_root_causes.py",
            "--autonomous-decisions-jsonl",
            str(decisions),
            "--autonomous-manifest-json",
            str(manifest),
            "--hard-case-benchmark-jsonl",
            str(benchmark),
            "--dashboard-json",
            str(dashboard),
            "--golden-jsonl",
            str(golden),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "root_cause_plan.jsonl").exists()
