from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.plan_missing_evidence_collection import build_missing_evidence_plan, write_outputs


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


def source_manifest(
    path: Path,
    *,
    with_audio: bool = True,
    with_reference: bool = True,
    with_metrics: bool = False,
    with_alignment: bool = False,
) -> Path:
    payload: dict = {
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
    }
    if with_audio:
        payload["source_audio_path"] = "/already/in/manifest/source.wav"
        payload["selected_artifacts"] = {"diagnostic_vocals_flac": "/already/in/manifest/vocals.flac"}
    if with_reference:
        payload["ground_truth_manifest"] = "/already/in/manifest/ground_truth_manifest.json"
        payload["reference_stem_path"] = "/already/in/manifest/reference_vocals.flac"
    if with_metrics:
        payload["objective_metrics"] = {
            "sdr_db": 12.0,
            "si_sdr_db": 11.0,
            "sir_db": 20.0,
            "sar_db": 15.0,
            "reference_likeness_percent": 93.0,
            "residual_rms_dbfs": -45.0,
            "residual_peak_dbfs": -20.0,
            "leakage_score": 0.05,
        }
    if with_alignment:
        payload["report_alignment_contract"] = {"status": "accepted"}
        payload["full_mix_reconstruction_contract"] = {"status": "accepted"}
    return write_json(path, payload)


def root_row(
    idx: int,
    *,
    case_id: str | None = None,
    track_hash: str | None = None,
    split: str = "train",
    stem_type: str = "vocal",
    bucket: str = "unknown_needs_more_evidence",
    action: str = "collect_missing_reference_evidence",
    reason: str = "unknown_blocked_reason",
    labels: list[str] | None = None,
) -> dict:
    failure_labels = labels if labels is not None else [reason]
    return {
        "case_id": case_id or f"bench-{idx:03d}",
        "track_hash": track_hash or f"track-{idx:03d}",
        "stem_type": stem_type,
        "split": split,
        "original_decision": "hard_case",
        "dominant_failure_reason": reason,
        "failure_labels": failure_labels,
        "judge_mode": "manifest_only",
        "confidence": 0.9,
        "root_cause_bucket": bucket,
        "safe_next_action": action,
        "forbidden_actions": ["no_audio_export", "no_policy_auto_change"],
        "required_evidence_before_execution": ["source_manifest"],
        "can_execute_automatically": False,
        "can_enter_finetune_pool": False,
        "can_change_policy": False,
        "can_request_new_model": False,
        "human_review_required": False,
    }


def make_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    src_a = source_manifest(tmp_path / "manifests" / "a.json", with_metrics=False, with_alignment=False)
    src_holdout = source_manifest(tmp_path / "manifests" / "holdout.json", with_metrics=False)
    src_golden = source_manifest(tmp_path / "manifests" / "golden.json", with_metrics=False)
    src_complete = source_manifest(
        tmp_path / "manifests" / "complete.json",
        with_metrics=True,
        with_alignment=True,
    )
    rows = [
        root_row(1),
        root_row(2, case_id="bench-missing-source", stem_type="unknown"),
        root_row(3, case_id="bench-holdout", split="holdout"),
        root_row(4, case_id="bench-golden", track_hash="golden-track"),
        root_row(5, case_id="bench-skip", bucket="vocal_specific_failure", action="investigate_vocal_specific_path"),
        root_row(6, case_id="bench-complete"),
    ]
    root_plan = write_jsonl(tmp_path / "root_cause_plan.jsonl", rows)
    root_manifest = write_json(tmp_path / "root_cause_plan_manifest.json", {"schema": "root_cause_plan_manifest.v1", "row_count": len(rows)})

    source_by_case = {
        "bench-001": str(src_a),
        "bench-missing-source": "",
        "bench-holdout": str(src_holdout),
        "bench-golden": str(src_golden),
        "bench-skip": str(src_a),
        "bench-complete": str(src_complete),
    }
    decisions = write_jsonl(
        tmp_path / "autonomous_decision_queue.jsonl",
        [
            {
                "case_id": row["case_id"],
                "source_manifest": source_by_case[row["case_id"]],
                "track_hash": row["track_hash"],
                "stem_type": row["stem_type"],
                "split": row["split"],
                "dominant_failure_reason": row["dominant_failure_reason"],
                "failure_labels": row["failure_labels"],
                "failed_checks": row["failure_labels"],
                "available_evidence": {"reference_metrics": {}, "source_manifest_loaded": bool(source_by_case[row["case_id"]])},
                "decision": row["original_decision"],
                "forbidden_actions": row["forbidden_actions"],
            }
            for row in rows
        ],
    )
    benchmark = write_jsonl(
        tmp_path / "hard_case_benchmark.jsonl",
        [
            {
                "benchmark_case_id": row["case_id"],
                "source_manifest": source_by_case[row["case_id"]],
                "track_hash": row["track_hash"],
                "split": row["split"],
                "stem_type": row["stem_type"],
                "dominant_failure_reason": row["dominant_failure_reason"],
                "failure_labels": row["failure_labels"],
                "failed_checks": row["failure_labels"],
                "window_start_sec": None if row["case_id"] == "bench-missing-source" else 10.0,
                "window_end_sec": None if row["case_id"] == "bench-missing-source" else 15.0,
            }
            for row in rows
        ],
    )
    dashboard = write_json(tmp_path / "control_plane_dashboard.json", {"schema": "control_plane_dashboard.v1", "summary": {}})
    golden = write_jsonl(
        tmp_path / "golden_corpus.jsonl",
        [{"track_hash": "golden-track", "training_allowed": False, "threshold_tuning_allowed": False}],
    )
    return root_plan, root_manifest, decisions, benchmark, dashboard, golden


def by_case(rows: list[dict]) -> dict[str, dict]:
    return {row["case_id"]: row for row in rows}


def test_builds_only_unknown_missing_evidence_rows_and_routes_tools(tmp_path: Path) -> None:
    root_plan, root_manifest, decisions, benchmark, dashboard, golden = make_inputs(tmp_path)

    result = build_missing_evidence_plan(
        root_cause_plan_jsonl=root_plan,
        root_cause_manifest_json=root_manifest,
        autonomous_decisions_jsonl=decisions,
        hard_case_benchmark_jsonl=benchmark,
        dashboard_json=dashboard,
        golden_jsonl=golden,
    )

    planned = by_case(result["rows"])
    assert set(planned) == {"bench-001", "bench-missing-source", "bench-holdout", "bench-golden", "bench-complete"}
    assert "missing_reference_metrics" in planned["bench-001"]["missing_evidence_types"]
    assert "run_ground_truth_quality_gate" in planned["bench-001"]["required_tools_to_run"]
    assert "waveform_forensic_compare" in planned["bench-001"]["required_tools_to_run"]
    assert "missing_source_manifest" in planned["bench-missing-source"]["missing_evidence_types"]
    assert planned["bench-missing-source"]["can_collect_automatically"] is False
    assert planned["bench-missing-source"]["blocked_reason"]
    assert planned["bench-holdout"]["can_use_for_finetune_after_collection"] is False
    assert planned["bench-golden"]["can_use_for_finetune_after_collection"] is False
    assert planned["bench-complete"]["missing_evidence_types"] == ["unknown_manifest_inconsistency"]
    assert all(row["missing_evidence_types"] for row in result["rows"])
    assert all(row["required_tools_to_run"] for row in result["rows"])


def test_output_is_deterministic_summary_matches_and_no_audio_is_copied(tmp_path: Path) -> None:
    root_plan, root_manifest, decisions, benchmark, dashboard, golden = make_inputs(tmp_path)

    first = build_missing_evidence_plan(
        root_cause_plan_jsonl=root_plan,
        root_cause_manifest_json=root_manifest,
        autonomous_decisions_jsonl=decisions,
        hard_case_benchmark_jsonl=benchmark,
        dashboard_json=dashboard,
        golden_jsonl=golden,
    )
    second = build_missing_evidence_plan(
        root_cause_plan_jsonl=root_plan,
        root_cause_manifest_json=root_manifest,
        autonomous_decisions_jsonl=decisions,
        hard_case_benchmark_jsonl=benchmark,
        dashboard_json=dashboard,
        golden_jsonl=golden,
    )

    assert first["rows"] == second["rows"]
    output_dir = tmp_path / "out"
    write_outputs(first, output_dir)
    output_rows = [
        json.loads(line)
        for line in (output_dir / "missing_evidence_plan.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_payload = json.loads((output_dir / "missing_evidence_plan_manifest.json").read_text(encoding="utf-8"))
    assert len(output_rows) == first["summary"]["row_count"]
    assert manifest_payload["row_count"] == len(output_rows)
    assert manifest_payload["rows_by_missing_evidence_type"] == first["summary"]["rows_by_missing_evidence_type"]
    assert not list(output_dir.glob("*.flac"))
    assert not list(output_dir.glob("*.wav"))
    assert manifest_payload["safety"]["audio_copied"] is False
    assert manifest_payload["safety"]["training_started"] is False
    assert manifest_payload["safety"]["policy_changed"] is False
    assert manifest_payload["safety"]["models_changed"] is False
    assert manifest_payload["safety"]["weights_added"] is False
    assert manifest_payload["safety"]["cleanup_routes_added"] is False
    assert manifest_payload["safety"]["thresholds_changed"] is False


def test_direct_script_invocation(tmp_path: Path) -> None:
    root_plan, root_manifest, decisions, benchmark, dashboard, golden = make_inputs(tmp_path)
    output_dir = tmp_path / "direct"

    result = subprocess.run(
        [
            sys.executable,
            "tools/plan_missing_evidence_collection.py",
            "--root-cause-plan-jsonl",
            str(root_plan),
            "--autonomous-decisions-jsonl",
            str(decisions),
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
    assert (output_dir / "missing_evidence_plan.jsonl").exists()
    assert (output_dir / "missing_evidence_plan_summary.md").exists()
    assert (output_dir / "missing_evidence_plan_manifest.json").exists()
