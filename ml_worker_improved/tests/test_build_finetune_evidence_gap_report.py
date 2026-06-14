from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.build_finetune_evidence_gap_report import build_finetune_evidence_gap_report, write_outputs


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


def preflight_row(
    case_id: str,
    *,
    status: str = "blocked_missing_reference",
    stem_type: str = "vocal",
    split: str = "train",
    track_hash: str | None = None,
    has_reference: bool = False,
    has_source: bool = True,
    has_gt: bool = False,
    has_alignment: bool = False,
    has_residual: bool = False,
    has_leakage: bool = False,
) -> dict:
    return {
        "case_id": case_id,
        "track_hash": track_hash or f"track-{case_id}",
        "stem_type": stem_type,
        "split": split,
        "source_manifest": f"/app/lab_artifacts/{case_id}/manifest.json",
        "original_root_cause_bucket": "vocal_specific_failure",
        "original_safe_next_action": "investigate_vocal_specific_path",
        "has_reference_path": has_reference,
        "has_source_audio_path": has_source,
        "has_gt_bundle": has_gt,
        "has_alignment_evidence": has_alignment,
        "has_residual_metrics": has_residual,
        "has_leakage_metrics": has_leakage,
        "eligible_for_future_finetune": False,
        "preflight_status": status,
        "required_before_training": ["recover_or_attach_verified_reference_path"],
        "forbidden_actions": ["no_training_start", "no_audio_copy"],
        "can_build_dataset_later": False,
        "can_start_training_now": False,
    }


def make_inputs(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    preflight = write_jsonl(tmp_path / "finetune_candidate_preflight.jsonl", rows)
    preflight_manifest = write_json(
        tmp_path / "finetune_candidate_preflight_manifest.json",
        {"schema": "finetune_candidate_preflight_manifest.v1", "total_preflight_rows": len(rows)},
    )
    root = write_jsonl(tmp_path / "root_cause_plan.jsonl", [{"case_id": row["case_id"]} for row in rows])
    decisions = write_jsonl(tmp_path / "autonomous_decision_queue.jsonl", [{"case_id": row["case_id"]} for row in rows])
    benchmark = write_jsonl(
        tmp_path / "hard_case_benchmark.jsonl",
        [{"benchmark_case_id": row["case_id"], "track_hash": row["track_hash"]} for row in rows],
    )
    golden = write_jsonl(
        tmp_path / "golden_corpus.jsonl",
        [{"track_hash": "golden-track", "training_allowed": False, "threshold_tuning_allowed": False}],
    )
    reference = write_jsonl(tmp_path / "reference_path_recovery_plan.jsonl", [])
    return preflight, preflight_manifest, root, decisions, benchmark, golden, reference


def build_with_rows(tmp_path: Path, rows: list[dict]) -> dict:
    preflight, preflight_manifest, root, decisions, benchmark, golden, reference = make_inputs(tmp_path, rows)
    return build_finetune_evidence_gap_report(
        finetune_preflight_jsonl=preflight,
        finetune_preflight_manifest_json=preflight_manifest,
        root_cause_plan_jsonl=root,
        autonomous_decisions_jsonl=decisions,
        hard_case_benchmark_jsonl=benchmark,
        golden_jsonl=golden,
        reference_path_recovery_jsonl=reference,
    )


def rows_by_case(result: dict) -> dict[str, dict]:
    return {row["case_id"]: row for row in result["rows"]}


def test_includes_all_preflight_rows(tmp_path: Path) -> None:
    rows = [
        preflight_row("bench-001"),
        preflight_row("bench-002", status="blocked_insufficient_evidence", stem_type="unknown", has_reference=True, has_source=True, has_gt=True),
    ]

    result = build_with_rows(tmp_path, rows)

    assert [row["case_id"] for row in result["rows"]] == ["bench-001", "bench-002"]
    assert result["summary"]["total_rows"] == 2


def test_blocked_missing_reference_rows_have_reference_blocker(tmp_path: Path) -> None:
    result = build_with_rows(tmp_path, [preflight_row("bench-vocal"), preflight_row("bench-minus", stem_type="minus")])
    rows = rows_by_case(result)

    assert rows["bench-vocal"]["exact_blocker"] == "missing_reference_vocal"
    assert rows["bench-vocal"]["missing_reference_types"] == ["reference_vocal"]
    assert "recover_or_attach_verified_reference_path" in rows["bench-vocal"]["required_to_unblock"]
    assert rows["bench-vocal"]["can_be_unblocked_by_reference_path_fix"] is True
    assert rows["bench-minus"]["exact_blocker"] == "missing_reference_minus"
    assert rows["bench-minus"]["missing_reference_types"] == ["reference_minus"]


def test_blocked_insufficient_evidence_rows_have_metric_or_provenance_blocker(tmp_path: Path) -> None:
    row = preflight_row(
        "bench-metrics",
        status="blocked_insufficient_evidence",
        stem_type="unknown",
        has_reference=True,
        has_source=True,
        has_gt=True,
        has_alignment=False,
        has_residual=False,
        has_leakage=False,
    )

    result = build_with_rows(tmp_path, [row])
    planned = rows_by_case(result)["bench-metrics"]

    assert planned["exact_blocker"] == "missing_stem_type"
    assert "missing_stem_type" in planned["missing_evidence_reasons"]
    assert "missing_alignment_evidence" in planned["missing_metric_types"]
    assert "missing_residual_metrics" in planned["missing_metric_types"]
    assert "missing_leakage_metrics" in planned["missing_metric_types"]
    assert "resolve_stem_type" in planned["required_to_unblock"]


def test_every_row_has_blocker_required_action_and_no_training(tmp_path: Path) -> None:
    rows = [
        preflight_row("bench-ref"),
        preflight_row("bench-source", status="blocked_missing_source_audio", has_reference=True, has_source=False),
        preflight_row("bench-gt", status="blocked_missing_gt_bundle", has_reference=True, has_source=True, has_gt=False),
        preflight_row("bench-holdout", status="blocked_holdout", split="holdout"),
        preflight_row("bench-golden", status="blocked_golden", track_hash="golden-track"),
        preflight_row("bench-policy", status="blocked_policy_issue"),
    ]

    result = build_with_rows(tmp_path, rows)

    for row in result["rows"]:
        assert row["exact_blocker"]
        assert row["required_to_unblock"]
        assert row["can_start_training_now"] is False
    assert rows_by_case(result)["bench-holdout"]["missing_evidence_reasons"] == ["blocked_by_holdout"]
    assert rows_by_case(result)["bench-golden"]["missing_evidence_reasons"] == ["blocked_by_golden"]
    assert rows_by_case(result)["bench-policy"]["missing_evidence_reasons"] == ["blocked_by_policy"]


def test_output_is_deterministic_summary_matches_and_no_audio_or_dataset_is_created(tmp_path: Path) -> None:
    rows = [
        preflight_row("bench-ref"),
        preflight_row("bench-metrics", status="blocked_insufficient_evidence", stem_type="unknown", has_reference=True, has_source=True, has_gt=True),
    ]
    first = build_with_rows(tmp_path, rows)
    second = build_with_rows(tmp_path, rows)
    assert first["rows"] == second["rows"]
    assert first["summary"] == second["summary"]

    output_dir = tmp_path / "out"
    write_outputs(first, output_dir)
    output_rows = [
        json.loads(line)
        for line in (output_dir / "finetune_evidence_gap_report.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads((output_dir / "finetune_evidence_gap_manifest.json").read_text(encoding="utf-8"))
    assert len(output_rows) == first["summary"]["total_rows"]
    assert manifest["total_rows"] == len(output_rows)
    assert not list(output_dir.glob("*.flac"))
    assert not list(output_dir.glob("*.wav"))
    assert not list(output_dir.glob("*.npz"))
    assert not list(output_dir.glob("*dataset*"))
    assert manifest["safety"]["audio_copied"] is False
    assert manifest["safety"]["audio_read"] is False
    assert manifest["safety"]["audio_hashed"] is False
    assert manifest["safety"]["dataset_built"] is False
    assert manifest["safety"]["gt_bundle_built"] is False
    assert manifest["safety"]["training_started"] is False
    assert manifest["safety"]["policy_changed"] is False


def test_direct_script_invocation(tmp_path: Path) -> None:
    preflight, preflight_manifest, root, decisions, benchmark, golden, reference = make_inputs(
        tmp_path,
        [preflight_row("bench-direct")],
    )
    output_dir = tmp_path / "direct"

    result = subprocess.run(
        [
            sys.executable,
            "tools/build_finetune_evidence_gap_report.py",
            "--finetune-preflight-jsonl",
            str(preflight),
            "--finetune-preflight-manifest-json",
            str(preflight_manifest),
            "--root-cause-plan-jsonl",
            str(root),
            "--autonomous-decisions-jsonl",
            str(decisions),
            "--hard-case-benchmark-jsonl",
            str(benchmark),
            "--golden-jsonl",
            str(golden),
            "--reference-path-recovery-jsonl",
            str(reference),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "finetune_evidence_gap_report.jsonl").exists()
    assert (output_dir / "finetune_evidence_gap_manifest.json").exists()
    assert (output_dir / "finetune_evidence_gap_summary.md").exists()
