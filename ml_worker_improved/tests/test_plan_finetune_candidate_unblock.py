from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.plan_finetune_candidate_unblock import plan_finetune_candidate_unblock, write_outputs


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def evidence_row(
    case_id: str,
    *,
    exact_blocker: str,
    stem_type: str = "vocal",
    missing_reasons: list[str] | None = None,
    missing_reference_types: list[str] | None = None,
    missing_source_types: list[str] | None = None,
    missing_metric_types: list[str] | None = None,
    missing_bundle_types: list[str] | None = None,
    split: str = "train",
) -> dict:
    return {
        "case_id": case_id,
        "track_hash": f"track-{case_id}",
        "stem_type": stem_type,
        "split": split,
        "source_manifest": f"/app/manifests/{case_id}.json",
        "preflight_status": "blocked_missing_reference" if exact_blocker.startswith("missing_reference") else "blocked_insufficient_evidence",
        "original_root_cause_bucket": "current_model_insufficient",
        "original_safe_next_action": "create_candidate_for_future_finetune_pool",
        "missing_evidence_reasons": missing_reasons or [exact_blocker],
        "missing_reference_types": missing_reference_types or [],
        "missing_source_types": missing_source_types or [],
        "missing_metric_types": missing_metric_types or [],
        "missing_bundle_types": missing_bundle_types or [],
        "exact_blocker": exact_blocker,
        "required_to_unblock": ["recover_or_attach_verified_reference_path"]
        if exact_blocker.startswith("missing_reference")
        else ["resolve_stem_type"],
        "can_be_unblocked_by_manifest_fix": exact_blocker == "missing_stem_type",
        "can_be_unblocked_by_reference_path_fix": exact_blocker.startswith("missing_reference"),
        "can_be_unblocked_by_docker_mount": bool(missing_source_types),
        "can_be_unblocked_by_gt_bundle_build": bool(missing_bundle_types),
        "can_be_unblocked_automatically": False,
        "can_start_training_now": False,
        "forbidden_actions": ["do_not_start_training", "do_not_copy_or_export_audio"],
    }


def make_56_evidence_rows() -> list[dict]:
    rows: list[dict] = []
    for index in range(30):
        rows.append(
            evidence_row(
                f"bench-vocal-{index:02d}",
                exact_blocker="missing_reference_vocal",
                stem_type="vocal",
                missing_reasons=["missing_reference_vocal", "missing_gt_bundle", "missing_alignment_evidence"],
                missing_reference_types=["reference_vocal"],
                missing_metric_types=["missing_alignment_evidence"],
                missing_bundle_types=["gt_bundle"],
            )
        )
    for index in range(18):
        rows.append(
            evidence_row(
                f"bench-minus-{index:02d}",
                exact_blocker="missing_reference_minus",
                stem_type="minus",
                missing_reasons=["missing_reference_minus", "missing_gt_bundle"],
                missing_reference_types=["reference_minus"],
                missing_bundle_types=["gt_bundle"],
            )
        )
    for index in range(8):
        rows.append(
            evidence_row(
                f"bench-stem-{index:02d}",
                exact_blocker="missing_stem_type",
                stem_type="unknown",
                missing_reasons=["missing_stem_type", "missing_residual_metrics"],
                missing_source_types=["stem_type"],
                missing_metric_types=["missing_residual_metrics"],
            )
        )
    return rows


def make_inputs(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    evidence_jsonl = write_jsonl(tmp_path / "finetune_evidence_gap_report.jsonl", rows)
    evidence_manifest = write_json(
        tmp_path / "finetune_evidence_gap_manifest.json",
        {"schema": "finetune_evidence_gap_report.v1", "row_count": len(rows), "total_rows": len(rows)},
    )
    preflight_jsonl = write_jsonl(tmp_path / "finetune_candidate_preflight.jsonl", rows)
    root_jsonl = write_jsonl(tmp_path / "root_cause_plan.jsonl", [])
    benchmark_jsonl = write_jsonl(tmp_path / "hard_case_benchmark.jsonl", rows)
    golden_jsonl = write_jsonl(tmp_path / "golden_corpus.jsonl", [])
    reference_recovery_jsonl = write_jsonl(tmp_path / "reference_path_recovery_plan.jsonl", [])
    return (
        evidence_jsonl,
        evidence_manifest,
        preflight_jsonl,
        root_jsonl,
        benchmark_jsonl,
        golden_jsonl,
        reference_recovery_jsonl,
    )


def build_with_rows(tmp_path: Path, rows: list[dict]) -> dict:
    inputs = make_inputs(tmp_path, rows)
    return plan_finetune_candidate_unblock(
        evidence_gap_report_jsonl=inputs[0],
        evidence_gap_manifest_json=inputs[1],
        finetune_preflight_jsonl=inputs[2],
        root_cause_plan_jsonl=inputs[3],
        hard_case_benchmark_jsonl=inputs[4],
        golden_jsonl=inputs[5],
        reference_path_recovery_jsonl=inputs[6],
    )


def rows_by_case(result: dict) -> dict[str, dict]:
    return {row["case_id"]: row for row in result["rows"]}


def test_includes_all_56_rows_and_maps_required_actions(tmp_path: Path) -> None:
    result = build_with_rows(tmp_path, make_56_evidence_rows())
    rows = result["rows"]
    assert len(rows) == 56
    assert result["summary"]["unblock_action_type_counts"] == {
        "provide_reference_minus_path": 18,
        "provide_reference_vocal_path": 30,
        "repair_stem_type_metadata": 8,
    }
    assert result["summary"]["required_reference_vocal_count"] == 30
    assert result["summary"]["required_reference_minus_count"] == 18
    assert result["summary"]["required_manifest_fix_count"] == 8


def test_reference_rows_require_user_file_and_reference_path(tmp_path: Path) -> None:
    result = build_with_rows(
        tmp_path,
        [
            evidence_row(
                "bench-vocal",
                exact_blocker="missing_reference_vocal",
                missing_reference_types=["reference_vocal"],
            ),
            evidence_row(
                "bench-minus",
                exact_blocker="missing_reference_minus",
                stem_type="minus",
                missing_reference_types=["reference_minus"],
            ),
        ],
    )
    rows = rows_by_case(result)
    vocal = rows["bench-vocal"]
    minus = rows["bench-minus"]
    assert vocal["unblock_action_type"] == "provide_reference_vocal_path"
    assert vocal["unblock_status"] == "blocked_waiting_reference_vocal"
    assert vocal["required_reference_type"] == "reference_vocal"
    assert vocal["requires_user_file"] is True
    assert vocal["requires_reference_path"] is True
    assert minus["unblock_action_type"] == "provide_reference_minus_path"
    assert minus["unblock_status"] == "blocked_waiting_reference_minus"
    assert minus["required_reference_type"] == "reference_minus"


def test_missing_stem_type_maps_to_manifest_fix(tmp_path: Path) -> None:
    result = build_with_rows(
        tmp_path,
        [
            evidence_row(
                "bench-stem",
                exact_blocker="missing_stem_type",
                stem_type="unknown",
                missing_reasons=["missing_stem_type"],
                missing_source_types=["stem_type"],
            )
        ],
    )
    row = result["rows"][0]
    assert row["unblock_action_type"] == "repair_stem_type_metadata"
    assert row["unblock_status"] == "blocked_waiting_manifest_fix"
    assert row["required_manifest_fields"] == ["stem_type"]
    assert row["requires_manifest_edit"] is True
    assert row["can_start_training_now"] is False


def test_no_training_audio_dataset_or_gt_bundle_outputs(tmp_path: Path) -> None:
    result = build_with_rows(tmp_path, make_56_evidence_rows())
    assert result["summary"]["can_start_training_now_count"] == 0
    assert result["summary"]["safety"] == {
        "audio_copied": False,
        "audio_read": False,
        "audio_hashed": False,
        "dataset_built": False,
        "gt_bundle_built": False,
        "training_started": False,
        "policy_changed": False,
        "models_changed": False,
        "weights_added": False,
        "cleanup_routes_added": False,
        "thresholds_changed": False,
    }
    output_dir = tmp_path / "out"
    write_outputs(result, output_dir)
    assert not list(output_dir.rglob("*.flac"))
    assert not list(output_dir.rglob("*.wav"))
    assert not list(output_dir.rglob("*.mp3"))
    assert not [path for path in output_dir.rglob("*") if "dataset" in path.name.lower()]
    assert not [path for path in output_dir.rglob("*") if "gt_bundle" in path.name.lower()]


def test_output_deterministic_and_summary_matches_jsonl(tmp_path: Path) -> None:
    rows = make_56_evidence_rows()
    first = build_with_rows(tmp_path, rows)
    second = build_with_rows(tmp_path, rows)
    assert first["rows"] == second["rows"]
    assert first["summary"] == second["summary"]

    output_dir = tmp_path / "out"
    write_outputs(first, output_dir)
    output_rows = [
        json.loads(line)
        for line in (output_dir / "finetune_candidate_unblock_plan.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads((output_dir / "finetune_candidate_unblock_manifest.json").read_text(encoding="utf-8"))
    assert len(output_rows) == first["summary"]["total_rows"]
    assert manifest["total_rows"] == len(output_rows)
    assert manifest["summary"]["unblock_action_type_counts"] == first["summary"]["unblock_action_type_counts"]


def test_direct_script_invocation(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, make_56_evidence_rows())
    output_dir = tmp_path / "cli-out"
    script = Path(__file__).resolve().parents[1] / "tools" / "plan_finetune_candidate_unblock.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--evidence-gap-report-jsonl",
            str(inputs[0]),
            "--evidence-gap-manifest-json",
            str(inputs[1]),
            "--finetune-preflight-jsonl",
            str(inputs[2]),
            "--root-cause-plan-jsonl",
            str(inputs[3]),
            "--hard-case-benchmark-jsonl",
            str(inputs[4]),
            "--golden-jsonl",
            str(inputs[5]),
            "--reference-path-recovery-jsonl",
            str(inputs[6]),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"total_rows": 56' in completed.stdout
    assert (output_dir / "finetune_candidate_unblock_plan.jsonl").exists()
