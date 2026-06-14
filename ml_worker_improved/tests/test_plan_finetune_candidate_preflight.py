from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.plan_finetune_candidate_preflight import build_finetune_candidate_preflight, write_outputs


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


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")
    return path


def source_manifest(
    path: Path,
    *,
    source_audio: Path | None = None,
    reference_audio: Path | None = None,
    gt_manifest: Path | None = None,
    include_metrics: bool = True,
) -> Path:
    payload: dict = {
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
    }
    if source_audio is not None:
        payload["source_audio_path"] = str(source_audio)
    if reference_audio is not None:
        payload["reference_vocal_path"] = str(reference_audio)
    if gt_manifest is not None:
        payload["inputs"] = {"ground_truth_manifest": str(gt_manifest)}
    if include_metrics:
        payload["objective_evidence"] = {
            "alignment": {"status": "checked"},
            "residual": {"rms_dbfs": -70.0},
            "leakage": {"max_source": "vocals"},
        }
    return write_json(path, payload)


def gt_manifest(path: Path, *, source_audio: Path, reference_audio: Path) -> Path:
    return write_json(
        path,
        {
            "ground_truth_contract": {"verified": True},
            "mix": {"path": str(source_audio)},
            "stems": {"vocals": {"path": str(reference_audio)}},
        },
    )


def root_row(
    case_id: str,
    *,
    track_hash: str | None = None,
    split: str = "train",
    stem_type: str = "vocal",
    can_enter: bool = True,
    bucket: str = "vocal_specific_failure",
    safe_next_action: str = "investigate_vocal_specific_path",
    can_change_policy: bool = False,
) -> dict:
    return {
        "case_id": case_id,
        "track_hash": track_hash or f"track-{case_id}",
        "stem_type": stem_type,
        "split": split,
        "original_decision": "hard_case",
        "dominant_failure_reason": "selected_candidate_still_has_severe_issue",
        "failure_labels": ["selected_candidate_still_has_severe_issue"],
        "root_cause_bucket": bucket,
        "safe_next_action": safe_next_action,
        "can_enter_finetune_pool": can_enter,
        "can_change_policy": can_change_policy,
        "forbidden_actions": ["no_training_start", "no_audio_export"],
        "required_evidence_before_execution": ["reference_path", "source_audio_path"],
    }


def benchmark_row(case_id: str, *, source_manifest_path: Path, split: str = "train", track_hash: str | None = None) -> dict:
    return {
        "benchmark_case_id": case_id,
        "track_hash": track_hash or f"track-{case_id}",
        "stem_type": "vocal",
        "split": split,
        "source_manifest": str(source_manifest_path),
        "dominant_failure_reason": "selected_candidate_still_has_severe_issue",
        "failure_labels": ["selected_candidate_still_has_severe_issue"],
        "failed_checks": [],
        "window_start_sec": 0.0,
        "window_end_sec": 5.0,
    }


def make_inputs(tmp_path: Path, root_rows: list[dict], benchmark_rows: list[dict], reference_rows: list[dict] | None = None) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    root = write_jsonl(tmp_path / "root_cause_plan.jsonl", root_rows)
    root_manifest = write_json(tmp_path / "root_cause_plan_manifest.json", {"schema": "root_cause_plan_manifest.v1", "row_count": len(root_rows)})
    decisions = write_jsonl(
        tmp_path / "autonomous_decision_queue.jsonl",
        [{"case_id": row["case_id"], "source_manifest": next((b["source_manifest"] for b in benchmark_rows if b["benchmark_case_id"] == row["case_id"]), "")} for row in root_rows],
    )
    benchmark = write_jsonl(tmp_path / "hard_case_benchmark.jsonl", benchmark_rows)
    golden = write_jsonl(tmp_path / "golden_corpus.jsonl", [{"track_hash": "golden-track", "training_allowed": False, "threshold_tuning_allowed": False}])
    reference = write_jsonl(tmp_path / "reference_path_recovery_plan.jsonl", reference_rows or [])
    reference_manifest = write_json(tmp_path / "reference_path_recovery_manifest.json", {"schema": "reference_path_recovery_manifest.v1", "total_rows": len(reference_rows or [])})
    return root, root_manifest, decisions, benchmark, golden, reference, reference_manifest


def build_with_inputs(tmp_path: Path, root_rows: list[dict], benchmark_rows: list[dict], reference_rows: list[dict] | None = None) -> dict:
    root, root_manifest, decisions, benchmark, golden, reference, reference_manifest = make_inputs(
        tmp_path, root_rows, benchmark_rows, reference_rows
    )
    return build_finetune_candidate_preflight(
        root_cause_plan_jsonl=root,
        root_cause_manifest_json=root_manifest,
        autonomous_decisions_jsonl=decisions,
        hard_case_benchmark_jsonl=benchmark,
        golden_jsonl=golden,
        reference_path_recovery_jsonl=reference,
        reference_path_recovery_manifest_json=reference_manifest,
    )


def rows_by_case(result: dict) -> dict[str, dict]:
    return {row["case_id"]: row for row in result["rows"]}


def test_only_can_enter_finetune_pool_rows_are_included(tmp_path: Path) -> None:
    source = touch(tmp_path / "audio" / "mix.wav")
    reference = touch(tmp_path / "audio" / "reference_vocals.flac")
    gt = gt_manifest(tmp_path / "gt" / "ground_truth_manifest.json", source_audio=source, reference_audio=reference)
    manifest = source_manifest(tmp_path / "manifest.json", source_audio=source, reference_audio=reference, gt_manifest=gt)

    result = build_with_inputs(
        tmp_path,
        [root_row("bench-included"), root_row("bench-excluded", can_enter=False)],
        [
            benchmark_row("bench-included", source_manifest_path=manifest),
            benchmark_row("bench-excluded", source_manifest_path=manifest),
        ],
    )

    assert [row["case_id"] for row in result["rows"]] == ["bench-included"]


def test_holdout_golden_and_policy_rows_are_blocked(tmp_path: Path) -> None:
    source = touch(tmp_path / "audio" / "mix.wav")
    reference = touch(tmp_path / "audio" / "reference_vocals.flac")
    gt = gt_manifest(tmp_path / "gt" / "ground_truth_manifest.json", source_audio=source, reference_audio=reference)
    manifest = source_manifest(tmp_path / "manifest.json", source_audio=source, reference_audio=reference, gt_manifest=gt)

    result = build_with_inputs(
        tmp_path,
        [
            root_row("bench-holdout", split="holdout"),
            root_row("bench-golden", track_hash="golden-track"),
            root_row("bench-policy", bucket="policy_classification_problem", safe_next_action="improve_taxonomy_mapping", can_change_policy=True),
        ],
        [
            benchmark_row("bench-holdout", source_manifest_path=manifest, split="holdout"),
            benchmark_row("bench-golden", source_manifest_path=manifest, track_hash="golden-track"),
            benchmark_row("bench-policy", source_manifest_path=manifest),
        ],
    )
    rows = rows_by_case(result)

    assert rows["bench-holdout"]["preflight_status"] == "blocked_holdout"
    assert rows["bench-golden"]["preflight_status"] == "blocked_golden"
    assert rows["bench-policy"]["preflight_status"] == "blocked_policy_issue"


def test_reference_and_source_audio_blockers_are_enforced(tmp_path: Path) -> None:
    source = touch(tmp_path / "audio" / "mix.wav")
    generated_manifest = source_manifest(tmp_path / "missing_ref.json", source_audio=source, reference_audio=None)
    reference = touch(tmp_path / "audio" / "reference_vocals.flac")
    no_source_manifest = source_manifest(tmp_path / "missing_source.json", source_audio=None, reference_audio=reference)
    ambiguous_manifest = source_manifest(tmp_path / "ambiguous_ref.json", source_audio=source, reference_audio=reference)
    outside_manifest = source_manifest(tmp_path / "outside_ref.json", source_audio=source, reference_audio=reference)

    result = build_with_inputs(
        tmp_path,
        [
            root_row("bench-ref-blocked"),
            root_row("bench-no-source"),
            root_row("bench-ambiguous-ref"),
            root_row("bench-outside-ref"),
        ],
        [
            benchmark_row("bench-ref-blocked", source_manifest_path=generated_manifest),
            benchmark_row("bench-no-source", source_manifest_path=no_source_manifest),
            benchmark_row("bench-ambiguous-ref", source_manifest_path=ambiguous_manifest),
            benchmark_row("bench-outside-ref", source_manifest_path=outside_manifest),
        ],
        [
            {
                "case_id": "bench-ref-blocked",
                "recovery_status": "blocked_missing_reference",
                "can_recover_reference_audio": False,
            },
            {
                "case_id": "bench-ambiguous-ref",
                "recovery_status": "blocked_ambiguous_reference",
                "can_recover_reference_audio": False,
            },
            {
                "case_id": "bench-outside-ref",
                "recovery_status": "blocked_reference_outside_allowed_roots",
                "can_recover_reference_audio": False,
            },
        ],
    )
    rows = rows_by_case(result)

    assert rows["bench-ref-blocked"]["preflight_status"] == "blocked_missing_reference"
    assert rows["bench-ref-blocked"]["has_reference_path"] is False
    assert rows["bench-no-source"]["preflight_status"] == "blocked_missing_source_audio"
    assert rows["bench-no-source"]["has_source_audio_path"] is False
    assert rows["bench-ambiguous-ref"]["preflight_status"] == "blocked_missing_reference"
    assert rows["bench-outside-ref"]["preflight_status"] == "blocked_missing_reference"


def test_gt_bundle_and_metric_evidence_blockers_are_enforced(tmp_path: Path) -> None:
    source = touch(tmp_path / "audio" / "mix.wav")
    reference = touch(tmp_path / "audio" / "reference_vocals.flac")
    no_gt = source_manifest(tmp_path / "no_gt.json", source_audio=source, reference_audio=reference, gt_manifest=None)
    gt = gt_manifest(tmp_path / "gt" / "ground_truth_manifest.json", source_audio=source, reference_audio=reference)
    no_metrics = source_manifest(
        tmp_path / "no_metrics.json",
        source_audio=source,
        reference_audio=reference,
        gt_manifest=gt,
        include_metrics=False,
    )

    result = build_with_inputs(
        tmp_path,
        [root_row("bench-no-gt"), root_row("bench-no-metrics")],
        [
            benchmark_row("bench-no-gt", source_manifest_path=no_gt),
            benchmark_row("bench-no-metrics", source_manifest_path=no_metrics),
        ],
    )
    rows = rows_by_case(result)

    assert rows["bench-no-gt"]["preflight_status"] == "blocked_missing_gt_bundle"
    assert rows["bench-no-metrics"]["preflight_status"] == "blocked_insufficient_evidence"


def test_metadata_eligible_rows_never_start_training_or_copy_audio(tmp_path: Path) -> None:
    source = touch(tmp_path / "audio" / "mix.wav")
    reference = touch(tmp_path / "audio" / "reference_vocals.flac")
    gt = gt_manifest(tmp_path / "gt" / "ground_truth_manifest.json", source_audio=source, reference_audio=reference)
    manifest = source_manifest(tmp_path / "manifest.json", source_audio=source, reference_audio=reference, gt_manifest=gt)

    result = build_with_inputs(
        tmp_path,
        [root_row("bench-eligible")],
        [benchmark_row("bench-eligible", source_manifest_path=manifest)],
    )
    row = rows_by_case(result)["bench-eligible"]

    assert row["preflight_status"] == "eligible_metadata_only"
    assert row["eligible_for_future_finetune"] is True
    assert row["can_build_dataset_later"] is True
    assert row["can_start_training_now"] is False


def test_output_is_deterministic_summary_matches_and_no_audio_or_dataset_is_created(tmp_path: Path) -> None:
    source = touch(tmp_path / "audio" / "mix.wav")
    reference = touch(tmp_path / "audio" / "reference_vocals.flac")
    gt = gt_manifest(tmp_path / "gt" / "ground_truth_manifest.json", source_audio=source, reference_audio=reference)
    manifest = source_manifest(tmp_path / "manifest.json", source_audio=source, reference_audio=reference, gt_manifest=gt)
    root_rows = [root_row("bench-eligible")]
    benchmark_rows = [benchmark_row("bench-eligible", source_manifest_path=manifest)]

    first = build_with_inputs(tmp_path, root_rows, benchmark_rows)
    second = build_with_inputs(tmp_path, root_rows, benchmark_rows)
    assert first["rows"] == second["rows"]
    assert first["summary"] == second["summary"]

    output_dir = tmp_path / "out"
    write_outputs(first, output_dir)
    output_rows = [
        json.loads(line)
        for line in (output_dir / "finetune_candidate_preflight.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_payload = json.loads((output_dir / "finetune_candidate_preflight_manifest.json").read_text(encoding="utf-8"))
    assert len(output_rows) == first["summary"]["total_preflight_rows"]
    assert manifest_payload["total_preflight_rows"] == len(output_rows)
    assert not list(output_dir.glob("*.flac"))
    assert not list(output_dir.glob("*.wav"))
    assert not list(output_dir.glob("*.npz"))
    assert not list(output_dir.glob("*dataset*"))
    assert manifest_payload["safety"]["audio_copied"] is False
    assert manifest_payload["safety"]["dataset_built"] is False
    assert manifest_payload["safety"]["gt_bundle_built"] is False
    assert manifest_payload["safety"]["training_started"] is False
    assert manifest_payload["safety"]["policy_changed"] is False


def test_direct_script_invocation(tmp_path: Path) -> None:
    source = touch(tmp_path / "audio" / "mix.wav")
    reference = touch(tmp_path / "audio" / "reference_vocals.flac")
    gt = gt_manifest(tmp_path / "gt" / "ground_truth_manifest.json", source_audio=source, reference_audio=reference)
    manifest = source_manifest(tmp_path / "manifest.json", source_audio=source, reference_audio=reference, gt_manifest=gt)
    root, root_manifest, decisions, benchmark, golden, reference_recovery, reference_manifest = make_inputs(
        tmp_path,
        [root_row("bench-direct")],
        [benchmark_row("bench-direct", source_manifest_path=manifest)],
    )
    output_dir = tmp_path / "direct"

    result = subprocess.run(
        [
            sys.executable,
            "tools/plan_finetune_candidate_preflight.py",
            "--root-cause-plan-jsonl",
            str(root),
            "--root-cause-manifest-json",
            str(root_manifest),
            "--autonomous-decisions-jsonl",
            str(decisions),
            "--hard-case-benchmark-jsonl",
            str(benchmark),
            "--golden-jsonl",
            str(golden),
            "--reference-path-recovery-jsonl",
            str(reference_recovery),
            "--reference-path-recovery-manifest-json",
            str(reference_manifest),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "finetune_candidate_preflight.jsonl").exists()
    assert (output_dir / "finetune_candidate_preflight_manifest.json").exists()
    assert (output_dir / "finetune_candidate_preflight_summary.md").exists()
