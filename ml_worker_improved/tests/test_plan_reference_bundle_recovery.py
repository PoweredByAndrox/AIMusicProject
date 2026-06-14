from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.plan_reference_bundle_recovery import build_reference_bundle_recovery_plan, write_outputs


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
    generated_vocal: Path | None = None,
    reference_vocal: Path | None = None,
    generated_minus: Path | None = None,
    flstudio_stem: Path | None = None,
) -> Path:
    payload: dict = {
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
    }
    if source_audio is not None:
        payload["source_audio_path"] = str(source_audio)
    selected_artifacts: dict[str, str] = {}
    if generated_vocal is not None:
        selected_artifacts["diagnostic_vocals_flac"] = str(generated_vocal)
    if generated_minus is not None:
        selected_artifacts["diagnostic_minus_flac"] = str(generated_minus)
    if selected_artifacts:
        payload["selected_artifacts"] = selected_artifacts
    if reference_vocal is not None:
        payload["reference_vocal_path"] = str(reference_vocal)
    if flstudio_stem is not None:
        payload["flstudio_stem_export"] = str(flstudio_stem)
    return write_json(path, payload)


def evidence_row(
    idx: int,
    *,
    case_id: str | None = None,
    source_manifest_path: Path | str = "",
    track_hash: str | None = None,
    split: str = "train",
    stem_type: str = "vocal",
    missing: list[str] | None = None,
) -> dict:
    return {
        "case_id": case_id or f"bench-{idx:03d}",
        "track_hash": track_hash or f"track-{idx:03d}",
        "stem_type": stem_type,
        "split": split,
        "source_manifest": str(source_manifest_path),
        "tools_requested": ["build_flstudio_ground_truth_bundle", "waveform_forensic_compare"],
        "tools_executed": ["stem_quality_control"],
        "tools_skipped": ["build_flstudio_ground_truth_bundle", "waveform_forensic_compare"],
        "evidence_collected": {"source_manifest": {"exists": bool(source_manifest_path)}},
        "evidence_missing_after_run": missing
        or [
            "missing_ground_truth_bundle",
            "missing_reference_metrics",
            "missing_alignment_check",
            "missing_residual_metrics",
            "missing_leakage_metrics",
        ],
        "collection_status": "partially_collected",
        "can_reclassify_after_collection": True,
        "can_enter_finetune_pool_after_collection": split == "train",
        "can_promote_after_collection": False,
        "audio_exported": False,
        "training_started": False,
        "policy_changed": False,
    }


def make_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    source_audio = touch(tmp_path / "audio" / "mix.wav")
    generated_vocal = touch(tmp_path / "audio" / "generated_vocals.flac")
    reference_vocal = touch(tmp_path / "audio" / "reference_vocals.flac")
    flstudio_stem = touch(tmp_path / "flstudio_exports" / "track - vocal.wav")
    generated_minus = touch(tmp_path / "audio" / "generated_minus.flac")

    recoverable_manifest = source_manifest(
        tmp_path / "manifests" / "recoverable.json",
        source_audio=source_audio,
        generated_vocal=generated_vocal,
        reference_vocal=reference_vocal,
        generated_minus=generated_minus,
        flstudio_stem=flstudio_stem,
    )
    partial_manifest = source_manifest(tmp_path / "manifests" / "partial.json", source_audio=source_audio)
    no_source_manifest = source_manifest(
        tmp_path / "manifests" / "no_source.json",
        generated_vocal=generated_vocal,
        reference_vocal=reference_vocal,
    )
    no_reference_manifest = source_manifest(
        tmp_path / "manifests" / "no_reference.json",
        source_audio=source_audio,
        generated_vocal=generated_vocal,
    )

    rows = [
        evidence_row(1, source_manifest_path=recoverable_manifest),
        evidence_row(2, case_id="bench-partial", source_manifest_path=partial_manifest),
        evidence_row(3, case_id="bench-no-source", source_manifest_path=no_source_manifest),
        evidence_row(4, case_id="bench-no-reference", source_manifest_path=no_reference_manifest),
        evidence_row(5, case_id="bench-holdout", source_manifest_path=recoverable_manifest, split="holdout"),
        evidence_row(6, case_id="bench-golden", source_manifest_path=recoverable_manifest, track_hash="golden-track"),
    ]
    collected = write_jsonl(tmp_path / "collected_evidence.jsonl", rows)
    collected_manifest = write_json(
        tmp_path / "collected_evidence_manifest.json",
        {"schema": "collected_evidence_manifest.v1", "eligible_rows_processed": len(rows)},
    )
    failed = write_jsonl(tmp_path / "failed_evidence_collection.jsonl", [{"case_id": "blocked", "collection_status": "skipped_blocked"}])
    missing = write_jsonl(tmp_path / "missing_evidence_plan.jsonl", [{"case_id": row["case_id"]} for row in rows])
    benchmark = write_jsonl(tmp_path / "hard_case_benchmark.jsonl", [{"benchmark_case_id": row["case_id"]} for row in rows])
    golden = write_jsonl(
        tmp_path / "golden_corpus.jsonl",
        [{"track_hash": "golden-track", "training_allowed": False, "threshold_tuning_allowed": False}],
    )
    return collected, collected_manifest, failed, missing, benchmark, golden


def by_case(rows: list[dict]) -> dict[str, dict]:
    return {row["case_id"]: row for row in rows}


def test_classifies_recoverable_partial_and_blocked_rows(tmp_path: Path) -> None:
    collected, collected_manifest, failed, missing, benchmark, golden = make_inputs(tmp_path)

    result = build_reference_bundle_recovery_plan(
        collected_evidence_jsonl=collected,
        collected_evidence_manifest_json=collected_manifest,
        failed_evidence_jsonl=failed,
        missing_evidence_plan_jsonl=missing,
        hard_case_benchmark_jsonl=benchmark,
        golden_jsonl=golden,
    )

    rows = by_case(result["rows"])
    assert rows["bench-001"]["recovery_status"] == "recoverable"
    assert rows["bench-001"]["can_run_alignment_check"] is True
    assert rows["bench-001"]["can_run_reference_metrics"] is True
    assert rows["bench-001"]["can_build_gt_bundle"] is True
    assert rows["bench-partial"]["recovery_status"] == "partially_recoverable"
    assert rows["bench-no-source"]["recovery_status"] == "blocked_missing_source_audio"
    assert rows["bench-no-reference"]["recovery_status"] == "blocked_missing_reference"
    assert rows["bench-no-reference"]["can_run_reference_metrics"] is False


def test_holdout_and_golden_protection_is_preserved(tmp_path: Path) -> None:
    collected, collected_manifest, failed, missing, benchmark, golden = make_inputs(tmp_path)

    result = build_reference_bundle_recovery_plan(
        collected_evidence_jsonl=collected,
        collected_evidence_manifest_json=collected_manifest,
        failed_evidence_jsonl=failed,
        missing_evidence_plan_jsonl=missing,
        hard_case_benchmark_jsonl=benchmark,
        golden_jsonl=golden,
    )

    rows = by_case(result["rows"])
    assert "holdout_protected" in rows["bench-holdout"]["forbidden_actions"]
    assert "golden_protected_no_training_or_tuning" in rows["bench-golden"]["forbidden_actions"]


def test_output_is_deterministic_summary_matches_and_no_audio_or_bundle_is_created(tmp_path: Path) -> None:
    collected, collected_manifest, failed, missing, benchmark, golden = make_inputs(tmp_path)

    first = build_reference_bundle_recovery_plan(
        collected_evidence_jsonl=collected,
        collected_evidence_manifest_json=collected_manifest,
        failed_evidence_jsonl=failed,
        missing_evidence_plan_jsonl=missing,
        hard_case_benchmark_jsonl=benchmark,
        golden_jsonl=golden,
    )
    second = build_reference_bundle_recovery_plan(
        collected_evidence_jsonl=collected,
        collected_evidence_manifest_json=collected_manifest,
        failed_evidence_jsonl=failed,
        missing_evidence_plan_jsonl=missing,
        hard_case_benchmark_jsonl=benchmark,
        golden_jsonl=golden,
    )
    assert first["rows"] == second["rows"]

    output_dir = tmp_path / "out"
    write_outputs(first, output_dir)
    output_rows = [
        json.loads(line)
        for line in (output_dir / "reference_bundle_recovery_plan.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_payload = json.loads((output_dir / "reference_bundle_recovery_manifest.json").read_text(encoding="utf-8"))
    assert len(output_rows) == first["summary"]["total_recovery_rows"]
    assert manifest_payload["total_recovery_rows"] == len(output_rows)
    assert not list(output_dir.glob("*.flac"))
    assert not list(output_dir.glob("*.wav"))
    assert not list(output_dir.glob("ground_truth_manifest.json"))
    assert manifest_payload["safety"]["audio_copied"] is False
    assert manifest_payload["safety"]["gt_bundle_built"] is False
    assert manifest_payload["safety"]["training_started"] is False
    assert manifest_payload["safety"]["policy_changed"] is False


def test_direct_script_invocation(tmp_path: Path) -> None:
    collected, collected_manifest, failed, missing, benchmark, golden = make_inputs(tmp_path)
    output_dir = tmp_path / "direct"

    result = subprocess.run(
        [
            sys.executable,
            "tools/plan_reference_bundle_recovery.py",
            "--collected-evidence-jsonl",
            str(collected),
            "--collected-evidence-manifest-json",
            str(collected_manifest),
            "--failed-evidence-jsonl",
            str(failed),
            "--missing-evidence-plan-jsonl",
            str(missing),
            "--hard-case-benchmark-jsonl",
            str(benchmark),
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
    assert (output_dir / "reference_bundle_recovery_plan.jsonl").exists()
    assert (output_dir / "reference_bundle_recovery_manifest.json").exists()
    assert (output_dir / "reference_bundle_recovery_summary.md").exists()
