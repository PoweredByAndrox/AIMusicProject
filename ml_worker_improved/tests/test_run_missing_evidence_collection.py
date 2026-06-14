from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import wave
from pathlib import Path

from tools.run_missing_evidence_collection import run_missing_evidence_collection, write_outputs


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


def write_tone_wav(path: Path, *, seconds: float = 0.2, sample_rate: int = 8000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        for index in range(frame_count):
            sample = int(12000 * math.sin(2.0 * math.pi * 440.0 * index / sample_rate))
            handle.writeframes(struct.pack("<h", sample))
    return path


def plan_row(
    idx: int,
    *,
    case_id: str | None = None,
    source_manifest: str = "",
    audio_paths: list[str] | None = None,
    reference_paths: list[str] | None = None,
    can_collect: bool = True,
    split: str = "train",
    track_hash: str | None = None,
    stem_type: str = "vocal",
    tools: list[str] | None = None,
    missing: list[str] | None = None,
) -> dict:
    return {
        "case_id": case_id or f"bench-{idx:03d}",
        "track_hash": track_hash or f"track-{idx:03d}",
        "stem_type": stem_type,
        "split": split,
        "source_manifest": source_manifest,
        "dominant_failure_reason": "unknown_blocked_reason",
        "failure_labels": ["unknown_blocked_reason"],
        "original_decision": "hard_case",
        "root_cause_bucket": "unknown_needs_more_evidence",
        "missing_evidence_types": missing
        or ["missing_reference_metrics", "missing_residual_metrics", "missing_leakage_metrics"],
        "required_tools_to_run": tools
        or ["stem_quality_control", "waveform_forensic_compare", "build_flstudio_ground_truth_bundle"],
        "required_inputs": {
            "source_manifest": source_manifest,
            "track_hash": track_hash or f"track-{idx:03d}",
            "stem_type": stem_type,
            "window_start_sec": 1.0,
            "window_end_sec": 2.0,
            "audio_paths_already_in_manifest": audio_paths or [],
            "reference_or_ground_truth_paths_already_in_manifest": reference_paths or [],
        },
        "can_collect_automatically": can_collect,
        "can_use_for_finetune_after_collection": can_collect and split == "train",
        "blocked_reason": "" if can_collect else "stem_type_missing_or_unknown",
        "forbidden_actions": ["no_audio_export_as_result", "no_training_start"],
    }


def make_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    audio_path = write_tone_wav(tmp_path / "audio" / "candidate.wav")
    source_manifest = write_json(
        tmp_path / "manifests" / "source.json",
        {
            "delivery_status": "blocked",
            "delivery_allowed": False,
            "presentation_status": "blocked_from_listen_pack",
            "selected_artifacts": {"diagnostic_vocals_flac": str(audio_path)},
        },
    )
    holdout_manifest = write_json(
        tmp_path / "manifests" / "holdout.json",
        {
            "delivery_status": "blocked",
            "delivery_allowed": False,
            "presentation_status": "blocked_from_listen_pack",
            "selected_artifacts": {"diagnostic_vocals_flac": str(audio_path)},
        },
    )
    rows = [
        plan_row(1, source_manifest=str(source_manifest), audio_paths=[str(audio_path)]),
        plan_row(2, case_id="bench-blocked", can_collect=False, stem_type="unknown"),
        plan_row(3, case_id="bench-missing-input", source_manifest=str(tmp_path / "missing.json"), audio_paths=[]),
        plan_row(
            4,
            case_id="bench-holdout",
            source_manifest=str(holdout_manifest),
            audio_paths=[str(audio_path)],
            split="holdout",
            tools=["stem_quality_control"],
        ),
        plan_row(
            5,
            case_id="bench-golden",
            source_manifest=str(source_manifest),
            audio_paths=[str(audio_path)],
            track_hash="golden-track",
            tools=["stem_quality_control"],
        ),
    ]
    plan = write_jsonl(tmp_path / "missing_evidence_plan.jsonl", rows)
    manifest = write_json(
        tmp_path / "missing_evidence_plan_manifest.json",
        {"schema": "missing_evidence_plan_manifest.v1", "row_count": len(rows), "can_collect_automatically_count": 4},
    )
    root = write_jsonl(tmp_path / "root_cause_plan.jsonl", [{"case_id": row["case_id"]} for row in rows])
    decisions = write_jsonl(tmp_path / "autonomous_decision_queue.jsonl", [{"case_id": row["case_id"]} for row in rows])
    benchmark = write_jsonl(tmp_path / "hard_case_benchmark.jsonl", [{"benchmark_case_id": row["case_id"]} for row in rows])
    golden = write_jsonl(
        tmp_path / "golden_corpus.jsonl",
        [{"track_hash": "golden-track", "training_allowed": False, "threshold_tuning_allowed": False}],
    )
    return plan, manifest, root, decisions, benchmark, golden


def by_case(rows: list[dict]) -> dict[str, dict]:
    return {row["case_id"]: row for row in rows}


def test_processes_only_eligible_rows_and_skips_blocked_rows(tmp_path: Path) -> None:
    plan, manifest, root, decisions, benchmark, golden = make_inputs(tmp_path)

    result = run_missing_evidence_collection(
        missing_evidence_plan_jsonl=plan,
        missing_evidence_manifest_json=manifest,
        root_cause_plan_jsonl=root,
        autonomous_decisions_jsonl=decisions,
        hard_case_benchmark_jsonl=benchmark,
        golden_jsonl=golden,
    )

    collected = by_case(result["collected_rows"])
    failed = by_case(result["failed_rows"])
    assert "bench-001" in collected
    assert "bench-blocked" not in collected
    assert failed["bench-blocked"]["collection_status"] == "skipped_blocked"
    assert failed["bench-blocked"]["tools_executed"] == []
    assert result["summary"]["blocked_rows_skipped"] == 1


def test_missing_inputs_skip_tools_without_crashing_and_never_promote(tmp_path: Path) -> None:
    plan, manifest, root, decisions, benchmark, golden = make_inputs(tmp_path)

    result = run_missing_evidence_collection(
        missing_evidence_plan_jsonl=plan,
        missing_evidence_manifest_json=manifest,
        root_cause_plan_jsonl=root,
        autonomous_decisions_jsonl=decisions,
        hard_case_benchmark_jsonl=benchmark,
        golden_jsonl=golden,
    )

    collected = by_case(result["collected_rows"])
    eligible = collected["bench-001"]
    assert "stem_quality_control" in eligible["tools_executed"]
    assert "waveform_forensic_compare" in eligible["tools_skipped"]
    assert "stem_quality_control" in eligible["evidence_collected"]
    assert eligible["collection_status"] == "partially_collected"
    assert eligible["can_promote_after_collection"] is False
    assert eligible["audio_exported"] is False
    assert eligible["training_started"] is False
    assert eligible["policy_changed"] is False

    missing_input = collected["bench-missing-input"]
    assert missing_input["collection_status"] == "failed"
    assert missing_input["tools_executed"] == []
    assert missing_input["evidence_missing_after_run"]


def test_holdout_and_golden_rows_cannot_enter_finetune_after_collection(tmp_path: Path) -> None:
    plan, manifest, root, decisions, benchmark, golden = make_inputs(tmp_path)

    result = run_missing_evidence_collection(
        missing_evidence_plan_jsonl=plan,
        missing_evidence_manifest_json=manifest,
        root_cause_plan_jsonl=root,
        autonomous_decisions_jsonl=decisions,
        hard_case_benchmark_jsonl=benchmark,
        golden_jsonl=golden,
    )

    collected = by_case(result["collected_rows"])
    assert collected["bench-holdout"]["can_enter_finetune_pool_after_collection"] is False
    assert collected["bench-golden"]["can_enter_finetune_pool_after_collection"] is False


def test_output_is_deterministic_summary_matches_and_no_audio_exported(tmp_path: Path) -> None:
    plan, manifest, root, decisions, benchmark, golden = make_inputs(tmp_path)

    first = run_missing_evidence_collection(
        missing_evidence_plan_jsonl=plan,
        missing_evidence_manifest_json=manifest,
        root_cause_plan_jsonl=root,
        autonomous_decisions_jsonl=decisions,
        hard_case_benchmark_jsonl=benchmark,
        golden_jsonl=golden,
    )
    second = run_missing_evidence_collection(
        missing_evidence_plan_jsonl=plan,
        missing_evidence_manifest_json=manifest,
        root_cause_plan_jsonl=root,
        autonomous_decisions_jsonl=decisions,
        hard_case_benchmark_jsonl=benchmark,
        golden_jsonl=golden,
    )
    assert first["collected_rows"] == second["collected_rows"]
    assert first["failed_rows"] == second["failed_rows"]

    output_dir = tmp_path / "out"
    write_outputs(first, output_dir)
    collected_rows = [
        json.loads(line)
        for line in (output_dir / "collected_evidence.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failed_rows = [
        json.loads(line)
        for line in (output_dir / "failed_evidence_collection.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_payload = json.loads((output_dir / "collected_evidence_manifest.json").read_text(encoding="utf-8"))
    assert len(collected_rows) == first["summary"]["eligible_rows_processed"]
    assert len(failed_rows) == first["summary"]["blocked_rows_skipped"]
    assert manifest_payload["eligible_rows_processed"] == len(collected_rows)
    assert manifest_payload["blocked_rows_skipped"] == len(failed_rows)
    assert not list(output_dir.glob("*.flac"))
    assert not list(output_dir.glob("*.wav"))
    assert manifest_payload["safety"]["audio_exported"] is False
    assert manifest_payload["safety"]["audio_copied"] is False
    assert manifest_payload["safety"]["training_started"] is False
    assert manifest_payload["safety"]["policy_changed"] is False


def test_direct_script_invocation(tmp_path: Path) -> None:
    plan, manifest, root, decisions, benchmark, golden = make_inputs(tmp_path)
    output_dir = tmp_path / "direct"

    result = subprocess.run(
        [
            sys.executable,
            "tools/run_missing_evidence_collection.py",
            "--missing-evidence-plan-jsonl",
            str(plan),
            "--missing-evidence-manifest-json",
            str(manifest),
            "--root-cause-plan-jsonl",
            str(root),
            "--autonomous-decisions-jsonl",
            str(decisions),
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
    assert (output_dir / "collected_evidence.jsonl").exists()
    assert (output_dir / "failed_evidence_collection.jsonl").exists()
    assert (output_dir / "collected_evidence_manifest.json").exists()
