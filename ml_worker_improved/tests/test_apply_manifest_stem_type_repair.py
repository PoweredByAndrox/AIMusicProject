from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.apply_manifest_stem_type_repair import apply_manifest_stem_type_repair, write_outputs


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


def source_manifest(path: Path, stem_names: list[str]) -> Path:
    return write_json(path, {"inputs": {"stem_names": stem_names}, "existing": "unchanged"})


def unblock_row(
    case_id: str,
    *,
    source_manifest_path: Path,
    can_apply: bool = True,
    action: str = "repair_stem_type_metadata",
    requires_user_file: bool = False,
) -> dict:
    return {
        "case_id": case_id,
        "track_hash": f"track-{case_id}",
        "stem_type": "unknown",
        "split": "train",
        "source_manifest": str(source_manifest_path),
        "exact_blocker": "missing_stem_type",
        "required_to_unblock": ["resolve_stem_type"],
        "unblock_action_type": action,
        "unblock_status": "blocked_waiting_manifest_fix",
        "required_reference_type": "",
        "required_manifest_fields": ["stem_type"],
        "required_docker_mount": "",
        "required_gt_bundle_inputs": [],
        "can_apply_automatically": can_apply,
        "requires_user_file": requires_user_file,
        "requires_manifest_edit": True,
        "requires_reference_path": False,
        "requires_gt_bundle_build_later": False,
        "can_enter_dataset_after_unblock": can_apply,
        "can_start_training_now": False,
        "forbidden_actions": ["do_not_start_training", "do_not_copy_or_export_audio"],
    }


def make_inputs(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path, Path, Path, Path, Path]:
    unblock_jsonl = write_jsonl(tmp_path / "finetune_candidate_unblock_plan.jsonl", rows)
    unblock_manifest = write_json(
        tmp_path / "finetune_candidate_unblock_manifest.json",
        {"schema": "finetune_candidate_unblock_plan.v1", "row_count": len(rows), "total_rows": len(rows)},
    )
    evidence_jsonl = write_jsonl(tmp_path / "finetune_evidence_gap_report.jsonl", rows)
    preflight_jsonl = write_jsonl(tmp_path / "finetune_candidate_preflight.jsonl", rows)
    benchmark_jsonl = write_jsonl(tmp_path / "hard_case_benchmark.jsonl", rows)
    golden_jsonl = write_jsonl(tmp_path / "golden_corpus.jsonl", [])
    return unblock_jsonl, unblock_manifest, evidence_jsonl, preflight_jsonl, benchmark_jsonl, golden_jsonl


def build_with_rows(tmp_path: Path, rows: list[dict]) -> dict:
    inputs = make_inputs(tmp_path, rows)
    return apply_manifest_stem_type_repair(
        unblock_plan_jsonl=inputs[0],
        unblock_manifest_json=inputs[1],
        evidence_gap_report_jsonl=inputs[2],
        finetune_preflight_jsonl=inputs[3],
        hard_case_benchmark_jsonl=inputs[4],
        golden_jsonl=inputs[5],
    )


def rows_by_case(result: dict) -> dict[str, dict]:
    return {row["case_id"]: row for row in result["rows"]}


def test_only_repair_stem_type_metadata_rows_are_considered(tmp_path: Path) -> None:
    manifest = source_manifest(tmp_path / "manifest.json", ["drums"])
    result = build_with_rows(
        tmp_path,
        [
            unblock_row("repair", source_manifest_path=manifest),
            unblock_row("reference", source_manifest_path=manifest, action="provide_reference_vocal_path", requires_user_file=True),
        ],
    )
    assert [row["case_id"] for row in result["rows"]] == ["repair"]
    assert result["summary"]["total_repair_candidates"] == 1


def test_overlay_patch_created_for_safe_automatic_row_without_mutating_manifest(tmp_path: Path) -> None:
    manifest = source_manifest(tmp_path / "manifest.json", ["drums"])
    before = manifest.read_text(encoding="utf-8")
    result = build_with_rows(tmp_path, [unblock_row("repair", source_manifest_path=manifest)])
    row = result["rows"][0]
    assert row["repair_status"] == "repaired_overlay_created"
    assert row["original_stem_type"] == "unknown"
    assert row["inferred_stem_type"] == "drums"
    assert row["overlay_patch"] == {
        "op": "add_or_replace",
        "path": "/stem_type",
        "value": "drums",
    }
    assert row["can_re_run_preflight_after_overlay"] is True
    assert manifest.read_text(encoding="utf-8") == before
    assert result["overlays"][0]["overlay_patch"] == row["overlay_patch"]


def test_can_apply_false_is_skipped_and_ambiguous_stem_type_is_blocked(tmp_path: Path) -> None:
    safe_manifest = source_manifest(tmp_path / "safe.json", ["drums"])
    ambiguous_manifest = source_manifest(tmp_path / "ambiguous.json", ["drums", "bass"])
    result = build_with_rows(
        tmp_path,
        [
            unblock_row("skip", source_manifest_path=safe_manifest, can_apply=False),
            unblock_row("ambiguous", source_manifest_path=ambiguous_manifest),
        ],
    )
    rows = rows_by_case(result)
    assert rows["skip"]["repair_status"] == "skipped_not_automatic"
    assert rows["skip"]["overlay_patch"] == {}
    assert rows["ambiguous"]["repair_status"] == "blocked_ambiguous_stem_type"
    assert rows["ambiguous"]["overlay_patch"] == {}


def test_no_audio_training_dataset_or_gt_bundle_outputs_and_summary_matches(tmp_path: Path) -> None:
    manifest = source_manifest(tmp_path / "manifest.json", ["drums"])
    result = build_with_rows(tmp_path, [unblock_row("repair", source_manifest_path=manifest)])
    assert result["summary"]["can_start_training_now_count"] == 0
    assert result["summary"]["safety"] == {
        "audio_copied": False,
        "audio_read": False,
        "audio_hashed": False,
        "dataset_built": False,
        "gt_bundle_built": False,
        "training_started": False,
        "policy_changed": False,
        "source_manifests_mutated": False,
        "models_changed": False,
        "weights_added": False,
        "cleanup_routes_added": False,
        "thresholds_changed": False,
    }

    output_dir = tmp_path / "out"
    write_outputs(result, output_dir)
    output_rows = [
        json.loads(line)
        for line in (output_dir / "manifest_stem_type_repair_report.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_json = json.loads((output_dir / "manifest_stem_type_repair_manifest.json").read_text(encoding="utf-8"))
    assert len(output_rows) == manifest_json["total_rows"]
    assert manifest_json["summary"]["overlay_repairs_created"] == 1
    assert not list(output_dir.rglob("*.flac"))
    assert not list(output_dir.rglob("*.wav"))
    assert not [path for path in output_dir.rglob("*") if "dataset" in path.name.lower()]
    assert not [path for path in output_dir.rglob("*") if "gt_bundle" in path.name.lower()]


def test_output_is_deterministic(tmp_path: Path) -> None:
    manifest = source_manifest(tmp_path / "manifest.json", ["drums"])
    rows = [unblock_row("repair", source_manifest_path=manifest)]
    first = build_with_rows(tmp_path, rows)
    second = build_with_rows(tmp_path, rows)
    assert first["rows"] == second["rows"]
    assert first["summary"] == second["summary"]
    assert first["overlays"] == second["overlays"]


def test_direct_script_invocation(tmp_path: Path) -> None:
    manifest = source_manifest(tmp_path / "manifest.json", ["drums"])
    inputs = make_inputs(tmp_path, [unblock_row("repair", source_manifest_path=manifest)])
    output_dir = tmp_path / "cli-out"
    script = Path(__file__).resolve().parents[1] / "tools" / "apply_manifest_stem_type_repair.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--unblock-plan-jsonl",
            str(inputs[0]),
            "--unblock-manifest-json",
            str(inputs[1]),
            "--evidence-gap-report-jsonl",
            str(inputs[2]),
            "--finetune-preflight-jsonl",
            str(inputs[3]),
            "--hard-case-benchmark-jsonl",
            str(inputs[4]),
            "--golden-jsonl",
            str(inputs[5]),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"overlay_repairs_created": 1' in completed.stdout
    assert (output_dir / "repaired_manifest_overlay.jsonl").exists()
