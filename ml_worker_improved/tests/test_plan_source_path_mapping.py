from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import tools.plan_source_path_mapping as source_mapping
from tools.plan_source_path_mapping import build_source_path_mapping_plan, write_outputs


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


def recovery_row(
    case_id: str,
    *,
    track_hash: str = "track-hash",
    split: str = "train",
    source_manifest: str = "/app/manifests/source.json",
    source_path: str = "/host_testing_stuff_music/song.mp3",
    source_exists: bool = False,
    generated_path: str = "/app/lab_artifacts/generated/vocals.flac",
    generated_exists: bool = True,
    reference_path: str | None = None,
    reference_exists: bool = False,
    recovery_status: str = "blocked_missing_source_audio",
) -> dict:
    reference_inputs = []
    if reference_path is not None:
        reference_inputs.append(
            {
                "type": "reference_vocal",
                "path": reference_path,
                "exists": reference_exists,
                "source_key": "reference_vocal_path",
            }
        )
    return {
        "case_id": case_id,
        "track_hash": track_hash,
        "stem_type": "vocal",
        "split": split,
        "source_manifest": source_manifest,
        "current_missing_evidence": ["missing_reference_metrics"],
        "discovered_audio_inputs": [
            {
                "type": "original_mix",
                "path": source_path,
                "exists": source_exists,
                "source_key": "source_audio_path",
            },
            {
                "type": "generated_vocal",
                "path": generated_path,
                "exists": generated_exists,
                "source_key": "selected_artifacts.selected_vocals_flac",
            },
        ],
        "discovered_reference_inputs": reference_inputs,
        "discovered_manifest_inputs": [
            {
                "type": "source_manifest",
                "path": source_manifest,
                "exists": True,
                "source_key": "source_manifest",
            }
        ],
        "required_bundle_inputs": {},
        "recovery_status": recovery_status,
        "can_build_gt_bundle": False,
        "can_run_alignment_check": False,
        "can_run_reference_metrics": False,
        "can_run_residual_metrics": False,
        "can_run_leakage_metrics": False,
        "required_next_tool": "manual_source_manifest_inspection",
        "blocked_reason": "original_mix_missing_or_unavailable",
        "forbidden_actions": ["no_audio_copy", "no_training_start"],
    }


def make_common_inputs(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path, Path, Path, Path, Path]:
    recovery_plan = write_jsonl(tmp_path / "reference_bundle_recovery_plan.jsonl", rows)
    recovery_manifest = write_json(
        tmp_path / "reference_bundle_recovery_manifest.json",
        {"schema": "reference_bundle_recovery_manifest.v1", "total_recovery_rows": len(rows)},
    )
    collected = write_jsonl(tmp_path / "collected_evidence.jsonl", [{"case_id": row["case_id"]} for row in rows])
    missing = write_jsonl(tmp_path / "missing_evidence_plan.jsonl", [{"case_id": row["case_id"]} for row in rows])
    benchmark = write_jsonl(tmp_path / "hard_case_benchmark.jsonl", [{"benchmark_case_id": row["case_id"]} for row in rows])
    golden = write_jsonl(
        tmp_path / "golden_corpus.jsonl",
        [{"track_hash": "golden-track", "training_allowed": False, "threshold_tuning_allowed": False}],
    )
    return recovery_plan, recovery_manifest, collected, missing, benchmark, golden


def build_with_rows(tmp_path: Path, rows: list[dict], **kwargs) -> dict:
    recovery_plan, recovery_manifest, collected, missing, benchmark, golden = make_common_inputs(tmp_path, rows)
    return build_source_path_mapping_plan(
        reference_recovery_plan_jsonl=recovery_plan,
        reference_recovery_manifest_json=recovery_manifest,
        collected_evidence_jsonl=collected,
        missing_evidence_plan_jsonl=missing,
        hard_case_benchmark_jsonl=benchmark,
        golden_jsonl=golden,
        **kwargs,
    )


def rows_by_case(result: dict) -> dict[str, dict]:
    return {row["case_id"]: row for row in result["rows"]}


def test_detects_missing_host_testing_stuff_music_mount(tmp_path: Path, monkeypatch) -> None:
    original_exists = source_mapping.Path.exists

    def fake_exists(path: Path) -> bool:
        if str(path) == "/host_testing_stuff_music":
            return False
        return original_exists(path)

    monkeypatch.setattr(source_mapping.Path, "exists", fake_exists)
    result = build_with_rows(tmp_path, [recovery_row("bench-missing-mount")])
    row = rows_by_case(result)["bench-missing-mount"]

    assert "/host_testing_stuff_music" in row["path_prefixes_detected"]
    assert row["mapping_status"] == "blocked_missing_docker_mount"
    assert row["required_docker_mount"] == "./testing_stuff_music:/host_testing_stuff_music:ro"
    assert row["can_repair_manifest_paths"] is False
    assert row["can_run_phase10e_after_mapping"] is False
    assert result["summary"]["rows_needing_docker_mount_count"] == 1


def test_maps_known_prefix_to_docker_candidate_when_configured(tmp_path: Path) -> None:
    docker_audio = touch(tmp_path / "docker_visible" / "song.mp3")
    result = build_with_rows(
        tmp_path,
        [recovery_row("bench-mapped")],
        configured_prefix_mappings={"/host_testing_stuff_music": str(tmp_path / "docker_visible")},
        docker_search_roots=[tmp_path / "docker_visible"],
    )
    row = rows_by_case(result)["bench-mapped"]

    assert row["mapping_status"] == "mapping_candidate_found"
    assert row["docker_visible_candidate_paths"] == [str(docker_audio)]
    assert row["can_repair_manifest_paths"] is True
    assert row["can_recover_source_audio"] is True


def test_ambiguous_mappings_are_blocked(tmp_path: Path) -> None:
    left = touch(tmp_path / "left" / "song.mp3")
    right = touch(tmp_path / "right" / "song.mp3")

    result = build_with_rows(
        tmp_path,
        [recovery_row("bench-ambiguous")],
        docker_search_roots=[left.parent, right.parent],
    )
    row = rows_by_case(result)["bench-ambiguous"]

    assert row["mapping_status"] == "blocked_ambiguous_mapping"
    assert sorted(row["docker_visible_candidate_paths"]) == sorted([str(left), str(right)])
    assert row["can_repair_manifest_paths"] is False


def test_missing_reference_path_is_blocked_even_when_source_exists(tmp_path: Path) -> None:
    source = touch(tmp_path / "source.wav")
    result = build_with_rows(
        tmp_path,
        [
            recovery_row(
                "bench-missing-reference",
                source_path=str(source),
                source_exists=True,
                reference_path=None,
                recovery_status="blocked_missing_reference",
            )
        ],
    )
    row = rows_by_case(result)["bench-missing-reference"]

    assert row["mapping_status"] == "blocked_missing_reference_path"
    assert row["can_recover_source_audio"] is True
    assert row["can_recover_reference_audio"] is False
    assert row["can_run_phase10e_after_mapping"] is False


def test_output_is_deterministic_summary_matches_and_no_audio_or_bundle_is_created(tmp_path: Path) -> None:
    rows = [
        recovery_row("bench-missing-mount"),
        recovery_row("bench-golden", track_hash="golden-track"),
    ]
    first = build_with_rows(tmp_path, rows)
    second = build_with_rows(tmp_path, rows)
    assert first["rows"] == second["rows"]
    assert first["summary"] == second["summary"]

    output_dir = tmp_path / "out"
    write_outputs(first, output_dir)
    output_rows = [
        json.loads(line)
        for line in (output_dir / "source_path_mapping_plan.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads((output_dir / "source_path_mapping_manifest.json").read_text(encoding="utf-8"))
    assert len(output_rows) == first["summary"]["total_rows"]
    assert manifest["total_rows"] == len(output_rows)
    assert not list(output_dir.glob("*.flac"))
    assert not list(output_dir.glob("*.wav"))
    assert not list(output_dir.glob("ground_truth_manifest.json"))
    plan_text = (output_dir / "source_path_mapping_plan.jsonl").read_text(encoding="utf-8")
    assert '"sha256"' not in plan_text
    assert manifest["safety"]["audio_copied"] is False
    assert manifest["safety"]["audio_read"] is False
    assert manifest["safety"]["audio_hashed"] is False
    assert manifest["safety"]["gt_bundle_built"] is False
    assert manifest["safety"]["metrics_run"] is False
    assert "golden_protected_no_training_or_tuning" in rows_by_case(first)["bench-golden"]["forbidden_actions"]


def test_direct_script_invocation(tmp_path: Path) -> None:
    recovery_plan, recovery_manifest, collected, missing, benchmark, golden = make_common_inputs(
        tmp_path,
        [recovery_row("bench-direct")],
    )
    output_dir = tmp_path / "direct"

    result = subprocess.run(
        [
            sys.executable,
            "tools/plan_source_path_mapping.py",
            "--reference-recovery-plan-jsonl",
            str(recovery_plan),
            "--reference-recovery-manifest-json",
            str(recovery_manifest),
            "--collected-evidence-jsonl",
            str(collected),
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
    assert (output_dir / "source_path_mapping_plan.jsonl").exists()
    assert (output_dir / "source_path_mapping_manifest.json").exists()
    assert (output_dir / "source_path_mapping_summary.md").exists()
    assert (output_dir / "docker_mount_suggestions.md").exists()
