from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.plan_reference_path_recovery import build_reference_path_recovery_plan, write_outputs


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


def source_manifest(path: Path, *, source_audio: Path, generated_vocal: Path, **extra_paths: Path) -> Path:
    payload: dict = {
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "source_audio_path": str(source_audio),
        "selected_artifacts": {"selected_vocals_flac": str(generated_vocal)},
    }
    for key, value in extra_paths.items():
        payload[key] = str(value)
    return write_json(path, payload)


def mapping_row(
    case_id: str,
    *,
    source_manifest_path: Path,
    source_audio: Path,
    generated_vocal: Path,
    mapping_status: str = "blocked_missing_reference_path",
    track_hash: str | None = None,
    split: str = "train",
) -> dict:
    return {
        "case_id": case_id,
        "track_hash": track_hash or f"track-{case_id}",
        "stem_type": "vocal",
        "split": split,
        "source_manifest": str(source_manifest_path),
        "original_paths_found": [
            {
                "type": "original_mix",
                "path": str(source_audio),
                "exists_in_docker": source_audio.exists(),
                "source_key": "source_audio_path",
            },
            {
                "type": "generated_vocal",
                "path": str(generated_vocal),
                "exists_in_docker": generated_vocal.exists(),
                "source_key": "selected_artifacts.selected_vocals_flac",
            },
        ],
        "missing_paths": [],
        "path_prefixes_detected": [],
        "docker_visible_candidate_paths": [],
        "host_path_candidates": [],
        "suggested_mapping_rule": {},
        "mapping_status": mapping_status,
        "can_repair_manifest_paths": False,
        "can_recover_source_audio": source_audio.exists(),
        "can_recover_reference_audio": False,
        "can_run_phase10e_after_mapping": False,
        "required_docker_mount": "",
        "forbidden_actions": ["no_audio_copy", "no_metrics_run"],
    }


def make_inputs(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    mapping_plan = write_jsonl(tmp_path / "source_path_mapping_plan.jsonl", rows)
    mapping_manifest = write_json(
        tmp_path / "source_path_mapping_manifest.json",
        {"schema": "source_path_mapping_manifest.v1", "total_rows": len(rows)},
    )
    docker_verification = write_json(
        tmp_path / "docker_mount_verification.json",
        {"schema": "docker_mount_verification.v1", "host_testing_stuff_music_visible": True},
    )
    reference_recovery = write_jsonl(
        tmp_path / "reference_bundle_recovery_plan.jsonl",
        [
            {
                "case_id": row["case_id"],
                "track_hash": row["track_hash"],
                "stem_type": row["stem_type"],
                "split": row["split"],
                "source_manifest": row["source_manifest"],
                "recovery_status": "blocked_missing_reference",
                "discovered_audio_inputs": [],
                "discovered_reference_inputs": [],
                "forbidden_actions": ["no_audio_copy"],
            }
            for row in rows
        ],
    )
    collected = write_jsonl(tmp_path / "collected_evidence.jsonl", [{"case_id": row["case_id"]} for row in rows])
    benchmark = write_jsonl(
        tmp_path / "hard_case_benchmark.jsonl",
        [{"benchmark_case_id": row["case_id"], "track_hash": row["track_hash"]} for row in rows],
    )
    golden = write_jsonl(
        tmp_path / "golden_corpus.jsonl",
        [{"track_hash": "golden-track", "training_allowed": False, "threshold_tuning_allowed": False}],
    )
    return mapping_plan, mapping_manifest, docker_verification, reference_recovery, collected, benchmark, golden


def build_with_rows(tmp_path: Path, rows: list[dict], *, reference_search_roots: list[Path] | None = None) -> dict:
    mapping_plan, mapping_manifest, docker_verification, reference_recovery, collected, benchmark, golden = make_inputs(
        tmp_path, rows
    )
    return build_reference_path_recovery_plan(
        source_path_mapping_jsonl=mapping_plan,
        source_path_mapping_manifest_json=mapping_manifest,
        docker_mount_verification_json=docker_verification,
        reference_recovery_plan_jsonl=reference_recovery,
        collected_evidence_jsonl=collected,
        hard_case_benchmark_jsonl=benchmark,
        golden_jsonl=golden,
        reference_search_roots=reference_search_roots,
    )


def rows_by_case(result: dict) -> dict[str, dict]:
    return {row["case_id"]: row for row in result["rows"]}


def test_only_blocked_missing_reference_path_rows_are_included(tmp_path: Path) -> None:
    source_audio = touch(tmp_path / "audio" / "mix.wav")
    generated_vocal = touch(tmp_path / "audio" / "generated_vocals.flac")
    manifest = source_manifest(tmp_path / "manifests" / "source.json", source_audio=source_audio, generated_vocal=generated_vocal)
    result = build_with_rows(
        tmp_path,
        [
            mapping_row("bench-included", source_manifest_path=manifest, source_audio=source_audio, generated_vocal=generated_vocal),
            mapping_row(
                "bench-excluded",
                source_manifest_path=manifest,
                source_audio=source_audio,
                generated_vocal=generated_vocal,
                mapping_status="mapping_found",
            ),
        ],
    )

    assert [row["case_id"] for row in result["rows"]] == ["bench-included"]
    assert result["summary"]["total_rows"] == 1


def test_exact_reference_candidate_under_allowed_root_is_candidate_found(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    source_audio = touch(allowed_root / "mix.wav")
    generated_vocal = touch(allowed_root / "generated_vocals.flac")
    reference_vocal = touch(allowed_root / "reference_vocals.flac")
    manifest = source_manifest(
        tmp_path / "manifests" / "source.json",
        source_audio=source_audio,
        generated_vocal=generated_vocal,
        reference_vocal_path=reference_vocal,
    )

    result = build_with_rows(
        tmp_path,
        [mapping_row("bench-reference", source_manifest_path=manifest, source_audio=source_audio, generated_vocal=generated_vocal)],
        reference_search_roots=[allowed_root],
    )
    row = rows_by_case(result)["bench-reference"]

    assert row["recovery_status"] == "reference_candidate_found"
    assert row["can_recover_reference_audio"] is True
    assert row["can_run_metrics_after_reference_recovery"] is True
    assert row["discovered_reference_candidates"][0]["path"] == str(reference_vocal)
    assert row["candidate_match_method"] == "manifest_reference_path"


def test_ambiguous_reference_candidates_are_blocked(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    source_audio = touch(allowed_root / "mix.wav")
    generated_vocal = touch(allowed_root / "generated_vocals.flac")
    reference_a = touch(allowed_root / "reference_vocals_a.flac")
    reference_b = touch(allowed_root / "ground_truth_vocals_b.flac")
    manifest = source_manifest(
        tmp_path / "manifests" / "source.json",
        source_audio=source_audio,
        generated_vocal=generated_vocal,
        reference_vocal_path=reference_a,
        ground_truth_vocal_path=reference_b,
    )

    result = build_with_rows(
        tmp_path,
        [mapping_row("bench-ambiguous", source_manifest_path=manifest, source_audio=source_audio, generated_vocal=generated_vocal)],
        reference_search_roots=[allowed_root],
    )
    row = rows_by_case(result)["bench-ambiguous"]

    assert row["recovery_status"] == "blocked_ambiguous_reference"
    assert row["can_recover_reference_audio"] is False
    assert len(row["discovered_reference_candidates"]) == 2


def test_missing_reference_candidates_remain_blocked(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    source_audio = touch(allowed_root / "mix.wav")
    generated_vocal = touch(allowed_root / "generated_vocals.flac")
    manifest = source_manifest(tmp_path / "manifests" / "source.json", source_audio=source_audio, generated_vocal=generated_vocal)

    result = build_with_rows(
        tmp_path,
        [mapping_row("bench-missing", source_manifest_path=manifest, source_audio=source_audio, generated_vocal=generated_vocal)],
        reference_search_roots=[allowed_root],
    )
    row = rows_by_case(result)["bench-missing"]

    assert row["recovery_status"] == "blocked_missing_reference"
    assert row["missing_reference_types"] == ["reference_vocal"]
    assert row["can_recover_reference_audio"] is False
    assert row["required_next_tool"] == "manual_source_manifest_inspection"


def test_outside_root_candidate_is_blocked(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    outside_root = tmp_path / "outside"
    source_audio = touch(allowed_root / "mix.wav")
    generated_vocal = touch(allowed_root / "generated_vocals.flac")
    outside_reference = touch(outside_root / "reference_vocals.flac")
    manifest = source_manifest(
        tmp_path / "manifests" / "source.json",
        source_audio=source_audio,
        generated_vocal=generated_vocal,
        reference_vocal_path=outside_reference,
    )

    result = build_with_rows(
        tmp_path,
        [mapping_row("bench-outside", source_manifest_path=manifest, source_audio=source_audio, generated_vocal=generated_vocal)],
        reference_search_roots=[allowed_root],
    )
    row = rows_by_case(result)["bench-outside"]

    assert row["recovery_status"] == "blocked_reference_outside_allowed_roots"
    assert row["can_recover_reference_audio"] is False
    assert row["discovered_reference_candidates"][0]["allowed_root"] is False


def test_output_is_deterministic_summary_matches_and_no_audio_or_metrics_are_created(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    source_audio = touch(allowed_root / "mix.wav")
    generated_vocal = touch(allowed_root / "generated_vocals.flac")
    manifest = source_manifest(tmp_path / "manifests" / "source.json", source_audio=source_audio, generated_vocal=generated_vocal)
    rows = [
        mapping_row("bench-missing", source_manifest_path=manifest, source_audio=source_audio, generated_vocal=generated_vocal),
        mapping_row(
            "bench-golden",
            source_manifest_path=manifest,
            source_audio=source_audio,
            generated_vocal=generated_vocal,
            track_hash="golden-track",
        ),
    ]

    first = build_with_rows(tmp_path, rows, reference_search_roots=[allowed_root])
    second = build_with_rows(tmp_path, rows, reference_search_roots=[allowed_root])
    assert first["rows"] == second["rows"]
    assert first["summary"] == second["summary"]

    output_dir = tmp_path / "out"
    write_outputs(first, output_dir)
    output_rows = [
        json.loads(line)
        for line in (output_dir / "reference_path_recovery_plan.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_payload = json.loads((output_dir / "reference_path_recovery_manifest.json").read_text(encoding="utf-8"))
    assert len(output_rows) == first["summary"]["total_rows"]
    assert manifest_payload["total_rows"] == len(output_rows)
    assert not list(output_dir.glob("*.flac"))
    assert not list(output_dir.glob("*.wav"))
    assert not list(output_dir.glob("*.npz"))
    assert not list(output_dir.glob("*ground_truth*"))
    assert not list(output_dir.glob("*objective_report*"))
    assert manifest_payload["safety"]["audio_copied"] is False
    assert manifest_payload["safety"]["audio_read"] is False
    assert manifest_payload["safety"]["audio_hashed"] is False
    assert manifest_payload["safety"]["gt_bundle_built"] is False
    assert manifest_payload["safety"]["metrics_run"] is False
    assert "golden_protected_no_training_or_tuning" in rows_by_case(first)["bench-golden"]["forbidden_actions"]


def test_direct_script_invocation(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    source_audio = touch(allowed_root / "mix.wav")
    generated_vocal = touch(allowed_root / "generated_vocals.flac")
    manifest = source_manifest(tmp_path / "manifests" / "source.json", source_audio=source_audio, generated_vocal=generated_vocal)
    mapping_plan, mapping_manifest, docker_verification, reference_recovery, collected, benchmark, golden = make_inputs(
        tmp_path,
        [mapping_row("bench-direct", source_manifest_path=manifest, source_audio=source_audio, generated_vocal=generated_vocal)],
    )
    output_dir = tmp_path / "direct"

    result = subprocess.run(
        [
            sys.executable,
            "tools/plan_reference_path_recovery.py",
            "--source-path-mapping-jsonl",
            str(mapping_plan),
            "--source-path-mapping-manifest-json",
            str(mapping_manifest),
            "--docker-mount-verification-json",
            str(docker_verification),
            "--reference-recovery-plan-jsonl",
            str(reference_recovery),
            "--collected-evidence-jsonl",
            str(collected),
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
    assert (output_dir / "reference_path_recovery_plan.jsonl").exists()
    assert (output_dir / "reference_path_recovery_manifest.json").exists()
    assert (output_dir / "reference_path_recovery_summary.md").exists()
