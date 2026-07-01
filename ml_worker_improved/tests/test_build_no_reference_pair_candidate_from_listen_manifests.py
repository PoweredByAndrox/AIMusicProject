from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.build_no_reference_pair_candidate_from_listen_manifests import (
    build_pair_candidate_manifest,
    main,
)


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def accepted_vocal_manifest(track_id: str = "track-a", mix_path: str = "/tmp/mix.wav") -> dict:
    return {
        "manifest_type": "no_reference_vocal_listen_preview",
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": "",
        "stem_type": "vocal",
        "track_id": track_id,
        "candidate_id": "vocal-a",
        "mix_path": mix_path,
        "selected_vocal_path": "/tmp/vocal.wav",
        "selected_audio_path": "/tmp/vocal.wav",
        "accepted_count": 1,
        "can_mark_release_ready": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": {
            "training_started": False,
            "fine_tune_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "cleanup_routes_added": False,
            "thresholds_relaxed": False,
        },
    }


def accepted_minus_manifest(track_id: str = "track-a", mix_path: str = "/tmp/mix.wav") -> dict:
    return {
        "manifest_type": "no_reference_minus_listen_preview",
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": "",
        "stem_type": "minus",
        "track_id": track_id,
        "candidate_id": "minus-a",
        "mix_path": mix_path,
        "selected_minus_path": "/tmp/minus.wav",
        "selected_audio_path": "/tmp/minus.wav",
        "accepted_count": 1,
        "can_mark_release_ready": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": {
            "training_started": False,
            "fine_tune_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "cleanup_routes_added": False,
            "thresholds_relaxed": False,
        },
    }


def test_pair_candidate_accepts_shared_mix_legacy_track_ids(tmp_path: Path) -> None:
    vocal_json = write_json(tmp_path / "vocal_manifest.json", accepted_vocal_manifest("legacy-vocal"))
    minus_json = write_json(tmp_path / "minus_manifest.json", accepted_minus_manifest("legacy-minus"))

    manifest = build_pair_candidate_manifest(
        vocal_manifest_json=vocal_json,
        minus_manifest_json=minus_json,
    )

    assert manifest["delivery_status"] == "ready_for_pair_judge"
    assert manifest["dominant_failure_reason"] == ""
    assert manifest["track_identity_match_method"] == "shared_mix_path"
    assert manifest["candidate_count"] == 1
    row = manifest["candidate_rows"][0]
    assert row["track_id"] == "legacy-vocal"
    assert row["mix_path"] == "/tmp/mix.wav"
    assert row["vocal_path"] == "/tmp/vocal.wav"
    assert row["instrumental_path"] == "/tmp/minus.wav"
    assert row["selected_vocal_path"] == "/tmp/vocal.wav"
    assert row["selected_minus_path"] == "/tmp/minus.wav"
    assert row["requires_pair_compatibility_judge"] is True
    assert row["can_mark_release_ready"] is False
    assert row["audio_exported"] is False


def test_pair_candidate_blocks_mismatched_track_and_mix(tmp_path: Path) -> None:
    vocal_json = write_json(tmp_path / "vocal_manifest.json", accepted_vocal_manifest("track-a", "/tmp/a.wav"))
    minus_json = write_json(tmp_path / "minus_manifest.json", accepted_minus_manifest("track-b", "/tmp/b.wav"))

    manifest = build_pair_candidate_manifest(
        vocal_manifest_json=vocal_json,
        minus_manifest_json=minus_json,
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["dominant_failure_reason"] == "track_id_mismatch"
    assert manifest["candidate_count"] == 0
    assert manifest["candidate_rows"] == []
    assert manifest["can_mark_release_ready"] is False


def test_pair_candidate_blocks_non_accepted_source(tmp_path: Path) -> None:
    vocal = accepted_vocal_manifest()
    vocal["delivery_status"] = "blocked"
    vocal["delivery_allowed"] = False
    vocal["presentation_status"] = "blocked_from_listen_pack"
    vocal_json = write_json(tmp_path / "vocal_manifest.json", vocal)
    minus_json = write_json(tmp_path / "minus_manifest.json", accepted_minus_manifest())

    manifest = build_pair_candidate_manifest(
        vocal_manifest_json=vocal_json,
        minus_manifest_json=minus_json,
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["dominant_failure_reason"] == "source_stem_manifest_not_accepted"
    assert manifest["candidate_rows"] == []


def test_pair_candidate_blocks_accepted_source_with_semantic_failure_labels(tmp_path: Path) -> None:
    vocal = accepted_vocal_manifest()
    vocal["quality_audit"] = {
        "issue_labels": [
            "foreign_source_recovered:vocal:minus,mix_conservation_failed",
            "candidate_manifest_created",
        ],
    }
    vocal_json = write_json(tmp_path / "vocal_manifest.json", vocal)
    minus_json = write_json(tmp_path / "minus_manifest.json", accepted_minus_manifest())

    manifest = build_pair_candidate_manifest(
        vocal_manifest_json=vocal_json,
        minus_manifest_json=minus_json,
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["dominant_failure_reason"] == "source_stem_manifest_not_accepted"
    assert manifest["candidate_rows"] == []


def test_cli_writes_candidate_jsonl_and_summary(tmp_path: Path) -> None:
    output_jsonl = tmp_path / "pair_candidate_rows.jsonl"
    summary_json = tmp_path / "pair_candidate_manifest.json"
    summary_md = tmp_path / "pair_candidate_summary.md"

    exit_code = main(
        [
            "--vocal-manifest-json",
            str(write_json(tmp_path / "vocal_manifest.json", accepted_vocal_manifest())),
            "--minus-manifest-json",
            str(write_json(tmp_path / "minus_manifest.json", accepted_minus_manifest())),
            "--output-jsonl",
            str(output_jsonl),
            "--summary-json",
            str(summary_json),
            "--summary-md",
            str(summary_md),
        ]
    )

    assert exit_code == 0
    rows = read_jsonl(output_jsonl)
    assert len(rows) == 1
    assert rows[0]["candidate_family"] == "listen_manifest_vocal_minus_pair"
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["delivery_status"] == "ready_for_pair_judge"
    assert "delivery_status: ready_for_pair_judge" in summary_md.read_text(encoding="utf-8")


def test_cli_writes_empty_jsonl_when_blocked(tmp_path: Path) -> None:
    output_jsonl = tmp_path / "pair_candidate_rows.jsonl"
    summary_json = tmp_path / "pair_candidate_manifest.json"
    summary_md = tmp_path / "pair_candidate_summary.md"

    exit_code = main(
        [
            "--vocal-manifest-json",
            str(write_json(tmp_path / "vocal_manifest.json", accepted_vocal_manifest("track-a", "/tmp/a.wav"))),
            "--minus-manifest-json",
            str(write_json(tmp_path / "minus_manifest.json", accepted_minus_manifest("track-b", "/tmp/b.wav"))),
            "--output-jsonl",
            str(output_jsonl),
            "--summary-json",
            str(summary_json),
            "--summary-md",
            str(summary_md),
        ]
    )

    assert exit_code == 1
    assert read_jsonl(output_jsonl) == []
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["delivery_status"] == "blocked"
    assert summary["dominant_failure_reason"] == "track_id_mismatch"


def test_file_script_entrypoint_runs_with_tools_imports(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "build_no_reference_pair_candidate_from_listen_manifests.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--vocal-manifest-json",
            str(write_json(tmp_path / "vocal_manifest.json", accepted_vocal_manifest())),
            "--minus-manifest-json",
            str(write_json(tmp_path / "minus_manifest.json", accepted_minus_manifest())),
            "--output-jsonl",
            str(tmp_path / "pair_candidate_rows.jsonl"),
            "--summary-json",
            str(tmp_path / "pair_candidate_manifest.json"),
            "--summary-md",
            str(tmp_path / "pair_candidate_summary.md"),
        ],
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
