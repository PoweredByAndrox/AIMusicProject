from __future__ import annotations

import json
from pathlib import Path

from tools.filter_no_reference_replay_runs import classify_run, filter_replay_runs


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def make_contract_run(tmp_path: Path, name: str = "complete-run") -> Path:
    run_dir = tmp_path / name
    preview_path = run_dir / "audio_previews" / "minus.flac"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_bytes(b"fake-preview")
    write_json(run_dir / "run_manifest.json", {"run_id": name, "track_id": "track-a", "release_ready": False})
    write_json(
        run_dir / "preview_manifest.json",
        {
            "preview_status": "audio_preview_ready",
            "accepted_pair_ready": True,
            "manifest_preview_ready": True,
            "audio_preview_ready": True,
            "rendered_preview_audio_count": 1,
            "rendered_preview_audio": [str(preview_path)],
            "validation_status": "validated",
            "can_publish_as_result": False,
            "can_mark_release_ready": False,
            "release_ready": False,
        },
    )
    write_json(
        run_dir / "preview_render_contract.json",
        {
            "contract_version": "no_reference_audio_preview_render_v1",
            "preview_status": "audio_preview_ready",
            "rendered_preview_audio_count": 1,
            "release_ready": False,
        },
    )
    write_jsonl(
        run_dir / "rendered_preview_audio.jsonl",
        [{"render_status": "rendered", "validation_status": "validated", "rendered_path": str(preview_path), "release_ready": False}],
    )
    return run_dir


def test_complete_run_is_accepted(tmp_path: Path) -> None:
    run_dir = make_contract_run(tmp_path)

    result = classify_run(run_dir)

    assert result["preview_contract_complete"] is True
    assert result["audio_preview_ready"] is True
    assert result["incomplete_reasons"] == []
    assert result["release_ready"] is False


def test_legacy_preview_ready_without_preview_1_fields_is_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "legacy-run"
    write_json(run_dir / "run_manifest.json", {"release_ready": False})
    write_json(run_dir / "preview_manifest.json", {"runtime_status": "preview_ready", "release_ready": False, "can_publish_as_result": False})

    result = classify_run(run_dir)

    assert result["preview_contract_complete"] is False
    assert "preview_render_contract_missing" in result["incomplete_reasons"]
    assert "ambiguous_legacy_preview_ready" in result["incomplete_reasons"]


def test_explicit_skipped_or_failed_without_rendered_rows_is_contract_complete(tmp_path: Path) -> None:
    run_dir = tmp_path / "skipped-run"
    write_json(run_dir / "run_manifest.json", {"release_ready": False})
    write_json(
        run_dir / "preview_manifest.json",
        {
            "preview_status": "preview_render_skipped",
            "accepted_pair_ready": True,
            "manifest_preview_ready": True,
            "audio_preview_ready": False,
            "preview_render_skipped": True,
            "skip_reason": "audio_preview_render_disabled",
            "can_publish_as_result": False,
            "can_mark_release_ready": False,
            "release_ready": False,
        },
    )
    write_json(run_dir / "preview_render_contract.json", {"preview_status": "preview_render_skipped", "release_ready": False})

    result = classify_run(run_dir)

    assert result["preview_contract_complete"] is True
    assert result["audio_preview_ready"] is False
    assert result["rendered_preview_audio_count"] == 0
    assert result["preview_status"] == "preview_render_skipped"


def test_release_ready_true_anywhere_disqualifies_run(tmp_path: Path) -> None:
    run_dir = make_contract_run(tmp_path)
    write_jsonl(run_dir / "selected_candidates.jsonl", [{"candidate_id": "bad", "release_ready": True}])

    result = classify_run(run_dir)

    assert result["preview_contract_complete"] is False
    assert "release_ready_true_found" in result["incomplete_reasons"]


def test_filter_writes_complete_and_incomplete_outputs(tmp_path: Path) -> None:
    complete_run = make_contract_run(tmp_path, "complete-run")
    legacy_run = tmp_path / "legacy-run"
    write_json(legacy_run / "run_manifest.json", {"release_ready": False})
    write_json(legacy_run / "preview_manifest.json", {"runtime_status": "preview_ready", "release_ready": False, "can_publish_as_result": False})

    summary = filter_replay_runs(run_roots=[complete_run, legacy_run], output_dir=tmp_path / "filter")

    complete_rows = [json.loads(line) for line in (tmp_path / "filter" / "preview_contract_complete_runs.jsonl").read_text(encoding="utf-8").splitlines()]
    incomplete_rows = [json.loads(line) for line in (tmp_path / "filter" / "preview_contract_incomplete_runs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["total_runs_scanned"] == 2
    assert summary["preview_contract_complete_runs"] == 1
    assert summary["preview_contract_incomplete_runs"] == 1
    assert complete_rows[0]["run_root"] == str(complete_run)
    assert incomplete_rows[0]["run_root"] == str(legacy_run)
    assert (tmp_path / "filter" / "preview_contract_filter_summary.md").exists()
