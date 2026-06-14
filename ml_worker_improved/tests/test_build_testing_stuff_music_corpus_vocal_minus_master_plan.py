from __future__ import annotations

import json
from pathlib import Path

from tools.build_testing_stuff_music_corpus_vocal_minus_master_plan import build_plan, main


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def accepted_manifest(track_id: str, stem_type: str, *, path_key: str, path: str) -> dict:
    return {
        "manifest_type": f"no_reference_{stem_type}_listen_preview",
        "track_id": track_id,
        "stem_type": stem_type,
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "release_ready": False,
        path_key: path,
    }


def corpus_gate_report(*rows: dict, planned_tracks: list[str] | None = None) -> dict:
    planned = planned_tracks or sorted({str(row["track_id"]) for row in rows})
    return {
        "manifest_type": "testing_stuff_music_corpus_gate_report",
        "release_ready": False,
        "delivery_allowed": False,
        "planned_target_stems_by_track": {track_id: ["vocal", "minus", "melody", "drums", "bass", "other"] for track_id in planned},
        "summary": {"candidate_count": len(rows)},
        "rows": list(rows),
    }


def accepted_stem_row(track_id: str, stem_type: str) -> dict:
    return {
        "track_id": track_id,
        "stem_type": stem_type,
        "model_id": "model",
        "gate_status": "accepted_for_listen_pack_not_release",
        "delivery_allowed": True,
        "release_ready": False,
        "selected_audio_path": f"/app/{track_id}/{stem_type}.flac",
    }


def test_master_plan_requires_vocal_gate_before_minus_and_master(tmp_path: Path) -> None:
    gate_report = write_json(
        tmp_path / "gate_report.json",
        corpus_gate_report(accepted_stem_row("track_a", "melody"), planned_tracks=["track_a"]),
    )

    plan = build_plan(corpus_gate_report_json=gate_report)

    assert plan["delivery_allowed"] is False
    assert plan["release_ready"] is False
    assert plan["summary"]["run_vocal_delivery_pipeline"] == 1
    assert plan["tracks"][0]["next_gate"] == "run_vocal_delivery_pipeline"
    assert plan["tracks"][0]["missing_gates"] == ["vocal_delivery", "minus_delivery", "master_manifest", "nonvocal_stems"]


def test_master_plan_advances_to_master_when_vocal_and_minus_are_accepted(tmp_path: Path) -> None:
    gate_report = write_json(
        tmp_path / "gate_report.json",
        corpus_gate_report(accepted_stem_row("track_a", "melody"), planned_tracks=["track_a"]),
    )
    vocal_root = tmp_path / "vocal"
    minus_root = tmp_path / "minus"
    write_json(vocal_root / "no_reference_vocal_listen_preview_manifest.json", accepted_manifest("track_a", "vocal", path_key="selected_vocal_path", path="/app/vocal.flac"))
    write_json(minus_root / "no_reference_minus_listen_preview_manifest.json", accepted_manifest("track_a", "minus", path_key="selected_minus_path", path="/app/minus.flac"))

    plan = build_plan(
        corpus_gate_report_json=gate_report,
        vocal_manifest_roots=[vocal_root],
        minus_manifest_roots=[minus_root],
    )

    assert plan["summary"]["run_pair_compatibility_and_master_manifest"] == 1
    assert plan["tracks"][0]["next_gate"] == "run_pair_compatibility_and_master_manifest"
    assert plan["tracks"][0]["ready_for_master"] is True


def test_master_plan_marks_existing_full_pack_listen_preview_without_release(tmp_path: Path) -> None:
    gate_report = write_json(
        tmp_path / "gate_report.json",
        corpus_gate_report(planned_tracks=["track_a"]),
    )
    full_root = tmp_path / "full"
    write_json(
        full_root / "no_reference_full_stem_pack_manifest.json",
        {
            "manifest_type": "no_reference_full_stem_pack",
            "track_id": "track_a",
            "stem_type": "full_stem_pack",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_status": "listen_preview_only",
            "release_ready": False,
            "accepted_stem_count": 6,
        },
    )

    plan = build_plan(corpus_gate_report_json=gate_report, full_stem_pack_roots=[full_root])

    assert plan["delivery_allowed"] is False
    assert plan["release_ready"] is False
    assert plan["summary"]["full_stem_pack_listen_preview_ready_not_release"] == 1
    assert plan["tracks"][0]["next_gate"] == "full_stem_pack_listen_preview_ready_not_release"
    assert plan["tracks"][0]["missing_gates"] == []


def test_master_plan_matches_existing_full_pack_by_source_audio_when_track_ids_drift(tmp_path: Path) -> None:
    corpus_plan_jsonl = tmp_path / "corpus_plan.jsonl"
    corpus_plan_jsonl.write_text(
        json.dumps({"track_id": "sanitized-track", "audio_path": "/host/same-track.mp3"}) + "\n",
        encoding="utf-8",
    )
    gate_report = write_json(
        tmp_path / "gate_report.json",
        corpus_gate_report(planned_tracks=["sanitized-track"])
        | {"source_corpus_plan_jsonl": {"path": str(corpus_plan_jsonl), "sha256": "test"}},
    )
    full_root = tmp_path / "full"
    write_json(
        full_root / "no_reference_full_stem_pack_manifest.json",
        {
            "manifest_type": "no_reference_full_stem_pack",
            "track_id": "filename-derived-track",
            "source_mix_path": "/host/same-track.mp3",
            "stem_type": "full_stem_pack",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_status": "listen_preview_only",
            "release_ready": False,
            "accepted_stem_count": 6,
        },
    )

    plan = build_plan(corpus_gate_report_json=gate_report, full_stem_pack_roots=[full_root])

    assert plan["tracks"][0]["track_identity_match_method"] == "source_audio_path"
    assert plan["tracks"][0]["next_gate"] == "full_stem_pack_listen_preview_ready_not_release"


def test_master_plan_cli_writes_manifest_and_summary(tmp_path: Path) -> None:
    gate_report = write_json(
        tmp_path / "gate_report.json",
        corpus_gate_report(planned_tracks=["track_a"]),
    )
    output_manifest = tmp_path / "plan.json"
    summary_md = tmp_path / "summary.md"

    assert (
        main(
            [
                "--corpus-gate-report-json",
                str(gate_report),
                "--output-manifest-json",
                str(output_manifest),
                "--summary-md",
                str(summary_md),
            ]
        )
        == 0
    )

    manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert manifest["tracks"][0]["next_gate"] == "run_corpus_job_matrix_for_track"
    assert "run_corpus_job_matrix_for_track: 1" in summary_md.read_text(encoding="utf-8")
