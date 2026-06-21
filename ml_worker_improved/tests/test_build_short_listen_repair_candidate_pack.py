from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.build_short_listen_repair_candidate_pack import build_short_listen_repair_candidate_pack


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _audio(path: Path, value: float, *, seconds: float = 0.05) -> Path:
    sample_rate = 8000
    frames = max(1, int(sample_rate * seconds))
    data = np.full((frames, 2), value, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, sample_rate)
    return path


def _repair_queue(tmp_path: Path, minus_manifest: Path) -> Path:
    rows_path = tmp_path / "queue" / "rows.jsonl"
    rows = [
        {
            "schema": "short_listen_repair_queue_item.v1",
            "queue_item_id": "q001",
            "audio_path": "audio/voc/v002.flac",
            "route_intent": "vocal_candidate_replacement",
            "stem_type": "vocal",
            "track_alias": "voc",
            "source_row": {"track_id": "Square a Saw - Oxygen (Tevlo Vocals Resynthesized)"},
            "issue_tags": ["bad_vocal_quality", "user_reported_unacceptable"],
            "release_ready": False,
            "can_publish_as_result": False,
        },
        {
            "schema": "short_listen_repair_queue_item.v1",
            "queue_item_id": "q002",
            "audio_path": "audio/noe/minus.flac",
            "route_intent": "minus_repair",
            "stem_type": "minus_preview",
            "track_alias": "noe",
            "source_manifest": str(minus_manifest),
            "issue_tags": ["muffled_minus", "residual_vocal_middle", "user_reported_unacceptable"],
            "diagnostic_context": {
                "target_windows": [
                    {"alternate_candidate_id": "c002", "start_sec": 48.0, "duration_sec": 12.0},
                ]
            },
            "release_ready": False,
            "can_publish_as_result": False,
        },
    ]
    _write_jsonl(rows_path, rows)
    return _write_json(
        tmp_path / "queue" / "queue.json",
        {
            "schema": "short_listen_repair_queue_manifest.v1",
            "queue_status": "ready_for_repair_cycle",
            "queue_item_count": 2,
            "rows_jsonl": str(rows_path),
            "failed_checks": [],
            "release_ready": False,
            "can_publish_as_result": False,
            "safety": {"release_ready": False, "training_started": False, "thresholds_relaxed": False},
        },
    )


def _vocal_examples(tmp_path: Path) -> Path:
    target = _audio(tmp_path / "ref" / "vocals.flac", 0.2)
    good = _audio(tmp_path / "src" / "good.flac", 0.2)
    good_copy = _audio(tmp_path / "src" / "good_copy.flac", 0.2)
    poor = _audio(tmp_path / "src" / "poor.flac", 0.8)
    other = _audio(tmp_path / "src" / "other.flac", 0.2)
    return _write_jsonl(
        tmp_path / "examples.jsonl",
        [
            {
                "id": "candidate_conditioned__scnet__good",
                "track_id": "Square a Saw - Oxygen (Tevlo Vocals Resynthesized)",
                "stem_type": "vocal",
                "source_audio_path": str(good),
                "target_audio_path": str(target),
                "source_candidate_family": "scnet",
                "source_candidate_model_id": "scnet_xl_ihf_musdb18",
                "reference_verified": True,
                "candidate_source_status": "completed",
                "release_ready": False,
            },
            {
                "id": "candidate_conditioned__melband__good_copy",
                "track_id": "Square a Saw - Oxygen (Tevlo Vocals Resynthesized)",
                "stem_type": "vocal",
                "source_audio_path": str(good_copy),
                "target_audio_path": str(target),
                "source_candidate_family": "melband",
                "source_candidate_model_id": "melband_roformer_vocals",
                "reference_verified": True,
                "candidate_source_status": "completed",
                "release_ready": False,
            },
            {
                "id": "candidate_conditioned__melband__poor",
                "track_id": "Square a Saw - Oxygen (Tevlo Vocals Resynthesized)",
                "stem_type": "vocal",
                "source_audio_path": str(poor),
                "target_audio_path": str(target),
                "source_candidate_family": "melband",
                "source_candidate_model_id": "melband_roformer_vocals",
                "reference_verified": True,
                "candidate_source_status": "completed",
                "release_ready": False,
            },
            {
                "id": "candidate_conditioned__scnet__other",
                "track_id": "other-track",
                "stem_type": "vocal",
                "source_audio_path": str(other),
                "target_audio_path": str(target),
                "source_candidate_family": "scnet",
                "source_candidate_model_id": "scnet_xl_ihf_musdb18",
                "reference_verified": True,
                "candidate_source_status": "completed",
                "release_ready": False,
            },
        ],
    )


def _minus_manifest(tmp_path: Path) -> Path:
    candidates = tmp_path / "minus_candidates"
    c001 = _audio(candidates / "c001.flac", 0.1)
    c002 = _audio(candidates / "c002.flac", 0.15)
    mix = _audio(tmp_path / "minus" / "mix.flac", 0.2)
    vocal = _audio(tmp_path / "minus" / "vocal.flac", 0.05)
    result_judge = _write_jsonl(
        tmp_path / "minus" / "result_judge" / "no_reference_result_judge.jsonl",
        [{"mix_path": str(mix), "vocal_path": str(vocal)}],
    )
    render_manifest = _write_json(
        tmp_path / "minus" / "multi_window_blend_manifest.json",
        {
            "base_candidate_id": "c001",
            "base_minus_path": str(c001),
            "alternate_candidate_ids": ["c002"],
            "applied_windows": [
                {"alternate_candidate_id": "c002", "window_start_sec": 48.0, "window_duration_sec": 12.0},
            ],
            "rendered_minus_path": str(tmp_path / "minus" / "audio" / "minus.flac"),
            "release_ready": False,
            "can_publish_as_result": False,
        },
    )
    _audio(tmp_path / "minus" / "audio" / "minus.flac", 0.12)
    return _write_json(
        tmp_path / "minus" / "listen_check_manifest.json",
        {
            "schema": "listen_check_manifest.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "release_ready": False,
            "can_publish_as_result": False,
            "render_manifest": render_manifest.name,
            "result_judge": str(result_judge.relative_to(tmp_path / "minus")),
        },
    )


def _minus_only_queue(tmp_path: Path, source_manifest: Path) -> Path:
    rows_path = tmp_path / "queue" / "rows.jsonl"
    _write_jsonl(
        rows_path,
        [
            {
                "schema": "short_listen_repair_queue_item.v1",
                "queue_item_id": "q001",
                "audio_path": "audio/min/m001.flac",
                "route_intent": "minus_repair",
                "stem_type": "minus",
                "source_manifest": str(source_manifest),
                "source_row": {"track_id": "no_escape"},
                "issue_tags": ["residual_vocal"],
                "release_ready": False,
                "can_publish_as_result": False,
            }
        ],
    )
    return _write_json(
        tmp_path / "queue" / "queue.json",
        {
            "schema": "short_listen_repair_queue_manifest.v1",
            "queue_status": "ready_for_machine_warning_cycle",
            "queue_source": "minus_audit_warning",
            "queue_item_count": 1,
            "rows_jsonl": str(rows_path),
            "failed_checks": [],
            "release_ready": False,
            "can_publish_as_result": False,
            "safety": {"release_ready": False, "training_started": False, "thresholds_relaxed": False},
        },
    )


def _full_track_switch_pack(tmp_path: Path) -> Path:
    fts_dir = tmp_path / "fts"
    base_minus = _audio(fts_dir / "base_min.wav", 0.10)
    switch_minus = _audio(fts_dir / "fts_min.wav", 0.14)
    switch_vocal = _audio(fts_dir / "fts_voc.wav", 0.02)
    window_minus = _audio(tmp_path / "sel" / "audio" / "min" / "m001.flac", 0.11)
    mix = _audio(tmp_path / "src" / "mix.flac", 0.20)
    manifest = _write_json(
        fts_dir / "manifest.json",
        {
            "manifest_type": "no_reference_full_track_segment_switch_candidate",
            "base_minus_path": str(base_minus),
            "source_minus_path": str(base_minus),
            "rendered_minus_candidate_path": str(switch_minus),
            "rendered_vocal_candidate_path": str(switch_vocal),
            "mix_path": str(mix),
            "selected_windows": [
                {
                    "candidate_id": "q009_c003",
                    "selected_minus_path": str(window_minus),
                    "selected_vocal_path": str(switch_vocal),
                }
            ],
            "release_ready": False,
            "can_publish_as_result": False,
            "candidate_row": {"base_minus_path": str(base_minus), "source_minus_path": str(base_minus)},
        },
    )
    return _write_json(
        tmp_path / "flp" / "pack.json",
        {
            "schema": "full_track_switch_listen_pack.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "release_ready": False,
            "can_publish_as_result": False,
            "switch_manifest_json": str(manifest),
            "listen_audio_files": [
                {
                    "stem_type": "minus",
                    "source_audio_path": str(switch_minus),
                    "source_switch_manifest_json": str(manifest),
                    "release_ready": False,
                    "can_publish_as_result": False,
                }
            ],
        },
    )


def _stem_audio_pack(tmp_path: Path) -> Path:
    mix = _audio(tmp_path / "stem" / "mix.flac", 0.20)
    vocal = _audio(tmp_path / "stem" / "audio" / "t001" / "vocal.flac", 0.02)
    minus = _audio(tmp_path / "stem" / "audio" / "t001" / "minus.flac", 0.11)
    return _write_json(
        tmp_path / "stem" / "pack.json",
        {
            "schema": "stem_manifest_audio_pack_manifest.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "release_ready": False,
            "can_publish_as_result": False,
            "source_audio_path": str(mix),
            "stems": [
                {
                    "stem_type": "vocal",
                    "listen_audio_path": str(vocal),
                    "source_audio_path": str(vocal),
                    "release_ready": False,
                    "can_publish_as_result": False,
                },
                {
                    "stem_type": "minus",
                    "listen_audio_path": str(minus),
                    "source_audio_path": str(minus),
                    "release_ready": False,
                    "can_publish_as_result": False,
                },
            ],
        },
    )


def _no_reference_minus_preview(tmp_path: Path) -> Path:
    lineage_dir = tmp_path / "nrp"
    base_minus = _audio(lineage_dir / "base.flac", 0.11)
    repaired_minus = _audio(lineage_dir / "repair.flac", 0.13)
    mix = _audio(lineage_dir / "mix.flac", 0.20)
    vocal = _audio(lineage_dir / "vocal.flac", 0.03)
    source_candidate_manifest = _write_json(
        lineage_dir / "cand.json",
        {
            "delivery_status": "candidate_only",
            "delivery_allowed": False,
            "presentation_status": "blocked_until_judged",
            "candidate_only": True,
            "listenable": False,
            "release_ready": False,
            "can_publish_as_result": False,
            "candidate_id": "repair_candidate",
            "base_minus_path": str(base_minus),
            "candidate_row": {
                "selected_minus_path": str(repaired_minus),
                "selected_vocal_path": str(vocal),
                "mix_path": str(mix),
                "track_id": "trk_no_ref",
            },
            "repair": {
                "selected_windows": [
                    {"start_sec": 12.0, "duration_sec": 6.0},
                ]
            },
            "track_id": "trk_no_ref",
        },
    )
    return _write_json(
        lineage_dir / "preview.json",
        {
            "manifest_type": "no_reference_minus_listen_preview",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_ready": False,
            "can_publish_as_result": False,
            "source_candidate_manifest": {"path": str(source_candidate_manifest)},
            "selected_audio_path": str(repaired_minus),
            "mix_path": str(mix),
            "vocal_path": str(vocal),
            "instrumental_path": str(repaired_minus),
            "track_id": "trk_no_ref",
        },
    )


def _judge_guided_leakage_repair_manifest(tmp_path: Path) -> Path:
    lineage_dir = tmp_path / "jgl"
    raw = _audio(lineage_dir / "mix.flac", 0.20)
    vocal = _audio(lineage_dir / "vocal.flac", 0.03)
    base_minus = _audio(lineage_dir / "base.flac", 0.11)
    repaired_minus = _audio(lineage_dir / "repair.flac", 0.12)
    rows_path = lineage_dir / "rows.jsonl"
    candidate_row = {
        "track_id": "trk_jgl",
        "candidate_id": "q001_source_minus__judge_guided_leakage_repair",
        "mix_path": str(raw),
        "vocal_path": str(vocal),
        "instrumental_path": str(repaired_minus),
        "minus_path": str(repaired_minus),
        "selected_minus_path": str(repaired_minus),
        "base_minus_path": str(base_minus),
        "delivery_allowed": False,
        "delivery_status": "blocked",
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
    }
    _write_jsonl(rows_path, [candidate_row])
    return _write_json(
        lineage_dir / "manifest.json",
        {
            "schema_version": 1,
            "mode": "judge_guided_minus_vocal_leakage_repair",
            "stem_type": "minus",
            "track_id": "trk_jgl",
            "candidate_id": "q001_source_minus__judge_guided_leakage_repair",
            "candidate_only": True,
            "delivery_status": "candidate_only",
            "delivery_allowed": False,
            "presentation_status": "blocked_until_judged",
            "release_ready": False,
            "can_publish_as_result": False,
            "rendered_minus_candidate_path": str(repaired_minus),
            "raw_audio_path": str(raw),
            "base_minus_path": str(base_minus),
            "primary_vocals_path": str(vocal),
            "candidate_row_jsonl": str(rows_path),
            "candidate_row": candidate_row,
            "repair": {
                "selected_window_count": 1,
                "selected_windows": [
                    {"start_sec": 12.0, "duration_sec": 6.0, "minus_blockers": ["severe_vocal_bleed_in_minus"]},
                ],
            },
            "safety": {"release_ready": False, "training_started": False, "thresholds_relaxed": False},
        },
    )


def test_build_short_listen_repair_candidate_pack_from_queue(tmp_path: Path) -> None:
    minus_manifest = _minus_manifest(tmp_path)
    queue_manifest = _repair_queue(tmp_path, minus_manifest)
    examples = _vocal_examples(tmp_path)

    manifest = build_short_listen_repair_candidate_pack(
        repair_queue_manifest_json=queue_manifest,
        output_dir=tmp_path / "pack",
        vocal_training_examples_jsonl=examples,
        max_vocal_candidates=2,
        max_minus_candidates=2,
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["release_ready"] is False
    assert manifest["candidate_count"] == 4
    assert manifest["route_intent_counts"] == {"minus_repair": 2, "vocal_candidate_replacement": 2}
    rows = [json.loads(line) for line in (tmp_path / "pack" / "selected.jsonl").read_text(encoding="utf-8").splitlines()]
    paths = [row["artifact_relative_path"] for row in rows]
    assert paths == [
        "audio/q001/v001.flac",
        "audio/q001/v002.flac",
        "audio/q002/m001.flac",
        "audio/q002/m002.flac",
    ]
    assert "Square" not in "\n".join(paths)
    assert rows[0]["reference_metrics"]["reference_sdr_db"] > rows[1]["reference_metrics"]["reference_sdr_db"]
    assert rows[0]["listen_audio_sha256"] != rows[1]["listen_audio_sha256"]
    assert rows[2]["minus_candidate_id"] == "c001"
    assert rows[3]["minus_candidate_id"] == "c002"
    assert rows[3]["applied_window_count"] == 1
    judge_rows = [
        json.loads(line)
        for line in (tmp_path / "pack" / "minus_judge_candidates.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(judge_rows) == 2
    assert judge_rows[0]["instrumental_path"] == rows[2]["listen_audio_path"]
    assert judge_rows[0]["mix_path"].endswith("mix.flac")
    assert judge_rows[0]["vocal_path"].endswith("vocal.flac")


def test_build_short_listen_repair_candidate_pack_accepts_machine_warning_queue(tmp_path: Path) -> None:
    minus_manifest = _minus_manifest(tmp_path)
    queue_manifest = _repair_queue(tmp_path, minus_manifest)
    queue = json.loads(queue_manifest.read_text(encoding="utf-8"))
    queue["queue_status"] = "ready_for_machine_warning_cycle"
    queue["queue_source"] = "minus_audit_warning"
    queue_manifest.write_text(json.dumps(queue, sort_keys=True), encoding="utf-8")
    rows_path = Path(queue["rows_jsonl"])
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    _write_jsonl(rows_path, [row for row in rows if row.get("route_intent") == "minus_repair"])

    manifest = build_short_listen_repair_candidate_pack(
        repair_queue_manifest_json=queue_manifest,
        output_dir=tmp_path / "pack",
        vocal_training_examples_jsonl=_vocal_examples(tmp_path),
        max_vocal_candidates=0,
        max_minus_candidates=2,
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["queue_source"] == "minus_audit_warning"
    assert manifest["candidate_count"] == 2
    rows = [json.loads(line) for line in (tmp_path / "pack" / "selected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["artifact_relative_path"] for row in rows] == ["audio/q002/m001.flac", "audio/q002/m002.flac"]
    assert all(row["route_intent"] == "minus_repair" for row in rows)


def test_build_short_listen_repair_candidate_pack_accepts_full_pack_machine_gate_queue(tmp_path: Path) -> None:
    source_manifest = _stem_audio_pack(tmp_path)
    rows_path = tmp_path / "queue" / "rows.jsonl"
    _write_jsonl(
        rows_path,
        [
            {
                "schema": "short_listen_repair_queue_item.v1",
                "queue_item_id": "q001",
                "audio_path": "audio/stm/t001/minus.flac",
                "route_intent": "minus_repair",
                "stem_type": "minus",
                "source_manifest": str(source_manifest),
                "source_audio_path": str(tmp_path / "stem" / "audio" / "t001" / "minus.flac"),
                "source_row": {
                    "track_id": "trk_full",
                    "source_manifest": str(source_manifest),
                },
                "issue_tags": ["minus_gate_blocked", "residual_vocal"],
                "release_ready": False,
                "can_publish_as_result": False,
            }
        ],
    )
    queue_manifest = _write_json(
        tmp_path / "queue" / "queue.json",
        {
            "schema": "short_listen_repair_queue_manifest.v1",
            "queue_status": "ready_for_full_pack_loop",
            "queue_source": "machine_full_pack_gate",
            "queue_item_count": 1,
            "rows_jsonl": str(rows_path),
            "failed_checks": [],
            "release_ready": False,
            "can_publish_as_result": False,
            "safety": {"release_ready": False, "training_started": False, "thresholds_relaxed": False},
        },
    )

    manifest = build_short_listen_repair_candidate_pack(
        repair_queue_manifest_json=queue_manifest,
        output_dir=tmp_path / "pack",
        max_vocal_candidates=0,
        max_minus_candidates=3,
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["queue_source"] == "machine_full_pack_gate"
    assert manifest["candidate_count"] == 1
    rows = [json.loads(line) for line in (tmp_path / "pack" / "selected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["source_lineage"] == "stem_manifest_audio_pack"
    assert rows[0]["source_manifest"] == str(source_manifest)
    assert rows[0]["track_id"] == "trk_full"
    assert rows[0]["artifact_relative_path"] == "audio/q001/m001.flac"


def test_build_short_listen_repair_candidate_pack_uses_source_row_minus_preview_lineage(tmp_path: Path) -> None:
    source_manifest = _no_reference_minus_preview(tmp_path)
    master_manifest = _write_json(
        tmp_path / "master" / "manifest.json",
        {
            "schema": "aggregate_short_listen_pack_manifest.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "release_ready": False,
            "can_publish_as_result": False,
        },
    )
    rows_path = tmp_path / "queue" / "rows.jsonl"
    _write_jsonl(
        rows_path,
        [
            {
                "schema": "short_listen_repair_queue_item.v1",
                "queue_item_id": "q001",
                "audio_path": "audio/mst/minus.flac",
                "route_intent": "minus_repair",
                "stem_type": "minus",
                "source_manifest": str(master_manifest),
                "source_row": {"track_id": "trk_no_ref", "source_manifest": str(source_manifest)},
                "issue_tags": ["residual_vocal", "muffled_minus"],
                "release_ready": False,
                "can_publish_as_result": False,
            }
        ],
    )
    queue_manifest = _write_json(
        tmp_path / "queue" / "queue.json",
        {
            "schema": "short_listen_repair_queue_manifest.v1",
            "queue_status": "ready_for_machine_warning_cycle",
            "queue_source": "minus_audit_warning",
            "queue_item_count": 1,
            "rows_jsonl": str(rows_path),
            "failed_checks": [],
            "release_ready": False,
            "can_publish_as_result": False,
            "safety": {"release_ready": False, "training_started": False, "thresholds_relaxed": False},
        },
    )

    manifest = build_short_listen_repair_candidate_pack(
        repair_queue_manifest_json=queue_manifest,
        output_dir=tmp_path / "pack",
        max_vocal_candidates=0,
        max_minus_candidates=3,
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["candidate_count"] == 2
    rows = [json.loads(line) for line in (tmp_path / "pack" / "selected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["minus_candidate_id"] for row in rows] == ["base_minus", "repair_candidate"]
    assert {row["source_lineage"] for row in rows} == {"no_reference_minus_preview_source_candidate_manifest"}
    assert {row["source_manifest"] for row in rows} == {str(source_manifest)}
    assert [row["artifact_relative_path"] for row in rows] == ["audio/q001/m001.flac", "audio/q001/m002.flac"]
    assert all(row["mix_path"].endswith("mix.flac") for row in rows)
    assert all(row["vocal_path"].endswith("vocal.flac") for row in rows)


def test_build_short_listen_repair_candidate_pack_uses_nested_source_row_minus_preview_lineage(tmp_path: Path) -> None:
    source_manifest = _no_reference_minus_preview(tmp_path)
    selection_manifest = _write_json(
        tmp_path / "selection" / "pack.json",
        {
            "schema": "short_listen_repair_selection_pack_manifest.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_ready": False,
            "can_publish_as_result": False,
            "listen_audio_files": [],
        },
    )
    rows_path = tmp_path / "queue" / "rows.jsonl"
    _write_jsonl(
        rows_path,
        [
            {
                "schema": "short_listen_repair_queue_item.v1",
                "queue_item_id": "q001",
                "audio_path": "audio/mst/minus.flac",
                "route_intent": "minus_repair",
                "stem_type": "minus",
                "source_manifest": str(selection_manifest),
                "source_row": {
                    "track_id": "trk_no_ref",
                    "source_row": {"track_id": "trk_no_ref", "source_manifest": str(source_manifest)},
                },
                "issue_tags": ["residual_vocal", "muffled_minus"],
                "release_ready": False,
                "can_publish_as_result": False,
            }
        ],
    )
    queue_manifest = _write_json(
        tmp_path / "queue" / "queue.json",
        {
            "schema": "short_listen_repair_queue_manifest.v1",
            "queue_status": "ready_for_machine_warning_cycle",
            "queue_source": "minus_audit_warning",
            "queue_item_count": 1,
            "rows_jsonl": str(rows_path),
            "failed_checks": [],
            "release_ready": False,
            "can_publish_as_result": False,
            "safety": {"release_ready": False, "training_started": False, "thresholds_relaxed": False},
        },
    )

    manifest = build_short_listen_repair_candidate_pack(
        repair_queue_manifest_json=queue_manifest,
        output_dir=tmp_path / "pack",
        max_vocal_candidates=0,
        max_minus_candidates=3,
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["candidate_count"] == 2
    rows = [json.loads(line) for line in (tmp_path / "pack" / "selected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["minus_candidate_id"] for row in rows] == ["base_minus", "repair_candidate"]
    assert {row["source_manifest"] for row in rows} == {str(source_manifest)}


def test_build_short_listen_repair_candidate_pack_collects_judge_guided_leakage_repair_lineage(tmp_path: Path) -> None:
    source_manifest = _judge_guided_leakage_repair_manifest(tmp_path)
    queue_manifest = _minus_only_queue(tmp_path, source_manifest)

    manifest = build_short_listen_repair_candidate_pack(
        repair_queue_manifest_json=queue_manifest,
        output_dir=tmp_path / "pack",
        max_vocal_candidates=0,
        max_minus_candidates=3,
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["candidate_count"] == 1
    rows = [json.loads(line) for line in (tmp_path / "pack" / "selected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["source_lineage"] == "judge_guided_minus_vocal_leakage_repair"
    assert rows[0]["minus_candidate_id"] == "q001_source_minus__judge_guided_leakage_repair"
    assert rows[0]["track_id"] == "trk_jgl"
    assert rows[0]["source_minus_path"].endswith("base.flac")
    assert rows[0]["mix_path"].endswith("mix.flac")
    assert rows[0]["vocal_path"].endswith("vocal.flac")


def test_build_short_listen_repair_candidate_pack_rejects_judge_guided_leakage_repair_without_base_minus(
    tmp_path: Path,
) -> None:
    source_manifest = _judge_guided_leakage_repair_manifest(tmp_path)
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest.pop("base_minus_path", None)
    manifest["candidate_row"].pop("base_minus_path", None)
    source_manifest.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    queue_manifest = _minus_only_queue(tmp_path, source_manifest)

    result = build_short_listen_repair_candidate_pack(
        repair_queue_manifest_json=queue_manifest,
        output_dir=tmp_path / "pack",
        max_vocal_candidates=0,
        max_minus_candidates=3,
    )

    assert result["delivery_status"] == "blocked"
    assert "judge_guided_leakage_repair_base_minus_missing" in result["failed_checks"]


def test_build_short_listen_repair_candidate_pack_collects_full_track_switch_lineage(tmp_path: Path) -> None:
    queue_manifest = _minus_only_queue(tmp_path, _full_track_switch_pack(tmp_path))

    manifest = build_short_listen_repair_candidate_pack(
        repair_queue_manifest_json=queue_manifest,
        output_dir=tmp_path / "pack",
        max_vocal_candidates=0,
        max_minus_candidates=3,
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["candidate_count"] == 2
    rows = [json.loads(line) for line in (tmp_path / "pack" / "selected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["source_lineage"] for row in rows] == ["full_track_switch_manifest", "full_track_switch_manifest"]
    assert [row["minus_candidate_id"] for row in rows] == ["fts", "q009_c003"]
    assert all(row["source_minus_path"].endswith("base_min.wav") for row in rows)
    assert all(row["base_minus_path"].endswith("base_min.wav") for row in rows)
    assert all(row["mix_path"].endswith("mix.flac") for row in rows)
    assert all(row["vocal_path"].endswith("fts_voc.wav") for row in rows)


def test_build_short_listen_repair_candidate_pack_rejects_full_track_switch_without_base_minus(
    tmp_path: Path,
) -> None:
    source_manifest = _full_track_switch_pack(tmp_path)
    pack = json.loads(source_manifest.read_text(encoding="utf-8"))
    switch_manifest = Path(pack["switch_manifest_json"])
    switch = json.loads(switch_manifest.read_text(encoding="utf-8"))
    switch.pop("base_minus_path", None)
    switch.pop("source_minus_path", None)
    switch.get("candidate_row", {}).pop("base_minus_path", None)
    switch.get("candidate_row", {}).pop("source_minus_path", None)
    switch_manifest.write_text(json.dumps(switch, sort_keys=True), encoding="utf-8")
    queue_manifest = _minus_only_queue(tmp_path, source_manifest)

    manifest = build_short_listen_repair_candidate_pack(
        repair_queue_manifest_json=queue_manifest,
        output_dir=tmp_path / "pack",
        max_vocal_candidates=0,
        max_minus_candidates=3,
    )

    assert manifest["delivery_status"] == "blocked"
    assert "full_track_switch_base_minus_missing" in manifest["failed_checks"]


def test_build_short_listen_repair_candidate_pack_deduplicates_repeated_minus_sources(tmp_path: Path) -> None:
    source_manifest = _full_track_switch_pack(tmp_path)
    rows_path = tmp_path / "queue" / "rows.jsonl"
    _write_jsonl(
        rows_path,
        [
            {
                "schema": "short_listen_repair_queue_item.v1",
                "queue_item_id": "q001",
                "audio_path": "audio/min/m001.flac",
                "route_intent": "minus_repair",
                "stem_type": "minus",
                "source_manifest": str(source_manifest),
                "source_row": {"track_id": "no_escape"},
                "issue_tags": ["residual_vocal"],
                "release_ready": False,
                "can_publish_as_result": False,
            },
            {
                "schema": "short_listen_repair_queue_item.v1",
                "queue_item_id": "q002",
                "audio_path": "audio/min/m002.flac",
                "route_intent": "minus_repair",
                "stem_type": "minus",
                "source_manifest": str(source_manifest),
                "source_row": {"track_id": "no_escape"},
                "issue_tags": ["residual_vocal"],
                "release_ready": False,
                "can_publish_as_result": False,
            },
        ],
    )
    queue_manifest = _write_json(
        tmp_path / "queue" / "queue.json",
        {
            "schema": "short_listen_repair_queue_manifest.v1",
            "queue_status": "ready_for_machine_warning_cycle",
            "queue_source": "minus_audit_warning",
            "queue_item_count": 2,
            "rows_jsonl": str(rows_path),
            "failed_checks": [],
            "release_ready": False,
            "can_publish_as_result": False,
            "safety": {"release_ready": False, "training_started": False, "thresholds_relaxed": False},
        },
    )

    manifest = build_short_listen_repair_candidate_pack(
        repair_queue_manifest_json=queue_manifest,
        output_dir=tmp_path / "pack",
        max_vocal_candidates=0,
        max_minus_candidates=3,
    )

    rows = [json.loads(line) for line in (tmp_path / "pack" / "selected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert manifest["delivery_status"] == "accepted"
    assert manifest["candidate_count"] == 2
    assert manifest["deduplicated_candidate_count"] == 2
    assert [row["queue_item_id"] for row in rows] == ["q001", "q001"]
    assert all(row["track_id"] == "no_escape" for row in rows)
    assert len({row["listen_audio_sha256"] for row in rows}) == 2
    assert [row["artifact_relative_path"] for row in rows] == ["audio/q001/m001.flac", "audio/q001/m002.flac"]


def test_build_short_listen_repair_candidate_pack_collects_stem_audio_pack_lineage(tmp_path: Path) -> None:
    queue_manifest = _minus_only_queue(tmp_path, _stem_audio_pack(tmp_path))

    manifest = build_short_listen_repair_candidate_pack(
        repair_queue_manifest_json=queue_manifest,
        output_dir=tmp_path / "pack",
        max_vocal_candidates=0,
        max_minus_candidates=3,
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["candidate_count"] == 1
    rows = [json.loads(line) for line in (tmp_path / "pack" / "selected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["source_lineage"] == "stem_manifest_audio_pack"
    assert rows[0]["minus_candidate_id"] == "source_minus"
    assert rows[0]["mix_path"].endswith("mix.flac")
    assert rows[0]["vocal_path"].endswith("vocal.flac")


def test_build_short_listen_repair_candidate_pack_blocks_release_ready_queue(tmp_path: Path) -> None:
    minus_manifest = _minus_manifest(tmp_path)
    queue_manifest = _repair_queue(tmp_path, minus_manifest)
    queue = json.loads(queue_manifest.read_text(encoding="utf-8"))
    queue["release_ready"] = True
    queue_manifest.write_text(json.dumps(queue, sort_keys=True), encoding="utf-8")

    manifest = build_short_listen_repair_candidate_pack(
        repair_queue_manifest_json=queue_manifest,
        output_dir=tmp_path / "pack",
        vocal_training_examples_jsonl=_vocal_examples(tmp_path),
    )

    assert manifest["delivery_status"] == "blocked"
    assert "repair_queue_release_ready_true" in manifest["failed_checks"]
    assert manifest["delivery_allowed"] is False
    assert manifest["candidate_count"] == 0


def test_build_short_listen_repair_candidate_pack_blocks_missing_minus_render_manifest(tmp_path: Path) -> None:
    minus_manifest = _write_json(
        tmp_path / "minus" / "listen_check_manifest.json",
        {
            "schema": "listen_check_manifest.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "release_ready": False,
            "can_publish_as_result": False,
        },
    )
    rows_path = tmp_path / "queue" / "rows.jsonl"
    _write_jsonl(
        rows_path,
        [
            {
                "schema": "short_listen_repair_queue_item.v1",
                "queue_item_id": "q001",
                "audio_path": "audio/mpr/min/m001.flac",
                "route_intent": "minus_repair",
                "stem_type": "minus",
                "source_manifest": str(minus_manifest),
                "issue_tags": ["muffled_minus", "residual_vocal_middle"],
                "release_ready": False,
                "can_publish_as_result": False,
            }
        ],
    )
    queue_manifest = _write_json(
        tmp_path / "queue" / "queue.json",
        {
            "schema": "short_listen_repair_queue_manifest.v1",
            "queue_status": "ready_for_repair_cycle",
            "queue_item_count": 1,
            "rows_jsonl": str(rows_path),
            "failed_checks": [],
            "release_ready": False,
            "can_publish_as_result": False,
            "safety": {"release_ready": False, "training_started": False, "thresholds_relaxed": False},
        },
    )

    manifest = build_short_listen_repair_candidate_pack(
        repair_queue_manifest_json=queue_manifest,
        output_dir=tmp_path / "pack",
        max_vocal_candidates=0,
        max_minus_candidates=2,
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["candidate_count"] == 0
    assert "minus_render_manifest_missing:q001" in manifest["failed_checks"]


def test_build_short_listen_repair_candidate_pack_uses_phase_selection_lineage(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "source" / "audio"
    c001 = _audio(candidate_dir / "c001.flac", 0.1)
    c002 = _audio(candidate_dir / "c002.flac", 0.2)
    phase_manifest = _write_json(
        tmp_path / "phase" / "phase.json",
        {
            "schema": "no_reference_minus_phase_repair_manifest.v1",
            "delivery_status": "diagnostic_only",
            "delivery_allowed": False,
            "release_ready": False,
            "can_publish_as_result": False,
            "source_candidate_id": "q001_c001",
            "source_minus_path": str(c001),
            "source_mix_path": str(_audio(tmp_path / "phase" / "mix.flac", 0.4)),
            "source_vocal_path": str(_audio(tmp_path / "phase" / "vocal.flac", 0.05)),
            "training_started": False,
            "thresholds_relaxed": False,
        },
    )
    selection_manifest = _write_json(
        tmp_path / "selection" / "pack.json",
        {
            "schema": "short_minus_phase_repair_selection_pack_manifest.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_ready": False,
            "can_publish_as_result": False,
            "failed_checks": [],
            "source_phase_repair_manifest_json": str(phase_manifest),
            "listen_audio_files": [
                {
                    "source_audio_path": str(c002),
                    "source_phase_repair_manifest_json": str(phase_manifest),
                    "release_ready": False,
                    "can_publish_as_result": False,
                }
            ],
        },
    )
    rows_path = tmp_path / "queue" / "rows.jsonl"
    _write_jsonl(
        rows_path,
        [
            {
                "schema": "short_listen_repair_queue_item.v1",
                "queue_item_id": "q001",
                "audio_path": "audio/mpr/min/m001.flac",
                "route_intent": "minus_repair",
                "stem_type": "minus",
                "source_manifest": str(selection_manifest),
                "issue_tags": ["muffled_minus"],
                "release_ready": False,
                "can_publish_as_result": False,
            }
        ],
    )
    queue_manifest = _write_json(
        tmp_path / "queue" / "queue.json",
        {
            "schema": "short_listen_repair_queue_manifest.v1",
            "queue_status": "ready_for_repair_cycle",
            "queue_item_count": 1,
            "rows_jsonl": str(rows_path),
            "failed_checks": [],
            "release_ready": False,
            "can_publish_as_result": False,
            "safety": {"release_ready": False, "training_started": False, "thresholds_relaxed": False},
        },
    )

    manifest = build_short_listen_repair_candidate_pack(
        repair_queue_manifest_json=queue_manifest,
        output_dir=tmp_path / "pack",
        max_vocal_candidates=0,
        max_minus_candidates=2,
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["candidate_count"] == 2
    rows = [json.loads(line) for line in (tmp_path / "pack" / "selected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["minus_candidate_id"] for row in rows] == ["c001", "c002"]
    assert {row["source_lineage"] for row in rows} == {"phase_repair_selection_manifest"}
    assert all(Path(row["listen_audio_path"]).exists() for row in rows)
    judge_rows = [
        json.loads(line)
        for line in (tmp_path / "pack" / "minus_judge_candidates.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(judge_rows) == 2
    assert all(row["mix_path"].endswith("mix.flac") for row in judge_rows)
    assert all(row["vocal_path"].endswith("vocal.flac") for row in judge_rows)


def test_build_short_listen_repair_candidate_pack_uses_phase_variant_paths(tmp_path: Path) -> None:
    source_minus = _audio(tmp_path / "phase" / "source_minus.flac", 0.1)
    mid_mono = _audio(tmp_path / "phase" / "minus_mid_mono.flac", 0.12)
    side_25 = _audio(tmp_path / "phase" / "minus_side_attenuate_25.flac", 0.14)
    phase_manifest = _write_json(
        tmp_path / "phase" / "phase.json",
        {
            "schema": "no_reference_minus_phase_repair_manifest.v1",
            "delivery_status": "diagnostic_only",
            "delivery_allowed": False,
            "release_ready": False,
            "can_publish_as_result": False,
            "source_candidate_id": "q001_source_minus",
            "source_minus_path": str(source_minus),
            "source_mix_path": str(_audio(tmp_path / "phase" / "mix.flac", 0.4)),
            "source_vocal_path": str(_audio(tmp_path / "phase" / "vocal.flac", 0.05)),
            "track_id": "trk_phase",
            "variants": [
                {"variant": "minus_mid_mono", "path": str(mid_mono)},
                {"variant": "minus_side_attenuate_25", "path": str(side_25)},
            ],
            "training_started": False,
            "thresholds_relaxed": False,
        },
    )
    selection_manifest = _write_json(
        tmp_path / "selection" / "pack.json",
        {
            "schema": "short_minus_phase_repair_selection_pack_manifest.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_ready": False,
            "can_publish_as_result": False,
            "failed_checks": [],
            "source_phase_repair_manifest_json": str(phase_manifest),
            "listen_audio_files": [
                {
                    "source_audio_path": str(side_25),
                    "source_phase_repair_manifest_json": str(phase_manifest),
                    "release_ready": False,
                    "can_publish_as_result": False,
                }
            ],
        },
    )
    rows_path = tmp_path / "queue" / "rows.jsonl"
    _write_jsonl(
        rows_path,
        [
            {
                "schema": "short_listen_repair_queue_item.v1",
                "queue_item_id": "q001",
                "audio_path": "audio/mpr/min/m001.flac",
                "route_intent": "minus_repair",
                "stem_type": "minus",
                "source_manifest": str(selection_manifest),
                "issue_tags": ["residual_vocal"],
                "release_ready": False,
                "can_publish_as_result": False,
            }
        ],
    )
    queue_manifest = _write_json(
        tmp_path / "queue" / "queue.json",
        {
            "schema": "short_listen_repair_queue_manifest.v1",
            "queue_status": "ready_for_machine_warning_cycle",
            "queue_source": "minus_audit_warning",
            "queue_item_count": 1,
            "rows_jsonl": str(rows_path),
            "failed_checks": [],
            "release_ready": False,
            "can_publish_as_result": False,
            "safety": {"release_ready": False, "training_started": False, "thresholds_relaxed": False},
        },
    )

    manifest = build_short_listen_repair_candidate_pack(
        repair_queue_manifest_json=queue_manifest,
        output_dir=tmp_path / "pack",
        max_vocal_candidates=0,
        max_minus_candidates=3,
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["candidate_count"] == 3
    rows = [json.loads(line) for line in (tmp_path / "pack" / "selected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["minus_candidate_id"] for row in rows] == [
        "source_minus",
        "minus_mid_mono",
        "minus_side_attenuate_25",
    ]
    assert {row["source_lineage"] for row in rows} == {"phase_repair_selection_manifest"}
    assert [row["artifact_relative_path"] for row in rows] == [
        "audio/q001/m001.flac",
        "audio/q001/m002.flac",
        "audio/q001/m003.flac",
    ]
    assert {row["track_id"] for row in rows} == {"trk_phase"}
    judge_rows = [
        json.loads(line)
        for line in (tmp_path / "pack" / "minus_judge_candidates.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["track_id"] for row in judge_rows} == {"trk_phase"}
