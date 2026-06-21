from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.build_short_minus_phase_repair_selection_pack import build_short_minus_phase_repair_selection_pack


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


def _audio(path: Path, value: float) -> Path:
    sample_rate = 8000
    data = np.full((sample_rate // 20, 2), value, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, sample_rate)
    return path


def _phase_repair(tmp_path: Path, *, release_ready: bool = False) -> tuple[Path, Path, Path]:
    root = tmp_path / "mpr"
    accepted_audio = _audio(root / "minus_side_attenuate_50.flac", 0.2)
    blocked_audio = _audio(root / "minus_mid_mono.flac", 0.4)
    candidate_rows = [
        {
            "candidate_id": "q002_c009__minus_mid_mono",
            "minus_path": str(blocked_audio),
            "instrumental_path": str(blocked_audio),
            "track_id": "no_escape",
            "delivery_status": "diagnostic_only",
            "delivery_allowed": False,
            "release_ready": False,
        },
        {
            "candidate_id": "q002_c009__minus_side_attenuate_50",
            "minus_path": str(accepted_audio),
            "instrumental_path": str(accepted_audio),
            "track_id": "no_escape",
            "delivery_status": "diagnostic_only",
            "delivery_allowed": False,
            "release_ready": False,
        },
    ]
    candidates = _write_jsonl(root / "candidate_pair.jsonl", candidate_rows)
    manifest = _write_json(
        root / "minus_phase_repair_manifest.json",
        {
            "schema": "no_reference_minus_phase_repair_manifest.v1",
            "mode": "no_reference_minus_phase_repair",
            "track_id": "no_escape",
            "source_candidate_id": "q002_c009",
            "source_minus_route_id": "mpr001",
            "source_mix_path": "/app/mix.flac",
            "source_vocal_path": "/app/vocal.flac",
            "source_minus_path": "/app/source_minus.flac",
            "candidate_pair_jsonl": str(candidates),
            "variant_count": len(candidate_rows),
            "variants": [
                {"variant": "minus_mid_mono", "path": str(blocked_audio), "phase_consistency_after": 1.0},
                {"variant": "minus_side_attenuate_50", "path": str(accepted_audio), "phase_consistency_after": 0.88},
            ],
            "delivery_status": "diagnostic_only",
            "delivery_allowed": False,
            "release_ready": release_ready,
            "can_publish_as_release": False,
            "can_mark_release_ready": False,
            "safety": {"release_ready": False, "training_started": False, "thresholds_relaxed": False},
        },
    )
    result_judge = _write_jsonl(
        root / "result.jsonl",
        [
            {"candidate_id": "q002_c009__minus_mid_mono", "decision": "blocked_hard_case", "minus_decision": "blocked_hard_case"},
            {"candidate_id": "q002_c009__minus_side_attenuate_50", "decision": "accept_candidate", "minus_decision": "accept_candidate"},
        ],
    )
    windowed_judge = _write_jsonl(
        root / "windowed.jsonl",
        [
            {
                "candidate_id": "q002_c009__minus_mid_mono",
                "decision": "blocked_hard_case",
                "minus_decision": "blocked_hard_case",
                "minus_score": 0.84,
                "mixture_consistency_score": 0.58,
                "vocal_leakage_in_minus_score": 0.25,
                "instrumental_phase_consistency_score": 1.0,
                "phase_consistency_score": 0.35,
                "reasons": ["minus_detail_gate_blocked"],
                "minus_path": str(blocked_audio),
            },
            {
                "candidate_id": "q002_c009__minus_side_attenuate_50",
                "decision": "accept_candidate",
                "minus_decision": "accept_candidate",
                "minus_score": 0.87,
                "mixture_consistency_score": 0.78,
                "vocal_leakage_in_minus_score": 0.24,
                "instrumental_phase_consistency_score": 0.88,
                "phase_consistency_score": 0.35,
                "reasons": ["no_reference_proxy_passed"],
                "minus_path": str(accepted_audio),
            },
        ],
    )
    return manifest, result_judge, windowed_judge


def test_build_short_minus_phase_repair_selection_pack_selects_accepted_variant(tmp_path: Path) -> None:
    manifest_json, result_judge, windowed_judge = _phase_repair(tmp_path)

    manifest = build_short_minus_phase_repair_selection_pack(
        phase_repair_manifest_json=manifest_json,
        result_judge_jsonl=result_judge,
        windowed_judge_jsonl=windowed_judge,
        output_dir=tmp_path / "out",
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["presentation_status"] == "accepted_for_listen_pack"
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert manifest["selected_count"] == 1
    assert manifest["source_phase_repair_manifest_json"] == str(manifest_json)
    rows = [json.loads(line) for line in (tmp_path / "out" / "selected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["artifact_relative_path"] for row in rows] == ["audio/min/m001.flac"]
    assert rows[0]["source_candidate_id"] == "q002_c009__minus_side_attenuate_50"
    assert rows[0]["quality_warning_reasons"] == []
    assert rows[0]["phase_repair_metrics"]["minus_score"] == 0.87
    assert "q002_c009" not in rows[0]["artifact_relative_path"]


def test_build_short_minus_phase_repair_selection_pack_blocks_release_ready_source(tmp_path: Path) -> None:
    manifest_json, result_judge, windowed_judge = _phase_repair(tmp_path, release_ready=True)

    manifest = build_short_minus_phase_repair_selection_pack(
        phase_repair_manifest_json=manifest_json,
        result_judge_jsonl=result_judge,
        windowed_judge_jsonl=windowed_judge,
        output_dir=tmp_path / "out",
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert "phase_repair_release_ready_true" in manifest["failed_checks"]
    assert manifest["selected_count"] == 0


def test_build_short_minus_phase_repair_selection_pack_blocks_pair_incompatible_warning(tmp_path: Path) -> None:
    manifest_json, result_judge, windowed_judge = _phase_repair(tmp_path)
    rows = [json.loads(line) for line in windowed_judge.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["candidate_id"] == "q002_c009__minus_side_attenuate_50":
            row["pair_decision"] = "blocked_pair_incompatible"
            row["reasons"] = ["phase_or_stereo_inconsistency"]
    _write_jsonl(windowed_judge, rows)

    manifest = build_short_minus_phase_repair_selection_pack(
        phase_repair_manifest_json=manifest_json,
        result_judge_jsonl=result_judge,
        windowed_judge_jsonl=windowed_judge,
        output_dir=tmp_path / "out",
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["selected_count"] == 0
    assert "no_accepted_phase_repair_variants" in manifest["failed_checks"]
