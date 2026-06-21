from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.build_short_minus_quality_audit import build_short_minus_quality_audit, quality_warning_reasons


def write_audio(path: Path, mono: np.ndarray, sample_rate: int = 16_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.column_stack([mono, mono]).astype(np.float32), sample_rate)
    return path


def sine(freq_hz: float, *, seconds: float = 2.0, amp: float = 0.08, sample_rate: int = 16_000) -> np.ndarray:
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / float(sample_rate)
    return np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32) * amp


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def accepted_pack(items: list[dict], path: Path) -> Path:
    return write_json(
        path,
        {
            "schema": "aggregate_short_listen_pack_manifest.v1",
            "pack_id": "lp_test",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_ready": False,
            "can_publish_as_result": False,
            "items": items,
            "failed_checks": [],
        },
    )


def test_audit_judges_short_pack_minus_with_sibling_pair(tmp_path: Path) -> None:
    vocal = sine(440.0, amp=0.04)
    minus = sine(120.0, amp=0.10)
    mix_path = write_audio(tmp_path / "src" / "mix.flac", vocal + minus)
    vocal_path = write_audio(tmp_path / "audio" / "grp" / "vocal.flac", vocal)
    minus_path = write_audio(tmp_path / "audio" / "grp" / "minus.flac", minus)
    source_pack = write_json(
        tmp_path / "src" / "pack.json",
        {
            "schema": "stem_manifest_audio_pack_manifest.v1",
            "source_audio_path": str(mix_path),
            "release_ready": False,
        },
    )
    pack = accepted_pack(
        [
            {
                "schema": "aggregate_short_listen_pack_audio.v1",
                "source_alias": "grp",
                "track_alias": "t001",
                "stem_type": "vocal",
                "audio_path": "audio/grp/vocal.flac",
                "source_manifest": str(source_pack),
                "release_ready": False,
                "can_publish_as_result": False,
            },
            {
                "schema": "aggregate_short_listen_pack_audio.v1",
                "source_alias": "grp",
                "track_alias": "t001",
                "stem_type": "minus",
                "audio_path": "audio/grp/minus.flac",
                "source_manifest": str(source_pack),
                "source_audio_path": str(minus_path),
                "source_row": {"source_stem_key": "minus", "track_id": "grp_track"},
                "release_ready": False,
                "can_publish_as_result": False,
            },
        ],
        tmp_path / "pack.json",
    )

    manifest = build_short_minus_quality_audit(
        pack_manifest_json=pack,
        output_dir=tmp_path / "audit",
        window_seconds=0.5,
        max_windows_per_candidate=2,
    )

    assert manifest["audit_status"] == "completed"
    assert manifest["minus_item_count"] == 1
    assert manifest["audited_minus_count"] == 1
    assert manifest["accepted_minus_count"] == 1
    assert manifest["blocked_minus_count"] == 0
    assert manifest["release_ready"] is False
    rows = [json.loads(line) for line in Path(manifest["rows_jsonl"]).read_text(encoding="utf-8").splitlines()]
    assert rows[0]["audit_decision"] == "passed"
    assert rows[0]["result_minus_decision"] == "accept_candidate"
    assert rows[0]["windowed_minus_decision"] == "accept_candidate"
    assert rows[0]["short_audio_path"] == "audio/grp/minus.flac"
    assert rows[0]["source_manifest"] == str(source_pack)
    assert rows[0]["source_audio_path"] == str(minus_path)
    assert rows[0]["source_row"]["source_stem_key"] == "minus"
    assert not list((tmp_path / "audit").rglob("*.wav"))
    assert not list((tmp_path / "audit").rglob("*.flac"))


def test_audit_uses_existing_source_judge_jsonl_when_pair_audio_is_not_available(tmp_path: Path) -> None:
    result_jsonl = tmp_path / "src" / "result.jsonl"
    window_jsonl = tmp_path / "src" / "window.jsonl"
    gate = {"status": "proxy_pass", "blockers": []}
    result_jsonl.parent.mkdir(parents=True, exist_ok=True)
    result_jsonl.write_text(
        json.dumps(
            {
                "candidate_id": "prior-minus",
                "minus_decision": "accept_candidate",
                "minus_blockers": [],
                "minus_score": 0.91,
                "vocal_leakage_in_minus_score": 0.04,
                "minus_musical_detail_preservation_gate": gate,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    window_jsonl.write_text(
        json.dumps(
            {
                "candidate_id": "prior-minus",
                "minus_decision": "accept_candidate",
                "minus_blockers": [],
                "minus_score": 0.90,
                "vocal_leakage_in_minus_score": 0.05,
                "minus_musical_detail_preservation_gate": gate,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    minus_path = write_audio(tmp_path / "audio" / "mfs" / "minus.flac", sine(120.0))
    pack = accepted_pack(
        [
            {
                "schema": "aggregate_short_listen_pack_audio.v1",
                "source_alias": "mfs",
                "track_alias": "t001",
                "stem_type": "minus",
                "audio_path": "audio/mfs/minus.flac",
                "source_audio_path": str(minus_path),
                "source_row": {
                    "result_judge_jsonl": str(result_jsonl),
                    "windowed_judge_jsonl": str(window_jsonl),
                },
                "release_ready": False,
                "can_publish_as_result": False,
            }
        ],
        tmp_path / "pack.json",
    )

    manifest = build_short_minus_quality_audit(pack_manifest_json=pack, output_dir=tmp_path / "audit")
    row = json.loads(Path(manifest["rows_jsonl"]).read_text(encoding="utf-8").splitlines()[0])

    assert row["audit_decision"] == "passed"
    assert row["evidence_source"] == "source_judge_jsonl"
    assert row["source_audio_path"] == str(minus_path)
    assert row["source_row"]["result_judge_jsonl"] == str(result_jsonl)
    assert row["release_ready"] is False


def test_audit_judges_selection_pack_minus_from_source_row_pair(tmp_path: Path) -> None:
    vocal = sine(440.0, amp=0.04)
    minus = sine(120.0, amp=0.10)
    mix_path = write_audio(tmp_path / "src" / "mix.flac", vocal + minus)
    vocal_path = write_audio(tmp_path / "src" / "vocal.flac", vocal)
    minus_path = write_audio(tmp_path / "audio" / "min" / "m001.flac", minus)
    pack = write_json(
        tmp_path / "pack.json",
        {
            "schema": "short_listen_repair_selection_pack_manifest.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_ready": False,
            "can_publish_as_result": False,
            "failed_checks": [],
            "listen_audio_files": [
                {
                    "schema": "short_listen_repair_selection_audio.v1",
                    "artifact_relative_path": "audio/min/m001.flac",
                    "stem_type": "minus",
                    "release_ready": False,
                    "can_publish_as_result": False,
                    "source_audio_path": str(minus_path),
                    "source_row": {
                        "candidate_id": "q001_fts",
                        "mix_path": str(mix_path),
                        "vocal_path": str(vocal_path),
                        "stem_type": "minus",
                    },
                }
            ],
        },
    )

    manifest = build_short_minus_quality_audit(
        pack_manifest_json=pack,
        output_dir=tmp_path / "audit",
        window_seconds=0.5,
        max_windows_per_candidate=2,
    )
    rows = [json.loads(line) for line in Path(manifest["rows_jsonl"]).read_text(encoding="utf-8").splitlines()]

    assert manifest["minus_item_count"] == 1
    assert manifest["audited_minus_count"] == 1
    assert manifest["accepted_minus_count"] == 1
    assert manifest["evidence_missing_count"] == 0
    assert rows[0]["short_audio_path"] == "audio/min/m001.flac"
    assert rows[0]["evidence_source"] == "fresh_pack_pair_judge"
    assert rows[0]["source_audio_path"] == str(minus_path)
    assert rows[0]["source_row"]["vocal_path"] == str(vocal_path)
    assert rows[0]["release_ready"] is False


def test_audit_judges_phase_repair_selection_lineage_from_aggregate_pack(tmp_path: Path) -> None:
    vocal = sine(440.0, amp=0.04)
    minus = sine(120.0, amp=0.10)
    mix_path = write_audio(tmp_path / "src" / "mix.flac", vocal + minus)
    vocal_path = write_audio(tmp_path / "src" / "vocal.flac", vocal)
    source_minus_path = write_audio(tmp_path / "src" / "source_minus.flac", minus)
    selected_minus_path = write_audio(tmp_path / "audio" / "m01" / "min" / "m001.flac", minus)
    pack = accepted_pack(
        [
            {
                "schema": "aggregate_short_listen_pack_audio.v1",
                "source_alias": "m01",
                "track_alias": "m01",
                "stem_type": "minus",
                "audio_path": "audio/m01/min/m001.flac",
                "source_audio_path": str(selected_minus_path),
                "source_manifest": str(tmp_path / "mps" / "pack.json"),
                "source_row": {
                    "schema": "short_minus_phase_repair_selection_audio.v1",
                    "source_audio_path": str(source_minus_path),
                    "source_candidate_id": "q001_source_minus__minus_side_attenuate_50",
                    "source_phase_repair_manifest": {
                        "track_id": "trk007",
                        "source_candidate_id": "q001_source_minus",
                        "source_mix_path": str(mix_path),
                        "source_vocal_path": str(vocal_path),
                        "source_minus_path": str(source_minus_path),
                    },
                    "source_row": {
                        "candidate_id": "q001_source_minus__minus_side_attenuate_50",
                        "mix_path": str(mix_path),
                        "vocal_path": str(vocal_path),
                        "minus_path": str(source_minus_path),
                    },
                },
                "release_ready": False,
                "can_publish_as_result": False,
            }
        ],
        tmp_path / "pack.json",
    )

    manifest = build_short_minus_quality_audit(
        pack_manifest_json=pack,
        output_dir=tmp_path / "audit",
        window_seconds=0.5,
        max_windows_per_candidate=2,
    )
    rows = [json.loads(line) for line in Path(manifest["rows_jsonl"]).read_text(encoding="utf-8").splitlines()]

    assert manifest["audit_status"] == "completed"
    assert manifest["audited_minus_count"] == 1
    assert manifest["accepted_minus_count"] == 1
    assert manifest["evidence_missing_count"] == 0
    assert rows[0]["short_audio_path"] == "audio/m01/min/m001.flac"
    assert rows[0]["evidence_source"] == "fresh_pack_pair_judge"
    assert rows[0]["source_audio_path"] == str(selected_minus_path)
    assert rows[0]["source_row"]["source_phase_repair_manifest"]["source_vocal_path"] == str(vocal_path)
    assert rows[0]["release_ready"] is False


def test_audit_marks_minus_without_pair_or_prior_judge_as_evidence_missing(tmp_path: Path) -> None:
    minus_path = write_audio(tmp_path / "audio" / "solo" / "minus.flac", sine(120.0))
    pack = accepted_pack(
        [
            {
                "schema": "aggregate_short_listen_pack_audio.v1",
                "source_alias": "solo",
                "track_alias": "t001",
                "stem_type": "minus",
                "audio_path": "audio/solo/minus.flac",
                "source_audio_path": str(minus_path),
                "release_ready": False,
                "can_publish_as_result": False,
            }
        ],
        tmp_path / "pack.json",
    )

    manifest = build_short_minus_quality_audit(pack_manifest_json=pack, output_dir=tmp_path / "audit")
    row = json.loads(Path(manifest["rows_jsonl"]).read_text(encoding="utf-8").splitlines()[0])

    assert manifest["audit_status"] == "completed_with_blockers"
    assert manifest["audited_minus_count"] == 0
    assert manifest["accepted_minus_count"] == 0
    assert "minus_audit_blocked:audio/solo/minus.flac:minus_pair_evidence_missing" in manifest["failed_checks"]
    assert row["audit_decision"] == "evidence_missing"
    assert row["audit_blockers"] == ["minus_pair_evidence_missing"]
    assert row["source_audio_path"] == str(minus_path)


def test_quality_warning_reasons_flag_residual_vocal_watch_without_blocking() -> None:
    warnings = quality_warning_reasons(
        {
            "vocal_leakage_in_minus_score": 0.18,
            "minus_musical_detail_preservation_gate": {
                "status": "proxy_pass",
                "metrics": {"instrumental_integrity": 0.72},
                "thresholds": {"instrumental_integrity": 0.70},
                "blockers": [],
            },
        }
    )

    assert "residual_vocal_watch" in warnings
    assert "instrumental_integrity_margin_watch" in warnings
