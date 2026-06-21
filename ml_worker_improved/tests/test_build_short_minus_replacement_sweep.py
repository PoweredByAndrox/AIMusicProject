from __future__ import annotations

import json
from pathlib import Path

from tools import build_short_minus_replacement_sweep as sweep


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


def _base_pack(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "base" / "pack.json",
        {
            "schema": "aggregate_short_listen_pack_manifest.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_ready": False,
            "can_publish_as_result": False,
            "failed_checks": [],
            "items": [],
        },
    )


def _candidate_pack(tmp_path: Path, *, release_ready: bool = False) -> Path:
    selected = _write_jsonl(
        tmp_path / "cand" / "selected.jsonl",
        [
            {
                "schema": "short_listen_repair_candidate.v1",
                "queue_item_id": "q001",
                "route_intent": "minus_repair",
                "stem_type": "minus",
                "artifact_relative_path": "audio/q001/m001.flac",
                "listen_audio_path": str(tmp_path / "cand" / "audio" / "q001" / "m001.flac"),
                "source_audio_path": str(tmp_path / "source" / "minus_a.flac"),
                "candidate_id": "q001_a",
                "track_id": "trk001",
                "delivery_status": "accepted",
                "delivery_allowed": True,
                "release_ready": False,
                "can_publish_as_result": False,
            },
            {
                "schema": "short_listen_repair_candidate.v1",
                "queue_item_id": "q001",
                "route_intent": "minus_repair",
                "stem_type": "minus",
                "artifact_relative_path": "audio/q001/m002.flac",
                "listen_audio_path": str(tmp_path / "cand" / "audio" / "q001" / "m002.flac"),
                "source_audio_path": str(tmp_path / "source" / "minus_b.flac"),
                "candidate_id": "q001_b",
                "track_id": "trk001",
                "delivery_status": "accepted",
                "delivery_allowed": True,
                "release_ready": False,
                "can_publish_as_result": False,
            },
        ],
    )
    return _write_json(
        tmp_path / "cand" / "pack.json",
        {
            "schema": "short_listen_repair_candidate_pack_manifest.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_repair_listen_pack",
            "release_ready": release_ready,
            "can_publish_as_result": False,
            "failed_checks": [],
            "selected_jsonl": str(selected),
        },
    )


def _judges(tmp_path: Path) -> tuple[Path, Path]:
    result = _write_jsonl(
        tmp_path / "rj.jsonl",
        [
            {"candidate_id": "q001_a", "decision": "accept_candidate", "minus_decision": "accept_candidate"},
            {"candidate_id": "q001_b", "decision": "blocked_hard_case", "minus_decision": "blocked_hard_case"},
        ],
    )
    windowed = _write_jsonl(
        tmp_path / "wj.jsonl",
        [
            {
                "candidate_id": "q001_a",
                "decision": "accept_candidate",
                "minus_decision": "accept_candidate",
                "minus_score": 0.81,
                "vocal_leakage_in_minus_score": 0.22,
            },
            {
                "candidate_id": "q001_b",
                "decision": "blocked_hard_case",
                "minus_decision": "blocked_hard_case",
                "minus_score": 0.91,
                "vocal_leakage_in_minus_score": 0.08,
            },
        ],
    )
    return result, windowed


def test_replacement_sweep_writes_ranked_rows_and_skips_blocked_candidates(tmp_path: Path, monkeypatch) -> None:
    calls: list[Path] = []

    def fake_replace(*, base_pack_json, replacement_pack_jsons, output_dir, pack_id, command_args=None):
        calls.append(replacement_pack_jsons[0])
        return _write_json(
            output_dir / "pack.json",
            {
                "schema": "aggregate_short_listen_pack_manifest.v1",
                "delivery_status": "accepted",
                "delivery_allowed": True,
                "presentation_status": "accepted_for_listen_pack",
                "release_ready": False,
                "can_publish_as_result": False,
                "failed_checks": [],
                "item_count": 7,
                "replacement_count": 1,
            },
        ) and {
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "failed_checks": [],
            "item_count": 7,
            "replacement_count": 1,
        }

    def fake_audit(*, pack_manifest_json, output_dir, window_seconds=12.0, max_windows_per_candidate=5, command_args=None):
        return _write_json(
            output_dir / "audit.json",
            {
                "schema": "short_minus_quality_audit.v1",
                "audit_status": "completed",
                "accepted_minus_count": 4,
                "blocked_minus_count": 0,
                "evidence_missing_count": 0,
                "warning_count": 1,
                "failed_checks": [],
                "release_ready": False,
                "can_publish_as_result": False,
            },
        ) and {
            "audit_status": "completed",
            "accepted_minus_count": 4,
            "blocked_minus_count": 0,
            "evidence_missing_count": 0,
            "warning_count": 1,
            "failed_checks": [],
            "release_ready": False,
            "can_publish_as_result": False,
        }

    monkeypatch.setattr(sweep, "build_short_listen_replacement_pack", fake_replace)
    monkeypatch.setattr(sweep, "build_short_minus_quality_audit", fake_audit)
    rj, wj = _judges(tmp_path)

    manifest = sweep.build_short_minus_replacement_sweep(
        base_pack_json=_base_pack(tmp_path),
        repair_candidate_pack_json=_candidate_pack(tmp_path),
        minus_result_judge_jsonl=rj,
        minus_windowed_judge_jsonl=wj,
        output_dir=tmp_path / "sweep",
        baseline_audit_json=_write_json(tmp_path / "baseline" / "audit.json", {"warning_count": 2, "blocked_minus_count": 0}),
    )

    assert manifest["sweep_status"] == "candidate_improved"
    assert manifest["evaluated_candidate_count"] == 1
    assert manifest["skipped_candidate_count"] == 1
    assert manifest["best_candidate"]["candidate_id"] == "q001_a"
    assert manifest["best_candidate"]["warning_delta"] == -1
    assert len(calls) == 1
    rows = [json.loads(line) for line in (tmp_path / "sweep" / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["candidate_id"] for row in rows] == ["q001_a", "q001_b"]
    assert rows[1]["evaluation_status"] == "skipped"
    assert rows[1]["skip_reason"] == "candidate_not_accepted_by_minus_judges"
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False


def test_replacement_sweep_blocks_release_ready_sources(tmp_path: Path) -> None:
    rj, wj = _judges(tmp_path)

    manifest = sweep.build_short_minus_replacement_sweep(
        base_pack_json=_base_pack(tmp_path),
        repair_candidate_pack_json=_candidate_pack(tmp_path, release_ready=True),
        minus_result_judge_jsonl=rj,
        minus_windowed_judge_jsonl=wj,
        output_dir=tmp_path / "sweep",
    )

    assert manifest["sweep_status"] == "blocked"
    assert "repair_candidate_pack_release_ready_true" in manifest["failed_checks"]
    assert manifest["evaluated_candidate_count"] == 0
    assert manifest["release_ready"] is False
