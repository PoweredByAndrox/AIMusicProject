from __future__ import annotations

import json
from pathlib import Path

from tools import apply_mac_residual_damage_limited_repair as route


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_existing_residual_damage_limited_route_preserves_contract_and_provenance(tmp_path: Path) -> None:
    guard_dir = tmp_path / "guard"
    guard_audio = guard_dir / "audio"
    guard_audio.mkdir(parents=True)
    for name in ("v.flac", "i.flac", "raw.flac"):
        (guard_audio / name).write_bytes(b"audio")
    guard_manifest = guard_dir / "manifest.json"
    _write_json(
        guard_manifest,
        {
            "schema": "mac_mel_residual_guard.v1",
            "audio": {
                "vocals": str(guard_audio / "v.flac"),
                "instrumental": str(guard_audio / "i.flac"),
                "raw_ref": str(guard_audio / "raw.flac"),
            },
            "score": {"regions": [{"start": 1.0, "end": 1.5}], "config": {"window_sec": 0.75}},
            "delivery_allowed": False,
            "listen_pack_created": False,
        },
    )
    candidate = tmp_path / "train" / "audio" / "i.flac"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"audio")
    residual_manifest = tmp_path / "residual" / "manifest.json"
    _write_json(
        residual_manifest,
        {
            "schema": "mac_mel_residual_refiner_training.v1",
            "status": "residual_batch_trained_damage_failed",
            "training_status": "trained_repair_passed_pitch_guard",
            "guard_manifest": str(guard_manifest),
            "training_manifest": str(tmp_path / "train" / "manifest.json"),
            "audio": {"instrumental": str(candidate)},
            "damage_audit": {
                "schema": "mac_instrumental_damage_audit.v1",
                "verdict": "fail",
                "failed_region_count": 1,
                "regions": [
                    {
                        "start": 1.0,
                        "end": 1.5,
                        "delta_ratio": 0.445517,
                        "blocker_reasons": ["instrumental_region_overprocessed"],
                    }
                ],
                "blocker_reasons": ["instrumental_region_overprocessed"],
                "delivery_allowed": False,
            },
            "delivery_allowed": False,
            "listen_pack_created": False,
        },
    )
    calls: list[dict[str, object]] = []

    def fake_run_damage_limited_repair(**kwargs):
        calls.append(kwargs)
        return {
            "schema": "mac_residual_damage_limited_repair.v1",
            "status": "damage_limited_repair_passed",
            "audio": {"vocals": "v.flac", "instrumental": "i.flac", "raw_ref": "raw.flac"},
            "after_score": {"verdict": "pass", "regions": []},
            "damage_audit": {"verdict": "pass", "failed_region_count": 0, "blocker_reasons": []},
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "damage_limited_repair_diagnostic_only",
        }

    original = route.run_damage_limited_repair
    try:
        route.run_damage_limited_repair = fake_run_damage_limited_repair
        manifest = route.run_existing_residual_damage_limited_repair(
            residual_manifest_path=residual_manifest,
            output_dir=tmp_path / "out",
        )
    finally:
        route.run_damage_limited_repair = original

    assert len(calls) == 1
    assert calls[0]["reference_vocals_path"] == guard_audio / "v.flac"
    assert calls[0]["reference_instrumental_path"] == guard_audio / "i.flac"
    assert calls[0]["reference_raw_path"] == guard_audio / "raw.flac"
    assert calls[0]["candidate_instrumental_path"] == candidate
    assert manifest["status"] == "damage_limited_repair_passed"
    assert manifest["source_residual_manifest"] == str(residual_manifest)
    assert manifest["source_guard_manifest"] == str(guard_manifest)
    assert manifest["source_training_status"] == "trained_repair_passed_pitch_guard"
    assert manifest["delivery_allowed"] is False
    assert manifest["listen_pack_created"] is False
    assert manifest["presentation_status"] == "damage_limited_repair_diagnostic_only"
    assert (tmp_path / "out" / "manifest.json").is_file()
    progress_first = json.loads((tmp_path / "out" / "progress.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert progress_first["acceleration_mode"] == "cpu_dsp_scoring"
    assert progress_first["gpu_inference_skipped"] is True


def test_existing_residual_damage_limited_route_accepts_damage_audit_route_source(tmp_path: Path) -> None:
    guard_dir = tmp_path / "guard"
    guard_audio = guard_dir / "audio"
    guard_audio.mkdir(parents=True)
    for name in ("v.flac", "i.flac", "raw.flac"):
        (guard_audio / name).write_bytes(b"audio")
    guard_manifest = guard_dir / "manifest.json"
    _write_json(
        guard_manifest,
        {
            "schema": "mac_mel_residual_guard.v1",
            "audio": {
                "vocals": str(guard_audio / "v.flac"),
                "instrumental": str(guard_audio / "i.flac"),
                "raw_ref": str(guard_audio / "raw.flac"),
            },
            "score": {"regions": [{"start": 22.17, "end": 23.08}], "config": {"window_sec": 0.75}},
            "delivery_allowed": False,
            "listen_pack_created": False,
        },
    )
    candidate = tmp_path / "train" / "audio" / "i.flac"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"audio")
    source_residual_manifest = tmp_path / "residual" / "manifest.json"
    _write_json(
        source_residual_manifest,
        {
            "schema": "mac_mel_residual_refiner_training.v1",
            "status": "residual_batch_trained_passed_pitch_guard",
            "training_status": "trained_repair_passed_pitch_guard",
            "guard_manifest": str(guard_manifest),
            "training_manifest": str(tmp_path / "train" / "manifest.json"),
            "audio": {"instrumental": str(candidate)},
            "delivery_allowed": False,
            "listen_pack_created": False,
        },
    )
    damage_audit_route_manifest = tmp_path / "audit" / "manifest.json"
    damage_audit = {
        "schema": "mac_instrumental_damage_audit.v1",
        "verdict": "fail",
        "failed_region_count": 1,
        "regions": [
            {
                "start": 22.17,
                "end": 23.08,
                "delta_ratio": 0.51,
                "blocker_reasons": ["instrumental_region_overprocessed"],
            }
        ],
        "blocker_reasons": ["instrumental_region_overprocessed"],
        "delivery_allowed": False,
    }
    _write_json(
        damage_audit_route_manifest,
        {
            "schema": "mac_residual_damage_audit_route.v1",
            "status": "residual_damage_audit_failed",
            "source_residual_manifest": str(source_residual_manifest),
            "source_guard_manifest": str(guard_manifest),
            "source_training_manifest": str(tmp_path / "train" / "manifest.json"),
            "source_training_status": "trained_repair_passed_pitch_guard",
            "damage_audit": damage_audit,
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "damage_audit_route_diagnostic_only",
        },
    )
    calls: list[dict[str, object]] = []

    def fake_run_damage_limited_repair(**kwargs):
        calls.append(kwargs)
        return {
            "schema": "mac_residual_damage_limited_repair.v1",
            "status": "damage_limited_repair_passed",
            "audio": {"vocals": "v.flac", "instrumental": "i.flac", "raw_ref": "raw.flac"},
            "after_score": {"verdict": "pass", "regions": []},
            "damage_audit": {"verdict": "pass", "failed_region_count": 0, "blocker_reasons": []},
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "damage_limited_repair_diagnostic_only",
        }

    original = route.run_damage_limited_repair
    try:
        route.run_damage_limited_repair = fake_run_damage_limited_repair
        manifest = route.run_existing_residual_damage_limited_repair(
            residual_manifest_path=damage_audit_route_manifest,
            output_dir=tmp_path / "out",
        )
    finally:
        route.run_damage_limited_repair = original

    assert len(calls) == 1
    assert calls[0]["reference_instrumental_path"] == guard_audio / "i.flac"
    assert calls[0]["candidate_instrumental_path"] == candidate
    assert calls[0]["initial_damage_audit"] == damage_audit
    assert manifest["status"] == "damage_limited_repair_passed"
    assert manifest["source_damage_audit_route_manifest"] == str(damage_audit_route_manifest)
    assert manifest["source_residual_manifest"] == str(source_residual_manifest)
    assert manifest["source_guard_manifest"] == str(guard_manifest)
    assert manifest["source_training_status"] == "trained_repair_passed_pitch_guard"
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "damage_limited_repair_diagnostic_only"


def test_existing_residual_damage_limited_route_requires_pitch_pass_source(tmp_path: Path) -> None:
    residual_manifest = tmp_path / "residual" / "manifest.json"
    _write_json(
        residual_manifest,
        {
            "schema": "mac_mel_residual_refiner_training.v1",
            "status": "residual_trained_damage_failed",
            "training_status": "trained_repair_still_blocked",
            "damage_audit": {"verdict": "fail"},
            "delivery_allowed": False,
            "listen_pack_created": False,
        },
    )
    calls: list[dict[str, object]] = []

    def fake_run_damage_limited_repair(**kwargs):
        calls.append(kwargs)
        return {"status": "damage_limited_repair_passed", "delivery_allowed": False}

    original = route.run_damage_limited_repair
    try:
        route.run_damage_limited_repair = fake_run_damage_limited_repair
        manifest = route.run_existing_residual_damage_limited_repair(
            residual_manifest_path=residual_manifest,
            output_dir=tmp_path / "out",
        )
    finally:
        route.run_damage_limited_repair = original

    assert calls == []
    assert manifest["status"] == "blocked_source_residual_not_pitch_passed"
    assert manifest["failed_checks"] == ["source_residual_not_pitch_passed"]
    assert manifest["delivery_allowed"] is False


def test_existing_residual_damage_limited_route_blocks_malformed_audit_route(tmp_path: Path) -> None:
    damage_audit_route_manifest = tmp_path / "audit" / "manifest.json"
    _write_json(
        damage_audit_route_manifest,
        {
            "schema": "mac_residual_damage_audit_route.v1",
            "status": "residual_damage_audit_failed",
            "source_training_status": "trained_repair_passed_pitch_guard",
            "damage_audit": {"verdict": "fail", "failed_region_count": 1},
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "damage_audit_route_diagnostic_only",
        },
    )
    calls: list[dict[str, object]] = []

    def fake_run_damage_limited_repair(**kwargs):
        calls.append(kwargs)
        return {"status": "damage_limited_repair_passed"}

    original = route.run_damage_limited_repair
    try:
        route.run_damage_limited_repair = fake_run_damage_limited_repair
        manifest = route.run_existing_residual_damage_limited_repair(
            residual_manifest_path=damage_audit_route_manifest,
            output_dir=tmp_path / "out",
        )
    finally:
        route.run_damage_limited_repair = original

    assert calls == []
    assert manifest["status"] == "blocked_source_damage_audit_route_manifest"
    assert manifest["failed_checks"] == ["source_damage_audit_route_manifest"]
    assert manifest["delivery_allowed"] is False
