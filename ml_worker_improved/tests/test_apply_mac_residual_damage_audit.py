from __future__ import annotations

import importlib
import json
from pathlib import Path

from tools.select_mac_residual_pareto_candidate import run_selection


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _route_module():
    return importlib.import_module("tools.apply_mac_residual_damage_audit")


def test_existing_residual_damage_audit_route_preserves_contract_and_selector_flow(tmp_path: Path) -> None:
    route = _route_module()
    guard_audio = tmp_path / "guard" / "audio"
    guard_audio.mkdir(parents=True)
    reference_instrumental = guard_audio / "i.flac"
    reference_instrumental.write_bytes(b"reference")
    guard_manifest = _write_json(
        tmp_path / "guard" / "manifest.json",
        {
            "schema": "mac_mel_residual_guard.v1",
            "audio": {"instrumental": str(reference_instrumental)},
            "score": {
                "schema": "mac_melband_pitch_guard_score.v1",
                "regions": [{"start": 22.17, "end": 23.08, "score": 0.91}],
                "config": {"window_sec": 0.75},
            },
            "delivery_allowed": False,
            "listen_pack_created": False,
        },
    )
    candidate = tmp_path / "train" / "audio" / "i.flac"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"candidate")
    residual_manifest = _write_json(
        tmp_path / "residual" / "manifest.json",
        {
            "schema": "mac_mel_residual_refiner_training.v1",
            "status": "residual_batch_trained_passed_pitch_guard",
            "training_status": "trained_repair_passed_pitch_guard",
            "guard_manifest": str(guard_manifest),
            "training_manifest": str(tmp_path / "train" / "manifest.json"),
            "audio": {"instrumental": str(candidate)},
            "before_region_count": 4,
            "after_region_count": 0,
            "source_audio_path": "/mac/src/source.mp3",
            "source_track_identity": {"track_id": "source", "identity_status": "actual_source_verified"},
            "objective_profile": "pitch_region_detector_objective_v3",
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "residual_training_diagnostic_only",
        },
    )
    calls: list[dict[str, object]] = []

    def fake_run_damage_audit(**kwargs):
        calls.append(kwargs)
        return {
            "schema": "mac_instrumental_damage_audit.v1",
            "verdict": "pass",
            "failed_region_count": 0,
            "blocker_reasons": [],
            "regions": [],
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "damage_audit_diagnostic_only",
        }

    original = route.run_damage_audit
    try:
        route.run_damage_audit = fake_run_damage_audit
        manifest = route.run_existing_residual_damage_audit(
            residual_manifest_path=residual_manifest,
            output_dir=tmp_path / "out",
        )
    finally:
        route.run_damage_audit = original

    score_json = tmp_path / "out" / "score.json"
    assert len(calls) == 1
    assert calls[0]["reference_instrumental_path"] == reference_instrumental
    assert calls[0]["candidate_instrumental_path"] == candidate
    assert calls[0]["score_json_path"] == score_json
    assert calls[0]["output_dir"] == tmp_path / "out" / "damage"
    assert json.loads(score_json.read_text(encoding="utf-8"))["regions"] == [
        {"start": 22.17, "end": 23.08, "score": 0.91}
    ]
    assert manifest["schema"] == "mac_residual_damage_audit_route.v1"
    assert manifest["status"] == "residual_damage_audit_passed"
    assert manifest["source_residual_manifest"] == str(residual_manifest)
    assert manifest["source_guard_manifest"] == str(guard_manifest)
    assert manifest["source_training_status"] == "trained_repair_passed_pitch_guard"
    assert manifest["before_region_count"] == 4
    assert manifest["after_region_count"] == 0
    assert manifest["damage_audit_manifest"] == str(tmp_path / "out" / "damage" / "damage.json")
    assert manifest["delivery_allowed"] is False
    assert manifest["listen_pack_created"] is False
    assert manifest["presentation_status"] == "damage_audit_route_diagnostic_only"
    assert (tmp_path / "out" / "manifest.json").is_file()

    selection = run_selection(candidates=[{"id": "it7da001", "manifest": str(tmp_path / "out" / "manifest.json")}], output_dir=tmp_path / "sel")
    assert selection["status"] == "pareto_candidate_selected_for_full_holdout"
    assert selection["selected_candidate_id"] == "it7da001"
    assert selection["selected_candidate"]["next_action"] == "run_full_holdout_validation"
    assert selection["delivery_allowed"] is False


def test_existing_residual_damage_audit_route_requires_pitch_pass_source(tmp_path: Path) -> None:
    route = _route_module()
    residual_manifest = _write_json(
        tmp_path / "residual" / "manifest.json",
        {
            "schema": "mac_mel_residual_refiner_training.v1",
            "status": "residual_trained_still_blocked",
            "training_status": "trained_repair_still_blocked",
            "before_region_count": 4,
            "after_region_count": 1,
            "delivery_allowed": False,
            "listen_pack_created": False,
        },
    )
    calls: list[dict[str, object]] = []

    def fake_run_damage_audit(**kwargs):
        calls.append(kwargs)
        return {"verdict": "pass"}

    original = route.run_damage_audit
    try:
        route.run_damage_audit = fake_run_damage_audit
        manifest = route.run_existing_residual_damage_audit(
            residual_manifest_path=residual_manifest,
            output_dir=tmp_path / "out",
        )
    finally:
        route.run_damage_audit = original

    assert calls == []
    assert manifest["status"] == "blocked_source_residual_not_pitch_passed"
    assert manifest["failed_checks"] == ["source_residual_not_pitch_passed"]
    assert manifest["delivery_allowed"] is False


def test_existing_residual_damage_audit_route_rejects_already_audited_source(tmp_path: Path) -> None:
    route = _route_module()
    residual_manifest = _write_json(
        tmp_path / "residual" / "manifest.json",
        {
            "schema": "mac_mel_residual_refiner_training.v1",
            "status": "residual_batch_trained_damage_failed",
            "training_status": "trained_repair_passed_pitch_guard",
            "before_region_count": 4,
            "after_region_count": 0,
            "damage_audit": {"verdict": "fail", "failed_region_count": 1},
            "delivery_allowed": False,
            "listen_pack_created": False,
        },
    )

    manifest = route.run_existing_residual_damage_audit(
        residual_manifest_path=residual_manifest,
        output_dir=tmp_path / "out",
    )

    assert manifest["status"] == "blocked_source_residual_already_damage_audited"
    assert manifest["failed_checks"] == ["source_residual_already_damage_audited"]
    assert manifest["next_action"] == "run_damage_limited_repair"
    assert manifest["delivery_allowed"] is False
