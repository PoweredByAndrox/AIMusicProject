from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import select_mac_residual_pareto_candidate as selector
from tools.select_mac_residual_pareto_candidate import run_selection


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _candidate(tmp_path: Path, candidate_id: str, *, before: int, after: int, damage_verdict: str, damage_failed: int = 0) -> dict[str, str]:
    root = tmp_path / candidate_id
    manifest = _write_json(
        root / "manifest.json",
        {
            "status": "residual_batch_trained_damage_failed" if damage_verdict != "pass" else "residual_trained_still_blocked",
            "training_status": "trained_repair_passed_pitch_guard" if after == 0 else "trained_repair_still_blocked",
            "before_region_count": before,
            "after_region_count": after,
            "delivery_allowed": False,
            "presentation_status": "residual_training_diagnostic_only",
        },
    )
    regions = [
        {
            "start": 22.17 + index * 0.25,
            "end": 23.08 + index * 0.25,
            "delta_ratio": 0.55 + index * 0.1,
            "blocker_reasons": ["instrumental_region_overprocessed"],
        }
        for index in range(damage_failed)
    ]
    damage = _write_json(
        root / "damage.json",
        {
            "schema": "mac_instrumental_damage_audit.v1",
            "verdict": damage_verdict,
            "failed_region_count": damage_failed,
            "blocker_reasons": [] if damage_verdict == "pass" else ["instrumental_region_overprocessed"],
            "regions": regions,
            "delivery_allowed": False,
            "presentation_status": "damage_audit_diagnostic_only",
        },
    )
    return {"id": candidate_id, "manifest": str(manifest), "damage": str(damage)}


def _manifest_only_candidate(tmp_path: Path, candidate_id: str, *, before: int, after: int, damage_audit: dict[str, object] | None = None) -> dict[str, str]:
    manifest = {
        "status": "residual_batch_trained_passed_pitch_guard" if after == 0 else "residual_trained_still_blocked",
        "training_status": "trained_repair_passed_pitch_guard" if after == 0 else "trained_repair_still_blocked",
        "before_region_count": before,
        "after_region_count": after,
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "residual_training_diagnostic_only",
        "full_holdout_resolution_status": "not_evaluated",
    }
    if damage_audit is not None:
        manifest["damage_audit"] = damage_audit
    return {
        "id": candidate_id,
        "manifest": str(_write_json(tmp_path / candidate_id / "manifest.json", manifest)),
    }


def test_pareto_selection_blocks_when_no_candidate_passes_pitch_and_damage(tmp_path: Path) -> None:
    result = run_selection(
        candidates=[
            _candidate(tmp_path, "pitch_only", before=4, after=0, damage_verdict="fail", damage_failed=2),
            _candidate(tmp_path, "damage_only", before=4, after=1, damage_verdict="pass"),
        ],
        output_dir=tmp_path / "sel",
    )

    assert result["status"] == "blocked_no_pareto_candidate"
    assert result["selected_candidate_id"] == ""
    assert result["delivery_allowed"] is False
    assert "no_candidate_passes_pitch_and_damage" in result["blocker_reasons"]
    assert {row["candidate_id"]: row["eligible_for_full_holdout"] for row in result["candidates"]} == {
        "pitch_only": False,
        "damage_only": False,
    }


def test_pareto_selection_selects_only_pitch_and_damage_pass_candidate(tmp_path: Path) -> None:
    result = run_selection(
        candidates=[
            _candidate(tmp_path, "pitch_only", before=4, after=0, damage_verdict="fail", damage_failed=2),
            _candidate(tmp_path, "pareto_ok", before=4, after=0, damage_verdict="pass"),
        ],
        output_dir=tmp_path / "sel",
    )

    assert result["status"] == "pareto_candidate_selected_for_full_holdout"
    assert result["selected_candidate_id"] == "pareto_ok"
    assert result["selected_candidate"]["pitch_pass"] is True
    assert result["selected_candidate"]["damage_pass"] is True
    assert result["delivery_allowed"] is False
    assert result["presentation_status"] == "pareto_selection_diagnostic_only"


def test_pareto_selection_routes_pitch_pass_without_retraining_loop(tmp_path: Path) -> None:
    result = run_selection(
        candidates=[
            _manifest_only_candidate(tmp_path, "pitch_without_damage", before=4, after=0),
            _candidate(tmp_path, "pitch_damage_failed", before=4, after=0, damage_verdict="fail", damage_failed=1),
            _candidate(tmp_path, "needs_more_pitch_work", before=4, after=1, damage_verdict="pass"),
            _candidate(tmp_path, "ready_for_full_holdout", before=4, after=0, damage_verdict="pass"),
        ],
        output_dir=tmp_path / "sel",
    )

    rows = {row["candidate_id"]: row for row in result["candidates"]}
    assert rows["pitch_without_damage"]["next_action"] == "run_damage_audit"
    assert rows["pitch_without_damage"]["stop_retraining_reason"] == "pitch_pass_waiting_for_damage_audit"
    assert rows["pitch_damage_failed"]["next_action"] == "run_damage_limited_repair"
    assert rows["pitch_damage_failed"]["stop_retraining_reason"] == "pitch_pass_damage_failed"
    assert rows["needs_more_pitch_work"]["next_action"] == "continue_residual_training_or_rebatch"
    assert rows["ready_for_full_holdout"]["next_action"] == "run_full_holdout_validation"
    assert result["next_action_counts"] == {
        "run_full_holdout_validation": 1,
        "run_damage_limited_repair": 1,
        "run_damage_audit": 1,
        "continue_residual_training_or_rebatch": 1,
    }


def test_pareto_selection_accepts_damage_limited_route_for_full_holdout(tmp_path: Path) -> None:
    root = tmp_path / "dlim"
    manifest = _write_json(
        root / "manifest.json",
        {
            "schema": "mac_residual_damage_limited_repair_route.v1",
            "status": "damage_limited_repair_passed",
            "source_residual_status": "residual_batch_trained_damage_failed",
            "source_training_status": "trained_repair_passed_pitch_guard",
            "source_audio_path": "/mac/src/be-nicegreenhorn.mp3",
            "source_track_identity": {
                "track_id": "be-nicegreenhorn",
                "identity_status": "actual_source_verified",
            },
            "source_residual_manifest": "/mac/it7rr005/r001/manifest.json",
            "source_damage_audit_route_manifest": "/mac/it7da001/r001/manifest.json",
            "source_guard_manifest": "/mac/it7rr005/r001/guard/manifest.json",
            "after_score": {
                "schema": "mac_melband_pitch_guard_score.v1",
                "verdict": "pass",
                "regions": [],
            },
            "damage_audit": {
                "schema": "mac_instrumental_damage_audit.v1",
                "verdict": "pass",
                "failed_region_count": 0,
                "blocker_reasons": [],
                "delivery_allowed": False,
            },
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "damage_limited_repair_diagnostic_only",
            "full_holdout_resolution_status": "not_evaluated",
        },
    )

    result = run_selection(
        candidates=[_candidate(tmp_path, "pitch_only", before=4, after=0, damage_verdict="fail", damage_failed=1), {"id": "it7dl003", "manifest": str(manifest)}],
        output_dir=tmp_path / "sel",
    )

    selected = result["selected_candidate"]
    assert result["status"] == "pareto_candidate_selected_for_full_holdout"
    assert result["selected_candidate_id"] == "it7dl003"
    assert selected["schema"] == "mac_residual_damage_limited_repair_route.v1"
    assert selected["pitch_pass"] is True
    assert selected["damage_pass"] is True
    assert selected["eligible_for_full_holdout"] is True
    assert selected["eligible_for_delivery"] is False
    assert selected["eligible_for_promotion"] is False
    assert "full_holdout_not_passed" in selected["delivery_blocker_reasons"]
    assert "reference_bundle_not_verified_for_actual_source" in selected["delivery_blocker_reasons"]
    assert selected["source_training_status"] == "trained_repair_passed_pitch_guard"
    assert selected["source_residual_manifest"] == "/mac/it7rr005/r001/manifest.json"
    assert selected["source_damage_audit_route_manifest"] == "/mac/it7da001/r001/manifest.json"
    assert selected["source_audio_path"] == "/mac/src/be-nicegreenhorn.mp3"
    assert selected["source_track_identity"]["track_id"] == "be-nicegreenhorn"
    assert selected["source_guard_manifest"] == "/mac/it7rr005/r001/guard/manifest.json"
    assert selected["reference_bundle_status"] == "missing_for_actual_source"
    assert result["delivery_allowed"] is False


def test_cli_accepts_manifest_glob_for_residual_run_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _manifest_only_candidate(tmp_path / "runs" / "it7rr002", "r001", before=4, after=0)
    _manifest_only_candidate(
        tmp_path / "runs" / "it7rr005",
        "r001",
        before=4,
        after=0,
        damage_audit={
            "schema": "mac_instrumental_damage_audit.v1",
            "verdict": "fail",
            "failed_region_count": 1,
            "blocker_reasons": ["instrumental_region_overprocessed"],
            "delivery_allowed": False,
            "listen_pack_created": False,
        },
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "select_mac_residual_pareto_candidate.py",
            "--manifest-glob",
            str(tmp_path / "runs" / "it7rr*" / "r*" / "manifest.json"),
            "--out",
            str(tmp_path / "sel"),
        ],
    )

    assert selector.main() == 0
    result = json.loads((tmp_path / "sel" / "selection.json").read_text(encoding="utf-8"))
    assert {row["candidate_id"]: row["next_action"] for row in result["candidates"]} == {
        "it7rr002_r001": "run_damage_audit",
        "it7rr005_r001": "run_damage_limited_repair",
    }


@pytest.mark.parametrize(
    ("manifest_patch", "damage_patch", "expected_field"),
    [
        ({"status": "damage_limited_repair_failed"}, {}, "status"),
        ({"presentation_status": "accepted_for_listen_pack"}, {}, "presentation_status"),
        ({"full_holdout_resolution_status": "passed"}, {}, "full_holdout_resolution_status"),
        ({}, {"delivery_allowed": True}, "damage_delivery_allowed"),
        ({}, {"listen_pack_created": True}, "damage_listen_pack_created"),
    ],
)
def test_pareto_selection_rejects_malformed_damage_limited_route(
    tmp_path: Path,
    manifest_patch: dict[str, object],
    damage_patch: dict[str, object],
    expected_field: str,
) -> None:
    damage = {
        "schema": "mac_instrumental_damage_audit.v1",
        "verdict": "pass",
        "failed_region_count": 0,
        "blocker_reasons": [],
        "delivery_allowed": False,
        "listen_pack_created": False,
    }
    damage.update(damage_patch)
    manifest = {
        "schema": "mac_residual_damage_limited_repair_route.v1",
        "status": "damage_limited_repair_passed",
        "source_training_status": "trained_repair_passed_pitch_guard",
        "source_residual_manifest": "/mac/it7rr005/r001/manifest.json",
        "after_score": {"verdict": "pass", "regions": []},
        "damage_audit": damage,
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "damage_limited_repair_diagnostic_only",
        "full_holdout_resolution_status": "not_evaluated",
    }
    manifest.update(manifest_patch)
    manifest_path = _write_json(tmp_path / "dlim" / "manifest.json", manifest)

    result = run_selection(candidates=[{"id": "it7dl_bad", "manifest": str(manifest_path)}], output_dir=tmp_path / "sel")

    row = result["candidates"][0]
    assert result["status"] == "blocked_no_pareto_candidate"
    assert row["pitch_pass"] is True
    assert row["damage_pass"] is True
    assert row["eligible_for_promotion"] is False
    assert expected_field in row["promotion_blocker_reasons"]
    assert row["next_action"] == "fix_promotion_contract_blocker"


def test_pareto_selection_rejects_delivery_marked_candidate(tmp_path: Path) -> None:
    manifest = _write_json(
        tmp_path / "bad_delivery" / "manifest.json",
        {
            "status": "residual_trained_passed_pitch_guard",
            "training_status": "trained_repair_passed_pitch_guard",
            "before_region_count": 4,
            "after_region_count": 0,
            "damage_audit": {"verdict": "pass", "failed_region_count": 0, "blocker_reasons": []},
            "delivery_allowed": True,
            "listen_pack_created": False,
        },
    )

    result = run_selection(candidates=[{"id": "bad_delivery", "manifest": str(manifest)}], output_dir=tmp_path / "sel")

    assert result["status"] == "blocked_no_pareto_candidate"
    assert result["candidates"][0]["pitch_pass"] is True
    assert result["candidates"][0]["damage_pass"] is True
    assert result["candidates"][0]["source_delivery_allowed"] is True
    assert result["candidates"][0]["eligible_for_promotion"] is False
    assert result["candidates"][0]["next_action"] == "fix_promotion_contract_blocker"


def test_pareto_selection_rejects_malformed_damage_audit_route(tmp_path: Path) -> None:
    manifest = _write_json(
        tmp_path / "audit_route" / "manifest.json",
        {
            "schema": "mac_residual_damage_audit_route.v1",
            "status": "residual_damage_audit_failed",
            "training_status": "trained_repair_passed_pitch_guard",
            "source_training_status": "trained_repair_passed_pitch_guard",
            "source_residual_manifest": "/mac/it7rr002/r001/manifest.json",
            "source_guard_manifest": "/mac/it7rr002/r001/guard/manifest.json",
            "before_region_count": 4,
            "after_region_count": 0,
            "damage_audit": {
                "schema": "mac_instrumental_damage_audit.v1",
                "verdict": "pass",
                "failed_region_count": 0,
                "blocker_reasons": [],
                "delivery_allowed": True,
            },
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "damage_audit_route_diagnostic_only",
            "full_holdout_resolution_status": "not_evaluated",
        },
    )

    result = run_selection(candidates=[{"id": "bad_audit_route", "manifest": str(manifest)}], output_dir=tmp_path / "sel")

    row = result["candidates"][0]
    assert result["status"] == "blocked_no_pareto_candidate"
    assert row["pitch_pass"] is True
    assert row["damage_pass"] is True
    assert row["eligible_for_promotion"] is False
    assert "status" in row["promotion_blocker_reasons"]
    assert "damage_delivery_allowed" in row["promotion_blocker_reasons"]
    assert row["next_action"] == "fix_promotion_contract_blocker"


def test_cli_accepts_candidates_json_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    candidates_path = _write_json(
        tmp_path / "candidates.json",
        {
            "candidates": [
                _candidate(tmp_path, "pitch_only", before=4, after=0, damage_verdict="fail", damage_failed=2),
                _candidate(tmp_path, "pareto_ok", before=4, after=0, damage_verdict="pass"),
            ]
        },
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "select_mac_residual_pareto_candidate.py",
            "--candidates-json",
            str(candidates_path),
            "--out",
            str(tmp_path / "sel"),
        ],
    )

    assert selector.main() == 0
    result = json.loads((tmp_path / "sel" / "selection.json").read_text(encoding="utf-8"))
    assert result["selected_candidate_id"] == "pareto_ok"
