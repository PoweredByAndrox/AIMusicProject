from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from tools import train_mac_refiner_from_fulltrack_residuals as residual_train


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_failed_stage(root: Path, *, regions: list[dict[str, object]] | None = None) -> Path:
    replay_dir = root / "replay"
    audio_dir = replay_dir / "audio"
    audio_dir.mkdir(parents=True)
    for name in ("v.flac", "i.flac", "raw.flac"):
        (audio_dir / name).write_bytes(b"audio")
    _write_json(
        replay_dir / "manifest.json",
        {
            "audio": {
                "vocals": str(audio_dir / "v.flac"),
                "instrumental": str(audio_dir / "i.flac"),
                "raw_ref": str(audio_dir / "raw.flac"),
            },
            "delivery_allowed": False,
            "listen_pack_created": False,
        },
    )
    _write_json(
        replay_dir / "score" / "after" / "pitch.json",
        {
            "verdict": "fail" if regions else "pass",
            "blocker_reasons": ["pitch_shifted_vocal_in_instrumental"] if regions else [],
            "regions": regions or [],
            "config": {"window_sec": 0.75, "hop_sec": 0.25},
        },
    )
    _write_json(
        root / "manifest.json",
        {
            "status": "diagnostic_blocked_pitch_guard" if regions else "diagnostic_passed",
            "replay_manifest": str(replay_dir / "manifest.json"),
            "after_pitch_verdict": "fail" if regions else "pass",
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "fulltrack_diagnostic_only",
        },
    )
    return root


def _write_composition_stage(root: Path, *, regions: list[dict[str, object]]) -> Path:
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True)
    for name in ("v.flac", "i.flac", "raw.flac"):
        (audio_dir / name).write_bytes(b"audio")
    score_dir = root / "score" / "after"
    _write_json(
        score_dir / "pitch.json",
        {
            "verdict": "fail" if regions else "pass",
            "blocker_reasons": ["pitch_shifted_vocal_in_instrumental"] if regions else [],
            "regions": regions,
            "total_fail_regions": len(regions),
            "selected_region_count": len(regions),
            "regions_capped": False,
            "failure_taxonomy": {
                "by_probe": {"0.0": len(regions)},
                "by_fail_mode": {"strong_projection_stft": len(regions)},
            },
            "config": {"window_sec": 0.75, "hop_sec": 0.25},
        },
    )
    _write_json(
        root / "manifest.json",
        {
            "schema": "mac_mel_residual_chunk_composition.v1",
            "status": "composition_blocked_pitch_guard" if regions else "composition_passed",
            "audio": {
                "vocals": str(audio_dir / "v.flac"),
                "instrumental": str(audio_dir / "i.flac"),
                "raw_ref": str(audio_dir / "raw.flac"),
            },
            "failed_checks": ["pitch_guard_failed"] if regions else [],
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "chunk_composition_diagnostic_only",
        },
    )
    return root


def test_residual_guard_accepts_composition_stage_as_remaining_region_source() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        failed_stage = _write_composition_stage(
            root / "stage",
            regions=[
                {"start": 22.25, "end": 23.0, "probe_semitones": 0.0, "fail_mode": "strong_projection_stft", "score": 0.49},
                {"start": 143.5, "end": 144.25, "probe_semitones": 7.0, "fail_mode": "standard_score", "score": 0.61},
            ],
        )

        result = residual_train.build_residual_guard_from_failed_stage(
            failed_stage_dir=failed_stage,
            output_dir=root / "out",
        )

        guard_manifest = result["guard_manifest"]
        assert result["status"] == "ok"
        assert result["residual_region_count"] == 2
        assert guard_manifest["source_stage_manifest"] == str(failed_stage / "manifest.json")
        assert guard_manifest["source_replay_manifest"] == ""
        assert guard_manifest["source_stage_schema"] == "mac_mel_residual_chunk_composition.v1"
        assert guard_manifest["score"]["source_after_score"] == str(failed_stage / "score" / "after" / "pitch.json")
        assert [region["start"] for region in guard_manifest["score"]["regions"]] == [22.25, 143.5]
        assert guard_manifest["delivery_allowed"] is False
        assert guard_manifest["listen_pack_created"] is False


def test_residual_guard_accepts_training_stage_as_remaining_region_source() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        failed_stage = _write_composition_stage(
            root / "stage",
            regions=[
                {"start": 22.25, "end": 23.0, "probe_semitones": 0.0, "fail_mode": "strong_projection_stft", "score": 0.54},
            ],
        )
        stage_manifest_path = failed_stage / "manifest.json"
        stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
        stage_manifest["schema"] = "mac_melband_refiner_training.v1"
        stage_manifest["status"] = "trained_repair_still_blocked"
        stage_manifest_path.write_text(json.dumps(stage_manifest), encoding="utf-8")

        result = residual_train.build_residual_guard_from_failed_stage(
            failed_stage_dir=failed_stage,
            output_dir=root / "out",
        )

        guard_manifest = result["guard_manifest"]
        assert result["status"] == "ok"
        assert result["residual_region_count"] == 1
        assert guard_manifest["source_stage_schema"] == "mac_melband_refiner_training.v1"
        assert guard_manifest["source_replay_manifest"] == ""
        assert guard_manifest["score"]["regions"][0]["start"] == 22.25
        assert guard_manifest["delivery_allowed"] is False


def test_residual_training_requires_remaining_pitch_regions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        failed_stage = _write_failed_stage(root / "stage", regions=[])

        manifest = residual_train.run_residual_training(
            failed_stage_dir=failed_stage,
            output_dir=root / "out",
            target_gain=2.5,
            epochs=1,
            require_mps=True,
        )

        assert manifest["status"] == "blocked_no_residual_pitch_regions"
        assert manifest["delivery_allowed"] is False
        assert manifest["listen_pack_created"] is False


def test_residual_training_builds_guard_and_keeps_diagnostic_contract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        failed_stage = _write_failed_stage(
            root / "stage",
            regions=[{"start": 15.5, "end": 16.25, "semitones": -12.0, "score": 0.48}],
        )
        calls: list[dict[str, object]] = []
        candidate = root / "candidate.flac"
        candidate.write_bytes(b"audio")

        def fake_run_training(**kwargs):
            calls.append(kwargs)
            train_out = Path(kwargs["output_dir"])
            checkpoint = train_out / "model.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"pt")
            return {
                "schema": "mac_melband_refiner_training.v1",
                "status": "trained_repair_passed_pitch_guard",
                "checkpoint": str(checkpoint),
                "label_type": "pseudo_teacher",
                "is_ground_truth": False,
                "usable_for_promotion": False,
                "delivery_allowed": False,
                "listen_pack_created": False,
                "audio": {"instrumental": str(candidate)},
                "metrics": {"before_region_count": 1, "after_region_count": 0},
            }

        def fake_damage_audit(**kwargs):
            return {
                "schema": "mac_instrumental_damage_audit.v1",
                "verdict": "pass",
                "failed_region_count": 0,
                "blocker_reasons": [],
                "delivery_allowed": False,
                "presentation_status": "damage_audit_diagnostic_only",
            }

        original = residual_train.run_training
        original_damage = residual_train.run_damage_audit
        try:
            residual_train.run_training = fake_run_training
            residual_train.run_damage_audit = fake_damage_audit
            manifest = residual_train.run_residual_training(
                failed_stage_dir=failed_stage,
                output_dir=root / "out",
                target_gain=2.5,
                epochs=3,
                require_mps=True,
            )
        finally:
            residual_train.run_training = original
            residual_train.run_damage_audit = original_damage

        assert len(calls) == 1
        assert Path(calls[0]["mel_guard_dir"]).name == "guard"
        assert calls[0]["target_gain"] == 2.5
        assert manifest["status"] == "residual_trained_passed_pitch_guard"
        assert manifest["damage_audit"]["verdict"] == "pass"
        assert manifest["residual_region_count"] == 1
        assert manifest["label_type"] == "pseudo_teacher"
        assert manifest["is_ground_truth"] is False
        assert manifest["usable_for_promotion"] is False
        assert manifest["delivery_allowed"] is False
        assert manifest["listen_pack_created"] is False


def test_residual_training_filters_regions_by_taxonomy_batch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        failed_stage = _write_failed_stage(
            root / "stage",
            regions=[
                {"start": 1.0, "end": 1.5, "probe_semitones": 0.0, "fail_mode": "strong_projection_stft", "score": 0.61},
                {"start": 2.0, "end": 2.5, "probe_semitones": -9.0, "fail_mode": "strong_projection_stft", "score": 0.51},
                {"start": 3.0, "end": 3.5, "probe_semitones": -9.0, "fail_mode": "standard_score", "score": 0.49},
            ],
        )
        calls: list[dict[str, object]] = []

        def fake_run_training(**kwargs):
            calls.append(kwargs)
            guard_manifest = json.loads((Path(kwargs["mel_guard_dir"]) / "manifest.json").read_text(encoding="utf-8"))
            assert [region["start"] for region in guard_manifest["score"]["regions"]] == [2.0, 3.0]
            assert guard_manifest["region_batch"] == {
                "axis": "probe",
                "value": "-9.0",
                "expected_region_count": 2,
                "batch_id": "b002",
            }
            return {
                "schema": "mac_melband_refiner_training.v1",
                "status": "trained_repair_still_blocked",
                "checkpoint": str(Path(kwargs["output_dir"]) / "model.pt"),
                "delivery_allowed": False,
                "listen_pack_created": False,
                "metrics": {"before_region_count": 2, "after_region_count": 2, "target_issue_improved": False},
            }

        original = residual_train.run_training
        try:
            residual_train.run_training = fake_run_training
            manifest = residual_train.run_residual_training(
                failed_stage_dir=failed_stage,
                output_dir=root / "out",
                target_gain=2.5,
                epochs=3,
                require_mps=True,
                region_batch={"axis": "probe", "value": "-9.0", "expected_region_count": 2, "batch_id": "b002"},
            )
        finally:
            residual_train.run_training = original

        assert len(calls) == 1
        assert manifest["status"] == "residual_trained_still_blocked"
        assert manifest["residual_region_count"] == 2
        assert manifest["source_residual_region_count"] == 3
        assert manifest["taxonomy_batch"]["batch_id"] == "b002"
        assert manifest["source_region_mode"] == "selected_regions"
        assert manifest["delivery_allowed"] is False


def test_residual_training_filters_regions_by_fail_mode_batch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        failed_stage = _write_failed_stage(
            root / "stage",
            regions=[
                {"start": 1.0, "end": 1.5, "probe_semitones": 0.0, "fail_mode": "strong_projection_stft", "score": 0.61},
                {"start": 2.0, "end": 2.5, "probe_semitones": -9.0, "fail_mode": "standard_score", "score": 0.51},
                {"start": 3.0, "end": 3.5, "probe_semitones": -9.0, "fail_mode": "strong_projection_stft", "score": 0.49},
            ],
        )
        calls: list[dict[str, object]] = []

        def fake_run_training(**kwargs):
            calls.append(kwargs)
            guard_manifest = json.loads((Path(kwargs["mel_guard_dir"]) / "manifest.json").read_text(encoding="utf-8"))
            assert [region["start"] for region in guard_manifest["score"]["regions"]] == [1.0, 3.0]
            assert guard_manifest["region_batch"] == {
                "axis": "fail_mode",
                "value": "strong_projection_stft",
                "expected_region_count": 2,
                "batch_id": "b001",
            }
            return {
                "schema": "mac_melband_refiner_training.v1",
                "status": "trained_repair_still_blocked",
                "checkpoint": str(Path(kwargs["output_dir"]) / "model.pt"),
                "delivery_allowed": False,
                "listen_pack_created": False,
                "metrics": {"before_region_count": 2, "after_region_count": 2, "target_issue_improved": False},
            }

        original = residual_train.run_training
        try:
            residual_train.run_training = fake_run_training
            manifest = residual_train.run_residual_training(
                failed_stage_dir=failed_stage,
                output_dir=root / "out",
                target_gain=2.5,
                epochs=3,
                require_mps=True,
                region_batch={
                    "axis": "fail_mode",
                    "value": "strong_projection_stft",
                    "expected_region_count": 2,
                    "batch_id": "b001",
                },
            )
        finally:
            residual_train.run_training = original

        assert len(calls) == 1
        assert manifest["status"] == "residual_trained_still_blocked"
        assert manifest["residual_region_count"] == 2
        assert manifest["source_residual_region_count"] == 3
        assert manifest["taxonomy_batch"]["axis"] == "fail_mode"
        assert manifest["delivery_allowed"] is False


def test_residual_training_passes_backend_objective_and_probe_values() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        failed_stage = _write_failed_stage(
            root / "stage",
            regions=[
                {"start": 22.25, "end": 23.0, "probe_semitones": 0.0, "fail_mode": "strong_projection_stft", "score": 0.49},
                {"start": 106.25, "end": 107.0, "probe_semitones": 0.0, "fail_mode": "strong_projection_stft", "score": 0.43},
            ],
        )
        calls: list[dict[str, object]] = []

        def fake_run_training(**kwargs):
            calls.append(kwargs)
            return {
                "schema": "mac_melband_refiner_training.v1",
                "status": "trained_repair_still_blocked",
                "checkpoint": str(Path(kwargs["output_dir"]) / "model.pt"),
                "delivery_allowed": False,
                "listen_pack_created": False,
                "metrics": {"before_region_count": 2, "after_region_count": 1, "target_issue_improved": True},
            }

        original = residual_train.run_training
        try:
            residual_train.run_training = fake_run_training
            manifest = residual_train.run_residual_training(
                failed_stage_dir=failed_stage,
                output_dir=root / "out",
                target_gain=3.5,
                epochs=5,
                require_mps=True,
                max_regions=2,
                objective_profile="pitch_region_detector_objective_v3",
                probe_semitones=[0.0],
                region_batch={
                    "axis": "fail_mode",
                    "value": "strong_projection_stft",
                    "expected_region_count": 2,
                    "batch_id": "bsp001",
                },
            )
        finally:
            residual_train.run_training = original

        assert len(calls) == 1
        assert calls[0]["objective_profile"] == "pitch_region_detector_objective_v3"
        assert calls[0]["probe_semitones"] == [0.0]
        assert calls[0]["max_regions"] == 2
        assert manifest["objective_profile"] == "pitch_region_detector_objective_v3"
        assert manifest["probe_semitones"] == [0.0]
        assert manifest["delivery_allowed"] is False


def test_residual_training_marks_batch_pass_as_batch_scoped_diagnostic() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        failed_stage = _write_failed_stage(
            root / "stage",
            regions=[
                {"start": 1.0, "end": 1.5, "probe_semitones": 0.0, "score": 0.61},
                {"start": 2.0, "end": 2.5, "probe_semitones": -9.0, "score": 0.51},
            ],
        )
        candidate = root / "candidate.flac"
        candidate.write_bytes(b"audio")

        def fake_run_training(**kwargs):
            return {
                "schema": "mac_melband_refiner_training.v1",
                "status": "trained_repair_passed_pitch_guard",
                "checkpoint": str(Path(kwargs["output_dir"]) / "model.pt"),
                "delivery_allowed": False,
                "listen_pack_created": False,
                "audio": {"instrumental": str(candidate)},
                "metrics": {"before_region_count": 1, "after_region_count": 0, "target_issue_improved": True},
            }

        def fake_damage_audit(**kwargs):
            return {
                "schema": "mac_instrumental_damage_audit.v1",
                "verdict": "pass",
                "failed_region_count": 0,
                "blocker_reasons": [],
                "delivery_allowed": False,
                "presentation_status": "damage_audit_diagnostic_only",
            }

        original = residual_train.run_training
        original_damage = residual_train.run_damage_audit
        try:
            residual_train.run_training = fake_run_training
            residual_train.run_damage_audit = fake_damage_audit
            manifest = residual_train.run_residual_training(
                failed_stage_dir=failed_stage,
                output_dir=root / "out",
                target_gain=2.5,
                epochs=3,
                require_mps=True,
                region_batch={"axis": "probe", "value": "-9.0", "expected_region_count": 1, "batch_id": "b002"},
            )
        finally:
            residual_train.run_training = original
            residual_train.run_damage_audit = original_damage

        assert manifest["status"] == "residual_batch_trained_passed_pitch_guard"
        assert manifest["damage_audit"]["verdict"] == "pass"
        assert manifest["resolution_scope"] == "taxonomy_batch"
        assert manifest["full_holdout_resolution_status"] == "not_evaluated"
        assert manifest["delivery_allowed"] is False
        assert manifest["presentation_status"] == "residual_training_diagnostic_only"


def test_residual_training_blocks_pitch_pass_when_damage_audit_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        failed_stage = _write_failed_stage(
            root / "stage",
            regions=[{"start": 22.25, "end": 23.0, "probe_semitones": 0.0, "fail_mode": "strong_projection_stft", "score": 0.54}],
        )
        candidate = root / "candidate.flac"
        candidate.write_bytes(b"audio")
        damage_calls: list[dict[str, object]] = []

        def fake_run_training(**kwargs):
            return {
                "schema": "mac_melband_refiner_training.v1",
                "status": "trained_repair_passed_pitch_guard",
                "checkpoint": str(Path(kwargs["output_dir"]) / "model.pt"),
                "delivery_allowed": False,
                "listen_pack_created": False,
                "audio": {"instrumental": str(candidate)},
                "metrics": {"before_region_count": 1, "after_region_count": 0, "target_issue_improved": True},
            }

        def fake_damage_audit(**kwargs):
            damage_calls.append(kwargs)
            return {
                "schema": "mac_instrumental_damage_audit.v1",
                "verdict": "fail",
                "failed_region_count": 1,
                "blocker_reasons": ["instrumental_region_overprocessed"],
                "delivery_allowed": False,
                "presentation_status": "damage_audit_diagnostic_only",
            }

        original_training = residual_train.run_training
        original_damage = getattr(residual_train, "run_damage_audit", None)
        try:
            residual_train.run_training = fake_run_training
            residual_train.run_damage_audit = fake_damage_audit
            manifest = residual_train.run_residual_training(
                failed_stage_dir=failed_stage,
                output_dir=root / "out",
                target_gain=4.0,
                epochs=3,
                require_mps=True,
                region_batch={"axis": "fail_mode", "value": "strong_projection_stft", "expected_region_count": 1, "batch_id": "bsp001"},
            )
        finally:
            residual_train.run_training = original_training
            if original_damage is None:
                delattr(residual_train, "run_damage_audit")
            else:
                residual_train.run_damage_audit = original_damage

        assert len(damage_calls) == 1
        assert damage_calls[0]["reference_instrumental_path"] == root / "out" / "guard" / "audio" / "i.flac"
        assert damage_calls[0]["candidate_instrumental_path"] == candidate
        assert Path(damage_calls[0]["score_json_path"]).is_file()
        assert manifest["status"] == "residual_batch_trained_damage_failed"
        assert manifest["failed_checks"] == ["instrumental_damage_audit_failed"]
        assert manifest["damage_audit"]["verdict"] == "fail"
        assert manifest["damage_audit"]["blocker_reasons"] == ["instrumental_region_overprocessed"]
        assert manifest["training_status"] == "trained_repair_passed_pitch_guard"
        assert manifest["delivery_allowed"] is False


def test_damage_limited_delta_attenuates_only_failed_regions_and_preserves_mix() -> None:
    sample_rate = 10
    frames = 30
    reference_vocals = np.full((frames, 2), 0.6, dtype=np.float32)
    reference_instrumental = np.full((frames, 2), 0.1, dtype=np.float32)
    candidate_instrumental = reference_instrumental.copy()
    candidate_instrumental[0:5] += 0.2
    candidate_instrumental[10:20] += 0.8

    limited_vocals, limited_instrumental, applied = residual_train._apply_damage_limited_delta(
        reference_vocals=reference_vocals,
        reference_instrumental=reference_instrumental,
        candidate_instrumental=candidate_instrumental,
        sample_rate=sample_rate,
        damage_regions=[
            {
                "start": 1.0,
                "end": 2.0,
                "delta_ratio": 0.84,
                "blocker_reasons": ["instrumental_region_overprocessed"],
            }
        ],
        max_region_delta_ratio=0.42,
        safety_margin=1.0,
        fade_sec=0.0,
    )

    assert applied == [
        {
            "start": 1.0,
            "end": 2.0,
            "source_delta_ratio": 0.84,
            "scale": 0.5,
            "blocker_reasons": ["instrumental_region_overprocessed"],
        }
    ]
    np.testing.assert_allclose(limited_instrumental[0:5], candidate_instrumental[0:5], atol=1.0e-6)
    np.testing.assert_allclose(limited_instrumental[10:20], reference_instrumental[10:20] + 0.4, atol=1.0e-6)
    np.testing.assert_allclose(limited_vocals + limited_instrumental, reference_vocals + reference_instrumental, atol=1.0e-6)


def test_residual_training_accepts_damage_limited_retry_as_diagnostic_pass() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        failed_stage = _write_failed_stage(
            root / "stage",
            regions=[{"start": 22.25, "end": 23.0, "probe_semitones": 0.0, "fail_mode": "strong_projection_stft", "score": 0.54}],
        )
        candidate = root / "candidate.flac"
        candidate.write_bytes(b"audio")
        limited_i = root / "out" / "dlim" / "audio" / "i.flac"
        limited_v = root / "out" / "dlim" / "audio" / "v.flac"
        limited_raw = root / "out" / "dlim" / "audio" / "raw.flac"
        for path in (limited_i, limited_v, limited_raw):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"audio")
        limiter_calls: list[dict[str, object]] = []

        def fake_run_training(**kwargs):
            return {
                "schema": "mac_melband_refiner_training.v1",
                "status": "trained_repair_passed_pitch_guard",
                "checkpoint": str(Path(kwargs["output_dir"]) / "model.pt"),
                "delivery_allowed": False,
                "listen_pack_created": False,
                "audio": {"instrumental": str(candidate)},
                "metrics": {"before_region_count": 1, "after_region_count": 0, "target_issue_improved": True},
            }

        def fake_damage_audit(**kwargs):
            return {
                "schema": "mac_instrumental_damage_audit.v1",
                "verdict": "fail",
                "failed_region_count": 1,
                "regions": [
                    {
                        "start": 22.17,
                        "end": 23.08,
                        "delta_ratio": 0.445517,
                        "blocker_reasons": ["instrumental_region_overprocessed"],
                    }
                ],
                "blocker_reasons": ["instrumental_region_overprocessed"],
                "delivery_allowed": False,
                "presentation_status": "damage_audit_diagnostic_only",
            }

        def fake_damage_limited_repair(**kwargs):
            limiter_calls.append(kwargs)
            return {
                "schema": "mac_residual_damage_limited_repair.v1",
                "status": "damage_limited_repair_passed",
                "audio": {"vocals": str(limited_v), "instrumental": str(limited_i), "raw_ref": str(limited_raw)},
                "after_score": {
                    "schema": "mac_melband_pitch_guard_score.v1",
                    "verdict": "pass",
                    "regions": [],
                    "total_fail_regions": 0,
                    "selected_region_count": 0,
                    "config": {"window_sec": 0.75, "hop_sec": 0.25},
                },
                "damage_audit": {
                    "schema": "mac_instrumental_damage_audit.v1",
                    "verdict": "pass",
                    "failed_region_count": 0,
                    "blocker_reasons": [],
                    "delivery_allowed": False,
                    "presentation_status": "damage_audit_diagnostic_only",
                },
                "damage_audit_manifest": str(root / "out" / "dlim" / "damage" / "damage.json"),
                "manifest_path": str(root / "out" / "dlim" / "manifest.json"),
                "delivery_allowed": False,
                "listen_pack_created": False,
            }

        original_training = residual_train.run_training
        original_damage = residual_train.run_damage_audit
        original_limiter = getattr(residual_train, "run_damage_limited_repair", None)
        try:
            residual_train.run_training = fake_run_training
            residual_train.run_damage_audit = fake_damage_audit
            residual_train.run_damage_limited_repair = fake_damage_limited_repair
            manifest = residual_train.run_residual_training(
                failed_stage_dir=failed_stage,
                output_dir=root / "out",
                target_gain=4.0,
                epochs=3,
                require_mps=True,
                region_batch={"axis": "fail_mode", "value": "strong_projection_stft", "expected_region_count": 1, "batch_id": "bsp001"},
            )
        finally:
            residual_train.run_training = original_training
            residual_train.run_damage_audit = original_damage
            if original_limiter is None:
                delattr(residual_train, "run_damage_limited_repair")
            else:
                residual_train.run_damage_limited_repair = original_limiter

        assert len(limiter_calls) == 1
        assert limiter_calls[0]["initial_damage_audit"]["verdict"] == "fail"
        assert manifest["status"] == "residual_batch_trained_damage_limited_passed"
        assert manifest["failed_checks"] == []
        assert manifest["damage_audit"]["verdict"] == "pass"
        assert manifest["damage_limited_repair"]["status"] == "damage_limited_repair_passed"
        assert manifest["audio"]["instrumental"] == str(limited_i)
        assert manifest["audio"]["vocals"] == str(limited_v)
        assert manifest["delivery_allowed"] is False
        assert manifest["presentation_status"] == "residual_training_diagnostic_only"


def test_residual_guard_preserves_uncapped_score_provenance_for_batch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        failed_stage = _write_failed_stage(
            root / "stage",
            regions=[
                {"start": 1.0, "end": 1.5, "probe_semitones": 0.0, "fail_mode": "strong_projection_stft", "score": 0.61},
                {"start": 2.0, "end": 2.5, "probe_semitones": -9.0, "fail_mode": "strong_projection_stft", "score": 0.51},
                {"start": 3.0, "end": 3.5, "probe_semitones": -9.0, "fail_mode": "standard_score", "score": 0.49},
            ],
        )
        score_path = failed_stage / "replay" / "score" / "after" / "pitch.json"
        score = json.loads(score_path.read_text(encoding="utf-8"))
        score.update(
            {
                "total_fail_regions": 46,
                "selected_region_count": 16,
                "regions_capped": True,
                "best_probe": {"score": 0.72, "semitones": 0.0},
                "failure_taxonomy": {
                    "by_probe": {"0.0": 39, "-9.0": 4, "-5.0": 2, "-12.0": 1},
                    "by_fail_mode": {"standard_score": 14, "strong_projection_stft": 32},
                },
            }
        )
        score_path.write_text(json.dumps(score), encoding="utf-8")

        result = residual_train.build_residual_guard_from_failed_stage(
            failed_stage_dir=failed_stage,
            output_dir=root / "out",
            region_batch={"axis": "probe", "value": "-9.0", "expected_region_count": 4, "batch_id": "b002"},
        )

        guard_score = result["guard_manifest"]["score"]
        assert [region["start"] for region in guard_score["regions"]] == [2.0, 3.0]
        assert guard_score["total_fail_regions"] == 46
        assert guard_score["selected_region_count"] == 16
        assert guard_score["regions_capped"] is True
        assert guard_score["best_probe"] == {"score": 0.72, "semitones": 0.0}
        assert guard_score["failure_taxonomy"]["by_probe"]["0.0"] == 39
        assert guard_score["batch_region_count"] == 2
        assert guard_score["source_selected_region_count"] == 16
        assert result["source_residual_region_count"] == 3


def test_residual_guard_prefers_uncapped_region_file_for_taxonomy_batch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        failed_stage = _write_failed_stage(
            root / "stage",
            regions=[
                {"start": 1.0, "end": 1.5, "probe_semitones": 0.0, "fail_mode": "strong_projection_stft", "score": 0.61},
            ],
        )
        score_path = failed_stage / "replay" / "score" / "after" / "pitch.json"
        all_regions_path = score_path.parent / "regions.jsonl"
        uncapped_regions = [
            {"start": 1.0, "end": 1.5, "probe_semitones": 0.0, "fail_mode": "strong_projection_stft", "score": 0.61},
            {"start": 2.0, "end": 2.5, "probe_semitones": 0.0, "fail_mode": "standard_score", "score": 0.59},
            {"start": 3.0, "end": 3.5, "probe_semitones": -9.0, "fail_mode": "standard_score", "score": 0.49},
        ]
        all_regions_path.write_text(
            "".join(json.dumps(region) + "\n" for region in uncapped_regions),
            encoding="utf-8",
        )
        score = json.loads(score_path.read_text(encoding="utf-8"))
        score.update(
            {
                "total_fail_regions": 3,
                "selected_region_count": 1,
                "regions_capped": True,
                "all_regions_path": str(all_regions_path),
                "failure_taxonomy": {
                    "by_probe": {"0.0": 2, "-9.0": 1},
                    "by_fail_mode": {"standard_score": 2, "strong_projection_stft": 1},
                },
            }
        )
        score_path.write_text(json.dumps(score), encoding="utf-8")

        result = residual_train.build_residual_guard_from_failed_stage(
            failed_stage_dir=failed_stage,
            output_dir=root / "out",
            region_batch={"axis": "probe", "value": "0.0", "expected_region_count": 2, "batch_id": "b001"},
        )

        guard_score = result["guard_manifest"]["score"]
        assert [region["start"] for region in guard_score["regions"]] == [1.0, 2.0]
        assert result["source_residual_region_count"] == 3
        assert result["residual_region_count"] == 2
        assert guard_score["source_all_regions_path"] == str(all_regions_path)


def test_residual_guard_applies_taxonomy_batch_chunk_window() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        failed_stage = _write_failed_stage(
            root / "stage",
            regions=[
                {"start": 1.0, "end": 1.5, "probe_semitones": 0.0, "fail_mode": "standard_score", "score": 0.61},
                {"start": 2.0, "end": 2.5, "probe_semitones": 0.0, "fail_mode": "standard_score", "score": 0.59},
                {"start": 3.0, "end": 3.5, "probe_semitones": 0.0, "fail_mode": "standard_score", "score": 0.57},
                {"start": 4.0, "end": 4.5, "probe_semitones": 0.0, "fail_mode": "standard_score", "score": 0.55},
            ],
        )
        score_path = failed_stage / "replay" / "score" / "after" / "pitch.json"
        all_regions_path = score_path.parent / "regions.jsonl"
        all_regions_path.write_text(
            "".join(json.dumps(region) + "\n" for region in json.loads(score_path.read_text(encoding="utf-8"))["regions"]),
            encoding="utf-8",
        )
        score = json.loads(score_path.read_text(encoding="utf-8"))
        score.update({"total_fail_regions": 4, "selected_region_count": 4, "all_regions_path": str(all_regions_path)})
        score_path.write_text(json.dumps(score), encoding="utf-8")

        result = residual_train.build_residual_guard_from_failed_stage(
            failed_stage_dir=failed_stage,
            output_dir=root / "out",
            region_batch={
                "axis": "probe",
                "value": "0.0",
                "expected_region_count": 4,
                "batch_id": "b001",
                "chunk_start": 2,
                "chunk_limit": 2,
                "chunk_index": 1,
                "chunk_count": 2,
                "chunk_expected_region_count": 2,
            },
        )

        guard_score = result["guard_manifest"]["score"]
        assert [region["start"] for region in guard_score["regions"]] == [3.0, 4.0]
        assert result["source_residual_region_count"] == 4
        assert result["residual_region_count"] == 2
        assert guard_score["batch_region_count"] == 2


def test_residual_training_blocks_empty_taxonomy_batch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        failed_stage = _write_failed_stage(
            root / "stage",
            regions=[{"start": 1.0, "end": 2.0, "probe_semitones": 0.0, "score": 0.5}],
        )

        manifest = residual_train.run_residual_training(
            failed_stage_dir=failed_stage,
            output_dir=root / "out",
            target_gain=2.5,
            epochs=1,
            require_mps=True,
            region_batch={"axis": "probe", "value": "-12.0", "expected_region_count": 1, "batch_id": "b003"},
        )

        assert manifest["status"] == "blocked_no_matching_residual_batch_regions"
        assert manifest["failed_checks"] == ["no_matching_residual_batch_regions"]
        assert manifest["source_residual_region_count"] == 1
        assert manifest["taxonomy_batch"]["batch_id"] == "b003"
        assert manifest["delivery_allowed"] is False


def test_residual_training_uses_local_replay_manifest_for_copied_absolute_stage_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        failed_stage = _write_failed_stage(
            root / "stage",
            regions=[{"start": 3.0, "end": 4.0, "semitones": -9.0, "score": 0.51}],
        )
        stage_manifest_path = failed_stage / "manifest.json"
        stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
        stage_manifest["replay_manifest"] = "/Users/andrejviluha/aimusic_local_artifacts/mac_api/it6h001/val/h001/replay/manifest.json"
        stage_manifest_path.write_text(json.dumps(stage_manifest), encoding="utf-8")

        result = residual_train.build_residual_guard_from_failed_stage(
            failed_stage_dir=failed_stage,
            output_dir=root / "out",
        )

        assert result["status"] == "ok"
        assert result["guard_manifest"]["source_replay_manifest"] == str(failed_stage / "replay" / "manifest.json")
        assert result["residual_region_count"] == 1


def test_residual_training_rejects_unexpected_copied_absolute_path_tail() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        failed_stage = _write_failed_stage(
            root / "stage",
            regions=[{"start": 3.0, "end": 4.0, "semitones": -9.0, "score": 0.51}],
        )
        wrong_dir = failed_stage / "other"
        wrong_dir.mkdir()
        (wrong_dir / "manifest.json").write_text("{}", encoding="utf-8")
        stage_manifest_path = failed_stage / "manifest.json"
        stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
        stage_manifest["replay_manifest"] = "/Users/andrejviluha/aimusic_local_artifacts/mac_api/it6h001/val/h001/other/manifest.json"
        stage_manifest_path.write_text(json.dumps(stage_manifest), encoding="utf-8")

        try:
            residual_train.build_residual_guard_from_failed_stage(
                failed_stage_dir=failed_stage,
                output_dir=root / "out",
            )
        except FileNotFoundError as exc:
            assert "other/manifest.json" in str(exc)
        else:
            raise AssertionError("unexpected absolute path tail must not be accepted")


def test_residual_training_blocks_invalid_training_delivery_contract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        failed_stage = _write_failed_stage(
            root / "stage",
            regions=[{"start": 1.0, "end": 2.0, "semitones": -12.0, "score": 0.5}],
        )

        def fake_run_training(**kwargs):
            return {
                "status": "trained_repair_passed_pitch_guard",
                "delivery_allowed": True,
                "listen_pack_created": False,
            }

        original = residual_train.run_training
        try:
            residual_train.run_training = fake_run_training
            manifest = residual_train.run_residual_training(
                failed_stage_dir=failed_stage,
                output_dir=root / "out",
                target_gain=2.5,
                epochs=1,
                require_mps=True,
            )
        finally:
            residual_train.run_training = original

        assert manifest["status"] == "blocked_invalid_training_contract"
        assert manifest["failed_checks"] == ["training_delivery_contract_invalid"]
        assert manifest["delivery_allowed"] is False
