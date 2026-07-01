from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from tools import run_mac_mel_refiner_fulltrack_stage as stage


class _FakeMpsDevice:
    type = "mps"

    def __str__(self) -> str:
        return "mps"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _promoted_benchmark(path: Path, selected_checkpoint_id: str = "g1.5") -> None:
    _write_json(
        path,
        {
            "schema": "mac_mel_refiner_benchmark.v1",
            "outcome": "promote_to_next_diagnostic_stage",
            "selected_checkpoint_id": selected_checkpoint_id,
            "delivery_allowed": False,
            "listen_pack_created": False,
            "resolved_device": "mps",
            "failure_taxonomy_summary": {
                "schema": "mac_mel_refiner_benchmark_taxonomy_summary.v1",
                "delivery_allowed": False,
                "total_before_regions": 4,
                "total_after_regions": 0,
            },
        },
    )


def test_fulltrack_stage_requires_promoted_benchmark() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        benchmark_manifest = root / "bench.json"
        _write_json(
            benchmark_manifest,
            {
                "outcome": "reject_as_overfit",
                "selected_checkpoint_id": None,
                "delivery_allowed": False,
                "listen_pack_created": False,
                "resolved_device": "mps",
            },
        )

        manifest = stage.run_fulltrack_diagnostic_stage(
            benchmark_manifest_path=benchmark_manifest,
            mel_guard_dir=root / "guard",
            checkpoint_root=root / "ckpt",
            output_dir=root / "out",
            require_mps=True,
        )

        assert manifest["status"] == "blocked_benchmark_not_promoted"
        assert manifest["delivery_allowed"] is False
        assert manifest["listen_pack_created"] is False


def test_fulltrack_stage_requires_benchmark_failure_taxonomy() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        benchmark_manifest = root / "bench.json"
        _write_json(
            benchmark_manifest,
            {
                "schema": "mac_mel_refiner_benchmark.v1",
                "outcome": "promote_to_next_diagnostic_stage",
                "selected_checkpoint_id": "g1.5",
                "delivery_allowed": False,
                "listen_pack_created": False,
                "resolved_device": "mps",
            },
        )

        manifest = stage.run_fulltrack_diagnostic_stage(
            benchmark_manifest_path=benchmark_manifest,
            mel_guard_dir=root / "guard",
            checkpoint_root=root / "ckpt",
            output_dir=root / "out",
            require_mps=False,
        )

        assert manifest["status"] == "blocked_benchmark_missing_failure_taxonomy"
        assert manifest["failed_checks"] == ["benchmark_missing_failure_taxonomy_summary"]
        assert manifest["delivery_allowed"] is False


def test_fulltrack_stage_uses_selected_checkpoint_and_never_delivers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        benchmark_manifest = root / "bench.json"
        _promoted_benchmark(benchmark_manifest)
        checkpoint = root / "ckpt" / "g1.5" / "model.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        calls: list[Path] = []

        def fake_replay(**kwargs):
            calls.append(Path(kwargs["checkpoint_path"]))
            replay_out = Path(kwargs["output_dir"])
            return {
                "schema": "mac_melband_refiner_replay.v1",
                "status": "repaired_passed_pitch_guard",
                "resolved_device": "mps",
                "before_score": {
                    "verdict": "fail",
                    "regions": [{"start": 0.0, "end": 1.0}],
                    "total_fail_regions": 2,
                    "selected_region_count": 1,
                    "failure_taxonomy": {
                        "total_fail_regions": 2,
                        "selected_region_count": 1,
                        "regions_capped": True,
                        "by_fail_mode": {"standard_score": 2},
                        "by_probe": {"-12.0": 2},
                    },
                },
                "after_score": {
                    "verdict": "pass",
                    "regions": [],
                    "total_fail_regions": 0,
                    "selected_region_count": 0,
                    "failure_taxonomy": {
                        "total_fail_regions": 0,
                        "selected_region_count": 0,
                        "regions_capped": False,
                        "by_fail_mode": {},
                        "by_probe": {},
                    },
                },
                "applied_regions": [{"start": 0.0, "end": 1.0}],
                "metrics": {
                    "target_issue_improved": True,
                    "original_reconstruction_rms": 0.001,
                    "reconstruction_rms": 0.001,
                    "instrumental_delta_rms": 0.02,
                },
                "audio": {
                    "vocals": str(replay_out / "audio" / "v.flac"),
                    "instrumental": str(replay_out / "audio" / "i.flac"),
                    "raw_ref": str(replay_out / "audio" / "raw.flac"),
                },
                "delivery_allowed": False,
                "listen_pack_created": False,
            }

        def fake_read_audio(path: Path):
            frames = 1024
            left = np.linspace(-0.05, 0.05, frames, dtype=np.float32)
            right = np.linspace(0.05, -0.05, frames, dtype=np.float32)
            return np.stack([left, right], axis=1), 44100

        original_replay = stage.run_refiner_replay
        original_read_audio = stage.read_audio_stereo
        original_resolve = stage.resolve_region_repair_training_device
        try:
            stage.run_refiner_replay = fake_replay
            stage.read_audio_stereo = fake_read_audio
            stage.resolve_region_repair_training_device = lambda _device: _FakeMpsDevice()
            manifest = stage.run_fulltrack_diagnostic_stage(
                benchmark_manifest_path=benchmark_manifest,
                mel_guard_dir=root / "guard",
                checkpoint_root=root / "ckpt",
                output_dir=root / "out",
                require_mps=True,
            )
        finally:
            stage.run_refiner_replay = original_replay
            stage.read_audio_stereo = original_read_audio
            stage.resolve_region_repair_training_device = original_resolve

        assert calls == [checkpoint]
        assert manifest["status"] == "diagnostic_passed"
        assert manifest["selected_checkpoint_id"] == "g1.5"
        assert manifest["delivery_allowed"] is False
        assert manifest["listen_pack_created"] is False
        assert manifest["presentation_status"] == "fulltrack_diagnostic_only"
        assert manifest["failure_taxonomy_summary"]["schema"] == "mac_mel_refiner_fulltrack_taxonomy_summary.v1"
        assert manifest["failure_taxonomy_summary"]["before_region_count"] == 2
        assert manifest["failure_taxonomy_summary"]["after_region_count"] == 0
        assert manifest["failure_taxonomy_summary"]["coverage_status"] == "fulltrack_clear"


def test_fulltrack_stage_blocks_missing_selected_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        benchmark_manifest = root / "bench.json"
        _promoted_benchmark(benchmark_manifest, selected_checkpoint_id="g2.0")

        original_resolve = stage.resolve_region_repair_training_device
        try:
            stage.resolve_region_repair_training_device = lambda _device: _FakeMpsDevice()
            manifest = stage.run_fulltrack_diagnostic_stage(
                benchmark_manifest_path=benchmark_manifest,
                mel_guard_dir=root / "guard",
                checkpoint_root=root / "ckpt",
                output_dir=root / "out",
                require_mps=True,
            )
        finally:
            stage.resolve_region_repair_training_device = original_resolve

        assert manifest["status"] == "blocked_missing_checkpoint"
        assert manifest["selected_checkpoint_id"] == "g2.0"
        assert manifest["delivery_allowed"] is False


def test_fulltrack_stage_blocks_when_replay_pitch_guard_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        benchmark_manifest = root / "bench.json"
        _promoted_benchmark(benchmark_manifest)
        checkpoint = root / "ckpt" / "g1.5" / "model.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")

        def fake_replay(**kwargs):
            return {
                "status": "repaired_but_still_blocked",
                "resolved_device": "mps",
                "after_score": {"verdict": "fail", "regions": [{"start": 0.0, "end": 1.0}]},
                "metrics": {"target_issue_improved": False},
                "audio": {},
                "delivery_allowed": False,
                "listen_pack_created": False,
            }

        original_replay = stage.run_refiner_replay
        original_resolve = stage.resolve_region_repair_training_device
        try:
            stage.run_refiner_replay = fake_replay
            stage.resolve_region_repair_training_device = lambda _device: _FakeMpsDevice()
            manifest = stage.run_fulltrack_diagnostic_stage(
                benchmark_manifest_path=benchmark_manifest,
                mel_guard_dir=root / "guard",
                checkpoint_root=root / "ckpt",
                output_dir=root / "out",
                require_mps=True,
            )
        finally:
            stage.run_refiner_replay = original_replay
            stage.resolve_region_repair_training_device = original_resolve

        assert manifest["status"] == "diagnostic_blocked_pitch_guard"
        assert manifest["delivery_allowed"] is False
        assert manifest["failed_checks"] == ["pitch_guard_failed"]


def test_fulltrack_stage_reports_remaining_failure_taxonomy() -> None:
    replay_manifest = {
        "before_score": {
            "total_fail_regions": 69,
            "selected_region_count": 16,
            "regions_capped": True,
            "failure_taxonomy": {
                "total_fail_regions": 69,
                "selected_region_count": 16,
                "regions_capped": True,
                "by_fail_mode": {"standard_score": 35, "strong_projection_stft": 34},
                "by_probe": {"-12.0": 20, "0.0": 49},
            },
        },
        "after_score": {
            "total_fail_regions": 46,
            "selected_region_count": 16,
            "regions_capped": True,
            "failure_taxonomy": {
                "total_fail_regions": 46,
                "selected_region_count": 16,
                "regions_capped": True,
                "by_fail_mode": {"standard_score": 14, "strong_projection_stft": 32},
                "by_probe": {"-12.0": 10, "0.0": 36},
            },
        },
        "applied_regions": [{"start": 0.0, "end": 1.0}] * 16,
    }

    summary = stage._fulltrack_failure_taxonomy_summary(replay_manifest, max_regions=16)

    assert summary["before_region_count"] == 69
    assert summary["after_region_count"] == 46
    assert summary["remaining_failure_count"] == 46
    assert summary["coverage_status"] == "fulltrack_remaining_failures"
    assert summary["after_failure_taxonomy"]["by_fail_mode"] == {"standard_score": 14, "strong_projection_stft": 32}
