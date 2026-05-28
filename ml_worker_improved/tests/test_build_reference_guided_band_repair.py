from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
ROOT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = TOOLS_ROOT / "build_reference_guided_band_repair.py"


def _load_module():
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    spec = importlib.util.spec_from_file_location("build_reference_guided_band_repair", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_active_band_repair_improves_missing_high_band() -> None:
    module = _load_module()
    sample_rate = 128
    frames = 128
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    low = (0.2 * np.sin(2.0 * np.pi * 6.0 * t)).astype(np.float32)[:, None]
    high = (0.2 * np.sin(2.0 * np.pi * 30.0 * t)).astype(np.float32)[:, None]
    reference = np.repeat(low + high, 2, axis=1)
    candidate = np.repeat(low, 2, axis=1)
    localization_report = {
        "windows": [
            {
                "start_frame": 0,
                "end_frame": frames,
                "window_class": "active_reference",
                "active_vocal_defect": True,
                "issue_labels": ["active_low_sdr", "active_low_si_sdr"],
                "top_error_bands": [{"low_hz": 12.0, "high_hz": 64.0, "error_share": 1.0}],
            }
        ]
    }

    repaired, diagnostics = module.repair_active_vocal_bands(
        reference_audio=reference,
        candidate_audio=candidate,
        sample_rate=sample_rate,
        localization_report=localization_report,
        config=module.BandRepairConfig(repair_strength=1.0, edge_fade_sec=0.0),
    )

    candidate_mse = float(np.mean((candidate - reference) ** 2))
    repaired_mse = float(np.mean((repaired - reference) ** 2))
    assert repaired_mse < candidate_mse * 0.05
    assert diagnostics["summary"]["active_repaired_window_count"] == 1


def test_near_silent_residual_floor_window_is_not_repaired() -> None:
    module = _load_module()
    reference = np.zeros((100, 2), dtype=np.float32)
    candidate = np.full((100, 2), 1.0e-6, dtype=np.float32)
    localization_report = {
        "windows": [
            {
                "start_frame": 0,
                "end_frame": 100,
                "window_class": "near_silent_reference",
                "active_vocal_defect": False,
                "issue_labels": ["near_silent_reference_residual_floor"],
                "top_error_bands": [{"low_hz": 0.0, "high_hz": 50.0, "error_share": 1.0}],
            }
        ]
    }

    repaired, diagnostics = module.repair_active_vocal_bands(
        reference_audio=reference,
        candidate_audio=candidate,
        sample_rate=100,
        localization_report=localization_report,
        config=module.BandRepairConfig(repair_strength=1.0, edge_fade_sec=0.0),
    )

    assert np.allclose(repaired, candidate)
    assert diagnostics["summary"]["active_repaired_window_count"] == 0


def test_repair_strength_scales_linearly_with_error_share() -> None:
    module = _load_module()
    reference = np.ones((64, 2), dtype=np.float32)
    candidate = np.zeros((64, 2), dtype=np.float32)

    _, repaired_bands = module.repair_window_frequency_bands(
        reference_window=reference,
        candidate_window=candidate,
        sample_rate=128,
        top_error_bands=[{"low_hz": 0.0, "high_hz": 64.0, "error_share": 0.05}],
        config=module.BandRepairConfig(repair_strength=0.8, min_error_share=0.05),
    )

    assert repaired_bands[0]["strength"] == 0.04


def test_edge_fade_has_nonzero_boundaries_when_enabled() -> None:
    module = _load_module()
    envelope = module.edge_fade_envelope(frame_count=10, fade_frames=3)

    assert 0.0 < float(envelope[0, 0]) < 1.0
    assert 0.0 < float(envelope[-1, 0]) < 1.0
    assert float(envelope[3, 0]) == 1.0


def test_clipping_diagnostics_report_pre_clip_peak_and_ratio() -> None:
    module = _load_module()
    summary = module.clipping_summary(np.array([[1.2, -1.1], [0.5, 0.0]], dtype=np.float32))

    assert summary["peak_abs"] == 1.2
    assert summary["clipped_sample_ratio"] == 0.5


def test_invalid_top_error_band_fails_with_context() -> None:
    module = _load_module()
    reference = np.ones((64, 2), dtype=np.float32)
    candidate = np.zeros((64, 2), dtype=np.float32)

    with pytest.raises(RuntimeError, match="Invalid top_error_bands"):
        module.repair_window_frequency_bands(
            reference_window=reference,
            candidate_window=candidate,
            sample_rate=128,
            top_error_bands=[{"low_hz": 0.0, "error_share": 0.5}],
            config=module.BandRepairConfig(),
        )


def test_non_finite_top_error_band_fails_with_context() -> None:
    module = _load_module()
    reference = np.ones((64, 2), dtype=np.float32)
    candidate = np.zeros((64, 2), dtype=np.float32)

    with pytest.raises(RuntimeError, match="Invalid top_error_bands"):
        module.repair_window_frequency_bands(
            reference_window=reference,
            candidate_window=candidate,
            sample_rate=128,
            top_error_bands=[{"low_hz": 0.0, "high_hz": float("inf"), "error_share": float("nan")}],
            config=module.BandRepairConfig(),
        )


def test_invalid_window_frame_bounds_fail_with_context() -> None:
    module = _load_module()
    reference = np.ones((64, 2), dtype=np.float32)
    candidate = np.zeros((64, 2), dtype=np.float32)
    localization_report = {
        "windows": [
            {
                "start_frame": "bad",
                "end_frame": 64,
                "window_class": "active_reference",
                "active_vocal_defect": True,
                "top_error_bands": [{"low_hz": 0.0, "high_hz": 64.0, "error_share": 0.5}],
            }
        ]
    }

    with pytest.raises(RuntimeError, match="Invalid localization window frame bounds"):
        module.repair_active_vocal_bands(
            reference_audio=reference,
            candidate_audio=candidate,
            sample_rate=128,
            localization_report=localization_report,
            config=module.BandRepairConfig(),
        )


def test_out_of_range_window_frame_bounds_fail_with_context() -> None:
    module = _load_module()
    reference = np.ones((64, 2), dtype=np.float32)
    candidate = np.zeros((64, 2), dtype=np.float32)
    localization_report = {
        "windows": [
            {
                "start_frame": 60,
                "end_frame": 80,
                "window_class": "active_reference",
                "active_vocal_defect": True,
                "top_error_bands": [{"low_hz": 0.0, "high_hz": 64.0, "error_share": 0.5}],
            }
        ]
    }

    with pytest.raises(RuntimeError, match="outside audio bounds"):
        module.repair_active_vocal_bands(
            reference_audio=reference,
            candidate_audio=candidate,
            sample_rate=128,
            localization_report=localization_report,
            config=module.BandRepairConfig(),
        )


def test_non_object_localization_window_fails_with_context() -> None:
    module = _load_module()
    reference = np.ones((64, 2), dtype=np.float32)
    candidate = np.zeros((64, 2), dtype=np.float32)

    with pytest.raises(RuntimeError, match="Invalid localization window"):
        module.repair_active_vocal_bands(
            reference_audio=reference,
            candidate_audio=candidate,
            sample_rate=128,
            localization_report={"windows": ["bad-window"]},
            config=module.BandRepairConfig(),
        )


def test_non_object_localization_report_fails_with_context() -> None:
    module = _load_module()
    reference = np.ones((64, 2), dtype=np.float32)
    candidate = np.zeros((64, 2), dtype=np.float32)

    with pytest.raises(RuntimeError, match="Localization report must be an object"):
        module.repair_active_vocal_bands(
            reference_audio=reference,
            candidate_audio=candidate,
            sample_rate=128,
            localization_report=[],
            config=module.BandRepairConfig(),
        )


def test_candidate_source_must_match_ground_truth_mix(tmp_path: Path) -> None:
    module = _load_module()
    ground_truth_manifest = tmp_path / "gt" / "ground_truth_manifest.json"
    candidate_manifest = tmp_path / "candidate" / "stem_manifest.json"
    ground_truth_manifest.parent.mkdir()
    candidate_manifest.parent.mkdir()
    ground_truth_manifest.write_text(json.dumps({"mix": {"path": "source_mix.flac"}}), encoding="utf-8")
    candidate_manifest.write_text(json.dumps({"source_audio_path": "/other/source_mix.flac"}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not match ground-truth mix"):
        module.validate_candidate_source_matches_ground_truth(
            ground_truth_manifest=ground_truth_manifest,
            candidate_manifest=candidate_manifest,
        )


def test_non_object_manifest_fails_with_context(tmp_path: Path) -> None:
    module = _load_module()
    ground_truth_manifest = tmp_path / "gt" / "ground_truth_manifest.json"
    candidate_manifest = tmp_path / "candidate" / "stem_manifest.json"
    ground_truth_manifest.parent.mkdir()
    candidate_manifest.parent.mkdir()
    ground_truth_manifest.write_text("[]", encoding="utf-8")
    candidate_manifest.write_text(json.dumps({"source_audio_path": "/tracks/source_mix.flac"}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Manifest must be an object"):
        module.validate_candidate_source_matches_ground_truth(
            ground_truth_manifest=ground_truth_manifest,
            candidate_manifest=candidate_manifest,
        )


def test_invalid_localization_report_does_not_write_outputs(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 128
    frames = 64
    audio = np.zeros((frames, 2), dtype=np.float32)
    source_path = tmp_path / "source_mix.flac"
    reference_path = tmp_path / "reference_vocals.flac"
    candidate_path = tmp_path / "candidate_vocals.flac"
    module.sf.write(str(source_path), audio, sample_rate)
    module.sf.write(str(reference_path), audio, sample_rate)
    module.sf.write(str(candidate_path), audio, sample_rate)
    ground_truth_manifest = tmp_path / "ground_truth_manifest.json"
    candidate_manifest = tmp_path / "candidate_manifest.json"
    localization_report = tmp_path / "bad_localization.json"
    output_dir = tmp_path / "output"
    ground_truth_manifest.write_text(
        json.dumps(
            {
                "ground_truth_contract": {
                    "verified": True,
                    "verification_method": "test_fixture",
                    "source": "unit_test",
                },
                "mix": {"path": str(source_path)},
                "stems": {"vocals": {"path": str(reference_path)}},
            }
        ),
        encoding="utf-8",
    )
    candidate_manifest.write_text(
        json.dumps({"source_audio_path": str(source_path), "stems": {"vocals": {"path": str(candidate_path)}}}),
        encoding="utf-8",
    )
    localization_report.write_text("[]", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Localization report must be an object"):
        module.build_reference_guided_band_repair(
            ground_truth_manifest=ground_truth_manifest,
            candidate_manifest=candidate_manifest,
            localization_report=localization_report,
            output_dir=output_dir,
            stem_name="vocals",
            config=module.BandRepairConfig(),
        )

    assert not (output_dir / "stem_manifest.json").exists()
    assert not (output_dir / "band_repair_diagnostics.json").exists()
    assert not (output_dir / "vocals.flac").exists()
    assert not (output_dir / "instrumental.flac").exists()


def test_band_repair_manifest_is_reference_assisted_not_no_reference(tmp_path: Path) -> None:
    module = _load_module()
    vocals_path = tmp_path / "vocals.flac"
    instrumental_path = tmp_path / "instrumental.flac"
    diagnostics_path = tmp_path / "band_repair_diagnostics.json"
    vocals_path.write_text("stub vocals", encoding="utf-8")
    instrumental_path.write_text("stub instrumental", encoding="utf-8")
    diagnostics_path.write_text(json.dumps({"summary": {"active_repaired_window_count": 1}}), encoding="utf-8")

    manifest = module.build_band_repair_manifest(
        source_audio_path=Path("/tracks/source.flac"),
        ground_truth_manifest=Path("/refs/ground_truth_manifest.json"),
        candidate_manifest=Path("/runs/base/stem_manifest.json"),
        localization_report=Path("/runs/localization/report.json"),
        vocals_path=vocals_path,
        instrumental_path=instrumental_path,
        diagnostics_path=diagnostics_path,
        sample_rate=44100,
        frames=44100,
        elapsed_sec=1.0,
    )

    assert manifest["model_id"] == "reference_guided_band_repair"
    assert manifest["contract"]["benchmark_compatible"] is True
    assert manifest["contract"]["target_reference_assisted"] is True
    assert manifest["contract"]["delivery_allowed_without_reference"] is False
    assert manifest["contract"]["fallback_used"] is False
    assert manifest["reference_assistance"]["not_a_no_reference_production_route"] is True
