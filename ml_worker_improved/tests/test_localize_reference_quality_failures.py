from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
ROOT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = TOOLS_ROOT / "localize_reference_quality_failures.py"


def _load_module():
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    spec = importlib.util.spec_from_file_location("localize_reference_quality_failures", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_near_silent_reference_residual_floor_is_not_active_vocal_defect() -> None:
    module = _load_module()
    sample_rate = 100
    reference = np.zeros((500, 2), dtype=np.float32)
    candidate = np.full((500, 2), 1.0e-6, dtype=np.float32)

    window = module.analyze_quality_window(
        reference_audio=reference,
        candidate_audio=candidate,
        sample_rate=sample_rate,
        start_frame=0,
        end_frame=500,
        config=module.LocalizationConfig(reference_silence_db=-80.0, candidate_residual_floor_db=-90.0),
        band_edges_hz=[0.0, 20.0, 50.0],
    )

    assert window["window_class"] == "near_silent_reference"
    assert "near_silent_reference_residual_floor" in window["issue_labels"]
    assert "active_low_sdr" not in window["issue_labels"]
    assert window["active_vocal_defect"] is False


def test_active_band_mismatch_reports_dominant_high_band_error() -> None:
    module = _load_module()
    sample_rate = 128
    frames = 128
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    low = (0.2 * np.sin(2.0 * np.pi * 6.0 * t)).astype(np.float32)[:, None]
    high = (0.2 * np.sin(2.0 * np.pi * 30.0 * t)).astype(np.float32)[:, None]
    reference = np.repeat(low + high, 2, axis=1)
    candidate = np.repeat(low, 2, axis=1)

    window = module.analyze_quality_window(
        reference_audio=reference,
        candidate_audio=candidate,
        sample_rate=sample_rate,
        start_frame=0,
        end_frame=frames,
        config=module.LocalizationConfig(min_sdr_db=20.0, min_si_sdr_db=20.0),
        band_edges_hz=[0.0, 12.0, 64.0],
    )

    assert window["window_class"] == "active_reference"
    assert "active_low_sdr" in window["issue_labels"]
    assert window["active_vocal_defect"] is True
    assert window["top_error_bands"][0]["band_hz"] == "12.0-64.0"


def test_localization_summary_separates_active_defects_from_silent_tail() -> None:
    module = _load_module()
    sample_rate = 100
    active_reference = np.ones((500, 2), dtype=np.float32) * 0.2
    active_candidate = active_reference * 0.7
    silent_reference = np.zeros((500, 2), dtype=np.float32)
    silent_candidate = np.full((500, 2), 1.0e-6, dtype=np.float32)
    reference = np.concatenate([active_reference, silent_reference], axis=0)
    candidate = np.concatenate([active_candidate, silent_candidate], axis=0)

    report = module.localize_reference_quality_failures_from_audio(
        reference_audio=reference,
        candidate_audio=candidate,
        sample_rate=sample_rate,
        config=module.LocalizationConfig(window_sec=5.0, reference_silence_db=-80.0),
        band_edges_hz=[0.0, 20.0, 50.0],
    )

    assert report["summary"]["window_count"] == 2
    assert report["summary"]["active_vocal_defect_window_count"] == 1
    assert report["summary"]["near_silent_reference_window_count"] == 1
    assert report["top_active_defect_windows"][0]["start_sec"] == 0.0
    assert report["top_silent_residual_windows"][0]["start_sec"] == 5.0


def test_cli_summary_omits_full_window_payload() -> None:
    module = _load_module()
    full_report = {
        "report_path": "/tmp/report.json",
        "sha256": "abc",
        "summary": {"window_count": 2},
        "top_active_defect_windows": [
            {
                "start_sec": 0.0,
                "end_sec": 5.0,
                "issue_labels": ["active_low_sdr"],
                "metrics": {"sdr_db": 10.0, "si_sdr_db": 9.0},
                "top_error_bands": [{"band_hz": "1000.0-2000.0", "error_share": 0.5}],
            }
        ],
        "top_silent_residual_windows": [
            {
                "start_sec": 5.0,
                "end_sec": 10.0,
                "issue_labels": ["near_silent_reference_residual_floor"],
                "metrics": {"candidate_rms_db": -110.0},
                "top_error_bands": [{"band_hz": "16000.0-22050.0", "error_share": 0.8}],
            }
        ],
        "windows": [{"large": "payload"}],
        "repair_rule_recommendations": [{"rule_id": "silence_aware_window_gate"}],
    }

    compact = module.build_cli_summary(full_report)

    assert compact["report_path"] == "/tmp/report.json"
    assert "windows" not in compact
    assert compact["top_active_defect_windows"][0]["top_error_bands"] == ["1000.0-2000.0"]
    assert compact["top_silent_residual_windows"][0]["top_error_bands"] == ["16000.0-22050.0"]


def test_localization_writes_full_report_from_manifests(tmp_path: Path) -> None:
    module = _load_module()
    manifests = module.quality_gate.create_synthetic_fixture(
        tmp_path / "fixture",
        sample_rate=8000,
        duration_seconds=6.0,
    )

    summary = module.localize_reference_quality_failures(
        ground_truth_manifest=manifests["ground_truth"],
        candidate_manifest=manifests["synthetic_degraded"],
        output_dir=tmp_path / "localization",
        stem_name="vocals",
        config=module.LocalizationConfig(window_sec=3.0),
        band_edges_hz=[0.0, 1000.0, 4000.0],
    )

    report_path = Path(summary["report_path"])
    report = __import__("json").loads(report_path.read_text(encoding="utf-8"))
    compact = module.build_cli_summary(summary)

    assert report_path.exists()
    assert summary["sha256"] == module.sha256_file(report_path)
    assert report["schema_version"] == 1
    assert report["inputs"]["stem"] == "vocals"
    assert report["config"]["window_sec"] == 3.0
    assert report["summary"]["window_count"] == 2
    assert "windows" in report
    assert "windows" not in compact
