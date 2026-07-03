from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "apply_actual_masked_residue_minus_cleanup.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("apply_actual_masked_residue_minus_cleanup", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_low_priority_windows_require_explicit_flag_and_score() -> None:
    module = _load_module()
    windows = [
        {
            "start_sec": 0.0,
            "end_sec": 1.0,
            "vocal_evidence_status": "low_priority",
            "suspicion_score": 0.15,
        },
        {
            "start_sec": 1.0,
            "end_sec": 2.0,
            "vocal_evidence_status": "low_priority",
            "suspicion_score": 0.05,
        },
    ]

    default_gate = module.build_review_window_gate(
        windows=windows,
        frame_count=200,
        sample_rate=100,
        smoothing_sec=0.0,
    )
    low_priority_gate = module.build_review_window_gate(
        windows=windows,
        frame_count=200,
        sample_rate=100,
        smoothing_sec=0.0,
        include_low_priority_windows=True,
        min_low_priority_score=0.12,
    )

    assert float(default_gate.max()) == 0.0
    assert float(low_priority_gate[:100].max()) == 1.0
    assert float(low_priority_gate[100:].max()) == 0.0


def test_base_overlap_power_reduces_broad_residue_mask() -> None:
    module = _load_module()
    sample_rate = 8000
    t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
    vocal = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    base = (vocal + 0.8 * np.sin(2.0 * np.pi * 880.0 * t)).astype(np.float32)
    review_gate = np.ones(sample_rate, dtype=np.float32)

    _, ungated = module.build_actual_residue_gain_mask(
        base_minus=base,
        primary_vocals=vocal,
        review_gate=review_gate,
        sample_rate=sample_rate,
        strength=0.1,
        mask_power=1.0,
        base_overlap_power=0.0,
    )
    _, overlap_gated = module.build_actual_residue_gain_mask(
        base_minus=base,
        primary_vocals=vocal,
        review_gate=review_gate,
        sample_rate=sample_rate,
        strength=0.1,
        mask_power=1.0,
        base_overlap_power=1.0,
    )

    assert overlap_gated["mean_residue_mask"] < ungated["mean_residue_mask"]
