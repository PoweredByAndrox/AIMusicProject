from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "apply_windowed_hybrid_minus_cleanup.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("apply_windowed_hybrid_minus_cleanup", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frame_score_gate_recovers_sparse_max_percentile_gate() -> None:
    module = _load_module()
    frame_score = np.zeros(64, dtype=np.float32)
    frame_score[[10, 24, 48]] = 1.0

    gate, metrics = module._frame_score_gate(
        frame_score,
        score_percentile=100.0,
        smooth_width=9,
        apply_threshold=0.05,
    )

    assert metrics["gate_mode"] in {"continuous_max_guard", "continuous_sparse_recovery"}
    assert metrics["frame_score_max"] == 1.0
    assert float(gate.max()) > 0.0
    assert float(gate.mean()) > 0.0


def test_frame_score_gate_stays_zero_without_positive_scores() -> None:
    module = _load_module()
    frame_score = np.zeros(32, dtype=np.float32)

    gate, metrics = module._frame_score_gate(
        frame_score,
        score_percentile=70.0,
        smooth_width=9,
        apply_threshold=0.14,
    )

    assert metrics["gate_mode"] == "empty_score"
    assert float(gate.max()) == 0.0
