from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "apply_actual_masked_residue_minus_cleanup.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "apply_actual_masked_residue_minus_cleanup", MODULE_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ActualMaskedResidueMinusCleanupTests(unittest.TestCase):
    def test_review_gate_excludes_projection_only_windows(self) -> None:
        module = _load_module()

        gate = module.build_review_window_gate(
            windows=[
                {
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                    "vocal_evidence_status": "rejected_as_projection_only",
                },
                {
                    "start_sec": 2.0,
                    "end_sec": 3.0,
                    "vocal_evidence_status": "needs_human_review",
                },
            ],
            frame_count=4000,
            sample_rate=1000,
            smoothing_sec=0.0,
        )

        self.assertEqual(float(np.max(gate[0:1000])), 0.0)
        self.assertEqual(float(np.min(gate[2000:3000])), 1.0)

    def test_review_gate_ignores_low_priority_windows(self) -> None:
        module = _load_module()

        gate = module.build_review_window_gate(
            windows=[
                {
                    "start_sec": 1.0,
                    "end_sec": 2.0,
                    "vocal_evidence_status": "low_priority",
                }
            ],
            frame_count=3000,
            sample_rate=1000,
            smoothing_sec=0.0,
        )

        self.assertEqual(float(np.max(gate)), 0.0)


if __name__ == "__main__":
    unittest.main()
