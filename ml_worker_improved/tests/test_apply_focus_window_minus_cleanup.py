from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "apply_focus_window_minus_cleanup.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("apply_focus_window_minus_cleanup", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FocusWindowMinusCleanupTests(unittest.TestCase):
    def test_focus_gate_uses_only_requested_windows(self) -> None:
        module = _load_module()

        gate = module.build_focus_window_gate(
            focus_windows=[{"start_sec": 1.0, "end_sec": 2.0}],
            frame_count=4000,
            sample_rate=1000,
            smoothing_sec=0.0,
        )

        self.assertEqual(float(np.max(gate[:1000])), 0.0)
        self.assertEqual(float(np.min(gate[1000:2000])), 1.0)
        self.assertEqual(float(np.max(gate[2000:])), 0.0)

    def test_focus_cleanup_changes_focus_region_more_than_outside(self) -> None:
        module = _load_module()
        sample_rate = 2000
        frames = sample_rate * 3
        t = np.arange(frames, dtype=np.float32) / float(sample_rate)
        vocal = np.sin(2.0 * np.pi * 300.0 * t).astype(np.float32)
        base = np.stack([vocal, vocal], axis=1)
        primary = np.stack([vocal, vocal], axis=1)
        secondary = np.zeros_like(primary)

        refined, report = module.apply_focus_window_cleanup(
            base_minus=base,
            primary_vocals=primary,
            secondary_vocals=secondary,
            focus_windows=[{"start_sec": 1.0, "end_sec": 2.0}],
            sample_rate=sample_rate,
            strength=0.25,
            secondary_weight=0.0,
            mask_power=1.0,
            smoothing_sec=0.0,
        )

        focus_delta = float(
            np.sqrt(
                np.mean(
                    (
                        refined[int(sample_rate * 1.25) : int(sample_rate * 1.75)]
                        - base[int(sample_rate * 1.25) : int(sample_rate * 1.75)]
                    )
                    ** 2
                )
            )
        )
        outside_delta = float(
            np.sqrt(
                np.mean(
                    (
                        refined[int(sample_rate * 2.5) : int(sample_rate * 2.9)]
                        - base[int(sample_rate * 2.5) : int(sample_rate * 2.9)]
                    )
                    ** 2
                )
            )
        )
        self.assertEqual(report["cleanup_status"], "applied_focus_window_minus_cleanup")
        self.assertGreater(focus_delta, outside_delta * 10.0)


if __name__ == "__main__":
    unittest.main()
