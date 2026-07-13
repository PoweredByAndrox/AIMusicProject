from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "build_spectral_minus_matrix.py"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
SPEC = importlib.util.spec_from_file_location("build_spectral_minus_matrix", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TestSpectralMinusMatrix(unittest.TestCase):
    def test_consensus_support_blocks_delta_explained_by_vocal_proxies(self):
        delta = np.ones((3, 4), dtype=np.float32)
        proxies = [np.ones_like(delta) for _ in range(5)]
        support, scales = MODULE.compute_consensus_support(
            delta,
            proxies,
            active_frames=np.ones(4, dtype=bool),
            consensus_quantile=0.5,
            min_proxy_scale=0.25,
            max_proxy_scale=4.0,
        )
        np.testing.assert_allclose(support, 1.0, atol=1e-6)
        self.assertEqual(len(scales), 5)

    def test_consensus_support_allows_delta_not_present_in_vocal_proxies(self):
        delta = np.ones((3, 4), dtype=np.float32)
        proxies = [np.zeros_like(delta) for _ in range(5)]
        support, _ = MODULE.compute_consensus_support(
            delta,
            proxies,
            active_frames=np.ones(4, dtype=bool),
            consensus_quantile=0.5,
            min_proxy_scale=0.25,
            max_proxy_scale=4.0,
        )
        np.testing.assert_allclose(support, 0.0, atol=1e-6)

    def test_median_requires_three_of_five_proxy_supports(self):
        delta = np.ones((1, 2), dtype=np.float32)
        proxies = [np.ones_like(delta) for _ in range(3)] + [np.zeros_like(delta) for _ in range(2)]
        support, _ = MODULE.compute_consensus_support(
            delta,
            proxies,
            active_frames=np.ones(2, dtype=bool),
            consensus_quantile=0.5,
            min_proxy_scale=1.0,
            max_proxy_scale=1.0,
        )
        np.testing.assert_allclose(support, 1.0)

    def test_sample_gate_is_full_at_track_start_and_zero_outside_window(self):
        gate = MODULE.build_sample_gate(
            1000,
            sample_rate=1000,
            windows=[{"window_sec": [0.0, 0.5]}],
            crossfade_ms=80,
        )
        self.assertEqual(float(gate[0]), 1.0)
        self.assertEqual(float(gate[500]), 0.0)
        self.assertEqual(float(gate[-1]), 0.0)

    def test_render_keeps_outside_window_identical(self):
        sample_rate = 4000
        base = np.full((4000, 1), 0.25, dtype=np.float32)
        zero_z = np.zeros((1025, 9), dtype=np.complex64)
        component = {
            "base_z": zero_z,
            "alternate_z": zero_z,
            "delta_z": zero_z,
            "consensus": np.ones_like(zero_z, dtype=np.float32),
        }
        rendered, _ = MODULE.render_spectral_candidate(
            base,
            [component],
            sample_rate=sample_rate,
            windows=[{"window_sec": [0.0, 0.5]}],
            restore_strength=1.0,
            mask_power=1.0,
            crossfade_ms=0,
        )
        np.testing.assert_array_equal(rendered[2000:], base[2000:])


if __name__ == "__main__":
    unittest.main()
