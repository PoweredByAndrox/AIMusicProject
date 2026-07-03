from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


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

    def test_main_writes_role_explicit_candidate_artifact_names(self) -> None:
        module = _load_module()
        original_guard = module.require_docker_runtime
        module.require_docker_runtime = lambda **_kwargs: None
        try:
            input_dir = self.temp_path / "in"
            output_dir = self.temp_path / "out"
            listen_dir = self.temp_path / "listen"
            for name in ("raw", "base", "guide", "primary", "secondary"):
                self._write_audio(input_dir / f"{name}.flac")
            focus_jsonl = self.temp_path / "focus.jsonl"
            focus_jsonl.write_text(
                json.dumps(
                    {
                        "track_id": "track-a",
                        "candidate_id": "cand-a",
                        "critic_decision": "blocked_hard_case",
                        "window_start_sec": 0.0,
                        "window_end_sec": 0.4,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            module.main(
                [
                    str(input_dir / "raw.flac"),
                    str(input_dir / "base.flac"),
                    str(input_dir / "guide.flac"),
                    str(input_dir / "primary.flac"),
                    str(input_dir / "secondary.flac"),
                    str(output_dir),
                    "--listen-dir",
                    str(listen_dir),
                    "--focus-windows-jsonl",
                    str(focus_jsonl),
                    "--focus-track-id",
                    "track-a",
                    "--focus-candidate-id",
                    "cand-a",
                ]
            )
        finally:
            module.require_docker_runtime = original_guard

        manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue((output_dir / "raw_mix.flac").is_file())
        self.assertTrue((output_dir / "base_product_minus.flac").is_file())
        self.assertTrue((output_dir / "guide_product_minus.flac").is_file())
        self.assertTrue((output_dir / "focus_product_minus.flac").is_file())
        self.assertFalse((output_dir / "raw_ref.flac").exists())
        self.assertFalse((output_dir / "minus_base.flac").exists())
        self.assertFalse((output_dir / "minus_guide.flac").exists())
        self.assertFalse((output_dir / "minus_focus_window.flac").exists())
        self.assertEqual(manifest["stem_type"], "minus")
        self.assertEqual(manifest["artifact_role"], "candidate_product_minus")
        self.assertEqual(manifest["candidate_product_artifact_key"], "focus_product_minus_flac")
        self.assertTrue(manifest["artifacts"]["focus_product_minus_flac"].endswith("focus_product_minus.flac"))
        self.assertTrue(manifest["artifacts"]["minus_focus_window_cleanup_flac"].endswith("focus_product_minus.flac"))
        self.assertTrue((listen_dir / "00_raw_mix.flac").is_file())
        self.assertTrue((listen_dir / "01_base_product_minus.flac").is_file())
        self.assertTrue((listen_dir / "02_candidate_product_minus.flac").is_file())

    def _write_audio(self, path: Path, sample_rate: int = 12000) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frames = sample_rate // 2
        t = np.linspace(0.0, 0.5, frames, endpoint=False, dtype=np.float32)
        audio = np.column_stack(
            [
                0.05 * np.sin(2.0 * np.pi * 440.0 * t),
                0.05 * np.sin(2.0 * np.pi * 660.0 * t),
            ]
        ).astype(np.float32)
        sf.write(path, audio, sample_rate, subtype="PCM_24")

    def setUp(self) -> None:
        import tempfile

        self._temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self._temp_dir.name)

    def tearDown(self) -> None:
        self._temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
