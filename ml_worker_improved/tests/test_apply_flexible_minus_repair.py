from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "apply_flexible_minus_repair.py"


def _load_repair_module():
    spec = importlib.util.spec_from_file_location("apply_flexible_minus_repair", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stereo(audio: np.ndarray) -> np.ndarray:
    return np.stack([audio, audio * 0.97], axis=1).astype(np.float32)


def _band_ratio_delta(module, repaired: np.ndarray, base: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    return module.band_energy_ratio_db(repaired, sample_rate, low_hz=low_hz, high_hz=high_hz) - module.band_energy_ratio_db(
        base,
        sample_rate,
        low_hz=low_hz,
        high_hz=high_hz,
    )


class TestApplyFlexibleMinusRepair(unittest.TestCase):
    def test_parse_focus_window_validates_contract(self):
        module = _load_repair_module()

        self.assertEqual(module.parse_focus_window("intro:0:35"), ("intro", 0.0, 35.0))

        with self.assertRaises(argparse.ArgumentTypeError):
            module.parse_focus_window("intro")
        with self.assertRaises(argparse.ArgumentTypeError):
            module.parse_focus_window(":0:35")
        with self.assertRaises(argparse.ArgumentTypeError):
            module.parse_focus_window("intro:35:0")

    def test_resample_to_sample_rate_preserves_channel_shape_and_target_length(self):
        module = _load_repair_module()
        source_sample_rate = 48000
        target_sample_rate = 44100
        frames = source_sample_rate
        audio = np.zeros((frames, 2), dtype=np.float32)

        resampled = module.resample_to_sample_rate(audio, source_sample_rate, target_sample_rate)

        self.assertEqual(resampled.shape[1], 2)
        self.assertLessEqual(abs(resampled.shape[0] - target_sample_rate), 1)
        self.assertEqual(resampled.dtype, np.float32)

    def test_ambience_floor_repair_lifts_presence_dropout_without_touching_low_band(self):
        module = _load_repair_module()
        sample_rate = 16000
        seconds = 4.0
        t = np.arange(int(sample_rate * seconds), dtype=np.float32) / float(sample_rate)
        low_body = 0.18 * np.sin(2.0 * np.pi * 220.0 * t)
        high_texture = 0.045 * np.sin(2.0 * np.pi * 5200.0 * t)
        envelope = np.ones_like(t)
        dropout = (t >= 1.2) & (t <= 1.8)
        envelope[dropout] = 0.12
        base_minus = _stereo(low_body + high_texture * envelope)

        repaired, debug = module.apply_ambience_floor_repair(
            base_minus,
            sample_rate,
            strength=0.85,
            max_gain_db=3.0,
            percentile_window_sec=1.2,
        )

        start = int(1.25 * sample_rate)
        end = int(1.75 * sample_rate)
        dropout_base = base_minus[start:end]
        dropout_repaired = repaired[start:end]

        self.assertEqual(repaired.shape, base_minus.shape)
        self.assertGreater(_band_ratio_delta(module, dropout_repaired, dropout_base, sample_rate, 4800.0, 6200.0), 0.15)
        self.assertLess(abs(_band_ratio_delta(module, dropout_repaired, dropout_base, sample_rate, 80.0, 400.0)), 0.08)
        self.assertGreater(max(channel["active_frame_ratio"] for channel in debug), 0.0)

    def test_texture_floor_inpainting_lifts_missing_texture_without_raw_audio(self):
        module = _load_repair_module()
        sample_rate = 16000
        seconds = 4.0
        t = np.arange(int(sample_rate * seconds), dtype=np.float32) / float(sample_rate)
        low_body = 0.12 * np.sin(2.0 * np.pi * 220.0 * t)
        texture = 0.025 * np.sin(2.0 * np.pi * 2600.0 * t) + 0.018 * np.sin(2.0 * np.pi * 5200.0 * t)
        envelope = np.ones_like(t)
        envelope[(t >= 1.2) & (t <= 1.8)] = 0.08
        base_minus = _stereo(low_body + texture * envelope)

        repaired, debug = module.apply_texture_floor_inpainting(
            base_minus,
            sample_rate,
            strength=0.85,
            percentile_window_sec=1.2,
            max_floor_ratio=0.35,
        )

        start = int(1.25 * sample_rate)
        end = int(1.75 * sample_rate)
        dropout_base = base_minus[start:end]
        dropout_repaired = repaired[start:end]

        self.assertEqual(repaired.shape, base_minus.shape)
        self.assertGreater(_band_ratio_delta(module, dropout_repaired, dropout_base, sample_rate, 1800.0, 6000.0), 0.2)
        self.assertLess(abs(_band_ratio_delta(module, dropout_repaired, dropout_base, sample_rate, 80.0, 400.0)), 0.08)
        self.assertGreater(max(channel["active_frame_ratio"] for channel in debug), 0.0)

    def test_clip_to_peak_does_not_apply_global_gain_reduction(self):
        module = _load_repair_module()
        audio = np.array([[0.5, -0.5], [1.2, -1.4]], dtype=np.float32)

        clipped = module.clip_to_peak(audio, max_peak=0.9)

        self.assertAlmostEqual(float(clipped[0, 0]), 0.5, places=6)
        self.assertAlmostEqual(float(clipped[0, 1]), -0.5, places=6)
        self.assertAlmostEqual(float(clipped[1, 0]), 0.9, places=6)
        self.assertAlmostEqual(float(clipped[1, 1]), -0.9, places=6)

    def test_added_layer_monitor_filters_low_phase_residual_without_changing_length(self):
        module = _load_repair_module()
        sample_rate = 16000
        seconds = 2.0
        t = np.arange(int(sample_rate * seconds), dtype=np.float32) / float(sample_rate)
        low_phase_residual = 0.05 * np.sin(2.0 * np.pi * 160.0 * t)
        vocal_check_band = 0.03 * np.sin(2.0 * np.pi * 2600.0 * t)
        added_layer = _stereo(low_phase_residual + vocal_check_band)

        monitored = module.build_added_layer_monitor(added_layer, sample_rate)

        self.assertEqual(monitored.shape, added_layer.shape)
        self.assertLess(module.band_energy_ratio_db(monitored, sample_rate, low_hz=80.0, high_hz=350.0), -20.0)
        self.assertGreater(module.band_energy_ratio_db(monitored, sample_rate, low_hz=1800.0, high_hz=4500.0), -3.0)

    def test_vocal_residue_proxy_metrics_detects_vocal_like_leakage_in_minus(self):
        module = _load_repair_module()
        sample_rate = 16000
        seconds = 2.0
        t = np.arange(int(sample_rate * seconds), dtype=np.float32) / float(sample_rate)
        instrumental = 0.16 * np.sin(2.0 * np.pi * 220.0 * t)
        vocal = 0.12 * np.sin(2.0 * np.pi * 980.0 * t)
        raw_audio = _stereo(instrumental + vocal)
        clean_minus = _stereo(instrumental)
        leaky_minus = _stereo(instrumental + vocal * 0.45)

        clean_metrics = module.vocal_residue_proxy_metrics(raw_audio, clean_minus)
        leaky_metrics = module.vocal_residue_proxy_metrics(raw_audio, leaky_minus)

        self.assertLess(clean_metrics["vocal_residue_corr"], 0.05)
        self.assertGreater(leaky_metrics["vocal_residue_corr"], 0.25)
        self.assertLess(clean_metrics["vocal_to_instrumental_rms_ratio_db"], -40.0)
        self.assertGreater(leaky_metrics["vocal_to_instrumental_rms_ratio_db"], -18.0)

    def test_kick_transient_restore_lifts_attentuated_low_band_with_capped_blend(self):
        module = _load_repair_module()
        sample_rate = 16000
        seconds = 4.0
        t = np.arange(int(sample_rate * seconds), dtype=np.float32) / float(sample_rate)
        background = 0.035 * np.sin(2.0 * np.pi * 440.0 * t)
        kick = np.zeros_like(t)
        for center_sec in (0.75, 1.5, 2.25, 3.0):
            start = int(center_sec * sample_rate)
            length = int(0.16 * sample_rate)
            local_t = np.arange(length, dtype=np.float32) / float(sample_rate)
            transient = 0.48 * np.sin(2.0 * np.pi * 72.0 * local_t) * np.exp(-local_t * 18.0)
            kick[start : start + length] += transient[: max(0, min(length, kick.shape[0] - start))]
        raw_audio = _stereo(background + kick)
        base_minus = _stereo(background + kick * 0.22)

        repaired, debug = module.apply_kick_transient_restore(
            raw_audio,
            base_minus,
            sample_rate,
            strength=0.28,
            max_blend=0.08,
            activity_threshold=1.05,
            window_sec=0.7,
        )

        focus_start = int(1.45 * sample_rate)
        focus_end = int(1.65 * sample_rate)
        base_focus = base_minus[focus_start:focus_end]
        repaired_focus = repaired[focus_start:focus_end]

        self.assertEqual(repaired.shape, base_minus.shape)
        self.assertGreater(_band_ratio_delta(module, repaired_focus, base_focus, sample_rate, 35.0, 180.0), 0.75)
        self.assertLess(module.rms_dbfs(repaired - base_minus) - module.rms_dbfs(base_minus), -17.5)
        self.assertLessEqual(max(channel["max_blend"] for channel in debug), 0.080001)
        self.assertGreater(max(channel["active_frame_ratio"] for channel in debug), 0.0)

    def test_kick_transient_restore_lifts_attack_band_when_low_body_is_present(self):
        module = _load_repair_module()
        sample_rate = 16000
        seconds = 4.0
        t = np.arange(int(sample_rate * seconds), dtype=np.float32) / float(sample_rate)
        background = 0.025 * np.sin(2.0 * np.pi * 440.0 * t)
        low_kick = np.zeros_like(t)
        attack_click = np.zeros_like(t)
        for center_sec in (0.75, 1.5, 2.25, 3.0):
            start = int(center_sec * sample_rate)
            length = int(0.16 * sample_rate)
            local_t = np.arange(length, dtype=np.float32) / float(sample_rate)
            low_transient = 0.42 * np.sin(2.0 * np.pi * 72.0 * local_t) * np.exp(-local_t * 18.0)
            attack_transient = 0.12 * np.sin(2.0 * np.pi * 2300.0 * local_t) * np.exp(-local_t * 35.0)
            available = max(0, min(length, low_kick.shape[0] - start))
            low_kick[start : start + available] += low_transient[:available]
            attack_click[start : start + available] += attack_transient[:available]
        raw_audio = _stereo(background + low_kick + attack_click)
        base_minus = _stereo(background + low_kick + attack_click * 0.18)

        repaired, debug = module.apply_kick_transient_restore(
            raw_audio,
            base_minus,
            sample_rate,
            strength=0.36,
            max_blend=0.12,
            activity_threshold=1.05,
            window_sec=0.7,
        )

        focus_start = int(1.45 * sample_rate)
        focus_end = int(1.65 * sample_rate)
        base_focus = base_minus[focus_start:focus_end]
        repaired_focus = repaired[focus_start:focus_end]

        self.assertEqual(repaired.shape, base_minus.shape)
        self.assertGreater(_band_ratio_delta(module, repaired_focus, base_focus, sample_rate, 1600.0, 4500.0), 0.25)
        self.assertLess(abs(_band_ratio_delta(module, repaired_focus, base_focus, sample_rate, 35.0, 180.0)), 0.2)
        self.assertLessEqual(max(channel["max_blend"] for channel in debug), 0.120001)
        self.assertGreater(max(channel["mean_blend_on_repair_band"] for channel in debug), 0.0)

    def test_cli_writes_focus_metrics_for_intro_and_kick_windows(self):
        module = _load_repair_module()
        sample_rate = 16000
        seconds = 4.0
        t = np.arange(int(sample_rate * seconds), dtype=np.float32) / float(sample_rate)
        low_body = 0.12 * np.sin(2.0 * np.pi * 220.0 * t)
        high_texture = 0.05 * np.sin(2.0 * np.pi * 5200.0 * t)
        kick = np.zeros_like(t)
        kick_start = int(2.35 * sample_rate)
        kick_length = int(0.16 * sample_rate)
        local_t = np.arange(kick_length, dtype=np.float32) / float(sample_rate)
        kick[kick_start : kick_start + kick_length] = 0.5 * np.sin(2.0 * np.pi * 72.0 * local_t) * np.exp(-local_t * 18.0)

        ambience_envelope = np.ones_like(t)
        ambience_envelope[(t >= 0.4) & (t <= 1.6)] = 0.12
        raw_audio = _stereo(low_body + high_texture + kick)
        base_minus = _stereo(low_body + high_texture * ambience_envelope + kick * 0.22)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_path = temp_path / "raw.flac"
            base_path = temp_path / "base.flac"
            output_dir = temp_path / "out"
            listen_dir = temp_path / "listen"
            sf.write(str(raw_path), raw_audio, sample_rate, subtype="PCM_24")
            sf.write(str(base_path), base_minus, sample_rate, subtype="PCM_24")

            argv = [
                "apply_flexible_minus_repair.py",
                str(raw_path),
                str(base_path),
                str(output_dir),
                "--listen-dir",
                str(listen_dir),
                "--focus-window",
                "intro:0:2",
                "--focus-window",
                "intro_dropout_4_5:1:2",
                "--focus-window",
                "kick:2:3",
                "--ambience-strength",
                "0.85",
                "--ambience-max-gain-db",
                "3.0",
                "--kick-activity-threshold",
                "1.05",
                "--kick-window-sec",
                "0.7",
            ]

            with patch.object(sys, "argv", argv):
                self.assertEqual(module.main(), 0)

            manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))

            self.assertTrue((output_dir / "focus_intro_02_flexible_minus_repaired.flac").exists())
            self.assertTrue((output_dir / "focus_kick_02_flexible_minus_repaired.flac").exists())
            self.assertTrue((output_dir / "04_added_layer_delta_raw.flac").exists())
            self.assertTrue((listen_dir / "focus_intro_repaired_minus.flac").exists())
            self.assertTrue((listen_dir / "focus_kick_repaired_minus.flac").exists())
            self.assertTrue((listen_dir / "04_added_layer_delta_raw.flac").exists())
            self.assertIn("focus_metrics", manifest)
            self.assertIn("added_layer_delta_raw_flac", manifest["artifacts"])
            self.assertIn("artifact_contract", manifest)
            self.assertIn("vocal_residue_corr", manifest["metrics"])
            self.assertIn("vocal_to_instrumental_rms_ratio_db", manifest["metrics"])
            repaired_audio, repaired_sample_rate = sf.read(str(output_dir / "02_flexible_minus_repaired.flac"), always_2d=True)
            base_audio, base_sample_rate = sf.read(str(output_dir / "01_base_minus.flac"), always_2d=True)
            delta_audio, delta_sample_rate = sf.read(str(output_dir / "04_added_layer_delta_raw.flac"), always_2d=True)
            self.assertEqual(repaired_sample_rate, sample_rate)
            self.assertEqual(base_sample_rate, sample_rate)
            self.assertEqual(delta_sample_rate, sample_rate)
            np.testing.assert_allclose(delta_audio, repaired_audio - base_audio, atol=3e-6)
            self.assertIn("kick_attack_band_ratio_db", manifest["focus_metrics"]["kick"]["drift_vs_base"])
            self.assertIn("intro_dropout_4_5", manifest["focus_metrics"])
            self.assertEqual(manifest["focus_metrics"]["intro_dropout_4_5"]["window_sec"]["start"], 1.0)
            self.assertEqual(manifest["focus_metrics"]["intro_dropout_4_5"]["window_sec"]["end"], 2.0)
            self.assertFalse(manifest["repair"]["tuning"]["enable_raw_kick_repair"])
            self.assertFalse(manifest["debug"]["kick"][0]["enabled"])
            self.assertGreater(manifest["focus_metrics"]["intro"]["drift_vs_base"]["presence_band_ratio_db"], 0.15)
            self.assertLess(manifest["focus_metrics"]["kick"]["drift_vs_base"]["added_layer_vs_base_rms_db"], -13.0)

    def test_cli_pure_base_bypasses_all_repair_layers(self):
        module = _load_repair_module()
        sample_rate = 16000
        seconds = 2.0
        t = np.arange(int(sample_rate * seconds), dtype=np.float32) / float(sample_rate)
        raw_audio = _stereo(
            0.25 * np.sin(2.0 * np.pi * 220.0 * t)
            + 0.08 * np.sin(2.0 * np.pi * 5200.0 * t)
        )
        base_minus = _stereo(
            0.25 * np.sin(2.0 * np.pi * 220.0 * t)
            + 0.01 * np.sin(2.0 * np.pi * 5200.0 * t)
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_path = temp_path / "raw.flac"
            base_path = temp_path / "base.flac"
            output_dir = temp_path / "out"
            listen_dir = temp_path / "listen"
            sf.write(str(raw_path), raw_audio, sample_rate, subtype="PCM_24")
            sf.write(str(base_path), base_minus, sample_rate, subtype="PCM_24")

            argv = [
                "apply_flexible_minus_repair.py",
                str(raw_path),
                str(base_path),
                str(output_dir),
                "--pure-base",
                "--enable-raw-kick-repair",
                "--listen-dir",
                str(listen_dir),
                "--focus-window",
                "intro:0:1",
            ]

            with patch.object(sys, "argv", argv):
                self.assertEqual(module.main(), 0)

            manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
            repaired_audio, repaired_sample_rate = sf.read(
                str(output_dir / "02_flexible_minus_repaired.flac"),
                always_2d=True,
            )
            base_audio, base_sample_rate = sf.read(str(output_dir / "01_base_minus.flac"), always_2d=True)
            delta_audio, delta_sample_rate = sf.read(str(output_dir / "04_added_layer_delta_raw.flac"), always_2d=True)

        self.assertEqual(repaired_sample_rate, sample_rate)
        self.assertEqual(base_sample_rate, sample_rate)
        self.assertEqual(delta_sample_rate, sample_rate)
        np.testing.assert_allclose(repaired_audio, base_audio, atol=3e-6)
        self.assertLess(float(np.max(np.abs(delta_audio))), 3e-6)
        self.assertTrue(manifest["repair"]["tuning"]["pure_base"])
        self.assertFalse(manifest["repair"]["tuning"]["enable_raw_kick_repair"])
        self.assertEqual(manifest["debug"]["ambience"][0]["reason"], "pure_base_bypass")
        self.assertEqual(manifest["debug"]["texture"][0]["reason"], "pure_base_bypass")
        self.assertEqual(manifest["debug"]["kick"][0]["reason"], "pure_base_bypass")
        self.assertLess(manifest["metrics"]["added_layer_rms_dbfs"], -140.0)
        self.assertEqual(manifest["focus_metrics"]["intro"]["drift_vs_base"]["rms_db"], 0.0)


if __name__ == "__main__":
    unittest.main()
