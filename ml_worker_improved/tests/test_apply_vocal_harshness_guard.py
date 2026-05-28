from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "apply_vocal_harshness_guard.py"


def _load_harshness_guard_module():
    spec = importlib.util.spec_from_file_location("apply_vocal_harshness_guard", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _band_rms(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    spectrum = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(mono.size, d=1.0 / sample_rate)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return 0.0
    band = np.fft.irfft(spectrum * mask.astype(np.float32), n=mono.size)
    return float(np.sqrt(np.mean(np.square(band, dtype=np.float32))))


class TestApplyVocalHarshnessGuard(unittest.TestCase):
    def test_build_window_decisions_targets_harshness_and_protects_vocal_loss(self):
        module = _load_harshness_guard_module()
        tuning = module.HarshnessGuardTuning(
            base_strength=0.60,
            max_strength=1.0,
            vocal_loss_strength_scale=0.40,
            min_segment_severity=0.20,
        )
        base_candidate = {
            "segment_issue_report": {
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 1.0,
                        "severity_score": 1.0,
                        "labels": ["high_noise_or_robotic_artifact"],
                        "metrics": {"high_noise_flatness_6000_16000": 0.24},
                    },
                    {
                        "start_sec": 1.0,
                        "end_sec": 2.0,
                        "severity_score": 1.0,
                        "labels": ["vocal_dropout", "high_noise_or_robotic_artifact"],
                        "metrics": {"high_noise_flatness_6000_16000": 0.24},
                    },
                    {
                        "start_sec": 2.0,
                        "end_sec": 3.0,
                        "severity_score": 1.2,
                        "labels": ["excessive_reverb_tail_smear"],
                        "metrics": {"high_noise_flatness_6000_16000": 0.18},
                    },
                    {
                        "start_sec": 3.0,
                        "end_sec": 4.0,
                        "severity_score": 2.0,
                        "labels": ["kick_punch_bleed"],
                        "metrics": {"high_noise_flatness_6000_16000": 0.10},
                    },
                ]
            }
        }

        decisions = module.build_window_decisions(base_candidate, tuning)

        self.assertEqual([decision["segment_index"] for decision in decisions], [0, 1, 2])
        self.assertGreater(decisions[0]["strength"], decisions[1]["strength"])
        self.assertIn("vocal_dropout", decisions[1]["labels"])
        self.assertIn("excessive_reverb_tail_smear", decisions[2]["labels"])

    def test_harshness_guard_reduces_marked_high_band_without_touching_unmarked_window(self):
        module = _load_harshness_guard_module()
        sample_rate = 16000
        duration_sec = 3
        t = np.arange(sample_rate * duration_sec, dtype=np.float32) / float(sample_rate)
        vocal_body = (
            0.16 * np.sin(2.0 * np.pi * 260.0 * t)
            + 0.06 * np.sin(2.0 * np.pi * 1700.0 * t)
        ).astype(np.float32)
        harsh_air = (0.08 * np.sin(2.0 * np.pi * 7000.0 * t)).astype(np.float32)
        anchor_air = (0.01 * np.sin(2.0 * np.pi * 7000.0 * t)).astype(np.float32)
        base_mono = vocal_body.copy()
        anchor_mono = vocal_body.copy()
        base_mono[:sample_rate] += harsh_air[:sample_rate]
        anchor_mono[:sample_rate] += anchor_air[:sample_rate]
        base_audio = np.stack([base_mono, base_mono], axis=1).astype(np.float32)
        anchor_audio = np.stack([anchor_mono, anchor_mono], axis=1).astype(np.float32)
        decisions = [
            {
                "start_sec": 0.0,
                "end_sec": 1.0,
                "strength": 1.0,
                "severity_score": 2.0,
                "labels": ["high_noise_or_robotic_artifact"],
            }
        ]

        guarded_audio, debug = module.apply_harshness_guard(
            base_audio,
            anchor_audio,
            sample_rate=sample_rate,
            decisions=decisions,
            tuning=module.HarshnessGuardTuning(
                high_band_start_hz=6200.0,
                high_band_end_hz=7600.0,
                anchor_tolerance=1.0,
                min_high_band_floor=0.45,
                base_strength=1.0,
                max_strength=1.0,
                n_fft=2048,
                hop_length=256,
            ),
        )

        self.assertEqual(debug["window_count"], 1)
        self.assertLess(
            _band_rms(guarded_audio[:sample_rate], sample_rate, 6200.0, 7600.0),
            0.72 * _band_rms(base_audio[:sample_rate], sample_rate, 6200.0, 7600.0),
        )
        self.assertAlmostEqual(
            _band_rms(guarded_audio[:sample_rate], sample_rate, 180.0, 4500.0),
            _band_rms(base_audio[:sample_rate], sample_rate, 180.0, 4500.0),
            delta=0.002,
        )
        np.testing.assert_allclose(
            guarded_audio[sample_rate * 2 :],
            base_audio[sample_rate * 2 :],
            atol=1e-4,
        )

    def test_listen_pack_names_anchor_file_from_anchor_profile(self):
        module = _load_harshness_guard_module()
        sample_rate = 8000
        t = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
        base_mono = (0.10 * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)
        anchor_mono = (0.08 * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)
        base_audio = np.stack([base_mono, base_mono], axis=1)
        anchor_audio = np.stack([anchor_mono, anchor_mono], axis=1)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.flac"
            base_path = temp_path / "base.flac"
            anchor_path = temp_path / "anchor.flac"
            manifest_path = temp_path / "adaptive_manifest.json"
            output_dir = temp_path / "output"
            listen_dir = temp_path / "listen"
            sf.write(source_path, base_audio, sample_rate, subtype="PCM_24")
            sf.write(base_path, base_audio, sample_rate, subtype="PCM_24")
            sf.write(anchor_path, anchor_audio, sample_rate, subtype="PCM_24")
            manifest_path.write_text(
                json.dumps(
                    {
                        "selected_profile": "issue_aware_segment_selection",
                        "source_audio_path": str(source_path),
                        "source_audio_sha256": "synthetic",
                        "candidates": [
                            {
                                "profile_name": "issue_aware_segment_selection",
                                "vocals_path": str(base_path),
                                "segment_issue_report": {"segments": []},
                            },
                            {"profile_name": "ft_tail_safe", "vocals_path": str(anchor_path)},
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            module.run_harshness_guard(
                adaptive_manifest_path=manifest_path,
                output_dir=output_dir,
                listen_dir=listen_dir,
                base_profile_name=None,
                anchor_profile_name="ft_tail_safe",
                tuning=module.HarshnessGuardTuning(n_fft=1024, hop_length=256),
                max_listen_windows=0,
            )
            candidate_map = json.loads((listen_dir / "candidate_map.json").read_text(encoding="utf-8"))

        self.assertIn("03_anchor_ft_tail_safe.flac", candidate_map)
        self.assertNotIn("03_anchor_bs_artifact_clean.flac", candidate_map)


if __name__ == "__main__":
    unittest.main()
