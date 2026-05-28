from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = SCRIPTS_ROOT / "run_nine_thou_quality_gate.py"


def _load_gate_module():
    spec = importlib.util.spec_from_file_location("run_nine_thou_quality_gate", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestRunNineThouQualityGate(unittest.TestCase):
    def test_scale_invariant_sdr_is_scale_invariant(self):
        module = _load_gate_module()
        reference = np.tile(np.array([1.0, -0.5, 0.25, -0.125], dtype=np.float32), 512)
        estimate = 2.0 * reference

        self.assertLess(module.signal_to_distortion_db(reference, estimate), 1.0)
        self.assertGreater(module.scale_invariant_signal_to_distortion_db(reference, estimate), 100.0)

    def test_bss_eval_like_metrics_separate_target_interference_and_artifact(self):
        module = _load_gate_module()
        target = np.tile(np.array([1.0, 0.0], dtype=np.float32), 1024)
        interference = np.tile(np.array([0.0, 1.0], dtype=np.float32), 1024)
        estimate = target + 0.1 * interference

        metrics = module.bss_eval_like_metrics(target, interference, estimate)

        self.assertGreater(metrics["bss_eval_like_sir_db"], 19.0)
        self.assertGreater(metrics["bss_eval_like_sar_db"], 100.0)
        self.assertAlmostEqual(metrics["target_projection_gain"], 1.0, places=6)
        self.assertAlmostEqual(metrics["interference_projection_gain"], 0.1, places=6)

    def test_pseudo_reference_warnings_flag_weak_source_contract(self):
        module = _load_gate_module()
        zeros = np.zeros(4, dtype=np.float32)
        references = module.PseudoReferenceSet(
            full_mix=zeros,
            reference_minus=zeros,
            instrumental_reference=zeros,
            vocal_residual_reference=zeros,
            sample_rate=16000,
            minus_offset_vs_full_sec=-0.85,
            envelope_score=0.8,
            minus_gain_to_full=-0.2,
            minus_cosine_to_full=-0.1,
        )

        warnings = module.pseudo_reference_warnings(references)
        warning_codes = {warning["code"] for warning in warnings}

        self.assertIn("weak_full_minus_envelope_alignment", warning_codes)
        self.assertIn("negative_minus_gain_to_full_mix", warning_codes)
        self.assertIn("low_full_minus_cosine", warning_codes)

    def test_align_pair_with_indices_reports_reference_and_candidate_windows(self):
        module = _load_gate_module()
        reference = np.arange(10, dtype=np.float32)
        candidate = np.arange(8, dtype=np.float32)

        reference_aligned, candidate_aligned, window = module.align_pair_with_indices(
            reference,
            candidate,
            sample_rate=10,
            candidate_offset_seconds=0.2,
        )

        np.testing.assert_array_equal(reference_aligned, reference[:6])
        np.testing.assert_array_equal(candidate_aligned, candidate[2:8])
        self.assertEqual(window.reference_start, 0)
        self.assertEqual(window.candidate_start, 2)
        self.assertEqual(window.sample_count, 6)

    def test_run_quality_gate_writes_summary_without_threshold_exit_gate(self):
        module = _load_gate_module()
        sample_rate = 16000
        seconds = 8
        t = np.arange(sample_rate * seconds, dtype=np.float32) / float(sample_rate)
        envelope = (0.25 + 0.08 * np.sin(2.0 * np.pi * 0.5 * t)).astype(np.float32)
        instrumental = (envelope * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
        vocal = (0.08 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        full_mix = (instrumental + vocal).astype(np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            full_path = temp_path / "full.mp3"
            minus_path = temp_path / "minus.mp3"
            candidate_path = temp_path / "candidate.flac"
            output_dir = temp_path / "gate"
            full_path.write_bytes(b"full")
            minus_path.write_bytes(b"minus")
            candidate_path.write_bytes(b"candidate")

            def fake_decode(path: Path, requested_sample_rate: int) -> np.ndarray:
                self.assertEqual(requested_sample_rate, sample_rate)
                if path == full_path:
                    return full_mix
                if path == minus_path or path == candidate_path:
                    return instrumental
                raise AssertionError(path)

            def fake_probe(path: Path) -> dict[str, object]:
                return {
                    "path": str(path),
                    "sha256": "fake",
                    "duration_sec": seconds,
                    "audio_codec": "fixture",
                }

            def fake_analyze_pair(*args, **kwargs) -> dict[str, object]:
                return {
                    "analysis_sample_rate": sample_rate,
                    "alignment": {
                        "candidate_b_offset_vs_a_sec": 0.0,
                        "energy_envelope_score": 1.0,
                        "aligned_duration_sec": seconds,
                    },
                }

            def fake_compare_candidate_to_reference(*args, **kwargs) -> dict[str, object]:
                return {
                    "alignment": {
                        "candidate_offset_vs_reference_sec": 0.0,
                        "energy_envelope_score": 1.0,
                        "aligned_duration_sec": seconds,
                    },
                    "metrics": {
                        "reference": {},
                        "candidate": {},
                        "residual_after_gain_match": {},
                    },
                    "comparisons": {
                        "cosine_similarity": 1.0,
                        "absolute_cosine_similarity": 1.0,
                        "gain_candidate_to_reference": 1.0,
                        "rms_drift_db": 0.0,
                        "peak_drift_db": 0.0,
                        "duration_drift_sec": 0.0,
                        "high_band_ratio_drift_4k": 0.0,
                        "air_band_ratio_drift_7k": 0.0,
                        "residual_vs_reference_rms_db": -120.0,
                    },
                }

            candidate = module.CandidateSpec(
                name="fixture_candidate",
                audio_path=candidate_path,
                report_name="fixture_candidate_gate.json",
            )

            with patch.object(module, "decode_mono", side_effect=fake_decode), patch.object(
                module,
                "probe_audio",
                side_effect=fake_probe,
            ), patch.object(module, "analyze_pair", side_effect=fake_analyze_pair), patch.object(
                module,
                "compare_candidate_to_reference",
                side_effect=fake_compare_candidate_to_reference,
            ):
                summary = module.run_quality_gate(
                    full_mix_path=full_path,
                    reference_minus_path=minus_path,
                    output_dir=output_dir,
                    candidates=[candidate],
                    thresholds=module.GateThresholds(),
                    sample_rate=sample_rate,
                    max_offset_seconds=1.0,
                )

            summary_path = output_dir / "quality_gate_summary.json"
            report_path = output_dir / "fixture_candidate_gate.json"
            self.assertTrue(summary_path.exists())
            self.assertTrue(report_path.exists())

            saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            saved_report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["overall"]["status"], "passed")
            self.assertEqual(saved_summary["execution"]["threshold_exit_gate"], "disabled")
            self.assertTrue(saved_summary["execution"]["does_not_call_fail_on_threshold"])
            self.assertEqual(saved_report["verdict"]["status"], "passed")
            self.assertEqual(
                saved_report["pseudo_objective_quality"]["ground_truth_mode"],
                "pseudo_from_full_mix_and_reference_minus",
            )
            self.assertGreater(saved_report["pseudo_objective_quality"]["metrics"]["si_sdr_db"], 100.0)


if __name__ == "__main__":
    unittest.main()
