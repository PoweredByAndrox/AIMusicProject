from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "waveform_forensic_compare.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("waveform_forensic_compare", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestWaveformForensicCompare(unittest.TestCase):
    def test_exact_match_has_silent_residual_and_no_defect_ranges(self) -> None:
        module = _load_module()
        sample_rate = 8000
        time = np.arange(sample_rate, dtype=np.float64) / sample_rate
        reference = (0.3 * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32)

        report = module.compare_waveforms_forensically(
            reference_audio=reference,
            candidate_audio=reference.copy(),
            sample_rate=sample_rate,
        )

        self.assertEqual(report["status"], "passed")
        self.assertLessEqual(report["residual"]["peak_dbfs"], -120.0)
        self.assertEqual(report["defect_ranges"], [])
        self.assertEqual(report["alignment"]["polarity"], "normal")

    def test_localizes_short_injected_leakage_with_sample_bounds(self) -> None:
        module = _load_module()
        sample_rate = 8000
        frame_count = sample_rate * 2
        time = np.arange(frame_count, dtype=np.float64) / sample_rate
        reference = (0.2 * np.sin(2.0 * np.pi * 110.0 * time)).astype(np.float32)
        candidate = reference.copy()
        leak_start = int(0.700 * sample_rate)
        leak_end = int(0.900 * sample_rate)
        candidate[leak_start:leak_end] += (0.12 * np.sin(2.0 * np.pi * 880.0 * time[: leak_end - leak_start])).astype(
            np.float32
        )

        report = module.compare_waveforms_forensically(
            reference_audio=reference,
            candidate_audio=candidate,
            sample_rate=sample_rate,
            defect_threshold_dbfs=-38.0,
            min_defect_duration_sec=0.050,
        )

        self.assertEqual(report["status"], "failed")
        self.assertGreaterEqual(len(report["defect_ranges"]), 1)
        top_range = report["defect_ranges"][0]
        self.assertLessEqual(abs(top_range["start_sample"] - leak_start), int(0.020 * sample_rate))
        self.assertLessEqual(abs(top_range["end_sample"] - leak_end), int(0.020 * sample_rate))
        self.assertEqual(top_range["labels"], ["sample_residual_above_threshold"])
        self.assertGreater(report["histogram"]["p99_abs_error"], report["histogram"]["median_abs_error"])

    def test_reports_polarity_correction_without_hiding_raw_mismatch(self) -> None:
        module = _load_module()
        sample_rate = 8000
        time = np.arange(sample_rate, dtype=np.float64) / sample_rate
        reference = (0.3 * np.sin(2.0 * np.pi * 330.0 * time)).astype(np.float32)

        report = module.compare_waveforms_forensically(
            reference_audio=reference,
            candidate_audio=-reference,
            sample_rate=sample_rate,
        )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["alignment"]["polarity"], "inverted")
        self.assertLess(report["alignment"]["raw_cosine_similarity"], -0.99)
        self.assertGreater(report["alignment"]["aligned_cosine_similarity"], 0.99)
        self.assertTrue(report["alignment"]["polarity_was_corrected"])

    def test_reports_sample_offset_without_hiding_raw_mismatch(self) -> None:
        module = _load_module()
        sample_rate = 8000
        frame_count = sample_rate
        time = np.arange(frame_count, dtype=np.float64) / sample_rate
        reference = (0.3 * np.sin(2.0 * np.pi * 330.0 * time)).astype(np.float32)
        offset_samples = 37
        candidate = np.zeros_like(reference)
        candidate[offset_samples:] = reference[:-offset_samples]

        report = module.compare_waveforms_forensically(
            reference_audio=reference,
            candidate_audio=candidate,
            sample_rate=sample_rate,
            max_offset_samples=64,
        )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["alignment"]["candidate_offset_samples"], offset_samples)
        self.assertTrue(report["alignment"]["offset_was_corrected"])
        self.assertLess(report["alignment"]["raw_cosine_similarity"], 0.99)
        self.assertGreater(report["alignment"]["aligned_cosine_similarity"], 0.99)
        self.assertLessEqual(report["residual"]["peak_dbfs"], -100.0)

    def test_rejects_unbounded_offset_search(self) -> None:
        module = _load_module()
        sample_rate = 8000
        reference = np.zeros(sample_rate, dtype=np.float32)

        with self.assertRaises(ValueError):
            module.compare_waveforms_forensically(
                reference_audio=reference,
                candidate_audio=reference,
                sample_rate=sample_rate,
                max_offset_samples=module.MAX_SAFE_OFFSET_SAMPLES + 1,
            )

    def test_attributes_residual_leakage_to_named_reference_stem(self) -> None:
        module = _load_module()
        sample_rate = 8000
        time = np.arange(sample_rate, dtype=np.float64) / sample_rate
        drum_reference = (0.22 * np.sin(2.0 * np.pi * 110.0 * time)).astype(np.float32)
        melody_reference = (0.12 * np.sin(2.0 * np.pi * 880.0 * time)).astype(np.float32)
        candidate = drum_reference + melody_reference

        report = module.compare_waveforms_forensically(
            reference_audio=drum_reference,
            candidate_audio=candidate,
            sample_rate=sample_rate,
            defect_threshold_dbfs=-55.0,
            reference_stems={"melody": melody_reference},
        )

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["leakage_attribution"]["max_source"], "melody")
        self.assertGreater(report["leakage_attribution"]["sources"]["melody"]["residual_projection_ratio"], 0.95)
        self.assertIn("named_reference_leakage", report["defect_ranges"][0]["labels"])

    def test_writes_forensic_evidence_pack(self) -> None:
        module = _load_module()
        sample_rate = 8000
        frame_count = sample_rate
        time = np.arange(frame_count, dtype=np.float64) / sample_rate
        reference = (0.20 * np.sin(2.0 * np.pi * 110.0 * time)).astype(np.float32)
        candidate = reference.copy()
        candidate[int(0.4 * sample_rate) : int(0.5 * sample_rate)] += 0.10

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            report = module.write_waveform_forensic_evidence_pack(
                reference_audio=reference,
                candidate_audio=candidate,
                sample_rate=sample_rate,
                output_dir=output_dir,
                stem_name="drums",
                defect_threshold_dbfs=-50.0,
            )

            artifacts = report["evidence_artifacts"]
            expected_names = {
                "report_json",
                "trace_npz",
                "residual_audio_wav",
                "waveform_overlay_svg",
                "residual_histogram_svg",
            }
            self.assertTrue(expected_names.issubset(set(artifacts)))
            for artifact_path in artifacts.values():
                self.assertTrue((output_dir / artifact_path).exists(), artifact_path)

            report_json = json.loads((output_dir / artifacts["report_json"]).read_text(encoding="utf-8"))
            self.assertEqual(report_json["stem_name"], "drums")
            self.assertEqual(report_json["waveform_forensics"]["status"], "failed")

            with np.load(output_dir / artifacts["trace_npz"]) as trace:
                self.assertEqual(trace["reference_aligned"].shape, trace["candidate_aligned"].shape)
                self.assertEqual(trace["residual"].shape, trace["abs_error"].shape)
                residual_frame_count = trace["residual"].shape[0]

            residual_audio, residual_sample_rate = sf.read(output_dir / artifacts["residual_audio_wav"])
            self.assertEqual(residual_sample_rate, sample_rate)
            self.assertEqual(residual_audio.shape[0], residual_frame_count)

    def test_evidence_pack_rejects_path_like_stem_name(self) -> None:
        module = _load_module()
        sample_rate = 8000
        reference = np.zeros(sample_rate, dtype=np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                module.write_waveform_forensic_evidence_pack(
                    reference_audio=reference,
                    candidate_audio=reference,
                    sample_rate=sample_rate,
                    output_dir=Path(temp_dir),
                    stem_name="../drums",
                )

    def test_cli_reference_stem_loader_rejects_sample_rate_mismatch(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            matching_path = temp_path / "matching.wav"
            mismatched_path = temp_path / "mismatched.wav"
            sf.write(matching_path, np.zeros(8000, dtype=np.float32), 8000, subtype="FLOAT")
            sf.write(mismatched_path, np.zeros(4000, dtype=np.float32), 4000, subtype="FLOAT")

            with self.assertRaises(ValueError):
                module._load_named_reference_stems(
                    [f"matching={matching_path}", f"mismatch={mismatched_path}"],
                    expected_sample_rate=8000,
                )


if __name__ == "__main__":
    unittest.main()
