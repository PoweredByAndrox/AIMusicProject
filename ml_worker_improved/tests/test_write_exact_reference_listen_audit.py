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
MODULE_PATH = TOOLS_ROOT / "write_exact_reference_listen_audit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("write_exact_reference_listen_audit", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestWriteExactReferenceListenAudit(unittest.TestCase):
    def test_accepts_exact_reference_match(self) -> None:
        module = _load_module()
        sample_rate = 8000
        time = np.arange(sample_rate, dtype=np.float64) / sample_rate
        reference = (0.25 * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            reference_path = temp_path / "reference.wav"
            candidate_path = temp_path / "candidate.wav"
            output_dir = temp_path / "audit"
            sf.write(reference_path, reference, sample_rate, subtype="FLOAT")
            sf.write(candidate_path, reference.copy(), sample_rate, subtype="FLOAT")

            report = module.write_exact_reference_listen_audit(
                reference_path=reference_path,
                candidate_path=candidate_path,
                output_dir=output_dir,
                stem_name="vocals",
                min_sdr_db=40.0,
                min_cosine_percent=99.0,
                max_residual_peak_dbfs=-90.0,
            )

            self.assertEqual(report["delivery"]["status"], "accepted")
            self.assertEqual(report["delivery"]["reason"], "exact_reference_threshold_passed")
            self.assertGreaterEqual(report["metrics"]["sdr_db"], 90.0)
            self.assertGreaterEqual(report["metrics"]["cosine_percent"], 99.999)
            self.assertTrue((output_dir / "exact_reference_listen_audit.json").exists())

    def test_rejects_non_matching_listen_candidate_and_writes_evidence(self) -> None:
        module = _load_module()
        sample_rate = 8000
        frame_count = sample_rate * 2
        time = np.arange(frame_count, dtype=np.float64) / sample_rate
        reference = (0.20 * np.sin(2.0 * np.pi * 110.0 * time)).astype(np.float32)
        candidate = reference.copy()
        candidate[int(0.70 * sample_rate) : int(1.10 * sample_rate)] += 0.14

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            reference_path = temp_path / "reference.wav"
            candidate_path = temp_path / "candidate.wav"
            output_dir = temp_path / "audit"
            sf.write(reference_path, reference, sample_rate, subtype="FLOAT")
            sf.write(candidate_path, candidate, sample_rate, subtype="FLOAT")

            report = module.write_exact_reference_listen_audit(
                reference_path=reference_path,
                candidate_path=candidate_path,
                output_dir=output_dir,
                stem_name="vocals",
                min_sdr_db=40.0,
                min_cosine_percent=99.0,
                max_residual_peak_dbfs=-90.0,
                defect_threshold_dbfs=-50.0,
                max_evidence_frames=1024,
            )

            self.assertEqual(report["delivery"]["status"], "rejected")
            self.assertEqual(report["delivery"]["reason"], "exact_reference_threshold_failed")
            self.assertIn("sdr_below_threshold", report["delivery"]["failed_checks"])
            self.assertIn("residual_peak_above_threshold", report["delivery"]["failed_checks"])
            self.assertIn("waveform_forensic_scope_truncated", report["delivery"]["failed_checks"])
            self.assertEqual(report["comparison_mode"], "exact_reference_listen_audit")
            self.assertEqual(report["reference"]["path"], str(reference_path))
            self.assertEqual(report["candidate"]["path"], str(candidate_path))

            evidence = report["forensic_evidence"]
            self.assertEqual(evidence["status"], "available")
            self.assertEqual(evidence["artifact_scope"]["mode"], "head_truncated")
            self.assertEqual(evidence["artifact_scope"]["aligned_frame_count"], frame_count)
            self.assertEqual(evidence["artifact_scope"]["artifact_frame_count"], 1024)
            self.assertEqual(evidence["artifact_scope"]["waveform_forensic_scope"], "evidence_head_only")
            self.assertEqual(evidence["artifact_scope"]["exact_metrics_scope"], "full_overlap_streaming")
            self.assertEqual(report["metrics"]["compared_frame_count"], frame_count)
            for artifact_path in evidence["evidence_artifacts"].values():
                self.assertTrue((output_dir / evidence["output_dir"] / artifact_path).exists(), artifact_path)

            persisted = json.loads((output_dir / "exact_reference_listen_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["delivery"]["status"], "rejected")
            persisted_evidence = json.loads(
                (output_dir / evidence["output_dir"] / evidence["evidence_artifacts"]["report_json"]).read_text(
                    encoding="utf-8",
                )
            )
            self.assertEqual(persisted_evidence["artifact_scope"]["mode"], "head_truncated")


if __name__ == "__main__":
    unittest.main()
