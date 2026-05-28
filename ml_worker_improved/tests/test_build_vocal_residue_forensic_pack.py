from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "build_vocal_residue_forensic_pack.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_vocal_residue_forensic_pack", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class VocalResidueForensicPackTests(unittest.TestCase):
    def test_projection_only_window_is_rejected_as_vocal_evidence(self) -> None:
        module = _load_module()

        classification = module.classify_vocal_residue_window(
            {
                "vocal_projection_ratio_db": -11.0,
                "vocal_residue_corr": 0.25,
                "masked_residue_ratio_db": -28.0,
            }
        )

        self.assertEqual(classification["label"], "projection_only_melody_contamination")
        self.assertEqual(classification["vocal_evidence_status"], "rejected_as_projection_only")

    def test_projection_and_masked_residue_window_requires_review(self) -> None:
        module = _load_module()

        classification = module.classify_vocal_residue_window(
            {
                "vocal_projection_ratio_db": -16.0,
                "vocal_residue_corr": 0.15,
                "masked_residue_ratio_db": -9.0,
            }
        )

        self.assertEqual(classification["label"], "needs_residue_review")
        self.assertEqual(classification["vocal_evidence_status"], "needs_human_review")

    def test_weak_projection_window_is_low_priority(self) -> None:
        module = _load_module()

        classification = module.classify_vocal_residue_window(
            {
                "vocal_projection_ratio_db": -24.0,
                "vocal_residue_corr": 0.05,
                "masked_residue_ratio_db": -30.0,
            }
        )

        self.assertEqual(classification["label"], "low_priority_residue")
        self.assertEqual(classification["vocal_evidence_status"], "low_priority")


if __name__ == "__main__":
    unittest.main()
