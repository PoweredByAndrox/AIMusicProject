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
MODULE_PATH = TOOLS_ROOT / "build_quality_guided_minus_matrix.py"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
SPEC = importlib.util.spec_from_file_location("build_quality_guided_minus_matrix", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def quality_row(*, direct_votes: int, issues: list[str], delta: float) -> dict:
    return {
        "segment_id": "s0000",
        "window_sec": [0.0, 1.0],
        "confirmed_issues": issues,
        "votes": {"direct_vocal": [f"p{i}" for i in range(direct_votes)]},
        "source_conservation": {"source_candidate_delta_rms_ratio": delta},
    }


class TestQualityGuidedMinusMatrix(unittest.TestCase):
    def test_selects_only_alternate_with_vote_reduction_and_no_detail_damage(self):
        selected, rejected = MODULE.select_repair_windows(
            [quality_row(direct_votes=4, issues=[MODULE.VOCAL_ISSUE], delta=0.0)],
            [quality_row(direct_votes=3, issues=[MODULE.VOCAL_ISSUE], delta=0.24)],
            min_direct_vote_reduction=1,
            max_alternate_delta_rms_ratio=0.35,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(rejected, [])

    def test_rejects_alternate_with_detail_damage(self):
        selected, rejected = MODULE.select_repair_windows(
            [quality_row(direct_votes=4, issues=[MODULE.VOCAL_ISSUE], delta=0.0)],
            [
                quality_row(
                    direct_votes=2,
                    issues=[MODULE.VOCAL_ISSUE, "presence_loss_proxy"],
                    delta=0.2,
                )
            ],
            min_direct_vote_reduction=1,
            max_alternate_delta_rms_ratio=0.35,
        )
        self.assertEqual(selected, [])
        self.assertIn("alternate_has_confirmed_detail_damage", rejected[0]["rejection_reasons"])

    def test_rms_match_removes_local_level_hole_without_touching_later_audio(self):
        sample_rate = 1000
        base = np.full((2000, 2), 0.4, dtype=np.float32)
        alternate = np.full((2000, 2), 0.2, dtype=np.float32)
        rendered, records = MODULE.render_candidate(
            base,
            alternate,
            sample_rate=sample_rate,
            windows=[{"segment_id": "s0000", "window_sec": [0.0, 1.0]}],
            blend=1.0,
            gain_mode="window_rms_match",
            max_gain_db=6.1,
            crossfade_ms=0,
        )
        self.assertAlmostEqual(float(np.sqrt(np.mean(rendered[:1000] ** 2))), 0.4, places=5)
        np.testing.assert_allclose(rendered[1000:], base[1000:])
        self.assertAlmostEqual(records[0]["applied_gain_db"], 6.0206, places=3)

    def test_start_boundary_does_not_fade_back_to_leaky_base(self):
        base = np.ones((1000, 1), dtype=np.float32)
        alternate = np.zeros((1000, 1), dtype=np.float32)
        rendered, _ = MODULE.render_candidate(
            base,
            alternate,
            sample_rate=1000,
            windows=[{"segment_id": "s0000", "window_sec": [0.0, 0.5]}],
            blend=1.0,
            gain_mode="none",
            max_gain_db=3.0,
            crossfade_ms=80,
        )
        self.assertEqual(float(rendered[0, 0]), 0.0)
        self.assertEqual(float(rendered[499, 0]), 1.0)
        self.assertEqual(float(rendered[500, 0]), 1.0)

    def test_end_to_end_matrix_keeps_delivery_blocked_and_short_names(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            artifacts = root / "artifacts"
            source_path = artifacts / "src.flac"
            base_path = artifacts / "base.flac"
            alternate_path = artifacts / "alt.flac"
            sample_rate = 1000
            t = np.arange(2000, dtype=np.float32) / sample_rate
            source = np.stack([0.4 * np.sin(2 * np.pi * 10 * t)] * 2, axis=1)
            base = source.copy()
            alternate = source.copy()
            alternate[:1000] *= 0.6
            artifacts.mkdir(parents=True)
            sf.write(source_path, source, sample_rate, subtype="PCM_24")
            sf.write(base_path, base, sample_rate, subtype="PCM_24")
            sf.write(alternate_path, alternate, sample_rate, subtype="PCM_24")
            proxy_rows = [
                {"valid": True, "model_id": "p1"},
                {"valid": True, "model_id": "p2"},
            ]
            manifests = []
            for name, candidate_path, row in (
                ("bq", base_path, quality_row(direct_votes=4, issues=[MODULE.VOCAL_ISSUE], delta=0.0)),
                ("aq", alternate_path, quality_row(direct_votes=3, issues=[MODULE.VOCAL_ISSUE], delta=0.24)),
            ):
                quality_dir = artifacts / name
                quality_dir.mkdir()
                rows_path = quality_dir / "rows.jsonl"
                rows_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                manifest_path = quality_dir / "manifest.json"
                manifest_path.write_text(
                    json.dumps(
                        {
                            "schema": "minus_quality_ensemble.v2",
                            "source": {"sha256": MODULE.sha256_file(source_path)},
                            "candidate_minus": {"sha256": MODULE.sha256_file(candidate_path)},
                            "proxies": proxy_rows,
                            "artifacts": {"rows": str(rows_path)},
                        }
                    ),
                    encoding="utf-8",
                )
                manifests.append(manifest_path)

            result = MODULE.build_quality_guided_minus_matrix(
                source_audio_path=source_path,
                base_minus_path=base_path,
                alternate_minus_path=alternate_path,
                base_quality_manifest_path=manifests[0],
                alternate_quality_manifest_path=manifests[1],
                output_dir=artifacts / "out",
                runtime_project_root=root,
                windows_project_root="D:\\AIMusicProject",
                blends=(0.5, 1.0),
                gain_modes=("none", "window_rms_match"),
            )
            self.assertEqual(result["matrix"]["candidate_count"], 4)
            self.assertFalse(result["delivery_contract"]["delivery_allowed"])
            self.assertEqual(
                [Path(row["path"]).name for row in result["matrix"]["candidates"]],
                ["c001.flac", "c002.flac", "c003.flac", "c004.flac"],
            )


if __name__ == "__main__":
    unittest.main()
