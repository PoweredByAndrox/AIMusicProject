from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "apply_judge_guided_minus_air_detail_repair.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "apply_judge_guided_minus_air_detail_repair", MODULE_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class JudgeGuidedMinusAirDetailRepairTests(unittest.TestCase):
    def test_selects_only_safe_air_detail_self_judge_windows(self) -> None:
        module = _load_module()
        rows = [
            {
                "segment_id": "segment_0000",
                "candidate_label": "minus-a",
                "window_sec": [0.0, 6.0],
                "blocker": True,
                "issues": ["air_loss_proxy"],
                "issue_severities": {"air_loss": 1.2},
                "metrics": {
                    "air_ratio_db_vs_source": -26.0,
                    "vocal_projection_db_vs_instrumental": -240.0,
                },
            },
            {
                "segment_id": "segment_0001",
                "candidate_label": "minus-a",
                "window_sec": [6.0, 12.0],
                "blocker": True,
                "issues": ["air_loss_proxy", "vocal_projection_proxy"],
                "issue_severities": {"air_loss": 1.0, "vocal_projection": 0.2},
            },
            {
                "segment_id": "segment_0002",
                "candidate_label": "minus-b",
                "window_sec": [12.0, 18.0],
                "blocker": True,
                "issues": ["presence_loss_proxy"],
                "issue_severities": {"presence_loss": 0.3},
            },
        ]

        windows = module.select_target_windows(
            rows,
            candidate_id="minus-a",
            minus_path="/app/minus-a.flac",
        )

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["segment_id"], "segment_0000")
        self.assertEqual(windows[0]["start_sec"], 0.0)
        self.assertEqual(windows[0]["duration_sec"], 6.0)
        self.assertEqual(windows[0]["target_window_source"], "instrumental_self_judge_gate")
        self.assertEqual(windows[0]["detail_issue_score"], 1.2)

    def test_air_detail_repair_restores_selected_high_band_without_touching_other_audio(self) -> None:
        module = _load_module()
        sample_rate = 8000
        times = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
        body = 0.25 * np.sin(2.0 * np.pi * 220.0 * times).astype(np.float32)
        missing_air = 0.12 * np.sin(2.0 * np.pi * 3000.0 * times).astype(np.float32)
        source = body.copy()
        source[:sample_rate] += missing_air[:sample_rate]
        base_minus = body.copy()
        stereo_source = np.column_stack([source, source])
        stereo_base = np.column_stack([base_minus, base_minus])

        repaired, metrics = module.apply_air_detail_repair(
            source_audio=stereo_source,
            base_minus=stereo_base,
            vocal_audio=None,
            sample_rate=sample_rate,
            windows=[
                {
                    "start_sec": 0.0,
                    "duration_sec": 1.0,
                    "metrics": {"rms_ratio": 0.7},
                }
            ],
            margin_sec=0.0,
            smoothing_sec=0.0,
            presence_delta_strength=0.0,
            air_delta_strength=1.0,
            body_delta_strength=0.0,
            base_gain_strength=0.0,
            low_rms_target_ratio=0.52,
            max_base_gain=1.0,
            vocal_suppression_strength=0.0,
            max_vocal_projection_gain=0.0,
            presence_low_hz=1200.0,
            presence_high_hz=2200.0,
            air_low_hz=2500.0,
            air_high_hz=3600.0,
            body_low_hz=120.0,
            body_high_hz=900.0,
        )

        before_air = module.rms(module.bandpass_audio(stereo_base[:sample_rate], sample_rate, low_hz=2500.0, high_hz=3600.0))
        after_air = module.rms(module.bandpass_audio(repaired[:sample_rate], sample_rate, low_hz=2500.0, high_hz=3600.0))
        untouched_delta = np.max(np.abs(repaired[sample_rate:] - stereo_base[sample_rate:]))

        self.assertGreater(after_air, before_air * 5.0)
        self.assertLess(untouched_delta, 1.0e-5)
        self.assertEqual(metrics["status"], "air_detail_repair_applied")
        self.assertEqual(metrics["selected_window_count"], 1)

    def test_vocal_guide_suppresses_added_detail_projection(self) -> None:
        module = _load_module()
        sample_rate = 8000
        times = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
        vocal_like_air = np.sin(2.0 * np.pi * 3000.0 * times).astype(np.float32)
        source = 0.20 * vocal_like_air
        base_minus = np.zeros_like(source)
        stereo_source = np.column_stack([source, source])
        stereo_base = np.column_stack([base_minus, base_minus])
        stereo_vocal = np.column_stack([vocal_like_air, vocal_like_air])
        windows = [{"start_sec": 0.0, "duration_sec": 1.0, "metrics": {"rms_ratio": 0.0}}]

        unsuppressed, _ = module.apply_air_detail_repair(
            source_audio=stereo_source,
            base_minus=stereo_base,
            vocal_audio=None,
            sample_rate=sample_rate,
            windows=windows,
            margin_sec=0.0,
            smoothing_sec=0.0,
            presence_delta_strength=0.0,
            air_delta_strength=1.0,
            body_delta_strength=0.0,
            base_gain_strength=0.0,
            low_rms_target_ratio=0.52,
            max_base_gain=1.0,
            vocal_suppression_strength=0.0,
            max_vocal_projection_gain=0.0,
            presence_low_hz=1200.0,
            presence_high_hz=2200.0,
            air_low_hz=2500.0,
            air_high_hz=3600.0,
            body_low_hz=120.0,
            body_high_hz=900.0,
        )
        suppressed, metrics = module.apply_air_detail_repair(
            source_audio=stereo_source,
            base_minus=stereo_base,
            vocal_audio=stereo_vocal,
            sample_rate=sample_rate,
            windows=windows,
            margin_sec=0.0,
            smoothing_sec=0.0,
            presence_delta_strength=0.0,
            air_delta_strength=1.0,
            body_delta_strength=0.0,
            base_gain_strength=0.0,
            low_rms_target_ratio=0.52,
            max_base_gain=1.0,
            vocal_suppression_strength=1.0,
            max_vocal_projection_gain=2.0,
            presence_low_hz=1200.0,
            presence_high_hz=2200.0,
            air_low_hz=2500.0,
            air_high_hz=3600.0,
            body_low_hz=120.0,
            body_high_hz=900.0,
        )

        before = module.projection_gain(unsuppressed, stereo_vocal)
        after = module.projection_gain(suppressed, stereo_vocal)

        self.assertLess(after, before * 0.25)
        self.assertGreater(metrics["vocal_projection_gain_removed"], 0.0)

    def test_manifest_is_candidate_only_and_requires_post_judge(self) -> None:
        module = _load_module()

        manifest = module.build_manifest(
            track_id="track-a",
            candidate_id="minus-a__air_detail_repair",
            raw_audio_path=Path("/app/raw.flac"),
            base_minus_path=Path("/app/base.flac"),
            vocal_guide_path=Path("/app/vocal.flac"),
            judge_jsonl_path=Path("/app/self_judge_segments.jsonl"),
            output_path=Path("/app/out/minus_air_detail_repair.flac"),
            candidate_row_jsonl=Path("/app/out/candidate_pair.jsonl"),
            sample_rate=44100,
            frames=44100,
            selected_windows=[{"start_sec": 0.0, "duration_sec": 1.0}],
            repair_metrics={"status": "air_detail_repair_applied"},
        )

        self.assertEqual(manifest["delivery_status"], "candidate_only")
        self.assertFalse(manifest["delivery_allowed"])
        self.assertEqual(manifest["presentation_status"], "blocked_until_judged")
        self.assertEqual(manifest["release_status"], "not_release_ready")
        self.assertTrue(manifest["requires_post_judge"])
        self.assertEqual(manifest["render_status"], "rendered_minus_air_detail_repair_candidate")
        self.assertEqual(
            manifest["candidate_row"]["render_status"],
            "rendered_minus_air_detail_repair_candidate",
        )
        self.assertEqual(
            manifest["rendered_minus_candidate_path"],
            "/app/out/minus_air_detail_repair.flac",
        )
        self.assertFalse(manifest["safety"]["training_started"])
        self.assertFalse(manifest["safety"]["thresholds_relaxed"])
        self.assertEqual(
            manifest["candidate_row"]["candidate_family"],
            "judge_guided_minus_air_detail_repair",
        )


if __name__ == "__main__":
    unittest.main()
