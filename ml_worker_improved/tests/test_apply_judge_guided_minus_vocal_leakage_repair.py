from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "apply_judge_guided_minus_vocal_leakage_repair.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "apply_judge_guided_minus_vocal_leakage_repair", MODULE_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class JudgeGuidedMinusVocalLeakageRepairTests(unittest.TestCase):
    def test_selects_only_judge_windows_with_minus_vocal_bleed_blocker(self) -> None:
        module = _load_module()
        rows = [
            {
                "candidate_id": "base_minus",
                "minus_path": "/app/base.flac",
                "window_metrics": [
                    {
                        "start_sec": 0.0,
                        "duration_sec": 4.0,
                        "vocal_leakage_in_minus_score": 0.25,
                        "minus_blockers": [],
                    },
                    {
                        "start_sec": 8.0,
                        "duration_sec": 4.0,
                        "vocal_leakage_in_minus_score": 0.79,
                        "minus_blockers": ["severe_vocal_bleed_in_minus"],
                    },
                ],
            }
        ]

        windows = module.select_target_windows(
            rows,
            candidate_id="base_minus",
            minus_path="/app/base.flac",
            leak_threshold=0.60,
        )

        self.assertEqual(
            windows,
            [
                {
                    "start_sec": 8.0,
                    "duration_sec": 4.0,
                    "vocal_leakage_in_minus_score": 0.79,
                    "minus_blockers": ["severe_vocal_bleed_in_minus"],
                }
            ],
        )

    def test_selects_self_judge_vocal_projection_blocker_windows(self) -> None:
        module = _load_module()
        rows = [
            {
                "segment_id": "segment_0000",
                "window_sec": [0.0, 6.0],
                "candidate_label": "base_minus__judge_guided_leakage_repair",
                "issues": ["vocal_projection_proxy"],
                "issue_severities": {"vocal_projection": 0.21},
                "blocker": True,
                "severity_score": 0.21,
            },
            {
                "segment_id": "segment_0001",
                "window_sec": [6.0, 12.0],
                "candidate_label": "base_minus__judge_guided_leakage_repair",
                "issues": [],
                "issue_severities": {},
                "blocker": False,
                "severity_score": 0.0,
            },
            {
                "segment_id": "segment_0002",
                "window_sec": [12.0, 18.0],
                "candidate_label": "base_minus__judge_guided_leakage_repair",
                "issues": ["presence_loss_proxy"],
                "issue_severities": {"presence_loss": 0.18},
                "blocker": True,
                "severity_score": 0.18,
            },
        ]

        windows = module.select_target_windows(
            rows,
            candidate_id="base_minus__judge_guided_leakage_repair",
            minus_path="/app/base.flac",
            leak_threshold=0.60,
        )

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["start_sec"], 0.0)
        self.assertEqual(windows[0]["duration_sec"], 6.0)
        self.assertEqual(windows[0]["target_window_source"], "instrumental_self_judge_gate")
        self.assertEqual(windows[0]["vocal_leakage_issue_score"], 0.21)

    def test_read_judge_rows_accepts_self_judge_manifest_json(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "instrumental_self_judge_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "mode": "instrumental_self_judge_gate",
                        "candidate": {
                            "label": "base_minus",
                            "instrumental_path": "/app/base.flac",
                            "vocal_path": "/app/vocals.flac",
                        },
                        "segments": [
                            {
                                "segment_id": "segment_0013",
                                "window_sec": [156.0, 168.0],
                                "issues": ["vocal_projection_proxy"],
                                "issue_severities": {"vocal_projection": 0.080557},
                                "blocker": True,
                                "severity_score": 0.080557,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            rows = module.read_judge_rows(manifest_path)
            windows = module.select_target_windows(
                rows,
                candidate_id="base_minus",
                minus_path="/app/base.flac",
                leak_threshold=0.01,
            )

        self.assertEqual(rows[0]["candidate_label"], "base_minus")
        self.assertEqual(rows[0]["instrumental_path"], "/app/base.flac")
        self.assertEqual(rows[0]["minus_path"], "/app/base.flac")
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["start_sec"], 156.0)
        self.assertEqual(windows[0]["duration_sec"], 12.0)
        self.assertEqual(windows[0]["segment_id"], "segment_0013")

    def test_projection_repair_reduces_vocal_projection_without_touching_unselected_audio(self) -> None:
        module = _load_module()
        sample_rate = 8000
        times = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
        vocal = np.sin(2.0 * np.pi * 440.0 * times).astype(np.float32)
        music = 0.45 * np.sin(2.0 * np.pi * 110.0 * times).astype(np.float32)
        base_minus = music + 0.50 * vocal
        base_minus[times >= 1.0] = music[times >= 1.0]
        stereo_base = np.column_stack([base_minus, base_minus])
        stereo_vocal = np.column_stack([vocal, vocal])
        gate = module.build_time_gate(
            frame_count=stereo_base.shape[0],
            sample_rate=sample_rate,
            windows=[{"start_sec": 0.0, "duration_sec": 1.0}],
            margin_sec=0.0,
            smoothing_sec=0.0,
        )

        repaired, metrics = module.apply_projection_repair(
            base_minus=stereo_base,
            vocal_audio=stereo_vocal,
            sample_rate=sample_rate,
            gate=gate,
            strength=0.85,
            max_projection_gain=0.75,
            low_hz=160.0,
            high_hz=2000.0,
        )

        before = module.projection_gain(stereo_base[:sample_rate, 0], stereo_vocal[:sample_rate, 0])
        after = module.projection_gain(repaired[:sample_rate, 0], stereo_vocal[:sample_rate, 0])
        unchanged_delta = np.max(np.abs(repaired[sample_rate:] - stereo_base[sample_rate:]))

        self.assertLess(after, before * 0.45)
        self.assertLess(unchanged_delta, 1.0e-5)
        self.assertGreater(metrics["projection_gain_before"], metrics["projection_gain_after"])
        self.assertLessEqual(metrics["applied_projection_gain"], 0.75)

    def test_segment_local_projection_repair_handles_opposite_signed_windows(self) -> None:
        module = _load_module()
        sample_rate = 8000
        times = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
        vocal = np.sin(2.0 * np.pi * 440.0 * times).astype(np.float32)
        music = 0.35 * np.sin(2.0 * np.pi * 110.0 * times).astype(np.float32)
        base_minus = music.copy()
        base_minus[:sample_rate] += 0.50 * vocal[:sample_rate]
        base_minus[sample_rate:] -= 0.50 * vocal[sample_rate:]
        stereo_base = np.column_stack([base_minus, base_minus])
        stereo_vocal = np.column_stack([vocal, vocal])

        repaired, metrics = module.apply_segment_local_projection_repair(
            base_minus=stereo_base,
            vocal_audio=stereo_vocal,
            sample_rate=sample_rate,
            windows=[
                {"start_sec": 0.0, "duration_sec": 1.0},
                {"start_sec": 1.0, "duration_sec": 1.0},
            ],
            margin_sec=0.0,
            smoothing_sec=0.0,
            strength=1.0,
            max_projection_gain=0.75,
            low_hz=160.0,
            high_hz=2000.0,
        )

        first_before = module.projection_gain(stereo_base[:sample_rate, 0], stereo_vocal[:sample_rate, 0])
        first_after = module.projection_gain(repaired[:sample_rate, 0], stereo_vocal[:sample_rate, 0])
        second_before = module.projection_gain(stereo_base[sample_rate:, 0], stereo_vocal[sample_rate:, 0])
        second_after = module.projection_gain(repaired[sample_rate:, 0], stereo_vocal[sample_rate:, 0])

        self.assertLess(first_after, first_before * 0.15)
        self.assertLess(second_after, second_before * 0.15)
        self.assertEqual(metrics["projection_mode"], "segment_local")
        self.assertEqual(metrics["window_projection_count"], 2)
        self.assertEqual(metrics["status"], "segment_local_projection_repair_applied")

    def test_segment_local_projection_repair_allows_high_gain_for_low_energy_guide(self) -> None:
        module = _load_module()
        sample_rate = 8000
        times = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
        true_vocal = np.sin(2.0 * np.pi * 440.0 * times).astype(np.float32)
        guide_vocal = 0.01 * true_vocal
        music = 0.35 * np.sin(2.0 * np.pi * 110.0 * times).astype(np.float32)
        base_minus = music.copy()
        base_minus[:sample_rate] += 0.35 * true_vocal[:sample_rate]
        stereo_base = np.column_stack([base_minus, base_minus])
        stereo_guide = np.column_stack([guide_vocal, guide_vocal])

        repaired, metrics = module.apply_segment_local_projection_repair(
            base_minus=stereo_base,
            vocal_audio=stereo_guide,
            sample_rate=sample_rate,
            windows=[{"start_sec": 0.0, "duration_sec": 1.0}],
            margin_sec=0.0,
            smoothing_sec=0.0,
            strength=1.0,
            max_projection_gain=64.0,
            low_hz=160.0,
            high_hz=2000.0,
        )

        first_before = module.projection_gain(stereo_base[:sample_rate, 0], stereo_guide[:sample_rate, 0])
        first_after = module.projection_gain(repaired[:sample_rate, 0], stereo_guide[:sample_rate, 0])
        unchanged_delta = np.max(np.abs(repaired[sample_rate:] - stereo_base[sample_rate:]))

        self.assertLess(first_after, first_before * 0.10)
        self.assertLess(unchanged_delta, 1.0e-5)
        self.assertGreater(metrics["applied_projection_gain"], 1.0)
        self.assertEqual(metrics["projection_mode"], "segment_local")

    def test_segment_local_peak_limit_does_not_scale_unselected_audio(self) -> None:
        module = _load_module()
        sample_rate = 8000
        times = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
        vocal = np.sin(2.0 * np.pi * 440.0 * times).astype(np.float32)
        music = 0.25 * np.sin(2.0 * np.pi * 110.0 * times).astype(np.float32)
        base_minus = music.copy()
        base_minus[:sample_rate] = 1.20 * vocal[:sample_rate]
        stereo_base = np.column_stack([base_minus, base_minus])
        stereo_vocal = np.column_stack([vocal, vocal])

        repaired, _metrics = module.apply_segment_local_projection_repair(
            base_minus=stereo_base,
            vocal_audio=stereo_vocal,
            sample_rate=sample_rate,
            windows=[{"start_sec": 0.0, "duration_sec": 1.0}],
            margin_sec=0.0,
            smoothing_sec=0.0,
            strength=0.0,
            max_projection_gain=64.0,
            low_hz=160.0,
            high_hz=2000.0,
        )

        unchanged_delta = np.max(np.abs(repaired[sample_rate:] - stereo_base[sample_rate:]))

        self.assertLess(unchanged_delta, 1.0e-5)

    def test_manifest_is_candidate_only_and_not_delivery_acceptance(self) -> None:
        module = _load_module()

        manifest = module.build_manifest(
            track_id="track-a",
            candidate_id="base_minus__judge_guided_leakage_repair",
            raw_audio_path=Path("/app/raw.flac"),
            base_minus_path=Path("/app/base.flac"),
            primary_vocals_path=Path("/app/vocals.flac"),
            judge_jsonl_path=Path("/app/judge.jsonl"),
            output_path=Path("/app/out/minus_judge_guided_leakage_repair.flac"),
            candidate_row_jsonl=Path("/app/out/candidate_pair.jsonl"),
            sample_rate=44100,
            frames=44100,
            selected_windows=[{"start_sec": 0.0, "duration_sec": 1.0}],
            repair_metrics={"projection_gain_before": 0.5, "projection_gain_after": 0.2},
        )

        self.assertEqual(manifest["delivery_status"], "candidate_only")
        self.assertFalse(manifest["delivery_allowed"])
        self.assertEqual(manifest["presentation_status"], "blocked_until_judged")
        self.assertEqual(manifest["release_status"], "not_release_ready")
        self.assertFalse(manifest["safety"]["thresholds_relaxed"])
        self.assertFalse(manifest["safety"]["training_started"])
        self.assertEqual(manifest["candidate_id"], "base_minus__judge_guided_leakage_repair")
        self.assertEqual(manifest["candidate_row_jsonl"], "/app/out/candidate_pair.jsonl")
        self.assertEqual(
            manifest["candidate_row"]["instrumental_path"],
            "/app/out/minus_judge_guided_leakage_repair.flac",
        )
        self.assertEqual(manifest["candidate_row"]["minus_path"], "/app/out/minus_judge_guided_leakage_repair.flac")
        self.assertEqual(manifest["candidate_row"]["vocal_path"], "/app/vocals.flac")


if __name__ == "__main__":
    unittest.main()
