from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import soundfile as sf


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

    def test_selects_residual_vocal_watch_windows_above_repair_threshold(self) -> None:
        module = _load_module()
        rows = [
            {
                "candidate_id": "warned_minus",
                "minus_path": "/app/warned.flac",
                "window_metrics": [
                    {
                        "start_sec": 12.0,
                        "duration_sec": 4.0,
                        "vocal_leakage_in_minus_score": 0.17,
                        "reasons": ["residual_vocal_watch"],
                    },
                    {
                        "start_sec": 24.0,
                        "duration_sec": 4.0,
                        "vocal_leakage_in_minus_score": 0.22,
                        "reasons": ["residual_vocal_watch"],
                    },
                ],
            }
        ]

        windows = module.select_target_windows(
            rows,
            candidate_id="warned_minus",
            minus_path="/app/warned.flac",
            leak_threshold=0.18,
        )

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["start_sec"], 24.0)
        self.assertEqual(windows[0]["duration_sec"], 4.0)
        self.assertEqual(windows[0]["reasons"], ["residual_vocal_watch"])

    def test_selects_minus_dominant_vocal_bleed_windows_above_repair_threshold(self) -> None:
        module = _load_module()
        rows = [
            {
                "candidate_id": "warned_minus",
                "minus_path": "/app/warned.flac",
                "window_metrics": [
                    {
                        "start_sec": 168.0,
                        "duration_sec": 12.0,
                        "vocal_leakage_in_minus_score": 0.49,
                        "minus_dominant_defect": "vocal_bleed_in_minus",
                        "reasons": ["no_reference_proxy_passed"],
                    }
                ],
            }
        ]

        windows = module.select_target_windows(
            rows,
            candidate_id="warned_minus",
            minus_path="/app/warned.flac",
            leak_threshold=0.15,
        )

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["start_sec"], 168.0)
        self.assertEqual(windows[0]["minus_dominant_defect"], "vocal_bleed_in_minus")

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

    def test_read_judge_rows_accepts_windowed_judge_jsonl_starting_with_object(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            judge_path = Path(tmp_dir) / "wj.jsonl"
            judge_path.write_text(
                json.dumps({"candidate_id": "minus", "minus_path": "/app/minus.flac"})
                + "\n"
                + json.dumps({"candidate_id": "alt", "minus_path": "/app/alt.flac"})
                + "\n",
                encoding="utf-8",
            )

            rows = module.read_judge_rows(judge_path)

        self.assertEqual([row["candidate_id"] for row in rows], ["minus", "alt"])

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
        self.assertEqual(metrics["memory_strategy"], "window_sliced_float32")
        self.assertFalse(metrics["full_track_gate_per_window"])
        self.assertLess(metrics["max_local_working_frames"], stereo_base.shape[0])

    def test_window_gate_slice_matches_full_track_gate(self) -> None:
        module = _load_module()
        sample_rate = 8000
        frame_count = sample_rate * 5
        window = {"start_sec": 2.0, "duration_sec": 1.0}

        full_gate = module.build_time_gate(
            frame_count=frame_count,
            sample_rate=sample_rate,
            windows=[window],
            margin_sec=0.04,
            smoothing_sec=0.03,
        )
        window_slice, local_gate = module.build_window_gate_slice(
            frame_count=frame_count,
            sample_rate=sample_rate,
            window=window,
            margin_sec=0.04,
            smoothing_sec=0.03,
        )
        reconstructed = np.zeros(frame_count, dtype=np.float32)
        reconstructed[window_slice] = local_gate

        np.testing.assert_allclose(reconstructed, full_gate, rtol=0.0, atol=1.0e-7)

    def test_segment_local_projection_bounds_filter_work_to_local_slice(self) -> None:
        module = _load_module()
        sample_rate = 8000
        times = np.arange(sample_rate * 20, dtype=np.float32) / float(sample_rate)
        vocal = np.sin(2.0 * np.pi * 440.0 * times).astype(np.float32)
        music = 0.25 * np.sin(2.0 * np.pi * 110.0 * times).astype(np.float32)
        stereo_base = np.column_stack([music + (0.25 * vocal), music + (0.25 * vocal)])
        stereo_vocal = np.column_stack([vocal, vocal])

        _repaired, metrics = module.apply_segment_local_projection_repair(
            base_minus=stereo_base,
            vocal_audio=stereo_vocal,
            sample_rate=sample_rate,
            windows=[
                {"start_sec": 2.0, "duration_sec": 1.0},
                {"start_sec": 15.0, "duration_sec": 1.0},
            ],
            margin_sec=0.04,
            smoothing_sec=0.03,
            strength=0.85,
            max_projection_gain=0.75,
            low_hz=160.0,
            high_hz=2000.0,
        )

        self.assertLess(metrics["max_local_working_frames"], sample_rate * 2)
        self.assertLess(metrics["max_local_working_frames"], stereo_base.shape[0] // 5)

    def test_local_bandpass_and_segment_repair_match_full_track_reference(self) -> None:
        module = _load_module()
        sample_rate = 8000
        times = np.arange(sample_rate * 6, dtype=np.float32) / float(sample_rate)
        envelope = (0.55 + (0.35 * np.sin(2.0 * np.pi * 0.7 * times))).astype(np.float32)
        vocal = envelope * (
            np.sin(2.0 * np.pi * 440.0 * times)
            + (0.25 * np.sin(2.0 * np.pi * 920.0 * times))
        ).astype(np.float32)
        music = (
            0.25 * np.sin(2.0 * np.pi * 110.0 * times)
            + 0.12 * np.sin(2.0 * np.pi * 1850.0 * times)
        ).astype(np.float32)
        stereo_vocal = np.column_stack([vocal, vocal * 0.92])
        stereo_base = np.column_stack([music, music * 0.96]) + (0.32 * stereo_vocal)
        windows = [
            {"start_sec": 0.02, "duration_sec": 0.72},
            {"start_sec": 2.00, "duration_sec": 0.90},
            {"start_sec": 2.45, "duration_sec": 0.85},
            {"start_sec": 5.15, "duration_sec": 0.80},
        ]

        full_guide = module.bandpass_audio(
            stereo_vocal,
            sample_rate,
            low_hz=160.0,
            high_hz=2000.0,
        )
        reference = stereo_base.astype(np.float32, copy=True)
        combined_gate = np.zeros(reference.shape[0], dtype=np.float32)
        for window in windows:
            gate = module.build_time_gate(
                frame_count=reference.shape[0],
                sample_rate=sample_rate,
                windows=[window],
                margin_sec=0.04,
                smoothing_sec=0.03,
            )
            active = gate > 1.0e-6
            np.maximum(combined_gate, gate, out=combined_gate)
            for channel_index in range(reference.shape[1]):
                gain = module.signed_projection_gain(
                    reference[active, channel_index],
                    full_guide[active, channel_index],
                    max_projection_gain=0.75,
                )
                reference[:, channel_index] -= (
                    0.85 * gain * full_guide[:, channel_index] * gate
                ).astype(np.float32)
        reference, _peak = module.limit_active_region_peaks(
            reference,
            combined_gate > 1.0e-6,
            copy=False,
        )

        repaired, _metrics = module.apply_segment_local_projection_repair(
            base_minus=stereo_base,
            vocal_audio=stereo_vocal,
            sample_rate=sample_rate,
            windows=windows,
            margin_sec=0.04,
            smoothing_sec=0.03,
            strength=0.85,
            max_projection_gain=0.75,
            low_hz=160.0,
            high_hz=2000.0,
        )

        error = np.asarray(repaired - reference, dtype=np.float64)
        signal_energy = float(np.sum(np.square(np.asarray(reference, dtype=np.float64))))
        error_energy = float(np.sum(np.square(error)))
        snr_db = 10.0 * np.log10(signal_energy / max(error_energy, 1.0e-20))
        self.assertGreater(snr_db, 55.0)

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
        self.assertEqual(manifest["execution_contract"]["runtime_environment"], "docker")
        self.assertFalse(manifest["execution_contract"]["native_runtime_allowed"])
        self.assertEqual(manifest["candidate_id"], "base_minus__judge_guided_leakage_repair")
        self.assertEqual(manifest["candidate_row_jsonl"], "/app/out/candidate_pair.jsonl")
        self.assertEqual(
            manifest["candidate_row"]["instrumental_path"],
            "/app/out/minus_judge_guided_leakage_repair.flac",
        )
        self.assertEqual(manifest["candidate_row"]["minus_path"], "/app/out/minus_judge_guided_leakage_repair.flac")
        self.assertEqual(manifest["candidate_row"]["vocal_path"], "/app/vocals.flac")

    def test_build_repair_can_write_short_output_names(self) -> None:
        module = _load_module()
        sample_rate = 8000
        times = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
        vocal = 0.25 * np.sin(2.0 * np.pi * 440.0 * times).astype(np.float32)
        music = 0.35 * np.sin(2.0 * np.pi * 110.0 * times).astype(np.float32)
        raw = np.column_stack([music + vocal, music + vocal])
        base_minus = np.column_stack([music + (0.45 * vocal), music + (0.45 * vocal)])
        primary_vocals = np.column_stack([vocal, vocal])

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_path = root / "mix.flac"
            base_path = root / "base.flac"
            vocal_path = root / "voc.flac"
            sf.write(raw_path, raw, sample_rate)
            sf.write(base_path, base_minus, sample_rate)
            sf.write(vocal_path, primary_vocals, sample_rate)
            judge_path = root / "wj.jsonl"
            judge_path.write_text(
                json.dumps(
                    {
                        "candidate_id": "base",
                        "minus_path": str(base_path),
                        "window_metrics": [
                            {
                                "start_sec": 0.0,
                                "duration_sec": 1.0,
                                "vocal_leakage_in_minus_score": 0.7,
                                "reasons": ["vocal_leakage_in_minus"],
                            }
                        ],
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "out"

            original_build_time_gate = module.build_time_gate

            def fail_full_track_gate(**_kwargs):
                raise AssertionError("segment_local orchestration must not allocate a full-track gate")

            module.build_time_gate = fail_full_track_gate
            try:
                manifest = module.build_judge_guided_minus_vocal_leakage_repair(
                    Namespace(
                        raw_audio_path=raw_path,
                        base_minus_path=base_path,
                        primary_vocals_path=vocal_path,
                        windowed_judge_jsonl=judge_path,
                        output_dir=output_dir,
                        candidate_id="base",
                        track_id="track-a",
                        listen_dir=None,
                        listen_label="m001.flac",
                        leak_threshold=0.60,
                        strength=0.85,
                        max_projection_gain=0.75,
                        gate_smoothing_sec=0.0,
                        window_margin_sec=0.0,
                        band_low_hz=160.0,
                        band_high_hz=2000.0,
                        projection_mode="segment_local",
                        short_output_names=True,
                    )
                )
            finally:
                module.build_time_gate = original_build_time_gate

            self.assertTrue((output_dir / "manifest.json").exists())
            self.assertTrue((output_dir / "rows.jsonl").exists())
            self.assertTrue((output_dir / "m001.flac").exists())
            self.assertFalse((output_dir / "judge_guided_minus_vocal_leakage_repair_manifest.json").exists())
            self.assertFalse((output_dir / "minus_judge_guided_leakage_repair.flac").exists())
            self.assertEqual(manifest["candidate_row_jsonl"], str(output_dir / "rows.jsonl"))
            self.assertEqual(manifest["rendered_minus_candidate_path"], str(output_dir / "m001.flac"))
            self.assertTrue(all(len(path.name) <= 32 for path in output_dir.iterdir()))

    def test_no_matching_repair_windows_writes_blocked_short_manifest(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            base_path = root / "base.flac"
            judge_path = root / "wj.jsonl"
            judge_path.write_text(
                json.dumps(
                    {
                        "candidate_id": "base",
                        "minus_path": str(base_path),
                        "window_metrics": [
                            {
                                "start_sec": 0.0,
                                "duration_sec": 1.0,
                                "vocal_leakage_in_minus_score": 0.4,
                                "reasons": ["silence_hallucination"],
                            }
                        ],
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "out"

            manifest = module.build_judge_guided_minus_vocal_leakage_repair(
                Namespace(
                    raw_audio_path=root / "mix.flac",
                    base_minus_path=base_path,
                    primary_vocals_path=root / "voc.flac",
                    windowed_judge_jsonl=judge_path,
                    output_dir=output_dir,
                    candidate_id="base",
                    track_id="track-a",
                    listen_dir=None,
                    listen_label="m001.flac",
                    leak_threshold=0.15,
                    strength=0.85,
                    max_projection_gain=0.75,
                    gate_smoothing_sec=0.0,
                    window_margin_sec=0.0,
                    band_low_hz=160.0,
                    band_high_hz=2000.0,
                    projection_mode="segment_local",
                    short_output_names=True,
                )
            )

            self.assertEqual(manifest["delivery_status"], "blocked")
            self.assertFalse(manifest["delivery_allowed"])
            self.assertEqual(manifest["failed_checks"], ["no_matching_repair_windows"])
            self.assertEqual(manifest["repair"]["selected_window_count"], 0)
            self.assertEqual(manifest["manifest_path"], str(output_dir / "manifest.json"))
            self.assertEqual(manifest["candidate_row_jsonl"], str(output_dir / "rows.jsonl"))
            self.assertTrue((output_dir / "manifest.json").exists())
            self.assertTrue((output_dir / "rows.jsonl").exists())
            self.assertTrue((output_dir / "summary.md").exists())
            self.assertFalse((output_dir / "m001.flac").exists())
            self.assertEqual((output_dir / "rows.jsonl").read_text(encoding="utf-8"), "")

    def test_cli_allows_explicit_native_runtime_and_records_blocked_contract(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            judge_path = root / "wj.jsonl"
            judge_path.write_text(
                json.dumps(
                    {
                        "candidate_id": "base",
                        "minus_path": str(root / "base.flac"),
                        "window_metrics": [
                            {
                                "start_sec": 0.0,
                                "duration_sec": 1.0,
                                "vocal_leakage_in_minus_score": 0.4,
                                "reasons": ["silence_hallucination"],
                            }
                        ],
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "out"
            calls: list[dict] = []

            def fake_require_docker_runtime(**kwargs):
                calls.append(kwargs)

            original_guard = module.require_docker_runtime
            original_contract = module.cli_runtime_contract
            module.require_docker_runtime = fake_require_docker_runtime
            module.cli_runtime_contract = lambda *, allow_native_runtime: {
                "runtime_environment": "native_mac_cpu_dsp",
                "native_runtime_allowed": bool(allow_native_runtime),
            }
            try:
                exit_code = module.main(
                    [
                        str(root / "mix.flac"),
                        str(root / "base.flac"),
                        str(root / "vocals.flac"),
                        str(judge_path),
                        str(output_dir),
                        "--candidate-id",
                        "base",
                        "--short-output-names",
                        "--allow-native-runtime",
                    ]
                )
            finally:
                module.require_docker_runtime = original_guard
                module.cli_runtime_contract = original_contract

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 2)
            self.assertEqual(
                calls,
                [
                    {
                        "tool_name": "apply_judge_guided_minus_vocal_leakage_repair.py",
                        "entrypoint_path": "/app/tools/apply_judge_guided_minus_vocal_leakage_repair.py",
                        "allow_native_macos_diagnostic": True,
                    }
                ],
            )
            self.assertEqual(manifest["execution_contract"]["runtime_environment"], "native_mac_cpu_dsp")
            self.assertTrue(manifest["execution_contract"]["native_runtime_allowed"])
            self.assertFalse(manifest["delivery_allowed"])


if __name__ == "__main__":
    unittest.main()
