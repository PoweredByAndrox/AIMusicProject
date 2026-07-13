from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


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


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def last_json_object(stdout: str) -> dict:
    decoder = json.JSONDecoder()
    objects = []
    index = 0
    while index < len(stdout):
        while index < len(stdout) and stdout[index].isspace():
            index += 1
        if index >= len(stdout):
            break
        payload, next_index = decoder.raw_decode(stdout, index)
        if isinstance(payload, dict):
            objects.append(payload)
        index = next_index
    assert objects
    return objects[-1]


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
        self.assertEqual(metrics["vocal_suppression_mode"], "global")

    def test_segment_local_vocal_suppression_cleans_each_selected_window_only(self) -> None:
        module = _load_module()
        sample_rate = 8000
        times = np.arange(sample_rate * 3, dtype=np.float32) / float(sample_rate)
        vocal_like_air = np.sin(2.0 * np.pi * 3000.0 * times).astype(np.float32)
        source = np.zeros_like(vocal_like_air)
        source[:sample_rate] = 0.20 * vocal_like_air[:sample_rate]
        source[sample_rate : sample_rate * 2] = -0.16 * vocal_like_air[sample_rate : sample_rate * 2]
        source[sample_rate * 2 :] = 0.10 * vocal_like_air[sample_rate * 2 :]
        base_minus = np.zeros_like(source)
        stereo_source = np.column_stack([source, source])
        stereo_base = np.column_stack([base_minus, base_minus])
        stereo_vocal = np.column_stack([vocal_like_air, vocal_like_air])
        windows = [
            {"start_sec": 0.0, "duration_sec": 1.0, "metrics": {"rms_ratio": 0.0}},
            {"start_sec": 1.0, "duration_sec": 1.0, "metrics": {"rms_ratio": 0.0}},
        ]

        unsuppressed, _ = module.apply_air_detail_repair(
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
            vocal_suppression_strength=0.0,
            max_vocal_projection_gain=2.0,
            presence_low_hz=1200.0,
            presence_high_hz=2200.0,
            air_low_hz=2500.0,
            air_high_hz=3600.0,
            body_low_hz=120.0,
            body_high_hz=900.0,
            vocal_suppression_mode="segment_local",
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
            vocal_suppression_mode="segment_local",
        )

        first_signed_before = module.signed_projection_gain(
            unsuppressed[:sample_rate, 0],
            stereo_vocal[:sample_rate, 0],
            max_projection_gain=4.0,
        )
        first_signed_after = module.signed_projection_gain(
            suppressed[:sample_rate, 0],
            stereo_vocal[:sample_rate, 0],
            max_projection_gain=4.0,
        )
        second_signed_before = module.signed_projection_gain(
            unsuppressed[sample_rate : sample_rate * 2, 0],
            stereo_vocal[sample_rate : sample_rate * 2, 0],
            max_projection_gain=4.0,
        )
        second_signed_after = module.signed_projection_gain(
            suppressed[sample_rate : sample_rate * 2, 0],
            stereo_vocal[sample_rate : sample_rate * 2, 0],
            max_projection_gain=4.0,
        )
        untouched_delta = np.max(np.abs(suppressed[sample_rate * 2 :] - stereo_base[sample_rate * 2 :]))

        self.assertGreater(first_signed_before, 0.0)
        self.assertLess(second_signed_before, 0.0)
        self.assertLess(abs(first_signed_after), abs(first_signed_before) * 0.25)
        self.assertLess(abs(second_signed_after), abs(second_signed_before) * 0.25)
        self.assertLess(untouched_delta, 1.0e-5)
        self.assertEqual(metrics["vocal_suppression_mode"], "segment_local")
        self.assertEqual(metrics["window_vocal_projection_count"], 2)
        self.assertEqual(len(metrics["window_vocal_projection_metrics"]), 2)
        self.assertTrue(
            all(
                item["vocal_projection_gain_removed"] > 0.0
                for item in metrics["window_vocal_projection_metrics"]
            )
        )

    def test_default_global_vocal_suppression_mode_matches_explicit_global(self) -> None:
        module = _load_module()
        sample_rate = 8000
        times = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
        vocal_like_air = np.sin(2.0 * np.pi * 3000.0 * times).astype(np.float32)
        source = 0.20 * vocal_like_air
        base_minus = np.zeros_like(source)
        stereo_source = np.column_stack([source, source])
        stereo_base = np.column_stack([base_minus, base_minus])
        stereo_vocal = np.column_stack([vocal_like_air, vocal_like_air])
        common_kwargs = dict(
            source_audio=stereo_source,
            base_minus=stereo_base,
            vocal_audio=stereo_vocal,
            sample_rate=sample_rate,
            windows=[{"start_sec": 0.0, "duration_sec": 1.0, "metrics": {"rms_ratio": 0.0}}],
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

        default_repaired, default_metrics = module.apply_air_detail_repair(**common_kwargs)
        explicit_repaired, explicit_metrics = module.apply_air_detail_repair(
            **common_kwargs,
            vocal_suppression_mode="global",
        )
        default_args = module.parse_args(
            [
                "/tmp/raw.flac",
                "/tmp/base.flac",
                "/tmp/rows.jsonl",
                "/tmp/out",
            ]
        )

        self.assertTrue(np.allclose(default_repaired, explicit_repaired, atol=1.0e-6))
        self.assertEqual(default_metrics["vocal_suppression_mode"], "global")
        self.assertEqual(explicit_metrics["vocal_suppression_mode"], "global")
        self.assertEqual(
            default_metrics["vocal_projection_gain_removed"],
            explicit_metrics["vocal_projection_gain_removed"],
        )
        self.assertEqual(default_args.vocal_suppression_mode, "global")

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
        self.assertEqual(manifest["repair"]["vocal_suppression_mode"], "global")
        self.assertEqual(manifest["execution_contract"]["runtime_environment"], "docker")
        self.assertFalse(manifest["execution_contract"]["native_runtime_allowed"])

    def test_manifest_records_explicit_vocal_suppression_mode(self) -> None:
        module = _load_module()
        cli_args = module.parse_args(
            [
                "/tmp/raw.flac",
                "/tmp/base.flac",
                "/tmp/rows.jsonl",
                "/tmp/out",
                "--vocal-suppression-mode",
                "segment_local",
            ]
        )
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
            repair_metrics={
                "status": "air_detail_repair_applied",
                "vocal_suppression_mode": "segment_local",
            },
        )

        self.assertEqual(cli_args.vocal_suppression_mode, "segment_local")
        self.assertEqual(manifest["repair"]["vocal_suppression_mode"], "segment_local")
        self.assertEqual(
            manifest["repair"]["metrics"]["vocal_suppression_mode"],
            "segment_local",
        )

    def test_cli_allows_explicit_native_runtime_and_records_contract(self) -> None:
        module = _load_module()
        captured: dict = {}
        calls: list[dict] = []

        def fake_require_docker_runtime(**kwargs):
            calls.append(kwargs)

        def fake_build(args):
            captured["runtime_environment"] = args.runtime_environment
            captured["native_runtime_allowed"] = args.native_runtime_allowed
            return {
                "manifest_path": str(Path(args.output_dir) / "manifest.json"),
                "artifacts": {"minus_air_detail_repair_flac": str(Path(args.output_dir) / "m001.flac")},
                "candidate_row_jsonl": str(Path(args.output_dir) / "rows.jsonl"),
                "repair": {"selected_window_count": 0},
                "delivery_status": "candidate_only",
                "delivery_allowed": False,
            }

        original_guard = module.require_docker_runtime
        original_build = module.build_judge_guided_minus_air_detail_repair
        original_contract = module.cli_runtime_contract
        module.require_docker_runtime = fake_require_docker_runtime
        module.build_judge_guided_minus_air_detail_repair = fake_build
        module.cli_runtime_contract = lambda *, allow_native_runtime: {
            "runtime_environment": "native_mac_cpu_dsp",
            "native_runtime_allowed": bool(allow_native_runtime),
        }
        try:
            exit_code = module.main(
                [
                    "/tmp/raw.flac",
                    "/tmp/base.flac",
                    "/tmp/rows.jsonl",
                    "/tmp/out",
                    "--allow-native-runtime",
                ]
            )
        finally:
            module.require_docker_runtime = original_guard
            module.build_judge_guided_minus_air_detail_repair = original_build
            module.cli_runtime_contract = original_contract

        self.assertEqual(exit_code, 0)
        self.assertEqual(captured["runtime_environment"], "native_mac_cpu_dsp")
        self.assertTrue(captured["native_runtime_allowed"])
        self.assertEqual(
            calls,
            [
                {
                    "tool_name": "apply_judge_guided_minus_air_detail_repair.py",
                    "entrypoint_path": "/app/tools/apply_judge_guided_minus_air_detail_repair.py",
                    "allow_native_macos_diagnostic": True,
                }
            ],
        )

    def test_air_detail_repair_supports_short_output_names(self) -> None:
        sample_rate = 44100
        duration_sec = 0.5
        time = np.arange(int(sample_rate * duration_sec), dtype=np.float32) / sample_rate
        raw = 0.08 * np.sin(2.0 * np.pi * 880.0 * time) + 0.03 * np.sin(2.0 * np.pi * 8000.0 * time)
        base = raw * 0.35

        import tempfile

        with tempfile.TemporaryDirectory() as tmp_name:
            tmp_path = Path(tmp_name)
            raw_path = tmp_path / "raw.flac"
            base_path = tmp_path / "minus.flac"
            sf.write(raw_path, raw, sample_rate, subtype="PCM_24")
            sf.write(base_path, base, sample_rate, subtype="PCM_24")
            judge_rows = write_jsonl(
                tmp_path / "rows.jsonl",
                [
                    {
                        "candidate_label": "short_minus",
                        "window_sec": [0.05, 0.35],
                        "issues": ["air_loss_proxy", "presence_loss_proxy"],
                        "issue_severities": {"air_loss": 0.8, "presence_loss": 0.4},
                        "blocker": True,
                        "severity_score": 0.8,
                    }
                ],
            )
            output_dir = tmp_path / "out"

            result = subprocess.run(
                [
                    sys.executable,
                    "tools/apply_judge_guided_minus_air_detail_repair.py",
                    str(raw_path),
                    str(base_path),
                    str(judge_rows),
                    str(output_dir),
                    "--candidate-id",
                    "short_minus",
                    "--track-id",
                    "t001",
                    "--vocal-suppression-mode",
                    "segment_local",
                    "--short-output-names",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            stdout = last_json_object(result.stdout)
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(stdout["manifest_path"], str(output_dir / "manifest.json"))
            self.assertTrue((output_dir / "m001.flac").exists())
            self.assertTrue((output_dir / "rows.jsonl").exists())
            self.assertEqual(manifest["manifest_path"], str(output_dir / "manifest.json"))
            self.assertTrue(manifest["short_output_names"])
            self.assertEqual(manifest["artifact_name_policy"]["audio_filename"], "m001.flac")
            self.assertFalse(manifest["artifact_name_policy"]["raw_track_or_candidate_ids_in_path"])
            self.assertFalse(manifest["delivery_allowed"])
            self.assertTrue(manifest["requires_post_judge"])
            self.assertEqual(manifest["repair"]["vocal_suppression_mode"], "segment_local")
            self.assertEqual(
                manifest["repair"]["metrics"]["vocal_suppression_mode"],
                "segment_local",
            )
            self.assertTrue(all(len(path.name) <= 32 for path in output_dir.rglob("*") if path.is_file()))


if __name__ == "__main__":
    unittest.main()
