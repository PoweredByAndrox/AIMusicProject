from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "run_ralph_minus_improvement_loop.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_ralph_minus_improvement_loop", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _base_summary() -> dict[str, float]:
    return {
        "rms_delta_db_vs_base": -240.0,
        "avg_vocal_residue_corr": 0.1078,
        "max_vocal_residue_corr": 0.2740,
        "avg_vocal_projection_ratio_db": -19.529,
        "worst_rms_dip_db": 0.0,
        "worst_presence_dip_db": 0.0,
        "worst_air_dip_db": 0.0,
    }


class RalphMinusImprovementLoopTests(unittest.TestCase):
    def test_default_recipes_include_actual_masked_residue_guarded_candidate(self) -> None:
        module = _load_module()

        recipes = {recipe.name: recipe for recipe in module.default_candidate_recipes()}

        self.assertIn("actual_masked_residue_guarded", recipes)
        self.assertEqual(
            recipes["actual_masked_residue_guarded"].output_artifact_key,
            "minus_actual_masked_residue_flac",
        )
        self.assertNotIn("focus_window_guarded", recipes)

    def test_focus_recipe_is_added_only_when_focus_windows_exist(self) -> None:
        module = _load_module()

        no_focus = {
            recipe.name
            for recipe in module.default_candidate_recipes(
                focus_windows_jsonl="focus.jsonl",
                focus_window_count=0,
            )
        }
        with_focus = {
            recipe.name: recipe
            for recipe in module.default_candidate_recipes(
                focus_windows_jsonl="focus.jsonl",
                focus_track_id="track-a",
                focus_candidate_id="cand-a",
                max_focus_windows=4,
                focus_window_count=2,
            )
        }

        self.assertNotIn("focus_window_guarded", no_focus)
        self.assertIn("focus_window_guarded", with_focus)
        self.assertEqual(
            with_focus["focus_window_guarded"].output_artifact_key,
            "minus_focus_window_cleanup_flac",
        )
        self.assertIn("--focus-windows-jsonl", with_focus["focus_window_guarded"].arguments)

    def test_no_change_candidate_is_rejected_even_if_release_like(self) -> None:
        module = _load_module()

        evaluation = module.evaluate_minus_candidate(
            candidate_name="strict_no_texture",
            candidate_summary={
                "rms_delta_db_vs_base": -57.256,
                "avg_vocal_residue_corr": 0.1078,
                "max_vocal_residue_corr": 0.2740,
                "avg_vocal_projection_ratio_db": -19.529,
                "worst_rms_dip_db": 0.0,
                "worst_presence_dip_db": 0.0,
                "worst_air_dip_db": 0.0,
            },
            base_summary=_base_summary(),
            policy=module.MinusAcceptancePolicy(),
        )

        labels = {blocker["label"] for blocker in evaluation["blockers"]}
        self.assertEqual(evaluation["status"], "rejected")
        self.assertIn("no_meaningful_minus_delta", labels)
        self.assertIn("insufficient_vocal_residue_improvement", labels)

    def test_destructive_cleanup_is_rejected_even_with_better_vocal_proxy(self) -> None:
        module = _load_module()

        evaluation = module.evaluate_minus_candidate(
            candidate_name="direct_vocal_mask",
            candidate_summary={
                "rms_delta_db_vs_base": -29.754,
                "avg_vocal_residue_corr": 0.09618,
                "max_vocal_residue_corr": 0.25215,
                "avg_vocal_projection_ratio_db": -20.537,
                "worst_rms_dip_db": -0.400,
                "worst_presence_dip_db": -0.426,
                "worst_air_dip_db": -0.366,
            },
            base_summary=_base_summary(),
            policy=module.MinusAcceptancePolicy(),
        )

        labels = {blocker["label"] for blocker in evaluation["blockers"]}
        self.assertEqual(evaluation["status"], "rejected")
        self.assertIn("rms_dip_damage", labels)
        self.assertIn("presence_dip_damage", labels)
        self.assertIn("air_dip_damage", labels)

    def test_overboosted_candidate_is_rejected_even_with_better_vocal_proxy(self) -> None:
        module = _load_module()

        evaluation = module.evaluate_minus_candidate(
            candidate_name="overboosted_minus",
            candidate_summary={
                "rms_delta_db_vs_base": -30.0,
                "avg_vocal_residue_corr": 0.09618,
                "max_vocal_residue_corr": 0.25215,
                "avg_vocal_projection_ratio_db": -20.537,
                "worst_rms_dip_db": 0.0,
                "worst_presence_dip_db": 0.0,
                "worst_air_dip_db": 0.0,
                "max_rms_boost_db": 24.0,
                "max_presence_boost_db": 12.0,
                "max_air_boost_db": 11.0,
            },
            base_summary=_base_summary(),
            policy=module.MinusAcceptancePolicy(),
        )

        labels = {blocker["label"] for blocker in evaluation["blockers"]}
        self.assertEqual(evaluation["status"], "rejected")
        self.assertIn("rms_boost_damage", labels)
        self.assertIn("presence_boost_damage", labels)
        self.assertIn("air_boost_damage", labels)

    def test_iterative_candidate_is_rejected_when_cumulative_dip_budget_is_exceeded(self) -> None:
        module = _load_module()

        evaluation = module.evaluate_minus_candidate(
            candidate_name="third_pass",
            candidate_summary={
                "rms_delta_db_vs_base": -42.0,
                "avg_vocal_residue_corr": 0.101,
                "max_vocal_residue_corr": 0.260,
                "avg_vocal_projection_ratio_db": -20.10,
                "worst_rms_dip_db": -0.040,
                "worst_presence_dip_db": -0.050,
                "worst_air_dip_db": -0.020,
            },
            base_summary=_base_summary(),
            policy=module.MinusAcceptancePolicy(),
            cumulative_summary={
                "rms_delta_db_vs_base": -33.0,
                "avg_vocal_residue_corr": 0.101,
                "max_vocal_residue_corr": 0.260,
                "avg_vocal_projection_ratio_db": -20.10,
                "worst_rms_dip_db": -0.130,
                "worst_presence_dip_db": -0.170,
                "worst_air_dip_db": -0.050,
            },
        )

        labels = {blocker["label"] for blocker in evaluation["blockers"]}
        self.assertEqual(evaluation["status"], "rejected")
        self.assertIn("cumulative_rms_dip_damage", labels)
        self.assertIn("cumulative_presence_dip_damage", labels)

    def test_windowed_candidate_is_accepted_when_it_improves_residue_without_dips(self) -> None:
        module = _load_module()

        evaluation = module.evaluate_minus_candidate(
            candidate_name="windowed_stronger",
            candidate_summary={
                "rms_delta_db_vs_base": -43.446,
                "avg_vocal_residue_corr": 0.10575,
                "max_vocal_residue_corr": 0.2720,
                "avg_vocal_projection_ratio_db": -19.700,
                "worst_rms_dip_db": -0.042,
                "worst_presence_dip_db": -0.064,
                "worst_air_dip_db": -0.014,
            },
            base_summary=_base_summary(),
            policy=module.MinusAcceptancePolicy(),
        )

        self.assertEqual(evaluation["status"], "accepted")
        self.assertEqual(evaluation["blockers"], [])
        self.assertGreater(evaluation["average_vocal_improvement_db"], 0.12)

    def test_blocked_manifest_returns_non_zero_exit_code(self) -> None:
        module = _load_module()

        self.assertEqual(
            module.exit_code_for_manifest(
                {
                    "delivery_status": "accepted",
                    "delivery_allowed": True,
                    "release_status": "release_ready",
                }
            ),
            0,
        )
        self.assertEqual(
            module.exit_code_for_manifest(
                {
                    "delivery_status": "blocked",
                    "delivery_allowed": False,
                    "release_status": "blocked_for_listening",
                }
            ),
            1,
        )

    def test_progress_log_records_external_only_cpu_scoring(self) -> None:
        module = _load_module()
        progress_path = self.temp_path / "ralph_minus_improvement_progress.jsonl"

        module.emit_progress(
            progress_path,
            phase="start",
            route_name="ralph_minus_improvement_loop",
            acceleration_mode="external_only_cpu_scoring",
            gpu_inference_skipped=True,
            reason="external_only_cpu_dsp_scoring",
        )

        rows = [
            json.loads(line)
            for line in progress_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["phase"], "start")
        self.assertEqual(rows[0]["route_name"], "ralph_minus_improvement_loop")
        self.assertEqual(rows[0]["acceleration_mode"], "external_only_cpu_scoring")
        self.assertTrue(rows[0]["gpu_inference_skipped"])
        self.assertEqual(rows[0]["reason"], "external_only_cpu_dsp_scoring")

    def test_focus_windows_load_only_blocked_rows_for_requested_candidate(self) -> None:
        module = _load_module()
        focus_path = self.temp_path / "focus.jsonl"
        rows = [
            {
                "track_id": "track-a",
                "candidate_id": "cand-a",
                "window_start_sec": 0.0,
                "window_end_sec": 6.0,
                "critic_decision": "blocked_hard_case",
                "predicted_blockers": ["severe_phase_collapse"],
                "true_label": 0,
            },
            {
                "track_id": "track-a",
                "candidate_id": "cand-a",
                "window_start_sec": 6.0,
                "window_end_sec": 12.0,
                "true_label": 0,
                "predicted_blockers": ["numeric_zero_label"],
            },
            {
                "track_id": "track-a",
                "candidate_id": "cand-a",
                "window_start_sec": 12.0,
                "window_end_sec": 18.0,
                "critic_decision": "accept_candidate",
                "true_label": 1,
            },
            {
                "track_id": "track-b",
                "candidate_id": "cand-b",
                "window_start_sec": 0.0,
                "window_end_sec": 6.0,
                "critic_decision": "blocked_hard_case",
                "true_label": 0,
            },
        ]
        focus_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

        focus_windows = module.load_focus_windows(
            focus_path,
            track_id="track-a",
            candidate_id="cand-a",
            max_windows=4,
        )

        self.assertEqual(len(focus_windows), 2)
        self.assertEqual(focus_windows[0]["start_sec"], 0.0)
        self.assertEqual(focus_windows[0]["end_sec"], 6.0)
        self.assertEqual(focus_windows[0]["predicted_blockers"], ["severe_phase_collapse"])
        self.assertEqual(focus_windows[1]["start_sec"], 6.0)
        self.assertEqual(focus_windows[1]["end_sec"], 12.0)
        self.assertEqual(focus_windows[1]["predicted_blockers"], ["numeric_zero_label"])

    def test_focus_summary_reports_local_improvement_without_delivery_claim(self) -> None:
        module = _load_module()
        sample_rate = 100
        frames = 1000
        t = np.linspace(0.0, 1.0, frames, dtype=np.float32)
        vocal = np.stack([np.sin(2.0 * np.pi * 8.0 * t), np.sin(2.0 * np.pi * 8.0 * t)], axis=1).astype(np.float32)
        base = (0.8 * vocal).astype(np.float32)
        candidate = (0.6 * vocal).astype(np.float32)
        focus_windows = [{"start_sec": 0.0, "end_sec": 2.0, "critic_decision": "blocked_hard_case"}]

        base_summary = module.calculate_focus_window_summary(
            candidate_audio=base,
            base_audio=base,
            vocal_audio=vocal,
            sample_rate=sample_rate,
            focus_windows=focus_windows,
        )
        candidate_summary = module.calculate_focus_window_summary(
            candidate_audio=candidate,
            base_audio=base,
            vocal_audio=vocal,
            sample_rate=sample_rate,
            focus_windows=focus_windows,
        )
        comparison = module.compare_focus_summaries(candidate_summary, base_summary)

        self.assertTrue(comparison["available"])
        self.assertEqual(comparison["window_count"], 1)
        self.assertGreater(comparison["average_vocal_improvement_db"], 0.0)
        self.assertLess(comparison["worst_rms_dip_db"], 0.0)

    def test_recipe_inputs_are_resampled_to_base_sample_rate(self) -> None:
        module = _load_module()
        inputs_dir = self.temp_path / "inputs"
        raw_path = self._write_audio(inputs_dir / "raw.flac", sample_rate=12000)
        base_minus_path = self._write_audio(inputs_dir / "base.flac", sample_rate=8000)
        guide_minus_path = self._write_audio(inputs_dir / "guide.flac", sample_rate=16000)
        primary_vocals_path = self._write_audio(inputs_dir / "primary.flac", sample_rate=11025)
        secondary_vocals_path = self._write_audio(inputs_dir / "secondary.flac", sample_rate=22050)

        prepared = module.prepare_recipe_inputs(
            raw_audio_path=raw_path,
            base_minus_path=base_minus_path,
            guide_minus_path=guide_minus_path,
            primary_vocals_path=primary_vocals_path,
            secondary_vocals_path=secondary_vocals_path,
            output_dir=self.temp_path / "normalized",
        )

        sample_rates = {
            key: sf.info(str(path)).samplerate
            for key, path in prepared.items()
            if key.endswith("_path")
        }
        self.assertEqual(set(sample_rates.values()), {8000})
        self.assertEqual(prepared["sample_rate"], 8000)
        self.assertTrue(str(prepared["raw_audio_path"]).endswith("raw_audio.flac"))

    def _write_audio(self, path: Path, *, sample_rate: int) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        frames = sample_rate // 2
        audio = np.zeros((frames, 2), dtype=np.float32)
        audio[:, 0] = np.linspace(-0.1, 0.1, frames, dtype=np.float32)
        audio[:, 1] = audio[:, 0]
        sf.write(path, audio, sample_rate)
        return path

    def setUp(self) -> None:
        import tempfile

        self._temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self._temp_dir.name)

    def tearDown(self) -> None:
        self._temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
