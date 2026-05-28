from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
