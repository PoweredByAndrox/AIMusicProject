from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "adaptive_minus_repair.py"


def _load_adaptive_module():
    spec = importlib.util.spec_from_file_location("adaptive_minus_repair", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_manifest(
    output_dir: Path,
    *,
    added_layer_db: float,
    presence_db: float,
    air_db: float,
    extra_metrics: dict[str, float] | None = None,
) -> dict[str, object]:
    repaired_path = output_dir / "02_flexible_minus_repaired.flac"
    repaired_path.parent.mkdir(parents=True, exist_ok=True)
    repaired_path.write_bytes(b"fake flac payload")
    metrics = {
        "rms_drift_db_vs_base": 0.002,
        "added_layer_vs_base_rms_db": added_layer_db,
        "vocal_residue_corr": 0.0,
        "vocal_to_instrumental_rms_ratio_db": -120.0,
        "kick_attack_band_ratio_drift_db": 0.01,
        "kick_punch_band_ratio_drift_db": 0.01,
        "presence_band_ratio_drift_db": presence_db,
        "air_band_ratio_drift_db": air_db,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    return {
        "schema_version": 1,
        "metrics": metrics,
        "focus_metrics": {
            "intro": {
                "drift_vs_base": {
                    "added_layer_vs_base_rms_db": added_layer_db,
                    "presence_band_ratio_drift_db": presence_db,
                    "air_band_ratio_drift_db": air_db,
                }
            }
        },
        "artifacts": {
            "repaired_minus_flac": {
                "path": str(repaired_path),
            }
        },
    }


def _write_fake_manifest(output_dir: Path, *, profile_name: str) -> None:
    if profile_name == "pure_base":
        manifest = _fake_manifest(output_dir, added_layer_db=-160.0, presence_db=0.0, air_db=0.0)
    elif profile_name == "strict_no_texture":
        manifest = _fake_manifest(output_dir, added_layer_db=-56.0, presence_db=0.04, air_db=0.05)
    elif profile_name == "soft_delta":
        manifest = _fake_manifest(output_dir, added_layer_db=-50.0, presence_db=0.12, air_db=0.16)
    else:
        manifest = _fake_manifest(output_dir, added_layer_db=-38.0, presence_db=0.30, air_db=0.72)
    (output_dir / "stem_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


class TestAdaptiveMinusRepair(unittest.TestCase):
    def test_default_profiles_are_vocal_safe_and_explicit(self):
        module = _load_adaptive_module()

        profiles = module.default_candidate_profiles()

        self.assertEqual(
            [profile.name for profile in profiles],
            ["pure_base", "strict_no_texture", "soft_delta", "vocal_safe_balanced"],
        )
        self.assertTrue(all(profile.enable_raw_kick_repair is False for profile in profiles))
        self.assertTrue(profiles[0].pure_base)
        self.assertEqual(profiles[0].texture_strength, 0.0)
        self.assertEqual(profiles[1].texture_strength, 0.0)
        self.assertGreater(profiles[2].texture_strength, profiles[1].texture_strength)

    def test_score_penalizes_loud_synthetic_layer(self):
        module = _load_adaptive_module()
        _, strict_profile, _, balanced_profile = module.default_candidate_profiles()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            strict_manifest = _fake_manifest(
                temp_path / "strict",
                added_layer_db=-56.0,
                presence_db=0.04,
                air_db=0.05,
            )
            loud_manifest = _fake_manifest(
                temp_path / "balanced",
                added_layer_db=-36.0,
                presence_db=0.30,
                air_db=0.80,
            )

            strict_score = module.score_candidate(
                manifest=strict_manifest,
                manifest_path=temp_path / "strict" / "stem_manifest.json",
                output_dir=temp_path / "strict",
                profile=strict_profile,
                feedback_adjustments={},
            )
            loud_score = module.score_candidate(
                manifest=loud_manifest,
                manifest_path=temp_path / "balanced" / "stem_manifest.json",
                output_dir=temp_path / "balanced",
                profile=balanced_profile,
                feedback_adjustments={},
            )

        self.assertGreater(strict_score.score, loud_score.score)
        self.assertTrue(
            any("synthetic layer loudness penalty" in reason for reason in loud_score.reasons),
        )

    def test_score_penalizes_vocal_residue_and_instrumental_dip(self):
        module = _load_adaptive_module()
        strict_profile = module.default_candidate_profiles()[1]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            clean_manifest = _fake_manifest(
                temp_path / "clean",
                added_layer_db=-56.0,
                presence_db=0.04,
                air_db=0.05,
            )
            damaged_manifest = _fake_manifest(
                temp_path / "damaged",
                added_layer_db=-56.0,
                presence_db=-0.90,
                air_db=-0.70,
                extra_metrics={
                    "vocal_residue_corr": 0.34,
                    "vocal_to_instrumental_rms_ratio_db": -10.0,
                },
            )

            clean_score = module.score_candidate(
                manifest=clean_manifest,
                manifest_path=temp_path / "clean" / "stem_manifest.json",
                output_dir=temp_path / "clean",
                profile=strict_profile,
                feedback_adjustments={},
            )
            damaged_score = module.score_candidate(
                manifest=damaged_manifest,
                manifest_path=temp_path / "damaged" / "stem_manifest.json",
                output_dir=temp_path / "damaged",
                profile=strict_profile,
                feedback_adjustments={},
            )

        self.assertGreater(clean_score.score, damaged_score.score)
        self.assertTrue(any("vocal residue" in reason for reason in damaged_score.reasons))
        self.assertTrue(any("instrumental presence dip" in reason for reason in damaged_score.reasons))

    def test_minus_release_gate_blocks_missing_vocal_residue_metrics(self):
        module = _load_adaptive_module()
        metrics = {
            "presence_band_ratio_drift_db": 0.0,
            "air_band_ratio_drift_db": 0.0,
        }

        gate = module.build_minus_release_gate(metrics, {})

        self.assertEqual(gate["status"], "blocked_for_listening")
        self.assertEqual(gate["release_status"], "diagnostic_only")
        self.assertEqual(
            [blocker["reason"] for blocker in gate["blockers"]],
            ["missing_minus_release_metrics"],
        )

    def test_minus_release_gate_blocks_non_finite_required_metrics(self):
        module = _load_adaptive_module()
        metrics = {
            "vocal_residue_corr": float("nan"),
            "vocal_to_instrumental_rms_ratio_db": -120.0,
            "presence_band_ratio_drift_db": 0.0,
            "air_band_ratio_drift_db": 0.0,
        }

        gate = module.build_minus_release_gate(metrics, {})

        self.assertEqual(gate["status"], "blocked_for_listening")
        self.assertEqual(gate["release_status"], "diagnostic_only")
        self.assertEqual(gate["blockers"][0]["label"], "missing_minus_release_metrics")
        self.assertEqual(gate["blockers"][0]["missing_metrics"], ["vocal_residue_corr"])

    def test_feedback_store_adjusts_future_profile_selection(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_store = Path(temp_dir) / "feedback.jsonl"
            feedback_store.write_text(
                json.dumps({"profile": "soft_delta", "rating": "accept", "score": 1.0})
                + "\n"
                + json.dumps({"profile": "strict_no_texture", "rating": "reject"})
                + "\n",
                encoding="utf-8",
            )

            adjustments = module.load_feedback_adjustments(feedback_store)

        self.assertGreater(adjustments["soft_delta"], 0.0)
        self.assertLess(adjustments["strict_no_texture"], 0.0)

    def test_holdout_feedback_does_not_adjust_minus_priors(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_store = Path(temp_dir) / "feedback.jsonl"
            feedback_store.write_text(
                json.dumps({"profile": "soft_delta", "rating": "accept", "feedback_split": "holdout"})
                + "\n",
                encoding="utf-8",
            )

            adjustments = module.load_feedback_adjustments(feedback_store)

        self.assertEqual(adjustments, {})

    def test_score_penalizes_focused_instrumental_dip(self):
        module = _load_adaptive_module()
        strict_profile = module.default_candidate_profiles()[1]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            clean_manifest = _fake_manifest(
                temp_path / "clean_focus",
                added_layer_db=-56.0,
                presence_db=0.04,
                air_db=0.05,
            )
            dipped_manifest = _fake_manifest(
                temp_path / "dipped_focus",
                added_layer_db=-56.0,
                presence_db=0.04,
                air_db=0.05,
            )
            dipped_manifest["focus_metrics"]["intro"]["drift_vs_base"]["presence_band_ratio_drift_db"] = -0.80
            dipped_manifest["focus_metrics"]["intro"]["drift_vs_base"]["air_band_ratio_drift_db"] = -0.75

            clean_score = module.score_candidate(
                manifest=clean_manifest,
                manifest_path=temp_path / "clean_focus" / "stem_manifest.json",
                output_dir=temp_path / "clean_focus",
                profile=strict_profile,
                feedback_adjustments={},
            )
            dipped_score = module.score_candidate(
                manifest=dipped_manifest,
                manifest_path=temp_path / "dipped_focus" / "stem_manifest.json",
                output_dir=temp_path / "dipped_focus",
                profile=strict_profile,
                feedback_adjustments={},
            )

        self.assertGreater(clean_score.score, dipped_score.score)
        self.assertTrue(any("focused instrumental dip" in reason for reason in dipped_score.reasons))

    def test_cli_runs_candidates_and_writes_adaptive_manifest(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_audio_path = temp_path / "raw.flac"
            base_minus_path = temp_path / "base.flac"
            repair_script = temp_path / "apply_flexible_minus_repair.py"
            output_dir = temp_path / "adaptive"
            listen_dir = temp_path / "listen"
            raw_audio_path.write_bytes(b"raw")
            base_minus_path.write_bytes(b"base")
            repair_script.write_text("# fake repair script\n", encoding="utf-8")
            commands: list[list[str]] = []

            def fake_run_command(command: list[str]) -> None:
                commands.append(command)
                candidate_output_dir = Path(command[4])
                _write_fake_manifest(candidate_output_dir, profile_name=candidate_output_dir.name)

            argv = [
                str(raw_audio_path),
                str(base_minus_path),
                str(output_dir),
                "--listen-dir",
                str(listen_dir),
                "--repair-script",
                str(repair_script),
                "--python-executable",
                sys.executable,
                "--focus-window",
                "intro:0:30",
            ]

            with patch.object(module, "_run_command", side_effect=fake_run_command):
                self.assertEqual(module.main(argv), 0)

            adaptive_manifest = json.loads(
                (output_dir / "adaptive_manifest.json").read_text(encoding="utf-8"),
            )
            output_selected_exists = (output_dir / "02_adaptive_selected_minus.flac").exists()
            listen_selected_exists = (listen_dir / "02_adaptive_selected_minus.flac").exists()

        self.assertEqual(len(commands), 4)
        self.assertEqual(adaptive_manifest["selected_profile"], "pure_base")
        self.assertTrue(output_selected_exists)
        self.assertFalse(listen_selected_exists)
        self.assertEqual(adaptive_manifest["selected_minus_release_gate"]["status"], "release_ready")
        self.assertEqual(adaptive_manifest["candidate_status"], "selected_candidate_only")
        self.assertEqual(adaptive_manifest["delivery_status"], "candidate_only")
        self.assertFalse(adaptive_manifest["delivery_allowed"])
        self.assertEqual(adaptive_manifest["presentation_status"], "blocked_until_downstream_delivery_gates")
        self.assertEqual(adaptive_manifest["release_status"], "not_release_ready")
        self.assertTrue(adaptive_manifest["requires_post_judge"])
        self.assertEqual(adaptive_manifest["selected_artifacts"]["candidate_status"], "selected_candidate_only")
        self.assertEqual(adaptive_manifest["selected_artifacts"]["delivery_status"], "candidate_only")
        self.assertFalse(adaptive_manifest["selected_artifacts"]["delivery_allowed"])
        self.assertEqual(
            adaptive_manifest["selected_artifacts"]["presentation_status"],
            "blocked_until_downstream_delivery_gates",
        )
        self.assertEqual(adaptive_manifest["selected_artifacts"]["release_status"], "not_release_ready")
        self.assertTrue(adaptive_manifest["selected_artifacts"]["requires_post_judge"])
        self.assertEqual(
            adaptive_manifest["selected_artifacts"]["internal_minus_release_status"],
            "release_ready",
        )
        self.assertIn("minus_release_gate", adaptive_manifest["candidates"][0])
        self.assertEqual(
            [candidate["profile_name"] for candidate in adaptive_manifest["candidates"]],
            ["pure_base", "soft_delta", "strict_no_texture", "vocal_safe_balanced"],
        )
        self.assertIn("--pure-base", commands[0])
        self.assertEqual(adaptive_manifest["learning_contract"]["type"], "feedback_weighted_profile_selection")
        self.assertEqual(
            adaptive_manifest["execution_contract"]["acceleration_mode"],
            "external_only_cpu_dsp_repair",
        )
        self.assertTrue(adaptive_manifest["execution_contract"]["gpu_inference_skipped"])
        self.assertEqual(adaptive_manifest["progress_events"][0]["phase"], "start")
        self.assertEqual(adaptive_manifest["progress_events"][0]["gpu_inference_skipped"], True)
        self.assertTrue(
            any(
                event["phase"] == "candidate_start" and event["profile"] == "pure_base"
                for event in adaptive_manifest["progress_events"]
            ),
        )

    def test_cli_keeps_selected_candidate_locked_until_downstream_delivery_gates(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_audio_path = temp_path / "raw.flac"
            base_minus_path = temp_path / "base.flac"
            repair_script = temp_path / "apply_flexible_minus_repair.py"
            output_dir = temp_path / "adaptive"
            listen_dir = temp_path / "listen"
            raw_audio_path.write_bytes(b"raw")
            base_minus_path.write_bytes(b"base")
            repair_script.write_text("# fake repair script\n", encoding="utf-8")

            def fake_run_command(command: list[str]) -> None:
                candidate_output_dir = Path(command[4])
                _write_fake_manifest(candidate_output_dir, profile_name=candidate_output_dir.name)

            argv = [
                str(raw_audio_path),
                str(base_minus_path),
                str(output_dir),
                "--profile",
                "pure_base",
                "--listen-dir",
                str(listen_dir),
                "--repair-script",
                str(repair_script),
                "--python-executable",
                sys.executable,
            ]

            with patch.object(module, "_run_command", side_effect=fake_run_command):
                self.assertEqual(module.main(argv), 0)

            adaptive_manifest = json.loads(
                (output_dir / "adaptive_manifest.json").read_text(encoding="utf-8"),
            )
            output_selected_exists = (output_dir / "02_adaptive_selected_minus.flac").exists()
            listen_selected_exists = (listen_dir / "02_adaptive_selected_minus.flac").exists()

        self.assertTrue(output_selected_exists)
        self.assertFalse(listen_selected_exists)
        self.assertEqual(adaptive_manifest["selected_minus_release_gate"]["status"], "release_ready")
        self.assertEqual(adaptive_manifest["candidate_status"], "selected_candidate_only")
        self.assertEqual(adaptive_manifest["release_status"], "not_release_ready")
        self.assertEqual(adaptive_manifest["delivery_status"], "candidate_only")
        self.assertFalse(adaptive_manifest["delivery_allowed"])
        self.assertEqual(adaptive_manifest["presentation_status"], "blocked_until_downstream_delivery_gates")
        self.assertTrue(adaptive_manifest["requires_post_judge"])
        self.assertEqual(adaptive_manifest["selected_artifacts"]["delivery_status"], "candidate_only")
        self.assertFalse(adaptive_manifest["selected_artifacts"]["delivery_allowed"])
        self.assertEqual(
            adaptive_manifest["selected_artifacts"]["presentation_status"],
            "blocked_until_downstream_delivery_gates",
        )
        self.assertEqual(adaptive_manifest["selected_artifacts"]["release_status"], "not_release_ready")
        self.assertTrue(adaptive_manifest["selected_artifacts"]["requires_post_judge"])
        self.assertEqual(
            adaptive_manifest["selected_artifacts"]["internal_minus_release_status"],
            "release_ready",
        )

    def test_cli_blocks_listen_copy_when_selected_minus_gate_blocks(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_audio_path = temp_path / "raw.flac"
            base_minus_path = temp_path / "base.flac"
            repair_script = temp_path / "apply_flexible_minus_repair.py"
            output_dir = temp_path / "adaptive"
            listen_dir = temp_path / "listen"
            raw_audio_path.write_bytes(b"raw")
            base_minus_path.write_bytes(b"base")
            repair_script.write_text("# fake repair script\n", encoding="utf-8")

            def fake_run_command(command: list[str]) -> None:
                candidate_output_dir = Path(command[4])
                manifest = _fake_manifest(
                    candidate_output_dir,
                    added_layer_db=-160.0,
                    presence_db=-0.90,
                    air_db=-0.80,
                    extra_metrics={
                        "vocal_residue_corr": 0.42,
                        "vocal_to_instrumental_rms_ratio_db": -8.0,
                    },
                )
                (candidate_output_dir / "stem_manifest.json").write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

            argv = [
                str(raw_audio_path),
                str(base_minus_path),
                str(output_dir),
                "--profile",
                "pure_base",
                "--listen-dir",
                str(listen_dir),
                "--repair-script",
                str(repair_script),
                "--python-executable",
                sys.executable,
            ]

            with patch.object(module, "_run_command", side_effect=fake_run_command):
                self.assertEqual(module.main(argv), 0)

            adaptive_manifest = json.loads(
                (output_dir / "adaptive_manifest.json").read_text(encoding="utf-8"),
            )
            output_selected_exists = (output_dir / "02_adaptive_selected_minus.flac").exists()
            listen_selected_exists = (listen_dir / "02_adaptive_selected_minus.flac").exists()

        self.assertTrue(output_selected_exists)
        self.assertFalse(listen_selected_exists)
        self.assertEqual(adaptive_manifest["selected_minus_release_gate"]["status"], "blocked_for_listening")
        self.assertEqual(adaptive_manifest["candidate_status"], "selected_candidate_only")
        self.assertEqual(adaptive_manifest["delivery_status"], "candidate_only")
        self.assertFalse(adaptive_manifest["delivery_allowed"])
        self.assertEqual(adaptive_manifest["presentation_status"], "blocked_until_downstream_delivery_gates")
        self.assertEqual(adaptive_manifest["release_status"], "not_release_ready")
        self.assertTrue(adaptive_manifest["requires_post_judge"])
        self.assertEqual(adaptive_manifest["selected_artifacts"]["candidate_status"], "selected_candidate_only")
        self.assertEqual(adaptive_manifest["selected_artifacts"]["delivery_status"], "candidate_only")
        self.assertFalse(adaptive_manifest["selected_artifacts"]["delivery_allowed"])
        self.assertEqual(
            adaptive_manifest["selected_artifacts"]["presentation_status"],
            "blocked_until_downstream_delivery_gates",
        )
        self.assertEqual(adaptive_manifest["selected_artifacts"]["release_status"], "not_release_ready")
        self.assertTrue(adaptive_manifest["selected_artifacts"]["requires_post_judge"])
        self.assertEqual(
            adaptive_manifest["selected_artifacts"]["internal_minus_release_status"],
            "diagnostic_only",
        )


if __name__ == "__main__":
    unittest.main()
