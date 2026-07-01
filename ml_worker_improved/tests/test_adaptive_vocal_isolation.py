from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "adaptive_vocal_isolation.py"
PROFILE_MODULE_PATH = TOOLS_ROOT / "adaptive_vocal_profiles.py"


def _load_adaptive_module():
    spec = importlib.util.spec_from_file_location("adaptive_vocal_isolation", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metrics(
    *,
    dropout: float,
    tail_dropout: float,
    rms_ratio_db: float,
    low_ratio_db: float,
    kick_ratio_db: float,
) -> dict[str, float]:
    return {
        "source_rms_dbfs": -10.0,
        "candidate_rms_dbfs": -18.0,
        "candidate_to_source_rms_db": rms_ratio_db,
        "candidate_peak_dbfs": -1.0,
        "dropout_frame_ratio": dropout,
        "tail_dropout_ratio": tail_dropout,
        "low_band_ratio_db_35_180": low_ratio_db,
        "kick_punch_ratio_db_250_900": kick_ratio_db,
        "presence_ratio_db_1800_4500": -18.0,
        "air_ratio_db_4500_9000": -24.0,
    }


def _write_fake_candidate(output_dir: Path, profile_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    vocals_path = output_dir / "vocals.flac"
    vocals_path.write_bytes(f"fake vocals {profile_name}".encode("utf-8"))
    manifest = {
        "schema_version": 1,
        "pipeline": {"candidate_mode": "hybrid_vocals"},
        "stems": {
            "vocals": {
                "path": str(vocals_path),
            }
        },
    }
    (output_dir / "stem_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _band_rms(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    spectrum = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(mono.size, d=1.0 / sample_rate)
    mask = (freqs >= low_hz) & (freqs < high_hz)
    values = spectrum[mask]
    return float(np.sqrt(np.mean(values * values))) if values.size else 0.0


class TestAdaptiveVocalIsolation(unittest.TestCase):
    def test_profile_module_exposes_defaults_without_audio_runtime_imports(self):
        script = f"""
import importlib.util
import json
import sys
from pathlib import Path

module_path = Path({str(PROFILE_MODULE_PATH)!r})
spec = importlib.util.spec_from_file_location("adaptive_vocal_profiles", module_path)
assert spec is not None
assert spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

profiles = module.default_candidate_profiles()
router_profiles = module.default_segment_router_profiles()
payload = {{
    "profile_names": [profile.name for profile in profiles],
    "router_names": [profile.name for profile in router_profiles],
    "ft_requires_bs": module.profiles_require_bs_roformer_runtime([profiles[2]]),
    "hybrid_requires_bs": module.profiles_require_bs_roformer_runtime([profiles[0]]),
    "librosa_loaded": "librosa" in sys.modules,
}}
print(json.dumps(payload, sort_keys=True))
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(
            payload["profile_names"],
            ["hybrid_tail_preserve", "hybrid_balanced_clean", "ft_tail_safe", "bs_artifact_clean"],
        )
        self.assertEqual(
            payload["router_names"],
            [
                "segment_adaptive_vocal_router",
                "busy_vocal_supported_router",
                "ir_residual_supported_router",
                "quality_consistent_ir_router",
            ],
        )
        self.assertFalse(payload["ft_requires_bs"])
        self.assertTrue(payload["hybrid_requires_bs"])
        self.assertFalse(payload["librosa_loaded"])

    def test_default_profiles_include_tail_preservation_and_clean_candidates(self):
        module = _load_adaptive_module()

        profiles = module.default_candidate_profiles()
        derived_profiles = module.default_anchor_suppression_profiles()
        body_repair_profiles = module.default_vocal_body_repair_profiles()
        segment_router_profiles = module.default_segment_router_profiles()

        self.assertEqual(
            [profile.name for profile in profiles],
            ["hybrid_tail_preserve", "hybrid_balanced_clean", "ft_tail_safe", "bs_artifact_clean"],
        )
        self.assertEqual(profiles[0].candidate_mode, "hybrid_vocals")
        self.assertLess(profiles[0].tuning["inactive_high_bs_weight"], profiles[1].tuning["inactive_high_bs_weight"])
        self.assertEqual(profiles[2].candidate_mode, "ft_vocals")
        self.assertEqual(
            [profile.name for profile in derived_profiles],
            ["ft_tail_safe_anchor_clean", "ft_tail_safe_anchor_strict", "ir_residual_anchor_clean"],
        )
        self.assertEqual(derived_profiles[0].source_profile_name, "ft_tail_safe")
        self.assertEqual(derived_profiles[0].anchor_profile_name, "bs_artifact_clean")
        self.assertEqual(
            [profile.name for profile in body_repair_profiles],
            [
                "ft_anchor_clean_vocal_body_repair",
                "ft_anchor_clean_phase_safe_drop_repair",
                "ir_residual_supported_body_repair",
            ],
        )
        self.assertEqual(body_repair_profiles[0].base_profile_name, "ft_tail_safe_anchor_clean")
        self.assertIn("bs_artifact_clean", body_repair_profiles[0].donor_profile_names)
        self.assertEqual(body_repair_profiles[1].tuning["phase_strategy"], "base")
        self.assertEqual(
            [profile.name for profile in segment_router_profiles],
            [
                "segment_adaptive_vocal_router",
                "busy_vocal_supported_router",
                "ir_residual_supported_router",
                "quality_consistent_ir_router",
            ],
        )
        self.assertEqual(segment_router_profiles[0].clean_profile_name, "ft_tail_safe_anchor_clean")
        self.assertEqual(segment_router_profiles[0].repair_profile_name, "ft_anchor_clean_phase_safe_drop_repair")
        self.assertLess(
            segment_router_profiles[1].tuning["body_support_floor"],
            segment_router_profiles[0].tuning["body_support_floor"],
        )
        self.assertGreater(
            segment_router_profiles[1].tuning["body_repair_strength"],
            segment_router_profiles[0].tuning["body_repair_strength"],
        )
        self.assertGreater(
            segment_router_profiles[3].tuning["body_support_floor"],
            segment_router_profiles[2].tuning["body_support_floor"],
        )
        self.assertLess(
            segment_router_profiles[3].tuning["body_repair_strength"],
            segment_router_profiles[2].tuning["body_repair_strength"],
        )
        segment_issue_repair_profile = module.default_segment_issue_repair_profile()
        self.assertEqual(segment_issue_repair_profile.name, "segment_issue_repair")
        self.assertEqual(segment_issue_repair_profile.candidate_mode, "segment_issue_repaired_vocals")
        self.assertLess(
            segment_issue_repair_profile.tuning["dropout_body_support_floor"],
            segment_issue_repair_profile.tuning["body_support_floor"],
        )
        self.assertIn("min_bleed_issue_score", segment_issue_repair_profile.tuning)
        self.assertGreater(
            segment_issue_repair_profile.tuning["bleed_core_debleed_strength"],
            segment_issue_repair_profile.tuning["combined_core_debleed_strength"],
        )
        no_vocal_cleanup_profile = module.default_no_vocal_leakage_cleanup_profile()
        self.assertEqual(no_vocal_cleanup_profile.name, "no_vocal_leakage_cleanup")
        self.assertEqual(no_vocal_cleanup_profile.candidate_mode, "no_vocal_leakage_cleaned_vocals")
        self.assertLess(
            no_vocal_cleanup_profile.tuning["suppression_gain"],
            no_vocal_cleanup_profile.tuning["bridge_suppression_gain"],
        )
        harshness_guard_profile = module.default_vocal_harshness_guard_profile()
        self.assertEqual(harshness_guard_profile.name, "vocal_harshness_guard")
        self.assertEqual(harshness_guard_profile.candidate_mode, "vocal_harshness_guarded_vocals")
        self.assertEqual(harshness_guard_profile.tuning["anchor_profile"], "ft_tail_safe")

    def test_bounded_thin_repair_body_profile_avoids_anchor_clean_dependency(self):
        module = _load_adaptive_module()

        profiles = module.default_bounded_thin_repair_body_profiles()

        self.assertEqual([profile.name for profile in profiles], ["ft_tail_safe_direct_body_presence_repair"])
        self.assertEqual(profiles[0].base_profile_name, "ft_tail_safe")
        self.assertNotIn("ft_tail_safe_anchor_clean", profiles[0].donor_profile_names)
        self.assertIn("hybrid_tail_preserve", profiles[0].donor_profile_names)
        self.assertLessEqual(profiles[0].tuning["max_gain"], 1.05)

    def test_adaptive_decision_contract_routes_blocked_vocal_dropout_to_repair(self):
        module = _load_adaptive_module()

        contract = module.build_adaptive_decision_contract(
            detected_issue_action_contract={
                "status": "pass",
                "summary": {
                    "covered_labels": ["vocal_dropout"],
                    "rejected_labels": [],
                    "uncovered_labels": [],
                },
            },
            defect_effectiveness_gate={
                "status": "blocked_for_listening",
                "release_status": "diagnostic_only",
                "blockers": [
                    {
                        "reason": "selected_candidate_still_has_severe_issue",
                        "labels": ["vocal_dropout"],
                    }
                ],
            },
            vocal_next_action_plan={
                "status": "requires_bounded_defect_specific_repair",
                "failed_labels": ["vocal_dropout"],
            },
        )

        self.assertEqual(contract["mode"], "adaptive_decision_contract")
        self.assertEqual(contract["dominant_failure_class"], "vocal_dropout")
        self.assertEqual(contract["decision"], "repair_once")
        self.assertIn("hard_vad_gate", contract["forbidden_actions"])
        self.assertEqual(contract["evidence"]["candidate_status"], "blocked_for_listening")

    def test_external_candidate_coverage_accepts_equal_duration_different_sample_rate(self):
        module = _load_adaptive_module()

        module._validate_external_candidate_coverage(
            profile_name="htdemucs_ft",
            source_audio=np.zeros((48000 * 10, 2), dtype=np.float32),
            candidate_audio=np.zeros((44100 * 10, 2), dtype=np.float32),
            sample_rate=48000,
            candidate_sample_rate=44100,
        )

    def test_external_candidate_coverage_rejects_short_candidate_at_different_sample_rate(self):
        module = _load_adaptive_module()

        with self.assertRaises(ValueError):
            module._validate_external_candidate_coverage(
                profile_name="short_external",
                source_audio=np.zeros((48000 * 10, 2), dtype=np.float32),
                candidate_audio=np.zeros((44100 * 8, 2), dtype=np.float32),
                sample_rate=48000,
                candidate_sample_rate=44100,
            )

    def test_blend_candidate_segment_clamps_to_shorter_candidate_length(self):
        module = _load_adaptive_module()
        output_audio = np.zeros((10, 2), dtype=np.float32)
        candidate_audio = np.ones((9, 2), dtype=np.float32)

        module._blend_candidate_segment(
            output_audio=output_audio,
            candidate_audio=candidate_audio,
            sample_rate=10,
            relative_start_sec=0.0,
            relative_end_sec=1.0,
            crossfade_sec=0.0,
        )

        np.testing.assert_allclose(output_audio[:9], 1.0)
        np.testing.assert_allclose(output_audio[9:], 0.0)

    def test_blend_candidate_segment_clamps_short_candidate_with_offset_crossfade(self):
        module = _load_adaptive_module()
        output_audio = np.zeros((10, 2), dtype=np.float32)
        candidate_audio = np.ones((8, 2), dtype=np.float32)

        module._blend_candidate_segment(
            output_audio=output_audio,
            candidate_audio=candidate_audio,
            sample_rate=10,
            relative_start_sec=0.2,
            relative_end_sec=1.0,
            crossfade_sec=0.1,
        )

        np.testing.assert_allclose(output_audio[:3], 0.0)
        np.testing.assert_allclose(output_audio[3:8], 1.0)
        np.testing.assert_allclose(output_audio[8:], 0.0)

    def test_score_prefers_tail_preservation_over_aggressive_dropout(self):
        module = _load_adaptive_module()
        preserve_profile, clean_profile, *_ = module.default_candidate_profiles()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            preserve_score = module.score_candidate(
                profile=preserve_profile,
                metrics=_metrics(
                    dropout=0.04,
                    tail_dropout=0.06,
                    rms_ratio_db=-14.0,
                    low_ratio_db=-22.0,
                    kick_ratio_db=-15.0,
                ),
                feedback_adjustments={},
                manifest_path=temp_path / "preserve" / "stem_manifest.json",
                output_dir=temp_path / "preserve",
                vocals_path=temp_path / "preserve" / "vocals.flac",
            )
            aggressive_score = module.score_candidate(
                profile=clean_profile,
                metrics=_metrics(
                    dropout=0.22,
                    tail_dropout=0.38,
                    rms_ratio_db=-24.0,
                    low_ratio_db=-26.0,
                    kick_ratio_db=-18.0,
                ),
                feedback_adjustments={},
                manifest_path=temp_path / "clean" / "stem_manifest.json",
                output_dir=temp_path / "clean",
                vocals_path=temp_path / "clean" / "vocals.flac",
            )

        self.assertGreater(preserve_score.score, aggressive_score.score)
        self.assertTrue(any("tail dropout penalty" in reason for reason in aggressive_score.reasons))

    def test_score_penalizes_unsupported_anchor_leakage(self):
        module = _load_adaptive_module()
        profile = module.default_candidate_profiles()[2]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            clean_metrics = _metrics(
                dropout=0.08,
                tail_dropout=0.08,
                rms_ratio_db=-14.0,
                low_ratio_db=-24.0,
                kick_ratio_db=-15.0,
            )
            clean_metrics.update(
                {
                    "unsupported_vs_anchor_ratio_db": -35.0,
                    "high_unsupported_vs_anchor_ratio_db": -34.0,
                    "high_noise_flatness_6000_16000": 0.08,
                    "stereo_side_mid_ratio_db": -14.0,
                }
            )
            leaky_metrics = dict(clean_metrics)
            leaky_metrics.update(
                {
                    "unsupported_vs_anchor_ratio_db": -14.0,
                    "high_unsupported_vs_anchor_ratio_db": -12.0,
                    "high_noise_flatness_6000_16000": 0.21,
                }
            )

            clean_score = module.score_candidate(
                profile=profile,
                metrics=clean_metrics,
                feedback_adjustments={},
                manifest_path=temp_path / "clean" / "stem_manifest.json",
                output_dir=temp_path / "clean",
                vocals_path=temp_path / "clean" / "vocals.flac",
            )
            leaky_score = module.score_candidate(
                profile=profile,
                metrics=leaky_metrics,
                feedback_adjustments={},
                manifest_path=temp_path / "leaky" / "stem_manifest.json",
                output_dir=temp_path / "leaky",
                vocals_path=temp_path / "leaky" / "vocals.flac",
            )

        self.assertGreater(clean_score.score, leaky_score.score)
        self.assertTrue(any("unsupported-vs-anchor" in reason for reason in leaky_score.reasons))

    def test_score_penalizes_segment_inconsistency(self):
        module = _load_adaptive_module()
        profile = module.default_segment_router_profiles()[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            stable_metrics = _metrics(
                dropout=0.05,
                tail_dropout=0.06,
                rms_ratio_db=-14.0,
                low_ratio_db=-24.0,
                kick_ratio_db=-15.0,
            )
            stable_metrics.update(
                {
                    "segment_consistency_worst_dropout_frame_ratio": 0.16,
                    "segment_consistency_worst_tail_dropout_ratio": 0.18,
                    "segment_consistency_dropout_std": 0.03,
                    "segment_consistency_tail_dropout_std": 0.04,
                    "segment_consistency_rms_ratio_range_db": 3.0,
                }
            )
            uneven_metrics = dict(stable_metrics)
            uneven_metrics.update(
                {
                    "segment_consistency_worst_dropout_frame_ratio": 0.82,
                    "segment_consistency_worst_tail_dropout_ratio": 0.84,
                    "segment_consistency_dropout_std": 0.26,
                    "segment_consistency_tail_dropout_std": 0.28,
                    "segment_consistency_rms_ratio_range_db": 14.0,
                }
            )

            stable_score = module.score_candidate(
                profile=profile,
                metrics=stable_metrics,
                feedback_adjustments={},
                manifest_path=temp_path / "stable" / "stem_manifest.json",
                output_dir=temp_path / "stable",
                vocals_path=temp_path / "stable" / "vocals.flac",
            )
            uneven_score = module.score_candidate(
                profile=profile,
                metrics=uneven_metrics,
                feedback_adjustments={},
                manifest_path=temp_path / "uneven" / "stem_manifest.json",
                output_dir=temp_path / "uneven",
                vocals_path=temp_path / "uneven" / "vocals.flac",
            )

        self.assertGreater(stable_score.score, uneven_score.score)
        self.assertTrue(any("segment worst dropout penalty" in reason for reason in uneven_score.reasons))
        self.assertTrue(any("segment rms consistency penalty" in reason for reason in uneven_score.reasons))

    def test_score_penalizes_segment_issue_hotspots(self):
        module = _load_adaptive_module()
        profile = module.default_segment_router_profiles()[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            clean_metrics = _metrics(
                dropout=0.05,
                tail_dropout=0.06,
                rms_ratio_db=-14.0,
                low_ratio_db=-24.0,
                kick_ratio_db=-15.0,
            )
            clean_metrics.update(
                {
                    "segment_issue_max_score": 0.10,
                    "segment_issue_mean_top3_score": 0.08,
                    "segment_issue_window_ratio": 0.10,
                    "source_stereo_side_mid_ratio_db": -20.0,
                    "backing_vocal_side_loss_db": 0.0,
                }
            )
            hotspot_metrics = dict(clean_metrics)
            hotspot_metrics.update(
                {
                    "segment_issue_max_score": 2.40,
                    "segment_issue_mean_top3_score": 1.70,
                    "segment_issue_window_ratio": 0.75,
                }
            )

            clean_score = module.score_candidate(
                profile=profile,
                metrics=clean_metrics,
                feedback_adjustments={},
                manifest_path=temp_path / "clean" / "stem_manifest.json",
                output_dir=temp_path / "clean",
                vocals_path=temp_path / "clean" / "vocals.flac",
            )
            hotspot_score = module.score_candidate(
                profile=profile,
                metrics=hotspot_metrics,
                feedback_adjustments={},
                manifest_path=temp_path / "hotspot" / "stem_manifest.json",
                output_dir=temp_path / "hotspot",
                vocals_path=temp_path / "hotspot" / "vocals.flac",
            )

        self.assertGreater(clean_score.score, hotspot_score.score)
        self.assertTrue(any("segment issue hotspot penalty" in reason for reason in hotspot_score.reasons))
        self.assertTrue(any("segment issue cluster penalty" in reason for reason in hotspot_score.reasons))
        self.assertTrue(any("segment issue coverage penalty" in reason for reason in hotspot_score.reasons))

    def test_classify_vocal_segment_issue_detects_reverb_tail_smear(self):
        module = _load_adaptive_module()
        track_metrics = _metrics(
            dropout=0.08,
            tail_dropout=0.10,
            rms_ratio_db=-14.0,
            low_ratio_db=-25.0,
            kick_ratio_db=-16.0,
        )
        smear_metrics = _metrics(
            dropout=0.10,
            tail_dropout=0.24,
            rms_ratio_db=-13.0,
            low_ratio_db=-26.0,
            kick_ratio_db=-16.0,
        )
        smear_metrics.update(
            {
                "air_ratio_db_4500_9000": -12.0,
                "high_noise_flatness_6000_16000": 0.31,
                "stereo_side_mid_ratio_db": -2.0,
                "source_stereo_side_mid_ratio_db": -10.0,
            }
        )
        clean_tail_metrics = dict(smear_metrics)
        clean_tail_metrics.update(
            {
                "high_noise_flatness_6000_16000": 0.08,
                "stereo_side_mid_ratio_db": -14.0,
            }
        )

        smear_issue = module.classify_vocal_segment_issue(smear_metrics, track_metrics=track_metrics)
        clean_tail_issue = module.classify_vocal_segment_issue(clean_tail_metrics, track_metrics=track_metrics)

        self.assertIn("excessive_reverb_tail_smear", smear_issue["labels"])
        self.assertNotIn("excessive_reverb_tail_smear", clean_tail_issue["labels"])

    def test_segment_issue_summary_reports_reverb_tail_smear_metrics(self):
        module = _load_adaptive_module()
        summary = module.summarize_vocal_segment_issue_report(
            {
                "summary": {
                    "max_issue_score": 1.40,
                    "mean_top3_issue_score": 0.80,
                    "issue_segments_count": 2.0,
                    "issue_window_ratio": 1.0,
                },
                "segments": [
                    {
                        "severity_score": 1.40,
                        "labels": ["excessive_reverb_tail_smear"],
                        "severity_parts": {"excessive_reverb_tail_smear": 1.20},
                    },
                    {
                        "severity_score": 0.20,
                        "labels": ["high_noise_or_robotic_artifact"],
                        "severity_parts": {"high_noise_or_robotic_artifact": 0.20},
                    },
                ],
            }
        )

        self.assertEqual(summary["segment_reverb_tail_smear_count"], 1.0)
        self.assertEqual(summary["segment_reverb_tail_smear_window_ratio"], 0.5)
        self.assertEqual(summary["segment_reverb_tail_smear_max_score"], 1.2)

    def test_score_penalizes_reverb_tail_smear_hotspots(self):
        module = _load_adaptive_module()
        profile = module.default_segment_router_profiles()[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            clean_metrics = _metrics(
                dropout=0.05,
                tail_dropout=0.06,
                rms_ratio_db=-14.0,
                low_ratio_db=-24.0,
                kick_ratio_db=-15.0,
            )
            clean_metrics.update(
                {
                    "segment_reverb_tail_smear_count": 0.0,
                    "segment_reverb_tail_smear_window_ratio": 0.0,
                    "segment_reverb_tail_smear_max_score": 0.0,
                }
            )
            smear_metrics = dict(clean_metrics)
            smear_metrics.update(
                {
                    "segment_reverb_tail_smear_count": 3.0,
                    "segment_reverb_tail_smear_window_ratio": 0.60,
                    "segment_reverb_tail_smear_max_score": 1.40,
                }
            )

            clean_score = module.score_candidate(
                profile=profile,
                metrics=clean_metrics,
                feedback_adjustments={},
                manifest_path=temp_path / "clean" / "stem_manifest.json",
                output_dir=temp_path / "clean",
                vocals_path=temp_path / "clean" / "vocals.flac",
            )
            smear_score = module.score_candidate(
                profile=profile,
                metrics=smear_metrics,
                feedback_adjustments={},
                manifest_path=temp_path / "smear" / "stem_manifest.json",
                output_dir=temp_path / "smear",
                vocals_path=temp_path / "smear" / "vocals.flac",
            )

        self.assertGreater(clean_score.score, smear_score.score)
        self.assertTrue(any("reverb tail smear penalty" in reason for reason in smear_score.reasons))

    def test_score_rewards_issue_aware_segment_improvements(self):
        module = _load_adaptive_module()
        profile_template = module.default_issue_aware_segment_selection_profile()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            baseline_metrics = _metrics(
                dropout=0.18,
                tail_dropout=0.22,
                rms_ratio_db=-16.0,
                low_ratio_db=-24.0,
                kick_ratio_db=-15.0,
            )
            repaired_metrics = dict(baseline_metrics)
            repaired_metrics.update(
                {
                    "issue_aware_replaced_window_count": 4.0,
                    "issue_aware_adjusted_improvement_sum": 0.40,
                    "issue_aware_adjusted_improvement_max": 0.15,
                    "issue_aware_base_segment_issue_max_score": 1.40,
                    "issue_aware_base_segment_issue_mean_top3_score": 1.00,
                    "segment_issue_max_score": 1.12,
                    "segment_issue_mean_top3_score": 0.72,
                }
            )

            baseline_score = module.score_candidate(
                profile=profile_template,
                metrics=baseline_metrics,
                feedback_adjustments={},
                manifest_path=temp_path / "baseline" / "stem_manifest.json",
                output_dir=temp_path / "baseline",
                vocals_path=temp_path / "baseline" / "vocals.flac",
            )
            repaired_score = module.score_candidate(
                profile=profile_template,
                metrics=repaired_metrics,
                feedback_adjustments={},
                manifest_path=temp_path / "repaired" / "stem_manifest.json",
                output_dir=temp_path / "repaired",
                vocals_path=temp_path / "repaired" / "vocals.flac",
            )

        self.assertGreater(repaired_score.score, baseline_score.score)
        self.assertTrue(
            any("issue-aware measured improvement bonus" in reason for reason in repaired_score.reasons)
        )

    def test_score_rewards_intentional_no_vocal_cleanup_windows(self):
        module = _load_adaptive_module()
        profile = module.default_no_vocal_leakage_cleanup_profile()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            baseline_metrics = _metrics(
                dropout=0.48,
                tail_dropout=0.58,
                rms_ratio_db=-17.0,
                low_ratio_db=-28.0,
                kick_ratio_db=-14.0,
            )
            baseline_metrics.update(
                {
                    "segment_issue_max_score": 6.0,
                    "segment_issue_mean_top3_score": 5.0,
                    "segment_issue_window_ratio": 1.0,
                }
            )
            cleanup_metrics = dict(baseline_metrics)
            cleanup_metrics.update(
                {
                    "no_vocal_cleanup_window_count": 5.0,
                    "no_vocal_cleanup_confidence_sum": 9.0,
                    "no_vocal_cleanup_suppressed_duration_sec": 25.0,
                }
            )

            baseline_score = module.score_candidate(
                profile=profile,
                metrics=baseline_metrics,
                feedback_adjustments={},
                manifest_path=temp_path / "baseline" / "stem_manifest.json",
                output_dir=temp_path / "baseline",
                vocals_path=temp_path / "baseline" / "vocals.flac",
            )
            cleanup_score = module.score_candidate(
                profile=profile,
                metrics=cleanup_metrics,
                feedback_adjustments={},
                manifest_path=temp_path / "cleanup" / "stem_manifest.json",
                output_dir=temp_path / "cleanup",
                vocals_path=temp_path / "cleanup" / "vocals.flac",
            )

        self.assertGreater(cleanup_score.score, baseline_score.score + 1.0)
        self.assertTrue(any("no-vocal leakage cleanup bonus" in reason for reason in cleanup_score.reasons))

    def test_score_rewards_vocal_harshness_guard_reduction(self):
        module = _load_adaptive_module()
        profile = module.default_vocal_harshness_guard_profile()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            baseline_metrics = _metrics(
                dropout=0.08,
                tail_dropout=0.10,
                rms_ratio_db=-16.0,
                low_ratio_db=-24.0,
                kick_ratio_db=-15.0,
            )
            baseline_metrics.update(
                {
                    "high_noise_flatness_6000_16000": 0.20,
                    "segment_issue_max_score": 1.2,
                    "segment_issue_mean_top3_score": 0.9,
                    "segment_issue_window_ratio": 0.25,
                }
            )
            guarded_metrics = dict(baseline_metrics)
            guarded_metrics.update(
                {
                    "vocal_harshness_guard_window_count": 3.0,
                    "vocal_harshness_guard_high_band_reduction": 0.01,
                    "vocal_harshness_guard_high_band_reduction_ratio": 0.35,
                    "vocal_harshness_guard_route_mean": 0.12,
                }
            )

            baseline_score = module.score_candidate(
                profile=profile,
                metrics=baseline_metrics,
                feedback_adjustments={},
                manifest_path=temp_path / "baseline" / "stem_manifest.json",
                output_dir=temp_path / "baseline",
                vocals_path=temp_path / "baseline" / "vocals.flac",
            )
            guarded_score = module.score_candidate(
                profile=profile,
                metrics=guarded_metrics,
                feedback_adjustments={},
                manifest_path=temp_path / "guarded" / "stem_manifest.json",
                output_dir=temp_path / "guarded",
                vocals_path=temp_path / "guarded" / "vocals.flac",
            )

        self.assertGreater(guarded_score.score, baseline_score.score)
        self.assertTrue(any("vocal harshness guard" in reason for reason in guarded_score.reasons))

    def test_score_does_not_reward_issue_aware_decisions_without_measured_improvement(self):
        module = _load_adaptive_module()
        profile_template = module.default_issue_aware_segment_selection_profile()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            metrics = _metrics(
                dropout=0.18,
                tail_dropout=0.22,
                rms_ratio_db=-16.0,
                low_ratio_db=-24.0,
                kick_ratio_db=-15.0,
            )
            metrics.update(
                {
                    "issue_aware_replaced_window_count": 4.0,
                    "issue_aware_adjusted_improvement_sum": 0.40,
                    "issue_aware_adjusted_improvement_max": 0.15,
                    "issue_aware_base_segment_issue_max_score": 1.00,
                    "issue_aware_base_segment_issue_mean_top3_score": 0.70,
                    "segment_issue_max_score": 1.12,
                    "segment_issue_mean_top3_score": 0.72,
                }
            )
            score = module.score_candidate(
                profile=profile_template,
                metrics=metrics,
                feedback_adjustments={},
                manifest_path=temp_path / "candidate" / "stem_manifest.json",
                output_dir=temp_path / "candidate",
                vocals_path=temp_path / "candidate" / "vocals.flac",
            )

        self.assertFalse(any("issue-aware measured improvement bonus" in reason for reason in score.reasons))
        self.assertTrue(
            any("did not improve recalculated issue metrics" in reason for reason in score.reasons)
        )

    def test_score_penalizes_backing_vocal_side_loss(self):
        module = _load_adaptive_module()
        profile = module.default_candidate_profiles()[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            preserved_metrics = _metrics(
                dropout=0.05,
                tail_dropout=0.06,
                rms_ratio_db=-14.0,
                low_ratio_db=-24.0,
                kick_ratio_db=-15.0,
            )
            preserved_metrics.update(
                {
                    "source_stereo_side_mid_ratio_db": -6.0,
                    "candidate_stereo_side_mid_ratio_db": -7.0,
                    "backing_vocal_side_loss_db": 1.0,
                }
            )
            narrow_metrics = dict(preserved_metrics)
            narrow_metrics.update(
                {
                    "candidate_stereo_side_mid_ratio_db": -22.0,
                    "backing_vocal_side_loss_db": 16.0,
                }
            )

            preserved_score = module.score_candidate(
                profile=profile,
                metrics=preserved_metrics,
                feedback_adjustments={},
                manifest_path=temp_path / "preserved" / "stem_manifest.json",
                output_dir=temp_path / "preserved",
                vocals_path=temp_path / "preserved" / "vocals.flac",
            )
            narrow_score = module.score_candidate(
                profile=profile,
                metrics=narrow_metrics,
                feedback_adjustments={},
                manifest_path=temp_path / "narrow" / "stem_manifest.json",
                output_dir=temp_path / "narrow",
                vocals_path=temp_path / "narrow" / "vocals.flac",
            )

        self.assertGreater(preserved_score.score, narrow_score.score)
        self.assertTrue(any("backing vocal side loss penalty" in reason for reason in narrow_score.reasons))

    def test_classify_vocal_segment_issue_covers_expected_labels(self):
        module = _load_adaptive_module()
        track_metrics = _metrics(
            dropout=0.05,
            tail_dropout=0.05,
            rms_ratio_db=-14.0,
            low_ratio_db=-24.0,
            kick_ratio_db=-15.0,
        )
        dropout_and_thin_metrics = _metrics(
            dropout=0.91,
            tail_dropout=0.92,
            rms_ratio_db=-35.0,
            low_ratio_db=-24.0,
            kick_ratio_db=-15.0,
        )
        dropout_and_thin_metrics.update(
            {
                "presence_ratio_db_1800_4500": -36.0,
                "air_ratio_db_4500_9000": -47.0,
            }
        )
        leaky_noise_metrics = _metrics(
            dropout=0.05,
            tail_dropout=0.05,
            rms_ratio_db=-2.0,
            low_ratio_db=-10.0,
            kick_ratio_db=-5.0,
        )
        leaky_noise_metrics.update(
            {
                "unsupported_vs_anchor_ratio_db": -10.0,
                "high_unsupported_vs_anchor_ratio_db": -8.0,
                "high_noise_flatness_6000_16000": 0.30,
                "stereo_side_mid_ratio_db": -2.0,
            }
        )
        backing_loss_metrics = _metrics(
            dropout=0.05,
            tail_dropout=0.05,
            rms_ratio_db=-16.0,
            low_ratio_db=-24.0,
            kick_ratio_db=-15.0,
        )
        backing_loss_metrics.update(
            {
                "source_stereo_side_mid_ratio_db": -5.0,
                "candidate_stereo_side_mid_ratio_db": -20.0,
                "backing_vocal_side_loss_db": 15.0,
            }
        )
        supported_body_dip_metrics = _metrics(
            dropout=0.33,
            tail_dropout=0.40,
            rms_ratio_db=-24.5,
            low_ratio_db=-24.0,
            kick_ratio_db=-15.0,
        )
        supported_body_dip_metrics.update(
            {
                "unsupported_vs_anchor_ratio_db": -28.0,
                "high_unsupported_vs_anchor_ratio_db": -30.0,
                "high_noise_flatness_6000_16000": 0.12,
            }
        )
        unsupported_body_dip_metrics = dict(supported_body_dip_metrics)
        unsupported_body_dip_metrics["unsupported_vs_anchor_ratio_db"] = -8.0

        dropout_and_thin = module.classify_vocal_segment_issue(
            dropout_and_thin_metrics,
            track_metrics=track_metrics,
        )
        leaky_noise = module.classify_vocal_segment_issue(
            leaky_noise_metrics,
            track_metrics=track_metrics,
        )
        backing_loss = module.classify_vocal_segment_issue(
            backing_loss_metrics,
            track_metrics=track_metrics,
        )
        supported_body_dip = module.classify_vocal_segment_issue(
            supported_body_dip_metrics,
            track_metrics=track_metrics,
        )
        unsupported_body_dip = module.classify_vocal_segment_issue(
            unsupported_body_dip_metrics,
            track_metrics=track_metrics,
        )
        labels = (
            set(dropout_and_thin["labels"])
            | set(leaky_noise["labels"])
            | set(backing_loss["labels"])
            | set(supported_body_dip["labels"])
        )

        self.assertTrue(
            {
                "vocal_dropout",
                "vocal_tail_loss",
                "thin_or_overcleaned",
                "leaky_full_mix_risk",
                "low_band_bleed",
                "kick_punch_bleed",
                "unsupported_anchor_leakage",
                "high_band_unsupported_leakage",
                "high_noise_or_robotic_artifact",
                "wide_stereo_non_vocal_leakage",
                "backing_vocal_loss_risk",
                "local_vocal_body_dip",
                "presence_loss",
                "air_detail_loss",
            }.issubset(labels)
        )
        self.assertIn("local_vocal_body_dip", supported_body_dip["labels"])
        self.assertNotIn("local_vocal_body_dip", unsupported_body_dip["labels"])

    def test_classify_vocal_segment_issue_ignores_ratio_bleed_on_both_silent_windows(self):
        module = _load_adaptive_module()
        track_metrics = _metrics(
            dropout=0.05,
            tail_dropout=0.05,
            rms_ratio_db=-14.0,
            low_ratio_db=-24.0,
            kick_ratio_db=-15.0,
        )
        both_silent_metrics = _metrics(
            dropout=0.0,
            tail_dropout=0.0,
            rms_ratio_db=0.0,
            low_ratio_db=0.0,
            kick_ratio_db=0.0,
        )
        both_silent_metrics.update(
            {
                "source_rms_dbfs": -200.0,
                "candidate_rms_dbfs": -200.0,
                "candidate_peak_dbfs": -200.0,
                "presence_ratio_db_1800_4500": 0.0,
                "air_ratio_db_4500_9000": 0.0,
            }
        )

        issue = module.classify_vocal_segment_issue(both_silent_metrics, track_metrics=track_metrics)

        self.assertEqual(issue["labels"], ["acceptable_proxy_window"])
        self.assertEqual(issue["severity_score"], 0.0)
        self.assertEqual(issue["severity_parts"], {})

    def test_classify_vocal_segment_issue_keeps_source_silent_candidate_active_leakage(self):
        module = _load_adaptive_module()
        track_metrics = _metrics(
            dropout=0.05,
            tail_dropout=0.05,
            rms_ratio_db=-14.0,
            low_ratio_db=-24.0,
            kick_ratio_db=-15.0,
        )
        leaked_metrics = _metrics(
            dropout=0.0,
            tail_dropout=0.0,
            rms_ratio_db=150.0,
            low_ratio_db=-4.0,
            kick_ratio_db=-3.0,
        )
        leaked_metrics.update(
            {
                "source_rms_dbfs": -200.0,
                "candidate_rms_dbfs": -50.0,
                "candidate_peak_dbfs": -32.0,
            }
        )

        issue = module.classify_vocal_segment_issue(leaked_metrics, track_metrics=track_metrics)

        self.assertIn("leaky_full_mix_risk", issue["labels"])
        self.assertIn("low_band_bleed", issue["labels"])
        self.assertIn("kick_punch_bleed", issue["labels"])
        self.assertGreater(issue["severity_score"], 0.0)

    def test_feedback_store_adjusts_future_profile_selection(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_store = Path(temp_dir) / "feedback.jsonl"
            feedback_store.write_text(
                json.dumps({"profile": "hybrid_tail_preserve", "rating": "accept", "score": 0.7})
                + "\n"
                + json.dumps({"profile": "bs_artifact_clean", "rating": "reject", "swallowed_endings": True})
                + "\n"
                + json.dumps({"profile": "ft_tail_safe", "rating": "reject", "backing_vocal_loss": True})
                + "\n",
                encoding="utf-8",
            )
            adjustments = module.load_feedback_adjustments(feedback_store)

        self.assertGreater(adjustments["hybrid_tail_preserve"], 0.0)
        self.assertLess(adjustments["bs_artifact_clean"], 0.0)
        self.assertLess(adjustments["ft_tail_safe"], 0.0)

    def test_listener_verified_feedback_is_context_scoped_and_strong(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_store = Path(temp_dir) / "feedback.jsonl"
            feedback_store.write_text(
                json.dumps(
                    {
                        "profile": "bs_artifact_clean",
                        "rating": "better",
                        "score": 1.0,
                        "listener_verified": True,
                        "confidence": 1.0,
                        "source_audio_sha256": "abc123",
                        "window_sec": {"start": 0.0, "end": 30.0},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            matching_adjustments = module.load_feedback_adjustments(
                feedback_store,
                source_audio_sha256="abc123",
                start_sec=0.0,
                end_sec=30.0,
            )
            mismatched_adjustments = module.load_feedback_adjustments(
                feedback_store,
                source_audio_sha256="different",
                start_sec=0.0,
                end_sec=30.0,
            )
            missing_window_adjustments = module.load_feedback_adjustments(
                feedback_store,
                source_audio_sha256="abc123",
            )

        self.assertGreater(matching_adjustments["bs_artifact_clean"], 6.0)
        self.assertEqual(mismatched_adjustments, {})
        self.assertEqual(missing_window_adjustments, {})

    def test_listener_verified_feedback_requires_complete_window(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_store = Path(temp_dir) / "feedback.jsonl"
            feedback_store.write_text(
                json.dumps(
                    {
                        "profile": "bs_artifact_clean",
                        "rating": "better",
                        "listener_verified": True,
                        "source_audio_sha256": "abc123",
                        "window_sec": {"start": 0.0},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            adjustments = module.load_feedback_adjustments(
                feedback_store,
                source_audio_sha256="abc123",
                start_sec=0.0,
                end_sec=30.0,
            )

        self.assertEqual(adjustments, {})

    def test_holdout_feedback_does_not_adjust_training_priors(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_store = Path(temp_dir) / "feedback.jsonl"
            feedback_store.write_text(
                json.dumps(
                    {
                        "profile": "bs_artifact_clean",
                        "rating": "better",
                        "listener_verified": True,
                        "confidence": 1.0,
                        "source_audio_sha256": "abc123",
                        "window_sec": {"start": 0.0, "end": 30.0},
                        "feedback_split": "holdout",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            adjustments = module.load_feedback_adjustments(
                feedback_store,
                source_audio_sha256="abc123",
                start_sec=0.0,
                end_sec=30.0,
            )

        self.assertEqual(adjustments, {})

    def test_proxy_metrics_use_requested_source_window(self):
        module = _load_adaptive_module()
        sample_rate = 16000
        t = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
        voice = 0.25 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
        silence = np.zeros_like(voice, dtype=np.float32)
        source = np.stack([np.concatenate([silence, voice]), np.concatenate([silence, voice])], axis=1)
        vocals = np.stack([voice, voice], axis=1).astype(np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            vocals_path = temp_path / "vocals.flac"
            sf.write(source_path, source, sample_rate)
            sf.write(vocals_path, vocals, sample_rate)

            metrics = module.calculate_vocal_proxy_metrics(
                source_path,
                vocals_path,
                start_sec=1.0,
                end_sec=2.0,
            )

        self.assertLess(metrics["dropout_frame_ratio"], 0.15)
        self.assertGreater(metrics["candidate_to_source_rms_db"], -0.2)
        self.assertLess(metrics["candidate_to_source_rms_db"], 0.2)

    def test_proxy_metrics_detect_backing_vocal_side_loss_from_audio(self):
        module = _load_adaptive_module()
        sample_rate = 16000
        t = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
        lead_body = (
            0.18 * np.sin(2.0 * np.pi * 330.0 * t)
            + 0.05 * np.sin(2.0 * np.pi * 1900.0 * t)
        ).astype(np.float32)
        wide_backing = (0.12 * np.sin(2.0 * np.pi * 660.0 * t)).astype(np.float32)
        source = np.stack([lead_body + wide_backing, lead_body - wide_backing], axis=1).astype(np.float32)
        narrowed_vocals = np.stack([lead_body, lead_body], axis=1).astype(np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.flac"
            vocals_path = temp_path / "vocals.flac"
            sf.write(source_path, source, sample_rate, subtype="PCM_24")
            sf.write(vocals_path, narrowed_vocals, sample_rate, subtype="PCM_24")

            metrics = module.calculate_vocal_proxy_metrics(
                source_path,
                vocals_path,
                start_sec=0.0,
                end_sec=1.0,
            )

        issue = module.classify_vocal_segment_issue(metrics, track_metrics=metrics)

        self.assertGreater(metrics["source_stereo_side_mid_ratio_db"], -12.0)
        self.assertLess(metrics["candidate_stereo_side_mid_ratio_db"], -40.0)
        self.assertGreater(metrics["backing_vocal_side_loss_db"], 6.0)
        self.assertIn("backing_vocal_loss_risk", issue["labels"])

    def test_proxy_metrics_detect_drop_melody_leakage_from_sustained_tonal_residue(self):
        module = _load_adaptive_module()
        sample_rate = 16000
        t = np.arange(sample_rate * 4, dtype=np.float32) / float(sample_rate)
        vocal_body = (
            0.10 * np.sin(2.0 * np.pi * (240.0 + 5.0 * np.sin(2.0 * np.pi * 4.3 * t)) * t)
            * (0.65 + 0.35 * np.sin(2.0 * np.pi * 5.7 * t))
        ).astype(np.float32)
        clean_vocals = np.stack([vocal_body, vocal_body], axis=1).astype(np.float32)
        drop_melody = (
            0.06 * np.sin(2.0 * np.pi * 1180.0 * t)
            + 0.035 * np.sin(2.0 * np.pi * 1770.0 * t)
        ).astype(np.float32)
        dirty_vocals = np.stack([vocal_body + drop_melody, vocal_body + drop_melody], axis=1).astype(np.float32)
        source = dirty_vocals.copy()

        clean_metrics = module._calculate_vocal_proxy_metrics_from_audio(source, clean_vocals, sample_rate)
        dirty_metrics = module._calculate_vocal_proxy_metrics_from_audio(source, dirty_vocals, sample_rate)
        clean_issue = module.classify_vocal_segment_issue(clean_metrics, track_metrics=dirty_metrics)
        dirty_issue = module.classify_vocal_segment_issue(dirty_metrics, track_metrics=dirty_metrics)

        self.assertLess(dirty_metrics["mid_tonal_flatness_700_1800"], 0.095)
        self.assertGreater(dirty_metrics["mid_tonal_peak_ratio_700_1800"], 0.18)
        self.assertIn("drop_melody_leakage", dirty_issue["labels"])
        self.assertNotIn("drop_melody_leakage", clean_issue["labels"])
        self.assertGreater(dirty_issue["severity_parts"]["drop_melody_leakage"], 0.20)

    def test_proxy_consistency_metrics_detect_segment_dropout(self):
        module = _load_adaptive_module()
        sample_rate = 8000
        t = np.arange(sample_rate * 4, dtype=np.float32) / float(sample_rate)
        voice = (0.25 * np.sin(2.0 * np.pi * 330.0 * t)).astype(np.float32)
        source = np.stack([voice, voice], axis=1).astype(np.float32)
        vocals = source.copy()
        vocals[sample_rate * 2 : sample_rate * 3] = 0.0

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.flac"
            vocals_path = temp_path / "vocals.flac"
            sf.write(source_path, source, sample_rate, subtype="PCM_24")
            sf.write(vocals_path, vocals, sample_rate, subtype="PCM_24")

            metrics = module.calculate_vocal_proxy_consistency_metrics(
                source_path,
                vocals_path,
                start_sec=0.0,
                end_sec=4.0,
                segment_duration_sec=1.0,
            )

        self.assertEqual(metrics["segment_consistency_count"], 4.0)
        self.assertGreater(metrics["segment_consistency_worst_dropout_frame_ratio"], 0.90)
        self.assertGreater(metrics["segment_consistency_rms_ratio_range_db"], 20.0)

    def test_segment_issue_report_ranks_dropout_and_unsupported_noise_windows(self):
        module = _load_adaptive_module()
        sample_rate = 16000
        t = np.arange(sample_rate * 4, dtype=np.float32) / float(sample_rate)
        voice = (
            0.20 * np.sin(2.0 * np.pi * 330.0 * t)
            + 0.06 * np.sin(2.0 * np.pi * 1900.0 * t)
        ).astype(np.float32)
        source = np.stack([voice, voice], axis=1).astype(np.float32)
        vocals = source.copy()
        anchor = source.copy()
        vocals[sample_rate : sample_rate * 2] = 0.0
        rng = np.random.default_rng(123)
        noisy_window = slice(sample_rate * 2, sample_rate * 3)
        high_noise = (0.10 * rng.standard_normal(sample_rate)).astype(np.float32)
        low_bleed = (0.16 * np.sin(2.0 * np.pi * 70.0 * t[:sample_rate])).astype(np.float32)
        vocals[noisy_window, 0] += high_noise + low_bleed
        vocals[noisy_window, 1] += high_noise - low_bleed

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.flac"
            vocals_path = temp_path / "vocals.flac"
            anchor_path = temp_path / "anchor.flac"
            sf.write(source_path, source, sample_rate, subtype="PCM_24")
            sf.write(vocals_path, vocals, sample_rate, subtype="PCM_24")
            sf.write(anchor_path, anchor, sample_rate, subtype="PCM_24")

            report = module.calculate_vocal_segment_issue_report(
                source_path,
                vocals_path,
                anchor_vocals_path=anchor_path,
                start_sec=0.0,
                end_sec=4.0,
                segment_duration_sec=1.0,
                max_ranked_segments=4,
            )

        self.assertEqual(report["segments_analyzed"], 4)
        self.assertGreater(report["summary"]["issue_segments_count"], 0.0)
        self.assertGreater(report["summary_metrics"]["segment_issue_max_score"], 0.0)
        dropout_segment = next(segment for segment in report["segments"] if segment["start_sec"] == 1.0)
        noisy_segment = next(segment for segment in report["segments"] if segment["start_sec"] == 2.0)
        self.assertIn("vocal_dropout", dropout_segment["labels"])
        self.assertIn("vocal_tail_loss", dropout_segment["labels"])
        self.assertIn("unsupported_anchor_leakage", noisy_segment["labels"])
        self.assertTrue(
            "high_noise_or_robotic_artifact" in noisy_segment["labels"]
            or "low_band_bleed" in noisy_segment["labels"]
        )

    def test_segment_issue_report_keeps_summary_when_ranked_window_limit_is_zero(self):
        module = _load_adaptive_module()
        sample_rate = 8000
        t = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
        source_mono = (0.25 * np.sin(2.0 * np.pi * 330.0 * t)).astype(np.float32)
        source = np.stack([source_mono, source_mono], axis=1).astype(np.float32)
        vocals = source.copy()
        vocals[sample_rate : sample_rate * 2] = 0.0

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.flac"
            vocals_path = temp_path / "vocals.flac"
            sf.write(source_path, source, sample_rate, subtype="PCM_24")
            sf.write(vocals_path, vocals, sample_rate, subtype="PCM_24")

            report = module.calculate_vocal_segment_issue_report(
                source_path,
                vocals_path,
                anchor_vocals_path=None,
                start_sec=0.0,
                end_sec=2.0,
                segment_duration_sec=1.0,
                max_ranked_segments=0,
            )

        self.assertEqual(report["segments_analyzed"], 2)
        self.assertEqual(report["top_segments"], [])
        self.assertGreater(report["summary"]["issue_segments_count"], 0.0)
        self.assertGreater(report["summary_metrics"]["segment_issue_max_score"], 0.0)

    def test_segment_issue_listen_clips_are_diagnostic_only(self):
        module = _load_adaptive_module()
        sample_rate = 8000
        source_audio = np.zeros((sample_rate * 4, 2), dtype=np.float32)
        source_audio[:sample_rate] = 0.10
        source_audio[sample_rate : sample_rate * 2] = 0.20
        source_audio[sample_rate * 2 : sample_rate * 3] = 0.30
        source_audio[sample_rate * 3 :] = 0.40
        selected_audio = np.zeros((sample_rate * 2, 2), dtype=np.float32)
        selected_audio[:sample_rate] = 0.50
        selected_audio[sample_rate:] = 0.75
        report = {
            "top_segments": [
                {
                    "start_sec": 2.0,
                    "end_sec": 2.5,
                    "relative_start_sec": 0.5,
                    "relative_end_sec": 1.0,
                    "severity_score": 1.2,
                    "labels": ["vocal_dropout", "vocal_tail_loss"],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.flac"
            vocals_path = temp_path / "vocals.flac"
            listen_dir = temp_path / "listen"
            sf.write(source_path, source_audio, sample_rate, subtype="PCM_24")
            sf.write(vocals_path, selected_audio, sample_rate, subtype="PCM_24")

            artifacts = module.write_segment_issue_listen_clips(
                source_audio_path=source_path,
                selected_vocals_path=vocals_path,
                issue_report=report,
                listen_dir=listen_dir,
                max_clips=1,
            )
            clips_manifest = json.loads(
                (listen_dir / "segment_issues" / "segment_issue_clips.json").read_text(encoding="utf-8"),
            )
            raw_clip_exists = Path(clips_manifest["clips"][0]["raw_clip"]).exists()
            selected_clip_exists = Path(clips_manifest["clips"][0]["selected_vocals_clip"]).exists()
            raw_clip, raw_sr = sf.read(clips_manifest["clips"][0]["raw_clip"], always_2d=True, dtype="float32")
            selected_clip, selected_sr = sf.read(
                clips_manifest["clips"][0]["selected_vocals_clip"],
                always_2d=True,
                dtype="float32",
            )

        self.assertIn("segment_issue_clips_manifest", artifacts)
        self.assertEqual(clips_manifest["mode"], "vocal_segment_issue_diagnostic_clips")
        self.assertEqual(clips_manifest["delivery_status"], "diagnostic_only")
        self.assertFalse(clips_manifest["delivery_allowed"])
        self.assertEqual(clips_manifest["presentation_status"], "diagnostic_segment_issue_evidence_only")
        self.assertEqual(Path(clips_manifest["clips"][0]["raw_clip"]).name, "r001.flac")
        self.assertEqual(Path(clips_manifest["clips"][0]["selected_vocals_clip"]).name, "v001.flac")
        self.assertEqual(len(clips_manifest["clips"]), 1)
        self.assertTrue(raw_clip_exists)
        self.assertTrue(selected_clip_exists)
        self.assertEqual(raw_sr, sample_rate)
        self.assertEqual(selected_sr, sample_rate)
        self.assertEqual(raw_clip.shape[0], sample_rate // 2)
        self.assertEqual(selected_clip.shape[0], sample_rate // 2)
        self.assertAlmostEqual(float(np.mean(raw_clip)), 0.30, places=2)
        self.assertAlmostEqual(float(np.mean(selected_clip)), 0.50, places=2)

    def test_issue_aware_segment_selection_replaces_only_improved_issue_windows(self):
        module = _load_adaptive_module()
        sample_rate = 8000
        t = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
        base_mono = (0.14 * np.sin(2.0 * np.pi * 240.0 * t)).astype(np.float32)
        alternate_mono = base_mono.copy()
        base_mono[:sample_rate] *= 0.25
        alternate_mono[sample_rate:] *= 0.60
        base_audio = np.stack([base_mono, base_mono], axis=1).astype(np.float32)
        alternate_audio = np.stack([alternate_mono, alternate_mono], axis=1).astype(np.float32)
        profile_template = module.default_issue_aware_segment_selection_profile()
        issue_profile = module.VocalCandidateProfile(
            name=profile_template.name,
            candidate_mode=profile_template.candidate_mode,
            prior_score=profile_template.prior_score,
            intent=profile_template.intent,
            tuning={**profile_template.tuning, "crossfade_sec": 0.0},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            base_dir = temp_path / "base"
            alternate_dir = temp_path / "alternate"
            output_dir = temp_path / "issue_aware"
            base_dir.mkdir()
            alternate_dir.mkdir()
            base_path = base_dir / "vocals.flac"
            alternate_path = alternate_dir / "vocals.flac"
            sf.write(base_path, base_audio, sample_rate, subtype="PCM_24")
            sf.write(alternate_path, alternate_audio, sample_rate, subtype="PCM_24")

            def write_manifest(candidate_dir: Path, profile_name: str, vocals_path: Path) -> dict[str, object]:
                manifest = {
                    "schema_version": 1,
                    "audio_path": str(temp_path / "source.flac"),
                    "sample_rate": sample_rate,
                    "preview_window_sec": {"start": 0.0, "end": 2.0},
                    "pipeline": {"candidate_mode": profile_name},
                    "tuning": {},
                    "stems": {"vocals": {"path": str(vocals_path)}},
                }
                (candidate_dir / "stem_manifest.json").write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                return manifest

            base_manifest = write_manifest(base_dir, "base_mode", base_path)
            alternate_manifest = write_manifest(alternate_dir, "alternate_mode", alternate_path)
            base_record = module.VocalCandidateRecord(
                profile=module.VocalCandidateProfile("base", "base_mode", 0.0, "base", {}),
                manifest_path=base_dir / "stem_manifest.json",
                output_dir=base_dir,
                vocals_path=base_path,
                metrics={},
                manifest=base_manifest,
            )
            alternate_record = module.VocalCandidateRecord(
                profile=module.VocalCandidateProfile("alternate", "alternate_mode", 0.0, "alternate", {}),
                manifest_path=alternate_dir / "stem_manifest.json",
                output_dir=alternate_dir,
                vocals_path=alternate_path,
                metrics={},
                manifest=alternate_manifest,
            )
            base_score = module.VocalCandidateScore(
                profile_name="base",
                score=1.0,
                reasons=[],
                metrics={},
                manifest_path=str(base_record.manifest_path),
                output_dir=str(base_dir),
                vocals_path=str(base_path),
            )
            alternate_score = module.VocalCandidateScore(
                profile_name="alternate",
                score=0.9,
                reasons=[],
                metrics={},
                manifest_path=str(alternate_record.manifest_path),
                output_dir=str(alternate_dir),
                vocals_path=str(alternate_path),
            )
            base_report = {
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 1.0,
                        "relative_start_sec": 0.0,
                        "relative_end_sec": 1.0,
                        "severity_score": 1.20,
                        "labels": ["vocal_dropout", "vocal_tail_loss"],
                    },
                    {
                        "start_sec": 1.0,
                        "end_sec": 2.0,
                        "relative_start_sec": 1.0,
                        "relative_end_sec": 2.0,
                        "severity_score": 0.10,
                        "labels": ["acceptable_proxy_window"],
                    },
                ],
            }
            alternate_report = {
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 1.0,
                        "relative_start_sec": 0.0,
                        "relative_end_sec": 1.0,
                        "severity_score": 0.18,
                        "labels": ["acceptable_proxy_window"],
                    },
                    {
                        "start_sec": 1.0,
                        "end_sec": 2.0,
                        "relative_start_sec": 1.0,
                        "relative_end_sec": 2.0,
                        "severity_score": 0.62,
                        "labels": ["thin_or_overcleaned"],
                    },
                ],
            }

            manifest = module.build_issue_aware_segment_selection_candidate(
                profile=issue_profile,
                base_score=base_score,
                candidate_scores=[base_score, alternate_score],
                candidate_records_by_name={"base": base_record, "alternate": alternate_record},
                candidate_issue_reports={"base": base_report, "alternate": alternate_report},
                dependency_fingerprints={
                    "base": {"score": "1.000000"},
                    "alternate": {"score": "0.900000"},
                },
                output_dir=output_dir,
            )
            routed, routed_sr = sf.read(output_dir / "vocals.flac", always_2d=True)

        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.assertEqual(routed_sr, sample_rate)
        self.assertEqual(manifest["stems"]["vocals"]["replaced_window_count"], 1)
        self.assertEqual(
            manifest["stems"]["vocals"]["window_decisions"][0]["selected_profile"],
            "alternate",
        )
        self.assertGreater(
            _band_rms(routed[:sample_rate], sample_rate, 180.0, 320.0),
            2.5 * _band_rms(base_audio[:sample_rate], sample_rate, 180.0, 320.0),
        )
        self.assertLess(
            abs(
                _band_rms(routed[sample_rate:], sample_rate, 180.0, 320.0)
                - _band_rms(base_audio[sample_rate:], sample_rate, 180.0, 320.0)
            ),
            0.02 * _band_rms(base_audio[sample_rate:], sample_rate, 180.0, 320.0),
        )

    def test_issue_aware_segment_selection_reclassifies_donor_windows_against_base_track_metrics(self):
        module = _load_adaptive_module()

        profile_template = module.default_issue_aware_segment_selection_profile()
        tuning = dict(profile_template.tuning)

        base_track_metrics = _metrics(
            dropout=0.24,
            tail_dropout=0.22,
            rms_ratio_db=-12.0,
            low_ratio_db=-10.0,
            kick_ratio_db=-4.0,
        )
        globally_weak_track_metrics = _metrics(
            dropout=0.48,
            tail_dropout=0.49,
            rms_ratio_db=-16.0,
            low_ratio_db=-15.0,
            kick_ratio_db=-7.0,
        )
        cleaner_track_metrics = _metrics(
            dropout=0.12,
            tail_dropout=0.14,
            rms_ratio_db=-7.0,
            low_ratio_db=-20.0,
            kick_ratio_db=-14.0,
        )

        def candidate_score(profile_name: str, score: float, metrics: dict[str, float]):
            return module.VocalCandidateScore(
                profile_name=profile_name,
                score=score,
                reasons=[],
                metrics=metrics,
                manifest_path=f"/tmp/{profile_name}/stem_manifest.json",
                output_dir=f"/tmp/{profile_name}",
                vocals_path=f"/tmp/{profile_name}/vocals.flac",
            )

        def segment(
            *,
            severity: float,
            labels: list[str],
            metrics: dict[str, float],
        ) -> dict[str, object]:
            return {
                "start_sec": 170.0,
                "end_sec": 175.0,
                "relative_start_sec": 170.0,
                "relative_end_sec": 175.0,
                "severity_score": severity,
                "labels": labels,
                "metrics": metrics,
            }

        base_score = candidate_score("base", 10.0, base_track_metrics)
        globally_weak_score = candidate_score("globally_weak", 8.0, globally_weak_track_metrics)
        cleaner_score = candidate_score("cleaner", 8.0, cleaner_track_metrics)

        base_segment = segment(
            severity=1.7,
            labels=["vocal_dropout", "vocal_tail_loss", "kick_punch_bleed"],
            metrics=_metrics(
                dropout=0.70,
                tail_dropout=0.75,
                rms_ratio_db=-12.0,
                low_ratio_db=-10.0,
                kick_ratio_db=-4.0,
            ),
        )
        globally_weak_segment = segment(
            severity=0.30,
            labels=["acceptable_proxy_window"],
            metrics=_metrics(
                dropout=0.66,
                tail_dropout=0.72,
                rms_ratio_db=-11.5,
                low_ratio_db=-14.0,
                kick_ratio_db=-5.6,
            ),
        )
        cleaner_segment = segment(
            severity=0.95,
            labels=["vocal_tail_loss"],
            metrics=_metrics(
                dropout=0.45,
                tail_dropout=0.52,
                rms_ratio_db=-7.0,
                low_ratio_db=-20.0,
                kick_ratio_db=-14.0,
            ),
        )

        decisions = module._build_issue_aware_window_decisions(
            base_score=base_score,
            candidate_scores=[base_score, globally_weak_score, cleaner_score],
            candidate_issue_reports={
                "base": {"segments": [base_segment]},
                "globally_weak": {"segments": [globally_weak_segment]},
                "cleaner": {"segments": [cleaner_segment]},
            },
            tuning=tuning,
        )

        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["selected_profile"], "cleaner")
        self.assertIn("shared_track_metrics", decisions[0]["selection_basis"])

    def test_segment_issue_repair_candidate_repairs_only_diagnosed_window(self):
        module = _load_adaptive_module()
        sample_rate = 16000
        t = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
        voice = (0.14 * np.sin(2.0 * np.pi * 260.0 * t)).astype(np.float32)
        high_noise = (0.05 * np.sin(2.0 * np.pi * 6500.0 * t)).astype(np.float32)
        base_mono = voice.copy()
        base_mono[:sample_rate] *= 0.25
        donor_mono = voice.copy()
        donor_mono[:sample_rate] += high_noise[:sample_rate]
        strict_mono = voice.copy()
        anchor_mono = voice.copy()
        source_mono = voice + 0.03 * np.sin(2.0 * np.pi * 120.0 * t).astype(np.float32)
        base_audio = np.stack([base_mono, base_mono], axis=1).astype(np.float32)
        donor_audio = np.stack([donor_mono, donor_mono], axis=1).astype(np.float32)
        strict_audio = np.stack([strict_mono, strict_mono], axis=1).astype(np.float32)
        anchor_audio = np.stack([anchor_mono, anchor_mono], axis=1).astype(np.float32)
        source_audio = np.stack([source_mono, source_mono], axis=1).astype(np.float32)
        profile_template = module.default_segment_issue_repair_profile()
        profile = module.VocalCandidateProfile(
            name=profile_template.name,
            candidate_mode=profile_template.candidate_mode,
            prior_score=profile_template.prior_score,
            intent=profile_template.intent,
            tuning={
                **profile_template.tuning,
                "crossfade_sec": 0.0,
                "segment_consistency_sec": 1.0,
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.flac"
            sf.write(source_path, source_audio, sample_rate, subtype="PCM_24")

            def write_candidate(
                profile_name: str,
                candidate_mode: str,
                audio: np.ndarray,
            ) -> module.VocalCandidateRecord:
                candidate_dir = temp_path / profile_name
                candidate_dir.mkdir()
                vocals_path = candidate_dir / "vocals.flac"
                sf.write(vocals_path, audio, sample_rate, subtype="PCM_24")
                manifest = {
                    "schema_version": 1,
                    "audio_path": str(source_path),
                    "source_audio_sha256": module._sha256_file(source_path),
                    "sample_rate": sample_rate,
                    "preview_window_sec": {"start": 0.0, "end": 2.0},
                    "pipeline": {"candidate_mode": candidate_mode},
                    "tuning": {},
                    "stems": {"vocals": {"path": str(vocals_path)}},
                }
                manifest_path = candidate_dir / "stem_manifest.json"
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return module.VocalCandidateRecord(
                    profile=module.VocalCandidateProfile(profile_name, candidate_mode, 0.0, profile_name, {}),
                    manifest_path=manifest_path,
                    output_dir=candidate_dir,
                    vocals_path=vocals_path,
                    metrics={},
                    manifest=manifest,
                )

            base_record = write_candidate("base", "base_mode", base_audio)
            donor_record = write_candidate("donor", "donor_mode", donor_audio)
            strict_record = write_candidate("ft_tail_safe_anchor_strict", "strict_mode", strict_audio)
            anchor_record = write_candidate("bs_artifact_clean", "anchor_mode", anchor_audio)
            records = {
                record.profile.name: record
                for record in (base_record, donor_record, strict_record, anchor_record)
            }
            base_score = module.VocalCandidateScore(
                profile_name="base",
                score=1.0,
                reasons=[],
                metrics={},
                manifest_path=str(base_record.manifest_path),
                output_dir=str(base_record.output_dir),
                vocals_path=str(base_record.vocals_path),
            )
            donor_score = module.VocalCandidateScore(
                profile_name="donor",
                score=0.9,
                reasons=[],
                metrics={},
                manifest_path=str(donor_record.manifest_path),
                output_dir=str(donor_record.output_dir),
                vocals_path=str(donor_record.vocals_path),
            )
            strict_score = module.VocalCandidateScore(
                profile_name="ft_tail_safe_anchor_strict",
                score=0.7,
                reasons=[],
                metrics={},
                manifest_path=str(strict_record.manifest_path),
                output_dir=str(strict_record.output_dir),
                vocals_path=str(strict_record.vocals_path),
            )
            anchor_score = module.VocalCandidateScore(
                profile_name="bs_artifact_clean",
                score=0.7,
                reasons=[],
                metrics={},
                manifest_path=str(anchor_record.manifest_path),
                output_dir=str(anchor_record.output_dir),
                vocals_path=str(anchor_record.vocals_path),
            )
            base_report = {
                "summary": {"max_issue_score": 1.2, "mean_top3_issue_score": 0.9},
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 1.0,
                        "relative_start_sec": 0.0,
                        "relative_end_sec": 1.0,
                        "severity_score": 1.20,
                        "labels": ["vocal_dropout", "high_noise_or_robotic_artifact"],
                        "severity_parts": {"vocal_dropout": 0.80, "high_noise_or_robotic_artifact": 0.40},
                    },
                    {
                        "start_sec": 1.0,
                        "end_sec": 2.0,
                        "relative_start_sec": 1.0,
                        "relative_end_sec": 2.0,
                        "severity_score": 0.10,
                        "labels": ["acceptable_proxy_window"],
                        "severity_parts": {},
                    },
                ],
            }
            donor_report = {
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 1.0,
                        "relative_start_sec": 0.0,
                        "relative_end_sec": 1.0,
                        "severity_score": 0.40,
                        "labels": ["high_noise_or_robotic_artifact"],
                        "severity_parts": {"high_noise_or_robotic_artifact": 0.40},
                    },
                    base_report["segments"][1],
                ],
            }
            output_dir = temp_path / "segment_issue_repair"
            progress_events: list[dict[str, object]] = []

            manifest = module.build_segment_issue_repair_candidate(
                profile=profile,
                source_audio_path=source_path,
                start_sec=0.0,
                end_sec=2.0,
                base_score=base_score,
                candidate_scores=[base_score, donor_score, strict_score, anchor_score],
                candidate_records_by_name=records,
                candidate_issue_reports={
                    "base": base_report,
                    "donor": donor_report,
                    "ft_tail_safe_anchor_strict": donor_report,
                    "bs_artifact_clean": donor_report,
                },
                dependency_fingerprints={name: {"score": "0.000000"} for name in records},
                output_dir=output_dir,
                progress=lambda **metadata: progress_events.append(metadata),
            )
            repaired, repaired_sr = sf.read(output_dir / "vocals.flac", always_2d=True)

        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.assertEqual(repaired_sr, sample_rate)
        self.assertEqual(manifest["stems"]["vocals"]["repaired_window_count"], 1)
        self.assertEqual(manifest["stems"]["vocals"]["window_decisions"][0]["repair_profile"], "donor")
        self.assertGreater(
            _band_rms(repaired[:sample_rate], sample_rate, 220.0, 310.0),
            1.4 * _band_rms(base_audio[:sample_rate], sample_rate, 220.0, 310.0),
        )
        self.assertLess(
            _band_rms(repaired[:sample_rate], sample_rate, 6200.0, 6800.0),
            0.45 * _band_rms(donor_audio[:sample_rate], sample_rate, 6200.0, 6800.0),
        )
        np.testing.assert_allclose(repaired[sample_rate:], base_audio[sample_rate:], atol=8e-4)
        self.assertTrue(
            any(
                event.get("status") == "window_start"
                and event.get("segment_index") == 0
                and event.get("window_index") == 1
                for event in progress_events
            )
        )
        self.assertTrue(
            any(
                event.get("status") == "window_finish"
                and event.get("segment_index") == 0
                and event.get("repair_profile") == "donor"
                for event in progress_events
            )
        )

    def test_segment_repair_support_selection_ignores_invalid_issue_summaries(self):
        module = _load_adaptive_module()

        def record(profile_name: str) -> module.VocalCandidateRecord:
            return module.VocalCandidateRecord(
                profile=module.VocalCandidateProfile(profile_name, "external_vocals", 0.0, profile_name, {}),
                manifest_path=Path(f"/tmp/{profile_name}/stem_manifest.json"),
                output_dir=Path(f"/tmp/{profile_name}"),
                vocals_path=Path(f"/tmp/{profile_name}/vocals.flac"),
                metrics={},
                manifest={},
            )

        records = {
            "base": record("base"),
            "invalid_support": record("invalid_support"),
            "valid_support": record("valid_support"),
        }

        selected, source = module._select_segment_repair_support_record(
            candidate_records_by_name=records,
            candidate_issue_reports={
                "invalid_support": {
                    "summary": {
                        "max_issue_score": "not-a-number",
                        "mean_top3_issue_score": "not-a-number",
                        "issue_window_ratio": "not-a-number",
                    }
                },
                "valid_support": {
                    "summary": {
                        "max_issue_score": 0.4,
                        "mean_top3_issue_score": 0.2,
                        "issue_window_ratio": 0.1,
                    }
                },
            },
            base_profile_name="base",
            preferred_names=("missing_strict",),
        )

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.profile.name, "valid_support")
        self.assertEqual(source, "adaptive_low_issue_support")

    def test_segment_repair_support_selection_reports_missing_without_scored_candidates(self):
        module = _load_adaptive_module()
        base_record = module.VocalCandidateRecord(
            profile=module.VocalCandidateProfile("base", "base_mode", 0.0, "base", {}),
            manifest_path=Path("/tmp/base/stem_manifest.json"),
            output_dir=Path("/tmp/base"),
            vocals_path=Path("/tmp/base/vocals.flac"),
            metrics={},
            manifest={},
        )

        selected, source = module._select_segment_repair_support_record(
            candidate_records_by_name={"base": base_record},
            candidate_issue_reports={},
            base_profile_name="base",
            preferred_names=("missing_strict",),
        )

        self.assertIsNone(selected)
        self.assertEqual(source, "missing_support")

    def test_defect_aware_candidate_selection_prefers_lower_measured_issue_over_proxy_score(self):
        module = _load_adaptive_module()

        def score(profile_name: str, proxy_score: float) -> module.VocalCandidateScore:
            return module.VocalCandidateScore(
                profile_name=profile_name,
                score=proxy_score,
                reasons=[],
                metrics={},
                manifest_path=f"/tmp/{profile_name}/stem_manifest.json",
                output_dir=f"/tmp/{profile_name}",
                vocals_path=f"/tmp/{profile_name}/vocals.flac",
            )

        high_proxy = score("high_proxy_unrepaired", 5.0)
        repaired = score("lower_proxy_repaired", 4.1)
        selected = module.select_defect_aware_candidate_score(
            [high_proxy, repaired],
            {
                "high_proxy_unrepaired": {
                    "summary": {
                        "max_issue_score": 1.13,
                        "mean_top3_issue_score": 0.88,
                        "issue_window_ratio": 0.975,
                    },
                },
                "lower_proxy_repaired": {
                    "summary": {
                        "max_issue_score": 1.09,
                        "mean_top3_issue_score": 0.87,
                        "issue_window_ratio": 0.975,
                    },
                },
            },
        )

        self.assertEqual(selected.profile_name, "lower_proxy_repaired")

    def test_defect_aware_candidate_selection_falls_back_to_proxy_without_issue_reports(self):
        module = _load_adaptive_module()
        first = module.VocalCandidateScore(
            profile_name="first",
            score=1.0,
            reasons=[],
            metrics={},
            manifest_path="/tmp/first/stem_manifest.json",
            output_dir="/tmp/first",
            vocals_path="/tmp/first/vocals.flac",
        )
        second = module.VocalCandidateScore(
            profile_name="second",
            score=2.0,
            reasons=[],
            metrics={},
            manifest_path="/tmp/second/stem_manifest.json",
            output_dir="/tmp/second",
            vocals_path="/tmp/second/vocals.flac",
        )

        selected = module.select_defect_aware_candidate_score([first, second], {})

        self.assertEqual(selected.profile_name, "second")

    def test_segment_issue_repair_candidate_uses_adaptive_external_support_when_named_support_missing(self):
        module = _load_adaptive_module()
        sample_rate = 16000
        t = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
        voice = (0.14 * np.sin(2.0 * np.pi * 260.0 * t)).astype(np.float32)
        high_noise = (0.05 * np.sin(2.0 * np.pi * 6500.0 * t)).astype(np.float32)
        base_mono = voice.copy()
        base_mono[:sample_rate] *= 0.25
        donor_mono = voice.copy()
        donor_mono[:sample_rate] += high_noise[:sample_rate]
        support_mono = voice.copy()
        source_mono = voice + 0.03 * np.sin(2.0 * np.pi * 120.0 * t).astype(np.float32)
        base_audio = np.stack([base_mono, base_mono], axis=1).astype(np.float32)
        donor_audio = np.stack([donor_mono, donor_mono], axis=1).astype(np.float32)
        support_audio = np.stack([support_mono, support_mono], axis=1).astype(np.float32)
        source_audio = np.stack([source_mono, source_mono], axis=1).astype(np.float32)
        profile_template = module.default_segment_issue_repair_profile()
        profile = module.VocalCandidateProfile(
            name=profile_template.name,
            candidate_mode=profile_template.candidate_mode,
            prior_score=profile_template.prior_score,
            intent=profile_template.intent,
            tuning={
                **profile_template.tuning,
                "crossfade_sec": 0.0,
                "segment_consistency_sec": 1.0,
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.flac"
            sf.write(source_path, source_audio, sample_rate, subtype="PCM_24")

            def write_candidate(
                profile_name: str,
                candidate_mode: str,
                audio: np.ndarray,
            ) -> module.VocalCandidateRecord:
                candidate_dir = temp_path / profile_name
                candidate_dir.mkdir()
                vocals_path = candidate_dir / "vocals.flac"
                sf.write(vocals_path, audio, sample_rate, subtype="PCM_24")
                manifest = {
                    "schema_version": 1,
                    "audio_path": str(source_path),
                    "source_audio_sha256": module._sha256_file(source_path),
                    "sample_rate": sample_rate,
                    "preview_window_sec": {"start": 0.0, "end": 2.0},
                    "pipeline": {"candidate_mode": candidate_mode},
                    "tuning": {},
                    "stems": {"vocals": {"path": str(vocals_path)}},
                }
                manifest_path = candidate_dir / "stem_manifest.json"
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return module.VocalCandidateRecord(
                    profile=module.VocalCandidateProfile(profile_name, candidate_mode, 0.0, profile_name, {}),
                    manifest_path=manifest_path,
                    output_dir=candidate_dir,
                    vocals_path=vocals_path,
                    metrics={},
                    manifest=manifest,
                )

            base_record = write_candidate("base", "base_mode", base_audio)
            donor_record = write_candidate("donor", "donor_mode", donor_audio)
            support_record = write_candidate("external_clean_support", "external_vocals", support_audio)
            records = {
                record.profile.name: record
                for record in (base_record, donor_record, support_record)
            }

            def score_for(record: module.VocalCandidateRecord, score: float) -> module.VocalCandidateScore:
                return module.VocalCandidateScore(
                    profile_name=record.profile.name,
                    score=score,
                    reasons=[],
                    metrics={},
                    manifest_path=str(record.manifest_path),
                    output_dir=str(record.output_dir),
                    vocals_path=str(record.vocals_path),
                )

            base_score = score_for(base_record, 1.0)
            donor_score = score_for(donor_record, 0.9)
            support_score = score_for(support_record, 0.8)
            base_report = {
                "summary": {"max_issue_score": 1.2, "mean_top3_issue_score": 0.9},
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 1.0,
                        "relative_start_sec": 0.0,
                        "relative_end_sec": 1.0,
                        "severity_score": 1.20,
                        "labels": ["vocal_dropout", "high_noise_or_robotic_artifact"],
                        "severity_parts": {"vocal_dropout": 0.80, "high_noise_or_robotic_artifact": 0.40},
                    },
                    {
                        "start_sec": 1.0,
                        "end_sec": 2.0,
                        "relative_start_sec": 1.0,
                        "relative_end_sec": 2.0,
                        "severity_score": 0.10,
                        "labels": ["acceptable_proxy_window"],
                        "severity_parts": {},
                    },
                ],
            }
            donor_report = {
                "summary": {"max_issue_score": 0.4, "mean_top3_issue_score": 0.4},
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 1.0,
                        "relative_start_sec": 0.0,
                        "relative_end_sec": 1.0,
                        "severity_score": 0.40,
                        "labels": ["high_noise_or_robotic_artifact"],
                        "severity_parts": {"high_noise_or_robotic_artifact": 0.40},
                    },
                    base_report["segments"][1],
                ],
            }
            support_report = {
                "summary": {"max_issue_score": 0.1, "mean_top3_issue_score": 0.1},
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 1.0,
                        "relative_start_sec": 0.0,
                        "relative_end_sec": 1.0,
                        "severity_score": 0.10,
                        "labels": ["acceptable_proxy_window"],
                        "severity_parts": {},
                    },
                    base_report["segments"][1],
                ],
            }
            output_dir = temp_path / "segment_issue_repair"

            manifest = module.build_segment_issue_repair_candidate(
                profile=profile,
                source_audio_path=source_path,
                start_sec=0.0,
                end_sec=2.0,
                base_score=base_score,
                candidate_scores=[base_score, donor_score, support_score],
                candidate_records_by_name=records,
                candidate_issue_reports={
                    "base": base_report,
                    "donor": donor_report,
                    "external_clean_support": support_report,
                },
                dependency_fingerprints={name: {"score": "0.000000"} for name in records},
                output_dir=output_dir,
            )

        self.assertIsNotNone(manifest)
        assert manifest is not None
        vocals = manifest["stems"]["vocals"]
        self.assertEqual(vocals["strict_profile"], "external_clean_support")
        self.assertEqual(vocals["anchor_profile"], "external_clean_support")
        self.assertEqual(vocals["support_selection"]["strict_profile_source"], "adaptive_low_issue_support")
        self.assertEqual(vocals["support_selection"]["anchor_profile_source"], "adaptive_low_issue_support")

    def test_gate_blocker_best_segment_splice_candidate_replaces_only_measured_blocker_window(self):
        module = _load_adaptive_module()
        sample_rate = 16000
        t = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
        voice = (0.14 * np.sin(2.0 * np.pi * 260.0 * t)).astype(np.float32)
        base_mono = voice.copy()
        base_mono[:sample_rate] *= 0.20
        donor_mono = voice.copy()
        base_audio = np.stack([base_mono, base_mono], axis=1).astype(np.float32)
        donor_audio = np.stack([donor_mono, donor_mono], axis=1).astype(np.float32)
        source_audio = np.stack([voice, voice], axis=1).astype(np.float32)

        profile_template = module.default_gate_blocker_best_segment_splice_profile()
        profile = module.VocalCandidateProfile(
            name=profile_template.name,
            candidate_mode=profile_template.candidate_mode,
            prior_score=profile_template.prior_score,
            intent=profile_template.intent,
            tuning={
                **profile_template.tuning,
                "crossfade_sec": 0.0,
                "min_donor_severity_improvement": 0.08,
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.flac"
            sf.write(source_path, source_audio, sample_rate, subtype="PCM_24")

            def write_candidate(profile_name: str, audio: np.ndarray) -> module.VocalCandidateRecord:
                candidate_dir = temp_path / profile_name
                candidate_dir.mkdir()
                vocals_path = candidate_dir / "vocals.flac"
                sf.write(vocals_path, audio, sample_rate, subtype="PCM_24")
                manifest = {
                    "schema_version": 1,
                    "audio_path": str(source_path),
                    "source_audio_sha256": module._sha256_file(source_path),
                    "sample_rate": sample_rate,
                    "preview_window_sec": {"start": 0.0, "end": 2.0},
                    "pipeline": {"candidate_mode": f"{profile_name}_mode"},
                    "tuning": {},
                    "stems": {"vocals": {"path": str(vocals_path)}},
                }
                manifest_path = candidate_dir / "stem_manifest.json"
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return module.VocalCandidateRecord(
                    profile=module.VocalCandidateProfile(profile_name, f"{profile_name}_mode", 0.0, profile_name, {}),
                    manifest_path=manifest_path,
                    output_dir=candidate_dir,
                    vocals_path=vocals_path,
                    metrics={},
                    manifest=manifest,
                )

            base_record = write_candidate("base", base_audio)
            donor_record = write_candidate("donor", donor_audio)
            records = {"base": base_record, "donor": donor_record}
            base_score = module.VocalCandidateScore(
                profile_name="base",
                score=1.0,
                reasons=[],
                metrics={},
                manifest_path=str(base_record.manifest_path),
                output_dir=str(base_record.output_dir),
                vocals_path=str(base_record.vocals_path),
            )
            donor_score = module.VocalCandidateScore(
                profile_name="donor",
                score=0.3,
                reasons=[],
                metrics={},
                manifest_path=str(donor_record.manifest_path),
                output_dir=str(donor_record.output_dir),
                vocals_path=str(donor_record.vocals_path),
            )
            base_report = {
                "summary": {"max_issue_score": 1.2, "mean_top3_issue_score": 0.8, "issue_window_ratio": 0.5},
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 1.0,
                        "relative_start_sec": 0.0,
                        "relative_end_sec": 1.0,
                        "severity_score": 1.20,
                        "labels": ["vocal_dropout", "vocal_tail_loss"],
                        "severity_parts": {"vocal_dropout": 0.80, "vocal_tail_loss": 0.40},
                    },
                    {
                        "start_sec": 1.0,
                        "end_sec": 2.0,
                        "relative_start_sec": 1.0,
                        "relative_end_sec": 2.0,
                        "severity_score": 0.05,
                        "labels": ["acceptable_proxy_window"],
                        "severity_parts": {},
                    },
                ],
            }
            donor_report = {
                "summary": {"max_issue_score": 0.45, "mean_top3_issue_score": 0.25, "issue_window_ratio": 0.5},
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 1.0,
                        "relative_start_sec": 0.0,
                        "relative_end_sec": 1.0,
                        "severity_score": 0.45,
                        "labels": ["vocal_tail_loss"],
                        "severity_parts": {"vocal_tail_loss": 0.45},
                    },
                    base_report["segments"][1],
                ],
            }
            gate = {
                "status": "blocked_for_listening",
                "release_status": "diagnostic_only",
                "blockers": [
                    {
                        "reason": "selected_candidate_still_has_severe_issue",
                        "start_sec": 0.0,
                        "end_sec": 1.0,
                        "severity_score": 1.2,
                        "labels": ["vocal_dropout", "vocal_tail_loss"],
                    }
                ],
            }
            output_dir = temp_path / "gate_blocker_best_segment_splice"

            manifest = module.build_gate_blocker_best_segment_splice_candidate(
                profile=profile,
                base_score=base_score,
                candidate_scores=[base_score, donor_score],
                candidate_records_by_name=records,
                candidate_issue_reports={"base": base_report, "donor": donor_report},
                defect_effectiveness_gate=gate,
                dependency_fingerprints={name: {"score": "0.000000"} for name in records},
                output_dir=output_dir,
            )
            spliced, spliced_sr = sf.read(output_dir / "vocals.flac", always_2d=True)

        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.assertEqual(spliced_sr, sample_rate)
        self.assertEqual(manifest["stems"]["vocals"]["spliced_window_count"], 1)
        self.assertEqual(manifest["stems"]["vocals"]["window_decisions"][0]["donor_profile"], "donor")
        np.testing.assert_allclose(spliced[:sample_rate], donor_audio[:sample_rate], atol=8e-4)
        np.testing.assert_allclose(spliced[sample_rate:], base_audio[sample_rate:], atol=8e-4)

    def test_gate_blocker_best_segment_splice_candidate_skips_without_measured_donor_improvement(self):
        module = _load_adaptive_module()
        profile = module.default_gate_blocker_best_segment_splice_profile()
        base_score = module.VocalCandidateScore(
            profile_name="base",
            score=1.0,
            reasons=[],
            metrics={},
            manifest_path="base/stem_manifest.json",
            output_dir="base",
            vocals_path="base/vocals.flac",
        )
        donor_score = module.VocalCandidateScore(
            profile_name="donor",
            score=0.9,
            reasons=[],
            metrics={},
            manifest_path="donor/stem_manifest.json",
            output_dir="donor",
            vocals_path="donor/vocals.flac",
        )
        base_segment = {
            "start_sec": 0.0,
            "end_sec": 1.0,
            "relative_start_sec": 0.0,
            "relative_end_sec": 1.0,
            "severity_score": 1.20,
            "labels": ["vocal_dropout"],
            "severity_parts": {"vocal_dropout": 1.20},
        }
        donor_segment = {**base_segment, "severity_score": 1.16}

        decisions = module._build_gate_blocker_best_segment_splice_window_decisions(
            base_score=base_score,
            candidate_scores=[base_score, donor_score],
            candidate_issue_reports={
                "base": {"segments": [base_segment]},
                "donor": {"segments": [donor_segment]},
            },
            defect_effectiveness_gate={
                "blockers": [
                    {
                        "reason": "selected_candidate_still_has_severe_issue",
                        "start_sec": 0.0,
                        "end_sec": 1.0,
                        "severity_score": 1.2,
                    }
                ]
            },
            tuning=profile.tuning,
        )

        self.assertEqual(decisions, [])

    def test_gate_blocker_best_segment_splice_reclassifies_donor_against_base_track_metrics(self):
        module = _load_adaptive_module()
        profile = module.default_gate_blocker_best_segment_splice_profile()
        base_score = module.VocalCandidateScore(
            profile_name="base",
            score=1.0,
            reasons=[],
            metrics={"dropout_frame_ratio": 0.10, "tail_dropout_ratio": 0.10},
            manifest_path="base/stem_manifest.json",
            output_dir="base",
            vocals_path="base/vocals.flac",
        )
        donor_score = module.VocalCandidateScore(
            profile_name="donor",
            score=0.9,
            reasons=[],
            metrics={"dropout_frame_ratio": 0.62, "tail_dropout_ratio": 0.70},
            manifest_path="donor/stem_manifest.json",
            output_dir="donor",
            vocals_path="donor/vocals.flac",
        )
        base_segment = {
            "start_sec": 0.0,
            "end_sec": 1.0,
            "relative_start_sec": 0.0,
            "relative_end_sec": 1.0,
            "severity_score": 1.10,
            "labels": ["vocal_dropout", "vocal_tail_loss"],
            "severity_parts": {"vocal_dropout": 0.60, "vocal_tail_loss": 0.50},
            "metrics": {
                "dropout_frame_ratio": 0.55,
                "tail_dropout_ratio": 0.58,
                "candidate_to_source_rms_db": -12.0,
                "low_band_ratio_db_35_180": -60.0,
                "kick_punch_ratio_db_250_900": -5.0,
                "presence_ratio_db_1800_4500": -12.0,
                "air_ratio_db_4500_9000": -18.0,
                "source_rms_dbfs": -19.0,
                "candidate_rms_dbfs": -31.0,
            },
        }
        donor_segment = {
            **base_segment,
            "severity_score": 0.42,
            "labels": ["vocal_tail_loss"],
            "severity_parts": {"vocal_tail_loss": 0.42},
            "metrics": {
                **base_segment["metrics"],
                "dropout_frame_ratio": 0.70,
                "tail_dropout_ratio": 0.76,
                "candidate_to_source_rms_db": -32.0,
            },
        }

        decisions = module._build_gate_blocker_best_segment_splice_window_decisions(
            base_score=base_score,
            candidate_scores=[base_score, donor_score],
            candidate_issue_reports={
                "base": {"segments": [base_segment]},
                "donor": {"segments": [donor_segment]},
            },
            defect_effectiveness_gate={
                "blockers": [
                    {
                        "reason": "selected_candidate_still_has_severe_issue",
                        "start_sec": 0.0,
                        "end_sec": 1.0,
                        "severity_score": 1.1,
                    }
                ]
            },
            tuning=profile.tuning,
        )

        self.assertEqual(decisions, [])

    def test_gate_blocker_presence_band_recovery_candidate_repairs_only_predicted_improving_window(self):
        module = _load_adaptive_module()
        sample_rate = 16000
        t = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
        source_voice = (0.18 * np.sin(2.0 * np.pi * 2400.0 * t)).astype(np.float32)
        kick_bleed = (0.12 * np.sin(2.0 * np.pi * 420.0 * t)).astype(np.float32)
        base_mono = source_voice.copy()
        base_mono[:sample_rate] = 0.08 * source_voice[:sample_rate] + kick_bleed[:sample_rate]
        donor_mono = source_voice.copy()
        donor_mono[:sample_rate] = 0.72 * source_voice[:sample_rate]
        base_audio = np.stack([base_mono, base_mono], axis=1).astype(np.float32)
        donor_audio = np.stack([donor_mono, donor_mono], axis=1).astype(np.float32)
        source_audio = np.stack([source_voice, source_voice], axis=1).astype(np.float32)

        profile_template = module.default_gate_blocker_presence_band_recovery_profile()
        profile = module.VocalCandidateProfile(
            name=profile_template.name,
            candidate_mode=profile_template.candidate_mode,
            prior_score=profile_template.prior_score,
            intent=profile_template.intent,
            tuning={
                **profile_template.tuning,
                "crossfade_sec": 0.0,
                "min_predicted_segment_improvement": 0.01,
                "presence_delta_gain": 0.6,
                "bleed_band_gain": 0.85,
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.flac"
            sf.write(source_path, source_audio, sample_rate, subtype="PCM_24")

            def write_candidate(profile_name: str, audio: np.ndarray) -> module.VocalCandidateRecord:
                candidate_dir = temp_path / profile_name
                candidate_dir.mkdir()
                vocals_path = candidate_dir / "vocals.flac"
                sf.write(vocals_path, audio, sample_rate, subtype="PCM_24")
                manifest = {
                    "schema_version": 1,
                    "audio_path": str(source_path),
                    "source_audio_sha256": module._sha256_file(source_path),
                    "sample_rate": sample_rate,
                    "preview_window_sec": {"start": 0.0, "end": 2.0},
                    "pipeline": {"candidate_mode": f"{profile_name}_mode"},
                    "tuning": {},
                    "stems": {"vocals": {"path": str(vocals_path)}},
                }
                manifest_path = candidate_dir / "stem_manifest.json"
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return module.VocalCandidateRecord(
                    profile=module.VocalCandidateProfile(profile_name, f"{profile_name}_mode", 0.0, profile_name, {}),
                    manifest_path=manifest_path,
                    output_dir=candidate_dir,
                    vocals_path=vocals_path,
                    metrics={},
                    manifest=manifest,
                )

            base_record = write_candidate("base", base_audio)
            donor_record = write_candidate("donor", donor_audio)
            records = {"base": base_record, "donor": donor_record}
            base_score = module.VocalCandidateScore(
                profile_name="base",
                score=1.0,
                reasons=[],
                metrics={},
                manifest_path=str(base_record.manifest_path),
                output_dir=str(base_record.output_dir),
                vocals_path=str(base_record.vocals_path),
            )
            donor_score = module.VocalCandidateScore(
                profile_name="donor",
                score=0.4,
                reasons=[],
                metrics={},
                manifest_path=str(donor_record.manifest_path),
                output_dir=str(donor_record.output_dir),
                vocals_path=str(donor_record.vocals_path),
            )
            base_segment_metrics = module._calculate_vocal_proxy_metrics_from_audio(
                source_audio[:sample_rate],
                base_audio[:sample_rate],
                sample_rate,
            )
            donor_segment_metrics = module._calculate_vocal_proxy_metrics_from_audio(
                source_audio[:sample_rate],
                donor_audio[:sample_rate],
                sample_rate,
            )
            base_report = {
                "summary": {"max_issue_score": 1.15, "mean_top3_issue_score": 0.75, "issue_window_ratio": 0.5},
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 1.0,
                        "relative_start_sec": 0.0,
                        "relative_end_sec": 1.0,
                        "severity_score": 1.15,
                        "labels": ["presence_loss", "kick_punch_bleed"],
                        "severity_parts": {"presence_loss": 0.75, "kick_punch_bleed": 0.40},
                        "metrics": base_segment_metrics,
                    },
                    {
                        "start_sec": 1.0,
                        "end_sec": 2.0,
                        "relative_start_sec": 1.0,
                        "relative_end_sec": 2.0,
                        "severity_score": 0.0,
                        "labels": ["acceptable_proxy_window"],
                        "severity_parts": {},
                    },
                ],
            }
            donor_report = {
                "summary": {"max_issue_score": 0.50, "mean_top3_issue_score": 0.25, "issue_window_ratio": 0.5},
                "segments": [
                    {
                        **base_report["segments"][0],
                        "severity_score": 0.50,
                        "labels": ["acceptable_proxy_window"],
                        "severity_parts": {},
                        "metrics": donor_segment_metrics,
                    },
                    base_report["segments"][1],
                ],
            }
            gate = {
                "status": "blocked_for_listening",
                "release_status": "diagnostic_only",
                "blockers": [
                    {
                        "reason": "selected_candidate_still_has_severe_issue",
                        "start_sec": 0.0,
                        "end_sec": 1.0,
                        "severity_score": 1.15,
                        "labels": ["presence_loss", "kick_punch_bleed"],
                    }
                ],
            }
            output_dir = temp_path / "gate_blocker_presence_band_recovery"

            manifest = module.build_gate_blocker_presence_band_recovery_candidate(
                profile=profile,
                base_score=base_score,
                candidate_scores=[base_score, donor_score],
                candidate_records_by_name=records,
                candidate_issue_reports={"base": base_report, "donor": donor_report},
                defect_effectiveness_gate=gate,
                dependency_fingerprints={name: {"score": "0.000000"} for name in records},
                output_dir=output_dir,
            )
            recovered, recovered_sr = sf.read(output_dir / "vocals.flac", always_2d=True)

        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.assertEqual(recovered_sr, sample_rate)
        vocals = manifest["stems"]["vocals"]
        self.assertEqual(vocals["presence_band_recovered_window_count"], 1)
        self.assertEqual(vocals["window_decisions"][0]["donor_profile"], "donor")
        self.assertGreater(
            _band_rms(recovered[:sample_rate], sample_rate, 1800.0, 4500.0),
            1.2 * _band_rms(base_audio[:sample_rate], sample_rate, 1800.0, 4500.0),
        )
        self.assertLess(
            _band_rms(recovered[:sample_rate], sample_rate, 250.0, 900.0),
            0.98 * _band_rms(base_audio[:sample_rate], sample_rate, 250.0, 900.0),
        )
        np.testing.assert_allclose(recovered[sample_rate:], base_audio[sample_rate:], atol=8e-4)

    def test_gate_blocker_presence_band_recovery_candidate_skips_without_predicted_improvement(self):
        module = _load_adaptive_module()
        sample_rate = 16000
        t = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
        voice = (0.10 * np.sin(2.0 * np.pi * 2400.0 * t)).astype(np.float32)
        base_audio = np.stack([voice, voice], axis=1).astype(np.float32)
        donor_audio = base_audio.copy()
        source_audio = np.stack([voice, voice], axis=1).astype(np.float32)

        profile_template = module.default_gate_blocker_presence_band_recovery_profile()
        profile = module.VocalCandidateProfile(
            name=profile_template.name,
            candidate_mode=profile_template.candidate_mode,
            prior_score=profile_template.prior_score,
            intent=profile_template.intent,
            tuning={**profile_template.tuning, "crossfade_sec": 0.0},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.flac"
            sf.write(source_path, source_audio, sample_rate, subtype="PCM_24")

            def write_candidate(profile_name: str, audio: np.ndarray) -> module.VocalCandidateRecord:
                candidate_dir = temp_path / profile_name
                candidate_dir.mkdir()
                vocals_path = candidate_dir / "vocals.flac"
                sf.write(vocals_path, audio, sample_rate, subtype="PCM_24")
                manifest = {
                    "schema_version": 1,
                    "audio_path": str(source_path),
                    "source_audio_sha256": module._sha256_file(source_path),
                    "sample_rate": sample_rate,
                    "preview_window_sec": {"start": 0.0, "end": 1.0},
                    "pipeline": {"candidate_mode": f"{profile_name}_mode"},
                    "tuning": {},
                    "stems": {"vocals": {"path": str(vocals_path)}},
                }
                manifest_path = candidate_dir / "stem_manifest.json"
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return module.VocalCandidateRecord(
                    profile=module.VocalCandidateProfile(profile_name, f"{profile_name}_mode", 0.0, profile_name, {}),
                    manifest_path=manifest_path,
                    output_dir=candidate_dir,
                    vocals_path=vocals_path,
                    metrics={},
                    manifest=manifest,
                )

            base_record = write_candidate("base", base_audio)
            donor_record = write_candidate("donor", donor_audio)
            base_score = module.VocalCandidateScore(
                profile_name="base",
                score=1.0,
                reasons=[],
                metrics={},
                manifest_path=str(base_record.manifest_path),
                output_dir=str(base_record.output_dir),
                vocals_path=str(base_record.vocals_path),
            )
            donor_score = module.VocalCandidateScore(
                profile_name="donor",
                score=0.4,
                reasons=[],
                metrics={},
                manifest_path=str(donor_record.manifest_path),
                output_dir=str(donor_record.output_dir),
                vocals_path=str(donor_record.vocals_path),
            )
            base_segment = {
                "start_sec": 0.0,
                "end_sec": 1.0,
                "relative_start_sec": 0.0,
                "relative_end_sec": 1.0,
                "severity_score": 1.20,
                "labels": ["presence_loss", "kick_punch_bleed"],
                "severity_parts": {"presence_loss": 0.80, "kick_punch_bleed": 0.40},
            }
            manifest = module.build_gate_blocker_presence_band_recovery_candidate(
                profile=profile,
                base_score=base_score,
                candidate_scores=[base_score, donor_score],
                candidate_records_by_name={"base": base_record, "donor": donor_record},
                candidate_issue_reports={
                    "base": {"segments": [base_segment]},
                    "donor": {"segments": [{**base_segment, "severity_score": 0.40, "severity_parts": {}}]},
                },
                defect_effectiveness_gate={
                    "blockers": [
                        {
                            "reason": "selected_candidate_still_has_severe_issue",
                            "start_sec": 0.0,
                            "end_sec": 1.0,
                            "severity_score": 1.20,
                        }
                    ]
                },
                dependency_fingerprints={"base": {"score": "0.000000"}, "donor": {"score": "0.000000"}},
                output_dir=temp_path / "gate_blocker_presence_band_recovery",
            )

        self.assertIsNone(manifest)

    def test_clustered_issue_window_ratio_repair_candidate_repairs_only_predicted_improving_windows(self):
        module = _load_adaptive_module()
        sample_rate = 16000
        t = np.arange(sample_rate * 3, dtype=np.float32) / float(sample_rate)
        source_voice = (0.16 * np.sin(2.0 * np.pi * 2400.0 * t)).astype(np.float32)
        kick_bleed = (0.10 * np.sin(2.0 * np.pi * 420.0 * t)).astype(np.float32)
        base_mono = source_voice.copy()
        base_mono[: sample_rate * 2] += kick_bleed[: sample_rate * 2]
        base_audio = np.stack([base_mono, base_mono], axis=1).astype(np.float32)
        source_audio = np.stack([source_voice, source_voice], axis=1).astype(np.float32)

        profile_template = module.default_clustered_issue_window_ratio_repair_profile()
        profile = module.VocalCandidateProfile(
            name=profile_template.name,
            candidate_mode=profile_template.candidate_mode,
            prior_score=profile_template.prior_score,
            intent=profile_template.intent,
            tuning={
                **profile_template.tuning,
                "crossfade_sec": 0.0,
                "min_predicted_segment_improvement": 0.01,
                "kick_band_gain": 0.20,
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.flac"
            sf.write(source_path, source_audio, sample_rate, subtype="PCM_24")
            candidate_dir = temp_path / "base"
            candidate_dir.mkdir()
            vocals_path = candidate_dir / "vocals.flac"
            sf.write(vocals_path, base_audio, sample_rate, subtype="PCM_24")
            manifest = {
                "schema_version": 1,
                "audio_path": str(source_path),
                "source_audio_sha256": module._sha256_file(source_path),
                "sample_rate": sample_rate,
                "preview_window_sec": {"start": 0.0, "end": 3.0},
                "pipeline": {"candidate_mode": "base_mode"},
                "tuning": {},
                "stems": {"vocals": {"path": str(vocals_path)}},
            }
            manifest_path = candidate_dir / "stem_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            base_record = module.VocalCandidateRecord(
                profile=module.VocalCandidateProfile("base", "base_mode", 0.0, "base", {}),
                manifest_path=manifest_path,
                output_dir=candidate_dir,
                vocals_path=vocals_path,
                metrics={},
                manifest=manifest,
            )
            base_score = module.VocalCandidateScore(
                profile_name="base",
                score=1.0,
                reasons=[],
                metrics={},
                manifest_path=str(manifest_path),
                output_dir=str(candidate_dir),
                vocals_path=str(vocals_path),
            )
            base_report = {
                "summary": {"max_issue_score": 0.70, "mean_top3_issue_score": 0.62, "issue_window_ratio": 0.67},
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 1.0,
                        "relative_start_sec": 0.0,
                        "relative_end_sec": 1.0,
                        "severity_score": 0.65,
                        "labels": ["kick_punch_bleed"],
                        "severity_parts": {"kick_punch_bleed": 0.65},
                    },
                    {
                        "start_sec": 1.0,
                        "end_sec": 2.0,
                        "relative_start_sec": 1.0,
                        "relative_end_sec": 2.0,
                        "severity_score": 0.70,
                        "labels": ["kick_punch_bleed"],
                        "severity_parts": {"kick_punch_bleed": 0.70},
                    },
                    {
                        "start_sec": 2.0,
                        "end_sec": 3.0,
                        "relative_start_sec": 2.0,
                        "relative_end_sec": 3.0,
                        "severity_score": 0.0,
                        "labels": ["acceptable_proxy_window"],
                        "severity_parts": {},
                    },
                ],
            }
            output_dir = temp_path / "clustered_issue_window_ratio_repair"

            repaired_manifest = module.build_clustered_issue_window_ratio_repair_candidate(
                profile=profile,
                base_score=base_score,
                candidate_records_by_name={"base": base_record},
                candidate_issue_reports={"base": base_report},
                defect_effectiveness_gate={
                    "blockers": [
                        {"reason": "selected_candidate_has_too_many_issue_windows", "issue_window_ratio": 0.67}
                    ]
                },
                dependency_fingerprints={"base": {"score": "0.000000"}},
                output_dir=output_dir,
            )
            repaired, repaired_sr = sf.read(output_dir / "vocals.flac", always_2d=True)

        self.assertIsNotNone(repaired_manifest)
        assert repaired_manifest is not None
        self.assertEqual(repaired_sr, sample_rate)
        vocals = repaired_manifest["stems"]["vocals"]
        self.assertEqual(vocals["clustered_repaired_window_count"], 2)
        self.assertLess(
            _band_rms(repaired[:sample_rate], sample_rate, 250.0, 900.0),
            0.55 * _band_rms(base_audio[:sample_rate], sample_rate, 250.0, 900.0),
        )
        self.assertLess(
            _band_rms(repaired[sample_rate : sample_rate * 2], sample_rate, 250.0, 900.0),
            0.55 * _band_rms(base_audio[sample_rate : sample_rate * 2], sample_rate, 250.0, 900.0),
        )
        np.testing.assert_allclose(repaired[sample_rate * 2 :], base_audio[sample_rate * 2 :], atol=8e-4)

    def test_clustered_issue_repair_variant_specs_include_extreme_drop_melody_reduction(self):
        module = _load_adaptive_module()
        profile = module.default_clustered_issue_window_ratio_repair_profile()

        variants = module._clustered_issue_repair_variant_specs(
            {"drop_melody_leakage"},
            profile.tuning,
        )

        extreme_variant = next(
            variant
            for variant in variants
            if variant["name"] == "drop_melody_mid_tonal_reduction_extreme"
        )
        self.assertEqual(extreme_variant["low_hz"], 700.0)
        self.assertEqual(extreme_variant["high_hz"], 1800.0)
        self.assertEqual(extreme_variant["gain"], profile.tuning["kick_band_gain_extreme"])

    def test_clustered_issue_repair_variant_specs_include_measured_kick_drop_combo_only_for_joint_labels(self):
        module = _load_adaptive_module()
        profile = module.default_clustered_issue_window_ratio_repair_profile()

        joint_variants = module._clustered_issue_repair_variant_specs(
            {"kick_punch_bleed", "drop_melody_leakage"},
            profile.tuning,
        )
        drop_only_variants = module._clustered_issue_repair_variant_specs(
            {"drop_melody_leakage"},
            profile.tuning,
        )

        combo_variant = next(
            variant
            for variant in joint_variants
            if variant["name"] == "kick_drop_melody_combo_reduction_extreme"
        )
        self.assertEqual(combo_variant["low_hz"], profile.tuning["kick_low_hz"])
        self.assertEqual(combo_variant["high_hz"], 1800.0)
        self.assertEqual(combo_variant["gain"], profile.tuning["kick_band_gain_extreme"])
        self.assertNotIn(
            "kick_drop_melody_combo_reduction_extreme",
            {variant["name"] for variant in drop_only_variants},
        )

    def test_clustered_issue_window_ratio_repair_candidate_materializes_joint_label_combo_variant(self):
        module = _load_adaptive_module()
        sample_rate = 16000
        base_audio = np.zeros((sample_rate, 2), dtype=np.float32)

        profile_template = module.default_clustered_issue_window_ratio_repair_profile()
        profile = module.VocalCandidateProfile(
            name=profile_template.name,
            candidate_mode=profile_template.candidate_mode,
            prior_score=profile_template.prior_score,
            intent=profile_template.intent,
            tuning={**profile_template.tuning, "crossfade_sec": 0.0},
        )

        def fake_decisions(**kwargs):
            self.assertIn(
                "kick_drop_melody_combo_reduction_extreme",
                kwargs["variant_audio_by_name"],
            )
            return [
                {
                    "segment_index": 0,
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                    "relative_start_sec": 0.0,
                    "relative_end_sec": 1.0,
                    "base_profile": "base",
                    "base_severity_score": 0.8,
                    "predicted_severity_score": 0.6,
                    "predicted_severity_improvement": 0.2,
                    "base_selection_basis": "reported_score",
                    "base_labels": ["drop_melody_leakage", "kick_punch_bleed"],
                    "predicted_labels": ["kick_punch_bleed"],
                    "predicted_severity_parts": {"kick_punch_bleed": 0.6},
                    "variant_name": "kick_drop_melody_combo_reduction_extreme",
                    "variant_low_hz": float(profile.tuning["kick_low_hz"]),
                    "variant_high_hz": 1800.0,
                    "variant_gain": float(profile.tuning["kick_band_gain_extreme"]),
                    "repair_policy": "measured_clustered_issue_band_reduction",
                }
            ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.flac"
            sf.write(source_path, base_audio, sample_rate, subtype="PCM_24")
            candidate_dir = temp_path / "base"
            candidate_dir.mkdir()
            vocals_path = candidate_dir / "vocals.flac"
            sf.write(vocals_path, base_audio, sample_rate, subtype="PCM_24")
            manifest = {
                "schema_version": 1,
                "audio_path": str(source_path),
                "source_audio_sha256": module._sha256_file(source_path),
                "sample_rate": sample_rate,
                "preview_window_sec": {"start": 0.0, "end": 1.0},
                "pipeline": {"candidate_mode": "base_mode"},
                "tuning": {},
                "stems": {"vocals": {"path": str(vocals_path)}},
            }
            manifest_path = candidate_dir / "stem_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            base_record = module.VocalCandidateRecord(
                profile=module.VocalCandidateProfile("base", "base_mode", 0.0, "base", {}),
                manifest_path=manifest_path,
                output_dir=candidate_dir,
                vocals_path=vocals_path,
                metrics={},
                manifest=manifest,
            )
            base_score = module.VocalCandidateScore(
                profile_name="base",
                score=1.0,
                reasons=[],
                metrics={},
                manifest_path=str(manifest_path),
                output_dir=str(candidate_dir),
                vocals_path=str(vocals_path),
            )
            base_report = {
                "summary": {"max_issue_score": 0.8, "mean_top3_issue_score": 0.8, "issue_window_ratio": 1.0},
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 1.0,
                        "relative_start_sec": 0.0,
                        "relative_end_sec": 1.0,
                        "severity_score": 0.8,
                        "labels": ["kick_punch_bleed", "drop_melody_leakage"],
                        "severity_parts": {"kick_punch_bleed": 0.4, "drop_melody_leakage": 0.4},
                    }
                ],
            }

            with patch.object(module, "_build_clustered_issue_window_ratio_repair_decisions", side_effect=fake_decisions):
                repaired_manifest = module.build_clustered_issue_window_ratio_repair_candidate(
                    profile=profile,
                    base_score=base_score,
                    candidate_records_by_name={"base": base_record},
                    candidate_issue_reports={"base": base_report},
                    defect_effectiveness_gate={
                        "blockers": [
                            {"reason": "selected_candidate_has_too_many_issue_windows", "issue_window_ratio": 1.0}
                        ]
                    },
                    dependency_fingerprints={"base": {"score": "0.000000"}},
                    output_dir=temp_path / "clustered_issue_window_ratio_repair",
                )

        self.assertIsNotNone(repaired_manifest)
        assert repaired_manifest is not None
        self.assertIn(
            "kick_drop_melody_combo_reduction_extreme",
            repaired_manifest["stems"]["vocals"]["variant_names"],
        )

    def test_clustered_issue_window_ratio_repair_candidate_skips_without_predicted_improvement(self):
        module = _load_adaptive_module()
        sample_rate = 16000
        t = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
        source_voice = (0.16 * np.sin(2.0 * np.pi * 2400.0 * t)).astype(np.float32)
        base_audio = np.stack([source_voice, source_voice], axis=1).astype(np.float32)

        profile = module.default_clustered_issue_window_ratio_repair_profile()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.flac"
            sf.write(source_path, base_audio, sample_rate, subtype="PCM_24")
            candidate_dir = temp_path / "base"
            candidate_dir.mkdir()
            vocals_path = candidate_dir / "vocals.flac"
            sf.write(vocals_path, base_audio, sample_rate, subtype="PCM_24")
            manifest = {
                "schema_version": 1,
                "audio_path": str(source_path),
                "source_audio_sha256": module._sha256_file(source_path),
                "sample_rate": sample_rate,
                "preview_window_sec": {"start": 0.0, "end": 1.0},
                "pipeline": {"candidate_mode": "base_mode"},
                "tuning": {},
                "stems": {"vocals": {"path": str(vocals_path)}},
            }
            manifest_path = candidate_dir / "stem_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            base_record = module.VocalCandidateRecord(
                profile=module.VocalCandidateProfile("base", "base_mode", 0.0, "base", {}),
                manifest_path=manifest_path,
                output_dir=candidate_dir,
                vocals_path=vocals_path,
                metrics={},
                manifest=manifest,
            )
            base_score = module.VocalCandidateScore(
                profile_name="base",
                score=1.0,
                reasons=[],
                metrics={},
                manifest_path=str(manifest_path),
                output_dir=str(candidate_dir),
                vocals_path=str(vocals_path),
            )
            base_segment = {
                "start_sec": 0.0,
                "end_sec": 1.0,
                "relative_start_sec": 0.0,
                "relative_end_sec": 1.0,
                "severity_score": 0.65,
                "labels": ["kick_punch_bleed"],
                "severity_parts": {"kick_punch_bleed": 0.65},
            }

            repaired_manifest = module.build_clustered_issue_window_ratio_repair_candidate(
                profile=profile,
                base_score=base_score,
                candidate_records_by_name={"base": base_record},
                candidate_issue_reports={"base": {"segments": [base_segment]}},
                defect_effectiveness_gate={
                    "blockers": [
                        {"reason": "selected_candidate_has_too_many_issue_windows", "issue_window_ratio": 1.0}
                    ]
                },
                dependency_fingerprints={"base": {"score": "0.000000"}},
                output_dir=temp_path / "clustered_issue_window_ratio_repair",
            )

        self.assertIsNone(repaired_manifest)

    def test_clustered_issue_window_ratio_repair_triggers_for_severe_drop_melody_blocker(self):
        module = _load_adaptive_module()

        self.assertTrue(
            module._has_clustered_issue_ratio_blocker(
                {
                    "blockers": [
                        {
                            "reason": "selected_candidate_still_has_severe_issue",
                            "labels": ["drop_melody_leakage"],
                        }
                    ]
                }
            )
        )
        self.assertFalse(
            module._has_clustered_issue_ratio_blocker(
                {
                    "blockers": [
                        {
                            "reason": "selected_candidate_still_has_severe_issue",
                            "labels": ["vocal_dropout", "vocal_tail_loss"],
                        }
                    ]
                }
            )
        )

    def test_segment_issue_repair_resolves_bleed_only_window_with_core_debleed(self):
        module = _load_adaptive_module()
        profile = module.default_segment_issue_repair_profile()
        base_score = module.VocalCandidateScore(
            profile_name="base",
            score=1.0,
            reasons=[],
            metrics={},
            manifest_path="base/stem_manifest.json",
            output_dir="base",
            vocals_path="base/vocals.flac",
        )
        donor_score = module.VocalCandidateScore(
            profile_name="donor",
            score=0.9,
            reasons=[],
            metrics={},
            manifest_path="donor/stem_manifest.json",
            output_dir="donor",
            vocals_path="donor/vocals.flac",
        )
        bleed_only_report = {
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                    "relative_start_sec": 0.0,
                    "relative_end_sec": 1.0,
                    "severity_score": 0.90,
                    "labels": ["kick_punch_bleed", "high_noise_or_robotic_artifact"],
                    "severity_parts": {
                        "kick_punch_bleed": 0.55,
                        "high_noise_or_robotic_artifact": 0.35,
                    },
                }
            ],
        }

        decisions = module._build_segment_issue_repair_window_decisions(
            base_score=base_score,
            candidate_scores=[base_score, donor_score],
            candidate_issue_reports={"base": bleed_only_report, "donor": bleed_only_report},
            tuning=profile.tuning,
        )

        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["repair_profile"], "base")
        self.assertEqual(decisions[0]["resolution_kind"], "bleed_control")
        self.assertGreater(decisions[0]["bleed_issue_score"], profile.tuning["min_bleed_issue_score"])
        self.assertEqual(decisions[0]["router_tuning"]["body_repair_strength"], 0.0)
        self.assertEqual(decisions[0]["router_tuning"]["preserve_tail_strength"], 0.0)
        self.assertGreater(decisions[0]["router_tuning"]["core_debleed_strength"], 0.0)

    def test_segment_issue_repair_skips_fragile_vocal_loss_without_recovery_donor(self):
        module = _load_adaptive_module()
        profile = module.default_segment_issue_repair_profile()
        base_score = module.VocalCandidateScore(
            profile_name="base",
            score=1.0,
            reasons=[],
            metrics={},
            manifest_path="base/stem_manifest.json",
            output_dir="base",
            vocals_path="base/vocals.flac",
        )
        fragile_report = {
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                    "relative_start_sec": 0.0,
                    "relative_end_sec": 1.0,
                    "severity_score": 4.20,
                    "labels": [
                        "vocal_dropout",
                        "vocal_tail_loss",
                        "kick_punch_bleed",
                        "high_noise_or_robotic_artifact",
                        "presence_loss",
                    ],
                    "severity_parts": {
                        "vocal_dropout": 0.80,
                        "vocal_tail_loss": 0.90,
                        "kick_punch_bleed": 0.75,
                        "high_noise_or_robotic_artifact": 0.50,
                        "presence_loss": 0.35,
                    },
                }
            ],
        }

        decisions = module._build_segment_issue_repair_window_decisions(
            base_score=base_score,
            candidate_scores=[base_score],
            candidate_issue_reports={"base": fragile_report},
            tuning=profile.tuning,
        )

        self.assertEqual(decisions, [])

    def test_segment_issue_repair_uses_local_vocal_body_dip_as_recovery_trigger(self):
        module = _load_adaptive_module()
        profile = module.default_segment_issue_repair_profile()
        base_score = module.VocalCandidateScore(
            profile_name="base",
            score=1.0,
            reasons=[],
            metrics={},
            manifest_path="base/stem_manifest.json",
            output_dir="base",
            vocals_path="base/vocals.flac",
        )
        donor_score = module.VocalCandidateScore(
            profile_name="donor",
            score=0.9,
            reasons=[],
            metrics={},
            manifest_path="donor/stem_manifest.json",
            output_dir="donor",
            vocals_path="donor/vocals.flac",
        )
        base_report = {
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                    "relative_start_sec": 0.0,
                    "relative_end_sec": 1.0,
                    "severity_score": 0.93,
                    "labels": ["local_vocal_body_dip", "kick_punch_bleed"],
                    "severity_parts": {
                        "local_vocal_body_dip": 0.48,
                        "kick_punch_bleed": 0.45,
                    },
                }
            ],
        }
        donor_report = {
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                    "relative_start_sec": 0.0,
                    "relative_end_sec": 1.0,
                    "severity_score": 0.20,
                    "labels": ["kick_punch_bleed"],
                    "severity_parts": {"kick_punch_bleed": 0.20},
                }
            ],
        }

        decisions = module._build_segment_issue_repair_window_decisions(
            base_score=base_score,
            candidate_scores=[base_score, donor_score],
            candidate_issue_reports={"base": base_report, "donor": donor_report},
            tuning=profile.tuning,
        )

        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["repair_profile"], "donor")
        self.assertEqual(decisions[0]["base_labels"], ["local_vocal_body_dip", "kick_punch_bleed"])
        self.assertEqual(decisions[0]["resolution_kind"], "combined_recovery_and_bleed_control")
        self.assertGreater(decisions[0]["router_tuning"]["core_debleed_strength"], 0.0)

    def test_no_vocal_leakage_cleanup_uses_consensus_and_skips_backing_risk(self):
        module = _load_adaptive_module()
        profile = module.default_no_vocal_leakage_cleanup_profile()
        base_score = module.VocalCandidateScore(
            profile_name="base",
            score=1.0,
            reasons=[],
            metrics={},
            manifest_path="base/stem_manifest.json",
            output_dir="base",
            vocals_path="base/vocals.flac",
        )
        donor_score = module.VocalCandidateScore(
            profile_name="donor",
            score=0.9,
            reasons=[],
            metrics={},
            manifest_path="donor/stem_manifest.json",
            output_dir="donor",
            vocals_path="donor/vocals.flac",
        )
        base_report = {
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 5.0,
                    "relative_start_sec": 0.0,
                    "relative_end_sec": 5.0,
                    "severity_score": 1.80,
                    "labels": ["thin_or_overcleaned", "presence_loss", "high_noise_or_robotic_artifact"],
                    "metrics": {"candidate_to_source_rms_db": -30.0, "kick_punch_ratio_db_250_900": -10.0},
                },
                {
                    "start_sec": 5.0,
                    "end_sec": 10.0,
                    "relative_start_sec": 5.0,
                    "relative_end_sec": 10.0,
                    "severity_score": 0.30,
                    "labels": ["high_noise_or_robotic_artifact"],
                    "metrics": {"candidate_to_source_rms_db": -25.0, "kick_punch_ratio_db_250_900": -10.0},
                },
                {
                    "start_sec": 10.0,
                    "end_sec": 15.0,
                    "relative_start_sec": 10.0,
                    "relative_end_sec": 15.0,
                    "severity_score": 2.00,
                    "labels": ["thin_or_overcleaned", "backing_vocal_loss_risk"],
                    "metrics": {"candidate_to_source_rms_db": -30.0},
                },
                {
                    "start_sec": 15.0,
                    "end_sec": 20.0,
                    "relative_start_sec": 15.0,
                    "relative_end_sec": 20.0,
                    "severity_score": 0.30,
                    "labels": ["high_noise_or_robotic_artifact"],
                    "metrics": {"candidate_to_source_rms_db": -25.0},
                },
                {
                    "start_sec": 20.0,
                    "end_sec": 25.0,
                    "relative_start_sec": 20.0,
                    "relative_end_sec": 25.0,
                    "severity_score": 2.20,
                    "labels": ["vocal_dropout", "vocal_tail_loss", "thin_or_overcleaned"],
                    "metrics": {"candidate_to_source_rms_db": -32.0},
                },
            ],
        }
        donor_report = {
            "segments": [
                {**base_report["segments"][0], "severity_score": 1.30},
                {**base_report["segments"][1], "severity_score": 0.20},
                {**base_report["segments"][2], "severity_score": 1.80},
                {**base_report["segments"][3], "severity_score": 0.20},
                {**base_report["segments"][4], "severity_score": 1.70},
            ],
        }

        decisions = module._build_no_vocal_leakage_cleanup_window_decisions(
            base_score=base_score,
            candidate_scores=[base_score, donor_score],
            candidate_issue_reports={"base": base_report, "donor": donor_report},
            tuning=profile.tuning,
        )

        self.assertEqual([decision["segment_index"] for decision in decisions], [0, 1])
        self.assertEqual(decisions[0]["cleanup_kind"], "consensus_no_vocal")
        self.assertEqual(decisions[1]["cleanup_kind"], "low_energy_no_vocal_bridge")
        self.assertEqual(decisions[0]["best_candidate_profile"], "donor")
        self.assertAlmostEqual(decisions[0]["suppression_gain"], profile.tuning["suppression_gain"])
        self.assertAlmostEqual(decisions[1]["suppression_gain"], profile.tuning["bridge_suppression_gain"])

    def test_no_vocal_leakage_cleanup_requires_low_energy_no_vocal_evidence(self):
        module = _load_adaptive_module()
        profile = module.default_no_vocal_leakage_cleanup_profile()
        base_score = module.VocalCandidateScore(
            profile_name="base",
            score=1.0,
            reasons=[],
            metrics={},
            manifest_path="base/stem_manifest.json",
            output_dir="base",
            vocals_path="base/vocals.flac",
        )
        donor_score = module.VocalCandidateScore(
            profile_name="donor",
            score=0.9,
            reasons=[],
            metrics={},
            manifest_path="donor/stem_manifest.json",
            output_dir="donor",
            vocals_path="donor/vocals.flac",
        )
        thin_but_active_report = {
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 5.0,
                    "relative_start_sec": 0.0,
                    "relative_end_sec": 5.0,
                    "severity_score": 2.20,
                    "labels": ["thin_or_overcleaned", "presence_loss", "high_noise_or_robotic_artifact"],
                    "metrics": {
                        "candidate_to_source_rms_db": -18.0,
                        "kick_punch_ratio_db_250_900": -4.0,
                    },
                }
            ],
        }
        donor_report = {
            "segments": [
                {
                    **thin_but_active_report["segments"][0],
                    "severity_score": 1.30,
                }
            ],
        }

        decisions = module._build_no_vocal_leakage_cleanup_window_decisions(
            base_score=base_score,
            candidate_scores=[base_score, donor_score],
            candidate_issue_reports={"base": thin_but_active_report, "donor": donor_report},
            tuning=profile.tuning,
        )

        self.assertEqual(decisions, [])

    def test_no_vocal_leakage_cleanup_does_not_treat_reverb_tail_smear_as_no_vocal(self):
        module = _load_adaptive_module()
        profile = module.default_no_vocal_leakage_cleanup_profile()
        base_score = module.VocalCandidateScore(
            profile_name="base",
            score=1.0,
            reasons=[],
            metrics={},
            manifest_path="base/stem_manifest.json",
            output_dir="base",
            vocals_path="base/vocals.flac",
        )
        donor_score = module.VocalCandidateScore(
            profile_name="donor",
            score=0.9,
            reasons=[],
            metrics={},
            manifest_path="donor/stem_manifest.json",
            output_dir="donor",
            vocals_path="donor/vocals.flac",
        )
        reverb_tail_smear_report = {
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 5.0,
                    "relative_start_sec": 0.0,
                    "relative_end_sec": 5.0,
                    "severity_score": 2.40,
                    "labels": ["excessive_reverb_tail_smear"],
                    "metrics": {
                        "candidate_to_source_rms_db": -30.0,
                        "kick_punch_ratio_db_250_900": -10.0,
                    },
                },
                {
                    "start_sec": 5.0,
                    "end_sec": 10.0,
                    "relative_start_sec": 5.0,
                    "relative_end_sec": 10.0,
                    "severity_score": 2.80,
                    "labels": ["excessive_reverb_tail_smear", "thin_or_overcleaned", "presence_loss"],
                    "metrics": {
                        "candidate_to_source_rms_db": -30.0,
                        "kick_punch_ratio_db_250_900": -10.0,
                    },
                },
            ],
        }
        donor_report = {
            "segments": [
                {
                    **segment,
                    "severity_score": 1.30,
                }
                for segment in reverb_tail_smear_report["segments"]
            ],
        }

        decisions = module._build_no_vocal_leakage_cleanup_window_decisions(
            base_score=base_score,
            candidate_scores=[base_score, donor_score],
            candidate_issue_reports={"base": reverb_tail_smear_report, "donor": donor_report},
            tuning=profile.tuning,
        )

        self.assertEqual(decisions, [])

    def test_no_vocal_leakage_cleanup_bridge_requires_low_kick_evidence(self):
        module = _load_adaptive_module()
        profile = module.default_no_vocal_leakage_cleanup_profile()
        base_score = module.VocalCandidateScore(
            profile_name="base",
            score=1.0,
            reasons=[],
            metrics={},
            manifest_path="base/stem_manifest.json",
            output_dir="base",
            vocals_path="base/vocals.flac",
        )
        donor_score = module.VocalCandidateScore(
            profile_name="donor",
            score=0.9,
            reasons=[],
            metrics={},
            manifest_path="donor/stem_manifest.json",
            output_dir="donor",
            vocals_path="donor/vocals.flac",
        )
        base_report = {
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 5.0,
                    "relative_start_sec": 0.0,
                    "relative_end_sec": 5.0,
                    "severity_score": 1.80,
                    "labels": ["thin_or_overcleaned", "presence_loss", "high_noise_or_robotic_artifact"],
                    "metrics": {"candidate_to_source_rms_db": -30.0, "kick_punch_ratio_db_250_900": -10.0},
                },
                {
                    "start_sec": 5.0,
                    "end_sec": 10.0,
                    "relative_start_sec": 5.0,
                    "relative_end_sec": 10.0,
                    "severity_score": 0.30,
                    "labels": ["high_noise_or_robotic_artifact"],
                    "metrics": {"candidate_to_source_rms_db": -25.0, "kick_punch_ratio_db_250_900": -3.0},
                },
            ],
        }
        donor_report = {
            "segments": [
                {**base_report["segments"][0], "severity_score": 1.30},
                {**base_report["segments"][1], "severity_score": 0.20},
            ],
        }

        decisions = module._build_no_vocal_leakage_cleanup_window_decisions(
            base_score=base_score,
            candidate_scores=[base_score, donor_score],
            candidate_issue_reports={"base": base_report, "donor": donor_report},
            tuning=profile.tuning,
        )

        self.assertEqual([decision["segment_index"] for decision in decisions], [0])
        self.assertEqual(decisions[0]["cleanup_kind"], "consensus_no_vocal")

    def test_no_vocal_leakage_cleanup_skips_vocal_loss_windows(self):
        module = _load_adaptive_module()
        profile = module.default_no_vocal_leakage_cleanup_profile()
        base_score = module.VocalCandidateScore(
            profile_name="base",
            score=1.0,
            reasons=[],
            metrics={},
            manifest_path="base/stem_manifest.json",
            output_dir="base",
            vocals_path="base/vocals.flac",
        )
        donor_score = module.VocalCandidateScore(
            profile_name="donor",
            score=0.9,
            reasons=[],
            metrics={},
            manifest_path="donor/stem_manifest.json",
            output_dir="donor",
            vocals_path="donor/vocals.flac",
        )
        vocal_loss_report = {
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 5.0,
                    "relative_start_sec": 0.0,
                    "relative_end_sec": 5.0,
                    "severity_score": 7.50,
                    "labels": [
                        "vocal_dropout",
                        "vocal_tail_loss",
                        "thin_or_overcleaned",
                        "high_noise_or_robotic_artifact",
                    ],
                    "metrics": {
                        "candidate_to_source_rms_db": -43.0,
                        "kick_punch_ratio_db_250_900": -4.0,
                    },
                }
            ],
        }
        donor_report = {
            "segments": [
                {
                    **vocal_loss_report["segments"][0],
                    "severity_score": 6.90,
                }
            ],
        }

        decisions = module._build_no_vocal_leakage_cleanup_window_decisions(
            base_score=base_score,
            candidate_scores=[base_score, donor_score],
            candidate_issue_reports={"base": vocal_loss_report, "donor": donor_report},
            tuning=profile.tuning,
        )

        self.assertEqual(decisions, [])

    def test_no_vocal_leakage_cleanup_allows_low_energy_drop_melody_only_residue(self):
        module = _load_adaptive_module()
        profile = module.default_no_vocal_leakage_cleanup_profile()
        base_score = module.VocalCandidateScore(
            profile_name="base",
            score=1.0,
            reasons=[],
            metrics={},
            manifest_path="base/stem_manifest.json",
            output_dir="base",
            vocals_path="base/vocals.flac",
        )
        donor_score = module.VocalCandidateScore(
            profile_name="donor",
            score=0.9,
            reasons=[],
            metrics={},
            manifest_path="donor/stem_manifest.json",
            output_dir="donor",
            vocals_path="donor/vocals.flac",
        )
        low_energy_drop_melody_report = {
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 5.0,
                    "relative_start_sec": 0.0,
                    "relative_end_sec": 5.0,
                    "severity_score": 0.97,
                    "labels": ["drop_melody_leakage", "local_vocal_body_dip"],
                    "metrics": {
                        "candidate_rms_dbfs": -35.0,
                        "candidate_to_source_rms_db": -23.0,
                        "kick_punch_ratio_db_250_900": -12.0,
                        "mid_tonal_band_ratio_db_700_1800": -1.5,
                        "mid_tonal_flatness_700_1800": 0.041,
                        "mid_tonal_flux_700_1800": 0.084,
                        "mid_tonal_peak_ratio_700_1800": 0.383,
                        "presence_ratio_db_1800_4500": -7.5,
                    },
                },
                {
                    "start_sec": 5.0,
                    "end_sec": 10.0,
                    "relative_start_sec": 5.0,
                    "relative_end_sec": 10.0,
                    "severity_score": 0.75,
                    "labels": ["drop_melody_leakage", "backing_vocal_loss_risk"],
                    "metrics": {
                        "candidate_rms_dbfs": -18.0,
                        "candidate_to_source_rms_db": -7.5,
                        "kick_punch_ratio_db_250_900": -12.0,
                        "mid_tonal_band_ratio_db_700_1800": -0.8,
                        "mid_tonal_flatness_700_1800": 0.022,
                        "mid_tonal_flux_700_1800": 0.125,
                        "mid_tonal_peak_ratio_700_1800": 0.369,
                        "presence_ratio_db_1800_4500": -9.5,
                    },
                },
            ],
        }
        donor_report = {
            "segments": [
                {**low_energy_drop_melody_report["segments"][0], "severity_score": 0.80},
                {**low_energy_drop_melody_report["segments"][1], "severity_score": 0.65},
            ],
        }

        decisions = module._build_no_vocal_leakage_cleanup_window_decisions(
            base_score=base_score,
            candidate_scores=[base_score, donor_score],
            candidate_issue_reports={"base": low_energy_drop_melody_report, "donor": donor_report},
            tuning=profile.tuning,
        )

        self.assertEqual([decision["segment_index"] for decision in decisions], [0])
        self.assertEqual(decisions[0]["cleanup_kind"], "drop_melody_only_residue")
        self.assertAlmostEqual(
            decisions[0]["suppression_gain"],
            profile.tuning["drop_melody_only_suppression_gain"],
        )

    def test_no_vocal_leakage_cleanup_with_drop_melody_dropout_tail_requires_narrow_path_only(self):
        module = _load_adaptive_module()
        profile = module.default_no_vocal_leakage_cleanup_profile()
        base_score = module.VocalCandidateScore(
            profile_name="base",
            score=1.0,
            reasons=[],
            metrics={},
            manifest_path="base/stem_manifest.json",
            output_dir="base",
            vocals_path="base/vocals.flac",
        )
        donor_score = module.VocalCandidateScore(
            profile_name="donor",
            score=0.9,
            reasons=[],
            metrics={},
            manifest_path="donor/stem_manifest.json",
            output_dir="donor",
            vocals_path="donor/vocals.flac",
        )
        dropout_tail_report = {
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 5.0,
                    "relative_start_sec": 0.0,
                    "relative_end_sec": 5.0,
                    "severity_score": 1.80,
                    "labels": ["drop_melody_leakage", "vocal_dropout", "vocal_tail_loss"],
                    "metrics": {
                        "candidate_rms_dbfs": -18.0,
                        "candidate_to_source_rms_db": -9.0,
                        "kick_punch_ratio_db_250_900": -7.0,
                        "mid_tonal_band_ratio_db_700_1800": -8.5,
                        "mid_tonal_flatness_700_1800": 0.14,
                        "mid_tonal_flux_700_1800": 0.20,
                        "mid_tonal_peak_ratio_700_1800": 0.19,
                        "presence_ratio_db_1800_4500": -0.7,
                    },
                }
            ],
        }
        donor_report = {
            "segments": [
                {
                    **dropout_tail_report["segments"][0],
                    "severity_score": 1.20,
                }
            ],
        }

        decisions = module._build_no_vocal_leakage_cleanup_window_decisions(
            base_score=base_score,
            candidate_scores=[base_score, donor_score],
            candidate_issue_reports={"base": dropout_tail_report, "donor": donor_report},
            tuning=profile.tuning,
        )

        self.assertEqual(decisions, [])

    def test_no_vocal_leakage_cleanup_skips_dense_kick_noise_without_no_vocal_evidence(self):
        module = _load_adaptive_module()
        profile = module.default_no_vocal_leakage_cleanup_profile()
        base_score = module.VocalCandidateScore(
            profile_name="base",
            score=1.0,
            reasons=[],
            metrics={},
            manifest_path="base/stem_manifest.json",
            output_dir="base",
            vocals_path="base/vocals.flac",
        )
        donor_score = module.VocalCandidateScore(
            profile_name="donor",
            score=0.9,
            reasons=[],
            metrics={},
            manifest_path="donor/stem_manifest.json",
            output_dir="donor",
            vocals_path="donor/vocals.flac",
        )
        dense_drop_report = {
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 5.0,
                    "relative_start_sec": 0.0,
                    "relative_end_sec": 5.0,
                    "severity_score": 3.20,
                    "labels": [
                        "kick_punch_bleed",
                        "unsupported_anchor_leakage",
                        "high_band_unsupported_leakage",
                        "high_noise_or_robotic_artifact",
                    ],
                    "metrics": {
                        "candidate_to_source_rms_db": -25.0,
                        "kick_punch_ratio_db_250_900": -0.8,
                    },
                }
            ],
        }
        donor_report = {
            "segments": [
                {
                    **dense_drop_report["segments"][0],
                    "severity_score": 1.30,
                }
            ],
        }

        decisions = module._build_no_vocal_leakage_cleanup_window_decisions(
            base_score=base_score,
            candidate_scores=[base_score, donor_score],
            candidate_issue_reports={"base": dense_drop_report, "donor": donor_report},
            tuning=profile.tuning,
        )

        self.assertEqual(decisions, [])

    def test_detected_issue_action_contract_maps_labels_to_actions_or_rejections(self):
        module = _load_adaptive_module()
        issue_report = {
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 5.0,
                    "severity_score": 1.5,
                    "labels": [
                        "vocal_dropout",
                        "kick_punch_bleed",
                        "high_noise_or_robotic_artifact",
                        "excessive_reverb_tail_smear",
                        "backing_vocal_loss_risk",
                    ],
                }
            ]
        }

        contract = module.build_detected_issue_action_contract(
            issue_report=issue_report,
            available_candidate_names={"segment_issue_repair", "vocal_harshness_guard"},
            skipped_candidates={},
        )
        serialized = json.dumps(contract, sort_keys=True)

        self.assertEqual(contract["status"], "pass")
        self.assertEqual(contract["summary"]["uncovered_labels"], [])
        self.assertIn("vocal_dropout", contract["summary"]["covered_labels"])
        self.assertIn("kick_punch_bleed", contract["summary"]["covered_labels"])
        self.assertIn("high_noise_or_robotic_artifact", contract["summary"]["covered_labels"])
        self.assertIn("excessive_reverb_tail_smear", contract["summary"]["covered_labels"])
        self.assertIn("backing_vocal_loss_risk", contract["summary"]["rejected_labels"])
        self.assertNotIn("fallback", serialized.lower())
        self.assertNotIn("timestamp", serialized.lower())

    def test_issue_effectiveness_ledger_links_windows_actions_and_blockers(self):
        module = _load_adaptive_module()
        issue_report = {
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 5.0,
                    "severity_score": 1.5,
                    "labels": ["high_noise_or_robotic_artifact"],
                    "severity_parts": {"high_noise_or_robotic_artifact": 1.5},
                    "metrics": {"high_noise_flatness_6000_16000": 0.24},
                }
            ],
            "summary": {
                "max_issue_score": 1.5,
                "mean_top3_issue_score": 1.5,
                "issue_window_ratio": 1.0,
            },
        }
        contract = module.build_detected_issue_action_contract(
            issue_report=issue_report,
            available_candidate_names={"vocal_harshness_guard"},
            skipped_candidates={},
        )
        gate = module.build_defect_effectiveness_gate(
            issue_report=issue_report,
            action_contract=contract,
            selected_metrics={"vocal_harshness_guard_high_band_reduction_ratio": 0.0},
        )

        ledger = module.build_issue_effectiveness_ledger(
            issue_report=issue_report,
            action_contract=contract,
            defect_effectiveness_gate=gate,
        )

        self.assertEqual(ledger["status"], "blocked_for_listening")
        self.assertEqual(ledger["summary"]["window_count"], 1)
        self.assertEqual(ledger["summary"]["blocked_window_count"], 1)
        self.assertEqual(ledger["windows"][0]["release_decision"], "blocked")
        self.assertEqual(ledger["windows"][0]["actions"][0]["candidate_profiles"], ["vocal_harshness_guard"])
        self.assertEqual(
            ledger["windows"][0]["blockers"][0]["reason"],
            "selected_candidate_still_has_severe_issue",
        )
        self.assertIn("high_noise_flatness_6000_16000", ledger["windows"][0]["metrics"])

    def test_defect_gate_ignores_stale_both_silent_severe_segment(self):
        module = _load_adaptive_module()
        issue_report = {
            "segments_analyzed": 1,
            "segments": [
                {
                    "start_sec": 195.0,
                    "end_sec": 198.087596,
                    "severity_score": 3.095,
                    "labels": ["leaky_full_mix_risk", "low_band_bleed", "kick_punch_bleed"],
                    "severity_parts": {
                        "leaky_full_mix_risk": 0.81,
                        "low_band_bleed": 1.48,
                        "kick_punch_bleed": 0.805,
                    },
                    "metrics": {
                        "source_rms_dbfs": -200.0,
                        "candidate_rms_dbfs": -200.0,
                        "candidate_to_source_rms_db": 0.0,
                        "low_band_ratio_db_35_180": 0.0,
                        "kick_punch_ratio_db_250_900": 0.0,
                    },
                }
            ],
            "summary": {
                "max_issue_score": 3.095,
                "mean_top3_issue_score": 3.095,
                "issue_window_ratio": 1.0,
            },
        }
        contract = module.build_detected_issue_action_contract(
            issue_report=issue_report,
            available_candidate_names={"segment_issue_repair"},
            skipped_candidates={},
        )

        gate = module.build_defect_effectiveness_gate(
            issue_report=issue_report,
            action_contract=contract,
            selected_metrics={},
        )
        ledger = module.build_issue_effectiveness_ledger(
            issue_report=issue_report,
            action_contract=contract,
            defect_effectiveness_gate=gate,
        )

        self.assertEqual(gate["status"], "release_ready")
        self.assertEqual(gate["summary"]["max_issue_score"], 0.0)
        self.assertEqual(gate["summary"]["severe_segment_count"], 0)
        self.assertEqual(gate["blockers"], [])
        self.assertEqual(ledger["status"], "release_ready")
        self.assertEqual(ledger["summary"]["window_count"], 0)
        self.assertEqual(ledger["summary"]["blocked_window_count"], 0)

    def test_defect_gate_blocks_global_vocal_instability_even_when_issue_scores_pass(self):
        module = _load_adaptive_module()
        issue_report = {
            "segments_analyzed": 5,
            "summary": {"max_issue_score": 0.60, "mean_top3_issue_score": 0.58, "issue_window_ratio": 0.20},
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 5.0,
                    "severity_score": 0.60,
                    "labels": ["kick_punch_bleed"],
                }
            ],
        }
        contract = module.build_detected_issue_action_contract(
            issue_report=issue_report,
            available_candidate_names={"clustered_issue_window_ratio_repair_pass2"},
            skipped_candidates={},
        )

        gate = module.build_defect_effectiveness_gate(
            issue_report=issue_report,
            action_contract=contract,
            selected_metrics={
                "dropout_frame_ratio": 0.259312,
                "tail_dropout_ratio": 0.24578,
                "segment_consistency_worst_dropout_frame_ratio": 0.505155,
                "segment_consistency_worst_tail_dropout_ratio": 0.578947,
                "segment_consistency_dropout_std": 0.161141,
                "segment_consistency_tail_dropout_std": 0.174906,
                "segment_consistency_rms_ratio_range_db": 21.90033,
            },
        )

        self.assertEqual(gate["status"], "blocked_for_listening")
        self.assertEqual(gate["release_status"], "diagnostic_only")
        self.assertIn(
            "selected_candidate_has_unstable_vocal_consistency",
            {blocker["reason"] for blocker in gate["blockers"]},
        )

    def test_unstable_vocal_consistency_requires_gpu_ladder_next_action(self):
        module = _load_adaptive_module()
        gate = {
            "status": "blocked_for_listening",
            "release_status": "diagnostic_only",
            "blockers": [
                {
                    "reason": "selected_candidate_has_unstable_vocal_consistency",
                    "metrics": {
                        "dropout_frame_ratio": 0.259312,
                        "tail_dropout_ratio": 0.24578,
                        "segment_consistency_rms_ratio_range_db": 21.90033,
                    },
                }
            ],
        }
        action_contract = {
            "status": "pass",
            "summary": {
                "covered_labels": ["drop_melody_leakage", "kick_punch_bleed", "vocal_dropout"],
                "uncovered_labels": [],
            },
        }

        plan = module.build_vocal_next_action_plan(
            defect_effectiveness_gate=gate,
            action_contract=action_contract,
            external_only=True,
        )

        self.assertEqual(plan["status"], "requires_new_model_candidates")
        self.assertEqual(plan["next_action"], "run_live_gpu_vocal_candidate_ladder")
        self.assertTrue(plan["forbid_cpu_dsp_repair_loop"])
        self.assertIn("audio_separator_bs_roformer_vocals", plan["recommended_model_ids"])
        self.assertIn("vocal_dropout", plan["failed_labels"])
        self.assertIn("unstable_vocal_consistency", plan["failed_labels"])

    def test_unstable_vocal_consistency_exhausts_gpu_ladder_once_all_models_evaluated(self):
        module = _load_adaptive_module()
        gate = {
            "status": "blocked_for_listening",
            "release_status": "diagnostic_only",
            "blockers": [{"reason": "selected_candidate_has_unstable_vocal_consistency"}],
        }
        action_contract = {
            "status": "pass",
            "summary": {
                "covered_labels": ["drop_melody_leakage", "vocal_dropout"],
                "uncovered_labels": [],
            },
        }

        plan = module.build_vocal_next_action_plan(
            defect_effectiveness_gate=gate,
            action_contract=action_contract,
            external_only=True,
            evaluated_model_ids=module.LIVE_GPU_VOCAL_LADDER_MODEL_IDS,
        )

        self.assertEqual(plan["status"], "candidate_space_exhausted")
        self.assertEqual(plan["next_action"], "escalate_to_new_vocal_route_or_teacher_calibration")
        self.assertEqual(plan["recommended_model_ids"], [])
        self.assertTrue(plan["forbid_cpu_dsp_repair_loop"])
        self.assertIn("run_teacher_calibrated_vocal_selector", plan["required_next_step_options"])

    def test_collect_evaluated_live_gpu_vocal_model_ids_uses_manifest_provenance(self):
        module = _load_adaptive_module()
        record = module.VocalCandidateRecord(
            profile=module.VocalCandidateProfile(
                name="external_mdx_alias",
                candidate_mode="external_vocals",
                prior_score=0.12,
                intent="test alias for registered model output",
                tuning={},
            ),
            manifest_path=Path("stem_manifest.json"),
            output_dir=Path("."),
            vocals_path=Path("vocals.flac"),
            metrics={},
            manifest={
                "pipeline": {
                    "mode": "external_vocal_candidate",
                    "source_candidate_path": "/app/lab_artifacts/golden_audio/run/mdx23c_vocals/vocals.flac",
                },
                "stems": {"vocals": {"path": "vocals.flac"}},
            },
        )

        self.assertEqual(
            module.collect_evaluated_live_gpu_vocal_model_ids([record]),
            {"mdx23c_vocals"},
        )

    def test_collect_evaluated_live_gpu_vocal_model_ids_does_not_treat_profile_alias_as_model_id(self):
        module = _load_adaptive_module()
        record = module.VocalCandidateRecord(
            profile=module.VocalCandidateProfile(
                name="ft_tail_safe",
                candidate_mode="model_profile",
                prior_score=0.24,
                intent="built-in adaptive profile alias",
                tuning={},
            ),
            manifest_path=Path("stem_manifest.json"),
            output_dir=Path("."),
            vocals_path=Path("vocals.flac"),
            metrics={},
            manifest={
                "pipeline": {
                    "mode": "adaptive_vocal_candidate",
                    "candidate_mode": "model_profile",
                },
                "stems": {"vocals": {"path": "vocals.flac"}},
            },
        )

        self.assertEqual(module.collect_evaluated_live_gpu_vocal_model_ids([record]), set())

    def test_live_gpu_vocal_ladder_references_registered_models(self):
        module = _load_adaptive_module()
        from worker.model_registry import default_model_registry

        registry_ids = {entry.model_id for entry in default_model_registry().entries}

        self.assertTrue(module.LIVE_GPU_VOCAL_LADDER_MODEL_IDS)
        self.assertIn(
            "audio_separator_melband_roformer_viperx_vocals",
            module.LIVE_GPU_VOCAL_LADDER_MODEL_IDS,
        )
        self.assertEqual(set(module.LIVE_GPU_VOCAL_LADDER_MODEL_IDS) - registry_ids, set())

    def test_vocal_next_action_allows_release_when_gate_is_ready(self):
        module = _load_adaptive_module()

        plan = module.build_vocal_next_action_plan(
            defect_effectiveness_gate={"status": "release_ready", "release_status": "release_ready", "blockers": []},
            action_contract={"status": "pass", "summary": {"covered_labels": [], "uncovered_labels": []}},
            external_only=False,
        )

        self.assertEqual(plan["status"], "release_ready")
        self.assertEqual(plan["next_action"], "publish_if_delivery_contract_accepts")
        self.assertFalse(plan["forbid_cpu_dsp_repair_loop"])

    def test_defect_gate_keeps_source_silent_candidate_active_segment_blocked(self):
        module = _load_adaptive_module()
        issue_report = {
            "segments_analyzed": 1,
            "segments": [
                {
                    "start_sec": 195.0,
                    "end_sec": 198.087596,
                    "severity_score": 3.095,
                    "labels": ["leaky_full_mix_risk", "low_band_bleed", "kick_punch_bleed"],
                    "severity_parts": {
                        "leaky_full_mix_risk": 0.81,
                        "low_band_bleed": 1.48,
                        "kick_punch_bleed": 0.805,
                    },
                    "metrics": {
                        "source_rms_dbfs": -200.0,
                        "candidate_rms_dbfs": -50.0,
                        "candidate_to_source_rms_db": 150.0,
                        "low_band_ratio_db_35_180": -4.0,
                        "kick_punch_ratio_db_250_900": -3.0,
                    },
                }
            ],
            "summary": {
                "max_issue_score": 3.095,
                "mean_top3_issue_score": 3.095,
                "issue_window_ratio": 1.0,
            },
        }
        contract = module.build_detected_issue_action_contract(
            issue_report=issue_report,
            available_candidate_names={"segment_issue_repair"},
            skipped_candidates={},
        )

        gate = module.build_defect_effectiveness_gate(
            issue_report=issue_report,
            action_contract=contract,
            selected_metrics={},
        )
        ledger = module.build_issue_effectiveness_ledger(
            issue_report=issue_report,
            action_contract=contract,
            defect_effectiveness_gate=gate,
        )

        self.assertEqual(gate["status"], "blocked_for_listening")
        self.assertEqual(gate["summary"]["max_issue_score"], 3.095)
        self.assertEqual(gate["summary"]["severe_segment_count"], 1)
        self.assertEqual(gate["blockers"][0]["reason"], "selected_candidate_still_has_severe_issue")
        self.assertEqual(ledger["status"], "blocked_for_listening")
        self.assertEqual(ledger["summary"]["window_count"], 1)
        self.assertEqual(ledger["summary"]["blocked_window_count"], 1)
        self.assertEqual(ledger["windows"][0]["release_decision"], "blocked")

    def test_defect_gate_uses_material_issue_ratio_for_many_tiny_labels(self):
        module = _load_adaptive_module()
        segments = [
            {
                "start_sec": float(index * 5),
                "end_sec": float(index * 5 + 5),
                "severity_score": 0.05,
                "labels": ["kick_punch_bleed"],
                "severity_parts": {"kick_punch_bleed": 0.05},
                "metrics": {"source_rms_dbfs": -18.0, "candidate_rms_dbfs": -18.0},
            }
            for index in range(8)
        ]
        issue_report = {
            "segments_analyzed": 8,
            "segments": segments,
            "summary": {
                "max_issue_score": 0.05,
                "mean_top3_issue_score": 0.05,
                "issue_window_ratio": 1.0,
            },
        }
        contract = module.build_detected_issue_action_contract(
            issue_report=issue_report,
            available_candidate_names={"segment_issue_repair"},
            skipped_candidates={},
        )

        gate = module.build_defect_effectiveness_gate(
            issue_report=issue_report,
            action_contract=contract,
            selected_metrics={},
        )

        self.assertEqual(gate["status"], "release_ready")
        self.assertEqual(gate["summary"]["raw_issue_window_ratio"], 1.0)
        self.assertEqual(gate["summary"]["issue_window_ratio"], 0.0)
        self.assertEqual(gate["blockers"], [])

    def test_defect_gate_blocks_many_material_issue_windows(self):
        module = _load_adaptive_module()
        segments = [
            {
                "start_sec": float(index * 5),
                "end_sec": float(index * 5 + 5),
                "severity_score": 0.25,
                "labels": ["kick_punch_bleed"],
                "severity_parts": {"kick_punch_bleed": 0.25},
                "metrics": {"source_rms_dbfs": -18.0, "candidate_rms_dbfs": -18.0},
            }
            for index in range(5)
        ] + [
            {
                "start_sec": float(index * 5),
                "end_sec": float(index * 5 + 5),
                "severity_score": 0.05,
                "labels": ["kick_punch_bleed"],
                "severity_parts": {"kick_punch_bleed": 0.05},
                "metrics": {"source_rms_dbfs": -18.0, "candidate_rms_dbfs": -18.0},
            }
            for index in range(5, 8)
        ]
        issue_report = {
            "segments_analyzed": 8,
            "segments": segments,
            "summary": {
                "max_issue_score": 0.25,
                "mean_top3_issue_score": 0.25,
                "issue_window_ratio": 1.0,
            },
        }
        contract = module.build_detected_issue_action_contract(
            issue_report=issue_report,
            available_candidate_names={"segment_issue_repair"},
            skipped_candidates={},
        )

        gate = module.build_defect_effectiveness_gate(
            issue_report=issue_report,
            action_contract=contract,
            selected_metrics={},
        )

        self.assertEqual(gate["status"], "blocked_for_listening")
        self.assertEqual(gate["summary"]["raw_issue_window_ratio"], 1.0)
        self.assertEqual(gate["summary"]["issue_window_ratio"], 0.625)
        self.assertEqual(gate["blockers"][0]["reason"], "selected_candidate_has_too_many_issue_windows")

    def test_issue_effectiveness_ledger_tolerates_small_window_jitter(self):
        module = _load_adaptive_module()
        issue_report = {
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 5.0,
                    "severity_score": 1.5,
                    "labels": ["high_noise_or_robotic_artifact"],
                    "severity_parts": {"high_noise_or_robotic_artifact": 1.5},
                    "metrics": {"high_noise_flatness_6000_16000": 0.24},
                }
            ],
            "summary": {"max_issue_score": 1.5, "mean_top3_issue_score": 1.5, "issue_window_ratio": 1.0},
        }
        contract = {
            "status": "pass",
            "segments": [
                {
                    "start_sec": 0.0004,
                    "end_sec": 5.0004,
                    "actions": [
                        {
                            "label": "high_noise_or_robotic_artifact",
                            "status": "action_available",
                            "candidate_profiles": ["vocal_harshness_guard"],
                        }
                    ],
                    "rejections": [],
                    "uncovered_labels": [],
                }
            ],
        }
        gate = module.build_defect_effectiveness_gate(
            issue_report=issue_report,
            action_contract=contract,
            selected_metrics={},
        )

        ledger = module.build_issue_effectiveness_ledger(
            issue_report=issue_report,
            action_contract=contract,
            defect_effectiveness_gate=gate,
        )

        self.assertEqual(ledger["windows"][0]["actions"][0]["candidate_profiles"], ["vocal_harshness_guard"])

    def test_issue_effectiveness_ledger_scores_action_candidate_delta_by_window(self):
        module = _load_adaptive_module()
        issue_report = {
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 5.0,
                    "severity_score": 1.2,
                    "labels": ["vocal_dropout", "high_noise_or_robotic_artifact"],
                    "severity_parts": {
                        "vocal_dropout": 0.8,
                        "high_noise_or_robotic_artifact": 0.4,
                    },
                    "metrics": {"dropout_frame_ratio": 0.72},
                }
            ],
            "summary": {"max_issue_score": 1.2, "mean_top3_issue_score": 1.2, "issue_window_ratio": 1.0},
        }
        candidate_issue_reports = {
            "segment_issue_repair": {
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 5.0,
                        "severity_score": 0.35,
                        "labels": ["high_noise_or_robotic_artifact"],
                        "severity_parts": {"high_noise_or_robotic_artifact": 0.35},
                    }
                ]
            },
            "vocal_harshness_guard": {
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 5.0,
                        "severity_score": 1.35,
                        "labels": ["vocal_dropout"],
                        "severity_parts": {"vocal_dropout": 1.35},
                    }
                ]
            },
        }
        contract = module.build_detected_issue_action_contract(
            issue_report=issue_report,
            available_candidate_names={"segment_issue_repair", "vocal_harshness_guard"},
            skipped_candidates={},
        )
        gate = module.build_defect_effectiveness_gate(
            issue_report=issue_report,
            action_contract=contract,
            selected_metrics={},
        )

        ledger = module.build_issue_effectiveness_ledger(
            issue_report=issue_report,
            action_contract=contract,
            defect_effectiveness_gate=gate,
            candidate_issue_reports=candidate_issue_reports,
        )

        effectiveness = ledger["windows"][0]["action_effectiveness"]
        self.assertEqual(ledger["summary"]["action_improved_window_count"], 1)
        self.assertEqual(effectiveness["best_candidate_profile"], "segment_issue_repair")
        self.assertEqual(effectiveness["status"], "improved")
        self.assertAlmostEqual(effectiveness["severity_delta"], 0.85, places=6)
        self.assertIn("vocal_dropout", effectiveness["resolved_labels"])

    def test_stage_profiler_records_named_stage_and_heartbeat_payload(self):
        module = _load_adaptive_module()
        profiler = module.StageProfiler(emit_heartbeat=False)

        with profiler.measure("proxy_metrics", candidate="hybrid_tail_preserve"):
            pass

        payload = profiler.to_manifest()
        self.assertEqual(payload["schema_version"], 1)
        self.assertIn("proxy_metrics", payload["stages"])
        self.assertGreaterEqual(payload["stages"]["proxy_metrics"]["elapsed_sec"], 0.0)
        self.assertEqual(payload["events"][0]["stage"], "proxy_metrics")
        self.assertEqual(payload["events"][0]["status"], "start")
        self.assertEqual(payload["events"][0]["metadata"]["candidate"], "hybrid_tail_preserve")

    def test_audio_analysis_cache_reuses_candidate_audio_for_metric_layers(self):
        module = _load_adaptive_module()
        sample_rate = 8000
        t = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
        vocal = (0.2 * np.sin(2.0 * np.pi * 330.0 * t)).astype(np.float32)
        source_audio = np.stack([vocal, vocal], axis=1).astype(np.float32)
        vocals_audio = source_audio.copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.flac"
            vocals_path = temp_path / "vocals.flac"
            sf.write(source_path, source_audio, sample_rate, subtype="PCM_24")
            sf.write(vocals_path, vocals_audio, sample_rate, subtype="PCM_24")
            cache = module.AudioAnalysisCache()

            original_read = module.sf.read
            read_paths: list[str] = []

            def counting_read(path, *args, **kwargs):
                read_paths.append(str(path))
                return original_read(path, *args, **kwargs)

            with patch.object(module.sf, "read", side_effect=counting_read):
                module.calculate_vocal_proxy_metrics(
                    source_path,
                    vocals_path,
                    start_sec=0.0,
                    end_sec=2.0,
                    analysis_cache=cache,
                )
                module.calculate_vocal_proxy_consistency_metrics(
                    source_path,
                    vocals_path,
                    start_sec=0.0,
                    end_sec=2.0,
                    segment_duration_sec=1.0,
                    analysis_cache=cache,
                )
                module.calculate_vocal_segment_issue_report(
                    source_path,
                    vocals_path,
                    anchor_vocals_path=None,
                    start_sec=0.0,
                    end_sec=2.0,
                    segment_duration_sec=1.0,
                    max_ranked_segments=2,
                    analysis_cache=cache,
                )

        self.assertEqual(read_paths.count(str(vocals_path)), 1)
        self.assertGreater(cache.stats()["audio_cache_hits"], 0)

    def test_detected_issue_action_contract_skips_malformed_segments(self):
        module = _load_adaptive_module()

        contract = module.build_detected_issue_action_contract(
            issue_report={"segments": [{"severity_score": 1.2, "labels": ["high_noise_or_robotic_artifact"]}]},
            available_candidate_names={"vocal_harshness_guard"},
            skipped_candidates={},
        )

        self.assertEqual(contract["status"], "pass")
        self.assertEqual(contract["summary"]["issue_segment_count"], 0)

    def test_detected_issue_action_contract_records_unavailable_candidate_reason(self):
        module = _load_adaptive_module()
        issue_report = {
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 5.0,
                    "severity_score": 1.5,
                    "labels": ["high_noise_or_robotic_artifact"],
                }
            ]
        }

        contract = module.build_detected_issue_action_contract(
            issue_report=issue_report,
            available_candidate_names=set(),
            skipped_candidates={
                "vocal_harshness_guard": [{"name": "vocal_harshness_guard", "reason": "no_harshness_guard_window"}],
            },
        )

        self.assertEqual(contract["status"], "pass")
        self.assertEqual(contract["summary"]["uncovered_labels"], [])
        self.assertEqual(contract["segments"][0]["rejections"][0]["status"], "candidate_unavailable")
        self.assertEqual(contract["segments"][0]["rejections"][0]["reason"], "no_harshness_guard_window")

    def test_detected_issue_action_contract_reports_defect_specific_action_families(self):
        module = _load_adaptive_module()
        issue_report = {
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 5.0,
                    "severity_score": 1.5,
                    "labels": [
                        "vocal_dropout",
                        "vocal_tail_loss",
                        "drop_melody_leakage",
                        "kick_punch_bleed",
                        "local_vocal_body_dip",
                    ],
                }
            ]
        }

        contract = module.build_detected_issue_action_contract(
            issue_report=issue_report,
            available_candidate_names={
                "segment_issue_repair",
                "issue_aware_segment_selection",
                "clustered_issue_window_ratio_repair",
            },
            skipped_candidates={},
        )

        actions_by_label = {
            action["label"]: action
            for action in contract["segments"][0]["actions"]
        }
        self.assertEqual(actions_by_label["vocal_dropout"]["action_family"], "vocal_presence_reconstruction")
        self.assertEqual(actions_by_label["vocal_tail_loss"]["action_family"], "vocal_presence_reconstruction")
        self.assertEqual(actions_by_label["drop_melody_leakage"]["action_family"], "melody_leakage_rejection")
        self.assertEqual(actions_by_label["kick_punch_bleed"]["action_family"], "percussive_bleed_rejection")
        self.assertEqual(actions_by_label["local_vocal_body_dip"]["action_family"], "vocal_body_recovery")

    def test_detected_issue_action_contract_reports_missing_issue_report(self):
        module = _load_adaptive_module()

        contract = module.build_detected_issue_action_contract(
            issue_report=None,
            available_candidate_names={"segment_issue_repair"},
            skipped_candidates={},
        )

        self.assertEqual(contract["status"], "missing_issue_report")
        self.assertEqual(contract["summary"]["issue_segment_count"], 0)
        self.assertEqual(contract["summary"]["uncovered_labels"], [])

    def test_detected_issue_action_contract_reports_uncovered_unknown_label(self):
        module = _load_adaptive_module()
        issue_report = {
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 5.0,
                    "severity_score": 1.5,
                    "labels": ["new_unmapped_audible_defect"],
                }
            ]
        }

        contract = module.build_detected_issue_action_contract(
            issue_report=issue_report,
            available_candidate_names={"segment_issue_repair"},
            skipped_candidates={},
        )

        self.assertEqual(contract["status"], "uncovered_labels")
        self.assertEqual(contract["summary"]["uncovered_labels"], ["new_unmapped_audible_defect"])
        self.assertEqual(contract["segments"][0]["uncovered_labels"], ["new_unmapped_audible_defect"])

    def test_detected_issue_action_contract_maps_no_vocal_evidence_to_cleanup_candidate(self):
        module = _load_adaptive_module()
        issue_report = {
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 5.0,
                    "severity_score": 1.5,
                    "labels": ["thin_or_overcleaned", "presence_loss"],
                }
            ]
        }

        contract = module.build_detected_issue_action_contract(
            issue_report=issue_report,
            available_candidate_names={"no_vocal_leakage_cleanup"},
            skipped_candidates={},
        )

        self.assertEqual(contract["status"], "pass")
        self.assertEqual(contract["summary"]["uncovered_labels"], [])
        self.assertEqual(
            contract["segments"][0]["actions"],
            [
                {
                    "label": "thin_or_overcleaned",
                    "status": "action_available",
                    "action_family": "vocal_body_recovery",
                    "candidate_profiles": ["no_vocal_leakage_cleanup"],
                },
                {
                    "label": "presence_loss",
                    "status": "action_available",
                    "action_family": "vocal_body_recovery",
                    "candidate_profiles": ["no_vocal_leakage_cleanup"],
                },
            ],
        )

    def test_no_vocal_leakage_cleanup_candidate_attenuates_only_detected_window(self):
        module = _load_adaptive_module()
        sample_rate = 8000
        t = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
        base_mono = (0.30 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        base_audio = np.stack([base_mono, base_mono], axis=1).astype(np.float32)
        profile_template = module.default_no_vocal_leakage_cleanup_profile()
        profile = module.VocalCandidateProfile(
            name=profile_template.name,
            candidate_mode=profile_template.candidate_mode,
            prior_score=profile_template.prior_score,
            intent=profile_template.intent,
            tuning={
                **profile_template.tuning,
                "crossfade_sec": 0.0,
                "min_region_duration_sec": 1.0,
                "suppression_gain": 0.10,
                "bridge_suppression_gain": 0.20,
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.flac"
            base_dir = temp_path / "base"
            base_dir.mkdir()
            base_vocals_path = base_dir / "vocals.flac"
            sf.write(source_path, base_audio, sample_rate, subtype="PCM_24")
            sf.write(base_vocals_path, base_audio, sample_rate, subtype="PCM_24")
            base_manifest = {
                "schema_version": 1,
                "audio_path": str(source_path),
                "source_audio_sha256": module._sha256_file(source_path),
                "sample_rate": sample_rate,
                "preview_window_sec": {"start": 0.0, "end": 2.0},
                "pipeline": {"candidate_mode": "base_mode"},
                "tuning": {},
                "stems": {"vocals": {"path": str(base_vocals_path)}},
            }
            base_manifest_path = base_dir / "stem_manifest.json"
            base_manifest_path.write_text(json.dumps(base_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            base_record = module.VocalCandidateRecord(
                profile=module.VocalCandidateProfile("base", "base_mode", 0.0, "base", {}),
                manifest_path=base_manifest_path,
                output_dir=base_dir,
                vocals_path=base_vocals_path,
                metrics={},
                manifest=base_manifest,
            )
            base_score = module.VocalCandidateScore(
                profile_name="base",
                score=1.0,
                reasons=[],
                metrics={},
                manifest_path=str(base_record.manifest_path),
                output_dir=str(base_record.output_dir),
                vocals_path=str(base_record.vocals_path),
            )
            donor_score = module.VocalCandidateScore(
                profile_name="donor",
                score=0.9,
                reasons=[],
                metrics={},
                manifest_path="donor/stem_manifest.json",
                output_dir="donor",
                vocals_path="donor/vocals.flac",
            )
            base_report = {
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 1.0,
                        "relative_start_sec": 0.0,
                        "relative_end_sec": 1.0,
                        "severity_score": 1.80,
                        "labels": ["thin_or_overcleaned", "presence_loss", "high_noise_or_robotic_artifact"],
                        "metrics": {"candidate_to_source_rms_db": -30.0, "kick_punch_ratio_db_250_900": -10.0},
                    },
                    {
                        "start_sec": 1.0,
                        "end_sec": 2.0,
                        "relative_start_sec": 1.0,
                        "relative_end_sec": 2.0,
                        "severity_score": 0.05,
                        "labels": ["acceptable_proxy_window"],
                        "metrics": {"candidate_to_source_rms_db": -8.0},
                    },
                ],
            }
            donor_report = {
                "segments": [
                    {**base_report["segments"][0], "severity_score": 1.30},
                    base_report["segments"][1],
                ],
            }
            output_dir = temp_path / "no_vocal_leakage_cleanup"

            manifest = module.build_no_vocal_leakage_cleanup_candidate(
                profile=profile,
                base_score=base_score,
                candidate_scores=[base_score, donor_score],
                candidate_records_by_name={"base": base_record},
                candidate_issue_reports={"base": base_report, "donor": donor_report},
                dependency_fingerprints={"base": {"score": "1.000000"}, "donor": {"score": "0.900000"}},
                output_dir=output_dir,
            )
            cleaned, cleaned_sr = sf.read(output_dir / "vocals.flac", always_2d=True)

        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.assertEqual(cleaned_sr, sample_rate)
        self.assertEqual(manifest["stems"]["vocals"]["cleaned_window_count"], 1)
        self.assertEqual(manifest["stems"]["vocals"]["window_decisions"][0]["cleanup_kind"], "consensus_no_vocal")
        self.assertLess(
            _band_rms(cleaned[:sample_rate], sample_rate, 400.0, 480.0),
            0.20 * _band_rms(base_audio[:sample_rate], sample_rate, 400.0, 480.0),
        )
        np.testing.assert_allclose(cleaned[sample_rate:], base_audio[sample_rate:], atol=8e-5)

    def test_ir_residual_candidate_builds_raw_minus_instrumental_proxy(self):
        module = _load_adaptive_module()
        sample_rate = 16000
        t = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
        vocal = (0.18 * np.sin(2.0 * np.pi * 330.0 * t)).astype(np.float32)
        instrumental = (0.12 * np.sin(2.0 * np.pi * 110.0 * t)).astype(np.float32)
        source = np.stack([vocal + instrumental, vocal + instrumental], axis=1).astype(np.float32)
        minus = np.stack([instrumental, instrumental], axis=1).astype(np.float32)
        profile = module.VocalCandidateProfile(
            name="ir_residual_vocals",
            candidate_mode="ir_residual_vocals",
            prior_score=0.18,
            intent="test residual",
            tuning={"instrumental_gain": 1.0},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.flac"
            minus_path = temp_path / "minus.flac"
            output_dir = temp_path / "candidate"
            sf.write(source_path, source, sample_rate, subtype="PCM_24")
            sf.write(minus_path, minus, sample_rate, subtype="PCM_24")

            manifest = module.build_ir_residual_vocal_candidate(
                profile=profile,
                source_audio_path=source_path,
                instrumental_minus_path=minus_path,
                output_dir=output_dir,
                start_sec=0.5,
                end_sec=1.5,
            )
            residual, residual_sr = sf.read(output_dir / "vocals.flac", always_2d=True, dtype="float32")

        expected = np.stack(
            [
                vocal[int(0.5 * sample_rate) : int(1.5 * sample_rate)],
                vocal[int(0.5 * sample_rate) : int(1.5 * sample_rate)],
            ],
            axis=1,
        )
        self.assertEqual(residual_sr, sample_rate)
        np.testing.assert_allclose(residual, expected, atol=2e-4)
        self.assertFalse(manifest["pipeline"]["ground_truth_contract"]["verified"])
        self.assertEqual(manifest["pipeline"]["candidate_mode"], "ir_residual_vocals")
        self.assertLess(manifest["residual_contract"]["reconstruction_error_rms_dbfs"], -100.0)

    def test_external_vocal_candidate_rejects_reserved_ir_residual_name(self):
        module = _load_adaptive_module()

        with self.assertRaises(SystemExit):
            module.parse_args(
                [
                    "source.flac",
                    "out",
                    "--external-vocal-candidate",
                    "ir_residual_vocals=/tmp/vocals.flac",
                ]
            )

    def test_cli_external_only_skips_builtin_candidate_inference(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sample_rate = 8000
            t = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
            vocal = 0.10 * np.sin(2.0 * np.pi * 330.0 * t)
            source_audio = np.stack([vocal, vocal], axis=1).astype(np.float32)
            source_path = temp_path / "source.flac"
            external_vocals_path = temp_path / "external_vocals.flac"
            export_script = temp_path / "export_hq_vocal_preview.py"
            output_dir = temp_path / "adaptive"
            sf.write(source_path, source_audio, sample_rate, subtype="PCM_24")
            sf.write(external_vocals_path, source_audio, sample_rate, subtype="PCM_24")
            export_script.write_text("# fake export script\n", encoding="utf-8")

            issue_report = {
                "schema_version": 1,
                "mode": "vocal_segment_issue_report",
                "segments": [],
                "top_segments": [],
                "summary": {
                    "max_issue_score": 0.0,
                    "mean_top3_issue_score": 0.0,
                    "issue_segments_count": 0.0,
                    "issue_window_ratio": 0.0,
                    "dominant_issue_labels": [],
                },
                "summary_metrics": {
                    "segment_issue_max_score": 0.0,
                    "segment_issue_mean_top3_score": 0.0,
                    "segment_issue_count": 0.0,
                    "segment_issue_window_ratio": 0.0,
                },
            }
            argv = [
                str(source_path),
                str(output_dir),
                "--export-script",
                str(export_script),
                "--python-executable",
                sys.executable,
                "--end-sec",
                "1.0",
                "--segment-consistency-sec",
                "0",
                "--external-only",
                "--external-vocal-candidate",
                f"model_zoo_vocals={external_vocals_path}",
                "--disable-anchor-suppression",
                "--disable-body-repair",
                "--disable-segment-router",
                "--disable-issue-aware-segment-selection",
                "--disable-segment-issue-repair",
                "--disable-no-vocal-leakage-cleanup",
                "--disable-vocal-harshness-guard",
            ]

            with patch.object(module, "ensure_bs_roformer_runtime") as ensure_runtime_mock, patch.object(
                module,
                "_run_command",
            ) as run_command_mock, patch.object(
                module,
                "calculate_vocal_proxy_metrics",
                return_value=_metrics(
                    dropout=0.01,
                    tail_dropout=0.01,
                    rms_ratio_db=-6.0,
                    low_ratio_db=-30.0,
                    kick_ratio_db=-30.0,
                ),
            ), patch.object(
                module,
                "calculate_vocal_segment_issue_report",
                return_value=issue_report,
            ), patch.object(
                module,
                "calculate_anchor_support_metrics",
                return_value={
                    "unsupported_vs_anchor_ratio_db": -40.0,
                    "high_unsupported_vs_anchor_ratio_db": -40.0,
                    "high_noise_flatness_6000_16000": 0.05,
                    "stereo_side_mid_ratio_db": -16.0,
                },
            ):
                self.assertEqual(module.main(argv), 0)

            adaptive_manifest = json.loads(
                (output_dir / "adaptive_vocal_manifest.json").read_text(encoding="utf-8"),
            )
            progress_log_path = output_dir / "adaptive_vocal_progress.jsonl"
            progress_log_exists = progress_log_path.exists()
            progress_events = [
                json.loads(line)
                for line in progress_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        ensure_runtime_mock.assert_not_called()
        run_command_mock.assert_not_called()
        self.assertTrue(adaptive_manifest["external_only"])
        self.assertEqual(adaptive_manifest["candidate_profiles"], [])
        self.assertEqual(adaptive_manifest["selected_profile"], "model_zoo_vocals")
        self.assertEqual(adaptive_manifest["external_candidate_profiles"][0]["name"], "model_zoo_vocals")
        self.assertEqual(
            adaptive_manifest["execution_contract"]["expected_acceleration_mode"],
            "external_only_cpu_dsp_scoring",
        )
        self.assertTrue(adaptive_manifest["execution_contract"]["gpu_inference_skipped"])
        self.assertEqual(
            adaptive_manifest["execution_contract"]["gpu_inference_skip_reason"],
            "external_only_precomputed_candidates",
        )
        self.assertEqual(adaptive_manifest["execution_contract"]["progress_log_path"], str(progress_log_path))
        self.assertTrue(progress_log_exists)
        self.assertTrue(
            any(
                event["stage"] == "execution_contract"
                and event["metadata"]["gpu_inference_skipped"] is True
                for event in progress_events
            )
        )
        self.assertTrue(
            any(event["stage"] == "proxy_metrics" and event["status"] == "start" for event in progress_events)
        )

    def test_cli_external_only_rejects_short_preview_candidate_for_full_window(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sample_rate = 8000
            source_t = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
            candidate_t = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
            source_audio = np.stack(
                [
                    0.10 * np.sin(2.0 * np.pi * 330.0 * source_t),
                    0.10 * np.sin(2.0 * np.pi * 330.0 * source_t),
                ],
                axis=1,
            ).astype(np.float32)
            candidate_audio = np.stack(
                [
                    0.10 * np.sin(2.0 * np.pi * 330.0 * candidate_t),
                    0.10 * np.sin(2.0 * np.pi * 330.0 * candidate_t),
                ],
                axis=1,
            ).astype(np.float32)
            source_path = temp_path / "source.flac"
            short_candidate_path = temp_path / "short_candidate.flac"
            export_script = temp_path / "export_hq_vocal_preview.py"
            output_dir = temp_path / "adaptive"
            sf.write(source_path, source_audio, sample_rate, subtype="PCM_24")
            sf.write(short_candidate_path, candidate_audio, sample_rate, subtype="PCM_24")
            export_script.write_text("# fake export script\n", encoding="utf-8")

            argv = [
                str(source_path),
                str(output_dir),
                "--export-script",
                str(export_script),
                "--python-executable",
                sys.executable,
                "--external-only",
                "--external-vocal-candidate",
                f"short_preview={short_candidate_path}",
                "--disable-anchor-suppression",
                "--disable-body-repair",
                "--disable-segment-router",
                "--disable-issue-aware-segment-selection",
                "--disable-segment-issue-repair",
                "--disable-no-vocal-leakage-cleanup",
                "--disable-vocal-harshness-guard",
            ]

            with patch.object(module, "ensure_bs_roformer_runtime") as ensure_runtime_mock, patch.object(
                module,
                "_run_command",
            ) as run_command_mock:
                with self.assertRaisesRegex(ValueError, "shorter than the requested analysis window"):
                    module.main(argv)

        ensure_runtime_mock.assert_not_called()
        run_command_mock.assert_not_called()

    def test_cli_ft_only_profile_does_not_require_bs_roformer_runtime(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            export_script = temp_path / "export_hq_vocal_preview.py"
            output_dir = temp_path / "adaptive"
            source_path.write_bytes(b"source")
            export_script.write_text("# fake export script\n", encoding="utf-8")

            def fake_run_command(command: list[str]) -> None:
                output_arg = command[command.index("--output-dir") + 1]
                output_path = Path(output_arg)
                _write_fake_candidate(output_path, output_path.name)
                manifest_path = output_path / "stem_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest.update(
                    {
                        "audio_path": str(source_path),
                        "source_audio_sha256": module._sha256_file(source_path),
                        "preview_window_sec": {"start": 0.0, "end": 12.0},
                        "tuning": module.default_candidate_profiles()[2].tuning,
                    }
                )
                manifest["pipeline"]["candidate_mode"] = "ft_vocals"
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            argv = [
                str(source_path),
                str(output_dir),
                "--export-script",
                str(export_script),
                "--python-executable",
                sys.executable,
                "--end-sec",
                "12.0",
                "--segment-consistency-sec",
                "0",
                "--profile",
                "ft_tail_safe",
                "--disable-anchor-suppression",
                "--disable-body-repair",
                "--disable-segment-router",
                "--disable-issue-aware-segment-selection",
                "--disable-segment-issue-repair",
                "--disable-no-vocal-leakage-cleanup",
                "--disable-vocal-harshness-guard",
            ]

            with patch.object(module, "ensure_bs_roformer_runtime") as ensure_runtime_mock, patch.object(
                module,
                "_run_command",
                side_effect=fake_run_command,
            ) as run_command_mock, patch.object(
                module,
                "calculate_vocal_proxy_metrics",
                return_value=_metrics(
                    dropout=0.04,
                    tail_dropout=0.05,
                    rms_ratio_db=-14.0,
                    low_ratio_db=-22.0,
                    kick_ratio_db=-15.0,
                ),
            ):
                self.assertEqual(module.main(argv), 0)

            adaptive_manifest = json.loads(
                (output_dir / "adaptive_vocal_manifest.json").read_text(encoding="utf-8"),
            )

        ensure_runtime_mock.assert_not_called()
        run_command_mock.assert_called_once()
        self.assertEqual(adaptive_manifest["selected_profile"], "ft_tail_safe")

    def test_profile_runtime_requirement_keeps_bs_profiles_guarded(self):
        module = _load_adaptive_module()
        profiles = {profile.name: profile for profile in module.default_candidate_profiles()}

        self.assertFalse(module.profiles_require_bs_roformer_runtime([profiles["ft_tail_safe"]]))
        self.assertTrue(module.profiles_require_bs_roformer_runtime([profiles["hybrid_tail_preserve"]]))
        self.assertTrue(module.profiles_require_bs_roformer_runtime([profiles["bs_artifact_clean"]]))

    def test_parse_args_defaults_to_fine_segment_consistency(self):
        module = _load_adaptive_module()

        args = module.parse_args(["source.flac", "out"])

        self.assertEqual(args.segment_consistency_sec, 5.0)

    def test_reusable_manifest_rejects_candidate_mode_and_tuning_mismatch(self):
        module = _load_adaptive_module()
        profile = module.default_candidate_profiles()[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.flac"
            manifest_path = temp_path / "stem_manifest.json"
            vocals_path = temp_path / "vocals.flac"
            source_path.write_bytes(b"source")
            vocals_path.write_bytes(b"vocals")
            manifest = {
                "audio_path": str(source_path),
                "source_audio_sha256": module._sha256_file(source_path),
                "preview_window_sec": {"start": 0.0, "end": 12.0},
                "pipeline": {"candidate_mode": "wrong_mode"},
                "tuning": profile.tuning,
                "stems": {"vocals": {"path": str(vocals_path)}},
            }
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            rejected: list[dict[str, str]] = []

            self.assertIsNone(
                module.load_reusable_candidate_manifest(
                    manifest_path=manifest_path,
                    profile=profile,
                    source_audio_path=source_path,
                    start_sec=0.0,
                    end_sec=12.0,
                    reuse_existing_candidates=True,
                    reuse_rejected_profiles=rejected,
                )
            )
            self.assertEqual(rejected[0]["reason"], "candidate_mode_mismatch")

            manifest["pipeline"]["candidate_mode"] = profile.candidate_mode
            manifest["tuning"] = {**profile.tuning, "low_bs_weight": 0.01}
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            rejected.clear()

            self.assertIsNone(
                module.load_reusable_candidate_manifest(
                    manifest_path=manifest_path,
                    profile=profile,
                    source_audio_path=source_path,
                    start_sec=0.0,
                    end_sec=12.0,
                    reuse_existing_candidates=True,
                    reuse_rejected_profiles=rejected,
                )
            )

        self.assertEqual(rejected[0]["reason"], "profile_tuning_mismatch")

    def test_reusable_manifest_rejects_missing_source_audio_sha256(self):
        module = _load_adaptive_module()
        profile = module.default_candidate_profiles()[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.flac"
            manifest_path = temp_path / "stem_manifest.json"
            vocals_path = temp_path / "vocals.flac"
            source_path.write_bytes(b"source")
            vocals_path.write_bytes(b"vocals")
            manifest = {
                "audio_path": str(source_path),
                "preview_window_sec": {"start": 0.0, "end": 12.0},
                "pipeline": {"candidate_mode": profile.candidate_mode},
                "tuning": profile.tuning,
                "stems": {"vocals": {"path": str(vocals_path)}},
            }
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            rejected: list[dict[str, str]] = []

            self.assertIsNone(
                module.load_reusable_candidate_manifest(
                    manifest_path=manifest_path,
                    profile=profile,
                    source_audio_path=source_path,
                    start_sec=0.0,
                    end_sec=12.0,
                    reuse_existing_candidates=True,
                    reuse_rejected_profiles=rejected,
                )
            )

        self.assertEqual(rejected[0]["reason"], "missing_source_audio_sha256")

    def test_reusable_manifest_rejects_source_audio_sha256_mismatch(self):
        module = _load_adaptive_module()
        profile = module.default_candidate_profiles()[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.flac"
            manifest_path = temp_path / "stem_manifest.json"
            vocals_path = temp_path / "vocals.flac"
            source_path.write_bytes(b"source")
            vocals_path.write_bytes(b"vocals")
            manifest = {
                "audio_path": str(source_path),
                "source_audio_sha256": "0" * 64,
                "preview_window_sec": {"start": 0.0, "end": 12.0},
                "pipeline": {"candidate_mode": profile.candidate_mode},
                "tuning": profile.tuning,
                "stems": {"vocals": {"path": str(vocals_path)}},
            }
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            rejected: list[dict[str, str]] = []

            self.assertIsNone(
                module.load_reusable_candidate_manifest(
                    manifest_path=manifest_path,
                    profile=profile,
                    source_audio_path=source_path,
                    start_sec=0.0,
                    end_sec=12.0,
                    reuse_existing_candidates=True,
                    reuse_rejected_profiles=rejected,
                )
            )

        self.assertEqual(rejected[0]["reason"], "source_audio_sha256_mismatch")

    def test_reusable_manifest_rejects_dependency_fingerprint_mismatch(self):
        module = _load_adaptive_module()
        profile = module.default_segment_router_profiles()[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.flac"
            manifest_path = temp_path / "stem_manifest.json"
            vocals_path = temp_path / "vocals.flac"
            source_path.write_bytes(b"source")
            vocals_path.write_bytes(b"vocals")
            manifest = {
                "audio_path": str(source_path),
                "source_audio_sha256": module._sha256_file(source_path),
                "preview_window_sec": {"start": 0.0, "end": 12.0},
                "pipeline": {
                    "candidate_mode": "segment_adaptive_routed_vocals",
                    "dependency_fingerprints": {"clean": {"manifest_sha256": "old"}},
                },
                "tuning": profile.tuning,
                "stems": {"vocals": {"path": str(vocals_path)}},
            }
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            rejected: list[dict[str, str]] = []

            self.assertIsNone(
                module.load_reusable_candidate_manifest(
                    manifest_path=manifest_path,
                    profile=profile,
                    source_audio_path=source_path,
                    start_sec=0.0,
                    end_sec=12.0,
                    reuse_existing_candidates=True,
                    reuse_rejected_profiles=rejected,
                    expected_dependency_fingerprints={"clean": {"manifest_sha256": "new"}},
                )
            )

        self.assertEqual(rejected[0]["reason"], "dependency_fingerprint_mismatch")

    def test_reusable_manifest_rejects_vocals_path_outside_candidate_dir(self):
        module = _load_adaptive_module()
        profile = module.default_candidate_profiles()[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.flac"
            current_candidate_dir = temp_path / "current" / "hybrid_tail_preserve"
            stale_candidate_dir = temp_path / "stale" / "hybrid_tail_preserve"
            manifest_path = current_candidate_dir / "stem_manifest.json"
            stale_vocals_path = stale_candidate_dir / "vocals.flac"
            current_candidate_dir.mkdir(parents=True)
            stale_candidate_dir.mkdir(parents=True)
            source_path.write_bytes(b"source")
            stale_vocals_path.write_bytes(b"stale vocals")
            manifest = {
                "audio_path": str(source_path),
                "source_audio_sha256": module._sha256_file(source_path),
                "preview_window_sec": {"start": 0.0, "end": 12.0},
                "pipeline": {"candidate_mode": profile.candidate_mode},
                "tuning": profile.tuning,
                "stems": {"vocals": {"path": str(stale_vocals_path)}},
            }
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            rejected: list[dict[str, str]] = []

            self.assertIsNone(
                module.load_reusable_candidate_manifest(
                    manifest_path=manifest_path,
                    profile=profile,
                    source_audio_path=source_path,
                    start_sec=0.0,
                    end_sec=12.0,
                    reuse_existing_candidates=True,
                    reuse_rejected_profiles=rejected,
                )
            )

        self.assertEqual(rejected[0]["reason"], "vocals_path_outside_candidate_dir")

    def test_dependency_fingerprints_are_written_to_derived_manifest(self):
        module = _load_adaptive_module()
        profile = module.default_anchor_suppression_profiles()[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_dir = temp_path / "source"
            anchor_dir = temp_path / "anchor"
            output_dir = temp_path / "derived"
            _write_fake_candidate(source_dir, "ft_tail_safe")
            _write_fake_candidate(anchor_dir, "bs_artifact_clean")
            source_manifest_path = source_dir / "stem_manifest.json"
            anchor_manifest_path = anchor_dir / "stem_manifest.json"
            source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            anchor_manifest = json.loads(anchor_manifest_path.read_text(encoding="utf-8"))
            source_record = module.VocalCandidateRecord(
                profile=module.default_candidate_profiles()[2],
                manifest_path=source_manifest_path,
                output_dir=source_dir,
                vocals_path=source_dir / "vocals.flac",
                metrics={},
                manifest=source_manifest,
            )
            anchor_record = module.VocalCandidateRecord(
                profile=module.default_candidate_profiles()[3],
                manifest_path=anchor_manifest_path,
                output_dir=anchor_dir,
                vocals_path=anchor_dir / "vocals.flac",
                metrics={},
                manifest=anchor_manifest,
            )

            def fake_suppression(*, output_vocals_path: Path, **_kwargs):
                output_vocals_path.parent.mkdir(parents=True, exist_ok=True)
                output_vocals_path.write_bytes(b"derived vocals")
                return {"sample_rate": 44100, "channels": 2, "frames": 1, "duration_sec": 0.0, "rms": 0.0, "peak": 0.0}

            with patch.object(module, "apply_anchor_guided_leakage_suppression", side_effect=fake_suppression):
                manifest = module.build_anchor_suppressed_candidate(
                    profile=profile,
                    source_record=source_record,
                    anchor_record=anchor_record,
                    output_dir=output_dir,
                )
            expected_source_manifest_sha256 = module._sha256_file(source_manifest_path)
            expected_anchor_vocals_sha256 = module._sha256_file(anchor_dir / "vocals.flac")

        fingerprints = manifest["pipeline"]["dependency_fingerprints"]
        self.assertEqual(
            fingerprints[profile.source_profile_name]["manifest_sha256"],
            expected_source_manifest_sha256,
        )
        self.assertEqual(
            fingerprints[profile.anchor_profile_name]["vocals_sha256"],
            expected_anchor_vocals_sha256,
        )

    def test_cli_runs_candidates_and_writes_adaptive_manifest(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            export_script = temp_path / "export_hq_vocal_preview.py"
            output_dir = temp_path / "adaptive"
            listen_dir = temp_path / "listen"
            source_path.write_bytes(b"source")
            export_script.write_text("# fake export script\n", encoding="utf-8")
            commands: list[list[str]] = []

            def fake_run_command(command: list[str]) -> None:
                commands.append(command)
                output_arg = command[command.index("--output-dir") + 1]
                output_path = Path(output_arg)
                _write_fake_candidate(output_path, output_path.name)

            def fake_metrics(_source_audio_path: Path, vocals_path: Path, **_kwargs) -> dict[str, float]:
                if vocals_path.parent.name == "hybrid_tail_preserve":
                    return _metrics(
                        dropout=0.05,
                        tail_dropout=0.06,
                        rms_ratio_db=-14.0,
                        low_ratio_db=-20.0,
                        kick_ratio_db=-14.0,
                    )
                if vocals_path.parent.name == "hybrid_balanced_clean":
                    return _metrics(
                        dropout=0.12,
                        tail_dropout=0.16,
                        rms_ratio_db=-18.0,
                        low_ratio_db=-24.0,
                        kick_ratio_db=-17.0,
                    )
                return _metrics(
                    dropout=0.30,
                    tail_dropout=0.42,
                    rms_ratio_db=-26.0,
                    low_ratio_db=-28.0,
                    kick_ratio_db=-18.0,
                )

            argv = [
                str(source_path),
                str(output_dir),
                "--listen-dir",
                str(listen_dir),
                "--export-script",
                str(export_script),
                "--python-executable",
                sys.executable,
                "--end-sec",
                "12.0",
                "--segment-consistency-sec",
                "0",
                "--disable-anchor-suppression",
            ]

            with patch.object(module, "ensure_bs_roformer_runtime"), patch.object(
                module,
                "_run_command",
                side_effect=fake_run_command,
            ), patch.object(
                module,
                "calculate_vocal_proxy_metrics",
                side_effect=fake_metrics,
            ), patch.object(
                module,
                "calculate_anchor_support_metrics",
                return_value={
                    "unsupported_vs_anchor_ratio_db": -40.0,
                    "high_unsupported_vs_anchor_ratio_db": -40.0,
                    "high_noise_flatness_6000_16000": 0.05,
                    "stereo_side_mid_ratio_db": -16.0,
                },
            ), patch.object(
                module,
                "write_listen_pack",
                return_value={"selected_vocals_flac": str(listen_dir / "01_adaptive_selected_vocals.flac")},
            ):
                self.assertEqual(module.main(argv), 0)

            adaptive_manifest = json.loads(
                (output_dir / "adaptive_vocal_manifest.json").read_text(encoding="utf-8"),
            )
            selected_exists = (output_dir / "01_adaptive_selected_vocals.flac").exists()

        self.assertEqual(len(commands), 4)
        self.assertEqual(adaptive_manifest["selected_profile"], "hybrid_tail_preserve")
        self.assertTrue(selected_exists)
        self.assertEqual(adaptive_manifest["learning_contract"]["type"], "feedback_weighted_vocal_profile_selection")
        self.assertEqual(
            adaptive_manifest["learning_contract"]["feedback_record_schema"]["backing_vocal_loss"],
            "optional boolean",
        )
        self.assertEqual(
            adaptive_manifest["learning_contract"]["feedback_record_schema"]["listener_verified"],
            "optional boolean; requires source_audio_sha256 and window_sec",
        )
        self.assertIn(
            "It can build a no-vocal leakage cleanup candidate from consensus no-vocal windows without fixed timestamps.",
            adaptive_manifest["learning_contract"]["limits"],
        )
        self.assertEqual(
            [candidate["profile_name"] for candidate in adaptive_manifest["candidates"]][0],
            "hybrid_tail_preserve",
        )

    def test_cli_feedback_store_changes_profile_ranking(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            export_script = temp_path / "export_hq_vocal_preview.py"
            feedback_store = temp_path / "feedback.jsonl"
            source_path.write_bytes(b"source")
            export_script.write_text("# fake export script\n", encoding="utf-8")
            feedback_store.write_text(
                json.dumps({"profile": "hybrid_tail_preserve", "rating": "reject", "backing_vocal_loss": True})
                + "\n",
                encoding="utf-8",
            )

            def fake_run_command(command: list[str]) -> None:
                output_arg = command[command.index("--output-dir") + 1]
                output_path = Path(output_arg)
                _write_fake_candidate(output_path, output_path.name)

            def fake_metrics(_source_audio_path: Path, vocals_path: Path, **_kwargs) -> dict[str, float]:
                if vocals_path.parent.name == "hybrid_tail_preserve":
                    return _metrics(
                        dropout=0.05,
                        tail_dropout=0.06,
                        rms_ratio_db=-14.0,
                        low_ratio_db=-22.0,
                        kick_ratio_db=-16.0,
                    )
                return _metrics(
                    dropout=0.07,
                    tail_dropout=0.08,
                    rms_ratio_db=-14.0,
                    low_ratio_db=-22.0,
                    kick_ratio_db=-16.0,
                )

            def run_controller(output_name: str, use_feedback: bool) -> dict[str, object]:
                argv = [
                    str(source_path),
                    str(temp_path / output_name),
                    "--profile",
                    "hybrid_tail_preserve",
                    "--profile",
                    "hybrid_balanced_clean",
                    "--export-script",
                    str(export_script),
                    "--python-executable",
                    sys.executable,
                    "--end-sec",
                    "12.0",
                    "--segment-consistency-sec",
                    "0",
                    "--disable-anchor-suppression",
                    "--disable-body-repair",
                    "--disable-segment-router",
                ]
                if use_feedback:
                    argv.extend(["--feedback-store", str(feedback_store)])

                with patch.object(module, "ensure_bs_roformer_runtime"), patch.object(
                    module,
                    "_run_command",
                    side_effect=fake_run_command,
                ), patch.object(
                    module,
                    "calculate_vocal_proxy_metrics",
                    side_effect=fake_metrics,
                ):
                    self.assertEqual(module.main(argv), 0)

                return json.loads(
                    (temp_path / output_name / "adaptive_vocal_manifest.json").read_text(encoding="utf-8"),
                )

            baseline_manifest = run_controller("without_feedback", use_feedback=False)
            feedback_manifest = run_controller("with_feedback", use_feedback=True)

        self.assertEqual(baseline_manifest["selected_profile"], "hybrid_tail_preserve")
        self.assertEqual(feedback_manifest["selected_profile"], "hybrid_balanced_clean")
        self.assertLess(feedback_manifest["feedback_adjustments"]["hybrid_tail_preserve"], 0.0)

    def test_cli_keeps_blocked_segment_issue_clips_diagnostic_only(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            export_script = temp_path / "export_hq_vocal_preview.py"
            output_dir = temp_path / "adaptive"
            listen_dir = temp_path / "listen"
            source_path.write_bytes(b"source")
            export_script.write_text("# fake export script\n", encoding="utf-8")

            def fake_run_command(command: list[str]) -> None:
                output_arg = command[command.index("--output-dir") + 1]
                output_path = Path(output_arg)
                _write_fake_candidate(output_path, output_path.name)

            def fake_metrics(_source_audio_path: Path, vocals_path: Path, **_kwargs) -> dict[str, float]:
                if vocals_path.parent.name == "hybrid_tail_preserve":
                    return _metrics(
                        dropout=0.05,
                        tail_dropout=0.06,
                        rms_ratio_db=-14.0,
                        low_ratio_db=-23.0,
                        kick_ratio_db=-16.0,
                    )
                return _metrics(
                    dropout=0.20,
                    tail_dropout=0.24,
                    rms_ratio_db=-19.0,
                    low_ratio_db=-22.0,
                    kick_ratio_db=-15.0,
                )

            issue_report = {
                "schema_version": 1,
                "mode": "vocal_segment_issue_report",
                "summary": {
                    "max_issue_score": 9.2,
                    "mean_top3_issue_score": 5.1,
                    "issue_window_ratio": 1.0,
                    "dominant_issue_labels": [{"label": "kick_punch_bleed", "count": 1}],
                },
                "summary_metrics": {
                    "segment_issue_max_score": 9.2,
                    "segment_issue_mean_top3_score": 5.1,
                    "segment_issue_count": 1.0,
                    "segment_issue_window_ratio": 1.0,
                },
                "segments": [
                    {
                        "start_sec": 2.0,
                        "end_sec": 4.0,
                        "relative_start_sec": 2.0,
                        "relative_end_sec": 4.0,
                        "severity_score": 9.2,
                        "labels": ["kick_punch_bleed", "vocal_dropout"],
                    }
                ],
                "top_segments": [
                    {
                        "start_sec": 2.0,
                        "end_sec": 4.0,
                        "relative_start_sec": 2.0,
                        "relative_end_sec": 4.0,
                        "severity_score": 9.2,
                        "labels": ["kick_punch_bleed", "vocal_dropout"],
                    }
                ],
            }

            argv = [
                str(source_path),
                str(output_dir),
                "--listen-dir",
                str(listen_dir),
                "--export-script",
                str(export_script),
                "--python-executable",
                sys.executable,
                "--end-sec",
                "12.0",
                "--segment-consistency-sec",
                "3.0",
                "--segment-issue-report-max-windows",
                "2",
                "--segment-issue-listen-clip-count",
                "1",
                "--disable-segment-issue-repair",
                "--disable-anchor-suppression",
            ]

            with patch.object(module, "ensure_bs_roformer_runtime"), patch.object(
                module,
                "_run_command",
                side_effect=fake_run_command,
            ), patch.object(
                module,
                "calculate_vocal_proxy_metrics",
                side_effect=fake_metrics,
            ), patch.object(
                module,
                "calculate_vocal_proxy_consistency_metrics",
                return_value={
                    "segment_consistency_worst_dropout_frame_ratio": 0.12,
                    "segment_consistency_worst_tail_dropout_ratio": 0.14,
                    "segment_consistency_dropout_std": 0.02,
                    "segment_consistency_tail_dropout_std": 0.02,
                    "segment_consistency_rms_ratio_range_db": 2.0,
                },
            ), patch.object(
                module,
                "calculate_vocal_segment_issue_report",
                return_value=issue_report,
            ), patch.object(
                module,
                "calculate_anchor_support_metrics",
                return_value={
                    "unsupported_vs_anchor_ratio_db": -40.0,
                    "high_unsupported_vs_anchor_ratio_db": -40.0,
                    "high_noise_flatness_6000_16000": 0.05,
                    "stereo_side_mid_ratio_db": -16.0,
                },
            ), patch.object(
                module,
                "write_listen_pack",
                return_value={"selected_vocals_flac": str(listen_dir / "01_adaptive_selected_vocals.flac")},
            ) as listen_pack_mock, patch.object(
                module,
                "write_segment_issue_listen_clips",
                return_value={
                    "segment_issue_clips_manifest": str(listen_dir / "segment_issues" / "segment_issue_clips.json"),
                    "segment_issue_clips_dir": str(listen_dir / "segment_issues"),
                },
            ):
                self.assertEqual(module.main(argv), 0)

            adaptive_manifest = json.loads(
                (output_dir / "adaptive_vocal_manifest.json").read_text(encoding="utf-8"),
            )
            selected_delivery_gate_path = Path(
                adaptive_manifest["selected_artifacts"]["selected_delivery_gate_manifest"],
            )
            self.assertTrue(selected_delivery_gate_path.exists())
            selected_delivery_gate = json.loads(
                selected_delivery_gate_path.read_text(encoding="utf-8"),
            )

        listen_pack_mock.assert_not_called()
        self.assertEqual(adaptive_manifest["selected_profile"], "hybrid_tail_preserve")
        self.assertEqual(adaptive_manifest["selected_segment_issue_report"], issue_report)
        self.assertEqual(adaptive_manifest["stage_profiler"]["mode"], "adaptive_vocal_stage_profiler")
        self.assertIn("stages", adaptive_manifest["stage_profiler"])
        self.assertIn("audio_reads", adaptive_manifest["analysis_cache"])
        self.assertEqual(
            adaptive_manifest["selected_artifacts"]["listen_selected_vocals_flac"],
            None,
        )
        self.assertEqual(
            adaptive_manifest["selected_artifacts"]["listen_segment_issue_clips_manifest"],
            None,
        )
        self.assertEqual(
            adaptive_manifest["selected_artifacts"]["diagnostic_segment_issue_clips_manifest"],
            str(listen_dir / "segment_issues" / "segment_issue_clips.json"),
        )
        self.assertFalse(adaptive_manifest["selected_artifacts"]["delivery_allowed"])
        self.assertEqual(
            adaptive_manifest["selected_artifacts"]["presentation_status"],
            "blocked_from_listen_pack",
        )
        self.assertEqual(
            adaptive_manifest["selected_artifacts"]["diagnostic_vocals_flac"],
            adaptive_manifest["selected_artifacts"]["selected_vocals_flac"],
        )
        self.assertEqual(selected_delivery_gate["selected_profile"], adaptive_manifest["selected_profile"])
        self.assertEqual(
            selected_delivery_gate["selected_vocals_flac"],
            adaptive_manifest["selected_artifacts"]["selected_vocals_flac"],
        )
        self.assertFalse(selected_delivery_gate["delivery_allowed"])
        self.assertEqual(selected_delivery_gate["presentation_status"], "blocked_from_listen_pack")
        self.assertEqual(selected_delivery_gate["release_status"], "diagnostic_only")
        self.assertEqual(
            selected_delivery_gate["diagnostic_segment_issue_clips_manifest"],
            str(listen_dir / "segment_issues" / "segment_issue_clips.json"),
        )
        self.assertIsNone(selected_delivery_gate["listen_segment_issue_clips_manifest"])
        self.assertEqual(selected_delivery_gate["defect_effectiveness_gate"]["status"], "blocked_for_listening")
        self.assertEqual(
            selected_delivery_gate["vocal_next_action_plan"]["next_action"],
            "run_bounded_defect_specific_repair_or_model_ladder",
        )
        self.assertEqual(
            adaptive_manifest["decision_contract"]["mode"],
            "adaptive_decision_contract",
        )
        self.assertEqual(
            adaptive_manifest["decision_contract"]["dominant_failure_class"],
            "vocal_dropout",
        )
        self.assertEqual(
            selected_delivery_gate["decision_contract"],
            adaptive_manifest["decision_contract"],
        )
        self.assertEqual(
            adaptive_manifest["candidates"][0]["segment_issue_report"],
            issue_report,
        )

    def test_cli_keeps_release_ready_segment_issue_clips_diagnostic_only(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            export_script = temp_path / "export_hq_vocal_preview.py"
            output_dir = temp_path / "adaptive"
            listen_dir = temp_path / "listen"
            source_path.write_bytes(b"source")
            export_script.write_text("# fake export script\n", encoding="utf-8")

            def fake_run_command(command: list[str]) -> None:
                output_arg = command[command.index("--output-dir") + 1]
                output_path = Path(output_arg)
                _write_fake_candidate(output_path, output_path.name)

            issue_report = {
                "schema_version": 1,
                "mode": "vocal_segment_issue_report",
                "summary": {
                    "max_issue_score": 0.05,
                    "mean_top3_issue_score": 0.05,
                    "issue_window_ratio": 1.0,
                    "dominant_issue_labels": [{"label": "acceptable_proxy_window", "count": 1}],
                },
                "summary_metrics": {
                    "segment_issue_max_score": 0.05,
                    "segment_issue_mean_top3_score": 0.05,
                    "segment_issue_count": 1.0,
                    "segment_issue_window_ratio": 1.0,
                },
                "segments": [
                    {
                        "start_sec": 2.0,
                        "end_sec": 4.0,
                        "relative_start_sec": 2.0,
                        "relative_end_sec": 4.0,
                        "severity_score": 0.05,
                        "labels": ["acceptable_proxy_window"],
                    }
                ],
                "top_segments": [
                    {
                        "start_sec": 2.0,
                        "end_sec": 4.0,
                        "relative_start_sec": 2.0,
                        "relative_end_sec": 4.0,
                        "severity_score": 0.05,
                        "labels": ["acceptable_proxy_window"],
                    }
                ],
            }

            argv = [
                str(source_path),
                str(output_dir),
                "--listen-dir",
                str(listen_dir),
                "--export-script",
                str(export_script),
                "--python-executable",
                sys.executable,
                "--end-sec",
                "12.0",
                "--segment-consistency-sec",
                "3.0",
                "--segment-issue-report-max-windows",
                "2",
                "--segment-issue-listen-clip-count",
                "1",
                "--disable-segment-issue-repair",
                "--disable-anchor-suppression",
            ]

            with patch.object(module, "ensure_bs_roformer_runtime"), patch.object(
                module,
                "_run_command",
                side_effect=fake_run_command,
            ), patch.object(
                module,
                "calculate_vocal_proxy_metrics",
                return_value=_metrics(
                    dropout=0.05,
                    tail_dropout=0.06,
                    rms_ratio_db=-14.0,
                    low_ratio_db=-23.0,
                    kick_ratio_db=-16.0,
                ),
            ), patch.object(
                module,
                "calculate_vocal_proxy_consistency_metrics",
                return_value={
                    "segment_consistency_worst_dropout_frame_ratio": 0.12,
                    "segment_consistency_worst_tail_dropout_ratio": 0.14,
                    "segment_consistency_dropout_std": 0.02,
                    "segment_consistency_tail_dropout_std": 0.02,
                    "segment_consistency_rms_ratio_range_db": 2.0,
                },
            ), patch.object(
                module,
                "calculate_vocal_segment_issue_report",
                return_value=issue_report,
            ), patch.object(
                module,
                "calculate_anchor_support_metrics",
                return_value={
                    "unsupported_vs_anchor_ratio_db": -40.0,
                    "high_unsupported_vs_anchor_ratio_db": -40.0,
                    "high_noise_flatness_6000_16000": 0.05,
                    "stereo_side_mid_ratio_db": -16.0,
                },
            ), patch.object(
                module,
                "write_listen_pack",
                return_value={"selected_vocals_flac": str(listen_dir / "01_adaptive_selected_vocals.flac")},
            ) as listen_pack_mock, patch.object(
                module,
                "write_segment_issue_listen_clips",
                return_value={
                    "segment_issue_clips_manifest": str(listen_dir / "segment_issues" / "segment_issue_clips.json"),
                    "segment_issue_clips_dir": str(listen_dir / "segment_issues"),
                },
            ):
                self.assertEqual(module.main(argv), 0)

            adaptive_manifest = json.loads(
                (output_dir / "adaptive_vocal_manifest.json").read_text(encoding="utf-8"),
            )
            selected_delivery_gate = json.loads(
                Path(adaptive_manifest["selected_artifacts"]["selected_delivery_gate_manifest"]).read_text(
                    encoding="utf-8",
                ),
            )

        listen_pack_mock.assert_called_once()
        self.assertTrue(adaptive_manifest["selected_artifacts"]["delivery_allowed"])
        self.assertEqual(
            adaptive_manifest["selected_artifacts"]["presentation_status"],
            "accepted_for_listen_pack",
        )
        self.assertIsNone(adaptive_manifest["selected_artifacts"]["listen_segment_issue_clips_manifest"])
        self.assertEqual(
            adaptive_manifest["selected_artifacts"]["diagnostic_segment_issue_clips_manifest"],
            str(listen_dir / "segment_issues" / "segment_issue_clips.json"),
        )
        self.assertIsNone(selected_delivery_gate["listen_segment_issue_clips_manifest"])
        self.assertEqual(
            selected_delivery_gate["diagnostic_segment_issue_clips_manifest"],
            str(listen_dir / "segment_issues" / "segment_issue_clips.json"),
        )

    def test_cli_writes_real_segment_issue_report_and_clips_from_audio_candidates(self):
        module = _load_adaptive_module()
        sample_rate = 8000
        t = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
        vocal = (0.18 * np.sin(2.0 * np.pi * 330.0 * t)).astype(np.float32)
        source_audio = np.stack([vocal, vocal], axis=1).astype(np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.flac"
            export_script = temp_path / "export_hq_vocal_preview.py"
            output_dir = temp_path / "adaptive"
            listen_dir = temp_path / "listen"
            sf.write(source_path, source_audio, sample_rate, subtype="PCM_24")
            export_script.write_text("# fake export script\n", encoding="utf-8")

            def fake_run_command(command: list[str]) -> None:
                output_arg = command[command.index("--output-dir") + 1]
                candidate_dir = Path(output_arg)
                candidate_dir.mkdir(parents=True, exist_ok=True)
                candidate_audio = source_audio.copy()
                if candidate_dir.name == "hybrid_balanced_clean":
                    candidate_audio[sample_rate : sample_rate * 2] = 0.0
                vocals_path = candidate_dir / "vocals.flac"
                sf.write(vocals_path, candidate_audio, sample_rate, subtype="PCM_24")
                manifest = {
                    "schema_version": 1,
                    "audio_path": str(source_path),
                    "preview_window_sec": {"start": 0.0, "end": 2.0},
                    "pipeline": {"candidate_mode": "hybrid_vocals"},
                    "stems": {"vocals": {"path": str(vocals_path)}},
                }
                if candidate_dir.name == "ft_tail_safe":
                    manifest["pipeline"]["candidate_mode"] = "ft_vocals"
                (candidate_dir / "stem_manifest.json").write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

            argv = [
                str(source_path),
                str(output_dir),
                "--listen-dir",
                str(listen_dir),
                "--export-script",
                str(export_script),
                "--python-executable",
                sys.executable,
                "--end-sec",
                "2.0",
                "--segment-consistency-sec",
                "1.0",
                "--segment-issue-report-max-windows",
                "2",
                "--segment-issue-listen-clip-count",
                "1",
                "--disable-anchor-suppression",
            ]

            with patch.object(module, "ensure_bs_roformer_runtime"), patch.object(
                module,
                "_run_command",
                side_effect=fake_run_command,
            ):
                self.assertEqual(module.main(argv), 0)

            adaptive_manifest = json.loads(
                (output_dir / "adaptive_vocal_manifest.json").read_text(encoding="utf-8"),
            )
            self.assertIsNone(adaptive_manifest["selected_artifacts"]["listen_segment_issue_clips_manifest"])
            clips_manifest_path = Path(
                adaptive_manifest["selected_artifacts"]["diagnostic_segment_issue_clips_manifest"],
            )
            clips_manifest = json.loads(clips_manifest_path.read_text(encoding="utf-8"))
            clips_manifest_exists = clips_manifest_path.exists()
            clip_count = len(clips_manifest["clips"])

        self.assertIsNotNone(adaptive_manifest["selected_segment_issue_report"])
        self.assertIn("summary_metrics", adaptive_manifest["selected_segment_issue_report"])
        self.assertFalse(clips_manifest["delivery_allowed"])
        self.assertEqual(clips_manifest["presentation_status"], "diagnostic_segment_issue_evidence_only")
        self.assertTrue(clips_manifest_exists)
        self.assertGreaterEqual(clip_count, 1)

    def test_cli_builds_segment_router_candidate_when_dependencies_exist(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            export_script = temp_path / "export_hq_vocal_preview.py"
            output_dir = temp_path / "adaptive"
            source_path.write_bytes(b"source")
            export_script.write_text("# fake export script\n", encoding="utf-8")
            commands: list[list[str]] = []

            def fake_run_command(command: list[str]) -> None:
                commands.append(command)
                output_arg = command[command.index("--output-dir") + 1]
                output_path = Path(output_arg)
                _write_fake_candidate(output_path, output_path.name)

            def fake_derived_candidate(*, profile, output_dir: Path, **_kwargs):
                _write_fake_candidate(output_dir, profile.name)
                manifest_path = output_dir / "stem_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["pipeline"]["candidate_mode"] = "derived_test_candidate"
                manifest["manifest_path"] = str(manifest_path)
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return manifest

            def fake_segment_candidate(*, profile, output_dir: Path, **_kwargs):
                _write_fake_candidate(output_dir, profile.name)
                manifest_path = output_dir / "stem_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["pipeline"]["candidate_mode"] = "segment_adaptive_routed_vocals"
                manifest["pipeline"]["vocal_postprocess"] = "segment_adaptive_vocal_router_v1"
                manifest["manifest_path"] = str(manifest_path)
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return manifest

            def fake_metrics(_source_audio_path: Path, vocals_path: Path, **_kwargs) -> dict[str, float]:
                if vocals_path.parent.name == "busy_vocal_supported_router":
                    return _metrics(
                        dropout=0.02,
                        tail_dropout=0.03,
                        rms_ratio_db=-14.0,
                        low_ratio_db=-23.0,
                        kick_ratio_db=-16.0,
                    )
                if vocals_path.parent.name == "segment_adaptive_vocal_router":
                    return _metrics(
                        dropout=0.03,
                        tail_dropout=0.04,
                        rms_ratio_db=-14.0,
                        low_ratio_db=-23.0,
                        kick_ratio_db=-16.0,
                    )
                return _metrics(
                    dropout=0.16,
                    tail_dropout=0.20,
                    rms_ratio_db=-17.0,
                    low_ratio_db=-23.0,
                    kick_ratio_db=-16.0,
                )

            argv = [
                str(source_path),
                str(output_dir),
                "--export-script",
                str(export_script),
                "--python-executable",
                sys.executable,
                "--end-sec",
                "12.0",
                "--segment-consistency-sec",
                "0",
            ]

            with patch.object(module, "ensure_bs_roformer_runtime"), patch.object(
                module,
                "_run_command",
                side_effect=fake_run_command,
            ), patch.object(
                module,
                "build_anchor_suppressed_candidate",
                side_effect=fake_derived_candidate,
            ), patch.object(
                module,
                "build_vocal_body_repair_candidate",
                side_effect=fake_derived_candidate,
            ), patch.object(
                module,
                "build_segment_router_candidate",
                side_effect=fake_segment_candidate,
            ) as segment_builder_mock, patch.object(
                module,
                "calculate_vocal_proxy_metrics",
                side_effect=fake_metrics,
            ), patch.object(
                module,
                "calculate_anchor_support_metrics",
                return_value={
                    "unsupported_vs_anchor_ratio_db": -40.0,
                    "high_unsupported_vs_anchor_ratio_db": -40.0,
                    "high_noise_flatness_6000_16000": 0.05,
                    "stereo_side_mid_ratio_db": -16.0,
                },
            ):
                self.assertEqual(module.main(argv), 0)

            adaptive_manifest = json.loads(
                (output_dir / "adaptive_vocal_manifest.json").read_text(encoding="utf-8"),
            )

        self.assertEqual(len(commands), 4)
        self.assertEqual(segment_builder_mock.call_count, 2)
        self.assertEqual(adaptive_manifest["selected_profile"], "busy_vocal_supported_router")
        self.assertEqual(
            adaptive_manifest["segment_router_candidate_profiles"][0]["candidate_mode"],
            "segment_adaptive_routed_vocals",
        )
        self.assertEqual(
            [profile["name"] for profile in adaptive_manifest["segment_router_candidate_profiles"]],
            [
                "segment_adaptive_vocal_router",
                "busy_vocal_supported_router",
                "ir_residual_supported_router",
                "quality_consistent_ir_router",
            ],
        )
        self.assertIn(
            "segment_adaptive_vocal_router",
            [candidate["profile_name"] for candidate in adaptive_manifest["candidates"]],
        )
        self.assertIn(
            "busy_vocal_supported_router",
            [candidate["profile_name"] for candidate in adaptive_manifest["candidates"]],
        )

    def test_cli_adds_issue_aware_segment_selection_candidate_to_ranking(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            export_script = temp_path / "export_hq_vocal_preview.py"
            output_dir = temp_path / "adaptive"
            source_path.write_bytes(b"source")
            export_script.write_text("# fake export script\n", encoding="utf-8")

            def fake_run_command(command: list[str]) -> None:
                output_arg = command[command.index("--output-dir") + 1]
                output_path = Path(output_arg)
                _write_fake_candidate(output_path, output_path.name)

            def fake_issue_aware_candidate(*, profile, output_dir: Path, **_kwargs):
                _write_fake_candidate(output_dir, profile.name)
                vocals_path = output_dir / "vocals.flac"
                manifest_path = output_dir / "stem_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["pipeline"]["candidate_mode"] = profile.candidate_mode
                manifest["pipeline"]["vocal_postprocess"] = "issue_aware_segment_selection_v1"
                manifest["manifest_path"] = str(manifest_path)
                manifest["stems"]["vocals"]["path"] = str(vocals_path)
                manifest["stems"]["vocals"]["vocal_postprocess"] = "issue_aware_segment_selection_v1"
                manifest["stems"]["vocals"]["replaced_window_count"] = 1
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return manifest

            def fake_metrics(_source_audio_path: Path, vocals_path: Path, **_kwargs) -> dict[str, float]:
                if vocals_path.parent.name == "issue_aware_segment_selection":
                    return _metrics(
                        dropout=0.01,
                        tail_dropout=0.02,
                        rms_ratio_db=-14.0,
                        low_ratio_db=-24.0,
                        kick_ratio_db=-16.0,
                    )
                return _metrics(
                    dropout=0.22,
                    tail_dropout=0.25,
                    rms_ratio_db=-17.0,
                    low_ratio_db=-23.0,
                    kick_ratio_db=-15.0,
                )

            issue_report = {
                "schema_version": 1,
                "mode": "vocal_segment_issue_report",
                "summary": {
                    "max_issue_score": 1.1,
                    "mean_top3_issue_score": 0.8,
                    "issue_segments_count": 1.0,
                    "issue_window_ratio": 0.25,
                    "dominant_issue_labels": [{"label": "vocal_dropout", "count": 1}],
                },
                "summary_metrics": {
                    "segment_issue_max_score": 1.1,
                    "segment_issue_mean_top3_score": 0.8,
                    "segment_issue_count": 1.0,
                    "segment_issue_window_ratio": 0.25,
                },
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 3.0,
                        "relative_start_sec": 0.0,
                        "relative_end_sec": 3.0,
                        "severity_score": 1.1,
                        "labels": ["vocal_dropout"],
                    }
                ],
                "top_segments": [],
            }

            argv = [
                str(source_path),
                str(output_dir),
                "--export-script",
                str(export_script),
                "--python-executable",
                sys.executable,
                "--end-sec",
                "12.0",
                "--segment-consistency-sec",
                "3.0",
                "--disable-anchor-suppression",
                "--disable-body-repair",
                "--disable-segment-router",
                "--disable-segment-issue-repair",
            ]

            with patch.object(module, "ensure_bs_roformer_runtime"), patch.object(
                module,
                "_run_command",
                side_effect=fake_run_command,
            ), patch.object(
                module,
                "calculate_vocal_proxy_metrics",
                side_effect=fake_metrics,
            ), patch.object(
                module,
                "calculate_vocal_proxy_consistency_metrics",
                return_value={
                    "segment_consistency_worst_dropout_frame_ratio": 0.12,
                    "segment_consistency_worst_tail_dropout_ratio": 0.14,
                    "segment_consistency_dropout_std": 0.02,
                    "segment_consistency_tail_dropout_std": 0.02,
                    "segment_consistency_rms_ratio_range_db": 2.0,
                },
            ), patch.object(
                module,
                "calculate_vocal_segment_issue_report",
                return_value=issue_report,
            ), patch.object(
                module,
                "calculate_anchor_support_metrics",
                return_value={
                    "unsupported_vs_anchor_ratio_db": -40.0,
                    "high_unsupported_vs_anchor_ratio_db": -40.0,
                    "high_noise_flatness_6000_16000": 0.05,
                    "stereo_side_mid_ratio_db": -16.0,
                },
            ), patch.object(
                module,
                "build_issue_aware_segment_selection_candidate",
                side_effect=fake_issue_aware_candidate,
            ) as issue_builder_mock:
                self.assertEqual(module.main(argv), 0)

            adaptive_manifest = json.loads(
                (output_dir / "adaptive_vocal_manifest.json").read_text(encoding="utf-8"),
            )

        issue_builder_mock.assert_called_once()
        self.assertEqual(adaptive_manifest["selected_profile"], "issue_aware_segment_selection")
        self.assertEqual(
            adaptive_manifest["issue_aware_segment_selection_profile"]["candidate_mode"],
            "issue_aware_segment_selected_vocals",
        )
        self.assertEqual(adaptive_manifest["skipped_issue_aware_segment_selection"], [])
        self.assertIn(
            "issue_aware_segment_selection",
            [candidate["profile_name"] for candidate in adaptive_manifest["candidates"]],
        )

    def test_cli_adds_segment_issue_repair_candidate_to_ranking(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            export_script = temp_path / "export_hq_vocal_preview.py"
            output_dir = temp_path / "adaptive"
            source_path.write_bytes(b"source")
            export_script.write_text("# fake export script\n", encoding="utf-8")

            def fake_run_command(command: list[str]) -> None:
                output_arg = command[command.index("--output-dir") + 1]
                output_path = Path(output_arg)
                _write_fake_candidate(output_path, output_path.name)

            def fake_segment_issue_repair_candidate(*, profile, output_dir: Path, **_kwargs):
                _write_fake_candidate(output_dir, profile.name)
                vocals_path = output_dir / "vocals.flac"
                manifest_path = output_dir / "stem_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["pipeline"]["candidate_mode"] = profile.candidate_mode
                manifest["pipeline"]["vocal_postprocess"] = "segment_issue_repair_v1"
                manifest["manifest_path"] = str(manifest_path)
                manifest["stems"]["vocals"]["path"] = str(vocals_path)
                manifest["stems"]["vocals"]["vocal_postprocess"] = "segment_issue_repair_v1"
                manifest["stems"]["vocals"]["repaired_window_count"] = 1
                manifest["stems"]["vocals"]["base_segment_issue_max_score"] = 1.2
                manifest["stems"]["vocals"]["base_segment_issue_mean_top3_score"] = 0.9
                manifest["stems"]["vocals"]["window_decisions"] = [
                    {
                        "segment_index": 0,
                        "repair_profile": "hybrid_tail_preserve",
                        "recovery_improvement": 0.35,
                    }
                ]
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return manifest

            def fake_metrics(_source_audio_path: Path, vocals_path: Path, **_kwargs) -> dict[str, float]:
                if vocals_path.parent.name == "segment_issue_repair":
                    return _metrics(
                        dropout=0.01,
                        tail_dropout=0.02,
                        rms_ratio_db=-14.0,
                        low_ratio_db=-24.0,
                        kick_ratio_db=-16.0,
                    )
                return _metrics(
                    dropout=0.24,
                    tail_dropout=0.28,
                    rms_ratio_db=-17.0,
                    low_ratio_db=-23.0,
                    kick_ratio_db=-15.0,
                )

            issue_report = {
                "schema_version": 1,
                "mode": "vocal_segment_issue_report",
                "summary": {
                    "max_issue_score": 1.2,
                    "mean_top3_issue_score": 0.9,
                    "issue_segments_count": 1.0,
                    "issue_window_ratio": 0.25,
                    "dominant_issue_labels": [{"label": "vocal_dropout", "count": 1}],
                },
                "summary_metrics": {
                    "segment_issue_max_score": 1.2,
                    "segment_issue_mean_top3_score": 0.9,
                    "segment_issue_count": 1.0,
                    "segment_issue_window_ratio": 0.25,
                },
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 3.0,
                        "relative_start_sec": 0.0,
                        "relative_end_sec": 3.0,
                        "severity_score": 1.2,
                        "labels": ["vocal_dropout"],
                        "severity_parts": {"vocal_dropout": 1.2},
                    }
                ],
                "top_segments": [],
            }

            argv = [
                str(source_path),
                str(output_dir),
                "--export-script",
                str(export_script),
                "--python-executable",
                sys.executable,
                "--end-sec",
                "12.0",
                "--segment-consistency-sec",
                "3.0",
                "--disable-anchor-suppression",
                "--disable-body-repair",
                "--disable-segment-router",
                "--disable-issue-aware-segment-selection",
            ]

            with patch.object(module, "ensure_bs_roformer_runtime"), patch.object(
                module,
                "_run_command",
                side_effect=fake_run_command,
            ), patch.object(
                module,
                "calculate_vocal_proxy_metrics",
                side_effect=fake_metrics,
            ), patch.object(
                module,
                "calculate_vocal_proxy_consistency_metrics",
                return_value={
                    "segment_consistency_worst_dropout_frame_ratio": 0.12,
                    "segment_consistency_worst_tail_dropout_ratio": 0.14,
                    "segment_consistency_dropout_std": 0.02,
                    "segment_consistency_tail_dropout_std": 0.02,
                    "segment_consistency_rms_ratio_range_db": 2.0,
                },
            ), patch.object(
                module,
                "calculate_vocal_segment_issue_report",
                return_value=issue_report,
            ), patch.object(
                module,
                "calculate_anchor_support_metrics",
                return_value={
                    "unsupported_vs_anchor_ratio_db": -40.0,
                    "high_unsupported_vs_anchor_ratio_db": -40.0,
                    "high_noise_flatness_6000_16000": 0.05,
                    "stereo_side_mid_ratio_db": -16.0,
                },
            ), patch.object(
                module,
                "build_segment_issue_repair_candidate",
                side_effect=fake_segment_issue_repair_candidate,
            ) as repair_builder_mock:
                self.assertEqual(module.main(argv), 0)

            adaptive_manifest = json.loads(
                (output_dir / "adaptive_vocal_manifest.json").read_text(encoding="utf-8"),
            )

        self.assertEqual(repair_builder_mock.call_count, 2)
        self.assertEqual(adaptive_manifest["selected_profile"], "segment_issue_repair")
        self.assertEqual(
            adaptive_manifest["segment_issue_repair_profile"]["candidate_mode"],
            "segment_issue_repaired_vocals",
        )
        self.assertEqual(adaptive_manifest["skipped_segment_issue_repair"], [])
        self.assertEqual(adaptive_manifest["skipped_post_gate_residual_issue_repair"], [])
        self.assertIn(
            "segment_issue_repair",
            [candidate["profile_name"] for candidate in adaptive_manifest["candidates"]],
        )
        self.assertIn(
            "post_gate_residual_issue_repair",
            [candidate["profile_name"] for candidate in adaptive_manifest["candidates"]],
        )

    def test_cli_builds_post_gate_repair_when_selected_candidate_remains_blocked(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            export_script = temp_path / "export_hq_vocal_preview.py"
            output_dir = temp_path / "adaptive"
            source_path.write_bytes(b"source")
            export_script.write_text("# fake export script\n", encoding="utf-8")

            def fake_run_command(command: list[str]) -> None:
                output_arg = command[command.index("--output-dir") + 1]
                output_path = Path(output_arg)
                _write_fake_candidate(output_path, output_path.name)

            def fake_segment_issue_repair_candidate(*, profile, output_dir: Path, **_kwargs):
                _write_fake_candidate(output_dir, profile.name)
                vocals_path = output_dir / "vocals.flac"
                manifest_path = output_dir / "stem_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["pipeline"]["candidate_mode"] = profile.candidate_mode
                manifest["pipeline"]["vocal_postprocess"] = "segment_issue_repair_v1"
                manifest["manifest_path"] = str(manifest_path)
                manifest["stems"]["vocals"]["path"] = str(vocals_path)
                manifest["stems"]["vocals"]["vocal_postprocess"] = "segment_issue_repair_v1"
                manifest["stems"]["vocals"]["repaired_window_count"] = 1
                manifest["stems"]["vocals"]["base_segment_issue_max_score"] = 1.4
                manifest["stems"]["vocals"]["base_segment_issue_mean_top3_score"] = 1.1
                manifest["stems"]["vocals"]["window_decisions"] = [
                    {
                        "segment_index": 0,
                        "base_profile": "segment_issue_repair",
                        "repair_profile": "hybrid_tail_preserve",
                        "recovery_improvement": 0.22,
                    }
                ]
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return manifest

            def fake_metrics(_source_audio_path: Path, vocals_path: Path, **_kwargs) -> dict[str, float]:
                if vocals_path.parent.name == "post_gate_residual_issue_repair":
                    return _metrics(
                        dropout=0.02,
                        tail_dropout=0.03,
                        rms_ratio_db=-14.0,
                        low_ratio_db=-24.0,
                        kick_ratio_db=-16.0,
                    )
                if vocals_path.parent.name == "segment_issue_repair":
                    return _metrics(
                        dropout=0.18,
                        tail_dropout=0.21,
                        rms_ratio_db=-17.0,
                        low_ratio_db=-23.0,
                        kick_ratio_db=-15.0,
                    )
                return _metrics(
                    dropout=0.08,
                    tail_dropout=0.12,
                    rms_ratio_db=-17.5,
                    low_ratio_db=-22.5,
                    kick_ratio_db=-14.5,
                )

            issue_report = {
                "schema_version": 1,
                "mode": "vocal_segment_issue_report",
                "summary": {
                    "max_issue_score": 1.4,
                    "mean_top3_issue_score": 1.1,
                    "issue_segments_count": 1.0,
                    "issue_window_ratio": 0.25,
                    "dominant_issue_labels": [{"label": "vocal_dropout", "count": 1}],
                },
                "summary_metrics": {
                    "segment_issue_max_score": 1.4,
                    "segment_issue_mean_top3_score": 1.1,
                    "segment_issue_count": 1.0,
                    "segment_issue_window_ratio": 0.25,
                },
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 3.0,
                        "relative_start_sec": 0.0,
                        "relative_end_sec": 3.0,
                        "severity_score": 1.4,
                        "labels": ["vocal_dropout", "vocal_tail_loss"],
                        "severity_parts": {"vocal_dropout": 1.0, "vocal_tail_loss": 0.4},
                    }
                ],
                "top_segments": [],
            }

            argv = [
                str(source_path),
                str(output_dir),
                "--export-script",
                str(export_script),
                "--python-executable",
                sys.executable,
                "--end-sec",
                "12.0",
                "--segment-consistency-sec",
                "3.0",
                "--disable-anchor-suppression",
                "--disable-body-repair",
                "--disable-segment-router",
                "--disable-issue-aware-segment-selection",
                "--disable-no-vocal-leakage-cleanup",
                "--disable-vocal-harshness-guard",
            ]

            with patch.object(module, "ensure_bs_roformer_runtime"), patch.object(
                module,
                "_run_command",
                side_effect=fake_run_command,
            ), patch.object(
                module,
                "calculate_vocal_proxy_metrics",
                side_effect=fake_metrics,
            ), patch.object(
                module,
                "calculate_vocal_proxy_consistency_metrics",
                return_value={
                    "segment_consistency_worst_dropout_frame_ratio": 0.12,
                    "segment_consistency_worst_tail_dropout_ratio": 0.14,
                    "segment_consistency_dropout_std": 0.02,
                    "segment_consistency_tail_dropout_std": 0.02,
                    "segment_consistency_rms_ratio_range_db": 2.0,
                },
            ), patch.object(
                module,
                "calculate_vocal_segment_issue_report",
                return_value=issue_report,
            ), patch.object(
                module,
                "calculate_anchor_support_metrics",
                return_value={
                    "unsupported_vs_anchor_ratio_db": -40.0,
                    "high_unsupported_vs_anchor_ratio_db": -40.0,
                    "high_noise_flatness_6000_16000": 0.05,
                    "stereo_side_mid_ratio_db": -16.0,
                },
            ), patch.object(
                module,
                "build_segment_issue_repair_candidate",
                side_effect=fake_segment_issue_repair_candidate,
            ) as repair_builder_mock:
                self.assertEqual(module.main(argv), 0)

            adaptive_manifest = json.loads(
                (output_dir / "adaptive_vocal_manifest.json").read_text(encoding="utf-8"),
            )

        self.assertEqual(repair_builder_mock.call_count, 2)
        self.assertEqual(adaptive_manifest["skipped_post_gate_residual_issue_repair"], [])
        self.assertEqual(adaptive_manifest["selected_profile"], "post_gate_residual_issue_repair")
        self.assertIn(
            "post_gate_residual_issue_repair",
            [candidate["profile_name"] for candidate in adaptive_manifest["candidates"]],
        )
        self.assertEqual(
            adaptive_manifest["post_gate_residual_issue_repair_profile"]["candidate_mode"],
            "post_gate_residual_issue_repaired_vocals",
        )
        self.assertEqual(adaptive_manifest["selected_artifacts"]["release_status"], "diagnostic_only")
        self.assertEqual(adaptive_manifest["defect_effectiveness_gate"]["status"], "blocked_for_listening")
        self.assertEqual(adaptive_manifest["detected_issue_action_contract"]["status"], "pass")
        self.assertEqual(
            adaptive_manifest["issue_effectiveness_ledger"]["windows"][0]["release_decision"],
            "blocked",
        )

    def test_cli_adds_gate_blocker_presence_band_recovery_candidate_without_bypassing_delivery_gate(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            export_script = temp_path / "export_hq_vocal_preview.py"
            output_dir = temp_path / "adaptive"
            source_path.write_bytes(b"source")
            export_script.write_text("# fake export script\n", encoding="utf-8")

            def fake_run_command(command: list[str]) -> None:
                output_arg = command[command.index("--output-dir") + 1]
                output_path = Path(output_arg)
                _write_fake_candidate(output_path, output_path.name)

            def fake_segment_issue_repair_candidate(*, profile, output_dir: Path, **_kwargs):
                _write_fake_candidate(output_dir, profile.name)
                vocals_path = output_dir / "vocals.flac"
                manifest_path = output_dir / "stem_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["pipeline"]["candidate_mode"] = profile.candidate_mode
                manifest["pipeline"]["vocal_postprocess"] = "segment_issue_repair_v1"
                manifest["manifest_path"] = str(manifest_path)
                manifest["stems"]["vocals"]["path"] = str(vocals_path)
                manifest["stems"]["vocals"]["vocal_postprocess"] = "segment_issue_repair_v1"
                manifest["stems"]["vocals"]["repaired_window_count"] = 1
                manifest["stems"]["vocals"]["base_segment_issue_max_score"] = 1.25
                manifest["stems"]["vocals"]["base_segment_issue_mean_top3_score"] = 0.95
                manifest["stems"]["vocals"]["window_decisions"] = [
                    {
                        "segment_index": 0,
                        "base_profile": "segment_issue_repair",
                        "repair_profile": "hybrid_tail_preserve",
                        "recovery_improvement": 0.18,
                    }
                ]
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return manifest

            def fake_presence_band_recovery_candidate(*, profile, output_dir: Path, **_kwargs):
                _write_fake_candidate(output_dir, profile.name)
                vocals_path = output_dir / "vocals.flac"
                manifest_path = output_dir / "stem_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["pipeline"]["candidate_mode"] = profile.candidate_mode
                manifest["pipeline"]["vocal_postprocess"] = "gate_blocker_presence_band_recovery_v1"
                manifest["pipeline"]["target_reference_assisted"] = False
                manifest["pipeline"]["target_reference_used_at_application"] = False
                manifest["pipeline"]["application_reference_free"] = True
                manifest["pipeline"]["delivery_allowed_without_reference"] = True
                manifest["pipeline"]["fallback_used"] = False
                manifest["pipeline"]["per_track_hardcode_allowed"] = False
                manifest["pipeline"]["not_timestamp_hardcoded"] = True
                manifest["manifest_path"] = str(manifest_path)
                manifest["stems"]["vocals"]["path"] = str(vocals_path)
                manifest["stems"]["vocals"]["vocal_postprocess"] = "gate_blocker_presence_band_recovery_v1"
                manifest["stems"]["vocals"]["presence_band_recovered_window_count"] = 1
                manifest["stems"]["vocals"]["base_segment_issue_max_score"] = 1.25
                manifest["stems"]["vocals"]["base_segment_issue_mean_top3_score"] = 0.95
                manifest["stems"]["vocals"]["window_decisions"] = [
                    {
                        "segment_index": 0,
                        "base_profile": "post_gate_residual_issue_repair",
                        "donor_profile": "hybrid_tail_preserve",
                        "predicted_severity_improvement": 0.22,
                    }
                ]
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return manifest

            def fake_metrics(_source_audio_path: Path, vocals_path: Path, **_kwargs) -> dict[str, float]:
                if vocals_path.parent.name == "gate_blocker_presence_band_recovery":
                    return _metrics(
                        dropout=0.03,
                        tail_dropout=0.04,
                        rms_ratio_db=-14.0,
                        low_ratio_db=-25.0,
                        kick_ratio_db=-16.0,
                    )
                if vocals_path.parent.name == "post_gate_residual_issue_repair":
                    return _metrics(
                        dropout=0.08,
                        tail_dropout=0.12,
                        rms_ratio_db=-15.0,
                        low_ratio_db=-24.0,
                        kick_ratio_db=-15.5,
                    )
                return _metrics(
                    dropout=0.26,
                    tail_dropout=0.30,
                    rms_ratio_db=-17.5,
                    low_ratio_db=-22.5,
                    kick_ratio_db=-14.5,
                )

            issue_report = {
                "schema_version": 1,
                "mode": "vocal_segment_issue_report",
                "summary": {
                    "max_issue_score": 1.25,
                    "mean_top3_issue_score": 0.95,
                    "issue_segments_count": 1.0,
                    "issue_window_ratio": 0.25,
                    "dominant_issue_labels": [{"label": "vocal_dropout", "count": 1}],
                },
                "summary_metrics": {
                    "segment_issue_max_score": 1.25,
                    "segment_issue_mean_top3_score": 0.95,
                    "segment_issue_count": 1.0,
                    "segment_issue_window_ratio": 0.25,
                },
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 3.0,
                        "relative_start_sec": 0.0,
                        "relative_end_sec": 3.0,
                        "severity_score": 1.25,
                        "labels": ["vocal_dropout", "presence_loss"],
                        "severity_parts": {"vocal_dropout": 0.85, "presence_loss": 0.40},
                    }
                ],
                "top_segments": [],
            }

            argv = [
                str(source_path),
                str(output_dir),
                "--export-script",
                str(export_script),
                "--python-executable",
                sys.executable,
                "--end-sec",
                "12.0",
                "--segment-consistency-sec",
                "3.0",
                "--disable-anchor-suppression",
                "--disable-body-repair",
                "--disable-segment-router",
                "--disable-issue-aware-segment-selection",
                "--disable-no-vocal-leakage-cleanup",
                "--disable-vocal-harshness-guard",
            ]

            with patch.object(module, "ensure_bs_roformer_runtime"), patch.object(
                module,
                "_run_command",
                side_effect=fake_run_command,
            ), patch.object(
                module,
                "calculate_vocal_proxy_metrics",
                side_effect=fake_metrics,
            ), patch.object(
                module,
                "calculate_vocal_proxy_consistency_metrics",
                return_value={
                    "segment_consistency_worst_dropout_frame_ratio": 0.12,
                    "segment_consistency_worst_tail_dropout_ratio": 0.14,
                    "segment_consistency_dropout_std": 0.02,
                    "segment_consistency_tail_dropout_std": 0.02,
                    "segment_consistency_rms_ratio_range_db": 2.0,
                },
            ), patch.object(
                module,
                "calculate_vocal_segment_issue_report",
                return_value=issue_report,
            ), patch.object(
                module,
                "calculate_anchor_support_metrics",
                return_value={
                    "unsupported_vs_anchor_ratio_db": -40.0,
                    "high_unsupported_vs_anchor_ratio_db": -40.0,
                    "high_noise_flatness_6000_16000": 0.05,
                    "stereo_side_mid_ratio_db": -16.0,
                },
            ), patch.object(
                module,
                "build_segment_issue_repair_candidate",
                side_effect=fake_segment_issue_repair_candidate,
            ), patch.object(
                module,
                "build_gate_blocker_best_segment_splice_candidate",
                return_value=None,
            ), patch.object(
                module,
                "build_gate_blocker_presence_band_recovery_candidate",
                side_effect=fake_presence_band_recovery_candidate,
            ) as presence_builder_mock:
                self.assertEqual(module.main(argv), 0)

            adaptive_manifest = json.loads(
                (output_dir / "adaptive_vocal_manifest.json").read_text(encoding="utf-8"),
            )

        presence_builder_mock.assert_called_once()
        self.assertEqual(adaptive_manifest["skipped_gate_blocker_presence_band_recovery"], [])
        self.assertEqual(adaptive_manifest["selected_profile"], "gate_blocker_presence_band_recovery")
        self.assertIn(
            "gate_blocker_presence_band_recovery",
            [candidate["profile_name"] for candidate in adaptive_manifest["candidates"]],
        )
        self.assertEqual(
            adaptive_manifest["gate_blocker_presence_band_recovery_profile"]["candidate_mode"],
            "gate_blocker_presence_band_recovered_vocals",
        )
        self.assertFalse(adaptive_manifest["selected_artifacts"]["delivery_allowed"])
        self.assertEqual(
            adaptive_manifest["selected_artifacts"]["presentation_status"],
            "blocked_from_listen_pack",
        )
        self.assertEqual(adaptive_manifest["defect_effectiveness_gate"]["status"], "blocked_for_listening")

    def test_cli_bounded_thin_repair_only_skips_extra_repairs_but_keeps_presence_recovery(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            export_script = temp_path / "export_hq_vocal_preview.py"
            output_dir = temp_path / "adaptive"
            source_path.write_bytes(b"source")
            export_script.write_text("# fake export script\n", encoding="utf-8")

            def fake_run_command(command: list[str]) -> None:
                output_arg = command[command.index("--output-dir") + 1]
                output_path = Path(output_arg)
                _write_fake_candidate(output_path, output_path.name)

            def fake_presence_band_recovery_candidate(*, profile, output_dir: Path, **_kwargs):
                _write_fake_candidate(output_dir, profile.name)
                manifest_path = output_dir / "stem_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["pipeline"]["candidate_mode"] = profile.candidate_mode
                manifest["pipeline"]["vocal_postprocess"] = "gate_blocker_presence_band_recovery_v1"
                manifest["manifest_path"] = str(manifest_path)
                manifest["stems"]["vocals"]["path"] = str(output_dir / "vocals.flac")
                manifest["stems"]["vocals"]["vocal_postprocess"] = "gate_blocker_presence_band_recovery_v1"
                manifest["stems"]["vocals"]["presence_band_recovered_window_count"] = 1
                manifest["stems"]["vocals"]["base_segment_issue_max_score"] = 1.25
                manifest["stems"]["vocals"]["base_segment_issue_mean_top3_score"] = 0.95
                manifest["stems"]["vocals"]["window_decisions"] = [
                    {
                        "segment_index": 0,
                        "base_profile": "hybrid_tail_preserve",
                        "donor_profile": "ft_tail_safe",
                        "predicted_severity_improvement": 0.22,
                    }
                ]
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return manifest

            issue_report = {
                "schema_version": 1,
                "mode": "vocal_segment_issue_report",
                "summary": {
                    "max_issue_score": 1.25,
                    "mean_top3_issue_score": 0.95,
                    "issue_segments_count": 1.0,
                    "issue_window_ratio": 0.25,
                    "dominant_issue_labels": [{"label": "presence_loss", "count": 1}],
                },
                "summary_metrics": {
                    "segment_issue_max_score": 1.25,
                    "segment_issue_mean_top3_score": 0.95,
                    "segment_issue_count": 1.0,
                    "segment_issue_window_ratio": 0.25,
                },
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 3.0,
                        "relative_start_sec": 0.0,
                        "relative_end_sec": 3.0,
                        "severity_score": 1.25,
                        "labels": ["thin_or_overcleaned", "presence_loss"],
                        "severity_parts": {"thin_or_overcleaned": 0.70, "presence_loss": 0.55},
                    }
                ],
                "top_segments": [],
            }

            argv = [
                str(source_path),
                str(output_dir),
                "--export-script",
                str(export_script),
                "--python-executable",
                sys.executable,
                "--end-sec",
                "12.0",
                "--segment-consistency-sec",
                "3.0",
                "--bounded-thin-repair-only",
                "--disable-anchor-suppression",
                "--disable-body-repair",
            ]

            with patch.object(module, "ensure_bs_roformer_runtime"), patch.object(
                module,
                "_run_command",
                side_effect=fake_run_command,
            ), patch.object(
                module,
                "calculate_vocal_proxy_metrics",
                return_value=_metrics(
                    dropout=0.22,
                    tail_dropout=0.16,
                    rms_ratio_db=-17.0,
                    low_ratio_db=-22.0,
                    kick_ratio_db=-14.5,
                ),
            ), patch.object(
                module,
                "calculate_vocal_proxy_consistency_metrics",
                return_value={
                    "segment_consistency_worst_dropout_frame_ratio": 0.12,
                    "segment_consistency_worst_tail_dropout_ratio": 0.14,
                    "segment_consistency_dropout_std": 0.02,
                    "segment_consistency_tail_dropout_std": 0.02,
                    "segment_consistency_rms_ratio_range_db": 2.0,
                },
            ), patch.object(
                module,
                "calculate_vocal_segment_issue_report",
                return_value=issue_report,
            ), patch.object(
                module,
                "calculate_anchor_support_metrics",
                return_value={
                    "unsupported_vs_anchor_ratio_db": -40.0,
                    "high_unsupported_vs_anchor_ratio_db": -40.0,
                    "high_noise_flatness_6000_16000": 0.05,
                    "stereo_side_mid_ratio_db": -16.0,
                },
            ), patch.object(
                module,
                "build_segment_issue_repair_candidate",
                return_value=None,
            ) as segment_repair_mock, patch.object(
                module,
                "build_gate_blocker_best_segment_splice_candidate",
                return_value=None,
            ) as splice_mock, patch.object(
                module,
                "build_gate_blocker_presence_band_recovery_candidate",
                side_effect=fake_presence_band_recovery_candidate,
            ) as presence_builder_mock, patch.object(
                module,
                "build_clustered_issue_window_ratio_repair_candidate",
                return_value=None,
            ) as clustered_repair_mock:
                self.assertEqual(module.main(argv), 0)

            adaptive_manifest = json.loads(
                (output_dir / "adaptive_vocal_manifest.json").read_text(encoding="utf-8"),
            )

        segment_repair_mock.assert_not_called()
        splice_mock.assert_not_called()
        clustered_repair_mock.assert_not_called()
        presence_builder_mock.assert_called_once()
        self.assertTrue(adaptive_manifest["bounded_thin_repair_only"])
        self.assertEqual(
            adaptive_manifest["skipped_post_gate_residual_issue_repair"][0]["reason"],
            "disabled_by_bounded_thin_repair_only",
        )
        self.assertEqual(
            adaptive_manifest["skipped_gate_blocker_best_segment_splice"][0]["reason"],
            "disabled_by_bounded_thin_repair_only",
        )
        self.assertEqual(
            adaptive_manifest["skipped_clustered_issue_window_ratio_repair"][0]["reason"],
            "disabled_by_bounded_thin_repair_only",
        )
        self.assertEqual(adaptive_manifest["skipped_gate_blocker_presence_band_recovery"], [])

    def test_cli_adds_clustered_issue_window_ratio_repair_candidate_without_bypassing_delivery_gate(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            export_script = temp_path / "export_hq_vocal_preview.py"
            output_dir = temp_path / "adaptive"
            source_path.write_bytes(b"source")
            export_script.write_text("# fake export script\n", encoding="utf-8")

            def fake_run_command(command: list[str]) -> None:
                output_arg = command[command.index("--output-dir") + 1]
                output_path = Path(output_arg)
                _write_fake_candidate(output_path, output_path.name)

            def fake_segment_issue_repair_candidate(*, profile, output_dir: Path, **_kwargs):
                _write_fake_candidate(output_dir, profile.name)
                manifest_path = output_dir / "stem_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["pipeline"]["candidate_mode"] = profile.candidate_mode
                manifest["manifest_path"] = str(manifest_path)
                manifest["stems"]["vocals"]["vocal_postprocess"] = "segment_issue_repair_v1"
                manifest["stems"]["vocals"]["window_decisions"] = [{"segment_index": 0}]
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return manifest

            def fake_clustered_repair_candidate(*, profile, output_dir: Path, **_kwargs):
                _write_fake_candidate(output_dir, profile.name)
                manifest_path = output_dir / "stem_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["pipeline"]["candidate_mode"] = profile.candidate_mode
                manifest["pipeline"]["vocal_postprocess"] = "clustered_issue_window_ratio_repair_v1"
                manifest["pipeline"]["target_reference_assisted"] = False
                manifest["pipeline"]["target_reference_used_at_application"] = False
                manifest["pipeline"]["application_reference_free"] = True
                manifest["pipeline"]["delivery_allowed_without_reference"] = True
                manifest["pipeline"]["fallback_used"] = False
                manifest["pipeline"]["per_track_hardcode_allowed"] = False
                manifest["pipeline"]["not_timestamp_hardcoded"] = True
                manifest["manifest_path"] = str(manifest_path)
                manifest["stems"]["vocals"]["vocal_postprocess"] = "clustered_issue_window_ratio_repair_v1"
                manifest["stems"]["vocals"]["clustered_repaired_window_count"] = 2
                manifest["stems"]["vocals"]["window_decisions"] = [
                    {"segment_index": 0, "predicted_severity_improvement": 0.11},
                    {"segment_index": 1, "predicted_severity_improvement": 0.10},
                ]
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return manifest

            def fake_metrics(_source_audio_path: Path, vocals_path: Path, **_kwargs) -> dict[str, float]:
                if vocals_path.parent.name in {
                    "clustered_issue_window_ratio_repair",
                    "clustered_issue_window_ratio_repair_pass2",
                }:
                    return _metrics(
                        dropout=0.04,
                        tail_dropout=0.05,
                        rms_ratio_db=-14.5,
                        low_ratio_db=-25.0,
                        kick_ratio_db=-17.0,
                    )
                return _metrics(
                    dropout=0.10,
                    tail_dropout=0.12,
                    rms_ratio_db=-16.5,
                    low_ratio_db=-23.0,
                    kick_ratio_db=-14.0,
                )

            issue_report = {
                "schema_version": 1,
                "mode": "vocal_segment_issue_report",
                "summary": {
                    "max_issue_score": 0.72,
                    "mean_top3_issue_score": 0.70,
                    "issue_segments_count": 2.0,
                    "issue_window_ratio": 0.50,
                    "dominant_issue_labels": [{"label": "kick_punch_bleed", "count": 2}],
                },
                "summary_metrics": {
                    "segment_issue_max_score": 0.72,
                    "segment_issue_mean_top3_score": 0.70,
                    "segment_issue_count": 2.0,
                    "segment_issue_window_ratio": 0.50,
                },
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 3.0,
                        "relative_start_sec": 0.0,
                        "relative_end_sec": 3.0,
                        "severity_score": 0.70,
                        "labels": ["kick_punch_bleed"],
                        "severity_parts": {"kick_punch_bleed": 0.70},
                    },
                    {
                        "start_sec": 3.0,
                        "end_sec": 6.0,
                        "relative_start_sec": 3.0,
                        "relative_end_sec": 6.0,
                        "severity_score": 0.72,
                        "labels": ["kick_punch_bleed"],
                        "severity_parts": {"kick_punch_bleed": 0.72},
                    },
                    {
                        "start_sec": 6.0,
                        "end_sec": 9.0,
                        "relative_start_sec": 6.0,
                        "relative_end_sec": 9.0,
                        "severity_score": 0.0,
                        "labels": ["acceptable_proxy_window"],
                        "severity_parts": {},
                    },
                    {
                        "start_sec": 9.0,
                        "end_sec": 12.0,
                        "relative_start_sec": 9.0,
                        "relative_end_sec": 12.0,
                        "severity_score": 0.0,
                        "labels": ["acceptable_proxy_window"],
                        "severity_parts": {},
                    },
                ],
                "top_segments": [],
            }
            first_clustered_issue_report = {
                **issue_report,
                "summary": {
                    **issue_report["summary"],
                    "max_issue_score": 0.68,
                    "mean_top3_issue_score": 0.66,
                    "issue_window_ratio": 0.50,
                },
                "summary_metrics": {
                    **issue_report["summary_metrics"],
                    "segment_issue_max_score": 0.68,
                    "segment_issue_mean_top3_score": 0.66,
                    "segment_issue_window_ratio": 0.50,
                },
            }
            second_clustered_issue_report = {
                **issue_report,
                "summary": {
                    **issue_report["summary"],
                    "max_issue_score": 0.62,
                    "mean_top3_issue_score": 0.60,
                    "issue_window_ratio": 0.50,
                },
                "summary_metrics": {
                    **issue_report["summary_metrics"],
                    "segment_issue_max_score": 0.62,
                    "segment_issue_mean_top3_score": 0.60,
                    "segment_issue_window_ratio": 0.50,
                },
            }

            def fake_segment_issue_report(_source_audio_path: Path, vocals_path: Path, **_kwargs):
                return {
                    "clustered_issue_window_ratio_repair": first_clustered_issue_report,
                    "clustered_issue_window_ratio_repair_pass2": second_clustered_issue_report,
                }.get(vocals_path.parent.name, issue_report)

            argv = [
                str(source_path),
                str(output_dir),
                "--export-script",
                str(export_script),
                "--python-executable",
                sys.executable,
                "--end-sec",
                "12.0",
                "--segment-consistency-sec",
                "3.0",
                "--disable-anchor-suppression",
                "--disable-body-repair",
                "--disable-segment-router",
                "--disable-issue-aware-segment-selection",
                "--disable-no-vocal-leakage-cleanup",
                "--disable-vocal-harshness-guard",
            ]

            with patch.object(module, "ensure_bs_roformer_runtime"), patch.object(
                module,
                "_run_command",
                side_effect=fake_run_command,
            ), patch.object(
                module,
                "calculate_vocal_proxy_metrics",
                side_effect=fake_metrics,
            ), patch.object(
                module,
                "calculate_vocal_proxy_consistency_metrics",
                return_value={
                    "segment_consistency_worst_dropout_frame_ratio": 0.12,
                    "segment_consistency_worst_tail_dropout_ratio": 0.14,
                    "segment_consistency_dropout_std": 0.02,
                    "segment_consistency_tail_dropout_std": 0.02,
                    "segment_consistency_rms_ratio_range_db": 2.0,
                },
            ), patch.object(
                module,
                "calculate_vocal_segment_issue_report",
                side_effect=fake_segment_issue_report,
            ), patch.object(
                module,
                "calculate_anchor_support_metrics",
                return_value={
                    "unsupported_vs_anchor_ratio_db": -40.0,
                    "high_unsupported_vs_anchor_ratio_db": -40.0,
                    "high_noise_flatness_6000_16000": 0.05,
                    "stereo_side_mid_ratio_db": -16.0,
                },
            ), patch.object(
                module,
                "build_segment_issue_repair_candidate",
                side_effect=fake_segment_issue_repair_candidate,
            ), patch.object(
                module,
                "build_gate_blocker_best_segment_splice_candidate",
                return_value=None,
            ), patch.object(
                module,
                "build_gate_blocker_presence_band_recovery_candidate",
                return_value=None,
            ), patch.object(
                module,
                "build_clustered_issue_window_ratio_repair_candidate",
                side_effect=fake_clustered_repair_candidate,
            ) as clustered_builder_mock:
                self.assertEqual(module.main(argv), 0)

            adaptive_manifest = json.loads(
                (output_dir / "adaptive_vocal_manifest.json").read_text(encoding="utf-8"),
            )

        self.assertEqual(clustered_builder_mock.call_count, 2)
        self.assertEqual(adaptive_manifest["skipped_clustered_issue_window_ratio_repair"], [])
        self.assertEqual(adaptive_manifest["skipped_clustered_issue_window_ratio_repair_pass2"], [])
        self.assertEqual(adaptive_manifest["selected_profile"], "clustered_issue_window_ratio_repair_pass2")
        self.assertIn(
            "clustered_issue_window_ratio_repair",
            [candidate["profile_name"] for candidate in adaptive_manifest["candidates"]],
        )
        self.assertIn(
            "clustered_issue_window_ratio_repair_pass2",
            [candidate["profile_name"] for candidate in adaptive_manifest["candidates"]],
        )
        self.assertFalse(adaptive_manifest["selected_artifacts"]["delivery_allowed"])
        self.assertEqual(
            adaptive_manifest["selected_artifacts"]["presentation_status"],
            "blocked_from_listen_pack",
        )
        self.assertEqual(adaptive_manifest["defect_effectiveness_gate"]["status"], "blocked_for_listening")

    def test_cli_adds_no_vocal_leakage_cleanup_candidate_to_ranking(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            export_script = temp_path / "export_hq_vocal_preview.py"
            output_dir = temp_path / "adaptive"
            source_path.write_bytes(b"source")
            export_script.write_text("# fake export script\n", encoding="utf-8")

            def fake_run_command(command: list[str]) -> None:
                output_arg = command[command.index("--output-dir") + 1]
                output_path = Path(output_arg)
                _write_fake_candidate(output_path, output_path.name)

            def fake_no_vocal_cleanup_candidate(*, profile, output_dir: Path, **_kwargs):
                _write_fake_candidate(output_dir, profile.name)
                vocals_path = output_dir / "vocals.flac"
                manifest_path = output_dir / "stem_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["pipeline"]["candidate_mode"] = profile.candidate_mode
                manifest["pipeline"]["vocal_postprocess"] = "no_vocal_leakage_cleanup_v1"
                manifest["manifest_path"] = str(manifest_path)
                manifest["stems"]["vocals"]["path"] = str(vocals_path)
                manifest["stems"]["vocals"]["vocal_postprocess"] = "no_vocal_leakage_cleanup_v1"
                manifest["stems"]["vocals"]["cleaned_window_count"] = 2
                manifest["stems"]["vocals"]["window_decisions"] = [
                    {
                        "segment_index": 0,
                        "cleanup_kind": "consensus_no_vocal",
                        "start_sec": 0.0,
                        "end_sec": 3.0,
                        "best_candidate_severity_score": 1.6,
                    },
                    {
                        "segment_index": 1,
                        "cleanup_kind": "low_energy_no_vocal_bridge",
                        "start_sec": 3.0,
                        "end_sec": 6.0,
                        "best_candidate_severity_score": 0.4,
                    },
                ]
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return manifest

            def fake_metrics(_source_audio_path: Path, vocals_path: Path, **_kwargs) -> dict[str, float]:
                if vocals_path.parent.name == "no_vocal_leakage_cleanup":
                    return _metrics(
                        dropout=0.01,
                        tail_dropout=0.02,
                        rms_ratio_db=-14.0,
                        low_ratio_db=-24.0,
                        kick_ratio_db=-16.0,
                    )
                return _metrics(
                    dropout=0.28,
                    tail_dropout=0.32,
                    rms_ratio_db=-17.0,
                    low_ratio_db=-23.0,
                    kick_ratio_db=-15.0,
                )

            issue_report = {
                "schema_version": 1,
                "mode": "vocal_segment_issue_report",
                "summary": {
                    "max_issue_score": 1.4,
                    "mean_top3_issue_score": 1.0,
                    "issue_segments_count": 1.0,
                    "issue_window_ratio": 0.25,
                    "dominant_issue_labels": [{"label": "vocal_dropout", "count": 1}],
                },
                "summary_metrics": {
                    "segment_issue_max_score": 1.4,
                    "segment_issue_mean_top3_score": 1.0,
                    "segment_issue_count": 1.0,
                    "segment_issue_window_ratio": 0.25,
                },
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 3.0,
                        "relative_start_sec": 0.0,
                        "relative_end_sec": 3.0,
                        "severity_score": 1.4,
                        "labels": ["vocal_dropout", "high_noise_or_robotic_artifact"],
                        "metrics": {
                            "candidate_to_source_rms_db": -24.0,
                            "kick_punch_ratio_db_250_900": -12.0,
                        },
                    }
                ],
                "top_segments": [],
            }

            argv = [
                str(source_path),
                str(output_dir),
                "--export-script",
                str(export_script),
                "--python-executable",
                sys.executable,
                "--end-sec",
                "12.0",
                "--segment-consistency-sec",
                "3.0",
                "--disable-anchor-suppression",
                "--disable-body-repair",
                "--disable-segment-router",
                "--disable-issue-aware-segment-selection",
                "--disable-segment-issue-repair",
                "--disable-vocal-harshness-guard",
            ]

            with patch.object(module, "ensure_bs_roformer_runtime"), patch.object(
                module,
                "_run_command",
                side_effect=fake_run_command,
            ), patch.object(
                module,
                "calculate_vocal_proxy_metrics",
                side_effect=fake_metrics,
            ), patch.object(
                module,
                "calculate_vocal_proxy_consistency_metrics",
                return_value={
                    "segment_consistency_worst_dropout_frame_ratio": 0.12,
                    "segment_consistency_worst_tail_dropout_ratio": 0.14,
                    "segment_consistency_dropout_std": 0.02,
                    "segment_consistency_tail_dropout_std": 0.02,
                    "segment_consistency_rms_ratio_range_db": 2.0,
                },
            ), patch.object(
                module,
                "calculate_vocal_segment_issue_report",
                return_value=issue_report,
            ), patch.object(
                module,
                "calculate_anchor_support_metrics",
                return_value={
                    "unsupported_vs_anchor_ratio_db": -40.0,
                    "high_unsupported_vs_anchor_ratio_db": -40.0,
                    "high_noise_flatness_6000_16000": 0.05,
                    "stereo_side_mid_ratio_db": -16.0,
                },
            ), patch.object(
                module,
                "build_no_vocal_leakage_cleanup_candidate",
                side_effect=fake_no_vocal_cleanup_candidate,
            ) as cleanup_builder_mock:
                self.assertEqual(module.main(argv), 0)

            adaptive_manifest = json.loads(
                (output_dir / "adaptive_vocal_manifest.json").read_text(encoding="utf-8"),
            )

        cleanup_builder_mock.assert_called_once()
        self.assertEqual(adaptive_manifest["selected_profile"], "no_vocal_leakage_cleanup")
        self.assertEqual(
            adaptive_manifest["no_vocal_leakage_cleanup_profile"]["candidate_mode"],
            "no_vocal_leakage_cleaned_vocals",
        )
        self.assertEqual(adaptive_manifest["skipped_no_vocal_leakage_cleanup"], [])
        self.assertIn(
            "no_vocal_leakage_cleanup",
            [candidate["profile_name"] for candidate in adaptive_manifest["candidates"]],
        )

    def test_cli_disable_no_vocal_leakage_cleanup_skips_candidate_builder(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            export_script = temp_path / "export_hq_vocal_preview.py"
            output_dir = temp_path / "adaptive"
            source_path.write_bytes(b"source")
            export_script.write_text("# fake export script\n", encoding="utf-8")

            def fake_run_command(command: list[str]) -> None:
                output_arg = command[command.index("--output-dir") + 1]
                output_path = Path(output_arg)
                _write_fake_candidate(output_path, output_path.name)

            issue_report = {
                "schema_version": 1,
                "mode": "vocal_segment_issue_report",
                "summary": {
                    "max_issue_score": 1.4,
                    "mean_top3_issue_score": 1.0,
                    "issue_segments_count": 1.0,
                    "issue_window_ratio": 0.25,
                    "dominant_issue_labels": [{"label": "vocal_dropout", "count": 1}],
                },
                "summary_metrics": {
                    "segment_issue_max_score": 1.4,
                    "segment_issue_mean_top3_score": 1.0,
                    "segment_issue_count": 1.0,
                    "segment_issue_window_ratio": 0.25,
                },
                "segments": [],
                "top_segments": [],
            }

            argv = [
                str(source_path),
                str(output_dir),
                "--export-script",
                str(export_script),
                "--python-executable",
                sys.executable,
                "--end-sec",
                "12.0",
                "--segment-consistency-sec",
                "3.0",
                "--disable-anchor-suppression",
                "--disable-body-repair",
                "--disable-segment-router",
                "--disable-issue-aware-segment-selection",
                "--disable-segment-issue-repair",
                "--disable-no-vocal-leakage-cleanup",
            ]

            with patch.object(module, "ensure_bs_roformer_runtime"), patch.object(
                module,
                "_run_command",
                side_effect=fake_run_command,
            ), patch.object(
                module,
                "calculate_vocal_proxy_metrics",
                return_value=_metrics(
                    dropout=0.18,
                    tail_dropout=0.20,
                    rms_ratio_db=-16.0,
                    low_ratio_db=-24.0,
                    kick_ratio_db=-16.0,
                ),
            ), patch.object(
                module,
                "calculate_vocal_proxy_consistency_metrics",
                return_value={
                    "segment_consistency_worst_dropout_frame_ratio": 0.12,
                    "segment_consistency_worst_tail_dropout_ratio": 0.14,
                    "segment_consistency_dropout_std": 0.02,
                    "segment_consistency_tail_dropout_std": 0.02,
                    "segment_consistency_rms_ratio_range_db": 2.0,
                },
            ), patch.object(
                module,
                "calculate_vocal_segment_issue_report",
                return_value=issue_report,
            ), patch.object(
                module,
                "calculate_anchor_support_metrics",
                return_value={
                    "unsupported_vs_anchor_ratio_db": -40.0,
                    "high_unsupported_vs_anchor_ratio_db": -40.0,
                    "high_noise_flatness_6000_16000": 0.05,
                    "stereo_side_mid_ratio_db": -16.0,
                },
            ), patch.object(
                module,
                "build_no_vocal_leakage_cleanup_candidate",
            ) as cleanup_builder_mock:
                self.assertEqual(module.main(argv), 0)

            adaptive_manifest = json.loads(
                (output_dir / "adaptive_vocal_manifest.json").read_text(encoding="utf-8"),
            )

        cleanup_builder_mock.assert_not_called()
        self.assertEqual(
            adaptive_manifest["skipped_no_vocal_leakage_cleanup"],
            [{"name": "no_vocal_leakage_cleanup", "reason": "disabled_by_flag"}],
        )
        self.assertNotIn(
            "no_vocal_leakage_cleanup",
            [candidate["profile_name"] for candidate in adaptive_manifest["candidates"]],
        )

    def test_cli_adds_vocal_harshness_guard_candidate_to_ranking_and_action_contract(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            export_script = temp_path / "export_hq_vocal_preview.py"
            output_dir = temp_path / "adaptive"
            source_path.write_bytes(b"source")
            export_script.write_text("# fake export script\n", encoding="utf-8")

            def fake_run_command(command: list[str]) -> None:
                output_arg = command[command.index("--output-dir") + 1]
                output_path = Path(output_arg)
                _write_fake_candidate(output_path, output_path.name)

            def fake_harshness_guard_candidate(*, profile, output_dir: Path, **_kwargs):
                _write_fake_candidate(output_dir, profile.name)
                vocals_path = output_dir / "vocals.flac"
                manifest_path = output_dir / "stem_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["pipeline"]["candidate_mode"] = profile.candidate_mode
                manifest["pipeline"]["vocal_postprocess"] = "vocal_harshness_guard_v1"
                manifest["manifest_path"] = str(manifest_path)
                manifest["stems"]["vocals"]["path"] = str(vocals_path)
                manifest["stems"]["vocals"]["vocal_postprocess"] = "vocal_harshness_guard_v1"
                manifest["stems"]["vocals"]["window_decisions"] = [
                    {
                        "segment_index": 0,
                        "labels": ["high_noise_or_robotic_artifact"],
                        "strength": 0.75,
                    }
                ]
                manifest["stems"]["vocals"]["debug"] = {
                    "base_high_band_rms": 0.20,
                    "guarded_high_band_rms": 0.10,
                    "route_mean": 0.12,
                }
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return manifest

            def fake_metrics(_source_audio_path: Path, vocals_path: Path, **_kwargs) -> dict[str, float]:
                if vocals_path.parent.name == "vocal_harshness_guard":
                    metrics = _metrics(
                        dropout=0.02,
                        tail_dropout=0.03,
                        rms_ratio_db=-14.0,
                        low_ratio_db=-24.0,
                        kick_ratio_db=-16.0,
                    )
                    metrics["high_noise_flatness_6000_16000"] = 0.10
                    return metrics
                metrics = _metrics(
                    dropout=0.04,
                    tail_dropout=0.05,
                    rms_ratio_db=-14.5,
                    low_ratio_db=-24.0,
                    kick_ratio_db=-16.0,
                )
                metrics["high_noise_flatness_6000_16000"] = 0.24
                return metrics

            issue_report = {
                "schema_version": 1,
                "mode": "vocal_segment_issue_report",
                "summary": {
                    "max_issue_score": 1.4,
                    "mean_top3_issue_score": 0.9,
                    "issue_segments_count": 1.0,
                    "issue_window_ratio": 0.25,
                    "dominant_issue_labels": [{"label": "high_noise_or_robotic_artifact", "count": 1}],
                },
                "summary_metrics": {
                    "segment_issue_max_score": 1.4,
                    "segment_issue_mean_top3_score": 0.9,
                    "segment_issue_count": 1.0,
                    "segment_issue_window_ratio": 0.25,
                },
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 3.0,
                        "relative_start_sec": 0.0,
                        "relative_end_sec": 3.0,
                        "severity_score": 1.4,
                        "labels": ["high_noise_or_robotic_artifact"],
                        "severity_parts": {"high_noise_or_robotic_artifact": 1.4},
                        "metrics": {"high_noise_flatness_6000_16000": 0.24},
                    }
                ],
                "top_segments": [],
            }

            argv = [
                str(source_path),
                str(output_dir),
                "--export-script",
                str(export_script),
                "--python-executable",
                sys.executable,
                "--end-sec",
                "12.0",
                "--segment-consistency-sec",
                "3.0",
                "--disable-anchor-suppression",
                "--disable-body-repair",
                "--disable-segment-router",
                "--disable-issue-aware-segment-selection",
                "--disable-segment-issue-repair",
                "--disable-no-vocal-leakage-cleanup",
            ]

            with patch.object(module, "ensure_bs_roformer_runtime"), patch.object(
                module,
                "_run_command",
                side_effect=fake_run_command,
            ), patch.object(
                module,
                "calculate_vocal_proxy_metrics",
                side_effect=fake_metrics,
            ), patch.object(
                module,
                "calculate_vocal_proxy_consistency_metrics",
                return_value={
                    "segment_consistency_worst_dropout_frame_ratio": 0.12,
                    "segment_consistency_worst_tail_dropout_ratio": 0.14,
                    "segment_consistency_dropout_std": 0.02,
                    "segment_consistency_tail_dropout_std": 0.02,
                    "segment_consistency_rms_ratio_range_db": 2.0,
                },
            ), patch.object(
                module,
                "calculate_vocal_segment_issue_report",
                return_value=issue_report,
            ), patch.object(
                module,
                "calculate_anchor_support_metrics",
                return_value={
                    "unsupported_vs_anchor_ratio_db": -40.0,
                    "high_unsupported_vs_anchor_ratio_db": -40.0,
                    "high_noise_flatness_6000_16000": 0.10,
                    "stereo_side_mid_ratio_db": -16.0,
                },
            ), patch.object(
                module,
                "build_vocal_harshness_guard_candidate",
                side_effect=fake_harshness_guard_candidate,
            ) as harshness_builder_mock:
                self.assertEqual(module.main(argv), 0)

            adaptive_manifest = json.loads(
                (output_dir / "adaptive_vocal_manifest.json").read_text(encoding="utf-8"),
            )

        harshness_builder_mock.assert_called_once()
        self.assertEqual(adaptive_manifest["selected_profile"], "vocal_harshness_guard")
        self.assertEqual(adaptive_manifest["skipped_vocal_harshness_guard"], [])
        self.assertIn(
            "vocal_harshness_guard",
            [candidate["profile_name"] for candidate in adaptive_manifest["candidates"]],
        )
        self.assertEqual(adaptive_manifest["detected_issue_action_contract"]["status"], "pass")
        self.assertIn(
            "high_noise_or_robotic_artifact",
            adaptive_manifest["detected_issue_action_contract"]["summary"]["covered_labels"],
        )
        self.assertEqual(adaptive_manifest["defect_effectiveness_gate"]["status"], "blocked_for_listening")
        self.assertEqual(
            adaptive_manifest["defect_effectiveness_gate"]["summary"]["blocker_count"],
            1,
        )
        self.assertEqual(
            adaptive_manifest["selected_artifacts"]["release_status"],
            "diagnostic_only",
        )

    def test_cli_adds_ir_residual_candidate_to_router_search_space(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sample_rate = 8000
            t = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
            vocal = 0.12 * np.sin(2.0 * np.pi * 330.0 * t)
            instrumental = 0.08 * np.sin(2.0 * np.pi * 110.0 * t)
            source_audio = np.stack([vocal + instrumental, vocal + instrumental], axis=1).astype(np.float32)
            minus_audio = np.stack([instrumental, instrumental], axis=1).astype(np.float32)
            source_path = temp_path / "source.flac"
            minus_path = temp_path / "instrumental_minus.flac"
            export_script = temp_path / "export_hq_vocal_preview.py"
            output_dir = temp_path / "adaptive"
            sf.write(source_path, source_audio, sample_rate, subtype="PCM_24")
            sf.write(minus_path, minus_audio, sample_rate, subtype="PCM_24")
            export_script.write_text("# fake export script\n", encoding="utf-8")
            commands: list[list[str]] = []

            def fake_run_command(command: list[str]) -> None:
                commands.append(command)
                output_arg = command[command.index("--output-dir") + 1]
                output_path = Path(output_arg)
                _write_fake_candidate(output_path, output_path.name)

            def fake_derived_candidate(*, profile, output_dir: Path, **_kwargs):
                _write_fake_candidate(output_dir, profile.name)
                manifest_path = output_dir / "stem_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["pipeline"]["candidate_mode"] = "derived_test_candidate"
                manifest["manifest_path"] = str(manifest_path)
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return manifest

            def fake_segment_candidate(*, profile, output_dir: Path, **_kwargs):
                _write_fake_candidate(output_dir, profile.name)
                manifest_path = output_dir / "stem_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["pipeline"]["candidate_mode"] = "segment_adaptive_routed_vocals"
                manifest["manifest_path"] = str(manifest_path)
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return manifest

            def fake_metrics(_source_audio_path: Path, vocals_path: Path, **_kwargs) -> dict[str, float]:
                if vocals_path.parent.name == "ir_residual_supported_router":
                    return _metrics(
                        dropout=0.02,
                        tail_dropout=0.02,
                        rms_ratio_db=-13.0,
                        low_ratio_db=-23.0,
                        kick_ratio_db=-16.0,
                    )
                return _metrics(
                    dropout=0.14,
                    tail_dropout=0.18,
                    rms_ratio_db=-17.0,
                    low_ratio_db=-23.0,
                    kick_ratio_db=-16.0,
                )

            argv = [
                str(source_path),
                str(output_dir),
                "--export-script",
                str(export_script),
                "--python-executable",
                sys.executable,
                "--end-sec",
                "1.0",
                "--segment-consistency-sec",
                "0",
                "--ir-instrumental-minus-path",
                str(minus_path),
            ]

            with patch.object(module, "ensure_bs_roformer_runtime"), patch.object(
                module,
                "_run_command",
                side_effect=fake_run_command,
            ), patch.object(
                module,
                "build_anchor_suppressed_candidate",
                side_effect=fake_derived_candidate,
            ), patch.object(
                module,
                "build_vocal_body_repair_candidate",
                side_effect=fake_derived_candidate,
            ), patch.object(
                module,
                "build_segment_router_candidate",
                side_effect=fake_segment_candidate,
            ) as segment_builder_mock, patch.object(
                module,
                "calculate_vocal_proxy_metrics",
                side_effect=fake_metrics,
            ), patch.object(
                module,
                "calculate_anchor_support_metrics",
                return_value={
                    "unsupported_vs_anchor_ratio_db": -40.0,
                    "high_unsupported_vs_anchor_ratio_db": -40.0,
                    "high_noise_flatness_6000_16000": 0.05,
                    "stereo_side_mid_ratio_db": -16.0,
                },
            ):
                self.assertEqual(module.main(argv), 0)

            adaptive_manifest = json.loads(
                (output_dir / "adaptive_vocal_manifest.json").read_text(encoding="utf-8"),
            )

        self.assertEqual(len(commands), 4)
        self.assertEqual(segment_builder_mock.call_count, 4)
        self.assertEqual(adaptive_manifest["selected_profile"], "ir_residual_supported_router")
        self.assertEqual(adaptive_manifest["external_candidate_profiles"][0]["name"], "ir_residual_vocals")
        self.assertIn(
            "ir_residual_vocals",
            [candidate["profile_name"] for candidate in adaptive_manifest["candidates"]],
        )
        self.assertIn(
            "ir_residual_supported_router",
            [candidate["profile_name"] for candidate in adaptive_manifest["candidates"]],
        )

    def test_cli_disable_segment_router_skips_router_candidate(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            export_script = temp_path / "export_hq_vocal_preview.py"
            output_dir = temp_path / "adaptive"
            source_path.write_bytes(b"source")
            export_script.write_text("# fake export script\n", encoding="utf-8")

            def fake_run_command(command: list[str]) -> None:
                output_arg = command[command.index("--output-dir") + 1]
                output_path = Path(output_arg)
                _write_fake_candidate(output_path, output_path.name)

            def fake_derived_candidate(*, profile, output_dir: Path, **_kwargs):
                _write_fake_candidate(output_dir, profile.name)
                manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
                manifest["manifest_path"] = str(output_dir / "stem_manifest.json")
                return manifest

            argv = [
                str(source_path),
                str(output_dir),
                "--export-script",
                str(export_script),
                "--python-executable",
                sys.executable,
                "--end-sec",
                "12.0",
                "--segment-consistency-sec",
                "0",
                "--disable-segment-router",
            ]

            with patch.object(module, "ensure_bs_roformer_runtime"), patch.object(
                module,
                "_run_command",
                side_effect=fake_run_command,
            ), patch.object(
                module,
                "build_anchor_suppressed_candidate",
                side_effect=fake_derived_candidate,
            ), patch.object(
                module,
                "build_vocal_body_repair_candidate",
                side_effect=fake_derived_candidate,
            ), patch.object(
                module,
                "build_segment_router_candidate",
            ) as segment_builder_mock, patch.object(
                module,
                "calculate_vocal_proxy_metrics",
                return_value=_metrics(
                    dropout=0.12,
                    tail_dropout=0.14,
                    rms_ratio_db=-17.0,
                    low_ratio_db=-23.0,
                    kick_ratio_db=-16.0,
                ),
            ), patch.object(
                module,
                "calculate_anchor_support_metrics",
                return_value={
                    "unsupported_vs_anchor_ratio_db": -40.0,
                    "high_unsupported_vs_anchor_ratio_db": -40.0,
                    "high_noise_flatness_6000_16000": 0.05,
                    "stereo_side_mid_ratio_db": -16.0,
                },
            ):
                self.assertEqual(module.main(argv), 0)

            adaptive_manifest = json.loads(
                (output_dir / "adaptive_vocal_manifest.json").read_text(encoding="utf-8"),
            )

        segment_builder_mock.assert_not_called()
        self.assertEqual(adaptive_manifest["segment_router_candidate_profiles"], [])
        self.assertEqual(adaptive_manifest["skipped_segment_router_profiles"], [])
        self.assertNotIn(
            "segment_adaptive_vocal_router",
            [candidate["profile_name"] for candidate in adaptive_manifest["candidates"]],
        )
        self.assertNotIn(
            "busy_vocal_supported_router",
            [candidate["profile_name"] for candidate in adaptive_manifest["candidates"]],
        )

    def test_cli_skips_segment_router_when_body_repair_dependency_missing(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            export_script = temp_path / "export_hq_vocal_preview.py"
            output_dir = temp_path / "adaptive"
            source_path.write_bytes(b"source")
            export_script.write_text("# fake export script\n", encoding="utf-8")

            def fake_run_command(command: list[str]) -> None:
                output_arg = command[command.index("--output-dir") + 1]
                output_path = Path(output_arg)
                _write_fake_candidate(output_path, output_path.name)

            def fake_anchor_candidate(*, profile, output_dir: Path, **_kwargs):
                _write_fake_candidate(output_dir, profile.name)
                manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
                manifest["manifest_path"] = str(output_dir / "stem_manifest.json")
                return manifest

            argv = [
                str(source_path),
                str(output_dir),
                "--export-script",
                str(export_script),
                "--python-executable",
                sys.executable,
                "--end-sec",
                "12.0",
                "--segment-consistency-sec",
                "0",
                "--disable-body-repair",
            ]

            with patch.object(module, "ensure_bs_roformer_runtime"), patch.object(
                module,
                "_run_command",
                side_effect=fake_run_command,
            ), patch.object(
                module,
                "build_anchor_suppressed_candidate",
                side_effect=fake_anchor_candidate,
            ), patch.object(
                module,
                "build_segment_router_candidate",
            ) as segment_builder_mock, patch.object(
                module,
                "calculate_vocal_proxy_metrics",
                return_value=_metrics(
                    dropout=0.12,
                    tail_dropout=0.14,
                    rms_ratio_db=-17.0,
                    low_ratio_db=-23.0,
                    kick_ratio_db=-16.0,
                ),
            ), patch.object(
                module,
                "calculate_anchor_support_metrics",
                return_value={
                    "unsupported_vs_anchor_ratio_db": -40.0,
                    "high_unsupported_vs_anchor_ratio_db": -40.0,
                    "high_noise_flatness_6000_16000": 0.05,
                    "stereo_side_mid_ratio_db": -16.0,
                },
            ):
                self.assertEqual(module.main(argv), 0)

            adaptive_manifest = json.loads(
                (output_dir / "adaptive_vocal_manifest.json").read_text(encoding="utf-8"),
            )

        segment_builder_mock.assert_not_called()
        self.assertEqual(adaptive_manifest["body_repair_candidate_profiles"], [])
        self.assertEqual(len(adaptive_manifest["skipped_segment_router_profiles"]), 4)
        self.assertEqual(
            [profile["name"] for profile in adaptive_manifest["skipped_segment_router_profiles"]],
            [
                "segment_adaptive_vocal_router",
                "busy_vocal_supported_router",
                "ir_residual_supported_router",
                "quality_consistent_ir_router",
            ],
        )
        self.assertIn(
            "repair_profile",
            adaptive_manifest["skipped_segment_router_profiles"][0]["missing_dependencies"],
        )
        self.assertNotIn(
            "segment_adaptive_vocal_router",
            [candidate["profile_name"] for candidate in adaptive_manifest["candidates"]],
        )
        self.assertNotIn(
            "busy_vocal_supported_router",
            [candidate["profile_name"] for candidate in adaptive_manifest["candidates"]],
        )

    def test_cli_reuses_existing_candidate_manifest_when_requested(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            export_script = temp_path / "export_hq_vocal_preview.py"
            output_dir = temp_path / "adaptive"
            candidate_dir = output_dir / "candidates" / "hybrid_tail_preserve"
            source_path.write_bytes(b"source")
            export_script.write_text("# fake export script\n", encoding="utf-8")
            _write_fake_candidate(candidate_dir, "hybrid_tail_preserve")
            reusable_manifest_path = candidate_dir / "stem_manifest.json"
            reusable_manifest = json.loads(reusable_manifest_path.read_text(encoding="utf-8"))
            reusable_manifest.update(
                {
                    "audio_path": str(source_path),
                    "source_audio_sha256": module._sha256_file(source_path),
                    "preview_window_sec": {"start": 0.0, "end": 12.0},
                    "tuning": module.default_candidate_profiles()[0].tuning,
                }
            )
            reusable_manifest_path.write_text(
                json.dumps(reusable_manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            argv = [
                str(source_path),
                str(output_dir),
                "--export-script",
                str(export_script),
                "--python-executable",
                sys.executable,
                "--end-sec",
                "12.0",
                "--segment-consistency-sec",
                "0",
                "--profile",
                "hybrid_tail_preserve",
                "--reuse-existing-candidates",
                "--disable-anchor-suppression",
            ]

            with patch.object(module, "ensure_bs_roformer_runtime"), patch.object(
                module,
                "_run_command",
            ) as run_command_mock, patch.object(
                module,
                "calculate_vocal_proxy_metrics",
                return_value=_metrics(
                    dropout=0.04,
                    tail_dropout=0.05,
                    rms_ratio_db=-14.0,
                    low_ratio_db=-22.0,
                    kick_ratio_db=-15.0,
                ),
            ):
                self.assertEqual(module.main(argv), 0)

            adaptive_manifest = json.loads(
                (output_dir / "adaptive_vocal_manifest.json").read_text(encoding="utf-8"),
            )

        run_command_mock.assert_not_called()
        self.assertEqual(adaptive_manifest["selected_profile"], "hybrid_tail_preserve")
        self.assertEqual(adaptive_manifest["reuse_rejected_profiles"], [])

    def test_cli_rebuilds_stale_reused_candidate_manifest(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            export_script = temp_path / "export_hq_vocal_preview.py"
            output_dir = temp_path / "adaptive"
            candidate_dir = output_dir / "candidates" / "hybrid_tail_preserve"
            source_path.write_bytes(b"source")
            export_script.write_text("# fake export script\n", encoding="utf-8")
            _write_fake_candidate(candidate_dir, "hybrid_tail_preserve")
            stale_manifest_path = candidate_dir / "stem_manifest.json"
            stale_manifest = json.loads(stale_manifest_path.read_text(encoding="utf-8"))
            stale_manifest.update(
                {
                    "audio_path": str(temp_path / "old_source.wav"),
                    "preview_window_sec": {"start": 0.0, "end": 12.0},
                    "tuning": module.default_candidate_profiles()[0].tuning,
                }
            )
            stale_manifest_path.write_text(
                json.dumps(stale_manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            def fake_run_command(command: list[str]) -> None:
                output_arg = command[command.index("--output-dir") + 1]
                output_path = Path(output_arg)
                _write_fake_candidate(output_path, output_path.name)
                manifest_path = output_path / "stem_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest.update(
                    {
                        "audio_path": str(source_path),
                        "source_audio_sha256": module._sha256_file(source_path),
                        "preview_window_sec": {"start": 0.0, "end": 12.0},
                        "tuning": module.default_candidate_profiles()[0].tuning,
                    }
                )
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            argv = [
                str(source_path),
                str(output_dir),
                "--export-script",
                str(export_script),
                "--python-executable",
                sys.executable,
                "--end-sec",
                "12.0",
                "--segment-consistency-sec",
                "0",
                "--profile",
                "hybrid_tail_preserve",
                "--reuse-existing-candidates",
                "--disable-anchor-suppression",
            ]

            with patch.object(module, "ensure_bs_roformer_runtime"), patch.object(
                module,
                "_run_command",
                side_effect=fake_run_command,
            ) as run_command_mock, patch.object(
                module,
                "calculate_vocal_proxy_metrics",
                return_value=_metrics(
                    dropout=0.04,
                    tail_dropout=0.05,
                    rms_ratio_db=-14.0,
                    low_ratio_db=-22.0,
                    kick_ratio_db=-15.0,
                ),
            ):
                self.assertEqual(module.main(argv), 0)

            adaptive_manifest = json.loads(
                (output_dir / "adaptive_vocal_manifest.json").read_text(encoding="utf-8"),
            )

        run_command_mock.assert_called_once()
        self.assertEqual(adaptive_manifest["reuse_rejected_profiles"][0]["name"], "hybrid_tail_preserve")
        self.assertEqual(adaptive_manifest["reuse_rejected_profiles"][0]["reason"], "source_audio_path_mismatch")

    def test_cli_rebuilds_dependent_candidate_when_upstream_was_rebuilt(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            export_script = temp_path / "export_hq_vocal_preview.py"
            output_dir = temp_path / "adaptive"
            source_path.write_bytes(b"source")
            export_script.write_text("# fake export script\n", encoding="utf-8")
            for profile in module.default_candidate_profiles():
                candidate_dir = output_dir / "candidates" / profile.name
                _write_fake_candidate(candidate_dir, profile.name)
                manifest_path = candidate_dir / "stem_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest.update(
                    {
                        "audio_path": str(source_path),
                        "source_audio_sha256": module._sha256_file(source_path),
                        "preview_window_sec": {"start": 0.0, "end": 12.0},
                        "tuning": profile.tuning,
                    }
                )
                manifest["pipeline"]["candidate_mode"] = profile.candidate_mode
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            stale_profile = module.default_candidate_profiles()[2]
            stale_manifest_path = output_dir / "candidates" / stale_profile.name / "stem_manifest.json"
            stale_manifest = json.loads(stale_manifest_path.read_text(encoding="utf-8"))
            stale_manifest["audio_path"] = str(temp_path / "old_source.wav")
            stale_manifest_path.write_text(json.dumps(stale_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            stale_dependent_dir = output_dir / "candidates" / "ft_tail_safe_anchor_clean"
            _write_fake_candidate(stale_dependent_dir, "ft_tail_safe_anchor_clean")
            stale_dependent_manifest_path = stale_dependent_dir / "stem_manifest.json"
            stale_dependent_manifest = json.loads(stale_dependent_manifest_path.read_text(encoding="utf-8"))
            stale_dependent_manifest.update(
                {
                    "audio_path": str(source_path),
                    "source_audio_sha256": module._sha256_file(source_path),
                    "preview_window_sec": {"start": 0.0, "end": 12.0},
                    "pipeline": {"candidate_mode": "anchor_suppressed_vocals"},
                    "tuning": module.default_anchor_suppression_profiles()[0].tuning,
                }
            )
            stale_dependent_manifest_path.write_text(
                json.dumps(stale_dependent_manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            def fake_run_command(command: list[str]) -> None:
                output_arg = command[command.index("--output-dir") + 1]
                output_path = Path(output_arg)
                _write_fake_candidate(output_path, output_path.name)
                manifest_path = output_path / "stem_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                profile = next(item for item in module.default_candidate_profiles() if item.name == output_path.name)
                manifest.update(
                    {
                        "audio_path": str(source_path),
                        "source_audio_sha256": module._sha256_file(source_path),
                        "preview_window_sec": {"start": 0.0, "end": 12.0},
                        "tuning": profile.tuning,
                    }
                )
                manifest["pipeline"]["candidate_mode"] = profile.candidate_mode
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            def fake_anchor_candidate(*, profile, output_dir: Path, **_kwargs):
                _write_fake_candidate(output_dir, profile.name)
                manifest_path = output_dir / "stem_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest.update(
                    {
                        "audio_path": str(source_path),
                        "source_audio_sha256": module._sha256_file(source_path),
                        "preview_window_sec": {"start": 0.0, "end": 12.0},
                        "pipeline": {"candidate_mode": "anchor_suppressed_vocals"},
                        "tuning": profile.tuning,
                    }
                )
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return manifest

            argv = [
                str(source_path),
                str(output_dir),
                "--export-script",
                str(export_script),
                "--python-executable",
                sys.executable,
                "--end-sec",
                "12.0",
                "--segment-consistency-sec",
                "0",
                "--reuse-existing-candidates",
                "--disable-body-repair",
                "--disable-segment-router",
            ]

            with patch.object(module, "ensure_bs_roformer_runtime"), patch.object(
                module,
                "_run_command",
                side_effect=fake_run_command,
            ) as run_command_mock, patch.object(
                module,
                "build_anchor_suppressed_candidate",
                side_effect=fake_anchor_candidate,
            ) as anchor_builder_mock, patch.object(
                module,
                "calculate_vocal_proxy_metrics",
                return_value=_metrics(
                    dropout=0.04,
                    tail_dropout=0.05,
                    rms_ratio_db=-14.0,
                    low_ratio_db=-22.0,
                    kick_ratio_db=-15.0,
                ),
            ), patch.object(
                module,
                "calculate_anchor_support_metrics",
                return_value={
                    "unsupported_vs_anchor_ratio_db": -40.0,
                    "high_unsupported_vs_anchor_ratio_db": -40.0,
                    "high_noise_flatness_6000_16000": 0.05,
                    "stereo_side_mid_ratio_db": -16.0,
                },
            ):
                self.assertEqual(module.main(argv), 0)

            adaptive_manifest = json.loads(
                (output_dir / "adaptive_vocal_manifest.json").read_text(encoding="utf-8"),
            )

        run_command_mock.assert_called_once()
        anchor_builder_mock.assert_called()
        self.assertIn(
            {
                "name": "ft_tail_safe_anchor_clean",
                "reason": "upstream_candidate_rebuilt",
                "manifest_path": str(stale_dependent_manifest_path),
            },
            adaptive_manifest["reuse_rejected_profiles"],
        )

    def test_listen_pack_includes_ranked_candidate_copies(self):
        module = _load_adaptive_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            listen_dir = temp_path / "listen"
            first_vocals = temp_path / "first.flac"
            second_vocals = temp_path / "second.flac"
            audio = np.zeros((1024, 2), dtype=np.float32)
            sf.write(source_path, audio, 44100)
            first_vocals.write_bytes(b"first")
            second_vocals.write_bytes(b"second")
            listen_dir.mkdir(parents=True, exist_ok=True)
            (listen_dir / "candidate_99_stale.flac").write_bytes(b"stale")
            candidate_scores = [
                module.VocalCandidateScore(
                    profile_name="second",
                    score=0.2,
                    reasons=[],
                    metrics={},
                    manifest_path="second.json",
                    output_dir="second",
                    vocals_path=str(second_vocals),
                ),
                module.VocalCandidateScore(
                    profile_name="first",
                    score=0.9,
                    reasons=[],
                    metrics={},
                    manifest_path="first.json",
                    output_dir="first",
                    vocals_path=str(first_vocals),
                ),
            ]

            module.write_listen_pack(
                source_audio_path=source_path,
                selected_vocals_path=first_vocals,
                candidate_scores=candidate_scores,
                listen_dir=listen_dir,
                selected_label="01_adaptive_selected_vocals.flac",
            )

            candidate_map = json.loads((listen_dir / "candidate_map.json").read_text(encoding="utf-8"))
            first_copy_exists = (listen_dir / "candidate_01_first.flac").exists()
            second_copy_exists = (listen_dir / "candidate_02_second.flac").exists()
            stale_copy_exists = (listen_dir / "candidate_99_stale.flac").exists()

        self.assertTrue(first_copy_exists)
        self.assertTrue(second_copy_exists)
        self.assertFalse(stale_copy_exists)
        self.assertEqual(candidate_map["candidate_01_first.flac"], str(first_vocals))

    def test_anchor_guided_suppression_reduces_unsupported_high_band_leakage(self):
        module = _load_adaptive_module()
        sample_rate = 44100
        t = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
        voice = (0.35 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
        leakage = (0.22 * np.sin(2.0 * np.pi * 7200.0 * t)).astype(np.float32)
        source = np.stack([voice + leakage, voice + leakage], axis=1).astype(np.float32)
        anchor = np.stack([voice, voice], axis=1).astype(np.float32)
        profile = module.default_anchor_suppression_profiles()[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.flac"
            anchor_path = temp_path / "anchor.flac"
            output_path = temp_path / "cleaned.flac"
            sf.write(source_path, source, sample_rate)
            sf.write(anchor_path, anchor, sample_rate)

            debug = module.apply_anchor_guided_leakage_suppression(
                source_vocals_path=source_path,
                anchor_vocals_path=anchor_path,
                output_vocals_path=output_path,
                tuning=profile.tuning,
            )
            cleaned, cleaned_sr = sf.read(output_path, always_2d=True)

        self.assertEqual(cleaned_sr, sample_rate)
        self.assertLess(
            _band_rms(cleaned, sample_rate, 6500.0, 8500.0),
            0.75 * _band_rms(source, sample_rate, 6500.0, 8500.0),
        )
        self.assertGreater(
            _band_rms(cleaned, sample_rate, 180.0, 300.0),
            0.45 * _band_rms(source, sample_rate, 180.0, 300.0),
        )
        self.assertEqual(debug["sample_rate"], sample_rate)

    def test_consensus_vocal_body_repair_fills_core_dip_without_boosting_high_band(self):
        module = _load_adaptive_module()
        sample_rate = 44100
        t = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
        voice = (0.34 * np.sin(2.0 * np.pi * 240.0 * t)).astype(np.float32)
        high_leak = (0.20 * np.sin(2.0 * np.pi * 7600.0 * t)).astype(np.float32)
        base_mono = voice.copy()
        dip_start = sample_rate // 3
        dip_end = sample_rate // 2
        base_mono[dip_start:dip_end] *= 0.35
        base = np.stack([base_mono, base_mono], axis=1).astype(np.float32)
        donor_one = np.stack([voice + high_leak, voice + high_leak], axis=1).astype(np.float32)
        donor_two = np.stack([voice, voice], axis=1).astype(np.float32)
        donor_three = np.stack([voice + 0.5 * high_leak, voice + 0.5 * high_leak], axis=1).astype(np.float32)
        profile = module.default_vocal_body_repair_profiles()[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            base_path = temp_path / "base.flac"
            donor_one_path = temp_path / "donor_one.flac"
            donor_two_path = temp_path / "donor_two.flac"
            donor_three_path = temp_path / "donor_three.flac"
            output_path = temp_path / "repaired.flac"
            sf.write(base_path, base, sample_rate)
            sf.write(donor_one_path, donor_one, sample_rate)
            sf.write(donor_two_path, donor_two, sample_rate)
            sf.write(donor_three_path, donor_three, sample_rate)

            debug = module.apply_consensus_vocal_body_repair(
                base_vocals_path=base_path,
                donor_vocals_paths=(donor_one_path, donor_two_path, donor_three_path),
                output_vocals_path=output_path,
                tuning=profile.tuning,
            )
            repaired, repaired_sr = sf.read(output_path, always_2d=True)

        self.assertEqual(repaired_sr, sample_rate)
        self.assertGreater(
            _band_rms(repaired[dip_start:dip_end], sample_rate, 180.0, 320.0),
            1.15 * _band_rms(base[dip_start:dip_end], sample_rate, 180.0, 320.0),
        )
        self.assertLess(
            _band_rms(repaired, sample_rate, 7000.0, 8200.0),
            1.15 * max(_band_rms(base, sample_rate, 7000.0, 8200.0), 1e-6),
        )
        self.assertGreater(debug["fill_mask_mean"], 0.0)

    def test_phase_safe_vocal_body_repair_preserves_base_phase_and_core_body(self):
        module = _load_adaptive_module()
        sample_rate = 44100
        t = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
        voice = (0.34 * np.sin(2.0 * np.pi * 240.0 * t)).astype(np.float32)
        shifted_voice = (0.34 * np.sin(2.0 * np.pi * 240.0 * t + np.pi / 2.0)).astype(np.float32)
        high_leak = (0.20 * np.sin(2.0 * np.pi * 7600.0 * t)).astype(np.float32)
        dip_start = sample_rate // 3
        dip_end = sample_rate // 2
        base_mono = voice.copy()
        base_mono[dip_start:dip_end] *= 0.35
        base = np.stack([base_mono, base_mono], axis=1).astype(np.float32)
        donor_one = np.stack([shifted_voice + high_leak, shifted_voice + high_leak], axis=1).astype(np.float32)
        donor_two = np.stack([voice, voice], axis=1).astype(np.float32)
        donor_three = np.stack([voice + 0.5 * high_leak, voice + 0.5 * high_leak], axis=1).astype(np.float32)
        profile = module.default_vocal_body_repair_profiles()[1]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            base_path = temp_path / "base.flac"
            donor_one_path = temp_path / "donor_one.flac"
            donor_two_path = temp_path / "donor_two.flac"
            donor_three_path = temp_path / "donor_three.flac"
            output_path = temp_path / "repaired.flac"
            sf.write(base_path, base, sample_rate)
            sf.write(donor_one_path, donor_one, sample_rate)
            sf.write(donor_two_path, donor_two, sample_rate)
            sf.write(donor_three_path, donor_three, sample_rate)

            debug = module.apply_consensus_vocal_body_repair(
                base_vocals_path=base_path,
                donor_vocals_paths=(donor_one_path, donor_two_path, donor_three_path),
                output_vocals_path=output_path,
                tuning=profile.tuning,
            )
            repaired, repaired_sr = sf.read(output_path, always_2d=True)

        self.assertEqual(repaired_sr, sample_rate)
        self.assertEqual(debug["phase_strategy"], "base")
        self.assertGreater(
            _band_rms(repaired[dip_start:dip_end], sample_rate, 180.0, 320.0),
            1.18 * _band_rms(base[dip_start:dip_end], sample_rate, 180.0, 320.0),
        )
        self.assertLess(
            _band_rms(repaired, sample_rate, 7000.0, 8200.0),
            1.15 * max(_band_rms(base, sample_rate, 7000.0, 8200.0), 1e-6),
        )
        self.assertGreater(
            np.corrcoef(
                repaired[dip_start:dip_end, 0],
                base[dip_start:dip_end, 0],
            )[0, 1],
            0.95,
        )

    def test_segment_adaptive_router_repairs_drop_without_global_high_boost(self):
        module = _load_adaptive_module()
        sample_rate = 16000
        t = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
        voice_body = (0.30 * np.sin(2.0 * np.pi * 240.0 * t)).astype(np.float32)
        voice_presence = (0.10 * np.sin(2.0 * np.pi * 1250.0 * t)).astype(np.float32)
        high_noise = (0.12 * np.sin(2.0 * np.pi * 7000.0 * t)).astype(np.float32)
        focus_start = int(0.80 * sample_rate)
        focus_end = int(1.20 * sample_rate)
        focus_slice = slice(focus_start, focus_end)
        pre_slice = slice(int(0.10 * sample_rate), int(0.50 * sample_rate))

        full_voice = voice_body + voice_presence
        clean_mono = full_voice.copy()
        clean_mono[focus_slice] *= 0.35
        clean_mono = (clean_mono + high_noise).astype(np.float32)
        strict_mono = full_voice.copy()
        strict_mono[focus_slice] *= 0.35
        repair_mono = (full_voice + high_noise).astype(np.float32)
        preserve_mono = (full_voice + 1.20 * high_noise).astype(np.float32)
        anchor_mono = full_voice.astype(np.float32)
        raw_mono = (full_voice + 0.18 * high_noise).astype(np.float32)

        clean = np.stack([clean_mono, clean_mono], axis=1).astype(np.float32)
        strict = np.stack([strict_mono, strict_mono], axis=1).astype(np.float32)
        repair = np.stack([repair_mono, repair_mono], axis=1).astype(np.float32)
        preserve = np.stack([preserve_mono, preserve_mono], axis=1).astype(np.float32)
        anchor = np.stack([anchor_mono, anchor_mono], axis=1).astype(np.float32)
        raw = np.stack([raw_mono, raw_mono], axis=1).astype(np.float32)
        profile = module.default_segment_router_profiles()[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_path = temp_path / "raw.flac"
            clean_path = temp_path / "clean.flac"
            strict_path = temp_path / "strict.flac"
            repair_path = temp_path / "repair.flac"
            preserve_path = temp_path / "preserve.flac"
            anchor_path = temp_path / "anchor.flac"
            output_path = temp_path / "routed.flac"
            sf.write(raw_path, raw, sample_rate)
            sf.write(clean_path, clean, sample_rate)
            sf.write(strict_path, strict, sample_rate)
            sf.write(repair_path, repair, sample_rate)
            sf.write(preserve_path, preserve, sample_rate)
            sf.write(anchor_path, anchor, sample_rate)

            debug = module.apply_segment_adaptive_vocal_router(
                source_audio_path=raw_path,
                start_sec=0.0,
                end_sec=2.0,
                clean_vocals_path=clean_path,
                strict_vocals_path=strict_path,
                repair_vocals_path=repair_path,
                preserve_vocals_path=preserve_path,
                anchor_vocals_path=anchor_path,
                output_vocals_path=output_path,
                tuning=profile.tuning,
            )
            routed, routed_sr = sf.read(output_path, always_2d=True)

        self.assertEqual(routed_sr, sample_rate)
        self.assertEqual(debug["phase_strategy"], "clean_base")
        self.assertGreater(debug["body_route_mean"], 0.0)
        self.assertGreater(debug["strict_route_mean"], 0.0)
        self.assertGreater(
            _band_rms(routed[focus_slice], sample_rate, 180.0, 320.0),
            1.15 * _band_rms(clean[focus_slice], sample_rate, 180.0, 320.0),
        )
        self.assertLess(
            _band_rms(routed[pre_slice], sample_rate, 180.0, 320.0),
            1.08 * _band_rms(clean[pre_slice], sample_rate, 180.0, 320.0),
        )
        self.assertLess(
            _band_rms(routed, sample_rate, 6600.0, 7400.0),
            0.90 * _band_rms(clean, sample_rate, 6600.0, 7400.0),
        )
        self.assertLessEqual(float(np.max(np.abs(routed))), 0.99901)

    def test_busy_vocal_supported_router_recovers_partial_anchor_drop_more_than_safe_router(self):
        module = _load_adaptive_module()
        sample_rate = 16000
        t = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
        voice_body = (0.30 * np.sin(2.0 * np.pi * 240.0 * t)).astype(np.float32)
        voice_presence = (0.10 * np.sin(2.0 * np.pi * 1250.0 * t)).astype(np.float32)
        high_noise = (0.12 * np.sin(2.0 * np.pi * 7000.0 * t)).astype(np.float32)
        focus_slice = slice(int(0.80 * sample_rate), int(1.20 * sample_rate))

        full_voice = (voice_body + voice_presence).astype(np.float32)
        clean_mono = full_voice.copy()
        clean_mono[focus_slice] *= 0.32
        strict_mono = full_voice.copy()
        strict_mono[focus_slice] *= 0.42
        anchor_mono = full_voice.copy()
        anchor_mono[focus_slice] *= 0.42
        repair_mono = (full_voice + high_noise).astype(np.float32)
        preserve_mono = (full_voice + 1.15 * high_noise).astype(np.float32)
        raw_mono = (full_voice + 0.20 * high_noise).astype(np.float32)

        clean = np.stack([clean_mono, clean_mono], axis=1).astype(np.float32)
        strict = np.stack([strict_mono, strict_mono], axis=1).astype(np.float32)
        repair = np.stack([repair_mono, repair_mono], axis=1).astype(np.float32)
        preserve = np.stack([preserve_mono, preserve_mono], axis=1).astype(np.float32)
        anchor = np.stack([anchor_mono, anchor_mono], axis=1).astype(np.float32)
        raw = np.stack([raw_mono, raw_mono], axis=1).astype(np.float32)
        safe_profile, supported_profile, *_ = module.default_segment_router_profiles()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_path = temp_path / "raw.flac"
            clean_path = temp_path / "clean.flac"
            strict_path = temp_path / "strict.flac"
            repair_path = temp_path / "repair.flac"
            preserve_path = temp_path / "preserve.flac"
            anchor_path = temp_path / "anchor.flac"
            safe_output_path = temp_path / "safe.flac"
            supported_output_path = temp_path / "supported.flac"
            sf.write(raw_path, raw, sample_rate)
            sf.write(clean_path, clean, sample_rate)
            sf.write(strict_path, strict, sample_rate)
            sf.write(repair_path, repair, sample_rate)
            sf.write(preserve_path, preserve, sample_rate)
            sf.write(anchor_path, anchor, sample_rate)

            safe_debug = module.apply_segment_adaptive_vocal_router(
                source_audio_path=raw_path,
                start_sec=0.0,
                end_sec=2.0,
                clean_vocals_path=clean_path,
                strict_vocals_path=strict_path,
                repair_vocals_path=repair_path,
                preserve_vocals_path=preserve_path,
                anchor_vocals_path=anchor_path,
                output_vocals_path=safe_output_path,
                tuning=safe_profile.tuning,
            )
            supported_debug = module.apply_segment_adaptive_vocal_router(
                source_audio_path=raw_path,
                start_sec=0.0,
                end_sec=2.0,
                clean_vocals_path=clean_path,
                strict_vocals_path=strict_path,
                repair_vocals_path=repair_path,
                preserve_vocals_path=preserve_path,
                anchor_vocals_path=anchor_path,
                output_vocals_path=supported_output_path,
                tuning=supported_profile.tuning,
            )
            safe_routed, safe_sr = sf.read(safe_output_path, always_2d=True)
            supported_routed, supported_sr = sf.read(supported_output_path, always_2d=True)

        self.assertEqual(safe_sr, sample_rate)
        self.assertEqual(supported_sr, sample_rate)
        self.assertGreater(supported_debug["body_route_mean"], safe_debug["body_route_mean"])
        self.assertGreater(
            _band_rms(supported_routed[focus_slice], sample_rate, 180.0, 320.0),
            1.18 * _band_rms(safe_routed[focus_slice], sample_rate, 180.0, 320.0),
        )
        self.assertLess(
            _band_rms(supported_routed, sample_rate, 6600.0, 7400.0),
            0.92 * _band_rms(repair, sample_rate, 6600.0, 7400.0),
        )
        self.assertLessEqual(float(np.max(np.abs(supported_routed))), 0.99901)

    def test_segment_router_core_debleed_reduces_unsupported_mid_band_leakage(self):
        module = _load_adaptive_module()
        sample_rate = 16000
        t = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
        voice_body = (0.30 * np.sin(2.0 * np.pi * 240.0 * t)).astype(np.float32)
        voice_presence = (0.08 * np.sin(2.0 * np.pi * 1300.0 * t)).astype(np.float32)
        kick_bleed = (0.24 * np.sin(2.0 * np.pi * 720.0 * t)).astype(np.float32)
        focus_slice = slice(int(0.60 * sample_rate), int(1.40 * sample_rate))

        strict_mono = (voice_body + voice_presence).astype(np.float32)
        clean_mono = strict_mono.copy()
        clean_mono[focus_slice] += kick_bleed[focus_slice]
        raw_mono = strict_mono.copy()
        raw_mono[focus_slice] += 1.35 * kick_bleed[focus_slice]

        clean = np.stack([clean_mono, clean_mono], axis=1).astype(np.float32)
        strict = np.stack([strict_mono, strict_mono], axis=1).astype(np.float32)
        repair = clean.copy()
        preserve = clean.copy()
        anchor = strict.copy()
        raw = np.stack([raw_mono, raw_mono], axis=1).astype(np.float32)
        profile = module.default_segment_issue_repair_profile()
        tuning = dict(profile.tuning)
        tuning["body_repair_strength"] = 0.0
        tuning["preserve_tail_strength"] = 0.0
        tuning["core_debleed_strength"] = 0.80
        tuning["core_debleed_low_hz"] = 150.0
        tuning["core_debleed_high_hz"] = 1200.0

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_path = temp_path / "raw.flac"
            clean_path = temp_path / "clean.flac"
            strict_path = temp_path / "strict.flac"
            repair_path = temp_path / "repair.flac"
            preserve_path = temp_path / "preserve.flac"
            anchor_path = temp_path / "anchor.flac"
            output_path = temp_path / "routed.flac"
            sf.write(raw_path, raw, sample_rate)
            sf.write(clean_path, clean, sample_rate)
            sf.write(strict_path, strict, sample_rate)
            sf.write(repair_path, repair, sample_rate)
            sf.write(preserve_path, preserve, sample_rate)
            sf.write(anchor_path, anchor, sample_rate)

            debug = module.apply_segment_adaptive_vocal_router(
                source_audio_path=raw_path,
                start_sec=0.0,
                end_sec=2.0,
                clean_vocals_path=clean_path,
                strict_vocals_path=strict_path,
                repair_vocals_path=repair_path,
                preserve_vocals_path=preserve_path,
                anchor_vocals_path=anchor_path,
                output_vocals_path=output_path,
                tuning=tuning,
            )
            routed, routed_sr = sf.read(output_path, always_2d=True)

        self.assertEqual(routed_sr, sample_rate)
        self.assertGreater(debug["core_debleed_route_mean"], 0.0)
        self.assertLess(
            _band_rms(routed[focus_slice], sample_rate, 650.0, 790.0),
            0.70 * _band_rms(clean[focus_slice], sample_rate, 650.0, 790.0),
        )
        self.assertGreater(
            _band_rms(routed[focus_slice], sample_rate, 180.0, 320.0),
            0.70 * _band_rms(strict[focus_slice], sample_rate, 180.0, 320.0),
        )
        self.assertLessEqual(float(np.max(np.abs(routed))), 0.99901)

    def test_segment_router_damps_busy_drop_without_anchor_vocal_support(self):
        module = _load_adaptive_module()
        sample_rate = 16000
        t = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
        voice_body = (0.28 * np.sin(2.0 * np.pi * 240.0 * t)).astype(np.float32)
        voice_presence = (0.09 * np.sin(2.0 * np.pi * 1250.0 * t)).astype(np.float32)
        drop_synth = (0.30 * np.sin(2.0 * np.pi * 720.0 * t)).astype(np.float32)
        focus_slice = slice(int(0.80 * sample_rate), int(1.20 * sample_rate))

        clean_mono = voice_body + voice_presence
        clean_mono[focus_slice] *= 0.35
        strict_mono = clean_mono.copy()
        anchor_mono = clean_mono.copy()
        repair_mono = voice_body + voice_presence
        repair_mono[focus_slice] += drop_synth[focus_slice]
        preserve_mono = voice_body + voice_presence
        preserve_mono[focus_slice] += 1.15 * drop_synth[focus_slice]
        raw_mono = voice_body + voice_presence
        raw_mono[focus_slice] += 1.50 * drop_synth[focus_slice]

        clean = np.stack([clean_mono, clean_mono], axis=1).astype(np.float32)
        strict = np.stack([strict_mono, strict_mono], axis=1).astype(np.float32)
        repair = np.stack([repair_mono, repair_mono], axis=1).astype(np.float32)
        preserve = np.stack([preserve_mono, preserve_mono], axis=1).astype(np.float32)
        anchor = np.stack([anchor_mono, anchor_mono], axis=1).astype(np.float32)
        raw = np.stack([raw_mono, raw_mono], axis=1).astype(np.float32)
        profile = module.default_segment_router_profiles()[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_path = temp_path / "raw.flac"
            clean_path = temp_path / "clean.flac"
            strict_path = temp_path / "strict.flac"
            repair_path = temp_path / "repair.flac"
            preserve_path = temp_path / "preserve.flac"
            anchor_path = temp_path / "anchor.flac"
            output_path = temp_path / "routed.flac"
            sf.write(raw_path, raw, sample_rate)
            sf.write(clean_path, clean, sample_rate)
            sf.write(strict_path, strict, sample_rate)
            sf.write(repair_path, repair, sample_rate)
            sf.write(preserve_path, preserve, sample_rate)
            sf.write(anchor_path, anchor, sample_rate)

            debug = module.apply_segment_adaptive_vocal_router(
                source_audio_path=raw_path,
                start_sec=0.0,
                end_sec=2.0,
                clean_vocals_path=clean_path,
                strict_vocals_path=strict_path,
                repair_vocals_path=repair_path,
                preserve_vocals_path=preserve_path,
                anchor_vocals_path=anchor_path,
                output_vocals_path=output_path,
                tuning=profile.tuning,
            )
            routed, routed_sr = sf.read(output_path, always_2d=True)

        self.assertEqual(routed_sr, sample_rate)
        self.assertLessEqual(float(np.max(np.abs(routed))), 0.99901)
        self.assertLess(
            _band_rms(routed[focus_slice], sample_rate, 650.0, 790.0),
            0.00015 * _band_rms(repair[focus_slice], sample_rate, 650.0, 790.0),
        )
        self.assertLess(
            _band_rms(routed[focus_slice], sample_rate, 180.0, 320.0),
            1.10 * _band_rms(clean[focus_slice], sample_rate, 180.0, 320.0),
        )

    def test_busy_vocal_supported_router_damps_unsupported_drop_synth(self):
        module = _load_adaptive_module()
        sample_rate = 16000
        t = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
        voice_body = (0.28 * np.sin(2.0 * np.pi * 240.0 * t)).astype(np.float32)
        voice_presence = (0.09 * np.sin(2.0 * np.pi * 1250.0 * t)).astype(np.float32)
        drop_synth = (0.30 * np.sin(2.0 * np.pi * 720.0 * t)).astype(np.float32)
        focus_slice = slice(int(0.80 * sample_rate), int(1.20 * sample_rate))

        clean_mono = voice_body + voice_presence
        clean_mono[focus_slice] *= 0.35
        strict_mono = clean_mono.copy()
        anchor_mono = clean_mono.copy()
        repair_mono = voice_body + voice_presence
        repair_mono[focus_slice] += drop_synth[focus_slice]
        preserve_mono = voice_body + voice_presence
        preserve_mono[focus_slice] += 1.15 * drop_synth[focus_slice]
        raw_mono = voice_body + voice_presence
        raw_mono[focus_slice] += 1.50 * drop_synth[focus_slice]

        clean = np.stack([clean_mono, clean_mono], axis=1).astype(np.float32)
        strict = np.stack([strict_mono, strict_mono], axis=1).astype(np.float32)
        repair = np.stack([repair_mono, repair_mono], axis=1).astype(np.float32)
        preserve = np.stack([preserve_mono, preserve_mono], axis=1).astype(np.float32)
        anchor = np.stack([anchor_mono, anchor_mono], axis=1).astype(np.float32)
        raw = np.stack([raw_mono, raw_mono], axis=1).astype(np.float32)
        supported_profile = module.default_segment_router_profiles()[1]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_path = temp_path / "raw.flac"
            clean_path = temp_path / "clean.flac"
            strict_path = temp_path / "strict.flac"
            repair_path = temp_path / "repair.flac"
            preserve_path = temp_path / "preserve.flac"
            anchor_path = temp_path / "anchor.flac"
            output_path = temp_path / "routed.flac"
            sf.write(raw_path, raw, sample_rate)
            sf.write(clean_path, clean, sample_rate)
            sf.write(strict_path, strict, sample_rate)
            sf.write(repair_path, repair, sample_rate)
            sf.write(preserve_path, preserve, sample_rate)
            sf.write(anchor_path, anchor, sample_rate)

            debug = module.apply_segment_adaptive_vocal_router(
                source_audio_path=raw_path,
                start_sec=0.0,
                end_sec=2.0,
                clean_vocals_path=clean_path,
                strict_vocals_path=strict_path,
                repair_vocals_path=repair_path,
                preserve_vocals_path=preserve_path,
                anchor_vocals_path=anchor_path,
                output_vocals_path=output_path,
                tuning=supported_profile.tuning,
            )
            routed, routed_sr = sf.read(output_path, always_2d=True)

        self.assertEqual(routed_sr, sample_rate)
        self.assertGreater(debug["body_route_mean"], 0.0)
        self.assertLessEqual(float(np.max(np.abs(routed))), 0.99901)
        self.assertLess(
            _band_rms(routed[focus_slice], sample_rate, 650.0, 790.0),
            0.00025 * _band_rms(repair[focus_slice], sample_rate, 650.0, 790.0),
        )
        self.assertGreater(
            _band_rms(routed[focus_slice], sample_rate, 180.0, 320.0),
            1.15 * _band_rms(clean[focus_slice], sample_rate, 180.0, 320.0),
        )


if __name__ == "__main__":
    unittest.main()
