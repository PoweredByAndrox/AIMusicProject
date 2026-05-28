from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = SCRIPTS_ROOT / "run_ground_truth_quality_gate.py"


def _load_gate_module():
    spec = importlib.util.spec_from_file_location("run_ground_truth_quality_gate", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestRunGroundTruthQualityGate(unittest.TestCase):
    def test_quality_gate_requires_docker_runtime_marker(self):
        module = _load_gate_module()

        class MissingDockerRuntimePath:
            def __init__(self, value):
                self.value = value

            def exists(self):
                return False

        original_path = module.require_docker_runtime.__globals__["Path"]
        module.require_docker_runtime.__globals__["Path"] = MissingDockerRuntimePath
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                "Docker-only.*docker compose run --rm --no-deps -T ml-worker",
            ):
                module.require_docker_runtime(
                    tool_name="run_ground_truth_quality_gate.py",
                    entrypoint_path="/app/scripts/run_ground_truth_quality_gate.py",
                )
        finally:
            module.require_docker_runtime.__globals__["Path"] = original_path

    def test_quality_gate_accepts_docker_runtime_marker(self):
        module = _load_gate_module()

        class PresentDockerRuntimePath:
            def __init__(self, value):
                self.value = value

            def exists(self):
                return self.value == "/.dockerenv"

        original_path = module.require_docker_runtime.__globals__["Path"]
        module.require_docker_runtime.__globals__["Path"] = PresentDockerRuntimePath
        try:
            module.require_docker_runtime(
                tool_name="run_ground_truth_quality_gate.py",
                entrypoint_path="/app/scripts/run_ground_truth_quality_gate.py",
            )
        finally:
            module.require_docker_runtime.__globals__["Path"] = original_path

    def test_main_checks_docker_runtime_before_parsing_args(self):
        module = _load_gate_module()
        calls = []

        def reject_non_docker_runtime(**kwargs):
            calls.append("require_docker_runtime")
            raise RuntimeError("Docker-only test sentinel")

        def fail_if_args_are_parsed(argv):
            raise AssertionError("parse_args must not run before Docker runtime validation")

        original_guard = module.require_docker_runtime
        original_parse_args = module.parse_args
        module.require_docker_runtime = reject_non_docker_runtime
        module.parse_args = fail_if_args_are_parsed
        try:
            with self.assertRaisesRegex(RuntimeError, "Docker-only test sentinel"):
                module.main(["--mode", "synthetic-smoke"])
            self.assertEqual(calls, ["require_docker_runtime"])
        finally:
            module.require_docker_runtime = original_guard
            module.parse_args = original_parse_args

    def test_synthetic_smoke_creates_verified_fixture_and_expected_control_results(self):
        module = _load_gate_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "ground_truth_gate"
            summary = module.run_synthetic_smoke(
                output_dir=output_dir,
                sample_rate=8000,
                duration_seconds=2.0,
                max_offset_seconds=0.0,
                thresholds=module.ObjectiveThresholds(),
            )

            ground_truth_manifest = json.loads(
                (output_dir / "ground_truth" / "ground_truth_manifest.json").read_text(encoding="utf-8")
            )
            perfect_report = json.loads(
                (output_dir / "synthetic_perfect_objective_report.json").read_text(encoding="utf-8")
            )
            degraded_report = json.loads(
                (output_dir / "synthetic_degraded_objective_report.json").read_text(encoding="utf-8")
            )

            self.assertTrue(ground_truth_manifest["ground_truth_contract"]["verified"])
            self.assertEqual(summary["overall"]["status"], "passed")
            self.assertEqual(summary["overall"]["control_gate"], "perfect_passes_degraded_fails")
            self.assertEqual(perfect_report["overall"]["status"], "passed")
            self.assertEqual(degraded_report["overall"]["status"], "failed")
            self.assertGreaterEqual(perfect_report["stems"][0]["reference_likeness"]["score_percent"], 99.0)
            self.assertLess(degraded_report["stems"][0]["reference_likeness"]["score_percent"], 95.0)
            self.assertIn(
                "reference_likeness_below_threshold",
                degraded_report["stems"][0]["verdict"]["failed_check_names"],
            )
            self.assertGreater(perfect_report["stems"][0]["metrics"]["si_sdr_db"], 100.0)
            self.assertIn("sir_db", degraded_report["stems"][0]["metrics"])
            self.assertEqual(perfect_report["ground_truth_contract"]["mode"], "verified_isolated_stems")
            self.assertEqual(perfect_report["delivery"]["status"], "accepted")
            self.assertEqual(perfect_report["delivery"]["forensic_contract_status"], "accepted")
            self.assertEqual(perfect_report["forensic_delivery_contract"]["status"], "accepted")
            self.assertEqual(perfect_report["forensic_delivery_contract"]["uncovered_labels"], [])
            self.assertEqual(degraded_report["delivery"]["status"], "rejected")
            self.assertEqual(degraded_report["delivery"]["reason"], "objective_threshold_failed")
            self.assertEqual(degraded_report["delivery"]["forensic_contract_status"], "rejected")
            self.assertEqual(degraded_report["forensic_delivery_contract"]["status"], "rejected")
            self.assertEqual(degraded_report["forensic_delivery_contract"]["uncovered_labels"], [])
            self.assertGreater(len(degraded_report["forensic_delivery_contract"]["issue_actions"]), 0)
            self.assertEqual(summary["candidates"][1]["delivery_status"], "rejected")

    def test_load_json_accepts_utf8_bom_manifest(self):
        module = _load_gate_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "stem_manifest.json"
            manifest_path.write_text("\ufeff{\"schema_version\": 1}", encoding="utf-8")

            manifest = module.load_json(manifest_path)

        self.assertEqual(manifest["schema_version"], 1)

    def test_failed_candidate_report_includes_per_window_issue_ledger(self):
        module = _load_gate_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "fixture"
            manifests = module.create_synthetic_fixture(output_dir, sample_rate=8000, duration_seconds=6.0)

            report = module.evaluate_candidate_manifest(
                ground_truth_manifest=manifests["ground_truth"],
                candidate_manifest=manifests["synthetic_degraded"],
                output_json=output_dir / "degraded_report.json",
                stem_names=["vocals"],
                sample_rate=8000,
                max_offset_seconds=0.0,
                thresholds=module.ObjectiveThresholds(),
            )

            vocal_report = report["stems"][0]
            ledger = vocal_report["quality_windows"]
            self.assertGreater(ledger["window_count"], 0)
            self.assertLessEqual(len(ledger["worst_windows"]), ledger["window_count"])
            self.assertTrue(ledger["worst_windows"][0]["issue_labels"])
            self.assertIn("reference_likeness", ledger["worst_windows"][0])
            self.assertIn("problem_summary", ledger["worst_windows"][0])
            self.assertIn("low_sdr", ledger["worst_windows"][0]["issue_labels"])
            self.assertIn("micro_quality_windows", vocal_report)
            self.assertEqual(vocal_report["micro_quality_windows"]["schema_version"], 1)
            self.assertGreater(vocal_report["micro_quality_windows"]["analyzed_window_count"], ledger["window_count"])
            self.assertGreater(vocal_report["micro_quality_windows"]["defect_window_count"], 0)
            contract = report["forensic_delivery_contract"]
            self.assertEqual(contract["status"], "rejected")
            self.assertEqual(contract["uncovered_labels"], [])
            self.assertIn("vocals", contract["rejected_stems"])
            self.assertTrue(
                all(action.get("action") for action in contract["issue_actions"]),
                contract["issue_actions"],
            )

    def test_objective_pass_does_not_override_candidate_delivery_block(self):
        module = _load_gate_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "fixture"
            manifests = module.create_synthetic_fixture(output_dir, sample_rate=8000, duration_seconds=2.0)
            perfect_manifest_path = manifests["synthetic_perfect"]
            perfect_manifest = json.loads(perfect_manifest_path.read_text(encoding="utf-8"))
            perfect_manifest["contract"] = {
                "delivery_allowed_without_reference": False,
                "requires_holdout_validation": True,
                "holdout_validation_status": "missing",
                "stores_timestamp_rules": True,
                "target_reference_reconstructive_prior": True,
            }
            perfect_manifest_path.write_text(json.dumps(perfect_manifest), encoding="utf-8")

            report = module.evaluate_candidate_manifest(
                ground_truth_manifest=manifests["ground_truth"],
                candidate_manifest=perfect_manifest_path,
                output_json=output_dir / "blocked_perfect_report.json",
                stem_names=["vocals"],
                sample_rate=8000,
                max_offset_seconds=0.0,
                thresholds=module.ObjectiveThresholds(),
            )

            self.assertEqual(report["stems"][0]["verdict"]["status"], "passed")
            self.assertEqual(report["candidate_manifest_delivery_contract"]["status"], "rejected")
            self.assertEqual(report["overall"]["status"], "failed")
            self.assertEqual(report["delivery"]["status"], "rejected")
            self.assertEqual(report["delivery"]["reason"], "candidate_manifest_delivery_blocked")
            self.assertIn("candidate_manifest", report["overall"]["contract_failed_stems"])

    def test_evaluate_candidate_manifest_writes_waveform_forensic_evidence_pack(self):
        module = _load_gate_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "fixture"
            manifests = module.create_synthetic_fixture(output_dir, sample_rate=8000, duration_seconds=3.0)
            evidence_dir = output_dir / "forensic_evidence"

            report = module.evaluate_candidate_manifest(
                ground_truth_manifest=manifests["ground_truth"],
                candidate_manifest=manifests["synthetic_degraded"],
                output_json=output_dir / "degraded_report.json",
                stem_names=["vocals"],
                sample_rate=8000,
                max_offset_seconds=0.0,
                thresholds=module.ObjectiveThresholds(),
                forensic_evidence_dir=evidence_dir,
            )

            stem_report = report["stems"][0]
            self.assertIn("waveform_forensic_evidence", stem_report)
            evidence = stem_report["waveform_forensic_evidence"]
            self.assertEqual(evidence["status"], "available")
            self.assertEqual(evidence["stem_name"], "vocals")
            for artifact_path in evidence["evidence_artifacts"].values():
                self.assertTrue((evidence_dir / "vocals" / artifact_path).exists(), artifact_path)
            self.assertIn(
                "waveform_forensic_evidence",
                report["forensic_delivery_contract"]["evidence_requirements"]["available_artifacts"],
            )

    def test_evaluate_stem_includes_sample_level_waveform_forensics(self):
        module = _load_gate_module()
        sample_rate = 8000
        stereo_reference_stems = module.build_synthetic_stems(sample_rate=sample_rate, duration_seconds=3.0)
        stereo_candidate_stems = module.build_degraded_candidate(stereo_reference_stems)
        reference_stems = {name: audio.mean(axis=1).astype("float32") for name, audio in stereo_reference_stems.items()}
        candidate_stems = {name: audio.mean(axis=1).astype("float32") for name, audio in stereo_candidate_stems.items()}

        report = module.evaluate_stem(
            stem_name="vocals",
            reference_stems=reference_stems,
            candidate_audio=candidate_stems["vocals"],
            sample_rate=sample_rate,
            max_offset_seconds=0.0,
            thresholds=module.ObjectiveThresholds(),
        )

        self.assertIn("waveform_forensics", report)
        self.assertEqual(report["waveform_forensics"]["schema_version"], 1)
        self.assertEqual(report["waveform_forensics"]["status"], "failed")
        self.assertTrue(report["waveform_forensics"]["defect_ranges"])

    def test_micro_quality_windows_localize_short_transient_defect(self):
        module = _load_gate_module()
        sample_rate = 8000
        frame_count = sample_rate * 2
        time = module.np.arange(frame_count, dtype=module.np.float64) / float(sample_rate)
        reference_drums = module.np.zeros(frame_count, dtype=module.np.float32)
        transient = module.np.hanning(96).astype(module.np.float32)
        transient_start = int(0.750 * sample_rate)
        reference_drums[transient_start : transient_start + len(transient)] = 0.45 * transient
        candidate_drums = reference_drums.copy()
        leak_start = int(0.800 * sample_rate)
        leak_end = int(0.920 * sample_rate)
        candidate_drums[leak_start:leak_end] += (
            0.16 * module.np.sin(2.0 * module.np.pi * 880.0 * time[: leak_end - leak_start])
        ).astype(module.np.float32)
        reference_stems = {
            "drums": reference_drums,
            "melody": module.np.zeros(frame_count, dtype=module.np.float32),
            "vocals": module.np.zeros(frame_count, dtype=module.np.float32),
        }

        report = module.evaluate_stem(
            stem_name="drums",
            reference_stems=reference_stems,
            candidate_audio=candidate_drums,
            sample_rate=sample_rate,
            max_offset_seconds=0.0,
            thresholds=module.ObjectiveThresholds(),
        )

        micro = report["micro_quality_windows"]
        self.assertEqual(micro["schema_version"], 1)
        self.assertGreater(micro["analyzed_window_count"], report["quality_windows"]["window_count"])
        self.assertEqual(micro["hop_ratio"], 0.5)
        top_window = micro["worst_windows"][0]
        self.assertLessEqual(abs(top_window["start_sample"] - leak_start), int(0.150 * sample_rate))
        self.assertIn("micro_residual_spike", top_window["issue_labels"])
        self.assertIn("micro_transient_mismatch", top_window["issue_labels"])

    def test_micro_quality_windows_use_residual_reference_leakage_for_drum_bleed(self):
        module = _load_gate_module()
        sample_rate = 8000
        frame_count = sample_rate * 2
        time = module.np.arange(frame_count, dtype=module.np.float64) / float(sample_rate)
        reference_drums = module.np.zeros(frame_count, dtype=module.np.float32)
        transient = module.np.hanning(128).astype(module.np.float32)
        transient_start = int(1.100 * sample_rate)
        reference_drums[transient_start : transient_start + len(transient)] = 0.95 * transient
        melody = module.np.zeros(frame_count, dtype=module.np.float32)
        leak_start = int(1.120 * sample_rate)
        leak_end = int(1.260 * sample_rate)
        melody[leak_start:leak_end] = (
            0.30 * module.np.sin(2.0 * module.np.pi * 660.0 * time[: leak_end - leak_start])
        ).astype(module.np.float32)
        candidate_drums = reference_drums.copy()
        candidate_drums[leak_start:leak_end] += 0.12 * melody[leak_start:leak_end]
        reference_stems = {
            "drums": reference_drums,
            "melody": melody,
            "vocals": module.np.zeros(frame_count, dtype=module.np.float32),
        }

        report = module.evaluate_stem(
            stem_name="drums",
            reference_stems=reference_stems,
            candidate_audio=candidate_drums,
            sample_rate=sample_rate,
            max_offset_seconds=0.0,
            thresholds=module.ObjectiveThresholds(),
        )

        micro = report["micro_quality_windows"]
        melody_windows = micro["worst_windows_by_residual_leakage_source"]["melody"]
        top_melody_window = melody_windows[0]
        self.assertLessEqual(abs(top_melody_window["start_sample"] - leak_start), int(0.150 * sample_rate))
        self.assertIn("micro_residual_reference_stem_leakage", top_melody_window["issue_labels"])
        self.assertEqual(top_melody_window["metrics"]["max_residual_leakage_source"], "melody")
        self.assertGreater(top_melody_window["metrics"]["max_residual_leakage_strength"], 0.50)

    def test_micro_quality_windows_keep_local_time_bucket_defects_when_global_worst_is_elsewhere(self):
        module = _load_gate_module()
        sample_rate = 8000
        frame_count = sample_rate * 30
        time = module.np.arange(frame_count, dtype=module.np.float64) / float(sample_rate)
        reference_drums = module.np.zeros(frame_count, dtype=module.np.float32)
        transient = module.np.hanning(128).astype(module.np.float32)
        reference_drums[int(1.0 * sample_rate) : int(1.0 * sample_rate) + len(transient)] = 0.95 * transient
        reference_drums[int(12.0 * sample_rate) : int(12.0 * sample_rate) + len(transient)] = 0.80 * transient
        melody = module.np.zeros(frame_count, dtype=module.np.float32)
        local_leak_start = int(12.050 * sample_rate)
        local_leak_end = int(12.180 * sample_rate)
        melody[local_leak_start:local_leak_end] = (
            0.40 * module.np.sin(2.0 * module.np.pi * 550.0 * time[: local_leak_end - local_leak_start])
        ).astype(module.np.float32)
        candidate_drums = reference_drums.copy()
        candidate_drums[int(1.0 * sample_rate) : int(1.0 * sample_rate) + 500] += 0.90
        candidate_drums[local_leak_start:local_leak_end] += 0.08 * melody[local_leak_start:local_leak_end]
        reference_stems = {
            "drums": reference_drums,
            "melody": melody,
            "vocals": module.np.zeros(frame_count, dtype=module.np.float32),
        }

        report = module.evaluate_stem(
            stem_name="drums",
            reference_stems=reference_stems,
            candidate_audio=candidate_drums,
            sample_rate=sample_rate,
            max_offset_seconds=0.0,
            thresholds=module.ObjectiveThresholds(),
        )

        bucket_windows = report["micro_quality_windows"]["worst_windows_by_time_bucket"]["10.000-20.000"]
        self.assertTrue(
            any(abs(window["start_sample"] - local_leak_start) <= int(0.150 * sample_rate) for window in bucket_windows)
        )

    def test_micro_quality_windows_flag_quiet_sustained_residual_leakage_without_transient_spike(self):
        module = _load_gate_module()
        sample_rate = 8000
        frame_count = sample_rate * 2
        time = module.np.arange(frame_count, dtype=module.np.float64) / float(sample_rate)
        reference_drums = (0.25 * module.np.sin(2.0 * module.np.pi * 110.0 * time)).astype(module.np.float32)
        melody = (0.35 * module.np.sin(2.0 * module.np.pi * 440.0 * time)).astype(module.np.float32)
        candidate_drums = reference_drums + (0.001 * melody).astype(module.np.float32)
        reference_stems = {
            "drums": reference_drums,
            "melody": melody,
            "vocals": module.np.zeros(frame_count, dtype=module.np.float32),
        }

        report = module.evaluate_stem(
            stem_name="drums",
            reference_stems=reference_stems,
            candidate_audio=candidate_drums,
            sample_rate=sample_rate,
            max_offset_seconds=0.0,
            thresholds=module.ObjectiveThresholds(),
        )

        micro = report["micro_quality_windows"]
        melody_windows = micro["worst_windows_by_residual_leakage_source"]["melody"]
        top_melody_window = melody_windows[0]
        self.assertIn("micro_residual_reference_stem_leakage", top_melody_window["issue_labels"])
        self.assertNotIn("micro_residual_spike", top_melody_window["issue_labels"])
        self.assertNotIn("micro_transient_mismatch", top_melody_window["issue_labels"])

    def test_phrase_quality_windows_flag_drop_melody_between_kicks(self):
        module = _load_gate_module()
        sample_rate = 8000
        frame_count = sample_rate * 8
        time = module.np.arange(frame_count, dtype=module.np.float64) / float(sample_rate)
        reference_drums = module.np.zeros(frame_count, dtype=module.np.float32)
        kick = module.np.hanning(160).astype(module.np.float32)
        for kick_sec in (0.50, 2.50, 4.50, 6.50):
            kick_start = int(kick_sec * sample_rate)
            reference_drums[kick_start : kick_start + len(kick)] += 0.85 * kick
        melody = module.np.zeros(frame_count, dtype=module.np.float32)
        leak_start = int(1.200 * sample_rate)
        leak_end = int(5.700 * sample_rate)
        melody[leak_start:leak_end] = (
            0.32 * module.np.sin(2.0 * module.np.pi * 523.25 * time[: leak_end - leak_start])
        ).astype(module.np.float32)
        candidate_drums = reference_drums.copy()
        candidate_drums[leak_start:leak_end] += 0.18 * melody[leak_start:leak_end]
        reference_stems = {
            "drums": reference_drums,
            "melody": melody,
            "vocals": module.np.zeros(frame_count, dtype=module.np.float32),
        }

        report = module.evaluate_stem(
            stem_name="drums",
            reference_stems=reference_stems,
            candidate_audio=candidate_drums,
            sample_rate=sample_rate,
            max_offset_seconds=0.0,
            thresholds=module.ObjectiveThresholds(),
        )

        micro = report["micro_quality_windows"]
        self.assertIn("bar_spanning_drop_melody_leakage", micro["issue_label_counts"])
        phrase_windows = micro["phrase_level_windows"]
        self.assertTrue(phrase_windows)
        phrase_bucket_windows = micro["phrase_level_windows_by_time_bucket"]["0.000-10.000"]
        self.assertTrue(
            any(
                "bar_spanning_drop_melody_leakage" in window["issue_labels"]
                and window["metrics"]["phrase_leakage_source"] == "melody"
                for window in phrase_bucket_windows
            )
        )
        top_phrase_window = phrase_windows[0]
        self.assertIn("bar_spanning_drop_melody_leakage", top_phrase_window["issue_labels"])
        self.assertEqual(top_phrase_window["metrics"]["phrase_leakage_source"], "melody")
        self.assertGreaterEqual(top_phrase_window["scale_window_seconds"], 1.0)
        self.assertLessEqual(top_phrase_window["start_sample"], leak_start + int(0.750 * sample_rate))
        self.assertGreaterEqual(top_phrase_window["end_sample"], leak_start + int(1.750 * sample_rate))
        self.assertIn(
            "bar_spanning_drop_melody_leakage",
            report["forensic_delivery_contract"]["blocking_issue_labels"],
        )

    def test_phrase_quality_windows_do_not_flag_clean_drum_resonance(self):
        module = _load_gate_module()
        sample_rate = 8000
        frame_count = sample_rate * 6
        time = module.np.arange(frame_count, dtype=module.np.float64) / float(sample_rate)
        kick_body = (0.18 * module.np.sin(2.0 * module.np.pi * 92.0 * time)).astype(module.np.float32)
        kick_body *= module.np.exp(-0.25 * time).astype(module.np.float32)
        reference_drums = kick_body.copy()
        candidate_drums = reference_drums.copy()
        reference_stems = {
            "drums": reference_drums,
            "melody": module.np.zeros(frame_count, dtype=module.np.float32),
            "vocals": module.np.zeros(frame_count, dtype=module.np.float32),
        }

        report = module.evaluate_stem(
            stem_name="drums",
            reference_stems=reference_stems,
            candidate_audio=candidate_drums,
            sample_rate=sample_rate,
            max_offset_seconds=0.0,
            thresholds=module.ObjectiveThresholds(),
        )

        micro = report["micro_quality_windows"]
        self.assertNotIn("bar_spanning_drop_melody_leakage", micro["issue_label_counts"])
        self.assertEqual(micro["phrase_level_windows"], [])
        self.assertEqual(micro["phrase_level_windows_by_time_bucket"], {})

    def test_forensic_contract_uses_all_micro_issue_labels_not_only_global_worst_windows(self):
        module = _load_gate_module()
        contract = module.build_stem_forensic_delivery_contract(
            stem_name="drums",
            verdict={"status": "passed", "failed_check_names": []},
            reference_likeness={"issue_labels": []},
            waveform_forensics={"status": "passed", "defect_ranges": []},
            quality_windows={"worst_windows": []},
            micro_quality_windows={
                "defect_window_count": 128,
                "issue_label_counts": {"micro_residual_reference_stem_leakage": 1},
                "worst_windows": [],
            },
        )

        self.assertEqual(contract["status"], "rejected")
        self.assertIn("micro_residual_reference_stem_leakage", contract["blocking_issue_labels"])

    def test_micro_quality_windows_do_not_flag_exact_candidate_leakage(self):
        module = _load_gate_module()
        sample_rate = 8000
        stereo_reference_stems = module.build_synthetic_stems(sample_rate=sample_rate, duration_seconds=2.0)
        reference_stems = {name: audio.mean(axis=1).astype("float32") for name, audio in stereo_reference_stems.items()}

        report = module.evaluate_stem(
            stem_name="vocals",
            reference_stems=reference_stems,
            candidate_audio=reference_stems["vocals"].copy(),
            sample_rate=sample_rate,
            max_offset_seconds=0.0,
            thresholds=module.ObjectiveThresholds(),
        )

        micro = report["micro_quality_windows"]
        self.assertEqual(micro["defect_window_count"], 0)
        self.assertEqual(micro["worst_windows"], [])

    def test_reference_likeness_explains_shape_spectrum_envelope_transient_and_leakage(self):
        module = _load_gate_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "fixture"
            manifests = module.create_synthetic_fixture(output_dir, sample_rate=8000, duration_seconds=4.0)

            report = module.evaluate_candidate_manifest(
                ground_truth_manifest=manifests["ground_truth"],
                candidate_manifest=manifests["synthetic_degraded"],
                output_json=output_dir / "degraded_report.json",
                stem_names=["vocals"],
                sample_rate=8000,
                max_offset_seconds=0.0,
                thresholds=module.ObjectiveThresholds(),
            )

            likeness = report["stems"][0]["reference_likeness"]
            self.assertEqual(likeness["threshold_percent"], 95.0)
            self.assertLess(likeness["score_percent"], 95.0)
            self.assertEqual(
                set(likeness["components"]),
                {
                    "waveform_shape",
                    "spectrum",
                    "energy_envelope",
                    "transient_envelope",
                    "leakage_cleanliness",
                },
            )
            self.assertIn("reference_likeness_below_threshold", likeness["issue_labels"])
            self.assertTrue(likeness["problem_summary"])

    def test_failed_candidate_report_includes_named_reference_leakage_matrix(self):
        module = _load_gate_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "fixture"
            manifests = module.create_synthetic_fixture(output_dir, sample_rate=8000, duration_seconds=4.0)

            report = module.evaluate_candidate_manifest(
                ground_truth_manifest=manifests["ground_truth"],
                candidate_manifest=manifests["synthetic_degraded"],
                output_json=output_dir / "degraded_report.json",
                stem_names=["vocals", "other"],
                sample_rate=8000,
                max_offset_seconds=0.0,
                thresholds=module.ObjectiveThresholds(),
            )

            vocal_report = next(stem for stem in report["stems"] if stem["stem"] == "vocals")
            leakage_sources = vocal_report["leakage_matrix"]["sources"]
            self.assertIn("other", leakage_sources)
            self.assertGreater(leakage_sources["other"]["projection_gain"], 0.01)
            self.assertEqual(vocal_report["leakage_matrix"]["target_stem"], "vocals")

    def test_leakage_matrix_uses_all_reference_stems_even_when_only_one_stem_is_requested(self):
        module = _load_gate_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "fixture"
            manifests = module.create_synthetic_fixture(output_dir, sample_rate=8000, duration_seconds=4.0)

            report = module.evaluate_candidate_manifest(
                ground_truth_manifest=manifests["ground_truth"],
                candidate_manifest=manifests["synthetic_degraded"],
                output_json=output_dir / "degraded_report.json",
                stem_names=["vocals"],
                sample_rate=8000,
                max_offset_seconds=0.0,
                thresholds=module.ObjectiveThresholds(),
            )

            sources = report["stems"][0]["leakage_matrix"]["sources"]
            self.assertEqual(set(sources), {"vocals", "drums", "bass", "other"})
            self.assertGreater(sources["other"]["projection_gain"], 0.01)

    def test_unverified_ground_truth_manifest_is_rejected(self):
        module = _load_gate_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "ground_truth_manifest.json"
            stem_path = temp_path / "vocals.flac"
            stem_path.write_bytes(b"not decoded in this test")
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "ground_truth_contract": {
                            "verified": False,
                            "verification_method": "none",
                        },
                        "stems": {
                            "vocals": {
                                "path": str(stem_path),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            bundle = module.resolve_stem_bundle(manifest_path)
            with self.assertRaisesRegex(RuntimeError, "verified=true"):
                module.validate_verified_ground_truth(bundle)

    def test_tampered_ground_truth_stem_hash_is_rejected_before_metrics(self):
        module = _load_gate_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "fixture"
            manifests = module.create_synthetic_fixture(output_dir, sample_rate=8000, duration_seconds=2.0)
            ground_truth_manifest = json.loads(manifests["ground_truth"].read_text(encoding="utf-8"))
            vocals_path = Path(ground_truth_manifest["stems"]["vocals"]["path"])
            with vocals_path.open("ab") as handle:
                handle.write(b"tamper")

            with self.assertRaisesRegex(RuntimeError, "sha256 mismatch"):
                module.evaluate_candidate_manifest(
                    ground_truth_manifest=manifests["ground_truth"],
                    candidate_manifest=manifests["synthetic_perfect"],
                    output_json=output_dir / "tampered_report.json",
                    stem_names=module.DEFAULT_STEMS,
                    sample_rate=8000,
                    max_offset_seconds=0.0,
                    thresholds=module.ObjectiveThresholds(),
                )

    def test_normalize_requested_stems_overrides_defaults_and_deduplicates(self):
        module = _load_gate_module()

        self.assertEqual(module.normalize_requested_stems(None), list(module.DEFAULT_STEMS))
        self.assertEqual(module.normalize_requested_stems(["vocals", "drums", "vocals"]), ["vocals", "drums"])

    def test_objective_gate_rejects_derived_instrumental_with_child_target_stems(self):
        module = _load_gate_module()
        bundle = module.StemBundle(
            manifest_path=Path("ground_truth_manifest.json"),
            manifest={},
            stems={
                "vocals": Path("vocals.flac"),
                "drums": Path("drums.flac"),
                "bass": Path("bass.flac"),
                "instrumental": Path("instrumental.flac"),
            },
        )

        with self.assertRaisesRegex(RuntimeError, "derived reference stem.*child target stems"):
            module.validate_objective_stem_selection(bundle, ["instrumental", "drums"])

    def test_objective_gate_rejects_drums_plus_percussive_fx_aux_with_child_target_stems(self):
        module = _load_gate_module()
        bundle = module.StemBundle(
            manifest_path=Path("ground_truth_manifest.json"),
            manifest={
                "target_definitions": {
                    "drums_plus_percussive_fx_aux": {
                        "child_stems": ["drums", "fx"],
                    }
                }
            },
            stems={
                "drums": Path("drums.flac"),
                "fx": Path("fx.flac"),
                "drums_plus_percussive_fx_aux": Path("drums_plus_percussive_fx_aux.flac"),
            },
        )

        with self.assertRaisesRegex(RuntimeError, "derived reference stem.*child target stems"):
            module.validate_objective_stem_selection(bundle, ["drums_plus_percussive_fx_aux", "drums"])

        with self.assertRaisesRegex(RuntimeError, "derived reference stem.*child target stems"):
            module.validate_objective_stem_selection(bundle, ["drums_plus_percussive_fx_aux", "fx"])

        module.validate_objective_stem_selection(bundle, ["drums_plus_percussive_fx_aux"])

    def test_objective_gate_uses_manifest_target_definitions_for_custom_composite_targets(self):
        module = _load_gate_module()
        bundle = module.StemBundle(
            manifest_path=Path("ground_truth_manifest.json"),
            manifest={
                "target_definitions": {
                    "drums_with_room_aux": {
                        "derived": True,
                        "child_stems": ["drums", "fx"],
                    }
                }
            },
            stems={
                "drums": Path("drums.flac"),
                "fx": Path("fx.flac"),
                "drums_with_room_aux": Path("drums_with_room_aux.flac"),
            },
        )

        self.assertNotIn(
            "drums_with_room_aux",
            module.reference_stem_names_for_quality(bundle, requested_stems=[]),
        )
        self.assertIn(
            "drums_with_room_aux",
            module.reference_stem_names_for_quality(bundle, requested_stems=["drums_with_room_aux"]),
        )
        with self.assertRaisesRegex(RuntimeError, "derived reference stem.*child target stems"):
            module.validate_objective_stem_selection(bundle, ["drums_with_room_aux", "drums"])


if __name__ == "__main__":
    unittest.main()
