from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = TOOLS_ROOT / "run_model_zoo_benchmark.py"


def _load_benchmark_module():
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    spec = importlib.util.spec_from_file_location("run_model_zoo_benchmark", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_gate_module():
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    spec = importlib.util.spec_from_file_location(
        "run_ground_truth_quality_gate",
        SCRIPTS_ROOT / "run_ground_truth_quality_gate.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _copy_manifest_with_model_id(source_manifest: Path, target_manifest: Path, model_id: str) -> Path:
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    payload["model"] = {"model_id": model_id}
    target_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target_manifest


def _copy_manifest_with_contract(source_manifest: Path, target_manifest: Path, contract_updates: dict[str, object]) -> Path:
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    contract = payload.setdefault("contract", {})
    contract.update(contract_updates)
    target_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target_manifest


class TestRunModelZooBenchmark(unittest.TestCase):
    def test_accepted_candidate_does_not_report_nonblocking_window_labels_as_failed_checks(self) -> None:
        benchmark = _load_benchmark_module()
        accepted_report = {
            "forensic_delivery_contract": {
                "status": "accepted",
                "blocking_issue_labels": [],
            },
            "stems": [
                {
                    "verdict": {
                        "status": "passed",
                        "failed_check_names": [],
                    },
                    "forensic_delivery_contract": {
                        "status": "accepted",
                        "blocking_issue_labels": [],
                    },
                    "quality_windows": {
                        "worst_windows": [
                            {
                                "severity_score": 1.0,
                                "issue_labels": ["reference_stem_leakage"],
                            }
                        ]
                    },
                    "micro_quality_windows": {
                        "worst_windows": [
                            {
                                "issue_labels": ["micro_residual_spike"],
                            }
                        ]
                    },
                }
            ],
        }

        self.assertEqual(benchmark.extract_failed_check_names(accepted_report), [])

    def test_candidate_summary_rejects_when_exact_reference_audit_is_incomplete(self) -> None:
        benchmark = _load_benchmark_module()
        summary = benchmark.candidate_summary(
            spec=benchmark.CandidateManifestSpec("candidate", Path("/app/candidate/stem_manifest.json")),
            report={
                "delivery": {"status": "accepted", "reason": "objective_threshold_passed"},
                "overall": {"status": "passed", "failed_stems": []},
                "stems": [],
            },
            report_path=Path("/app/out/objective_report.json"),
            exact_reference_audit={
                "status": "partial",
                "delivery_status": "skipped",
                "delivery_reason": "exact_reference_stem_path_missing",
                "output_dir": "/app/out/exact_reference_listen_audits/01_candidate",
                "failed_stems": [],
                "skipped_stems": ["vocals"],
                "stems": {
                    "vocals": {
                        "status": "skipped",
                        "reason": "stem_path_missing",
                    }
                },
            },
        )

        self.assertEqual(summary["delivery_status"], "rejected")
        self.assertFalse(summary["delivery_allowed"])
        self.assertEqual(summary["presentation_status"], "blocked_from_delivery")
        self.assertEqual(summary["ralph_ai_delivery_contract"]["status"], "blocked")
        self.assertFalse(summary["ralph_ai_delivery_contract"]["result_delivery_allowed"])
        self.assertTrue(summary["ralph_ai_delivery_contract"]["rejects_best_of_bad_candidate"])
        self.assertEqual(summary["delivery_reason"], "exact_reference_stem_path_missing")
        self.assertIn("exact_reference_stem_path_missing", summary["failed_check_names"])

    def test_selects_best_accepted_candidate_and_preserves_rejected_report(self) -> None:
        benchmark = _load_benchmark_module()
        quality_gate = _load_gate_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifests = quality_gate.create_synthetic_fixture(root / "fixture", sample_rate=8000, duration_seconds=3.0)

            report = benchmark.run_model_zoo_benchmark(
                ground_truth_manifest=manifests["ground_truth"],
                candidate_manifests=[
                    benchmark.CandidateManifestSpec("degraded_model", manifests["synthetic_degraded"]),
                    benchmark.CandidateManifestSpec("perfect_model", manifests["synthetic_perfect"]),
                ],
                output_dir=root / "benchmark",
                stem_names=["vocals"],
                sample_rate=8000,
                max_offset_seconds=0.0,
                thresholds=quality_gate.ObjectiveThresholds(),
            )

            self.assertEqual(report["overall"]["status"], "accepted_candidate_found")
            self.assertEqual(report["selected_candidate"]["label"], "perfect_model")
            self.assertEqual(report["selected_candidate"]["delivery_status"], "accepted")
            self.assertEqual(report["selected_candidate"]["forensic_contract_status"], "accepted")
            self.assertEqual(report["selected_candidate"]["exact_reference_audit_status"], "available")
            self.assertEqual(report["selected_candidate"]["exact_reference_delivery_status"], "accepted")
            self.assertTrue(report["selected_candidate"]["delivery_allowed"])
            self.assertEqual(report["selected_candidate"]["presentation_status"], "accepted_for_listen_pack")
            self.assertEqual(report["selected_candidate"]["ralph_ai_delivery_contract"]["status"], "accepted")
            self.assertTrue(report["selected_candidate"]["ralph_ai_delivery_contract"]["result_delivery_allowed"])
            self.assertTrue(report["contract"]["ralph_ai_delivery_contract_enforced"])
            self.assertTrue(report["contract"]["best_of_bad_candidate_delivery_forbidden"])
            self.assertTrue(report["contract"]["rejected_candidates_are_diagnostic_only"])
            self.assertEqual(report["ranked_candidates"][0]["label"], "perfect_model")
            self.assertEqual(report["ranked_candidates"][1]["label"], "degraded_model")
            self.assertEqual(report["ranked_candidates"][1]["delivery_status"], "rejected")
            self.assertEqual(report["ranked_candidates"][1]["forensic_contract_status"], "rejected")
            self.assertEqual(report["ranked_candidates"][1]["exact_reference_audit_status"], "available")
            self.assertEqual(report["ranked_candidates"][1]["exact_reference_delivery_status"], "rejected")
            self.assertIn("exact_reference_threshold_failed", report["ranked_candidates"][1]["failed_check_names"])
            self.assertEqual(report["ranked_candidates"][1]["forensic_uncovered_labels"], [])
            self.assertFalse(report["ranked_candidates"][1]["delivery_allowed"])
            self.assertEqual(report["ranked_candidates"][1]["presentation_status"], "blocked_from_delivery")
            self.assertEqual(report["ranked_candidates"][1]["ralph_ai_delivery_contract"]["status"], "blocked")
            self.assertIn(
                "reference_likeness_below_threshold",
                report["ranked_candidates"][1]["failed_check_names"],
            )
            self.assertIn("micro_residual_spike", report["ranked_candidates"][1]["failed_check_names"])
            self.assertFalse(report["contract"]["fallback_used"])
            self.assertTrue(report["contract"]["reference_assisted"])
            self.assertTrue(report["contract"]["acceptance_requires_delivery_accepted"])
            self.assertTrue((root / "benchmark" / "model_zoo_benchmark_manifest.json").exists())
            self.assertTrue(
                (root / "benchmark" / "objective_reports" / "01_degraded_model_objective_report.json").exists()
            )
            self.assertEqual(report["ranked_candidates"][1]["forensic_evidence_status"], "available")
            self.assertEqual(
                report["ranked_candidates"][1]["forensic_evidence_dir"],
                str(root / "benchmark" / "forensic_evidence" / "01_degraded_model"),
            )
            self.assertTrue(
                (
                    root
                    / "benchmark"
                    / "exact_reference_listen_audits"
                    / "01_degraded_model"
                    / "vocals"
                    / "exact_reference_listen_audit.json"
                ).exists()
            )
            degraded_report = json.loads(
                (root / "benchmark" / "objective_reports" / "01_degraded_model_objective_report.json").read_text(
                    encoding="utf-8"
                )
            )
            degraded_evidence = degraded_report["stems"][0]["waveform_forensic_evidence"]
            self.assertEqual(degraded_evidence["status"], "available")
            for artifact_path in degraded_evidence["evidence_artifacts"].values():
                self.assertTrue(
                    (root / "benchmark" / "forensic_evidence" / "01_degraded_model" / "vocals" / artifact_path).exists(),
                    artifact_path,
                )

    def test_require_no_reference_candidates_rejects_assisted_winner(self) -> None:
        benchmark = _load_benchmark_module()
        quality_gate = _load_gate_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifests = quality_gate.create_synthetic_fixture(root / "fixture", sample_rate=8000, duration_seconds=3.0)
            assisted_manifest = _copy_manifest_with_contract(
                manifests["synthetic_perfect"],
                manifests["synthetic_perfect"].with_name("synthetic_perfect_assisted_manifest.json"),
                {
                    "target_reference_assisted": True,
                    "target_reference_used_at_application": True,
                    "application_reference_free": False,
                    "fallback_used": False,
                    "per_track_hardcode_allowed": False,
                    "not_timestamp_hardcoded": True,
                },
            )

            report = benchmark.run_model_zoo_benchmark(
                ground_truth_manifest=manifests["ground_truth"],
                candidate_manifests=[benchmark.CandidateManifestSpec("assisted_perfect", assisted_manifest)],
                output_dir=root / "benchmark",
                stem_names=["vocals"],
                sample_rate=8000,
                max_offset_seconds=0.0,
                thresholds=quality_gate.ObjectiveThresholds(),
                require_no_reference_candidates=True,
                next_candidate_limit=0,
            )

            candidate = report["ranked_candidates"][0]
            self.assertEqual(report["overall"]["status"], "candidate_space_exhausted")
            self.assertIsNone(report["selected_candidate"])
            self.assertEqual(candidate["delivery_status"], "rejected")
            self.assertFalse(candidate["delivery_allowed"])
            self.assertEqual(candidate["presentation_status"], "blocked_from_delivery")
            self.assertEqual(candidate["ralph_ai_delivery_contract"]["status"], "blocked")
            self.assertEqual(candidate["delivery_reason"], "candidate_contract_not_no_reference_safe")
            self.assertEqual(candidate["no_reference_contract"]["status"], "rejected")
            self.assertIn("candidate_contract_not_no_reference_safe", candidate["failed_check_names"])

    def test_require_no_reference_candidates_accepts_explicit_reference_free_winner(self) -> None:
        benchmark = _load_benchmark_module()
        quality_gate = _load_gate_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifests = quality_gate.create_synthetic_fixture(root / "fixture", sample_rate=8000, duration_seconds=3.0)
            reference_free_manifest = _copy_manifest_with_contract(
                manifests["synthetic_perfect"],
                manifests["synthetic_perfect"].with_name("synthetic_perfect_reference_free_manifest.json"),
                {
                    "target_reference_assisted": False,
                    "target_reference_used_at_application": False,
                    "application_reference_free": True,
                    "delivery_allowed_without_reference": True,
                    "fallback_used": False,
                    "per_track_hardcode_allowed": False,
                    "not_timestamp_hardcoded": True,
                },
            )

            report = benchmark.run_model_zoo_benchmark(
                ground_truth_manifest=manifests["ground_truth"],
                candidate_manifests=[benchmark.CandidateManifestSpec("reference_free_perfect", reference_free_manifest)],
                output_dir=root / "benchmark",
                stem_names=["vocals"],
                sample_rate=8000,
                max_offset_seconds=0.0,
                thresholds=quality_gate.ObjectiveThresholds(),
                require_no_reference_candidates=True,
                next_candidate_limit=0,
            )

            candidate = report["selected_candidate"]
            self.assertEqual(report["overall"]["status"], "accepted_candidate_found")
            self.assertIsNotNone(candidate)
            self.assertEqual(candidate["label"], "reference_free_perfect")
            self.assertEqual(candidate["delivery_status"], "accepted")
            self.assertTrue(candidate["delivery_allowed"])
            self.assertEqual(candidate["no_reference_contract"]["status"], "accepted")

    def test_require_no_reference_candidates_rejects_missing_delivery_without_reference_claim(self) -> None:
        benchmark = _load_benchmark_module()
        quality_gate = _load_gate_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifests = quality_gate.create_synthetic_fixture(root / "fixture", sample_rate=8000, duration_seconds=3.0)
            incomplete_manifest = _copy_manifest_with_contract(
                manifests["synthetic_perfect"],
                manifests["synthetic_perfect"].with_name("synthetic_perfect_incomplete_no_ref_manifest.json"),
                {
                    "target_reference_assisted": False,
                    "target_reference_used_at_application": False,
                    "application_reference_free": True,
                    "fallback_used": False,
                    "per_track_hardcode_allowed": False,
                    "not_timestamp_hardcoded": True,
                },
            )

            report = benchmark.run_model_zoo_benchmark(
                ground_truth_manifest=manifests["ground_truth"],
                candidate_manifests=[benchmark.CandidateManifestSpec("incomplete_no_ref", incomplete_manifest)],
                output_dir=root / "benchmark",
                stem_names=["vocals"],
                sample_rate=8000,
                max_offset_seconds=0.0,
                thresholds=quality_gate.ObjectiveThresholds(),
                require_no_reference_candidates=True,
                next_candidate_limit=0,
            )

            candidate = report["ranked_candidates"][0]
            self.assertEqual(candidate["delivery_status"], "rejected")
            self.assertEqual(candidate["no_reference_contract"]["status"], "rejected")
            self.assertIn(
                "delivery_allowed_without_reference",
                {violation["field"] for violation in candidate["no_reference_contract"]["violations"]},
            )

    def test_require_no_reference_candidates_rejects_target_reference_used_flag(self) -> None:
        benchmark = _load_benchmark_module()
        quality_gate = _load_gate_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifests = quality_gate.create_synthetic_fixture(root / "fixture", sample_rate=8000, duration_seconds=3.0)
            tainted_manifest = _copy_manifest_with_contract(
                manifests["synthetic_perfect"],
                manifests["synthetic_perfect"].with_name("synthetic_perfect_target_ref_used_manifest.json"),
                {
                    "target_reference_assisted": False,
                    "target_reference_used": True,
                    "target_reference_used_at_application": False,
                    "application_reference_free": True,
                    "delivery_allowed_without_reference": True,
                    "fallback_used": False,
                    "per_track_hardcode_allowed": False,
                    "not_timestamp_hardcoded": True,
                },
            )

            report = benchmark.run_model_zoo_benchmark(
                ground_truth_manifest=manifests["ground_truth"],
                candidate_manifests=[benchmark.CandidateManifestSpec("target_ref_used", tainted_manifest)],
                output_dir=root / "benchmark",
                stem_names=["vocals"],
                sample_rate=8000,
                max_offset_seconds=0.0,
                thresholds=quality_gate.ObjectiveThresholds(),
                require_no_reference_candidates=True,
                next_candidate_limit=0,
            )

            candidate = report["ranked_candidates"][0]
            self.assertEqual(candidate["delivery_status"], "rejected")
            self.assertFalse(candidate["delivery_allowed"])
            self.assertEqual(candidate["delivery_reason"], "candidate_contract_not_no_reference_safe")
            self.assertEqual(candidate["no_reference_contract"]["status"], "rejected")
            self.assertIn(
                "target_reference_used",
                {violation["field"] for violation in candidate["no_reference_contract"]["violations"]},
            )

    def test_reports_next_action_when_no_candidate_passes_delivery_gate(self) -> None:
        benchmark = _load_benchmark_module()
        quality_gate = _load_gate_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifests = quality_gate.create_synthetic_fixture(root / "fixture", sample_rate=8000, duration_seconds=3.0)

            report = benchmark.run_model_zoo_benchmark(
                ground_truth_manifest=manifests["ground_truth"],
                candidate_manifests=[
                    benchmark.CandidateManifestSpec("degraded_a", manifests["synthetic_degraded"]),
                    benchmark.CandidateManifestSpec("degraded_b", manifests["synthetic_degraded"]),
                ],
                output_dir=root / "benchmark",
                stem_names=["vocals"],
                sample_rate=8000,
                max_offset_seconds=0.0,
                thresholds=quality_gate.ObjectiveThresholds(),
            )

            self.assertEqual(report["overall"]["status"], "no_accepted_candidate")
            self.assertIsNone(report["selected_candidate"])
            self.assertFalse(report["ranked_candidates"][0]["delivery_allowed"])
            self.assertEqual(report["ranked_candidates"][0]["presentation_status"], "blocked_from_delivery")
            self.assertEqual(report["ranked_candidates"][0]["ralph_ai_delivery_contract"]["status"], "blocked")
            self.assertEqual(
                report["next_action"]["action"],
                "generate_registered_model_candidates",
            )
            self.assertTrue(report["next_action"]["candidate_suggestions"])
            self.assertIn("reference_likeness_below_threshold", report["next_action"]["uncovered_labels"])
            self.assertEqual(report["overall"]["accepted_count"], 0)
            self.assertEqual(report["overall"]["rejected_count"], 2)

    def test_scnet_high_quality_rejection_produces_adaptive_repair_plan_before_more_model_suggestions(self) -> None:
        benchmark = _load_benchmark_module()
        scnet_candidate = {
            "label": "scnet_xl_ihf_musdb18",
            "candidate_manifest": "/app/out/scnet/stem_manifest.json",
            "evaluated_model_labels": ["scnet", "scnet_xl_ihf_musdb18"],
            "delivery_status": "rejected",
            "delivery_allowed": False,
            "presentation_status": "blocked_from_delivery",
            "no_reference_contract": {"status": "accepted", "violations": []},
            "forensic_uncovered_labels": [],
            "forensic_issue_actions": [
                {
                    "label": "micro_reference_stem_leakage",
                    "action": "debleed_or_reroute_required",
                    "reason": "short window contains reference-stem leakage",
                    "stems": ["drums"],
                    "evidence_sources": ["micro_quality_windows"],
                },
                {
                    "label": "sample_residual_above_threshold",
                    "action": "repair_candidate_required",
                    "reason": "sample-level residual exceeds forensic threshold",
                    "stems": ["drums"],
                    "evidence_sources": ["waveform_forensics"],
                },
                {
                    "label": "reference_likeness_below_threshold",
                    "action": "reroute_required",
                    "reason": "candidate failed the verified reference-likeness gate",
                    "stems": ["drums"],
                    "evidence_sources": ["thresholds"],
                },
            ],
            "forensic_evidence_status": "available",
            "forensic_evidence_dir": "/app/out/bench/forensic_evidence/01_scnet",
            "mean_reference_likeness_percent": 91.745626,
            "min_reference_likeness_percent": 91.745626,
            "min_sdr_db": 15.849678,
            "min_si_sdr_db": 15.735349,
            "failed_check_names": [
                "micro_reference_stem_leakage",
                "reference_likeness_below_threshold",
                "sample_residual_above_threshold",
            ],
        }

        next_action = benchmark.build_next_action(
            ranked_candidates=[scnet_candidate],
            candidate_suggestions=[
                {
                    "model_id": "htdemucs_6s",
                    "reason": "generic model can rerun low_sdr",
                }
            ],
        )

        self.assertEqual(next_action["action"], "repair_best_ranked_candidate")
        self.assertEqual(next_action["repair_plan"]["base_candidate_label"], "scnet_xl_ihf_musdb18")
        self.assertEqual(next_action["repair_plan"]["base_candidate_manifest"], "/app/out/scnet/stem_manifest.json")
        self.assertEqual(
            next_action["repair_plan"]["target_labels"],
            [
                "micro_reference_stem_leakage",
                "reference_likeness_below_threshold",
                "sample_residual_above_threshold",
            ],
        )
        self.assertFalse(next_action["repair_plan"]["guardrails"]["accepts_rejected_base_as_final"])
        self.assertFalse(next_action["repair_plan"]["guardrails"]["fallback_used"])
        self.assertEqual(next_action["candidate_suggestions"][0]["model_id"], "htdemucs_6s")

    def test_high_quality_no_reference_drum_repair_rejection_can_continue_repair_chain(self) -> None:
        benchmark = _load_benchmark_module()
        tonal_repair_candidate = {
            "label": "scnet_projection_tonal_default",
            "candidate_manifest": "/app/out/tonal/stem_manifest.json",
            "evaluated_model_labels": ["drum_tonal_leakage_repair", "scnet_projection_tonal_default"],
            "delivery_status": "rejected",
            "delivery_allowed": False,
            "presentation_status": "blocked_from_delivery",
            "no_reference_contract": {"status": "accepted", "violations": []},
            "forensic_uncovered_labels": [],
            "forensic_issue_actions": [
                {
                    "label": "micro_reference_stem_leakage",
                    "action": "debleed_or_reroute_required",
                    "reason": "short window contains reference-stem leakage",
                    "stems": ["drums"],
                    "evidence_sources": ["micro_quality_windows"],
                },
                {
                    "label": "micro_transient_mismatch",
                    "action": "repair_candidate_required",
                    "reason": "short attack/transient event differs from reference",
                    "stems": ["drums"],
                    "evidence_sources": ["micro_quality_windows"],
                },
            ],
            "forensic_evidence_status": "available",
            "forensic_evidence_dir": "/app/out/bench/forensic_evidence/01_tonal",
            "mean_reference_likeness_percent": 92.139455,
            "min_reference_likeness_percent": 92.139455,
            "min_sdr_db": 16.174866,
            "min_si_sdr_db": 16.069559,
            "failed_check_names": ["micro_reference_stem_leakage", "micro_transient_mismatch"],
        }

        next_action = benchmark.build_next_action(
            ranked_candidates=[tonal_repair_candidate],
            candidate_suggestions=[{"model_id": "htdemucs_ft", "reason": "generic rerun"}],
        )

        self.assertEqual(next_action["action"], "repair_best_ranked_candidate")
        self.assertEqual(next_action["repair_plan"]["base_candidate_label"], "scnet_projection_tonal_default")
        self.assertEqual(next_action["repair_plan"]["base_candidate_manifest"], "/app/out/tonal/stem_manifest.json")
        self.assertEqual(
            next_action["repair_plan"]["target_labels"],
            ["micro_reference_stem_leakage", "micro_transient_mismatch"],
        )
        self.assertFalse(next_action["repair_plan"]["guardrails"]["accepts_rejected_base_as_final"])
        self.assertTrue(next_action["repair_plan"]["guardrails"]["requires_rebenchmark"])

    def test_phrase_drop_melody_leakage_can_continue_adaptive_repair_chain(self) -> None:
        benchmark = _load_benchmark_module()
        phrase_leakage_candidate = {
            "label": "scnet_projection_tonal_default",
            "candidate_manifest": "/app/out/tonal/stem_manifest.json",
            "evaluated_model_labels": ["drum_tonal_leakage_repair", "scnet_projection_tonal_default"],
            "delivery_status": "rejected",
            "delivery_allowed": False,
            "presentation_status": "blocked_from_delivery",
            "no_reference_contract": {"status": "accepted", "violations": []},
            "forensic_uncovered_labels": [],
            "forensic_issue_actions": [
                {
                    "label": "bar_spanning_drop_melody_leakage",
                    "action": "debleed_or_reroute_required",
                    "reason": "long phrase-level non-drum harmonic content remains in target stem",
                    "stems": ["drums"],
                    "evidence_sources": ["micro_quality_windows"],
                },
            ],
            "forensic_evidence_status": "available",
            "forensic_evidence_dir": "/app/out/bench/forensic_evidence/01_tonal",
            "mean_reference_likeness_percent": 92.139455,
            "min_reference_likeness_percent": 92.139455,
            "min_sdr_db": 16.174866,
            "min_si_sdr_db": 16.069559,
            "failed_check_names": ["bar_spanning_drop_melody_leakage"],
        }

        next_action = benchmark.build_next_action(
            ranked_candidates=[phrase_leakage_candidate],
            candidate_suggestions=[{"model_id": "htdemucs_ft", "reason": "generic rerun"}],
        )

        self.assertEqual(next_action["action"], "repair_best_ranked_candidate")
        self.assertEqual(next_action["repair_plan"]["base_candidate_label"], "scnet_projection_tonal_default")
        self.assertEqual(
            next_action["repair_plan"]["target_labels"],
            ["bar_spanning_drop_melody_leakage"],
        )
        self.assertFalse(next_action["repair_plan"]["guardrails"]["fallback_used"])
        self.assertTrue(next_action["repair_plan"]["guardrails"]["requires_rebenchmark"])

    def test_reports_exhausted_candidate_space_when_no_registry_action_remains(self) -> None:
        benchmark = _load_benchmark_module()
        quality_gate = _load_gate_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifests = quality_gate.create_synthetic_fixture(root / "fixture", sample_rate=8000, duration_seconds=3.0)

            report = benchmark.run_model_zoo_benchmark(
                ground_truth_manifest=manifests["ground_truth"],
                candidate_manifests=[
                    benchmark.CandidateManifestSpec("degraded_a", manifests["synthetic_degraded"]),
                ],
                output_dir=root / "benchmark",
                stem_names=["vocals"],
                sample_rate=8000,
                max_offset_seconds=0.0,
                thresholds=quality_gate.ObjectiveThresholds(),
                next_candidate_limit=0,
            )

            self.assertEqual(report["overall"]["status"], "candidate_space_exhausted")
            self.assertEqual(report["next_action"]["action"], "reject_delivery_candidate_space_exhausted")
            self.assertEqual(report["next_action"]["candidate_suggestions"], [])

    def test_registry_suggestions_exclude_evaluated_model_id_from_candidate_manifest(self) -> None:
        benchmark = _load_benchmark_module()
        quality_gate = _load_gate_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifests = quality_gate.create_synthetic_fixture(root / "fixture", sample_rate=8000, duration_seconds=3.0)
            candidate_manifest = _copy_manifest_with_model_id(
                manifests["synthetic_degraded"],
                manifests["synthetic_degraded"].with_name("synthetic_degraded_htdemucs_6s_manifest.json"),
                "htdemucs_6s",
            )

            report = benchmark.run_model_zoo_benchmark(
                ground_truth_manifest=manifests["ground_truth"],
                candidate_manifests=[
                    benchmark.CandidateManifestSpec("custom_user_label", candidate_manifest),
                ],
                output_dir=root / "benchmark",
                stem_names=["vocals"],
                sample_rate=8000,
                max_offset_seconds=0.0,
                thresholds=quality_gate.ObjectiveThresholds(),
                next_candidate_limit=10,
            )

            suggested_model_ids = {
                suggestion["model_id"] for suggestion in report["next_action"]["candidate_suggestions"]
            }
            self.assertNotIn("htdemucs_6s", suggested_model_ids)
            self.assertIn("htdemucs_6s", report["ranked_candidates"][0]["evaluated_model_labels"])

            suggestion_by_id = {
                suggestion["model_id"]: suggestion
                for suggestion in report["next_action"]["candidate_suggestions"]
            }
            audio_separator_suggestion = suggestion_by_id["audio_separator_bs_roformer_vocals"]
            self.assertEqual(audio_separator_suggestion["task_scope"], "vocal_instrumental")
            self.assertEqual(audio_separator_suggestion["license_status"], "checkpoint_license_unverified")
            self.assertEqual(audio_separator_suggestion["weights_license"], "checkpoint_specific")
            self.assertEqual(audio_separator_suggestion["commercial_use_status"], "requires_checkpoint_rights_review")
            self.assertEqual(
                audio_separator_suggestion["benchmark_score_context"]["directly_comparable_with"],
                "same_task_local_benchmark",
            )
            self.assertTrue(audio_separator_suggestion["report_alignment_notes"])

    def test_registry_suggestions_exclude_becruily_karaoke_manifest_identity(self) -> None:
        benchmark = _load_benchmark_module()
        quality_gate = _load_gate_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifests = quality_gate.create_synthetic_fixture(root / "fixture", sample_rate=8000, duration_seconds=3.0)
            candidate_manifest = _copy_manifest_with_model_id(
                manifests["synthetic_degraded"],
                manifests["synthetic_degraded"].with_name("synthetic_degraded_becruily_manifest.json"),
                "becruily_karaoke_vocals",
            )

            report = benchmark.run_model_zoo_benchmark(
                ground_truth_manifest=manifests["ground_truth"],
                candidate_manifests=[
                    benchmark.CandidateManifestSpec("becruily_user_label", candidate_manifest),
                ],
                output_dir=root / "benchmark",
                stem_names=["vocals"],
                sample_rate=8000,
                max_offset_seconds=0.0,
                thresholds=quality_gate.ObjectiveThresholds(),
                next_candidate_limit=20,
            )

            suggested_model_ids = {
                suggestion["model_id"] for suggestion in report["next_action"]["candidate_suggestions"]
            }
            evaluated_labels = set(report["ranked_candidates"][0]["evaluated_model_labels"])
            self.assertNotIn("becruily_karaoke_vocals", suggested_model_ids)
            self.assertIn("becruily_karaoke_vocals", evaluated_labels)
            self.assertIn("becruily_karaoke", evaluated_labels)

    def test_cli_returns_zero_and_writes_manifest_without_threshold_exit_gate(self) -> None:
        benchmark = _load_benchmark_module()
        quality_gate = _load_gate_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifests = quality_gate.create_synthetic_fixture(root / "fixture", sample_rate=8000, duration_seconds=3.0)
            exit_code = benchmark.main(
                [
                    "--ground-truth-manifest",
                    str(manifests["ground_truth"]),
                    "--candidate-manifest",
                    f"degraded={manifests['synthetic_degraded']}",
                    "--output-dir",
                    str(root / "benchmark"),
                    "--stem",
                    "vocals",
                    "--sample-rate",
                    "8000",
                ]
            )

            self.assertEqual(exit_code, 0)
            manifest_path = root / "benchmark" / "model_zoo_benchmark_manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metric_contract = manifest["metric_stack_contract"]
            self.assertEqual(metric_contract["source_document"], "deep-research-report.md")
            self.assertEqual(metric_contract["computed_layers"]["separation_fidelity"], ["sdr_db", "si_sdr_db", "sir_db", "sar_db"])
            self.assertEqual(metric_contract["not_computed_layers"]["perceptual_full_reference"]["metric"], "visqol_audio_48khz")
            self.assertEqual(metric_contract["not_computed_layers"]["vocal_specific_secondary"]["metric"], "pesq")
            self.assertEqual(metric_contract["human_listening_gate"]["status"], "not_run")
            self.assertFalse(metric_contract["human_listening_gate"]["substituted_by_objective_metrics"])


if __name__ == "__main__":
    unittest.main()
