from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tools.adaptive_policy import assert_action_not_forbidden, decision_from_evidence


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"


def _load_tool_module(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS_ROOT / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_space_exhausted_stops_more_candidates() -> None:
    decision = decision_from_evidence(issue_labels=["candidate_space_exhausted"], failed_checks=[])
    assert decision["dominant_failure_class"] == "candidate_space_exhausted"
    assert decision["decision"] == "hard_case"
    assert "generate_more_ab_candidates" in decision["forbidden_actions"]


def test_candidate_space_exhausted_dominates_metric_noise_and_forbids_backbone_switch() -> None:
    decision = decision_from_evidence(
        issue_labels=["candidate_space_exhausted"],
        failed_checks=["sample_residual_above_threshold"],
    )
    assert decision["dominant_failure_class"] == "candidate_space_exhausted"
    with pytest.raises(ValueError):
        assert_action_not_forbidden("switch_backbone", decision)


def test_thin_or_overcleaned_forbids_stronger_debleed() -> None:
    decision = decision_from_evidence(issue_labels=["thin_or_overcleaned"], failed_checks=[])
    assert decision["dominant_failure_class"] == "thin_or_overcleaned"
    assert "body_repair" in decision["allowed_actions"]
    with pytest.raises(ValueError):
        assert_action_not_forbidden("stronger_debleed", decision)


def test_vocal_tail_loss_forbids_hard_vad_gate() -> None:
    decision = decision_from_evidence(issue_labels=["vocal_tail_loss"], failed_checks=[])
    assert decision["dominant_failure_class"] == "vocal_tail_loss"
    assert "tail_preserving_blend" in decision["allowed_actions"]
    with pytest.raises(ValueError):
        assert_action_not_forbidden("hard_vad_gate", decision)


def test_robotic_artifact_forbids_stacked_cleanup() -> None:
    decision = decision_from_evidence(issue_labels=["high_noise_or_robotic_artifact"], failed_checks=[])
    assert decision["dominant_failure_class"] == "robotic_or_noise_artifact"
    with pytest.raises(ValueError):
        assert_action_not_forbidden("stack_more_cleanup", decision)


def test_alignment_failure_forbids_model_switch() -> None:
    decision = decision_from_evidence(failed_checks=["sample_residual_above_threshold"], issue_labels=[])
    assert decision["dominant_failure_class"] == "alignment_or_reference_failure"
    with pytest.raises(ValueError):
        assert_action_not_forbidden("switch_backbone", decision)


def test_clean_pass_should_be_preserved() -> None:
    decision = decision_from_evidence(issue_labels=[], failed_checks=[], candidate_status="release_ready")
    assert decision["dominant_failure_class"] == "pass_or_no_dominant_issue"
    assert decision["decision"] == "promote"
    assert "unnecessary_repair" in decision["forbidden_actions"]


def test_report_tools_skip_huge_json_instead_of_reading_it(tmp_path: Path) -> None:
    report_tool = _load_tool_module("build_adaptive_decision_report")
    dashboard_tool = _load_tool_module("build_hard_case_dashboard")
    huge_json = tmp_path / "huge.json"
    huge_json.write_text(json.dumps({"status": "failed"}), encoding="utf-8")

    report_tool.MAX_JSON_BYTES = 1
    dashboard_tool.MAX_JSON_BYTES = 1

    report_payload = report_tool.read_json(huge_json)
    dashboard_payload = dashboard_tool.read_json(huge_json)

    assert report_payload["status"] == "read_skipped_large_json"
    assert dashboard_payload["status"] == "read_skipped_large_json"
    assert report_payload["_large_json_skipped"] is True
    assert dashboard_payload["_large_json_skipped"] is True


def test_report_tools_include_model_zoo_manifests() -> None:
    report_tool = _load_tool_module("build_adaptive_decision_report")
    dashboard_tool = _load_tool_module("build_hard_case_dashboard")

    path = Path("model_zoo_benchmark_manifest.json")

    assert report_tool.is_result_json(path)
    assert dashboard_tool.is_result_json(path)


def test_report_tools_extract_model_zoo_exhaustion_contract(tmp_path: Path) -> None:
    report_tool = _load_tool_module("build_adaptive_decision_report")
    dashboard_tool = _load_tool_module("build_hard_case_dashboard")
    manifest_path = tmp_path / "model_zoo_benchmark_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "contract": {
                    "bounded_auto_rerun": {
                        "exhaustion_action": "reject_delivery_candidate_space_exhausted",
                    },
                },
                "selected_candidate": None,
                "next_action": {"status": "no_accepted_candidate"},
                "overall": {"status": "no_accepted_candidate", "accepted_count": 0},
            }
        ),
        encoding="utf-8",
    )

    payload = report_tool.read_json(manifest_path)
    assert "candidate_space_exhausted" in report_tool.collect_issue_labels(payload)
    assert "no_accepted_candidate" in report_tool.collect_failed_checks(payload)
    assert report_tool.infer_status(payload) == "no_accepted_candidate"

    rows = dashboard_tool.build_rows(tmp_path)
    assert rows[0]["dominant_failure_class"] == "candidate_space_exhausted"
    assert rows[0]["decision"] == "hard_case"
    assert rows[0]["status"] == "no_accepted_candidate"


def test_report_tools_prefer_embedded_adaptive_decision_contract(tmp_path: Path) -> None:
    report_tool = _load_tool_module("build_adaptive_decision_report")
    dashboard_tool = _load_tool_module("build_hard_case_dashboard")
    manifest_path = tmp_path / "adaptive_vocal_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "mode": "adaptive_vocal_isolation_controller",
                "decision_contract": {
                    "schema_version": 1,
                    "mode": "adaptive_decision_contract",
                    "dominant_failure_class": "candidate_space_exhausted",
                    "decision": "hard_case",
                    "allowed_actions": ["mark_hard_case"],
                    "forbidden_actions": ["generate_more_ab_candidates"],
                    "required_next_action": "stop_candidate_generation_and_route_to_hard_case_dashboard",
                    "rationale": "The current candidate space already failed.",
                },
            }
        ),
        encoding="utf-8",
    )

    report = report_tool.build_report(tmp_path)
    assert report["summary"]["dominant_failure_class_counts"]["candidate_space_exhausted"] == 1
    assert report["manifests"][0]["decision_contract"]["decision"] == "hard_case"

    rows = dashboard_tool.build_rows(tmp_path)
    assert rows[0]["dominant_failure_class"] == "candidate_space_exhausted"
    assert rows[0]["decision"] == "hard_case"
