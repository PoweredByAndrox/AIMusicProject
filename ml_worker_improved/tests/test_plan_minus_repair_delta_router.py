from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "plan_minus_repair_delta_router.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("plan_minus_repair_delta_router", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def delta_gate(
    tmp_path: Path,
    *,
    gate_status: str,
    rows: list[dict],
    delta: dict | None = None,
    failure_taxonomy: list[str] | None = None,
) -> Path:
    if failure_taxonomy is None:
        if gate_status == "passed_for_candidate_routing":
            failure_taxonomy = []
        elif gate_status == "blocked_remaining_self_judge_blockers":
            failure_taxonomy = ["remaining_self_judge_blockers"]
        else:
            failure_taxonomy = ["blocker_count_not_reduced"]
    rows_path = write_jsonl(tmp_path / "gate" / "rows.jsonl", rows)
    return write_json(
        tmp_path / "gate" / "gate.json",
        {
            "schema": "minus_repair_delta_gate.v1",
            "gate_status": gate_status,
            "candidate_id": "it019-candidate",
            "repair_manifest": "/host_artifacts/loop/it017/cmd/air/manifest.json",
            "delta": {
                "blocker_count_delta": 0,
                "unsafe_issue_count_delta": 0,
                "persisting_blocker_count": len(rows),
                **(delta or {}),
            },
            "failure_taxonomy": failure_taxonomy,
            "rows_jsonl": str(rows_path),
            "delivery_allowed": False,
            "presentation_status": "delta_gate_not_for_listen_pack",
            "route_provenance": {"after_candidate_label": "it017_air"},
        },
    )


def segment(segment_id: str, *, after_issues: list[str], transition: str = "persisting_blocker") -> dict:
    return {
        "schema": "minus_repair_delta_gate_segment.v1",
        "segment_id": segment_id,
        "transition": transition,
        "before_blocker": True,
        "after_blocker": transition != "resolved_blocker",
        "before_issues": after_issues,
        "after_issues": after_issues,
        "persisting_issues": after_issues,
        "new_issues": [],
        "resolved_issues": [],
    }


def test_router_switches_to_leakage_repair_for_persistent_unsafe_blockers(tmp_path: Path) -> None:
    module = _load_module()
    gate = delta_gate(
        tmp_path,
        gate_status="blocked_no_blocker_reduction",
        rows=[
            segment("segment_0000", after_issues=["vocal_projection_proxy"]),
            segment("segment_0001", after_issues=["low_rms_muffling_proxy"]),
        ],
    )

    manifest = module.plan_minus_repair_delta_router(
        delta_gate_manifest=gate,
        output_dir=tmp_path / "router",
        source_route="judge_guided_minus_air_detail_repair",
    )

    assert manifest["router_status"] == "ready_for_next_candidate_route"
    assert manifest["selected_next_action"] == "switch_to_leakage_repair"
    assert manifest["selected_route"] == "judge_guided_minus_vocal_leakage_repair"
    assert manifest["blocked_repeat_routes"] == ["judge_guided_minus_air_detail_repair"]
    assert manifest["delivery_allowed"] is False
    assert manifest["can_publish_as_result"] is False
    row = read_json(tmp_path / "router" / "plan.json")["rows"][0]
    assert row["required_actions"][0] == "run_judge_guided_minus_vocal_leakage_repair"
    assert (tmp_path / "router" / "rows.jsonl").exists()
    assert (tmp_path / "router" / "summary.md").exists()


def test_router_retunes_detail_repair_for_detail_only_persistent_blockers(tmp_path: Path) -> None:
    module = _load_module()
    gate = delta_gate(
        tmp_path,
        gate_status="blocked_no_blocker_reduction",
        rows=[
            segment("segment_0000", after_issues=["low_rms_muffling_proxy"]),
            segment("segment_0001", after_issues=["presence_loss_proxy"]),
        ],
    )

    manifest = module.plan_minus_repair_delta_router(
        delta_gate_manifest=gate,
        output_dir=tmp_path / "router",
        source_route="judge_guided_minus_air_detail_repair",
    )

    assert manifest["router_status"] == "ready_for_next_candidate_route"
    assert manifest["selected_next_action"] == "retune_detail_repair"
    assert manifest["selected_route"] == "judge_guided_minus_air_detail_repair"
    assert manifest["blocked_repeat_routes"] == ["judge_guided_minus_air_detail_repair:repeat_same_parameters"]
    assert manifest["failure_taxonomy"] == ["blocker_count_not_reduced", "persistent_detail_loss_blockers"]
    assert manifest["delivery_allowed"] is False


def test_router_for_passed_delta_sends_candidate_to_downstream_gates(tmp_path: Path) -> None:
    module = _load_module()
    gate = delta_gate(
        tmp_path,
        gate_status="passed_for_candidate_routing",
        rows=[segment("segment_0000", after_issues=[], transition="resolved_blocker")],
        delta={"blocker_count_delta": -1, "unsafe_issue_count_delta": 0, "persisting_blocker_count": 0},
    )

    manifest = module.plan_minus_repair_delta_router(
        delta_gate_manifest=gate,
        output_dir=tmp_path / "router",
        source_route="judge_guided_minus_air_detail_repair",
    )

    assert manifest["router_status"] == "candidate_ready_for_downstream_gates"
    assert manifest["selected_next_action"] == "run_downstream_acceptance_gates"
    assert manifest["selected_route"] == ""
    assert manifest["delivery_allowed"] is False
    assert manifest["requires_downstream_acceptance"] is True


def test_router_continues_detail_repair_for_remaining_detail_blockers_after_improvement(tmp_path: Path) -> None:
    module = _load_module()
    gate = delta_gate(
        tmp_path,
        gate_status="blocked_remaining_self_judge_blockers",
        rows=[
            segment("segment_0000", after_issues=[], transition="resolved_blocker"),
            segment("segment_0001", after_issues=["presence_loss_proxy"], transition="persisting_blocker"),
        ],
        delta={"blocker_count_delta": -1, "unsafe_issue_count_delta": 0, "persisting_blocker_count": 1},
    )

    manifest = module.plan_minus_repair_delta_router(
        delta_gate_manifest=gate,
        output_dir=tmp_path / "router",
        source_route="judge_guided_minus_air_detail_repair",
    )

    assert manifest["router_status"] == "ready_for_next_candidate_route"
    assert manifest["selected_next_action"] == "retune_detail_repair"
    assert manifest["selected_route"] == "judge_guided_minus_air_detail_repair"
    assert manifest["failure_taxonomy"] == ["remaining_self_judge_blockers", "persistent_detail_loss_blockers"]
    assert manifest["delivery_allowed"] is False


def test_router_halts_when_delta_rows_are_missing(tmp_path: Path) -> None:
    module = _load_module()
    gate = write_json(
        tmp_path / "gate" / "gate.json",
        {
            "schema": "minus_repair_delta_gate.v1",
            "gate_status": "blocked_no_blocker_reduction",
            "delta": {"blocker_count_delta": 0},
            "failure_taxonomy": ["blocker_count_not_reduced"],
            "delivery_allowed": False,
        },
    )

    manifest = module.plan_minus_repair_delta_router(
        delta_gate_manifest=gate,
        output_dir=tmp_path / "router",
        source_route="judge_guided_minus_air_detail_repair",
    )

    assert manifest["router_status"] == "halt_missing_evidence"
    assert manifest["selected_next_action"] == "halt_missing_evidence"
    assert "delta_rows_jsonl_missing" in manifest["failure_taxonomy"]
    assert manifest["delivery_allowed"] is False


def test_cli_allows_explicit_native_runtime_and_records_contract(tmp_path: Path) -> None:
    module = _load_module()
    gate = delta_gate(
        tmp_path,
        gate_status="blocked_no_blocker_reduction",
        rows=[segment("segment_0000", after_issues=["vocal_projection_proxy"])],
    )
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
                "--delta-gate-manifest",
                str(gate),
                "--output-dir",
                str(tmp_path / "router"),
                "--source-route",
                "judge_guided_minus_air_detail_repair",
                "--allow-native-runtime",
            ]
        )
    finally:
        module.require_docker_runtime = original_guard
        module.cli_runtime_contract = original_contract

    manifest = read_json(tmp_path / "router" / "plan.json")
    assert exit_code == 0
    assert calls == [
        {
            "tool_name": "plan_minus_repair_delta_router.py",
            "entrypoint_path": "/app/tools/plan_minus_repair_delta_router.py",
            "allow_native_macos_diagnostic": True,
        }
    ]
    assert manifest["runtime_environment"] == "native_mac_cpu_dsp"
    assert manifest["native_runtime_allowed"] is True
    assert manifest["delivery_allowed"] is False
