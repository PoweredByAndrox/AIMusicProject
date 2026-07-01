from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
ROOT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = TOOLS_ROOT / "materialize_minus_delta_model_candidate_route.py"


def _load_module():
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    spec = importlib.util.spec_from_file_location("materialize_minus_delta_model_candidate_route", MODULE_PATH)
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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def router_plan(tmp_path: Path, *, selected_route: str = "no_reference_minus_candidate_model_router") -> Path:
    row = {
        "schema": "minus_repair_delta_router_item.v1",
        "queue_item_id": "dr001",
        "candidate_id": "it029_lkg",
        "stem_type": "minus",
        "route_intent": "minus_repair_delta_router",
        "action_status": "ready_for_next_candidate_route",
        "next_action": "route_to_alternative_model_candidate",
        "selected_route": selected_route,
        "source_route": "judge_guided_minus_vocal_leakage_repair",
        "blocked_repeat_routes": ["judge_guided_minus_vocal_leakage_repair:no_blocker_reduction"],
        "required_actions": [
            "mine_alternative_minus_model_candidate",
            "rerun_instrumental_self_judge_gate",
            "rerun_minus_repair_delta_gate",
        ],
        "failure_taxonomy": [
            "blocker_count_not_reduced",
            "persistent_unsafe_vocal_or_full_mix_blockers",
            "failed_leakage_repair_no_blocker_reduction",
        ],
        "issue_summary": {
            "issue_counts": {"vocal_projection_proxy": 1, "presence_loss_proxy": 1},
            "persistent_unsafe_blocker_segments": 1,
        },
        "delta_gate_manifest": str(tmp_path / "gate" / "gate.json"),
        "delivery_allowed": False,
        "can_publish_as_result": False,
        "release_ready": False,
        "training_started": False,
    }
    return write_json(
        tmp_path / "router" / "plan.json",
        {
            "schema": "minus_repair_delta_router_plan.v1",
            "router_status": "ready_for_next_candidate_route",
            "candidate_id": "it029_lkg",
            "stem_type": "minus",
            "mode": "minus_repair_delta_router",
            "selected_next_action": "route_to_alternative_model_candidate",
            "selected_route": selected_route,
            "blocked_repeat_routes": ["judge_guided_minus_vocal_leakage_repair:no_blocker_reduction"],
            "required_actions": row["required_actions"],
            "failure_taxonomy": row["failure_taxonomy"],
            "issue_summary": row["issue_summary"],
            "rows": [row],
            "delivery_allowed": False,
            "can_publish_as_result": False,
        },
    )


def source_repair_manifest(tmp_path: Path) -> Path:
    source_audio = tmp_path / "host_artifacts" / "src" / "mix.flac"
    source_audio.parent.mkdir(parents=True, exist_ok=True)
    source_audio.write_bytes(b"stub")
    return write_json(
        tmp_path / "repair" / "manifest.json",
        {
            "mode": "judge_guided_minus_vocal_leakage_repair",
            "delivery_status": "candidate_only",
            "delivery_allowed": False,
            "raw_audio_path": str(source_audio),
        },
    )


def demo_registry(tmp_path: Path):
    from worker.model_registry import ModelRegistry, ModelRegistryEntry

    return ModelRegistry(
        runtime_root=tmp_path,
        entries=(
            ModelRegistryEntry(
                model_id="demo_minus_alt",
                display_name="Demo minus alternative",
                family="test",
                target_stems=("instrumental",),
                output_stems=("vocals", "instrumental"),
                command_template=("python", "demo.py", "{source_audio_path}", "{output_dir}", "--device", "{device}"),
                expected_manifest="{output_dir}/stem_manifest.json",
                issue_labels_addressed=("vocal_residue_in_minus",),
            ),
            ModelRegistryEntry(
                model_id="demo_unavailable_minus_alt",
                display_name="Unavailable demo minus alternative",
                family="test",
                target_stems=("instrumental",),
                output_stems=("vocals", "instrumental"),
                command_template=("python", "missing.py", "{source_audio_path}", "{output_dir}"),
                expected_manifest="{output_dir}/stem_manifest.json",
                required_paths=("missing.ckpt",),
                issue_labels_addressed=("vocal_residue_in_minus",),
            ),
        ),
    )


def test_model_candidate_materializer_builds_registered_minus_command_plan(tmp_path: Path) -> None:
    module = _load_module()
    router = router_plan(tmp_path)
    repair = source_repair_manifest(tmp_path)

    manifest = module.materialize_minus_delta_model_candidate_route(
        router_plan_json=router,
        source_repair_manifest=repair,
        output_dir=tmp_path / "host_artifacts" / "out",
        registry=demo_registry(tmp_path),
        max_candidates=3,
        native_runtime_allowed=True,
    )

    assert manifest["schema"] == "minus_delta_model_candidate_command_plan.v1"
    assert manifest["command_plan_status"] == "ready"
    assert manifest["selected_route"] == "no_reference_minus_candidate_model_router"
    assert manifest["ready_command_count"] == 1
    assert manifest["blocked_command_count"] == 0
    assert manifest["delivery_allowed"] is False
    assert manifest["can_publish_as_result"] is False
    assert manifest["execution_started"] is False
    assert manifest["requires_post_judge"] is True
    assert manifest["requires_result_judge"] is True
    assert manifest["requires_delta_gate"] is True
    assert manifest["gpu_inference_skipped"] is True
    assert manifest["native_runtime_allowed"] is True

    command_rows = read_jsonl(tmp_path / "host_artifacts" / "out" / "cmds.jsonl")
    assert len(command_rows) == 1
    row = command_rows[0]
    assert row["schema"] == "minus_delta_model_candidate_command_row.v1"
    assert row["command_status"] == "ready"
    assert row["model_id"] == "demo_minus_alt"
    assert row["target_stem"] == "minus"
    assert row["expected_acceleration_mode"] == "gpu_registered_model_inference"
    assert row["native_runtime_allowed"] is False
    assert row["materializer_native_runtime_allowed"] is True
    assert row["requires_post_judge"] is True
    assert row["delivery_allowed"] is False
    assert row["can_publish_as_result"] is False
    assert row["docker_command"] == [
        "python",
        "/app/tools/run_registered_model_candidate.py",
        "/host_artifacts/src/mix.flac",
        "/host_artifacts/out/mdl/m001",
        "--model-id",
        "demo_minus_alt",
        "--device",
        "cuda",
        "--target-stem",
        "minus",
    ]
    assert "persistent_unsafe_vocal_or_full_mix_blockers" in manifest["source_failure_taxonomy"]
    assert "vocal_residue_in_minus" in manifest["expanded_failure_labels"]
    assert all(len(path.name) <= 32 for path in (tmp_path / "host_artifacts" / "out").rglob("*"))
    assert row["source_audio_path"] == str(tmp_path / "host_artifacts" / "src" / "mix.flac")
    assert row["output_dir"] == str(tmp_path / "host_artifacts" / "out" / "mdl" / "m001")
    assert row["docker_source_audio_path"] == "/host_artifacts/src/mix.flac"
    assert row["docker_output_dir"] == "/host_artifacts/out/mdl/m001"


def test_model_candidate_materializer_blocks_without_docker_visible_paths(tmp_path: Path) -> None:
    module = _load_module()
    source_audio = tmp_path / "src" / "mix.flac"
    source_audio.parent.mkdir(parents=True, exist_ok=True)
    source_audio.write_bytes(b"stub")
    repair = write_json(
        tmp_path / "repair" / "manifest.json",
        {
            "mode": "judge_guided_minus_vocal_leakage_repair",
            "delivery_status": "candidate_only",
            "delivery_allowed": False,
            "raw_audio_path": str(source_audio),
        },
    )

    manifest = module.materialize_minus_delta_model_candidate_route(
        router_plan_json=router_plan(tmp_path),
        source_repair_manifest=repair,
        output_dir=tmp_path / "out",
        registry=demo_registry(tmp_path),
    )

    assert manifest["command_plan_status"] == "blocked_missing_docker_path_mapping"
    assert manifest["ready_command_count"] == 0
    assert manifest["blocked_command_count"] == 1
    assert manifest["failed_checks"] == ["docker_path_mapping_missing"]
    assert manifest["docker_path_mapping"] == {
        "source_audio_path": str(source_audio),
        "output_dir": str(tmp_path / "out"),
        "docker_source_audio_path": "",
        "docker_output_dir": "",
        "failed_path_fields": ["source_audio_path", "output_dir"],
    }
    assert read_jsonl(tmp_path / "out" / "cmds.jsonl") == []


def test_model_candidate_materializer_blocks_when_router_did_not_select_route(tmp_path: Path) -> None:
    module = _load_module()
    manifest = module.materialize_minus_delta_model_candidate_route(
        router_plan_json=router_plan(tmp_path, selected_route="judge_guided_minus_vocal_leakage_repair"),
        source_repair_manifest=source_repair_manifest(tmp_path),
        output_dir=tmp_path / "host_artifacts" / "out",
        registry=demo_registry(tmp_path),
    )

    assert manifest["command_plan_status"] == "blocked_route_not_selected"
    assert manifest["ready_command_count"] == 0
    assert manifest["blocked_command_count"] == 1
    assert manifest["failed_checks"] == ["router_selected_route_not_model_candidate_router"]
    assert manifest["delivery_allowed"] is False
    assert read_jsonl(tmp_path / "host_artifacts" / "out" / "cmds.jsonl") == []


def test_model_candidate_materializer_blocks_without_source_audio(tmp_path: Path) -> None:
    module = _load_module()
    repair = write_json(
        tmp_path / "repair" / "manifest.json",
        {
            "mode": "judge_guided_minus_vocal_leakage_repair",
            "delivery_status": "candidate_only",
            "delivery_allowed": False,
        },
    )

    manifest = module.materialize_minus_delta_model_candidate_route(
        router_plan_json=router_plan(tmp_path),
        source_repair_manifest=repair,
        output_dir=tmp_path / "host_artifacts" / "out",
        registry=demo_registry(tmp_path),
    )

    assert manifest["command_plan_status"] == "blocked_missing_evidence"
    assert manifest["failed_checks"] == ["source_audio_path_missing"]
    assert manifest["ready_command_count"] == 0
    assert manifest["delivery_allowed"] is False


def test_model_candidate_materializer_resolves_relative_delta_gate_from_router_plan(tmp_path: Path) -> None:
    module = _load_module()
    router = router_plan(tmp_path)
    router_payload = json.loads(router.read_text(encoding="utf-8"))
    router_payload["delta_gate_manifest"] = "gate/gate.json"
    router_payload["rows"][0]["delta_gate_manifest"] = "gate/gate.json"
    write_json(router, router_payload)

    source_audio = tmp_path / "host_artifacts" / "src" / "mix.flac"
    source_audio.parent.mkdir(parents=True, exist_ok=True)
    source_audio.write_bytes(b"stub")
    after_manifest = write_json(
        tmp_path / "self" / "manifest.json",
        {
            "schema_version": 1,
            "mode": "instrumental_self_judge_gate",
            "source_audio_path": str(source_audio),
            "delivery_contract": {"delivery_allowed": False},
        },
    )
    write_json(
        tmp_path / "router" / "gate" / "gate.json",
        {
            "schema": "minus_repair_delta_gate.v1",
            "gate_status": "blocked_no_blocker_reduction",
            "after_self_judge_manifest": str(after_manifest),
            "delivery_allowed": False,
        },
    )
    repair = write_json(
        tmp_path / "repair" / "manifest.json",
        {
            "mode": "judge_guided_minus_vocal_leakage_repair",
            "delivery_status": "candidate_only",
            "delivery_allowed": False,
        },
    )

    manifest = module.materialize_minus_delta_model_candidate_route(
        router_plan_json=router,
        source_repair_manifest=repair,
        output_dir=tmp_path / "host_artifacts" / "out",
        registry=demo_registry(tmp_path),
    )

    assert manifest["command_plan_status"] == "ready"
    assert manifest["source_audio_path"] == str(source_audio)
    command_rows = read_jsonl(tmp_path / "host_artifacts" / "out" / "cmds.jsonl")
    assert command_rows[0]["source_audio_path"] == str(source_audio)
    assert command_rows[0]["docker_source_audio_path"] == "/host_artifacts/src/mix.flac"
