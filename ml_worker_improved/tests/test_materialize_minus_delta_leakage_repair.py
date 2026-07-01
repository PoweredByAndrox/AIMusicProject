import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "materialize_minus_delta_leakage_repair.py"


def _load_module():
    tools_dir = str(MODULE_PATH.parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    spec = importlib.util.spec_from_file_location("materialize_minus_delta_leakage_repair", MODULE_PATH)
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
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def fixture_router(tmp_path: Path, *, selected_route: str = "judge_guided_minus_vocal_leakage_repair") -> tuple[Path, Path]:
    raw_ref = touch(tmp_path / "src" / "raw.flac")
    base_minus = touch(tmp_path / "cmd" / "base.flac")
    repaired_minus = touch(tmp_path / "cmd" / "m001.flac")
    vocals = touch(tmp_path / "src" / "vocals.flac")
    repair_manifest = write_json(
        tmp_path / "cmd" / "air" / "manifest.json",
        {
            "mode": "judge_guided_minus_air_detail_repair",
            "delivery_allowed": False,
            "delivery_status": "candidate_only",
            "presentation_status": "blocked_until_judged",
            "raw_audio_path": str(raw_ref),
            "base_minus_path": str(base_minus),
            "rendered_minus_candidate_path": str(repaired_minus),
            "vocal_guide_path": str(vocals),
        },
    )
    before_manifest = write_json(
        tmp_path / "self" / "before" / "manifest.json",
        {
            "mode": "instrumental_self_judge_gate",
            "segments_jsonl": str(write_jsonl(tmp_path / "self" / "before" / "rows.jsonl", [])),
            "delivery_contract": {"delivery_allowed": False},
        },
    )
    after_rows = write_jsonl(
        tmp_path / "self" / "after" / "rows.jsonl",
        [
            {
                "segment_id": "segment_0000",
                "window_sec": [0.0, 5.0],
                "candidate_label": "it017_air",
                "blocker": True,
                "issues": ["low_rms_muffling_proxy", "vocal_projection_proxy", "vocal_stem_corr_proxy"],
                "issue_severities": {
                    "low_rms_ratio": 9.0,
                    "vocal_projection": 0.42,
                    "vocal_stem_corr": 0.33,
                },
                "severity_score": 9.0,
            },
            {
                "segment_id": "segment_0001",
                "window_sec": [5.0, 10.0],
                "candidate_label": "it017_air",
                "blocker": True,
                "issues": ["presence_loss_proxy"],
                "issue_severities": {"presence_loss": 0.4},
                "severity_score": 0.4,
            },
            {
                "segment_id": "segment_0003",
                "window_sec": [15.0, 20.0],
                "candidate_label": "it017_air",
                "blocker": True,
                "issues": ["vocal_stem_corr_proxy"],
                "issue_severities": {"vocal_stem_corr": 0.77},
                "severity_score": 0.77,
            },
        ],
    )
    after_manifest = write_json(
        tmp_path / "self" / "after" / "manifest.json",
        {
            "mode": "instrumental_self_judge_gate",
            "source_audio_path": str(raw_ref),
            "candidate": {
                "label": "it017_air",
                "instrumental_path": str(repaired_minus),
                "vocal_path": str(vocals),
            },
            "overall": {"status": "proxy_blocked", "blocker_count": 3, "segment_count": 3},
            "segments_jsonl": str(after_rows),
            "delivery_contract": {"delivery_allowed": False},
        },
    )
    delta_rows = write_jsonl(
        tmp_path / "gate" / "rows.jsonl",
        [
            {
                "segment_id": "segment_0000",
                "transition": "persisting_blocker",
                "before_blocker": True,
                "after_blocker": True,
                "before_issues": ["low_rms_muffling_proxy"],
                "after_issues": ["low_rms_muffling_proxy", "vocal_projection_proxy"],
            },
            {
                "segment_id": "segment_0001",
                "transition": "persisting_blocker",
                "before_blocker": True,
                "after_blocker": True,
                "before_issues": ["presence_loss_proxy"],
                "after_issues": ["presence_loss_proxy"],
            },
            {
                "segment_id": "segment_0002",
                "transition": "resolved_blocker",
                "before_blocker": True,
                "after_blocker": False,
                "before_issues": ["vocal_projection_proxy"],
                "after_issues": [],
            },
            {
                "segment_id": "segment_0003",
                "transition": "new_blocker",
                "before_blocker": False,
                "after_blocker": True,
                "before_issues": [],
                "after_issues": ["vocal_stem_corr_proxy"],
            },
        ],
    )
    delta_gate = write_json(
        tmp_path / "gate" / "gate.json",
        {
            "schema": "minus_repair_delta_gate.v1",
            "mode": "minus_repair_delta_gate",
            "candidate_id": "it017_air",
            "gate_status": "blocked_no_blocker_reduction",
            "before_self_judge_manifest": str(before_manifest),
            "after_self_judge_manifest": str(after_manifest),
            "repair_manifest": str(repair_manifest),
            "rows_jsonl": str(delta_rows),
            "delivery_allowed": False,
        },
    )
    router_rows = write_jsonl(
        tmp_path / "router" / "rows.jsonl",
        [
            {
                "schema": "minus_repair_delta_router_item.v1",
                "candidate_id": "it017_air",
                "selected_route": selected_route,
                "next_action": "switch_to_leakage_repair",
                "blocked_repeat_routes": ["judge_guided_minus_air_detail_repair"],
                "delta_gate_manifest": str(delta_gate),
                "delivery_allowed": False,
            }
        ],
    )
    router_manifest = write_json(
        tmp_path / "router" / "plan.json",
        {
            "schema": "minus_repair_delta_router_plan.v1",
            "mode": "minus_repair_delta_router",
            "candidate_id": "it017_air",
            "router_status": "ready_for_next_candidate_route",
            "selected_next_action": "switch_to_leakage_repair",
            "selected_route": selected_route,
            "blocked_repeat_routes": ["judge_guided_minus_air_detail_repair"],
            "delta_gate_manifest": str(delta_gate),
            "rows": read_jsonl(router_rows),
            "rows_jsonl": str(router_rows),
            "delivery_allowed": False,
        },
    )
    return router_manifest, repair_manifest


def test_materializer_converts_persistent_unsafe_delta_segments_to_leakage_windows(tmp_path: Path) -> None:
    module = _load_module()
    router_manifest, repair_manifest = fixture_router(tmp_path)

    manifest = module.materialize_minus_delta_leakage_repair(
        router_plan_json=router_manifest,
        source_repair_manifest=repair_manifest,
        output_dir=tmp_path / "out",
        candidate_id=None,
        command_args=[],
    )

    assert manifest["schema"] == "minus_delta_leakage_repair_command_plan.v1"
    assert manifest["command_plan_status"] == "ready"
    assert manifest["selected_route"] == "judge_guided_minus_vocal_leakage_repair"
    assert manifest["selected_window_count"] == 2
    assert manifest["execution_started"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["can_publish_as_result"] is False
    assert manifest["blocked_repeat_routes"] == ["judge_guided_minus_air_detail_repair"]

    command_rows = read_jsonl(tmp_path / "out" / "cmds.jsonl")
    assert len(command_rows) == 1
    command = command_rows[0]["docker_command"]
    assert command[:2] == ["python", "tools/apply_judge_guided_minus_vocal_leakage_repair.py"]
    assert command[2:6] == [
        str(tmp_path / "src" / "raw.flac"),
        str(tmp_path / "cmd" / "m001.flac"),
        str(tmp_path / "src" / "vocals.flac"),
        str(tmp_path / "out" / "rows.jsonl"),
    ]
    assert command[-5:] == ["--candidate-id", "it017_air", "--track-id", "it017_air", "--short-output-names"]

    judge_rows = read_jsonl(tmp_path / "out" / "rows.jsonl")
    assert len(judge_rows) == 1
    judge_row = judge_rows[0]
    assert judge_row["schema"] == "windowed_minus_judge_row.v1"
    assert judge_row["candidate_id"] == "it017_air"
    assert judge_row["source_audio_path"] == str(tmp_path / "src" / "raw.flac")
    assert judge_row["minus_path"] == str(tmp_path / "cmd" / "m001.flac")
    assert judge_row["instrumental_path"] == str(tmp_path / "cmd" / "m001.flac")
    assert judge_row["primary_vocals_path"] == str(tmp_path / "src" / "vocals.flac")
    assert judge_row["judge_source"] == "minus_repair_delta_gate_after_self_judge"
    windows = judge_row["window_metrics"]
    assert [window["segment_id"] for window in windows] == ["segment_0000", "segment_0003"]
    assert windows[0]["window_sec"] == [0.0, 5.0]
    assert windows[0]["start_sec"] == 0.0
    assert windows[0]["duration_sec"] == 5.0
    assert windows[0]["source_delta_transition"] == "persisting_blocker"
    assert windows[0]["minus_blockers"] == ["vocal_leakage_in_minus"]
    assert windows[0]["vocal_leakage_in_minus_score"] == 0.42
    assert windows[1]["source_delta_transition"] == "new_blocker"
    assert windows[1]["vocal_leakage_in_minus_score"] == 0.77
    assert all(len(path.name) <= 32 for path in (tmp_path / "out").rglob("*") if path.is_file())


def test_materializer_blocks_when_router_did_not_select_leakage_route(tmp_path: Path) -> None:
    module = _load_module()
    router_manifest, repair_manifest = fixture_router(
        tmp_path,
        selected_route="judge_guided_minus_air_detail_repair",
    )

    manifest = module.materialize_minus_delta_leakage_repair(
        router_plan_json=router_manifest,
        source_repair_manifest=repair_manifest,
        output_dir=tmp_path / "out",
        candidate_id=None,
        command_args=[],
    )

    assert manifest["command_plan_status"] == "blocked_route_not_selected"
    assert manifest["selected_route"] == "judge_guided_minus_air_detail_repair"
    assert manifest["ready_command_count"] == 0
    assert manifest["selected_window_count"] == 0
    assert manifest["failed_checks"] == ["router_selected_route_not_leakage_repair"]
    assert manifest["delivery_allowed"] is False


def test_materializer_blocks_without_persistent_unsafe_delta_windows(tmp_path: Path) -> None:
    module = _load_module()
    router_manifest, repair_manifest = fixture_router(tmp_path)
    router = read_json(router_manifest)
    delta_gate = Path(router["delta_gate_manifest"])
    delta = read_json(delta_gate)
    write_jsonl(
        Path(delta["rows_jsonl"]),
        [
            {
                "segment_id": "segment_0001",
                "transition": "persisting_blocker",
                "before_blocker": True,
                "after_blocker": True,
                "before_issues": ["presence_loss_proxy"],
                "after_issues": ["presence_loss_proxy"],
            }
        ],
    )

    manifest = module.materialize_minus_delta_leakage_repair(
        router_plan_json=router_manifest,
        source_repair_manifest=repair_manifest,
        output_dir=tmp_path / "out",
        candidate_id=None,
        command_args=[],
    )

    assert manifest["command_plan_status"] == "blocked_no_persistent_unsafe_delta_windows"
    assert manifest["ready_command_count"] == 0
    assert manifest["selected_window_count"] == 0
    assert manifest["failed_checks"] == ["persistent_unsafe_delta_windows_missing"]
    assert manifest["delivery_allowed"] is False


def test_cli_allows_explicit_native_runtime_and_materializes_native_command(tmp_path: Path) -> None:
    module = _load_module()
    router_manifest, repair_manifest = fixture_router(tmp_path)
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
                "--router-plan-json",
                str(router_manifest),
                "--source-repair-manifest",
                str(repair_manifest),
                "--output-dir",
                str(tmp_path / "out"),
                "--allow-native-runtime",
            ]
        )
    finally:
        module.require_docker_runtime = original_guard
        module.cli_runtime_contract = original_contract

    manifest = read_json(tmp_path / "out" / "plan.json")
    command_row = read_jsonl(tmp_path / "out" / "cmds.jsonl")[0]
    assert exit_code == 0
    assert calls == [
        {
            "tool_name": "materialize_minus_delta_leakage_repair.py",
            "entrypoint_path": "/app/tools/materialize_minus_delta_leakage_repair.py",
            "allow_native_macos_diagnostic": True,
        }
    ]
    assert manifest["runtime_environment"] == "native_mac_cpu_dsp"
    assert manifest["native_runtime_allowed"] is True
    assert command_row["runtime_environment"] == "native_mac_cpu_dsp"
    assert command_row["native_runtime_allowed"] is True
    assert command_row["native_command"][-1] == "--allow-native-runtime"
    assert manifest["delivery_allowed"] is False
