from __future__ import annotations

import importlib.util
from pathlib import Path


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"


def _load_module():
    module_path = TOOLS_ROOT / "mac_windows_mount_preflight.py"
    spec = importlib.util.spec_from_file_location("mac_windows_mount_preflight", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mount_preflight_blocks_missing_mount(tmp_path: Path) -> None:
    module = _load_module()

    preflight = module.preflight_mac_windows_project_mount(
        mac_windows_project_mount=tmp_path / "missing-share",
        output_path=tmp_path / "missing-share" / "artifacts" / "mac_api" / "m001",
        windows_project_root=r"D:\AIMusicProject",
    )

    assert preflight["preflight_status"] == "blocked"
    assert preflight["blocked_reason"] == "mac_windows_project_mount_missing"
    assert preflight["can_write_route_artifacts"] is False


def test_mount_preflight_blocks_output_outside_allowed_project_roots(tmp_path: Path) -> None:
    module = _load_module()
    mount = tmp_path / "win_project"
    mount.mkdir()

    preflight = module.preflight_mac_windows_project_mount(
        mac_windows_project_mount=mount,
        output_path=mount / "tmp" / "mac_api" / "m001",
        windows_project_root=r"D:\AIMusicProject",
    )

    assert preflight["preflight_status"] == "blocked"
    assert preflight["blocked_reason"] == "mac_output_path_not_in_allowed_project_artifact_root"
    assert preflight["can_write_route_artifacts"] is False


def test_mount_preflight_blocks_mac_local_artifacts_parent(tmp_path: Path) -> None:
    module = _load_module()
    mount = tmp_path / "AIMusicProject_win"
    mount.mkdir()

    preflight = module.preflight_mac_windows_project_mount(
        mac_windows_project_mount=mount,
        output_path="/Users/andrejviluha/aimusic_local_artifacts/scratch/r001",
        windows_project_root=r"D:\AIMusicProject",
    )

    assert preflight["preflight_status"] == "blocked"
    assert preflight["blocked_reason"] == "mac_output_path_uses_forbidden_local_artifact_root"
    assert preflight["forbidden_mac_local_artifact_root"] == "/Users/andrejviluha/aimusic_local_artifacts"
    assert preflight["can_write_route_artifacts"] is False


def test_mount_preflight_records_windows_path_and_removes_probe(tmp_path: Path) -> None:
    module = _load_module()
    mount = tmp_path / "win_project"
    output = mount / "artifacts" / "mac_api" / "m001"
    output.mkdir(parents=True)

    preflight = module.preflight_mac_windows_project_mount(
        mac_windows_project_mount=mount,
        output_path=output,
        windows_project_root=r"D:\AIMusicProject",
    )

    assert preflight["preflight_status"] == "passed"
    assert preflight["can_write_route_artifacts"] is True
    assert preflight["windows_output_path"] == r"D:\AIMusicProject\artifacts\mac_api\m001"
    assert preflight["mac_output_path"] == str(output.resolve())
    assert preflight["probe_status"] == "write_read_delete_passed"
    assert not list((mount / ".aimusic_mount_probe").glob("*"))
