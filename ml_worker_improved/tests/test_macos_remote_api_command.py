from __future__ import annotations

import subprocess

import pytest

from tools.run_macos_remote_api_command import (
    CANONICAL_MAC_ARTIFACT_ROOT,
    CANONICAL_MAC_WINDOWS_PROJECT_MOUNT,
    PRESENTATION_STATUS,
    build_remote_script,
    parse_args,
    run_macos_remote_api_command,
    validate_artifact_subdir,
)


def test_validate_artifact_subdir_allows_short_nested_aliases() -> None:
    assert validate_artifact_subdir("mac_api/c001") == "mac_api/c001"
    assert validate_artifact_subdir("/mac_api/c001/") == "mac_api/c001"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "../x",
        "mac_api/../x",
        "mac api/c001",
        "mac_api/c001/raw-track-title-that-is-far-too-long",
        "mac_api\\c001",
    ],
)
def test_validate_artifact_subdir_rejects_unsafe_segments(value: str) -> None:
    with pytest.raises(ValueError):
        validate_artifact_subdir(value)


def test_remote_script_writes_diagnostic_manifest_and_progress() -> None:
    script = build_remote_script(
        command="/usr/local/bin/container run --rm alpine uname -a",
        mac_artifact_root="/Users/andrejviluha/AIMusicProject_win/artifacts",
        mac_windows_project_mount="/Users/andrejviluha/AIMusicProject_win",
        windows_project_root=r"D:\AIMusicProject",
        artifact_subdir="mac_api/c001",
        metadata={"purpose": "container_smoke"},
    )

    assert "/Users/andrejviluha/AIMusicProject_win/artifacts/mac_api/c001" in script
    assert 'PROBE_DIR="$MOUNT/.aimusic_mount_probe"' in script
    assert "write_read_delete_passed" in script
    assert r"D:\AIMusicProject\artifacts\mac_api\c001" in script
    assert "manifest.json" in script
    assert "progress.jsonl" in script
    assert '"delivery_allowed": False' in script
    assert PRESENTATION_STATUS in script
    assert "/usr/local/bin/container run --rm alpine uname -a" not in script
    assert "CMD_B64=" in script


def test_remote_api_runner_uses_ssh_stdin_and_fails_closed() -> None:
    calls = []

    def fake_runner(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="ok", stderr="")

    result = run_macos_remote_api_command(
        ssh_target="andrejviluha@192.168.0.138",
        ssh_key="C:/Users/PBAX/.ssh/id_ed25519_phone",
        mac_artifact_root="/Users/andrejviluha/AIMusicProject_win/artifacts",
        mac_windows_project_mount="/Users/andrejviluha/AIMusicProject_win",
        windows_project_root=r"D:\AIMusicProject",
        artifact_subdir="mac_api/c002",
        command="printf hello",
        runner=fake_runner,
    )

    assert result["status"] == "passed"
    assert result["delivery_allowed"] is False
    assert result["presentation_status"] == PRESENTATION_STATUS
    assert result["remote_manifest_path"].endswith("/mac_api/c002/manifest.json")
    assert result["windows_output_path"] == r"D:\AIMusicProject\artifacts\mac_api\c002"
    assert result["mount_preflight"]["windows_output_path"] == r"D:\AIMusicProject\artifacts\mac_api\c002"
    ssh_command = calls[0][0][0]
    kwargs = calls[0][1]
    assert ssh_command[-1] == "sh -s"
    assert ssh_command[ssh_command.index("-o") + 1] == "IdentitiesOnly=yes"
    assert kwargs["input"].startswith(b"#!/bin/sh")
    assert kwargs["capture_output"] is True
    assert kwargs["check"] is False


def test_remote_api_runner_reports_failed_command_without_accepting() -> None:
    def fake_runner(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=7, stdout="", stderr="bad")

    result = run_macos_remote_api_command(
        ssh_target="mac",
        ssh_key="key",
        mac_artifact_root="/mnt/artifacts",
        mac_windows_project_mount="/mnt",
        artifact_subdir="mac_api/fail01",
        command="false",
        runner=fake_runner,
    )

    assert result["status"] == "failed"
    assert result["returncode"] == 7
    assert result["delivery_allowed"] is False


def test_remote_api_runner_reports_missing_ssh_client_fail_closed() -> None:
    def fake_runner(*args, **kwargs):
        raise FileNotFoundError("ssh")

    result = run_macos_remote_api_command(
        ssh_target="mac",
        ssh_key="key",
        mac_artifact_root="/mnt/artifacts",
        mac_windows_project_mount="/mnt",
        artifact_subdir="mac_api/no_ssh",
        command="true",
        runner=fake_runner,
    )

    assert result["status"] == "failed"
    assert result["returncode"] == 127
    assert result["error"]["type"] == "ssh_client_unavailable"
    assert result["delivery_allowed"] is False


def test_remote_api_runner_blocks_without_windows_project_mount_before_ssh() -> None:
    def fake_runner(*_args, **_kwargs):
        raise AssertionError("runner must not be called")

    result = run_macos_remote_api_command(
        ssh_target="mac",
        ssh_key="key",
        mac_artifact_root="/Users/andrejviluha/AIMusicProject_win/artifacts",
        artifact_subdir="mac_api/no_mount",
        command="true",
        runner=fake_runner,
    )

    assert result["status"] == "blocked"
    assert result["returncode"] == 78
    assert result["blocked_reason"] == "mac_windows_project_mount_missing"
    assert result["delivery_allowed"] is False
    assert result["mount_preflight"]["can_write_route_artifacts"] is False


def test_remote_api_runner_blocks_mac_local_result_root_before_ssh() -> None:
    def fake_runner(*_args, **_kwargs):
        raise AssertionError("runner must not be called")

    result = run_macos_remote_api_command(
        ssh_target="mac",
        ssh_key="key",
        mac_artifact_root="/Users/andrejviluha/aimusic_local_artifacts/host_artifacts",
        mac_windows_project_mount="/Users/andrejviluha/AIMusicProject_win",
        artifact_subdir="loop/it999_mac",
        command="true",
        runner=fake_runner,
    )

    assert result["status"] == "blocked"
    assert result["returncode"] == 78
    assert result["blocked_reason"] == "mac_output_path_uses_forbidden_local_artifact_root"
    assert result["delivery_allowed"] is False
    assert result["mount_preflight"]["can_write_route_artifacts"] is False


def test_remote_api_runner_blocks_mac_local_artifacts_parent_before_ssh() -> None:
    def fake_runner(*_args, **_kwargs):
        raise AssertionError("runner must not be called")

    result = run_macos_remote_api_command(
        ssh_target="mac",
        ssh_key="key",
        mac_artifact_root="/Users/andrejviluha/aimusic_local_artifacts",
        mac_windows_project_mount="/Users/andrejviluha/AIMusicProject_win",
        artifact_subdir="scratch/r001",
        command="true",
        runner=fake_runner,
    )

    assert result["status"] == "blocked"
    assert result["returncode"] == 78
    assert result["blocked_reason"] == "mac_output_path_uses_forbidden_local_artifact_root"
    assert result["mount_preflight"]["forbidden_mac_local_artifact_root"] == (
        "/Users/andrejviluha/aimusic_local_artifacts"
    )
    assert result["delivery_allowed"] is False


def test_parse_args_accepts_command_file_instead_of_inline_command(tmp_path: Path) -> None:
    command_file = tmp_path / "cmd.sh"
    command_file.write_text("python3 - <<'PY'\nprint('quoted')\nPY\n", encoding="utf-8")

    args = parse_args(
        [
            "--ssh-target",
            "mac",
            "--ssh-key",
            "key",
            "--artifact-subdir",
            "mac_api/cmd01",
            "--mac-windows-project-mount",
            "/Users/andrejviluha/AIMusicProject_win",
            "--command-file",
            str(command_file),
        ]
    )

    assert args.command is None
    assert args.command_file == str(command_file)
    assert args.mac_windows_project_mount == "/Users/andrejviluha/AIMusicProject_win"


def test_parse_args_defaults_to_shared_windows_artifact_mount() -> None:
    args = parse_args(
        [
            "--ssh-target",
            "mac",
            "--ssh-key",
            "key",
            "--artifact-subdir",
            "mac_api/cmd02",
            "--command",
            "printf ok",
        ]
    )

    assert args.mac_windows_project_mount == CANONICAL_MAC_WINDOWS_PROJECT_MOUNT
    assert args.mac_artifact_root == CANONICAL_MAC_ARTIFACT_ROOT
