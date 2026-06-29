from __future__ import annotations

import subprocess

import pytest

from tools.run_macos_remote_api_command import (
    PRESENTATION_STATUS,
    build_remote_script,
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
        mac_artifact_root="/Users/andrejviluha/win_artifacts",
        artifact_subdir="mac_api/c001",
        metadata={"purpose": "container_smoke"},
    )

    assert "/Users/andrejviluha/win_artifacts/mac_api/c001" in script
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
        ssh_target="andrejviluha@192.168.0.167",
        ssh_key="C:/Users/PBAX/.ssh/id_ed25519_phone",
        mac_artifact_root="/Users/andrejviluha/win_artifacts",
        artifact_subdir="mac_api/c002",
        command="printf hello",
        runner=fake_runner,
    )

    assert result["status"] == "passed"
    assert result["delivery_allowed"] is False
    assert result["presentation_status"] == PRESENTATION_STATUS
    assert result["remote_manifest_path"].endswith("/mac_api/c002/manifest.json")
    ssh_command = calls[0][0][0]
    kwargs = calls[0][1]
    assert ssh_command[-1] == "sh -s"
    assert kwargs["input"].startswith(b"#!/bin/sh")
    assert kwargs["capture_output"] is True
    assert kwargs["check"] is False


def test_remote_api_runner_reports_failed_command_without_accepting() -> None:
    def fake_runner(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=7, stdout="", stderr="bad")

    result = run_macos_remote_api_command(
        ssh_target="mac",
        ssh_key="key",
        mac_artifact_root="/mnt",
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
        mac_artifact_root="/mnt",
        artifact_subdir="mac_api/no_ssh",
        command="true",
        runner=fake_runner,
    )

    assert result["status"] == "failed"
    assert result["returncode"] == 127
    assert result["error"]["type"] == "ssh_client_unavailable"
    assert result["delivery_allowed"] is False
