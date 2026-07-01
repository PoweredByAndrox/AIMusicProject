from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"


def _load_module():
    module_path = TOOLS_ROOT / "run_apple_container_project_smoke.py"
    spec = importlib.util.spec_from_file_location("run_apple_container_project_smoke", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestAppleContainerProjectSmoke(unittest.TestCase):
    def test_blocks_missing_windows_mount_without_creating_local_output(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            project_root = tmp_path / "project"
            project_root.mkdir()
            container_cli = tmp_path / "container"
            container_cli.write_text("#!/bin/sh\n", encoding="utf-8")
            output_dir = tmp_path / "local_out"

            manifest = module.run_apple_container_project_smoke(
                project_root=project_root,
                output_dir=output_dir,
                container_cli=container_cli,
                runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("runner must not be called")),
            )

            self.assertEqual(manifest["status"], "blocked")
            self.assertEqual(manifest["blocked_reason"], "mac_windows_project_mount_missing")
            self.assertFalse(manifest["delivery_allowed"])
            self.assertFalse(manifest["mount_preflight"]["can_write_route_artifacts"])
            self.assertFalse(output_dir.exists())

    def test_builds_readonly_project_test_command(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            project_root = tmp_path / "project"
            project_root.mkdir()
            output_dir = tmp_path / "artifacts" / "acsm001"
            container_cli = tmp_path / "container"
            container_cli.write_text("#!/bin/sh\n", encoding="utf-8")
            calls = []

            def fake_runner(command, **kwargs):
                calls.append((command, kwargs))
                return subprocess.CompletedProcess(command, 0, stdout="OK\n", stderr="")

            manifest = module.run_apple_container_project_smoke(
                project_root=project_root,
                output_dir=output_dir,
                container_cli=container_cli,
                image="docker.io/library/python:3.11-alpine",
                test_selectors=(
                    "tests.test_docker_runtime_policy.TestDockerRuntimePolicy."
                    "test_cli_runtime_contract_preserves_explicit_apple_container_runtime_marker",
                ),
                mac_windows_project_mount=tmp_path,
                runner=fake_runner,
            )

            self.assertEqual(manifest["schema"], "apple_container_project_smoke.v1")
            self.assertEqual(manifest["status"], "passed")
            self.assertFalse(manifest["delivery_allowed"])
            self.assertEqual(manifest["container_runtime"], "apple_container")
            self.assertEqual(manifest["runtime_environment"], "apple_container_project_smoke")
            self.assertTrue(manifest["gpu_inference_skipped"])
            self.assertEqual(
                manifest["test_selectors"],
                [
                    "tests.test_docker_runtime_policy.TestDockerRuntimePolicy."
                    "test_cli_runtime_contract_preserves_explicit_apple_container_runtime_marker"
                ],
            )
            command = calls[0][0]
            self.assertEqual(command[:4], [str(container_cli), "run", "--rm", "--mount"])
            self.assertIn(f"type=bind,source={project_root},target=/work,readonly", command)
            self.assertIn("-e", command)
            self.assertIn("AIMUSIC_CONTAINER_RUNTIME=apple_container", command)
            self.assertIn("--workdir", command)
            self.assertIn("/work/ml_worker_improved", command)
            self.assertIn("docker.io/library/python:3.11-alpine", command)
            self.assertNotIn("docker", Path(command[0]).name.lower())

    def test_fails_closed_when_container_cli_missing(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            project_root = tmp_path / "project"
            project_root.mkdir()
            output_dir = tmp_path / "artifacts" / "acsm001"
            missing_cli = tmp_path / "missing-container"

            manifest = module.run_apple_container_project_smoke(
                project_root=project_root,
                output_dir=output_dir,
                container_cli=missing_cli,
                mac_windows_project_mount=tmp_path,
                runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("runner must not be called")),
            )

            self.assertEqual(manifest["status"], "blocked")
            self.assertEqual(manifest["blocked_reason"], "apple_container_cli_missing")
            self.assertFalse(manifest["delivery_allowed"])
            self.assertEqual(manifest["returncode"], 127)
            self.assertIn(str(missing_cli), manifest["command"])
            self.assertIn(f"type=bind,source={project_root},target=/work,readonly", manifest["command"])
            self.assertIn("/work/ml_worker_improved", manifest["command"])
            cmd_json = module.json.loads((output_dir / "cmd.json").read_text(encoding="utf-8"))
            self.assertEqual(cmd_json["command"], manifest["command"])

    def test_records_command_provenance_when_project_root_missing(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            missing_project = tmp_path / "missing-project"
            output_dir = tmp_path / "artifacts" / "acsm001"
            container_cli = tmp_path / "container"
            container_cli.write_text("#!/bin/sh\n", encoding="utf-8")

            manifest = module.run_apple_container_project_smoke(
                project_root=missing_project,
                output_dir=output_dir,
                container_cli=container_cli,
                mac_windows_project_mount=tmp_path,
                runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("runner must not be called")),
            )

            self.assertEqual(manifest["status"], "blocked")
            self.assertEqual(manifest["blocked_reason"], "project_root_missing")
            self.assertIn(str(container_cli), manifest["command"])
            self.assertIn(f"type=bind,source={missing_project},target=/work,readonly", manifest["command"])
            self.assertIn("AIMUSIC_CONTAINER_RUNTIME=apple_container", manifest["command"])
            cmd_json = module.json.loads((output_dir / "cmd.json").read_text(encoding="utf-8"))
            self.assertEqual(cmd_json["command"], manifest["command"])

    def test_fails_closed_when_container_cli_cannot_execute(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            project_root = tmp_path / "project"
            project_root.mkdir()
            output_dir = tmp_path / "artifacts" / "acsm001"
            container_cli = tmp_path / "container"
            container_cli.write_text("#!/bin/sh\n", encoding="utf-8")

            def fake_runner(command, **kwargs):
                raise PermissionError("not executable")

            manifest = module.run_apple_container_project_smoke(
                project_root=project_root,
                output_dir=output_dir,
                container_cli=container_cli,
                mac_windows_project_mount=tmp_path,
                runner=fake_runner,
            )

            self.assertEqual(manifest["status"], "blocked")
            self.assertEqual(manifest["blocked_reason"], "apple_container_cli_unavailable")
            self.assertFalse(manifest["delivery_allowed"])
            self.assertEqual(manifest["returncode"], 126)
            self.assertIn(
                "not executable",
                (output_dir / "stderr.txt").read_text(encoding="utf-8"),
            )

    def test_records_failed_test_run_without_delivery(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            project_root = tmp_path / "project"
            project_root.mkdir()
            output_dir = tmp_path / "artifacts" / "acsm001"
            container_cli = tmp_path / "container"
            container_cli.write_text("#!/bin/sh\n", encoding="utf-8")

            def fake_runner(command, **kwargs):
                return subprocess.CompletedProcess(command, 3, stdout="FAIL\n", stderr="trace\n")

            manifest = module.run_apple_container_project_smoke(
                project_root=project_root,
                output_dir=output_dir,
                container_cli=container_cli,
                mac_windows_project_mount=tmp_path,
                runner=fake_runner,
            )

            self.assertEqual(manifest["status"], "failed")
            self.assertFalse(manifest["delivery_allowed"])
            self.assertEqual(manifest["returncode"], 3)
            self.assertEqual((output_dir / "stdout.txt").read_text(encoding="utf-8"), "FAIL\n")
            self.assertEqual((output_dir / "stderr.txt").read_text(encoding="utf-8"), "trace\n")
