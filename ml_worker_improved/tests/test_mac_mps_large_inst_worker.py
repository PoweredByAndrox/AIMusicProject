from __future__ import annotations

import importlib.util
import json
import socket
import sys
import tempfile
import threading
import unittest
import stat
from pathlib import Path


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
WORKER_PATH = TOOLS_ROOT / "mac_mps_large_inst_worker.py"
CLIENT_PATH = TOOLS_ROOT / "mac_mps_worker_client.py"


def _load(path: Path, name: str):
    tools_root = str(TOOLS_ROOT)
    if tools_root not in sys.path:
        sys.path.insert(0, tools_root)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _config(module, root: Path):
    project_root = root / "AIMusicProject_win"
    artifacts_root = project_root / "artifacts"
    state_dir = artifacts_root / "mac_api" / "wrk" / "state"
    runtime_repo = project_root / "ml_worker_improved" / "vendor" / "runtime"
    runtime_repo.mkdir(parents=True)
    implementation_path = runtime_repo / "bs_roformer.py"
    config_path = runtime_repo / "config.yaml"
    checkpoint_path = runtime_repo / "model.ckpt"
    for path in (implementation_path, config_path, checkpoint_path):
        path.write_bytes(b"fixture")
    return module.LargeInstWorkerConfig(
        socket_path=root / "worker.sock",
        project_root=project_root,
        artifacts_root=artifacts_root,
        state_dir=state_dir,
        windows_project_root=r"D:\AIMusicProject",
        runtime_repo=runtime_repo,
        model_implementation_path=implementation_path,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    ).validated()


class TestMacMpsLargeInstWorker(unittest.TestCase):
    def test_runtime_environment_forbids_implicit_fallback(self) -> None:
        module = _load(WORKER_PATH, "mac_mps_large_inst_worker_env")
        module.require_mps_environment(
            {
                "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                "AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC": "1",
            }
        )
        with self.assertRaisesRegex(RuntimeError, "PYTORCH_ENABLE_MPS_FALLBACK=0"):
            module.require_mps_environment({"AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC": "1"})
        with self.assertRaisesRegex(RuntimeError, "AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC=1"):
            module.require_mps_environment({"PYTORCH_ENABLE_MPS_FALLBACK": "0"})

    def test_request_policy_requires_project_source_and_mounted_artifact_output(self) -> None:
        module = _load(WORKER_PATH, "mac_mps_large_inst_worker_policy")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _config(module, root)
            source_path = config.project_root / "testing_stuff_music" / "mix.flac"
            source_path.parent.mkdir(parents=True)
            source_path.write_bytes(b"audio")
            output_dir = config.artifacts_root / "mac_api" / "wrk" / "r001"

            request_id, resolved_source, resolved_output, preflight = module.validate_separate_request(
                {
                    "request_id": "r001",
                    "source_audio_path": str(source_path),
                    "output_dir": str(output_dir),
                },
                config=config,
            )

            self.assertEqual(request_id, "r001")
            self.assertEqual(resolved_source, source_path.resolve())
            self.assertEqual(resolved_output, output_dir.resolve())
            self.assertEqual(preflight["preflight_status"], "passed")
            self.assertEqual(
                preflight["windows_output_path"],
                r"D:\AIMusicProject\artifacts\mac_api\wrk\r001",
            )

            outside_source = root / "outside.flac"
            outside_source.write_bytes(b"audio")
            with self.assertRaisesRegex(ValueError, "inside project_root"):
                module.validate_separate_request(
                    {
                        "request_id": "r002",
                        "source_audio_path": str(outside_source),
                        "output_dir": str(config.artifacts_root / "mac_api" / "wrk" / "r002"),
                    },
                    config=config,
                )

    def test_request_policy_rejects_unsafe_alias_and_nonempty_output(self) -> None:
        module = _load(WORKER_PATH, "mac_mps_large_inst_worker_alias")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _config(module, root)
            source_path = config.project_root / "testing_stuff_music" / "mix.flac"
            source_path.parent.mkdir(parents=True)
            source_path.write_bytes(b"audio")
            output_dir = config.artifacts_root / "mac_api" / "wrk" / "r001"
            output_dir.mkdir(parents=True)
            (output_dir / "existing.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Windows-safe alias"):
                module.validate_separate_request(
                    {
                        "request_id": "raw title with spaces",
                        "source_audio_path": str(source_path),
                        "output_dir": str(config.artifacts_root / "mac_api" / "wrk" / "r002"),
                    },
                    config=config,
                )
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                module.validate_separate_request(
                    {
                        "request_id": "r001",
                        "source_audio_path": str(source_path),
                        "output_dir": str(output_dir),
                    },
                    config=config,
                )

    def test_request_policy_allows_explicit_disposable_scratch(self) -> None:
        module = _load(WORKER_PATH, "mac_mps_large_inst_worker_scratch")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _config(module, root)
            scratch_root = Path(tempfile.gettempdir())
            source_path = scratch_root / "worker_source_fixture.flac"
            output_dir = scratch_root / "worker_output_fixture"
            source_path.write_bytes(b"audio")
            try:
                request_id, _, _, contract = module.validate_separate_request(
                    {
                        "request_id": "r003",
                        "source_audio_path": str(source_path),
                        "source_scope": "disposable_scratch",
                        "output_dir": str(output_dir),
                        "output_scope": "disposable_scratch",
                    },
                    config=config,
                )
            finally:
                source_path.unlink(missing_ok=True)

            self.assertEqual(request_id, "r003")
            self.assertEqual(contract["output_scope"], "disposable_scratch")
            self.assertFalse(contract["can_write_route_artifacts"])
            self.assertTrue(contract["cleanup_required"])
            self.assertFalse(contract["delivery_allowed"])

    def test_unix_server_health_separate_and_shutdown_protocol(self) -> None:
        worker = _load(WORKER_PATH, "mac_mps_large_inst_worker_server")
        client = _load(CLIENT_PATH, "mac_mps_worker_client_test")

        class FakeRuntime:
            def __init__(self) -> None:
                self.calls = 0

            def health(self):
                return {
                    "schema": worker.RESPONSE_SCHEMA,
                    "status": "ready",
                    "model_load_count": 1,
                    "delivery_allowed": False,
                }

            def separate(self, payload):
                self.calls += 1
                return {
                    "schema": worker.RESPONSE_SCHEMA,
                    "status": "ok",
                    "request_id": payload["request_id"],
                    "request_count": self.calls,
                    "model_load_count": 1,
                    "delivery_allowed": False,
                }

            def release_cache(self):
                return {
                    "schema": worker.RESPONSE_SCHEMA,
                    "status": "ready",
                    "action": "release_cache",
                    "model_load_count": 1,
                    "delivery_allowed": False,
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            socket_path = Path(temp_dir) / "worker.sock"
            runtime = FakeRuntime()
            server = worker.LargeInstWorkerServer(socket_path, runtime)
            self.assertEqual(stat.S_IMODE(socket_path.stat().st_mode), 0o600)
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
            thread.start()
            try:
                health = client.send_request(socket_path, {"action": "health"}, 2.0)
                response = client.send_request(
                    socket_path,
                    {"action": "separate", "request_id": "r001"},
                    2.0,
                )
                released = client.send_request(socket_path, {"action": "release_cache"}, 2.0)
                stopping = client.send_request(socket_path, {"action": "shutdown"}, 2.0)
                thread.join(timeout=2.0)
            finally:
                server.server_close()

            self.assertEqual(health["status"], "ready")
            self.assertEqual(health["model_load_count"], 1)
            self.assertEqual(response["status"], "ok")
            self.assertEqual(response["request_count"], 1)
            self.assertFalse(response["delivery_allowed"])
            self.assertEqual(released["action"], "release_cache")
            self.assertEqual(released["model_load_count"], 1)
            self.assertEqual(stopping["status"], "stopping")
            self.assertFalse(thread.is_alive())
            self.assertFalse(socket_path.exists())

    def test_runtime_requires_exact_expected_model_fingerprint_before_inference(self) -> None:
        module = _load(WORKER_PATH, "mac_mps_large_inst_worker_fingerprint")
        runtime = object.__new__(module.LargeInstModelRuntime)
        runtime.model = object()
        runtime.run_folder = object()
        runtime.poisoned_reason = None
        runtime.model_fingerprint = {
            "model_id": "bs_roformer_large_inst",
            "config_sha256": "config",
            "checkpoint_sha256": "checkpoint",
            "implementation_sha256": "implementation",
        }

        with self.assertRaisesRegex(ValueError, "expected_model_fingerprint is required"):
            runtime.separate({})
        with self.assertRaisesRegex(ValueError, "model_fingerprint_mismatch:config_sha256"):
            runtime.separate(
                {
                    "expected_model_fingerprint": {
                        **runtime.model_fingerprint,
                        "config_sha256": "wrong",
                    }
                }
            )

    def test_poisoned_runtime_preserves_restart_required_response(self) -> None:
        module = _load(WORKER_PATH, "mac_mps_large_inst_worker_poison")
        runtime = object.__new__(module.LargeInstModelRuntime)
        runtime.model = object()
        runtime.run_folder = object()
        runtime.poisoned_reason = "RuntimeError: synthetic MPS failure"
        runtime.model_load_count = 1
        runtime.request_count = 3

        response = runtime.separate({"request_id": "r004"})

        self.assertEqual(response["status"], "failed")
        self.assertTrue(response["restart_required"])
        self.assertEqual(response["request_id"], "r004")
        self.assertIn("worker_poisoned_restart_required", response["reason"])
        self.assertFalse(response["delivery_allowed"])

    def test_poisoned_runtime_rejects_release_cache_without_ready_state(self) -> None:
        module = _load(WORKER_PATH, "mac_mps_large_inst_worker_poison_cache")
        runtime = object.__new__(module.LargeInstModelRuntime)
        runtime._torch = object()
        runtime.model = object()
        runtime.poisoned_reason = "RuntimeError: synthetic MPS failure"
        runtime.model_load_count = 1
        runtime.request_count = 3

        response = runtime.release_cache()

        self.assertEqual(response["status"], "failed")
        self.assertEqual(response["action"], "release_cache")
        self.assertTrue(response["restart_required"])
        self.assertIn("worker_poisoned_restart_required", response["reason"])

    def test_unterminated_client_request_cannot_stall_worker(self) -> None:
        worker = _load(WORKER_PATH, "mac_mps_large_inst_worker_read_timeout")
        client = _load(CLIENT_PATH, "mac_mps_worker_client_read_timeout")

        class FakeRuntime:
            def health(self):
                return {
                    "schema": worker.RESPONSE_SCHEMA,
                    "status": "ready",
                    "model_load_count": 1,
                    "delivery_allowed": False,
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            socket_path = Path(temp_dir) / "worker.sock"
            server = worker.LargeInstWorkerServer(socket_path, FakeRuntime(), read_timeout_sec=0.05)
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
            thread.start()
            slow_client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            slow_client.settimeout(1.0)
            try:
                slow_client.connect(str(socket_path))
                slow_client.sendall(b'{"action":"health"')
                timeout_response = json.loads(slow_client.recv(4096).decode("utf-8"))
                health = client.send_request(socket_path, {"action": "health"}, 1.0)
            finally:
                slow_client.close()
                server.shutdown()
                thread.join(timeout=2.0)
                server.server_close()

            self.assertEqual(timeout_response["reason"], "request_read_timeout")
            self.assertEqual(health["status"], "ready")
            self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
