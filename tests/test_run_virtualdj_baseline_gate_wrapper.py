from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = PROJECT_ROOT / "scripts" / "run-virtualdj-baseline-gate.ps1"


def write_minimal_virtualdj_data(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "settings.xml").write_text("<settings />", encoding="utf-8")
    (root / "database.xml").write_text('<VirtualDJ_Database Version="2026" />', encoding="utf-8")
    (root / "Cache").mkdir()
    (root / "History").mkdir()
    (root / "Plugins64").mkdir()


def create_fake_docker(fake_bin: Path, calls_path: Path, project_root: Path) -> None:
    fake_bin.mkdir(parents=True, exist_ok=True)
    fake_docker_py = fake_bin / "fake_docker.py"
    fake_docker_py.write_text(
        """
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


argv = sys.argv[1:]
calls_path = Path(os.environ["FAKE_DOCKER_CALLS"])
project_root = Path(os.environ["FAKE_PROJECT_ROOT"])
calls_path.parent.mkdir(parents=True, exist_ok=True)
with calls_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"argv": argv}) + "\\n")

if "/app/scripts/virtualdj_baseline_manifest.py" in argv:
    output_json = Path("/app/lab_artifacts/virtualdj/baseline_manifest.json")
    if "--output-json" in argv:
        output_json = Path(argv[argv.index("--output-json") + 1].replace("\\\\", "/"))
    if not str(output_json).replace("\\\\", "/").startswith("/app/"):
        raise SystemExit(f"unexpected output-json: {output_json}")
    host_output = project_root / str(output_json).replace("\\\\", "/")[len("/app/"):]
    host_output.parent.mkdir(parents=True, exist_ok=True)
    host_output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "quality_gate_hint": {
                    "can_run_nine_thou_proxy_gate": True,
                    "reference_minus_docker_path": "/virtualdj_export/Nine Thou Instrumental.flac",
                    "reason": "instrumental_mix_export_found",
                },
            },
            indent=2,
        )
        + "\\n",
        encoding="utf-8",
    )
    print(host_output)
elif "/app/scripts/run_nine_thou_quality_gate.py" in argv:
    print("fake quality gate")
else:
    raise SystemExit(f"unexpected docker argv: {argv}")
""".lstrip(),
        encoding="utf-8",
    )
    python_exe = str(sys.executable).replace("'", "''")
    fake_script = str(fake_docker_py).replace("'", "''")
    docker_ps1 = fake_bin / "docker.ps1"
    docker_ps1.write_text(
        f"$ErrorActionPreference = 'Stop'\n& '{python_exe}' '{fake_script}' @args\nexit $LASTEXITCODE\n",
        encoding="utf-8",
    )


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class NonVirtualDjHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"plain localhost service")

    def log_message(self, format: str, *args: object) -> None:
        return


@unittest.skipIf(os.name != "nt", "PowerShell wrapper test is Windows-only")
class TestRunVirtualDjBaselineGateWrapper(unittest.TestCase):
    def run_wrapper(self, temp_root: Path, fake_bin: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
        powershell = shutil.which("powershell")
        self.assertIsNotNone(powershell)
        env = os.environ.copy()
        env["PATH"] = str(fake_bin) + os.pathsep + env.get("PATH", "")
        env["FAKE_DOCKER_CALLS"] = str(temp_root / "fake_docker_calls.jsonl")
        env["FAKE_PROJECT_ROOT"] = str(temp_root)
        return subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(WRAPPER_PATH),
                *args,
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def make_common_fixture(self, temp_root: Path) -> dict[str, Path]:
        virtualdj_data = temp_root / "VirtualDJ"
        export_dir = temp_root / "exports"
        write_minimal_virtualdj_data(virtualdj_data)
        export_dir.mkdir(parents=True)
        (export_dir / "Nine Thou Instrumental.flac").write_bytes(b"fixture")
        full_mix = temp_root / "full.mp3"
        full_mix.write_bytes(b"full")
        virtualdj_exe = temp_root / "virtualdj.exe"
        virtualdj_exe.write_bytes(b"fake exe")
        fake_bin = temp_root / "fake_bin"
        create_fake_docker(fake_bin, temp_root / "fake_docker_calls.jsonl", temp_root)
        return {
            "virtualdj_data": virtualdj_data,
            "export_dir": export_dir,
            "full_mix": full_mix,
            "virtualdj_exe": virtualdj_exe,
            "fake_bin": fake_bin,
        }

    def test_run_quality_gate_preserves_reference_minus_path_mapping_and_gate_args(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            fixture = self.make_common_fixture(temp_root)

            completed = self.run_wrapper(
                temp_root,
                fixture["fake_bin"],
                [
                    "-ProjectRoot",
                    str(temp_root),
                    "-VirtualDjExe",
                    str(fixture["virtualdj_exe"]),
                    "-VirtualDjDataRoot",
                    str(fixture["virtualdj_data"]),
                    "-ExportDir",
                    str(fixture["export_dir"]),
                    "-FullMix",
                    str(fixture["full_mix"]),
                    "-ManifestOutput",
                    "/app/lab_artifacts/virtualdj/test_manifest.json",
                    "-OutputDir",
                    "/app/lab_artifacts/virtualdj/test_quality_gate",
                    "-RunQualityGate",
                    "--sample-rate",
                    "8000",
                ],
            )

            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            calls = load_jsonl(temp_root / "fake_docker_calls.jsonl")
            self.assertEqual(len(calls), 2)
            manifest_call = calls[0]["argv"]
            gate_call = calls[1]["argv"]
            self.assertIn("/app/scripts/virtualdj_baseline_manifest.py", manifest_call)
            self.assertIn("/app/scripts/run_nine_thou_quality_gate.py", gate_call)
            self.assertEqual(
                gate_call[gate_call.index("--reference-minus") + 1].replace("\\", "/"),
                "/virtualdj_export/Nine Thou Instrumental.flac",
            )
            self.assertEqual(gate_call[gate_call.index("--full-mix") + 1].replace("\\", "/"), "/host_project/full.mp3")
            self.assertEqual(gate_call[-2:], ["--sample-rate", "8000"])

    def test_manifest_output_cannot_escape_lab_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            fixture = self.make_common_fixture(temp_root)

            completed = self.run_wrapper(
                temp_root,
                fixture["fake_bin"],
                [
                    "-ProjectRoot",
                    str(temp_root),
                    "-VirtualDjExe",
                    str(fixture["virtualdj_exe"]),
                    "-VirtualDjDataRoot",
                    str(fixture["virtualdj_data"]),
                    "-ExportDir",
                    str(fixture["export_dir"]),
                    "-ManifestOutput",
                    "/app/lab_artifacts/../escape.json",
                ],
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse((temp_root / "fake_docker_calls.jsonl").exists())
            self.assertIn("ManifestOutput must not contain", completed.stderr + completed.stdout)

    def test_network_control_probe_rejects_plain_localhost_http_service(self) -> None:
        server = HTTPServer(("127.0.0.1", 0), NonVirtualDjHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_root = Path(temp_dir)
                fixture = self.make_common_fixture(temp_root)

                completed = self.run_wrapper(
                    temp_root,
                    fixture["fake_bin"],
                    [
                        "-ProjectRoot",
                        str(temp_root),
                        "-VirtualDjExe",
                        str(fixture["virtualdj_exe"]),
                        "-VirtualDjDataRoot",
                        str(fixture["virtualdj_data"]),
                        "-ExportDir",
                        str(fixture["export_dir"]),
                        "-ProbeNetworkControl",
                        "-NetworkControlHost",
                        "127.0.0.1",
                        "-NetworkControlPorts",
                        str(server.server_port),
                    ],
                )

                self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
                host_probe = json.loads(
                    (temp_root / "lab_artifacts" / "virtualdj" / "host_probe.json").read_text(encoding="utf-8-sig")
                )
                network_control = host_probe["virtualdj"]["network_control"]
                self.assertFalse(network_control["available"])
                self.assertEqual(network_control["status"], "unavailable")
                self.assertGreaterEqual(len(network_control["attempts"]), 2)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
