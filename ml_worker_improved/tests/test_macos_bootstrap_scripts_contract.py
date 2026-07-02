from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_repo_text(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8-sig")


class TestMacosBootstrapScriptsContract(unittest.TestCase):
    def test_macos_bootstrap_and_connect_scripts_pin_current_runtime_contract(self) -> None:
        bootstrap_and_ssh = REPO_ROOT / "scripts" / "macos-bootstrap-and-ssh.ps1"
        self.assertTrue(bootstrap_and_ssh.is_file())

        bootstrap_and_ssh_text = bootstrap_and_ssh.read_text(encoding="utf-8-sig")
        bootstrap_mount_text = read_repo_text("scripts/macos-bootstrap-workspace-mount.ps1")
        ssh_text = read_repo_text("scripts/macos-ssh.ps1")

        self.assertIn("macos-bootstrap-workspace-mount.ps1", bootstrap_and_ssh_text)
        self.assertIn("macos-ssh.ps1", bootstrap_and_ssh_text)
        self.assertIn("/usr/local/bin/container --version", bootstrap_and_ssh_text)
        self.assertIn("192.168.0.138", bootstrap_and_ssh_text)
        self.assertIn("id_ed25519_phone", bootstrap_and_ssh_text)
        self.assertIn("/Users/andrejviluha/AIMusicProject_win", bootstrap_and_ssh_text)

        self.assertIn("AIMusicProject", bootstrap_mount_text)
        self.assertIn("MacAudioApi", bootstrap_mount_text)
        self.assertIn("/Users/andrejviluha/AIMusicProject_win", bootstrap_mount_text)
        self.assertIn("mount_smbfs", bootstrap_mount_text)
        self.assertIn(".aimusic_mount_probe", bootstrap_mount_text)

        self.assertIn("id_ed25519_phone", ssh_text)
        self.assertIn("IdentitiesOnly=yes", ssh_text)
        self.assertIn("BatchMode=yes", ssh_text)
        self.assertIn("/Users/andrejviluha/AIMusicProject_win", ssh_text)
        self.assertIn("aimusic_local_artifacts/host_artifacts", ssh_text)
        self.assertIn("Forbidden Mac-local result artifact root", ssh_text)
        self.assertIn("$MacProjectRoot/artifacts", ssh_text)

    def test_macos_runtime_docs_expose_only_current_bootstrap_commands(self) -> None:
        runtime_doc = read_repo_text("docs/MACOS_8GB_AUDIO_RUNTIME.md")
        remote_api_doc = read_repo_text("ml_worker_improved/docs/MACOS_REMOTE_API_COMMAND_README.md")
        mac_mps_api_doc = read_repo_text("docs/MAC_MPS_SEPARATOR_API.md")

        for doc in (runtime_doc, remote_api_doc, mac_mps_api_doc):
            self.assertIn("powershell -ExecutionPolicy Bypass -File scripts\\macos-bootstrap-and-ssh.ps1", doc)
            self.assertIn("powershell -ExecutionPolicy Bypass -File scripts\\macos-ssh.ps1", doc)
            self.assertIn("192.168.0.138", doc)
            self.assertIn("/Users/andrejviluha/AIMusicProject_win", doc)
            self.assertNotIn("192.168.0.167", doc)
            self.assertIn("/Users/andrejviluha/AIMusicProject_win/artifacts", doc)

        for doc in (runtime_doc, remote_api_doc):
            self.assertIn("powershell -ExecutionPolicy Bypass -File scripts\\macos-bootstrap-workspace-mount.ps1", doc)
            self.assertIn("id_ed25519_phone", doc)
            self.assertIn("/Users/andrejviluha/aimusic_local_artifacts/host_artifacts", doc)
            self.assertIn("/Users/andrejviluha/win_artifacts", doc)
            self.assertIn("forbidden", doc.lower())


if __name__ == "__main__":
    unittest.main()
