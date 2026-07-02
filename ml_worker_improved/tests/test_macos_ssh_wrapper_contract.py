from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MACOS_SSH_SCRIPT = REPO_ROOT / "scripts" / "macos-ssh.ps1"


def test_macos_ssh_exports_windows_mount_artifact_env_contract() -> None:
    script = MACOS_SSH_SCRIPT.read_text(encoding="utf-8-sig")

    assert '$MacArtifactRoot = Join-RemotePosixPath -Root $MacProjectRoot -Child "artifacts"' in script
    assert "export AIMUSIC_MAC_PROJECT_ROOT=$remoteRoot" in script
    assert "export AIMUSIC_MAC_ARTIFACT_ROOT=$remoteArtifactRoot" in script
    assert "export AIMUSIC_MAC_LAB_ARTIFACTS_ROOT=$remoteLabArtifactsRoot" in script
    assert "export AIMUSIC_MAC_ARTIFACT_RAG_ROOT=$remoteArtifactRagRoot" in script


def test_macos_ssh_blocks_home_relative_local_artifact_roots() -> None:
    script = MACOS_SSH_SCRIPT.read_text(encoding="utf-8-sig")

    assert "'$HOME/aimusic_local_artifacts'" in script
    assert "'${HOME}/aimusic_local_artifacts'" in script
    assert "'~/aimusic_local_artifacts'" in script
    assert "Forbidden Mac-local result artifact root in RemoteCommand" in script
