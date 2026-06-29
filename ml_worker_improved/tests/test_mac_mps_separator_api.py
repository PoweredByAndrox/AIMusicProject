from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"


def _load_module():
    module_path = TOOLS_ROOT / "mac_mps_separator_api.py"
    spec = importlib.util.spec_from_file_location("mac_mps_separator_api", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mac_api_job_id_is_short_and_windows_safe():
    module = _load_module()

    job_id = module.make_job_id("ytmp3free.cc_karak-yani-rainbow-youtubemp3free.org.mp3")

    assert job_id.startswith("m")
    assert len(job_id) <= 16
    assert "karak" not in job_id
    assert all(ch.isalnum() or ch == "_" for ch in job_id)


def test_mac_api_rejects_output_outside_artifacts_root(tmp_path):
    module = _load_module()
    config = module.MacMpsApiConfig(
        project_root=tmp_path / "project",
        artifacts_root=tmp_path / "artifacts",
        python_bin=tmp_path / "venv" / "bin" / "python",
    )

    with pytest.raises(ValueError, match="outside artifacts root"):
        module.resolve_job_output_dir(config, "../artifact_rag/ar99")


def test_mac_api_resolves_safe_relative_output_dir(tmp_path):
    module = _load_module()
    config = module.MacMpsApiConfig(
        project_root=tmp_path / "project",
        artifacts_root=tmp_path / "artifacts",
        python_bin=tmp_path / "venv" / "bin" / "python",
    )

    output_dir = module.resolve_job_output_dir(config, "mac_api/t001")

    assert output_dir == (tmp_path / "artifacts" / "mac_api" / "t001").resolve()


def test_mac_api_export_command_uses_mps_native_diagnostic(tmp_path):
    module = _load_module()
    config = module.MacMpsApiConfig(
        project_root=tmp_path / "project",
        artifacts_root=tmp_path / "artifacts",
        python_bin=tmp_path / "venv" / "bin" / "python",
    )

    command, env = module.build_demucs_export_command(
        config=config,
        source_audio_path=tmp_path / "mix.flac",
        output_dir=tmp_path / "artifacts" / "mac_api" / "t001" / "run",
        model_name="htdemucs_6s",
    )

    assert env["AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC"] == "1"
    assert str(config.python_bin) == command[0]
    assert "tools/export_hq_stems_full.py" in command[1].replace("\\", "/")
    assert "--device" in command
    assert command[command.index("--device") + 1] == "mps"
    assert "--no-melody-stem" in command


def test_mac_api_job_manifest_is_diagnostic_only(tmp_path):
    module = _load_module()

    manifest = module.build_job_manifest(
        job_id="m000001",
        status="queued",
        source_audio_path=tmp_path / "mix.flac",
        output_dir=tmp_path / "artifacts" / "mac_api" / "m000001",
    )

    assert manifest["delivery_allowed"] is False
    assert manifest["listen_pack_created"] is False
    assert manifest["presentation_status"] == "diagnostic_mps_api_only"


def test_mac_api_config_preserves_venv_python_path(tmp_path):
    module = _load_module()
    python_bin = tmp_path / "venv" / "bin" / "python"

    config = module.build_config(
        project_root=tmp_path / "project",
        artifacts_root=tmp_path / "artifacts",
        python_bin=python_bin,
    )

    assert config.project_root == (tmp_path / "project").resolve()
    assert config.artifacts_root == (tmp_path / "artifacts").resolve()
    assert config.python_bin == python_bin


def test_mac_api_get_job_finds_manifest_after_restart(tmp_path):
    module = _load_module()
    artifacts_root = tmp_path / "artifacts"
    manifest_path = artifacts_root / "mac_api" / "api_full001" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        '{"schema":"mac_mps_separator_api_job.v1","job_id":"mabc123","status":"ok"}',
        encoding="utf-8",
    )
    runner = module.MacMpsJobRunner(
        module.MacMpsApiConfig(
            project_root=tmp_path / "project",
            artifacts_root=artifacts_root,
            python_bin=tmp_path / "venv" / "bin" / "python",
        )
    )

    manifest = runner.get_job("mabc123")

    assert manifest is not None
    assert manifest["status"] == "ok"
