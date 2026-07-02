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


def test_mac_api_mdx23c_route_uses_local_vendor_paths(tmp_path):
    module = _load_module()
    config = module.MacMpsApiConfig(
        project_root=tmp_path / "project",
        artifacts_root=tmp_path / "artifacts",
        python_bin=tmp_path / "venv" / "bin" / "python",
    )

    command, env = module.build_separator_export_command(
        config=config,
        route_id="mdx23c_vocals",
        source_audio_path=tmp_path / "mix.flac",
        output_dir=tmp_path / "artifacts" / "mac_api" / "t001" / "run",
    )

    assert env["AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC"] == "1"
    assert "tools/export_mdx23c_vocal_full.py" in command[1].replace("\\", "/")
    assert "--repo-path" in command
    assert str(tmp_path / "project" / "ml_worker_improved" / "vendor" / "music_source_separation_training") in command
    assert "--config-path" in command
    assert str(tmp_path / "project" / "ml_worker_improved" / "vendor" / "mdx23c_vocals_official" / "config_vocals_mdx23c.yaml") in command
    assert "--model-path" in command
    assert str(tmp_path / "project" / "ml_worker_improved" / "vendor" / "mdx23c_vocals_official" / "model_vocals_mdx23c_sdr_10.17.ckpt") in command
    assert "--timeout-sec" not in command


def test_mac_api_becruily_instrumental_route_uses_dedicated_minus_checkpoint(tmp_path):
    module = _load_module()
    config = module.MacMpsApiConfig(
        project_root=tmp_path / "project",
        artifacts_root=tmp_path / "artifacts",
        python_bin=tmp_path / "venv" / "bin" / "python",
    )

    command, env = module.build_separator_export_command(
        config=config,
        route_id="becruily_instrumental_vocals",
        source_audio_path=tmp_path / "mix.flac",
        output_dir=tmp_path / "artifacts" / "mac_api" / "t001" / "run",
    )

    vendor_root = tmp_path / "project" / "ml_worker_improved" / "vendor"
    assert env["AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC"] == "1"
    assert "tools/export_melband_vocal_full.py" in command[1].replace("\\", "/")
    assert "--config-path" in command
    assert str(vendor_root / "becruily_instrumental" / "config_instrumental_becruily.yaml") in command
    assert "--model-path" in command
    assert str(vendor_root / "becruily_instrumental" / "mel_band_roformer_instrumental_becruily.ckpt") in command
    assert "--manifest-model-id" in command
    assert command[command.index("--manifest-model-id") + 1] == "becruily_instrumental_vocals"


def test_mac_api_scnet_route_uses_extended_mps_timeout(tmp_path):
    module = _load_module()
    config = module.MacMpsApiConfig(
        project_root=tmp_path / "project",
        artifacts_root=tmp_path / "artifacts",
        python_bin=tmp_path / "venv" / "bin" / "python",
    )

    command, env = module.build_separator_export_command(
        config=config,
        route_id="scnet_xl_ihf_musdb18",
        source_audio_path=tmp_path / "mix.flac",
        output_dir=tmp_path / "artifacts" / "mac_api" / "t001" / "run",
    )

    assert env["AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC"] == "1"
    assert "tools/export_scnet_full.py" in command[1].replace("\\", "/")
    assert "--timeout-sec" in command
    assert command[command.index("--timeout-sec") + 1] == module.SCNET_MPS_TIMEOUT_SEC


def test_mac_api_rejects_unknown_route(tmp_path):
    module = _load_module()
    config = module.MacMpsApiConfig(
        project_root=tmp_path / "project",
        artifacts_root=tmp_path / "artifacts",
        python_bin=tmp_path / "venv" / "bin" / "python",
    )

    with pytest.raises(ValueError, match="unsupported Mac MPS route"):
        module.build_separator_export_command(
            config=config,
            route_id="raw_track_title_bad_route",
            source_audio_path=tmp_path / "mix.flac",
            output_dir=tmp_path / "artifacts" / "mac_api" / "t001" / "run",
        )


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
    project_root = tmp_path / "AIMusicProject_win"
    project_root.mkdir()
    python_bin = tmp_path / "venv" / "bin" / "python"

    config = module.build_config(
        project_root=project_root,
        artifacts_root=project_root / "artifacts",
        python_bin=python_bin,
        mac_windows_project_mount=project_root,
    )

    assert config.project_root == project_root.resolve()
    assert config.artifacts_root == (project_root / "artifacts").resolve()
    assert config.python_bin == python_bin
    assert config.mac_windows_project_mount == project_root.resolve()


def test_mac_api_config_blocks_forbidden_mac_local_artifact_root(tmp_path):
    module = _load_module()
    project_root = tmp_path / "AIMusicProject_win"
    project_root.mkdir()

    with pytest.raises(ValueError, match="mac_output_path_uses_forbidden_local_artifact_root"):
        module.build_config(
            project_root=project_root,
            artifacts_root=Path("/Users/andrejviluha/aimusic_local_artifacts/mac_api"),
            python_bin=tmp_path / "venv" / "bin" / "python",
            mac_windows_project_mount=project_root,
        )


def test_mac_api_config_requires_windows_project_artifacts_root(tmp_path):
    module = _load_module()
    project_root = tmp_path / "AIMusicProject_win"
    project_root.mkdir()

    with pytest.raises(ValueError, match="mac_mps_api_artifacts_root_must_be_under_windows_artifacts"):
        module.build_config(
            project_root=project_root,
            artifacts_root=project_root / "lab_artifacts",
            python_bin=tmp_path / "venv" / "bin" / "python",
            mac_windows_project_mount=project_root,
        )


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


def test_mac_api_start_job_blocks_without_windows_project_mount(tmp_path):
    module = _load_module()
    source_audio = tmp_path / "mix.flac"
    source_audio.write_bytes(b"audio")
    artifacts_root = tmp_path / "local_artifacts"
    runner = module.MacMpsJobRunner(
        module.MacMpsApiConfig(
            project_root=tmp_path / "project",
            artifacts_root=artifacts_root,
            python_bin=tmp_path / "venv" / "bin" / "python",
        )
    )

    manifest = runner.start_job({"source_audio_path": str(source_audio), "output_subdir": "mac_api/m001"})

    assert manifest["status"] == "blocked"
    assert manifest["blocked_reason"] == "mac_windows_project_mount_missing"
    assert manifest["delivery_allowed"] is False
    assert manifest["mount_preflight"]["can_write_route_artifacts"] is False
    assert not (artifacts_root / "mac_api").exists()
