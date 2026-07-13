from __future__ import annotations

import importlib.util
import http.client
import json
import threading
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


def test_request_timeout_defaults_and_rejects_unbounded_values():
    module = _load_module()

    assert module.resolve_request_timeout_sec({}, maximum_sec=600) == 600
    assert module.resolve_request_timeout_sec({"timeout_sec": 0}, maximum_sec=600) == 600
    assert module.resolve_request_timeout_sec({"timeout_sec": 30}, maximum_sec=600) == 30
    with pytest.raises(ValueError, match="must be positive"):
        module.resolve_request_timeout_sec({"timeout_sec": -1}, maximum_sec=600)
    with pytest.raises(ValueError, match="exceeds configured maximum"):
        module.resolve_request_timeout_sec({"timeout_sec": 601}, maximum_sec=600)


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
        timeout_sec=321,
    )

    assert env["AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC"] == "1"
    assert "tools/export_mdx23c_vocal_full.py" in command[1].replace("\\", "/")
    assert "--repo-path" in command
    assert str(tmp_path / "project" / "ml_worker_improved" / "vendor" / "music_source_separation_training") in command
    assert "--config-path" in command
    assert str(tmp_path / "project" / "ml_worker_improved" / "vendor" / "mdx23c_vocals_official" / "config_vocals_mdx23c.yaml") in command
    assert "--model-path" in command
    assert str(tmp_path / "project" / "ml_worker_improved" / "vendor" / "mdx23c_vocals_official" / "model_vocals_mdx23c_sdr_10.17.ckpt") in command
    assert "--timeout-sec" in command
    assert command[command.index("--timeout-sec") + 1] == "321"


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


def test_mac_api_fv7z_route_uses_verified_dedicated_instrumental_checkpoint(tmp_path):
    module = _load_module()
    config = module.MacMpsApiConfig(
        project_root=tmp_path / "project",
        artifacts_root=tmp_path / "artifacts",
        python_bin=tmp_path / "venv" / "bin" / "python",
    )

    command, env = module.build_separator_export_command(
        config=config,
        route_id="melband_inst_fv7z",
        source_audio_path=tmp_path / "mix.flac",
        output_dir=tmp_path / "artifacts" / "mac_api" / "t001" / "run",
    )

    model_root = tmp_path / "project" / "ml_worker_improved" / "vendor" / "melband_inst_fv7z_diag"
    assert env["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    assert "tools/export_melband_vocal_full.py" in command[1].replace("\\", "/")
    assert command[command.index("--config-path") + 1] == str(model_root / "config.yaml")
    assert command[command.index("--model-path") + 1] == str(
        model_root / "mel_band_roformer_instrumental_fv7z_gabox.ckpt"
    )
    assert command[command.index("--model-source-metadata-path") + 1] == str(model_root / "source.json")
    assert command[command.index("--manifest-model-id") + 1] == module.FV7Z_INSTRUMENTAL_ROUTE_ID
    assert command[command.index("--inference-chunk-size") + 1] == "485100"
    assert command[command.index("--inference-batch-size") + 1] == "1"
    assert command[command.index("--num-overlap") + 1] == "2"


def test_mac_api_becruily_karaoke_route_uses_registered_vocal_checkpoint(tmp_path):
    module = _load_module()
    config = module.MacMpsApiConfig(
        project_root=tmp_path / "project",
        artifacts_root=tmp_path / "artifacts",
        python_bin=tmp_path / "venv" / "bin" / "python",
    )

    command, env = module.build_separator_export_command(
        config=config,
        route_id="becruily_karaoke_vocals",
        source_audio_path=tmp_path / "mix.flac",
        output_dir=tmp_path / "artifacts" / "mac_api" / "t001" / "run",
    )

    vendor_root = tmp_path / "project" / "ml_worker_improved" / "vendor"
    assert env["AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC"] == "1"
    assert env["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    assert "tools/export_melband_vocal_full.py" in command[1].replace("\\", "/")
    assert str(vendor_root / "becruily_karaoke" / "config_karaoke_becruily.yaml") in command
    assert str(vendor_root / "becruily_karaoke" / "mel_band_roformer_karaoke_becruily.ckpt") in command
    assert command[command.index("--manifest-model-id") + 1] == "becruily_karaoke_vocals"


def test_mac_api_viperx_melband_route_uses_memory_safe_registered_checkpoint(tmp_path):
    module = _load_module()
    config = module.MacMpsApiConfig(
        project_root=tmp_path / "project",
        artifacts_root=tmp_path / "artifacts",
        python_bin=tmp_path / "venv" / "bin" / "python",
    )

    command, env = module.build_separator_export_command(
        config=config,
        route_id="audio_separator_melband_roformer_viperx_vocals",
        source_audio_path=tmp_path / "mix.flac",
        output_dir=tmp_path / "artifacts" / "mac_api" / "t001" / "run",
    )

    vendor_root = tmp_path / "project" / "ml_worker_improved" / "vendor"
    assert env["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    assert "tools/export_melband_vocal_full.py" in command[1].replace("\\", "/")
    assert str(
        vendor_root / "melband_roformer_viperx_11436" / "model_mel_band_roformer_ep_3005_sdr_11.4360.yaml"
    ) in command
    assert str(
        vendor_root / "melband_roformer_viperx_11436" / "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"
    ) in command
    assert command[command.index("--manifest-model-id") + 1] == module.VIPERX_MELBAND_ROUTE_ID
    assert command[command.index("--inference-batch-size") + 1] == "1"


def test_mac_api_scnet_route_uses_memory_safe_mps_profile(tmp_path):
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
    assert command[command.index("--chunk-size") + 1] == module.SCNET_MPS_CHUNK_SIZE
    assert command[command.index("--batch-size") + 1] == module.SCNET_MPS_BATCH_SIZE
    assert command[command.index("--num-overlap") + 1] == module.SCNET_MPS_NUM_OVERLAP


def test_mac_api_large_inst_route_uses_exact_architecture_and_no_mps_fallback(tmp_path):
    module = _load_module()
    config = module.MacMpsApiConfig(
        project_root=tmp_path / "project",
        artifacts_root=tmp_path / "artifacts",
        python_bin=tmp_path / "venv" / "bin" / "python",
    )

    command, env = module.build_separator_export_command(
        config=config,
        route_id="bs_roformer_large_inst",
        source_audio_path=tmp_path / "mix.flac",
        output_dir=tmp_path / "artifacts" / "mac_api" / "t001" / "run",
    )

    model_root = tmp_path / "project" / "ml_worker_improved" / "vendor" / "bs_roformer_large_inst_diag"
    assert env["AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC"] == "1"
    assert env["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    assert "tools/export_instrumental_resurrection_minus.py" in command[1].replace("\\", "/")
    assert command[command.index("--model-implementation-path") + 1] == str(model_root / "bs_roformer.py")
    assert command[command.index("--model-path") + 1] == str(model_root / "bs_large_v2_inst.ckpt")
    assert command[command.index("--model-source-metadata-path") + 1] == str(model_root / "source.json")
    assert command[command.index("--target-instrument") + 1] == "instrument"
    assert command[command.index("--device") + 1] == "mps"
    assert command[command.index("--inference-chunk-size") + 1] == "529200"
    assert command[command.index("--inference-batch-size") + 1] == "1"
    assert command[command.index("--inference-num-overlap") + 1] == "1"


def test_mac_api_resurrection_route_uses_official_dedicated_instrumental_checkpoint(tmp_path):
    module = _load_module()
    config = module.MacMpsApiConfig(
        project_root=tmp_path / "project",
        artifacts_root=tmp_path / "artifacts",
        python_bin=tmp_path / "venv" / "bin" / "python",
    )

    command, env = module.build_separator_export_command(
        config=config,
        route_id="bs_roformer_inst_res",
        source_audio_path=tmp_path / "mix.flac",
        output_dir=tmp_path / "artifacts" / "mac_api" / "t001" / "run",
    )

    model_root = (
        tmp_path / "project" / "ml_worker_improved" / "vendor" / "bs_roformer_instrumental_resurrection_manual"
    )
    assert env["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    assert "tools/export_instrumental_resurrection_minus.py" in command[1].replace("\\", "/")
    assert command[command.index("--config-path") + 1] == str(
        model_root / "config_bs_roformer_instrumental_resurrection_unwa.yaml"
    )
    assert command[command.index("--model-path") + 1] == str(
        model_root / "bs_roformer_instrumental_resurrection_unwa.ckpt"
    )
    assert command[command.index("--target-instrument") + 1] == "other"
    assert command[command.index("--device") + 1] == "mps"
    assert command[command.index("--inference-chunk-size") + 1] == "176400"
    assert command[command.index("--inference-batch-size") + 1] == "1"
    assert command[command.index("--inference-num-overlap") + 1] == "2"


def test_mac_api_large_inst_route_uses_configured_resident_worker_without_fallback(tmp_path):
    module = _load_module()
    socket_path = tmp_path / "li.sock"
    config = module.MacMpsApiConfig(
        project_root=tmp_path / "project",
        artifacts_root=tmp_path / "artifacts",
        python_bin=tmp_path / "venv" / "bin" / "python",
        large_inst_worker_socket=socket_path,
        large_inst_worker_timeout_sec=321,
    )

    command, env = module.build_separator_export_command(
        config=config,
        route_id="bs_roformer_large_inst",
        source_audio_path=tmp_path / "mix.flac",
        output_dir=tmp_path / "artifacts" / "mac_api" / "t001" / "run",
    )

    assert env["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    assert command[command.index("--resident-worker-socket") + 1] == str(socket_path)
    assert command[command.index("--resident-worker-timeout-sec") + 1] == "321"


def test_large_inst_minus_manifest_can_satisfy_instrumental_alias_without_stem_sum(tmp_path):
    module = _load_module()
    minus_path = tmp_path / "instrumental.flac"
    minus_path.write_bytes(b"dedicated minus")
    summary = {
        "target_stem": "minus",
        "stems": {
            "minus": {
                "path": str(minus_path),
                "source": "instrumental_resurrection_model_output",
                "source_stems": ["instrument"],
            }
        },
    }

    resolved = module._source_info_for_requested_stem(summary, "instrumental")

    assert resolved is not None
    assert resolved["path"] == str(minus_path)
    assert resolved["source"] == "instrumental_resurrection_model_output"


def test_official_resurrection_minus_manifest_can_satisfy_instrumental_alias(tmp_path):
    module = _load_module()
    minus_path = tmp_path / "instrumental.flac"
    minus_path.write_bytes(b"official dedicated minus")
    summary = {
        "target_stem": "minus",
        "stems": {
            "minus": {
                "path": str(minus_path),
                "source": "instrumental_resurrection_model_output",
                "source_stems": ["other"],
            }
        },
    }

    resolved = module._source_info_for_requested_stem(summary, "instrumental")

    assert resolved is not None
    assert resolved["path"] == str(minus_path)
    assert resolved["source_stems"] == ["other"]


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


def test_mac_api_stem_summary_preserves_residual_companion_role(tmp_path):
    module = _load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    vocals_path = run_dir / "vocals.flac"
    instrumental_path = run_dir / "instrumental.flac"
    vocals_path.write_bytes(b"vocals")
    instrumental_path.write_bytes(b"instrumental")
    (run_dir / "stem_manifest.json").write_text(
        """
{
  "stems": {
    "vocals": {
      "path": "VOCALS_PATH",
      "role": "companion",
      "source": "residual_from_raw_minus_instrumental",
      "model_output": false,
      "residual_companion": true
    },
    "instrumental": {
      "path": "INSTRUMENTAL_PATH",
      "role": "target",
      "source": "model_output",
      "model_output": true
    }
  }
}
""".replace("VOCALS_PATH", str(vocals_path).replace("\\", "\\\\")).replace(
            "INSTRUMENTAL_PATH", str(instrumental_path).replace("\\", "\\\\")
        ),
        encoding="utf-8",
    )

    summary = module._summarize_stems(run_dir)

    assert summary["stems"]["vocals"]["role"] == "companion"
    assert summary["stems"]["vocals"]["source"] == "residual_from_raw_minus_instrumental"
    assert summary["stems"]["vocals"]["model_output"] is False
    assert summary["stems"]["vocals"]["residual_companion"] is True
    assert summary["stems"]["instrumental"]["role"] == "target"
    assert summary["stems"]["instrumental"]["model_output"] is True


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


def test_mac_api_separate_adapts_persistent_contract_to_mps_export(tmp_path, monkeypatch):
    module = _load_module()
    project_root = tmp_path / "AIMusicProject_win"
    source_audio = project_root / "testing_stuff_music" / "mix.flac"
    source_audio.parent.mkdir(parents=True)
    source_audio.write_bytes(b"mix")
    artifacts_root = project_root / "artifacts"
    output_dir = artifacts_root / "rank" / "r001" / "c001"
    runner = module.MacMpsJobRunner(
        module.MacMpsApiConfig(
            project_root=project_root,
            artifacts_root=artifacts_root,
            python_bin=tmp_path / "venv" / "bin" / "python",
            mac_windows_project_mount=project_root,
        )
    )

    def fake_run(command, **_kwargs):
        run_dir = output_dir / "run"
        audio_dir = run_dir / "audio"
        audio_dir.mkdir(parents=True)
        vocals_path = audio_dir / "vocals.flac"
        instrumental_path = audio_dir / "instrumental.flac"
        vocals_path.write_bytes(b"vocals")
        instrumental_path.write_bytes(b"instrumental")
        (run_dir / "stem_manifest.json").write_text(
            """
{
  "target_stem": "instrumental",
  "target_product_stem": "minus",
  "product_aliases": {
    "minus": {
      "source_stem": "instrumental",
      "source_audio_path": "INSTRUMENTAL_PATH",
      "alias_of": "instrumental"
    }
  },
  "stems": {
    "vocals": {"path": "VOCALS_PATH", "role": "companion"},
    "instrumental": {"path": "INSTRUMENTAL_PATH", "role": "target"}
  }
}
""".replace("VOCALS_PATH", str(vocals_path).replace("\\", "\\\\")).replace(
                "INSTRUMENTAL_PATH", str(instrumental_path).replace("\\", "\\\\")
            ),
            encoding="utf-8",
        )
        return module.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    status_code, response = runner.separate(
        {
            "route_id": "becruily_instrumental_vocals",
            "source_audio_path": str(source_audio),
            "output_dir": str(output_dir),
            "device": "cuda",
            "require_gpu": True,
            "stems": ["instrumental", "vocals"],
            "timeout_sec": 30,
        }
    )

    assert status_code == 200
    assert response["status"] == "ok"
    assert response["actual_device"] == "mps"
    assert response["requested_device"] == "cuda"
    assert response["delivery_allowed"] is False
    assert response["can_accept_delivery"] is False
    assert response["missing_stems"] == []
    assert (output_dir / "stems" / "instrumental.flac").read_bytes() == b"instrumental"
    assert (output_dir / "stems" / "vocals.flac").read_bytes() == b"vocals"
    assert response["mount_preflight"]["preflight_status"] == "passed"


def test_mac_api_separate_maps_requested_minus_from_native_instrumental(tmp_path, monkeypatch):
    module = _load_module()
    project_root = tmp_path / "AIMusicProject_win"
    source_audio = project_root / "testing_stuff_music" / "mix.flac"
    source_audio.parent.mkdir(parents=True)
    source_audio.write_bytes(b"mix")
    artifacts_root = project_root / "artifacts"
    output_dir = artifacts_root / "rank" / "r001" / "c001"
    runner = module.MacMpsJobRunner(
        module.MacMpsApiConfig(
            project_root=project_root,
            artifacts_root=artifacts_root,
            python_bin=tmp_path / "venv" / "bin" / "python",
            mac_windows_project_mount=project_root,
        )
    )

    def fake_run(command, **_kwargs):
        run_dir = output_dir / "run"
        run_dir.mkdir(parents=True)
        vocals_path = run_dir / "vocals.flac"
        instrumental_path = run_dir / "instrumental.flac"
        vocals_path.write_bytes(b"vocals")
        instrumental_path.write_bytes(b"instrumental")
        (run_dir / "stem_manifest.json").write_text(
            """
{
  "target_stem": "instrumental",
  "target_product_stem": "minus",
  "product_aliases": {
    "minus": {
      "source_stem": "instrumental",
      "source_audio_path": "INSTRUMENTAL_PATH",
      "alias_of": "instrumental"
    }
  },
  "stems": {
    "vocals": {"path": "VOCALS_PATH", "role": "companion"},
    "instrumental": {"path": "INSTRUMENTAL_PATH", "role": "target"}
  }
}
""".replace("VOCALS_PATH", str(vocals_path).replace("\\", "\\\\")).replace(
                "INSTRUMENTAL_PATH", str(instrumental_path).replace("\\", "\\\\")
            ),
            encoding="utf-8",
        )
        return module.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    status_code, response = runner.separate(
        {
            "route_id": "becruily_instrumental_vocals",
            "source_audio_path": str(source_audio),
            "output_dir": str(output_dir),
            "device": "cuda",
            "require_gpu": True,
            "stems": ["minus", "vocals"],
            "timeout_sec": 30,
        }
    )

    assert status_code == 200
    assert response["status"] == "ok"
    assert response["missing_stems"] == []
    assert not (output_dir / "run" / "minus.flac").exists()
    assert (output_dir / "stems" / "minus.flac").read_bytes() == b"instrumental"
    assert (output_dir / "stems" / "vocals.flac").read_bytes() == b"vocals"
    assert response["copied_stems"]["minus"].endswith("stems/minus.flac")


def test_mac_api_separate_rejects_poisoned_minus_alias_to_vocals(tmp_path, monkeypatch):
    module = _load_module()
    project_root = tmp_path / "AIMusicProject_win"
    source_audio = project_root / "testing_stuff_music" / "mix.flac"
    source_audio.parent.mkdir(parents=True)
    source_audio.write_bytes(b"mix")
    artifacts_root = project_root / "artifacts"
    output_dir = artifacts_root / "rank" / "r001" / "c001"
    runner = module.MacMpsJobRunner(
        module.MacMpsApiConfig(
            project_root=project_root,
            artifacts_root=artifacts_root,
            python_bin=tmp_path / "venv" / "bin" / "python",
            mac_windows_project_mount=project_root,
        )
    )

    def fake_run(command, **_kwargs):
        run_dir = output_dir / "run"
        run_dir.mkdir(parents=True)
        vocals_path = run_dir / "vocals.flac"
        instrumental_path = run_dir / "instrumental.flac"
        vocals_path.write_bytes(b"vocals")
        instrumental_path.write_bytes(b"instrumental")
        (run_dir / "stem_manifest.json").write_text(
            """
{
  "stems": {
    "vocals": {"path": "VOCALS_PATH", "role": "companion"},
    "instrumental": {"path": "INSTRUMENTAL_PATH", "role": "target"},
    "minus": {
      "path": "VOCALS_PATH",
      "role": "product_alias",
      "alias_of": "vocals",
      "model_output_audio_path": "VOCALS_PATH"
    }
  }
}
""".replace("VOCALS_PATH", str(vocals_path).replace("\\", "\\\\")).replace(
                "INSTRUMENTAL_PATH", str(instrumental_path).replace("\\", "\\\\")
            ),
            encoding="utf-8",
        )
        return module.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    status_code, response = runner.separate(
        {
            "route_id": "becruily_instrumental_vocals",
            "source_audio_path": str(source_audio),
            "output_dir": str(output_dir),
            "device": "cuda",
            "require_gpu": True,
            "stems": ["minus"],
            "timeout_sec": 30,
        }
    )

    assert status_code == 500
    assert response["status"] == "blocked"
    assert response["reason"] == "missing_required_stems"
    assert response["missing_stems"] == ["minus"]
    assert not (output_dir / "stems" / "minus.flac").exists()


def test_mac_api_separate_blocks_unsupported_route_without_delivery(tmp_path):
    module = _load_module()
    project_root = tmp_path / "AIMusicProject_win"
    source_audio = project_root / "testing_stuff_music" / "mix.flac"
    source_audio.parent.mkdir(parents=True)
    source_audio.write_bytes(b"mix")
    output_dir = project_root / "artifacts" / "rank" / "r001" / "bad"
    runner = module.MacMpsJobRunner(
        module.MacMpsApiConfig(
            project_root=project_root,
            artifacts_root=project_root / "artifacts",
            python_bin=tmp_path / "venv" / "bin" / "python",
            mac_windows_project_mount=project_root,
        )
    )

    status_code, response = runner.separate(
        {
            "route_id": "raw_track_title_bad_route",
            "source_audio_path": str(source_audio),
            "output_dir": str(output_dir),
            "device": "cuda",
            "require_gpu": True,
            "stems": ["instrumental"],
        }
    )

    assert status_code == 400
    assert response["status"] == "blocked"
    assert "unsupported Mac MPS route" in response["reason"]
    assert response["delivery_allowed"] is False


def test_api_health_reports_resident_worker_state(tmp_path, monkeypatch):
    module = _load_module()
    runner = module.MacMpsJobRunner(
        module.MacMpsApiConfig(
            project_root=tmp_path / "project",
            artifacts_root=tmp_path / "artifacts",
            python_bin=tmp_path / "venv" / "bin" / "python",
            large_inst_worker_socket=tmp_path / "li.sock",
        )
    )
    monkeypatch.setattr(
        module,
        "send_request",
        lambda *_args, **_kwargs: {
            "status": "ready",
            "model_load_count": 1,
            "model_fingerprint": {"model_id": "bs_roformer_large_inst"},
        },
    )

    health = runner.health()

    assert health["status"] == "ok"
    assert health["large_inst_worker_configured"] is True
    assert health["large_inst_worker"]["model_load_count"] == 1


def test_async_job_manifest_preserves_resident_worker_evidence(tmp_path):
    module = _load_module()
    output_dir = tmp_path / "artifacts" / "mac_api" / "t001"
    run_dir = output_dir / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "stem_manifest.json").write_text(
        json.dumps(
            {
                "target_stem": "minus",
                "stems": {},
                "execution_contract": {
                    "resident_worker_evidence": {
                        "model_load_count": 1,
                        "request_count": 7,
                        "scratch_cleanup_verified": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    runner = module.MacMpsJobRunner(
        module.MacMpsApiConfig(
            project_root=tmp_path / "project",
            artifacts_root=tmp_path / "artifacts",
            python_bin=tmp_path / "venv" / "bin" / "python",
            large_inst_worker_socket=tmp_path / "li.sock",
        )
    )

    runner._finish(
        "m123",
        tmp_path / "source.flac",
        output_dir,
        "bs_roformer_large_inst",
        "bs_roformer_large_inst",
        status="ok",
        returncode=0,
        error=None,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["resident_capable"] is True
    assert manifest["resident_backend_used"] is True
    assert manifest["resident_worker_evidence"]["request_count"] == 7


def test_instrumental_alias_rejects_untrusted_minus_manifest_source(tmp_path):
    module = _load_module()
    minus_path = tmp_path / "minus.flac"
    minus_path.write_bytes(b"untrusted")
    summary = {
        "target_stem": "minus",
        "stems": {
            "minus": {
                "path": str(minus_path),
                "source": "stem_model_non_vocal_sum",
                "source_stems": ["drums", "bass", "other"],
            }
        },
    }

    assert module._source_info_for_requested_stem(summary, "instrumental") is None


def test_separate_mount_block_does_not_write_rejected_output_path(tmp_path):
    module = _load_module()
    project_root = tmp_path / "AIMusicProject_win"
    source_audio = project_root / "testing_stuff_music" / "mix.flac"
    source_audio.parent.mkdir(parents=True)
    source_audio.write_bytes(b"mix")
    output_dir = tmp_path / "forbidden_local_output"
    runner = module.MacMpsJobRunner(
        module.MacMpsApiConfig(
            project_root=project_root,
            artifacts_root=project_root / "artifacts",
            python_bin=tmp_path / "venv" / "bin" / "python",
            mac_windows_project_mount=project_root,
        )
    )

    status_code, response = runner.separate(
        {
            "route_id": "bs_roformer_large_inst",
            "source_audio_path": str(source_audio),
            "output_dir": str(output_dir),
            "device": "mps",
            "require_gpu": True,
            "stems": ["instrumental"],
        }
    )

    assert status_code == 400
    assert response["status"] == "blocked"
    assert not output_dir.exists()


def test_separate_rejects_existing_output_without_running_or_reusing_stale_stems(tmp_path, monkeypatch):
    module = _load_module()
    project_root = tmp_path / "AIMusicProject_win"
    source_audio = project_root / "testing_stuff_music" / "mix.flac"
    source_audio.parent.mkdir(parents=True)
    source_audio.write_bytes(b"mix")
    output_dir = project_root / "artifacts" / "mac_api" / "stale"
    stale_run = output_dir / "run"
    stale_run.mkdir(parents=True)
    stale_manifest = stale_run / "stem_manifest.json"
    stale_manifest.write_text('{"target_stem":"minus"}', encoding="utf-8")
    runner = module.MacMpsJobRunner(
        module.MacMpsApiConfig(
            project_root=project_root,
            artifacts_root=project_root / "artifacts",
            python_bin=tmp_path / "venv" / "bin" / "python",
            mac_windows_project_mount=project_root,
        )
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("subprocess must not run")),
    )

    status_code, response = runner.separate(
        {
            "route_id": "bs_roformer_large_inst",
            "source_audio_path": str(source_audio),
            "output_dir": str(output_dir),
            "device": "mps",
            "require_gpu": True,
            "stems": ["instrumental"],
        }
    )

    assert status_code == 409
    assert response["reason"] == "output_dir_already_exists"
    assert stale_manifest.read_text(encoding="utf-8") == '{"target_stem":"minus"}'
    assert not (output_dir / "persistent_separator.json").exists()


def test_concurrent_separate_requests_cannot_share_output_dir(tmp_path, monkeypatch):
    module = _load_module()
    project_root = tmp_path / "AIMusicProject_win"
    source_audio = project_root / "testing_stuff_music" / "mix.flac"
    source_audio.parent.mkdir(parents=True)
    source_audio.write_bytes(b"mix")
    output_dir = project_root / "artifacts" / "mac_api" / "race"
    runner = module.MacMpsJobRunner(
        module.MacMpsApiConfig(
            project_root=project_root,
            artifacts_root=project_root / "artifacts",
            python_bin=tmp_path / "venv" / "bin" / "python",
            mac_windows_project_mount=project_root,
        )
    )
    inference_started = threading.Event()
    allow_inference_finish = threading.Event()
    call_count = 0

    def fake_run(command, **_kwargs):
        nonlocal call_count
        call_count += 1
        inference_started.set()
        assert allow_inference_finish.wait(timeout=2.0)
        return module.subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    payload = {
        "route_id": "bs_roformer_large_inst",
        "source_audio_path": str(source_audio),
        "output_dir": str(output_dir),
        "device": "mps",
        "require_gpu": True,
        "stems": ["instrumental"],
        "timeout_sec": 30,
    }
    first_result = []
    first_thread = threading.Thread(target=lambda: first_result.append(runner.separate(payload)))
    first_thread.start()
    assert inference_started.wait(timeout=2.0)

    second_status, second_response = runner.separate(payload)
    allow_inference_finish.set()
    first_thread.join(timeout=2.0)

    assert second_status == 409
    assert second_response["reason"] == "output_dir_already_exists"
    assert call_count == 1
    assert len(first_result) == 1


def test_separate_timeout_is_persisted_as_fail_closed_response(tmp_path, monkeypatch):
    module = _load_module()
    project_root = tmp_path / "AIMusicProject_win"
    source_audio = project_root / "testing_stuff_music" / "mix.flac"
    source_audio.parent.mkdir(parents=True)
    source_audio.write_bytes(b"mix")
    output_dir = project_root / "artifacts" / "mac_api" / "timeout"
    runner = module.MacMpsJobRunner(
        module.MacMpsApiConfig(
            project_root=project_root,
            artifacts_root=project_root / "artifacts",
            python_bin=tmp_path / "venv" / "bin" / "python",
            mac_windows_project_mount=project_root,
        )
    )

    def fake_run(command, **_kwargs):
        raise module.subprocess.TimeoutExpired(command, 30)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    status_code, response = runner.separate(
        {
            "route_id": "bs_roformer_large_inst",
            "source_audio_path": str(source_audio),
            "output_dir": str(output_dir),
            "device": "mps",
            "require_gpu": True,
            "stems": ["instrumental"],
            "timeout_sec": 30,
        }
    )

    persisted = json.loads((output_dir / "persistent_separator.json").read_text(encoding="utf-8"))
    assert status_code == 504
    assert response["status"] == "blocked"
    assert response["reason"] == "export_timeout:30s"
    assert response["returncode"] == 124
    assert persisted["reason"] == "export_timeout:30s"
    assert response["delivery_allowed"] is False


def test_async_job_timeout_finishes_manifest_instead_of_running_forever(tmp_path, monkeypatch):
    module = _load_module()
    project_root = tmp_path / "AIMusicProject_win"
    source_audio = project_root / "testing_stuff_music" / "mix.flac"
    source_audio.parent.mkdir(parents=True)
    source_audio.write_bytes(b"mix")
    output_dir = project_root / "artifacts" / "mac_api" / "async_timeout"
    output_dir.mkdir(parents=True)
    runner = module.MacMpsJobRunner(
        module.MacMpsApiConfig(
            project_root=project_root,
            artifacts_root=project_root / "artifacts",
            python_bin=tmp_path / "venv" / "bin" / "python",
            mac_windows_project_mount=project_root,
        )
    )

    def fake_run(command, **kwargs):
        assert kwargs["timeout"] == 45
        timeout_index = command.index("--timeout-sec")
        assert command[timeout_index + 1] == "45"
        raise module.subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    runner._run_job(
        "m123",
        source_audio,
        output_dir,
        "mdx23c_vocals",
        "mdx23c_vocals",
        {"preflight_status": "passed"},
        45,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["returncode"] == 124
    assert manifest["error"] == "export_timeout:45s"
    assert manifest["delivery_allowed"] is False


def test_http_health_survives_revoked_stderr_and_records_access_log(tmp_path, monkeypatch):
    module = _load_module()
    runner = module.MacMpsJobRunner(
        module.MacMpsApiConfig(
            project_root=tmp_path / "project",
            artifacts_root=tmp_path / "artifacts",
            python_bin=tmp_path / "venv" / "bin" / "python",
        )
    )

    class BoundHandler(module.MacMpsApiHandler):
        pass

    class RevokedStream:
        def write(self, _value):
            raise OSError("revoked")

        def flush(self):
            raise OSError("revoked")

    BoundHandler.runner = runner
    server = module.ThreadingHTTPServer(("127.0.0.1", 0), BoundHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    monkeypatch.setattr(module.sys, "stderr", RevokedStream())
    server_thread.start()

    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        connection.request("GET", "/health")
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)

    assert response.status == 200
    assert payload["status"] == "ok"
    access_log = tmp_path / "artifacts" / "mac_api" / "server.log"
    assert '"GET /health HTTP/1.1" 200' in access_log.read_text(encoding="utf-8")


def test_access_log_write_failure_is_exposed_by_health(tmp_path, monkeypatch):
    module = _load_module()
    runner = module.MacMpsJobRunner(
        module.MacMpsApiConfig(
            project_root=tmp_path / "project",
            artifacts_root=tmp_path / "artifacts",
            python_bin=tmp_path / "venv" / "bin" / "python",
        )
    )

    def fail_open(*_args, **_kwargs):
        raise OSError("access log unavailable")

    monkeypatch.setattr(module.Path, "open", fail_open)
    runner.record_access_log("request")
    health = runner.health()

    assert health["status"] == "degraded"
    assert health["access_log"]["status"] == "degraded"
    assert health["access_log"]["write_failure_count"] == 1
    assert health["access_log"]["last_error"] == "OSError: access log unavailable"
