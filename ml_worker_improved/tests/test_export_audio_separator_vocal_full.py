from __future__ import annotations

from pathlib import Path

from tools.export_audio_separator_vocal_full import (
    _resolve_named_output,
    build_full_track_stem_manifest,
    build_runtime_contract_status,
)


def test_audio_separator_runtime_contract_blocks_cpu_only_when_cuda_is_required() -> None:
    contract = build_runtime_contract_status(
        audio_separator_python=Path("/opt/audio-separator-venv/bin/python"),
        audio_separator_python_exists=True,
        model_file=Path("/app/models/audio_separator/model.ckpt"),
        model_file_exists=True,
        model_sidecar=Path("/app/models/audio_separator/model.yaml"),
        model_sidecar_exists=True,
        onnxruntime_version="1.26.0",
        available_providers=("AzureExecutionProvider", "CPUExecutionProvider"),
        require_cuda_provider=True,
        probe_error=None,
    )

    assert contract["status"] == "blocked"
    assert contract["cuda_provider_available"] is False
    assert "cuda_execution_provider_missing" in contract["blockers"]


def test_audio_separator_runtime_contract_accepts_cuda_provider_when_required() -> None:
    contract = build_runtime_contract_status(
        audio_separator_python=Path("/opt/audio-separator-venv/bin/python"),
        audio_separator_python_exists=True,
        model_file=Path("/app/models/audio_separator/model.ckpt"),
        model_file_exists=True,
        model_sidecar=Path("/app/models/audio_separator/model.yaml"),
        model_sidecar_exists=True,
        onnxruntime_version="1.26.0",
        available_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        require_cuda_provider=True,
        probe_error=None,
    )

    assert contract["status"] == "ready"
    assert contract["cuda_provider_available"] is True
    assert contract["blockers"] == []


def test_audio_separator_manifest_records_runtime_contract(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "source.flac"
    vocals_path = tmp_path / "vocals.flac"
    instrumental_path = tmp_path / "instrumental.flac"
    raw_ref_path = tmp_path / "raw_ref.flac"
    runtime_contract = {
        "status": "ready",
        "cuda_provider_available": True,
        "available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "blockers": [],
    }

    monkeypatch.setattr(
        "tools.export_audio_separator_vocal_full._audio_file_summary",
        lambda path: {
            "sample_rate": 44100,
            "frames": 44100,
            "channels": 2,
            "duration_sec": 1.0,
            "sha256": f"stub-{path.name}",
        },
    )

    manifest = build_full_track_stem_manifest(
        source_audio_path=source_path,
        vocals_flac_path=vocals_path,
        instrumental_flac_path=instrumental_path,
        raw_ref_flac_path=raw_ref_path,
        model_filename="model.ckpt",
        model_file_dir=Path("/app/models/audio_separator"),
        model_id="audio_separator_bs_roformer_vocals",
        model_display_name="Audio Separator BS-RoFormer vocal checkpoint",
        output_format="FLAC",
        normalization_threshold=0.9,
        mdx_overlap=0.25,
        mdx_batch_size=1,
        enable_denoise=False,
        sample_rate=44100,
        frames=44100,
        elapsed_sec=1.25,
        runtime_contract=runtime_contract,
    )

    assert manifest["runtime"]["audio_separator"] == runtime_contract


def test_audio_separator_resolves_other_output_as_instrumental(tmp_path: Path) -> None:
    model_output_dir = tmp_path / "separator_output"
    model_output_dir.mkdir()
    other_output = model_output_dir / "raw_ref_(other)_model_mel_band_roformer_ep_3005_sdr_11.flac"
    other_output.write_bytes(b"stub")

    resolved = _resolve_named_output(
        ["raw_ref_(other)_model_mel_band_roformer_ep_3005_sdr_11.flac"],
        model_output_dir,
        "instrumental",
    )

    assert resolved == other_output
