from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.no_reference_candidate_job_registry import (
    CandidateJobConfig,
    build_candidate_job_registry,
)
from tools.run_local_separator_candidate_job import run_local_separator_candidate_job


SAMPLE_RATE = 16_000


def sine(freq: float, seconds: float, amp: float = 0.25) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return str(path)


def write_wav_at(path: Path, *, sample_rate: int, seconds: float, channels: int = 1) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / float(sample_rate)
    audio = (0.25 * np.sin(2.0 * math.pi * 330.0 * t)).astype(np.float32)
    if channels > 1:
        audio = np.stack([audio for _ in range(channels)], axis=1)
    sf.write(path, audio, sample_rate)
    return str(path)


def unit_config(
    *,
    config_id: str = "unit-vocal",
    target_mode: str = "vocal",
    entrypoint_type: str = "python_script",
) -> CandidateJobConfig:
    return CandidateJobConfig(
        candidate_config_id=config_id,
        candidate_family=f"{target_mode}_focused" if target_mode != "both" else "balanced_pair",
        target_mode=target_mode,
        source_model_id="unit-existing-model",
        source_tool="unit-existing-separator",
        entrypoint_type=entrypoint_type,
        command_template=("unit-runner", "{input_audio}", "{output_dir}"),
        expected_outputs={
            "vocal_path": "{output_dir}/vocals.wav" if target_mode in {"vocal", "both"} else "",
            "minus_path": "{output_dir}/minus.wav" if target_mode in {"minus", "both"} else "",
        },
        gpu_required=False,
        timeout_sec=5,
        max_retries=0,
        profile_allowed=("safe",),
        safety_notes=("unit test config",),
    )


def test_job_registry_references_supported_configs_or_marks_unavailable() -> None:
    configs = build_candidate_job_registry(profile="safe")

    assert configs
    assert len(configs) >= 2
    assert len({config.candidate_config_id for config in configs}) == len(configs)
    for config in configs:
        manifest = config.to_manifest()
        assert config.source_config_hash
        assert "supported_parameters" in manifest
        assert config.target_mode in {"vocal", "minus", "both"}
        assert config.candidate_family in {
            "balanced_pair_default",
            "vocal_focused_default",
            "minus_focused_default",
            "artifact_safe_default",
            "chunk_overlap_variant",
            "high_overlap_variant",
            "low_overlap_variant",
        }
        assert config.entrypoint_type in {
            "python_module",
            "python_script",
            "existing_cli",
            "unavailable",
        }
        if config.entrypoint_type != "unavailable":
            assert config.source_tool == "tools/run_registered_model_candidate.py"
        else:
            assert config.safety_notes


def test_unavailable_job_is_skipped_not_faked(tmp_path: Path) -> None:
    input_audio = Path(write_wav(tmp_path / "input.wav", sine(220, 1.0)))
    output_dir = tmp_path / "candidate"

    manifest = run_local_separator_candidate_job(
        input_audio=input_audio,
        candidate_config_id="unit-unavailable",
        output_dir=output_dir,
        job_manifest_json=output_dir / "job_manifest.json",
        job_progress_jsonl=output_dir / "job_progress.jsonl",
        registry_configs=[unit_config(config_id="unit-unavailable", entrypoint_type="unavailable")],
    )

    assert manifest["generation_status"] == "skipped_unavailable_entrypoint"
    assert manifest["audio_rendered"] is False
    assert manifest["release_ready"] is False
    assert (output_dir / "job_manifest.json").exists()
    assert (output_dir / "job_progress.jsonl").exists()


def test_local_separator_job_writes_manifest_and_progress(tmp_path: Path) -> None:
    input_audio = Path(write_wav(tmp_path / "input.wav", sine(220, 1.0)))
    output_dir = tmp_path / "candidate"

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        assert command[0] == "unit-runner"
        assert timeout_sec == 5
        write_wav(output_dir / "vocals.wav", sine(440, 1.0))

    manifest = run_local_separator_candidate_job(
        input_audio=input_audio,
        candidate_config_id="unit-vocal",
        output_dir=output_dir,
        job_manifest_json=output_dir / "job_manifest.json",
        job_progress_jsonl=output_dir / "job_progress.jsonl",
        registry_configs=[unit_config()],
        command_runner=fake_runner,
    )

    progress_rows = [
        json.loads(line)
        for line in (output_dir / "job_progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    written_manifest = json.loads((output_dir / "job_manifest.json").read_text(encoding="utf-8"))

    assert manifest["generation_status"] == "generated"
    assert manifest["output_paths"]["vocal_path"].endswith("vocals.wav")
    assert manifest["audio_rendered"] is True
    assert manifest["release_ready"] is False
    assert written_manifest == manifest
    assert [row["phase"] for row in progress_rows] == [
        "start",
        "validate_input_audio",
        "execute_separator_command",
        "execute_separator_command",
        "validate_outputs",
        "complete",
    ]


def test_local_separator_job_normalizes_recoverable_audio_shape(tmp_path: Path) -> None:
    input_audio = Path(write_wav_at(tmp_path / "input.wav", sample_rate=48_000, seconds=1.0, channels=2))
    output_dir = tmp_path / "candidate"

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        write_wav_at(output_dir / "vocals.wav", sample_rate=44_100, seconds=1.0, channels=1)

    manifest = run_local_separator_candidate_job(
        input_audio=input_audio,
        candidate_config_id="unit-vocal",
        output_dir=output_dir,
        job_manifest_json=output_dir / "job_manifest.json",
        job_progress_jsonl=output_dir / "job_progress.jsonl",
        registry_configs=[unit_config()],
        command_runner=fake_runner,
        allow_shape_normalization=True,
    )

    raw_vocal = output_dir / "vocals.wav"
    normalized_vocal = Path(manifest["normalized_vocal_path"])
    normalized_info = sf.info(str(normalized_vocal))

    assert manifest["generation_status"] == "generated_with_shape_normalization"
    assert manifest["shape_normalization_applied"] is True
    assert manifest["shape_normalization_reason"]
    assert manifest["input_sample_rate"] == 48_000
    assert manifest["input_channels"] == 2
    assert manifest["raw_output_sample_rate"] == 44_100
    assert manifest["raw_output_channels"] == 1
    assert manifest["normalized_output_sample_rate"] == 48_000
    assert manifest["normalized_output_channels"] == 2
    assert manifest["raw_vocal_path"] == str(raw_vocal)
    assert manifest["normalized_vocal_path"] == manifest["output_paths"]["vocal_path"]
    assert manifest["raw_vocal_path"] != manifest["normalized_vocal_path"]
    assert manifest["raw_minus_path"] == ""
    assert manifest["normalized_minus_path"] == ""
    assert manifest["original_files_overwritten"] is False
    assert manifest["release_ready"] is False
    assert raw_vocal.exists()
    assert normalized_vocal.exists()
    assert normalized_info.samplerate == 48_000
    assert normalized_info.channels == 2


def test_missing_expected_output_fails_closed(tmp_path: Path) -> None:
    input_audio = Path(write_wav(tmp_path / "input.wav", sine(220, 1.0)))
    output_dir = tmp_path / "candidate"

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        return None

    manifest = run_local_separator_candidate_job(
        input_audio=input_audio,
        candidate_config_id="unit-vocal",
        output_dir=output_dir,
        job_manifest_json=output_dir / "job_manifest.json",
        job_progress_jsonl=output_dir / "job_progress.jsonl",
        registry_configs=[unit_config()],
        command_runner=fake_runner,
    )

    assert manifest["generation_status"] == "failed_missing_output"
    assert manifest["audio_rendered"] is False
    assert manifest["release_ready"] is False


def test_minus_job_ignores_vocal_stem_from_registered_manifest(tmp_path: Path) -> None:
    input_audio = Path(write_wav(tmp_path / "input.wav", sine(220, 1.0)))
    output_dir = tmp_path / "candidate"
    config = CandidateJobConfig(
        candidate_config_id="unit-minus-registered",
        candidate_family="minus_focused",
        target_mode="minus",
        source_model_id="unit-existing-model",
        source_tool="unit-existing-separator",
        entrypoint_type="python_script",
        command_template=("unit-runner", "{input_audio}", "{registered_output_dir}"),
        expected_outputs={"vocal_path": "", "minus_path": ""},
        gpu_required=False,
        timeout_sec=5,
        max_retries=0,
        profile_allowed=("safe",),
        safety_notes=("unit registered manifest",),
        target_stem="minus",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        registered = output_dir / "registered"
        vocal_path = Path(write_wav(registered / "vocals.wav", sine(440, 1.0)))
        minus_path = Path(write_wav(registered / "minus.wav", sine(120, 1.0)))
        stem_manifest = {
            "stems": {
                "vocal": {"path": str(vocal_path)},
                "minus": {"path": str(minus_path)},
            }
        }
        (registered / "stem_manifest.json").write_text(json.dumps(stem_manifest, sort_keys=True) + "\n", encoding="utf-8")
        execution_manifest = {
            "candidate_manifest": {
                "path": str(registered / "stem_manifest.json"),
                "target_audio_path": str(minus_path),
            }
        }
        (registered / "registry_candidate_execution_manifest.json").write_text(
            json.dumps(execution_manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    manifest = run_local_separator_candidate_job(
        input_audio=input_audio,
        candidate_config_id="unit-minus-registered",
        output_dir=output_dir,
        job_manifest_json=output_dir / "job_manifest.json",
        job_progress_jsonl=output_dir / "job_progress.jsonl",
        registry_configs=[config],
        command_runner=fake_runner,
    )

    assert manifest["generation_status"] == "generated"
    assert manifest["output_paths"]["vocal_path"] == ""
    assert manifest["output_paths"]["minus_path"].endswith("minus.wav")
    assert manifest["release_ready"] is False


def test_timeout_fails_closed(tmp_path: Path) -> None:
    input_audio = Path(write_wav(tmp_path / "input.wav", sine(220, 1.0)))
    output_dir = tmp_path / "candidate"

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        raise subprocess.TimeoutExpired(cmd=command, timeout=timeout_sec)

    manifest = run_local_separator_candidate_job(
        input_audio=input_audio,
        candidate_config_id="unit-vocal",
        output_dir=output_dir,
        job_manifest_json=output_dir / "job_manifest.json",
        job_progress_jsonl=output_dir / "job_progress.jsonl",
        registry_configs=[unit_config()],
        command_runner=fake_runner,
    )

    assert manifest["generation_status"] == "failed_timeout_before_any_output"
    assert manifest["timeout_sec"] == 5
    assert manifest["elapsed_sec"] >= 0
    assert manifest["audio_rendered"] is False
    assert manifest["release_ready"] is False


def test_timeout_with_valid_partial_output_is_recorded_honestly(tmp_path: Path) -> None:
    input_audio = Path(write_wav(tmp_path / "input.wav", sine(220, 1.0)))
    output_dir = tmp_path / "candidate"

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        write_wav(output_dir / "vocals.wav", sine(440, 1.0))
        raise subprocess.TimeoutExpired(cmd=command, timeout=timeout_sec)

    manifest = run_local_separator_candidate_job(
        input_audio=input_audio,
        candidate_config_id="unit-vocal",
        output_dir=output_dir,
        job_manifest_json=output_dir / "job_manifest.json",
        job_progress_jsonl=output_dir / "job_progress.jsonl",
        registry_configs=[unit_config()],
        command_runner=fake_runner,
    )

    assert manifest["generation_status"] == "generated_with_late_timeout"
    assert manifest["timeout_sec"] == 5
    assert manifest["elapsed_sec"] >= 0
    assert manifest["audio_rendered"] is True
    assert manifest["output_paths"]["vocal_path"].endswith("vocals.wav")
    assert manifest["release_ready"] is False
