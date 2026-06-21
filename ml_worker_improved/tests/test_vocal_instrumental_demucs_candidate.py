from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.run_vocal_instrumental_demucs_candidate import run_vocal_instrumental_demucs_candidate
from tools.stem_contamination_common import NON_VOCAL_STEMS, STEMS


SAMPLE_RATE = 16_000


def sine(freq: float, seconds: float = 0.25, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def write_audio(path: Path, audio: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return path


def read_audio(path: Path) -> np.ndarray:
    audio, _ = sf.read(path, dtype="float32", always_2d=False)
    return np.asarray(audio, dtype=np.float32)


def test_vocal_instrumental_demucs_route_splits_instrumental_not_mix(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, amp=0.04),
        "drums": sine(110.0, amp=0.12),
        "bass": sine(220.0, amp=0.08),
        "guitar": sine(330.0, amp=0.06),
        "piano": sine(440.0, amp=0.05),
        "other": sine(550.0, amp=0.03),
    }
    mix_path = write_audio(tmp_path / "mix.flac", sum(source_audio.values()))
    expected_instrumental = sum(source_audio[stem] for stem in NON_VOCAL_STEMS)
    seen_demucs_inputs: list[str] = []

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        if any(part.endswith("export_audio_separator_vocal_full.py") for part in command):
            route_stems = Path(command[3])
            write_audio(route_stems / "vocals.flac", source_audio["vocals"])
            write_audio(route_stems / "instrumental.flac", expected_instrumental)
            return
        if any(part.endswith("run_demucs_six_stem_candidate.py") for part in command):
            input_audio = command[command.index("--input-audio") + 1]
            route_stems = Path(command[command.index("--stems-dir") + 1])
            seen_demucs_inputs.append(input_audio)
            for stem in STEMS:
                audio = source_audio[stem] if stem in NON_VOCAL_STEMS else np.zeros_like(source_audio["vocals"])
                write_audio(route_stems / f"{stem}.flac", audio)
            return
        raise AssertionError(command)

    manifest = run_vocal_instrumental_demucs_candidate(
        input_audio=mix_path,
        stems_dir=tmp_path / "stems",
        work_dir=tmp_path / "work",
        manifest_path=tmp_path / "manifest.json",
        command_runner=fake_runner,
        timeout_sec=120,
    )

    assert manifest["generation_status"] == "completed"
    assert seen_demucs_inputs == [manifest["instrumental_path"]]
    assert Path(manifest["instrumental_path"]).name == "instrumental.flac"
    np.testing.assert_allclose(read_audio(tmp_path / "stems" / "vocals.flac"), source_audio["vocals"], atol=2e-4)
    for stem in NON_VOCAL_STEMS:
        np.testing.assert_allclose(read_audio(tmp_path / "stems" / f"{stem}.flac"), source_audio[stem], atol=2e-4)
    saved_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["schema"] == "vocal_instrumental_demucs_candidate.v1"


def test_vocal_instrumental_demucs_route_passes_vocal_model_identity(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, amp=0.04),
        "drums": sine(110.0, amp=0.12),
        "bass": sine(220.0, amp=0.08),
        "guitar": sine(330.0, amp=0.06),
        "piano": sine(440.0, amp=0.05),
        "other": sine(550.0, amp=0.03),
    }
    mix_path = write_audio(tmp_path / "mix.flac", sum(source_audio.values()))
    expected_instrumental = sum(source_audio[stem] for stem in NON_VOCAL_STEMS)
    vocal_commands: list[list[str]] = []

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        if any(part.endswith("export_audio_separator_vocal_full.py") for part in command):
            vocal_commands.append(command)
            route_stems = Path(command[3])
            write_audio(route_stems / "vocals.flac", source_audio["vocals"])
            write_audio(route_stems / "instrumental.flac", expected_instrumental)
            return
        if any(part.endswith("run_demucs_six_stem_candidate.py") for part in command):
            route_stems = Path(command[command.index("--stems-dir") + 1])
            for stem in STEMS:
                audio = source_audio[stem] if stem in NON_VOCAL_STEMS else np.zeros_like(source_audio["vocals"])
                write_audio(route_stems / f"{stem}.flac", audio)
            return
        raise AssertionError(command)

    manifest = run_vocal_instrumental_demucs_candidate(
        input_audio=mix_path,
        stems_dir=tmp_path / "stems",
        work_dir=tmp_path / "work",
        manifest_path=tmp_path / "manifest.json",
        command_runner=fake_runner,
        timeout_sec=120,
        vocal_model_filename="MelBandRoformer.ckpt",
        vocal_model_file_dir="/app/vendor/mel_band_roformer_vocal_model_weights",
        vocal_model_id="audio_separator_melband_roformer_vocals",
        vocal_model_display_name="Audio Separator MelBand RoFormer vocal checkpoint",
    )

    vocal_command = vocal_commands[0]
    assert vocal_command[vocal_command.index("--model-filename") + 1] == "MelBandRoformer.ckpt"
    assert (
        vocal_command[vocal_command.index("--model-file-dir") + 1]
        == "/app/vendor/mel_band_roformer_vocal_model_weights"
    )
    assert vocal_command[vocal_command.index("--manifest-model-id") + 1] == "audio_separator_melband_roformer_vocals"
    assert "MelBand RoFormer" in vocal_command[vocal_command.index("--manifest-model-display-name") + 1]
    assert manifest["vocal_model_id"] == "audio_separator_melband_roformer_vocals"
