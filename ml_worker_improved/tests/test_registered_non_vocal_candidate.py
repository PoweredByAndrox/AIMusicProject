from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.run_registered_non_vocal_candidate import run_registered_non_vocal_candidate


SAMPLE_RATE = 16_000
REQUIRED_NON_VOCAL_STEMS = ("drums", "bass", "other")


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


def write_registered_manifest(output_dir: Path, stems: dict[str, Path]) -> Path:
    stem_manifest = output_dir / "stem_manifest.json"
    stem_manifest.write_text(
        json.dumps(
            {
                "schema": "test_registered_non_vocal_candidate.v1",
                "stems": {stem: {"path": str(path)} for stem, path in stems.items()},
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "registry_candidate_execution_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "candidate_manifest": {
                    "path": str(stem_manifest),
                    "exists": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return stem_manifest


def test_registered_non_vocal_candidate_copies_four_stem_non_vocal_outputs(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, amp=0.04),
        "drums": sine(110.0, amp=0.12),
        "bass": sine(220.0, amp=0.08),
        "other": sine(550.0, amp=0.03),
    }
    mix_path = write_audio(tmp_path / "mix.flac", sum(source_audio.values()))
    commands: list[list[str]] = []

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        commands.append(command)
        registered_dir = Path(command[3])
        registered_dir.mkdir(parents=True, exist_ok=True)
        stem_paths = {stem: write_audio(registered_dir / f"{stem}.flac", audio) for stem, audio in source_audio.items()}
        write_registered_manifest(registered_dir, stem_paths)

    manifest = run_registered_non_vocal_candidate(
        input_audio=mix_path,
        stems_dir=tmp_path / "stems",
        work_dir=tmp_path / "work",
        manifest_path=tmp_path / "manifest.json",
        command_runner=fake_runner,
        timeout_sec=120,
        model_id="scnet_xl_ihf_musdb18",
    )

    assert manifest["generation_status"] == "completed"
    assert manifest["model_id"] == "scnet_xl_ihf_musdb18"
    assert commands[0][commands[0].index("--model-id") + 1] == "scnet_xl_ihf_musdb18"
    assert "--target-stem" not in commands[0]
    assert manifest["required_stems"] == list(REQUIRED_NON_VOCAL_STEMS)
    assert set(manifest["stem_paths"]) == set(REQUIRED_NON_VOCAL_STEMS)
    assert not (tmp_path / "stems" / "vocals.flac").exists()
    assert not (tmp_path / "stems" / "guitar.flac").exists()
    assert not (tmp_path / "stems" / "piano.flac").exists()
    assert not (tmp_path / "stems" / "minus.flac").exists()
    for stem in REQUIRED_NON_VOCAL_STEMS:
        np.testing.assert_allclose(read_audio(tmp_path / "stems" / f"{stem}.flac"), source_audio[stem], atol=2e-4)


def test_registered_non_vocal_candidate_blocks_missing_required_non_vocal_stem(tmp_path: Path) -> None:
    source_audio = {
        "drums": sine(110.0, amp=0.12),
        "other": sine(550.0, amp=0.03),
    }
    mix_path = write_audio(tmp_path / "mix.flac", sum(source_audio.values()))

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        registered_dir = Path(command[3])
        registered_dir.mkdir(parents=True, exist_ok=True)
        stem_paths = {stem: write_audio(registered_dir / f"{stem}.flac", audio) for stem, audio in source_audio.items()}
        write_registered_manifest(registered_dir, stem_paths)

    manifest = run_registered_non_vocal_candidate(
        input_audio=mix_path,
        stems_dir=tmp_path / "stems",
        work_dir=tmp_path / "work",
        manifest_path=tmp_path / "manifest.json",
        command_runner=fake_runner,
        timeout_sec=120,
    )

    assert manifest["generation_status"] == "failed_missing_stem"
    assert manifest["missing_stems"] == ["bass"]
    assert not (tmp_path / "stems" / "minus.flac").exists()
