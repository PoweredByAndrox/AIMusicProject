from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.run_registered_six_stem_candidate import run_registered_six_stem_candidate
from tools.stem_contamination_common import STEMS


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


def write_registered_manifest(output_dir: Path, stems: dict[str, Path]) -> Path:
    stem_manifest = output_dir / "stem_manifest.json"
    stem_manifest.write_text(
        json.dumps(
            {
                "schema": "test_registered_candidate.v1",
                "stems": {
                    stem: {"path": str(path)}
                    for stem, path in stems.items()
                },
                "target_stem_artifact": {
                    "target_stem": "minus",
                    "path": str(output_dir / "minus.flac"),
                    "source": "sum_non_vocal_stems",
                },
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
                    "target_stem": "minus",
                    "target_audio_path": str(output_dir / "minus.flac"),
                    "target_source": "sum_non_vocal_stems",
                },
            }
        ),
        encoding="utf-8",
    )
    return stem_manifest


def test_registered_six_stem_candidate_copies_required_stems(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, amp=0.04),
        "drums": sine(110.0, amp=0.12),
        "bass": sine(220.0, amp=0.08),
        "guitar": sine(330.0, amp=0.06),
        "piano": sine(440.0, amp=0.05),
        "other": sine(550.0, amp=0.03),
    }
    mix_path = write_audio(tmp_path / "mix.flac", sum(source_audio.values()))
    commands: list[list[str]] = []

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        commands.append(command)
        registered_dir = Path(command[3])
        registered_dir.mkdir(parents=True, exist_ok=True)
        stem_paths = {
            stem: write_audio(registered_dir / f"{stem}.flac", audio)
            for stem, audio in source_audio.items()
        }
        write_audio(registered_dir / "minus.flac", sum(source_audio[stem] for stem in STEMS if stem != "vocals"))
        write_registered_manifest(registered_dir, stem_paths)

    manifest = run_registered_six_stem_candidate(
        input_audio=mix_path,
        stems_dir=tmp_path / "stems",
        work_dir=tmp_path / "work",
        manifest_path=tmp_path / "manifest.json",
        command_runner=fake_runner,
        timeout_sec=120,
        model_id="hybrid_htdemucs_ft_bs_roformer_6s",
    )

    assert manifest["generation_status"] == "completed"
    assert manifest["model_id"] == "hybrid_htdemucs_ft_bs_roformer_6s"
    assert commands[0][commands[0].index("--target-stem") + 1] == "minus"
    assert manifest["registered_target_source"] == "sum_non_vocal_stems"
    assert manifest["required_stems"] == list(STEMS)
    assert "minus" not in manifest["stem_paths"]
    for stem in STEMS:
        np.testing.assert_allclose(read_audio(tmp_path / "stems" / f"{stem}.flac"), source_audio[stem], atol=2e-4)


def test_registered_six_stem_candidate_accepts_hyb001_shaped_extra_stems(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, amp=0.04),
        "drums": sine(110.0, amp=0.12),
        "bass": sine(220.0, amp=0.08),
        "guitar": sine(330.0, amp=0.06),
        "piano": sine(440.0, amp=0.05),
        "other": sine(550.0, amp=0.03),
    }
    mix_path = write_audio(tmp_path / "mix.flac", sum(source_audio.values()))

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        registered_dir = Path(command[3])
        registered_dir.mkdir(parents=True, exist_ok=True)
        stem_paths = {
            stem: write_audio(registered_dir / f"{stem}.flac", audio)
            for stem, audio in source_audio.items()
        }
        minus_path = write_audio(registered_dir / "minus.flac", sum(source_audio[stem] for stem in STEMS if stem != "vocals"))
        stem_manifest = registered_dir / "stem_manifest.json"
        stem_manifest.write_text(
            json.dumps(
                {
                    "stems": {
                        **{stem: {"path": str(path)} for stem, path in stem_paths.items()},
                        "minus": {
                            "path": str(minus_path),
                            "source": "sum_non_vocal_stems",
                            "source_stems": ["drums", "bass", "other", "guitar", "piano"],
                        },
                        "instrumental": {
                            "path": str(minus_path),
                            "source": "sum_non_vocal_stems",
                            "source_stems": ["drums", "bass", "other", "guitar", "piano"],
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        (registered_dir / "registry_candidate_execution_manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "candidate_manifest": {
                        "path": str(stem_manifest),
                        "exists": True,
                        "target_stem": "minus",
                        "target_audio_path": str(minus_path),
                        "target_source": "sum_non_vocal_stems",
                    },
                }
            ),
            encoding="utf-8",
        )

    manifest = run_registered_six_stem_candidate(
        input_audio=mix_path,
        stems_dir=tmp_path / "stems",
        work_dir=tmp_path / "work",
        manifest_path=tmp_path / "manifest.json",
        command_runner=fake_runner,
        timeout_sec=120,
    )

    assert manifest["generation_status"] == "completed"
    assert manifest["registered_target_source"] == "sum_non_vocal_stems"
    assert set(manifest["stem_paths"]) == set(STEMS)


def test_registered_six_stem_candidate_merges_declared_melody_into_guitar(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, amp=0.04),
        "drums": sine(110.0, amp=0.12),
        "bass": sine(220.0, amp=0.08),
        "guitar": sine(330.0, amp=0.06),
        "piano": sine(440.0, amp=0.05),
        "other": sine(550.0, amp=0.03),
    }
    melody = sine(660.0, amp=0.02)
    mix_path = write_audio(tmp_path / "mix.flac", sum(source_audio.values()) + melody)

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        registered_dir = Path(command[3])
        registered_dir.mkdir(parents=True, exist_ok=True)
        stem_paths = {
            stem: write_audio(registered_dir / f"{stem}.flac", audio)
            for stem, audio in source_audio.items()
        }
        melody_path = write_audio(registered_dir / "melody.flac", melody)
        minus_path = write_audio(
            registered_dir / "minus.flac",
            sum(source_audio[stem] for stem in STEMS if stem != "vocals") + melody,
        )
        stem_manifest = registered_dir / "stem_manifest.json"
        stem_manifest.write_text(
            json.dumps(
                {
                    "stems": {
                        **{stem: {"path": str(path)} for stem, path in stem_paths.items()},
                        "melody": {"path": str(melody_path)},
                        "minus": {
                            "path": str(minus_path),
                            "source": "sum_non_vocal_stems",
                            "source_stems": ["drums", "bass", "other", "guitar", "piano", "melody"],
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        (registered_dir / "registry_candidate_execution_manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "candidate_manifest": {
                        "path": str(stem_manifest),
                        "exists": True,
                        "target_stem": "minus",
                        "target_audio_path": str(minus_path),
                        "target_source": "sum_non_vocal_stems",
                    },
                }
            ),
            encoding="utf-8",
        )

    manifest = run_registered_six_stem_candidate(
        input_audio=mix_path,
        stems_dir=tmp_path / "stems",
        work_dir=tmp_path / "work",
        manifest_path=tmp_path / "manifest.json",
        command_runner=fake_runner,
        timeout_sec=120,
        model_id="hybrid_htdemucs_ft_bs_roformer_6s_aggressive_melody_first",
    )

    assert manifest["generation_status"] == "completed"
    assert manifest["merged_extra_stems"] == [
        {
            "source_stem": "melody",
            "target_stem": "guitar",
            "source_path": str(tmp_path / "work" / "registered" / "melody.flac"),
            "target_path": str(tmp_path / "stems" / "guitar.flac"),
        }
    ]
    np.testing.assert_allclose(
        read_audio(tmp_path / "stems" / "guitar.flac"),
        source_audio["guitar"] + melody,
        atol=2e-4,
    )


def test_registered_six_stem_candidate_blocks_unmapped_declared_extra_stem(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, amp=0.04),
        "drums": sine(110.0, amp=0.12),
        "bass": sine(220.0, amp=0.08),
        "guitar": sine(330.0, amp=0.06),
        "piano": sine(440.0, amp=0.05),
        "other": sine(550.0, amp=0.03),
    }
    lead = sine(660.0, amp=0.02)
    mix_path = write_audio(tmp_path / "mix.flac", sum(source_audio.values()) + lead)

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        registered_dir = Path(command[3])
        registered_dir.mkdir(parents=True, exist_ok=True)
        stem_paths = {
            stem: write_audio(registered_dir / f"{stem}.flac", audio)
            for stem, audio in source_audio.items()
        }
        lead_path = write_audio(registered_dir / "lead.flac", lead)
        minus_path = write_audio(
            registered_dir / "minus.flac",
            sum(source_audio[stem] for stem in STEMS if stem != "vocals") + lead,
        )
        stem_manifest = registered_dir / "stem_manifest.json"
        stem_manifest.write_text(
            json.dumps(
                {
                    "stems": {
                        **{stem: {"path": str(path)} for stem, path in stem_paths.items()},
                        "lead": {"path": str(lead_path)},
                        "minus": {
                            "path": str(minus_path),
                            "source": "sum_non_vocal_stems",
                            "source_stems": ["drums", "bass", "other", "guitar", "piano", "lead"],
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        (registered_dir / "registry_candidate_execution_manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "candidate_manifest": {
                        "path": str(stem_manifest),
                        "exists": True,
                        "target_stem": "minus",
                        "target_audio_path": str(minus_path),
                        "target_source": "sum_non_vocal_stems",
                    },
                }
            ),
            encoding="utf-8",
        )

    manifest = run_registered_six_stem_candidate(
        input_audio=mix_path,
        stems_dir=tmp_path / "stems",
        work_dir=tmp_path / "work",
        manifest_path=tmp_path / "manifest.json",
        command_runner=fake_runner,
        timeout_sec=120,
    )

    assert manifest["generation_status"] == "failed_unmapped_extra_stem"
    assert manifest["unmapped_declared_extra_stems"] == ["lead"]
    assert manifest["merged_extra_stems"] == []


def test_registered_six_stem_candidate_blocks_missing_required_stem(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, amp=0.04),
        "drums": sine(110.0, amp=0.12),
    }
    mix_path = write_audio(tmp_path / "mix.flac", sum(source_audio.values()))

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        registered_dir = Path(command[3])
        registered_dir.mkdir(parents=True, exist_ok=True)
        stem_paths = {
            stem: write_audio(registered_dir / f"{stem}.flac", audio)
            for stem, audio in source_audio.items()
        }
        write_registered_manifest(registered_dir, stem_paths)

    manifest = run_registered_six_stem_candidate(
        input_audio=mix_path,
        stems_dir=tmp_path / "stems",
        work_dir=tmp_path / "work",
        manifest_path=tmp_path / "manifest.json",
        command_runner=fake_runner,
        timeout_sec=120,
    )

    assert manifest["generation_status"] == "failed_missing_stem"
    assert manifest["missing_stems"] == ["bass", "guitar", "piano", "other"]
    assert not (tmp_path / "stems" / "minus.flac").exists()
