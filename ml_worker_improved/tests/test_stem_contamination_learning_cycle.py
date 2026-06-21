from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tools.build_procedural_stem_clean_pairs import build_procedural_stem_clean_pairs
from tools.run_stem_contamination_learning_cycle import run_stem_contamination_learning_cycle
from tools.stem_contamination_common import FORBIDDEN_SOURCES, STEMS


SAMPLE_RATE = 16_000


def sine(freq: float, seconds: float = 0.25, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return path


def read_wav(path: Path) -> np.ndarray:
    audio, _ = sf.read(path, dtype="float32", always_2d=False)
    return np.asarray(audio, dtype=np.float32)


def required_check_count() -> int:
    return sum(len(sources) for sources in FORBIDDEN_SOURCES.values())


def test_procedural_clean_pairs_cover_full_forbidden_matrix(tmp_path: Path) -> None:
    manifest_path = build_procedural_stem_clean_pairs(
        output_dir=tmp_path / "pairs",
        sample_rate=SAMPLE_RATE,
        duration_sec=0.25,
        examples_per_pair=1,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    pairs = manifest["pairs"]
    pair_set = {(pair["base_stem"], pair["foreign_source"]) for pair in pairs}
    expected_pairs = {
        (tested_stem, foreign_source)
        for tested_stem, foreign_sources in FORBIDDEN_SOURCES.items()
        for foreign_source in foreign_sources
    }
    assert manifest["schema"] == "stem_contamination_procedural_clean_pairs.v1"
    assert manifest["source_kind"] == "procedural_synthetic_clean"
    assert manifest["generated_separator_output_used"] is False
    assert len(pairs) == required_check_count()
    assert pair_set == expected_pairs
    assert set(manifest["sources"]) == set(STEMS)
    assert all(Path(pair["base_path"]).exists() and Path(pair["foreign_path"]).exists() for pair in pairs)


def test_learning_cycle_with_procedural_full_matrix_accepts_clean_fixture(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    clean_pairs_manifest = build_procedural_stem_clean_pairs(
        output_dir=tmp_path / "pairs",
        sample_rate=SAMPLE_RATE,
        duration_sec=0.25,
        examples_per_pair=2,
    )
    clean_pairs = json.loads(clean_pairs_manifest.read_text(encoding="utf-8"))
    stems = {
        stem: read_wav(Path(clean_pairs["sources"][stem]["path"]))
        for stem in STEMS
    }
    mix = write_wav(tmp_path / "mix.wav", sum(stems.values()))
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [{"candidate_id": "c001", "command_template": ["fixture-write-clean", "{window_mix}"]}],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        for stem, audio in stems.items():
            sf.write(stems_dir / f"{stem}.flac", audio[: len(window_audio)].astype(np.float32), mix_rate)

    manifest = run_stem_contamination_learning_cycle(
        mix_path=mix,
        clean_pairs_manifest=clean_pairs_manifest,
        output_dir=tmp_path / "cycle",
        recipe_path=recipe,
        command_runner=fake_runner,
        contamination_db=-10.0,
        training_device="cpu",
        training_epochs=220,
        training_hidden_size=16,
        training_min_positive_rows=2,
        max_candidates=1,
        require_gpu=False,
        no_artifact_rag=True,
    )

    model_manifest = json.loads(Path(manifest["model_manifest_json"]).read_text(encoding="utf-8"))
    search_manifest = json.loads(Path(manifest["search_run_json"]).read_text(encoding="utf-8"))

    assert model_manifest["training_status"] == "trained"
    assert model_manifest["pair_count"] == required_check_count()
    assert search_manifest["verdict"] == "auto_accept"
    assert search_manifest["delivery_allowed"] is True
    assert search_manifest["listen_pack"]


def test_stem_contamination_learning_cycle_trains_model_and_blocks_uncovered_matrix(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    stems = {
        "vocals": sine(880.0, amp=0.04),
        "drums": sine(110.0, amp=0.12),
        "bass": sine(220.0, amp=0.08),
        "guitar": sine(330.0, amp=0.06),
        "piano": sine(440.0, amp=0.05),
        "other": sine(550.0, amp=0.03),
    }
    mix = write_wav(tmp_path / "mix.wav", sum(stems.values()))
    clean_manifest = tmp_path / "clean_pairs.json"
    clean_manifest.write_text(
        json.dumps(
            {
                "schema": "stem_contamination_clean_pairs.v1",
                "pairs": [
                    {
                        "base_stem": "minus",
                        "base_path": str(
                            write_wav(
                                tmp_path / "clean" / "minus.flac",
                                sum(stems[stem] for stem in STEMS if stem != "vocals"),
                            )
                        ),
                        "foreign_source": "vocals",
                        "foreign_path": str(write_wav(tmp_path / "clean" / "vocals.flac", stems["vocals"])),
                    },
                    {
                        "base_stem": "guitar",
                        "base_path": str(write_wav(tmp_path / "clean" / "guitar.flac", stems["guitar"])),
                        "foreign_source": "vocals",
                        "foreign_path": str(tmp_path / "clean" / "vocals.flac"),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [{"candidate_id": "c001", "command_template": ["fixture-write-stems", "{window_mix}"]}],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        for stem in STEMS:
            sf.write(stems_dir / f"{stem}.flac", stems[stem][: len(window_audio)].astype(np.float32), mix_rate)

    manifest = run_stem_contamination_learning_cycle(
        mix_path=mix,
        clean_pairs_manifest=clean_manifest,
        output_dir=tmp_path / "cycle",
        recipe_path=recipe,
        command_runner=fake_runner,
        contamination_db=-12.0,
        training_device="cpu",
        training_epochs=120,
        training_hidden_size=8,
        training_min_positive_rows=2,
        require_gpu=False,
        no_artifact_rag=True,
    )

    model_manifest = json.loads(Path(manifest["model_manifest_json"]).read_text(encoding="utf-8"))
    search_manifest = json.loads(Path(manifest["search_run_json"]).read_text(encoding="utf-8"))

    assert manifest["cycle_status"] == "blocked_no_clean_candidate"
    assert manifest["delivery_allowed"] is False
    assert model_manifest["training_status"] == "trained"
    assert model_manifest["pair_count"] == 2
    assert search_manifest["verdict"] == "blocked_no_clean_candidate"
    assert search_manifest["delivery_allowed"] is False
    assert search_manifest["listen_pack"] == {}
    assert not (tmp_path / "cycle" / "search" / "listen").exists()
    assert any(
        "model_pair_untrained" in reason
        for candidate in search_manifest["candidates"]
        for reason in candidate["contamination_score"]["blocker_reasons"]
    )
