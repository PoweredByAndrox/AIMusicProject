from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tools.run_fully_automatic_stem_search import (
    build_candidate_manifest,
    build_default_from_scratch_recipe,
    choose_next_action,
    run_fully_automatic_stem_search,
    select_accepted_candidate,
)
from tools.stem_contamination_common import (
    FORBIDDEN_SOURCES,
    STEMS,
    VERDICTS,
    missing_required_stems,
    validate_no_artifact_rag_output,
)


SAMPLE_RATE = 16_000


def sine(freq: float, seconds: float = 0.25, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return path


def test_all_six_stems_are_required() -> None:
    assert STEMS == ("vocals", "drums", "bass", "guitar", "piano", "other")
    assert missing_required_stems({"vocals": "v.flac", "drums": "d.flac"}) == [
        "bass",
        "guitar",
        "piano",
        "other",
    ]


def test_no_human_review_state_exists() -> None:
    assert "human_review" not in VERDICTS
    assert all("human" not in verdict for verdict in VERDICTS)
    assert "vocals" in FORBIDDEN_SOURCES["minus"]


def test_unknown_blocks_instead_of_review() -> None:
    action = choose_next_action({"verdict": "blocked_unknown", "reason": "detector_confidence_insufficient"})

    assert action["action"] == "block"
    assert action["verdict"] == "blocked_unknown"
    assert "human" not in json.dumps(action)


def test_candidate_untrusted_by_default(tmp_path: Path) -> None:
    manifest = build_candidate_manifest(candidate_id="c001", output_dir=tmp_path / "c001")

    assert manifest["candidate_id"] == "c001"
    assert manifest["candidate_untrusted"] is True
    assert manifest["verdict"] == "blocked_unknown"


def test_best_bad_candidate_is_not_accepted() -> None:
    result = select_accepted_candidate(
        [
            {"candidate_id": "c001", "verdict": "blocked_leak", "score": 0.9},
            {"candidate_id": "c002", "verdict": "blocked_damage", "score": 0.8},
        ]
    )

    assert result["verdict"] == "blocked_no_clean_candidate"
    assert result["accepted_candidate_id"] == ""


def test_no_artifact_rag_write_in_scratch_search(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="artifact_rag"):
        validate_no_artifact_rag_output(Path("D:/AIMusicProject/artifact_rag/ar47"))

    mix = tmp_path / "mix.wav"
    mix.write_bytes(b"not-a-real-audio-file")
    out = tmp_path / "scratch" / "t001"

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=out,
        window_sec=6.0,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        allow_default_recipe=False,
    )

    assert result["verdict"] == "blocked_no_clean_candidate"
    assert (out / "run.json").exists()
    assert result["artifact_rag_write_allowed"] is False
    assert not (out / "artifact_rag").exists()
    assert "human" not in json.dumps(result)


def test_default_from_scratch_recipe_uses_demucs_six_stem_route() -> None:
    recipe = build_default_from_scratch_recipe(demucs_available=True)

    assert recipe["schema"] == "stem_search_recipe.v1"
    assert recipe["candidates"][0]["candidate_id"] == "c001"
    assert recipe["candidates"][0]["route_family"] == "demucs_htdemucs_6s"
    assert "run_demucs_six_stem_candidate.py" in " ".join(recipe["candidates"][0]["command_template"])
    assert recipe["candidates"][0]["required_stems"] == list(STEMS)
    assert len(recipe["candidates"]) >= 5
    assert any("--residual-correction-stem" in candidate["command_template"] for candidate in recipe["candidates"])


def test_no_default_separator_blocks_without_old_recipe(tmp_path: Path) -> None:
    mix = write_wav(tmp_path / "mix.wav", sine(220.0, seconds=0.25))
    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        demucs_available=False,
    )

    assert result["verdict"] == "blocked_no_available_separator"
    assert result["failed_checks"] == ["blocked_no_available_separator"]
    assert result["candidate_count"] == 0


def test_recipe_candidate_executes_and_blocks_without_contamination_proof(tmp_path: Path) -> None:
    mix = write_wav(tmp_path / "mix.wav", sine(220.0, seconds=0.25))
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-six-stems"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        assert command == ["fixture-write-six-stems"]
        stems_dir = fake_runner.stems_dir
        mix_audio, mix_rate = sf.read(mix, dtype="float32", always_2d=False)
        for stem in STEMS:
            audio = mix_audio if stem == "vocals" else np.zeros_like(mix_audio, dtype=np.float32)
            sf.write(stems_dir / f"{stem}.flac", np.asarray(audio, dtype=np.float32), mix_rate)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    assert result["candidate_count"] == 1
    assert result["verdict"] == "blocked_no_clean_candidate"
    assert candidate["candidate_id"] == "c001"
    assert candidate["execution_status"] == "completed"
    assert candidate["verdict"] == "blocked_unknown"
    assert Path(candidate["stem_paths"]["minus"]).exists()
    assert result["windows"][0]["window_id"] == "w000"


def test_recipe_candidate_missing_stem_blocks_candidate(tmp_path: Path) -> None:
    mix = write_wav(tmp_path / "mix.wav", sine(220.0, seconds=0.25))
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps({"candidates": [{"candidate_id": "c001", "command_template": ["fixture-write-incomplete"]}]}),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        stems_dir = fake_runner.stems_dir
        for stem in ("vocals", "drums"):
            write_wav(stems_dir / f"{stem}.flac", sine(330.0, seconds=0.25, amp=0.02))

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    assert candidate["verdict"] == "blocked_missing_stem"
    assert "bass" in candidate["missing_stems"]
    assert result["verdict"] == "blocked_no_clean_candidate"


def test_bad_reconstruction_blocks_candidate(tmp_path: Path) -> None:
    mix = write_wav(tmp_path / "mix.wav", sine(220.0, seconds=0.25))
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps({"candidates": [{"candidate_id": "c001", "command_template": ["fixture-write-bad-recon"]}]}),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        stems_dir = fake_runner.stems_dir
        for stem in STEMS:
            write_wav(stems_dir / f"{stem}.flac", np.zeros(int(SAMPLE_RATE * 0.25), dtype=np.float32))

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    assert candidate["verdict"] == "blocked_bad_reconstruction"
    assert candidate["reconstruction"]["ok"] is False
    assert result["verdict"] == "blocked_no_clean_candidate"
