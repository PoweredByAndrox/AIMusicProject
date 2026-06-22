from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import tools.run_fully_automatic_stem_search as stem_search
from tools.run_fully_automatic_stem_search import (
    _candidate_contamination_checks,
    _copy_target_specific_outputs,
    _discover_default_contamination_model_manifest,
    apply_stem_merges,
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
from tools.train_stem_contamination_detector import train_stem_contamination_detector


SAMPLE_RATE = 16_000


def sine(freq: float, seconds: float = 0.25, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return path


def write_full_foreign_references(tmp_path: Path, source_audio: dict[str, np.ndarray]) -> dict[str, dict[str, str]]:
    ref_paths = {
        stem: write_wav(tmp_path / "refs" / f"{stem}.flac", audio)
        for stem, audio in source_audio.items()
    }
    return {
        tested_stem: {foreign_source: str(ref_paths[foreign_source]) for foreign_source in foreign_sources}
        for tested_stem, foreign_sources in FORBIDDEN_SOURCES.items()
    }


def find_leak_result(candidate: dict, tested_stem: str, foreign_source: str) -> dict:
    return next(
        result
        for result in candidate["contamination_score"]["leak_results"]
        if result["tested_stem"] == tested_stem and result["foreign_source"] == foreign_source
    )


def test_apply_stem_merges_moves_other_into_guitar(tmp_path: Path) -> None:
    guitar_audio = sine(330.0, amp=0.05)
    other_audio = sine(550.0, amp=0.02)
    guitar_path = write_wav(tmp_path / "guitar.wav", guitar_audio)
    other_path = write_wav(tmp_path / "other.wav", other_audio)

    result = apply_stem_merges(
        stem_paths={"guitar": str(guitar_path), "other": str(other_path)},
        merge_specs=[{"source_stem": "other", "target_stem": "guitar", "mode": "move"}],
    )

    merged_guitar, _ = sf.read(guitar_path, dtype="float32", always_2d=False)
    emptied_other, _ = sf.read(other_path, dtype="float32", always_2d=False)

    assert result["applied"] is True
    assert result["blocker_reasons"] == []
    assert result["operations"] == [
        {
            "source_stem": "other",
            "target_stem": "guitar",
            "mode": "move",
            "source_path": str(other_path),
            "target_path": str(guitar_path),
        }
    ]
    np.testing.assert_allclose(merged_guitar, guitar_audio + other_audio, atol=1e-4)
    np.testing.assert_allclose(emptied_other, np.zeros_like(other_audio), atol=1e-4)


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


def test_run_accept_requires_every_window_to_have_clean_candidate() -> None:
    result = select_accepted_candidate(
        [
            {"candidate_id": "c001", "window_id": "w000", "verdict": "blocked_leak"},
            {"candidate_id": "c001", "window_id": "w001", "verdict": "auto_accept"},
        ],
        required_window_ids=["w000", "w001"],
    )

    assert result["verdict"] == "blocked_no_clean_candidate"
    assert result["delivery_allowed"] is False
    assert result["accepted_candidate_id"] == ""
    assert result["blocked_window_ids"] == ["w000"]


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
    assert result["delivery_status"] == "blocked"
    assert result["presentation_status"] == "blocked_from_listen_pack"
    assert result["listen_pack"] == {}
    assert (out / "run.json").exists()
    assert not (out / "listen").exists()
    assert result["artifact_rag_write_allowed"] is False
    assert not (out / "artifact_rag").exists()
    assert "human" not in json.dumps(result)


def test_split_audio_windows_merges_tiny_tail_into_previous_window(tmp_path: Path) -> None:
    mix = write_wav(tmp_path / "mix.wav", sine(220.0, seconds=1.025))

    windows = stem_search.split_audio_windows(mix, tmp_path / "split", window_sec=1.0)

    assert len(windows) == 1
    assert windows[0]["window_id"] == "w000"
    assert windows[0]["start_sec"] == 0.0
    assert windows[0]["end_sec"] == pytest.approx(1.025, abs=1.0 / SAMPLE_RATE)
    assert windows[0]["sample_count"] == int(round(1.025 * SAMPLE_RATE))
    assert windows[0]["short_tail_merged"] is True
    assert windows[0]["merged_tail_sec"] == pytest.approx(0.025, abs=1.0 / SAMPLE_RATE)


def test_default_from_scratch_recipe_uses_target_specific_real_routes() -> None:
    recipe = build_default_from_scratch_recipe(
        demucs_available=True,
        available_vocal_routes=("audio_separator_bs_roformer_vocals", "mdx23c_vocals"),
    )

    assert recipe["schema"] == "stem_search_recipe.v1"
    assert recipe["candidates"][0]["candidate_id"] == "c001"
    assert recipe["candidates"][0]["route_family"] == "target_specific_demucs_6s_audio_separator_bs_roformer_vocals"
    assert recipe["candidates"][0]["required_stems"] == list(STEMS)
    assert [route["role"] for route in recipe["candidates"][0]["target_specific_commands"]] == [
        "six_stems",
        "four_stem_non_vocal",
        "vocals",
    ]
    first_commands = recipe["candidates"][0]["target_specific_commands"]
    assert "run_demucs_six_stem_candidate.py" in " ".join(first_commands[0]["command_template"])
    assert "run_registered_non_vocal_candidate.py" in " ".join(first_commands[1]["command_template"])
    assert "export_audio_separator_vocal_full.py" in " ".join(first_commands[2]["command_template"])
    assert first_commands[1]["command_template"][first_commands[1]["command_template"].index("--model-id") + 1] == "scnet_xl_ihf_musdb18"
    assert [candidate["vocal_route_id"] for candidate in recipe["candidates"][:2]] == [
        "audio_separator_bs_roformer_vocals",
        "mdx23c_vocals",
    ]
    assert all(
        candidate["conservation_repair_stem"] == "other"
        for candidate in recipe["candidates"]
        if candidate["candidate_id"] != "c010"
    )
    assert all(candidate["max_conservation_repair_ratio"] == 0.30 for candidate in recipe["candidates"])
    assert all("contamination_checks" not in candidate for candidate in recipe["candidates"])
    assert len(recipe["candidates"]) == 10
    assert any(candidate.get("minus_vocal_debleed", {}).get("strength") == 1.0 for candidate in recipe["candidates"])
    assert any(candidate.get("vocal_projection_debleed", {}).get("strength") == 1.0 for candidate in recipe["candidates"])
    assert any(
        "run_registered_six_stem_candidate.py" in " ".join(candidate.get("command_template") or [])
        for candidate in recipe["candidates"]
    )
    assert any(
        route.get("role") == "four_stem_non_vocal"
        for candidate in recipe["candidates"]
        for route in candidate.get("target_specific_commands", [])
    )
    c009 = next(candidate for candidate in recipe["candidates"] if candidate["candidate_id"] == "c009")
    c010 = next(candidate for candidate in recipe["candidates"] if candidate["candidate_id"] == "c010")
    assert c009["vocal_route_id"] == "audio_separator_bs_roformer_vocals"
    assert c010["route_family"] == "registered_six_stem_hybrid_htdemucs_ft_bs_roformer_6s_aggressive_melody_first"
    assert c010["registered_model_id"] == "hybrid_htdemucs_ft_bs_roformer_6s_aggressive_melody_first"
    assert c010["conservation_repair_stem"] == "guitar"
    assert c010["stem_merges"] == [
        {
            "source_stem": "other",
            "target_stem": "guitar",
            "mode": "move",
            "reason": "aggressive_melody_first_harmonic_bus",
        }
    ]
    assert (
        c010["command_template"][c010["command_template"].index("--model-id") + 1]
        == "hybrid_htdemucs_ft_bs_roformer_6s_aggressive_melody_first"
    )
    assert any(
        "--residual-correction-stem" in route["command_template"]
        for candidate in recipe["candidates"]
        for route in candidate.get("target_specific_commands", [])
    )


def test_default_from_scratch_recipe_records_contamination_model_manifest(tmp_path: Path) -> None:
    model_manifest = tmp_path / "cycle001" / "model" / "manifest.json"
    model_manifest.parent.mkdir(parents=True)
    model_manifest.write_text("{}", encoding="utf-8")

    recipe = build_default_from_scratch_recipe(
        demucs_available=True,
        available_vocal_routes=("audio_separator_bs_roformer_vocals",),
        contamination_model_manifest=model_manifest,
    )

    assert recipe["contamination_model_manifest"] == str(model_manifest)


def test_default_contamination_model_manifest_discovery_uses_latest_cycle(tmp_path: Path) -> None:
    old_manifest = tmp_path / "stem_eval" / "cycle002" / "model" / "manifest.json"
    new_manifest = tmp_path / "stem_eval" / "cycle010" / "model" / "manifest.json"
    old_manifest.parent.mkdir(parents=True)
    new_manifest.parent.mkdir(parents=True)
    old_manifest.write_text("{}", encoding="utf-8")
    new_manifest.write_text("{}", encoding="utf-8")

    discovered = _discover_default_contamination_model_manifest((tmp_path / "stem_eval",))

    assert discovered == new_manifest


def test_default_from_scratch_recipe_blocks_without_specialized_vocal_route() -> None:
    recipe = build_default_from_scratch_recipe(demucs_available=True, available_vocal_routes=())

    assert recipe["available"] is False
    assert recipe["unavailable_reason"] == "missing_target_specific_vocal_route"
    assert recipe["candidates"] == []


def test_four_stem_non_vocal_role_overwrites_only_drums_bass_other(tmp_path: Path) -> None:
    role_dir = tmp_path / "role" / "stems"
    final_dir = tmp_path / "final"
    original = {
        "vocals": sine(880.0, amp=0.02),
        "drums": sine(100.0, amp=0.03),
        "bass": sine(140.0, amp=0.04),
        "guitar": sine(330.0, amp=0.05),
        "piano": sine(440.0, amp=0.06),
        "other": sine(550.0, amp=0.07),
    }
    replacement = {
        "drums": sine(111.0, amp=0.11),
        "bass": sine(155.0, amp=0.12),
        "other": sine(600.0, amp=0.13),
    }
    for stem, audio in original.items():
        write_wav(final_dir / f"{stem}.flac", audio)
    for stem, audio in replacement.items():
        write_wav(role_dir / f"{stem}.flac", audio)

    blockers = _copy_target_specific_outputs(
        role="four_stem_non_vocal",
        role_stems_dir=role_dir,
        final_stems_dir=final_dir,
        target_sample_rate=SAMPLE_RATE,
        target_sample_count=len(next(iter(original.values()))),
        target_channel_count=1,
    )

    assert blockers == []
    for stem, audio in replacement.items():
        actual, _ = sf.read(final_dir / f"{stem}.flac", dtype="float32", always_2d=False)
        np.testing.assert_allclose(actual, audio, atol=2e-4)
    for stem in ("vocals", "guitar", "piano"):
        actual, _ = sf.read(final_dir / f"{stem}.flac", dtype="float32", always_2d=False)
        np.testing.assert_allclose(actual, original[stem], atol=2e-4)


def test_contamination_model_manifest_applies_to_full_matrix() -> None:
    checks, hints_used = _candidate_contamination_checks(
        {
            "contamination_model_manifest": "artifacts/scd/m001/manifest.json",
            "contamination_checks": [
                {
                    "tested_stem": "minus",
                    "foreign_source": "vocals",
                    "model_manifest": "artifacts/scd/m002/manifest.json",
                }
            ],
        }
    )

    required_check_count = sum(len(sources) for sources in FORBIDDEN_SOURCES.values())
    assert hints_used is True
    assert len(checks) == required_check_count
    assert all(check["model_manifest"] for check in checks)
    assert next(
        check for check in checks if check["tested_stem"] == "minus" and check["foreign_source"] == "vocals"
    )["model_manifest"] == "artifacts/scd/m002/manifest.json"
    assert next(
        check for check in checks if check["tested_stem"] == "vocals" and check["foreign_source"] == "drums"
    )["model_manifest"] == "artifacts/scd/m001/manifest.json"


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


def test_recipe_candidate_blocks_clean_minus_check_when_full_matrix_unproven(tmp_path: Path) -> None:
    vocal = sine(880.0, seconds=0.25, amp=0.04)
    drums = sine(110.0, seconds=0.25, amp=0.12)
    mix = write_wav(tmp_path / "mix.wav", vocal + drums)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-clean", "{window_mix}"],
                        "contamination_checks": [{"tested_stem": "minus", "foreign_source": "vocals"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        assert command[0] == "fixture-write-clean"
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        write_wav(stems_dir / "vocals.flac", vocal[: len(window_audio)])
        write_wav(stems_dir / "drums.flac", drums[: len(window_audio)])
        for stem in ("bass", "guitar", "piano", "other"):
            sf.write(stems_dir / f"{stem}.flac", np.zeros_like(window_audio, dtype=np.float32), mix_rate)

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
    assert result["verdict"] == "blocked_no_clean_candidate"
    assert result["delivery_allowed"] is False
    assert result["listen_pack"] == {}
    assert candidate["verdict"] == "blocked_unknown"
    assert candidate["contamination_score"]["check_mode"] == "full_forbidden_matrix"
    assert "contamination_detector_unknown:vocals:drums" in candidate["contamination_score"]["blocker_reasons"]
    assert find_leak_result(candidate, "minus", "vocals")["detector_mode"] == "no_reference_spectral_v1"


def test_recipe_level_contamination_model_manifest_reaches_candidate_checks(tmp_path: Path) -> None:
    vocal = sine(880.0, seconds=0.25, amp=0.04)
    drums = sine(110.0, seconds=0.25, amp=0.12)
    mix = write_wav(tmp_path / "mix.wav", vocal + drums)
    recipe = tmp_path / "recipe.json"
    missing_manifest = tmp_path / "missing_model" / "manifest.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "contamination_model_manifest": str(missing_manifest),
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-clean-with-model", "{window_mix}"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        write_wav(stems_dir / "vocals.flac", vocal[: len(window_audio)])
        write_wav(stems_dir / "drums.flac", drums[: len(window_audio)])
        for stem in ("bass", "guitar", "piano", "other"):
            sf.write(stems_dir / f"{stem}.flac", np.zeros_like(window_audio, dtype=np.float32), mix_rate)

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
    assert result["verdict"] == "blocked_no_clean_candidate"
    assert candidate["contamination_score"]["candidate_check_hints_used"] is True
    assert candidate["contamination_score"]["attempted_checks"] == sum(len(sources) for sources in FORBIDDEN_SOURCES.values())
    assert any("no_reference_model_unavailable" in reason for reason in candidate["contamination_score"]["blocker_reasons"])


def test_recipe_candidate_accepts_only_with_full_forbidden_matrix_references(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(100.0, seconds=0.5, amp=0.03),
        "drums": sine(200.0, seconds=0.5, amp=0.04),
        "bass": sine(300.0, seconds=0.5, amp=0.05),
        "guitar": sine(400.0, seconds=0.5, amp=0.02),
        "piano": sine(500.0, seconds=0.5, amp=0.02),
        "other": sine(600.0, seconds=0.5, amp=0.01),
    }
    mix = write_wav(tmp_path / "mix.wav", sum(source_audio.values()))
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-full-matrix-clean", "{window_mix}"],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        for stem, audio in source_audio.items():
            sf.write(stems_dir / f"{stem}.flac", audio[: len(window_audio)].astype(np.float32), mix_rate)

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
    required_check_count = sum(len(sources) for sources in FORBIDDEN_SOURCES.values())
    assert result["verdict"] == "auto_accept"
    assert result["delivery_allowed"] is True
    assert candidate["verdict"] == "auto_accept"
    assert candidate["contamination_score"]["check_mode"] == "full_forbidden_matrix"
    assert candidate["contamination_score"]["attempted_checks"] == required_check_count
    assert candidate["contamination_score"]["blocker_reasons"] == []


def test_target_specific_vocal_route_overrides_six_stem_vocals(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.04),
        "drums": sine(110.0, seconds=0.5, amp=0.12),
        "bass": sine(220.0, seconds=0.5, amp=0.08),
        "guitar": sine(330.0, seconds=0.5, amp=0.06),
        "piano": sine(440.0, seconds=0.5, amp=0.05),
        "other": sine(550.0, seconds=0.5, amp=0.03),
    }
    bad_vocal = sine(660.0, seconds=0.5, amp=0.03)
    mix = write_wav(tmp_path / "mix.wav", sum(source_audio.values()))
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "target_specific_commands": [
                            {
                                "role": "six_stems",
                                "command_template": ["fixture-write-six-stem-route", "{window_mix}"],
                            },
                            {
                                "role": "vocals",
                                "command_template": ["fixture-write-vocal-route", "{window_mix}"],
                            },
                        ],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        if command[0] == "fixture-write-six-stem-route":
            for stem, audio in source_audio.items():
                route_audio = bad_vocal if stem == "vocals" else audio
                sf.write(stems_dir / f"{stem}.flac", route_audio[: len(window_audio)].astype(np.float32), mix_rate)
        elif command[0] == "fixture-write-vocal-route":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][: len(window_audio)].astype(np.float32), mix_rate)
        else:
            raise AssertionError(command)

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
    accepted_vocal, _ = sf.read(candidate["stem_paths"]["vocals"], dtype="float32", always_2d=False)
    assert result["verdict"] == "auto_accept"
    assert candidate["target_specific_composition"]["applied"] is True
    assert [route["role"] for route in candidate["target_specific_composition"]["routes"]] == ["six_stems", "vocals"]
    np.testing.assert_allclose(accepted_vocal, source_audio["vocals"], atol=2e-4)


def test_target_specific_missing_route_output_blocks(tmp_path: Path) -> None:
    vocal = sine(880.0, seconds=0.25, amp=0.04)
    drums = sine(110.0, seconds=0.25, amp=0.12)
    mix = write_wav(tmp_path / "mix.wav", vocal + drums)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "target_specific_commands": [
                            {
                                "role": "six_stems",
                                "command_template": ["fixture-write-six-stem-route", "{window_mix}"],
                            },
                            {
                                "role": "vocals",
                                "command_template": ["fixture-write-missing-vocal-route", "{window_mix}"],
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        if command[0] == "fixture-write-six-stem-route":
            write_wav(stems_dir / "vocals.flac", vocal[: len(window_audio)])
            write_wav(stems_dir / "drums.flac", drums[: len(window_audio)])
            for stem in ("bass", "guitar", "piano", "other"):
                sf.write(stems_dir / f"{stem}.flac", np.zeros_like(window_audio, dtype=np.float32), mix_rate)
        elif command[0] == "fixture-write-missing-vocal-route":
            stems_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise AssertionError(command)

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
    assert result["verdict"] == "blocked_no_clean_candidate"
    assert candidate["verdict"] == "blocked_missing_stem"
    assert "missing_target_specific_output:vocals:vocals" in candidate["target_specific_composition"]["blocker_reasons"]


def test_target_specific_output_is_normalized_to_window_shape(tmp_path: Path) -> None:
    source_dir = tmp_path / "route" / "stems"
    final_dir = tmp_path / "final"
    low_rate = SAMPLE_RATE // 2
    low_t = np.arange(int(low_rate * 0.5), dtype=np.float32) / float(low_rate)
    low_rate_vocal = (0.04 * np.sin(2.0 * math.pi * 440.0 * low_t)).astype(np.float32)
    source_dir.mkdir(parents=True, exist_ok=True)
    sf.write(source_dir / "vocals.flac", low_rate_vocal, low_rate)

    blockers = _copy_target_specific_outputs(
        role="vocals",
        role_stems_dir=source_dir,
        final_stems_dir=final_dir,
        target_sample_rate=SAMPLE_RATE,
        target_sample_count=int(SAMPLE_RATE * 0.5),
    )

    info = sf.info(final_dir / "vocals.flac")
    assert blockers == []
    assert int(info.samplerate) == SAMPLE_RATE
    assert int(info.frames) == int(SAMPLE_RATE * 0.5)


def test_listen_pack_build_failure_removes_partial_listen_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_audio = {
        "vocals": sine(100.0, seconds=0.5, amp=0.03),
        "drums": sine(200.0, seconds=0.5, amp=0.04),
        "bass": sine(300.0, seconds=0.5, amp=0.05),
        "guitar": sine(400.0, seconds=0.5, amp=0.02),
        "piano": sine(500.0, seconds=0.5, amp=0.02),
        "other": sine(600.0, seconds=0.5, amp=0.01),
    }
    mix = write_wav(tmp_path / "mix.wav", sum(source_audio.values()))
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-accepted", "{window_mix}"],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        for stem, audio in source_audio.items():
            sf.write(stems_dir / f"{stem}.flac", audio[: len(window_audio)].astype(np.float32), mix_rate)

    original_write_audio = stem_search._write_audio

    def fail_on_listen_drums(path: Path, audio: np.ndarray, sample_rate: int) -> Path:
        if path.name == "drums.flac" and "listen" in path.parts:
            raise RuntimeError("forced listen pack write failure")
        return original_write_audio(path, audio, sample_rate)

    monkeypatch.setattr(stem_search, "_write_audio", fail_on_listen_drums)
    out = tmp_path / "scratch" / "t001"

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=out,
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    assert result["verdict"] == "blocked_unknown"
    assert result["delivery_status"] == "blocked"
    assert result["presentation_status"] == "blocked_from_listen_pack"
    assert result["listen_pack"] == {}
    assert result["listen_pack_error"]["message"] == "forced listen pack write failure"
    assert not (out / "listen").exists()


def test_recipe_candidate_blocks_no_reference_vocal_leak_in_minus(tmp_path: Path) -> None:
    vocal = sine(880.0, seconds=0.25, amp=0.04)
    drums = sine(110.0, seconds=0.25, amp=0.12)
    mix = write_wav(tmp_path / "mix.wav", vocal + drums)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-leaky", "{window_mix}"],
                        "contamination_checks": [{"tested_stem": "minus", "foreign_source": "vocals"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        assert command[0] == "fixture-write-leaky"
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        leak = 0.5 * vocal[: len(window_audio)]
        stems_dir = fake_runner.stems_dir
        write_wav(stems_dir / "vocals.flac", vocal[: len(window_audio)] - leak)
        write_wav(stems_dir / "drums.flac", drums[: len(window_audio)] + leak)
        for stem in ("bass", "guitar", "piano", "other"):
            sf.write(stems_dir / f"{stem}.flac", np.zeros_like(window_audio, dtype=np.float32), mix_rate)

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
    leak_result = find_leak_result(candidate, "minus", "vocals")
    assert result["verdict"] == "blocked_no_clean_candidate"
    assert candidate["verdict"] == "blocked_leak"
    assert leak_result["verdict"] == "fail"
    assert leak_result["leak_audio"]
    assert Path(candidate["output_dir"], "score", leak_result["leak_audio"]).exists()


def test_recipe_candidate_uses_trainable_no_reference_model(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    rows = []
    for idx, freq in enumerate((820.0, 880.0, 960.0), start=1):
        clean = sine(110.0 + idx * 20.0, seconds=0.25, amp=0.18) + sine(220.0, seconds=0.25, amp=0.04)
        leak = sine(freq, seconds=0.25, amp=0.045)
        row_id = f"d{idx:06d}"
        rows.append(
            {
                "row_id": row_id,
                "input_stem": str(write_wav(tmp_path / "train" / f"{row_id}_in.flac", clean + leak)),
                "target_leak": str(write_wav(tmp_path / "train" / f"{row_id}_leak.flac", leak)),
                "base_stem": "minus",
                "foreign_source": "vocals",
                "label_type": "synthetic_clean",
                "is_ground_truth": True,
                "split": "train",
            }
        )
    rows_jsonl = tmp_path / "rows.jsonl"
    rows_jsonl.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    model_dir = tmp_path / "model"
    train_stem_contamination_detector(
        rows_jsonl=rows_jsonl,
        output_dir=model_dir,
        device="cpu",
        epochs=120,
        hidden_size=8,
        min_positive_rows=2,
    )

    vocal = sine(880.0, seconds=0.25, amp=0.04)
    drums = sine(110.0, seconds=0.25, amp=0.12)
    mix = write_wav(tmp_path / "mix.wav", vocal + drums)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-model-clean", "{window_mix}"],
                        "contamination_checks": [
                            {
                                "tested_stem": "minus",
                                "foreign_source": "vocals",
                                "model_manifest": str(model_dir / "manifest.json"),
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        write_wav(stems_dir / "vocals.flac", vocal[: len(window_audio)])
        write_wav(stems_dir / "drums.flac", drums[: len(window_audio)])
        for stem in ("bass", "guitar", "piano", "other"):
            sf.write(stems_dir / f"{stem}.flac", np.zeros_like(window_audio, dtype=np.float32), mix_rate)

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
    leak_result = find_leak_result(candidate, "minus", "vocals")
    assert result["verdict"] == "blocked_no_clean_candidate"
    assert candidate["verdict"] == "blocked_unknown"
    assert leak_result["detector_mode"] == "no_reference_neural_v1"
    assert leak_result["model_manifest"] == str(model_dir / "manifest.json")
    assert "contamination_detector_unknown:vocals:drums" in candidate["contamination_score"]["blocker_reasons"]


def test_conservation_repair_assigns_residual_before_reconstruction_gate(tmp_path: Path) -> None:
    vocal = sine(880.0, seconds=0.25, amp=0.04)
    drums = sine(110.0, seconds=0.25, amp=0.12)
    missing_other = sine(330.0, seconds=0.25, amp=0.03)
    mix = write_wav(tmp_path / "mix.wav", vocal + drums + missing_other)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-needs-repair", "{window_mix}"],
                        "conservation_repair_stem": "other",
                        "foreign_references": write_full_foreign_references(
                            tmp_path,
                            {
                                "vocals": vocal,
                                "drums": drums,
                                "bass": np.zeros_like(vocal, dtype=np.float32),
                                "guitar": np.zeros_like(vocal, dtype=np.float32),
                                "piano": np.zeros_like(vocal, dtype=np.float32),
                                "other": missing_other,
                            },
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        write_wav(stems_dir / "vocals.flac", vocal[: len(window_audio)])
        write_wav(stems_dir / "drums.flac", drums[: len(window_audio)])
        for stem in ("bass", "guitar", "piano", "other"):
            sf.write(stems_dir / f"{stem}.flac", np.zeros_like(window_audio, dtype=np.float32), mix_rate)

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
    assert result["verdict"] == "auto_accept"
    assert candidate["reconstruction_repair"]["applied"] is True
    assert candidate["reconstruction_repair"]["repair_owner"] == "runner_conservation_repair_v1"
    assert candidate["reconstruction_repair"]["target_stem"] == "other"
    assert candidate["reconstruction_repair"]["pre_repair_residual_rms"] > 0.0
    assert candidate["reconstruction"]["ok"] is True


def test_conservation_repair_rejects_vocal_target(tmp_path: Path) -> None:
    vocal = sine(880.0, seconds=0.25, amp=0.04)
    drums = sine(110.0, seconds=0.25, amp=0.12)
    missing_other = sine(330.0, seconds=0.25, amp=0.03)
    mix = write_wav(tmp_path / "mix.wav", vocal + drums + missing_other)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-unsafe-repair", "{window_mix}"],
                        "conservation_repair_stem": "vocals",
                        "contamination_checks": [{"tested_stem": "minus", "foreign_source": "vocals"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        write_wav(stems_dir / "vocals.flac", vocal[: len(window_audio)])
        write_wav(stems_dir / "drums.flac", drums[: len(window_audio)])
        for stem in ("bass", "guitar", "piano", "other"):
            sf.write(stems_dir / f"{stem}.flac", np.zeros_like(window_audio, dtype=np.float32), mix_rate)

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
    assert result["verdict"] == "blocked_no_clean_candidate"
    assert candidate["verdict"] == "blocked_bad_reconstruction"
    assert candidate["reconstruction_repair"]["applied"] is False
    assert candidate["reconstruction_repair"]["reason"] == "unsupported_conservation_repair_stem"


def test_conservation_repair_rejects_excessive_residual_ratio(tmp_path: Path) -> None:
    vocal = sine(880.0, seconds=0.25, amp=0.02)
    drums = sine(110.0, seconds=0.25, amp=0.02)
    missing_other = sine(330.0, seconds=0.25, amp=0.15)
    mix = write_wav(tmp_path / "mix.wav", vocal + drums + missing_other)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-excessive-repair", "{window_mix}"],
                        "conservation_repair_stem": "other",
                        "max_conservation_repair_ratio": 0.10,
                        "foreign_references": write_full_foreign_references(
                            tmp_path,
                            {
                                "vocals": vocal,
                                "drums": drums,
                                "bass": np.zeros_like(vocal, dtype=np.float32),
                                "guitar": np.zeros_like(vocal, dtype=np.float32),
                                "piano": np.zeros_like(vocal, dtype=np.float32),
                                "other": missing_other,
                            },
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        write_wav(stems_dir / "vocals.flac", vocal[: len(window_audio)])
        write_wav(stems_dir / "drums.flac", drums[: len(window_audio)])
        for stem in ("bass", "guitar", "piano", "other"):
            sf.write(stems_dir / f"{stem}.flac", np.zeros_like(window_audio, dtype=np.float32), mix_rate)

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
    assert result["verdict"] == "blocked_no_clean_candidate"
    assert candidate["verdict"] == "blocked_bad_reconstruction"
    assert candidate["reconstruction_repair"]["applied"] is False
    assert candidate["reconstruction_repair"]["reason"] == "conservation_repair_ratio_too_high"
    assert candidate["reconstruction_repair"]["max_residual_ratio"] == 0.10


def test_minus_vocal_debleed_does_not_publish_without_full_matrix_proof(tmp_path: Path) -> None:
    vocal_band_residue = sine(900.0, seconds=0.25, amp=0.04)
    bass = sine(110.0, seconds=0.25, amp=0.12)
    mix = write_wav(tmp_path / "mix.wav", vocal_band_residue + bass)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-debleedable", "{window_mix}"],
                        "minus_vocal_debleed": {"strength": 1.0, "low_hz": 700.0, "high_hz": 1200.0},
                        "contamination_checks": [{"tested_stem": "minus", "foreign_source": "vocals"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        sf.write(stems_dir / "vocals.flac", np.zeros_like(window_audio, dtype=np.float32), mix_rate)
        write_wav(stems_dir / "bass.flac", bass[: len(window_audio)])
        write_wav(stems_dir / "other.flac", vocal_band_residue[: len(window_audio)])
        for stem in ("drums", "guitar", "piano"):
            sf.write(stems_dir / f"{stem}.flac", np.zeros_like(window_audio, dtype=np.float32), mix_rate)

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
    assert result["verdict"] == "blocked_no_clean_candidate"
    assert result["delivery_status"] == "blocked"
    assert result["delivery_allowed"] is False
    assert result["presentation_status"] == "blocked_from_listen_pack"
    assert result["listen_pack"] == {}
    assert candidate["minus_vocal_debleed"]["applied"] is True
    assert candidate["minus_vocal_debleed"]["moved_rms"] > 0.0
    assert candidate["reconstruction"]["ok"] is True
    assert candidate["contamination_score"]["verdict"] == "blocked_unknown"


def test_vocal_projection_debleed_moves_correlated_vocal_leak_back_to_vocals(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.04),
        "drums": sine(110.0, seconds=0.5, amp=0.12),
        "bass": sine(220.0, seconds=0.5, amp=0.08),
        "guitar": sine(330.0, seconds=0.5, amp=0.06),
        "piano": sine(440.0, seconds=0.5, amp=0.05),
        "other": sine(550.0, seconds=0.5, amp=0.03),
    }
    drum_leak = 0.3 * source_audio["vocals"]
    guitar_leak = 0.2 * source_audio["vocals"]
    mix = write_wav(tmp_path / "mix.wav", sum(source_audio.values()))
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-projection-debleed", "{window_mix}"],
                        "vocal_projection_debleed": {
                            "strength": 1.0,
                            "target_stems": ["drums", "guitar"],
                        },
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        sf.write(stems_dir / "vocals.flac", (source_audio["vocals"] - drum_leak - guitar_leak)[: len(window_audio)], mix_rate)
        sf.write(stems_dir / "drums.flac", (source_audio["drums"] + drum_leak)[: len(window_audio)], mix_rate)
        sf.write(stems_dir / "bass.flac", source_audio["bass"][: len(window_audio)], mix_rate)
        sf.write(stems_dir / "guitar.flac", (source_audio["guitar"] + guitar_leak)[: len(window_audio)], mix_rate)
        sf.write(stems_dir / "piano.flac", source_audio["piano"][: len(window_audio)], mix_rate)
        sf.write(stems_dir / "other.flac", source_audio["other"][: len(window_audio)], mix_rate)

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
    assert result["verdict"] == "auto_accept"
    assert candidate["vocal_projection_debleed"]["applied"] is True
    assert candidate["vocal_projection_debleed"]["per_stem"]["drums"]["projection_coeff"] == pytest.approx(0.6, abs=0.03)
    assert candidate["vocal_projection_debleed"]["per_stem"]["guitar"]["projection_coeff"] == pytest.approx(0.4, abs=0.03)
    assert candidate["reconstruction"]["ok"] is True


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
