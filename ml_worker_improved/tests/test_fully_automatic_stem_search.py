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
    apply_minus_vocal_debleed,
    apply_stem_merges,
    build_candidate_manifest,
    build_default_from_scratch_recipe,
    build_model_split_v2_recipe,
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
from tools.stem_contamination_model import band_limited_audio
from tools.train_stem_contamination_detector import train_stem_contamination_detector
from tools.stem_region_repair_common import ISSUE_STEREO_COLLAPSE, make_region_mask
from tools.stem_region_repair_model import (
    StemRegionRepairTrainConfig,
    save_region_repair_checkpoint,
    train_region_repair_model_from_arrays,
)


SAMPLE_RATE = 16_000


def sine(freq: float, seconds: float = 0.25, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def bright_voice_like_vocal(seconds: float = 0.5) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    f0 = 155.0 + 24.0 * np.sin(2.0 * math.pi * 5.3 * t)
    phase = np.cumsum(2.0 * math.pi * f0 / float(SAMPLE_RATE)).astype(np.float32)
    audio = np.zeros_like(t)
    for harmonic in range(1, 18):
        audio += (0.12 / float(harmonic) ** 0.82) * np.sin(float(harmonic) * phase + 0.11 * harmonic).astype(np.float32)
    return (audio * (0.46 + 0.54 * np.sin(2.0 * math.pi * 2.7 * t) ** 2)).astype(np.float32)


def cymbal_like_drums(seconds: float = 0.5) -> np.ndarray:
    frames = int(SAMPLE_RATE * seconds)
    audio = np.zeros(frames, dtype=np.float32)
    rng = np.random.default_rng(4242)
    pulse_spacing = int(SAMPLE_RATE * 0.16)
    decay_len = int(SAMPLE_RATE * 0.05)
    decay = np.exp(-np.arange(decay_len, dtype=np.float32) / float(SAMPLE_RATE * 0.008))
    for start in range(0, frames, pulse_spacing):
        end = min(frames, start + decay_len)
        audio[start:end] += rng.normal(0.0, 0.25, end - start).astype(np.float32) * decay[: end - start]
    return audio.astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return path


def write_stereo_wav(path: Path, left: np.ndarray, right: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.stack([left, right], axis=1).astype(np.float32)
    sf.write(path, audio, SAMPLE_RATE)
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
    assert result["training_first_root_cause_queue_manifest"].endswith("training/root_cause/manifest.json")
    assert (out / "training" / "root_cause" / "manifest.json").exists()
    assert (out / "training" / "root_cause" / "queue.jsonl").exists()


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
    assert c010["minus_vocal_debleed"]["passes"] == 2
    assert c010["post_minus_vocal_debleed"]["passes"] == 1
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


def test_default_contamination_model_manifest_searches_project_artifacts_mount() -> None:
    assert stem_search.DEFAULT_CONTAMINATION_MODEL_SEARCH_ROOTS[0] == Path("/project_artifacts/stem_eval")


def test_default_from_scratch_recipe_blocks_without_specialized_vocal_route() -> None:
    recipe = build_default_from_scratch_recipe(demucs_available=True, available_vocal_routes=())

    assert recipe["available"] is False
    assert recipe["unavailable_reason"] == "missing_target_specific_vocal_route"
    assert recipe["candidates"] == []


def test_model_split_v2_recipe_builds_separate_lanes() -> None:
    recipe = build_model_split_v2_recipe(
        demucs_available=True,
        available_vocal_routes=("audio_separator_bs_roformer_vocals", "mdx23c_vocals"),
    )

    assert recipe["schema"] == "stem_search_recipe.v2"
    assert recipe["profile"] == "model_split_v2"
    assert recipe["target_specific_routes"]["vocal_lane"] == [
        "bs_mamba2_vocals",
        "audio_separator_bs_roformer_vocals",
        "mdx23c_vocals",
        "melband_roformer_vocals",
    ]
    assert recipe["target_specific_routes"]["minus_lane"] == [
        "bs_mamba2_vocals:instrumental",
        "becruily_instrumental_vocals",
        "audio_separator_bs_roformer_vocals:instrumental",
        "mdx23c_vocals:instrumental",
        "htdemucs_ft:instrumental",
    ]
    assert recipe["target_specific_routes"]["stem_lane"] == [
        "htdemucs_6s",
        "hybrid_htdemucs_ft_bs_roformer_6s",
        "hybrid_htdemucs_ft_bs_roformer_6s_aggressive_melody_first",
    ]
    assert recipe["target_specific_routes"]["bass_support_lane"] == [
        "scnet_xl_ihf_musdb18:bass",
    ]
    assert recipe["target_specific_routes"]["non_vocal_support_lane"] == [
        "scnet_xl_ihf_musdb18:drums_bass_other",
    ]
    assert recipe["model_split_composite_max_candidates"] == 864
    first = recipe["candidates"][0]
    assert first["minus_source"] == "minus_lane"
    assert first["minus_rebuilt_from_non_vocal_stems"] is False
    assert first["delivery_pair_reconstruction"] == {
        "enabled": True,
        "max_delivery_pair_residual_ratio": 0.06,
        "six_stem_diagnostic_residual_ratio": 0.01,
    }
    assert first["model_split_lanes"] == {
        "vocals": "vocal_lane",
        "minus": "minus_lane",
        "stems": "stem_lane",
        "bass_support": "bass_support_lane",
        "non_vocal_support": "non_vocal_support_lane",
    }
    assert [option["repair_id"] for option in first["model_split_composite_repair_options"]] == [
        "none",
        "vp100",
        "bs250",
        "bs400",
        "pg08",
        "pr16",
        "bs400_pr16",
        "vp100_bs400",
        "bs400_pg08",
        "vp100_pg08",
        "vp100_bs400_pg08",
    ]
    assert [route["role"] for route in first["target_specific_commands"]] == [
        "six_stems",
        "vocals",
        "minus",
        "non_vocal_support",
    ]
    non_vocal_support = first["target_specific_commands"][-1]
    assert non_vocal_support["lane"] == "non_vocal_support_lane"
    assert "run_registered_non_vocal_candidate.py" in " ".join(non_vocal_support["command_template"])
    assert non_vocal_support["command_template"][non_vocal_support["command_template"].index("--model-id") + 1] == "scnet_xl_ihf_musdb18"


def test_model_split_v2_recipe_materializes_support_lanes_beyond_first_candidate() -> None:
    recipe = build_model_split_v2_recipe(
        demucs_available=True,
        available_vocal_routes=("audio_separator_bs_roformer_vocals", "mdx23c_vocals"),
    )
    support_roles_by_candidate = {
        candidate["candidate_id"]: {
            command["role"]
            for command in candidate["target_specific_commands"]
            if command["role"] in {"bass_support", "non_vocal_support"}
        }
        for candidate in recipe["candidates"]
    }

    assert any("bass_support" in roles for roles in support_roles_by_candidate.values())
    assert any(
        "non_vocal_support" in roles
        for candidate_id, roles in support_roles_by_candidate.items()
        if candidate_id != "c001"
    )


def test_model_split_v2_recipe_can_inject_learned_region_repair_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model" / "model.pt"
    recipe = build_model_split_v2_recipe(
        demucs_available=True,
        stem_region_repair_checkpoints=[
            {
                "repair_id": "sr001",
                "stem": "bass",
                "issue_type": ISSUE_STEREO_COLLAPSE,
                "checkpoint": str(checkpoint),
            }
        ],
    )

    repairs = recipe["candidates"][0]["model_split_composite_repair_options"]

    assert repairs[0] == {"repair_id": "none", "type": "none"}
    assert repairs[1] == {
        "repair_id": "sr001",
        "type": "stem_region_repair",
        "stem": "bass",
        "issue_type": ISSUE_STEREO_COLLAPSE,
        "checkpoint": str(checkpoint),
    }
    assert repairs[2] == {
        "repair_id": "sr001_vp100_pr16",
        "type": "stem_region_repair+vocal_projection_debleed+delivery_pair_residual",
        "stem_region_repair": {
            "stem": "bass",
            "issue_type": ISSUE_STEREO_COLLAPSE,
            "checkpoint": str(checkpoint),
        },
        "vocal_projection_debleed": {
            "strength": 1.0,
            "target_stems": ["drums", "bass", "guitar", "piano", "other"],
        },
        "delivery_pair_residual": {
            "target_stem": "minus",
            "max_residual_ratio": 0.16,
        },
    }
    assert repairs[3] == {
        "repair_id": "sr001_vp100",
        "type": "stem_region_repair+vocal_projection_debleed",
        "stem_region_repair": {
            "stem": "bass",
            "issue_type": ISSUE_STEREO_COLLAPSE,
            "checkpoint": str(checkpoint),
        },
        "vocal_projection_debleed": {
            "strength": 1.0,
            "target_stems": ["drums", "bass", "guitar", "piano", "other"],
        },
    }


def test_model_split_v2_recipe_can_inject_learned_region_repair_sequence(tmp_path: Path) -> None:
    checkpoints = [
        tmp_path / "m1" / "model.pt",
        tmp_path / "m2" / "model.pt",
        tmp_path / "m3" / "model.pt",
    ]
    recipe = build_model_split_v2_recipe(
        demucs_available=True,
        stem_region_repair_checkpoints=[
            {"repair_id": "sr001", "stem": "bass", "issue_type": "vocal_leak", "checkpoint": str(checkpoints[0])},
            {"repair_id": "sr002", "stem": "minus", "issue_type": "vocal_leak", "checkpoint": str(checkpoints[1])},
            {"repair_id": "sr003", "stem": "other", "issue_type": "instrument_leak", "checkpoint": str(checkpoints[2])},
        ],
    )

    repairs = recipe["candidates"][0]["model_split_composite_repair_options"]
    sequence = next(repair for repair in repairs if repair["repair_id"] == "srseq_vp100_pr16")

    assert sequence["type"] == "stem_region_repair+vocal_projection_debleed+delivery_pair_residual"
    assert sequence["stem_region_repairs"] == [
        {"stem": "bass", "issue_type": "vocal_leak", "checkpoint": str(checkpoints[0])},
        {"stem": "minus", "issue_type": "vocal_leak", "checkpoint": str(checkpoints[1])},
        {"stem": "other", "issue_type": "instrument_leak", "checkpoint": str(checkpoints[2])},
    ]
    assert sequence["vocal_projection_debleed"]["target_stems"] == ["drums", "bass", "guitar", "piano", "other"]
    assert sequence["delivery_pair_residual"]["target_stem"] == "minus"


def test_apply_learned_stem_region_repairs_runs_sequence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    applied: list[str] = []
    stem_paths = {
        "bass": str(tmp_path / "bass.flac"),
        "minus": str(tmp_path / "minus.flac"),
        "other": str(tmp_path / "other.flac"),
    }

    def fake_apply_learned_stem_region_repair(*, mix_path: Path, stem_paths: dict[str, str], config: dict[str, str]) -> dict:
        applied.append(f"{config['stem']}:{config['issue_type']}")
        return {
            "applied": True,
            "stem": config["stem"],
            "issue_type": config["issue_type"],
            "checkpoint": config["checkpoint"],
            "blockers": [],
        }

    monkeypatch.setattr(stem_search, "apply_learned_stem_region_repair", fake_apply_learned_stem_region_repair)

    result = stem_search.apply_learned_stem_region_repairs(
        mix_path=tmp_path / "mix.flac",
        stem_paths=stem_paths,
        configs=[
            {"stem": "bass", "issue_type": "vocal_leak", "checkpoint": "b.pt"},
            {"stem": "minus", "issue_type": "vocal_leak", "checkpoint": "m.pt"},
            {"stem": "other", "issue_type": "instrument_leak", "checkpoint": "o.pt"},
        ],
    )

    assert applied == ["bass:vocal_leak", "minus:vocal_leak", "other:instrument_leak"]
    assert result["applied"] is True
    assert result["blockers"] == []
    assert [item["stem"] for item in result["repairs"]] == ["bass", "minus", "other"]


def test_model_split_v2_composite_prioritizes_learned_region_repair_before_dsp_repairs() -> None:
    pairs = stem_search._prioritized_model_split_support_repair_pairs(
        support_options=[
            {},
            {"bass": {"role": "bass_support"}},
            {"drums": {"role": "non_vocal_support"}, "bass": {"role": "non_vocal_support"}},
        ],
        repair_options=[
            {"repair_id": "none", "type": "none"},
            {"repair_id": "sr001", "type": "stem_region_repair"},
            {"repair_id": "pr16", "type": "delivery_pair_residual"},
            {"repair_id": "bs400_pr16", "type": "bass_stereo_width+delivery_pair_residual"},
        ],
    )

    ordered = [
        (repair["repair_id"], "bass" in support, any((route.get("role") == "non_vocal_support") for route in support.values()))
        for support, repair in pairs
    ]
    first_six = ordered[:6]

    assert ("sr001", True, False) in first_six
    assert ("sr001", True, True) in first_six
    assert ordered.index(("sr001", True, False)) < ordered.index(("pr16", True, False))


def test_parse_stem_region_repair_checkpoint_spec_preserves_windows_path() -> None:
    parsed = stem_search._parse_stem_region_repair_checkpoint_spec("bass:stereo_collapse:D:/models/model.pt")

    assert parsed == {
        "repair_id": "sr001",
        "stem": "bass",
        "issue_type": ISSUE_STEREO_COLLAPSE,
        "checkpoint": "D:/models/model.pt",
    }


def test_default_model_split_v2_composite_limit_respects_runtime_max_candidates() -> None:
    recipe = {"model_split_composite_max_candidates": 864}

    assert (
        stem_search._effective_model_split_composite_limit(
            recipe=recipe,
            max_candidates=10,
            default_recipe=True,
        )
        == 10
    )
    assert (
        stem_search._effective_model_split_composite_limit(
            recipe=recipe,
            max_candidates=10,
            default_recipe=False,
        )
        == 864
    )


def test_model_split_v2_htdemucs_ft_minus_uses_registered_cli_contract() -> None:
    recipe = build_model_split_v2_recipe(demucs_available=True)

    htdemucs_ft_minus = next(
        route
        for candidate in recipe["candidates"]
        for route in candidate["target_specific_commands"]
        if route["route_id"] == "htdemucs_ft:instrumental"
    )

    command_template = htdemucs_ft_minus["command_template"]
    assert command_template[:4] == [
        "python",
        "tools/run_registered_model_candidate.py",
        "{input_audio}",
        "{stems_dir}",
    ]
    assert "--input-audio" not in command_template
    assert "--output-dir" not in command_template
    assert command_template[command_template.index("--model-id") + 1] == "htdemucs_ft"
    assert command_template[command_template.index("--target-stem") + 1] == "minus"
    assert htdemucs_ft_minus["output_map"] == {"minus": "minus"}


def test_model_split_v2_bs_mamba2_uses_registered_vocal_minus_lanes() -> None:
    recipe = build_model_split_v2_recipe(demucs_available=True)

    assert "bs_mamba2_vocals" in recipe["target_specific_routes"]["vocal_lane"]
    assert "bs_mamba2_vocals:instrumental" in recipe["target_specific_routes"]["minus_lane"]

    bs_mamba2_vocal = next(
        route
        for candidate in recipe["candidates"]
        for route in candidate["target_specific_commands"]
        if route["route_id"] == "bs_mamba2_vocals"
    )
    bs_mamba2_minus = next(
        route
        for candidate in recipe["candidates"]
        for route in candidate["target_specific_commands"]
        if route["route_id"] == "bs_mamba2_vocals:instrumental"
    )

    vocal_command = bs_mamba2_vocal["command_template"]
    minus_command = bs_mamba2_minus["command_template"]
    assert "tools/run_registered_model_candidate.py" in vocal_command
    assert "tools/run_registered_model_candidate.py" in minus_command
    assert vocal_command[vocal_command.index("--model-id") + 1] == "bs_mamba2_vocals"
    assert minus_command[minus_command.index("--model-id") + 1] == "bs_mamba2_vocals"
    assert vocal_command[vocal_command.index("--target-stem") + 1] == "vocals"
    assert minus_command[minus_command.index("--target-stem") + 1] == "minus"
    assert bs_mamba2_minus["output_map"] == {"minus": "instrumental"}


def test_model_split_v2_minus_delivery_cannot_be_rebuilt_from_six_stems(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.06),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    rebuilt_minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    lane_minus = rebuilt_minus * np.float32(0.5)
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + lane_minus)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "delivery_pair_reconstruction": {"max_delivery_pair_residual_ratio": 2.0},
                        "target_specific_commands": [
                            {"role": "six_stems", "lane": "stem_lane", "command_template": ["fixture-six", "{window_mix}"]},
                            {"role": "vocals", "lane": "vocal_lane", "command_template": ["fixture-vocal", "{window_mix}"]},
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "route_id": "audio_separator_bs_roformer_vocals:instrumental",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus", "{window_mix}"],
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
        if command[0] == "fixture-six":
            for stem, audio in source_audio.items():
                sf.write(stems_dir / f"{stem}.flac", audio[: len(window_audio)], mix_rate)
        elif command[0] == "fixture-vocal":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][: len(window_audio)], mix_rate)
        elif command[0] == "fixture-minus":
            sf.write(stems_dir / "instrumental.flac", lane_minus[: len(window_audio)], mix_rate)
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
    accepted_minus, _ = sf.read(candidate["stem_paths"]["minus"], dtype="float32", always_2d=False)
    assert result["verdict"] == "auto_accept"
    assert candidate["minus_source"] == "minus_lane"
    assert candidate["minus_rebuilt_from_non_vocal_stems"] is False
    np.testing.assert_allclose(accepted_minus, lane_minus, atol=2e-4)
    assert not np.allclose(accepted_minus, rebuilt_minus, atol=2e-4)
    assert result["listen_pack"]["lane_provenance"]["minus"]["lane"] == "minus_lane"
    assert candidate["reconstruction"]["mode"] == "model_split_v2_delivery_pair"
    assert candidate["reconstruction"]["delivery_pair"]["ok"] is True
    assert candidate["reconstruction"]["six_stem_diagnostic"]["ok"] is False
    minus_route = next(
        route
        for route in candidate["target_specific_composition"]["routes"]
        if route["role"] == "minus"
    )
    assert minus_route["route_id"] == "audio_separator_bs_roformer_vocals:instrumental"
    assert minus_route["route_alias"] == "audio_separator_bs_roformer_vocals_instrumental"
    assert ":" not in Path(minus_route["output_dir"]).name


def test_model_split_v2_blocks_delivery_pair_mismatch(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.06),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    rebuilt_minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    lane_minus = rebuilt_minus * np.float32(0.25)
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + rebuilt_minus)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "delivery_pair_reconstruction": {"max_delivery_pair_residual_ratio": 0.06},
                        "target_specific_commands": [
                            {"role": "six_stems", "lane": "stem_lane", "command_template": ["fixture-six", "{window_mix}"]},
                            {"role": "vocals", "lane": "vocal_lane", "command_template": ["fixture-vocal", "{window_mix}"]},
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus", "{window_mix}"],
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
        if command[0] == "fixture-six":
            for stem, audio in source_audio.items():
                sf.write(stems_dir / f"{stem}.flac", audio[: len(window_audio)], mix_rate)
        elif command[0] == "fixture-vocal":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][: len(window_audio)], mix_rate)
        elif command[0] == "fixture-minus":
            sf.write(stems_dir / "instrumental.flac", lane_minus[: len(window_audio)], mix_rate)
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
    assert candidate["verdict"] == "blocked_bad_reconstruction"
    assert candidate["reconstruction"]["mode"] == "model_split_v2_delivery_pair"
    assert candidate["reconstruction"]["delivery_pair"]["ok"] is False


def test_model_split_v2_mandatory_gates_cannot_be_disabled(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.06),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + minus)
    recipe = tmp_path / "recipe.json"
    base_candidate = {
        "profile": "model_split_v2",
        "minus_source": "minus_lane",
        "target_specific_commands": [
            {"role": "six_stems", "lane": "stem_lane", "command_template": ["fixture-six", "{window_mix}"]},
            {"role": "vocals", "lane": "vocal_lane", "command_template": ["fixture-vocal", "{window_mix}"]},
            {
                "role": "minus",
                "lane": "minus_lane",
                "output_map": {"minus": "instrumental"},
                "command_template": ["fixture-minus", "{window_mix}"],
            },
        ],
        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
    }
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "candidates": [
                    {**base_candidate, "candidate_id": "c001", "damage_gates": {"enabled": False}},
                    {**base_candidate, "candidate_id": "c002", "voice_presence_gate": {"enabled": False}},
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        if command[0] == "fixture-six":
            for stem, audio in source_audio.items():
                sf.write(stems_dir / f"{stem}.flac", audio[: len(window_audio)], mix_rate)
        elif command[0] == "fixture-vocal":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][: len(window_audio)], mix_rate)
        elif command[0] == "fixture-minus":
            sf.write(stems_dir / "instrumental.flac", minus[: len(window_audio)], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=2,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    by_id = {candidate["candidate_id"]: candidate for candidate in result["candidates"]}
    assert result["verdict"] == "blocked_no_clean_candidate"
    assert by_id["c001"]["verdict"] == "blocked_damage"
    assert "v2_damage_gate_disabled" in by_id["c001"]["damage_score"]["blocker_reasons"]
    assert by_id["c002"]["verdict"] == "blocked_unknown"
    assert "v2_voice_presence_gate_disabled" in by_id["c002"]["voice_presence_score"]["blocker_reasons"]


def test_model_split_v2_cross_stem_forensic_blocks_vocal_drum_imprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drums = cymbal_like_drums(seconds=0.5)
    source_audio = {
        "vocals": bright_voice_like_vocal(seconds=0.5) + 0.18 * drums,
        "drums": drums,
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + minus)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "delivery_pair_reconstruction": {"max_delivery_pair_residual_ratio": 2.0},
                        "target_specific_commands": [
                            {"role": "six_stems", "lane": "stem_lane", "command_template": ["fixture-six", "{window_mix}"]},
                            {"role": "vocals", "lane": "vocal_lane", "command_template": ["fixture-vocal", "{window_mix}"]},
                            {"role": "minus", "lane": "minus_lane", "command_template": ["fixture-minus", "{window_mix}"]},
                        ],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def old_detector_pass(**_kwargs: object) -> dict[str, object]:
        return {"verdict": "auto_accept", "attempted_checks": 1, "blocker_reasons": [], "leak_results": []}

    monkeypatch.setattr(stem_search, "_score_candidate_contamination", old_detector_pass)

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        if command[0] == "fixture-six":
            for stem, audio in source_audio.items():
                sf.write(stems_dir / f"{stem}.flac", audio[: len(window_audio)], mix_rate)
        elif command[0] == "fixture-vocal":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][: len(window_audio)], mix_rate)
        elif command[0] == "fixture-minus":
            sf.write(stems_dir / "minus.flac", minus[: len(window_audio)], mix_rate)
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
    assert candidate["verdict"] == "blocked_leak"
    assert candidate["delivery_allowed"] is False
    assert "vocal_contains_drum_transient_imprint" in candidate["cross_stem_forensic_score"]["blocker_reasons"]
    assert candidate["cross_stem_forensic_score"]["verdict"] == "fail"
    assert candidate["contamination_score"]["verdict"] == "auto_accept"
    assert result["accepted_window_candidates"] == {}
    assert not (tmp_path / "scratch" / "t001" / "listen").exists()


def test_listen_pack_forensic_failure_deletes_listen_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_audio = {
        "vocals": bright_voice_like_vocal(seconds=0.5),
        "drums": cymbal_like_drums(seconds=0.5),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + minus)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "delivery_pair_reconstruction": {"max_delivery_pair_residual_ratio": 2.0},
                        "target_specific_commands": [
                            {"role": "six_stems", "lane": "stem_lane", "command_template": ["fixture-six", "{window_mix}"]},
                            {"role": "vocals", "lane": "vocal_lane", "command_template": ["fixture-vocal", "{window_mix}"]},
                            {"role": "minus", "lane": "minus_lane", "command_template": ["fixture-minus", "{window_mix}"]},
                        ],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        stem_search,
        "_score_candidate_contamination",
        lambda **_kwargs: {"verdict": "auto_accept", "attempted_checks": 1, "blocker_reasons": [], "leak_results": []},
    )
    monkeypatch.setattr(
        stem_search,
        "score_cross_stem_forensic_matrix",
        lambda **_kwargs: {"verdict": "auto_accept", "blocker_reasons": [], "pair_results": []},
    )
    monkeypatch.setattr(
        stem_search,
        "validate_listen_pack_forensic",
        lambda **_kwargs: {
            "verdict": "fail",
            "blocker_reasons": ["vocal_contains_drum_transient_imprint"],
            "pair_results": [],
        },
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        if command[0] == "fixture-six":
            for stem, audio in source_audio.items():
                sf.write(stems_dir / f"{stem}.flac", audio[: len(window_audio)], mix_rate)
        elif command[0] == "fixture-vocal":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][: len(window_audio)], mix_rate)
        elif command[0] == "fixture-minus":
            sf.write(stems_dir / "minus.flac", minus[: len(window_audio)], mix_rate)
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

    assert result["verdict"] == "blocked_unknown"
    assert result["delivery_allowed"] is False
    assert result["presentation_status"] == "blocked_from_listen_pack"
    assert result["failed_checks"] == ["listen_pack_forensic_failed"]
    assert result["listen_pack"] == {}
    assert result["listen_pack_forensic_score"]["blocker_reasons"] == ["vocal_contains_drum_transient_imprint"]
    assert not (tmp_path / "scratch" / "t001" / "listen").exists()


def test_listen_pack_forensic_uses_accepted_candidate_gate_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_audio = {
        "vocals": bright_voice_like_vocal(seconds=0.5),
        "drums": cymbal_like_drums(seconds=0.5),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + minus)
    candidate_forensic_gate = {
        "probe_gains": [1.0, 3.0],
        "vocal_drum_score_threshold": 0.91,
    }
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "delivery_pair_reconstruction": {"max_delivery_pair_residual_ratio": 2.0},
                        "cross_stem_forensic_gate": candidate_forensic_gate,
                        "target_specific_commands": [
                            {"role": "six_stems", "lane": "stem_lane", "command_template": ["fixture-six", "{window_mix}"]},
                            {"role": "vocals", "lane": "vocal_lane", "command_template": ["fixture-vocal", "{window_mix}"]},
                            {"role": "minus", "lane": "minus_lane", "command_template": ["fixture-minus", "{window_mix}"]},
                        ],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        stem_search,
        "_score_candidate_contamination",
        lambda **_kwargs: {"verdict": "auto_accept", "attempted_checks": 1, "blocker_reasons": [], "leak_results": []},
    )
    candidate_configs: list[dict] = []
    listen_configs: list[dict] = []

    def candidate_forensic_pass(**kwargs: object) -> dict[str, object]:
        candidate_configs.append(dict(kwargs.get("config") or {}))
        return {"verdict": "auto_accept", "blocker_reasons": [], "pair_results": []}

    def listen_forensic_pass(**kwargs: object) -> dict[str, object]:
        listen_configs.append(dict(kwargs.get("config") or {}))
        return {"verdict": "auto_accept", "blocker_reasons": [], "pair_results": []}

    monkeypatch.setattr(stem_search, "score_cross_stem_forensic_matrix", candidate_forensic_pass)
    monkeypatch.setattr(stem_search, "validate_listen_pack_forensic", listen_forensic_pass)

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        if command[0] == "fixture-six":
            for stem, audio in source_audio.items():
                sf.write(stems_dir / f"{stem}.flac", audio[: len(window_audio)], mix_rate)
        elif command[0] == "fixture-vocal":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][: len(window_audio)], mix_rate)
        elif command[0] == "fixture-minus":
            sf.write(stems_dir / "minus.flac", minus[: len(window_audio)], mix_rate)
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

    assert result["verdict"] == "auto_accept"
    assert result["delivery_allowed"] is True
    assert candidate_configs == [candidate_forensic_gate]
    assert listen_configs == [candidate_forensic_gate]


def test_model_split_v2_builds_composite_from_independent_lane_winners(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.06),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + minus)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "target_specific_commands": [
                            {
                                "role": "six_stems",
                                "lane": "stem_lane",
                                "route_id": "stem_good",
                                "command_template": ["fixture-six", "{window_mix}", "{output_dir}"],
                            },
                            {
                                "role": "vocals",
                                "lane": "vocal_lane",
                                "route_id": "vocal_good",
                                "command_template": ["fixture-vocal-good", "{window_mix}"],
                            },
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "route_id": "minus_bad",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus-bad", "{window_mix}"],
                            },
                        ],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    },
                    {
                        "candidate_id": "c002",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "target_specific_commands": [
                            {
                                "role": "six_stems",
                                "lane": "stem_lane",
                                "route_id": "stem_good",
                                "command_template": ["fixture-six", "{window_mix}", "{output_dir}"],
                            },
                            {
                                "role": "vocals",
                                "lane": "vocal_lane",
                                "route_id": "vocal_bad",
                                "command_template": ["fixture-vocal-bad", "{window_mix}"],
                            },
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "route_id": "minus_good",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus-good", "{window_mix}"],
                            },
                        ],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        if command[0] == "fixture-six":
            for stem, audio in source_audio.items():
                sf.write(stems_dir / f"{stem}.flac", audio[: len(window_audio)], mix_rate)
        elif command[0] == "fixture-vocal-good":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][: len(window_audio)], mix_rate)
        elif command[0] == "fixture-vocal-bad":
            sf.write(stems_dir / "vocals.flac", (source_audio["vocals"] * np.float32(0.5))[: len(window_audio)], mix_rate)
        elif command[0] == "fixture-minus-good":
            sf.write(stems_dir / "instrumental.flac", minus[: len(window_audio)], mix_rate)
        elif command[0] == "fixture-minus-bad":
            sf.write(stems_dir / "instrumental.flac", (minus * np.float32(0.25))[: len(window_audio)], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=2,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    assert result["verdict"] == "auto_accept"
    assert result["accepted_candidate_id"].startswith("c10")
    accepted = next(candidate for candidate in result["candidates"] if candidate["candidate_id"] == result["accepted_candidate_id"])
    assert accepted["model_split_composite"] is True
    lanes = accepted["target_specific_composition"]["routes"]
    assert next(route for route in lanes if route["role"] == "vocals")["source_candidate_id"] == "c001"
    assert next(route for route in lanes if route["role"] == "minus")["source_candidate_id"] == "c002"
    reused_stem_route = next(
        route for route in result["candidates"][1]["target_specific_composition"]["routes"] if route["role"] == "six_stems"
    )
    assert reused_stem_route["reused_route"] is True
    assert reused_stem_route["reused_from_candidate_id"] == "c001"
    assert accepted["reconstruction"]["delivery_pair"]["ok"] is True
    assert result["listen_pack"]["lane_provenance"]["vocals"]["route_id"] == "vocal_good"
    assert result["listen_pack"]["lane_provenance"]["minus"]["route_id"] == "minus_good"


def test_model_split_v2_composite_can_use_independent_non_vocal_support_lane(tmp_path: Path) -> None:
    def wide(mono: np.ndarray) -> np.ndarray:
        return np.stack([mono, mono * np.float32(0.2)], axis=1).astype(np.float32)

    def mono_stereo(mono: np.ndarray) -> np.ndarray:
        return np.stack([mono, mono], axis=1).astype(np.float32)

    source_audio = {
        "vocals": mono_stereo(sine(880.0, seconds=0.5, amp=0.12)),
        "drums": wide(sine(110.0, seconds=0.5, amp=0.06)),
        "bass": wide(sine(80.0, seconds=0.5, amp=0.05)),
        "guitar": wide(sine(330.0, seconds=0.5, amp=0.04)),
        "piano": wide(sine(440.0, seconds=0.5, amp=0.03)),
        "other": wide(sine(550.0, seconds=0.5, amp=0.02)),
    }
    bad_bass = mono_stereo(sine(80.0, seconds=0.5, amp=0.05))
    bad_stems = dict(source_audio)
    bad_stems["bass"] = bad_bass
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + minus)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "model_split_composite_max_candidates": 8,
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "damage_gates": {"enabled": True, "min_relative_stereo_width": 0.5},
                        "target_specific_commands": [
                            {
                                "role": "six_stems",
                                "lane": "stem_lane",
                                "route_id": "stem_bad_bass",
                                "command_template": ["fixture-six-bad-bass", "{window_mix}"],
                            },
                            {
                                "role": "vocals",
                                "lane": "vocal_lane",
                                "route_id": "vocal_good",
                                "command_template": ["fixture-vocal-good", "{window_mix}"],
                            },
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "route_id": "minus_bad",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus-bad", "{window_mix}"],
                            },
                            {
                                "role": "non_vocal_support",
                                "lane": "non_vocal_support_lane",
                                "route_id": "scnet_xl_ihf_musdb18:drums_bass_other",
                                "command_template": ["fixture-non-vocal-support", "{window_mix}"],
                            },
                        ],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    },
                    {
                        "candidate_id": "c002",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "damage_gates": {"enabled": True, "min_relative_stereo_width": 0.5},
                        "target_specific_commands": [
                            {
                                "role": "six_stems",
                                "lane": "stem_lane",
                                "route_id": "stem_bad_bass",
                                "command_template": ["fixture-six-bad-bass", "{window_mix}"],
                            },
                            {
                                "role": "vocals",
                                "lane": "vocal_lane",
                                "route_id": "vocal_bad",
                                "command_template": ["fixture-vocal-bad", "{window_mix}"],
                            },
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "route_id": "minus_good",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus-good", "{window_mix}"],
                            },
                        ],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=True)
        stems_dir = fake_runner.stems_dir
        frames = len(window_audio)
        if command[0] == "fixture-six-bad-bass":
            for stem, audio in bad_stems.items():
                sf.write(stems_dir / f"{stem}.flac", audio[:frames], mix_rate)
        elif command[0] == "fixture-vocal-good":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][:frames], mix_rate)
        elif command[0] == "fixture-vocal-bad":
            sf.write(stems_dir / "vocals.flac", np.zeros_like(source_audio["vocals"][:frames]), mix_rate)
        elif command[0] == "fixture-minus-good":
            sf.write(stems_dir / "instrumental.flac", minus[:frames], mix_rate)
        elif command[0] == "fixture-minus-bad":
            sf.write(stems_dir / "instrumental.flac", np.zeros_like(minus[:frames]), mix_rate)
        elif command[0] == "fixture-non-vocal-support":
            for stem in ("drums", "bass", "other"):
                sf.write(stems_dir / f"{stem}.flac", source_audio[stem][:frames], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=2,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    assert result["verdict"] == "auto_accept"
    accepted = next(candidate for candidate in result["candidates"] if candidate["candidate_id"] == result["accepted_candidate_id"])
    assert accepted["bass_support_applied"] is True
    assert accepted["stem_support_applied"] is True
    assert accepted["non_vocal_support_applied"] is True
    assert accepted["target_specific_composition"]["bass_support_applied"] is True
    assert accepted["target_specific_composition"]["non_vocal_support_applied"] is True
    routes = accepted["target_specific_composition"]["routes"]
    assert next(route for route in routes if route["role"] == "non_vocal_support")["lane"] == "non_vocal_support_lane"
    assert accepted["damage_score"]["verdict"] == "auto_accept"
    assert "stereo_width_collapse:bass" not in accepted["damage_score"]["blocker_reasons"]


def test_model_split_v2_composite_projection_repair_can_clean_vocal_leak(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.06),
        "drums": sine(110.0, seconds=0.5, amp=0.07),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    drum_leak = source_audio["vocals"] * np.float32(0.04)
    other_leak = source_audio["vocals"] * np.float32(0.03)
    vocal_lane = source_audio["vocals"]
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + minus)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "model_split_composite_max_candidates": 2,
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "model_split_composite_repair_options": [
                            {"repair_id": "none", "type": "none"},
                            {
                                "repair_id": "vp100",
                                "type": "vocal_projection_debleed",
                                "strength": 1.0,
                                "target_stems": ["drums", "other"],
                            },
                        ],
                        "target_specific_commands": [
                            {
                                "role": "six_stems",
                                "lane": "stem_lane",
                                "route_id": "stem_dirty_vocal_bleed",
                                "command_template": ["fixture-six-dirty-vocal-bleed", "{window_mix}"],
                            },
                            {
                                "role": "vocals",
                                "lane": "vocal_lane",
                                "route_id": "vocal_good",
                                "command_template": ["fixture-vocal-good", "{window_mix}"],
                            },
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "route_id": "minus_good",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus-good", "{window_mix}"],
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
        frames = len(window_audio)
        if command[0] == "fixture-six-dirty-vocal-bleed":
            sf.write(stems_dir / "vocals.flac", vocal_lane[:frames], mix_rate)
            sf.write(stems_dir / "drums.flac", (source_audio["drums"] + drum_leak)[:frames], mix_rate)
            sf.write(stems_dir / "bass.flac", source_audio["bass"][:frames], mix_rate)
            sf.write(stems_dir / "guitar.flac", source_audio["guitar"][:frames], mix_rate)
            sf.write(stems_dir / "piano.flac", source_audio["piano"][:frames], mix_rate)
            sf.write(stems_dir / "other.flac", (source_audio["other"] + other_leak)[:frames], mix_rate)
        elif command[0] == "fixture-vocal-good":
            sf.write(stems_dir / "vocals.flac", vocal_lane[:frames], mix_rate)
        elif command[0] == "fixture-minus-good":
            sf.write(stems_dir / "instrumental.flac", minus[:frames], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=1,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    assert result["verdict"] == "auto_accept"
    rejected_none = next(
        candidate
        for candidate in result["candidates"]
        if candidate.get("model_split_composite") and candidate.get("model_split_repair_id") == "none"
    )
    accepted = next(candidate for candidate in result["candidates"] if candidate["candidate_id"] == result["accepted_candidate_id"])
    assert rejected_none["verdict"] == "blocked_leak"
    assert accepted["model_split_repair_id"] == "vp100"
    assert accepted["model_split_repair_type"] == "vocal_projection_debleed"
    assert accepted["vocal_projection_debleed"]["applied"] is True
    assert accepted["contamination_score"]["verdict"] == "auto_accept"
    assert accepted["reconstruction"]["delivery_pair"]["ok"] is True


def test_model_split_v2_composite_bass_stereo_repair_can_clear_damage_gate(tmp_path: Path) -> None:
    def stereo_pair(mono: np.ndarray, right_gain: float) -> np.ndarray:
        return np.stack([mono, mono * np.float32(right_gain)], axis=1).astype(np.float32)

    source_audio = {
        "vocals": stereo_pair(sine(880.0, seconds=0.5, amp=0.03), 0.6),
        "drums": stereo_pair(sine(110.0, seconds=0.5, amp=0.07), 0.4),
        "bass": stereo_pair(sine(80.0, seconds=0.5, amp=0.06), -0.7),
        "guitar": stereo_pair(sine(330.0, seconds=0.5, amp=0.04), 0.3),
        "piano": stereo_pair(sine(440.0, seconds=0.5, amp=0.03), 0.5),
        "other": stereo_pair(sine(550.0, seconds=0.5, amp=0.02), 0.7),
    }
    damaged_bass = stereo_pair(sine(80.0, seconds=0.5, amp=0.06), 0.8)
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + minus)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "model_split_composite_max_candidates": 2,
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "damage_gates": {"enabled": True},
                        "model_split_composite_repair_options": [
                            {"repair_id": "none", "type": "none"},
                            {"repair_id": "bs400", "type": "bass_stereo_width", "side_gain": 4.0},
                        ],
                        "target_specific_commands": [
                            {
                                "role": "six_stems",
                                "lane": "stem_lane",
                                "route_id": "stem_bass_narrow",
                                "command_template": ["fixture-six-bass-narrow", "{window_mix}"],
                            },
                            {
                                "role": "vocals",
                                "lane": "vocal_lane",
                                "route_id": "vocal_good",
                                "command_template": ["fixture-vocal-good", "{window_mix}"],
                            },
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "route_id": "minus_good",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus-good", "{window_mix}"],
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
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=True)
        stems_dir = fake_runner.stems_dir
        frames = len(window_audio)
        if command[0] == "fixture-six-bass-narrow":
            for stem, audio in source_audio.items():
                sf.write(stems_dir / f"{stem}.flac", (damaged_bass if stem == "bass" else audio)[:frames], mix_rate)
        elif command[0] == "fixture-vocal-good":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][:frames], mix_rate)
        elif command[0] == "fixture-minus-good":
            sf.write(stems_dir / "instrumental.flac", minus[:frames], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=1,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    assert result["verdict"] == "auto_accept"
    rejected_none = next(
        candidate
        for candidate in result["candidates"]
        if candidate.get("model_split_composite") and candidate.get("model_split_repair_id") == "none"
    )
    accepted = next(candidate for candidate in result["candidates"] if candidate["candidate_id"] == result["accepted_candidate_id"])
    assert rejected_none["verdict"] == "blocked_damage"
    assert "stereo_width_collapse:bass" in rejected_none["damage_score"]["blocker_reasons"]
    assert accepted["model_split_repair_id"] == "bs400"
    assert accepted["bass_stereo_width_repair"]["applied"] is True
    assert accepted["bass_stereo_width_repair"]["after_width"] > accepted["bass_stereo_width_repair"]["before_width"]
    assert accepted["damage_score"]["verdict"] == "auto_accept"


def test_model_split_v2_composite_learned_stem_region_repair_can_clear_damage_gate(tmp_path: Path) -> None:
    def stereo_pair(mono: np.ndarray, side: np.ndarray) -> np.ndarray:
        return np.stack([mono + side, mono - side], axis=1).astype(np.float32)

    bass_mid = sine(80.0, seconds=0.5, amp=0.06)
    bass_side = sine(113.0, seconds=0.5, amp=0.025)
    repaired_target_bass = stereo_pair(bass_mid, bass_side)
    damaged_bass = stereo_pair(bass_mid, bass_side * np.float32(0.02))
    source_audio = {
        "vocals": stereo_pair(sine(880.0, seconds=0.5, amp=0.03), sine(910.0, seconds=0.5, amp=0.004)),
        "drums": stereo_pair(sine(110.0, seconds=0.5, amp=0.07), sine(150.0, seconds=0.5, amp=0.01)),
        "bass": repaired_target_bass,
        "guitar": stereo_pair(sine(330.0, seconds=0.5, amp=0.04), sine(350.0, seconds=0.5, amp=0.008)),
        "piano": stereo_pair(sine(440.0, seconds=0.5, amp=0.03), sine(460.0, seconds=0.5, amp=0.008)),
        "other": stereo_pair(sine(550.0, seconds=0.5, amp=0.02), sine(570.0, seconds=0.5, amp=0.006)),
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix_audio = source_audio["vocals"] + minus
    mix = write_wav(tmp_path / "mix.wav", mix_audio)
    mask = make_region_mask(
        frame_count=damaged_bass.shape[0],
        sample_rate=SAMPLE_RATE,
        start_sec=0.0,
        end_sec=damaged_bass.shape[0] / SAMPLE_RATE,
    )
    model, history = train_region_repair_model_from_arrays(
        mix_audio=mix_audio,
        bad_stem_audio=damaged_bass,
        target_stem_audio=repaired_target_bass,
        region_mask=mask,
        sample_rate=SAMPLE_RATE,
        issue_type=ISSUE_STEREO_COLLAPSE,
        config=StemRegionRepairTrainConfig(epochs=80, hidden_channels=16, learning_rate=0.01, device="cpu"),
    )
    checkpoint = save_region_repair_checkpoint(model=model, path=tmp_path / "model" / "model.pt", training_history=history)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "model_split_composite_max_candidates": 2,
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "damage_gates": {"enabled": True},
                        "model_split_composite_repair_options": [
                            {"repair_id": "none", "type": "none"},
                            {
                                "repair_id": "sr001",
                                "type": "stem_region_repair",
                                "stem": "bass",
                                "issue_type": ISSUE_STEREO_COLLAPSE,
                                "checkpoint": str(checkpoint),
                            },
                        ],
                        "target_specific_commands": [
                            {
                                "role": "six_stems",
                                "lane": "stem_lane",
                                "route_id": "stem_bass_collapsed",
                                "command_template": ["fixture-six-bass-collapsed", "{window_mix}"],
                            },
                            {
                                "role": "vocals",
                                "lane": "vocal_lane",
                                "route_id": "vocal_good",
                                "command_template": ["fixture-vocal-good", "{window_mix}"],
                            },
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "route_id": "minus_good",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus-good", "{window_mix}"],
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
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=True)
        stems_dir = fake_runner.stems_dir
        frames = len(window_audio)
        if command[0] == "fixture-six-bass-collapsed":
            for stem, audio in source_audio.items():
                sf.write(stems_dir / f"{stem}.flac", (damaged_bass if stem == "bass" else audio)[:frames], mix_rate)
        elif command[0] == "fixture-vocal-good":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][:frames], mix_rate)
        elif command[0] == "fixture-minus-good":
            sf.write(stems_dir / "instrumental.flac", minus[:frames], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=1,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    assert result["verdict"] == "auto_accept"
    accepted = next(candidate for candidate in result["candidates"] if candidate["candidate_id"] == result["accepted_candidate_id"])
    assert accepted["model_split_repair_id"] == "sr001"
    assert accepted["stem_region_repair"]["applied"] is True
    assert accepted["stem_region_repair"]["diagnostics"]["target_issue_improved"] is True
    assert accepted["damage_score"]["verdict"] == "auto_accept"


def test_model_split_v2_composite_pair_gain_repair_can_clear_reconstruction_gate(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.07),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    weak_minus = minus * np.float32(0.93)
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + minus)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "model_split_composite_max_candidates": 2,
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "model_split_composite_repair_options": [
                            {"repair_id": "none", "type": "none"},
                            {
                                "repair_id": "pg08",
                                "type": "delivery_pair_gain",
                                "target_stem": "minus",
                                "max_abs_gain_delta": 0.08,
                            },
                        ],
                        "target_specific_commands": [
                            {
                                "role": "six_stems",
                                "lane": "stem_lane",
                                "route_id": "stem_good",
                                "command_template": ["fixture-six-good", "{window_mix}"],
                            },
                            {
                                "role": "vocals",
                                "lane": "vocal_lane",
                                "route_id": "vocal_good",
                                "command_template": ["fixture-vocal-good", "{window_mix}"],
                            },
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "route_id": "minus_weak",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus-weak", "{window_mix}"],
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
        frames = len(window_audio)
        if command[0] == "fixture-six-good":
            for stem, audio in source_audio.items():
                sf.write(stems_dir / f"{stem}.flac", audio[:frames], mix_rate)
        elif command[0] == "fixture-vocal-good":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][:frames], mix_rate)
        elif command[0] == "fixture-minus-weak":
            sf.write(stems_dir / "instrumental.flac", weak_minus[:frames], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=1,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    assert result["verdict"] == "auto_accept"
    rejected_none = next(
        candidate
        for candidate in result["candidates"]
        if candidate.get("model_split_composite") and candidate.get("model_split_repair_id") == "none"
    )
    accepted = next(candidate for candidate in result["candidates"] if candidate["candidate_id"] == result["accepted_candidate_id"])
    assert rejected_none["verdict"] == "blocked_bad_reconstruction"
    assert accepted["model_split_repair_id"] == "pg08"
    assert accepted["delivery_pair_gain_repair"]["applied"] is True
    assert accepted["delivery_pair_gain_repair"]["after_residual_ratio"] < accepted["delivery_pair_gain_repair"]["before_residual_ratio"]
    assert accepted["reconstruction"]["delivery_pair"]["ok"] is True


def test_model_split_v2_composite_pair_residual_repair_can_clear_reconstruction_gate(tmp_path: Path) -> None:
    residual = sine(760.0, seconds=0.5, amp=0.008)
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.07),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02) + residual,
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    missing_residual_minus = minus - residual
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + minus)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "model_split_composite_max_candidates": 2,
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "model_split_composite_repair_options": [
                            {"repair_id": "none", "type": "none"},
                            {
                                "repair_id": "pr08",
                                "type": "delivery_pair_residual",
                                "target_stem": "minus",
                                "max_residual_ratio": 0.08,
                            },
                        ],
                        "target_specific_commands": [
                            {
                                "role": "six_stems",
                                "lane": "stem_lane",
                                "route_id": "stem_good",
                                "command_template": ["fixture-six-good", "{window_mix}"],
                            },
                            {
                                "role": "vocals",
                                "lane": "vocal_lane",
                                "route_id": "vocal_good",
                                "command_template": ["fixture-vocal-good", "{window_mix}"],
                            },
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "route_id": "minus_missing_residual",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus-missing-residual", "{window_mix}"],
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
        frames = len(window_audio)
        if command[0] == "fixture-six-good":
            for stem, audio in source_audio.items():
                sf.write(stems_dir / f"{stem}.flac", audio[:frames], mix_rate)
        elif command[0] == "fixture-vocal-good":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][:frames], mix_rate)
        elif command[0] == "fixture-minus-missing-residual":
            sf.write(stems_dir / "instrumental.flac", missing_residual_minus[:frames], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=1,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    assert result["verdict"] == "auto_accept"
    rejected_none = next(
        candidate
        for candidate in result["candidates"]
        if candidate.get("model_split_composite") and candidate.get("model_split_repair_id") == "none"
    )
    accepted = next(candidate for candidate in result["candidates"] if candidate["candidate_id"] == result["accepted_candidate_id"])
    assert rejected_none["verdict"] == "blocked_bad_reconstruction"
    assert accepted["model_split_repair_id"] == "pr08"
    assert accepted["delivery_pair_residual_repair"]["applied"] is True
    assert accepted["delivery_pair_residual_repair"]["after_residual_ratio"] < accepted["delivery_pair_residual_repair"]["before_residual_ratio"]
    assert accepted["reconstruction"]["delivery_pair"]["ok"] is True


def test_model_split_v2_composite_builder_reports_inner_progress(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.07),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    stem_dir = tmp_path / "routes" / "stems"
    vocal_dir = tmp_path / "routes" / "vocal"
    minus_dir = tmp_path / "routes" / "minus"
    for stem, audio in source_audio.items():
        write_wav(stem_dir / f"{stem}.flac", audio)
    write_wav(vocal_dir / "vocals.flac", source_audio["vocals"])
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    write_wav(minus_dir / "minus.flac", minus)
    window_mix = write_wav(tmp_path / "w000.wav", source_audio["vocals"] + minus)
    window_candidates = [
        {
            "target_specific_composition": {
                "routes": [
                    {
                        "role": "six_stems",
                        "lane": "stem_lane",
                        "route_id": "stem_fixture",
                        "copied_outputs": [
                            {"stem": stem, "source_path": str(stem_dir / f"{stem}.flac")}
                            for stem in ("drums", "bass", "guitar", "piano", "other")
                        ],
                    },
                    {
                        "role": "vocals",
                        "lane": "vocal_lane",
                        "route_id": "vocal_fixture",
                        "copied_outputs": [{"stem": "vocals", "source_path": str(vocal_dir / "vocals.flac")}],
                    },
                    {
                        "role": "minus",
                        "lane": "minus_lane",
                        "route_id": "minus_fixture",
                        "copied_outputs": [{"stem": "minus", "source_path": str(minus_dir / "minus.flac")}],
                    },
                ]
            }
        }
    ]
    events: list[dict] = []

    result = stem_search.build_model_split_v2_composite_candidates(
        window={
            "window_id": "w000",
            "mix_path": str(window_mix),
            "sample_rate": SAMPLE_RATE,
            "sample_count": len(source_audio["vocals"]),
            "channel_count": 1,
        },
        window_dir=tmp_path / "run" / "w000",
        window_candidates=window_candidates,
        candidate_template={
            "profile": "model_split_v2",
            "foreign_references": write_full_foreign_references(tmp_path, source_audio),
        },
        max_composites=1,
        progress_callback=events.append,
    )

    assert result
    assert events[0]["event"] == "composite_candidate_start"
    assert events[0]["window_id"] == "w000"
    assert events[0]["candidate_id"] == "c101"
    assert events[-1]["event"] == "composite_candidate_end"
    assert events[-1]["candidate_id"] == "c101"
    assert events[-1]["verdict"] in VERDICTS
    starts = [event for event in events if event["event"] == "composite_candidate_start"]
    ends = [event for event in events if event["event"] == "composite_candidate_end"]
    assert [event["candidate_id"] for event in starts] == ["c101"]
    assert [event["candidate_id"] for event in ends] == ["c101"]
    assert len({(event["event"], event["candidate_id"]) for event in starts + ends}) == 2

    events.clear()
    limited = stem_search.build_model_split_v2_composite_candidates(
        window={"window_id": "w000", "mix_path": str(window_mix)},
        window_dir=tmp_path / "run_limit" / "w000",
        window_candidates=window_candidates,
        candidate_template={
            "profile": "model_split_v2",
            "foreign_references": write_full_foreign_references(tmp_path, source_audio),
        },
        max_composites=0,
        progress_callback=events.append,
    )

    assert limited == []
    assert events == [{"event": "composite_candidate_limit_reached", "window_id": "w000", "max_composites": 0}]


def test_model_split_v2_composite_reaches_support_routes_before_repair_exhaustion(tmp_path: Path) -> None:
    def wide(mono: np.ndarray) -> np.ndarray:
        return np.stack([mono, mono * np.float32(0.2)], axis=1).astype(np.float32)

    def mono_stereo(mono: np.ndarray) -> np.ndarray:
        return np.stack([mono, mono], axis=1).astype(np.float32)

    source_audio = {
        "vocals": mono_stereo(sine(880.0, seconds=0.5, amp=0.03)),
        "drums": wide(sine(110.0, seconds=0.5, amp=0.07)),
        "bass": wide(sine(220.0, seconds=0.5, amp=0.05)),
        "guitar": wide(sine(330.0, seconds=0.5, amp=0.04)),
        "piano": wide(sine(440.0, seconds=0.5, amp=0.03)),
        "other": wide(sine(550.0, seconds=0.5, amp=0.02)),
    }
    damaged_stem_audio = dict(source_audio)
    damaged_stem_audio["bass"] = mono_stereo(sine(220.0, seconds=0.5, amp=0.05))
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    window_mix = write_wav(tmp_path / "w000.wav", source_audio["vocals"] + minus)
    stem_dir = tmp_path / "routes" / "stems"
    vocal_dir = tmp_path / "routes" / "vocal"
    minus_dir = tmp_path / "routes" / "minus"
    bass_support_dir = tmp_path / "routes" / "bass_support"
    non_vocal_support_dir = tmp_path / "routes" / "non_vocal_support"
    for stem, audio in damaged_stem_audio.items():
        write_wav(stem_dir / f"{stem}.flac", audio)
    write_wav(vocal_dir / "vocals.flac", source_audio["vocals"])
    write_wav(minus_dir / "minus.flac", minus)
    write_wav(bass_support_dir / "bass.flac", source_audio["bass"])
    for stem in ("drums", "bass", "other"):
        write_wav(non_vocal_support_dir / f"{stem}.flac", source_audio[stem])
    window_candidates = [
        {
            "target_specific_composition": {
                "routes": [
                    {
                        "role": "six_stems",
                        "lane": "stem_lane",
                        "route_id": "stem_fixture",
                        "copied_outputs": [
                            {"stem": stem, "source_path": str(stem_dir / f"{stem}.flac")}
                            for stem in ("drums", "bass", "guitar", "piano", "other")
                        ],
                    },
                    {
                        "role": "vocals",
                        "lane": "vocal_lane",
                        "route_id": "vocal_fixture",
                        "copied_outputs": [{"stem": "vocals", "source_path": str(vocal_dir / "vocals.flac")}],
                    },
                    {
                        "role": "minus",
                        "lane": "minus_lane",
                        "route_id": "minus_fixture",
                        "copied_outputs": [{"stem": "minus", "source_path": str(minus_dir / "minus.flac")}],
                    },
                    {
                        "role": "bass_support",
                        "lane": "bass_support_lane",
                        "route_id": "bass_support_fixture",
                        "copied_outputs": [{"stem": "bass", "source_path": str(bass_support_dir / "bass.flac")}],
                    },
                    {
                        "role": "non_vocal_support",
                        "lane": "non_vocal_support_lane",
                        "route_id": "non_vocal_support_fixture",
                        "copied_outputs": [
                            {"stem": stem, "source_path": str(non_vocal_support_dir / f"{stem}.flac")}
                            for stem in ("drums", "bass", "other")
                        ],
                    },
                ]
            }
        }
    ]
    repair_options = [
        {"repair_id": "none", "type": "none"},
        {"repair_id": "pg01", "type": "delivery_pair_gain", "max_gain": 1.01},
        {"repair_id": "pg02", "type": "delivery_pair_gain", "max_gain": 1.02},
        {"repair_id": "pg03", "type": "delivery_pair_gain", "max_gain": 1.03},
        {"repair_id": "pg04", "type": "delivery_pair_gain", "max_gain": 1.04},
    ]
    events: list[dict] = []

    result = stem_search.build_model_split_v2_composite_candidates(
        window={
            "window_id": "w000",
            "mix_path": str(window_mix),
            "sample_rate": SAMPLE_RATE,
            "sample_count": len(source_audio["vocals"]),
            "channel_count": 1,
        },
        window_dir=tmp_path / "run" / "w000",
        window_candidates=window_candidates,
        candidate_template={
            "profile": "model_split_v2",
            "damage_gates": {"enabled": True, "min_relative_stereo_width": 0.5},
            "foreign_references": write_full_foreign_references(tmp_path, source_audio),
            "model_split_composite_repair_options": repair_options,
        },
        max_composites=3,
        progress_callback=events.append,
    )

    assert [candidate["model_split_repair_id"] for candidate in result] == ["none", "none", "none"]
    assert [candidate["stem_support_applied"] for candidate in result] == [False, True, True]
    assert result[1]["bass_support_applied"] is True
    assert result[2]["non_vocal_support_applied"] is True
    starts = [event for event in events if event["event"] == "composite_candidate_start"]
    assert [event["repair_id"] for event in starts] == ["none", "none", "none"]


def test_model_split_v2_composite_reaches_support_key_repairs_before_budget_exhaustion(tmp_path: Path) -> None:
    def wide(mono: np.ndarray) -> np.ndarray:
        return np.stack([mono, mono * np.float32(0.2)], axis=1).astype(np.float32)

    def mono_stereo(mono: np.ndarray) -> np.ndarray:
        return np.stack([mono, mono], axis=1).astype(np.float32)

    source_audio = {
        "vocals": mono_stereo(sine(880.0, seconds=0.5, amp=0.03)),
        "drums": wide(sine(110.0, seconds=0.5, amp=0.07)),
        "bass": wide(sine(220.0, seconds=0.5, amp=0.05)),
        "guitar": wide(sine(330.0, seconds=0.5, amp=0.04)),
        "piano": wide(sine(440.0, seconds=0.5, amp=0.03)),
        "other": wide(sine(550.0, seconds=0.5, amp=0.02)),
    }
    damaged_stem_audio = dict(source_audio)
    damaged_stem_audio["bass"] = mono_stereo(sine(220.0, seconds=0.5, amp=0.05))
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    window_mix = write_wav(tmp_path / "w000.wav", source_audio["vocals"] + minus)
    stem_dir = tmp_path / "routes" / "stems"
    vocal_dir = tmp_path / "routes" / "vocal"
    minus_dir = tmp_path / "routes" / "minus"
    bass_support_dir = tmp_path / "routes" / "bass_support"
    non_vocal_support_dir = tmp_path / "routes" / "non_vocal_support"
    for stem, audio in damaged_stem_audio.items():
        write_wav(stem_dir / f"{stem}.flac", audio)
    write_wav(vocal_dir / "vocals.flac", source_audio["vocals"])
    write_wav(minus_dir / "minus.flac", minus)
    write_wav(bass_support_dir / "bass.flac", source_audio["bass"])
    for stem in ("drums", "bass", "other"):
        write_wav(non_vocal_support_dir / f"{stem}.flac", source_audio[stem])
    window_candidates = [
        {
            "target_specific_composition": {
                "routes": [
                    {
                        "role": "six_stems",
                        "lane": "stem_lane",
                        "route_id": "stem_fixture",
                        "copied_outputs": [
                            {"stem": stem, "source_path": str(stem_dir / f"{stem}.flac")}
                            for stem in ("drums", "bass", "guitar", "piano", "other")
                        ],
                    },
                    {
                        "role": "vocals",
                        "lane": "vocal_lane",
                        "route_id": "vocal_fixture",
                        "copied_outputs": [{"stem": "vocals", "source_path": str(vocal_dir / "vocals.flac")}],
                    },
                    {
                        "role": "minus",
                        "lane": "minus_lane",
                        "route_id": "minus_fixture",
                        "copied_outputs": [{"stem": "minus", "source_path": str(minus_dir / "minus.flac")}],
                    },
                    {
                        "role": "bass_support",
                        "lane": "bass_support_lane",
                        "route_id": "bass_support_fixture",
                        "copied_outputs": [{"stem": "bass", "source_path": str(bass_support_dir / "bass.flac")}],
                    },
                    {
                        "role": "non_vocal_support",
                        "lane": "non_vocal_support_lane",
                        "route_id": "non_vocal_support_fixture",
                        "copied_outputs": [
                            {"stem": stem, "source_path": str(non_vocal_support_dir / f"{stem}.flac")}
                            for stem in ("drums", "bass", "other")
                        ],
                    },
                ]
            }
        }
    ]
    repair_options = [
        {"repair_id": "none", "type": "none"},
        {"repair_id": "vp100", "type": "vocal_projection_debleed", "strength": 1.0},
        {"repair_id": "bs250", "type": "bass_stereo_width", "side_gain": 2.5},
        {"repair_id": "bs400", "type": "bass_stereo_width", "side_gain": 4.0},
        {"repair_id": "pg08", "type": "delivery_pair_gain", "target_stem": "minus", "max_abs_gain_delta": 0.08},
        {"repair_id": "pr16", "type": "delivery_pair_residual", "target_stem": "minus", "max_residual_ratio": 0.16},
        {
            "repair_id": "bs400_pr16",
            "type": "bass_stereo_width+delivery_pair_residual",
            "bass_stereo_width": {"side_gain": 4.0},
            "delivery_pair_residual": {"target_stem": "minus", "max_residual_ratio": 0.16},
        },
    ]
    events: list[dict] = []

    stem_search.build_model_split_v2_composite_candidates(
        window={
            "window_id": "w000",
            "mix_path": str(window_mix),
            "sample_rate": SAMPLE_RATE,
            "sample_count": len(source_audio["vocals"]),
            "channel_count": 2,
        },
        window_dir=tmp_path / "run" / "w000",
        window_candidates=window_candidates,
        candidate_template={
            "profile": "model_split_v2",
            "damage_gates": {"enabled": True, "min_relative_stereo_width": 10.0},
            "foreign_references": write_full_foreign_references(tmp_path, source_audio),
            "model_split_composite_repair_options": repair_options,
        },
        max_composites=10,
        progress_callback=events.append,
    )

    starts = [event for event in events if event["event"] == "composite_candidate_start"]
    first_ten = [
        (event["repair_id"], event["bass_support_applied"], event["non_vocal_support_applied"])
        for event in starts[:10]
    ]
    assert ("pr16", True, False) in first_ten
    assert ("pr16", True, True) in first_ten
    assert ("bs400_pr16", True, False) in first_ten
    assert ("vp100", True, False) in first_ten
    assert not any(repair_id in {"bs250", "pg08"} and not bass for repair_id, bass, _ in first_ten)


def test_model_split_v2_composite_prioritizes_low_residual_vocal_minus_pair(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.07),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = source_audio["vocals"] + minus
    window_mix = write_wav(tmp_path / "w000.wav", mix)
    stem_dir = tmp_path / "routes" / "stems"
    vocal_bad_dir = tmp_path / "routes" / "vocal_bad"
    vocal_good_dir = tmp_path / "routes" / "vocal_good"
    minus_bad_dir = tmp_path / "routes" / "minus_bad"
    minus_good_dir = tmp_path / "routes" / "minus_good"
    for stem, audio in source_audio.items():
        write_wav(stem_dir / f"{stem}.flac", audio)
    write_wav(vocal_bad_dir / "vocals.flac", np.zeros_like(source_audio["vocals"]))
    write_wav(vocal_good_dir / "vocals.flac", source_audio["vocals"])
    write_wav(minus_bad_dir / "minus.flac", np.zeros_like(minus))
    write_wav(minus_good_dir / "minus.flac", minus)
    window_candidates = [
        {
            "target_specific_composition": {
                "routes": [
                    {
                        "role": "six_stems",
                        "lane": "stem_lane",
                        "route_id": "stem_good",
                        "copied_outputs": [
                            {"stem": stem, "source_path": str(stem_dir / f"{stem}.flac")}
                            for stem in ("drums", "bass", "guitar", "piano", "other")
                        ],
                    },
                    {
                        "role": "vocals",
                        "lane": "vocal_lane",
                        "route_id": "vocal_bad_first",
                        "copied_outputs": [{"stem": "vocals", "source_path": str(vocal_bad_dir / "vocals.flac")}],
                    },
                    {
                        "role": "minus",
                        "lane": "minus_lane",
                        "route_id": "minus_bad_first",
                        "copied_outputs": [{"stem": "minus", "source_path": str(minus_bad_dir / "minus.flac")}],
                    },
                    {
                        "role": "vocals",
                        "lane": "vocal_lane",
                        "route_id": "vocal_good_later",
                        "copied_outputs": [{"stem": "vocals", "source_path": str(vocal_good_dir / "vocals.flac")}],
                    },
                    {
                        "role": "minus",
                        "lane": "minus_lane",
                        "route_id": "minus_good_later",
                        "copied_outputs": [{"stem": "minus", "source_path": str(minus_good_dir / "minus.flac")}],
                    },
                ]
            }
        }
    ]
    events: list[dict] = []

    result = stem_search.build_model_split_v2_composite_candidates(
        window={"window_id": "w000", "mix_path": str(window_mix)},
        window_dir=tmp_path / "run" / "w000",
        window_candidates=window_candidates,
        candidate_template={
            "profile": "model_split_v2",
            "foreign_references": write_full_foreign_references(tmp_path, source_audio),
        },
        max_composites=1,
        progress_callback=events.append,
    )

    assert len(result) == 1
    candidate = result[0]
    routes = candidate["target_specific_composition"]["routes"]
    assert candidate["verdict"] == "auto_accept"
    assert next(route for route in routes if route["role"] == "vocals")["route_id"] == "vocal_good_later"
    assert next(route for route in routes if route["role"] == "minus")["route_id"] == "minus_good_later"
    assert candidate["model_split_pair_residual_ratio"] == pytest.approx(0.0, abs=1e-3)
    assert events[0]["pair_residual_ratio"] == pytest.approx(0.0, abs=1e-3)


def test_model_split_v2_composite_keeps_distinct_audio_for_same_route_id(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.07),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = source_audio["vocals"] + minus
    window_mix = write_wav(tmp_path / "w000.wav", mix)
    stem_dir = tmp_path / "routes" / "stems"
    vocal_bad_dir = tmp_path / "routes" / "vocal_bad"
    vocal_good_dir = tmp_path / "routes" / "vocal_good"
    minus_bad_dir = tmp_path / "routes" / "minus_bad"
    minus_good_dir = tmp_path / "routes" / "minus_good"
    for stem, audio in source_audio.items():
        write_wav(stem_dir / f"{stem}.flac", audio)
    write_wav(vocal_bad_dir / "vocals.flac", np.zeros_like(source_audio["vocals"]))
    write_wav(vocal_good_dir / "vocals.flac", source_audio["vocals"])
    write_wav(minus_bad_dir / "minus.flac", np.zeros_like(minus))
    write_wav(minus_good_dir / "minus.flac", minus)
    window_candidates = [
        {
            "target_specific_composition": {
                "routes": [
                    {
                        "role": "six_stems",
                        "lane": "stem_lane",
                        "route_id": "stem_good",
                        "copied_outputs": [
                            {"stem": stem, "source_path": str(stem_dir / f"{stem}.flac")}
                            for stem in ("drums", "bass", "guitar", "piano", "other")
                        ],
                    },
                    {
                        "role": "vocals",
                        "lane": "vocal_lane",
                        "route_id": "shared_vocal_model",
                        "source_candidate_id": "bad_first",
                        "copied_outputs": [{"stem": "vocals", "source_path": str(vocal_bad_dir / "vocals.flac")}],
                    },
                    {
                        "role": "minus",
                        "lane": "minus_lane",
                        "route_id": "shared_minus_model",
                        "source_candidate_id": "bad_first",
                        "copied_outputs": [{"stem": "minus", "source_path": str(minus_bad_dir / "minus.flac")}],
                    },
                    {
                        "role": "vocals",
                        "lane": "vocal_lane",
                        "route_id": "shared_vocal_model",
                        "source_candidate_id": "good_later",
                        "copied_outputs": [{"stem": "vocals", "source_path": str(vocal_good_dir / "vocals.flac")}],
                    },
                    {
                        "role": "minus",
                        "lane": "minus_lane",
                        "route_id": "shared_minus_model",
                        "source_candidate_id": "good_later",
                        "copied_outputs": [{"stem": "minus", "source_path": str(minus_good_dir / "minus.flac")}],
                    },
                ]
            }
        }
    ]

    result = stem_search.build_model_split_v2_composite_candidates(
        window={"window_id": "w000", "mix_path": str(window_mix)},
        window_dir=tmp_path / "run" / "w000",
        window_candidates=window_candidates,
        candidate_template={
            "profile": "model_split_v2",
            "foreign_references": write_full_foreign_references(tmp_path, source_audio),
        },
        max_composites=1,
    )

    assert len(result) == 1
    candidate = result[0]
    routes = candidate["target_specific_composition"]["routes"]
    assert candidate["verdict"] == "auto_accept"
    assert next(route for route in routes if route["role"] == "vocals")["source_candidate_id"] == "good_later"
    assert next(route for route in routes if route["role"] == "minus")["source_candidate_id"] == "good_later"
    assert candidate["model_split_pair_residual_ratio"] == pytest.approx(0.0, abs=1e-3)


def test_model_split_v2_composite_pair_ranking_uses_aligned_output_paths(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.07),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = source_audio["vocals"] + minus
    window_mix = write_wav(tmp_path / "w000.wav", mix)
    stem_dir = tmp_path / "routes" / "stems"
    bad_dir = tmp_path / "routes" / "bad"
    good_raw_dir = tmp_path / "routes" / "good_raw"
    good_out_dir = tmp_path / "routes" / "good_out"
    for stem, audio in source_audio.items():
        write_wav(stem_dir / f"{stem}.flac", audio)
    write_wav(bad_dir / "vocals.flac", np.zeros_like(source_audio["vocals"]))
    write_wav(bad_dir / "minus.flac", np.zeros_like(minus))
    write_wav(good_raw_dir / "vocals.flac", np.concatenate([source_audio["vocals"], np.zeros(1000, dtype=np.float32)]))
    write_wav(good_raw_dir / "minus.flac", np.concatenate([minus, np.zeros(1000, dtype=np.float32)]))
    write_wav(good_out_dir / "vocals.flac", source_audio["vocals"])
    write_wav(good_out_dir / "minus.flac", minus)
    window_candidates = [
        {
            "target_specific_composition": {
                "routes": [
                    {
                        "role": "six_stems",
                        "lane": "stem_lane",
                        "route_id": "stem_good",
                        "copied_outputs": [
                            {"stem": stem, "source_path": str(stem_dir / f"{stem}.flac")}
                            for stem in ("drums", "bass", "guitar", "piano", "other")
                        ],
                    },
                    {
                        "role": "vocals",
                        "lane": "vocal_lane",
                        "route_id": "bad_vocal_first",
                        "copied_outputs": [{"stem": "vocals", "source_path": str(bad_dir / "vocals.flac")}],
                    },
                    {
                        "role": "minus",
                        "lane": "minus_lane",
                        "route_id": "bad_minus_first",
                        "copied_outputs": [{"stem": "minus", "source_path": str(bad_dir / "minus.flac")}],
                    },
                    {
                        "role": "vocals",
                        "lane": "vocal_lane",
                        "route_id": "good_vocal_later",
                        "source_candidate_id": "good_aligned",
                        "copied_outputs": [
                            {
                                "stem": "vocals",
                                "source_path": str(good_raw_dir / "vocals.flac"),
                                "output_path": str(good_out_dir / "vocals.flac"),
                            }
                        ],
                    },
                    {
                        "role": "minus",
                        "lane": "minus_lane",
                        "route_id": "good_minus_later",
                        "source_candidate_id": "good_aligned",
                        "copied_outputs": [
                            {
                                "stem": "minus",
                                "source_path": str(good_raw_dir / "minus.flac"),
                                "output_path": str(good_out_dir / "minus.flac"),
                            }
                        ],
                    },
                ]
            }
        }
    ]

    result = stem_search.build_model_split_v2_composite_candidates(
        window={
            "window_id": "w000",
            "mix_path": str(window_mix),
            "sample_rate": SAMPLE_RATE,
            "sample_count": len(mix),
            "channel_count": 1,
        },
        window_dir=tmp_path / "run" / "w000",
        window_candidates=window_candidates,
        candidate_template={
            "profile": "model_split_v2",
            "foreign_references": write_full_foreign_references(tmp_path, source_audio),
        },
        max_composites=1,
    )

    assert len(result) == 1
    candidate = result[0]
    routes = candidate["target_specific_composition"]["routes"]
    assert candidate["verdict"] == "auto_accept"
    assert next(route for route in routes if route["role"] == "vocals")["source_candidate_id"] == "good_aligned"
    assert next(route for route in routes if route["role"] == "minus")["source_candidate_id"] == "good_aligned"
    assert candidate["model_split_pair_residual_ratio"] == pytest.approx(0.0, abs=1e-3)


def test_model_split_v2_recipe_level_no_reference_manifest_can_accept_full_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.01),
        "drums": sine(110.0, seconds=0.5, amp=0.07),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + minus)
    model_manifest = tmp_path / "model" / "manifest.json"
    model_manifest.parent.mkdir(parents=True, exist_ok=True)
    model_manifest.write_text("{}", encoding="utf-8")
    calls: list[dict] = []

    def fake_score_stem_contamination(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        assert kwargs["foreign_reference_path"] is None
        assert kwargs["model_manifest_path"] == model_manifest
        return {
            "tested_stem": kwargs["tested_stem"],
            "foreign_source": kwargs["foreign_source"],
            "verdict": "pass",
            "leak_audio": "",
            "regions": [],
            "leak_dbfs": -90.0,
            "detector_mode": "fixture_no_reference_v1",
        }

    monkeypatch.setattr(stem_search, "score_stem_contamination", fake_score_stem_contamination)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "contamination_model_manifest": str(model_manifest),
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "target_specific_commands": [
                            {
                                "role": "six_stems",
                                "lane": "stem_lane",
                                "route_id": "stem_good",
                                "command_template": ["fixture-six-good", "{window_mix}"],
                            },
                            {
                                "role": "vocals",
                                "lane": "vocal_lane",
                                "route_id": "vocal_good",
                                "command_template": ["fixture-vocal-good", "{window_mix}"],
                            },
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "route_id": "minus_good",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus-good", "{window_mix}"],
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
        frames = len(window_audio)
        if command[0] == "fixture-six-good":
            for stem, audio in source_audio.items():
                sf.write(stems_dir / f"{stem}.flac", audio[:frames], mix_rate)
        elif command[0] == "fixture-vocal-good":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][:frames], mix_rate)
        elif command[0] == "fixture-minus-good":
            sf.write(stems_dir / "instrumental.flac", minus[:frames], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=1,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    assert result["verdict"] == "auto_accept"
    assert result["delivery_status"] == "accepted"
    assert result["delivery_allowed"] is True
    assert result["presentation_status"] == "accepted_for_listen_pack"
    assert len(calls) == sum(len(sources) for sources in FORBIDDEN_SOURCES.values())
    accepted = next(candidate for candidate in result["candidates"] if candidate["candidate_id"] == result["accepted_candidate_id"])
    assert accepted["contamination_score"]["check_mode"] == "full_forbidden_matrix"
    assert accepted["contamination_score"]["candidate_check_hints_used"] is True


def test_model_split_v2_blocks_t040_style_post_debleed_and_other_merge(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.06),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    mix = write_wav(tmp_path / "mix.wav", sum(source_audio.values()))
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "candidates": [
                    {
                        "candidate_id": "c010",
                        "profile": "model_split_v2",
                        "command_template": ["fixture-write-legacy-repair", "{window_mix}"],
                        "stem_merges": [{"source_stem": "other", "target_stem": "guitar", "mode": "move"}],
                        "post_minus_vocal_debleed": {
                            "strength": 1.0,
                            "low_hz": 700.0,
                            "high_hz": 1200.0,
                            "target_stems": ["drums"],
                            "passes": 1,
                        },
                        "damage_gates": {"enabled": True},
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
            sf.write(stems_dir / f"{stem}.flac", audio[: len(window_audio)], mix_rate)

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
    assert candidate["verdict"] == "blocked_damage"
    assert "v2_repair_route_not_publishable:stem_merges" in candidate["damage_score"]["blocker_reasons"]
    assert "v2_repair_route_not_publishable:post_minus_vocal_debleed" in candidate["damage_score"]["blocker_reasons"]
    assert result["listen_pack"] == {}


def test_model_split_v2_stereo_damage_blocks_even_when_contamination_passes(tmp_path: Path) -> None:
    vocal = sine(880.0, seconds=0.5, amp=0.02)
    left_drums = sine(110.0, seconds=0.5, amp=0.07)
    right_drums = -left_drums
    bass = sine(220.0, seconds=0.5, amp=0.05)
    guitar = sine(330.0, seconds=0.5, amp=0.04)
    piano = sine(440.0, seconds=0.5, amp=0.03)
    other = sine(550.0, seconds=0.5, amp=0.02)
    mono_non_drums = bass + guitar + piano + other
    mix = write_stereo_wav(tmp_path / "mix.wav", vocal + left_drums + mono_non_drums, vocal + right_drums + mono_non_drums)
    refs = write_full_foreign_references(
        tmp_path,
        {
            "vocals": vocal,
            "drums": left_drums,
            "bass": bass,
            "guitar": guitar,
            "piano": piano,
            "other": other,
        },
    )
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "delivery_pair_reconstruction": {"max_delivery_pair_residual_ratio": 2.0},
                        "target_specific_commands": [
                            {"role": "six_stems", "lane": "stem_lane", "command_template": ["fixture-six", "{window_mix}"]},
                            {"role": "vocals", "lane": "vocal_lane", "command_template": ["fixture-vocal", "{window_mix}"]},
                            {"role": "minus", "lane": "minus_lane", "command_template": ["fixture-minus", "{window_mix}"]},
                        ],
                        "damage_gates": {
                            "enabled": True,
                            "min_relative_stereo_width": 0.5,
                            "min_non_vocal_rms": 0.0,
                        },
                        "foreign_references": refs,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=True)
        stems_dir = fake_runner.stems_dir
        frames = len(window_audio)
        if command[0] == "fixture-six":
            sf.write(stems_dir / "vocals.flac", np.stack([vocal, vocal], axis=1)[:frames], mix_rate)
            sf.write(stems_dir / "drums.flac", np.stack([left_drums, right_drums], axis=1)[:frames], mix_rate)
            for stem, audio in (("bass", bass), ("guitar", guitar), ("piano", piano), ("other", other)):
                sf.write(stems_dir / f"{stem}.flac", np.stack([audio, audio], axis=1)[:frames], mix_rate)
        elif command[0] == "fixture-vocal":
            sf.write(stems_dir / "vocals.flac", np.stack([vocal, vocal], axis=1)[:frames], mix_rate)
        elif command[0] == "fixture-minus":
            collapsed = left_drums + mono_non_drums
            sf.write(stems_dir / "minus.flac", np.stack([collapsed, collapsed], axis=1)[:frames], mix_rate)
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
    assert candidate["verdict"] == "blocked_damage"
    assert "stereo_width_collapse:minus" in candidate["damage_score"]["blocker_reasons"]


def test_model_split_v2_bass_stereo_uses_bass_band_reference(tmp_path: Path) -> None:
    bass = sine(80.0, seconds=0.5, amp=0.06)
    wide_high = sine(2200.0, seconds=0.5, amp=0.04)
    mix_path = write_stereo_wav(tmp_path / "mix.wav", bass + wide_high, bass - wide_high)
    bass_path = write_stereo_wav(tmp_path / "bass.wav", bass, bass)
    minus_path = write_stereo_wav(tmp_path / "minus.wav", bass + wide_high, bass - wide_high)

    score = stem_search.score_model_split_v2_damage_gates(
        candidate={
            "profile": "model_split_v2",
            "minus_source": "minus_lane",
            "damage_gates": {
                "enabled": True,
                "min_relative_stereo_width": 0.5,
                "min_non_vocal_rms": 0.0,
                "min_mix_stereo_width": 0.02,
            },
        },
        mix_path=mix_path,
        stem_paths={"bass": str(bass_path), "minus": str(minus_path)},
    )

    assert score["verdict"] == "auto_accept"
    assert "stereo_width_collapse:bass" not in score["blocker_reasons"]
    assert score["metrics"]["bass"]["stereo_reference_surface"] == "bass_band_mix"
    assert score["metrics"]["mix"]["stereo_width"] > 0.02
    assert score["metrics"]["mix"]["bass_band_stereo_width"] < 0.02


def test_model_split_v2_bass_stereo_blocks_when_bass_band_collapses(tmp_path: Path) -> None:
    bass = sine(80.0, seconds=0.5, amp=0.06)
    mix_path = write_stereo_wav(tmp_path / "mix.wav", bass, bass * 0.2)
    bass_path = write_stereo_wav(tmp_path / "bass.wav", bass * 0.6, bass * 0.6)
    minus_path = write_stereo_wav(tmp_path / "minus.wav", bass, bass * 0.2)

    score = stem_search.score_model_split_v2_damage_gates(
        candidate={
            "profile": "model_split_v2",
            "minus_source": "minus_lane",
            "damage_gates": {
                "enabled": True,
                "min_relative_stereo_width": 0.5,
                "min_non_vocal_rms": 0.0,
                "min_mix_stereo_width": 0.02,
            },
        },
        mix_path=mix_path,
        stem_paths={"bass": str(bass_path), "minus": str(minus_path)},
    )

    assert score["verdict"] == "blocked_damage"
    assert "stereo_width_collapse:bass" in score["blocker_reasons"]
    assert score["metrics"]["bass"]["stereo_reference_surface"] == "bass_band_mix"


def test_model_split_v2_other_cannot_be_silently_emptied(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.06),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    mix = write_wav(tmp_path / "mix.wav", sum(source_audio.values()))
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "command_template": ["fixture-write-empty-other", "{window_mix}"],
                        "damage_gates": {"enabled": True, "min_non_vocal_rms": 1e-4},
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
            output = np.zeros_like(audio, dtype=np.float32) if stem == "other" else audio
            sf.write(stems_dir / f"{stem}.flac", output[: len(window_audio)], mix_rate)

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
    assert candidate["verdict"] == "blocked_damage"
    assert "stem_detail_floor:other" in candidate["damage_score"]["blocker_reasons"]


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


def test_minus_vocal_debleed_passes_iteratively_reduce_vocal_band(tmp_path: Path) -> None:
    vocals = sine(880.0, seconds=0.5, amp=0.03)
    drums = sine(110.0, seconds=0.5, amp=0.08)
    bass = sine(55.0, seconds=0.5, amp=0.04)
    vocal_residue = sine(900.0, seconds=0.5, amp=0.05)
    paths = {
        "vocals": str(write_wav(tmp_path / "vocals.flac", vocals)),
        "drums": str(write_wav(tmp_path / "drums.flac", drums + vocal_residue)),
        "bass": str(write_wav(tmp_path / "bass.flac", bass)),
    }
    for stem in ("guitar", "piano", "other"):
        paths[stem] = str(write_wav(tmp_path / f"{stem}.flac", np.zeros_like(vocals, dtype=np.float32)))

    result = apply_minus_vocal_debleed(
        stem_paths=paths,
        config={
            "strength": 1.0,
            "low_hz": 700.0,
            "high_hz": 1200.0,
            "target_stems": ["drums"],
            "passes": 2,
        },
    )

    cleaned_drums, _ = sf.read(paths["drums"], dtype="float32", always_2d=False)
    before_band_rms = np.sqrt(np.mean(np.square(vocal_residue.astype(np.float64))))
    after_band_input = cleaned_drums[:, None] if cleaned_drums.ndim == 1 else cleaned_drums
    after_band = band_limited_audio(after_band_input, SAMPLE_RATE, 700.0, 1200.0)
    after_band_rms = np.sqrt(np.mean(np.square(after_band.astype(np.float64))))
    assert result["applied"] is True
    assert result["passes"] == 2
    assert result["per_stem"]["drums"]["passes"] == 2
    assert after_band_rms < before_band_rms * 0.05


def test_minus_vocal_debleed_invalid_passes_blocks_fail_closed(tmp_path: Path) -> None:
    audio = sine(880.0, seconds=0.25, amp=0.03)
    paths = {
        "vocals": str(write_wav(tmp_path / "vocals.flac", audio)),
        "drums": str(write_wav(tmp_path / "drums.flac", audio)),
    }

    result = apply_minus_vocal_debleed(
        stem_paths=paths,
        config={
            "strength": 1.0,
            "low_hz": 700.0,
            "high_hz": 1200.0,
            "target_stems": ["drums"],
            "passes": 0,
        },
    )

    assert result["applied"] is False
    assert result["reason"] == "invalid_debleed_config"
    assert "invalid_debleed_passes" in result["blockers"]


def test_recipe_candidate_applies_post_debleed_before_rebuilt_minus_scoring(tmp_path: Path) -> None:
    vocal_residue = sine(900.0, seconds=0.25, amp=0.05)
    drums = sine(110.0, seconds=0.25, amp=0.08)
    source_audio = {
        "vocals": vocal_residue,
        "drums": drums,
        "bass": np.zeros_like(vocal_residue, dtype=np.float32),
        "guitar": np.zeros_like(vocal_residue, dtype=np.float32),
        "piano": np.zeros_like(vocal_residue, dtype=np.float32),
        "other": np.zeros_like(vocal_residue, dtype=np.float32),
    }
    mix = write_wav(tmp_path / "mix.wav", drums + vocal_residue)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-post-debleedable", "{window_mix}"],
                        "post_minus_vocal_debleed": {
                            "strength": 1.0,
                            "low_hz": 700.0,
                            "high_hz": 1200.0,
                            "target_stems": ["drums"],
                            "passes": 1,
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
        sf.write(stems_dir / "vocals.flac", np.zeros_like(window_audio, dtype=np.float32), mix_rate)
        sf.write(stems_dir / "drums.flac", (drums + vocal_residue)[: len(window_audio)], mix_rate)
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
    assert candidate["post_minus_vocal_debleed"]["applied"] is True
    assert candidate["post_minus_vocal_debleed"]["passes"] == 1
    assert candidate["reconstruction"]["ok"] is True
    assert candidate["contamination_score"]["verdict"] == "auto_accept"


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
