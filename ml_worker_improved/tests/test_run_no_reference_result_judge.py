from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.run_no_reference_result_judge import (
    build_summary,
    candidate_row_from_manifest,
    judge_audio,
    judge_candidate_rows,
    write_outputs,
)


SAMPLE_RATE = 16_000


def write_wav(path: Path, audio: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return str(path)


def sine(freq: float, seconds: float = 1.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def candidate_row(tmp_path: Path, *, candidate_id: str, mix: np.ndarray, vocal: np.ndarray, minus: np.ndarray) -> dict:
    return {
        "track_id": "track-a",
        "candidate_id": candidate_id,
        "mix_path": write_wav(tmp_path / candidate_id / "mix.wav", mix),
        "vocal_path": write_wav(tmp_path / candidate_id / "vocal.wav", vocal),
        "instrumental_path": write_wav(tmp_path / candidate_id / "minus.wav", minus),
    }


def test_candidate_row_from_manifest_prefers_exact_stem_paths_over_flat_fuzzy_matches(tmp_path: Path) -> None:
    manifest_path = tmp_path / "track-a" / "minus" / "becruily_instrumental_vocals" / "stem_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "source_audio_path": "/host_testing_stuff_music/track-a.mp3",
                "model_id": "becruily_instrumental_vocals",
                "stems": {
                    "vocals": {"path": "/app/out/vocals.flac"},
                    "instrumental": {"path": "/app/out/instrumental.flac", "source": "model_output"},
                    "minus": {"path": "/app/out/instrumental.flac", "source": "existing_manifest_stem"},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    row = candidate_row_from_manifest(manifest_path, 1)

    assert row["track_id"] == "track-a"
    assert row["candidate_id"] == "becruily_instrumental_vocals"
    assert row["mix_path"] == "/host_testing_stuff_music/track-a.mp3"
    assert row["vocal_path"] == "/app/out/vocals.flac"
    assert row["instrumental_path"] == "/app/out/instrumental.flac"


def test_candidate_row_from_manifest_prefers_rendered_repair_path_over_base_minus(tmp_path: Path) -> None:
    manifest_path = tmp_path / "air_detail_repair" / "judge_guided_minus_air_detail_repair_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "mode": "judge_guided_minus_air_detail_repair",
                "track_id": "track-a",
                "candidate_id": "minus-a__judge_guided_air_detail_repair",
                "raw_audio_path": "/app/raw.mp3",
                "base_minus_path": "/app/base/minus_judge_guided_air_detail_repair.flac",
                "rendered_minus_candidate_path": "/app/repaired/minus_judge_guided_air_detail_repair.flac",
                "candidate_row": {
                    "track_id": "track-a",
                    "candidate_id": "minus-a__judge_guided_air_detail_repair",
                    "mix_path": "/app/raw.mp3",
                    "vocal_path": "/app/vocal.flac",
                    "instrumental_path": "/app/repaired/minus_judge_guided_air_detail_repair.flac",
                    "minus_path": "/app/repaired/minus_judge_guided_air_detail_repair.flac",
                    "selected_minus_path": "/app/repaired/minus_judge_guided_air_detail_repair.flac",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    row = candidate_row_from_manifest(manifest_path, 1)

    assert row["track_id"] == "track-a"
    assert row["candidate_id"] == "minus-a__judge_guided_air_detail_repair"
    assert row["mix_path"] == "/app/raw.mp3"
    assert row["vocal_path"] == "/app/vocal.flac"
    assert row["instrumental_path"] == "/app/repaired/minus_judge_guided_air_detail_repair.flac"


def test_no_reference_or_stems_are_required_for_good_candidate(tmp_path: Path) -> None:
    vocal = sine(440)
    instrumental = sine(120)
    mix = vocal + instrumental
    result = judge_candidate_rows([candidate_row(tmp_path, candidate_id="good", mix=mix, vocal=vocal, minus=instrumental)])
    row = result["rows"][0]
    assert row["judge_mode"] == "no_reference"
    assert row["decision"] == "accept_candidate"
    assert row["mixture_consistency_score"] > 0.95
    assert row["confidence"] > 0.65
    assert "reference_path" not in row


def test_mixture_consistency_catches_bad_reconstruction(tmp_path: Path) -> None:
    mix = sine(440)
    zero = np.zeros_like(mix)
    result = judge_candidate_rows([candidate_row(tmp_path, candidate_id="bad", mix=mix, vocal=zero, minus=zero)])
    row = result["rows"][0]
    assert row["mixture_consistency_score"] < 0.55
    assert row["decision"] == "blocked_hard_case"
    assert "bad_mixture_reconstruction" in row["reasons"]


def test_leakage_score_blocks_vocal_heavy_minus(tmp_path: Path) -> None:
    vocal = sine(440)
    instrumental = sine(120)
    mix = vocal + instrumental
    minus = instrumental + 0.95 * vocal
    result = judge_candidate_rows([candidate_row(tmp_path, candidate_id="leaky", mix=mix, vocal=vocal, minus=minus)])
    row = result["rows"][0]
    assert row["vocal_leakage_in_minus_score"] > 0.60
    assert row["decision"] == "blocked_hard_case"
    assert "vocal_leakage_in_minus" in row["reasons"]


def test_minus_score_mode_adds_minus_matrix_fields(tmp_path: Path) -> None:
    vocal = sine(440)
    instrumental = sine(120)
    mix = vocal + instrumental

    result = judge_candidate_rows(
        [candidate_row(tmp_path, candidate_id="minus-mode", mix=mix, vocal=vocal, minus=instrumental)],
        score_mode="minus",
    )
    row = result["rows"][0]

    assert row["score_mode"] == "minus"
    assert "minus_score" in row
    assert "minus_decision" in row
    assert row["minus_blockers"] == []
    assert row["minus_musical_detail_preservation_gate"]["status"] == "proxy_pass"
    assert row["minus_musical_detail_preservation_gate"]["blockers"] == []
    assert result["summary"]["score_mode"] == "minus"


def test_minus_detail_gate_blocker_overrides_accept_decision(tmp_path: Path) -> None:
    vocal = sine(440)
    instrumental = sine(120) + sine(700)
    mix = vocal + instrumental
    music_damaged_minus = instrumental * 0.50

    result = judge_candidate_rows(
        [candidate_row(tmp_path, candidate_id="detail-damaged", mix=mix, vocal=vocal, minus=music_damaged_minus)],
        score_mode="minus",
    )
    row = result["rows"][0]

    assert row["minus_musical_detail_preservation_gate"]["status"] == "proxy_blocked"
    assert row["decision"] == "blocked_hard_case"
    assert row["minus_decision"] == "blocked_hard_case"
    assert "minus_detail_gate_blocked" in row["reasons"]
    assert "minus_detail_gate_blocked" in row["minus_blockers"]
    assert result["summary"]["accepted_candidates"] == 0
    assert result["summary"]["blocked_hard_case"] == 1


def test_non_minus_score_modes_do_not_add_minus_detail_gate(tmp_path: Path) -> None:
    vocal = sine(440)
    instrumental = sine(120)
    mix = vocal + instrumental

    for score_mode in ("vocal", "both"):
        result = judge_candidate_rows(
            [candidate_row(tmp_path, candidate_id=f"{score_mode}-mode", mix=mix, vocal=vocal, minus=instrumental)],
            score_mode=score_mode,
        )
        row = result["rows"][0]

        assert row["score_mode"] == score_mode
        assert "minus_musical_detail_preservation_gate" not in row


def test_score_mode_summary_counts_mode_decision_not_generic_decision() -> None:
    summary = build_summary(
        [
            {
                "track_id": "track-a",
                "decision": "accept_candidate",
                "minus_decision": "blocked_hard_case",
                "reasons": ["vocal_leakage_in_minus"],
            }
        ],
        score_mode="minus",
    )

    assert summary["decision_counts"] == {"blocked_hard_case": 1}
    assert summary["accepted_candidates"] == 0
    assert summary["blocked_hard_case"] == 1
    assert summary["usable_without_stems"] == 0


def test_clean_minus_is_not_marked_leaky_when_vocal_proxy_is_bad() -> None:
    true_vocal = sine(440)
    instrumental = sine(120)
    mix = true_vocal + instrumental
    bad_vocal_proxy = 0.02 * instrumental
    clean_minus = instrumental

    row = judge_audio(mix[:, None], bad_vocal_proxy[:, None], clean_minus[:, None], SAMPLE_RATE)

    assert row["vocal_leakage_in_minus_score"] < 0.60
    assert "vocal_leakage_in_minus" not in row["reasons"]


def test_low_energy_clean_minus_with_bad_vocal_proxy_is_not_full_mix_passthrough() -> None:
    quiet_vocal = sine(440, amp=0.015)
    instrumental = sine(120, amp=0.30) + sine(700, amp=0.15)
    mix = quiet_vocal + instrumental
    bad_vocal_proxy = 0.01 * instrumental
    clean_minus = instrumental

    row = judge_audio(mix[:, None], bad_vocal_proxy[:, None], clean_minus[:, None], SAMPLE_RATE)

    assert row["vocal_basis_supported"] is False
    assert 0.03 <= row["minus_removed_energy_ratio"] < 0.08
    assert row["mix_passthrough_similarity"] > 0.92
    assert row["passthrough_leakage_score"] < 0.60
    assert row["vocal_leakage_in_minus_score"] < 0.60
    assert "vocal_leakage_in_minus" not in row["reasons"]


def test_mix_passthrough_minus_is_still_marked_leaky_with_missing_vocal() -> None:
    true_vocal = sine(440)
    instrumental = sine(120)
    mix = true_vocal + instrumental
    missing_vocal = np.zeros_like(mix)
    passthrough_minus = mix.copy()

    row = judge_audio(mix[:, None], missing_vocal[:, None], passthrough_minus[:, None], SAMPLE_RATE)

    assert row["vocal_leakage_in_minus_score"] > 0.60
    assert "vocal_leakage_in_minus" in row["reasons"]


def test_silence_hallucination_blocks_fake_vocal_in_silent_area(tmp_path: Path) -> None:
    silence = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
    active = sine(160, seconds=0.5)
    mix = np.concatenate([silence, active])
    vocal = np.concatenate([sine(440, seconds=0.5), np.zeros_like(active)])
    minus = mix.copy()
    result = judge_candidate_rows([candidate_row(tmp_path, candidate_id="hallucination", mix=mix, vocal=vocal, minus=minus)])
    row = result["rows"][0]
    assert row["silence_hallucination_score"] > 0.80
    assert row["decision"] == "blocked_hard_case"
    assert "silence_hallucination" in row["reasons"]


def test_single_quiet_subwindow_does_not_saturate_silence_hallucination() -> None:
    window = int(SAMPLE_RATE * 0.05)
    mix = sine(160, seconds=12.0, amp=0.30)
    vocal = sine(440, seconds=12.0, amp=0.10)
    mix[100 * window : 101 * window] = 0.0
    minus = mix - vocal

    row = judge_audio(mix[:, None], vocal[:, None], minus[:, None], SAMPLE_RATE)

    assert row["silence_hallucination_score"] < 0.75
    assert "silence_hallucination" not in row["reasons"]


def test_sustained_quiet_area_vocal_still_counts_as_silence_hallucination() -> None:
    silence = np.zeros(SAMPLE_RATE * 6, dtype=np.float32)
    active = sine(160, seconds=6.0, amp=0.30)
    mix = np.concatenate([silence, active])
    vocal = np.concatenate([sine(440, seconds=6.0, amp=0.10), np.zeros_like(active)])
    minus = mix - vocal

    row = judge_audio(mix[:, None], vocal[:, None], minus[:, None], SAMPLE_RATE)

    assert row["silence_hallucination_score"] > 0.80
    assert "silence_hallucination" in row["reasons"]


def test_zero_vocal_in_quiet_area_is_not_silence_hallucination() -> None:
    silence = np.zeros(SAMPLE_RATE * 6, dtype=np.float32)
    active = sine(160, seconds=6.0, amp=0.30)
    mix = np.concatenate([silence, active])
    vocal = np.zeros_like(mix)
    minus = mix.copy()

    row = judge_audio(mix[:, None], vocal[:, None], minus[:, None], SAMPLE_RATE)

    assert row["silence_hallucination_score"] < 0.10
    assert "silence_hallucination" not in row["reasons"]


def test_output_is_deterministic_and_no_audio_is_exported(tmp_path: Path) -> None:
    vocal = sine(440)
    instrumental = sine(120)
    mix = vocal + instrumental
    rows = [candidate_row(tmp_path, candidate_id="good", mix=mix, vocal=vocal, minus=instrumental)]
    first = judge_candidate_rows(rows)
    second = judge_candidate_rows(rows)
    assert first["rows"] == second["rows"]
    assert first["summary"] == second["summary"]

    output_dir = tmp_path / "out"
    write_outputs(first, output_dir)
    assert (output_dir / "no_reference_result_judge.jsonl").exists()
    assert (output_dir / "no_reference_result_summary.md").exists()
    assert not list(output_dir.rglob("*.wav"))
    assert first["summary"]["safety"]["reference_required"] is False
    assert first["summary"]["safety"]["human_csv_required"] is False
    assert first["summary"]["safety"]["training_started"] is False
    assert first["summary"]["safety"]["models_changed"] is False
    assert first["summary"]["safety"]["cleanup_routes_added"] is False
    assert first["summary"]["safety"]["thresholds_relaxed"] is False
