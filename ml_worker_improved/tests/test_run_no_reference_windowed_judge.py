from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.run_no_reference_windowed_judge import (
    aggregate_metrics,
    judge_candidate_rows_windowed,
    select_windows,
    window_metric_summary,
    write_outputs,
    write_outputs_streaming,
)


SAMPLE_RATE = 16_000


def sine(freq: float, seconds: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return str(path)


def candidate_row(tmp_path: Path, *, candidate_id: str, mix: np.ndarray, vocal: np.ndarray, minus: np.ndarray) -> dict:
    return {
        "track_id": "track-a",
        "candidate_id": candidate_id,
        "mix_path": write_wav(tmp_path / candidate_id / "mix.wav", mix),
        "vocal_path": write_wav(tmp_path / candidate_id / "vocal.wav", vocal),
        "instrumental_path": write_wav(tmp_path / candidate_id / "minus.wav", minus),
    }


def test_deterministic_window_selection(tmp_path: Path) -> None:
    quiet = sine(90, 12.0, amp=0.02)
    loud = sine(220, 12.0, amp=0.7)
    mid = sine(140, 24.0, amp=0.2)
    mix_path = Path(write_wav(tmp_path / "mix.wav", np.concatenate([quiet, mid, loud])))

    first = select_windows(mix_path, window_seconds=12.0, max_windows=5)
    second = select_windows(mix_path, window_seconds=12.0, max_windows=5)

    assert first == second
    assert len(first) <= 5
    assert {window["kind"] for window in first}.issuperset({"start", "middle", "tail", "high_energy", "low_energy"})


def test_dense_window_selection_uses_step_and_unlimited_max(tmp_path: Path) -> None:
    mix_path = Path(write_wav(tmp_path / "dense.wav", sine(220, 31.0)))

    windows = select_windows(mix_path, window_seconds=10.0, max_windows=0, window_step_seconds=6.0)

    assert [window["kind"] for window in windows] == [
        "grid_0000",
        "grid_0001",
        "grid_0002",
        "grid_0003",
        "grid_0004",
    ]
    assert [window["start_sec"] for window in windows] == [0.0, 6.0, 12.0, 18.0, 21.0]
    assert all(window["duration_sec"] == 10.0 for window in windows)


def test_dense_window_selection_respects_positive_max_windows(tmp_path: Path) -> None:
    mix_path = Path(write_wav(tmp_path / "dense_limited.wav", sine(220, 31.0)))

    windows = select_windows(mix_path, window_seconds=10.0, max_windows=3, window_step_seconds=6.0)

    assert [window["start_sec"] for window in windows] == [0.0, 6.0, 12.0]


def test_short_audio_uses_full_audio(tmp_path: Path) -> None:
    mix_path = Path(write_wav(tmp_path / "short.wav", sine(220, 3.0)))

    windows = select_windows(mix_path, window_seconds=12.0, max_windows=5)

    assert windows == [{"kind": "full", "start_sec": 0.0, "duration_sec": 3.0}]


def test_cache_hit_avoids_recomputation(tmp_path: Path) -> None:
    vocal = sine(440, 18.0)
    minus = sine(120, 18.0)
    mix = vocal + minus
    row = candidate_row(tmp_path, candidate_id="good", mix=mix, vocal=vocal, minus=minus)
    cache_path = tmp_path / "cache.jsonl"

    first = judge_candidate_rows_windowed([row], cache_jsonl=cache_path, window_seconds=6.0, max_windows_per_candidate=3)
    second = judge_candidate_rows_windowed([row], cache_jsonl=cache_path, window_seconds=6.0, max_windows_per_candidate=3)

    assert first["summary"]["cache_misses"] > 0
    assert second["summary"]["cache_hits"] == first["summary"]["cache_misses"]
    assert second["summary"]["cache_misses"] == 0
    assert first["rows"] == second["rows"]


def test_severe_defect_in_one_window_blocks_candidate(tmp_path: Path) -> None:
    vocal = sine(440, 36.0)
    clean_minus = sine(120, 36.0)
    leaky_window = clean_minus.copy()
    leaky_window[SAMPLE_RATE * 12 : SAMPLE_RATE * 24] += 1.1 * vocal[SAMPLE_RATE * 12 : SAMPLE_RATE * 24]
    mix = vocal + clean_minus
    row = candidate_row(tmp_path, candidate_id="leaky-window", mix=mix, vocal=vocal, minus=leaky_window)

    result = judge_candidate_rows_windowed([row], cache_jsonl=tmp_path / "cache.jsonl", window_seconds=12.0, max_windows_per_candidate=5)
    judged = result["rows"][0]

    assert judged["decision"] == "blocked_hard_case"
    assert "vocal_leakage_in_minus" in judged["reasons"]
    assert judged["vocal_leakage_in_minus_score"] > 0.60


def test_windowed_output_is_optimizer_compatible_and_no_audio_exported(tmp_path: Path) -> None:
    vocal = sine(440, 15.0)
    minus = sine(120, 15.0)
    mix = vocal + minus
    row = candidate_row(tmp_path, candidate_id="good", mix=mix, vocal=vocal, minus=minus)

    result = judge_candidate_rows_windowed([row], cache_jsonl=tmp_path / "cache.jsonl", window_seconds=6.0, max_windows_per_candidate=3)
    judged = result["rows"][0]
    for key in (
        "track_id",
        "candidate_id",
        "mix_path",
        "vocal_path",
        "instrumental_path",
        "mixture_consistency_score",
        "vocal_leakage_in_minus_score",
        "instrumental_leakage_in_vocal_score",
        "vocal_activity_confidence",
        "silence_hallucination_score",
        "robotic_artifact_score",
        "thin_vocal_score",
        "transient_damage_score",
        "phase_consistency_score",
        "overall_no_reference_score",
        "confidence",
        "decision",
        "reasons",
        "forbidden_actions",
    ):
        assert key in judged
    assert judged["judge_mode"] == "no_reference_windowed"
    assert len(judged["window_metrics"]) == judged["window_count"]
    assert {
        "kind",
        "start_sec",
        "duration_sec",
        "vocal_score",
        "minus_score",
        "artifact_score",
        "consistency_score",
        "window_fitness",
        "reasons",
    }.issubset(judged["window_metrics"][0])
    assert "reference_path" not in judged
    assert result["summary"]["safety"]["training_started"] is False
    assert result["summary"]["safety"]["models_changed"] is False
    assert result["summary"]["safety"]["audio_exported"] is False

    output_dir = tmp_path / "out"
    write_outputs(
        result,
        output_jsonl=output_dir / "no_reference_windowed_judge.jsonl",
        summary_md=output_dir / "no_reference_windowed_judge_summary.md",
        cache_jsonl=output_dir / "no_reference_metric_cache.jsonl",
    )
    assert not list(output_dir.rglob("*.wav"))
    assert not list(output_dir.rglob("*.flac"))
    assert (output_dir / "no_reference_windowed_judge.jsonl").exists()
    assert (output_dir / "no_reference_metric_cache.jsonl").exists()


def test_windowed_output_preserves_candidate_route_metadata(tmp_path: Path) -> None:
    vocal = sine(440, 15.0)
    minus = sine(120, 15.0)
    mix = vocal + minus
    row = {
        **candidate_row(tmp_path, candidate_id="registered-minus", mix=mix, vocal=vocal, minus=minus),
        "stem_type": "minus",
        "source_stem": "vocal",
        "base_candidate_id": "base-pair",
        "alternate_candidate_id": "alt-pair",
        "window_start_sec": 12.0,
        "window_duration_sec": 6.0,
        "selected_vocal_path": str(tmp_path / "registered-minus" / "vocal.wav"),
        "selected_minus_path": str(tmp_path / "registered-minus" / "minus.wav"),
        "route_id": "htdemucs_6s",
        "model_id": "htdemucs_6s",
        "minus_route_id": "htdemucs_6s",
        "route_family": "demucs",
        "candidate_manifest": "/app/artifacts/track/minus/htdemucs_6s/stem_manifest.json",
        "source_manifest": "/app/artifacts/track/minus/htdemucs_6s/registry_candidate_execution_manifest.json",
    }

    result = judge_candidate_rows_windowed(
        [row],
        cache_jsonl=tmp_path / "cache.jsonl",
        window_seconds=6.0,
        max_windows_per_candidate=3,
        score_mode="minus",
    )
    judged = result["rows"][0]

    assert judged["stem_type"] == "minus"
    assert judged["source_stem"] == "vocal"
    assert judged["base_candidate_id"] == "base-pair"
    assert judged["alternate_candidate_id"] == "alt-pair"
    assert judged["window_start_sec"] == 12.0
    assert judged["window_duration_sec"] == 6.0
    assert judged["selected_vocal_path"] == str(tmp_path / "registered-minus" / "vocal.wav")
    assert judged["selected_minus_path"] == str(tmp_path / "registered-minus" / "minus.wav")
    assert judged["route_id"] == "htdemucs_6s"
    assert judged["model_id"] == "htdemucs_6s"
    assert judged["minus_route_id"] == "htdemucs_6s"
    assert judged["route_family"] == "demucs"


def test_minus_mode_uses_vocal_companion_without_selected_vocal_alias(tmp_path: Path) -> None:
    vocal = sine(440, 15.0)
    minus = sine(120, 15.0)
    mix = vocal + minus
    base = candidate_row(tmp_path, candidate_id="instrumental-route", mix=mix, vocal=vocal, minus=minus)
    row = {
        "track_id": base["track_id"],
        "candidate_id": base["candidate_id"],
        "mix_path": base["mix_path"],
        "vocal_companion_path": base["vocal_path"],
        "vocal_companion_role": "residual_companion",
        "vocal_companion_source": "residual_from_raw_minus_instrumental",
        "instrumental_path": base["instrumental_path"],
        "stem_type": "minus",
        "selected_minus_path": base["instrumental_path"],
        "source_manifest_target_stem": "instrumental",
        "canonical_stem_type": "instrumental",
        "user_facing_stem_role": "minus",
        "manifest_stem_name": "instrumental",
    }

    result = judge_candidate_rows_windowed(
        [row],
        cache_jsonl=tmp_path / "cache.jsonl",
        window_seconds=6.0,
        max_windows_per_candidate=3,
        score_mode="minus",
    )
    judged = result["rows"][0]

    assert judged["vocal_path"] == ""
    assert "selected_vocal_path" not in judged
    assert judged["vocal_companion_path"] == base["vocal_path"]
    assert judged["vocal_analysis_path"] == base["vocal_path"]
    assert judged["vocal_companion_role"] == "residual_companion"
    assert judged["vocal_companion_source"] == "residual_from_raw_minus_instrumental"
    assert judged["selected_minus_path"] == base["instrumental_path"]
    assert judged["source_manifest_target_stem"] == "instrumental"
    assert judged["canonical_stem_type"] == "instrumental"
    assert judged["user_facing_stem_role"] == "minus"
    assert judged["manifest_stem_name"] == "instrumental"


def test_streaming_output_resumes_existing_checkpoint_rows(tmp_path: Path) -> None:
    vocal = sine(440, 15.0)
    minus = sine(120, 15.0)
    mix = vocal + minus
    first_row = candidate_row(tmp_path, candidate_id="first", mix=mix, vocal=vocal, minus=minus)
    second_row = candidate_row(tmp_path, candidate_id="second", mix=mix, vocal=vocal, minus=minus)
    output_jsonl = tmp_path / "streamed.jsonl"
    summary_md = tmp_path / "streamed.md"
    cache_jsonl = tmp_path / "cache.jsonl"

    write_outputs_streaming(
        [first_row],
        output_jsonl=output_jsonl,
        summary_md=summary_md,
        cache_jsonl=cache_jsonl,
        window_seconds=6.0,
        max_windows_per_candidate=3,
    )
    write_outputs_streaming(
        [first_row, second_row],
        output_jsonl=output_jsonl,
        summary_md=summary_md,
        cache_jsonl=cache_jsonl,
        window_seconds=6.0,
        max_windows_per_candidate=3,
    )

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["candidate_id"] for row in rows] == ["first", "second"]
    assert len(rows) == 2


def test_aggregate_metrics_adds_vocal_score_mode_fields() -> None:
    result = aggregate_metrics(
        [
            {
                "mixture_consistency_score": 0.92,
                "vocal_leakage_in_minus_score": 0.85,
                "instrumental_leakage_in_vocal_score": 0.05,
                "vocal_activity_confidence": 0.86,
                "silence_hallucination_score": 0.04,
                "robotic_artifact_score": 0.04,
                "thin_vocal_score": 0.08,
                "transient_damage_score": 0.06,
                "phase_consistency_score": 0.92,
                "confidence": 0.82,
                "reasons": ["no_reference_proxy_passed"],
            }
        ],
        score_mode="vocal",
    )

    assert result["score_mode"] == "vocal"
    assert result["vocal_decision"] == "accept_candidate"
    assert result["decision"] == "accept_candidate"
    assert result["minus_decision"] != "accept_candidate"


def test_aggregate_vocal_mode_uses_vocal_phase_not_pair_phase() -> None:
    result = aggregate_metrics(
        [
            {
                "mixture_consistency_score": 0.92,
                "vocal_leakage_in_minus_score": 0.05,
                "instrumental_leakage_in_vocal_score": 0.05,
                "vocal_activity_confidence": 0.86,
                "silence_hallucination_score": 0.04,
                "robotic_artifact_score": 0.04,
                "thin_vocal_score": 0.08,
                "transient_damage_score": 0.06,
                "phase_consistency_score": 0.12,
                "mix_phase_consistency_score": 0.92,
                "vocal_phase_consistency_score": 0.92,
                "instrumental_phase_consistency_score": 0.12,
                "pair_phase_consistency_score": 0.12,
                "confidence": 0.82,
                "reasons": ["no_reference_proxy_passed"],
            }
        ],
        score_mode="vocal",
    )

    assert result["score_mode"] == "vocal"
    assert result["vocal_decision"] == "accept_candidate"
    assert result["decision"] == "accept_candidate"
    assert result["vocal_phase_consistency_score"] == 0.92


def test_aggregate_vocal_mode_ignores_single_low_activity_instrumental_gap() -> None:
    good_window = {
        "mixture_consistency_score": 0.92,
        "vocal_leakage_in_minus_score": 0.05,
        "instrumental_leakage_in_vocal_score": 0.05,
        "vocal_activity_confidence": 0.86,
        "silence_hallucination_score": 0.04,
        "robotic_artifact_score": 0.04,
        "thin_vocal_score": 0.08,
        "transient_damage_score": 0.06,
        "phase_consistency_score": 0.92,
        "mix_phase_consistency_score": 0.92,
        "vocal_phase_consistency_score": 0.92,
        "instrumental_phase_consistency_score": 0.92,
        "pair_phase_consistency_score": 0.92,
        "confidence": 0.82,
        "reasons": ["no_reference_proxy_passed"],
    }
    instrumental_gap = {
        **good_window,
        "vocal_activity_confidence": 0.12,
        "reasons": ["no_reference_proxy_passed"],
    }

    result = aggregate_metrics([good_window, good_window, good_window, instrumental_gap], score_mode="vocal")

    assert result["vocal_decision"] == "accept_candidate"
    assert "vocal_missing_on_active_region" not in result["vocal_blockers"]
    assert result["decision"] == "accept_candidate"


def test_aggregate_metrics_adds_minus_score_mode_fields() -> None:
    result = aggregate_metrics(
        [
            {
                "mixture_consistency_score": 0.92,
                "vocal_leakage_in_minus_score": 0.85,
                "instrumental_leakage_in_vocal_score": 0.05,
                "vocal_activity_confidence": 0.86,
                "silence_hallucination_score": 0.04,
                "robotic_artifact_score": 0.04,
                "thin_vocal_score": 0.08,
                "transient_damage_score": 0.06,
                "phase_consistency_score": 0.92,
                "confidence": 0.82,
                "reasons": ["no_reference_proxy_passed"],
            }
        ],
        score_mode="minus",
    )

    assert result["score_mode"] == "minus"
    assert "severe_vocal_bleed_in_minus" in result["minus_blockers"]
    assert result["minus_decision"] == "blocked_hard_case"
    assert result["decision"] == "blocked_hard_case"


def test_aggregate_minus_detail_gate_blocker_overrides_accept_decision() -> None:
    result = aggregate_metrics(
        [
            {
                "mixture_consistency_score": 0.60,
                "vocal_leakage_in_minus_score": 0.03,
                "instrumental_leakage_in_vocal_score": 0.05,
                "vocal_activity_confidence": 0.86,
                "silence_hallucination_score": 0.04,
                "robotic_artifact_score": 0.04,
                "thin_vocal_score": 0.08,
                "transient_damage_score": 0.06,
                "phase_consistency_score": 0.92,
                "instrumental_phase_consistency_score": 0.92,
                "confidence": 0.82,
                "reasons": ["no_reference_proxy_passed"],
            }
        ],
        score_mode="minus",
    )

    assert result["minus_musical_detail_preservation_gate"]["status"] == "proxy_blocked"
    assert result["minus_decision"] == "blocked_hard_case"
    assert result["decision"] == "blocked_hard_case"
    assert "minus_detail_gate_blocked" in result["minus_blockers"]
    assert "minus_detail_gate_blocked" in result["reasons"]


def test_aggregate_minus_mode_uses_instrumental_phase_not_pair_phase() -> None:
    result = aggregate_metrics(
        [
            {
                "mixture_consistency_score": 0.92,
                "vocal_leakage_in_minus_score": 0.05,
                "instrumental_leakage_in_vocal_score": 0.05,
                "vocal_activity_confidence": 0.86,
                "silence_hallucination_score": 0.04,
                "robotic_artifact_score": 0.04,
                "thin_vocal_score": 0.08,
                "transient_damage_score": 0.06,
                "phase_consistency_score": 0.12,
                "mix_phase_consistency_score": 0.92,
                "vocal_phase_consistency_score": 0.12,
                "instrumental_phase_consistency_score": 0.92,
                "pair_phase_consistency_score": 0.12,
                "confidence": 0.82,
                "reasons": ["no_reference_proxy_passed"],
            }
        ],
        score_mode="minus",
    )

    assert result["score_mode"] == "minus"
    assert result["minus_decision"] == "accept_candidate"
    assert result["decision"] == "accept_candidate"
    assert result["instrumental_phase_consistency_score"] == 0.92


def test_aggregate_both_mode_accepts_pair_compatible_without_shared_score_cutoff() -> None:
    result = aggregate_metrics(
        [
            {
                "mixture_consistency_score": 0.56,
                "vocal_leakage_in_minus_score": 0.26,
                "instrumental_leakage_in_vocal_score": 0.0,
                "vocal_activity_confidence": 0.42,
                "silence_hallucination_score": 0.24,
                "robotic_artifact_score": 0.25,
                "thin_vocal_score": 0.0,
                "transient_damage_score": 0.0,
                "phase_consistency_score": 0.73,
                "mix_phase_consistency_score": 0.75,
                "vocal_phase_consistency_score": 0.95,
                "instrumental_phase_consistency_score": 0.73,
                "pair_phase_consistency_score": 0.73,
                "confidence": 0.72,
                "reasons": ["no_reference_proxy_passed"],
            }
        ],
        score_mode="both",
    )

    assert result["pair_decision"] == "pair_compatible"
    assert result["overall_no_reference_score"] < 0.72
    assert result["decision"] == "accept_candidate"


def test_aggregate_both_mode_blocks_pair_blockers_despite_high_shared_score() -> None:
    result = aggregate_metrics(
        [
            {
                "mixture_consistency_score": 0.92,
                "vocal_leakage_in_minus_score": 0.05,
                "instrumental_leakage_in_vocal_score": 0.05,
                "vocal_activity_confidence": 0.86,
                "silence_hallucination_score": 0.04,
                "robotic_artifact_score": 0.04,
                "thin_vocal_score": 0.08,
                "transient_damage_score": 0.06,
                "phase_consistency_score": 0.92,
                "pair_phase_consistency_score": math.inf,
                "confidence": 0.82,
                "reasons": ["no_reference_proxy_passed"],
            }
        ],
        score_mode="both",
    )

    assert "invalid_no_reference_metric:pair_phase_consistency_score" in result["pair_blockers"]
    assert result["pair_decision"] == "blocked_pair_incompatible"
    assert result["overall_no_reference_score"] > 0.72
    assert result["decision"] == "blocked_hard_case"
    assert "invalid_no_reference_metric:pair_phase_consistency_score" in result["reasons"]


def test_window_metric_summary_uses_mode_specific_fitness() -> None:
    window = {"kind": "middle", "start_sec": 0.0, "duration_sec": 12.0}
    metrics = {
        "mixture_consistency_score": 0.92,
        "vocal_leakage_in_minus_score": 0.45,
        "instrumental_leakage_in_vocal_score": 0.05,
        "vocal_activity_confidence": 0.86,
        "silence_hallucination_score": 0.04,
        "robotic_artifact_score": 0.04,
        "thin_vocal_score": 0.08,
        "transient_damage_score": 0.06,
        "phase_consistency_score": 0.92,
        "confidence": 0.82,
        "reasons": ["no_reference_proxy_passed"],
    }

    vocal = window_metric_summary(window, metrics, score_mode="vocal")
    minus = window_metric_summary(window, metrics, score_mode="minus")

    assert vocal["window_fitness"] == vocal["vocal_score"]
    assert minus["window_fitness"] == minus["minus_score"]
    assert vocal["window_fitness"] > minus["window_fitness"]


def test_window_metric_summary_vocal_blocker_zeroes_window_fitness() -> None:
    window = {"kind": "middle", "start_sec": 0.0, "duration_sec": 12.0}
    metrics = {
        "mixture_consistency_score": 0.92,
        "vocal_leakage_in_minus_score": 0.05,
        "instrumental_leakage_in_vocal_score": 0.05,
        "vocal_activity_confidence": 0.86,
        "silence_hallucination_score": 0.04,
        "robotic_artifact_score": 0.04,
        "thin_vocal_score": 0.08,
        "transient_damage_score": 0.06,
        "phase_consistency_score": 0.92,
        "vocal_phase_consistency_score": math.inf,
        "confidence": 0.82,
        "reasons": ["no_reference_proxy_passed"],
    }

    vocal = window_metric_summary(window, metrics, score_mode="vocal")

    assert "invalid_no_reference_metric:vocal_phase_consistency_score" in vocal["vocal_blockers"]
    assert vocal["vocal_decision"] == "blocked_hard_case"
    assert vocal["vocal_score"] > 0.70
    assert vocal["window_fitness"] == 0.0


def test_window_metric_summary_pair_blocker_zeroes_window_fitness() -> None:
    window = {"kind": "middle", "start_sec": 0.0, "duration_sec": 12.0}
    metrics = {
        "mixture_consistency_score": 0.92,
        "vocal_leakage_in_minus_score": 0.05,
        "instrumental_leakage_in_vocal_score": 0.05,
        "vocal_activity_confidence": 0.86,
        "silence_hallucination_score": 0.04,
        "robotic_artifact_score": 0.04,
        "thin_vocal_score": 0.08,
        "transient_damage_score": 0.06,
        "phase_consistency_score": 0.92,
        "pair_phase_consistency_score": math.inf,
        "confidence": 0.82,
        "reasons": ["no_reference_proxy_passed"],
    }

    pair = window_metric_summary(window, metrics, score_mode="both")

    assert "invalid_no_reference_metric:pair_phase_consistency_score" in pair["pair_blockers"]
    assert pair["pair_decision"] == "blocked_pair_incompatible"
    assert pair["pair_compatibility_score"] > 0.70
    assert pair["window_fitness"] == 0.0
    assert pair["pair_window_fitness"] == 0.0


def test_window_metric_summary_minus_detail_gate_blocks_window() -> None:
    window = {"kind": "middle", "start_sec": 0.0, "duration_sec": 12.0}
    metrics = {
        "mixture_consistency_score": 0.60,
        "vocal_leakage_in_minus_score": 0.03,
        "instrumental_leakage_in_vocal_score": 0.05,
        "vocal_activity_confidence": 0.86,
        "silence_hallucination_score": 0.04,
        "robotic_artifact_score": 0.04,
        "thin_vocal_score": 0.08,
        "transient_damage_score": 0.06,
        "phase_consistency_score": 0.92,
        "instrumental_phase_consistency_score": 0.92,
        "confidence": 0.82,
        "decision": "accept_candidate",
        "reasons": ["no_reference_proxy_passed"],
    }

    minus = window_metric_summary(window, metrics, score_mode="minus")

    assert minus["minus_musical_detail_preservation_gate"]["status"] == "proxy_blocked"
    assert minus["minus_decision"] == "blocked_hard_case"
    assert minus["decision"] == "blocked_hard_case"
    assert minus["window_fitness"] == 0.0
    assert "minus_detail_gate_blocked" in minus["reasons"]


def test_streaming_checkpoint_key_depends_on_score_mode(tmp_path: Path) -> None:
    vocal = sine(440, 15.0)
    minus = sine(120, 15.0)
    mix = vocal + minus
    row = candidate_row(tmp_path, candidate_id="leaky-minus", mix=mix, vocal=vocal, minus=minus + 0.9 * vocal)
    output_jsonl = tmp_path / "streamed.jsonl"
    summary_md = tmp_path / "streamed.md"
    cache_jsonl = tmp_path / "cache.jsonl"

    write_outputs_streaming(
        [row],
        output_jsonl=output_jsonl,
        summary_md=summary_md,
        cache_jsonl=cache_jsonl,
        window_seconds=6.0,
        max_windows_per_candidate=3,
        score_mode="vocal",
    )
    write_outputs_streaming(
        [row],
        output_jsonl=output_jsonl,
        summary_md=summary_md,
        cache_jsonl=cache_jsonl,
        window_seconds=6.0,
        max_windows_per_candidate=3,
        score_mode="minus",
    )

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["score_mode"] for row in rows] == ["vocal", "minus"]


def test_streaming_checkpoint_key_depends_on_role_safe_stem_metadata(tmp_path: Path) -> None:
    vocal = sine(440, 15.0)
    minus = sine(120, 15.0)
    mix = vocal + minus
    row = {
        **candidate_row(tmp_path, candidate_id="instrumental-route", mix=mix, vocal=vocal, minus=minus),
        "stem_type": "minus",
        "selected_minus_path": str(tmp_path / "instrumental-route" / "minus.wav"),
    }
    role_safe_row = {
        **row,
        "source_manifest_target_stem": "instrumental",
        "canonical_stem_type": "instrumental",
        "user_facing_stem_role": "minus",
        "manifest_stem_name": "instrumental",
    }
    output_jsonl = tmp_path / "streamed_role_safe.jsonl"
    summary_md = tmp_path / "streamed_role_safe.md"
    cache_jsonl = tmp_path / "cache_role_safe.jsonl"

    write_outputs_streaming(
        [row],
        output_jsonl=output_jsonl,
        summary_md=summary_md,
        cache_jsonl=cache_jsonl,
        window_seconds=6.0,
        max_windows_per_candidate=3,
        score_mode="minus",
    )
    write_outputs_streaming(
        [role_safe_row],
        output_jsonl=output_jsonl,
        summary_md=summary_md,
        cache_jsonl=cache_jsonl,
        window_seconds=6.0,
        max_windows_per_candidate=3,
        score_mode="minus",
    )

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert "source_manifest_target_stem" not in rows[0]
    assert rows[1]["source_manifest_target_stem"] == "instrumental"
    assert rows[1]["canonical_stem_type"] == "instrumental"
    assert rows[1]["user_facing_stem_role"] == "minus"
    assert rows[1]["manifest_stem_name"] == "instrumental"


def test_streaming_checkpoint_key_depends_on_window_step_seconds(tmp_path: Path) -> None:
    vocal = sine(440, 31.0)
    minus = sine(120, 31.0)
    mix = vocal + minus
    row = candidate_row(tmp_path, candidate_id="dense-audit", mix=mix, vocal=vocal, minus=minus)
    output_jsonl = tmp_path / "streamed_dense.jsonl"
    summary_md = tmp_path / "streamed_dense.md"
    cache_jsonl = tmp_path / "cache_dense.jsonl"

    write_outputs_streaming(
        [row],
        output_jsonl=output_jsonl,
        summary_md=summary_md,
        cache_jsonl=cache_jsonl,
        window_seconds=10.0,
        max_windows_per_candidate=3,
        window_step_seconds=0.0,
        score_mode="vocal",
    )
    write_outputs_streaming(
        [row],
        output_jsonl=output_jsonl,
        summary_md=summary_md,
        cache_jsonl=cache_jsonl,
        window_seconds=10.0,
        max_windows_per_candidate=0,
        window_step_seconds=6.0,
        score_mode="vocal",
    )

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["window_step_seconds"] for row in rows] == [0.0, 6.0]
    assert [row["window_count"] for row in rows] == [3, 5]
