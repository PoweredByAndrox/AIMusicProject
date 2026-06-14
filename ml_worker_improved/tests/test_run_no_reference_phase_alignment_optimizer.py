from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

import tools.run_no_reference_phase_alignment_optimizer as optimizer
from tools.run_no_reference_phase_alignment_optimizer import build_phase_alignment_result, main


SAMPLE_RATE = 16_000


def sine(freq: float, seconds: float = 1.0, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
    return str(path)


def phase_problem_candidate(tmp_path: Path, *, candidate_id: str = "phase-wide") -> dict:
    left = sine(440.0, seconds=1.5)
    right = sine(880.0, seconds=1.5)
    flawed_vocal = np.stack([left, right], axis=1)
    corrected_vocal = np.repeat(((left + right) * 0.5)[:, None], 2, axis=1)
    instrumental = np.repeat(sine(140.0, seconds=1.5, amp=0.12)[:, None], 2, axis=1)
    mix = corrected_vocal + instrumental
    compensated_minus = mix - flawed_vocal
    return {
        "track_id": "track-a",
        "candidate_id": candidate_id,
        "mix_path": write_wav(tmp_path / candidate_id / "mix.wav", mix),
        "vocal_path": write_wav(tmp_path / candidate_id / "vocal.wav", flawed_vocal),
        "instrumental_path": write_wav(tmp_path / candidate_id / "minus.wav", compensated_minus),
    }


def stable_candidate(tmp_path: Path, *, candidate_id: str = "stable") -> dict:
    vocal = np.repeat(sine(440.0, seconds=1.5)[:, None], 2, axis=1)
    instrumental = np.repeat(sine(140.0, seconds=1.5, amp=0.12)[:, None], 2, axis=1)
    mix = vocal + instrumental
    return {
        "track_id": "track-a",
        "candidate_id": candidate_id,
        "mix_path": write_wav(tmp_path / candidate_id / "mix.wav", mix),
        "vocal_path": write_wav(tmp_path / candidate_id / "vocal.wav", vocal),
        "instrumental_path": write_wav(tmp_path / candidate_id / "minus.wav", instrumental),
    }


def localized_phase_problem_candidate(tmp_path: Path, *, candidate_id: str = "localized-phase") -> dict:
    seconds = 12.0
    left = sine(440.0, seconds=seconds)
    right = left.copy()
    issue_start = int(SAMPLE_RATE * 4.0)
    issue_end = int(SAMPLE_RATE * 4.5)
    right[issue_start:issue_end] = sine(880.0, seconds=0.5)
    flawed_vocal = np.stack([left, right], axis=1)
    corrected_vocal = flawed_vocal.copy()
    local_mid = np.mean(flawed_vocal[issue_start:issue_end], axis=1, keepdims=True)
    corrected_vocal[issue_start:issue_end] = np.repeat(local_mid, 2, axis=1)
    instrumental = np.repeat(sine(140.0, seconds=seconds, amp=0.12)[:, None], 2, axis=1)
    mix = corrected_vocal + instrumental
    compensated_minus = mix - flawed_vocal
    return {
        "track_id": "track-a",
        "candidate_id": candidate_id,
        "mix_path": write_wav(tmp_path / candidate_id / "mix.wav", mix),
        "vocal_path": write_wav(tmp_path / candidate_id / "vocal.wav", flawed_vocal),
        "instrumental_path": write_wav(tmp_path / candidate_id / "minus.wav", compensated_minus),
    }


def test_phase_optimizer_selects_improved_bounded_candidate(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    result = build_phase_alignment_result([phase_problem_candidate(tmp_path)], output_dir=output_dir, min_delta=0.01)

    row = result["rows"][0]
    assert row["final_decision"] == "phase_corrected_candidate_selected"
    assert row["input_candidate_id"] == "phase-wide"
    assert row["selected_candidate_id"].endswith("__vocal_mid_mono_compensated")
    assert row["candidate_id"] == row["selected_candidate_id"]
    assert row["mix_path"].endswith("mix.wav")
    assert row["vocal_path"] == row["selected_vocal_path"]
    assert row["instrumental_path"] == row["selected_minus_path"]
    assert row["vocal_phase_before"] < 0.30
    assert row["vocal_phase_after"] > 0.95
    assert row["minus_score_after"] >= row["minus_score_before"]
    assert row["mixture_consistency_after"] >= row["mixture_consistency_before"] - 0.001
    assert row["improvement_delta"] >= 0.01
    assert Path(row["selected_vocal_path"]).exists()
    assert Path(row["selected_minus_path"]).exists()
    assert output_dir / "phase_alignment_candidates" == Path(row["selected_vocal_path"]).parent
    assert row["can_publish_as_result"] is False
    assert row["can_mark_release_ready"] is False
    assert row["requires_post_correction_judge"] is True
    assert result["summary"]["phase_corrected_candidate_selected"] == 1
    assert result["summary"]["safety"]["reference_required"] is False
    assert result["summary"]["safety"]["training_started"] is False
    assert result["summary"]["safety"]["models_changed"] is False
    assert result["summary"]["safety"]["cleanup_routes_added"] is False
    assert result["summary"]["safety"]["thresholds_relaxed"] is False
    assert result["summary"]["safety"]["audio_exported_as_result"] is False


def test_phase_optimizer_keeps_original_when_no_improvement(tmp_path: Path) -> None:
    candidate = stable_candidate(tmp_path)
    result = build_phase_alignment_result([candidate], output_dir=tmp_path / "out", min_delta=0.01)

    row = result["rows"][0]
    assert row["final_decision"] == "original_candidate_kept"
    assert row["selected_candidate_id"] == "stable"
    assert row["candidate_id"] == "stable"
    assert row["mix_path"] == candidate["mix_path"]
    assert row["vocal_path"] == candidate["vocal_path"]
    assert row["instrumental_path"] == candidate["instrumental_path"]
    assert row["selected_vocal_path"] == candidate["vocal_path"]
    assert row["can_publish_as_result"] is False
    assert row["rollback_reason"] in {"fitness_improvement_below_min_delta", "no_phase_candidate_variant_needed"}


def test_phase_optimizer_uses_windowed_phase_blocker_as_trigger(tmp_path: Path) -> None:
    candidate = stable_candidate(tmp_path, candidate_id="stable-windowed-phase")
    windowed_row = {
        "track_id": "track-a",
        "candidate_id": "stable-windowed-phase",
        "score_mode": "vocal",
        "decision": "blocked_hard_case",
        "reasons": ["severe_phase_comb_filtering"],
        "vocal_blockers": ["severe_phase_comb_filtering"],
        "window_start_sec": 12.0,
        "window_duration_sec": 6.0,
    }

    result = build_phase_alignment_result(
        [candidate],
        output_dir=tmp_path / "out",
        windowed_judge_rows=[windowed_row],
    )

    row = result["rows"][0]
    assert row["final_decision"] == "original_candidate_kept"
    assert row["rollback_reason"] != "no_phase_candidate_variant_needed"
    assert row["optimization_iterations"] > 0
    assert row["windowed_phase_evidence_count"] == 1
    assert "windowed_phase_evidence_triggered" in row["reasons"]
    assert row["can_publish_as_result"] is False
    assert row["can_mark_release_ready"] is False


def test_phase_optimizer_builds_window_local_phase_variant_from_window_bounds(tmp_path: Path) -> None:
    candidate = stable_candidate(tmp_path, candidate_id="stable-window-local-phase")
    windowed_row = {
        "track_id": "track-a",
        "candidate_id": "stable-window-local-phase",
        "score_mode": "vocal",
        "decision": "blocked_hard_case",
        "reasons": ["severe_phase_comb_filtering"],
        "vocal_blockers": ["severe_phase_comb_filtering"],
        "window_start_sec": 0.25,
        "window_duration_sec": 0.50,
    }

    result = build_phase_alignment_result(
        [candidate],
        output_dir=tmp_path / "out",
        windowed_judge_rows=[windowed_row],
    )

    row = result["rows"][0]
    assert row["windowed_phase_evidence_count"] == 1
    assert row["windowed_phase_windows_used"] == 1
    assert row["optimization_iterations"] >= 6
    assert "bounded_window_local_phase_alignment_candidate" in row["actions_tried"]
    assert row["can_publish_as_result"] is False
    assert row["can_mark_release_ready"] is False


def test_phase_optimizer_builds_window_local_variant_from_nested_window_metrics(tmp_path: Path) -> None:
    candidate = stable_candidate(tmp_path, candidate_id="stable-nested-window-phase")
    windowed_row = {
        "track_id": "track-a",
        "candidate_id": "stable-nested-window-phase",
        "score_mode": "vocal",
        "decision": "blocked_hard_case",
        "reasons": ["severe_phase_comb_filtering"],
        "vocal_blockers": ["severe_phase_comb_filtering"],
        "window_metrics": [
            {
                "kind": "grid_0000",
                "start_sec": 0.0,
                "duration_sec": 0.25,
                "reasons": ["bad_mixture_reconstruction"],
            },
            {
                "kind": "grid_0001",
                "start_sec": 0.25,
                "duration_sec": 0.50,
                "reasons": ["phase_or_stereo_inconsistency"],
                "vocal_blockers": ["severe_phase_comb_filtering"],
            },
        ],
    }

    result = build_phase_alignment_result(
        [candidate],
        output_dir=tmp_path / "out",
        windowed_judge_rows=[windowed_row],
    )

    row = result["rows"][0]
    assert row["windowed_phase_evidence_count"] == 1
    assert row["windowed_phase_windows_used"] == 1
    assert "bounded_window_local_phase_alignment_candidate" in row["actions_tried"]


def test_phase_optimizer_preserves_zero_second_window_bounds(tmp_path: Path) -> None:
    candidate = stable_candidate(tmp_path, candidate_id="stable-zero-window-phase")
    windowed_row = {
        "track_id": "track-a",
        "candidate_id": "stable-zero-window-phase",
        "score_mode": "vocal",
        "decision": "blocked_hard_case",
        "reasons": ["severe_phase_comb_filtering"],
        "vocal_blockers": ["severe_phase_comb_filtering"],
        "window_start_sec": 0.0,
        "window_duration_sec": 0.25,
    }

    result = build_phase_alignment_result(
        [candidate],
        output_dir=tmp_path / "out",
        windowed_judge_rows=[windowed_row],
    )

    row = result["rows"][0]
    assert row["windowed_phase_windows_used"] == 1
    assert "bounded_window_local_phase_alignment_candidate" in row["actions_tried"]


def test_phase_optimizer_selects_window_local_candidate_by_local_delta(tmp_path: Path) -> None:
    candidate = localized_phase_problem_candidate(tmp_path)
    windowed_row = {
        "track_id": "track-a",
        "candidate_id": "localized-phase",
        "score_mode": "vocal",
        "decision": "blocked_hard_case",
        "reasons": ["severe_phase_comb_filtering"],
        "vocal_blockers": ["severe_phase_comb_filtering"],
        "window_metrics": [
            {
                "kind": "grid_0004",
                "start_sec": 4.0,
                "duration_sec": 0.5,
                "reasons": ["phase_or_stereo_inconsistency"],
                "vocal_blockers": ["severe_phase_comb_filtering"],
            },
        ],
    }

    result = build_phase_alignment_result(
        [candidate],
        output_dir=tmp_path / "out",
        min_delta=0.01,
        windowed_judge_rows=[windowed_row],
    )

    row = result["rows"][0]
    assert row["final_decision"] == "phase_corrected_candidate_selected"
    assert row["selected_candidate_id"].endswith("__window_local")
    assert row["windowed_phase_windows_used"] == 1
    assert row["local_window_improvement_delta"] >= 0.01
    assert row["can_publish_as_result"] is False
    assert row["can_mark_release_ready"] is False


def test_phase_optimizer_rejects_window_local_without_whole_track_viability(tmp_path: Path, monkeypatch) -> None:
    audio = np.zeros((32, 2), dtype=np.float32)
    candidate = {
        "track_id": "track-a",
        "candidate_id": "local-no-global",
        "mix_path": "mix.wav",
        "vocal_path": "vocal.wav",
        "instrumental_path": "minus.wav",
    }
    baseline_metrics = {
        "phase_alignment_fitness": 0.800000,
        "vocal_score": 0.70,
        "minus_score": 0.70,
        "pair_compatibility_score": 0.70,
        "overall_no_reference_score": 0.70,
        "mixture_consistency_score": 0.80,
        "vocal_phase_consistency_score": 0.25,
        "instrumental_phase_consistency_score": 0.25,
        "reasons": ["phase_or_stereo_inconsistency"],
        "vocal_dominant_defect": ["phase_or_stereo_inconsistency"],
    }
    local_only_metrics = dict(baseline_metrics, phase_alignment_fitness=0.800500)
    evaluated = iter([baseline_metrics, local_only_metrics])

    monkeypatch.setattr(optimizer, "load_aligned_audio", lambda row: (audio, audio, audio, SAMPLE_RATE))
    monkeypatch.setattr(optimizer, "has_phase_issue", lambda metrics, phase_trigger, windowed_rows: True)
    monkeypatch.setattr(
        optimizer,
        "window_local_compensated_variants",
        lambda vocal, minus, *, intervals, sample_rate: [
            {
                "variant_id": "vocal_mid_mono_compensated__window_local",
                "corrected_vocal": audio,
                "corrected_minus": audio,
                "action": "bounded_window_local_phase_alignment_candidate",
            }
        ],
    )
    monkeypatch.setattr(optimizer, "compensated_variants", lambda vocal, minus: [])
    monkeypatch.setattr(optimizer, "evaluate_pair", lambda mix, vocal, minus, sample_rate: next(evaluated))
    interval_scores = iter([0.50, 0.54, 0.54])
    monkeypatch.setattr(optimizer, "interval_phase_fitness", lambda mix, vocal, minus, sample_rate, intervals: next(interval_scores))
    rendered_candidates = []

    def record_render(output_dir, row, variant_id, vocal, minus, sample_rate):
        rendered_candidates.append(variant_id)
        return str(output_dir / "should_not_select_vocal.wav"), str(output_dir / "should_not_select_minus.wav")

    monkeypatch.setattr(optimizer, "write_candidate_audio", record_render)

    result = build_phase_alignment_result(
        [candidate],
        output_dir=tmp_path / "out",
        min_delta=0.01,
        windowed_judge_rows=[
            {
                "track_id": "track-a",
                "candidate_id": "local-no-global",
                "score_mode": "vocal",
                "decision": "blocked_hard_case",
                "reasons": ["phase_or_stereo_inconsistency"],
                "vocal_blockers": ["severe_phase_comb_filtering"],
                "window_start_sec": 0.0,
                "window_duration_sec": 0.001,
            }
        ],
    )

    row = result["rows"][0]
    assert row["final_decision"] == "original_candidate_kept"
    assert row["rollback_reason"] == "window_local_whole_track_delta_below_floor"
    assert row["requires_post_correction_judge"] is False
    assert row["selected_vocal_path"] == "vocal.wav"
    assert row["selected_minus_path"] == "minus.wav"
    assert row["actions_rejected"] == [
        {
            "variant_id": "vocal_mid_mono_compensated__window_local",
            "action": "bounded_window_local_phase_alignment_candidate",
            "reason": "window_local_whole_track_delta_below_floor",
            "fitness_delta": 0.04,
            "whole_track_fitness_delta": 0.0005,
            "local_window_fitness_delta": 0.04,
            "min_window_local_whole_track_delta": optimizer.DEFAULT_MIN_WINDOW_LOCAL_WHOLE_TRACK_DELTA,
        }
    ]
    assert result["summary"]["phase_corrected_candidate_selected"] == 0
    assert result["summary"]["safety"]["preview_candidate_audio_rendered"] is False
    assert rendered_candidates == []


def test_phase_optimizer_does_not_accept_best_of_bad(tmp_path: Path) -> None:
    row = phase_problem_candidate(tmp_path)
    row["candidate_id"] = "bad-paths"
    row["vocal_path"] = ""

    result = build_phase_alignment_result([row], output_dir=tmp_path / "out", min_delta=0.01)

    judged = result["rows"][0]
    assert judged["final_decision"] == "blocked_hard_case"
    assert judged["selected_candidate_id"] is None
    assert "missing_required_audio_path" in judged["reasons"]
    assert judged["can_publish_as_result"] is False
    assert judged["can_mark_release_ready"] is False


def test_phase_optimizer_output_is_deterministic(tmp_path: Path) -> None:
    rows = [phase_problem_candidate(tmp_path, candidate_id="b"), stable_candidate(tmp_path, candidate_id="a")]
    output_dir = tmp_path / "out"

    first = build_phase_alignment_result(list(reversed(rows)), output_dir=output_dir, min_delta=0.01)
    second = build_phase_alignment_result(rows, output_dir=output_dir, min_delta=0.01)

    assert first["rows"] == second["rows"]
    assert first["summary"] == second["summary"]


def test_phase_optimizer_cli_writes_outputs(tmp_path: Path) -> None:
    candidate_jsonl = tmp_path / "candidates.jsonl"
    candidate_jsonl.write_text(json.dumps(phase_problem_candidate(tmp_path), sort_keys=True) + "\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    exit_code = main(["--candidates-jsonl", str(candidate_jsonl), "--output-dir", str(output_dir), "--min-delta", "0.01"])

    assert exit_code == 0
    assert (output_dir / "no_reference_phase_alignment_optimizer_result.jsonl").exists()
    assert (output_dir / "no_reference_phase_alignment_optimizer_summary.md").exists()
    rows = [json.loads(line) for line in (output_dir / "no_reference_phase_alignment_optimizer_result.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["final_decision"] == "phase_corrected_candidate_selected"


def test_phase_optimizer_cli_accepts_windowed_judge_evidence(tmp_path: Path) -> None:
    candidate_jsonl = tmp_path / "candidates.jsonl"
    candidate_jsonl.write_text(json.dumps(stable_candidate(tmp_path, candidate_id="stable-windowed-phase"), sort_keys=True) + "\n", encoding="utf-8")
    windowed_jsonl = tmp_path / "windowed.jsonl"
    windowed_jsonl.write_text(
        json.dumps(
            {
                "track_id": "track-a",
                "candidate_id": "stable-windowed-phase",
                "score_mode": "vocal",
                "decision": "blocked_hard_case",
                "reasons": ["severe_phase_comb_filtering"],
                "vocal_blockers": ["severe_phase_comb_filtering"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            "--candidates-jsonl",
            str(candidate_jsonl),
            "--windowed-judge-jsonl",
            str(windowed_jsonl),
            "--output-dir",
            str(output_dir),
            "--min-delta",
            "0.01",
        ]
    )

    assert exit_code == 0
    rows = [json.loads(line) for line in (output_dir / "no_reference_phase_alignment_optimizer_result.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["windowed_phase_evidence_count"] == 1
    assert rows[0]["optimization_iterations"] > 0
    assert "windowed_phase_evidence_triggered" in rows[0]["reasons"]
