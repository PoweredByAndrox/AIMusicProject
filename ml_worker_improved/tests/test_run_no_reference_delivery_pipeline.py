from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.artifact_status_contract import can_promote_to_listen_result, status_view_from_manifest
import tools.run_no_reference_delivery_pipeline as pipeline
from tools.run_no_reference_delivery_pipeline import main


SAMPLE_RATE = 16_000


def sine(freq: float, seconds: float, amp: float = 0.25) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return str(path)


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def accepted_optimizer_row(tmp_path: Path) -> dict:
    vocal = sine(440.0, 6.0)
    minus = sine(120.0, 6.0)
    mix = vocal + minus
    return {
        "track_id": "track-a",
        "candidate_id": "mode_specific_pair:vocal-a:minus-b",
        "final_decision": "mode_specific_pair_selected",
        "pair_compatibility_decision": "pair_compatible",
        "needs_pair_reconciliation": False,
        "segment_blend_needed": False,
        "mix_path": write_wav(tmp_path / "mix.wav", mix),
        "selected_vocal_path": write_wav(tmp_path / "vocal.wav", vocal),
        "selected_minus_path": write_wav(tmp_path / "minus.wav", minus),
        "selected_vocal_candidate_id": "vocal-a",
        "selected_minus_candidate_id": "minus-b",
        "selected_vocal_score": 0.83,
        "selected_minus_score": 0.84,
        "pair_compatibility_score": 0.76,
    }


def phase_problem_optimizer_row(tmp_path: Path) -> dict:
    seconds = 6.0
    left = sine(440.0, seconds)
    right = sine(880.0, seconds)
    flawed_vocal = np.stack([left, right], axis=1)
    corrected_vocal = np.repeat(((left + right) * 0.5)[:, None], 2, axis=1)
    instrumental = np.repeat(sine(120.0, seconds, amp=0.12)[:, None], 2, axis=1)
    mix = corrected_vocal + instrumental
    compensated_minus = mix - flawed_vocal
    return {
        "track_id": "track-phase",
        "candidate_id": "mode_specific_pair:phase-vocal:phase-minus",
        "final_decision": "mode_specific_pair_selected",
        "pair_compatibility_decision": "pair_compatible",
        "needs_pair_reconciliation": False,
        "segment_blend_needed": False,
        "mix_path": write_wav(tmp_path / "phase_mix.wav", mix),
        "selected_vocal_path": write_wav(tmp_path / "phase_vocal.wav", flawed_vocal),
        "selected_minus_path": write_wav(tmp_path / "phase_minus.wav", compensated_minus),
        "selected_vocal_candidate_id": "phase-vocal",
        "selected_minus_candidate_id": "phase-minus",
        "selected_vocal_score": 0.78,
        "selected_minus_score": 0.80,
        "pair_compatibility_score": 0.74,
    }


def duration_mismatch_optimizer_row(tmp_path: Path) -> dict:
    vocal = sine(440.0, 6.0)
    minus = sine(120.0, 6.0)
    mix = np.concatenate([vocal + minus, sine(220.0, 6.0, amp=0.05)])
    return {
        "track_id": "track-duration-mismatch",
        "candidate_id": "mode_specific_pair:short-vocal:short-minus",
        "final_decision": "mode_specific_pair_selected",
        "pair_compatibility_decision": "pair_compatible",
        "needs_pair_reconciliation": False,
        "segment_blend_needed": False,
        "mix_path": write_wav(tmp_path / "long_mix.wav", mix),
        "selected_vocal_path": write_wav(tmp_path / "short_vocal.wav", vocal),
        "selected_minus_path": write_wav(tmp_path / "short_minus.wav", minus),
        "selected_vocal_candidate_id": "short-vocal",
        "selected_minus_candidate_id": "short-minus",
        "selected_vocal_score": 0.80,
        "selected_minus_score": 0.80,
        "pair_compatibility_score": 0.74,
    }


def test_delivery_pipeline_runs_dense_gate_and_listen_preview(tmp_path: Path) -> None:
    optimizer = write_jsonl(tmp_path / "optimizer.jsonl", [accepted_optimizer_row(tmp_path)])
    output_dir = tmp_path / "delivery"

    assert main(
        [
            "--optimizer-result-jsonl",
            str(optimizer),
            "--output-dir",
            str(output_dir),
            "--window-seconds",
            "1",
            "--window-step-seconds",
            "1",
            "--max-windows-per-candidate",
            "0",
            "--min-window-count",
            "5",
        ]
    ) == 0

    final_manifest = output_dir / "no_reference_listen_preview_manifest.json"
    pipeline_manifest = output_dir / "no_reference_delivery_pipeline_manifest.json"
    assert final_manifest.exists()
    assert pipeline_manifest.exists()

    final_payload = json.loads(final_manifest.read_text(encoding="utf-8"))
    assert final_payload["judge_mode"] == "no_reference_dense_windowed"
    assert final_payload["delivery_allowed"] is True
    assert final_payload["accepted_count"] == 1
    assert final_payload["can_mark_release_ready"] is False
    assert final_payload["audio_copied"] is False
    assert final_payload["audio_exported"] is False
    assert "dense_delivery_gate" in final_payload["accepted_listen_preview_rows"][0]

    view = status_view_from_manifest(final_manifest)
    assert view.listenable
    assert not view.deliverable
    assert can_promote_to_listen_result(final_manifest)

    pipeline_payload = json.loads(pipeline_manifest.read_text(encoding="utf-8"))
    assert pipeline_payload["final_status"] == "accepted_for_listen_pack"
    assert pipeline_payload["safety"]["audio_exported"] is False
    assert pipeline_payload["safety"]["training_started"] is False
    assert pipeline_payload["safety"]["models_changed"] is False
    assert not list(output_dir.rglob("*.flac"))
    assert not list(output_dir.rglob("*.wav"))


def test_delivery_pipeline_blocks_optimizer_only_bad_rows(tmp_path: Path) -> None:
    row = accepted_optimizer_row(tmp_path)
    row["final_decision"] = "blocked_hard_case"
    optimizer = write_jsonl(tmp_path / "optimizer.jsonl", [row])
    output_dir = tmp_path / "blocked_delivery"

    assert main(
        [
            "--optimizer-result-jsonl",
            str(optimizer),
            "--output-dir",
            str(output_dir),
            "--window-seconds",
            "1",
            "--window-step-seconds",
            "1",
            "--max-windows-per-candidate",
            "0",
            "--min-window-count",
            "5",
        ]
    ) == 1

    final_manifest = output_dir / "no_reference_listen_preview_manifest.json"
    payload = json.loads(final_manifest.read_text(encoding="utf-8"))
    assert payload["delivery_allowed"] is False
    assert payload["presentation_status"] == "blocked_from_listen_pack"
    assert payload["accepted_count"] == 0
    assert payload["safety"]["audio_exported"] is False
    assert not list(output_dir.rglob("*.flac"))
    assert not list(output_dir.rglob("*.wav"))


def test_delivery_pipeline_blocks_accepted_row_missing_mix_path_without_crash(tmp_path: Path) -> None:
    row = accepted_optimizer_row(tmp_path)
    row["mix_path"] = ""
    optimizer = write_jsonl(tmp_path / "missing_mix_optimizer.jsonl", [row])
    output_dir = tmp_path / "missing_mix_delivery"

    assert main(
        [
            "--optimizer-result-jsonl",
            str(optimizer),
            "--output-dir",
            str(output_dir),
            "--window-seconds",
            "1",
            "--window-step-seconds",
            "1",
            "--max-windows-per-candidate",
            "0",
            "--min-window-count",
            "5",
        ]
    ) == 1

    final_manifest = output_dir / "no_reference_listen_preview_manifest.json"
    payload = json.loads(final_manifest.read_text(encoding="utf-8"))
    assert payload["delivery_allowed"] is False
    assert payload["accepted_count"] == 0
    assert payload["presentation_status"] == "blocked_from_listen_pack"
    assert payload["safety"]["audio_exported"] is False


def test_delivery_pipeline_can_rescue_phase_blocked_candidate_after_dense_rejudge(tmp_path: Path) -> None:
    optimizer = write_jsonl(tmp_path / "phase_optimizer.jsonl", [phase_problem_optimizer_row(tmp_path)])
    output_dir = tmp_path / "phase_delivery"

    assert main(
        [
            "--optimizer-result-jsonl",
            str(optimizer),
            "--output-dir",
            str(output_dir),
            "--window-seconds",
            "1",
            "--window-step-seconds",
            "1",
            "--max-windows-per-candidate",
            "0",
            "--min-window-count",
            "5",
            "--enable-phase-alignment-rescue",
        ]
    ) == 0

    phase_optimizer_result = output_dir / "phase_alignment_rescue" / "no_reference_phase_alignment_optimizer_result.jsonl"
    assert phase_optimizer_result.exists()
    phase_rows = [
        json.loads(line)
        for line in phase_optimizer_result.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert phase_rows[0]["final_decision"] == "phase_corrected_candidate_selected"
    assert phase_rows[0]["requires_post_correction_judge"] is True
    assert Path(phase_rows[0]["selected_vocal_path"]).parent.name == "phase_alignment_candidates"
    assert Path(phase_rows[0]["selected_minus_path"]).parent.name == "phase_alignment_candidates"

    final_manifest = output_dir / "no_reference_listen_preview_manifest.json"
    payload = json.loads(final_manifest.read_text(encoding="utf-8"))
    assert payload["delivery_allowed"] is True
    assert payload["presentation_status"] == "accepted_for_listen_pack"
    assert payload["accepted_count"] == 1
    assert payload["accepted_listen_preview_rows"][0]["source_optimizer_result"] == str(phase_optimizer_result)
    assert payload["can_mark_release_ready"] is False
    assert payload["audio_copied"] is False
    assert payload["audio_exported"] is False
    assert can_promote_to_listen_result(final_manifest)

    pipeline_manifest = output_dir / "no_reference_delivery_pipeline_manifest.json"
    pipeline_payload = json.loads(pipeline_manifest.read_text(encoding="utf-8"))
    assert pipeline_payload["final_status"] == "accepted_for_listen_pack"
    assert pipeline_payload["phase_alignment_rescue"]["attempted"] is True
    assert pipeline_payload["phase_alignment_rescue"]["selected_count"] == 1
    assert pipeline_payload["phase_alignment_rescue"]["rescue_promotable"] is True
    assert not list(output_dir.glob("*.wav"))


def test_phase_rescue_stops_when_optimizer_selects_no_corrected_candidate(tmp_path: Path, monkeypatch) -> None:
    candidate = accepted_optimizer_row(tmp_path)
    initial_paths = {
        "dense_vocal": write_jsonl(tmp_path / "dense_vocal.jsonl", []),
        "dense_minus": write_jsonl(tmp_path / "dense_minus.jsonl", []),
        "dense_both": write_jsonl(tmp_path / "dense_both.jsonl", []),
    }

    def fake_phase_result(*args, **kwargs):
        return {
            "rows": [
                {
                    "track_id": candidate["track_id"],
                    "input_candidate_id": candidate["candidate_id"],
                    "candidate_id": candidate["candidate_id"],
                    "selected_candidate_id": candidate["candidate_id"],
                    "final_decision": "original_candidate_kept",
                    "mix_path": candidate["mix_path"],
                    "vocal_path": candidate["selected_vocal_path"],
                    "instrumental_path": candidate["selected_minus_path"],
                    "minus_path": candidate["selected_minus_path"],
                    "selected_vocal_path": candidate["selected_vocal_path"],
                    "selected_minus_path": candidate["selected_minus_path"],
                    "requires_post_correction_judge": False,
                    "rollback_reason": "window_local_whole_track_delta_below_floor",
                }
            ],
            "summary": {
                "total_candidates": 1,
                "phase_corrected_candidate_selected": 0,
                "original_candidate_kept": 1,
                "blocked_hard_case": 0,
                "average_fitness_before": 0.0,
                "average_fitness_after": 0.0,
                "average_improvement_delta": 0.0,
                "top_rejected_actions": {},
                "can_publish_as_result_count": 0,
                "can_mark_release_ready_count": 0,
                "requires_post_correction_judge_count": 0,
                "safety": {},
            },
        }

    dense_judge_calls = []
    monkeypatch.setattr(pipeline, "build_phase_alignment_result", fake_phase_result)
    monkeypatch.setattr(
        pipeline,
        "run_windowed_judges",
        lambda **kwargs: dense_judge_calls.append(kwargs),
    )

    rescue = pipeline.run_phase_alignment_rescue(
        candidates=[candidate],
        output_dir=tmp_path / "delivery",
        initial_paths=initial_paths,
        window_seconds=1.0,
        window_step_seconds=1.0,
        max_windows_per_candidate=0,
        max_candidates=1,
        min_window_count=5,
    )

    assert rescue["attempted"] is True
    assert rescue["selected_count"] == 0
    assert rescue["accepted_delivery_candidate_rows"] == 0
    assert rescue["rescue_promotable"] is False
    assert rescue["listen_preview_manifest"] == ""
    assert rescue["reason"] == "no_phase_corrected_delivery_candidate"
    assert dense_judge_calls == []


def test_delivery_pipeline_blocks_duration_mismatch_before_dense_scoring_and_rescue(tmp_path: Path) -> None:
    optimizer = write_jsonl(tmp_path / "duration_mismatch_optimizer.jsonl", [duration_mismatch_optimizer_row(tmp_path)])
    output_dir = tmp_path / "duration_mismatch_delivery"

    assert main(
        [
            "--optimizer-result-jsonl",
            str(optimizer),
            "--output-dir",
            str(output_dir),
            "--window-seconds",
            "1",
            "--window-step-seconds",
            "1",
            "--max-windows-per-candidate",
            "0",
            "--min-window-count",
            "5",
            "--enable-phase-alignment-rescue",
        ]
    ) == 1

    pipeline_manifest = output_dir / "no_reference_delivery_pipeline_manifest.json"
    pipeline_payload = json.loads(pipeline_manifest.read_text(encoding="utf-8"))
    assert pipeline_payload["final_status"] == "blocked_from_listen_pack"
    assert pipeline_payload["candidate_preflight"]["duration_mismatch_count"] == 1
    assert pipeline_payload["candidate_preflight"]["accepted_count"] == 0
    assert pipeline_payload["phase_alignment_rescue"]["attempted"] is False

    rejected = output_dir / "no_reference_delivery_rejected_candidates.jsonl"
    rejected_rows = [json.loads(line) for line in rejected.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rejected_rows[0]["skip_reason"] == "candidate_duration_mismatch"
    assert rejected_rows[0]["delivery_allowed"] is False
    assert rejected_rows[0]["delivery_status"] == "rejected"
    assert rejected_rows[0]["presentation_status"] == "blocked_from_listen_pack"
    assert rejected_rows[0]["mix_duration_sec"] == 12.0
    assert rejected_rows[0]["vocal_duration_sec"] == 6.0
    assert rejected_rows[0]["minus_duration_sec"] == 6.0
