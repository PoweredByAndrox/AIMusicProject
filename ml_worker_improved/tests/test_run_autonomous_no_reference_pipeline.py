from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tools import run_autonomous_no_reference_pipeline as controller
from tools.train_no_reference_critic import MODEL_FILES, dump_bundle
from tools.run_autonomous_no_reference_pipeline import (
    ALLOWED_RUNTIME_STATUSES,
    evaluate_pair_compatibility,
    main,
    select_best_candidate,
)


SAMPLE_RATE = 16_000


def sine(freq: float, seconds: float, amp: float = 0.25) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return str(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def create_shadow_model_dir(model_dir: Path, threshold: float = 0.05) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    binary = {
        "model_type": "threshold_fallback",
        "threshold": threshold,
        "feature_list": ["overall_no_reference_score"],
    }
    for key in ("vocal_critic", "minus_critic", "pair_critic"):
        dump_bundle(dict(binary), model_dir / MODEL_FILES[key])
    dump_bundle(
        {
            "model_type": "majority_policy_fallback",
            "majority_action": "accept_candidate",
            "feature_list": ["overall_no_reference_score"],
        },
        model_dir / MODEL_FILES["policy_controller"],
    )
    (model_dir / "model_manifest.json").write_text(
        json.dumps(
            {
                "model_family": "local_no_reference_weak_label_critic",
                "model_version": "unit-test-shadow",
                "training_data_hash": "unit-test-hash",
                "safety": {"release_ready": False, "training_started": False},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_autonomous_controller_runs_full_mocked_pipeline_from_one_command(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    model_dir = tmp_path / "models"
    create_shadow_model_dir(model_dir)
    vocal = sine(440, 2.0)
    minus = sine(120, 2.0)
    mix = vocal + minus
    mix_path = write_wav(tmp_path / "audio" / "song.wav", mix)
    vocal_path = write_wav(tmp_path / "audio" / "cand" / "vocal.wav", vocal)
    minus_path = write_wav(tmp_path / "audio" / "cand" / "minus.wav", minus)
    write_jsonl(
        output_dir / "candidate_pool.jsonl",
        [
            {
                "track_id": "song-001",
                "candidate_id": "candidate-a",
                "mix_path": mix_path,
                "vocal_path": vocal_path,
                "instrumental_path": minus_path,
                "minus_path": minus_path,
                "is_complete_pair": True,
            }
        ],
    )

    exit_code = main(
        [
            "--input-audio",
            mix_path,
            "--output-dir",
            str(output_dir),
            "--model-dir",
            str(model_dir),
            "--profile",
            "safe",
            "--generate-candidates",
            "false",
            "--window-seconds",
            "0.5",
            "--max-windows-per-candidate",
            "1",
        ]
    )

    assert exit_code == 0
    required = [
        "run_manifest.json",
        "runtime_summary.md",
        "selected_candidates.jsonl",
        "learned_critic_scores.jsonl",
        "pair_compatibility.jsonl",
        "preview_manifest.json",
        "post_render_judge.jsonl",
        "learning_outcome.jsonl",
    ]
    for name in required:
        assert (output_dir / name).exists(), name
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    preview = json.loads((output_dir / "preview_manifest.json").read_text(encoding="utf-8"))
    learning_rows = [json.loads(line) for line in (output_dir / "learning_outcome.jsonl").read_text(encoding="utf-8").splitlines()]

    assert manifest["codex_required_at_runtime"] is False
    assert manifest["manual_phase_execution_required"] is False
    assert manifest["output_status"] in ALLOWED_RUNTIME_STATUSES
    assert manifest["output_status"] == "preview_ready"
    assert manifest["release_ready"] is False
    assert manifest["learned_model"]["model_version"] == "unit-test-shadow"
    assert manifest["selected_vocal_candidate"] == "candidate-a"
    assert manifest["selected_minus_candidate"] == "candidate-a"
    assert preview["release_ready"] is False
    assert preview["can_mark_release_ready"] is False
    assert preview["can_publish_as_result"] is False
    assert learning_rows and learning_rows[0]["verified_outcome"] is False
    assert learning_rows[0]["raw_prediction_only"] is True
    assert manifest["candidate_wave_executed"] is False


def test_learned_accept_cannot_bypass_script_blocker() -> None:
    rows = [
        {
            "track_id": "track-a",
            "candidate_id": "blocked",
            "score_mode": "minus",
            "script_decision": "blocked_hard_case",
            "learned_decision": "accept_candidate",
            "final_guarded_decision": "blocked_hard_case",
            "script_blockers": ["severe_vocal_bleed_in_minus"],
            "hybrid_score": 0.99,
        }
    ]

    assert select_best_candidate(rows, "minus") is None


def test_failed_pair_compatibility_does_not_publish_output() -> None:
    vocal_row = {
        "track_id": "track-a",
        "candidate_id": "vocal-a",
        "sample_rate": 44_100,
        "channels": 2,
        "duration_sec": 10.0,
        "mixture_consistency_score": 0.9,
        "phase_consistency_score": 0.9,
    }
    minus_row = {
        "track_id": "track-a",
        "candidate_id": "minus-b",
        "sample_rate": 48_000,
        "channels": 2,
        "duration_sec": 10.0,
        "mixture_consistency_score": 0.9,
        "phase_consistency_score": 0.9,
    }

    result = evaluate_pair_compatibility(vocal_row, minus_row)

    assert result["runtime_status"] == "needs_pair_reconciliation"
    assert result["pair_decision"] == "blocked_pair_incompatible"
    assert result["can_publish_as_result"] is False
    assert result["can_mark_release_ready"] is False
    assert result["release_ready"] is False


def test_segment_blend_branch_renders_bounded_preview_without_publish(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "run"
    model_dir = tmp_path / "models"
    create_shadow_model_dir(model_dir)
    vocal = sine(440, 2.0)
    minus = sine(120, 2.0)
    alt_vocal = sine(660, 2.0)
    alt_minus = sine(180, 2.0)
    mix = vocal + minus
    mix_path = write_wav(tmp_path / "audio" / "song.wav", mix)
    vocal_path = write_wav(tmp_path / "audio" / "cand" / "vocal.wav", vocal)
    minus_path = write_wav(tmp_path / "audio" / "cand" / "minus.wav", minus)
    alt_vocal_path = write_wav(tmp_path / "audio" / "cand" / "alt_vocal.wav", alt_vocal)
    alt_minus_path = write_wav(tmp_path / "audio" / "cand" / "alt_minus.wav", alt_minus)
    write_jsonl(
        output_dir / "candidate_pool.jsonl",
        [
            {
                "track_id": "song-001",
                "candidate_id": "candidate-a",
                "mix_path": mix_path,
                "vocal_path": vocal_path,
                "instrumental_path": minus_path,
                "minus_path": minus_path,
                "is_complete_pair": True,
            }
        ],
    )

    def forced_pair_incompatible(vocal_row: dict, minus_row: dict) -> dict:
        return {
            "track_id": "song-001",
            "selected_vocal_candidate_id": vocal_row["candidate_id"],
            "selected_minus_candidate_id": minus_row["candidate_id"],
            "pair_decision": "blocked_pair_incompatible",
            "pair_blockers": ["forced_pair_incompatible"],
            "runtime_status": "needs_pair_reconciliation",
            "can_publish_as_result": False,
            "can_mark_release_ready": False,
            "release_ready": False,
        }

    def forced_segment_plan(candidate_rows: list[dict], windowed_rows: list[dict], optimizer_rows: list[dict]) -> dict:
        return {
            "rows": [
                {
                    "track_id": "song-001",
                    "base_candidate_id": "candidate-a",
                    "alternate_candidate_id": "candidate-b",
                    "source_stem": "both",
                    "window_start_sec": 0.2,
                    "window_duration_sec": 0.6,
                    "base_vocal_path": vocal_path,
                    "base_minus_path": minus_path,
                    "alternate_vocal_path": alt_vocal_path,
                    "alternate_minus_path": alt_minus_path,
                    "expected_improvement_delta": 0.2,
                }
            ],
            "summary": {"planned_segments": 1},
        }

    def forced_dry_run(segment_rows: list[dict], **kwargs) -> dict:
        row = {
            **segment_rows[0],
            "alt_vocal_path": alt_vocal_path,
            "alt_minus_path": alt_minus_path,
            "crossfade_ms": 40,
            "pre_roll_ms": 80,
            "post_roll_ms": 80,
            "render_status": "render_ready",
            "can_render_later": True,
            "can_publish_as_result": False,
            "can_mark_release_ready": False,
            "release_ready": False,
        }
        return {"rows": [row], "render_contract_rows": [row], "summary": {"render_ready": 1}}

    monkeypatch.setattr(controller, "evaluate_pair_compatibility", forced_pair_incompatible)
    monkeypatch.setattr(controller, "build_segment_blend_plan", forced_segment_plan)
    monkeypatch.setattr(controller, "build_dry_run_contract", forced_dry_run)

    exit_code = controller.main(
        [
            "--input-audio",
            mix_path,
            "--output-dir",
            str(output_dir),
            "--model-dir",
            str(model_dir),
            "--profile",
            "safe",
            "--generate-candidates",
            "false",
            "--window-seconds",
            "0.5",
            "--max-windows-per-candidate",
            "1",
        ]
    )

    assert exit_code == 0
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    preview = json.loads((output_dir / "preview_manifest.json").read_text(encoding="utf-8"))
    post_rows = [json.loads(line) for line in (output_dir / "post_render_judge.jsonl").read_text(encoding="utf-8").splitlines()]

    assert manifest["output_status"] == "segment_blend_needed"
    assert manifest["release_ready"] is False
    assert preview["rendered_preview_audio"]
    assert all(Path(path).exists() for path in preview["rendered_preview_audio"])
    assert preview["can_publish_as_result"] is False
    assert preview["can_mark_release_ready"] is False
    assert post_rows
    assert all(row["can_publish_as_result"] is False for row in post_rows)
    assert all(row["can_mark_release_ready"] is False for row in post_rows)


def test_controller_runs_candidate_wave_when_no_pool_is_provided(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    model_dir = tmp_path / "models"
    create_shadow_model_dir(model_dir)
    vocal = sine(440, 2.0)
    minus = sine(120, 2.0)
    mix = vocal + minus
    mix_path = write_wav(tmp_path / "audio" / "song.wav", mix)
    vocal_path = write_wav(tmp_path / "mocked_separator" / "vocal.wav", vocal)
    minus_path = write_wav(tmp_path / "mocked_separator" / "minus.wav", minus)
    write_jsonl(
        output_dir / "candidate_wave_sources.jsonl",
        [
            {"candidate_config_id": "v-safe", "candidate_family": "vocal_focused", "target_mode": "vocal", "vocal_path": vocal_path},
            {"candidate_config_id": "m-safe", "candidate_family": "minus_focused", "target_mode": "minus", "minus_path": minus_path},
        ],
    )

    exit_code = main(
        [
            "--input-audio",
            mix_path,
            "--output-dir",
            str(output_dir),
            "--model-dir",
            str(model_dir),
            "--profile",
            "safe",
            "--generate-candidates",
            "true",
            "--candidate-profile",
            "safe",
            "--max-candidates",
            "6",
            "--window-seconds",
            "0.5",
            "--max-windows-per-candidate",
            "1",
        ]
    )

    assert exit_code == 0
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    pool_rows = [json.loads(line) for line in (output_dir / "candidate_pool.jsonl").read_text(encoding="utf-8").splitlines()]

    assert manifest["candidate_wave_executed"] is True
    assert manifest["candidate_wave_summary"]["complete_pairs"] == 1
    assert manifest["output_status"] == "preview_ready"
    assert manifest["release_ready"] is False
    assert pool_rows and pool_rows[0]["target_mode"] == "both"


def test_controller_uses_provided_pool_when_generation_is_disabled(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    model_dir = tmp_path / "models"
    create_shadow_model_dir(model_dir)
    vocal = sine(440, 2.0)
    minus = sine(120, 2.0)
    mix = vocal + minus
    mix_path = write_wav(tmp_path / "audio" / "song.wav", mix)
    vocal_path = write_wav(tmp_path / "provided_pool" / "vocal.wav", vocal)
    minus_path = write_wav(tmp_path / "provided_pool" / "minus.wav", minus)
    pool = tmp_path / "provided_pool.jsonl"
    write_jsonl(
        pool,
        [
            {
                "track_id": "provided-song",
                "candidate_id": "provided-candidate",
                "mix_path": mix_path,
                "vocal_path": vocal_path,
                "instrumental_path": minus_path,
                "minus_path": minus_path,
                "is_complete_pair": True,
                "target_mode": "both",
            }
        ],
    )

    exit_code = main(
        [
            "--input-audio",
            mix_path,
            "--output-dir",
            str(output_dir),
            "--model-dir",
            str(model_dir),
            "--profile",
            "safe",
            "--candidate-pool-jsonl",
            str(pool),
            "--generate-candidates",
            "false",
            "--window-seconds",
            "0.5",
            "--max-windows-per-candidate",
            "1",
        ]
    )

    assert exit_code == 0
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["candidate_wave_executed"] is False
    assert manifest["candidate_pool_jsonl"] == str(pool)
    assert manifest["selected_vocal_candidate"] == "provided-candidate"
    assert manifest["selected_minus_candidate"] == "provided-candidate"


def test_controller_zero_generated_candidates_stops_with_needs_more_candidates(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    model_dir = tmp_path / "models"
    create_shadow_model_dir(model_dir)
    mix_path = write_wav(tmp_path / "audio" / "song.wav", sine(220, 1.0))

    exit_code = main(
        [
            "--input-audio",
            mix_path,
            "--output-dir",
            str(output_dir),
            "--model-dir",
            str(model_dir),
            "--profile",
            "safe",
            "--generate-candidates",
            "true",
            "--candidate-profile",
            "safe",
            "--max-candidates",
            "6",
        ]
    )

    assert exit_code == 0
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["candidate_wave_executed"] is True
    assert manifest["output_status"] == "needs_more_candidates"
    assert manifest["release_ready"] is False
    assert manifest["candidate_wave_summary"]["complete_pairs"] == 0
