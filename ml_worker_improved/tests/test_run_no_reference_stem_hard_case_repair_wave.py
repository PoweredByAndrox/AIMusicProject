from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.run_no_reference_stem_hard_case_repair_wave import plan_hard_case_repair_jobs, run_repair_wave


SAFETY_FLAGS_FOR_TEST = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "diagnostic_audio_exported": False,
    "audio_copied": False,
    "audio_exported": False,
    "release_ready": False,
}


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def write_audio(path: Path, *, freq_hz: float = 440.0) -> Path:
    sample_rate = 44_100
    t = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
    mono = np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32) * 0.08
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.column_stack([mono, mono]), sample_rate)
    return path


def write_stem_manifest(path: Path, *, track_id: str, stem: str = "melody") -> Path:
    audio = write_audio(path.parent / f"{stem}.flac", freq_hz=440.0)
    payload = {
        "audio_path": str(path.parent / "source.wav"),
        "model_name": "model_stack_melody_v1",
        "contract": {
            "application_reference_free": True,
            "registered_model_candidate": True,
            "fallback_used": False,
            "hidden_inference": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
            "route_name": "model_stack_melody_v1",
        },
        "execution_contract": {
            "docker_runtime_required": True,
            "expected_acceleration_mode": "gpu_inference_then_cpu_dsp_scoring",
            "gpu_inference_required": True,
            "gpu_inference_skipped": False,
            "cuda_preflight": {
                "cuda_available": True,
                "requested_device": "cuda",
                "resolved_device": "cuda",
                "device_count": 1,
                "device_name": "unit GPU",
            },
        },
        "stems": {stem: {"path": str(audio), "source": "existing_manifest_stem", "source_stems": [stem]}},
        "target_stem": stem,
        "target_stem_artifact": {"target_stem": stem, "path": str(audio), "source": "existing_manifest_stem", "source_stems": [stem]},
        "delivery_allowed": False,
        "presentation_status": "diagnostic_candidate_not_for_listen_pack",
        "release_status": "not_release_ready",
        "release_ready": False,
        "track_id": track_id,
    }
    return write_json(path, payload)


def write_blocked_semantic_batch(tmp_path: Path) -> tuple[Path, Path]:
    track_id = "track-hard-melody"
    source_audio = write_audio(tmp_path / "source" / "track.wav", freq_hz=220.0)
    blocked_audio = write_audio(tmp_path / "blocked" / "melody.flac", freq_hz=57.0)
    listen_manifest = tmp_path / "listen" / track_id / "melody" / "no_reference_stem_listen_preview_manifest.json"
    write_json(
        listen_manifest,
        {
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_status": "listen_preview_only",
            "release_ready": False,
            "stem_type": "melody",
            "track_id": track_id,
            "model_name": "htdemucs_6s",
            "selected_audio_path": str(blocked_audio),
            "source_audio_path": str(source_audio),
            "source_stem_manifest": {
                "path": str(tmp_path / "old" / track_id / "melody" / "htdemucs_6s_aggressive_melody_first" / "stem_manifest.json"),
                "sha256": "old",
            },
            "safety": {
                "reference_required": False,
                "human_csv_required": False,
                "training_started": False,
                "fine_tune_started": False,
                "models_changed": False,
                "model_registry_edited": False,
                "cleanup_routes_added": False,
                "thresholds_relaxed": False,
                "diagnostic_audio_exported": False,
                "audio_copied": False,
                "audio_exported": False,
                "release_ready": False,
            },
        },
    )
    judge_jsonl = tmp_path / "semantic_batch" / track_id / "stem_semantic_judge" / "no_reference_stem_semantic_judge.jsonl"
    write_jsonl(
        judge_jsonl,
        [
            {
                "decision": "blocked_hard_case",
                "stem_semantic_decision": "blocked_hard_case",
                "stem_type": "melody",
                "track_id": track_id,
                "model_name": "htdemucs_6s",
                "selected_audio_path": str(blocked_audio),
                "stem_semantic_blockers": ["melody_low_pitch_confidence"],
                "source_listen_manifest": {"path": str(listen_manifest), "sha256": "listen"},
            }
        ],
    )
    summary = tmp_path / "semantic_batch" / "no_reference_stem_semantic_batch_summary.json"
    write_json(
        summary,
        {
            "batch_status": "completed",
            "track_count": 1,
            "candidate_count": 1,
            "accepted_candidates": 0,
            "blocked_hard_case": 1,
            "release_ready": False,
            "tracks": [
                {
                    "track_id": track_id,
                    "candidate_count": 1,
                    "accepted_candidates": 0,
                    "blocked_hard_case": 1,
                    "judge_jsonl": str(judge_jsonl),
                    "matrix_json": str(tmp_path / "semantic_batch" / track_id / "stem_score_matrix" / "no_reference_stem_candidate_score_matrix.json"),
                    "release_ready": False,
                }
            ],
        },
    )
    return summary, listen_manifest.parent.parent


def test_plan_hard_case_repair_jobs_selects_alternative_melody_route(tmp_path: Path) -> None:
    summary, _listen_root = write_blocked_semantic_batch(tmp_path)

    jobs = plan_hard_case_repair_jobs(
        semantic_batch_summary_json=summary,
        output_root=tmp_path / "repair",
        max_repair_jobs=1,
    )

    assert len(jobs) == 1
    job = jobs[0]
    assert job["target_stem"] == "melody"
    assert job["model_id"] == "pitch_salience_melody_v1"
    assert job["repair_failed_labels"] == ["melody_low_pitch_confidence"]
    assert "htdemucs_6s_aggressive_melody_first" in job["repair_evaluated_labels"]
    assert job["expected_acceleration_mode"] == "gpu_inference_then_cpu_dsp_scoring"
    assert job["gpu_inference_required"] is True
    assert "--target-stem" in job["command"]
    assert job["command"][job["command"].index("--target-stem") + 1] == "melody"


def test_plan_hard_case_repair_jobs_excludes_all_same_stem_evaluated_routes(tmp_path: Path) -> None:
    summary, _listen_root = write_blocked_semantic_batch(tmp_path)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    track = payload["tracks"][0]
    judge_jsonl = Path(track["judge_jsonl"])
    existing_rows = [json.loads(line) for line in judge_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    track_id = existing_rows[0]["track_id"]
    model_stack_audio = write_audio(tmp_path / "repair" / "model_stack" / "melody.flac", freq_hz=58.0)
    model_stack_listen = (
        tmp_path
        / "repair_listen"
        / track_id
        / "melody"
        / "no_reference_stem_listen_preview_manifest.json"
    )
    write_json(
        model_stack_listen,
        {
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_status": "listen_preview_only",
            "release_ready": False,
            "stem_type": "melody",
            "track_id": track_id,
            "model_name": "model_stack_melody_v1",
            "selected_audio_path": str(model_stack_audio),
            "source_audio_path": str(tmp_path / "source" / "track.wav"),
            "source_stem_manifest": {
                "path": str(
                    tmp_path
                    / "candidate"
                    / track_id
                    / "melody"
                    / "model_stack_melody_v1"
                    / "stem_manifest.json"
                ),
                "sha256": "model-stack",
            },
            "safety": SAFETY_FLAGS_FOR_TEST,
        },
    )
    write_jsonl(
        judge_jsonl,
        existing_rows
        + [
            {
                "decision": "blocked_hard_case",
                "stem_semantic_decision": "blocked_hard_case",
                "stem_type": "melody",
                "track_id": track_id,
                "model_name": "model_stack_melody_v1",
                "selected_audio_path": str(model_stack_audio),
                "stem_semantic_blockers": ["melody_low_pitch_confidence"],
                "source_listen_manifest": {"path": str(model_stack_listen), "sha256": "listen-model-stack"},
            }
        ],
    )

    jobs = plan_hard_case_repair_jobs(
        semantic_batch_summary_json=summary,
        output_root=tmp_path / "repair-next",
        max_repair_jobs=1,
    )

    assert len(jobs) == 1
    job = jobs[0]
    assert "htdemucs_6s_aggressive_melody_first" in job["repair_evaluated_labels"]
    assert "model_stack_melody_v1" in job["repair_evaluated_labels"]
    assert job["model_id"] != "model_stack_melody_v1"


def test_run_repair_wave_executes_repairs_and_reruns_semantic_batch(tmp_path: Path, monkeypatch) -> None:
    summary, listen_root = write_blocked_semantic_batch(tmp_path)
    repair_manifest = write_stem_manifest(
        tmp_path / "candidate" / "track-hard-melody" / "melody" / "model_stack_melody_v1" / "stem_manifest.json",
        track_id="track-hard-melody",
    )
    captured = {}

    def fake_execute_job_matrix(**kwargs):
        captured["execute_kwargs"] = kwargs
        return {
            "rows": [
                {
                    "track_id": "track-hard-melody",
                    "target_stem": "melody",
                    "execution_status": "completed",
                    "candidate_manifest": {"exists": True, "path": str(repair_manifest), "target_stem": "melody"},
                }
            ],
            "summary": {
                "selected_jobs": 1,
                "completed_jobs": 1,
                "failed_jobs": 0,
                "release_ready": False,
                "gpu_inference_required": True,
                "expected_acceleration_mode": "gpu_inference_then_cpu_dsp_scoring",
            },
            "manifest": {"release_ready": False, "delivery_allowed": False},
        }

    def fake_run_semantic_batch(**kwargs):
        captured["semantic_kwargs"] = kwargs
        return {
            "batch_status": "completed",
            "track_count": 1,
            "candidate_count": 2,
            "accepted_candidates": 1,
            "blocked_hard_case": 1,
            "semantic_quality_verified_count": 1,
            "release_ready": False,
        }

    monkeypatch.setattr("tools.run_no_reference_stem_hard_case_repair_wave.execute_job_matrix", fake_execute_job_matrix)
    monkeypatch.setattr("tools.run_no_reference_stem_hard_case_repair_wave.run_semantic_batch", fake_run_semantic_batch)

    result = run_repair_wave(
        semantic_batch_summary_json=summary,
        existing_listen_roots=[listen_root],
        output_root=tmp_path / "repair",
        max_repair_jobs=1,
        execute_generation=True,
    )

    assert result["repair_status"] == "completed"
    assert result["planned_repair_jobs"] == 1
    assert result["generation_started"] is True
    assert result["listen_preview_count"] == 1
    assert result["semantic_batch_after_repair"]["candidate_count"] == 2
    assert result["release_ready"] is False
    assert captured["execute_kwargs"]["max_jobs"] == 1
    assert len(captured["semantic_kwargs"]["listen_manifest_roots"]) == 2
    assert (tmp_path / "repair" / "hard_case_repair_jobs.jsonl").exists()
    assert (tmp_path / "repair" / "listen_preview" / "track-hard-melody" / "melody" / "no_reference_stem_listen_preview_manifest.json").exists()
