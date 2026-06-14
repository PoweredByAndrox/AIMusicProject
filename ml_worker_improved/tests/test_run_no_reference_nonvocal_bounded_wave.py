from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

import tools.run_no_reference_nonvocal_bounded_wave as bounded_wave
from tools.run_no_reference_nonvocal_bounded_wave import main, run_wave, select_next_track_ids


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def write_audio(path: Path, *, freq_hz: float = 220.0) -> Path:
    sample_rate = 16_000
    t = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
    mono = np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32) * 0.08
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.column_stack([mono, mono]), sample_rate)
    return path


def job_rows(track_id: str, output_root: Path) -> list[dict]:
    rows = []
    for stem in ("bass", "drums", "melody", "other"):
        rows.append(
            {
                "track_id": track_id,
                "target_stem": stem,
                "model_id": "htdemucs_6s",
                "job_status": "queued",
                "execution_ready": True,
                "command": ["python", "/app/tools/run_registered_model_candidate.py"],
                "expected_candidate_manifest": str(output_root / track_id / stem / "stem_manifest.json"),
            }
        )
    return rows


def write_existing_listen_manifest(root: Path, *, track_id: str, stem: str = "bass") -> Path:
    audio = write_audio(root / track_id / stem / f"{stem}.flac", freq_hz=80.0)
    manifest = root / track_id / stem / "no_reference_stem_listen_preview_manifest.json"
    payload = {
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "stem_type": stem,
        "track_id": track_id,
        "selected_audio_path": str(audio),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def write_stem_manifest(path: Path, *, track_id: str, stem: str, freq_hz: float) -> Path:
    audio = write_audio(path.parent / f"{stem}.flac", freq_hz=freq_hz)
    payload = {
        "audio_path": "/host_testing_stuff_music/source.mp3",
        "model_name": "htdemucs_6s",
        "contract": {
            "application_reference_free": True,
            "registered_model_candidate": True,
            "fallback_used": False,
            "hidden_inference": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
            "route_name": "full_res_stem_export",
        },
        "execution_contract": {
            "docker_runtime_required": True,
            "expected_acceleration_mode": "gpu_inference_then_cpu_dsp_scoring",
            "gpu_inference_required": True,
            "gpu_inference_skipped": False,
            "cuda_preflight": {
                "cuda_available": True,
                "resolved_device": "cuda",
                "requested_device": "cuda",
                "device_count": 1,
                "device_name": "unit GPU",
            },
        },
        "stems": {stem: {"path": str(audio), "source": "existing_manifest_stem", "source_stems": [stem]}},
        "target_stem": stem,
        "target_stem_artifact": {"target_stem": stem, "path": str(audio)},
        "delivery_allowed": False,
        "presentation_status": "diagnostic_candidate_not_for_listen_pack",
        "release_status": "not_release_ready",
        "release_ready": False,
        "track_id": track_id,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_select_next_track_ids_excludes_already_evaluated_tracks(tmp_path: Path) -> None:
    job_matrix = write_jsonl(
        tmp_path / "jobs.jsonl",
        job_rows("track-a", tmp_path / "out") + job_rows("track-b", tmp_path / "out"),
    )
    listen_root = tmp_path / "listen"
    write_existing_listen_manifest(listen_root, track_id="track-a")

    selected = select_next_track_ids(
        job_matrix_jsonl=job_matrix,
        existing_listen_roots=[listen_root],
        max_tracks=1,
    )

    assert selected == ["track-b"]


def test_wave_dry_run_selects_tracks_without_execution(tmp_path: Path) -> None:
    job_matrix = write_jsonl(tmp_path / "jobs.jsonl", job_rows("track-a", tmp_path / "out"))
    output_root = tmp_path / "wave"

    summary = run_wave(
        job_matrix_jsonl=job_matrix,
        output_root=output_root,
        existing_listen_roots=[],
        max_tracks=1,
        execute_generation=False,
    )

    assert summary["wave_status"] == "dry_run"
    assert summary["selected_track_ids"] == ["track-a"]
    assert summary["generation_started"] is False
    assert summary["release_ready"] is False
    assert (output_root / "no_reference_nonvocal_bounded_wave_summary.json").exists()
    assert (output_root / "no_reference_nonvocal_bounded_wave_progress.jsonl").exists()


def test_wave_execute_builds_listen_preview_and_semantic_batch(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    job_matrix = write_jsonl(tmp_path / "jobs.jsonl", job_rows("track-a", source_root))
    for stem, freq in {"bass": 80.0, "drums": 80.0, "melody": 440.0, "other": 330.0}.items():
        write_stem_manifest(source_root / "track-a" / stem / "stem_manifest.json", track_id="track-a", stem=stem, freq_hz=freq)

    def fake_execute_job_matrix(**kwargs):
        rows = []
        for stem in ("bass", "drums", "melody", "other"):
            manifest = source_root / "track-a" / stem / "stem_manifest.json"
            rows.append(
                {
                    "track_id": "track-a",
                    "target_stem": stem,
                    "execution_status": "completed",
                    "candidate_manifest": {"exists": True, "path": str(manifest), "target_stem": stem},
                }
            )
        return {
            "rows": rows,
            "summary": {
                "selected_jobs": 4,
                "completed_jobs": 4,
                "failed_jobs": 0,
                "release_ready": False,
                "gpu_inference_required": True,
                "expected_acceleration_mode": "gpu_inference_then_cpu_dsp_scoring",
            },
            "manifest": {"release_ready": False, "delivery_allowed": False},
        }

    monkeypatch.setattr("tools.run_no_reference_nonvocal_bounded_wave.execute_job_matrix", fake_execute_job_matrix)
    output_root = tmp_path / "wave"

    summary = run_wave(
        job_matrix_jsonl=job_matrix,
        output_root=output_root,
        existing_listen_roots=[],
        max_tracks=1,
        execute_generation=True,
    )

    assert summary["wave_status"] == "completed"
    assert summary["generation_started"] is True
    assert summary["listen_preview_count"] == 4
    assert summary["semantic_batch"]["track_count"] == 1
    assert summary["semantic_batch"]["candidate_count"] == 4
    assert summary["release_ready"] is False
    assert summary["generation_scope"] == "selected_tracks_only"
    assert summary["semantic_batch_scope"] == "cumulative_existing_plus_new_listen_roots"
    progress_events = [
        json.loads(line)["event"]
        for line in (output_root / "no_reference_nonvocal_bounded_wave_progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert progress_events == [
        "no_reference_nonvocal_bounded_wave_start",
        "generation_start",
        "generation_end",
        "listen_preview_start",
        "listen_preview_end",
        "semantic_batch_start",
        "semantic_batch_end",
        "no_reference_nonvocal_bounded_wave_end",
    ]
    assert (output_root / "listen_preview" / "track-a" / "bass" / "no_reference_stem_listen_preview_manifest.json").exists()
    assert (output_root / "semantic_batch" / "track-a" / "stem_score_matrix" / "no_reference_stem_candidate_score_matrix.json").exists()


def test_wave_execute_reports_cumulative_semantic_scope(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    job_matrix = write_jsonl(tmp_path / "jobs.jsonl", job_rows("track-a", source_root))
    for stem, freq in {"bass": 80.0, "drums": 80.0, "melody": 440.0, "other": 330.0}.items():
        write_stem_manifest(source_root / "track-a" / stem / "stem_manifest.json", track_id="track-a", stem=stem, freq_hz=freq)

    existing_source = tmp_path / "existing-source"
    existing_manifest = write_stem_manifest(
        existing_source / "track-existing" / "bass" / "stem_manifest.json",
        track_id="track-existing",
        stem="bass",
        freq_hz=80.0,
    )
    existing_listen = bounded_wave.build_listen_preview_manifest(
        stem_manifest_json=existing_manifest,
        target_stem="bass",
        track_id="track-existing",
    )
    existing_listen_root = tmp_path / "existing-listen"
    bounded_wave.write_listen_preview_json(
        existing_listen_root / "track-existing" / "bass" / "no_reference_stem_listen_preview_manifest.json",
        existing_listen,
    )

    def fake_execute_job_matrix(**kwargs):
        rows = []
        for stem in ("bass", "drums", "melody", "other"):
            manifest = source_root / "track-a" / stem / "stem_manifest.json"
            rows.append(
                {
                    "track_id": "track-a",
                    "target_stem": stem,
                    "execution_status": "completed",
                    "candidate_manifest": {"exists": True, "path": str(manifest), "target_stem": stem},
                }
            )
        return {
            "rows": rows,
            "summary": {"selected_jobs": 4, "completed_jobs": 4, "failed_jobs": 0, "release_ready": False},
            "manifest": {"release_ready": False, "delivery_allowed": False},
        }

    monkeypatch.setattr("tools.run_no_reference_nonvocal_bounded_wave.execute_job_matrix", fake_execute_job_matrix)

    summary = run_wave(
        job_matrix_jsonl=job_matrix,
        output_root=tmp_path / "wave",
        existing_listen_roots=[existing_listen_root],
        max_tracks=1,
        execute_generation=True,
    )

    assert summary["selected_track_count"] == 1
    assert summary["semantic_batch_scope"] == "cumulative_existing_plus_new_listen_roots"
    assert summary["semantic_batch_track_filter"] is None
    assert summary["semantic_batch"]["track_count"] == 2
    assert summary["semantic_batch"]["candidate_count"] == 5


def test_cli_dry_run_writes_summary(tmp_path: Path) -> None:
    job_matrix = write_jsonl(tmp_path / "jobs.jsonl", job_rows("track-a", tmp_path / "out"))
    output_root = tmp_path / "wave"

    exit_code = main(
        [
            "--job-matrix-jsonl",
            str(job_matrix),
            "--output-root",
            str(output_root),
            "--max-tracks",
            "1",
        ]
    )

    assert exit_code == 0
    summary = json.loads((output_root / "no_reference_nonvocal_bounded_wave_summary.json").read_text(encoding="utf-8"))
    assert summary["wave_status"] == "dry_run"
    assert summary["release_ready"] is False


def test_cli_requires_docker_runtime(tmp_path: Path, monkeypatch) -> None:
    job_matrix = write_jsonl(tmp_path / "jobs.jsonl", job_rows("track-a", tmp_path / "out"))
    output_root = tmp_path / "wave"
    calls = []

    def fake_require_docker_runtime(*, tool_name: str, entrypoint_path: str) -> None:
        calls.append({"tool_name": tool_name, "entrypoint_path": entrypoint_path})

    monkeypatch.setattr(bounded_wave, "require_docker_runtime", fake_require_docker_runtime)

    exit_code = bounded_wave.main(
        [
            "--job-matrix-jsonl",
            str(job_matrix),
            "--output-root",
            str(output_root),
            "--max-tracks",
            "1",
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "tool_name": "run_no_reference_nonvocal_bounded_wave.py",
            "entrypoint_path": "/app/tools/run_no_reference_nonvocal_bounded_wave.py",
        }
    ]
