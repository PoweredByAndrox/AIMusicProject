from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.run_no_reference_stem_semantic_batch import run_batch, main


def write_audio(path: Path, *, freq_hz: float = 220.0, amplitude: float = 0.08) -> Path:
    sample_rate = 16_000
    t = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
    mono = np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32) * amplitude
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.column_stack([mono, mono]), sample_rate)
    return path


def write_listen_manifest(root: Path, *, track_id: str, stem_type: str, audio_path: Path) -> Path:
    manifest = root / track_id / stem_type / "no_reference_stem_listen_preview_manifest.json"
    payload = {
        "manifest_type": "no_reference_stem_listen_preview",
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "stem_type": stem_type,
        "track_id": track_id,
        "model_name": "unit_model",
        "selected_audio_path": str(audio_path),
        "audio_integrity_gate": {"status": "passed", "duration_sec": 2.0, "rms": 0.05, "peak": 0.5},
        "gpu_provenance_gate": {
            "status": "passed",
            "gpu_inference_required": True,
            "gpu_inference_skipped": False,
            "cuda_available": True,
            "resolved_device": "cuda",
        },
        "source_contract_gate": {"status": "passed"},
        "semantic_quality_verified": False,
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": {
            "training_started": False,
            "fine_tune_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "cleanup_routes_added": False,
            "thresholds_relaxed": False,
            "release_ready": False,
        },
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def test_batch_runs_semantic_judge_and_matrix_per_track(tmp_path: Path) -> None:
    listen_root = tmp_path / "listen"
    audio_root = tmp_path / "audio"
    write_listen_manifest(listen_root, track_id="track-a", stem_type="bass", audio_path=write_audio(audio_root / "a_bass.flac", freq_hz=80.0))
    write_listen_manifest(listen_root, track_id="track-a", stem_type="melody", audio_path=write_audio(audio_root / "a_melody.flac", freq_hz=440.0))
    write_listen_manifest(listen_root, track_id="track-b", stem_type="melody", audio_path=write_audio(audio_root / "b_melody.flac", freq_hz=330.0))
    output_root = tmp_path / "batch"

    summary = run_batch(listen_manifest_roots=[listen_root], output_root=output_root)

    assert summary["track_count"] == 2
    assert summary["candidate_count"] == 3
    assert summary["release_ready"] is False
    assert summary["semantic_quality_verified_count"] == 3
    assert (output_root / "track-a" / "stem_semantic_judge" / "no_reference_stem_semantic_judge.jsonl").exists()
    assert (output_root / "track-a" / "stem_score_matrix" / "no_reference_stem_candidate_score_matrix.json").exists()
    assert (output_root / "track-b" / "stem_score_matrix" / "no_reference_stem_candidate_score_matrix.json").exists()
    assert not list(output_root.rglob("*.wav"))
    assert not list(output_root.rglob("*.flac"))


def test_batch_track_filter_limits_processed_groups(tmp_path: Path) -> None:
    listen_root = tmp_path / "listen"
    audio_root = tmp_path / "audio"
    write_listen_manifest(listen_root, track_id="track-a", stem_type="bass", audio_path=write_audio(audio_root / "a_bass.flac", freq_hz=80.0))
    write_listen_manifest(listen_root, track_id="track-b", stem_type="melody", audio_path=write_audio(audio_root / "b_melody.flac", freq_hz=440.0))
    output_root = tmp_path / "batch"

    summary = run_batch(listen_manifest_roots=[listen_root], output_root=output_root, track_ids=["track-b"])

    assert summary["track_count"] == 1
    assert summary["candidate_count"] == 1
    assert summary["tracks"][0]["track_id"] == "track-b"
    assert not (output_root / "track-a").exists()
    assert (output_root / "track-b" / "stem_score_matrix" / "no_reference_stem_candidate_score_matrix.json").exists()


def test_batch_empty_root_is_blocked_not_release_ready(tmp_path: Path) -> None:
    summary = run_batch(listen_manifest_roots=[tmp_path / "missing"], output_root=tmp_path / "batch")

    assert summary["batch_status"] == "blocked_no_selected_manifests"
    assert summary["track_count"] == 0
    assert summary["candidate_count"] == 0
    assert summary["release_ready"] is False
    assert summary["safety"]["audio_exported"] is False


def test_cli_writes_summary_progress_and_no_audio(tmp_path: Path) -> None:
    listen_root = tmp_path / "listen"
    audio_root = tmp_path / "audio"
    write_listen_manifest(listen_root, track_id="track-a", stem_type="bass", audio_path=write_audio(audio_root / "a_bass.flac", freq_hz=80.0))
    output_root = tmp_path / "batch"

    exit_code = main(["--listen-manifest-root", str(listen_root), "--output-root", str(output_root)])

    assert exit_code == 0
    summary_json = output_root / "no_reference_stem_semantic_batch_summary.json"
    progress_jsonl = output_root / "no_reference_stem_semantic_batch_progress.jsonl"
    summary_md = output_root / "no_reference_stem_semantic_batch_summary.md"
    assert summary_json.exists()
    assert progress_jsonl.exists()
    assert summary_md.exists()
    assert json.loads(summary_json.read_text(encoding="utf-8"))["release_ready"] is False
    assert "no_reference_stem_semantic_batch_start" in progress_jsonl.read_text(encoding="utf-8")
    assert not list(output_root.rglob("*.wav"))
    assert not list(output_root.rglob("*.flac"))


def test_file_script_entrypoint_runs_with_tools_imports(tmp_path: Path) -> None:
    listen_root = tmp_path / "listen"
    audio_root = tmp_path / "audio"
    write_listen_manifest(listen_root, track_id="track-a", stem_type="melody", audio_path=write_audio(audio_root / "melody.flac", freq_hz=440.0))
    output_root = tmp_path / "batch"
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "tools/run_no_reference_stem_semantic_batch.py",
            "--listen-manifest-root",
            str(listen_root),
            "--output-root",
            str(output_root),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_root / "no_reference_stem_semantic_batch_summary.json").exists()
