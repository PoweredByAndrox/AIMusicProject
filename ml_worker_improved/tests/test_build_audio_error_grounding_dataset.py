import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.build_audio_error_grounding_dataset import build_audio_error_grounding_dataset


SAMPLE_RATE = 8_000


def sine(freq: float, seconds: float, amp: float = 0.25) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def stereo(audio: np.ndarray) -> np.ndarray:
    return np.column_stack([audio, audio * 0.8]).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return str(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def window_metric(*, start: float, duration: float, decision: str, blockers: list[str]) -> dict:
    return {
        "kind": "grid_0000",
        "start_sec": start,
        "duration_sec": duration,
        "decision": decision,
        "vocal_decision": decision,
        "minus_decision": decision,
        "vocal_blockers": blockers if "vocal" in " ".join(blockers) else [],
        "minus_blockers": blockers if "minus" in " ".join(blockers) or "phase" in " ".join(blockers) else [],
        "reasons": blockers,
        "pair_decision": "pair_compatible",
        "pair_compatibility_status": "pair_compatible",
        "mixture_consistency_score": 0.92,
        "phase_consistency_score": 0.83,
        "vocal_leakage_in_minus_score": 0.78 if blockers else 0.04,
        "instrumental_leakage_in_vocal_score": 0.03,
        "transient_damage_score": 0.02,
        "robotic_artifact_score": 0.01,
        "window_fitness": 0.91 if not blockers else 0.23,
    }


def create_run(root: Path, *, missing_minus: bool = False) -> Path:
    run_dir = root / "run-a"
    vocal = sine(440, 1.5, amp=0.18)
    minus = sine(120, 1.5, amp=0.24)
    mix = vocal + minus
    mix_path = write_wav(run_dir / "audio" / "mix.wav", stereo(mix))
    vocal_path = write_wav(run_dir / "audio" / "vocal.wav", stereo(vocal))
    minus_path = str(run_dir / "audio" / "missing-minus.wav") if missing_minus else write_wav(
        run_dir / "audio" / "minus.wav",
        stereo(minus),
    )
    candidate = {
        "track_id": "track-a",
        "run_id": "run-a",
        "candidate_id": "cand-a",
        "route_id": "route-safe",
        "source_candidate_family": "registered_separator",
        "mix_path": mix_path,
        "input_audio_path": mix_path,
        "vocal_path": vocal_path,
        "minus_path": minus_path,
        "instrumental_path": minus_path,
        "is_complete_pair": True,
        "release_ready": False,
    }
    write_jsonl(run_dir / "candidate_pool.jsonl", [candidate])
    write_jsonl(
        run_dir / "vocal_windowed_judge.jsonl",
        [
            {
                **candidate,
                "score_mode": "vocal",
                "decision": "accept_candidate",
                "vocal_decision": "accept_candidate",
                "vocal_blockers": [],
                "window_metrics": [
                    window_metric(start=0.0, duration=0.5, decision="accept_candidate", blockers=[]),
                ],
            }
        ],
    )
    write_jsonl(
        run_dir / "minus_windowed_judge.jsonl",
        [
            {
                **candidate,
                "score_mode": "minus",
                "decision": "blocked_hard_case",
                "minus_decision": "blocked_hard_case",
                "minus_blockers": ["severe_vocal_bleed_in_minus"],
                "window_metrics": [
                    window_metric(
                        start=0.5,
                        duration=0.5,
                        decision="blocked_hard_case",
                        blockers=["severe_vocal_bleed_in_minus"],
                    ),
                ],
            }
        ],
    )
    write_jsonl(
        run_dir / "pair_compatibility.jsonl",
        [
            {
                "track_id": "track-a",
                "candidate_id": "cand-a",
                "pair_decision": "pair_compatible",
                "pair_compatibility_status": "pair_compatible",
                "release_ready": False,
            }
        ],
    )
    write_json(run_dir / "preview_render_contract.json", {"release_ready": False, "audio_preview_ready": True})
    return run_dir


def test_candidate_config_id_is_used_as_explicit_route_provenance(tmp_path: Path) -> None:
    run_dir = create_run(tmp_path)
    for name in ("candidate_pool.jsonl", "vocal_windowed_judge.jsonl", "minus_windowed_judge.jsonl"):
        rows = read_jsonl(run_dir / name)
        for row in rows:
            row.pop("route_id", None)
            row["candidate_config_id"] = "local_registered__htdemucs_ft__both__shared"
        write_jsonl(run_dir / name, rows)
    output_dir = tmp_path / "dataset"

    manifest = build_audio_error_grounding_dataset(
        run_roots=[run_dir],
        output_dir=output_dir,
        window_sec=0.5,
        hop_sec=0.5,
        map_max_bins=16,
        map_max_frames=8,
        max_waveform_samples=64,
    )

    rows = read_jsonl(output_dir / "audio_error_window_dataset.jsonl")
    assert rows
    assert manifest["safe_for_training_count"] == len(rows)
    assert all(row["route_id"] == "local_registered__htdemucs_ft__both__shared" for row in rows)
    assert all(row["safe_for_training"] is True for row in rows)
    assert all(row["error_map_paths"]["npz"].endswith(".npz") for row in rows)


def test_window_pair_decision_is_used_as_explicit_pair_provenance(tmp_path: Path) -> None:
    run_dir = create_run(tmp_path)
    write_jsonl(run_dir / "pair_compatibility.jsonl", [])
    output_dir = tmp_path / "dataset"

    manifest = build_audio_error_grounding_dataset(
        run_roots=[run_dir],
        output_dir=output_dir,
        window_sec=0.5,
        hop_sec=0.5,
        map_max_bins=16,
        map_max_frames=8,
        max_waveform_samples=64,
    )

    rows = read_jsonl(output_dir / "audio_error_window_dataset.jsonl")
    assert rows
    assert manifest["safe_for_training_count"] == len(rows)
    assert all(row["pair_compatibility_status"] == "pair_compatible" for row in rows)
    assert all(row["safe_for_training"] is True for row in rows)
    assert all(row["error_map_paths"]["npz"].endswith(".npz") for row in rows)


def test_build_audio_error_grounding_dataset_writes_rows_maps_and_manifest(tmp_path: Path) -> None:
    run_dir = create_run(tmp_path)
    output_dir = tmp_path / "dataset"

    manifest = build_audio_error_grounding_dataset(
        run_roots=[run_dir],
        output_dir=output_dir,
        window_sec=0.5,
        hop_sec=0.5,
        map_max_bins=32,
        map_max_frames=16,
        max_waveform_samples=128,
    )

    rows = read_jsonl(output_dir / "audio_error_window_dataset.jsonl")
    maps_index = read_jsonl(output_dir / "audio_error_maps_index.jsonl")
    assert (output_dir / "audio_error_grounding_manifest.json").exists()
    assert (output_dir / "audio_error_dataset_summary.md").exists()
    assert manifest["tracks_included"] == 1
    assert manifest["candidates_included"] == 1
    assert manifest["windows_total"] == 2
    assert manifest["accepted_windows"] == 1
    assert manifest["blocked_windows"] == 1
    assert manifest["vocal_bleed_windows"] == 1
    assert manifest["release_ready"] is False
    assert manifest["training_started"] is False
    assert manifest["melody_transcription_ran"] is False
    assert manifest["recommendation_index_updated"] is False

    assert {row["stem_type"] for row in rows} == {"vocal", "minus"}
    assert all(row["mix_path"] and row["vocal_candidate_path"] and row["minus_candidate_path"] for row in rows)
    assert all(row["window_end_sec"] > row["window_start_sec"] for row in rows)
    assert all(row["safe_for_training"] is True for row in rows)
    assert all(row["pair_compatibility_status"] == "pair_compatible" for row in rows)
    assert all(row["error_map_paths"]["npz"].endswith(".npz") for row in rows)
    assert all(Path(row["error_map_paths"]["npz"]).exists() for row in rows)
    assert len(maps_index) == 2


def assert_minus_grounding_uses_analysis_audio(tmp_path: Path, *, path_field: str) -> None:
    run_dir = create_run(tmp_path)
    (run_dir / "vocal_windowed_judge.jsonl").unlink()
    candidates = read_jsonl(run_dir / "candidate_pool.jsonl")
    analysis_path = candidates[0].pop("vocal_path")
    candidates[0].pop("selected_vocal_path", None)
    candidates[0][path_field] = analysis_path
    if path_field == "vocal_companion_path":
        candidates[0]["vocal_companion_role"] = "residual_companion"
        candidates[0]["vocal_companion_source"] = "residual_from_raw_minus_instrumental"
    write_jsonl(run_dir / "candidate_pool.jsonl", candidates)
    minus_rows = read_jsonl(run_dir / "minus_windowed_judge.jsonl")
    for row in minus_rows:
        row.pop("vocal_path", None)
        row.pop("selected_vocal_path", None)
        row[path_field] = analysis_path
        if path_field == "vocal_companion_path":
            row["vocal_companion_role"] = "residual_companion"
            row["vocal_companion_source"] = "residual_from_raw_minus_instrumental"
    write_jsonl(run_dir / "minus_windowed_judge.jsonl", minus_rows)
    output_dir = tmp_path / "dataset"

    manifest = build_audio_error_grounding_dataset(
        run_roots=[run_dir],
        output_dir=output_dir,
        window_sec=0.5,
        hop_sec=0.5,
        map_max_bins=16,
        map_max_frames=8,
        max_waveform_samples=64,
    )

    rows = read_jsonl(output_dir / "audio_error_window_dataset.jsonl")
    assert manifest["windows_total"] == 1
    assert manifest["safe_for_training_count"] == 1
    assert manifest["error_maps_written"] == 1
    assert rows[0]["safe_for_training"] is True
    assert rows[0]["unsafe_reason"] == ""
    assert rows[0]["vocal_candidate_path"] == analysis_path
    assert rows[0]["vocal_analysis_path"] == analysis_path
    if path_field == "vocal_companion_path":
        assert rows[0]["vocal_companion_path"] == analysis_path
        assert rows[0]["vocal_companion_role"] == "residual_companion"
        assert rows[0]["vocal_companion_source"] == "residual_from_raw_minus_instrumental"
    else:
        assert rows[0]["vocal_companion_path"] == ""
        assert rows[0]["vocal_companion_role"] == ""
        assert rows[0]["vocal_companion_source"] == ""
    assert rows[0]["error_map_paths"]["npz"].endswith(".npz")
    assert Path(rows[0]["error_map_paths"]["npz"]).exists()


def test_minus_grounding_uses_vocal_analysis_path(tmp_path: Path) -> None:
    assert_minus_grounding_uses_analysis_audio(tmp_path, path_field="vocal_analysis_path")


def test_minus_grounding_uses_vocal_companion_path(tmp_path: Path) -> None:
    assert_minus_grounding_uses_analysis_audio(tmp_path, path_field="vocal_companion_path")


def test_missing_audio_is_kept_but_marked_unsafe(tmp_path: Path) -> None:
    run_dir = create_run(tmp_path, missing_minus=True)
    output_dir = tmp_path / "dataset"

    manifest = build_audio_error_grounding_dataset(
        run_roots=[run_dir],
        output_dir=output_dir,
        window_sec=0.5,
        hop_sec=0.5,
    )

    rows = read_jsonl(output_dir / "audio_error_window_dataset.jsonl")
    assert rows
    assert all(row["safe_for_training"] is False for row in rows)
    assert all("missing_audio_file:minus_candidate_path" in row["unsafe_reason"] for row in rows)
    assert all(row["error_map_paths"] == {} for row in rows)
    assert manifest["unsafe_count"] == len(rows)
    assert manifest["error_maps_written"] == 0


def test_builder_discovers_nested_run_directories(tmp_path: Path) -> None:
    parent = tmp_path / "parent-root"
    nested = create_run(parent / "nested")
    output_dir = tmp_path / "dataset"

    manifest = build_audio_error_grounding_dataset(
        run_roots=[parent],
        output_dir=output_dir,
        window_sec=0.5,
        hop_sec=0.5,
        map_max_bins=16,
        map_max_frames=8,
        max_waveform_samples=64,
    )

    rows = read_jsonl(output_dir / "audio_error_window_dataset.jsonl")
    assert manifest["run_roots_expanded"] == [str(nested)]
    assert len(rows) == 2


def test_missing_candidate_id_is_not_replaced_with_synthetic_value(tmp_path: Path) -> None:
    run_dir = create_run(tmp_path)
    for name in ("candidate_pool.jsonl", "vocal_windowed_judge.jsonl", "minus_windowed_judge.jsonl"):
        rows = read_jsonl(run_dir / name)
        for row in rows:
            row.pop("candidate_id", None)
        write_jsonl(run_dir / name, rows)
    output_dir = tmp_path / "dataset"

    manifest = build_audio_error_grounding_dataset(
        run_roots=[run_dir],
        output_dir=output_dir,
        window_sec=0.5,
        hop_sec=0.5,
    )

    rows = read_jsonl(output_dir / "audio_error_window_dataset.jsonl")
    assert rows
    assert all(row["candidate_id"] == "" for row in rows)
    assert all(row["safe_for_training"] is False for row in rows)
    assert all("missing_candidate_id" in row["unsafe_reason"] for row in rows)
    assert all(row["error_map_paths"] == {} for row in rows)
    assert manifest["error_maps_written"] == 0
