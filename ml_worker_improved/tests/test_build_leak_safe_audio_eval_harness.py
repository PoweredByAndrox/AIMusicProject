import json
import math
import wave
from pathlib import Path

import numpy as np


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_wave(path: Path, *, frequency: float, amplitude: float = 0.35, seconds: float = 0.25, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    for index in range(int(sample_rate * seconds)):
        value = amplitude * math.sin(2.0 * math.pi * frequency * index / sample_rate)
        samples.append(int(max(-1.0, min(1.0, value)) * 32767.0))
    array = np.asarray(samples, dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(array.tobytes())


def view_row(
    tmp_path: Path,
    *,
    track_id: str,
    candidate_id: str,
    target: str,
    route_id: str,
    label: int,
    blockers: list[str] | None = None,
    frequency: float = 440.0,
    window_start_sec: float = 0.0,
) -> dict:
    mix_path = tmp_path / "audio" / f"{track_id}_mix.wav"
    candidate_path = tmp_path / "audio" / f"{candidate_id}_{target}.wav"
    if not mix_path.exists():
        write_wave(mix_path, frequency=frequency)
    if not candidate_path.exists():
        write_wave(candidate_path, frequency=frequency + (10.0 if target == "minus" else 20.0))
    return {
        "row_id": f"{target}-{track_id}-{candidate_id}-{window_start_sec}",
        "track_id": track_id,
        "run_id": f"run-{track_id}",
        "candidate_id": candidate_id,
        "route_id": route_id,
        "candidate_family": route_id.split("__", 1)[0],
        "source_model_family": route_id.split("__", 1)[0],
        "stem_type": target,
        "window_start_sec": window_start_sec,
        "window_end_sec": window_start_sec + 12.0,
        "candidate_audio_path": str(candidate_path),
        "mix_path": str(mix_path),
        "label": label,
        "label_source": "windowed_judge_exact",
        "blocker_labels": blockers or [],
        "safe_for_this_view": True,
        "release_ready": False,
    }


def build_views(tmp_path: Path) -> Path:
    views_dir = tmp_path / "views"
    minus_rows: list[dict] = []
    vocal_rows: list[dict] = []
    for index in range(8):
        label = 0 if index in {2, 5} else 1
        blockers = ["severe_phase_collapse", "severe_vocal_bleed_in_minus"] if label == 0 else []
        track_id = f"track-{index}"
        route_id = "htdemucs_ft__balanced" if index % 2 == 0 else "scnet__balanced"
        minus_rows.append(
            view_row(
                tmp_path,
                track_id=track_id,
                candidate_id=f"cand-{index}",
                target="minus",
                route_id=route_id,
                label=label,
                blockers=blockers,
                frequency=320.0 + index,
            )
        )
        vocal_rows.append(
            view_row(
                tmp_path,
                track_id=track_id,
                candidate_id=f"cand-{index}",
                target="vocal",
                route_id=route_id,
                label=1,
                frequency=520.0 + index,
            )
        )
    write_jsonl(views_dir / "minus_critic_training_view.jsonl", minus_rows)
    write_jsonl(views_dir / "vocal_critic_training_view.jsonl", vocal_rows)
    write_json(views_dir / "audio_error_training_views_manifest.json", {"schema": "audio_error_training_views.v1"})
    return views_dir


def test_audio_fingerprint_has_exact_and_near_duplicate_hashes(tmp_path: Path) -> None:
    from tools.build_leak_safe_audio_eval_harness import extract_audio_fingerprint

    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    write_wave(first, frequency=440.0, amplitude=0.25)
    write_wave(second, frequency=440.0, amplitude=0.60)

    first_fp = extract_audio_fingerprint(first)
    second_fp = extract_audio_fingerprint(second)

    assert first_fp["fingerprint_status"] == "ok"
    assert second_fp["fingerprint_status"] == "ok"
    assert first_fp["exact_audio_sha256"] != second_fp["exact_audio_sha256"]
    assert first_fp["near_duplicate_audio_hash"] == second_fp["near_duplicate_audio_hash"]
    assert first_fp["release_ready"] is False


def test_builds_locked_split_with_zero_critical_leakage(tmp_path: Path) -> None:
    from tools.build_leak_safe_audio_eval_harness import build_leak_safe_audio_eval_harness

    views_dir = build_views(tmp_path)
    output_dir = tmp_path / "leak_safe_eval"

    manifest = build_leak_safe_audio_eval_harness(views_dir=views_dir, output_dir=output_dir, model_dir=None)

    assert manifest["schema"] == "leak_safe_audio_eval_harness.v1"
    assert manifest["release_ready"] is False
    assert manifest["training_started"] is False
    assert manifest["separator_training_started"] is False
    assert manifest["critical_leakage_count"] == 0
    assert manifest["split_counts"]["train"] > 0
    assert manifest["split_counts"]["test"] > 0
    assert Path(manifest["leak_safe_split_manifest_path"]).exists()
    assert (output_dir / "audio_fingerprint_index.jsonl").exists()
    assert (output_dir / "leakage_report.json").exists()
    assert (output_dir / "hardcase_holdout_manifest.json").exists()
    assert (output_dir / "locked_eval_summary.md").exists()

    leakage = read_json(output_dir / "leakage_report.json")
    assert leakage["critical_leakage_count"] == 0
    assert leakage["checks"]["track_id_overlap"]["critical"] is False
    assert leakage["checks"]["candidate_id_overlap"]["critical"] is False
    assert leakage["checks"]["window_overlap"]["critical"] is False
    assert "route_id_overlap" in leakage["checks"]
    assert "candidate_family_overlap" in leakage["checks"]
    assert "source_model_family_overlap" in leakage["checks"]

    hardcase = read_json(output_dir / "hardcase_holdout_manifest.json")
    assert hardcase["locked"] is True
    assert hardcase["release_ready"] is False
    assert hardcase["hardcase_rows_count"] > 0
    assert all(row["train_allowed"] is False for row in hardcase["hardcase_rows"])

    split_rows = read_jsonl(output_dir / "leak_safe_split_rows.jsonl")
    split_by_track: dict[str, set[str]] = {}
    for row in split_rows:
        split_by_track.setdefault(row["track_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in split_by_track.values())


def test_detect_leakage_reports_candidate_audio_and_window_overlap(tmp_path: Path) -> None:
    from tools.build_leak_safe_audio_eval_harness import detect_leakage

    audio_path = tmp_path / "candidate.wav"
    mix_path = tmp_path / "mix.wav"
    write_wave(audio_path, frequency=440.0)
    write_wave(mix_path, frequency=550.0)
    base = {
        "row_uid": "row-a",
        "track_id": "track-a",
        "candidate_id": "candidate-a",
        "route_id": "route-a",
        "candidate_family": "family-a",
        "source_model_family": "model-a",
        "candidate_audio_path": str(audio_path),
        "mix_path": str(mix_path),
        "input_fingerprint": "mix-fp",
        "candidate_exact_audio_sha256": "candidate-exact",
        "candidate_near_duplicate_audio_hash": "candidate-near",
        "window_start_sec": 0.0,
        "window_end_sec": 12.0,
        "split": "train",
    }
    other = {**base, "row_uid": "row-b", "split": "test", "window_start_sec": 6.0, "window_end_sec": 18.0}

    report = detect_leakage([base, other])

    assert report["critical_leakage_count"] >= 1
    assert report["checks"]["same_candidate_train_test"]["critical"] is True
    assert report["checks"]["candidate_audio_hash_overlap"]["critical"] is True
    assert report["checks"]["near_duplicate_audio_hash_overlap"]["critical"] is True
    assert report["checks"]["window_overlap"]["critical"] is True


def test_model2c_locked_eval_is_diagnostic_when_model_dir_missing(tmp_path: Path) -> None:
    from tools.build_leak_safe_audio_eval_harness import build_leak_safe_audio_eval_harness

    manifest = build_leak_safe_audio_eval_harness(views_dir=build_views(tmp_path), output_dir=tmp_path / "out", model_dir=None)

    re_eval = manifest["model_2c_locked_re_eval"]
    assert re_eval["status"] == "skipped_missing_model_dir"
    assert re_eval["training_started"] is False
    assert re_eval["separator_training_started"] is False
    assert re_eval["release_ready"] is False


def test_locked_eval_metrics_accept_structured_model2c_predictions() -> None:
    from tools.build_leak_safe_audio_eval_harness import _metrics_from_predictions

    rows = [{"label": 0}, {"label": 1}, {"label": 1}]
    predictions = [
        {"predicted_accept_probability": 0.8},
        {"predicted_accept_probability": 0.7},
        {"predicted_accept_probability": 0.2},
    ]

    metrics = _metrics_from_predictions(rows, predictions)

    assert metrics["false_accept"] == 1
    assert metrics["false_reject"] == 1
    assert metrics["false_accept_rate"] == 1.0
    assert metrics["false_reject_rate"] == 0.5
