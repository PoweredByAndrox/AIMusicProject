from __future__ import annotations

import json
from pathlib import Path

from tools.build_testing_stuff_music_corpus_vocal_judge_inputs import build_inputs, main


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def execution_row(
    tmp_path: Path,
    *,
    track_id: str = "track_a",
    target_stem: str,
    model_id: str,
    status: str = "completed",
) -> dict:
    audio_path = tmp_path / track_id / target_stem / model_id / f"{target_stem}.flac"
    manifest_path = tmp_path / track_id / target_stem / model_id / "stem_manifest.json"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"audio")
    write_json(
        manifest_path,
        {
            "model_name": model_id,
            "target_stem": target_stem,
            "execution_contract": {
                "expected_acceleration_mode": "gpu_inference_then_cpu_dsp_scoring",
                "gpu_inference_required": True,
                "gpu_inference_skipped": False,
                "cuda_preflight": {"cuda_available": True, "device_name": "NVIDIA test GPU"},
            },
            "delivery_allowed": False,
            "release_ready": False,
        },
    )
    return {
        "track_id": track_id,
        "source_audio_path": f"/host/{track_id}.mp3",
        "target_stem": target_stem,
        "model_id": model_id,
        "route_family": "test",
        "execution_status": status,
        "execution_started": status == "completed",
        "candidate_manifest": {
            "exists": status == "completed",
            "path": str(manifest_path),
            "target_audio_path": str(audio_path),
            "target_stem": target_stem,
        },
        "delivery_allowed": False,
        "release_ready": False,
    }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_builds_vocal_windowed_judge_candidate_rows_from_completed_vocal_minus_execution(tmp_path: Path) -> None:
    execution_jsonl = write_jsonl(
        tmp_path / "execution.jsonl",
        [
            execution_row(tmp_path, target_stem="vocal", model_id="vocal_a"),
            execution_row(tmp_path, target_stem="minus", model_id="minus_a"),
        ],
    )

    result = build_inputs(execution_results_jsonls=[execution_jsonl])

    assert result["summary"]["candidate_count"] == 1
    assert result["summary"]["paired_track_count"] == 1
    assert result["summary"]["release_ready"] is False
    row = result["rows"][0]
    assert row["track_id"] == "track_a"
    assert row["candidate_id"] == "vocal_a__minus_a"
    assert row["mix_path"] == "/host/track_a.mp3"
    assert row["vocal_path"].endswith("vocal.flac")
    assert row["instrumental_path"].endswith("minus.flac")
    assert row["minus_path"] == row["instrumental_path"]
    assert row["candidate_manifest"].endswith("stem_manifest.json")
    assert row["score_mode"] == "vocal"
    assert row["delivery_allowed"] is False
    assert row["release_ready"] is False
    assert row["forbidden_actions"] == ["present_as_listen_result", "mark_release_ready", "start_training"]


def test_blocks_unpaired_tracks_without_fallback_pairing(tmp_path: Path) -> None:
    execution_jsonl = write_jsonl(
        tmp_path / "execution.jsonl",
        [execution_row(tmp_path, target_stem="vocal", model_id="vocal_a")],
    )

    result = build_inputs(execution_results_jsonls=[execution_jsonl])

    assert result["summary"]["candidate_count"] == 0
    assert result["summary"]["unpaired_vocal_track_count"] == 1
    assert result["summary"]["unpaired_minus_track_count"] == 0
    assert result["summary"]["delivery_allowed"] is False
    assert result["summary"]["release_ready"] is False
    assert result["rows"] == []


def test_cli_writes_candidate_jsonl_manifest_and_summary(tmp_path: Path) -> None:
    execution_jsonl = write_jsonl(
        tmp_path / "execution.jsonl",
        [
            execution_row(tmp_path, target_stem="vocal", model_id="vocal_a"),
            execution_row(tmp_path, target_stem="minus", model_id="minus_a"),
        ],
    )
    output_dir = tmp_path / "out"

    assert main(["--execution-results-jsonl", str(execution_jsonl), "--output-dir", str(output_dir)]) == 0

    rows = read_jsonl(output_dir / "no_reference_vocal_windowed_judge_candidates.jsonl")
    manifest = read_json(output_dir / "no_reference_vocal_windowed_judge_candidates_manifest.json")
    assert len(rows) == 1
    assert manifest["summary"]["candidate_count"] == 1
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert (output_dir / "no_reference_vocal_windowed_judge_candidates_summary.md").exists()
