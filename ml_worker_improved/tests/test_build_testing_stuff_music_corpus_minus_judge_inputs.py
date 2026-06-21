from __future__ import annotations

import json
from pathlib import Path

from tools.build_testing_stuff_music_corpus_minus_judge_inputs import build_inputs, main


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
    target_stem: str = "minus",
    model_id: str = "minus_a",
    route_family: str = "test",
    output_subdir: str | None = None,
    status: str = "completed",
    include_vocal_companion: bool = True,
    include_minus_stem: bool = True,
) -> dict:
    audio_dir = tmp_path / track_id / target_stem / (output_subdir or model_id)
    minus_path = audio_dir / "minus.flac"
    vocal_path = audio_dir / "vocals.flac"
    melody_path = audio_dir / "melody.flac"
    manifest_path = audio_dir / "stem_manifest.json"
    audio_dir.mkdir(parents=True, exist_ok=True)
    if include_minus_stem:
        minus_path.write_bytes(b"minus")
    if include_vocal_companion:
        vocal_path.write_bytes(b"vocals")
    melody_path.write_bytes(b"melody")
    stems = {"melody": {"path": str(melody_path)}}
    if include_minus_stem:
        stems["minus"] = {"path": str(minus_path)}
        stems["instrumental"] = {"path": str(minus_path)}
    if include_vocal_companion:
        stems["vocals"] = {"path": str(vocal_path)}
    target_audio_path = vocal_path if target_stem in {"vocal", "vocals"} else minus_path
    write_json(
        manifest_path,
        {
            "model_name": model_id,
            "audio_path": f"/host/{track_id}.mp3",
            "target_stem": target_stem,
            "stems": stems,
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
        "route_family": route_family,
        "execution_status": status,
        "execution_started": status == "completed",
        "candidate_manifest": {
            "exists": status == "completed",
            "path": str(manifest_path),
            "target_audio_path": str(target_audio_path),
            "target_stem": target_stem,
        },
        "delivery_allowed": False,
        "release_ready": False,
    }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_builds_minus_windowed_judge_candidate_rows_from_completed_minus_execution(tmp_path: Path) -> None:
    execution_jsonl = write_jsonl(tmp_path / "execution.jsonl", [execution_row(tmp_path)])

    result = build_inputs(execution_results_jsonls=[execution_jsonl])

    assert result["summary"]["candidate_count"] == 1
    assert result["summary"]["skipped_candidate_count"] == 0
    assert result["summary"]["release_ready"] is False
    row = result["rows"][0]
    assert row["schema"] == "testing_stuff_music_minus_windowed_judge_candidate_row.v1"
    assert row["track_id"] == "track_a"
    assert row["candidate_id"] == "track_a__minus__minus_a"
    assert row["mix_path"] == "/host/track_a.mp3"
    assert row["minus_path"].endswith("minus.flac")
    assert row["instrumental_path"] == row["minus_path"]
    assert row["selected_minus_path"] == row["minus_path"]
    assert row["vocal_path"].endswith("vocals.flac")
    assert row["selected_vocal_path"] == row["vocal_path"]
    assert row["stem_paths"]["melody"].endswith("melody.flac")
    assert row["candidate_manifest"].endswith("stem_manifest.json")
    assert row["score_mode"] == "minus"
    assert row["delivery_allowed"] is False
    assert row["release_ready"] is False


def test_skips_minus_candidate_without_vocal_companion(tmp_path: Path) -> None:
    execution_jsonl = write_jsonl(
        tmp_path / "execution.jsonl",
        [execution_row(tmp_path, include_vocal_companion=False)],
    )

    result = build_inputs(execution_results_jsonls=[execution_jsonl])

    assert result["rows"] == []
    assert result["summary"]["candidate_count"] == 0
    assert result["summary"]["skipped_candidate_count"] == 1
    assert result["summary"]["skipped_candidates"][0]["skip_reason"] == "missing_vocal_companion_path"
    assert result["summary"]["delivery_allowed"] is False
    assert result["summary"]["release_ready"] is False


def test_builds_minus_rows_from_vocal_instrumental_execution_manifest(tmp_path: Path) -> None:
    execution_jsonl = write_jsonl(
        tmp_path / "execution.jsonl",
        [execution_row(tmp_path, target_stem="vocal", model_id="melband_vocals")],
    )

    result = build_inputs(execution_results_jsonls=[execution_jsonl])

    assert result["summary"]["candidate_count"] == 1
    assert result["summary"]["skipped_candidate_count"] == 0
    row = result["rows"][0]
    assert row["candidate_id"] == "track_a__minus__melband_vocals"
    assert row["minus_path"].endswith("minus.flac")
    assert row["selected_minus_path"] == row["minus_path"]
    assert row["vocal_path"].endswith("vocals.flac")
    assert row["stem_type"] == "minus"
    assert row["score_mode"] == "minus"
    assert row["delivery_allowed"] is False
    assert row["release_ready"] is False


def test_pairs_minus_candidate_with_separate_vocal_companion(tmp_path: Path) -> None:
    execution_jsonl = write_jsonl(
        tmp_path / "execution.jsonl",
        [
            execution_row(tmp_path, target_stem="minus", model_id="ir_minus", include_vocal_companion=False),
            execution_row(tmp_path, target_stem="vocal", model_id="htdemucs_6s", include_minus_stem=False),
        ],
    )

    result = build_inputs(execution_results_jsonls=[execution_jsonl])

    assert result["summary"]["candidate_count"] == 1
    assert result["summary"]["skipped_candidate_count"] == 0
    row = result["rows"][0]
    assert row["candidate_id"] == "track_a__minus__ir_minus"
    assert row["minus_path"].endswith("minus.flac")
    assert row["selected_minus_path"] == row["minus_path"]
    assert row["vocal_path"].endswith("vocals.flac")
    assert row["selected_vocal_path"] == row["vocal_path"]
    assert row["minus_path"] != row["vocal_path"]
    assert row["candidate_manifest"].endswith("minus/ir_minus/stem_manifest.json")


def test_pairs_minus_candidate_with_all_stems_companion_manifest(tmp_path: Path) -> None:
    execution_jsonl = write_jsonl(
        tmp_path / "execution.jsonl",
        [
            execution_row(tmp_path, target_stem="minus", model_id="ir_minus", include_vocal_companion=False),
            execution_row(tmp_path, target_stem="minus", model_id="htdemucs_6s", include_vocal_companion=True),
        ],
    )

    result = build_inputs(execution_results_jsonls=[execution_jsonl])

    candidate_ids = [row["candidate_id"] for row in result["rows"]]
    assert "track_a__minus__ir_minus" in candidate_ids
    ir_row = next(row for row in result["rows"] if row["candidate_id"] == "track_a__minus__ir_minus")
    assert ir_row["minus_path"].endswith("minus/ir_minus/minus.flac")
    assert ir_row["vocal_path"].endswith("minus/htdemucs_6s/vocals.flac")
    assert ir_row["minus_path"] != ir_row["vocal_path"]
    assert result["summary"]["delivery_allowed"] is False
    assert result["summary"]["release_ready"] is False


def test_pairs_minus_candidate_with_deterministic_vocal_companion(tmp_path: Path) -> None:
    execution_jsonl = write_jsonl(
        tmp_path / "execution.jsonl",
        [
            execution_row(tmp_path, target_stem="minus", model_id="ir_minus", include_vocal_companion=False),
            execution_row(tmp_path, target_stem="vocal", model_id="z_vocal", include_minus_stem=False),
            execution_row(tmp_path, target_stem="vocal", model_id="a_vocal", include_minus_stem=False),
        ],
    )

    result = build_inputs(execution_results_jsonls=[execution_jsonl])

    row = result["rows"][0]
    assert row["vocal_path"].endswith("vocal/a_vocal/vocals.flac")


def test_candidate_ids_distinguish_same_model_different_minus_routes(tmp_path: Path) -> None:
    execution_jsonl = write_jsonl(
        tmp_path / "execution.jsonl",
        [
            execution_row(
                tmp_path,
                model_id="instrumental_resurrection_minus",
                route_family="direct_clean_minus_baseline",
                output_subdir="direct",
            ),
            execution_row(
                tmp_path,
                model_id="instrumental_resurrection_minus",
                route_family="cpu_dsp_minus_phase_repair",
                output_subdir="phase_repair",
            ),
        ],
    )

    result = build_inputs(execution_results_jsonls=[execution_jsonl])

    candidate_ids = [row["candidate_id"] for row in result["rows"]]
    assert len(candidate_ids) == 2
    assert len(set(candidate_ids)) == 2
    assert any(candidate_id.endswith("__cpu_dsp_minus_phase_repair") for candidate_id in candidate_ids)


def test_does_not_turn_vocal_target_audio_into_minus_candidate(tmp_path: Path) -> None:
    execution_jsonl = write_jsonl(
        tmp_path / "execution.jsonl",
        [execution_row(tmp_path, target_stem="vocal", model_id="htdemucs_6s", include_minus_stem=False)],
    )

    result = build_inputs(execution_results_jsonls=[execution_jsonl])

    assert result["rows"] == []
    assert result["summary"]["candidate_count"] == 0
    assert result["summary"]["skipped_candidate_count"] == 0
    assert result["summary"]["delivery_allowed"] is False
    assert result["summary"]["release_ready"] is False


def test_cli_writes_minus_candidate_jsonl_manifest_and_summary(tmp_path: Path) -> None:
    execution_jsonl = write_jsonl(tmp_path / "execution.jsonl", [execution_row(tmp_path)])
    output_dir = tmp_path / "out"

    assert main(["--execution-results-jsonl", str(execution_jsonl), "--output-dir", str(output_dir)]) == 0

    rows = read_jsonl(output_dir / "no_reference_minus_windowed_judge_candidates.jsonl")
    manifest = read_json(output_dir / "no_reference_minus_windowed_judge_candidates_manifest.json")
    assert len(rows) == 1
    assert manifest["summary"]["candidate_count"] == 1
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert (output_dir / "no_reference_minus_windowed_judge_candidates_summary.md").exists()
