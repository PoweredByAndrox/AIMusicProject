import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools import run_no_reference_minus_self_judge_matrix as matrix


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def candidate_row(tmp_path: Path, *, candidate_id: str = "candidate-a") -> dict:
    mix = tmp_path / f"{candidate_id}_mix.wav"
    vocal = tmp_path / f"{candidate_id}_vocal.wav"
    minus = tmp_path / f"{candidate_id}_minus.wav"
    for path in (mix, vocal, minus):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"audio-placeholder")
    return {
        "track_id": "track-a",
        "candidate_id": candidate_id,
        "mix_path": str(mix),
        "vocal_path": str(vocal),
        "instrumental_path": str(minus),
        "minus_path": str(minus),
    }


def fake_judge(*, status: str, blocker_count: int):
    def _judge(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "mode": "instrumental_self_judge_gate",
            "source_audio_path": str(kwargs["source_audio_path"]),
            "candidate": {
                "label": str(kwargs["candidate_label"]),
                "instrumental_path": str(kwargs["candidate_instrumental_path"]),
                "vocal_path": str(kwargs["candidate_vocal_path"]),
            },
            "overall": {
                "status": status,
                "blocker_count": blocker_count,
                "segment_count": 4,
                "blocker_ratio": blocker_count / 4,
                "issue_counts": {"presence_loss_proxy": blocker_count} if blocker_count else {},
            },
            "delivery_contract": {
                "delivery_allowed": False,
                "release_status": "diagnostic_self_judge_only",
            },
            "execution_contract": {
                "expected_acceleration_mode": "cpu_dsp_self_judge",
                "gpu_inference_skipped": True,
            },
        }
        matrix.write_json(output_dir / "instrumental_self_judge_manifest.json", manifest)
        return manifest

    return _judge


def test_self_judge_matrix_marks_proxy_pass_candidate_eligible(tmp_path: Path) -> None:
    rows_path = write_jsonl(tmp_path / "candidates.jsonl", [candidate_row(tmp_path)])

    result = matrix.run_self_judge_matrix(
        candidate_jsonl=rows_path,
        output_root=tmp_path / "matrix",
        judge_runner=fake_judge(status="proxy_pass", blocker_count=0),
    )

    assert result["summary"]["eligible_candidates"] == 1
    assert result["rows"][0]["self_judge_status"] == "proxy_pass"
    assert result["rows"][0]["listen_candidate_eligible"] is True
    assert result["manifest"]["delivery_allowed"] is False
    assert result["manifest"]["release_ready"] is False


def test_self_judge_matrix_blocks_proxy_blocked_candidate(tmp_path: Path) -> None:
    rows_path = write_jsonl(tmp_path / "candidates.jsonl", [candidate_row(tmp_path)])

    result = matrix.run_self_judge_matrix(
        candidate_jsonl=rows_path,
        output_root=tmp_path / "matrix",
        judge_runner=fake_judge(status="proxy_blocked", blocker_count=3),
    )

    assert result["summary"]["eligible_candidates"] == 0
    assert result["summary"]["blocked_candidates"] == 1
    assert result["rows"][0]["listen_candidate_eligible"] is False
    assert result["rows"][0]["blocker_count"] == 3
    assert result["rows"][0]["dominant_failure_reason"] == "self_judge_proxy_blocked"


def test_self_judge_matrix_blocks_missing_audio_without_running_judge(tmp_path: Path) -> None:
    row = candidate_row(tmp_path)
    Path(row["instrumental_path"]).unlink()
    rows_path = write_jsonl(tmp_path / "candidates.jsonl", [row])
    calls = {"count": 0}

    def runner(**kwargs):
        calls["count"] += 1
        return fake_judge(status="proxy_pass", blocker_count=0)(**kwargs)

    result = matrix.run_self_judge_matrix(
        candidate_jsonl=rows_path,
        output_root=tmp_path / "matrix",
        judge_runner=runner,
    )

    assert calls["count"] == 0
    assert result["rows"][0]["self_judge_status"] == "blocked_missing_audio"
    assert result["rows"][0]["listen_candidate_eligible"] is False
    assert result["summary"]["blocked_missing_audio"] == 1


def test_self_judge_matrix_allows_missing_vocal_context(tmp_path: Path) -> None:
    row = candidate_row(tmp_path)
    Path(row["vocal_path"]).unlink()
    row["vocal_path"] = ""
    rows_path = write_jsonl(tmp_path / "candidates.jsonl", [row])

    result = matrix.run_self_judge_matrix(
        candidate_jsonl=rows_path,
        output_root=tmp_path / "matrix",
        judge_runner=fake_judge(status="proxy_pass", blocker_count=0),
    )

    assert result["summary"]["eligible_candidates"] == 1
    assert result["rows"][0]["self_judge_status"] == "proxy_pass"
    assert result["rows"][0]["missing_audio_inputs"] == []


def test_self_judge_matrix_writes_outputs_and_safety_flags(tmp_path: Path) -> None:
    rows_path = write_jsonl(tmp_path / "candidates.jsonl", [candidate_row(tmp_path)])
    output_root = tmp_path / "matrix"

    result = matrix.run_self_judge_matrix(
        candidate_jsonl=rows_path,
        output_root=output_root,
        judge_runner=fake_judge(status="proxy_pass", blocker_count=0),
    )

    assert (output_root / "no_reference_minus_self_judge_matrix.jsonl").exists()
    assert (output_root / "no_reference_minus_self_judge_matrix_manifest.json").exists()
    assert (output_root / "no_reference_minus_self_judge_matrix_summary.md").exists()
    assert result["summary"]["expected_acceleration_mode"] == "cpu_dsp_self_judge_matrix"
    assert result["summary"]["gpu_inference_skipped"] is True
    assert result["manifest"]["training_started"] is False
    assert result["manifest"]["fine_tune_started"] is False
    assert result["manifest"]["models_changed"] is False
    assert result["manifest"]["thresholds_relaxed"] is False


def test_self_judge_matrix_filters_by_track_id(tmp_path: Path) -> None:
    keep = candidate_row(tmp_path, candidate_id="keep")
    keep["track_id"] = "target-track"
    skip = candidate_row(tmp_path, candidate_id="skip")
    skip["track_id"] = "other-track"
    rows_path = write_jsonl(tmp_path / "candidates.jsonl", [skip, keep])

    result = matrix.run_self_judge_matrix(
        candidate_jsonl=rows_path,
        output_root=tmp_path / "matrix",
        track_id="target-track",
        judge_runner=fake_judge(status="proxy_pass", blocker_count=0),
    )

    assert result["summary"]["source_candidate_rows"] == 2
    assert result["summary"]["filtered_out_candidates"] == 1
    assert result["summary"]["evaluated_candidates"] == 1
    assert result["rows"][0]["candidate_id"] == "keep"


def test_self_judge_matrix_filters_by_minus_path_substring(tmp_path: Path) -> None:
    keep = candidate_row(tmp_path / "keep", candidate_id="keep")
    skip = candidate_row(tmp_path / "skip", candidate_id="skip")
    rows_path = write_jsonl(tmp_path / "candidates.jsonl", [skip, keep])

    result = matrix.run_self_judge_matrix(
        candidate_jsonl=rows_path,
        output_root=tmp_path / "matrix",
        minus_path_contains="keep",
        judge_runner=fake_judge(status="proxy_pass", blocker_count=0),
    )

    assert result["summary"]["source_candidate_rows"] == 2
    assert result["summary"]["filtered_out_candidates"] == 1
    assert result["summary"]["evaluated_candidates"] == 1
    assert result["rows"][0]["candidate_id"] == "keep"
