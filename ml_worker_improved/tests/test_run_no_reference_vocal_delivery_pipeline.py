from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.artifact_status_contract import can_promote_to_listen_result
from tools.run_no_reference_vocal_delivery_pipeline import (
    choose_base_candidate_id_from_score_matrix,
    main,
    run_vocal_delivery_pipeline,
)


SAMPLE_RATE = 8000


def write_wav(path: Path, value: float, *, seconds: float = 4.0) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.full((int(seconds * SAMPLE_RATE), 2), value, dtype=np.float32)
    sf.write(path, audio, SAMPLE_RATE, subtype="FLOAT")
    return str(path)


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def vocal_judge_row(tmp_path: Path, candidate_id: str = "vocal-a") -> dict:
    vocal_path = write_wav(tmp_path / candidate_id / "vocal.wav", 0.2)
    return {
        "track_id": "track-a",
        "candidate_id": candidate_id,
        "mix_path": write_wav(tmp_path / "mix.wav", 0.5),
        "vocal_path": vocal_path,
        "instrumental_path": write_wav(tmp_path / candidate_id / "minus.wav", 0.3),
        "score_mode": "vocal",
        "decision": "accept_candidate",
        "vocal_decision": "accept_candidate",
        "vocal_score": 0.88,
        "vocal_blockers": [],
        "window_count": 12,
        "overall_no_reference_score": 0.88,
        "confidence": 0.86,
        "minus_decision": "blocked_hard_case",
        "minus_score": 0.2,
        "minus_blockers": ["severe_vocal_bleed_in_minus"],
        "reasons": ["no_reference_proxy_passed"],
    }


def test_vocal_delivery_pipeline_accepts_vocal_without_minus_or_both_gate(tmp_path: Path) -> None:
    vocal_judge_jsonl = write_jsonl(tmp_path / "vocal_judge.jsonl", [vocal_judge_row(tmp_path)])

    manifest = run_vocal_delivery_pipeline(
        vocal_judge_jsonl=vocal_judge_jsonl,
        output_dir=tmp_path / "out",
        base_candidate_id=None,
    )

    listen_manifest = Path(manifest["final_listen_preview_manifest"])
    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["presentation_status"] == "accepted_for_listen_pack"
    assert manifest["release_status"] == "listen_preview_only"
    assert manifest["stem_type"] == "vocal"
    assert manifest["accepted_count"] == 1
    assert manifest["score_modes_used"] == ["vocal"]
    assert manifest["safety"]["reference_required"] is False
    assert manifest["safety"]["training_started"] is False
    assert manifest["safety"]["models_changed"] is False
    assert manifest["safety"]["cleanup_routes_added"] is False
    assert manifest["safety"]["thresholds_relaxed"] is False
    assert can_promote_to_listen_result(listen_manifest) is True


def test_vocal_delivery_pipeline_accepts_mode_specific_vocal_decision_with_generic_minus_reasons(
    tmp_path: Path,
) -> None:
    row = vocal_judge_row(tmp_path)
    row["decision"] = "select_alternative_candidate"
    row["vocal_decision"] = "accept_candidate"
    row["vocal_score"] = 0.82
    row["vocal_blockers"] = []
    row["reasons"] = ["vocal_leakage_in_minus", "silence_hallucination"]
    vocal_judge_jsonl = write_jsonl(tmp_path / "vocal_judge.jsonl", [row])

    manifest = run_vocal_delivery_pipeline(
        vocal_judge_jsonl=vocal_judge_jsonl,
        output_dir=tmp_path / "out",
        base_candidate_id=None,
    )

    listen_manifest = json.loads(Path(manifest["final_listen_preview_manifest"]).read_text(encoding="utf-8"))
    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["score_modes_used"] == ["vocal"]
    assert listen_manifest["candidate_id"] == row["candidate_id"]
    assert listen_manifest["accepted_listen_preview_rows"][0]["vocal_score"] == 0.82


def test_vocal_delivery_pipeline_preserves_rendered_source_quiet_repair_status(tmp_path: Path) -> None:
    row = vocal_judge_row(tmp_path)
    row["candidate_family"] = "source_quiet_vocal_silence_hallucination_repair"
    row["candidate_origin"] = "deterministic_no_reference_source_quiet_vocal_repair"
    row["render_status"] = "rendered_vocal_source_quiet_silence_repair_candidate"
    vocal_judge_jsonl = write_jsonl(tmp_path / "vocal_judge.jsonl", [row])

    manifest = run_vocal_delivery_pipeline(
        vocal_judge_jsonl=vocal_judge_jsonl,
        output_dir=tmp_path / "out",
        base_candidate_id=None,
    )

    candidate_manifest = json.loads(Path(manifest["candidate_manifest_path"]).read_text(encoding="utf-8"))
    listen_manifest = json.loads(Path(manifest["final_listen_preview_manifest"]).read_text(encoding="utf-8"))
    assert manifest["delivery_status"] == "accepted"
    assert candidate_manifest["render_status"] == "rendered_vocal_source_quiet_silence_repair_candidate"
    assert candidate_manifest["candidate_row"]["render_status"] == "rendered_vocal_source_quiet_silence_repair_candidate"
    assert listen_manifest["delivery_status"] == "accepted"


def test_vocal_delivery_pipeline_selects_by_vocal_score_not_overall_score(tmp_path: Path) -> None:
    vocal_better = vocal_judge_row(tmp_path, "vocal-better")
    vocal_better["vocal_score"] = 0.95
    vocal_better["overall_no_reference_score"] = 0.70
    vocal_better["confidence"] = 0.70

    overall_better = vocal_judge_row(tmp_path, "overall-better")
    overall_better["vocal_score"] = 0.80
    overall_better["overall_no_reference_score"] = 0.99
    overall_better["confidence"] = 0.99

    vocal_judge_jsonl = write_jsonl(tmp_path / "vocal_judge.jsonl", [overall_better, vocal_better])

    manifest = run_vocal_delivery_pipeline(
        vocal_judge_jsonl=vocal_judge_jsonl,
        output_dir=tmp_path / "out",
        base_candidate_id=None,
    )

    listen_manifest = json.loads(Path(manifest["final_listen_preview_manifest"]).read_text(encoding="utf-8"))
    assert manifest["delivery_status"] == "accepted"
    assert manifest["base_candidate_id"] == "vocal-better"
    assert manifest["selected_vocal_path"] == vocal_better["vocal_path"]
    assert listen_manifest["candidate_id"] == "vocal-better"
    assert listen_manifest["accepted_listen_preview_rows"][0]["vocal_score"] == 0.95


def test_vocal_delivery_pipeline_uses_score_matrix_for_base_candidate(tmp_path: Path) -> None:
    score_winner = vocal_judge_row(tmp_path, "score-winner")
    score_winner["vocal_score"] = 0.95
    score_winner["confidence"] = 0.95
    matrix_vocal = vocal_judge_row(tmp_path, "matrix-vocal")
    matrix_vocal["vocal_score"] = 0.86
    matrix_vocal["confidence"] = 0.86
    vocal_judge_jsonl = write_jsonl(tmp_path / "vocal_judge.jsonl", [score_winner, matrix_vocal])
    matrix_json = tmp_path / "matrix" / "no_reference_vocal_candidate_score_matrix.json"
    matrix_json.parent.mkdir(parents=True, exist_ok=True)
    matrix_json.write_text(
        json.dumps(
            {
                "summary": {
                    "matrix_type": "no_reference_vocal_candidate_score_matrix",
                    "score_mode": "vocal",
                    "top_candidate_id": "matrix-vocal",
                    "top_next_action": "candidate_ready_for_listen_preview_contract",
                },
                "rows": [
                    {
                        "rank": 1,
                        "canonical_candidate_id": "matrix-vocal",
                        "aliases": ["matrix-vocal"],
                        "next_action": "candidate_ready_for_listen_preview_contract",
                        "selection_priority": 92,
                        "vocal_judge": {"accepted": True, "vocal_score": 0.86, "confidence": 0.86},
                        "selector": {"accepted": True},
                        "delivery_allowed": False,
                        "release_ready": False,
                    }
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    manifest = run_vocal_delivery_pipeline(
        vocal_judge_jsonl=vocal_judge_jsonl,
        output_dir=tmp_path / "out",
        base_candidate_id=None,
        candidate_score_matrix_json=matrix_json,
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["base_candidate_id"] == "matrix-vocal"
    assert manifest["base_selection_source"] == "candidate_score_matrix"


def test_vocal_score_matrix_base_selection_prefers_specific_candidate_over_model_alias(tmp_path: Path) -> None:
    matrix_json = tmp_path / "matrix.json"
    matrix_json.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "rank": 1,
                        "canonical_candidate_id": "vocal_model_variant",
                        "aliases": ["vocal_model", "vocal_model_variant"],
                        "next_action": "candidate_ready_for_listen_preview_contract",
                        "selection_priority": 92,
                    }
                ]
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    rows = [
        {"candidate_id": "vocal_model", "score_mode": "vocal", "decision": "accept_candidate", "vocal_decision": "accept_candidate"},
        {
            "candidate_id": "vocal_model_variant",
            "score_mode": "vocal",
            "decision": "accept_candidate",
            "vocal_decision": "accept_candidate",
        },
    ]

    assert choose_base_candidate_id_from_score_matrix(matrix_json=matrix_json, vocal_rows=rows) == "vocal_model_variant"


def test_vocal_delivery_pipeline_blocks_non_vocal_rows(tmp_path: Path) -> None:
    row = vocal_judge_row(tmp_path)
    row["score_mode"] = "both"
    row["decision"] = "accept_candidate"
    vocal_judge_jsonl = write_jsonl(tmp_path / "both_judge.jsonl", [row])

    manifest = run_vocal_delivery_pipeline(
        vocal_judge_jsonl=vocal_judge_jsonl,
        output_dir=tmp_path / "out",
        base_candidate_id=None,
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "blocked_from_listen_pack"
    assert manifest["dominant_failure_reason"] == "no_accepted_vocal_candidate"
    assert manifest["score_modes_used"] == ["vocal"]


def test_vocal_delivery_pipeline_blocks_generic_only_acceptance(tmp_path: Path) -> None:
    row = vocal_judge_row(tmp_path)
    del row["vocal_score"]
    del row["vocal_decision"]
    del row["vocal_blockers"]
    row["decision"] = "accept_candidate"
    row["overall_no_reference_score"] = 0.99

    manifest = run_vocal_delivery_pipeline(
        vocal_judge_jsonl=write_jsonl(tmp_path / "vocal_judge.jsonl", [row]),
        output_dir=tmp_path / "out",
        base_candidate_id=None,
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "no_accepted_vocal_candidate"
    assert manifest["score_modes_used"] == ["vocal"]


def test_vocal_delivery_pipeline_blocks_missing_audio_before_selector(tmp_path: Path) -> None:
    row = vocal_judge_row(tmp_path)
    row["vocal_path"] = str(tmp_path / "missing_vocal.wav")
    vocal_judge_jsonl = write_jsonl(tmp_path / "vocal_judge.jsonl", [row])

    manifest = run_vocal_delivery_pipeline(
        vocal_judge_jsonl=vocal_judge_jsonl,
        output_dir=tmp_path / "out",
        base_candidate_id=None,
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "missing_audio_path_for_vocal_delivery"
    assert str(tmp_path / "missing_vocal.wav") in manifest["missing_audio_inputs"]


def test_vocal_delivery_pipeline_cli_writes_final_manifest(tmp_path: Path) -> None:
    vocal_judge_jsonl = write_jsonl(tmp_path / "vocal_judge.jsonl", [vocal_judge_row(tmp_path)])
    output_dir = tmp_path / "out"

    assert (
        main(
            [
                "--vocal-judge-jsonl",
                str(vocal_judge_jsonl),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    final_manifest = output_dir / "no_reference_vocal_delivery_pipeline_manifest.json"
    summary_md = output_dir / "no_reference_vocal_delivery_pipeline_summary.md"
    assert final_manifest.exists()
    assert summary_md.exists()
    payload = json.loads(final_manifest.read_text(encoding="utf-8"))
    assert payload["delivery_status"] == "accepted"
    assert payload["delivery_allowed"] is True
