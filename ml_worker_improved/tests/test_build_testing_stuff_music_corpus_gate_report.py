from __future__ import annotations

import json
from pathlib import Path

from tools.build_testing_stuff_music_corpus_gate_report import build_report, main


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def test_corpus_gate_report_keeps_vocal_candidate_diagnostic_until_mode_gates_exist(tmp_path: Path) -> None:
    plan_jsonl = write_jsonl(
        tmp_path / "plan.jsonl",
        [
            {"track_id": "track_a", "audio_path": "/host/a.mp3", "target_stems": ["vocal", "melody"]},
        ],
    )
    execution_jsonl = write_jsonl(
        tmp_path / "execution.jsonl",
        [
            {
                "track_id": "track_a",
                "target_stem": "vocal",
                "model_id": "htdemucs_6s",
                "execution_status": "completed",
                "candidate_manifest": {"exists": True, "path": "/app/artifacts/vocal/stem_manifest.json"},
            }
        ],
    )

    report = build_report(
        corpus_plan_jsonl=plan_jsonl,
        execution_results_jsonls=[execution_jsonl],
        score_matrix_jsons=[],
    )

    assert report["delivery_allowed"] is False
    assert report["release_ready"] is False
    assert report["summary"]["diagnostic_only_candidates"] == 1
    assert report["summary"]["missing_mode_specific_gate_count"] == 1
    assert report["rows"][0]["gate_status"] == "diagnostic_only_missing_mode_specific_gate"
    assert report["rows"][0]["required_next_gate"] == "vocal_judge_selector_listen_preview"


def test_corpus_gate_report_accepts_only_score_matrix_accepted_stem_candidate(tmp_path: Path) -> None:
    plan_jsonl = write_jsonl(
        tmp_path / "plan.jsonl",
        [
            {"track_id": "track_a", "audio_path": "/host/a.mp3", "target_stems": ["melody"]},
        ],
    )
    execution_jsonl = write_jsonl(
        tmp_path / "execution.jsonl",
        [
            {
                "track_id": "track_a",
                "target_stem": "melody",
                "model_id": "htdemucs_6s",
                "execution_status": "completed",
                "candidate_manifest": {"exists": True, "path": "/app/artifacts/melody/stem_manifest.json"},
            }
        ],
    )
    matrix_json = write_json(
        tmp_path / "stem_score_matrix.json",
        {
            "matrix_type": "no_reference_stem_candidate_score_matrix",
            "score_mode": "stem",
            "ranked_candidates": [
                {
                    "track_id": "track_a",
                    "stem_type": "melody",
                    "model_name": "htdemucs_6s",
                    "delivery_status": "accepted",
                    "delivery_allowed": True,
                    "presentation_status": "accepted_for_listen_pack",
                    "release_status": "listen_preview_only",
                    "release_ready": False,
                    "selected_audio_path": "/app/artifacts/melody/melody.flac",
                    "semantic_quality_verified": True,
                    "stem_semantic_blockers": [],
                    "selection_score": 0.91,
                    "next_action": "accepted_for_listen_pack_not_release",
                }
            ],
        },
    )

    report = build_report(
        corpus_plan_jsonl=plan_jsonl,
        execution_results_jsonls=[execution_jsonl],
        score_matrix_jsons=[matrix_json],
    )

    assert report["summary"]["accepted_listen_preview_candidates"] == 1
    assert report["summary"]["diagnostic_only_candidates"] == 0
    assert report["rows"][0]["gate_status"] == "accepted_for_listen_pack_not_release"
    assert report["rows"][0]["selected_audio_path"] == "/app/artifacts/melody/melody.flac"
    assert report["rows"][0]["score_matrix_path"] == str(matrix_json)


def test_corpus_gate_report_does_not_apply_one_accepted_matrix_row_to_other_models(tmp_path: Path) -> None:
    plan_jsonl = write_jsonl(
        tmp_path / "plan.jsonl",
        [
            {"track_id": "track_a", "audio_path": "/host/a.mp3", "target_stems": ["melody"]},
        ],
    )
    execution_jsonl = write_jsonl(
        tmp_path / "execution.jsonl",
        [
            {
                "track_id": "track_a",
                "target_stem": "melody",
                "model_id": "model_a",
                "execution_status": "completed",
                "candidate_manifest": {"exists": True, "path": "/app/artifacts/track_a/melody/model_a/stem_manifest.json"},
            },
            {
                "track_id": "track_a",
                "target_stem": "melody",
                "model_id": "model_b",
                "execution_status": "completed",
                "candidate_manifest": {"exists": True, "path": "/app/artifacts/track_a/melody/model_b/stem_manifest.json"},
            },
        ],
    )
    matrix_json = write_json(
        tmp_path / "stem_score_matrix.json",
        {
            "matrix_type": "no_reference_stem_candidate_score_matrix",
            "score_mode": "stem",
            "ranked_candidates": [
                {
                    "track_id": "track_a",
                    "stem_type": "melody",
                    "model_name": "model_b",
                    "delivery_status": "accepted",
                    "delivery_allowed": True,
                    "presentation_status": "accepted_for_listen_pack",
                    "release_status": "listen_preview_only",
                    "release_ready": False,
                    "selected_audio_path": "/app/artifacts/track_a/melody/model_b/melody.flac",
                    "semantic_quality_verified": True,
                    "stem_semantic_blockers": [],
                    "selection_score": 0.95,
                    "next_action": "accepted_for_listen_pack_not_release",
                }
            ],
        },
    )

    report = build_report(
        corpus_plan_jsonl=plan_jsonl,
        execution_results_jsonls=[execution_jsonl],
        score_matrix_jsons=[matrix_json],
    )

    statuses = {(row["model_id"], row["gate_status"]) for row in report["rows"]}
    assert ("model_a", "diagnostic_only_missing_stem_score_matrix") in statuses
    assert ("model_b", "accepted_for_listen_pack_not_release") in statuses
    assert report["summary"]["accepted_listen_preview_candidates"] == 1
    assert report["summary"]["diagnostic_only_candidates"] == 1


def test_corpus_gate_report_cli_writes_manifest_and_summary(tmp_path: Path) -> None:
    plan_jsonl = write_jsonl(tmp_path / "plan.jsonl", [{"track_id": "track_a", "target_stems": ["minus"]}])
    execution_jsonl = write_jsonl(
        tmp_path / "execution.jsonl",
        [
            {
                "track_id": "track_a",
                "target_stem": "minus",
                "model_id": "instrumental_resurrection_minus",
                "execution_status": "completed",
                "candidate_manifest": {"exists": True, "path": "/app/artifacts/minus/stem_manifest.json"},
            }
        ],
    )
    output_manifest = tmp_path / "report.json"
    summary_md = tmp_path / "summary.md"

    assert (
        main(
            [
                "--corpus-plan-jsonl",
                str(plan_jsonl),
                "--execution-results-jsonl",
                str(execution_jsonl),
                "--output-manifest-json",
                str(output_manifest),
                "--summary-md",
                str(summary_md),
            ]
        )
        == 0
    )

    manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert manifest["rows"][0]["required_next_gate"] == "minus_self_result_selector_listen_preview"
    assert "diagnostic_only_candidates: 1" in summary_md.read_text(encoding="utf-8")
