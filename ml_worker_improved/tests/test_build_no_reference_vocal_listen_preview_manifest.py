from __future__ import annotations

import json
from pathlib import Path

from tools.artifact_status_contract import can_promote_to_listen_result
from tools.build_no_reference_vocal_listen_preview_manifest import build_manifest, main


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def candidate_manifest(vocal_path: str = "/tmp/vocal.wav") -> dict:
    return {
        "manifest_type": "no_reference_vocal_existing_candidate",
        "render_status": "existing_vocal_candidate_proxy_pass",
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "preview_only_not_for_listen_pack",
        "release_status": "not_release_ready",
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "requires_post_render_vocal_judge": True,
        "rendered_vocal_candidate_path": vocal_path,
        "candidate_row": {
            "track_id": "track-a",
            "candidate_id": "vocal-a",
            "mix_path": "/tmp/mix.wav",
            "vocal_path": vocal_path,
            "instrumental_path": "/tmp/minus.wav",
            "minus_path": "/tmp/minus.wav",
            "selected_vocal_path": vocal_path,
            "selected_minus_path": "/tmp/minus.wav",
        },
    }


def vocal_silence_repair_candidate_manifest(vocal_path: str = "/tmp/vocal.wav") -> dict:
    candidate = candidate_manifest(vocal_path)
    candidate["manifest_type"] = "no_reference_vocal_silence_hallucination_repair"
    candidate["render_status"] = "rendered_vocal_source_quiet_silence_repair_candidate"
    candidate["repair_status"] = "rendered_source_quiet_vocal_silence_hallucination_repair"
    candidate["candidate_row"]["candidate_family"] = "source_quiet_vocal_silence_hallucination_repair"
    candidate["candidate_row"]["candidate_origin"] = "deterministic_no_reference_source_quiet_vocal_repair"
    candidate["candidate_row"]["render_status"] = "rendered_vocal_source_quiet_silence_repair_candidate"
    candidate["candidate_row"]["requires_post_render_vocal_judge"] = True
    return candidate


def judge_guided_vocal_silence_repair_candidate_manifest(vocal_path: str = "/tmp/vocal.wav") -> dict:
    candidate = candidate_manifest(vocal_path)
    candidate["manifest_type"] = "judge_guided_vocal_silence_hallucination_repair"
    candidate["render_status"] = "rendered_vocal_judge_guided_silence_repair_candidate"
    candidate["candidate_row"]["candidate_family"] = "judge_guided_vocal_silence_hallucination_repair"
    candidate["candidate_row"]["candidate_origin"] = "windowed_judge_guided_silence_repair_candidate"
    candidate["candidate_row"]["render_status"] = "rendered_vocal_judge_guided_silence_repair_candidate"
    candidate["candidate_row"]["requires_post_render_vocal_judge"] = True
    return candidate


def vocal_judge_row(
    vocal_path: str = "/tmp/vocal.wav",
    *,
    decision: str = "accept_candidate",
    score_mode: str = "vocal",
) -> dict:
    return {
        "track_id": "track-a",
        "candidate_id": "vocal-a",
        "mix_path": "/tmp/mix.wav",
        "vocal_path": vocal_path,
        "instrumental_path": "/tmp/minus.wav",
        "score_mode": score_mode,
        "decision": decision,
        "vocal_decision": decision,
        "vocal_score": 0.88,
        "vocal_blockers": [],
        "overall_no_reference_score": 0.88,
        "confidence": 0.86,
        "reasons": ["no_reference_proxy_passed"],
    }


def selector_row(
    vocal_path: str = "/tmp/vocal.wav",
    *,
    decision: str = "accept_candidate",
    score_mode: str = "vocal",
) -> dict:
    return {
        "track_id": "track-a",
        "score_mode": score_mode,
        "decision": decision,
        "selected_candidate": vocal_judge_row(vocal_path),
        "reasons": ["selected_by_no_reference_score", "no_reference_proxy_passed"],
        "candidate_count": 1,
        "accepted_candidate_count": 1 if decision in {"accept_candidate", "select_alternative_candidate"} else 0,
    }


def test_vocal_listen_preview_accepts_only_vocal_mode_evidence(tmp_path: Path) -> None:
    manifest = build_manifest(
        candidate_manifest_json=write_json(tmp_path / "candidate.json", candidate_manifest()),
        vocal_judge_jsonl=write_jsonl(tmp_path / "vocal_judge.jsonl", [vocal_judge_row()]),
        selector_jsonl=write_jsonl(tmp_path / "selector.jsonl", [selector_row()]),
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["presentation_status"] == "accepted_for_listen_pack"
    assert manifest["release_status"] == "listen_preview_only"
    assert manifest["stem_type"] == "vocal"
    assert manifest["accepted_count"] == 1
    assert manifest["selected_vocal_path"] == "/tmp/vocal.wav"
    assert manifest["can_mark_release_ready"] is False
    assert manifest["audio_copied"] is False
    assert manifest["audio_exported"] is False


def test_vocal_listen_preview_accepts_mode_specific_vocal_decision_with_generic_minus_reasons(
    tmp_path: Path,
) -> None:
    vocal_row = vocal_judge_row(decision="select_alternative_candidate")
    vocal_row["vocal_decision"] = "accept_candidate"
    vocal_row["vocal_blockers"] = []
    vocal_row["vocal_score"] = 0.82
    vocal_row["reasons"] = ["vocal_leakage_in_minus", "silence_hallucination"]

    manifest = build_manifest(
        candidate_manifest_json=write_json(tmp_path / "candidate.json", candidate_manifest()),
        vocal_judge_jsonl=write_jsonl(tmp_path / "vocal_judge.jsonl", [vocal_row]),
        selector_jsonl=write_jsonl(
            tmp_path / "selector.jsonl",
            [selector_row(decision="select_alternative_candidate")],
        ),
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["accepted_listen_preview_rows"][0]["vocal_score"] == 0.82
    assert manifest["can_mark_release_ready"] is False


def test_vocal_listen_preview_accepts_source_quiet_silence_repair_candidate(tmp_path: Path) -> None:
    manifest = build_manifest(
        candidate_manifest_json=write_json(tmp_path / "candidate.json", vocal_silence_repair_candidate_manifest()),
        vocal_judge_jsonl=write_jsonl(tmp_path / "vocal_judge.jsonl", [vocal_judge_row()]),
        selector_jsonl=write_jsonl(tmp_path / "selector.jsonl", [selector_row()]),
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["presentation_status"] == "accepted_for_listen_pack"
    assert manifest["selected_vocal_path"] == "/tmp/vocal.wav"
    assert manifest["can_mark_release_ready"] is False


def test_vocal_listen_preview_accepts_judge_guided_silence_repair_candidate(tmp_path: Path) -> None:
    manifest = build_manifest(
        candidate_manifest_json=write_json(tmp_path / "candidate.json", judge_guided_vocal_silence_repair_candidate_manifest()),
        vocal_judge_jsonl=write_jsonl(tmp_path / "vocal_judge.jsonl", [vocal_judge_row()]),
        selector_jsonl=write_jsonl(tmp_path / "selector.jsonl", [selector_row()]),
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["presentation_status"] == "accepted_for_listen_pack"
    assert manifest["selected_vocal_path"] == "/tmp/vocal.wav"


def test_vocal_listen_preview_blocks_non_vocal_judge_mode(tmp_path: Path) -> None:
    manifest = build_manifest(
        candidate_manifest_json=write_json(tmp_path / "candidate.json", candidate_manifest()),
        vocal_judge_jsonl=write_jsonl(tmp_path / "vocal_judge.jsonl", [vocal_judge_row(score_mode="both")]),
        selector_jsonl=write_jsonl(tmp_path / "selector.jsonl", [selector_row()]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "vocal_judge_not_vocal_mode"


def test_vocal_listen_preview_blocks_non_vocal_selector_mode(tmp_path: Path) -> None:
    manifest = build_manifest(
        candidate_manifest_json=write_json(tmp_path / "candidate.json", candidate_manifest()),
        vocal_judge_jsonl=write_jsonl(tmp_path / "vocal_judge.jsonl", [vocal_judge_row()]),
        selector_jsonl=write_jsonl(tmp_path / "selector.jsonl", [selector_row(score_mode="both")]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "selector_not_vocal_mode"


def test_vocal_listen_preview_blocks_selector_candidate_mismatch(tmp_path: Path) -> None:
    wrong_selector = selector_row()
    wrong_selector["selected_candidate"]["candidate_id"] = "other-vocal"
    manifest = build_manifest(
        candidate_manifest_json=write_json(tmp_path / "candidate.json", candidate_manifest()),
        vocal_judge_jsonl=write_jsonl(tmp_path / "vocal_judge.jsonl", [vocal_judge_row()]),
        selector_jsonl=write_jsonl(tmp_path / "selector.jsonl", [wrong_selector]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "selector_candidate_mismatch"


def test_vocal_listen_preview_blocks_source_safety_violation(tmp_path: Path) -> None:
    candidate = candidate_manifest()
    candidate["safety_flags"] = {"training_started": True}

    manifest = build_manifest(
        candidate_manifest_json=write_json(tmp_path / "candidate.json", candidate),
        vocal_judge_jsonl=write_jsonl(tmp_path / "vocal_judge.jsonl", [vocal_judge_row()]),
        selector_jsonl=write_jsonl(tmp_path / "selector.jsonl", [selector_row()]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "source_candidate_safety_violation"
    assert manifest["safety"]["training_started"] is True


def test_vocal_listen_preview_blocks_non_preview_source_candidate(tmp_path: Path) -> None:
    candidate = candidate_manifest()
    candidate["delivery_status"] = "accepted"
    candidate["delivery_allowed"] = True
    candidate["presentation_status"] = "accepted_for_listen_pack"
    candidate["release_status"] = "listen_preview_only"

    manifest = build_manifest(
        candidate_manifest_json=write_json(tmp_path / "candidate.json", candidate),
        vocal_judge_jsonl=write_jsonl(tmp_path / "vocal_judge.jsonl", [vocal_judge_row()]),
        selector_jsonl=write_jsonl(tmp_path / "selector.jsonl", [selector_row()]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "source_candidate_not_preview_only"


def test_vocal_listen_preview_manifest_is_promotable_by_artifact_contract(tmp_path: Path) -> None:
    output_manifest = tmp_path / "vocal_listen_preview_manifest.json"
    summary_md = tmp_path / "summary.md"

    assert (
        main(
            [
                "--candidate-manifest-json",
                str(write_json(tmp_path / "candidate.json", candidate_manifest())),
                "--vocal-judge-jsonl",
                str(write_jsonl(tmp_path / "vocal_judge.jsonl", [vocal_judge_row()])),
                "--selector-jsonl",
                str(write_jsonl(tmp_path / "selector.jsonl", [selector_row()])),
                "--output-manifest-json",
                str(output_manifest),
                "--summary-md",
                str(summary_md),
            ]
        )
        == 0
    )
    assert can_promote_to_listen_result(output_manifest) is True
    assert "delivery_status: accepted" in summary_md.read_text(encoding="utf-8")
