from __future__ import annotations

import json
from pathlib import Path

from tools.build_no_reference_vocal_candidate_score_matrix import (
    build_vocal_candidate_score_matrix,
    write_outputs,
)


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def stem_manifest(path: Path, *, model_name: str, gpu: bool = True) -> Path:
    return write_json(
        path,
        {
            "model_name": model_name,
            "target_stem": "vocal",
            "execution_contract": {
                "docker_runtime_required": True,
                "expected_acceleration_mode": "gpu_inference_then_cpu_dsp_scoring",
                "gpu_inference_required": True,
                "gpu_inference_skipped": False,
                "cuda_preflight": {"cuda_available": gpu, "device_name": "NVIDIA test GPU"},
            },
            "delivery_allowed": False,
            "release_ready": False,
        },
    )


def vocal_result_row(
    candidate_id: str,
    *,
    manifest: Path,
    vocal_path: Path,
    vocal_score: float,
    confidence: float,
    decision: str,
    blockers: list[str] | None = None,
    reasons: list[str] | None = None,
) -> dict:
    return {
        "track_id": "track-a",
        "candidate_id": candidate_id,
        "candidate_manifest": str(manifest),
        "mix_path": "/input/mix.wav",
        "vocal_path": str(vocal_path),
        "instrumental_path": str(vocal_path.with_name("minus.flac")),
        "score_mode": "vocal",
        "decision": decision,
        "vocal_decision": decision,
        "vocal_score": vocal_score,
        "overall_no_reference_score": vocal_score,
        "confidence": confidence,
        "vocal_blockers": blockers or [],
        "reasons": reasons or ([] if decision == "accept_candidate" else ["instrumental_leakage_in_vocal"]),
    }


def test_vocal_matrix_ranks_ready_vocal_above_blocked_candidate(tmp_path: Path) -> None:
    ready_manifest = stem_manifest(tmp_path / "ready" / "stem_manifest.json", model_name="ready_vocal")
    blocked_manifest = stem_manifest(tmp_path / "blocked" / "stem_manifest.json", model_name="blocked_vocal")
    result_jsonl = write_jsonl(
        tmp_path / "vocal_result.jsonl",
        [
            vocal_result_row(
                "blocked-id",
                manifest=blocked_manifest,
                vocal_path=tmp_path / "blocked" / "vocals.flac",
                vocal_score=0.91,
                confidence=0.9,
                decision="blocked_hard_case",
                blockers=["severe_instrumental_bleed_in_vocal"],
            ),
            vocal_result_row(
                "ready-id",
                manifest=ready_manifest,
                vocal_path=tmp_path / "ready" / "vocals.flac",
                vocal_score=0.86,
                confidence=0.82,
                decision="accept_candidate",
                reasons=["no_reference_proxy_passed"],
            ),
        ],
    )
    selector_jsonl = write_jsonl(
        tmp_path / "selector.jsonl",
        [
            {
                "track_id": "track-a",
                "score_mode": "vocal",
                "decision": "accept_candidate",
                "selected_candidate_id": "ready-id",
                "ranked_candidates": [{"candidate_id": "ready-id", "rank": 1, "selection_score": 0.86}],
            }
        ],
    )

    matrix = build_vocal_candidate_score_matrix(
        vocal_judge_jsonls=[result_jsonl],
        selector_jsonls=[selector_jsonl],
        listen_manifests=[],
    )

    rows = matrix["rows"]
    assert rows[0]["canonical_candidate_id"] == "ready"
    assert rows[0]["vocal_judge"]["accepted"] is True
    assert rows[0]["selector"]["accepted"] is True
    assert rows[0]["next_action"] == "candidate_ready_for_listen_preview_contract"
    assert rows[0]["has_gpu_proof"] is True
    assert rows[0]["delivery_allowed"] is False
    assert rows[0]["release_ready"] is False
    assert rows[1]["canonical_candidate_id"] == "blocked"
    assert rows[1]["next_action"] == "do_not_release_select_cleaner_vocal_or_new_route"
    assert matrix["summary"]["ready_for_listen_preview_contract"] == 1
    assert matrix["summary"]["release_ready"] is False


def test_vocal_matrix_records_accepted_listen_preview_without_release(tmp_path: Path) -> None:
    manifest = stem_manifest(tmp_path / "ready" / "stem_manifest.json", model_name="ready_vocal")
    vocal_path = tmp_path / "ready" / "vocals.flac"
    result_jsonl = write_jsonl(
        tmp_path / "vocal_result.jsonl",
        [
            vocal_result_row(
                "ready-id",
                manifest=manifest,
                vocal_path=vocal_path,
                vocal_score=0.9,
                confidence=0.88,
                decision="accept_candidate",
            )
        ],
    )
    listen_manifest = write_json(
        tmp_path / "listen.json",
        {
            "candidate_id": "ready-id",
            "track_id": "track-a",
            "selected_vocal_path": str(vocal_path),
            "delivery_allowed": True,
            "delivery_status": "accepted",
            "presentation_status": "accepted_for_listen_pack",
            "release_status": "listen_preview_only",
        },
    )

    matrix = build_vocal_candidate_score_matrix(
        vocal_judge_jsonls=[result_jsonl],
        selector_jsonls=[],
        listen_manifests=[listen_manifest],
    )

    row = matrix["rows"][0]
    assert row["next_action"] == "accepted_for_listen_pack_not_release"
    assert row["delivery_allowed"] is True
    assert row["release_ready"] is False

    output_dir = tmp_path / "out"
    write_outputs(matrix, output_dir)
    assert (output_dir / "no_reference_vocal_candidate_score_matrix.json").exists()
    assert (output_dir / "no_reference_vocal_candidate_score_matrix.jsonl").exists()
    assert (output_dir / "no_reference_vocal_candidate_score_matrix_summary.md").exists()
    assert not list(output_dir.rglob("*.wav"))
