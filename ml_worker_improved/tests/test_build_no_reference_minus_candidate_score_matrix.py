from __future__ import annotations

import json
from pathlib import Path

from tools.build_no_reference_minus_candidate_score_matrix import (
    build_minus_candidate_score_matrix,
    write_outputs,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def stem_manifest(path: Path, *, model_name: str, gpu: bool = True) -> None:
    write_json(
        path,
        {
            "model_name": model_name,
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


def result_row(
    candidate_id: str,
    *,
    manifest: Path,
    minus_path: Path,
    minus_score: float,
    confidence: float,
    decision: str,
    reasons: list[str] | None = None,
    detail_status: str = "proxy_pass",
    pair_decision: str = "pair_compatible",
    pair_blockers: list[str] | None = None,
) -> dict:
    return {
        "track_id": "track-a",
        "candidate_id": candidate_id,
        "candidate_manifest": str(manifest),
        "instrumental_path": str(minus_path),
        "minus_path": str(minus_path),
        "vocal_path": str(minus_path.with_name("vocals.flac")),
        "score_mode": "minus",
        "decision": decision,
        "minus_decision": decision,
        "pair_decision": pair_decision,
        "pair_blockers": pair_blockers or [],
        "minus_score": minus_score,
        "confidence": confidence,
        "minus_blockers": [] if decision == "accept_candidate" else ["severe_vocal_bleed_in_minus"],
        "reasons": reasons or ([] if decision == "accept_candidate" else ["vocal_leakage_in_minus"]),
        "minus_musical_detail_preservation_gate": {
            "status": detail_status,
            "blockers": [] if detail_status == "proxy_pass" else [{"reason": "minus_candidate_loses_music_detail"}],
        },
    }


def self_manifest(path: Path, *, label: str, minus_path: Path, status: str, blockers: int) -> None:
    write_json(
        path,
        {
            "source_audio_path": "/input/mix.wav",
            "candidate": {
                "label": label,
                "instrumental_path": str(minus_path),
                "vocal_path": str(minus_path.with_name("vocals.flac")),
            },
            "overall": {
                "status": status,
                "blocker_count": blockers,
                "blocker_ratio": round(blockers / 10.0, 6),
                "segment_count": 10,
                "top_blockers": [
                    {
                        "segment_id": "segment_0001",
                        "issues": ["vocal_projection_proxy"],
                        "severity_score": 0.08,
                        "window_sec": [6.0, 12.0],
                    }
                ]
                if blockers
                else [],
            },
            "segments": [],
            "execution_contract": {
                "docker_runtime_required": True,
                "expected_acceleration_mode": "cpu_dsp_self_judge",
                "gpu_inference_skipped": True,
            },
        },
    )


def test_matrix_ranks_result_accepted_self_blocked_candidate_for_router_before_self_pass_result_blocked(
    tmp_path: Path,
) -> None:
    aggressive_manifest = tmp_path / "aggressive" / "stem_manifest.json"
    normal_manifest = tmp_path / "normal" / "stem_manifest.json"
    repair_manifest = tmp_path / "repair" / "stem_manifest.json"
    stem_manifest(aggressive_manifest, model_name="hybrid_aggressive")
    stem_manifest(normal_manifest, model_name="hybrid_normal")
    stem_manifest(repair_manifest, model_name="repair")

    aggressive_minus = tmp_path / "aggressive" / "minus.flac"
    normal_minus = tmp_path / "normal" / "minus.flac"
    repair_minus = tmp_path / "repair" / "minus.flac"
    result_jsonl = tmp_path / "judge.jsonl"
    write_jsonl(
        result_jsonl,
        [
            result_row(
                "normal",
                manifest=normal_manifest,
                minus_path=normal_minus,
                minus_score=0.668,
                confidence=0.54,
                decision="blocked_hard_case",
            ),
            result_row(
                "aggressive-result-id",
                manifest=aggressive_manifest,
                minus_path=aggressive_minus,
                minus_score=0.797,
                confidence=0.75,
                decision="accept_candidate",
            ),
            result_row(
                "repair",
                manifest=repair_manifest,
                minus_path=repair_minus,
                minus_score=0.730,
                confidence=0.72,
                decision="blocked_hard_case",
            ),
        ],
    )
    aggressive_self = tmp_path / "aggressive-self" / "instrumental_self_judge_manifest.json"
    normal_self = tmp_path / "normal-self" / "instrumental_self_judge_manifest.json"
    repair_self = tmp_path / "repair-self" / "instrumental_self_judge_manifest.json"
    self_manifest(aggressive_self, label="hybrid_aggressive", minus_path=aggressive_minus, status="proxy_blocked", blockers=2)
    self_manifest(normal_self, label="hybrid_normal", minus_path=normal_minus, status="proxy_blocked", blockers=1)
    self_manifest(repair_self, label="repair", minus_path=repair_minus, status="proxy_pass", blockers=0)

    matrix = build_minus_candidate_score_matrix(
        result_judge_jsonls=[result_jsonl],
        self_judge_manifests=[aggressive_self, normal_self, repair_self],
        selector_jsonls=[],
        listen_manifests=[],
    )

    rows = matrix["rows"]
    assert rows[0]["canonical_candidate_id"] == "aggressive"
    assert rows[0]["result_judge"]["accepted"] is True
    assert rows[0]["self_judge"]["status"] == "proxy_blocked"
    assert rows[0]["next_action"] == "run_segment_router_for_self_blockers"
    assert rows[0]["delivery_allowed"] is False
    assert rows[0]["release_ready"] is False
    assert rows[1]["canonical_candidate_id"] == "repair"
    assert rows[1]["next_action"] == "do_not_release_consider_leakage_repair_or_new_model_route"
    assert matrix["summary"]["needs_segment_router"] == 1
    assert matrix["summary"]["release_ready"] is False
    assert matrix["summary"]["safety"]["thresholds_relaxed"] is False


def test_matrix_prefers_fewer_self_blockers_between_result_accepted_candidates(tmp_path: Path) -> None:
    high_score_manifest = tmp_path / "high_score_many_blockers" / "stem_manifest.json"
    low_blocker_manifest = tmp_path / "low_score_few_blockers" / "stem_manifest.json"
    high_score_minus = tmp_path / "high_score_many_blockers" / "minus.flac"
    low_blocker_minus = tmp_path / "low_score_few_blockers" / "minus.flac"
    stem_manifest(high_score_manifest, model_name="high_score_many_blockers")
    stem_manifest(low_blocker_manifest, model_name="low_score_few_blockers")
    result_jsonl = tmp_path / "judge.jsonl"
    write_jsonl(
        result_jsonl,
        [
            result_row(
                "high-score",
                manifest=high_score_manifest,
                minus_path=high_score_minus,
                minus_score=0.98,
                confidence=0.95,
                decision="accept_candidate",
            ),
            result_row(
                "low-blocker",
                manifest=low_blocker_manifest,
                minus_path=low_blocker_minus,
                minus_score=0.84,
                confidence=0.80,
                decision="accept_candidate",
            ),
        ],
    )
    high_score_self = tmp_path / "high-score-self" / "instrumental_self_judge_manifest.json"
    low_blocker_self = tmp_path / "low-blocker-self" / "instrumental_self_judge_manifest.json"
    self_manifest(
        high_score_self,
        label="high_score_many_blockers",
        minus_path=high_score_minus,
        status="proxy_blocked",
        blockers=8,
    )
    self_manifest(
        low_blocker_self,
        label="low_score_few_blockers",
        minus_path=low_blocker_minus,
        status="proxy_blocked",
        blockers=1,
    )

    matrix = build_minus_candidate_score_matrix(
        result_judge_jsonls=[result_jsonl],
        self_judge_manifests=[high_score_self, low_blocker_self],
        selector_jsonls=[],
        listen_manifests=[],
    )

    rows = matrix["rows"]
    assert rows[0]["canonical_candidate_id"] == "low_score_few_blockers"
    assert rows[0]["self_judge"]["blocker_count"] == 1
    assert rows[1]["canonical_candidate_id"] == "high_score_many_blockers"
    assert rows[1]["self_judge"]["blocker_count"] == 8
    assert matrix["summary"]["top_candidate_id"] == "low_score_few_blockers"
    assert matrix["summary"]["top_next_action"] == "run_segment_router_for_self_blockers"


def test_matrix_blocks_minus_accept_when_pair_gate_is_incompatible(tmp_path: Path) -> None:
    manifest = tmp_path / "pair_blocked" / "stem_manifest.json"
    minus_path = tmp_path / "pair_blocked" / "minus.flac"
    stem_manifest(manifest, model_name="ir_minus")
    result_jsonl = tmp_path / "judge.jsonl"
    write_jsonl(
        result_jsonl,
        [
            result_row(
                "pair-blocked",
                manifest=manifest,
                minus_path=minus_path,
                minus_score=0.879,
                confidence=0.66,
                decision="accept_candidate",
                reasons=["phase_or_stereo_inconsistency"],
                pair_decision="blocked_pair_incompatible",
                pair_blockers=["phase_collapse"],
            )
        ],
    )

    matrix = build_minus_candidate_score_matrix(
        result_judge_jsonls=[result_jsonl],
        self_judge_manifests=[],
        selector_jsonls=[],
        listen_manifests=[],
    )

    row = matrix["rows"][0]
    assert row["result_judge"]["accepted"] is False
    assert row["result_judge"]["pair_decision"] == "blocked_pair_incompatible"
    assert row["result_judge"]["pair_blockers"] == ["phase_collapse"]
    assert "pair_incompatible" in row["result_judge"]["reasons"]
    assert row["next_action"] == "blocked_generate_or_select_better_minus_candidate"
    assert row["delivery_allowed"] is False
    assert row["release_ready"] is False


def test_matrix_keeps_cross_product_rows_separate_when_manifest_is_jsonl(tmp_path: Path) -> None:
    candidate_pool = tmp_path / "candidate_pool.jsonl"
    write_jsonl(
        candidate_pool,
        [
            {"candidate_id": "source-minus-a", "model_id": "minus-a"},
            {"candidate_id": "source-vocal-a", "model_id": "vocal-a"},
        ],
    )
    result_jsonl = tmp_path / "cross_judge.jsonl"
    write_jsonl(
        result_jsonl,
        [
            result_row(
                "cross-pair-a",
                manifest=candidate_pool,
                minus_path=tmp_path / "cross-a" / "minus.flac",
                minus_score=0.88,
                confidence=0.72,
                decision="accept_candidate",
            ),
            result_row(
                "cross-pair-b",
                manifest=candidate_pool,
                minus_path=tmp_path / "cross-b" / "minus.flac",
                minus_score=0.84,
                confidence=0.70,
                decision="accept_candidate",
            ),
        ],
    )

    matrix = build_minus_candidate_score_matrix(
        result_judge_jsonls=[result_jsonl],
        self_judge_manifests=[],
        selector_jsonls=[],
        listen_manifests=[],
    )

    rows = matrix["rows"]
    assert {row["canonical_candidate_id"] for row in rows} == {"cross-pair-a", "cross-pair-b"}
    assert len(rows) == 2
    assert all(row["manifest_kind"] == "jsonl_candidate_pool" for row in rows)
    assert all(row["manifest_readable"] is True for row in rows)


def test_matrix_marks_candidate_ready_only_when_result_self_and_selector_accept(tmp_path: Path) -> None:
    manifest = tmp_path / "ready" / "stem_manifest.json"
    minus_path = tmp_path / "ready" / "minus.flac"
    stem_manifest(manifest, model_name="ready_model")
    result_jsonl = tmp_path / "judge.jsonl"
    selector_jsonl = tmp_path / "selector.jsonl"
    self_json = tmp_path / "self" / "instrumental_self_judge_manifest.json"
    write_jsonl(
        result_jsonl,
        [
            result_row(
                "ready-candidate",
                manifest=manifest,
                minus_path=minus_path,
                minus_score=0.82,
                confidence=0.8,
                decision="accept_candidate",
            )
        ],
    )
    write_jsonl(
        selector_jsonl,
        [
            {
                "track_id": "track-a",
                "score_mode": "minus",
                "decision": "accept_candidate",
                "selected_candidate_id": "ready-candidate",
                "ranked_candidates": [{"candidate_id": "ready-candidate", "rank": 1, "selection_score": 0.82}],
            }
        ],
    )
    self_manifest(self_json, label="ready", minus_path=minus_path, status="proxy_pass", blockers=0)

    matrix = build_minus_candidate_score_matrix(
        result_judge_jsonls=[result_jsonl],
        self_judge_manifests=[self_json],
        selector_jsonls=[selector_jsonl],
        listen_manifests=[],
    )

    row = matrix["rows"][0]
    assert row["canonical_candidate_id"] == "ready"
    assert row["next_action"] == "candidate_ready_for_listen_preview_contract"
    assert row["selection_priority"] > 80
    assert matrix["summary"]["ready_for_listen_preview_contract"] == 1

    output_dir = tmp_path / "out"
    write_outputs(matrix, output_dir)
    assert (output_dir / "no_reference_minus_candidate_score_matrix.json").exists()
    assert (output_dir / "no_reference_minus_candidate_score_matrix.jsonl").exists()
    assert (output_dir / "no_reference_minus_candidate_score_matrix_summary.md").exists()
    assert not list(output_dir.rglob("*.wav"))


def test_matrix_preserves_result_judge_route_provenance_without_manifest(tmp_path: Path) -> None:
    result_jsonl = tmp_path / "judge.jsonl"
    minus_path = tmp_path / "minus.flac"
    row = result_row(
        "candidate-a",
        manifest=tmp_path / "missing_manifest.json",
        minus_path=minus_path,
        minus_score=0.92,
        confidence=0.7,
        decision="blocked_hard_case",
        reasons=["severe_phase_collapse"],
    )
    row["candidate_manifest"] = ""
    row["route_id"] = "local_registered__becruily_instrumental_vocals__both__minus_hard_case"
    row["route_family"] = "other_existing"
    row["candidate_config_id"] = "local_registered__becruily_instrumental_vocals__both__minus_hard_case"
    write_jsonl(result_jsonl, [row])

    matrix = build_minus_candidate_score_matrix(
        result_judge_jsonls=[result_jsonl],
        self_judge_manifests=[],
        selector_jsonls=[],
        listen_manifests=[],
    )

    matrix_row = matrix["rows"][0]
    assert matrix_row["model_name"] == "becruily_instrumental_vocals"
    assert matrix_row["route_id"] == "local_registered__becruily_instrumental_vocals__both__minus_hard_case"
    assert matrix_row["route_family"] == "other_existing"
    assert matrix_row["candidate_config_id"] == "local_registered__becruily_instrumental_vocals__both__minus_hard_case"
