from __future__ import annotations

import json
from pathlib import Path

from tools.artifact_status_contract import can_promote_to_listen_result
from tools.build_no_reference_minus_listen_preview_manifest import build_manifest, main


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


def candidate_manifest(minus_path: str = "/tmp/minus.wav") -> dict:
    return {
        "manifest_type": "no_reference_minus_segment_switch_candidate",
        "render_status": "rendered_minus_segment_switch_candidate",
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "preview_only_not_for_listen_pack",
        "release_status": "not_release_ready",
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "requires_post_render_minus_self_judge": True,
        "rendered_minus_candidate_path": minus_path,
        "candidate_row": {
            "track_id": "track-a",
            "candidate_id": "minus-switch-a",
            "mix_path": "/tmp/mix.wav",
            "vocal_path": "/tmp/vocal.wav",
            "instrumental_path": minus_path,
            "minus_path": minus_path,
            "selected_vocal_path": "/tmp/vocal.wav",
            "selected_minus_path": minus_path,
        },
    }


def existing_candidate_manifest(minus_path: str = "/tmp/minus.wav") -> dict:
    payload = candidate_manifest(minus_path)
    payload["manifest_type"] = "no_reference_minus_existing_candidate"
    payload["render_status"] = "existing_minus_candidate_proxy_pass"
    payload["audio_exported"] = False
    payload["rendered_minus_candidate_path"] = minus_path
    return payload


def air_detail_candidate_manifest(minus_path: str = "/tmp/minus.wav") -> dict:
    payload = candidate_manifest(minus_path)
    payload["manifest_type"] = "judge_guided_minus_air_detail_repair_candidate"
    payload["render_status"] = "rendered_minus_air_detail_repair_candidate"
    payload["candidate_row"]["candidate_family"] = "judge_guided_minus_air_detail_repair"
    payload["candidate_row"]["candidate_origin"] = "self_judge_guided_air_detail_repair_candidate"
    return payload


def leakage_repair_candidate_manifest(minus_path: str = "/tmp/minus.wav") -> dict:
    payload = candidate_manifest(minus_path)
    payload.pop("render_status", None)
    payload.pop("requires_post_render_minus_self_judge", None)
    payload["manifest_type"] = "judge_guided_minus_vocal_leakage_repair_candidate"
    payload["mode"] = "judge_guided_minus_vocal_leakage_repair"
    payload["candidate_only"] = True
    payload["delivery_status"] = "candidate_only"
    payload["delivery_allowed"] = False
    payload["presentation_status"] = "blocked_until_judged"
    payload["release_status"] = "not_release_ready"
    payload["listenable"] = False
    payload["deliverable"] = False
    payload["requires_post_judge"] = True
    payload["candidate_row"]["candidate_family"] = "judge_guided_minus_vocal_leakage_repair"
    payload["candidate_row"]["candidate_origin"] = "judge_guided_projection_repair_candidate"
    payload["candidate_row"]["requires_post_render_minus_self_judge"] = True
    payload["candidate_row"]["requires_dense_post_reconciliation_judge"] = True
    return payload


def self_judge_manifest(minus_path: str = "/tmp/minus.wav", *, status: str = "proxy_pass", blockers: int = 0) -> dict:
    return {
        "mode": "instrumental_self_judge_gate",
        "candidate": {
            "label": "minus-switch-a",
            "instrumental_path": minus_path,
            "vocal_path": "/tmp/vocal.wav",
        },
        "overall": {
            "status": status,
            "blocker_count": blockers,
            "segment_count": 40,
        },
        "delivery_contract": {
            "delivery_allowed": False,
            "presentation_status": "diagnostic_self_judge_not_for_listen_pack",
        },
    }


def result_judge_row(
    minus_path: str = "/tmp/minus.wav",
    *,
    decision: str = "accept_candidate",
    score_mode: str = "minus",
    include_detail_gate: bool = True,
    detail_gate_status: str = "proxy_pass",
    detail_gate_blockers: list[dict] | None = None,
) -> dict:
    row = {
        "track_id": "track-a",
        "candidate_id": "minus-switch-a",
        "mix_path": "/tmp/mix.wav",
        "vocal_path": "/tmp/vocal.wav",
        "instrumental_path": minus_path,
        "score_mode": score_mode,
        "decision": decision,
        "minus_decision": decision,
        "minus_score": 0.91,
        "minus_blockers": [],
        "window_count": 12,
        "overall_no_reference_score": 0.91,
        "confidence": 0.88,
        "reasons": ["no_reference_proxy_passed"] if decision == "accept_candidate" else ["vocal_leakage_in_minus"],
    }
    if include_detail_gate:
        blockers = detail_gate_blockers or []
        row["minus_musical_detail_preservation_gate"] = {
            "status": detail_gate_status,
            "evidence_type": "no_reference_minus_detail_proxy",
            "required_metrics": [
                "instrumental_integrity",
                "transient_preservation",
                "spectral_balance",
                "phase_stereo_stability",
                "mixture_consistency_score",
            ],
            "metrics": {
                "instrumental_integrity": 0.91,
                "transient_preservation": 0.88,
                "spectral_balance": 0.86,
                "phase_stereo_stability": 0.9,
                "mixture_consistency_score": 0.92,
            },
            "blockers": blockers,
        }
    return row


def selector_row(
    minus_path: str = "/tmp/minus.wav",
    *,
    decision: str = "accept_candidate",
    score_mode: str = "minus",
) -> dict:
    return {
        "track_id": "track-a",
        "score_mode": score_mode,
        "decision": decision,
        "selected_candidate": result_judge_row(minus_path, decision="accept_candidate"),
        "reasons": ["selected_by_no_reference_score", "no_reference_proxy_passed"],
        "candidate_count": 1,
        "accepted_candidate_count": 1 if decision == "accept_candidate" else 0,
    }


def test_minus_listen_preview_blocks_result_judge_candidate_mismatch(tmp_path: Path) -> None:
    wrong_row = result_judge_row()
    wrong_row["candidate_id"] = "other-candidate"
    manifest = build_manifest(
        candidate_manifest_json=write_json(tmp_path / "candidate.json", candidate_manifest()),
        self_judge_manifest_json=write_json(tmp_path / "self_judge.json", self_judge_manifest()),
        result_judge_jsonl=write_jsonl(tmp_path / "result_judge.jsonl", [wrong_row]),
        selector_jsonl=write_jsonl(tmp_path / "selector.jsonl", [selector_row()]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "result_judge_candidate_mismatch"


def test_minus_listen_preview_blocks_non_minus_result_judge_mode(tmp_path: Path) -> None:
    manifest = build_manifest(
        candidate_manifest_json=write_json(tmp_path / "candidate.json", candidate_manifest()),
        self_judge_manifest_json=write_json(tmp_path / "self_judge.json", self_judge_manifest()),
        result_judge_jsonl=write_jsonl(tmp_path / "result_judge.jsonl", [result_judge_row(score_mode="both")]),
        selector_jsonl=write_jsonl(tmp_path / "selector.jsonl", [selector_row()]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "result_judge_not_minus_mode"


def test_minus_listen_preview_blocks_non_minus_selector_mode(tmp_path: Path) -> None:
    manifest = build_manifest(
        candidate_manifest_json=write_json(tmp_path / "candidate.json", candidate_manifest()),
        self_judge_manifest_json=write_json(tmp_path / "self_judge.json", self_judge_manifest()),
        result_judge_jsonl=write_jsonl(tmp_path / "result_judge.jsonl", [result_judge_row()]),
        selector_jsonl=write_jsonl(tmp_path / "selector.jsonl", [selector_row(score_mode="both")]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "selector_not_minus_mode"


def test_minus_listen_preview_blocks_selector_candidate_mismatch(tmp_path: Path) -> None:
    wrong_selector = selector_row()
    wrong_selector["selected_candidate"]["candidate_id"] = "other-candidate"
    manifest = build_manifest(
        candidate_manifest_json=write_json(tmp_path / "candidate.json", candidate_manifest()),
        self_judge_manifest_json=write_json(tmp_path / "self_judge.json", self_judge_manifest()),
        result_judge_jsonl=write_jsonl(tmp_path / "result_judge.jsonl", [result_judge_row()]),
        selector_jsonl=write_jsonl(tmp_path / "selector.jsonl", [wrong_selector]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "selector_candidate_mismatch"


def test_minus_listen_preview_blocks_generic_only_result_judge_evidence(tmp_path: Path) -> None:
    row = result_judge_row()
    del row["minus_score"]
    del row["minus_decision"]
    del row["minus_blockers"]
    row["overall_no_reference_score"] = 0.99

    manifest = build_manifest(
        candidate_manifest_json=write_json(tmp_path / "candidate.json", candidate_manifest()),
        self_judge_manifest_json=write_json(tmp_path / "self_judge.json", self_judge_manifest()),
        result_judge_jsonl=write_jsonl(tmp_path / "result_judge.jsonl", [row]),
        selector_jsonl=write_jsonl(tmp_path / "selector.jsonl", [selector_row()]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "result_judge_missing_minus_score"


def test_minus_listen_preview_blocks_insufficient_result_judge_windows(tmp_path: Path) -> None:
    row = result_judge_row()
    row["window_count"] = 1

    manifest = build_manifest(
        candidate_manifest_json=write_json(tmp_path / "candidate.json", candidate_manifest()),
        self_judge_manifest_json=write_json(tmp_path / "self_judge.json", self_judge_manifest()),
        result_judge_jsonl=write_jsonl(tmp_path / "result_judge.jsonl", [row]),
        selector_jsonl=write_jsonl(tmp_path / "selector.jsonl", [selector_row()]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "result_judge_insufficient_window_count"


def test_minus_listen_preview_accepts_only_when_all_evidence_passes(tmp_path: Path) -> None:
    candidate_path = write_json(tmp_path / "candidate.json", candidate_manifest())
    self_judge_path = write_json(tmp_path / "self_judge.json", self_judge_manifest())
    result_judge_path = write_jsonl(tmp_path / "result_judge.jsonl", [result_judge_row()])
    selector_path = write_jsonl(tmp_path / "selector.jsonl", [selector_row()])

    manifest = build_manifest(
        candidate_manifest_json=candidate_path,
        self_judge_manifest_json=self_judge_path,
        result_judge_jsonl=result_judge_path,
        selector_jsonl=selector_path,
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["presentation_status"] == "accepted_for_listen_pack"
    assert manifest["release_status"] == "listen_preview_only"
    assert manifest["stem_type"] == "minus"
    assert manifest["accepted_count"] == 1
    assert manifest["selected_minus_path"] == "/tmp/minus.wav"
    assert manifest["can_mark_release_ready"] is False
    assert manifest["audio_copied"] is False
    assert manifest["audio_exported"] is False


def test_minus_listen_preview_blocks_missing_musical_detail_preservation_gate(tmp_path: Path) -> None:
    manifest = build_manifest(
        candidate_manifest_json=write_json(tmp_path / "candidate.json", candidate_manifest()),
        self_judge_manifest_json=write_json(tmp_path / "self_judge.json", self_judge_manifest()),
        result_judge_jsonl=write_jsonl(
            tmp_path / "result_judge.jsonl", [result_judge_row(include_detail_gate=False)]
        ),
        selector_jsonl=write_jsonl(
            tmp_path / "selector.jsonl", [selector_row()]
        ),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "minus_detail_preservation_gate_missing"


def test_minus_listen_preview_blocks_failed_musical_detail_preservation_gate(tmp_path: Path) -> None:
    manifest = build_manifest(
        candidate_manifest_json=write_json(tmp_path / "candidate.json", candidate_manifest()),
        self_judge_manifest_json=write_json(tmp_path / "self_judge.json", self_judge_manifest()),
        result_judge_jsonl=write_jsonl(
            tmp_path / "result_judge.jsonl",
            [
                result_judge_row(
                    detail_gate_status="proxy_blocked",
                    detail_gate_blockers=[
                        {
                            "label": "instrumental_presence_dip",
                            "reason": "minus_candidate_loses_music_detail",
                        }
                    ],
                )
            ],
        ),
        selector_jsonl=write_jsonl(tmp_path / "selector.jsonl", [selector_row()]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "minus_detail_preservation_gate_blockers"


def test_minus_listen_preview_blocks_incomplete_musical_detail_metrics(tmp_path: Path) -> None:
    row = result_judge_row()
    row["minus_musical_detail_preservation_gate"]["required_metrics"] = []
    row["minus_musical_detail_preservation_gate"]["metrics"] = {
        "instrumental_integrity": 0.91,
    }

    manifest = build_manifest(
        candidate_manifest_json=write_json(tmp_path / "candidate.json", candidate_manifest()),
        self_judge_manifest_json=write_json(tmp_path / "self_judge.json", self_judge_manifest()),
        result_judge_jsonl=write_jsonl(tmp_path / "result_judge.jsonl", [row]),
        selector_jsonl=write_jsonl(tmp_path / "selector.jsonl", [selector_row()]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "minus_detail_preservation_metrics_missing"


def test_minus_listen_preview_blocks_non_finite_musical_detail_metrics(tmp_path: Path) -> None:
    row = result_judge_row()
    row["minus_musical_detail_preservation_gate"]["metrics"]["transient_preservation"] = float("nan")

    manifest = build_manifest(
        candidate_manifest_json=write_json(tmp_path / "candidate.json", candidate_manifest()),
        self_judge_manifest_json=write_json(tmp_path / "self_judge.json", self_judge_manifest()),
        result_judge_jsonl=write_jsonl(tmp_path / "result_judge.jsonl", [row]),
        selector_jsonl=write_jsonl(tmp_path / "selector.jsonl", [selector_row()]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "minus_detail_preservation_metrics_missing"


def test_minus_listen_preview_accepts_existing_proxy_pass_candidate(tmp_path: Path) -> None:
    candidate_path = write_json(tmp_path / "candidate.json", existing_candidate_manifest())
    self_judge_path = write_json(tmp_path / "self_judge.json", self_judge_manifest())
    result_judge_path = write_jsonl(tmp_path / "result_judge.jsonl", [result_judge_row()])
    selector_path = write_jsonl(tmp_path / "selector.jsonl", [selector_row()])

    manifest = build_manifest(
        candidate_manifest_json=candidate_path,
        self_judge_manifest_json=self_judge_path,
        result_judge_jsonl=result_judge_path,
        selector_jsonl=selector_path,
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["presentation_status"] == "accepted_for_listen_pack"
    assert manifest["selected_minus_path"] == "/tmp/minus.wav"


def test_minus_listen_preview_accepts_air_detail_repair_candidate(tmp_path: Path) -> None:
    candidate_path = write_json(tmp_path / "candidate.json", air_detail_candidate_manifest())
    self_judge_path = write_json(tmp_path / "self_judge.json", self_judge_manifest())
    result_judge_path = write_jsonl(tmp_path / "result_judge.jsonl", [result_judge_row()])
    selector_path = write_jsonl(tmp_path / "selector.jsonl", [selector_row()])

    manifest = build_manifest(
        candidate_manifest_json=candidate_path,
        self_judge_manifest_json=self_judge_path,
        result_judge_jsonl=result_judge_path,
        selector_jsonl=selector_path,
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["presentation_status"] == "accepted_for_listen_pack"
    assert manifest["selected_minus_path"] == "/tmp/minus.wav"


def test_minus_listen_preview_accepts_judge_guided_leakage_repair_after_post_gates(tmp_path: Path) -> None:
    candidate_path = write_json(tmp_path / "candidate.json", leakage_repair_candidate_manifest())
    self_judge_path = write_json(tmp_path / "self_judge.json", self_judge_manifest())
    result_judge_path = write_jsonl(tmp_path / "result_judge.jsonl", [result_judge_row()])
    selector_path = write_jsonl(tmp_path / "selector.jsonl", [selector_row()])

    manifest = build_manifest(
        candidate_manifest_json=candidate_path,
        self_judge_manifest_json=self_judge_path,
        result_judge_jsonl=result_judge_path,
        selector_jsonl=selector_path,
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["presentation_status"] == "accepted_for_listen_pack"
    assert manifest["release_status"] == "listen_preview_only"
    assert manifest["can_mark_release_ready"] is False
    assert manifest["selected_minus_path"] == "/tmp/minus.wav"


def test_minus_listen_preview_blocks_candidate_row_delivery_claims(tmp_path: Path) -> None:
    candidate = leakage_repair_candidate_manifest()
    candidate["candidate_row"]["can_publish_as_result"] = True
    manifest = build_manifest(
        candidate_manifest_json=write_json(tmp_path / "candidate.json", candidate),
        self_judge_manifest_json=write_json(tmp_path / "self_judge.json", self_judge_manifest()),
        result_judge_jsonl=write_jsonl(tmp_path / "result_judge.jsonl", [result_judge_row()]),
        selector_jsonl=write_jsonl(tmp_path / "selector.jsonl", [selector_row()]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "candidate_manifest_claims_delivery"


def test_minus_listen_preview_manifest_is_promotable_by_artifact_contract(tmp_path: Path) -> None:
    candidate_path = write_json(tmp_path / "candidate.json", candidate_manifest())
    self_judge_path = write_json(tmp_path / "self_judge.json", self_judge_manifest())
    result_judge_path = write_jsonl(tmp_path / "result_judge.jsonl", [result_judge_row()])
    selector_path = write_jsonl(tmp_path / "selector.jsonl", [selector_row()])
    output_manifest = tmp_path / "minus_listen_preview_manifest.json"
    summary_md = tmp_path / "summary.md"

    assert (
        main(
            [
                "--candidate-manifest-json",
                str(candidate_path),
                "--self-judge-manifest-json",
                str(self_judge_path),
                "--result-judge-jsonl",
                str(result_judge_path),
                "--selector-jsonl",
                str(selector_path),
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


def test_minus_listen_preview_blocks_failed_self_judge(tmp_path: Path) -> None:
    manifest = build_manifest(
        candidate_manifest_json=write_json(tmp_path / "candidate.json", candidate_manifest()),
        self_judge_manifest_json=write_json(
            tmp_path / "self_judge.json", self_judge_manifest(status="proxy_blocked", blockers=1)
        ),
        result_judge_jsonl=write_jsonl(tmp_path / "result_judge.jsonl", [result_judge_row()]),
        selector_jsonl=write_jsonl(tmp_path / "selector.jsonl", [selector_row()]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "post_render_self_judge_not_passed"
    assert manifest["accepted_count"] == 0


def test_minus_listen_preview_blocks_non_accepted_selector(tmp_path: Path) -> None:
    manifest = build_manifest(
        candidate_manifest_json=write_json(tmp_path / "candidate.json", candidate_manifest()),
        self_judge_manifest_json=write_json(tmp_path / "self_judge.json", self_judge_manifest()),
        result_judge_jsonl=write_jsonl(tmp_path / "result_judge.jsonl", [result_judge_row()]),
        selector_jsonl=write_jsonl(tmp_path / "selector.jsonl", [selector_row(decision="blocked_hard_case")]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "selector_not_accepted"


def test_minus_listen_preview_blocks_path_mismatch(tmp_path: Path) -> None:
    manifest = build_manifest(
        candidate_manifest_json=write_json(tmp_path / "candidate.json", candidate_manifest("/tmp/minus.wav")),
        self_judge_manifest_json=write_json(tmp_path / "self_judge.json", self_judge_manifest("/tmp/other.wav")),
        result_judge_jsonl=write_jsonl(tmp_path / "result_judge.jsonl", [result_judge_row()]),
        selector_jsonl=write_jsonl(tmp_path / "selector.jsonl", [selector_row()]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["dominant_failure_reason"] == "evidence_path_mismatch"


def test_minus_listen_preview_output_is_deterministic(tmp_path: Path) -> None:
    candidate_path = write_json(tmp_path / "candidate.json", candidate_manifest())
    self_judge_path = write_json(tmp_path / "self_judge.json", self_judge_manifest())
    result_judge_path = write_jsonl(tmp_path / "result_judge.jsonl", [result_judge_row()])
    selector_path = write_jsonl(tmp_path / "selector.jsonl", [selector_row()])

    first = build_manifest(
        candidate_manifest_json=candidate_path,
        self_judge_manifest_json=self_judge_path,
        result_judge_jsonl=result_judge_path,
        selector_jsonl=selector_path,
    )
    second = build_manifest(
        candidate_manifest_json=candidate_path,
        self_judge_manifest_json=self_judge_path,
        result_judge_jsonl=result_judge_path,
        selector_jsonl=selector_path,
    )

    assert second == first
