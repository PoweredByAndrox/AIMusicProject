from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.artifact_status_contract import can_promote_to_listen_result, status_view_from_manifest
from tools.build_no_reference_delivery_gate import build_delivery_gate_manifest, main


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def accepted_optimizer_row() -> dict:
    return {
        "track_id": "track-a",
        "candidate_id": "mode_specific_pair:vocal-a:minus-b",
        "final_decision": "mode_specific_pair_selected",
        "pair_compatibility_decision": "pair_compatible",
        "needs_pair_reconciliation": False,
        "segment_blend_needed": False,
        "mix_path": "/app/mix.wav",
        "selected_vocal_path": "/app/vocal.flac",
        "selected_minus_path": "/app/minus.flac",
        "selected_vocal_candidate_id": "vocal-a",
        "selected_minus_candidate_id": "minus-b",
        "selected_vocal_score": 0.83,
        "selected_minus_score": 0.84,
        "pair_compatibility_score": 0.76,
    }


def accepted_reconciliation_row() -> dict:
    row = accepted_optimizer_row()
    row.update(
        {
            "candidate_id": "reconciled:vocal-a:minus-b",
            "final_decision": "pair_reconciliation_candidate",
            "requires_dense_post_reconciliation_judge": True,
            "candidate_family": "reconciled_vocal_existing_minus",
            "candidate_origin": "no_reference_phase_vocal_existing_minus_reconciliation",
        }
    )
    return row


def accepted_phase_correction_row() -> dict:
    row = accepted_optimizer_row()
    row.update(
        {
            "candidate_id": "phase-corrected:vocal-a:minus-b",
            "final_decision": "phase_corrected_candidate_selected",
            "requires_post_correction_judge": True,
            "selected_candidate_id": "phase-corrected:vocal-a:minus-b",
        }
    )
    return row


def judge_row(mode: str, *, decision: str = "accept_candidate", candidate_id: str = "mode_specific_pair:vocal-a:minus-b") -> dict:
    row = {
        "track_id": "track-a",
        "candidate_id": candidate_id,
        "score_mode": mode,
        "judge_mode": "no_reference_windowed",
        "decision": decision,
        "window_count": 12,
        "window_seconds": 6.0,
        "window_step_seconds": 6.0,
        "reasons": ["no_reference_proxy_passed"] if decision == "accept_candidate" else ["severe_blocker"],
        "vocal_blockers": [] if mode == "vocal" and decision == "accept_candidate" else (["severe_vocal_body_loss"] if mode == "vocal" else []),
        "minus_blockers": [] if mode == "minus" and decision == "accept_candidate" else (["severe_vocal_bleed_in_minus"] if mode == "minus" else []),
        "pair_blockers": [] if mode == "both" and decision == "accept_candidate" else (["phase_collapse"] if mode == "both" else []),
    }
    if mode == "vocal":
        row["vocal_score"] = 0.83 if decision == "accept_candidate" else 0.20
        row["vocal_decision"] = decision
    elif mode == "minus":
        row["minus_score"] = 0.84 if decision == "accept_candidate" else 0.20
        row["minus_decision"] = decision
    elif mode == "both":
        row["pair_compatibility_score"] = 0.76 if decision == "accept_candidate" else 0.20
        row["pair_decision"] = "pair_compatible" if decision == "accept_candidate" else "blocked_pair_incompatible"
    return row


def write_gate_inputs(tmp_path: Path, *, minus_decision: str = "accept_candidate", include_both: bool = True) -> dict[str, Path]:
    optimizer = write_jsonl(tmp_path / "optimizer.jsonl", [accepted_optimizer_row()])
    vocal = write_jsonl(tmp_path / "vocal.jsonl", [judge_row("vocal")])
    minus = write_jsonl(tmp_path / "minus.jsonl", [judge_row("minus", decision=minus_decision)])
    both = write_jsonl(tmp_path / "both.jsonl", [judge_row("both")]) if include_both else tmp_path / "missing_both.jsonl"
    return {"optimizer": optimizer, "vocal": vocal, "minus": minus, "both": both}


def test_gate_accepts_only_when_optimizer_and_all_dense_modes_pass(tmp_path: Path) -> None:
    paths = write_gate_inputs(tmp_path)

    manifest = build_delivery_gate_manifest(
        optimizer_result_jsonl=paths["optimizer"],
        vocal_judge_jsonl=paths["vocal"],
        minus_judge_jsonl=paths["minus"],
        both_judge_jsonl=paths["both"],
        min_window_count=5,
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["presentation_status"] == "accepted_for_listen_pack"
    assert manifest["release_status"] == "listen_preview_only"
    assert manifest["accepted_count"] == 1
    assert manifest["skipped_count"] == 0
    assert manifest["audio_copied"] is False
    assert manifest["audio_exported"] is False
    row = manifest["accepted_listen_preview_rows"][0]
    assert row["dense_delivery_gate"]["vocal"]["decision"] == "accept_candidate"
    assert row["dense_delivery_gate"]["vocal"]["vocal_blockers"] == []
    assert row["dense_delivery_gate"]["minus"]["decision"] == "accept_candidate"
    assert row["dense_delivery_gate"]["minus"]["minus_blockers"] == []
    assert row["dense_delivery_gate"]["both"]["decision"] == "accept_candidate"
    assert row["dense_delivery_gate"]["both"]["pair_blockers"] == []
    assert row["can_mark_release_ready"] is False
    assert manifest["safety"]["audio_copied"] is False
    assert manifest["safety"]["audio_exported"] is False


def test_gate_accepts_reconciliation_candidate_only_after_all_dense_modes_pass(tmp_path: Path) -> None:
    optimizer = write_jsonl(tmp_path / "optimizer.jsonl", [accepted_reconciliation_row()])
    vocal = write_jsonl(tmp_path / "vocal.jsonl", [judge_row("vocal", candidate_id="reconciled:vocal-a:minus-b")])
    minus = write_jsonl(tmp_path / "minus.jsonl", [judge_row("minus", candidate_id="reconciled:vocal-a:minus-b")])
    both = write_jsonl(tmp_path / "both.jsonl", [judge_row("both", candidate_id="reconciled:vocal-a:minus-b")])

    manifest = build_delivery_gate_manifest(
        optimizer_result_jsonl=optimizer,
        vocal_judge_jsonl=vocal,
        minus_judge_jsonl=minus,
        both_judge_jsonl=both,
        min_window_count=5,
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    row = manifest["accepted_listen_preview_rows"][0]
    assert row["candidate_id"] == "reconciled:vocal-a:minus-b"
    assert row["dense_delivery_gate"]["vocal"]["decision"] == "accept_candidate"
    assert row["can_mark_release_ready"] is False


def test_gate_accepts_phase_corrected_candidate_only_after_all_dense_modes_pass(tmp_path: Path) -> None:
    optimizer = write_jsonl(tmp_path / "optimizer.jsonl", [accepted_phase_correction_row()])
    candidate_id = "phase-corrected:vocal-a:minus-b"
    vocal = write_jsonl(tmp_path / "vocal.jsonl", [judge_row("vocal", candidate_id=candidate_id)])
    minus = write_jsonl(tmp_path / "minus.jsonl", [judge_row("minus", candidate_id=candidate_id)])
    both = write_jsonl(tmp_path / "both.jsonl", [judge_row("both", candidate_id=candidate_id)])

    manifest = build_delivery_gate_manifest(
        optimizer_result_jsonl=optimizer,
        vocal_judge_jsonl=vocal,
        minus_judge_jsonl=minus,
        both_judge_jsonl=both,
        min_window_count=5,
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    row = manifest["accepted_listen_preview_rows"][0]
    assert row["candidate_id"] == candidate_id
    assert row["dense_delivery_gate"]["both"]["decision"] == "accept_candidate"
    assert row["can_mark_release_ready"] is False


def test_gate_blocks_when_one_dense_mode_fails(tmp_path: Path) -> None:
    paths = write_gate_inputs(tmp_path, minus_decision="blocked_hard_case")

    manifest = build_delivery_gate_manifest(
        optimizer_result_jsonl=paths["optimizer"],
        vocal_judge_jsonl=paths["vocal"],
        minus_judge_jsonl=paths["minus"],
        both_judge_jsonl=paths["both"],
        min_window_count=5,
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "blocked_from_listen_pack"
    assert manifest["accepted_count"] == 0
    assert manifest["skipped_rows"][0]["skip_reason"] == "dense_delivery_gate_failed"
    assert "minus" in manifest["skipped_rows"][0]["failed_modes"]


def test_gate_blocks_accept_decision_with_semantic_blocker_reason(tmp_path: Path) -> None:
    paths = write_gate_inputs(tmp_path)
    minus_rows = [judge_row("minus")]
    minus_rows[0]["reasons"] = ["foreign_source_recovered:minus:vocals"]
    paths["minus"] = write_jsonl(tmp_path / "minus.jsonl", minus_rows)

    manifest = build_delivery_gate_manifest(
        optimizer_result_jsonl=paths["optimizer"],
        vocal_judge_jsonl=paths["vocal"],
        minus_judge_jsonl=paths["minus"],
        both_judge_jsonl=paths["both"],
        min_window_count=5,
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["accepted_count"] == 0
    assert manifest["skipped_rows"][0]["skip_reason"] == "dense_delivery_gate_failed"
    assert "minus_blocking_semantic_label:foreign_source_recovered:minus:vocals" in manifest["skipped_rows"][0][
        "failed_checks"
    ]


def test_gate_blocks_missing_dense_evidence(tmp_path: Path) -> None:
    paths = write_gate_inputs(tmp_path, include_both=False)

    manifest = build_delivery_gate_manifest(
        optimizer_result_jsonl=paths["optimizer"],
        vocal_judge_jsonl=paths["vocal"],
        minus_judge_jsonl=paths["minus"],
        both_judge_jsonl=paths["both"],
        min_window_count=5,
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "dense_delivery_gate_failed"
    assert "both" in manifest["skipped_rows"][0]["failed_modes"]


def test_gate_blocks_vocal_dense_row_missing_mode_specific_fields(tmp_path: Path) -> None:
    paths = write_gate_inputs(tmp_path)
    vocal_row = judge_row("vocal")
    del vocal_row["vocal_score"]
    del vocal_row["vocal_decision"]
    del vocal_row["vocal_blockers"]
    paths["vocal"] = write_jsonl(tmp_path / "vocal.jsonl", [vocal_row])

    manifest = build_delivery_gate_manifest(
        optimizer_result_jsonl=paths["optimizer"],
        vocal_judge_jsonl=paths["vocal"],
        minus_judge_jsonl=paths["minus"],
        both_judge_jsonl=paths["both"],
        min_window_count=5,
    )

    skipped = manifest["skipped_rows"][0]
    assert manifest["delivery_allowed"] is False
    assert "vocal" in skipped["failed_modes"]
    assert {
        "missing_mode_specific_score:vocal_score",
        "missing_mode_specific_decision:vocal_decision",
        "missing_mode_specific_blockers:vocal_blockers",
    }.issubset(set(skipped["failed_checks"]))


def test_gate_blocks_minus_dense_row_missing_mode_blockers(tmp_path: Path) -> None:
    paths = write_gate_inputs(tmp_path)
    minus_row = judge_row("minus")
    del minus_row["minus_blockers"]
    paths["minus"] = write_jsonl(tmp_path / "minus.jsonl", [minus_row])

    manifest = build_delivery_gate_manifest(
        optimizer_result_jsonl=paths["optimizer"],
        vocal_judge_jsonl=paths["vocal"],
        minus_judge_jsonl=paths["minus"],
        both_judge_jsonl=paths["both"],
        min_window_count=5,
    )

    skipped = manifest["skipped_rows"][0]
    assert manifest["delivery_allowed"] is False
    assert "minus" in skipped["failed_modes"]
    assert "missing_mode_specific_blockers:minus_blockers" in skipped["failed_checks"]


@pytest.mark.parametrize(
    ("mode", "missing_key", "expected_failure"),
    [
        ("minus", "minus_score", "missing_mode_specific_score:minus_score"),
        ("minus", "minus_decision", "missing_mode_specific_decision:minus_decision"),
        ("both", "pair_compatibility_score", "missing_mode_specific_score:pair_compatibility_score"),
        ("both", "pair_blockers", "missing_mode_specific_blockers:pair_blockers"),
    ],
)
def test_gate_blocks_missing_individual_mode_specific_fields(
    tmp_path: Path,
    mode: str,
    missing_key: str,
    expected_failure: str,
) -> None:
    paths = write_gate_inputs(tmp_path)
    row = judge_row(mode)
    del row[missing_key]
    paths[mode] = write_jsonl(tmp_path / f"{mode}.jsonl", [row])

    manifest = build_delivery_gate_manifest(
        optimizer_result_jsonl=paths["optimizer"],
        vocal_judge_jsonl=paths["vocal"],
        minus_judge_jsonl=paths["minus"],
        both_judge_jsonl=paths["both"],
        min_window_count=5,
    )

    skipped = manifest["skipped_rows"][0]
    assert manifest["delivery_allowed"] is False
    assert mode in skipped["failed_modes"]
    assert expected_failure in skipped["failed_checks"]


def test_gate_blocks_generic_both_accept_only_evidence(tmp_path: Path) -> None:
    paths = write_gate_inputs(tmp_path)
    both_row = judge_row("both")
    del both_row["pair_compatibility_score"]
    del both_row["pair_decision"]
    del both_row["pair_blockers"]
    both_row["overall_no_reference_score"] = 0.99
    paths["both"] = write_jsonl(tmp_path / "both.jsonl", [both_row])

    manifest = build_delivery_gate_manifest(
        optimizer_result_jsonl=paths["optimizer"],
        vocal_judge_jsonl=paths["vocal"],
        minus_judge_jsonl=paths["minus"],
        both_judge_jsonl=paths["both"],
        min_window_count=5,
    )

    skipped = manifest["skipped_rows"][0]
    assert manifest["delivery_allowed"] is False
    assert "both" in skipped["failed_modes"]
    assert {
        "missing_mode_specific_score:pair_compatibility_score",
        "missing_mode_specific_decision:pair_decision",
        "missing_mode_specific_blockers:pair_blockers",
    }.issubset(set(skipped["failed_checks"]))


def test_gate_blocks_both_dense_row_missing_pair_decision(tmp_path: Path) -> None:
    paths = write_gate_inputs(tmp_path)
    both_row = judge_row("both")
    del both_row["pair_decision"]
    paths["both"] = write_jsonl(tmp_path / "both.jsonl", [both_row])

    manifest = build_delivery_gate_manifest(
        optimizer_result_jsonl=paths["optimizer"],
        vocal_judge_jsonl=paths["vocal"],
        minus_judge_jsonl=paths["minus"],
        both_judge_jsonl=paths["both"],
        min_window_count=5,
    )

    skipped = manifest["skipped_rows"][0]
    assert manifest["delivery_allowed"] is False
    assert "both" in skipped["failed_modes"]
    assert "missing_mode_specific_decision:pair_decision" in skipped["failed_checks"]


def test_gate_rejects_min_window_count_below_contract(tmp_path: Path) -> None:
    paths = write_gate_inputs(tmp_path)

    with pytest.raises(ValueError, match="min_window_count must be >= 5"):
        build_delivery_gate_manifest(
            optimizer_result_jsonl=paths["optimizer"],
            vocal_judge_jsonl=paths["vocal"],
            minus_judge_jsonl=paths["minus"],
            both_judge_jsonl=paths["both"],
            min_window_count=4,
        )


def test_gate_cli_writes_promotable_manifest_only_for_pass(tmp_path: Path) -> None:
    paths = write_gate_inputs(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    summary_path = tmp_path / "summary.md"

    assert main(
        [
            "--optimizer-result-jsonl",
            str(paths["optimizer"]),
            "--vocal-judge-jsonl",
            str(paths["vocal"]),
            "--minus-judge-jsonl",
            str(paths["minus"]),
            "--both-judge-jsonl",
            str(paths["both"]),
            "--output-manifest-json",
            str(manifest_path),
            "--summary-md",
            str(summary_path),
            "--min-window-count",
            "5",
        ]
    ) == 0

    view = status_view_from_manifest(manifest_path)
    assert view.listenable
    assert not view.deliverable
    assert can_promote_to_listen_result(manifest_path)
    assert summary_path.exists()
    assert not list(tmp_path.rglob("*.flac"))


def test_gate_cli_fails_for_blocked_manifest_after_writing_evidence(tmp_path: Path) -> None:
    paths = write_gate_inputs(tmp_path, minus_decision="blocked_hard_case")
    manifest_path = tmp_path / "blocked_manifest.json"
    summary_path = tmp_path / "blocked_summary.md"

    with pytest.raises(SystemExit, match="dense_delivery_gate_not_promotable"):
        main(
            [
                "--optimizer-result-jsonl",
                str(paths["optimizer"]),
                "--vocal-judge-jsonl",
                str(paths["vocal"]),
                "--minus-judge-jsonl",
                str(paths["minus"]),
                "--both-judge-jsonl",
                str(paths["both"]),
                "--output-manifest-json",
                str(manifest_path),
                "--summary-md",
                str(summary_path),
                "--min-window-count",
                "5",
            ]
        )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["delivery_allowed"] is False
    assert summary_path.exists()
