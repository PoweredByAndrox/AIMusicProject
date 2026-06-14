from __future__ import annotations

import json
from pathlib import Path

from tools.propose_policy_taxonomy_fixes import build_policy_taxonomy_proposals, write_outputs


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def root_row(
    idx: int,
    *,
    stem_type: str = "vocal",
    split: str = "train",
    bucket: str = "policy_classification_problem",
    action: str = "improve_taxonomy_mapping",
    can_change_policy: bool = True,
    reason: str = "selected_candidate_has_unstable_vocal_consistency",
    labels: list[str] | None = None,
) -> dict:
    failure_labels = labels if labels is not None else [reason]
    return {
        "case_id": f"bench-{idx:03d}",
        "track_hash": f"track-{idx:03d}",
        "stem_type": stem_type,
        "split": split,
        "original_decision": "blocked_from_listen_pack",
        "dominant_failure_reason": reason,
        "failure_labels": failure_labels,
        "judge_mode": "no_reference_proxy",
        "confidence": 0.9,
        "root_cause_bucket": bucket,
        "safe_next_action": action,
        "forbidden_actions": ["no_audio_export", "no_policy_auto_change"],
        "required_evidence_before_execution": ["policy_issue_labels"],
        "can_execute_automatically": False,
        "can_enter_finetune_pool": False,
        "can_change_policy": can_change_policy,
        "can_request_new_model": False,
        "human_review_required": False,
    }


def make_inputs(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path]:
    root_plan = write_jsonl(tmp_path / "root_cause_plan.jsonl", rows)
    manifest = write_json(
        tmp_path / "root_cause_plan_manifest.json",
        {
            "schema": "root_cause_plan_manifest.v1",
            "row_count": len(rows),
            "root_cause_bucket_counts": {},
            "safety": {
                "audio_copied": False,
                "training_started": False,
                "policy_changed": False,
                "models_changed": False,
                "weights_added": False,
                "cleanup_routes_added": False,
                "thresholds_changed": False,
            },
        },
    )
    return root_plan, manifest


def test_policy_proposals_include_only_policy_mapping_rows(tmp_path: Path) -> None:
    rows = [
        root_row(1),
        root_row(2, can_change_policy=False, action="improve_taxonomy_mapping"),
        root_row(3, bucket="vocal_specific_failure", action="investigate_vocal_specific_path", can_change_policy=False),
    ]
    root_plan, manifest = make_inputs(tmp_path, rows)

    result = build_policy_taxonomy_proposals(root_cause_plan_jsonl=root_plan, root_cause_manifest_json=manifest)

    assert [row["case_id"] for row in result["rows"]] == ["bench-001", "bench-002"]
    assert all(row["requires_human_approval"] is True for row in result["rows"])
    assert all(row["can_apply_automatically"] is False for row in result["rows"])
    assert all(row["proposed_test_name"] for row in result["rows"])
    assert result["summary"]["total_proposals"] == len(result["rows"])
    assert result["manifest"]["total_proposals"] == len(result["rows"])


def test_minus_and_holdout_rows_are_marked_safely(tmp_path: Path) -> None:
    rows = [
        root_row(
            1,
            stem_type="minus",
            reason="selected_candidate_has_unstable_vocal_consistency",
            labels=["selected_candidate_has_unstable_vocal_consistency", "vocal_tail_loss"],
        ),
        root_row(2, split="holdout", reason="selected_candidate_still_has_clustered_issues"),
    ]
    root_plan, manifest = make_inputs(tmp_path, rows)

    result = build_policy_taxonomy_proposals(root_cause_plan_jsonl=root_plan, root_cause_manifest_json=manifest)
    by_case = {row["case_id"]: row for row in result["rows"]}

    assert by_case["bench-001"]["proposed_taxonomy_mapping"]["stem_scope"] == "minus"
    assert by_case["bench-001"]["proposed_taxonomy_mapping"]["target_root_cause_bucket"] != "vocal_specific_failure"
    assert by_case["bench-002"]["proposed_taxonomy_mapping"]["protected_holdout"] is True
    assert by_case["bench-002"]["risk_level"] == "high"


def test_unknown_reasons_are_not_automatic_finetune_candidates(tmp_path: Path) -> None:
    rows = [
        root_row(
            1,
            reason="unknown_blocked_reason",
            labels=["unknown_blocked_reason"],
            can_change_policy=True,
        )
    ]
    root_plan, manifest = make_inputs(tmp_path, rows)

    result = build_policy_taxonomy_proposals(root_cause_plan_jsonl=root_plan, root_cause_manifest_json=manifest)
    proposal = result["rows"][0]

    assert proposal["proposed_policy_change_type"] == "clarify_unknown_reason"
    assert proposal["proposed_taxonomy_mapping"]["target_root_cause_bucket"] != "current_model_insufficient"
    assert "fine-tune" not in proposal["expected_behavior_after_fix"].lower()


def test_output_is_deterministic_and_summary_counts_match_jsonl(tmp_path: Path) -> None:
    rows = [
        root_row(2, reason="selected_candidate_still_has_clustered_issues"),
        root_row(1, labels=["hard_vad_gate", "stronger_debleed"]),
    ]
    root_plan, manifest = make_inputs(tmp_path, rows)

    first = build_policy_taxonomy_proposals(root_cause_plan_jsonl=root_plan, root_cause_manifest_json=manifest)
    second = build_policy_taxonomy_proposals(root_cause_plan_jsonl=root_plan, root_cause_manifest_json=manifest)

    assert first["rows"] == second["rows"]
    output_dir = tmp_path / "out"
    write_outputs(first, output_dir)
    output_rows = [
        json.loads(line)
        for line in (output_dir / "policy_taxonomy_fix_proposal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_payload = json.loads((output_dir / "policy_taxonomy_fix_manifest.json").read_text(encoding="utf-8"))
    assert len(output_rows) == first["summary"]["total_proposals"]
    assert manifest_payload["total_proposals"] == len(output_rows)
    assert (output_dir / "proposed_tests_patch.md").exists()
    assert not list(output_dir.glob("*.flac"))
    assert manifest_payload["safety"]["policy_changed"] is False
