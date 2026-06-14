from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


PROPOSED_POLICY_CHANGE_TYPES = {
    "relabel_failure_reason",
    "add_alias_mapping",
    "add_forbidden_action",
    "add_allowed_next_action",
    "tighten_promotion_rule",
    "clarify_unknown_reason",
}
SAFETY_FLAGS = {
    "audio_copied": False,
    "training_started": False,
    "policy_changed": False,
    "models_changed": False,
    "weights_added": False,
    "cleanup_routes_added": False,
    "thresholds_changed": False,
}
FORBIDDEN_ACTION_LABELS = {
    "hard_vad_gate",
    "hard_vad_gate_on_tails",
    "stack_more_cleanup",
    "stronger_debleed",
    "stronger_denoise",
    "switch_backbone",
    "global_cleanup",
    "full_track_denoise",
    "aggressive_cleanup",
}
PROMOTION_BLOCK_LABELS = {
    "blocked",
    "blocked_for_listening",
    "blocked_from_delivery",
    "blocked_from_listen_pack",
    "delivery_allowed_false",
    "diagnostic_only",
    "rejected",
    "skipped",
}
POLICY_ALIAS_LABELS = {
    "selected_candidate_has_unstable_vocal_consistency",
    "selected_candidate_still_has_clustered_issues",
    "selected_candidate_has_too_many_issue_windows",
    "candidate_proxy_regressed",
    "proxy_blocked",
}
UNKNOWN_LABELS = {"unknown", "unknown_blocked_reason"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"json_must_be_object:{path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"jsonl_row_must_be_object:{path}:{line_number}")
            yield payload


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.replace(",", ";").split(";") if item.strip()]
    text = str(value).strip()
    return [text] if text else []


def should_include(row: dict[str, Any]) -> bool:
    return bool(row.get("can_change_policy")) or row.get("safe_next_action") == "improve_taxonomy_mapping"


def normalized_signals(row: dict[str, Any]) -> set[str]:
    signals = set(as_list(row.get("failure_labels")))
    signals.add(str(row.get("dominant_failure_reason") or ""))
    signals.add(str(row.get("current_root_cause_bucket") or row.get("root_cause_bucket") or ""))
    signals.add(str(row.get("current_safe_next_action") or row.get("safe_next_action") or ""))
    return {signal for signal in signals if signal}


def choose_change_type(row: dict[str, Any], signals: set[str]) -> str:
    if signals & UNKNOWN_LABELS:
        return "clarify_unknown_reason"
    if signals & PROMOTION_BLOCK_LABELS:
        return "tighten_promotion_rule"
    if signals & FORBIDDEN_ACTION_LABELS:
        return "add_forbidden_action"
    if signals & POLICY_ALIAS_LABELS:
        return "add_alias_mapping"
    if row.get("safe_next_action") == "improve_taxonomy_mapping":
        return "relabel_failure_reason"
    return "add_alias_mapping"


def target_bucket(row: dict[str, Any], signals: set[str]) -> str:
    stem_type = str(row.get("stem_type") or "unknown")
    if signals & UNKNOWN_LABELS:
        return "unknown_needs_more_evidence"
    if stem_type == "minus":
        return "minus_specific_failure"
    return "policy_classification_problem"


def suspected_gap(row: dict[str, Any], change_type: str, signals: set[str]) -> str:
    stem_type = str(row.get("stem_type") or "unknown")
    if change_type == "clarify_unknown_reason":
        return "unknown_or_ambiguous_failure_reason_needs_explicit_non_training_taxonomy"
    if change_type == "tighten_promotion_rule":
        return "blocked_or_diagnostic_status_needs_stronger_taxonomy_guard_before_policy_routing"
    if change_type == "add_forbidden_action":
        return "forbidden_cleanup_or_backbone_action_label_needs_policy_mapping"
    if stem_type == "minus":
        return "minus_failure_labels_need_minus_scoped_policy_taxonomy"
    if change_type == "add_alias_mapping":
        return "proxy_or_clustered_issue_label_needs_alias_to_policy_classification_problem"
    return "dominant_failure_reason_needs_relabeling_to_existing_policy_class"


def risk_level(row: dict[str, Any], change_type: str) -> str:
    if row.get("split") == "holdout":
        return "high"
    if change_type == "tighten_promotion_rule":
        return "high"
    if change_type in {"add_forbidden_action", "clarify_unknown_reason"}:
        return "medium"
    return "medium"


def test_name(case_id: str, change_type: str) -> str:
    safe_case = re.sub(r"[^a-zA-Z0-9_]+", "_", case_id).strip("_").lower()
    return f"test_policy_taxonomy_{change_type}_{safe_case}"


def expected_behavior(row: dict[str, Any], change_type: str, mapping: dict[str, Any]) -> str:
    if change_type == "clarify_unknown_reason":
        return "Unknown/ambiguous policy evidence remains blocked for evidence collection and is not routed to training automatically."
    if change_type == "tighten_promotion_rule":
        return "Diagnostic or blocked status remains non-promotable and policy routing preserves blocked/listen safety."
    if change_type == "add_forbidden_action":
        return "Forbidden cleanup/debleed/denoise/backbone action labels are preserved as forbidden policy actions."
    if mapping["stem_scope"] == "minus":
        return "Minus-scoped policy labels route to minus taxonomy and never to vocal-specific failure."
    return "Equivalent clustered/unstable proxy labels map to policy_classification_problem with no automatic policy application."


def proposal_from_row(row: dict[str, Any]) -> dict[str, Any]:
    signals = normalized_signals(row)
    change_type = choose_change_type(row, signals)
    if change_type not in PROPOSED_POLICY_CHANGE_TYPES:
        raise ValueError(f"invalid_proposed_policy_change_type:{change_type}")
    stem_type = str(row.get("stem_type") or "unknown")
    split = str(row.get("split") or "")
    case_id = str(row.get("case_id") or "")
    mapping = {
        "source_failure_labels": sorted(as_list(row.get("failure_labels"))),
        "source_dominant_failure_reason": str(row.get("dominant_failure_reason") or ""),
        "target_root_cause_bucket": target_bucket(row, signals),
        "target_failure_class": "policy_classification_problem" if stem_type != "minus" else "minus_specific_failure",
        "stem_scope": stem_type,
        "protected_holdout": split == "holdout",
        "must_not_enter_training_or_tuning": split == "holdout",
    }
    return {
        "case_id": case_id,
        "track_hash": str(row.get("track_hash") or ""),
        "stem_type": stem_type,
        "split": split,
        "dominant_failure_reason": str(row.get("dominant_failure_reason") or ""),
        "failure_labels": as_list(row.get("failure_labels")),
        "current_root_cause_bucket": str(row.get("root_cause_bucket") or ""),
        "current_safe_next_action": str(row.get("safe_next_action") or ""),
        "suspected_policy_gap": suspected_gap(row, change_type, signals),
        "proposed_taxonomy_mapping": mapping,
        "proposed_policy_change_type": change_type,
        "risk_level": risk_level(row, change_type),
        "requires_human_approval": True,
        "can_apply_automatically": False,
        "proposed_test_name": test_name(case_id, change_type),
        "expected_behavior_after_fix": expected_behavior(row, change_type, mapping),
    }


def count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(Counter(str(row.get(field) or "") for row in rows).most_common())


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "policy_taxonomy_fix_summary.v1",
        "total_proposals": len(rows),
        "proposals_by_stem_type": count_by(rows, "stem_type"),
        "proposals_by_policy_change_type": count_by(rows, "proposed_policy_change_type"),
        "proposals_by_risk_level": count_by(rows, "risk_level"),
        "holdout_protected_proposals": sum(
            1 for row in rows if row["proposed_taxonomy_mapping"]["protected_holdout"]
        ),
        "high_risk_proposals": sum(1 for row in rows if row["risk_level"] == "high"),
        "can_apply_automatically_count": sum(1 for row in rows if row["can_apply_automatically"]),
        "requires_human_approval_count": sum(1 for row in rows if row["requires_human_approval"]),
        "safety": dict(SAFETY_FLAGS),
    }


def build_manifest(
    *,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    root_cause_plan_jsonl: Path,
    root_cause_manifest_json: Path,
) -> dict[str, Any]:
    return {
        "schema": "policy_taxonomy_fix_manifest.v1",
        "total_proposals": len(rows),
        "proposals_by_stem_type": summary["proposals_by_stem_type"],
        "proposals_by_policy_change_type": summary["proposals_by_policy_change_type"],
        "proposals_by_risk_level": summary["proposals_by_risk_level"],
        "holdout_protected_proposals": summary["holdout_protected_proposals"],
        "high_risk_proposals": summary["high_risk_proposals"],
        "can_apply_automatically_count": summary["can_apply_automatically_count"],
        "requires_human_approval_count": summary["requires_human_approval_count"],
        "input_files": {
            "root_cause_plan_jsonl": str(root_cause_plan_jsonl),
            "root_cause_manifest_json": str(root_cause_manifest_json),
        },
        "input_sha256": {
            "root_cause_plan_jsonl": sha256_file(root_cause_plan_jsonl),
            "root_cause_manifest_json": sha256_file(root_cause_manifest_json),
        },
        "safety": dict(SAFETY_FLAGS),
    }


def build_policy_taxonomy_proposals(
    *,
    root_cause_plan_jsonl: Path,
    root_cause_manifest_json: Path,
) -> dict[str, Any]:
    _manifest = read_json(root_cause_manifest_json)
    rows = [proposal_from_row(row) for row in iter_jsonl(root_cause_plan_jsonl) if should_include(row)]
    rows.sort(key=lambda row: (row["case_id"], row["proposed_policy_change_type"]))
    summary = build_summary(rows)
    manifest = build_manifest(
        rows=rows,
        summary=summary,
        root_cause_plan_jsonl=root_cause_plan_jsonl,
        root_cause_manifest_json=root_cause_manifest_json,
    )
    return {"rows": rows, "summary": summary, "manifest": manifest}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Policy Taxonomy Fix Proposal Summary",
        "",
        f"- `total_proposals`: {summary['total_proposals']}",
        f"- `holdout_protected_proposals`: {summary['holdout_protected_proposals']}",
        f"- `high_risk_proposals`: {summary['high_risk_proposals']}",
        f"- `can_apply_automatically_count`: {summary['can_apply_automatically_count']}",
        f"- `requires_human_approval_count`: {summary['requires_human_approval_count']}",
        "",
        "## Proposals By Stem Type",
        "",
    ]
    for key, value in summary["proposals_by_stem_type"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Proposals By Policy Change Type", ""])
    for key, value in summary["proposals_by_policy_change_type"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Proposals By Risk Level", ""])
    for key, value in summary["proposals_by_risk_level"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Safety", ""])
    for key, value in summary["safety"].items():
        lines.append(f"- `{key}`: {str(value).lower()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_proposed_tests_patch(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Proposed Tests Patch",
        "",
        "This is an auditable test proposal only. No production policy code is changed in Phase 9A.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## `{row['proposed_test_name']}`",
                "",
                f"- case_id: `{row['case_id']}`",
                f"- change_type: `{row['proposed_policy_change_type']}`",
                f"- expected: {row['expected_behavior_after_fix']}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "policy_taxonomy_fix_proposal.jsonl", result["rows"])
    write_summary_md(output_dir / "policy_taxonomy_fix_summary.md", result["summary"])
    (output_dir / "policy_taxonomy_fix_manifest.json").write_text(
        json.dumps(result["manifest"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_proposed_tests_patch(output_dir / "proposed_tests_patch.md", result["rows"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Create auditable policy/taxonomy fix proposals.")
    parser.add_argument("--root-cause-plan-jsonl", type=Path, required=True)
    parser.add_argument("--root-cause-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_policy_taxonomy_proposals(
        root_cause_plan_jsonl=args.root_cause_plan_jsonl,
        root_cause_manifest_json=args.root_cause_manifest_json,
    )
    write_outputs(result, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
