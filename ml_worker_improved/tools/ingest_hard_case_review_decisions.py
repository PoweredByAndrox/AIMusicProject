from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ALLOWED_REVIEWER_DECISIONS = {
    "true_failure",
    "proxy_false_positive",
    "acceptable_artifact",
    "needs_finetune",
    "needs_policy_fix",
    "needs_reference_fix",
    "needs_manual_ab",
}
QUEUE_NAMES = (
    "policy_fix_queue",
    "finetune_candidate_verified",
    "reference_fix_queue",
    "proxy_false_positive_queue",
    "manual_ab_followup",
)
PACK_VALIDATION_FIELDS = (
    "benchmark_case_id",
    "track_hash",
    "stem_type",
    "dominant_failure_reason",
    "severity_score",
    "recommended_review_type",
    "review_priority",
    "expected_audio_problem",
    "allowed_next_step",
)


class IngestValidationError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise IngestValidationError(f"jsonl_row_must_be_object:{path}:{line_number}")
            yield payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_list_from_sheet(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(";") if item.strip()]


def normalized_scalar(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_pack(review_pack_jsonl: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(review_pack_jsonl):
        case_id = normalized_scalar(row.get("benchmark_case_id"))
        if not case_id:
            raise IngestValidationError("review_pack_row_missing_benchmark_case_id")
        if case_id in rows:
            raise IngestValidationError(f"duplicate_benchmark_case_id:{case_id}")
        rows[case_id] = row
    return rows


def load_golden_track_hashes(golden_jsonl: Path) -> set[str]:
    hashes: set[str] = set()
    for row in iter_jsonl(golden_jsonl):
        track_hash = normalized_scalar(row.get("track_hash"))
        if track_hash:
            hashes.add(track_hash)
    return hashes


def validate_sheet_row_against_pack(sheet_row: dict[str, str], pack_row: dict[str, Any]) -> None:
    case_id = normalized_scalar(pack_row.get("benchmark_case_id"))
    for field in PACK_VALIDATION_FIELDS:
        sheet_value = normalized_scalar(sheet_row.get(field))
        pack_value = normalized_scalar(pack_row.get(field))
        if sheet_value and sheet_value != pack_value:
            raise IngestValidationError(f"review_sheet_field_mismatch:{case_id}:{field}")


def evidence_row(sheet_row: dict[str, str], pack_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_case_id": normalized_scalar(pack_row.get("benchmark_case_id")),
        "track_hash": normalized_scalar(pack_row.get("track_hash")),
        "stem_type": normalized_scalar(pack_row.get("stem_type")),
        "dominant_failure_reason": normalized_scalar(pack_row.get("dominant_failure_reason")),
        "failure_labels": list(pack_row.get("failure_labels") or as_list_from_sheet(sheet_row.get("failure_labels"))),
        "severity_score": pack_row.get("severity_score"),
        "recommended_review_type": normalized_scalar(pack_row.get("recommended_review_type")),
        "review_priority": normalized_scalar(pack_row.get("review_priority")),
        "expected_audio_problem": normalized_scalar(pack_row.get("expected_audio_problem")),
        "allowed_next_step": normalized_scalar(pack_row.get("allowed_next_step")),
        "human_rating": normalized_scalar(sheet_row.get("human_rating")),
        "human_notes": normalized_scalar(sheet_row.get("human_notes")),
        "reviewer_decision": normalized_scalar(sheet_row.get("reviewer_decision")),
    }


def route_decision(row: dict[str, Any]) -> str | None:
    decision = row["reviewer_decision"]
    next_step = row["allowed_next_step"]
    if next_step == "holdout_only" and decision in {"needs_finetune"}:
        raise IngestValidationError(f"holdout_cannot_enter_finetune_queue:{row['benchmark_case_id']}")
    if decision == "needs_finetune":
        return "finetune_candidate_verified"
    if decision == "needs_policy_fix":
        return "policy_fix_queue"
    if decision == "needs_reference_fix":
        return "reference_fix_queue"
    if decision == "proxy_false_positive":
        return "proxy_false_positive_queue"
    if decision == "needs_manual_ab":
        return "manual_ab_followup"
    if decision == "true_failure":
        if next_step == "finetune_candidate":
            return "finetune_candidate_verified"
        return "manual_ab_followup"
    return None


def ingest_review_decisions(
    *,
    review_pack_jsonl: Path,
    review_sheet_csv: Path,
    review_summary_md: Path,
    golden_jsonl: Path,
) -> dict[str, Any]:
    pack_rows = load_pack(review_pack_jsonl)
    golden_hashes = load_golden_track_hashes(golden_jsonl)
    queues: dict[str, list[dict[str, Any]]] = {name: [] for name in QUEUE_NAMES}
    validated: list[dict[str, Any]] = []
    seen_sheet_ids: set[str] = set()

    for sheet_row in read_csv(review_sheet_csv):
        case_id = normalized_scalar(sheet_row.get("benchmark_case_id"))
        if not case_id:
            continue
        if case_id not in pack_rows:
            raise IngestValidationError(f"unknown_benchmark_case_id:{case_id}")
        seen_sheet_ids.add(case_id)
        pack_row = pack_rows[case_id]
        validate_sheet_row_against_pack(sheet_row, pack_row)
        decision = normalized_scalar(sheet_row.get("reviewer_decision"))
        if not decision:
            continue
        if decision not in ALLOWED_REVIEWER_DECISIONS:
            raise IngestValidationError(f"invalid_reviewer_decision:{case_id}:{decision}")
        row = evidence_row(sheet_row, pack_row)
        if row["track_hash"] in golden_hashes and row["allowed_next_step"] in {"finetune_candidate", "policy_fix_candidate"}:
            raise IngestValidationError(f"golden_track_hash_cannot_enter_training_or_policy_queue:{case_id}")
        queue_name = route_decision(row)
        validated.append(row)
        if queue_name:
            queues[queue_name].append(row)

    unreviewed_rows = len(pack_rows) - len(validated)
    decision_counts = dict(Counter(row["reviewer_decision"] for row in validated).most_common())
    summary = {
        "schema": "hard_case_review_ingest_summary.v1",
        "reviewed_rows": len(validated),
        "unreviewed_rows": unreviewed_rows,
        "sheet_rows_seen": len(seen_sheet_ids),
        "pack_rows": len(pack_rows),
        "rows_by_reviewer_decision": decision_counts,
        "verified_finetune_candidates": len(queues["finetune_candidate_verified"]),
        "policy_fixes": len(queues["policy_fix_queue"]),
        "reference_fixes": len(queues["reference_fix_queue"]),
        "proxy_false_positives": len(queues["proxy_false_positive_queue"]),
        "manual_ab_followups": len(queues["manual_ab_followup"]),
        "holdout_protected_count": sum(1 for row in validated if row["allowed_next_step"] == "holdout_only"),
        "audio_copied": False,
        "training_started": False,
        "policy_changed": False,
        "input_files": {
            "review_pack_jsonl": str(review_pack_jsonl),
            "review_sheet_csv": str(review_sheet_csv),
            "review_summary_md": str(review_summary_md),
            "golden_jsonl": str(golden_jsonl),
        },
        "input_sha256": {
            "review_pack_jsonl": sha256_file(review_pack_jsonl),
            "review_sheet_csv": sha256_file(review_sheet_csv),
            "review_summary_md": sha256_file(review_summary_md),
            "golden_jsonl": sha256_file(golden_jsonl),
        },
    }
    return {"validated": validated, "queues": queues, "summary": summary}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Review Decision Ingest Summary",
        "",
        f"- `reviewed_rows`: {summary['reviewed_rows']}",
        f"- `unreviewed_rows`: {summary['unreviewed_rows']}",
        "",
        "## Rows By Reviewer Decision",
        "",
    ]
    for key, value in summary["rows_by_reviewer_decision"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Queues",
            "",
            f"- `verified_finetune_candidates`: {summary['verified_finetune_candidates']}",
            f"- `policy_fixes`: {summary['policy_fixes']}",
            f"- `reference_fixes`: {summary['reference_fixes']}",
            f"- `proxy_false_positives`: {summary['proxy_false_positives']}",
            f"- `manual_ab_followups`: {summary['manual_ab_followups']}",
            f"- `holdout_protected_count`: {summary['holdout_protected_count']}",
            "",
            "## Safety",
            "",
            f"- `audio_copied`: {str(summary['audio_copied']).lower()}",
            f"- `training_started`: {str(summary['training_started']).lower()}",
            f"- `policy_changed`: {str(summary['policy_changed']).lower()}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "validated_review_decisions.jsonl", result["validated"])
    write_jsonl(output_dir / "policy_fix_queue.jsonl", result["queues"]["policy_fix_queue"])
    write_jsonl(output_dir / "finetune_candidate_verified.jsonl", result["queues"]["finetune_candidate_verified"])
    write_jsonl(output_dir / "reference_fix_queue.jsonl", result["queues"]["reference_fix_queue"])
    write_jsonl(output_dir / "proxy_false_positive_queue.jsonl", result["queues"]["proxy_false_positive_queue"])
    write_jsonl(output_dir / "manual_ab_followup_queue.jsonl", result["queues"]["manual_ab_followup"])
    write_summary_md(output_dir / "review_ingest_summary.md", result["summary"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest filled hard-case review sheet into action queues.")
    parser.add_argument("--review-pack-jsonl", type=Path, required=True)
    parser.add_argument("--review-sheet-csv", type=Path, required=True)
    parser.add_argument("--review-summary-md", type=Path, required=True)
    parser.add_argument("--golden-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    result = ingest_review_decisions(
        review_pack_jsonl=args.review_pack_jsonl,
        review_sheet_csv=args.review_sheet_csv,
        review_summary_md=args.review_summary_md,
        golden_jsonl=args.golden_jsonl,
    )
    write_outputs(result, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
