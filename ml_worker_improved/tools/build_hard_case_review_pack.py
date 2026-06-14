from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REVIEW_DECISIONS = (
    "true_failure",
    "proxy_false_positive",
    "acceptable_artifact",
    "needs_finetune",
    "needs_policy_fix",
    "needs_reference_fix",
    "needs_manual_ab",
)
ALLOWED_NEXT_STEPS = (
    "manual_ab_only",
    "taxonomy_relabel",
    "policy_fix_candidate",
    "finetune_candidate",
    "holdout_only",
)
REVIEW_ROW_FIELDS = (
    "benchmark_case_id",
    "track_hash",
    "stem_type",
    "dominant_failure_reason",
    "failure_labels",
    "severity_score",
    "recommended_review_type",
    "review_priority",
    "expected_audio_problem",
    "human_rating",
    "human_notes",
    "reviewer_decision",
    "allowed_next_step",
)

P0_SIGNALS = {
    "no_accepted_candidate",
    "candidate_space_exhausted",
    "selected_candidate_still_has_severe_issue",
}
P1_SIGNALS = {
    "selected_candidate_has_unstable_vocal_consistency",
    "selected_candidate_still_has_clustered_issues",
    "unknown_blocked_reason",
}
P2_SIGNALS = {
    "thin_or_overcleaned",
    "robotic_or_noise_artifact",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
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
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def row_signals(row: dict[str, Any]) -> set[str]:
    signals = {
        str(row.get("dominant_failure_reason") or "").strip(),
        str(row.get("recommended_review_type") or "").strip(),
        str(row.get("stem_type") or "").strip(),
    }
    signals.update(as_list(row.get("failure_labels")))
    signals.update(as_list(row.get("failed_checks")))
    return {signal for signal in signals if signal}


def load_golden_track_hashes(golden_jsonl: Path) -> set[str]:
    hashes: set[str] = set()
    for row in iter_jsonl(golden_jsonl):
        track_hash = str(row.get("track_hash") or "").strip()
        if track_hash:
            hashes.add(track_hash)
    return hashes


def review_priority(row: dict[str, Any]) -> str:
    signals = row_signals(row)
    if signals & P0_SIGNALS:
        return "P0"
    if signals & P1_SIGNALS:
        return "P1"
    return "P2"


def expected_audio_problem(row: dict[str, Any]) -> str:
    signals = row_signals(row)
    reason = str(row.get("dominant_failure_reason") or "")
    if signals & {"no_accepted_candidate", "candidate_space_exhausted"}:
        return "candidate space failed to produce a listenable accepted result"
    if "selected_candidate_still_has_severe_issue" in signals:
        return "selected candidate still has severe audible issue windows"
    if "selected_candidate_has_unstable_vocal_consistency" in signals:
        return "vocal consistency is unstable across reviewed windows"
    if "selected_candidate_still_has_clustered_issues" in signals:
        return "audible defects are clustered in multiple windows"
    if "unknown_blocked_reason" in signals:
        return "failure taxonomy is unknown and needs manual labeling"
    if "thin_or_overcleaned" in signals:
        return "stem may be thin, overcleaned, or missing body/presence"
    if "robotic_or_noise_artifact" in signals:
        return "stem may contain robotic noise or musical artifacts"
    if row.get("stem_type") == "minus":
        return "minus stem may contain vocal residue or instrumental damage"
    return f"manual review required for {reason or 'unclassified hard case'}"


def allowed_next_step(row: dict[str, Any]) -> str:
    if row.get("split") == "holdout":
        return "holdout_only"
    signals = row_signals(row)
    if "unknown_blocked_reason" in signals:
        return "taxonomy_relabel"
    if signals & {"no_accepted_candidate", "candidate_space_exhausted"}:
        return "manual_ab_only"
    if "selected_candidate_still_has_severe_issue" in signals:
        return "finetune_candidate"
    if signals & {
        "selected_candidate_has_unstable_vocal_consistency",
        "selected_candidate_still_has_clustered_issues",
        "thin_or_overcleaned",
        "robotic_or_noise_artifact",
    }:
        return "policy_fix_candidate"
    if row.get("stem_type") == "minus":
        return "policy_fix_candidate"
    return "manual_ab_only"


def review_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "benchmark_case_id": str(row.get("benchmark_case_id") or ""),
        "track_hash": str(row.get("track_hash") or ""),
        "stem_type": str(row.get("stem_type") or "unknown"),
        "dominant_failure_reason": str(row.get("dominant_failure_reason") or "unknown_blocked_reason"),
        "failure_labels": as_list(row.get("failure_labels")),
        "severity_score": row.get("severity_score"),
        "recommended_review_type": str(row.get("recommended_review_type") or "manual_ab_hard_case_review"),
        "review_priority": review_priority(row),
        "expected_audio_problem": expected_audio_problem(row),
        "human_rating": None,
        "human_notes": None,
        "reviewer_decision": None,
        "allowed_next_step": allowed_next_step(row),
    }
    return {field: payload[field] for field in REVIEW_ROW_FIELDS}


def count_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key) or "unknown") for row in rows).most_common())


def build_summary(
    rows: list[dict[str, Any]],
    *,
    skipped_golden_overlap: int,
    benchmark_manifest: dict[str, Any],
    dashboard: dict[str, Any],
    input_files: dict[str, str],
    input_sha256: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema": "hard_case_review_pack_summary.v1",
        "row_count": len(rows),
        "priority_counts": count_by(rows, "review_priority"),
        "allowed_next_step_counts": count_by(rows, "allowed_next_step"),
        "rows_by_stem_type": count_by(rows, "stem_type"),
        "unknown_reason_review_count": sum(
            1
            for row in rows
            if row["dominant_failure_reason"] == "unknown_blocked_reason"
            or "unknown_blocked_reason" in row["failure_labels"]
        ),
        "finetune_candidate_count": sum(1 for row in rows if row["allowed_next_step"] == "finetune_candidate"),
        "policy_fix_candidate_count": sum(1 for row in rows if row["allowed_next_step"] == "policy_fix_candidate"),
        "holdout_only_count": sum(1 for row in rows if row["allowed_next_step"] == "holdout_only"),
        "skipped_golden_overlap": skipped_golden_overlap,
        "review_decision_values": list(REVIEW_DECISIONS),
        "allowed_next_step_values": list(ALLOWED_NEXT_STEPS),
        "golden_usage": {
            "usage": "protected_holdout_exclusion_only",
            "training_allowed": False,
            "threshold_tuning_allowed": False,
        },
        "holdout_rule": "holdout rows use allowed_next_step=holdout_only and are not route-tuning rows",
        "audio_copied": False,
        "input_files": input_files,
        "input_sha256": input_sha256,
        "benchmark_row_count": benchmark_manifest.get("row_count"),
        "dashboard_summary": dashboard.get("summary", {}),
    }


def build_review_pack(
    *,
    benchmark_jsonl: Path,
    benchmark_manifest: Path,
    benchmark_summary_md: Path,
    golden_jsonl: Path,
    dashboard_json: Path,
) -> dict[str, Any]:
    benchmark_manifest_payload = read_json(benchmark_manifest)
    dashboard_payload = read_json(dashboard_json)
    golden_track_hashes = load_golden_track_hashes(golden_jsonl)

    rows: list[dict[str, Any]] = []
    skipped_golden_overlap = 0
    for row in iter_jsonl(benchmark_jsonl):
        if str(row.get("track_hash") or "").strip() in golden_track_hashes:
            skipped_golden_overlap += 1
            continue
        rows.append(review_row(row))

    input_files = {
        "benchmark_jsonl": str(benchmark_jsonl),
        "benchmark_manifest": str(benchmark_manifest),
        "benchmark_summary_md": str(benchmark_summary_md),
        "golden_jsonl": str(golden_jsonl),
        "control_plane_dashboard_json": str(dashboard_json),
    }
    input_sha256 = {
        "benchmark_jsonl": sha256_file(benchmark_jsonl),
        "benchmark_manifest": sha256_file(benchmark_manifest),
        "benchmark_summary_md": sha256_file(benchmark_summary_md),
        "golden_jsonl": sha256_file(golden_jsonl),
        "control_plane_dashboard_json": sha256_file(dashboard_json),
    }
    summary = build_summary(
        rows,
        skipped_golden_overlap=skipped_golden_overlap,
        benchmark_manifest=benchmark_manifest_payload,
        dashboard=dashboard_payload,
        input_files=input_files,
        input_sha256=input_sha256,
    )
    return {"rows": rows, "summary": summary}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def csv_value(value: Any) -> Any:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_ROW_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in REVIEW_ROW_FIELDS})


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Hard Case Review Pack Summary",
        "",
        f"Review rows: `{summary['row_count']}`",
        "",
        "## Priority Counts",
        "",
    ]
    for key, value in summary["priority_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Allowed Next Steps", ""])
    for key, value in summary["allowed_next_step_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Key Counts",
            "",
            f"- `unknown_reason_review_count`: {summary['unknown_reason_review_count']}",
            f"- `finetune_candidate_count`: {summary['finetune_candidate_count']}",
            f"- `policy_fix_candidate_count`: {summary['policy_fix_candidate_count']}",
            f"- `holdout_only_count`: {summary['holdout_only_count']}",
            f"- `skipped_golden_overlap`: {summary['skipped_golden_overlap']}",
            "",
            "## Golden And Holdout Rules",
            "",
            "- Golden corpus is used only for protected holdout exclusion.",
            "- `training_allowed=false`.",
            "- `threshold_tuning_allowed=false`.",
            f"- {summary['holdout_rule']}.",
            "- No audio is copied or exported.",
            "",
            "## Reviewer Decisions",
            "",
        ]
    )
    for decision in summary["review_decision_values"]:
        lines.append(f"- `{decision}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(pack: dict[str, Any], output_jsonl: Path, output_csv: Path, output_summary_md: Path) -> None:
    write_jsonl(output_jsonl, pack["rows"])
    write_csv(output_csv, pack["rows"])
    write_markdown(output_summary_md, pack["summary"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Build manual/perceptual review pack from hard-case benchmark.")
    parser.add_argument("--benchmark-jsonl", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--benchmark-summary-md", type=Path, required=True)
    parser.add_argument("--golden-jsonl", type=Path, required=True)
    parser.add_argument("--dashboard-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary-md", type=Path, required=True)
    args = parser.parse_args()

    pack = build_review_pack(
        benchmark_jsonl=args.benchmark_jsonl,
        benchmark_manifest=args.benchmark_manifest,
        benchmark_summary_md=args.benchmark_summary_md,
        golden_jsonl=args.golden_jsonl,
        dashboard_json=args.dashboard_json,
    )
    write_outputs(pack, args.output_jsonl, args.output_csv, args.output_summary_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
