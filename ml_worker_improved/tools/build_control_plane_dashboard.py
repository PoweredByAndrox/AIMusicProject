from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


UNKNOWN_REASONS = {"", "none", "unknown", "unknown_blocked_reason"}
DIAGNOSTIC_STATUSES = {"diagnostic_only", "blocked_for_listening"}
LISTENABLE_PRESENTATION_STATUSES = {"accepted_for_listen_pack"}
DELIVERABLE_RELEASE_STATUSES = {"release_ready"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"jsonl_row_must_be_object:{path}:{line_number}")
            rows.append(payload)
    return rows


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def row_strings(row: dict[str, Any]) -> set[str]:
    values: set[str] = set()

    def add(value: Any) -> None:
        if isinstance(value, str):
            text = value.strip()
            if text:
                values.add(text)
        elif isinstance(value, list):
            for item in value:
                add(item)

    for key in (
        "delivery_status",
        "presentation_status",
        "release_status",
        "dominant_failure_reason",
        "failure_labels",
    ):
        add(row.get(key))
    return values


def has_signal(row: dict[str, Any], signal: str) -> bool:
    return signal in row_strings(row)


def has_unknown_reason(row: dict[str, Any]) -> bool:
    dominant = str(row.get("dominant_failure_reason") or "").strip()
    if dominant.lower() in UNKNOWN_REASONS:
        return True
    labels = {str(item).strip().lower() for item in row.get("failure_labels") or []}
    return bool(labels & UNKNOWN_REASONS)


def is_unsafe_diagnostic(row: dict[str, Any]) -> bool:
    if as_bool(row.get("delivery_allowed")):
        return False
    statuses = row_strings(row)
    if statuses & DIAGNOSTIC_STATUSES:
        return True
    has_audio_path = bool(str(row.get("audio_path") or "").strip())
    if not has_audio_path:
        return False
    return (
        row.get("presentation_status") not in LISTENABLE_PRESENTATION_STATUSES
        and row.get("release_status") not in DELIVERABLE_RELEASE_STATUSES
    )


def count_rows(rows: Iterable[dict[str, Any]], predicate) -> int:
    return sum(1 for row in rows if predicate(row))


def build_dashboard(golden_jsonl: Path, hard_case_jsonl: Path) -> dict[str, Any]:
    golden_rows = read_jsonl(golden_jsonl)
    hard_case_rows = read_jsonl(hard_case_jsonl)
    stem_counts = Counter(str(row.get("stem_type") or "unknown") for row in hard_case_rows)
    reason_counts = Counter(
        str(row.get("dominant_failure_reason") or "unknown_blocked_reason") for row in hard_case_rows
    )

    summary = {
        "golden_rows": len(golden_rows),
        "hard_case_rows": len(hard_case_rows),
        "blocked_vocal": stem_counts.get("vocal", 0),
        "blocked_minus": stem_counts.get("minus", 0),
        "unknown_reason": count_rows(hard_case_rows, has_unknown_reason),
        "unsafe_diagnostic": count_rows(hard_case_rows, is_unsafe_diagnostic),
        "no_accepted_candidate": count_rows(hard_case_rows, lambda row: has_signal(row, "no_accepted_candidate")),
        "candidate_space_exhausted": count_rows(
            hard_case_rows, lambda row: has_signal(row, "candidate_space_exhausted")
        ),
    }
    return {
        "schema": "control_plane_dashboard.v1",
        "inputs": {
            "golden_jsonl": str(golden_jsonl),
            "hard_case_jsonl": str(hard_case_jsonl),
        },
        "summary": summary,
        "hard_case_stem_type_counts": dict(stem_counts.most_common()),
        "hard_case_dominant_failure_reason_counts": dict(reason_counts.most_common()),
        "definitions": {
            "unsafe_diagnostic": (
                "hard-case row with delivery_allowed=false and diagnostic-only status, "
                "or blocked audio_path that is neither listenable nor release-ready"
            ),
            "unknown_reason": "dominant_failure_reason or failure_labels contains unknown_blocked_reason/unknown/none",
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_markdown(path: Path, dashboard: dict[str, Any]) -> None:
    summary = dashboard["summary"]
    lines = [
        "# Control Plane Dashboard",
        "",
        "## Summary",
        "",
        "| metric | count |",
        "|---|---:|",
    ]
    for key in (
        "golden_rows",
        "hard_case_rows",
        "blocked_vocal",
        "blocked_minus",
        "unknown_reason",
        "unsafe_diagnostic",
        "no_accepted_candidate",
        "candidate_space_exhausted",
    ):
        lines.append(f"| `{key}` | {summary[key]} |")

    lines.extend(["", "## Hard-Case Stem Types", ""])
    for key, value in dashboard["hard_case_stem_type_counts"].items():
        lines.append(f"- `{key}`: {value}")

    lines.extend(["", "## Dominant Failure Reasons", ""])
    for key, value in dashboard["hard_case_dominant_failure_reason_counts"].items():
        lines.append(f"- `{key}`: {value}")

    lines.extend(["", "## Definitions", ""])
    for key, value in dashboard["definitions"].items():
        lines.append(f"- `{key}`: {value}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build control-plane dashboard from golden/hard-case corpora.")
    parser.add_argument("--golden-jsonl", type=Path, required=True)
    parser.add_argument("--hard-case-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    dashboard = build_dashboard(args.golden_jsonl, args.hard_case_jsonl)
    write_json(args.output_json, dashboard)
    write_markdown(args.output_md, dashboard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
