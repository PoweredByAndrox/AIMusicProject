from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

from tools.artifact_status_contract import stable_track_split


DEFAULT_TARGET_SIZE = 120
DEFAULT_MIN_SIZE = 50
DEFAULT_MAX_SIZE = 200
DEFAULT_MIN_VOCAL = 20
DEFAULT_MIN_MINUS = 10
DEFAULT_MIN_UNKNOWN = 10

UNKNOWN_REASONS = {"", "none", "unknown", "unknown_blocked_reason"}
HIGH_SEVERITY_SIGNALS = {
    "candidate_space_exhausted",
    "no_accepted_candidate",
    "selected_candidate_still_has_severe_issue",
}
ELEVATED_SEVERITY_REASONS = {
    "selected_candidate_has_unstable_vocal_consistency",
    "selected_candidate_still_has_clustered_issues",
    "selected_candidate_has_too_many_issue_windows",
}
DIAGNOSTIC_STATUSES = {"diagnostic_only", "blocked_for_listening"}
LISTENABLE_PRESENTATION_STATUSES = {"accepted_for_listen_pack"}
DELIVERABLE_RELEASE_STATUSES = {"release_ready"}
REQUIRED_ROW_KEYS = (
    "benchmark_case_id",
    "source_hard_case_id",
    "track_hash",
    "split",
    "stem_type",
    "dominant_failure_reason",
    "failure_labels",
    "failed_checks",
    "severity_score",
    "window_start_sec",
    "window_end_sec",
    "source_manifest",
    "recommended_review_type",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(*parts: Any, length: int = 24) -> str:
    text = "|".join(str(part or "").strip().lower().replace("\\", "/") for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


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


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def row_signals(row: dict[str, Any]) -> set[str]:
    signals: set[str] = set()
    for key in (
        "delivery_status",
        "presentation_status",
        "release_status",
        "dominant_failure_reason",
    ):
        value = str(row.get(key) or "").strip()
        if value:
            signals.add(value)
    signals.update(as_list(row.get("failure_labels")))
    signals.update(as_list(row.get("failed_checks")))
    return signals


def is_unknown_reason(row: dict[str, Any]) -> bool:
    dominant = str(row.get("dominant_failure_reason") or "").strip().lower()
    if dominant in UNKNOWN_REASONS:
        return True
    labels = {label.lower() for label in as_list(row.get("failure_labels"))}
    return bool(labels & UNKNOWN_REASONS)


def is_unsafe_diagnostic(row: dict[str, Any]) -> bool:
    if as_bool(row.get("delivery_allowed")):
        return False
    signals = row_signals(row)
    if signals & DIAGNOSTIC_STATUSES:
        return True
    has_audio_path = bool(str(row.get("audio_path") or "").strip())
    return (
        has_audio_path
        and row.get("presentation_status") not in LISTENABLE_PRESENTATION_STATUSES
        and row.get("release_status") not in DELIVERABLE_RELEASE_STATUSES
    )


def severity_score(row: dict[str, Any]) -> int:
    signals = row_signals(row)
    if signals & HIGH_SEVERITY_SIGNALS:
        return 95
    if is_unsafe_diagnostic(row):
        return 90
    if is_unknown_reason(row):
        return 80
    if str(row.get("dominant_failure_reason") or "") in ELEVATED_SEVERITY_REASONS:
        return 78
    return 65


def recommended_review_type(row: dict[str, Any]) -> str:
    signals = row_signals(row)
    if signals & {"candidate_space_exhausted", "no_accepted_candidate"}:
        return "manual_ab_hard_case_review"
    if is_unknown_reason(row):
        return "taxonomy_label_review"
    if is_unsafe_diagnostic(row):
        return "diagnostic_safety_review"
    if row.get("stem_type") == "minus":
        return "minus_release_review"
    return "vocal_hard_window_review"


def normalized_benchmark_row(source: dict[str, Any], ordinal: int) -> dict[str, Any]:
    track_hash = str(source.get("track_hash") or "").strip()
    if not track_hash:
        raise ValueError(f"hard_case_row_missing_track_hash:{source.get('hard_case_id')}")
    source_id = str(source.get("hard_case_id") or "").strip()
    if not source_id:
        source_id = stable_id(track_hash, source.get("source_manifest"), source.get("window_start_sec"), ordinal)
    failure_labels = as_list(source.get("failure_labels"))
    failed_checks = as_list(source.get("failed_checks"))
    row = {
        "benchmark_case_id": f"bench_{stable_id(source_id, track_hash, source.get('source_manifest'))}",
        "source_hard_case_id": source_id,
        "track_hash": track_hash,
        "split": stable_track_split(track_hash),
        "stem_type": str(source.get("stem_type") or "unknown"),
        "dominant_failure_reason": str(source.get("dominant_failure_reason") or "unknown_blocked_reason"),
        "failure_labels": failure_labels,
        "failed_checks": failed_checks,
        "severity_score": severity_score(source),
        "window_start_sec": source.get("window_start_sec"),
        "window_end_sec": source.get("window_end_sec"),
        "source_manifest": str(source.get("source_manifest") or ""),
        "recommended_review_type": recommended_review_type(source),
    }
    return {key: row[key] for key in REQUIRED_ROW_KEYS}


def sort_key(row: dict[str, Any], salt: str) -> tuple[int, str]:
    return (
        -int(row.get("severity_score") or 0),
        stable_id(salt, row.get("source_hard_case_id"), row.get("track_hash"), row.get("source_manifest"), length=64),
    )


def add_rows(
    selected: list[dict[str, Any]],
    selected_ids: set[str],
    candidates: list[dict[str, Any]],
    quota: int,
    *,
    salt: str,
    max_size: int,
) -> None:
    if quota <= 0 or len(selected) >= max_size:
        return
    added = 0
    for row in sorted(candidates, key=lambda item: sort_key(item, salt)):
        source_id = str(row["source_hard_case_id"])
        if source_id in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(source_id)
        added += 1
        if len(selected) >= max_size or added >= quota:
            return


def rows_matching(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
    return [row for row in rows if predicate(row)]


def load_golden_track_hashes(golden_jsonl: Path) -> set[str]:
    hashes: set[str] = set()
    for row in iter_jsonl(golden_jsonl):
        track_hash = str(row.get("track_hash") or "").strip()
        if track_hash:
            hashes.add(track_hash)
    return hashes


def build_rows(
    hard_case_jsonl: Path,
    golden_jsonl: Path,
    *,
    target_size: int = DEFAULT_TARGET_SIZE,
    min_size: int = DEFAULT_MIN_SIZE,
    max_size: int = DEFAULT_MAX_SIZE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target_size < min_size or target_size > max_size:
        raise ValueError(f"target_size_must_be_between_{min_size}_and_{max_size}")

    golden_track_hashes = load_golden_track_hashes(golden_jsonl)
    normalized: list[dict[str, Any]] = []
    skipped_golden_overlap = 0
    skipped_missing_track_hash = 0
    for ordinal, source in enumerate(iter_jsonl(hard_case_jsonl)):
        if str(source.get("track_hash") or "").strip() in golden_track_hashes:
            skipped_golden_overlap += 1
            continue
        try:
            normalized.append(normalized_benchmark_row(source, ordinal))
        except ValueError:
            skipped_missing_track_hash += 1

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    candidate_space = rows_matching(normalized, lambda row: "candidate_space_exhausted" in row_signals(row))
    no_accepted = rows_matching(normalized, lambda row: "no_accepted_candidate" in row_signals(row))
    unknown = rows_matching(normalized, is_unknown_reason)
    minus = rows_matching(normalized, lambda row: row["stem_type"] == "minus")
    vocal = rows_matching(normalized, lambda row: row["stem_type"] == "vocal")

    add_rows(selected, selected_ids, candidate_space, min(10, len(candidate_space)), salt="candidate_space", max_size=target_size)
    add_rows(selected, selected_ids, no_accepted, min(10, len(no_accepted)), salt="no_accepted", max_size=target_size)
    add_rows(selected, selected_ids, unknown, min(DEFAULT_MIN_UNKNOWN, len(unknown)), salt="unknown", max_size=target_size)
    add_rows(selected, selected_ids, minus, min(DEFAULT_MIN_MINUS, len(minus)), salt="minus", max_size=target_size)
    add_rows(selected, selected_ids, vocal, min(DEFAULT_MIN_VOCAL, len(vocal)), salt="vocal", max_size=target_size)

    by_reason: dict[str, list[dict[str, Any]]] = {}
    for row in normalized:
        by_reason.setdefault(str(row["dominant_failure_reason"]), []).append(row)
    for reason in sorted(by_reason, key=lambda key: (-len(by_reason[key]), key)):
        if len(selected) >= target_size:
            break
        add_rows(selected, selected_ids, by_reason[reason], 2, salt=f"reason:{reason}", max_size=target_size)

    if len(selected) < target_size:
        add_rows(selected, selected_ids, normalized, target_size - len(selected), salt="fill", max_size=target_size)

    selected.sort(key=lambda item: sort_key(item, "final_order"))
    for index, row in enumerate(selected):
        row["benchmark_case_id"] = f"bench_{index:04d}_{stable_id(row['source_hard_case_id'], row['track_hash'])[:12]}"

    metadata = {
        "candidate_rows_after_golden_exclusion": len(normalized),
        "skipped_golden_track_hash_overlap": skipped_golden_overlap,
        "skipped_missing_track_hash": skipped_missing_track_hash,
        "available_candidate_space_exhausted": len(candidate_space),
        "available_no_accepted_candidate": len(no_accepted),
        "available_unknown_reason": len(unknown),
        "available_vocal": len(vocal),
        "available_minus": len(minus),
    }
    return selected, metadata


def count_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key) or "unknown") for row in rows).most_common())


def build_manifest(
    *,
    rows: list[dict[str, Any]],
    hard_case_jsonl: Path,
    hard_case_manifest: Path,
    dashboard_json: Path,
    golden_jsonl: Path,
    metadata: dict[str, Any],
    target_size: int,
    min_size: int,
    max_size: int,
) -> dict[str, Any]:
    return {
        "schema": "hard_case_benchmark_manifest.v1",
        "row_count": len(rows),
        "target_size": target_size,
        "min_size": min_size,
        "max_size": max_size,
        "input_files": {
            "hard_case_jsonl": str(hard_case_jsonl),
            "hard_case_manifest": str(hard_case_manifest),
            "control_plane_dashboard_json": str(dashboard_json),
            "golden_jsonl": str(golden_jsonl),
        },
        "input_sha256": {
            "hard_case_jsonl": sha256_file(hard_case_jsonl),
            "hard_case_manifest": sha256_file(hard_case_manifest),
            "control_plane_dashboard_json": sha256_file(dashboard_json),
            "golden_jsonl": sha256_file(golden_jsonl),
        },
        "sampling": {
            "method": "deterministic_stratified_hash_sampling",
            "split_rule": "stable_track_split(track_hash)",
            "audio_copied": False,
            "representative_case_target": "50_to_200_rows",
        },
        "golden_usage": {
            "usage": "protected_holdout_exclusion_only",
            "training_allowed": False,
            "threshold_tuning_allowed": False,
        },
        "coverage": {
            "rows_by_stem_type": count_by(rows, "stem_type"),
            "rows_by_dominant_failure_reason": count_by(rows, "dominant_failure_reason"),
            "split_counts": count_by(rows, "split"),
            "unknown_reason_rows": sum(1 for row in rows if is_unknown_reason(row)),
            "candidate_space_exhausted_rows": sum(
                1 for row in rows if "candidate_space_exhausted" in row_signals(row)
            ),
            "no_accepted_candidate_rows": sum(1 for row in rows if "no_accepted_candidate" in row_signals(row)),
        },
        "source_metadata": metadata,
    }


def build_summary_markdown(manifest: dict[str, Any]) -> str:
    coverage = manifest["coverage"]
    lines = [
        "# Hard Case Benchmark Summary",
        "",
        f"Total benchmark rows: `{manifest['row_count']}`",
        "",
        "## Rows By Stem Type",
        "",
    ]
    for key, value in coverage["rows_by_stem_type"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Rows By Dominant Failure Reason", ""])
    for key, value in coverage["rows_by_dominant_failure_reason"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Split Counts", ""])
    for key, value in coverage["split_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Required Signals",
            "",
            f"- `unknown_reason_rows`: {coverage['unknown_reason_rows']}",
            f"- `candidate_space_exhausted_rows`: {coverage['candidate_space_exhausted_rows']}",
            f"- `no_accepted_candidate_rows`: {coverage['no_accepted_candidate_rows']}",
            "",
            "## Golden Corpus Use",
            "",
            "- Golden rows are used only as protected holdout exclusion metadata.",
            "- `training_allowed=false`.",
            "- `threshold_tuning_allowed=false`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_benchmark(
    *,
    hard_case_jsonl: Path,
    hard_case_manifest: Path,
    dashboard_json: Path,
    golden_jsonl: Path,
    target_size: int = DEFAULT_TARGET_SIZE,
    min_size: int = DEFAULT_MIN_SIZE,
    max_size: int = DEFAULT_MAX_SIZE,
) -> dict[str, Any]:
    hard_manifest_payload = read_json(hard_case_manifest)
    dashboard_payload = read_json(dashboard_json)
    rows, metadata = build_rows(
        hard_case_jsonl,
        golden_jsonl,
        target_size=target_size,
        min_size=min_size,
        max_size=max_size,
    )
    metadata["hard_case_manifest_row_count"] = hard_manifest_payload.get("row_count")
    metadata["dashboard_summary"] = dashboard_payload.get("summary", {})
    manifest = build_manifest(
        rows=rows,
        hard_case_jsonl=hard_case_jsonl,
        hard_case_manifest=hard_case_manifest,
        dashboard_json=dashboard_json,
        golden_jsonl=golden_jsonl,
        metadata=metadata,
        target_size=target_size,
        min_size=min_size,
        max_size=max_size,
    )
    return {
        "rows": rows,
        "manifest": manifest,
        "summary_markdown": build_summary_markdown(manifest),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_outputs(
    benchmark: dict[str, Any],
    output_jsonl: Path,
    output_manifest: Path,
    output_summary_md: Path,
) -> None:
    write_jsonl(output_jsonl, benchmark["rows"])
    write_json(output_manifest, benchmark["manifest"])
    output_summary_md.parent.mkdir(parents=True, exist_ok=True)
    output_summary_md.write_text(benchmark["summary_markdown"], encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic representative hard-case benchmark.")
    parser.add_argument("--hard-case-jsonl", type=Path, required=True)
    parser.add_argument("--hard-case-manifest", type=Path, required=True)
    parser.add_argument("--dashboard-json", type=Path, required=True)
    parser.add_argument("--golden-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-summary-md", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--min-size", type=int, default=DEFAULT_MIN_SIZE)
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE)
    args = parser.parse_args()

    benchmark = build_benchmark(
        hard_case_jsonl=args.hard_case_jsonl,
        hard_case_manifest=args.hard_case_manifest,
        dashboard_json=args.dashboard_json,
        golden_jsonl=args.golden_jsonl,
        target_size=args.target_size,
        min_size=args.min_size,
        max_size=args.max_size,
    )
    write_outputs(benchmark, args.output_jsonl, args.output_manifest, args.output_summary_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
