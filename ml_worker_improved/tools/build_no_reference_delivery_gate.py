from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.artifact_status_contract import ArtifactContractError, can_promote_to_listen_result, sha256_file
from tools.build_no_reference_listen_preview_manifest import (
    SAFETY_FLAGS,
    is_accepted_optimizer_row,
    listen_row_from_optimizer,
    write_json,
)


MODES = ("vocal", "minus", "both")
MIN_DENSE_WINDOW_COUNT = 5


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def stable_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("track_id") or ""), str(row.get("candidate_id") or ""))


def load_judge_index(path: Path, expected_mode: str) -> dict[tuple[str, str], dict[str, Any]]:
    rows = list(iter_jsonl(path))
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = row_key(row)
        if not all(key):
            continue
        if str(row.get("score_mode") or expected_mode) != expected_mode:
            continue
        indexed[key] = row
    return indexed


def gate_source(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else "",
    }


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def blocker_keys_for_mode(mode: str) -> tuple[str, ...]:
    if mode == "vocal":
        return ("vocal_blockers",)
    if mode == "minus":
        return ("minus_blockers",)
    return ("pair_blockers",)


def dense_gate_view(mode: str, row: dict[str, Any] | None, *, min_window_count: int) -> tuple[dict[str, Any], list[str]]:
    if row is None:
        return {"decision": "missing_dense_judge_row", "window_count": 0}, ["missing_dense_judge_row"]

    failures: list[str] = []
    decision = str(row.get("decision") or "")
    if decision != "accept_candidate":
        failures.append(f"{mode}_decision_not_accepted")
    if int(row.get("window_count") or 0) < min_window_count:
        failures.append(f"{mode}_insufficient_window_count")
    for key in blocker_keys_for_mode(mode):
        if as_list(row.get(key)):
            failures.append(f"{mode}_{key}_present")
    if mode == "both":
        pair_decision = str(row.get("pair_decision") or row.get("pair_compatibility_decision") or "")
        if pair_decision and pair_decision != "pair_compatible":
            failures.append("both_pair_not_compatible")

    view = {
        "decision": decision,
        "score_mode": str(row.get("score_mode") or mode),
        "window_count": int(row.get("window_count") or 0),
        "window_seconds": float(row.get("window_seconds") or 0.0),
        "window_step_seconds": float(row.get("window_step_seconds") or 0.0),
        "reasons": as_list(row.get("reasons")),
    }
    for key in ("vocal_score", "minus_score", "pair_compatibility_score", "overall_no_reference_score"):
        if key in row:
            view[key] = float(row.get(key) or 0.0)
    return view, failures


def optimizer_skip(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "track_id": str(row.get("track_id") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "final_decision": str(row.get("final_decision") or ""),
        "skip_reason": "not_accepted_optimizer_row",
        "failed_modes": [],
        "failed_checks": ["optimizer_row_not_accepted_for_listen_preview"],
    }


def dense_gate_row(
    row: dict[str, Any],
    *,
    optimizer_result_jsonl: Path,
    indexes: dict[str, dict[tuple[str, str], dict[str, Any]]],
    min_window_count: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not is_accepted_optimizer_row(row):
        return None, optimizer_skip(row)

    key = row_key(row)
    gate: dict[str, Any] = {}
    failed_modes: list[str] = []
    failed_checks: list[str] = []
    for mode in MODES:
        view, failures = dense_gate_view(mode, indexes[mode].get(key), min_window_count=min_window_count)
        gate[mode] = view
        if failures:
            failed_modes.append(mode)
            failed_checks.extend(failures)

    if failed_checks:
        return None, {
            "track_id": str(row.get("track_id") or ""),
            "candidate_id": str(row.get("candidate_id") or ""),
            "final_decision": str(row.get("final_decision") or ""),
            "skip_reason": "dense_delivery_gate_failed",
            "failed_modes": stable_unique(failed_modes),
            "failed_checks": stable_unique(failed_checks),
            "dense_delivery_gate": gate,
        }

    accepted = listen_row_from_optimizer(row, optimizer_result_jsonl)
    accepted["dense_delivery_gate"] = gate
    return accepted, None


def build_delivery_gate_manifest(
    *,
    optimizer_result_jsonl: Path,
    vocal_judge_jsonl: Path,
    minus_judge_jsonl: Path,
    both_judge_jsonl: Path,
    min_window_count: int = 5,
) -> dict[str, Any]:
    if min_window_count < MIN_DENSE_WINDOW_COUNT:
        raise ValueError(f"min_window_count must be >= {MIN_DENSE_WINDOW_COUNT}")
    optimizer_rows = list(iter_jsonl(optimizer_result_jsonl))
    indexes = {
        "vocal": load_judge_index(vocal_judge_jsonl, "vocal"),
        "minus": load_judge_index(minus_judge_jsonl, "minus"),
        "both": load_judge_index(both_judge_jsonl, "both"),
    }
    accepted_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for row in optimizer_rows:
        accepted, skipped = dense_gate_row(
            row,
            optimizer_result_jsonl=optimizer_result_jsonl,
            indexes=indexes,
            min_window_count=min_window_count,
        )
        if accepted is not None:
            accepted_rows.append(accepted)
        if skipped is not None:
            skipped_rows.append(skipped)

    delivery_allowed = bool(accepted_rows) and not skipped_rows
    return {
        "manifest_type": "no_reference_dense_delivery_gate",
        "delivery_status": "accepted" if delivery_allowed else "blocked",
        "delivery_allowed": delivery_allowed,
        "presentation_status": "accepted_for_listen_pack" if delivery_allowed else "blocked_from_listen_pack",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": "" if delivery_allowed else "dense_delivery_gate_failed",
        "stem_type": "both",
        "judge_mode": "no_reference_dense_windowed",
        "source_manifest": str(optimizer_result_jsonl),
        "source_manifest_sha256": sha256_file(optimizer_result_jsonl) if optimizer_result_jsonl.exists() else "",
        "dense_gate_sources": {
            "vocal": gate_source(vocal_judge_jsonl),
            "minus": gate_source(minus_judge_jsonl),
            "both": gate_source(both_judge_jsonl),
        },
        "min_window_count": int(min_window_count),
        "accepted_count": len(accepted_rows),
        "skipped_count": len(skipped_rows),
        "accepted_listen_preview_rows": sorted(accepted_rows, key=lambda item: (item["track_id"], item["candidate_id"])),
        "skipped_rows": sorted(skipped_rows, key=lambda item: (item["track_id"], item["candidate_id"])),
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": dict(SAFETY_FLAGS),
    }


def count_failed_modes(skipped_rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(mode for row in skipped_rows for mode in as_list(row.get("failed_modes")))
    return dict(sorted(counter.items()))


def build_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    accepted_rows = list(manifest.get("accepted_listen_preview_rows") or [])
    skipped_rows = list(manifest.get("skipped_rows") or [])
    return {
        "accepted_count": int(manifest.get("accepted_count") or 0),
        "skipped_count": int(manifest.get("skipped_count") or 0),
        "delivery_allowed": bool(manifest.get("delivery_allowed")),
        "presentation_status": str(manifest.get("presentation_status") or ""),
        "release_status": str(manifest.get("release_status") or ""),
        "can_mark_release_ready": bool(manifest.get("can_mark_release_ready")),
        "audio_copied": bool(manifest.get("safety", {}).get("audio_copied")),
        "audio_exported": bool(manifest.get("safety", {}).get("audio_exported")),
        "accepted_candidate_ids": [str(row.get("candidate_id") or "") for row in accepted_rows],
        "failed_modes": count_failed_modes(skipped_rows),
        "min_window_count": int(manifest.get("min_window_count") or 0),
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# No-Reference Dense Delivery Gate Summary",
        "",
        f"- accepted_count: {summary['accepted_count']}",
        f"- skipped_count: {summary['skipped_count']}",
        f"- delivery_allowed: {str(summary['delivery_allowed']).lower()}",
        f"- presentation_status: {summary['presentation_status']}",
        f"- release_status: {summary['release_status']}",
        f"- can_mark_release_ready: {str(summary['can_mark_release_ready']).lower()}",
        f"- audio_copied: {str(summary['audio_copied']).lower()}",
        f"- audio_exported: {str(summary['audio_exported']).lower()}",
        f"- min_window_count: {summary['min_window_count']}",
        "",
        "## Accepted Candidates",
        "",
    ]
    for candidate_id in summary["accepted_candidate_ids"]:
        lines.append(f"- {candidate_id}")
    lines.extend(["", "## Failed Modes", ""])
    for mode, count in summary["failed_modes"].items():
        lines.append(f"- {mode}: {count}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a no-reference listen-preview manifest gated by dense vocal/minus/both evidence.")
    parser.add_argument("--optimizer-result-jsonl", type=Path, required=True)
    parser.add_argument("--vocal-judge-jsonl", type=Path, required=True)
    parser.add_argument("--minus-judge-jsonl", type=Path, required=True)
    parser.add_argument("--both-judge-jsonl", type=Path, required=True)
    parser.add_argument("--output-manifest-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--min-window-count", type=int, default=5)
    args = parser.parse_args(argv)

    manifest = build_delivery_gate_manifest(
        optimizer_result_jsonl=args.optimizer_result_jsonl,
        vocal_judge_jsonl=args.vocal_judge_jsonl,
        minus_judge_jsonl=args.minus_judge_jsonl,
        both_judge_jsonl=args.both_judge_jsonl,
        min_window_count=args.min_window_count,
    )
    write_json(args.output_manifest_json, manifest)
    summary = build_summary(manifest)
    write_summary_md(args.summary_md, summary)
    try:
        promotable = can_promote_to_listen_result(args.output_manifest_json)
    except ArtifactContractError as exc:
        raise SystemExit("dense_delivery_gate_not_promotable") from exc
    if not promotable:
        raise SystemExit("dense_delivery_gate_not_promotable")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
