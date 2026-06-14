from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "diagnostic_audio_exported": False,
    "audio_copied": False,
    "audio_exported": False,
    "audio_exported_as_result": False,
}

BASE_FORBIDDEN_ACTIONS = [
    "do_not_require_reference_stems",
    "do_not_require_reference_vocal_minus",
    "do_not_require_human_csv",
    "do_not_start_finetune",
    "do_not_train",
    "do_not_add_models",
    "do_not_edit_model_registry",
    "do_not_add_cleanup_debleed_denoise",
    "do_not_relax_gates",
    "do_not_export_diagnostic_audio_as_result",
    "do_not_accept_without_dense_post_reconciliation_judge",
]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_unique(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def short_id(*parts: str) -> str:
    value = "::".join(parts)
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    clean = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
    clean = "_".join(part for part in clean.split("_") if part)
    if len(clean) > 80:
        clean = clean[:80].rstrip("_-.")
    return f"{clean or 'pair'}_{digest}"


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def vocal_path(row: dict[str, Any]) -> str:
    return str(row.get("selected_vocal_path") or row.get("vocal_path") or "")


def minus_path(row: dict[str, Any]) -> str:
    return str(row.get("instrumental_path") or row.get("minus_path") or "")


def skip_vocal_reason(row: dict[str, Any]) -> str | None:
    if str(row.get("final_decision") or "") != "phase_corrected_candidate_selected":
        return "vocal_not_phase_corrected_candidate"
    if not vocal_path(row):
        return "vocal_missing_path"
    if as_bool(row.get("can_publish_as_result")) or as_bool(row.get("can_mark_release_ready")):
        return "vocal_row_unexpectedly_promotable"
    return None


def skip_minus_reason(row: dict[str, Any]) -> str | None:
    if not str(row.get("track_id") or ""):
        return "minus_missing_track_id"
    if not str(row.get("mix_path") or ""):
        return "minus_missing_mix_path"
    if not minus_path(row):
        return "minus_missing_path"
    if row.get("is_complete_pair") is False:
        return "minus_not_complete_pair"
    return None


def skipped_row(row: dict[str, Any], *, skip_reason: str, stem_role: str) -> dict[str, Any]:
    return {
        "track_id": str(row.get("track_id") or ""),
        "candidate_id": str(row.get("candidate_id") or row.get("selected_candidate_id") or ""),
        "stem_role": stem_role,
        "skip_reason": skip_reason,
    }


def build_reconciled_pairs(
    vocal_rows: Sequence[dict[str, Any]],
    minus_rows: Sequence[dict[str, Any]],
    *,
    vocal_source_manifest: str,
    minus_source_manifest: str,
    max_pairs_per_track: int = 0,
) -> dict[str, Any]:
    skipped_rows: list[dict[str, Any]] = []
    selected_vocals: list[dict[str, Any]] = []
    for row in vocal_rows:
        reason = skip_vocal_reason(row)
        if reason:
            skipped_rows.append(skipped_row(row, skip_reason=reason, stem_role="vocal"))
            continue
        selected_vocals.append(row)

    minus_by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in minus_rows:
        reason = skip_minus_reason(row)
        if reason:
            skipped_rows.append(skipped_row(row, skip_reason=reason, stem_role="minus"))
            continue
        minus_by_track[str(row.get("track_id") or "")].append(row)

    rows: list[dict[str, Any]] = []
    deduplicated = 0
    seen_pairs: set[tuple[str, str, str]] = set()
    pair_counts: Counter[str] = Counter()
    ordered_vocals = sorted(
        selected_vocals,
        key=lambda row: (str(row.get("track_id") or ""), str(row.get("selected_candidate_id") or row.get("candidate_id") or ""), vocal_path(row)),
    )
    for vocal_row in ordered_vocals:
        track_id = str(vocal_row.get("track_id") or "")
        for minus_row in sorted(minus_by_track.get(track_id, []), key=lambda row: (str(row.get("candidate_id") or ""), minus_path(row))):
            if max_pairs_per_track > 0 and pair_counts[track_id] >= max_pairs_per_track:
                break
            selected_vocal = vocal_path(vocal_row)
            selected_minus = minus_path(minus_row)
            pair_key = (track_id, selected_vocal, selected_minus)
            if pair_key in seen_pairs:
                deduplicated += 1
                continue
            seen_pairs.add(pair_key)
            pair_counts[track_id] += 1
            vocal_candidate_id = str(vocal_row.get("selected_candidate_id") or vocal_row.get("candidate_id") or "")
            minus_candidate_id = str(minus_row.get("candidate_id") or "")
            candidate_id = short_id("reconciled", track_id, vocal_candidate_id, minus_candidate_id)
            rows.append(
                {
                    "track_id": track_id,
                    "candidate_id": candidate_id,
                    "mix_path": str(vocal_row.get("mix_path") or minus_row.get("mix_path") or ""),
                    "vocal_path": selected_vocal,
                    "instrumental_path": selected_minus,
                    "minus_path": selected_minus,
                    "selected_vocal_path": selected_vocal,
                    "selected_minus_path": selected_minus,
                    "final_decision": "pair_reconciliation_candidate",
                    "pair_compatibility_decision": "",
                    "needs_pair_reconciliation": False,
                    "segment_blend_needed": False,
                    "candidate_family": "reconciled_vocal_existing_minus",
                    "candidate_origin": "no_reference_phase_vocal_existing_minus_reconciliation",
                    "is_complete_pair": True,
                    "source_manifest": vocal_source_manifest,
                    "candidate_manifest": "",
                    "source_vocal_candidate_id": vocal_candidate_id,
                    "source_minus_candidate_id": minus_candidate_id,
                    "requires_dense_post_reconciliation_judge": True,
                    "can_publish_as_result": False,
                    "can_mark_release_ready": False,
                    "audio_copied": False,
                    "audio_exported": False,
                    "forbidden_actions": BASE_FORBIDDEN_ACTIONS,
                    "pair_reconciliation_provenance": {
                        "vocal_source_manifest": vocal_source_manifest,
                        "minus_source_manifest": minus_source_manifest,
                        "vocal_candidate_id": vocal_candidate_id,
                        "minus_candidate_id": minus_candidate_id,
                        "vocal_path": selected_vocal,
                        "minus_path": selected_minus,
                    },
                }
            )

    rows = sorted(rows, key=lambda row: (row["track_id"], row["candidate_id"], row["vocal_path"], row["minus_path"]))
    skipped_rows = sorted(skipped_rows, key=lambda row: (row["stem_role"], row["track_id"], row["candidate_id"], row["skip_reason"]))
    summary = {
        "selected_vocal_rows": len(selected_vocals),
        "eligible_minus_rows": sum(len(items) for items in minus_by_track.values()),
        "reconciled_pair_count": len(rows),
        "skipped_count": len(skipped_rows),
        "deduplicated_pair_count": deduplicated,
        "tracks_with_reconciled_pairs": len({row["track_id"] for row in rows}),
        "max_pairs_per_track": max_pairs_per_track,
        "can_publish_as_result_count": sum(1 for row in rows if row.get("can_publish_as_result") is True),
        "can_mark_release_ready_count": sum(1 for row in rows if row.get("can_mark_release_ready") is True),
        "safety": dict(SAFETY_FLAGS),
    }
    return {"rows": rows, "skipped_rows": skipped_rows, "summary": summary}


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# No-Reference Vocal/Minus Pair Reconciliation Summary",
        "",
        f"- selected_vocal_rows: {summary['selected_vocal_rows']}",
        f"- eligible_minus_rows: {summary['eligible_minus_rows']}",
        f"- reconciled_pair_count: {summary['reconciled_pair_count']}",
        f"- skipped_count: {summary['skipped_count']}",
        f"- deduplicated_pair_count: {summary['deduplicated_pair_count']}",
        f"- tracks_with_reconciled_pairs: {summary['tracks_with_reconciled_pairs']}",
        f"- max_pairs_per_track: {summary['max_pairs_per_track']}",
        f"- can_publish_as_result_count: {summary['can_publish_as_result_count']}",
        f"- can_mark_release_ready_count: {summary['can_mark_release_ready_count']}",
        "",
        "## Safety",
        "",
    ]
    for key, value in summary["safety"].items():
        lines.append(f"- {key}: {str(value).lower()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build no-reference reconciled vocal/minus candidate pairs from existing artifacts.")
    parser.add_argument("--vocal-candidates-jsonl", type=Path, required=True)
    parser.add_argument("--minus-candidates-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--skipped-jsonl", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--max-pairs-per-track", type=int, default=0)
    args = parser.parse_args(argv)

    vocal_rows = list(iter_jsonl(args.vocal_candidates_jsonl))
    minus_rows = list(iter_jsonl(args.minus_candidates_jsonl))
    result = build_reconciled_pairs(
        vocal_rows,
        minus_rows,
        vocal_source_manifest=str(args.vocal_candidates_jsonl),
        minus_source_manifest=str(args.minus_candidates_jsonl),
        max_pairs_per_track=args.max_pairs_per_track,
    )
    write_jsonl(args.output_jsonl, result["rows"])
    write_jsonl(args.skipped_jsonl, result["skipped_rows"])
    write_summary_md(args.summary_md, result["summary"])
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
