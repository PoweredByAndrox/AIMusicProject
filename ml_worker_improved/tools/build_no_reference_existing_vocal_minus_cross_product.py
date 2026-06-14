from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Sequence


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
    "do_not_accept_without_dense_post_cross_product_judge",
]


class StemCandidate(NamedTuple):
    track_id: str
    candidate_id: str
    mix_path: str
    path: str
    source_manifest: str
    source_origin: str
    source_family: str


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


def short_id(*parts: str) -> str:
    value = "::".join(parts)
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    clean = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
    clean = "_".join(part for part in clean.split("_") if part)
    if len(clean) > 96:
        clean = clean[:96].rstrip("_-.")
    return f"{clean or 'cross_pair'}_{digest}"


def row_candidate_id(row: dict[str, Any]) -> str:
    return str(row.get("selected_candidate_id") or row.get("candidate_id") or row.get("input_candidate_id") or "")


def row_source_manifest(row: dict[str, Any], default: str) -> str:
    return str(row.get("source_manifest") or row.get("candidate_manifest") or default)


def row_vocal_path(row: dict[str, Any]) -> str:
    return str(row.get("selected_vocal_path") or row.get("vocal_path") or "")


def row_minus_path(row: dict[str, Any]) -> str:
    return str(row.get("selected_minus_path") or row.get("instrumental_path") or row.get("minus_path") or "")


def stem_candidate_from_row(row: dict[str, Any], *, role: str, source_manifest: str) -> StemCandidate | None:
    path = row_vocal_path(row) if role == "vocal" else row_minus_path(row)
    track_id = str(row.get("track_id") or "")
    mix_path = str(row.get("mix_path") or "")
    if not track_id or not mix_path or not path:
        return None
    return StemCandidate(
        track_id=track_id,
        candidate_id=row_candidate_id(row),
        mix_path=mix_path,
        path=path,
        source_manifest=row_source_manifest(row, source_manifest),
        source_origin=str(row.get("candidate_origin") or ""),
        source_family=str(row.get("candidate_family") or ""),
    )


def skipped_row(row: dict[str, Any], *, skip_reason: str, stem_role: str) -> dict[str, Any]:
    return {
        "track_id": str(row.get("track_id") or ""),
        "candidate_id": row_candidate_id(row),
        "stem_role": stem_role,
        "skip_reason": skip_reason,
    }


def pair_row(
    *,
    track_id: str,
    vocal: StemCandidate,
    minus: StemCandidate,
    track_identity_match_method: str,
    vocal_source_manifest: str,
    minus_source_manifest: str,
) -> dict[str, Any]:
    candidate_id = short_id("cross", track_id, vocal.candidate_id, minus.candidate_id, vocal.path, minus.path)
    return {
        "track_id": track_id,
        "candidate_id": candidate_id,
        "mix_path": vocal.mix_path,
        "track_identity_match_method": track_identity_match_method,
        "vocal_path": vocal.path,
        "instrumental_path": minus.path,
        "minus_path": minus.path,
        "selected_vocal_path": vocal.path,
        "selected_minus_path": minus.path,
        "final_decision": "cross_product_candidate",
        "pair_compatibility_decision": "",
        "needs_pair_reconciliation": True,
        "segment_blend_needed": False,
        "candidate_family": "existing_vocal_minus_cross_product",
        "candidate_origin": "no_reference_existing_vocal_minus_cross_product",
        "is_complete_pair": True,
        "source_manifest": vocal_source_manifest,
        "candidate_manifest": "",
        "source_vocal_candidate_id": vocal.candidate_id,
        "source_minus_candidate_id": minus.candidate_id,
        "requires_dense_post_cross_product_judge": True,
        "requires_dense_post_reconciliation_judge": True,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "audio_copied": False,
        "audio_exported": False,
        "forbidden_actions": list(BASE_FORBIDDEN_ACTIONS),
        "cross_product_provenance": {
            "vocal_source_manifest": vocal_source_manifest,
            "minus_source_manifest": minus_source_manifest,
            "vocal_row_source_manifest": vocal.source_manifest,
            "minus_row_source_manifest": minus.source_manifest,
            "source_vocal_track_id": vocal.track_id,
            "source_minus_track_id": minus.track_id,
            "vocal_candidate_id": vocal.candidate_id,
            "minus_candidate_id": minus.candidate_id,
            "vocal_candidate_origin": vocal.source_origin,
            "minus_candidate_origin": minus.source_origin,
            "vocal_candidate_family": vocal.source_family,
            "minus_candidate_family": minus.source_family,
            "vocal_path": vocal.path,
            "minus_path": minus.path,
        },
    }


def build_cross_product_pairs(
    vocal_rows: Sequence[dict[str, Any]],
    minus_rows: Sequence[dict[str, Any]],
    *,
    vocal_source_manifest: str,
    minus_source_manifest: str,
    max_pairs_per_track: int = 0,
) -> dict[str, Any]:
    skipped_rows: list[dict[str, Any]] = []
    vocals_by_track: dict[str, list[StemCandidate]] = defaultdict(list)
    minuses_by_track: dict[str, list[StemCandidate]] = defaultdict(list)

    for row in vocal_rows:
        candidate = stem_candidate_from_row(row, role="vocal", source_manifest=vocal_source_manifest)
        if candidate is None:
            skipped_rows.append(skipped_row(row, skip_reason="vocal_missing_track_mix_or_path", stem_role="vocal"))
            continue
        vocals_by_track[candidate.track_id].append(candidate)

    for row in minus_rows:
        candidate = stem_candidate_from_row(row, role="minus", source_manifest=minus_source_manifest)
        if candidate is None:
            skipped_rows.append(skipped_row(row, skip_reason="minus_missing_track_mix_or_path", stem_role="minus"))
            continue
        minuses_by_track[candidate.track_id].append(candidate)

    rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str, str]] = set()
    deduplicated = 0
    pair_counts: Counter[str] = Counter()
    vocals = sorted(
        [candidate for candidates in vocals_by_track.values() for candidate in candidates],
        key=lambda item: (item.track_id, item.candidate_id, item.path, item.mix_path),
    )
    minuses = sorted(
        [candidate for candidates in minuses_by_track.values() for candidate in candidates],
        key=lambda item: (item.track_id, item.candidate_id, item.path, item.mix_path),
    )
    for vocal in vocals:
        for minus in minuses:
            same_track = vocal.track_id == minus.track_id
            same_mix = vocal.mix_path == minus.mix_path
            if same_track and not same_mix:
                skipped_rows.append(
                    {
                        "track_id": vocal.track_id,
                        "candidate_id": short_id("skipped_cross", vocal.track_id, vocal.candidate_id, minus.candidate_id),
                        "stem_role": "cross_pair",
                        "skip_reason": "mix_path_mismatch",
                        "vocal_mix_path": vocal.mix_path,
                        "minus_mix_path": minus.mix_path,
                        "source_vocal_candidate_id": vocal.candidate_id,
                        "source_minus_candidate_id": minus.candidate_id,
                    }
                )
                continue
            if not same_track and not same_mix:
                continue

            track_id = vocal.track_id
            match_method = "track_id_and_mix_path" if same_track else "shared_mix_path"
            if max_pairs_per_track > 0 and pair_counts[track_id] >= max_pairs_per_track:
                skipped_rows.append(
                    {
                        "track_id": track_id,
                        "candidate_id": short_id("skipped_cross_cap", track_id, vocal.candidate_id, minus.candidate_id),
                        "stem_role": "cross_pair",
                        "skip_reason": "max_pairs_per_track_reached",
                        "source_vocal_candidate_id": vocal.candidate_id,
                        "source_minus_candidate_id": minus.candidate_id,
                    }
                )
                continue
            key = (track_id, vocal.path, minus.path)
            if key in seen_pairs:
                deduplicated += 1
                continue
            seen_pairs.add(key)
            pair_counts[track_id] += 1
            rows.append(
                pair_row(
                    track_id=track_id,
                    vocal=vocal,
                    minus=minus,
                    track_identity_match_method=match_method,
                    vocal_source_manifest=vocal_source_manifest,
                    minus_source_manifest=minus_source_manifest,
                )
            )

    rows = sorted(rows, key=lambda row: (row["track_id"], row["candidate_id"], row["vocal_path"], row["minus_path"]))
    skipped_rows = sorted(skipped_rows, key=lambda row: (row["stem_role"], row["track_id"], row["candidate_id"], row["skip_reason"]))
    summary = {
        "eligible_vocal_rows": sum(len(items) for items in vocals_by_track.values()),
        "eligible_minus_rows": sum(len(items) for items in minuses_by_track.values()),
        "cross_product_pair_count": len(rows),
        "skipped_count": len(skipped_rows),
        "deduplicated_pair_count": deduplicated,
        "tracks_with_cross_product_pairs": len({row["track_id"] for row in rows}),
        "max_pairs_per_track": max_pairs_per_track,
        "can_publish_as_result_count": sum(1 for row in rows if row.get("can_publish_as_result") is True),
        "can_mark_release_ready_count": sum(1 for row in rows if row.get("can_mark_release_ready") is True),
        "safety": dict(SAFETY_FLAGS),
    }
    return {"rows": rows, "skipped_rows": skipped_rows, "summary": summary}


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# No-Reference Existing Vocal/Minus Cross-Product Summary",
        "",
        f"- eligible_vocal_rows: {summary['eligible_vocal_rows']}",
        f"- eligible_minus_rows: {summary['eligible_minus_rows']}",
        f"- cross_product_pair_count: {summary['cross_product_pair_count']}",
        f"- skipped_count: {summary['skipped_count']}",
        f"- deduplicated_pair_count: {summary['deduplicated_pair_count']}",
        f"- tracks_with_cross_product_pairs: {summary['tracks_with_cross_product_pairs']}",
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
    parser = argparse.ArgumentParser(description="Build no-reference cross-product vocal/minus candidate pairs from existing artifacts.")
    parser.add_argument("--vocal-candidates-jsonl", type=Path, required=True)
    parser.add_argument("--minus-candidates-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--skipped-jsonl", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--max-pairs-per-track", type=int, default=0)
    args = parser.parse_args(argv)

    vocal_rows = list(iter_jsonl(args.vocal_candidates_jsonl))
    minus_rows = list(iter_jsonl(args.minus_candidates_jsonl))
    result = build_cross_product_pairs(
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
