from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from collect_no_reference_candidate_pool import (  # noqa: E402
    SAFETY_FLAGS,
    candidate_name,
    flatten,
    iter_jsonl,
    iter_manifest_paths,
    minus_paths,
    path_exists,
    read_json,
    sanitize,
    score_hints,
    source_paths,
    track_id_from_mix_path,
    vocal_paths,
    write_jsonl,
)


DEFAULT_MAX_CANDIDATES_PER_TRACK = 0
KNOWN_NON_VOCAL_STEMS = {
    "bass",
    "drums",
    "guitar",
    "melody",
    "other",
    "piano",
    "percussion",
    "percussive",
    "strings",
    "synth",
}


def existing_mix_path(flat: list[tuple[str, Any]]) -> str:
    mixes = sorted(path for path in source_paths(flat) if path_exists(path))
    return mixes[0] if mixes else ""


def sibling_vocal_path(minus_path: str) -> str:
    parent = Path(minus_path).parent
    for name in ("vocals.flac", "vocal.flac", "vocals.wav", "vocal.wav"):
        candidate = parent / name
        if candidate.exists():
            return str(candidate)
    return ""


def has_stem_only_minus_inputs(flat: list[tuple[str, Any]]) -> bool:
    for key, value in flat:
        if not path_exists(str(value)):
            continue
        key_lower = key.lower()
        name_lower = Path(str(value)).name.lower()
        if any(token in key_lower or name_lower == f"{token}.flac" for token in KNOWN_NON_VOCAL_STEMS):
            return True
    return False


def manifest_minus_rows(manifest_path: Path, payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    flat = flatten(payload)
    mix_path = existing_mix_path(flat)
    if not mix_path:
        return [], []
    track_id = track_id_from_mix_path(mix_path, manifest_path.parent.name)
    candidate_id_base = candidate_name(payload, manifest_path, "minus")
    hints = score_hints(payload)
    paired_vocal_paths = sorted(path for path in vocal_paths(flat) if path_exists(path))
    manifest_vocal_path = paired_vocal_paths[0] if paired_vocal_paths else ""
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index, minus_path in enumerate(sorted(path for path in minus_paths(flat) if path_exists(path)), start=1):
        vocal_path = manifest_vocal_path or sibling_vocal_path(minus_path)
        rows.append(
            {
                "track_id": track_id,
                "candidate_id": sanitize(candidate_id_base if index == 1 else f"{candidate_id_base}_{index}"),
                "mix_path": mix_path,
                "vocal_path": vocal_path,
                "minus_path": minus_path,
                "instrumental_path": minus_path,
                "selected_minus_path": minus_path,
                "source_manifest": str(manifest_path),
                "candidate_manifest": str(manifest_path),
                "candidate_family": "minus",
                "candidate_origin": "existing_generated_minus_manifest_scan",
                "score_hints": hints,
                "is_complete_pair": True,
                "is_complete_minus_candidate": True,
                "skip_reason": "",
            }
        )
    if not rows and has_stem_only_minus_inputs(flat):
        skipped.append(
            {
                "track_id": track_id,
                "candidate_id": sanitize(candidate_id_base),
                "source_manifest": str(manifest_path),
                "mix_path": mix_path,
                "minus_path": "",
                "skip_reason": "stem_only_requires_render_route",
            }
        )
    return rows, skipped


def seed_minus_rows(seed_candidate_pool_jsonls: Sequence[Path] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for seed_path in seed_candidate_pool_jsonls or []:
        for row in iter_jsonl(seed_path):
            track_id = str(row.get("track_id") or "")
            mix_path = str(row.get("mix_path") or "")
            minus_path = str(row.get("minus_path") or row.get("instrumental_path") or row.get("selected_minus_path") or "")
            candidate_id = sanitize(str(row.get("candidate_id") or row.get("selected_candidate_id") or seed_path.stem))
            missing: list[str] = []
            if not track_id:
                missing.append("track_id")
            if not mix_path or not path_exists(mix_path):
                missing.append("mix_path")
            if not minus_path or not path_exists(minus_path):
                missing.append("minus_path")
            if missing:
                skipped.append(
                    {
                        "track_id": track_id,
                        "candidate_id": candidate_id,
                        "source_manifest": str(seed_path),
                        "mix_path": mix_path,
                        "minus_path": minus_path,
                        "skip_reason": "missing_" + "_".join(missing),
                    }
                )
                continue
            rows.append(
                {
                    "track_id": track_id,
                    "candidate_id": candidate_id,
                    "mix_path": mix_path,
                    "vocal_path": "",
                    "minus_path": minus_path,
                    "instrumental_path": minus_path,
                    "selected_minus_path": minus_path,
                    "source_manifest": str(seed_path),
                    "candidate_manifest": str(row.get("candidate_manifest") or row.get("source_manifest") or seed_path),
                    "candidate_family": str(row.get("candidate_family") or "minus"),
                    "candidate_origin": "existing_generated_minus_seed_pool",
                    "score_hints": row.get("score_hints") if isinstance(row.get("score_hints"), dict) else {},
                    "is_complete_pair": True,
                    "is_complete_minus_candidate": True,
                    "skip_reason": "",
                }
            )
    return rows, skipped


def dedupe_rows(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    deduplicated = 0
    for row in sorted(rows, key=lambda item: (item["track_id"], item["candidate_id"], item["minus_path"])):
        key = (str(row["track_id"]), str(row["mix_path"]), str(row["minus_path"]))
        if key in seen:
            deduplicated += 1
            continue
        seen.add(key)
        unique.append(row)
    return unique, deduplicated


def build_summary(
    rows: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    *,
    seed_minus_candidates: int,
    deduplicated_minus_candidates: int,
    capped_candidates: int,
) -> dict[str, Any]:
    candidates_by_track = Counter(row["track_id"] for row in rows)
    tracks_discovered = len(set(candidates_by_track) | {row["track_id"] for row in skipped})
    return {
        "total_minus_rows_discovered": len(rows) + len(skipped) + deduplicated_minus_candidates + capped_candidates,
        "complete_minus_candidates": len(rows),
        "skipped_incomplete": len(skipped),
        "seed_minus_candidates": seed_minus_candidates,
        "deduplicated_minus_candidates": deduplicated_minus_candidates,
        "capped_candidates": capped_candidates,
        "tracks_discovered": tracks_discovered,
        "tracks_with_1_minus_candidate": sum(1 for count in candidates_by_track.values() if count == 1),
        "tracks_with_2plus_minus_candidates": sum(1 for count in candidates_by_track.values() if count >= 2),
        "skip_reasons": dict(sorted(Counter(row["skip_reason"] for row in skipped).items())),
        "safety": dict(SAFETY_FLAGS),
    }


def collect_minus_candidate_pool(
    lab_artifacts_root: Path,
    *,
    seed_candidate_pool_jsonls: Sequence[Path] | None = None,
    max_candidates_per_track: int = DEFAULT_MAX_CANDIDATES_PER_TRACK,
) -> dict[str, Any]:
    manifest_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for manifest_path in iter_manifest_paths(lab_artifacts_root):
        payload = read_json(manifest_path)
        if payload is None:
            continue
        rows, manifest_skipped = manifest_minus_rows(manifest_path, payload)
        manifest_rows.extend(rows)
        skipped.extend(manifest_skipped)
    seed_rows, seed_skipped = seed_minus_rows(seed_candidate_pool_jsonls)
    skipped.extend(seed_skipped)
    rows, deduplicated = dedupe_rows([*manifest_rows, *seed_rows])
    capped_candidates = 0
    if max_candidates_per_track > 0:
        capped_rows: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        for row in rows:
            track_id = str(row["track_id"])
            if counts[track_id] >= max_candidates_per_track:
                capped_candidates += 1
                continue
            counts[track_id] += 1
            capped_rows.append(row)
        rows = capped_rows
    rows = sorted(rows, key=lambda row: (row["track_id"], row["candidate_id"], row["minus_path"]))
    skipped = sorted(skipped, key=lambda row: (row["track_id"], row["candidate_id"], row["skip_reason"], row["source_manifest"]))
    summary = build_summary(
        rows,
        skipped,
        seed_minus_candidates=len(seed_rows),
        deduplicated_minus_candidates=deduplicated,
        capped_candidates=capped_candidates,
    )
    return {"rows": rows, "skipped": skipped, "summary": summary}


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# No-Reference Minus Candidate Pool Summary",
        "",
        f"- total_minus_rows_discovered: {summary['total_minus_rows_discovered']}",
        f"- complete_minus_candidates: {summary['complete_minus_candidates']}",
        f"- skipped_incomplete: {summary['skipped_incomplete']}",
        f"- seed_minus_candidates: {summary['seed_minus_candidates']}",
        f"- deduplicated_minus_candidates: {summary['deduplicated_minus_candidates']}",
        f"- capped_candidates: {summary['capped_candidates']}",
        f"- tracks_discovered: {summary['tracks_discovered']}",
        f"- tracks_with_1_minus_candidate: {summary['tracks_with_1_minus_candidate']}",
        f"- tracks_with_2plus_minus_candidates: {summary['tracks_with_2plus_minus_candidates']}",
        "",
        "## Skip Reasons",
        "",
    ]
    for key, value in summary["skip_reasons"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Safety", ""])
    for key, value in summary["safety"].items():
        lines.append(f"- {key}: {str(value).lower()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(result: dict[str, Any], output_jsonl: Path, summary_md: Path, skipped_jsonl: Path) -> None:
    write_jsonl(output_jsonl, result["rows"])
    write_jsonl(skipped_jsonl, result["skipped"])
    write_summary_md(summary_md, result["summary"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect existing generated minus candidates for no-reference pair reconciliation.")
    parser.add_argument("--lab-artifacts-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--skipped-jsonl", type=Path, required=True)
    parser.add_argument("--candidate-pool-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--max-candidates-per-track", type=int, default=DEFAULT_MAX_CANDIDATES_PER_TRACK)
    args = parser.parse_args(argv)
    result = collect_minus_candidate_pool(
        args.lab_artifacts_root,
        seed_candidate_pool_jsonls=args.candidate_pool_jsonl,
        max_candidates_per_track=args.max_candidates_per_track,
    )
    write_outputs(result, args.output_jsonl, args.summary_md, args.skipped_jsonl)
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
