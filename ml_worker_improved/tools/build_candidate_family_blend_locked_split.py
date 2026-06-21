#!/usr/bin/env python3
"""Build a locked split manifest for candidate-family blend TRAIN decisions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402
from train_teacher_mask_separator import validate_stem_name  # noqa: E402


START_TIME = time.monotonic()
MANIFEST_NAME = "candidate_family_blend_locked_split_manifest.json"
ROWS_NAME = "candidate_family_blend_locked_split_rows.jsonl"
PROGRESS_NAME = "candidate_family_blend_locked_split_progress.jsonl"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid jsonl row: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object row")
            yield row


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_metadata_validation",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "locked_split_metadata_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "locked_split_metadata_only",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _row_stem(row: dict[str, Any]) -> str:
    return str(row.get("stem_type") or row.get("stem") or "").strip()


def _row_family(row: dict[str, Any]) -> str:
    return str(row.get("source_candidate_family") or row.get("candidate_family") or row.get("family") or "").strip()


def _row_verified(row: dict[str, Any]) -> bool:
    return row.get("reference_verified") is True or row.get("human_verified") is True


def _row_safety_failed_checks(row: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    if row.get("release_ready") is True:
        failed.append("release_ready_input_row")
    if row.get("delivery_allowed") is True:
        failed.append("delivery_allowed_input_row")
    if row.get("pseudo_teacher") is True:
        failed.append("pseudo_teacher_input_row")
    if row.get("source_candidate_is_ground_truth") is True:
        failed.append("source_candidate_marked_ground_truth")
    if not _row_verified(row):
        failed.append("unverified_input_row")
    return failed


def _complete_track_groups(
    rows: Sequence[dict[str, Any]],
    *,
    stem: str,
    families: Sequence[str],
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[str]]:
    by_track: dict[str, dict[str, dict[str, Any]]] = {}
    failed: list[str] = []
    family_set = set(families)
    for row in rows:
        if _row_stem(row) != stem:
            continue
        family = _row_family(row)
        if family not in family_set:
            continue
        track_id = str(row.get("track_id") or "").strip()
        if not track_id:
            failed.append("missing_track_id")
            continue
        by_track.setdefault(track_id, {})[family] = row
        failed.extend(_row_safety_failed_checks(row))

    complete: dict[str, dict[str, dict[str, Any]]] = {}
    for track_id, by_family in sorted(by_track.items()):
        missing = [family for family in families if family not in by_family]
        if missing:
            failed.append("incomplete_required_family_group")
            continue
        complete[track_id] = {family: by_family[family] for family in families}
    return complete, sorted(set(failed))


def _split_rows(
    *,
    complete_groups: dict[str, dict[str, dict[str, Any]]],
    train_track_ids: set[str],
    holdout_track_ids: set[str],
    families: Sequence[str],
    stem: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, track_ids in (("train", train_track_ids), ("holdout", holdout_track_ids)):
        for track_id in sorted(track_ids):
            by_family = complete_groups.get(track_id) or {}
            for family in families:
                source_row = by_family.get(family) or {}
                rows.append(
                    {
                        "schema": "candidate_family_blend_locked_split_row.v1",
                        "split": split,
                        "track_id": track_id,
                        "stem": stem,
                        "source_candidate_family": family,
                        "source_row_id": str(source_row.get("id") or ""),
                        "reference_verified": source_row.get("reference_verified") is True,
                        "human_verified": source_row.get("human_verified") is True,
                        "pseudo_teacher": source_row.get("pseudo_teacher") is True,
                        "source_candidate_is_ground_truth": source_row.get("source_candidate_is_ground_truth") is True,
                        "release_ready": False,
                        "delivery_allowed": False,
                    }
                )
    return rows


def build_candidate_family_blend_locked_split(
    *,
    audio_training_examples_jsonl: Path,
    output_dir: Path,
    stem: str,
    families: Sequence[str],
    holdout_track_ids: set[str],
    force_train_track_ids: set[str] | None = None,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    stem = validate_stem_name(stem)
    normalized_families = [str(family).strip() for family in families if str(family).strip()]
    if len(normalized_families) < 2 or len(set(normalized_families)) != len(normalized_families):
        raise ValueError("families must contain at least two unique values")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    holdout_track_ids = {str(track_id).strip() for track_id in holdout_track_ids if str(track_id).strip()}
    forced_train = {str(track_id).strip() for track_id in (force_train_track_ids or set()) if str(track_id).strip()}
    progress(
        progress_path,
        "start",
        audio_training_examples_jsonl=str(audio_training_examples_jsonl),
        stem=stem,
        families=normalized_families,
        holdout_track_ids=sorted(holdout_track_ids),
        force_train_track_ids=sorted(forced_train),
    )
    rows = list(iter_jsonl(audio_training_examples_jsonl))
    complete_groups, failed_checks = _complete_track_groups(rows, stem=stem, families=normalized_families)
    complete_track_ids = set(complete_groups)
    missing_holdouts = sorted(holdout_track_ids - complete_track_ids)
    if missing_holdouts:
        failed_checks.append("holdout_track_missing_complete_group")
    train_track_ids = (complete_track_ids - holdout_track_ids) | forced_train
    missing_forced_train = sorted(forced_train - complete_track_ids)
    if missing_forced_train:
        failed_checks.append("forced_train_track_missing_complete_group")
    split_overlap = train_track_ids & holdout_track_ids
    if split_overlap:
        failed_checks.append("track_split_overlap")
    train_complete = {track_id for track_id in train_track_ids if track_id in complete_groups}
    holdout_complete = {track_id for track_id in holdout_track_ids if track_id in complete_groups}
    if not train_complete:
        failed_checks.append("train_track_missing")
    if not holdout_track_ids:
        failed_checks.append("holdout_track_missing")
    if holdout_track_ids and len(holdout_track_ids) < 2:
        failed_checks.append("multi_holdout_track_count_below_two")
    split_rows = _split_rows(
        complete_groups=complete_groups,
        train_track_ids=train_complete,
        holdout_track_ids=holdout_complete,
        families=normalized_families,
        stem=stem,
    )
    failed_checks = sorted(set(failed_checks))
    critical_leakage_count = 1 if failed_checks else 0
    manifest = {
        "schema": "candidate_family_blend_locked_split_manifest.v1",
        "audio_training_examples_jsonl": str(audio_training_examples_jsonl),
        "output_dir": str(output_dir),
        "split_rows_jsonl": str(output_dir / ROWS_NAME),
        "progress_jsonl": str(progress_path),
        "stem": stem,
        "families": normalized_families,
        "locked": not failed_checks,
        "failed_checks": failed_checks,
        "critical_leakage_count": critical_leakage_count,
        "warning_leakage_count": 0,
        "complete_track_count": len(complete_track_ids),
        "train_track_count": len(train_complete),
        "holdout_track_count": len(holdout_complete),
        "train_track_ids": sorted(train_complete),
        "holdout_track_ids": sorted(holdout_complete),
        "split_overlap_track_ids": sorted(split_overlap),
        "release_ready": False,
        "delivery_allowed": False,
        "promotion_allowed": False,
        "model_registry_edited": False,
        "runtime_gates_changed": False,
        "thresholds_relaxed": False,
        "fallback_used": False,
        "pseudo_teacher_rows_used": False,
        "command_args": command_args or [],
        "contract": {
            "holdout_tracks_excluded_from_training": not split_overlap,
            "complete_family_groups_required": True,
            "multi_holdout_tracks_required": True,
            "dirty_stems_never_ground_truth": True,
            "release_ready_forced_false": True,
        },
    }
    write_jsonl(output_dir / ROWS_NAME, split_rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress(
        progress_path,
        "finish",
        locked=manifest["locked"],
        critical_leakage_count=critical_leakage_count,
        failed_checks=failed_checks,
        train_track_count=len(train_complete),
        holdout_track_count=len(holdout_complete),
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a locked split manifest for candidate-family blend TRAIN decisions.")
    parser.add_argument("--audio-training-examples-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="vocal")
    parser.add_argument("--family", action="append", required=True)
    parser.add_argument("--holdout-track-id", action="append", default=[])
    parser.add_argument("--force-train-track-id", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_candidate_family_blend_locked_split.py",
        entrypoint_path="/app/tools/build_candidate_family_blend_locked_split.py",
    )
    args = parse_args(argv)
    command_args = list(argv if argv is not None else sys.argv[1:])
    manifest = build_candidate_family_blend_locked_split(
        audio_training_examples_jsonl=args.audio_training_examples_jsonl,
        output_dir=args.output_dir,
        stem=str(args.stem),
        families=[str(item) for item in args.family],
        holdout_track_ids={str(item) for item in args.holdout_track_id},
        force_train_track_ids={str(item) for item in args.force_train_track_id},
        command_args=command_args,
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "locked": manifest["locked"],
                "critical_leakage_count": manifest["critical_leakage_count"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["locked"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
