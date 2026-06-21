from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.audio_quality_critic_common import (
    BLOCKED_FILE,
    DATASET_FILE,
    DATASET_MANIFEST_SCHEMA,
    DATASET_SCHEMA,
    LABEL_TYPES,
    MANIFEST_FILE,
    OUTPUT_NAMES,
    PROGRESS_FILE,
    PROMOTION_LABEL_TYPES,
    SAFETY_FLAGS,
    WEAK_LABEL_TYPES,
    append_progress,
    as_float,
    as_list,
    iter_jsonl,
    label_type_counts,
    load_rag_current,
    normalize_label_type,
    normalize_stem_type,
    row_identity,
    split_counts,
    split_for_track,
    target_payload,
    write_json,
    write_jsonl,
)
from tools.docker_runtime import require_docker_runtime


def _unsafe_reasons(row: dict[str, Any], label_type: str) -> list[str]:
    reasons: list[str] = []
    if not label_type:
        reasons.append("invalid_label_type")
    if row.get("pseudo_teacher_targets_are_ground_truth") is True:
        reasons.append("pseudo_teacher_claims_ground_truth")
    if row.get("generated_stem_claims_ground_truth") is True:
        reasons.append("generated_stem_claims_ground_truth")
    if row.get("release_ready") is True:
        reasons.append("source_release_ready_true")
    if not as_list(row.get("source_manifest_refs")):
        reasons.append("missing_source_manifest_refs")
    if not as_list(row.get("rag_refs")):
        reasons.append("missing_rag_refs")
    if not str(row.get("tensor_path") or row.get("features") or ""):
        reasons.append("missing_tensor_or_features")
    if not str(row.get("route_id") or "").strip():
        reasons.append("missing_route_id")
    if as_float(row.get("window_end_sec")) <= as_float(row.get("window_start_sec")):
        reasons.append("invalid_window_bounds")
    return reasons


def make_dataset_row(row: dict[str, Any], *, source_jsonl: Path, index: int) -> dict[str, Any]:
    label_type = normalize_label_type(row.get("label_type"))
    if row.get("pseudo_teacher_targets_are_ground_truth") is True or row.get("generated_stem_claims_ground_truth") is True:
        raise ValueError("pseudo_teacher_claims_ground_truth")
    unsafe_reasons = _unsafe_reasons(row, label_type)
    track_id = str(row.get("track_id") or "")
    split = str(row.get("split") or split_for_track(track_id))
    label = int(row.get("label") or 0)
    safe_for_training = not unsafe_reasons
    promotion_eligible = safe_for_training and label_type in PROMOTION_LABEL_TYPES
    row_id = f"aqc_{row_identity(row)}"
    return {
        "schema": DATASET_SCHEMA,
        "row_id": row_id,
        "source_jsonl": str(source_jsonl),
        "source_row_index": index,
        "track_id": track_id,
        "candidate_id": str(row.get("candidate_id") or ""),
        "stem_type": normalize_stem_type(row.get("stem_type")),
        "window_start_sec": as_float(row.get("window_start_sec")),
        "window_end_sec": as_float(row.get("window_end_sec")),
        "mix_path": str(row.get("mix_path") or ""),
        "vocal_path": str(row.get("vocal_path") or ""),
        "minus_path": str(row.get("minus_path") or ""),
        "tensor_path": str(row.get("tensor_path") or ""),
        "features": row.get("features") if isinstance(row.get("features"), (dict, list)) else None,
        "route_id": str(row.get("route_id") or ""),
        "label": label,
        "targets": target_payload({"label": label, "targets": row.get("targets") if isinstance(row.get("targets"), dict) else {}}),
        "label_type": label_type,
        "label_confidence": as_float(row.get("label_confidence"), 0.0),
        "split": split,
        "source_manifest_refs": [str(item) for item in as_list(row.get("source_manifest_refs"))],
        "rag_refs": [item for item in as_list(row.get("rag_refs"))],
        "source_lock_pack_json": str(row.get("source_lock_pack_json") or ""),
        "issue_tags": [str(item) for item in as_list(row.get("issue_tags"))],
        "blocker_reasons": [str(item) for item in as_list(row.get("blocker_reasons"))],
        "safe_for_training": safe_for_training,
        "unsafe_reasons": unsafe_reasons,
        "train_allowed": safe_for_training and split == "train",
        "promotion_eligible": promotion_eligible,
        "weak_label_only": label_type in WEAK_LABEL_TYPES,
        "delivery_allowed": False,
        "release_ready": False,
    }


def build_audio_quality_critic_dataset(
    *,
    source_jsonls: Sequence[Path],
    output_dir: Path,
    rag_current_json: Path | None,
) -> dict[str, Any]:
    start_time = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_FILE
    if progress_path.exists():
        progress_path.unlink()
    append_progress(progress_path, "start", start_time=start_time, source_jsonls=[str(path) for path in source_jsonls])
    rag = load_rag_current(rag_current_json)
    rows: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for source_jsonl in source_jsonls:
        for index, row in enumerate(iter_jsonl(source_jsonl), start=1):
            dataset_row = make_dataset_row(row, source_jsonl=source_jsonl, index=index)
            rows.append(dataset_row)
            if not dataset_row["safe_for_training"]:
                blocked.append(
                    {
                        "row_id": dataset_row["row_id"],
                        "track_id": dataset_row["track_id"],
                        "candidate_id": dataset_row["candidate_id"],
                        "unsafe_reasons": dataset_row["unsafe_reasons"],
                        "delivery_allowed": False,
                        "release_ready": False,
                    }
                )
    write_jsonl(output_dir / DATASET_FILE, rows)
    write_jsonl(output_dir / BLOCKED_FILE, blocked)
    safe_rows = [row for row in rows if row["safe_for_training"]]
    source_lock_refs = sorted({str(row.get("source_lock_pack_json") or "") for row in rows if row.get("source_lock_pack_json")})
    manifest = {
        "schema": DATASET_MANIFEST_SCHEMA,
        "manifest_type": "audio_quality_critic_dataset",
        "dataset_id": f"aqd_{row_identity({'track_id': [row['row_id'] for row in rows]})}",
        "current_rag_id": rag["rag_id"],
        "rag_current_json": str(rag_current_json or ""),
        "rag_idx": rag["idx"],
        "rag_idx_exists": bool(rag.get("idx_exists")),
        "source_jsonls": [str(path) for path in source_jsonls],
        "source_lock_pack_json": source_lock_refs[0] if len(source_lock_refs) == 1 else "",
        "source_lock_pack_jsons": source_lock_refs,
        "row_count": len(rows),
        "safe_training_row_count": len(safe_rows),
        "unsafe_row_count": len(blocked),
        "promotion_eligible_row_count": sum(1 for row in safe_rows if row["promotion_eligible"]),
        "weak_label_row_count": sum(1 for row in safe_rows if row["label_type"] in WEAK_LABEL_TYPES),
        "label_types_allowed": sorted(LABEL_TYPES),
        "promotion_label_types": sorted(PROMOTION_LABEL_TYPES),
        "label_type_counts": label_type_counts(rows),
        "split_counts": split_counts(rows),
        "outputs": {key: str(output_dir / value) for key, value in OUTPUT_NAMES.items()},
        "presentation_status": "audio_quality_critic_dataset_not_for_listen_pack",
        "training_started": False,
        "separator_training_started": False,
        "safety": {
            "generated_stems_never_ground_truth": True,
            "weak_labels_cannot_promote": True,
            "rag_provenance_required_for_training": True,
            "short_windows_safe_artifact_names": True,
        },
        **SAFETY_FLAGS,
    }
    write_json(output_dir / MANIFEST_FILE, manifest)
    append_progress(
        progress_path,
        "finish",
        start_time=start_time,
        row_count=len(rows),
        safe_training_row_count=len(safe_rows),
        unsafe_row_count=len(blocked),
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a provenance-safe audio quality critic dataset.")
    parser.add_argument("--source-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rag-current-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_audio_quality_critic_dataset.py",
        entrypoint_path="/app/tools/build_audio_quality_critic_dataset.py",
    )
    args = parse_args(argv)
    manifest = build_audio_quality_critic_dataset(
        source_jsonls=args.source_jsonl,
        output_dir=args.output_dir,
        rag_current_json=args.rag_current_json,
    )
    print(json.dumps({"manifest_json": str(args.output_dir / MANIFEST_FILE), "row_count": manifest["row_count"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
