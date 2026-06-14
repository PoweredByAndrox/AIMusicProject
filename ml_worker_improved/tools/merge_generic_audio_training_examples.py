from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime


MERGED_NAME = "merged_audio_training_examples.jsonl"
REJECTED_NAME = "rejected_audio_training_examples.jsonl"
DUPLICATE_NAME = "duplicate_audio_training_examples.jsonl"
MANIFEST_NAME = "merged_audio_training_examples_manifest.json"
PROGRESS_NAME = "merge_generic_audio_training_examples_progress.jsonl"
START_TIME = time.monotonic()

TRAINABLE_PERMISSION_STATUSES = {"owned", "licensed", "cleared", "project_owned"}
VOCAL_STEMS = {"vocal", "vocals", "voice", "lead_vocal", "backing_vocal"}
MINUS_STEMS = {"minus", "instrumental", "no_vocals", "karaoke"}


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
                raise ValueError(f"{path}:{line_number}: expected object row")
            yield row


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "metadata_only_audio_training_dataset_merge",
        "expected_acceleration_mode": "external_only_cpu_metadata_curation",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_stem(stem: Any) -> str:
    value = str(stem or "").strip().lower()
    if value in VOCAL_STEMS:
        return "vocal"
    if value in MINUS_STEMS:
        return "minus"
    return value or "unknown"


def input_sha256(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths}


def rejection_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    permission_status = str(row.get("permission_status") or "").strip().lower()
    source_audio_path = str(row.get("source_audio_path") or "")
    target_audio_path = str(row.get("target_audio_path") or "")
    if row.get("trainable") is not True:
        reasons.append("not_marked_trainable")
    if permission_status not in TRAINABLE_PERMISSION_STATUSES:
        reasons.append("permission_not_trainable")
    if not source_audio_path:
        reasons.append("missing_source_audio_path")
    elif not Path(source_audio_path).exists():
        reasons.append("source_audio_not_found")
    if not target_audio_path:
        reasons.append("missing_target_audio_path")
    elif not Path(target_audio_path).exists():
        reasons.append("target_audio_not_found")
    return reasons


def normalized_example(
    row: dict[str, Any],
    *,
    merge_source: str,
    input_path: Path,
    input_index: int,
) -> dict[str, Any]:
    target_path = Path(str(row["target_audio_path"]))
    target_sha = str(row.get("target_audio_sha256") or "").strip() or sha256_file(target_path)
    original_id = str(row.get("id") or "")
    normalized = dict(row)
    normalized.update(
        {
            "schema": "generic_finetuning_audio_training_example.v1",
            "id": original_id or stable_hash([merge_source, str(input_path), input_index, target_sha])[:16],
            "stem_type": normalize_stem(row.get("stem_type")),
            "permission_status": str(row.get("permission_status") or "").strip().lower(),
            "target_audio_sha256": target_sha,
            "trainable": True,
            "training_target": "audio_separator_or_stem_adapter",
            "merge_source": merge_source,
            "merge_input_path": str(input_path),
            "merge_input_index": input_index,
            "merge_original_id": original_id,
        }
    )
    safety = dict(normalized.get("safety") or {})
    safety.update(
        {
            "not_release_artifact": True,
            "requires_eval_before_promotion": True,
            "merged_training_dataset_row": True,
        }
    )
    normalized["safety"] = safety
    return normalized


def dedupe_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        normalize_stem(row.get("stem_type")),
        str(row.get("source_audio_path") or ""),
        str(row.get("target_audio_path") or ""),
        str(row.get("target_audio_sha256") or ""),
    )


def collect_rows(paths: Sequence[Path], *, merge_source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for input_path in paths:
        for input_index, row in enumerate(iter_jsonl(input_path), start=1):
            reasons = rejection_reasons(row)
            if reasons:
                rejected.append(
                    {
                        "schema": "generic_audio_training_merge_rejected_row.v1",
                        "id": str(row.get("id") or stable_hash([str(input_path), input_index])[:16]),
                        "track_id": row.get("track_id"),
                        "stem_type": normalize_stem(row.get("stem_type")),
                        "merge_source": merge_source,
                        "merge_input_path": str(input_path),
                        "merge_input_index": input_index,
                        "source_audio_path": row.get("source_audio_path"),
                        "target_audio_path": row.get("target_audio_path"),
                        "rejection_reasons": reasons,
                    }
                )
                continue
            accepted.append(
                normalized_example(
                    row,
                    merge_source=merge_source,
                    input_path=input_path,
                    input_index=input_index,
                )
            )
    return accepted, rejected


def merge_generic_audio_training_examples(
    *,
    base_audio_training_examples_jsonls: Sequence[Path],
    approved_audio_training_examples_jsonls: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    input_paths = [*base_audio_training_examples_jsonls, *approved_audio_training_examples_jsonls]
    progress(
        progress_path,
        "start",
        base_audio_training_examples_jsonls=[str(path) for path in base_audio_training_examples_jsonls],
        approved_audio_training_examples_jsonls=[str(path) for path in approved_audio_training_examples_jsonls],
    )

    base_rows, base_rejected = collect_rows(base_audio_training_examples_jsonls, merge_source="base")
    approved_rows, approved_rejected = collect_rows(
        approved_audio_training_examples_jsonls,
        merge_source="approved_recovery",
    )
    progress(progress_path, "rows_loaded", base_rows=len(base_rows), approved_rows=len(approved_rows))

    merged_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    for row in [*base_rows, *approved_rows]:
        key = dedupe_key(row)
        existing = merged_by_key.get(key)
        if existing is not None:
            duplicates.append(
                {
                    "schema": "generic_audio_training_merge_duplicate_row.v1",
                    "id": row["id"],
                    "duplicate_of": existing["id"],
                    "track_id": row.get("track_id"),
                    "stem_type": row.get("stem_type"),
                    "merge_source": row.get("merge_source"),
                    "merge_input_path": row.get("merge_input_path"),
                    "dedupe_key_hash": stable_hash(key)[:16],
                }
            )
            continue
        merged_by_key[key] = row

    merged_rows = sorted(
        merged_by_key.values(),
        key=lambda item: (str(item.get("track_id") or ""), normalize_stem(item.get("stem_type")), str(item.get("id") or "")),
    )
    rejected_rows = sorted([*base_rejected, *approved_rejected], key=lambda item: (item["id"], item["merge_input_path"]))
    duplicates.sort(key=lambda item: (item["duplicate_of"], item["id"]))
    stem_counts = dict(sorted(Counter(normalize_stem(row.get("stem_type")) for row in merged_rows).items()))
    merge_source_counts = dict(sorted(Counter(str(row.get("merge_source") or "") for row in merged_rows).items()))
    unique_tracks = sorted({str(row.get("track_id") or row.get("manifest_path") or row.get("target_audio_path")) for row in merged_rows})
    summary = {
        "input_rows": len(base_rows) + len(approved_rows) + len(rejected_rows),
        "accepted_input_rows": len(base_rows) + len(approved_rows),
        "merged_examples": len(merged_rows),
        "duplicate_examples": len(duplicates),
        "rejected_rows": len(rejected_rows),
        "unique_track_count": len(unique_tracks),
        "stem_counts": stem_counts,
        "merge_source_counts": merge_source_counts,
    }
    manifest = {
        "schema": "merged_audio_training_examples_manifest.v1",
        "manifest_type": "merged_audio_training_examples",
        "output_dir": str(output_dir),
        "fingerprint": stable_hash(
            {
                "rows": [
                    {
                        "id": row["id"],
                        "track_id": row.get("track_id"),
                        "stem_type": row.get("stem_type"),
                        "target_audio_sha256": row.get("target_audio_sha256"),
                    }
                    for row in merged_rows
                ],
                "duplicates": duplicates,
                "rejected": rejected_rows,
            }
        ),
        "summary": summary,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "merged_training_dataset_not_for_listen_pack",
        "release_status": "training_dataset_artifact_only",
        "release_ready": False,
        "inputs": {
            "base_audio_training_examples_jsonls": [str(path) for path in base_audio_training_examples_jsonls],
            "approved_audio_training_examples_jsonls": [str(path) for path in approved_audio_training_examples_jsonls],
            "input_sha256": input_sha256(input_paths),
        },
        "outputs": {
            "merged_audio_training_examples_jsonl": str(output_dir / MERGED_NAME),
            "rejected_audio_training_examples_jsonl": str(output_dir / REJECTED_NAME),
            "duplicate_audio_training_examples_jsonl": str(output_dir / DUPLICATE_NAME),
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "progress_jsonl": str(progress_path),
        },
        "recommended_next_command": (
            "python /app/tools/plan_generic_audio_finetune_jobs.py "
            f"--audio-training-examples-jsonl {output_dir / MERGED_NAME} "
            f"--output-dir {output_dir / 'audio_finetune_plan'} "
            "--min-unique-tracks 3"
        ),
        "safety": {
            "metadata_only": True,
            "no_training_started": True,
            "not_release_artifact": True,
            "requires_holdout_eval_before_promotion": True,
            "fail_closed_on_untrainable_or_missing_audio": True,
        },
    }
    write_jsonl(output_dir / MERGED_NAME, merged_rows)
    write_jsonl(output_dir / REJECTED_NAME, rejected_rows)
    write_jsonl(output_dir / DUPLICATE_NAME, duplicates)
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress(progress_path, "finish", **summary)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge generic audio training examples into one planner-ready JSONL.")
    parser.add_argument("--base-audio-training-examples-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--approved-audio-training-examples-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="merge_generic_audio_training_examples.py",
        entrypoint_path="/app/tools/merge_generic_audio_training_examples.py",
    )
    args = parse_args(argv)
    manifest = merge_generic_audio_training_examples(
        base_audio_training_examples_jsonls=args.base_audio_training_examples_jsonl,
        approved_audio_training_examples_jsonls=args.approved_audio_training_examples_jsonl,
        output_dir=args.output_dir,
    )
    print(json.dumps({"summary": manifest["summary"], "fingerprint": manifest["fingerprint"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
