#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402
from pseudo_teacher_contract import PSEUDO_TEACHER_SCHEMA  # noqa: E402


MANIFEST_NAME = "accepted_vocal_teacher_training_view_manifest.json"
EXAMPLES_NAME = "accepted_vocal_teacher_training_examples.jsonl"
PSEUDO_EXAMPLES_NAME = "accepted_vocal_pseudo_teacher_training_examples.jsonl"
BLOCKED_NAME = "accepted_vocal_teacher_blocked_examples.jsonl"
PROGRESS_NAME = "accepted_vocal_teacher_training_view_progress.jsonl"
GENERIC_TRAINING_SCHEMA = "generic_finetuning_audio_training_example.v1"
START_TIME = time.monotonic()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            yield payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_metadata_curation",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "metadata_only_accepted_vocal_teacher_view_build",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:18]


def normalize_track_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    normalized = "".join(char if char.isalnum() else "_" for char in text)
    return "_".join(part for part in normalized.split("_") if part)


def source_alias_keys(row: dict[str, Any]) -> set[str]:
    keys = {normalize_track_key(row.get("track_id"))}
    for key in ("docker_audio_path", "source_audio_path", "relative_path"):
        path_text = str(row.get(key) or "").strip()
        if path_text:
            keys.add(normalize_track_key(Path(path_text).stem))
    return {key for key in keys if key}


def source_plan_by_track(rows: Iterable[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    result: dict[str, dict[str, Any]] = {}
    alias_candidates: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        track_id = str(row.get("track_id") or "").strip()
        source_path = str(row.get("docker_audio_path") or "").strip()
        if not track_id or not source_path:
            continue
        result.setdefault(track_id, row)
        for key in source_alias_keys(row):
            alias_candidates.setdefault(key, []).append(row)
    ambiguous_aliases: set[str] = set()
    for key, candidates in alias_candidates.items():
        unique_paths = {str(candidate.get("docker_audio_path") or "") for candidate in candidates}
        if len(unique_paths) != 1:
            ambiguous_aliases.add(key)
            continue
        result.setdefault(key, candidates[0])
    return result, ambiguous_aliases


def source_plan_match(
    track_id: str,
    source_by_track: dict[str, dict[str, Any]],
    ambiguous_aliases: set[str],
) -> tuple[dict[str, Any] | None, str]:
    if track_id in source_by_track:
        return source_by_track[track_id], "exact_track_id"
    normalized = normalize_track_key(track_id)
    if normalized in ambiguous_aliases:
        return None, "ambiguous_source_mix_reconciliation"
    if normalized in source_by_track:
        return source_by_track[normalized], "normalized_unique_source_alias"
    return None, "source_mix_not_reconciled"


def accepted_vocal_inventory_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("accepted_contract") is True
        and str(row.get("stem_type") or "").strip().lower() == "vocal"
        and row.get("reference_verified") is not True
    ]


def existing_file_reason(path_text: str, reason_name: str) -> list[str]:
    if not path_text:
        return [f"missing_{reason_name}"]
    path = Path(path_text)
    if not path.is_file():
        return [f"{reason_name}_not_found"]
    return []


def build_example_row(
    row: dict[str, Any],
    *,
    source_path: str,
    target_path: str,
    canonical_track_id: str,
    source_reconciliation_method: str,
    verification_source: str,
    permission_status: str,
) -> dict[str, Any]:
    inventory_id = str(row.get("id") or "").strip() or stable_id(
        [
            row.get("manifest_path"),
            row.get("selected_audio_path"),
            row.get("track_id"),
            row.get("model_id"),
        ]
    )
    original_track_id = str(row.get("track_id") or "")
    track_id = str(canonical_track_id or original_track_id)
    return {
        "schema": GENERIC_TRAINING_SCHEMA,
        "id": f"accepted_vocal_teacher__{inventory_id}",
        "track_id": track_id,
        "original_track_id": original_track_id,
        "source_plan_track_id": track_id,
        "stem_type": "vocal",
        "source_audio_path": source_path,
        "target_audio_path": target_path,
        "target_audio_sha256": sha256_file(Path(target_path)),
        "manifest_path": str(row.get("manifest_path") or ""),
        "model_id": str(row.get("model_id") or "accepted_no_reference_teacher"),
        "source_reconciliation_method": source_reconciliation_method,
        "rating": "ideal",
        "human_verified": True,
        "reference_verified": False,
        "permission_status": permission_status,
        "split": "train",
        "trainable": True,
        "training_target": "audio_separator_or_stem_adapter",
        "source_teacher_is_ground_truth": False,
        "target_audio_is_ground_truth": False,
        "accepted_no_reference_teacher": True,
        "human_verification_source": verification_source,
        "delivery_allowed": False,
        "release_ready": False,
        "safety": {
            "accepted_no_reference_teacher_not_ground_truth": True,
            "canonical_track_id_from_source_plan": True,
            "source_mix_reconciled_from_testing_stuff_plan": True,
            "fallback_used": False,
            "requires_locked_holdout_before_promotion": True,
            "not_release_artifact": True,
        },
    }


def build_pseudo_teacher_row(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": PSEUDO_TEACHER_SCHEMA,
        "id": f"pseudo__{example['id']}",
        "track_id": str(example.get("track_id") or ""),
        "original_track_id": str(example.get("original_track_id") or ""),
        "source_plan_track_id": str(example.get("source_plan_track_id") or example.get("track_id") or ""),
        "stem_type": "vocal",
        "source_audio_path": str(example.get("source_audio_path") or ""),
        "target_audio_path": str(example.get("target_audio_path") or ""),
        "target_audio_sha256": str(example.get("target_audio_sha256") or ""),
        "manifest_path": str(example.get("manifest_path") or ""),
        "model_id": str(example.get("model_id") or ""),
        "source_reconciliation_method": str(example.get("source_reconciliation_method") or ""),
        "split": "train",
        "trainable": True,
        "pseudo_teacher": True,
        "pseudo_teacher_source": "accepted_no_reference_vocal_teacher_view",
        "pseudo_teacher_targets_are_ground_truth": False,
        "accepted_no_reference_teacher_human_audited": True,
        "human_verification_source": str(example.get("human_verification_source") or ""),
        "human_verified": False,
        "reference_verified": False,
        "delivery_allowed": False,
        "release_ready": False,
        "safety": {
            "accepted_no_reference_teacher_not_ground_truth": True,
            "canonical_track_id_from_source_plan": True,
            "pseudo_projection_for_training_only": True,
            "human_audit_not_exported_as_ground_truth_claim": True,
            "fallback_used": False,
            "requires_locked_holdout_before_promotion": True,
            "not_release_artifact": True,
        },
    }


def blocked_row(row: dict[str, Any], *, reasons: Sequence[str]) -> dict[str, Any]:
    inventory_id = str(row.get("id") or "").strip() or stable_id(
        [
            row.get("manifest_path"),
            row.get("selected_audio_path"),
            row.get("track_id"),
            row.get("model_id"),
        ]
    )
    return {
        "schema": "accepted_vocal_teacher_blocked_example.v1",
        "id": inventory_id,
        "track_id": str(row.get("track_id") or ""),
        "stem_type": str(row.get("stem_type") or ""),
        "manifest_path": str(row.get("manifest_path") or ""),
        "selected_audio_path": str(row.get("selected_audio_path") or ""),
        "blocked_reasons": sorted(set(str(reason) for reason in reasons if str(reason))),
        "fallback_used": False,
        "release_ready": False,
    }


def build_accepted_vocal_teacher_training_view(
    *,
    inventory_jsonl: Path,
    source_plan_jsonl: Path,
    output_dir: Path,
    verification_source: str,
    permission_status: str = "project_owned",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    source_by_track, ambiguous_aliases = source_plan_by_track(iter_jsonl(source_plan_jsonl))
    inventory_rows = accepted_vocal_inventory_rows(iter_jsonl(inventory_jsonl))
    progress(
        progress_path,
        "start",
        inventory_jsonl=str(inventory_jsonl),
        source_plan_jsonl=str(source_plan_jsonl),
        accepted_vocal_inventory_count=len(inventory_rows),
        source_track_count=len(source_by_track),
        ambiguous_source_alias_count=len(ambiguous_aliases),
        verification_source=verification_source,
    )

    examples: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for row in inventory_rows:
        reasons: list[str] = []
        track_id = str(row.get("track_id") or "").strip()
        if not track_id:
            reasons.append("missing_track_id")
        source_plan, source_reconciliation_method = source_plan_match(track_id, source_by_track, ambiguous_aliases)
        if source_plan is None:
            reasons.append(source_reconciliation_method)
            source_path = ""
        else:
            source_path = str(source_plan.get("docker_audio_path") or "").strip()
            reasons.extend(existing_file_reason(source_path, "source_audio"))
        canonical_track_id = str((source_plan or {}).get("track_id") or track_id).strip()
        target_path = str(row.get("selected_audio_path") or "").strip()
        reasons.extend(existing_file_reason(target_path, "target_audio"))
        if reasons:
            blocked.append(blocked_row(row, reasons=reasons))
            continue
        examples.append(
            build_example_row(
                row,
                source_path=source_path,
                target_path=target_path,
                canonical_track_id=canonical_track_id,
                source_reconciliation_method=source_reconciliation_method,
                verification_source=verification_source,
                permission_status=permission_status,
            )
        )

    write_jsonl(output_dir / EXAMPLES_NAME, examples)
    pseudo_examples = [build_pseudo_teacher_row(example) for example in examples]
    write_jsonl(output_dir / PSEUDO_EXAMPLES_NAME, pseudo_examples)
    write_jsonl(output_dir / BLOCKED_NAME, blocked)
    track_counts = Counter(str(row.get("track_id") or "") for row in examples)
    original_track_counts = Counter(str(row.get("original_track_id") or "") for row in examples)
    manifest = {
        "schema": "accepted_vocal_teacher_training_view_manifest.v1",
        "manifest_type": "accepted_vocal_teacher_training_view",
        "status": "passed" if examples else "blocked_no_teacher_examples",
        "inventory_jsonl": str(inventory_jsonl),
        "source_plan_jsonl": str(source_plan_jsonl),
        "teacher_examples_jsonl": str(output_dir / EXAMPLES_NAME),
        "pseudo_teacher_examples_jsonl": str(output_dir / PSEUDO_EXAMPLES_NAME),
        "blocked_examples_jsonl": str(output_dir / BLOCKED_NAME),
        "progress_jsonl": str(progress_path),
        "teacher_example_count": len(examples),
        "pseudo_teacher_example_count": len(pseudo_examples),
        "blocked_example_count": len(blocked),
        "ambiguous_source_alias_count": len(ambiguous_aliases),
        "track_counts": dict(sorted(track_counts.items())),
        "original_track_counts": dict(sorted(original_track_counts.items())),
        "unique_track_count": len(track_counts),
        "unique_original_track_count": len(original_track_counts),
        "verification_source": verification_source,
        "permission_status": permission_status,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "thresholds_relaxed": False,
        "delivery_allowed": False,
        "release_ready": False,
        "safety": {
            "accepted_no_reference_teacher_not_ground_truth": True,
            "blocked_rows_for_missing_source_or_audio": True,
            "fallback_used": False,
            "release_ready_forced_false": True,
        },
    }
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress(
        progress_path,
        "finish",
        status=manifest["status"],
        teacher_example_count=len(examples),
        blocked_example_count=len(blocked),
        unique_track_count=len(track_counts),
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a human-verified accepted-vocal teacher training view with deterministic source-mix reconciliation."
    )
    parser.add_argument("--inventory-jsonl", type=Path, required=True)
    parser.add_argument("--source-plan-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verification-source", required=True)
    parser.add_argument("--permission-status", default="project_owned")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_accepted_vocal_teacher_training_view.py",
        entrypoint_path="/app/tools/build_accepted_vocal_teacher_training_view.py",
    )
    args = parse_args(argv)
    manifest = build_accepted_vocal_teacher_training_view(
        inventory_jsonl=args.inventory_jsonl,
        source_plan_jsonl=args.source_plan_jsonl,
        output_dir=args.output_dir,
        verification_source=str(args.verification_source),
        permission_status=str(args.permission_status),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["teacher_example_count"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
