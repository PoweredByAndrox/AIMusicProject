#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from build_candidate_conditioned_audio_training_view import (  # noqa: E402
    BLOCKED_NAME,
    EXAMPLES_NAME,
    GENERIC_TRAINING_SCHEMA,
    base_row_unsafe_reasons,
    candidate_audio_path_from_manifest,
    iter_jsonl,
    read_json,
    resolve_existing_file,
    row_is_candidate_view_eligible,
    row_stem,
    sha256_file,
    write_json,
    write_jsonl,
)
from docker_runtime import require_docker_runtime  # noqa: E402
from pseudo_teacher_contract import PSEUDO_TEACHER_SCHEMA  # noqa: E402
from train_teacher_mask_separator import validate_stem_name  # noqa: E402


MANIFEST_NAME = "multi_candidate_conditioned_audio_training_view_manifest.json"
PROGRESS_NAME = "multi_candidate_conditioned_audio_training_view_progress.jsonl"
FAMILY_RE = re.compile(r"[^A-Za-z0-9_.-]+")
START_TIME = time.monotonic()


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_metadata_curation",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "metadata_only_multi_candidate_conditioned_view_build",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def sanitize_family_label(raw: str) -> str:
    label = FAMILY_RE.sub("_", raw.strip()).strip("._-")
    if not label:
        raise ValueError("candidate root family label is empty")
    return label


def parse_candidate_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--candidate-root must use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    return sanitize_family_label(label), Path(raw_path)


def blocked_row(
    row: dict[str, Any],
    *,
    family: str,
    reasons: Sequence[str],
    candidate_manifest_path: Path,
) -> dict[str, Any]:
    return {
        "schema": "multi_candidate_conditioned_audio_training_blocked_example.v1",
        "id": str(row.get("id") or ""),
        "track_id": str(row.get("track_id") or ""),
        "stem_type": row_stem(row),
        "source_candidate_family": family,
        "candidate_execution_manifest_json": str(candidate_manifest_path),
        "original_source_audio_path": str(row.get("source_audio_path") or ""),
        "target_audio_path": str(row.get("target_audio_path") or ""),
        "blocked_reasons": sorted(set(reasons)),
        "fallback_used": False,
        "release_ready": False,
    }


def conditioned_row(
    row: dict[str, Any],
    *,
    family: str,
    candidate_manifest: dict[str, Any],
    candidate_manifest_path: Path,
    candidate_audio_path: Path,
    target_audio_path: Path,
    output_pseudo_teacher_rows: bool = False,
) -> dict[str, Any]:
    model_payload = candidate_manifest.get("model") if isinstance(candidate_manifest.get("model"), dict) else {}
    original_id = str(row.get("id") or "")
    new_row = dict(row)
    row_id = f"candidate_conditioned__{family}__{original_id}"
    new_row.update(
        {
            "schema": PSEUDO_TEACHER_SCHEMA if output_pseudo_teacher_rows else GENERIC_TRAINING_SCHEMA,
            "id": f"pseudo__{row_id}" if output_pseudo_teacher_rows else row_id,
            "source_audio_path": str(candidate_audio_path),
            "target_audio_path": str(target_audio_path),
            "target_audio_sha256": sha256_file(target_audio_path),
            "original_source_audio_path": str(row.get("source_audio_path") or ""),
            "source_candidate_conditioned": True,
            "source_candidate_is_ground_truth": False,
            "source_candidate_family": family,
            "source_candidate_model_id": str(model_payload.get("model_id") or ""),
            "candidate_execution_manifest_json": str(candidate_manifest_path),
            "candidate_source_status": "completed",
            "delivery_allowed": False,
            "release_ready": False,
        }
    )
    if output_pseudo_teacher_rows:
        new_row.update(
            {
                "pseudo_teacher": True,
                "pseudo_teacher_source": "candidate_conditioned_accepted_no_reference_teacher_view",
                "pseudo_teacher_targets_are_ground_truth": False,
                "accepted_no_reference_teacher_human_audited": row.get("accepted_no_reference_teacher") is True
                or row.get("accepted_no_reference_teacher_human_audited") is True,
                "human_verified": False,
                "reference_verified": False,
                "source_teacher_is_ground_truth": False,
                "target_audio_is_ground_truth": False,
                "safety": {
                    **(row.get("safety") if isinstance(row.get("safety"), dict) else {}),
                    "candidate_conditioned_pseudo_teacher_for_training_only": True,
                    "candidate_source_is_not_ground_truth": True,
                    "pseudo_teacher_targets_are_ground_truth": False,
                    "fallback_used": False,
                    "not_release_artifact": True,
                },
            }
        )
    return new_row


def _validate_candidate_roots(candidate_roots: Sequence[tuple[str, Path]]) -> list[tuple[str, Path]]:
    if not candidate_roots:
        raise ValueError("at least one --candidate-root is required")
    seen: set[str] = set()
    normalized: list[tuple[str, Path]] = []
    for raw_label, root in candidate_roots:
        label = sanitize_family_label(raw_label)
        if label in seen:
            raise ValueError(f"duplicate candidate root label: {label}")
        seen.add(label)
        normalized.append((label, root))
    return normalized


def build_multi_candidate_conditioned_audio_training_view(
    *,
    audio_training_examples_jsonl: Path,
    candidate_roots: Sequence[tuple[str, Path]],
    output_dir: Path,
    stem: str = "vocal",
    scope: str = "multi_candidate_conditioned_audio_training_view",
    include_example_ids: set[str] | None = None,
    output_pseudo_teacher_rows: bool = False,
) -> dict[str, Any]:
    stem = validate_stem_name(stem)
    candidate_roots = _validate_candidate_roots(candidate_roots)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    include_example_ids = set(include_example_ids or set())
    rows = list(iter_jsonl(audio_training_examples_jsonl))
    progress(
        progress_path,
        "start",
        scope=scope,
        stem=stem,
        audio_training_examples_jsonl=str(audio_training_examples_jsonl),
        candidate_roots={label: str(root) for label, root in candidate_roots},
        base_row_count=len(rows),
        include_example_ids=sorted(include_example_ids),
        output_pseudo_teacher_rows=bool(output_pseudo_teacher_rows),
    )

    conditioned_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    for family, candidate_root in candidate_roots:
        family_conditioned_count = 0
        family_blocked_count = 0
        progress(progress_path, "family_start", scope=scope, stem=stem, source_candidate_family=family)
        for row in rows:
            row_id = str(row.get("id") or "").strip()
            if include_example_ids and row_id not in include_example_ids:
                continue
            if not row_is_candidate_view_eligible(row, stem=stem):
                continue
            candidate_manifest_path = candidate_root / row_id / "registry_candidate_execution_manifest.json"
            reasons = base_row_unsafe_reasons(row)
            if not candidate_manifest_path.exists():
                blocked_rows.append(
                    blocked_row(
                        row,
                        family=family,
                        reasons=[*reasons, "candidate_execution_manifest_missing"],
                        candidate_manifest_path=candidate_manifest_path,
                    )
                )
                family_blocked_count += 1
                continue

            candidate_manifest = read_json(candidate_manifest_path)
            if str(candidate_manifest.get("status") or "") != "completed":
                reasons.append("candidate_execution_not_completed")
            if candidate_manifest.get("release_ready") is True:
                reasons.append("candidate_execution_release_ready_true")
            candidate_audio_raw = candidate_audio_path_from_manifest(candidate_manifest, stem=stem)
            candidate_audio_path, candidate_reasons = resolve_existing_file(candidate_audio_raw, reason_name="candidate_audio")
            target_audio_path, target_reasons = resolve_existing_file(row.get("target_audio_path"), reason_name="target_audio")
            reasons.extend(candidate_reasons)
            reasons.extend(target_reasons)
            if reasons or candidate_audio_path is None or target_audio_path is None:
                blocked_rows.append(
                    blocked_row(
                        row,
                        family=family,
                        reasons=reasons,
                        candidate_manifest_path=candidate_manifest_path,
                    )
                )
                family_blocked_count += 1
                continue
            conditioned_rows.append(
                conditioned_row(
                    row,
                    family=family,
                    candidate_manifest=candidate_manifest,
                    candidate_manifest_path=candidate_manifest_path,
                    candidate_audio_path=candidate_audio_path,
                    target_audio_path=target_audio_path,
                    output_pseudo_teacher_rows=bool(output_pseudo_teacher_rows),
                )
            )
            family_conditioned_count += 1
        progress(
            progress_path,
            "family_finish",
            scope=scope,
            stem=stem,
            source_candidate_family=family,
            conditioned_example_count=family_conditioned_count,
            blocked_example_count=family_blocked_count,
        )

    examples_path = output_dir / EXAMPLES_NAME
    blocked_path = output_dir / BLOCKED_NAME
    write_jsonl(examples_path, conditioned_rows)
    write_jsonl(blocked_path, blocked_rows)
    track_ids = sorted({str(row.get("track_id") or "") for row in conditioned_rows if str(row.get("track_id") or "")})
    family_counts = dict(sorted(Counter(str(row.get("source_candidate_family") or "") for row in conditioned_rows).items()))
    status = "passed" if conditioned_rows else "failed"
    manifest = {
        "schema": "multi_candidate_conditioned_audio_training_view_manifest.v1",
        "manifest_type": "multi_candidate_conditioned_audio_training_view",
        "status": status,
        "scope": scope,
        "stem": stem,
        "audio_training_examples_jsonl": str(audio_training_examples_jsonl),
        "candidate_roots": {label: str(root) for label, root in candidate_roots},
        "candidate_conditioned_examples_jsonl": str(examples_path),
        "blocked_examples_jsonl": str(blocked_path),
        "output_pseudo_teacher_rows": bool(output_pseudo_teacher_rows),
        "conditioned_example_count": len(conditioned_rows),
        "blocked_example_count": len(blocked_rows),
        "family_counts": family_counts,
        "track_ids": track_ids,
        "unique_track_count": len(track_ids),
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "multi_candidate_conditioned_training_view_not_for_listen_pack",
        "release_status": "training_view_only_requires_locked_holdout",
        "release_ready": False,
        "progress_jsonl": str(progress_path),
        "safety": {
            "metadata_only": True,
            "candidate_source_is_not_ground_truth": True,
            "verified_target_preserved": True,
            "pseudo_teacher_rows_not_ground_truth": bool(output_pseudo_teacher_rows),
            "fallback_to_raw_mix_forbidden": True,
            "release_ready_forced_false": True,
        },
        "provenance": {"tool_path": str(Path(__file__).resolve())},
    }
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress(
        progress_path,
        "finish",
        scope=scope,
        stem=stem,
        status=status,
        conditioned_example_count=len(conditioned_rows),
        blocked_example_count=len(blocked_rows),
        family_counts=family_counts,
        output_pseudo_teacher_rows=bool(output_pseudo_teacher_rows),
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a training view using multiple registered candidate families as sources and verified stems as targets."
    )
    parser.add_argument("--audio-training-examples-jsonl", type=Path, required=True)
    parser.add_argument("--candidate-root", action="append", default=[], help="Candidate source root in LABEL=PATH form.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="vocal")
    parser.add_argument("--scope", default="multi_candidate_conditioned_audio_training_view")
    parser.add_argument("--include-example-id", action="append", default=[])
    parser.add_argument("--output-pseudo-teacher-rows", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_multi_candidate_conditioned_audio_training_view.py",
        entrypoint_path="/app/tools/build_multi_candidate_conditioned_audio_training_view.py",
    )
    args = parse_args(argv)
    manifest = build_multi_candidate_conditioned_audio_training_view(
        audio_training_examples_jsonl=args.audio_training_examples_jsonl,
        candidate_roots=[parse_candidate_root(str(item)) for item in args.candidate_root],
        output_dir=args.output_dir,
        stem=str(args.stem),
        scope=str(args.scope),
        include_example_ids={str(item) for item in args.include_example_id},
        output_pseudo_teacher_rows=bool(args.output_pseudo_teacher_rows),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "status": manifest["status"],
                "conditioned_example_count": manifest["conditioned_example_count"],
                "blocked_example_count": manifest["blocked_example_count"],
                "family_counts": manifest["family_counts"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
