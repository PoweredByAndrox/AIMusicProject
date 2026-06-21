#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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


MANIFEST_NAME = "candidate_conditioned_audio_training_view_manifest.json"
EXAMPLES_NAME = "candidate_conditioned_audio_training_examples.jsonl"
BLOCKED_NAME = "candidate_conditioned_audio_training_blocked_examples.jsonl"
PROGRESS_NAME = "candidate_conditioned_audio_training_view_progress.jsonl"
GENERIC_TRAINING_SCHEMA = "generic_finetuning_audio_training_example.v1"
START_TIME = time.monotonic()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        yield payload


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
        "expected_acceleration_mode": "external_only_cpu_metadata_curation",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "metadata_only_candidate_conditioned_view_build",
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


def resolve_existing_file(value: Any, *, reason_name: str) -> tuple[Path | None, list[str]]:
    raw = str(value or "").strip()
    if not raw:
        return None, [f"missing_{reason_name}"]
    path = Path(raw)
    if not path.exists() or not path.is_file():
        return path, [f"{reason_name}_not_found"]
    return path, []


def row_stem(row: dict[str, Any]) -> str:
    return str(row.get("stem_type") or row.get("stem") or "").strip()


def row_is_candidate_view_eligible(row: dict[str, Any], *, stem: str) -> bool:
    if row.get("trainable") is False:
        return False
    if row_stem(row) != stem:
        return False
    return bool(str(row.get("id") or "").strip())


def base_row_unsafe_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if row.get("schema") != GENERIC_TRAINING_SCHEMA:
        reasons.append("unexpected_schema")
    if row.get("human_verified") is not True and row.get("reference_verified") is not True:
        reasons.append("target_not_human_or_reference_verified")
    if row.get("release_ready") is True:
        reasons.append("base_row_release_ready_true")
    if row.get("delivery_allowed") is True:
        reasons.append("base_row_delivery_allowed_true")
    return reasons


def candidate_audio_path_from_manifest(manifest: dict[str, Any], *, stem: str) -> str:
    candidate_manifest = manifest.get("candidate_manifest")
    if isinstance(candidate_manifest, dict):
        if str(candidate_manifest.get("target_stem") or "") == stem and str(candidate_manifest.get("target_audio_path") or ""):
            return str(candidate_manifest["target_audio_path"])
    return ""


def blocked_row(row: dict[str, Any], *, reasons: Sequence[str], candidate_manifest_path: Path) -> dict[str, Any]:
    return {
        "schema": "candidate_conditioned_audio_training_blocked_example.v1",
        "id": str(row.get("id") or ""),
        "track_id": str(row.get("track_id") or ""),
        "stem_type": row_stem(row),
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
    candidate_manifest: dict[str, Any],
    candidate_manifest_path: Path,
    candidate_audio_path: Path,
    target_audio_path: Path,
) -> dict[str, Any]:
    model_payload = candidate_manifest.get("model") if isinstance(candidate_manifest.get("model"), dict) else {}
    new_row = dict(row)
    original_id = str(row.get("id") or "")
    new_row.update(
        {
            "schema": GENERIC_TRAINING_SCHEMA,
            "id": f"candidate_conditioned__{original_id}",
            "source_audio_path": str(candidate_audio_path),
            "target_audio_path": str(target_audio_path),
            "target_audio_sha256": sha256_file(target_audio_path),
            "original_source_audio_path": str(row.get("source_audio_path") or ""),
            "source_candidate_conditioned": True,
            "source_candidate_is_ground_truth": False,
            "source_candidate_model_id": str(model_payload.get("model_id") or ""),
            "candidate_execution_manifest_json": str(candidate_manifest_path),
            "candidate_source_status": "completed",
            "delivery_allowed": False,
            "release_ready": False,
        }
    )
    return new_row


def build_candidate_conditioned_audio_training_view(
    *,
    audio_training_examples_jsonl: Path,
    candidate_root: Path,
    output_dir: Path,
    stem: str = "vocal",
    scope: str = "candidate_conditioned_audio_training_view",
    include_example_ids: set[str] | None = None,
) -> dict[str, Any]:
    stem = validate_stem_name(stem)
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
        candidate_root=str(candidate_root),
        base_row_count=len(rows),
        include_example_ids=sorted(include_example_ids),
    )

    conditioned_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(row.get("id") or "").strip()
        if include_example_ids and row_id not in include_example_ids:
            continue
        if not row_is_candidate_view_eligible(row, stem=stem):
            continue

        candidate_manifest_path = candidate_root / row_id / "registry_candidate_execution_manifest.json"
        reasons = base_row_unsafe_reasons(row)
        if not candidate_manifest_path.exists():
            blocked_rows.append(blocked_row(row, reasons=[*reasons, "candidate_execution_manifest_missing"], candidate_manifest_path=candidate_manifest_path))
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
            blocked_rows.append(blocked_row(row, reasons=reasons, candidate_manifest_path=candidate_manifest_path))
            continue
        conditioned_rows.append(
            conditioned_row(
                row,
                candidate_manifest=candidate_manifest,
                candidate_manifest_path=candidate_manifest_path,
                candidate_audio_path=candidate_audio_path,
                target_audio_path=target_audio_path,
            )
        )

    examples_path = output_dir / EXAMPLES_NAME
    blocked_path = output_dir / BLOCKED_NAME
    write_jsonl(examples_path, conditioned_rows)
    write_jsonl(blocked_path, blocked_rows)
    track_ids = sorted({str(row.get("track_id") or "") for row in conditioned_rows if str(row.get("track_id") or "")})
    status = "passed" if conditioned_rows else "failed"
    manifest = {
        "schema": "candidate_conditioned_audio_training_view_manifest.v1",
        "manifest_type": "candidate_conditioned_audio_training_view",
        "status": status,
        "scope": scope,
        "stem": stem,
        "audio_training_examples_jsonl": str(audio_training_examples_jsonl),
        "candidate_root": str(candidate_root),
        "candidate_conditioned_examples_jsonl": str(examples_path),
        "blocked_examples_jsonl": str(blocked_path),
        "conditioned_example_count": len(conditioned_rows),
        "blocked_example_count": len(blocked_rows),
        "track_ids": track_ids,
        "unique_track_count": len(track_ids),
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "candidate_conditioned_training_view_not_for_listen_pack",
        "release_status": "training_view_only_requires_locked_holdout",
        "release_ready": False,
        "progress_jsonl": str(progress_path),
        "safety": {
            "metadata_only": True,
            "candidate_source_is_not_ground_truth": True,
            "verified_target_preserved": True,
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
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a training view using registered vocal candidates as sources and verified stems as targets.")
    parser.add_argument("--audio-training-examples-jsonl", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="vocal")
    parser.add_argument("--scope", default="candidate_conditioned_audio_training_view")
    parser.add_argument("--include-example-id", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_candidate_conditioned_audio_training_view.py",
        entrypoint_path="/app/tools/build_candidate_conditioned_audio_training_view.py",
    )
    args = parse_args(argv)
    manifest = build_candidate_conditioned_audio_training_view(
        audio_training_examples_jsonl=args.audio_training_examples_jsonl,
        candidate_root=args.candidate_root,
        output_dir=args.output_dir,
        stem=str(args.stem),
        scope=str(args.scope),
        include_example_ids={str(item) for item in args.include_example_id},
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "status": manifest["status"],
                "conditioned_example_count": manifest["conditioned_example_count"],
                "blocked_example_count": manifest["blocked_example_count"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
