#!/usr/bin/env python3
"""Build the next metadata-only ARIS full-pack loop command from preflight."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime  # noqa: E402


START_TIME = time.monotonic()
MANIFEST_NAME = "next.json"
ARGS_NAME = "args.jsonl"
BLOCKED_NAME = "blocked.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
RUNNER_ENTRYPOINT = "/app/tools/run_full_pack_aris_metadata_loop.py"
DEFAULT_TRAINING_OUTPUT_ROOT = "/app/lab_artifacts/generic_finetuning_playbook_20260613/audio_finetune_runs"
READY_STATUSES = {"ready_for_reference_intake", "partial_ready_waiting_for_reference_bundles"}
BLOCKED_WAITING_STATUS = "blocked_waiting_for_reference_bundles"
NONFATAL_REFERENCE_PREFLIGHT_FAILED_CHECKS = {
    "handoff_status_not_ready",
    "handoff_failed_checks_not_empty",
    "handoff_empty",
}
BLOCKED_TRUE_FLAGS = (
    "delivery_allowed",
    "release_ready",
    "can_publish_as_result",
    "training_started",
    "fine_tune_started",
    "models_changed",
    "model_registry_edited",
    "audio_read",
    "audio_copied",
    "audio_decoded",
    "dataset_built",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        text = raw.strip()
        if not text:
            continue
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        yield payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_next_run_planning",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "next_run_plan_reads_metadata_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "next_run_plan_never_starts_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def collect_true_flag_failures(prefix: str, payload: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    for key in BLOCKED_TRUE_FLAGS:
        if payload.get(key) is True:
            failed.append(f"{prefix}_{key}_true")
    for nested_key in ("safety", "contract"):
        nested = payload.get(nested_key)
        if not isinstance(nested, dict):
            continue
        for key in BLOCKED_TRUE_FLAGS:
            if nested.get(key) is True:
                failed.append(f"{prefix}_{nested_key}_{key}_true")
    return failed


def resolved_output_path(payload: dict[str, Any], key: str, *, base_dir: Path) -> Path | None:
    outputs = payload.get("outputs")
    text = ""
    if isinstance(outputs, dict):
        text = str(outputs.get(key) or "")
    text = text or str(payload.get(key) or "")
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    return path


def load_blocked_rows(preflight_json: Path, preflight: dict[str, Any], failed: list[str]) -> list[dict[str, Any]]:
    blocked_count = int(preflight.get("blocked_bundle_count") or 0)
    path = resolved_output_path(preflight, "blocked_jsonl", base_dir=preflight_json.parent)
    if path is None:
        if blocked_count:
            failed.append("preflight_blocked_jsonl_missing")
        return []
    if not path.is_file():
        if blocked_count:
            failed.append("preflight_blocked_jsonl_not_found")
        return []
    return list(iter_jsonl(path))


def load_recovery_handoff_candidate_roots(
    handoff_json: Path | None,
    failed: list[str],
) -> list[str]:
    if handoff_json is None:
        return []
    if not handoff_json.is_file():
        failed.append("raw_reference_recovery_material_handoff_json_not_found")
        return []
    handoff = read_json(handoff_json)
    failed.extend(collect_true_flag_failures("raw_reference_recovery_material_handoff", handoff))
    if handoff.get("schema") != "full_pack_raw_reference_recovery_material_handoff_manifest.v1":
        failed.append("raw_reference_recovery_material_handoff_schema_unexpected")
    if handoff.get("failed_checks") not in ([], None):
        failed.append("raw_reference_recovery_material_handoff_failed_checks_not_empty")
    status = str(handoff.get("handoff_status") or "")
    if status not in {
        "ready_for_reference_corpus_intake",
        "partial_ready_for_reference_corpus_intake",
        "blocked_waiting_for_recovery_reference_material",
        "no_raw_reference_recovery_material_handoff_needed",
    }:
        failed.append("raw_reference_recovery_material_handoff_status_unexpected")
    manifest_roots = stable_unique(str(item) for item in handoff.get("candidate_roots") or [] if str(item or ""))
    ready_count = int(handoff.get("ready_handoff_count") or 0)
    if ready_count > 0 and not manifest_roots:
        failed.append("raw_reference_recovery_material_handoff_ready_without_candidate_roots")
    ready_path = resolved_output_path(handoff, "ready_jsonl", base_dir=handoff_json.parent)
    if ready_count > 0 and (ready_path is None or not ready_path.is_file()):
        failed.append("raw_reference_recovery_material_handoff_ready_jsonl_not_found")
        return []
    if ready_count == 0:
        return []

    ready_rows = list(iter_jsonl(ready_path)) if ready_path is not None else []
    if len(ready_rows) != ready_count:
        failed.append("raw_reference_recovery_material_handoff_ready_count_mismatch")
    row_roots: list[str] = []
    for index, row in enumerate(ready_rows, start=1):
        prefix = f"raw_reference_recovery_material_handoff_ready_row_{index:03d}"
        failed.extend(collect_true_flag_failures(prefix, row))
        if row.get("schema") != "full_pack_raw_reference_recovery_material_handoff_item.v1":
            failed.append(f"{prefix}_schema_unexpected")
        if row.get("handoff_status") != "ready_for_reference_corpus_intake":
            failed.append(f"{prefix}_status_not_ready")
        root = str(row.get("candidate_root") or "")
        if not root:
            failed.append(f"{prefix}_candidate_root_missing")
        else:
            row_roots.append(root)
    row_roots = stable_unique(row_roots)
    if manifest_roots and row_roots and manifest_roots != row_roots:
        failed.append("raw_reference_recovery_material_handoff_candidate_roots_mismatch")
    return row_roots or manifest_roots


def blocker_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        blockers = row.get("blockers")
        if not isinstance(blockers, list):
            continue
        for blocker in blockers:
            text = str(blocker or "")
            if text:
                counts[text] += 1
    return dict(sorted(counts.items()))


def stem_counts(rows: Sequence[dict[str, Any]], fallback: Any) -> dict[str, int]:
    counts: Counter[str] = Counter(str(row.get("stem_type") or "unknown") for row in rows)
    counts.pop("unknown", None)
    if counts:
        return dict(sorted(counts.items()))
    if isinstance(fallback, dict):
        return dict(sorted((str(key), int(value or 0)) for key, value in fallback.items() if int(value or 0) > 0))
    return {}


def build_next_command_args(
    *,
    machine_gate_json: Path,
    repair_queue_json: Path,
    full_pack_loop_plan_json: Path,
    reference_gap_plan_json: Path,
    existing_audio_training_examples_jsonl: Path,
    candidate_roots: Sequence[str],
    raw_reference_recovery_material_candidate_roots: Sequence[str] = (),
    next_output_dir: Path,
    min_unique_tracks: int,
    training_output_root: str,
) -> list[str]:
    args = [
        "--machine-gate-json",
        str(machine_gate_json),
        "--repair-queue-json",
        str(repair_queue_json),
        "--full-pack-loop-plan-json",
        str(full_pack_loop_plan_json),
        "--reference-gap-plan-json",
        str(reference_gap_plan_json),
        "--existing-audio-training-examples-jsonl",
        str(existing_audio_training_examples_jsonl),
    ]
    for candidate_root in candidate_roots:
        args.extend(["--candidate-root", candidate_root])
    for candidate_root in raw_reference_recovery_material_candidate_roots:
        args.extend(["--raw-reference-recovery-material-candidate-root", candidate_root])
    args.extend(
        [
            "--output-dir",
            str(next_output_dir),
            "--min-unique-tracks",
            str(int(min_unique_tracks)),
            "--training-output-root",
            str(training_output_root),
        ]
    )
    return args


def build_summary(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Full Pack ARIS Next Run Plan",
            "",
            f"- next_run_status: `{manifest['next_run_status']}`",
            f"- ready_reference_bundle_count: `{manifest['ready_reference_bundle_count']}`",
            f"- blocked_reference_bundle_count: `{manifest['blocked_reference_bundle_count']}`",
            f"- next_run_candidate_root_count: `{manifest['next_run_candidate_root_count']}`",
            f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
            f"- release_ready: `{str(manifest['release_ready']).lower()}`",
            "",
        ]
    )


def build_full_pack_aris_next_run_plan(
    *,
    preflight_json: Path,
    machine_gate_json: Path,
    repair_queue_json: Path,
    full_pack_loop_plan_json: Path,
    reference_gap_plan_json: Path,
    existing_audio_training_examples_jsonl: Path,
    output_dir: Path,
    next_output_dir: Path,
    raw_reference_recovery_material_handoff_json: Path | None = None,
    min_unique_tracks: int = 3,
    training_output_root: str = DEFAULT_TRAINING_OUTPUT_ROOT,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", preflight_json=str(preflight_json), next_output_dir=str(next_output_dir))

    preflight = read_json(preflight_json)
    failed = collect_true_flag_failures("preflight", preflight)
    if preflight.get("schema") != "full_pack_reference_bundle_preflight_manifest.v1":
        failed.append("preflight_schema_unexpected")
    preflight_failed = [str(item) for item in preflight.get("failed_checks") or []]
    nonfatal_preflight_failed = bool(preflight_failed) and all(
        item in NONFATAL_REFERENCE_PREFLIGHT_FAILED_CHECKS for item in preflight_failed
    )
    if preflight_failed and not nonfatal_preflight_failed:
        failed.append("preflight_failed_checks_not_empty")
    status = str(preflight.get("preflight_status") or "")
    if status not in {*READY_STATUSES, BLOCKED_WAITING_STATUS, "blocked"}:
        failed.append("preflight_status_unexpected")
    reference_bundle_candidate_roots = [str(item) for item in preflight.get("candidate_roots") or [] if str(item or "")]
    ready_bundle_count = int(preflight.get("ready_bundle_count") or 0)
    if status in READY_STATUSES and ready_bundle_count > 0 and not reference_bundle_candidate_roots:
        failed.append("preflight_ready_without_candidate_roots")
    if ready_bundle_count > 0 and not reference_bundle_candidate_roots:
        failed.append("preflight_ready_count_without_candidate_roots")

    blocked_rows = load_blocked_rows(preflight_json, preflight, failed)
    raw_reference_recovery_material_candidate_roots = load_recovery_handoff_candidate_roots(
        raw_reference_recovery_material_handoff_json,
        failed,
    )
    failed = stable_unique(failed)
    candidate_roots = [] if failed else stable_unique(
        [*reference_bundle_candidate_roots, *raw_reference_recovery_material_candidate_roots]
    )
    if failed:
        next_status = "blocked"
    elif candidate_roots:
        next_status = "ready_for_next_metadata_loop"
    else:
        next_status = BLOCKED_WAITING_STATUS

    next_args = []
    if next_status == "ready_for_next_metadata_loop":
        next_args = build_next_command_args(
            machine_gate_json=machine_gate_json,
            repair_queue_json=repair_queue_json,
            full_pack_loop_plan_json=full_pack_loop_plan_json,
            reference_gap_plan_json=reference_gap_plan_json,
            existing_audio_training_examples_jsonl=existing_audio_training_examples_jsonl,
            candidate_roots=candidate_roots,
            raw_reference_recovery_material_candidate_roots=raw_reference_recovery_material_candidate_roots,
            next_output_dir=next_output_dir,
            min_unique_tracks=min_unique_tracks,
            training_output_root=training_output_root,
        )
    arg_rows = [
        {"schema": "full_pack_aris_next_run_arg.v1", "index": index, "arg": arg}
        for index, arg in enumerate(next_args)
    ]
    blocked_by_stem = stem_counts(blocked_rows, preflight.get("blocked_by_stem"))
    manifest = {
        "schema": "full_pack_aris_next_run_plan_manifest.v1",
        "manifest_type": "full_pack_aris_next_run_plan",
        "next_run_status": next_status,
        "failed_checks": failed,
        "preflight_failed_checks": preflight_failed,
        "preflight_failed_checks_treated_as_nonfatal": nonfatal_preflight_failed,
        "source_preflight_json": str(preflight_json),
        "preflight_status": status,
        "reference_bundle_candidate_roots": [] if failed else stable_unique(reference_bundle_candidate_roots),
        "raw_reference_recovery_material_handoff_json": str(raw_reference_recovery_material_handoff_json or ""),
        "raw_reference_recovery_material_candidate_roots": []
        if failed
        else stable_unique(raw_reference_recovery_material_candidate_roots),
        "candidate_roots": candidate_roots,
        "ready_reference_bundle_count": 0 if failed else int(preflight.get("ready_bundle_count") or 0),
        "ready_raw_reference_recovery_material_count": 0
        if failed
        else len(raw_reference_recovery_material_candidate_roots),
        "blocked_reference_bundle_count": len(blocked_rows) if blocked_rows else int(preflight.get("blocked_bundle_count") or 0),
        "next_run_candidate_root_count": len(candidate_roots),
        "blocked_by_stem": blocked_by_stem,
        "blocked_reason_counts": blocker_counts(blocked_rows),
        "next_tool": RUNNER_ENTRYPOINT,
        "next_command": ["python", RUNNER_ENTRYPOINT, *next_args] if next_args else [],
        "next_command_args": next_args,
        "next_docker_service": "ml-worker",
        "inputs": {
            "machine_gate_json": str(machine_gate_json),
            "repair_queue_json": str(repair_queue_json),
            "full_pack_loop_plan_json": str(full_pack_loop_plan_json),
            "reference_gap_plan_json": str(reference_gap_plan_json),
            "existing_audio_training_examples_jsonl": str(existing_audio_training_examples_jsonl),
            "raw_reference_recovery_material_handoff_json": str(raw_reference_recovery_material_handoff_json or ""),
            "next_output_dir": str(next_output_dir),
            "min_unique_tracks": int(min_unique_tracks),
            "training_output_root": str(training_output_root),
        },
        "outputs": {
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "next_command_args_jsonl": str(output_dir / ARGS_NAME),
            "blocked_jsonl": str(output_dir / BLOCKED_NAME),
            "progress_jsonl": str(progress_path),
            "summary_md": str(output_dir / SUMMARY_NAME),
        },
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "full_pack_aris_next_run_plan_not_for_listen_pack",
        "release_status": "planning_artifact_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "safety": {
            "metadata_only": True,
            "audio_read": False,
            "audio_copied": False,
            "audio_decoded": False,
            "dataset_built": False,
            "training_started": False,
            "not_release_artifact": True,
        },
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / ARGS_NAME, arg_rows)
    write_jsonl(output_dir / BLOCKED_NAME, blocked_rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(build_summary(manifest), encoding="utf-8")
    progress(
        progress_path,
        "finish",
        next_run_status=next_status,
        next_run_candidate_root_count=len(candidate_roots),
        failed_checks=failed,
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the next metadata-only ARIS full-pack loop command from preflight.")
    parser.add_argument("--preflight-json", type=Path, required=True)
    parser.add_argument("--machine-gate-json", type=Path, required=True)
    parser.add_argument("--repair-queue-json", type=Path, required=True)
    parser.add_argument("--full-pack-loop-plan-json", type=Path, required=True)
    parser.add_argument("--reference-gap-plan-json", type=Path, required=True)
    parser.add_argument("--existing-audio-training-examples-jsonl", type=Path, required=True)
    parser.add_argument("--raw-reference-recovery-material-handoff-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--next-output-dir", type=Path, required=True)
    parser.add_argument("--min-unique-tracks", type=int, default=3)
    parser.add_argument("--training-output-root", default=DEFAULT_TRAINING_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_full_pack_aris_next_run_plan.py",
        entrypoint_path="/app/tools/build_full_pack_aris_next_run_plan.py",
    )
    args = parse_args(argv)
    manifest = build_full_pack_aris_next_run_plan(
        preflight_json=args.preflight_json,
        machine_gate_json=args.machine_gate_json,
        repair_queue_json=args.repair_queue_json,
        full_pack_loop_plan_json=args.full_pack_loop_plan_json,
        reference_gap_plan_json=args.reference_gap_plan_json,
        existing_audio_training_examples_jsonl=args.existing_audio_training_examples_jsonl,
        raw_reference_recovery_material_handoff_json=args.raw_reference_recovery_material_handoff_json,
        output_dir=args.output_dir,
        next_output_dir=args.next_output_dir,
        min_unique_tracks=int(args.min_unique_tracks),
        training_output_root=str(args.training_output_root),
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "next_run_status": manifest["next_run_status"],
                "next_run_candidate_root_count": manifest["next_run_candidate_root_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
