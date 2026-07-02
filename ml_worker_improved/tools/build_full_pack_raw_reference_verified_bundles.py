#!/usr/bin/env python3
"""Build verified FL Studio reference bundles from raw worklist rows."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_flstudio_ground_truth_bundle import build_flstudio_ground_truth_bundle  # noqa: E402
from tools.build_full_pack_reference_corpus_intake import stable_unique  # noqa: E402
from tools.docker_runtime import require_docker_runtime  # noqa: E402


START_TIME = time.monotonic()
MANIFEST_NAME = "bundles.json"
ROWS_NAME = "rows.jsonl"
READY_NAME = "ready.jsonl"
BLOCKED_NAME = "blocked.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
REJECTED_MANIFEST_NAME = "gt_rej.json"
ALLOWED_RIGHTS = {"owned", "licensed", "cleared", "project_owned"}
BUILDABLE_WORK_ITEM_STATUS = "needs_verified_ground_truth_manifest"
BLOCKED_MISSING_RAW_STATUS = "blocked_missing_raw_reference_candidate"
BLOCKED_PROVENANCE_STATUS = "blocked_raw_reference_provenance"
KNOWN_WORK_ITEM_STATUSES = {BUILDABLE_WORK_ITEM_STATUS, BLOCKED_MISSING_RAW_STATUS, BLOCKED_PROVENANCE_STATUS}
BLOCKED_TRUE_FLAGS = (
    "release_ready",
    "can_publish_as_result",
    "deliverable",
    "delivery_allowed",
    "thresholds_relaxed",
    "runtime_gates_changed",
    "model_registry_edited",
    "models_changed",
    "training_started",
    "fine_tune_started",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected object row")
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
        "expected_acceleration_mode": "cpu_audio_decode_reference_bundle_verification",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "reference_bundle_verification_uses_audio_decode_and_manifest_qc_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "verified_bundle_builder_never_starts_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def collect_true_flag_failures(prefix: str, payload: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    for key in BLOCKED_TRUE_FLAGS:
        if payload.get(key) is True:
            failed.append(f"{prefix}_{key}_true")
    nested = payload.get("safety")
    if isinstance(nested, dict):
        for key in BLOCKED_TRUE_FLAGS:
            if nested.get(key) is True:
                failed.append(f"{prefix}_safety_{key}_true")
    return failed


def normalized_stem(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "vocals":
        return "vocal"
    if text == "instrumental":
        return "minus"
    return text or "unknown"


def resolved_items_path(raw_worklist_json: Path, manifest: dict[str, Any]) -> Path | None:
    outputs = manifest.get("outputs")
    text = ""
    if isinstance(outputs, dict):
        text = str(outputs.get("items_jsonl") or "")
    text = text or str(manifest.get("items_jsonl") or "")
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = raw_worklist_json.parent / path
    return path


def load_work_items(raw_worklist_json: Path, manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    items_path = resolved_items_path(raw_worklist_json, manifest)
    if items_path is None or not items_path.is_file():
        return [], ["raw_worklist_items_jsonl_missing"]
    rows = list(iter_jsonl(items_path))
    failed: list[str] = []
    for row in rows:
        if row.get("schema") != "full_pack_raw_reference_bundle_work_item.v1":
            failed.append("raw_work_item_schema_unexpected")
        if row.get("work_item_status") not in KNOWN_WORK_ITEM_STATUSES:
            failed.append("raw_work_item_status_unexpected")
        failed.extend(collect_true_flag_failures("raw_work_item", row))
    expected_count = int(manifest.get("work_item_count") or 0)
    if expected_count != len(rows):
        failed.append("raw_worklist_item_count_mismatch")
    return (rows if not failed else []), stable_unique(failed)


def is_under_allowed_root(path: Path, allowed_roots: Sequence[Path]) -> bool:
    resolved = path.resolve()
    for root in allowed_roots:
        root_resolved = root.resolve()
        try:
            resolved.relative_to(root_resolved)
            return True
        except ValueError:
            continue
    return False


def safe_reset_dir(path: Path, *, output_dir: Path) -> None:
    output_resolved = output_dir.resolve()
    resolved = path.resolve()
    if resolved != output_resolved:
        try:
            resolved.relative_to(output_resolved)
        except ValueError as exc:
            raise ValueError(f"refusing_to_reset_outside_output_dir:{path}") from exc
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_remove_dir(path: Path, *, output_dir: Path) -> None:
    output_resolved = output_dir.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(output_resolved)
    except ValueError as exc:
        raise ValueError(f"refusing_to_remove_outside_output_dir:{path}") from exc
    if path.exists():
        shutil.rmtree(path)


def group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("candidate_directory") or ""),
        str(row.get("source_mix_candidate_path") or ""),
        str(row.get("track_id_hint") or ""),
    )


def grouped_items(rows: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(group_key(row), []).append(row)
    return list(groups.values())


def manifest_stems(payload: dict[str, Any]) -> set[str]:
    stems = payload.get("stems")
    if not isinstance(stems, dict):
        return set()
    return {normalized_stem(stem) for stem in stems if normalized_stem(stem) != "unknown"}


def short_error_detail(exc: Exception, *, limit: int = 240) -> str:
    text = f"{type(exc).__name__}: {exc}".replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def item_path_blockers(group: Sequence[dict[str, Any]], *, allowed_source_roots: Sequence[Path]) -> list[str]:
    blockers: list[str] = []
    first = group[0]
    candidate_dir = Path(str(first.get("candidate_directory") or ""))
    source_mix = Path(str(first.get("source_mix_candidate_path") or ""))
    source_mix_resolution = str(first.get("source_mix_resolution") or "")
    if source_mix_resolution == "pack_manifest_source_audio_path":
        blockers.append("source_mix_from_listen_pack_manifest_requires_clean_reference_bundle")
    if not candidate_dir.is_dir():
        blockers.append("candidate_directory_missing")
    if not source_mix.is_file():
        blockers.append("source_mix_candidate_missing")
    for row in group:
        target = Path(str(row.get("target_stem_candidate_path") or ""))
        stem = normalized_stem(row.get("stem_type"))
        if stem == "unknown":
            blockers.append("stem_type_unknown")
        if not target.is_file():
            blockers.append(f"target_stem_candidate_missing:{stem}")
        if target.exists() and not is_under_allowed_root(target, allowed_source_roots):
            blockers.append("source_not_under_allowed_root")
    for path in (candidate_dir, source_mix):
        if path.exists() and not is_under_allowed_root(path, allowed_source_roots):
            blockers.append("source_not_under_allowed_root")
    return stable_unique(blockers)


def blocked_work_item_row(row: dict[str, Any]) -> dict[str, Any]:
    stem = normalized_stem(row.get("stem_type"))
    blockers = [str(item) for item in row.get("blockers") or [] if str(item or "")]
    if not blockers:
        blockers = [str(row.get("work_item_status") or "raw_work_item_blocked")]
    return {
        "schema": "full_pack_raw_reference_verified_bundle_row.v1",
        "bundle_alias": str(row.get("bundle_alias") or ""),
        "verified_bundle_status": "blocked",
        "blockers": stable_unique(blockers),
        "candidate_root": "",
        "diagnostic_bundle_root": "",
        "ground_truth_manifest_json": "",
        "diagnostic_manifest_json": "",
        "source_work_item_aliases": [str(row.get("bundle_alias") or "")],
        "source_work_item_count": 1,
        "candidate_directory": str(row.get("candidate_directory") or ""),
        "source_mix_candidate_path": str(row.get("source_mix_candidate_path") or ""),
        "source_mix_resolution": str(row.get("source_mix_resolution") or ""),
        "source_mix_manifest_json": str(row.get("source_mix_manifest_json") or ""),
        "requested_stems": [] if stem == "unknown" else [stem],
        "verified_stems": [],
        "full_mix_reconstruction_status": "",
        "failure_detail": "",
        "rights_status": "",
        "bundle_build_attempted": False,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "training_started": False,
    }


def build_one_bundle(
    *,
    group: Sequence[dict[str, Any]],
    output_dir: Path,
    alias: str,
    allowed_source_roots: Sequence[Path],
    rights_status: str,
    require_full_mix_reconstruction: bool,
) -> dict[str, Any]:
    first = group[0]
    requested_stems = sorted({normalized_stem(row.get("stem_type")) for row in group if normalized_stem(row.get("stem_type")) != "unknown"})
    blockers = item_path_blockers(group, allowed_source_roots=allowed_source_roots)
    diag_dir = output_dir / "diag" / alias
    refs_dir = output_dir / "refs" / alias
    source_mix = Path(str(first.get("source_mix_candidate_path") or ""))
    candidate_dir = Path(str(first.get("candidate_directory") or ""))
    manifest: dict[str, Any] | None = None
    failure_detail = ""
    bundle_build_attempted = False

    if not blockers:
        safe_reset_dir(diag_dir, output_dir=output_dir)
        try:
            bundle_build_attempted = True
            manifest = build_flstudio_ground_truth_bundle(
                source_mix_path=source_mix,
                stem_dir=candidate_dir,
                output_dir=diag_dir,
                track_name=str(first.get("track_id_hint") or source_mix.stem),
                rights_status=rights_status,
                bundle_kind="clean",
                master_chain_status="flstudio_reference_export",
                allow_mixed_stem_dir=True,
            )
        except Exception as exc:  # pragma: no cover - exact exception depends on audio backend
            blockers.append(f"bundle_build_failed:{type(exc).__name__}")
            failure_detail = short_error_detail(exc)

    if manifest is not None:
        input_scope = manifest.get("input_scope_contract")
        input_scope_status = input_scope.get("status") if isinstance(input_scope, dict) else ""
        if input_scope_status != "accepted":
            blockers.append("input_scope_not_clean")
        available_stems = manifest_stems(manifest)
        for stem in requested_stems:
            if stem not in available_stems:
                blockers.append(f"verified_bundle_missing_requested_stem:{stem}")
        contract = manifest.get("full_mix_reconstruction_contract")
        contract_status = contract.get("status") if isinstance(contract, dict) else ""
        if require_full_mix_reconstruction and contract_status != "accepted":
            blockers.append("full_mix_reconstruction_not_accepted")

    blockers = stable_unique(blockers)
    if blockers:
        status = "blocked"
        candidate_root = ""
        diagnostic_manifest = ""
        if manifest is not None:
            manifest_path = diag_dir / "ground_truth_manifest.json"
            rejected_manifest_path = diag_dir / REJECTED_MANIFEST_NAME
            if rejected_manifest_path.exists():
                rejected_manifest_path.unlink()
            if manifest_path.exists():
                manifest_path.replace(rejected_manifest_path)
                diagnostic_manifest = str(rejected_manifest_path)
        manifest_json = ""
    else:
        diagnostic_manifest = ""
        safe_remove_dir(refs_dir, output_dir=output_dir)
        refs_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(diag_dir), str(refs_dir))
        status = "ready_for_reference_intake"
        candidate_root = str(refs_dir)
        manifest_json = str(refs_dir / "ground_truth_manifest.json")

    return {
        "schema": "full_pack_raw_reference_verified_bundle_row.v1",
        "bundle_alias": alias,
        "verified_bundle_status": status,
        "blockers": blockers,
        "candidate_root": candidate_root,
        "diagnostic_bundle_root": str(diag_dir) if status != "ready_for_reference_intake" else "",
        "ground_truth_manifest_json": manifest_json,
        "diagnostic_manifest_json": diagnostic_manifest,
        "source_work_item_aliases": [str(row.get("bundle_alias") or "") for row in group],
        "source_work_item_count": len(group),
        "candidate_directory": str(candidate_dir),
        "source_mix_candidate_path": str(source_mix),
        "source_mix_resolution": str(first.get("source_mix_resolution") or ""),
        "source_mix_manifest_json": str(first.get("source_mix_manifest_json") or ""),
        "requested_stems": requested_stems,
        "verified_stems": sorted(manifest_stems(manifest or {})),
        "full_mix_reconstruction_status": (
            str((manifest.get("full_mix_reconstruction_contract") or {}).get("status") or "") if manifest else ""
        ),
        "failure_detail": failure_detail,
        "rights_status": rights_status,
        "bundle_build_attempted": bundle_build_attempted,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "training_started": False,
    }


def build_summary(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Full Pack Raw Reference Verified Bundles",
            "",
            f"- bundle_status: `{manifest['bundle_status']}`",
            f"- verified_bundle_count: `{manifest['verified_bundle_count']}`",
            f"- blocked_bundle_count: `{manifest['blocked_bundle_count']}`",
            f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
            f"- release_ready: `{str(manifest['release_ready']).lower()}`",
            "",
        ]
    )


def status_for(*, failed: Sequence[str], ready_count: int, blocked_count: int) -> str:
    if failed:
        return "blocked"
    if ready_count:
        return "verified_reference_bundles_ready" if blocked_count == 0 else "partial_verified_reference_bundles_ready"
    if blocked_count:
        return "blocked_waiting_for_verified_raw_reference_bundles"
    return "no_raw_reference_bundle_items"


def build_full_pack_raw_reference_verified_bundles(
    *,
    raw_worklist_json: Path,
    output_dir: Path,
    allowed_source_roots: Sequence[Path],
    rights_status: str,
    require_full_mix_reconstruction: bool = True,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(
        progress_path,
        "start",
        raw_worklist_json=str(raw_worklist_json),
        allowed_source_roots=[str(path) for path in allowed_source_roots],
        rights_status=rights_status,
        require_full_mix_reconstruction=bool(require_full_mix_reconstruction),
    )

    raw_manifest = read_json(raw_worklist_json)
    failed = collect_true_flag_failures("raw_worklist", raw_manifest)
    if raw_manifest.get("schema") != "full_pack_raw_reference_bundle_worklist.v1":
        failed.append("raw_worklist_schema_unexpected")
    if raw_manifest.get("worklist_status") not in ("raw_reference_bundle_worklist_ready", "no_raw_reference_bundle_work_items"):
        failed.append("raw_worklist_status_unexpected")
    if raw_manifest.get("failed_checks") not in ([], None):
        failed.append("raw_worklist_failed_checks_not_empty")
    if rights_status not in ALLOWED_RIGHTS:
        failed.append("rights_status_not_trainable")
    if not allowed_source_roots:
        failed.append("allowed_source_roots_missing")
    rows, row_failed = load_work_items(raw_worklist_json, raw_manifest)
    failed.extend(row_failed)
    failed = stable_unique(failed)

    bundle_rows: list[dict[str, Any]] = []
    buildable_rows: list[dict[str, Any]] = []
    blocked_source_rows: list[dict[str, Any]] = []
    if not failed:
        buildable_rows = [row for row in rows if row.get("work_item_status") == BUILDABLE_WORK_ITEM_STATUS]
        blocked_source_rows = [row for row in rows if row.get("work_item_status") != BUILDABLE_WORK_ITEM_STATUS]
        for row in blocked_source_rows:
            bundle_rows.append(blocked_work_item_row(row))
        for index, group in enumerate(grouped_items(buildable_rows), start=1):
            alias = f"b{index:03d}"
            progress(progress_path, "bundle_start", bundle_alias=alias, source_work_item_count=len(group))
            row = build_one_bundle(
                group=group,
                output_dir=output_dir,
                alias=alias,
                allowed_source_roots=allowed_source_roots,
                rights_status=rights_status,
                require_full_mix_reconstruction=bool(require_full_mix_reconstruction),
            )
            bundle_rows.append(row)
            progress(
                progress_path,
                "bundle_finish",
                bundle_alias=alias,
                verified_bundle_status=row["verified_bundle_status"],
                blockers=row["blockers"],
            )

    ready_rows = [row for row in bundle_rows if row["verified_bundle_status"] == "ready_for_reference_intake"]
    blocked_rows = [row for row in bundle_rows if row["verified_bundle_status"] != "ready_for_reference_intake"]
    status = status_for(failed=failed, ready_count=len(ready_rows), blocked_count=len(blocked_rows))
    ready_by_stem = Counter(stem for row in ready_rows for stem in row["requested_stems"])
    blocked_by_stem = Counter(stem for row in blocked_rows for stem in row["requested_stems"])
    bundle_build_attempted = any(row.get("bundle_build_attempted") is True for row in bundle_rows)
    manifest = {
        "schema": "full_pack_raw_reference_verified_bundles_manifest.v1",
        "manifest_type": "full_pack_raw_reference_verified_bundles",
        "bundle_status": status,
        "failed_checks": failed,
        "raw_worklist_json": str(raw_worklist_json),
        "allowed_source_roots": [str(path) for path in allowed_source_roots],
        "rights_status": rights_status,
        "require_full_mix_reconstruction": bool(require_full_mix_reconstruction),
        "source_work_item_count": len(rows) if not failed else 0,
        "buildable_work_item_count": len(buildable_rows) if not failed else 0,
        "blocked_source_work_item_count": len(blocked_source_rows) if not failed else 0,
        "bundle_row_count": len(bundle_rows),
        "verified_bundle_count": len(ready_rows),
        "blocked_bundle_count": len(blocked_rows),
        "ready_by_stem": dict(sorted(ready_by_stem.items())),
        "blocked_by_stem": dict(sorted(blocked_by_stem.items())),
        "candidate_roots": [row["candidate_root"] for row in ready_rows],
        "outputs": {
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "rows_jsonl": str(output_dir / ROWS_NAME),
            "ready_jsonl": str(output_dir / READY_NAME),
            "blocked_jsonl": str(output_dir / BLOCKED_NAME),
            "progress_jsonl": str(progress_path),
            "summary_md": str(output_dir / SUMMARY_NAME),
        },
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "raw_reference_verified_bundles_not_for_listen_pack",
        "release_status": "training_reference_bundle_candidates_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "safety": {
            "metadata_only": not bundle_build_attempted,
            "audio_read": bundle_build_attempted,
            "audio_copied": bool(ready_rows),
            "audio_decoded": bundle_build_attempted,
            "dataset_built": False,
            "training_started": False,
            "ground_truth_manifest_written": bool(ready_rows),
            "not_release_artifact": True,
            "accepted_rows_not_imported": True,
        },
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / ROWS_NAME, bundle_rows)
    write_jsonl(output_dir / READY_NAME, ready_rows)
    write_jsonl(output_dir / BLOCKED_NAME, blocked_rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(build_summary(manifest), encoding="utf-8")
    progress(
        progress_path,
        "finish",
        bundle_status=status,
        verified_bundle_count=len(ready_rows),
        blocked_bundle_count=len(blocked_rows),
        failed_checks=failed,
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build verified reference bundles from a raw full-pack worklist.")
    parser.add_argument("--raw-worklist-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allowed-source-root", type=Path, action="append", required=True)
    parser.add_argument("--rights-status", choices=sorted(ALLOWED_RIGHTS), required=True)
    parser.add_argument(
        "--require-full-mix-reconstruction",
        dest="require_full_mix_reconstruction",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--allow-diagnostic-rejected-full-mix-reconstruction",
        dest="require_full_mix_reconstruction",
        action="store_false",
        help="Diagnostic-only escape hatch; produced bundles must still pass downstream intake before training.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_full_pack_raw_reference_verified_bundles.py",
        entrypoint_path="/app/tools/build_full_pack_raw_reference_verified_bundles.py",
    )
    args = parse_args(argv)
    manifest = build_full_pack_raw_reference_verified_bundles(
        raw_worklist_json=args.raw_worklist_json,
        output_dir=args.output_dir,
        allowed_source_roots=args.allowed_source_root,
        rights_status=args.rights_status,
        require_full_mix_reconstruction=bool(args.require_full_mix_reconstruction),
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "bundle_status": manifest["bundle_status"],
                "verified_bundle_count": manifest["verified_bundle_count"],
                "blocked_bundle_count": manifest["blocked_bundle_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
