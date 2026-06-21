#!/usr/bin/env python3
"""Evaluate accepted minus replacement candidates against a full short pack."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_short_listen_replacement_pack import build_short_listen_replacement_pack  # noqa: E402
from tools.build_short_minus_quality_audit import build_short_minus_quality_audit  # noqa: E402
from tools.docker_runtime import require_docker_runtime  # noqa: E402


START_TIME = time.monotonic()
MANIFEST_NAME = "sweep.json"
ROWS_NAME = "rows.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"

BLOCKED_TRUE_FLAGS = (
    "release_ready",
    "can_publish_as_result",
    "deliverable",
    "thresholds_relaxed",
    "runtime_gates_changed",
    "model_registry_edited",
    "models_changed",
    "training_started",
    "fine_tune_started",
    "separator_training_started",
    "cleanup_routes_added",
)

SAFETY = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "separator_training_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "thresholds_relaxed": False,
    "cleanup_routes_added": False,
    "release_ready": False,
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield payload


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
        "expected_acceleration_mode": "external_only_cpu_dsp_scoring",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "minus_replacement_sweep_reuses_existing_candidates_and_judges",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "minus_replacement_sweep_no_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_unique(values: Iterable[Any]) -> list[str]:
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
    safety = payload.get("safety")
    if isinstance(safety, dict):
        for key in BLOCKED_TRUE_FLAGS:
            if safety.get(key) is True:
                failed.append(f"{prefix}_safety_{key}_true")
    contract = payload.get("contract")
    if isinstance(contract, dict):
        for key in BLOCKED_TRUE_FLAGS:
            if contract.get(key) is True:
                failed.append(f"{prefix}_contract_{key}_true")
    return failed


def candidate_id(row: dict[str, Any]) -> str:
    return str(row.get("candidate_id") or row.get("source_candidate_id") or "")


def load_judge_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("candidate_id") or ""): row for row in iter_jsonl(path) if row.get("candidate_id")}


def minus_accepted(candidate_id_value: str, result_rows: dict[str, dict[str, Any]], window_rows: dict[str, dict[str, Any]]) -> bool:
    result = result_rows.get(candidate_id_value, {})
    window = window_rows.get(candidate_id_value, {})
    return (
        result.get("decision") == "accept_candidate"
        and result.get("minus_decision") == "accept_candidate"
        and window.get("decision") == "accept_candidate"
        and window.get("minus_decision") == "accept_candidate"
    )


def selected_jsonl_path(pack_path: Path, payload: dict[str, Any]) -> Path:
    path = Path(str(payload.get("selected_jsonl") or ""))
    if path.is_absolute():
        return path
    return pack_path.parent / path


def accepted_candidate_rows(pack_path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows_path = selected_jsonl_path(pack_path, payload)
    if not rows_path.exists():
        return []
    return [
        row
        for row in iter_jsonl(rows_path)
        if row.get("route_intent") == "minus_repair"
        and row.get("stem_type") == "minus"
        and row.get("delivery_status") == "accepted"
        and row.get("delivery_allowed") is True
        and row.get("release_ready") is not True
        and row.get("can_publish_as_result") is not True
    ]


def single_row_jsonl(path: Path, row: dict[str, Any]) -> None:
    write_jsonl(path, [row])


def replacement_selection_manifest(
    *,
    candidate_row: dict[str, Any],
    result_row: dict[str, Any],
    window_row: dict[str, Any],
    output_dir: Path,
) -> Path:
    selection_dir = output_dir / "sel"
    result_jsonl = selection_dir / "rj.jsonl"
    window_jsonl = selection_dir / "wj.jsonl"
    single_row_jsonl(result_jsonl, result_row)
    single_row_jsonl(window_jsonl, window_row)
    replacement_row = dict(candidate_row)
    replacement_row["source_row"] = {
        **(candidate_row.get("source_row") if isinstance(candidate_row.get("source_row"), dict) else {}),
        "source_audio_path": str(candidate_row.get("source_audio_path") or ""),
        "track_id": str(candidate_row.get("track_id") or ""),
        "source_track_id": str(candidate_row.get("source_track_id") or ""),
        "result_judge_jsonl": str(result_jsonl),
        "windowed_judge_jsonl": str(window_jsonl),
    }
    replacement_row["result_judge_jsonl"] = str(result_jsonl)
    replacement_row["windowed_judge_jsonl"] = str(window_jsonl)
    manifest_path = selection_dir / "pack.json"
    write_json(
        manifest_path,
        {
            "schema": "short_listen_repair_selection_pack_manifest.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_status": "listen_preview_only",
            "release_ready": False,
            "can_publish_as_result": False,
            "failed_checks": [],
            "selected_count": 1,
            "quality_warning_count": 0,
            "listen_audio_files": [replacement_row],
            "selected_jsonl": str(selection_dir / "selected.jsonl"),
        },
    )
    write_jsonl(selection_dir / "selected.jsonl", [replacement_row])
    return manifest_path


def metric_int(payload: dict[str, Any], key: str) -> int:
    try:
        return int(payload.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def candidate_result_row(
    *,
    index: int,
    candidate_row: dict[str, Any],
    result_row: dict[str, Any],
    window_row: dict[str, Any],
    baseline_warning_count: int,
    baseline_blocked_count: int,
    base_pack_json: Path,
    output_dir: Path,
    progress_path: Path,
) -> dict[str, Any]:
    alias = f"c{index:03d}"
    candidate_dir = output_dir / alias
    progress(progress_path, "candidate_start", candidate_alias=alias, candidate_id=candidate_id(candidate_row))
    selection_manifest = replacement_selection_manifest(
        candidate_row=candidate_row,
        result_row=result_row,
        window_row=window_row,
        output_dir=candidate_dir,
    )
    replacement_manifest = build_short_listen_replacement_pack(
        base_pack_json=base_pack_json,
        replacement_pack_jsons=[selection_manifest],
        output_dir=candidate_dir / "rep",
        pack_id=alias,
        command_args=["internal_minus_replacement_sweep"],
    )
    replacement_pack_json = candidate_dir / "rep" / "pack.json"
    if replacement_manifest.get("delivery_status") != "accepted" or replacement_manifest.get("failed_checks"):
        row = {
            "schema": "short_minus_replacement_sweep_row.v1",
            "candidate_alias": alias,
            "candidate_id": candidate_id(candidate_row),
            "evaluation_status": "replacement_blocked",
            "replacement_pack_json": str(replacement_pack_json),
            "audit_json": "",
            "replacement_count": metric_int(replacement_manifest, "replacement_count"),
            "warning_count": None,
            "blocked_minus_count": None,
            "warning_delta": None,
            "blocked_delta": None,
            "failed_checks": replacement_manifest.get("failed_checks") or [],
            "release_ready": False,
            "can_publish_as_result": False,
        }
        progress(progress_path, "candidate_finish", candidate_alias=alias, evaluation_status=row["evaluation_status"])
        return row
    audit = build_short_minus_quality_audit(
        pack_manifest_json=replacement_pack_json,
        output_dir=candidate_dir / "aud",
        command_args=["internal_minus_replacement_sweep"],
    )
    warning_count = metric_int(audit, "warning_count")
    blocked_count = metric_int(audit, "blocked_minus_count")
    row = {
        "schema": "short_minus_replacement_sweep_row.v1",
        "candidate_alias": alias,
        "candidate_id": candidate_id(candidate_row),
        "source_audio_path": str(candidate_row.get("source_audio_path") or ""),
        "listen_audio_path": str(candidate_row.get("listen_audio_path") or ""),
        "track_id": str(candidate_row.get("track_id") or candidate_row.get("source_track_id") or ""),
        "evaluation_status": "evaluated",
        "replacement_pack_json": str(replacement_pack_json),
        "audit_json": str(candidate_dir / "aud" / "audit.json"),
        "replacement_count": metric_int(replacement_manifest, "replacement_count"),
        "audit_status": audit.get("audit_status"),
        "accepted_minus_count": metric_int(audit, "accepted_minus_count"),
        "warning_count": warning_count,
        "blocked_minus_count": blocked_count,
        "evidence_missing_count": metric_int(audit, "evidence_missing_count"),
        "warning_delta": warning_count - baseline_warning_count,
        "blocked_delta": blocked_count - baseline_blocked_count,
        "failed_checks": audit.get("failed_checks") or [],
        "minus_judge": {
            "minus_score": window_row.get("minus_score"),
            "vocal_leakage_in_minus_score": window_row.get("vocal_leakage_in_minus_score"),
            "mixture_consistency_score": window_row.get("mixture_consistency_score"),
            "reasons": window_row.get("reasons") or [],
        },
        "release_ready": False,
        "can_publish_as_result": False,
    }
    progress(
        progress_path,
        "candidate_finish",
        candidate_alias=alias,
        evaluation_status=row["evaluation_status"],
        warning_count=warning_count,
        blocked_minus_count=blocked_count,
    )
    return row


def skipped_row(index: int, candidate_row: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "schema": "short_minus_replacement_sweep_row.v1",
        "candidate_alias": f"c{index:03d}",
        "candidate_id": candidate_id(candidate_row),
        "source_audio_path": str(candidate_row.get("source_audio_path") or ""),
        "listen_audio_path": str(candidate_row.get("listen_audio_path") or ""),
        "track_id": str(candidate_row.get("track_id") or candidate_row.get("source_track_id") or ""),
        "evaluation_status": "skipped",
        "skip_reason": reason,
        "replacement_pack_json": "",
        "audit_json": "",
        "release_ready": False,
        "can_publish_as_result": False,
    }


def best_candidate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [
        row
        for row in rows
        if row.get("evaluation_status") == "evaluated"
        and not row.get("failed_checks")
        and isinstance(row.get("warning_delta"), int)
        and isinstance(row.get("blocked_delta"), int)
    ]
    if not evaluated:
        return {}
    return sorted(
        evaluated,
        key=lambda row: (
            int(row.get("blocked_delta") or 0),
            int(row.get("warning_delta") or 0),
            int(row.get("blocked_minus_count") or 0),
            int(row.get("warning_count") or 0),
            -float((row.get("minus_judge") or {}).get("minus_score") or 0.0),
            str(row.get("candidate_id") or ""),
        ),
    )[0]


def summary_text(manifest: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# short minus replacement sweep",
        "",
        f"- sweep_status: `{manifest['sweep_status']}`",
        f"- evaluated_candidate_count: `{manifest['evaluated_candidate_count']}`",
        f"- skipped_candidate_count: `{manifest['skipped_candidate_count']}`",
        f"- release_ready: `{str(manifest['release_ready']).lower()}`",
        "",
        "## Rows",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['candidate_alias']}` `{row.get('candidate_id', '')}` "
            f"status=`{row['evaluation_status']}` warning_delta=`{row.get('warning_delta', '')}` "
            f"blocked_delta=`{row.get('blocked_delta', '')}`"
        )
    if manifest["failed_checks"]:
        lines.extend(["", "## Failed Checks", ""])
        lines.extend(f"- `{check}`" for check in manifest["failed_checks"])
    return "\n".join(lines) + "\n"


def build_short_minus_replacement_sweep(
    *,
    base_pack_json: Path,
    repair_candidate_pack_json: Path,
    minus_result_judge_jsonl: Path,
    minus_windowed_judge_jsonl: Path,
    output_dir: Path,
    baseline_audit_json: Path | None = None,
    max_candidates: int = 0,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", base_pack_json=str(base_pack_json), repair_candidate_pack_json=str(repair_candidate_pack_json))

    failed: list[str] = []
    base_pack = read_json(base_pack_json) if base_pack_json.exists() else {}
    candidate_pack = read_json(repair_candidate_pack_json) if repair_candidate_pack_json.exists() else {}
    baseline_audit = read_json(baseline_audit_json) if baseline_audit_json and baseline_audit_json.exists() else {}
    failed.extend(collect_true_flag_failures("base_pack", base_pack))
    failed.extend(collect_true_flag_failures("repair_candidate_pack", candidate_pack))
    failed.extend(collect_true_flag_failures("baseline_audit", baseline_audit))
    if base_pack.get("schema") != "aggregate_short_listen_pack_manifest.v1":
        failed.append("base_pack_schema_unexpected")
    if base_pack.get("delivery_status") != "accepted" or base_pack.get("delivery_allowed") is not True:
        failed.append("base_pack_not_accepted")
    if candidate_pack.get("schema") != "short_listen_repair_candidate_pack_manifest.v1":
        failed.append("repair_candidate_pack_schema_unexpected")
    if candidate_pack.get("delivery_status") != "accepted" or candidate_pack.get("delivery_allowed") is not True:
        failed.append("repair_candidate_pack_not_accepted")
    if candidate_pack.get("failed_checks") not in ([], None):
        failed.append("repair_candidate_pack_failed_checks_not_empty")
    if not minus_result_judge_jsonl.exists():
        failed.append("minus_result_judge_missing")
    if not minus_windowed_judge_jsonl.exists():
        failed.append("minus_windowed_judge_missing")

    rows: list[dict[str, Any]] = []
    if not failed:
        result_rows = load_judge_rows(minus_result_judge_jsonl)
        window_rows = load_judge_rows(minus_windowed_judge_jsonl)
        candidates = accepted_candidate_rows(repair_candidate_pack_json, candidate_pack)
        if max_candidates > 0:
            candidates = candidates[:max_candidates]
        baseline_warning_count = metric_int(baseline_audit, "warning_count")
        baseline_blocked_count = metric_int(baseline_audit, "blocked_minus_count")
        for index, candidate in enumerate(candidates, start=1):
            cid = candidate_id(candidate)
            result_row = result_rows.get(cid, {})
            window_row = window_rows.get(cid, {})
            if not minus_accepted(cid, result_rows, window_rows):
                rows.append(skipped_row(index, candidate, "candidate_not_accepted_by_minus_judges"))
                continue
            rows.append(
                candidate_result_row(
                    index=index,
                    candidate_row=candidate,
                    result_row=result_row,
                    window_row=window_row,
                    baseline_warning_count=baseline_warning_count,
                    baseline_blocked_count=baseline_blocked_count,
                    base_pack_json=base_pack_json,
                    output_dir=output_dir,
                    progress_path=progress_path,
                )
            )
    evaluated_count = sum(1 for row in rows if row.get("evaluation_status") == "evaluated")
    skipped_count = sum(1 for row in rows if row.get("evaluation_status") == "skipped")
    best = best_candidate(rows)
    improved = bool(best) and int(best.get("blocked_delta") or 0) <= 0 and int(best.get("warning_delta") or 0) < 0
    if failed:
        status = "blocked"
    elif improved:
        status = "candidate_improved"
    elif evaluated_count:
        status = "no_candidate_improved"
    else:
        status = "no_evaluable_candidates"
    manifest = {
        "schema": "short_minus_replacement_sweep.v1",
        "sweep_status": status,
        "base_pack_json": str(base_pack_json),
        "repair_candidate_pack_json": str(repair_candidate_pack_json),
        "baseline_audit_json": str(baseline_audit_json or ""),
        "minus_result_judge_jsonl": str(minus_result_judge_jsonl),
        "minus_windowed_judge_jsonl": str(minus_windowed_judge_jsonl),
        "evaluated_candidate_count": evaluated_count,
        "skipped_candidate_count": skipped_count,
        "best_candidate": best,
        "failed_checks": stable_unique(failed),
        "rows_jsonl": str(output_dir / ROWS_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "release_ready": False,
        "can_publish_as_result": False,
        "delivery_allowed": False,
        "presentation_status": "minus_replacement_sweep_not_listen_result",
        "safety": dict(SAFETY),
        "not_claimed": ["release_ready", "publishable_result", "audio_repaired", "training_started", "semantic_clean_minus"],
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / ROWS_NAME, rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(summary_text(manifest, rows), encoding="utf-8")
    progress(progress_path, "finish", sweep_status=status, evaluated_candidate_count=evaluated_count, skipped_candidate_count=skipped_count)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate minus replacement candidates against a full short listen pack.")
    parser.add_argument("--base-pack-json", type=Path, required=True)
    parser.add_argument("--repair-candidate-pack-json", type=Path, required=True)
    parser.add_argument("--minus-result-judge-jsonl", type=Path, required=True)
    parser.add_argument("--minus-windowed-judge-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-audit-json", type=Path)
    parser.add_argument("--max-candidates", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_short_minus_replacement_sweep.py",
        entrypoint_path="/app/tools/build_short_minus_replacement_sweep.py",
    )
    args = parse_args(argv)
    manifest = build_short_minus_replacement_sweep(
        base_pack_json=args.base_pack_json,
        repair_candidate_pack_json=args.repair_candidate_pack_json,
        minus_result_judge_jsonl=args.minus_result_judge_jsonl,
        minus_windowed_judge_jsonl=args.minus_windowed_judge_jsonl,
        output_dir=args.output_dir,
        baseline_audit_json=args.baseline_audit_json,
        max_candidates=int(args.max_candidates),
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "sweep_status": manifest["sweep_status"],
                "evaluated_candidate_count": manifest["evaluated_candidate_count"],
                "skipped_candidate_count": manifest["skipped_candidate_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["sweep_status"] != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
