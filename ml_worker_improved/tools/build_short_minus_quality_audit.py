#!/usr/bin/env python3
"""Build a short-name no-reference minus quality audit for listen packs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

TOOLS_ROOT = Path(__file__).resolve().parent
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.run_no_reference_result_judge import as_list, judge_candidate_row, resolve_path  # noqa: E402
from tools.run_no_reference_windowed_judge import judge_candidate_rows_windowed  # noqa: E402
from tools.docker_runtime import require_docker_runtime  # noqa: E402


START_TIME = time.monotonic()
MANIFEST_NAME = "audit.json"
ROWS_NAME = "rows.jsonl"
RESULT_ROWS_NAME = "rj.jsonl"
WINDOW_ROWS_NAME = "wj.jsonl"
CACHE_NAME = "cache.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"

RESIDUAL_VOCAL_WATCH_SCORE = 0.15
DETAIL_MARGIN_WATCH = 0.03

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
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_dsp_scoring",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "short_minus_quality_audit_reads_existing_audio_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "short_minus_quality_audit_no_training",
        **row,
    }
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


def pack_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = payload.get("items")
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    listen_rows = payload.get("listen_audio_files")
    if isinstance(listen_rows, list):
        return [
            {**row, "audio_path": str(row.get("artifact_relative_path") or row.get("audio_path") or "")}
            for row in listen_rows
            if isinstance(row, dict)
        ]
    return []


def resolve_artifact_path(raw: Any, *, base: Path) -> Path:
    path = Path(str(raw or ""))
    if not str(path):
        return path
    if path.is_absolute():
        return resolve_path(path)
    return base / path


def source_dicts(row: dict[str, Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    seen: set[int] = set()

    def visit(current: dict[str, Any]) -> None:
        marker = id(current)
        if marker in seen:
            return
        seen.add(marker)
        values.append(current)
        for key in ("source_row", "source_phase_repair_manifest"):
            nested = current.get(key)
            if isinstance(nested, dict):
                visit(nested)

    visit(row)
    return values


def resolve_json_path(raw: Any, *, base: Path) -> Path:
    path = Path(str(raw or ""))
    if not str(path):
        return path
    if path.is_absolute():
        return resolve_path(path)
    return base / path


def read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        return read_json(path)
    except Exception:
        return {}


def existing_audio_path(raw: Any, *, base: Path) -> str:
    path = resolve_artifact_path(raw, base=base)
    return str(path) if str(path) and path.exists() and path.is_file() else ""


def mix_path_from_manifest(manifest_path: Path) -> str:
    payload = read_optional_json(manifest_path)
    if not payload:
        return ""
    stem_manifest = payload.get("stem_manifest_json")
    if stem_manifest:
        nested = resolve_json_path(stem_manifest, base=manifest_path.parent)
        text = mix_path_from_manifest(nested)
        if text:
            return text
    for key in ("mix_path", "source_mix_path", "input_audio_path", "raw_audio_path", "source_audio_path", "audio_path"):
        text = existing_audio_path(payload.get(key), base=manifest_path.parent)
        if text:
            return text
    return ""


def mix_path_for_row(row: dict[str, Any], *, pack_path: Path) -> str:
    for source in source_dicts(row):
        for key in ("mix_path", "source_mix_path", "input_audio_path", "raw_audio_path"):
            text = existing_audio_path(source.get(key), base=pack_path.parent)
            if text:
                return text
    for source in source_dicts(row):
        for key in ("source_manifest", "upstream_source_manifest", "source_switch_manifest_json"):
            raw = source.get(key)
            if not raw:
                continue
            text = mix_path_from_manifest(resolve_json_path(raw, base=pack_path.parent))
            if text:
                return text
    return ""


def group_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("source_alias") or ""), str(row.get("track_alias") or ""))


def sibling_vocal_row(minus_row: dict[str, Any], items: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    key = group_key(minus_row)
    for row in items:
        if group_key(row) == key and str(row.get("stem_type") or "").lower() in {"vocal", "vocals"}:
            return row
    return None


def candidate_row_for_minus(minus_row: dict[str, Any], items: Sequence[dict[str, Any]], *, pack_path: Path) -> tuple[dict[str, Any] | None, str]:
    vocal_row = sibling_vocal_row(minus_row, items)
    mix_path = mix_path_for_row(minus_row, pack_path=pack_path)
    vocal_path = ""
    if vocal_row is not None:
        mix_path = mix_path or mix_path_for_row(vocal_row, pack_path=pack_path)
        vocal_path = existing_audio_path(vocal_row.get("audio_path") or vocal_row.get("source_audio_path"), base=pack_path.parent)
    if not vocal_path:
        for source in source_dicts(minus_row):
            vocal_path = existing_audio_path(source.get("vocal_path") or source.get("selected_vocal_path"), base=pack_path.parent)
            if vocal_path:
                break
    minus_path = existing_audio_path(minus_row.get("audio_path") or minus_row.get("source_audio_path"), base=pack_path.parent)
    if not mix_path or not vocal_path or not minus_path:
        return None, "minus_pair_evidence_missing"
    alias = str(minus_row.get("source_alias") or "src")
    track = str(minus_row.get("track_alias") or "trk")
    candidate_id = str(minus_row.get("source_candidate_id") or minus_row.get("candidate_id") or "")
    if not candidate_id:
        for source in source_dicts(minus_row):
            candidate_id = str(source.get("candidate_id") or source.get("minus_candidate_id") or "")
            if candidate_id:
                break
    return {
        "track_id": f"{alias}_{track}",
        "candidate_id": candidate_id or f"{alias}_{track}_minus",
        "mix_path": mix_path,
        "vocal_path": vocal_path,
        "instrumental_path": minus_path,
        "minus_path": minus_path,
    }, ""


def source_provenance(minus_row: dict[str, Any]) -> dict[str, Any]:
    source_row = minus_row.get("source_row") if isinstance(minus_row.get("source_row"), dict) else {}
    return {
        "source_manifest": str(minus_row.get("source_manifest") or source_row.get("source_manifest") or ""),
        "source_audio_path": str(minus_row.get("source_audio_path") or source_row.get("source_audio_path") or ""),
        "source_row": source_row,
    }


def source_judge_paths(row: dict[str, Any], *, pack_path: Path) -> tuple[Path | None, Path | None]:
    result_path: Path | None = None
    window_path: Path | None = None
    for source in source_dicts(row):
        if source.get("result_judge_jsonl") and result_path is None:
            result_path = resolve_json_path(source.get("result_judge_jsonl"), base=pack_path.parent)
        if source.get("windowed_judge_jsonl") and window_path is None:
            window_path = resolve_json_path(source.get("windowed_judge_jsonl"), base=pack_path.parent)
    return result_path, window_path


def first_jsonl_row(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    for row in iter_jsonl(path):
        return row
    return None


def gate_blocker_labels(row: dict[str, Any]) -> list[str]:
    gate = row.get("minus_musical_detail_preservation_gate")
    if not isinstance(gate, dict):
        return ["minus_detail_gate_missing"]
    labels: list[str] = []
    for blocker in gate.get("blockers") or []:
        if isinstance(blocker, dict):
            labels.append(str(blocker.get("label") or "minus_detail_gate_blocked"))
        else:
            labels.append(str(blocker))
    if gate.get("status") != "proxy_pass" and not labels:
        labels.append("minus_detail_gate_blocked")
    return stable_unique(labels)


def minus_decision(row: dict[str, Any] | None) -> str:
    if not isinstance(row, dict):
        return ""
    return str(row.get("minus_decision") or row.get("decision") or "")


def minus_blockers(row: dict[str, Any] | None) -> list[str]:
    if not isinstance(row, dict):
        return ["minus_judge_missing"]
    return stable_unique(as_list(row.get("minus_blockers")) + gate_blocker_labels(row))


def metric_float(row: dict[str, Any] | None, key: str) -> float:
    if not isinstance(row, dict):
        return 0.0
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def quality_warning_reasons(row: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if metric_float(row, "vocal_leakage_in_minus_score") >= RESIDUAL_VOCAL_WATCH_SCORE:
        warnings.append("residual_vocal_watch")
    gate = row.get("minus_musical_detail_preservation_gate")
    if isinstance(gate, dict):
        metrics = gate.get("metrics") if isinstance(gate.get("metrics"), dict) else {}
        thresholds = gate.get("thresholds") if isinstance(gate.get("thresholds"), dict) else {}
        for name, threshold_raw in thresholds.items():
            try:
                value = float(metrics.get(name))
                threshold = float(threshold_raw)
            except (TypeError, ValueError):
                continue
            if threshold <= value <= threshold + DETAIL_MARGIN_WATCH:
                warnings.append(f"{name}_margin_watch")
    return stable_unique(warnings)


def audit_decision(result_row: dict[str, Any] | None, window_row: dict[str, Any] | None) -> tuple[str, list[str]]:
    blockers = stable_unique(minus_blockers(result_row) + minus_blockers(window_row))
    if minus_decision(result_row) != "accept_candidate":
        blockers.append("result_minus_not_accepted")
    if minus_decision(window_row) != "accept_candidate":
        blockers.append("windowed_minus_not_accepted")
    blockers = stable_unique(blockers)
    return ("passed" if not blockers else "blocked", blockers)


def audit_row_from_evidence(
    minus_row: dict[str, Any],
    *,
    short_audio_path: str,
    result_row: dict[str, Any] | None,
    window_row: dict[str, Any] | None,
    evidence_source: str,
) -> dict[str, Any]:
    decision, blockers = audit_decision(result_row, window_row)
    warning_input = window_row if isinstance(window_row, dict) else result_row if isinstance(result_row, dict) else {}
    warnings = quality_warning_reasons(warning_input)
    return {
        "schema": "short_minus_quality_audit_row.v1",
        "source_alias": str(minus_row.get("source_alias") or ""),
        "track_alias": str(minus_row.get("track_alias") or ""),
        "stem_type": "minus",
        "short_audio_path": short_audio_path,
        **source_provenance(minus_row),
        "audit_decision": decision,
        "audit_blockers": blockers,
        "quality_warning_reasons": warnings,
        "evidence_source": evidence_source,
        "result_minus_decision": minus_decision(result_row),
        "windowed_minus_decision": minus_decision(window_row),
        "result_minus_score": metric_float(result_row, "minus_score"),
        "windowed_minus_score": metric_float(window_row, "minus_score"),
        "result_vocal_leakage_in_minus_score": metric_float(result_row, "vocal_leakage_in_minus_score"),
        "windowed_vocal_leakage_in_minus_score": metric_float(window_row, "vocal_leakage_in_minus_score"),
        "result_detail_gate_status": str((result_row or {}).get("minus_musical_detail_preservation_gate", {}).get("status") if isinstance((result_row or {}).get("minus_musical_detail_preservation_gate"), dict) else ""),
        "windowed_detail_gate_status": str((window_row or {}).get("minus_musical_detail_preservation_gate", {}).get("status") if isinstance((window_row or {}).get("minus_musical_detail_preservation_gate"), dict) else ""),
        "release_ready": False,
        "can_publish_as_result": False,
    }


def evidence_missing_row(minus_row: dict[str, Any], *, short_audio_path: str, reason: str) -> dict[str, Any]:
    return {
        "schema": "short_minus_quality_audit_row.v1",
        "source_alias": str(minus_row.get("source_alias") or ""),
        "track_alias": str(minus_row.get("track_alias") or ""),
        "stem_type": "minus",
        "short_audio_path": short_audio_path,
        **source_provenance(minus_row),
        "audit_decision": "evidence_missing",
        "audit_blockers": [reason],
        "quality_warning_reasons": [],
        "evidence_source": "",
        "result_minus_decision": "",
        "windowed_minus_decision": "",
        "release_ready": False,
        "can_publish_as_result": False,
    }


def build_summary_md(manifest: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# short minus quality audit",
        "",
        f"- audit_status: `{manifest['audit_status']}`",
        f"- minus_item_count: `{manifest['minus_item_count']}`",
        f"- audited_minus_count: `{manifest['audited_minus_count']}`",
        f"- accepted_minus_count: `{manifest['accepted_minus_count']}`",
        f"- blocked_minus_count: `{manifest['blocked_minus_count']}`",
        f"- evidence_missing_count: `{manifest['evidence_missing_count']}`",
        f"- warning_count: `{manifest['warning_count']}`",
        f"- release_ready: `{str(manifest['release_ready']).lower()}`",
        "",
        "## Rows",
        "",
    ]
    for row in rows:
        lines.append(
            "- "
            f"`{row['short_audio_path']}` "
            f"decision=`{row['audit_decision']}` "
            f"warnings=`{','.join(row.get('quality_warning_reasons') or [])}` "
            f"blockers=`{','.join(row.get('audit_blockers') or [])}`"
        )
    return "\n".join(lines) + "\n"


def build_short_minus_quality_audit(
    *,
    pack_manifest_json: Path,
    output_dir: Path,
    window_seconds: float = 12.0,
    max_windows_per_candidate: int = 5,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    append_jsonl(progress_path, {"event": "start", "pack_manifest_json": str(pack_manifest_json)})
    pack = read_json(pack_manifest_json)
    items = pack_items(pack)
    minus_items = [item for item in items if str(item.get("stem_type") or "").lower() == "minus"]
    rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for index, minus_row in enumerate(minus_items, start=1):
        short_audio_path = str(minus_row.get("audio_path") or "")
        append_jsonl(progress_path, {"event": "minus_audit_item_start", "item_index": index, "short_audio_path": short_audio_path})
        result_path, window_path = source_judge_paths(minus_row, pack_path=pack_manifest_json)
        prior_result = first_jsonl_row(result_path)
        prior_window = first_jsonl_row(window_path)
        if prior_result is not None and prior_window is not None:
            row = audit_row_from_evidence(
                minus_row,
                short_audio_path=short_audio_path,
                result_row=prior_result,
                window_row=prior_window,
                evidence_source="source_judge_jsonl",
            )
            result_rows.append(prior_result)
            window_rows.append(prior_window)
        else:
            candidate, missing_reason = candidate_row_for_minus(minus_row, items, pack_path=pack_manifest_json)
            if candidate is None:
                row = evidence_missing_row(minus_row, short_audio_path=short_audio_path, reason=missing_reason)
            else:
                result = judge_candidate_row(candidate, score_mode="minus")
                windowed = judge_candidate_rows_windowed(
                    [candidate],
                    cache_jsonl=output_dir / CACHE_NAME,
                    window_seconds=float(window_seconds),
                    max_windows_per_candidate=int(max_windows_per_candidate),
                    score_mode="minus",
                )
                window = windowed["rows"][0] if windowed.get("rows") else None
                row = audit_row_from_evidence(
                    minus_row,
                    short_audio_path=short_audio_path,
                    result_row=result,
                    window_row=window,
                    evidence_source="fresh_pack_pair_judge",
                )
                result_rows.append(result)
                if isinstance(window, dict):
                    window_rows.append(window)
        rows.append(row)
        append_jsonl(progress_path, {"event": "minus_audit_item_end", "item_index": index, "audit_decision": row["audit_decision"]})
    failed_checks = stable_unique(
        f"minus_audit_blocked:{row['short_audio_path']}:{blocker}"
        for row in rows
        if row["audit_decision"] != "passed"
        for blocker in row.get("audit_blockers") or ["minus_audit_not_passed"]
    )
    blocked_count = sum(1 for row in rows if row["audit_decision"] == "blocked")
    missing_count = sum(1 for row in rows if row["audit_decision"] == "evidence_missing")
    audited_count = sum(1 for row in rows if row["audit_decision"] != "evidence_missing")
    accepted_count = sum(1 for row in rows if row["audit_decision"] == "passed")
    warning_count = sum(1 for row in rows if row.get("quality_warning_reasons"))
    audit_status = "completed" if not failed_checks else "completed_with_blockers"
    manifest = {
        "schema": "short_minus_quality_audit.v1",
        "audit_status": audit_status,
        "pack_manifest_json": str(pack_manifest_json),
        "minus_item_count": len(minus_items),
        "audited_minus_count": audited_count,
        "accepted_minus_count": accepted_count,
        "blocked_minus_count": blocked_count,
        "evidence_missing_count": missing_count,
        "warning_count": warning_count,
        "failed_checks": failed_checks,
        "rows_jsonl": str(output_dir / ROWS_NAME),
        "result_judge_jsonl": str(output_dir / RESULT_ROWS_NAME),
        "windowed_judge_jsonl": str(output_dir / WINDOW_ROWS_NAME),
        "windowed_cache_jsonl": str(output_dir / CACHE_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "release_ready": False,
        "can_publish_as_result": False,
        "delivery_allowed": False,
        "presentation_status": "minus_quality_audit_not_listen_result",
        "not_claimed": ["release_ready", "publishable_result", "human_listen_verdict", "semantic_clean_minus"],
        "safety": dict(SAFETY),
        "elapsed_sec": round(time.monotonic() - started, 6),
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / ROWS_NAME, rows)
    write_jsonl(output_dir / RESULT_ROWS_NAME, result_rows)
    write_jsonl(output_dir / WINDOW_ROWS_NAME, window_rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(build_summary_md(manifest, rows), encoding="utf-8")
    append_jsonl(progress_path, {"event": "finish", "audit_status": audit_status, "failed_checks": failed_checks})
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a short-name minus quality audit for listen packs.")
    parser.add_argument("--pack-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=12.0)
    parser.add_argument("--max-windows-per-candidate", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_short_minus_quality_audit.py",
        entrypoint_path="/app/tools/build_short_minus_quality_audit.py",
    )
    args = parse_args(argv)
    manifest = build_short_minus_quality_audit(
        pack_manifest_json=args.pack_manifest_json,
        output_dir=args.output_dir,
        window_seconds=args.window_seconds,
        max_windows_per_candidate=args.max_windows_per_candidate,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "audit_status": manifest["audit_status"],
                "minus_item_count": manifest["minus_item_count"],
                "audited_minus_count": manifest["audited_minus_count"],
                "accepted_minus_count": manifest["accepted_minus_count"],
                "blocked_minus_count": manifest["blocked_minus_count"],
                "evidence_missing_count": manifest["evidence_missing_count"],
                "warning_count": manifest["warning_count"],
                "failed_checks": manifest["failed_checks"],
                "release_ready": manifest["release_ready"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not manifest["failed_checks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
