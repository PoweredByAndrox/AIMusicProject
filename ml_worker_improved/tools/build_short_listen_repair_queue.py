#!/usr/bin/env python3
"""Build a repair/mining queue from rejected short listen feedback."""

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

from tools.artifact_name_policy import indexed_alias, validate_artifact_segment  # noqa: E402
from tools.docker_runtime import require_docker_runtime  # noqa: E402


START_TIME = time.monotonic()
MANIFEST_NAME = "queue.json"
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

MINUS_ISSUES = {
    "muffled_minus",
    "residual_vocal_middle",
    "residual_vocal",
    "vocal_bleed",
    "severe_vocal_bleed_in_minus",
    "vocal_bleed_in_minus",
    "minus_detail_loss",
    "phase_stereo_damage",
    "mixture_reconstruction_error",
}
VOCAL_ISSUES = {"bad_vocal_quality", "vocal_artifact", "vocal_missing_detail", "vocal_bleed"}
TRAINING_ISSUES = {
    "insufficient_unique_tracks",
    "train_ready_model_missing",
    "training_job_blocked",
    "training_job_missing",
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(payload)
    return rows


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
        "expected_acceleration_mode": "external_only_cpu_repair_queue_planning",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "short_listen_repair_queue_builds_plan_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "short_listen_repair_queue_no_training",
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
        if value and value not in seen:
            seen.add(value)
            result.append(value)
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
    return failed


def canonical_audio_path(raw_path: Any) -> str:
    return str(raw_path or "").replace("\\", "/").strip()


def check_short_relative_path(audio_path: str) -> list[str]:
    path = Path(str(audio_path or ""))
    if path.is_absolute() or not audio_path:
        return ["queue_audio_path_not_relative"]
    failed: list[str] = []
    for segment in path.parts:
        try:
            validate_artifact_segment(segment)
        except ValueError:
            failed.append("queue_audio_path_not_short_windows_safe")
            break
    return failed


def issue_tag_counts(items: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        for tag in item.get("issue_tags") or []:
            key = str(tag)
            if key:
                counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def route_intent(item: dict[str, Any]) -> str:
    stem_type = str(item.get("stem_type") or "")
    tags = {str(tag) for tag in item.get("issue_tags") or []}
    if stem_type == "vocal":
        return "vocal_candidate_replacement"
    if stem_type.startswith("minus") and (tags.intersection(MINUS_ISSUES) or "minus_gate_blocked" in tags):
        return "minus_repair"
    if tags.intersection(MINUS_ISSUES):
        return "minus_repair"
    if tags.intersection(VOCAL_ISSUES):
        return "vocal_candidate_replacement"
    if tags.intersection(TRAINING_ISSUES):
        return "stem_training_coverage_gap"
    return "stem_quality_review"


def required_actions(intent: str, issue_tags: Sequence[str]) -> list[str]:
    tags = {str(tag) for tag in issue_tags}
    if intent == "minus_repair":
        actions = ["window_rejudge", "candidate_mining", "segment_blend_or_rerender", "listen_pack_rebuild"]
        if tags.intersection({"residual_vocal_middle", "residual_vocal", "vocal_bleed", "severe_vocal_bleed_in_minus"}):
            actions.insert(0, "localized_bleed_diagnostic")
        if "muffled_minus" in tags:
            actions.insert(0, "detail_loss_diagnostic")
        if tags.intersection(TRAINING_ISSUES):
            actions.extend(["verified_reference_corpus_expand", "train_ready_decision_gate"])
        return stable_unique(actions)
    if intent == "vocal_candidate_replacement":
        return ["candidate_family_remine", "full_length_vocal_regenerate", "listen_pack_rebuild"]
    if intent == "stem_training_coverage_gap":
        actions = ["verified_reference_corpus_expand"]
        if "training_job_missing" not in tags:
            actions.append("gpu_teacher_mask_train")
        actions.extend(["holdout_eval_gate", "train_ready_decision_gate", "full_pack_gate_rebuild"])
        return stable_unique(actions)
    return ["stem_quality_rejudge", "listen_pack_rebuild"]


def diagnostic_context(item: dict[str, Any]) -> dict[str, Any]:
    source_row = item.get("source_row") if isinstance(item.get("source_row"), dict) else {}
    judge_evidence = source_row.get("judge_evidence") if isinstance(source_row.get("judge_evidence"), dict) else {}
    windows = judge_evidence.get("applied_windows") if isinstance(judge_evidence.get("applied_windows"), list) else []
    context = {
        "judge_evidence": judge_evidence,
        "target_windows": [window for window in windows if isinstance(window, dict)],
    }
    if isinstance(item.get("minus_audit"), dict):
        context["minus_audit"] = item["minus_audit"]
    if isinstance(item.get("machine_gate"), dict):
        context["machine_gate"] = item["machine_gate"]
    return context


def route_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("route_intent") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def build_row(index: int, item: dict[str, Any]) -> dict[str, Any]:
    audio_path = canonical_audio_path(item.get("audio_path"))
    intent = route_intent(item)
    issue_tags = [str(tag) for tag in item.get("issue_tags") or [] if str(tag)]
    corrective_actions = [str(action) for action in item.get("corrective_action_labels") or [] if str(action)]
    required = stable_unique(required_actions(intent, issue_tags) + corrective_actions)
    return {
        "schema": "short_listen_repair_queue_item.v1",
        "queue_item_id": indexed_alias("q", index),
        "audio_path": audio_path,
        "source_alias": str(item.get("source_alias") or ""),
        "stem_type": str(item.get("stem_type") or ""),
        "track_alias": str(item.get("track_alias") or ""),
        "verdict": str(item.get("verdict") or ""),
        "issue_tags": issue_tags,
        "source_blocker_labels": [str(label) for label in item.get("source_blocker_labels") or [] if str(label)],
        "source_issue_labels": [str(label) for label in item.get("source_issue_labels") or [] if str(label)],
        "corrective_action_labels": corrective_actions,
        "corrective_target_lanes": [str(lane) for lane in item.get("corrective_target_lanes") or [] if str(lane)],
        "note": str(item.get("note") or ""),
        "route_intent": intent,
        "required_actions": required,
        "priority": "high" if item.get("verdict") == "machine_blocked" or "user_reported_unacceptable" in issue_tags else "normal",
        "source_manifest": str(item.get("source_manifest") or ""),
        "source_audio_path": str(item.get("source_audio_path") or ""),
        "source_row": item.get("source_row") if isinstance(item.get("source_row"), dict) else {},
        "diagnostic_context": diagnostic_context(item),
        "repair_started": False,
        "training_started": False,
        "release_ready": False,
        "can_publish_as_result": False,
    }


def audit_rows_path(audit: dict[str, Any], audit_json: Path) -> Path | None:
    rows_jsonl = Path(str(audit.get("rows_jsonl") or ""))
    if not str(rows_jsonl):
        return None
    return rows_jsonl if rows_jsonl.is_absolute() else audit_json.parent / rows_jsonl


def audit_warning_issue_tags(warnings: Sequence[Any]) -> list[str]:
    tags: list[str] = []
    warning_set = {str(warning) for warning in warnings}
    if "residual_vocal_watch" in warning_set:
        tags.append("residual_vocal")
    for warning in warning_set:
        if warning.endswith("_margin_watch"):
            tags.append("muffled_minus")
    return stable_unique(tags)


def audit_blocker_issue_tags(blockers: Sequence[Any]) -> list[str]:
    tags: list[str] = []
    for blocker in {str(blocker) for blocker in blockers}:
        if blocker == "severe_vocal_bleed_in_minus":
            tags.append("severe_vocal_bleed_in_minus")
        elif "vocal_bleed" in blocker:
            tags.append("residual_vocal")
        elif (
            blocker == "minus_detail_gate_blocked"
            or blocker.endswith("_below_detail_gate")
            or blocker.endswith("_margin_blocked")
        ):
            tags.append("muffled_minus")
        elif blocker:
            tags.append("minus_gate_blocked")
    return stable_unique(tags)


def audit_warning_items(minus_audit_json: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    audit = read_json(minus_audit_json)
    failed = collect_true_flag_failures("minus_audit", audit)
    if audit.get("schema") != "short_minus_quality_audit.v1":
        failed.append("minus_audit_schema_unexpected")
    rows_path = audit_rows_path(audit, minus_audit_json)
    if rows_path is None or not rows_path.exists():
        failed.append("minus_audit_rows_jsonl_missing")
        return audit, [], stable_unique(failed)
    items: list[dict[str, Any]] = []
    for row in read_jsonl(rows_path):
        failed.extend(collect_true_flag_failures("minus_audit_row", row))
        audit_decision = str(row.get("audit_decision") or "")
        warnings = list(row.get("quality_warning_reasons") or [])
        blockers = list(row.get("audit_blockers") or [])
        issue_tags = stable_unique(audit_warning_issue_tags(warnings) + audit_blocker_issue_tags(blockers))
        if audit_decision == "passed" and not warnings:
            continue
        if audit_decision != "passed" and not blockers:
            continue
        if not issue_tags:
            continue
        verdict = "machine_blocked" if audit_decision != "passed" else "machine_warning"
        items.append(
            {
                "schema": "short_listen_feedback_item.v1",
                "audio_path": canonical_audio_path(row.get("short_audio_path")),
                "verdict": verdict,
                "issue_tags": issue_tags,
                "note": (
                    "Machine minus audit blocker; not a human rejection."
                    if verdict == "machine_blocked"
                    else "Machine minus audit warning; not a human rejection."
                ),
                "source_alias": str(row.get("source_alias") or ""),
                "stem_type": "minus",
                "track_alias": str(row.get("track_alias") or ""),
                "source_manifest": str(row.get("source_manifest") or ""),
                "source_audio_path": str(row.get("source_audio_path") or ""),
                "source_row": row.get("source_row") if isinstance(row.get("source_row"), dict) else {},
                "minus_audit": row,
                "release_ready": False,
                "can_publish_as_result": False,
            }
        )
    return audit, items, stable_unique(failed)


def training_view_items(training_view_jsonl: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    failed: list[str] = []
    if not training_view_jsonl.exists():
        return {}, [], ["training_view_jsonl_missing"]
    rows = read_jsonl(training_view_jsonl)
    items: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if row.get("safe_for_this_view") is not True:
            continue
        if row.get("label") != 0:
            continue
        if str(row.get("view_name") or "") != "minus_critic" or str(row.get("target_name") or "") != "minus":
            continue
        blockers = [str(label) for label in row.get("blocker_labels") or [] if str(label)]
        issue_labels = [str(label) for label in row.get("issue_labels") or [] if str(label)]
        corrective_actions = [str(action) for action in row.get("corrective_action_labels") or [] if str(action)]
        corrective_lanes = [str(lane) for lane in row.get("corrective_target_lanes") or [] if str(lane)]
        if not blockers or not corrective_actions:
            continue
        items.append(
            {
                "schema": "short_listen_feedback_item.v1",
                "audio_path": f"audio/trv/{indexed_alias('q', index)}/minus.flac",
                "verdict": "machine_blocked",
                "issue_tags": stable_unique([*issue_labels, *blockers]),
                "source_blocker_labels": blockers,
                "source_issue_labels": issue_labels,
                "corrective_action_labels": corrective_actions,
                "corrective_target_lanes": corrective_lanes,
                "note": "Machine training-view hardcase blocker; not a human rejection.",
                "source_alias": "tview",
                "stem_type": "minus",
                "track_alias": str(row.get("track_id") or ""),
                "source_manifest": str(training_view_jsonl),
                "source_audio_path": str(row.get("candidate_audio_path") or ""),
                "source_row": row,
                "release_ready": False,
                "can_publish_as_result": False,
            }
        )
    if not items:
        failed.append("no_training_view_actionable_minus_items")
    return {
        "schema": "training_view_queue_source.v1",
        "training_view_jsonl": str(training_view_jsonl),
        "rows_scanned": len(rows),
    }, items, stable_unique(failed)


def normalize_stem(value: Any) -> str:
    return str(value or "").strip().lower().replace("vocals", "vocal").replace("instrumental", "minus")


def pack_audio_items(pack_manifest_json: Path | None) -> dict[str, dict[str, Any]]:
    if pack_manifest_json is None or not pack_manifest_json.exists():
        return {}
    pack = read_json(pack_manifest_json)
    raw_items = pack.get("items")
    if not isinstance(raw_items, list):
        raw_items = pack.get("stems")
    if not isinstance(raw_items, list):
        raw_items = pack.get("listen_audio_files")
    if not isinstance(raw_items, list):
        return {}
    items: dict[str, dict[str, Any]] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        stem = normalize_stem(item.get("stem_type") or item.get("artifact_kind"))
        audio_path = canonical_audio_path(item.get("audio_path") or item.get("short_audio_path"))
        if stem and audio_path and stem not in items:
            copied = dict(item)
            copied["_canonical_audio_path"] = audio_path
            items[stem] = copied
    return items


def pack_audio_paths(pack_manifest_json: Path | None) -> dict[str, str]:
    return {stem: str(item.get("_canonical_audio_path") or "") for stem, item in pack_audio_items(pack_manifest_json).items()}


def audit_rows_by_audio_path(minus_audit_json: Path | None) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if minus_audit_json is None or not str(minus_audit_json):
        return {}, []
    if not minus_audit_json.exists():
        return {}, ["minus_audit_json_missing_for_machine_gate_queue"]
    audit = read_json(minus_audit_json)
    failed = collect_true_flag_failures("minus_audit", audit)
    if audit.get("schema") != "short_minus_quality_audit.v1":
        failed.append("minus_audit_schema_unexpected")
    rows_path = audit_rows_path(audit, minus_audit_json)
    if rows_path is None or not rows_path.exists():
        failed.append("minus_audit_rows_jsonl_missing")
        return {}, stable_unique(failed)
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(rows_path):
        if not isinstance(row, dict):
            continue
        audio_path = canonical_audio_path(row.get("short_audio_path"))
        if audio_path and audio_path not in rows:
            rows[audio_path] = row
    return rows, stable_unique(failed)


def add_issue(stem_issues: dict[str, list[str]], stem: str, issue: str) -> None:
    stem = normalize_stem(stem)
    if stem and issue:
        stem_issues.setdefault(stem, []).append(issue)


def machine_gate_items(machine_gate_json: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    gate = read_json(machine_gate_json)
    failed = collect_true_flag_failures("machine_gate", gate)
    if gate.get("schema") != "machine_stem_acceptance_gate.v1":
        failed.append("machine_gate_schema_unexpected")
    if gate.get("gate_status") != "blocked":
        failed.append("machine_gate_not_blocked")
    if gate.get("final_result_allowed") is True:
        failed.append("machine_gate_final_result_allowed_true")
    if gate.get("delivery_allowed") is True:
        failed.append("machine_gate_delivery_allowed_true")

    failed_checks = [str(check) for check in gate.get("failed_checks") or [] if str(check)]
    if not failed_checks:
        failed.append("machine_gate_failed_checks_missing")
    train_ready_stems = {
        normalize_stem(stem)
        for stem in gate.get("train_ready_stems") or []
        if normalize_stem(stem)
    }

    pack_path = Path(str(gate.get("pack_manifest_json") or ""))
    if str(pack_path) and not pack_path.is_absolute():
        pack_path = machine_gate_json.parent / pack_path
    pack_items = pack_audio_items(pack_path if str(pack_path) else None)
    audio_paths = {stem: str(item.get("_canonical_audio_path") or "") for stem, item in pack_items.items()}

    minus_audit_raw = str(gate.get("minus_audit_json") or "").strip()
    minus_audit_path = Path(minus_audit_raw) if minus_audit_raw else None
    if minus_audit_path is not None and not minus_audit_path.is_absolute():
        minus_audit_path = machine_gate_json.parent / minus_audit_path
    minus_audit_rows, minus_audit_failed = audit_rows_by_audio_path(minus_audit_path)
    failed.extend(minus_audit_failed)

    stem_issues: dict[str, list[str]] = {}
    stem_failed_checks: dict[str, list[str]] = {}
    for check in failed_checks:
        if check.startswith("required_stem_train_ready_missing:"):
            stem = check.split(":", 1)[1]
            add_issue(stem_issues, stem, "train_ready_model_missing")
            stem_failed_checks.setdefault(normalize_stem(stem), []).append(check)
        elif check.startswith("training_job_blocked:"):
            parts = check.split(":")
            stem = parts[1] if len(parts) > 1 else ""
            status = parts[2] if len(parts) > 2 else ""
            add_issue(stem_issues, stem, "training_job_blocked")
            if "insufficient_unique_tracks" in status:
                add_issue(stem_issues, stem, "insufficient_unique_tracks")
            stem_failed_checks.setdefault(normalize_stem(stem), []).append(check)
        elif check.startswith("minus_audit"):
            add_issue(stem_issues, "minus", "minus_gate_blocked")
            if "vocal" in check:
                add_issue(stem_issues, "minus", "residual_vocal")
            stem_failed_checks.setdefault("minus", []).append(check)

    blocked_training = gate.get("blocked_training_job_stems")
    if isinstance(blocked_training, dict):
        for stem, statuses in blocked_training.items():
            if normalize_stem(stem) in train_ready_stems:
                continue
            if not isinstance(statuses, list):
                statuses = [statuses]
            for status in statuses:
                add_issue(stem_issues, str(stem), "training_job_blocked")
                if "insufficient_unique_tracks" in str(status):
                    add_issue(stem_issues, str(stem), "insufficient_unique_tracks")

    for stem, issues in list(stem_issues.items()):
        if "train_ready_model_missing" in issues and "training_job_blocked" not in issues:
            add_issue(stem_issues, stem, "training_job_missing")

    required_stems = [normalize_stem(stem) for stem in gate.get("required_stems") or [] if normalize_stem(stem)]
    ordered_stems = stable_unique(required_stems + sorted(stem_issues))
    items: list[dict[str, Any]] = []
    for index, stem in enumerate([stem for stem in ordered_stems if stem in stem_issues], start=1):
        issue_tags = stable_unique(stem_issues[stem])
        audio_path = audio_paths.get(stem) or f"audio/stm/t001/{stem}.flac"
        pack_item = pack_items.get(stem) or {}
        minus_audit_row = minus_audit_rows.get(audio_path) or {}
        nested_source_row = (
            minus_audit_row.get("source_row")
            if isinstance(minus_audit_row.get("source_row"), dict)
            else pack_item.get("source_row")
            if isinstance(pack_item.get("source_row"), dict)
            else {}
        )
        source_manifest = str(minus_audit_row.get("source_manifest") or pack_item.get("source_manifest") or "")
        source_audio_path = str(minus_audit_row.get("source_audio_path") or pack_item.get("source_audio_path") or "")
        source_row = {
            "machine_gate_json": str(machine_gate_json),
            "pack_manifest_json": str(gate.get("pack_manifest_json") or ""),
            "failed_checks": stem_failed_checks.get(stem, []),
            "blocked_training_job_statuses": blocked_training.get(stem, []) if isinstance(blocked_training, dict) else [],
        }
        if source_manifest:
            source_row["source_manifest"] = source_manifest
        if source_audio_path:
            source_row["source_audio_path"] = source_audio_path
        if nested_source_row:
            source_row["source_row"] = nested_source_row

        item = {
            "schema": "short_listen_feedback_item.v1",
            "audio_path": audio_path,
            "verdict": "machine_blocked",
            "issue_tags": issue_tags,
            "note": "Machine full-stem gate blocker; not a human rejection.",
            "source_alias": "fgate",
            "stem_type": stem,
            "track_alias": "t001",
            "source_manifest": source_manifest or str(machine_gate_json),
            "source_audio_path": source_audio_path,
            "source_row": source_row,
            "machine_gate": {
                "gate_status": gate.get("gate_status"),
                "final_result_allowed": gate.get("final_result_allowed"),
                "train_ready_stems": gate.get("train_ready_stems") if isinstance(gate.get("train_ready_stems"), list) else [],
            },
            "release_ready": False,
            "can_publish_as_result": False,
        }
        if minus_audit_row:
            item["minus_audit"] = minus_audit_row
        items.append(item)

    if not items:
        failed.append("no_machine_gate_action_items")
    return gate, items, stable_unique(failed)


def summary_text(manifest: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# short listen repair queue",
        "",
        f"- queue_status: `{manifest['queue_status']}`",
        f"- queue_item_count: `{manifest['queue_item_count']}`",
        f"- release_ready: `{str(manifest['release_ready']).lower()}`",
        f"- can_publish_as_result: `{str(manifest['can_publish_as_result']).lower()}`",
        "",
        "## Items",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['queue_item_id']}` `{row['audio_path']}` -> `{row['route_intent']}`")
    if manifest["failed_checks"]:
        lines.extend(["", "## Failed Checks", ""])
        for check in manifest["failed_checks"]:
            lines.append(f"- `{check}`")
    lines.append("")
    return "\n".join(lines)


def build_short_listen_repair_queue(
    *,
    feedback_manifest_json: Path | None = None,
    minus_audit_json: Path | None = None,
    machine_gate_json: Path | None = None,
    training_view_jsonl: Path | None = None,
    output_dir: Path,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(
        progress_path,
        "start",
        feedback_manifest_json=str(feedback_manifest_json or ""),
        minus_audit_json=str(minus_audit_json or ""),
        machine_gate_json=str(machine_gate_json or ""),
        training_view_jsonl=str(training_view_jsonl or ""),
    )

    failed: list[str] = []
    feedback: dict[str, Any] = {}
    queue_source = "human_feedback"
    source_items: list[dict[str, Any]] = []
    source_count = sum(1 for path in (feedback_manifest_json, minus_audit_json, machine_gate_json, training_view_jsonl) if path)
    if source_count > 1:
        failed.append("multiple_queue_sources")
    elif training_view_jsonl:
        queue_source = "training_view_hardcase"
        feedback, source_items, view_failed = training_view_items(training_view_jsonl)
        failed.extend(view_failed)
    elif machine_gate_json:
        queue_source = "machine_full_pack_gate"
        gate, source_items, gate_failed = machine_gate_items(machine_gate_json)
        failed.extend(gate_failed)
        feedback = {
            "schema": "machine_full_pack_gate_queue_source.v1",
            "pack_id": "",
            "pack_manifest_json": str(gate.get("pack_manifest_json") or ""),
            "review_coverage_status": "machine_full_pack_gate",
            "machine_gate_json": str(machine_gate_json),
            "gate_status": gate.get("gate_status"),
        }
    elif minus_audit_json:
        queue_source = "minus_audit_warning"
        audit, source_items, audit_failed = audit_warning_items(minus_audit_json)
        failed.extend(audit_failed)
        feedback = {
            "schema": "minus_audit_warning_queue_source.v1",
            "pack_id": "",
            "pack_manifest_json": "",
            "review_coverage_status": "machine_warning_only",
            "minus_audit_json": str(minus_audit_json),
            "minus_audit_status": audit.get("audit_status"),
        }
        if not source_items:
            failed.append("no_minus_audit_warning_items")
    elif feedback_manifest_json:
        feedback = read_json(feedback_manifest_json)
        failed.extend(collect_true_flag_failures("feedback", feedback))
        if feedback.get("schema") != "short_listen_feedback_manifest.v1":
            failed.append("feedback_schema_unexpected")
        if feedback.get("overall_listen_verdict") != "rejected":
            failed.append("feedback_overall_not_rejected")
        if feedback.get("failed_checks") not in ([], None):
            failed.append("feedback_failed_checks_not_empty")

        feedback_items = [item for item in feedback.get("items") or [] if isinstance(item, dict)]
        source_items = [item for item in feedback_items if item.get("verdict") == "rejected"]
        if not source_items:
            failed.append("no_rejected_feedback_items")
    else:
        failed.append("queue_source_missing")

    rows: list[dict[str, Any]] = []
    seen_audio_paths: set[str] = set()
    for index, item in enumerate(source_items, start=1):
        audio_path = canonical_audio_path(item.get("audio_path"))
        failed.extend(check_short_relative_path(audio_path))
        if audio_path in seen_audio_paths:
            failed.append(f"duplicate_repair_audio_path:{audio_path}")
            continue
        seen_audio_paths.add(audio_path)
        if item.get("release_ready") is True:
            failed.append(f"feedback_item_release_ready_true:{audio_path}")
        if item.get("can_publish_as_result") is True:
            failed.append(f"feedback_item_can_publish_as_result_true:{audio_path}")
        if not item.get("issue_tags"):
            failed.append(f"feedback_item_issue_tags_missing:{audio_path}")
        rows.append(build_row(index, item))

    failed = stable_unique(failed)
    accepted = bool(rows) and not failed
    if not accepted:
        rows = []
    manifest = {
        "schema": "short_listen_repair_queue_manifest.v1",
        "queue_status": (
            "ready_for_machine_warning_cycle"
            if queue_source == "minus_audit_warning"
            else "ready_for_full_pack_loop"
            if queue_source in {"machine_full_pack_gate", "training_view_hardcase"}
            else "ready_for_repair_cycle"
        )
        if accepted
        else "blocked",
        "queue_source": queue_source,
        "feedback_manifest_json": str(feedback_manifest_json or ""),
        "minus_audit_json": str(minus_audit_json or ""),
        "machine_gate_json": str(machine_gate_json or ""),
        "training_view_jsonl": str(training_view_jsonl or ""),
        "pack_id": str(feedback.get("pack_id") or ""),
        "pack_manifest_json": str(feedback.get("pack_manifest_json") or ""),
        "review_coverage_status": str(feedback.get("review_coverage_status") or "unknown"),
        "queue_item_count": len(rows),
        "route_intent_counts": route_counts(rows),
        "issue_tag_counts": issue_tag_counts(rows),
        "failed_checks": failed,
        "rows_jsonl": str(output_dir / ROWS_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "release_ready": False,
        "can_publish_as_result": False,
        "delivery_allowed": False,
        "presentation_status": "repair_queue_not_for_listen_pack",
        "not_claimed": ["release_ready", "publishable_result", "audio_repaired", "training_started"],
        "safety": {
            "reference_required": False,
            "human_csv_required": False,
            "training_started": False,
            "separator_training_started": False,
            "repair_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "thresholds_relaxed": False,
            "cleanup_routes_added": False,
            "release_ready": False,
        },
        "command_args": command_args or [],
    }
    write_json(output_dir / MANIFEST_NAME, manifest)
    write_jsonl(output_dir / ROWS_NAME, rows)
    (output_dir / SUMMARY_NAME).write_text(summary_text(manifest, rows), encoding="utf-8")
    progress(progress_path, "finish", queue_status=manifest["queue_status"], queue_item_count=len(rows), failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a repair queue from rejected short listen feedback.")
    parser.add_argument("--feedback-manifest-json", type=Path)
    parser.add_argument("--minus-audit-json", type=Path)
    parser.add_argument("--machine-gate-json", type=Path)
    parser.add_argument("--training-view-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_short_listen_repair_queue.py",
        entrypoint_path="/app/tools/build_short_listen_repair_queue.py",
    )
    args = parse_args(argv)
    manifest = build_short_listen_repair_queue(
        feedback_manifest_json=args.feedback_manifest_json,
        minus_audit_json=args.minus_audit_json,
        machine_gate_json=args.machine_gate_json,
        training_view_jsonl=args.training_view_jsonl,
        output_dir=args.output_dir,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "queue_status": manifest["queue_status"],
                "queue_source": manifest["queue_source"],
                "queue_item_count": manifest["queue_item_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if str(manifest["queue_status"]).startswith("ready_for_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
