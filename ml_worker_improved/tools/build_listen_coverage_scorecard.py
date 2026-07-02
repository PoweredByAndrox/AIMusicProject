#!/usr/bin/env python3
"""Build a deterministic scorecard for short listen pack coverage."""

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

from tools.artifact_name_policy import validate_artifact_segment
from tools.docker_runtime import require_docker_runtime

SCORE_NAME = "score.json"
GROUPS_NAME = "groups.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
START_TIME = time.monotonic()
AGGREGATE_VALIDATION_PENDING_STATUS = "aggregate_validation_pending"
AGGREGATE_VALIDATION_PENDING_PRESENTATION = "aggregate_validation_pending_not_for_listen_pack"

REQUIRED_6S_STEMS = {"bass", "drums", "guitar", "minus", "other", "piano", "vocal"}
BLOCKED_TRUE_FLAGS = (
    "release_ready",
    "can_publish_as_result",
    "training_started",
    "separator_training_started",
    "fine_tune_started",
    "models_changed",
    "model_registry_edited",
    "thresholds_relaxed",
    "cleanup_routes_added",
    "runtime_gates_changed",
)


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


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
        "expected_acceleration_mode": "external_only_cpu_manifest_scorecard",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "listen_coverage_scorecard_reads_existing_artifacts_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "listen_coverage_scorecard_no_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
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


def safety_flags() -> dict[str, bool]:
    return {
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


def check_output_names() -> None:
    for name in (SCORE_NAME, GROUPS_NAME, SUMMARY_NAME, PROGRESS_NAME):
        validate_artifact_segment(name)


def items(pack: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in pack.get("items") or pack.get("listen_audio_files") or [] if isinstance(row, dict)]


def stem_type_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        stem = str(row.get("stem_type") or "unknown")
        counts[stem] = counts.get(stem, 0) + 1
    return dict(sorted(counts.items()))


def full_length_minus_count(rows: Sequence[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        if str(row.get("stem_type") or "") != "minus":
            continue
        if float(row.get("duration_sec") or 0.0) >= 60.0:
            count += 1
    return count


def checks_by_id(goal_audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("id") or ""): row for row in goal_audit.get("checks") or [] if isinstance(row, dict)}


def resolve_review_input_path(raw: Any, *, base: Path | None) -> Path:
    path = Path(str(raw or ""))
    if not str(path):
        return path
    if path.is_absolute() or base is None:
        return path
    return base / path


def resolve_rows_jsonl(raw: Any, *, base: Path | None) -> Path:
    path = Path(str(raw or ""))
    if not str(path):
        return path
    if path.is_absolute() or base is None:
        return path
    return base / path


def semantic_verified_aliases(semantic_review: dict[str, Any], *, semantic_review_json: Path | None) -> set[str]:
    if not semantic_review:
        return set()
    if semantic_review.get("review_status") != "completed":
        return set()
    if int(semantic_review.get("pending_item_count") or 0) > 0:
        return set()
    if semantic_review.get("failed_checks") not in ([], None):
        return set()
    aliases = {str(item) for item in semantic_review.get("reviewed_group_aliases") or [] if str(item)}
    base = semantic_review_json.parent if semantic_review_json is not None else None
    for raw_pack in semantic_review.get("input_pack_jsons") or []:
        pack_path = resolve_review_input_path(raw_pack, base=base)
        if not pack_path.exists() or not pack_path.is_file():
            continue
        pack = read_json(pack_path)
        for item in pack.get("track_aliases") or []:
            if not isinstance(item, dict):
                continue
            track_alias = str(item.get("track_alias") or "")
            source_track_id = str(item.get("source_track_id") or "")
            if track_alias in aliases and source_track_id:
                aliases.add(source_track_id)
    return aliases


def group_rows(coverage: dict[str, Any], *, semantic_verified_group_aliases: set[str] | None = None) -> list[dict[str, Any]]:
    semantic_aliases = semantic_verified_group_aliases or set()
    rows: list[dict[str, Any]] = []
    for raw in coverage.get("stem_groups") or []:
        if not isinstance(raw, dict):
            continue
        stems = sorted(str(stem) for stem in raw.get("stem_types") or [])
        has = {f"has_{stem}": stem in stems for stem in sorted(REQUIRED_6S_STEMS)}
        group_alias = str(raw.get("group_alias") or "unknown")
        semantic_verified = raw.get("semantic_quality_verified") is True or group_alias in semantic_aliases
        full_6stem = REQUIRED_6S_STEMS.issubset(set(stems))
        if full_6stem and not semantic_verified:
            next_action = "semantic_stem_review"
        elif full_6stem:
            next_action = "human_listen_verdict"
        else:
            next_action = "expand_multi_track_cycle"
        rows.append(
            {
                "schema": "listen_coverage_scorecard_group.v1",
                "group_alias": group_alias,
                "item_count": int(raw.get("item_count") or 0),
                "stem_types": stems,
                "full_6stem_coverage": full_6stem,
                "semantic_quality_verified": semantic_verified,
                "review_status": "semantic_quality_unverified" if not semantic_verified else "machine_accepted_unverified",
                "next_action": next_action,
                "release_ready": False,
                "can_publish_as_result": False,
                **has,
            }
        )
    rows.sort(key=lambda row: row["group_alias"])
    return rows


def idea_funnel_summary(idea_funnel: dict[str, Any] | None) -> dict[str, Any]:
    if not idea_funnel:
        return {
            "required": True,
            "status": "missing",
            "idea_count": 0,
            "failed_checks": ["idea_funnel_missing"],
            "delivery_allowed": False,
            "presentation_status": "missing",
            "release_ready": False,
            "can_publish_as_result": False,
        }
    return {
        "required": True,
        "status": str(idea_funnel.get("funnel_status") or "unknown"),
        "idea_count": int(idea_funnel.get("idea_count") or 0),
        "failed_checks": [str(item) for item in idea_funnel.get("failed_checks") or []],
        "delivery_allowed": False,
        "presentation_status": "idea_funnel_not_listen_result",
        "release_ready": False,
        "can_publish_as_result": False,
    }


def semantic_review_summary(semantic_review: dict[str, Any] | None) -> dict[str, Any]:
    if not semantic_review:
        return {
            "required": True,
            "status": "missing",
            "pending_item_count": 0,
            "semantic_quality_verified_item_count": 0,
            "failed_checks": ["semantic_review_missing"],
            "release_ready": False,
            "can_publish_as_result": False,
        }
    return {
        "required": True,
        "status": str(semantic_review.get("review_status") or "unknown"),
        "pending_item_count": int(semantic_review.get("pending_item_count") or 0),
        "semantic_quality_verified_item_count": int(semantic_review.get("semantic_quality_verified_item_count") or 0),
        "failed_checks": [str(item) for item in semantic_review.get("failed_checks") or []],
        "release_ready": False,
        "can_publish_as_result": False,
    }


def minus_audit_summary(minus_audit: dict[str, Any] | None, *, minus_audit_json: Path | None) -> dict[str, Any]:
    if not minus_audit:
        return {
            "required": False,
            "status": "missing",
            "accepted_minus_count": 0,
            "blocked_minus_count": 0,
            "evidence_missing_count": 0,
            "warning_count": 0,
            "blocker_reasons": [],
            "blocked_audio_paths": [],
            "failed_checks": [],
            "release_ready": False,
            "can_publish_as_result": False,
        }
    rows_path = resolve_rows_jsonl(minus_audit.get("rows_jsonl"), base=minus_audit_json.parent if minus_audit_json else None)
    rows = read_jsonl(rows_path) if str(rows_path) else []
    blocker_rows = [row for row in rows if row.get("audit_blockers")]
    return {
        "required": True,
        "status": str(minus_audit.get("audit_status") or "unknown"),
        "accepted_minus_count": int(minus_audit.get("accepted_minus_count") or 0),
        "blocked_minus_count": int(minus_audit.get("blocked_minus_count") or 0),
        "evidence_missing_count": int(minus_audit.get("evidence_missing_count") or 0),
        "warning_count": int(minus_audit.get("warning_count") or 0),
        "blocker_reasons": stable_unique(reason for row in blocker_rows for reason in row.get("audit_blockers") or []),
        "blocked_audio_paths": stable_unique(row.get("short_audio_path") for row in blocker_rows),
        "failed_checks": [str(item) for item in minus_audit.get("failed_checks") or []],
        "release_ready": False,
        "can_publish_as_result": False,
    }


def schema_failures(pack: dict[str, Any], coverage: dict[str, Any], goal_audit: dict[str, Any], idea_funnel: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    if pack.get("schema") != "aggregate_short_listen_pack_manifest.v1":
        failed.append("pack_schema_unexpected")
    if pack.get("delivery_status") != AGGREGATE_VALIDATION_PENDING_STATUS:
        failed.append("pack_delivery_status_not_aggregate_validation_pending")
    if pack.get("delivery_allowed") is not False:
        failed.append("pack_delivery_allowed_not_false")
    if pack.get("presentation_status") != AGGREGATE_VALIDATION_PENDING_PRESENTATION:
        failed.append("pack_presentation_status_not_aggregate_validation_pending")
    if pack.get("release_status") != AGGREGATE_VALIDATION_PENDING_STATUS:
        failed.append("pack_release_status_not_aggregate_validation_pending")
    if pack.get("aggregate_validation_ready") is not True:
        failed.append("pack_aggregate_validation_ready_not_true")
    if pack.get("listenable") is not False:
        failed.append("pack_listenable_not_false")
    if coverage.get("schema") != "aggregate_short_listen_pack_coverage.v1":
        failed.append("coverage_schema_unexpected")
    if not goal_audit:
        failed.append("goal_audit_missing")
    elif goal_audit.get("schema") != "short_listen_goal_status_audit.v1":
        failed.append("goal_audit_schema_unexpected")
    if not idea_funnel:
        failed.append("idea_funnel_missing")
    elif idea_funnel.get("schema") != "short_listen_idea_funnel.v1":
        failed.append("idea_funnel_schema_unexpected")
    elif idea_funnel.get("funnel_status") != "ready":
        failed.append("idea_funnel_not_ready")
    elif int(idea_funnel.get("idea_count") or 0) < 1:
        failed.append("idea_funnel_empty")
    if idea_funnel:
        if idea_funnel.get("delivery_allowed") is not False:
            failed.append("idea_funnel_delivery_allowed_not_false")
        if idea_funnel.get("presentation_status") != "idea_funnel_not_listen_result":
            failed.append("idea_funnel_presentation_status_unexpected")
    return failed


def minus_audit_failures(minus_audit: dict[str, Any], minus_summary: dict[str, Any]) -> list[str]:
    if not minus_audit:
        return []
    failed: list[str] = []
    if minus_audit.get("schema") != "short_minus_quality_audit.v1":
        failed.append("minus_audit_schema_unexpected")
    if minus_summary["status"] not in {"completed", "completed_with_blockers"}:
        failed.append(f"minus_audit_status_unexpected:{minus_summary['status']}")
    blocked_count = int(minus_summary.get("blocked_minus_count") or 0)
    if blocked_count > 0:
        failed.append(f"minus_audit_blocked_minus_count:{blocked_count}")
    evidence_missing_count = int(minus_summary.get("evidence_missing_count") or 0)
    if evidence_missing_count > 0:
        failed.append(f"minus_audit_evidence_missing_count:{evidence_missing_count}")
    failed.extend(f"minus_audit_failed_check:{item}" for item in minus_summary.get("failed_checks") or [])
    if minus_audit.get("delivery_allowed") is True:
        failed.append("minus_audit_delivery_allowed_true")
    return stable_unique(failed)


def semantic_review_failures(
    *,
    complete_6stem_count: int,
    semantic_verified_group_count: int,
    semantic_review: dict[str, Any],
) -> list[str]:
    if complete_6stem_count <= 0 or semantic_verified_group_count >= complete_6stem_count:
        return []
    failed: list[str] = []
    if not semantic_review:
        return ["semantic_stem_review_missing"]
    if semantic_review.get("schema") != "semantic_stem_quality_review.v1":
        failed.append("semantic_stem_review_schema_unexpected")
    if semantic_review.get("review_status") != "completed":
        failed.append("semantic_stem_review_pending")
    if int(semantic_review.get("pending_item_count") or 0) > 0:
        failed.append("semantic_stem_review_pending_items")
    if semantic_review.get("release_ready") is True:
        failed.append("semantic_stem_review_release_ready_true")
    if semantic_review.get("can_publish_as_result") is True:
        failed.append("semantic_stem_review_can_publish_as_result_true")
    failed.extend(str(item) for item in semantic_review.get("failed_checks") or [])
    return stable_unique(failed)


def build_summary(manifest: dict[str, Any]) -> str:
    lines = [
        "# listen coverage scorecard",
        "",
        f"- scorecard_status: `{manifest['scorecard_status']}`",
        f"- total_item_count: `{manifest['total_item_count']}`",
        f"- complete_6stem_group_count: `{manifest['complete_6stem_group_count']}`",
        f"- idea_funnel_status: `{manifest['idea_funnel']['status']}`",
        f"- idea_count: `{manifest['idea_funnel']['idea_count']}`",
        f"- semantic_review_status: `{manifest['semantic_review']['status']}`",
        f"- semantic_review_pending_items: `{manifest['semantic_review']['pending_item_count']}`",
        f"- minus_audit_status: `{manifest['minus_audit']['status']}`",
        f"- minus_audit_blocked_minus_count: `{manifest['minus_audit']['blocked_minus_count']}`",
        f"- human_listen_pending: `{str(manifest['human_listen_pending']).lower()}`",
        f"- release_ready: `{str(manifest['release_ready']).lower()}`",
        f"- can_publish_as_result: `{str(manifest['can_publish_as_result']).lower()}`",
        "",
        "## Failed Checks",
        "",
    ]
    lines.extend(f"- `{item}`" for item in manifest["failed_checks"])
    if not manifest["failed_checks"]:
        lines.append("- none")
    lines.extend(["", "## Remaining Blockers", ""])
    lines.extend(f"- `{item}`" for item in manifest["remaining_blockers"])
    if not manifest["remaining_blockers"]:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def build_listen_coverage_scorecard(
    *,
    pack_manifest_json: Path,
    coverage_json: Path,
    output_dir: Path,
    goal_audit_json: Path | None = None,
    idea_funnel_json: Path | None = None,
    semantic_review_json: Path | None = None,
    minus_audit_json: Path | None = None,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    check_output_names()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(
        progress_path,
        "start",
        pack_manifest_json=str(pack_manifest_json),
        coverage_json=str(coverage_json),
        goal_audit_json=str(goal_audit_json or ""),
        idea_funnel_json=str(idea_funnel_json or ""),
        semantic_review_json=str(semantic_review_json or ""),
        minus_audit_json=str(minus_audit_json or ""),
    )

    pack = read_json(pack_manifest_json)
    coverage = read_json(coverage_json)
    goal_audit = read_json(goal_audit_json) if goal_audit_json else {}
    idea_funnel = read_json(idea_funnel_json) if idea_funnel_json else {}
    semantic_review = read_json(semantic_review_json) if semantic_review_json else {}
    minus_audit = read_json(minus_audit_json) if minus_audit_json else {}

    pack_items = items(pack)
    groups = group_rows(
        coverage,
        semantic_verified_group_aliases=semantic_verified_aliases(
            semantic_review,
            semantic_review_json=semantic_review_json,
        ),
    )
    complete_6stem_count = sum(1 for row in groups if row["full_6stem_coverage"])
    semantic_verified_group_count = sum(1 for row in groups if row["semantic_quality_verified"])
    goal_checks = checks_by_id(goal_audit)
    human_check = goal_checks.get("human_listen_verdict", {})
    human_blockers = human_check.get("blockers") if isinstance(human_check, dict) else []
    minus_summary = minus_audit_summary(minus_audit, minus_audit_json=minus_audit_json)
    minus_failed = minus_audit_failures(minus_audit, minus_summary)
    remaining_blockers = stable_unique([*(goal_audit.get("remaining_blockers") or []), *(human_blockers or []), *minus_failed])
    human_listen_pending = "human_listen_verdict_missing" in remaining_blockers

    failed = stable_unique(
        [
            *schema_failures(pack, coverage, goal_audit, idea_funnel),
            *collect_true_flag_failures("pack", pack),
            *collect_true_flag_failures("coverage", coverage),
            *collect_true_flag_failures("goal_audit", goal_audit),
            *collect_true_flag_failures("idea_funnel", idea_funnel),
            *collect_true_flag_failures("semantic_review", semantic_review),
            *collect_true_flag_failures("minus_audit", minus_audit),
            *minus_failed,
            *semantic_review_failures(
                complete_6stem_count=complete_6stem_count,
                semantic_verified_group_count=semantic_verified_group_count,
                semantic_review=semantic_review,
            ),
            *(coverage.get("failed_checks") or []),
            *(idea_funnel.get("failed_checks") or []),
        ]
    )
    if failed:
        status = "blocked"
    elif human_listen_pending:
        status = "machine_pass_human_listen_pending"
    else:
        status = "coverage_ready"

    manifest = {
        "schema": "listen_coverage_scorecard.v1",
        "scorecard_status": status,
        "pack_manifest_json": str(pack_manifest_json),
        "coverage_json": str(coverage_json),
        "goal_audit_json": str(goal_audit_json or ""),
        "idea_funnel_json": str(idea_funnel_json or ""),
        "semantic_review_json": str(semantic_review_json or ""),
        "minus_audit_json": str(minus_audit_json or ""),
        "total_item_count": len(pack_items),
        "stem_type_counts": stem_type_counts(pack_items),
        "full_length_minus_count": full_length_minus_count(pack_items),
        "group_count": len(groups),
        "complete_6stem_group_count": complete_6stem_count,
        "semantic_verified_group_count": semantic_verified_group_count,
        "semantic_unverified_group_count": len(groups) - semantic_verified_group_count,
        "human_listen_pending": human_listen_pending,
        "remaining_blockers": remaining_blockers,
        "failed_checks": failed,
        "idea_funnel": idea_funnel_summary(idea_funnel),
        "semantic_review": semantic_review_summary(semantic_review),
        "minus_audit": minus_summary,
        "groups_jsonl": str(output_dir / GROUPS_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "release_ready": False,
        "can_publish_as_result": False,
        "delivery_allowed": False,
        "presentation_status": "scorecard_not_listen_result",
        "safety": safety_flags(),
        "not_claimed": [
            "release_ready",
            "publishable_result",
            "semantic_quality_not_ground_truth",
            "human_listen_verdict",
        ],
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / GROUPS_NAME, groups)
    write_json(output_dir / SCORE_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(build_summary(manifest), encoding="utf-8")
    progress(
        progress_path,
        "finish",
        scorecard_status=status,
        failed_checks=failed,
        complete_6stem_group_count=complete_6stem_count,
        idea_funnel_status=manifest["idea_funnel"]["status"],
        minus_audit_status=manifest["minus_audit"]["status"],
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a short listen coverage scorecard with idea-funnel gate.")
    parser.add_argument("--pack-manifest-json", type=Path, required=True)
    parser.add_argument("--coverage-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--goal-audit-json", type=Path)
    parser.add_argument("--idea-funnel-json", type=Path)
    parser.add_argument("--semantic-review-json", type=Path)
    parser.add_argument("--minus-audit-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_listen_coverage_scorecard.py",
        entrypoint_path="/app/tools/build_listen_coverage_scorecard.py",
    )
    args = parse_args(argv)
    manifest = build_listen_coverage_scorecard(
        pack_manifest_json=args.pack_manifest_json,
        coverage_json=args.coverage_json,
        output_dir=args.output_dir,
        goal_audit_json=args.goal_audit_json,
        idea_funnel_json=args.idea_funnel_json,
        semantic_review_json=args.semantic_review_json,
        minus_audit_json=args.minus_audit_json,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "score_path": str(args.output_dir / SCORE_NAME),
                "scorecard_status": manifest["scorecard_status"],
                "failed_checks": manifest["failed_checks"],
                "complete_6stem_group_count": manifest["complete_6stem_group_count"],
                "idea_funnel_status": manifest["idea_funnel"]["status"],
                "semantic_review_status": manifest["semantic_review"]["status"],
                "minus_audit_status": manifest["minus_audit"]["status"],
                "minus_audit_blocked_minus_count": manifest["minus_audit"]["blocked_minus_count"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["scorecard_status"] != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
