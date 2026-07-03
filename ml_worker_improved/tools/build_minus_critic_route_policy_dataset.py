from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from .stem_route_policy_common import (
        ACTION_BLOCK_UNKNOWN,
        ACTION_QUARANTINE_ROUTE,
        ISSUE_MISSING_DETAIL,
        ISSUE_STEREO_COLLAPSE,
        ISSUE_UNKNOWN,
        ISSUE_VOCAL_LEAK,
        LANE_MINUS,
        normalize_issue_type,
        safe_policy_row_id,
        short_safe_segment,
        validate_policy_dataset_row,
    )
except ImportError:  # pragma: no cover - direct script execution
    from stem_route_policy_common import (  # type: ignore
        ACTION_BLOCK_UNKNOWN,
        ACTION_QUARANTINE_ROUTE,
        ISSUE_MISSING_DETAIL,
        ISSUE_STEREO_COLLAPSE,
        ISSUE_UNKNOWN,
        ISSUE_VOCAL_LEAK,
        LANE_MINUS,
        normalize_issue_type,
        safe_policy_row_id,
        short_safe_segment,
        validate_policy_dataset_row,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    return path


def _stable_alias(prefix: str, value: str) -> str:
    digest = hashlib.sha1(str(value or "").encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{prefix}{digest}"


def _window_id(row: Mapping[str, Any]) -> str:
    start_ms = int(round(float(row.get("window_start_sec") or 0.0) * 1000.0))
    end_ms = int(round(float(row.get("window_end_sec") or 0.0) * 1000.0))
    return f"w{start_ms:06d}_{end_ms:06d}"


def _scored_row_dedupe_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("row_id") or ""),
        str(row.get("route_id") or ""),
        str(row.get("true_label")),
        str(row.get("critic_decision") or ""),
        str(row.get("window_start_sec") or ""),
        str(row.get("window_end_sec") or ""),
    )


def _issue_from_blockers(blockers: list[str], *, true_label: int | None) -> str:
    lowered = [str(blocker or "").lower() for blocker in blockers]
    if any("vocal" in blocker or "bleed" in blocker for blocker in lowered):
        return ISSUE_VOCAL_LEAK
    if any("phase" in blocker or "stereo" in blocker for blocker in lowered):
        return ISSUE_STEREO_COLLAPSE
    if any("detail" in blocker or "integrity" in blocker or "consistency" in blocker or "damage" in blocker for blocker in lowered):
        return ISSUE_MISSING_DETAIL
    if true_label == 0:
        return ISSUE_UNKNOWN
    return ISSUE_UNKNOWN


def _root_cause_from_row(row: Mapping[str, Any], issue_type: str) -> str:
    decision = str(row.get("critic_decision") or "").strip().lower()
    blockers = [str(blocker or "").strip().lower() for blocker in row.get("predicted_blockers") or [] if str(blocker or "").strip()]
    if blockers:
        return f"minus_critic:{short_safe_segment(blockers[0], fallback='blocker')}"
    if decision == "accept_candidate" and int(row.get("true_label") or 0) == 0:
        return "minus_critic:false_accept_label"
    if issue_type == ISSUE_UNKNOWN:
        return "minus_critic:unknown_failure"
    return f"minus_critic:{issue_type}"


def _to_policy_row(
    *,
    index: int,
    row: Mapping[str, Any],
    source_run_alias: str,
) -> tuple[dict[str, Any], str | None]:
    target_name = str(row.get("target_name") or row.get("view_name") or "").strip()
    route_id = str(row.get("route_id") or "").strip()
    if target_name != "minus_critic":
        return {}, "non_minus_critic_row"
    if not route_id:
        return {}, "missing_route_id"
    try:
        true_label = int(row.get("true_label"))
    except (TypeError, ValueError):
        return {}, "missing_true_label"
    if true_label not in {0, 1}:
        return {}, "unsupported_true_label"

    blockers = [str(blocker or "").strip() for blocker in row.get("predicted_blockers") or [] if str(blocker or "").strip()]
    issue_type = normalize_issue_type(_issue_from_blockers(blockers, true_label=true_label))
    oracle_action = ACTION_QUARANTINE_ROUTE if true_label == 0 else ACTION_BLOCK_UNKNOWN
    policy_row = {
        "schema": "stem_route_policy_row.v1",
        "row_id": safe_policy_row_id(index),
        "source_run": source_run_alias,
        "source_target_critic_row_id": _stable_alias("tc", str(row.get("row_id") or "")),
        "window_id": _window_id(row),
        "candidate_id": _stable_alias("c", str(row.get("candidate_id") or row.get("row_id") or "")),
        "route_signature": f"{LANE_MINUS}:{route_id}",
        "lane_routes": {LANE_MINUS: [route_id]},
        "issue_type": issue_type,
        "root_cause": _root_cause_from_row(row, issue_type),
        "affected_stem": "minus",
        "gate_verdicts": {
            "critic_decision": str(row.get("critic_decision") or ""),
            "critic_target": "minus",
            "true_label": true_label,
            "label_source": str(row.get("label_source") or ""),
            "shadow_only": bool(row.get("shadow_only", True)),
            "release_ready": bool(row.get("release_ready", False)),
            "separator_training_started": bool(row.get("separator_training_started", False)),
        },
        "forensic_features": {
            "predicted_accept_probability": float(row.get("predicted_accept_probability") or 0.0),
            "predicted_block_probability": float(row.get("predicted_block_probability") or 0.0),
            "confidence": float(row.get("confidence") or 0.0),
        },
        "damage_features": {
            "predicted_blockers": blockers,
            "abstained": bool(row.get("abstained", False)),
        },
        "reconstruction_features": {
            "window_start_sec": float(row.get("window_start_sec") or 0.0),
            "window_end_sec": float(row.get("window_end_sec") or 0.0),
        },
        "current_action": ACTION_BLOCK_UNKNOWN,
        "oracle_action": oracle_action,
        "oracle_label_source": "target_audio_error_view_label",
        "usable_for_training": True,
        "usable_for_promotion": False,
        "is_ground_truth": False,
        "label_type": "target_audio_error_critic_view_label",
        "critic_shadow_only": True,
        "delivery_allowed": False,
        "release_ready": False,
        "separator_training_started": False,
    }
    try:
        validate_policy_dataset_row(policy_row)
    except ValueError as exc:
        policy_row["usable_for_training"] = False
        policy_row["validation_error"] = str(exc)
        return policy_row, "unsafe_policy_dataset_row"
    return policy_row, None


def _split_rows(rows: list[Mapping[str, Any]]) -> dict[str, list[str]]:
    train: list[str] = []
    holdout: list[str] = []
    for index, row in enumerate(rows):
        target = holdout if index % 5 == 4 else train
        target.append(str(row["row_id"]))
    return {"train": train, "holdout": holdout}


def build_minus_critic_route_policy_dataset(
    *,
    scores_jsonl_path: Path,
    output_dir: Path,
    source_run_alias: str = "scored_minus_critic_windows.jsonl",
    dedupe: bool = True,
    mount_preflight_json_path: Path | None = None,
    require_mount_preflight: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_json = output_dir / "manifest.json"
    rows_jsonl = output_dir / "rows.jsonl"
    splits_json = output_dir / "splits.json"
    mount_preflight = _read_json(mount_preflight_json_path)
    mount_preflight_status = str(mount_preflight.get("preflight_status") or "")
    mount_probe_status = str(mount_preflight.get("probe_status") or "")
    if require_mount_preflight and (
        mount_preflight_status != "passed" or not bool(mount_preflight.get("can_write_route_artifacts", False))
    ):
        manifest = {
            "schema": "minus_critic_route_policy_dataset_manifest.v1",
            "manifest_path": str(manifest_json),
            "rows_jsonl": str(rows_jsonl),
            "splits_json": str(splits_json),
            "source_scores_jsonl": str(scores_jsonl_path),
            "source_run_alias": source_run_alias,
            "source_target": "minus_critic",
            "row_count": 0,
            "usable_training_row_count": 0,
            "usable_for_promotion_row_count": 0,
            "unsafe_row_count": 0,
            "input_scored_row_count": 0,
            "duplicate_rows_skipped": 0,
            "skipped_reason_counts": {},
            "action_counts": {},
            "issue_counts": {},
            "true_label_counts": {},
            "shadow_only": True,
            "delivery_allowed": False,
            "listen_delivery_artifact": False,
            "can_publish_as_result": False,
            "release_ready": False,
            "separator_training_started": False,
            "runtime_gates_changed": False,
            "presentation_status": "blocked_from_listen_pack",
            "policy_effect": "blocked_mount_preflight",
            "output_status": "blocked_mount_preflight",
            "mount_preflight": mount_preflight,
            "mount_preflight_status": mount_preflight_status or "missing",
            "mount_probe_status": mount_probe_status,
        }
        _write_json(manifest_json, manifest)
        return manifest

    scored_rows = _read_jsonl(scores_jsonl_path)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    duplicate_rows_skipped = 0
    skipped_reasons: Counter[str] = Counter()
    unsafe_row_count = 0

    for scored_row in scored_rows:
        key = _scored_row_dedupe_key(scored_row)
        if dedupe and key in seen:
            duplicate_rows_skipped += 1
            continue
        seen.add(key)
        row, skip_reason = _to_policy_row(index=len(rows) + 1, row=scored_row, source_run_alias=source_run_alias)
        if not row:
            skipped_reasons[str(skip_reason or "skipped")] += 1
            continue
        if skip_reason:
            skipped_reasons[skip_reason] += 1
            unsafe_row_count += 1
        rows.append(row)

    usable_rows = [row for row in rows if bool(row.get("usable_for_training"))]
    splits = _split_rows(usable_rows)
    _write_jsonl(rows_jsonl, rows)
    _write_json(splits_json, {"schema": "stem_route_policy_splits.v1", **splits})

    action_counts = Counter(str(row.get("oracle_action") or "") for row in rows)
    issue_counts = Counter(str(row.get("issue_type") or "") for row in rows)
    true_label_counts = Counter(str(row.get("gate_verdicts", {}).get("true_label")) for row in rows)
    manifest = {
        "schema": "minus_critic_route_policy_dataset_manifest.v1",
        "manifest_path": str(manifest_json),
        "rows_jsonl": str(rows_jsonl),
        "splits_json": str(splits_json),
        "source_scores_jsonl": str(scores_jsonl_path),
        "source_run_alias": source_run_alias,
        "source_target": "minus_critic",
        "row_count": len(rows),
        "usable_training_row_count": len(usable_rows),
        "usable_for_promotion_row_count": 0,
        "unsafe_row_count": unsafe_row_count,
        "input_scored_row_count": len(scored_rows),
        "duplicate_rows_skipped": duplicate_rows_skipped,
        "skipped_reason_counts": dict(sorted(skipped_reasons.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "true_label_counts": dict(sorted(true_label_counts.items())),
        "shadow_only": True,
        "delivery_allowed": False,
        "listen_delivery_artifact": False,
        "can_publish_as_result": False,
        "release_ready": False,
        "separator_training_started": False,
        "runtime_gates_changed": False,
        "presentation_status": "blocked_from_listen_pack",
        "policy_effect": "shadow_route_quarantine_training_rows",
        "output_status": "policy_dataset_ready",
        "mount_preflight": mount_preflight,
        "mount_preflight_status": mount_preflight_status,
        "mount_probe_status": mount_probe_status,
    }
    _write_json(manifest_json, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-jsonl", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-run-alias", default="scored_minus_critic_windows.jsonl")
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--mount-preflight-json", type=Path)
    parser.add_argument("--require-mount-preflight", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_minus_critic_route_policy_dataset(
        scores_jsonl_path=args.scores_jsonl,
        output_dir=args.out,
        source_run_alias=args.source_run_alias,
        dedupe=not args.no_dedupe,
        mount_preflight_json_path=args.mount_preflight_json,
        require_mount_preflight=bool(args.require_mount_preflight),
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
