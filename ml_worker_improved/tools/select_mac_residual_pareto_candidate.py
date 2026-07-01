#!/usr/bin/env python3
"""Select a diagnostic Mac residual candidate that passes both pitch and damage gates."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def _int_value(*values: Any) -> int:
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _candidate_path(value: Any) -> Path:
    path = Path(str(value or ""))
    if not path.is_file():
        raise FileNotFoundError(f"candidate source file missing: {value}")
    return path


def _damage_payload(candidate: Mapping[str, Any], manifest: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    if candidate.get("damage"):
        path = _candidate_path(candidate.get("damage"))
        return _load_json(path), str(path)
    manifest_damage = manifest.get("damage_audit")
    if isinstance(manifest_damage, Mapping) and manifest_damage:
        return dict(manifest_damage), str(manifest.get("damage_audit_manifest") or "")
    if manifest.get("damage_audit_manifest"):
        path = _candidate_path(manifest.get("damage_audit_manifest"))
        return _load_json(path), str(path)
    return {}, ""


def _max_failed_delta_ratio(damage: Mapping[str, Any]) -> float:
    failed = [
        float(region.get("delta_ratio") or 0.0)
        for region in damage.get("regions") or []
        if isinstance(region, Mapping) and region.get("blocker_reasons")
    ]
    return max(failed) if failed else 0.0


def _score_region_count(score: Mapping[str, Any]) -> int:
    if str(score.get("verdict") or "") == "pass":
        return 0
    regions = score.get("regions")
    return _int_value(score.get("total_fail_regions"), score.get("all_region_count"), score.get("selected_region_count"), len(regions) if isinstance(regions, list) else None)


def _damage_limited_route_blockers(manifest: Mapping[str, Any], damage: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if str(manifest.get("status") or "") != "damage_limited_repair_passed":
        blockers.append("status")
    if str(manifest.get("presentation_status") or "") != "damage_limited_repair_diagnostic_only":
        blockers.append("presentation_status")
    if str(manifest.get("full_holdout_resolution_status") or "") != "not_evaluated":
        blockers.append("full_holdout_resolution_status")
    if bool(damage.get("delivery_allowed")):
        blockers.append("damage_delivery_allowed")
    if bool(damage.get("listen_pack_created")):
        blockers.append("damage_listen_pack_created")
    return blockers


def _damage_audit_route_blockers(manifest: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if str(manifest.get("status") or "") != "residual_damage_audit_passed":
        blockers.append("status")
    if str(manifest.get("presentation_status") or "") != "damage_audit_route_diagnostic_only":
        blockers.append("presentation_status")
    if str(manifest.get("full_holdout_resolution_status") or "") != "not_evaluated":
        blockers.append("full_holdout_resolution_status")
    return blockers


def _append_once(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _delivery_blockers(
    *,
    eligible_for_full_holdout: bool,
    full_holdout_resolution_status: str,
    reference_bundle_status: str,
    reference_mismatch_status: str,
) -> list[str]:
    blockers: list[str] = []
    if not eligible_for_full_holdout:
        blockers.append("not_pitch_and_damage_passed")
    if full_holdout_resolution_status != "passed":
        blockers.append("full_holdout_not_passed")
    if reference_bundle_status != "actual_source_verified":
        blockers.append("reference_bundle_not_verified_for_actual_source")
    if reference_mismatch_status:
        blockers.append("reference_mismatch_status")
    return blockers


def _mapping_value(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return dict(value) if isinstance(value, Mapping) else {}


def _candidate_next_action(row: Mapping[str, Any]) -> str:
    if bool(row.get("eligible_for_delivery")):
        return "ready_for_delivery_contract"
    if bool(row.get("eligible_for_full_holdout")):
        return "run_full_holdout_validation"
    if bool(row.get("pitch_pass")) and bool(row.get("damage_pass")) and row.get("promotion_blocker_reasons"):
        return "fix_promotion_contract_blocker"
    if bool(row.get("pitch_pass")) and not str(row.get("damage_verdict") or ""):
        return "run_damage_audit"
    if bool(row.get("pitch_pass")) and not bool(row.get("damage_pass")):
        return "run_damage_limited_repair"
    return "continue_residual_training_or_rebatch"


def _stop_retraining_reason(row: Mapping[str, Any]) -> str:
    if bool(row.get("eligible_for_delivery")):
        return "pitch_damage_full_holdout_and_reference_passed"
    if bool(row.get("eligible_for_full_holdout")):
        return "pitch_and_damage_pass_waiting_for_full_holdout"
    if bool(row.get("pitch_pass")) and bool(row.get("damage_pass")) and row.get("promotion_blocker_reasons"):
        return "pitch_and_damage_pass_contract_blocked"
    if bool(row.get("pitch_pass")) and not str(row.get("damage_verdict") or ""):
        return "pitch_pass_waiting_for_damage_audit"
    if bool(row.get("pitch_pass")) and not bool(row.get("damage_pass")):
        return "pitch_pass_damage_failed"
    return ""


def score_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    candidate_id = str(candidate.get("id") or "").strip()
    if not candidate_id:
        raise ValueError("candidate id is required")
    manifest_path = _candidate_path(candidate.get("manifest"))
    manifest = _load_json(manifest_path)
    damage, damage_path = _damage_payload(candidate, manifest)
    schema = str(manifest.get("schema") or "")
    metrics = manifest.get("metrics") if isinstance(manifest.get("metrics"), Mapping) else {}
    before_count = _int_value(manifest.get("before_region_count"), metrics.get("before_region_count"))
    after_count = _int_value(manifest.get("after_region_count"), metrics.get("after_region_count"))
    after_score = manifest.get("after_score") if isinstance(manifest.get("after_score"), Mapping) else {}
    route_pitch_pass = (
        schema == "mac_residual_damage_limited_repair_route.v1"
        and str(after_score.get("verdict") or "") == "pass"
        and str(manifest.get("source_training_status") or "") == "trained_repair_passed_pitch_guard"
    )
    if schema == "mac_residual_damage_limited_repair_route.v1":
        after_count = _score_region_count(after_score)
    damage_verdict = str(damage.get("verdict") or "")
    damage_failed_count = _int_value(damage.get("failed_region_count"))
    pitch_pass = (after_count == 0 and before_count > 0) or route_pitch_pass
    damage_pass = damage_verdict == "pass"
    source_delivery_allowed = bool(manifest.get("delivery_allowed"))
    source_listen_pack_created = bool(manifest.get("listen_pack_created"))
    if schema == "mac_residual_damage_limited_repair_route.v1":
        route_blockers = _damage_limited_route_blockers(manifest, damage)
    elif schema == "mac_residual_damage_audit_route.v1":
        route_blockers = _damage_audit_route_blockers(manifest)
    else:
        route_blockers = []
    promotion_blocker_reasons = list(route_blockers)
    if source_delivery_allowed:
        _append_once(promotion_blocker_reasons, "source_delivery_allowed")
    if source_listen_pack_created:
        _append_once(promotion_blocker_reasons, "source_listen_pack_created")
    if bool(damage.get("delivery_allowed")):
        _append_once(promotion_blocker_reasons, "damage_delivery_allowed")
    if bool(damage.get("listen_pack_created")):
        _append_once(promotion_blocker_reasons, "damage_listen_pack_created")
    eligible_for_full_holdout = pitch_pass and damage_pass and not promotion_blocker_reasons
    reference_bundle_status = str(manifest.get("reference_bundle_status") or "missing_for_actual_source")
    reference_mismatch_status = str(manifest.get("reference_mismatch_status") or "")
    full_holdout_resolution_status = str(manifest.get("full_holdout_resolution_status") or "")
    delivery_blocker_reasons = _delivery_blockers(
        eligible_for_full_holdout=eligible_for_full_holdout,
        full_holdout_resolution_status=full_holdout_resolution_status,
        reference_bundle_status=reference_bundle_status,
        reference_mismatch_status=reference_mismatch_status,
    )
    eligible_for_delivery = eligible_for_full_holdout and not delivery_blocker_reasons
    row = {
        "candidate_id": candidate_id,
        "manifest": str(manifest_path),
        "damage_json": damage_path,
        "schema": schema,
        "status": str(manifest.get("status") or ""),
        "training_status": str(manifest.get("training_status") or ""),
        "source_training_status": str(manifest.get("source_training_status") or ""),
        "source_audio_path": str(manifest.get("source_audio_path") or ""),
        "source_track_identity": _mapping_value(manifest, "source_track_identity"),
        "source_residual_manifest": str(manifest.get("source_residual_manifest") or ""),
        "source_damage_audit_route_manifest": str(manifest.get("source_damage_audit_route_manifest") or ""),
        "source_guard_manifest": str(manifest.get("source_guard_manifest") or manifest.get("guard_manifest") or ""),
        "reference_bundle_status": reference_bundle_status,
        "reference_mismatch_status": reference_mismatch_status,
        "objective_profile": str(manifest.get("objective_profile") or ""),
        "presentation_status": str(manifest.get("presentation_status") or ""),
        "full_holdout_resolution_status": full_holdout_resolution_status,
        "before_region_count": before_count,
        "after_region_count": after_count,
        "region_reduction": before_count - after_count,
        "pitch_pass": pitch_pass,
        "damage_verdict": damage_verdict,
        "damage_pass": damage_pass,
        "damage_failed_region_count": damage_failed_count,
        "damage_blocker_reasons": list(damage.get("blocker_reasons") or []),
        "max_failed_delta_ratio": _max_failed_delta_ratio(damage),
        "source_delivery_allowed": source_delivery_allowed,
        "source_listen_pack_created": source_listen_pack_created,
        "damage_delivery_allowed": bool(damage.get("delivery_allowed")),
        "damage_listen_pack_created": bool(damage.get("listen_pack_created")),
        "promotion_blocker_reasons": promotion_blocker_reasons,
        "delivery_blocker_reasons": delivery_blocker_reasons,
        "eligible_for_full_holdout": eligible_for_full_holdout,
        "eligible_for_delivery": eligible_for_delivery,
        "eligible_for_promotion": eligible_for_delivery,
    }
    row["next_action"] = _candidate_next_action(row)
    row["stop_retraining_reason"] = _stop_retraining_reason(row)
    return row


def _candidate_id_from_manifest_path(path: Path) -> str:
    parent = path.parent.name
    grandparent = path.parent.parent.name
    if parent and grandparent:
        return f"{grandparent}_{parent}"
    return parent or path.stem


def candidates_from_manifest_globs(patterns: Sequence[str]) -> list[dict[str, str]]:
    paths: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for raw_path in sorted(glob.glob(str(pattern), recursive=True)):
            path = Path(raw_path)
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
    return [{"id": _candidate_id_from_manifest_path(path), "manifest": str(path)} for path in paths]


def _selection_key(row: Mapping[str, Any]) -> tuple[int, int, int, float, int, str]:
    return (
        0 if row.get("eligible_for_full_holdout") else 1,
        int(row.get("after_region_count") or 0),
        int(row.get("damage_failed_region_count") or 0),
        float(row.get("max_failed_delta_ratio") or 0.0),
        -int(row.get("region_reduction") or 0),
        str(row.get("candidate_id") or ""),
    )


def run_selection(*, candidates: Sequence[Mapping[str, Any]], output_dir: Path) -> dict[str, Any]:
    rows = sorted((score_candidate(candidate) for candidate in candidates), key=_selection_key)
    selected = next((row for row in rows if row["eligible_for_full_holdout"]), None)
    blocker_reasons: list[str] = []
    if selected is None:
        blocker_reasons.append("no_candidate_passes_pitch_and_damage")
    next_action_counts = {action: sum(1 for row in rows if row["next_action"] == action) for action in sorted({row["next_action"] for row in rows})}
    result = {
        "schema": "mac_residual_pareto_selection.v1",
        "status": "pareto_candidate_selected_for_full_holdout" if selected else "blocked_no_pareto_candidate",
        "selected_candidate_id": str(selected.get("candidate_id") if selected else ""),
        "selected_candidate": dict(selected or {}),
        "candidate_count": len(rows),
        "candidates": rows,
        "next_action_counts": next_action_counts,
        "blocker_reasons": blocker_reasons,
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "pareto_selection_diagnostic_only",
        "contract": {
            "diagnostic_only": True,
            "no_listen_delivery": True,
            "requires_pitch_pass": True,
            "requires_damage_pass": True,
            "requires_full_holdout_after_selection": True,
            "requires_reference_before_delivery": True,
        },
    }
    _write_json(Path(output_dir) / "selection.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-json", action="append", default=[], help="JSON object with id, manifest, and optional damage")
    parser.add_argument("--candidates-json", type=Path, help="JSON file with a candidates array")
    parser.add_argument("--manifest-glob", action="append", default=[], help="Glob for residual run manifest.json files")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.candidate_json and args.candidates_json is None and not args.manifest_glob:
        parser.error("one of --candidate-json, --candidates-json, or --manifest-glob is required")
    return args


def main() -> int:
    args = parse_args()
    candidates = [json.loads(value) for value in args.candidate_json]
    if args.candidates_json is not None:
        payload = _load_json(args.candidates_json)
        candidates.extend(list(payload.get("candidates") or []))
    candidates.extend(candidates_from_manifest_globs(args.manifest_glob))
    result = run_selection(candidates=candidates, output_dir=args.out)
    print(json.dumps({"manifest": str(args.out / "selection.json"), "status": result["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
