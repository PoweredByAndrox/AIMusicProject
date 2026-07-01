from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.no_reference_candidate_job_registry import (  # noqa: E402
    CandidateJobConfig,
    build_candidate_job_registry,
)


FORBIDDEN_ACTIONS = (
    "do_not_require_reference_stems",
    "do_not_require_human_csv",
    "do_not_train_separator",
    "do_not_download_models",
    "do_not_add_unsupported_models",
    "do_not_add_cleanup_debleed_denoise",
    "do_not_relax_thresholds",
    "do_not_mark_release_ready",
    "do_not_publish_final_result",
)
SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "separator_training_started": False,
    "model_downloaded": False,
    "model_registry_edited": False,
    "thresholds_relaxed": False,
    "cleanup_routes_added": False,
    "release_ready": False,
}
BLOCKER_ROUTE = {
    "pitch_guard_full_region_stalled": "train_separator_attribution_objective",
    "scalar_score_only_pitch_guard_progress": "train_separator_attribution_objective",
    "severe_phase_collapse": "generate_phase_safe_variants",
    "severe_phase_comb_filtering": "generate_phase_safe_variants",
    "severe_vocal_bleed_in_minus": "generate_minus_focused_variants",
    "missing_safe_mode_candidate": "generate_minus_focused_variants",
    "only_one_minus_family": "generate_minus_focused_variants",
    "low_minus_candidate_diversity": "generate_minus_focused_variants",
    "pair_compatibility_not_reached": "segment_blend_needed",
}
ROUTE_PRIORITY = (
    "train_separator_attribution_objective",
    "generate_phase_safe_variants",
    "generate_minus_focused_variants",
    "generate_artifact_safe_variants",
    "segment_blend_needed",
    "blocked_hard_case",
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def candidate_minus_families(candidate_pool_rows: Iterable[dict[str, Any]]) -> list[str]:
    families = {
        str(row.get("candidate_family") or "")
        for row in candidate_pool_rows
        if row.get("can_use_for_minus") is True or row.get("minus_path") or row.get("instrumental_path")
    }
    return sorted(family for family in families if family)


def candidate_minus_count(candidate_pool_rows: Iterable[dict[str, Any]]) -> int:
    return sum(
        1
        for row in candidate_pool_rows
        if row.get("can_use_for_minus") is True or row.get("minus_path") or row.get("instrumental_path")
    )


def _first_number(row: dict[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            try:
                return float(value)
            except ValueError:
                continue
    return None


def pitch_guard_stall_blocker(row: dict[str, Any]) -> str | None:
    before_regions = _first_number(
        row,
        (
            "pitch_guard_before_region_count",
            "before_region_count",
            "before_fail_regions",
            "before_total_fail_regions",
        ),
    )
    after_regions = _first_number(
        row,
        (
            "pitch_guard_after_region_count",
            "after_region_count",
            "after_fail_regions",
            "after_total_fail_regions",
            "pitch_guard_after_repair_region_count",
            "after_repair_region_count",
        ),
    )
    if before_regions is None or after_regions is None:
        return None
    if before_regions <= 0 or after_regions <= 0:
        return None
    if int(round(before_regions)) != int(round(after_regions)):
        return None

    before_score = _first_number(
        row,
        (
            "pitch_guard_before_best_score",
            "before_best_score",
            "before_top_score",
            "score_before_best",
        ),
    )
    after_score = _first_number(
        row,
        (
            "pitch_guard_after_best_score",
            "after_best_score",
            "after_top_score",
            "score_after_best",
            "after_repair_top_score",
        ),
    )
    if before_score is not None and after_score is not None and after_score < before_score:
        return "scalar_score_only_pitch_guard_progress"
    return "pitch_guard_full_region_stalled"


def blockers_from_minus_rows(rows: Iterable[dict[str, Any]]) -> list[tuple[str, str]]:
    blockers: list[tuple[str, str]] = []
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        values: list[Any] = []
        for key in ("minus_blockers", "blockers", "failure_reasons", "top_failure_reasons"):
            value = row.get(key)
            if isinstance(value, list):
                values.extend(value)
            elif isinstance(value, dict):
                values.extend(value.keys())
            elif value:
                values.append(value)
        for key in ("minus_dominant_defect", "dominant_defect", "failure_reason", "judge_reason"):
            if row.get(key):
                values.append(row[key])
        for value in values:
            text = str(value)
            if text in BLOCKER_ROUTE:
                blockers.append((text, candidate_id))
        stall_blocker = pitch_guard_stall_blocker(row)
        if stall_blocker is not None:
            blockers.append((stall_blocker, candidate_id))
    return blockers


def available_configs_for_route(route: str, registry_configs: Iterable[CandidateJobConfig]) -> list[str]:
    return sorted(
        config.candidate_config_id
        for config in registry_configs
        if route in set(config.hard_case_routes)
        and config.entrypoint_type != "unavailable"
        and config.available is True
    )


def choose_route(blocker_types: set[str], registry_configs: Iterable[CandidateJobConfig]) -> tuple[str, list[str], bool]:
    routes = {BLOCKER_ROUTE.get(blocker, "blocked_hard_case") for blocker in blocker_types}
    for route in ROUTE_PRIORITY:
        if route not in routes:
            continue
        available = available_configs_for_route(route, registry_configs)
        return route, available, bool(available)
    return "blocked_hard_case", [], False


def build_rows(
    *,
    run_dir: Path,
    registry_configs: Iterable[CandidateJobConfig],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = read_json(run_dir / "run_manifest.json")
    minus_rows = read_jsonl(run_dir / "minus_windowed_judge.jsonl")
    selected_rows = read_jsonl(run_dir / "selected_candidates.jsonl")
    pair_rows = read_jsonl(run_dir / "pair_compatibility.jsonl")
    candidate_rows = read_jsonl(run_dir / "candidate_pool.jsonl")
    minus_count = candidate_minus_count(candidate_rows)
    minus_families = candidate_minus_families(candidate_rows)
    detected = blockers_from_minus_rows(candidate_rows + minus_rows)
    if str(manifest.get("failure_reason") or "") == "missing_safe_mode_candidate" or "missing_safe_mode_candidate" in set(manifest.get("safety_blockers") or []):
        detected.append(("missing_safe_mode_candidate", ""))
    if minus_count > 0 and len(minus_families) <= 1:
        detected.append(("only_one_minus_family", ""))
        detected.append(("low_minus_candidate_diversity", ""))
    if not selected_rows or not pair_rows:
        detected.append(("pair_compatibility_not_reached", ""))
    blocker_types = {blocker for blocker, _ in detected}
    route, available_configs, can_run = choose_route(blocker_types, registry_configs)
    rows: list[dict[str, Any]] = []
    track_id = ""
    for source_row in candidate_rows + minus_rows:
        if source_row.get("track_id"):
            track_id = str(source_row["track_id"])
            break
    run_id = str(manifest.get("run_id") or run_dir.name)
    for blocker, candidate_id in sorted(set(detected)):
        recommended_route = BLOCKER_ROUTE.get(blocker, "blocked_hard_case")
        route_configs = available_configs_for_route(recommended_route, registry_configs)
        rows.append(
            {
                "track_id": track_id,
                "run_id": run_id,
                "blocker_type": blocker,
                "affected_candidate_id": candidate_id,
                "minus_candidate_count": minus_count,
                "minus_candidate_families": minus_families,
                "recommended_route": recommended_route,
                "route_reason": f"{blocker}_detected",
                "can_run_without_new_models": bool(route_configs),
                "available_route_configs": route_configs,
                "forbidden_actions": list(FORBIDDEN_ACTIONS),
                **SAFETY_FLAGS,
            }
        )
    if not rows:
        rows.append(
            {
                "track_id": track_id,
                "run_id": run_id,
                "blocker_type": "no_minus_hard_case_detected",
                "affected_candidate_id": "",
                "minus_candidate_count": minus_count,
                "minus_candidate_families": minus_families,
                "recommended_route": "blocked_hard_case",
                "route_reason": "no_supported_hard_case_signature",
                "can_run_without_new_models": False,
                "available_route_configs": [],
                "forbidden_actions": list(FORBIDDEN_ACTIONS),
                **SAFETY_FLAGS,
            }
        )
    summary = {
        "track_id": track_id,
        "run_id": run_id,
        "minus_candidate_count": minus_count,
        "minus_candidate_families": minus_families,
        "blocker_types": sorted(blocker_types),
        "recommended_route": route,
        "available_route_configs": available_configs,
        "can_run_without_new_models": can_run,
        "rows": len(rows),
        **SAFETY_FLAGS,
    }
    return rows, summary


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Minus Hard-Case Router Summary",
        "",
        f"- recommended_route: {summary.get('recommended_route')}",
        f"- can_run_without_new_models: {str(summary.get('can_run_without_new_models') is True).lower()}",
        f"- available_route_configs: {', '.join(summary.get('available_route_configs') or []) or 'unavailable'}",
        f"- minus_candidate_count: {summary.get('minus_candidate_count')}",
        f"- minus_candidate_families: {', '.join(summary.get('minus_candidate_families') or [])}",
        f"- blocker_types: {', '.join(summary.get('blocker_types') or [])}",
        "- release_ready: false",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_minus_hard_case_router(
    *,
    run_dir: Path,
    output_jsonl: Path,
    summary_md: Path,
    registry_configs: Iterable[CandidateJobConfig] | None = None,
) -> dict[str, Any]:
    configs = list(registry_configs) if registry_configs is not None else build_candidate_job_registry(profile="balanced")
    rows, summary = build_rows(run_dir=run_dir, registry_configs=configs)
    write_jsonl(output_jsonl, rows)
    write_summary(summary_md, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Route no-reference minus hard cases without changing safety gates.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--summary-md", type=Path)
    args = parser.parse_args(argv)
    output_jsonl = args.output_jsonl or (args.run_dir / "minus_hard_case_router.jsonl")
    summary_md = args.summary_md or (args.run_dir / "minus_hard_case_router_summary.md")
    summary = run_minus_hard_case_router(run_dir=args.run_dir, output_jsonl=output_jsonl, summary_md=summary_md)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
