from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.discover_no_reference_minus_routes import SAFETY_FLAGS  # noqa: E402

HARD_FORBIDDEN_TOKENS = ("cleanup", "debleed", "denoise", "threshold", "download", "cloud")
EXPANSION_JSONL_ARTIFACTS = (
    "minus_route_candidate_pool.jsonl",
    "minus_route_jobs.jsonl",
    "minus_route_pair_compatibility.jsonl",
    "minus_route_failures.jsonl",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_summary_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Minus Route Expansion Validation",
        "",
        f"- ok: {str(result.get('ok') is True).lower()}",
        f"- discovered_routes_total: {result.get('discovered_routes_total', 0)}",
        f"- available_routes_total: {result.get('available_routes_total', 0)}",
        f"- unavailable_routes_total: {result.get('unavailable_routes_total', 0)}",
        f"- route_space_limited: {str(result.get('route_space_limited') is True).lower()}",
        f"- route_space_exhausted: {str(result.get('route_space_exhausted') is True).lower()}",
        f"- jobs_attempted: {result.get('jobs_attempted', 0)}",
        f"- jobs_generated: {result.get('jobs_generated', 0)}",
        f"- jobs_failed: {result.get('jobs_failed', 0)}",
        f"- accepted_minus_candidates: {result.get('accepted_minus_candidates', 0)}",
        f"- severe_phase_collapse_count: {result.get('severe_phase_collapse_count', 0)}",
        f"- severe_vocal_bleed_in_minus_count: {result.get('severe_vocal_bleed_in_minus_count', 0)}",
        f"- phase_comb_filtering_count: {result.get('phase_comb_filtering_count', 0)}",
        f"- release_ready_count: {result.get('release_ready_count', 0)}",
        "",
        "## Blockers",
    ]
    blockers = list(result.get("blockers") or [])
    lines.extend(f"- {blocker}" for blocker in blockers) if blockers else lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def as_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    summary = manifest.get("summary")
    return summary if isinstance(summary, dict) else {}


def is_hard_forbidden_reason(reason: str) -> bool:
    text = str(reason or "").lower()
    if text.startswith("unsupported_cli_parameter:"):
        return False
    return any(token in text for token in HARD_FORBIDDEN_TOKENS)


def add_int(target: dict[str, Any], key: str, value: Any) -> None:
    target[key] = int(target.get(key) or 0) + int(value or 0)


def collect_expansion_rows(expansion_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = expansion_dir / "minus_route_expansion_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {"summary": {}}
    rows: list[dict[str, Any]] = []
    for filename in EXPANSION_JSONL_ARTIFACTS:
        for row in iter_jsonl(expansion_dir / filename) or []:
            rows.append({**row, "_artifact": filename})
    return manifest, rows


def validate_minus_route_expansion(
    *,
    discovery_json: Path,
    expansion_dirs: Sequence[Path],
    output_json: Path | None = None,
    summary_md: Path | None = None,
) -> dict[str, Any]:
    discovery = load_json(discovery_json)
    discovery_summary = as_summary(discovery)
    route_rows = list(discovery.get("routes") or [])
    blockers: list[str] = []
    forbidden_routes = [
        str(row.get("route_id") or row.get("candidate_config_id") or "unknown_route")
        for row in route_rows
        if is_hard_forbidden_reason(str(row.get("forbidden_reason") or ""))
    ]
    blockers.extend(f"forbidden_route:{route_id}" for route_id in forbidden_routes)

    result: dict[str, Any] = {
        "discovered_routes_total": int(discovery_summary.get("discovered_routes_total") or len(route_rows)),
        "available_routes_total": int(discovery_summary.get("available_routes_total") or sum(1 for row in route_rows if row.get("available") is True)),
        "unavailable_routes_total": int(discovery_summary.get("unavailable_routes_total") or sum(1 for row in route_rows if row.get("available") is not True)),
        "route_space_limited": bool(discovery_summary.get("route_space_limited") is True),
        "route_space_exhausted": True,
        "jobs_attempted": 0,
        "jobs_generated": 0,
        "jobs_failed": 0,
        "minus_candidates_generated": 0,
        "accepted_minus_candidates": 0,
        "severe_phase_collapse_count": 0,
        "severe_vocal_bleed_in_minus_count": 0,
        "phase_comb_filtering_count": 0,
        "pair_compatible_count": 0,
        "audio_preview_ready_count": 0,
        "analysis_ready_stems_count": 0,
        "blocked_stems_count": 0,
        "release_ready_count": 0,
        "quality_improved_route_ids": [],
        "rejected_route_ids": [],
        **SAFETY_FLAGS,
    }
    quality_ids: set[str] = set()
    rejected_ids: set[str] = set()
    release_ready_rows = 0
    cleanup_rows = 0
    threshold_rows = 0
    training_rows = 0
    model_download_rows = 0
    model_registry_rows = 0
    for expansion_dir in expansion_dirs:
        manifest, rows = collect_expansion_rows(expansion_dir)
        summary = as_summary(manifest)
        for key in (
            "jobs_attempted",
            "jobs_generated",
            "jobs_failed",
            "minus_candidates_generated",
            "accepted_minus_candidates",
            "severe_phase_collapse_count",
            "severe_vocal_bleed_in_minus_count",
            "phase_comb_filtering_count",
            "pair_compatible_count",
            "audio_preview_ready_count",
            "analysis_ready_stems_count",
            "blocked_stems_count",
            "release_ready_count",
        ):
            add_int(result, key, summary.get(key))
        if summary.get("route_space_exhausted") is False:
            result["route_space_exhausted"] = False
        quality_ids.update(str(value) for value in summary.get("quality_improved_route_ids") or [] if str(value))
        rejected_ids.update(str(value) for value in summary.get("rejected_route_ids") or [] if str(value))
        safety_rows = [manifest, summary, *rows]
        release_ready_rows += sum(1 for row in safety_rows if row.get("release_ready") is True)
        cleanup_rows += sum(
            1
            for row in safety_rows
            if row.get("cleanup_routes_added") is True or is_hard_forbidden_reason(str(row.get("forbidden_reason") or ""))
        )
        threshold_rows += sum(1 for row in safety_rows if row.get("thresholds_relaxed") is True)
        training_rows += sum(1 for row in safety_rows if row.get("training_started") is True or row.get("separator_training_started") is True)
        model_download_rows += sum(1 for row in safety_rows if row.get("model_downloaded") is True)
        model_registry_rows += sum(1 for row in safety_rows if row.get("model_registry_edited") is True)

    result["quality_improved_route_ids"] = sorted(quality_ids)
    result["rejected_route_ids"] = sorted(rejected_ids)
    result["release_ready_count"] = int(result.get("release_ready_count") or 0) + release_ready_rows
    if result["release_ready_count"]:
        blockers.append("release_ready_violation")
    if cleanup_rows:
        blockers.append("cleanup_route_violation")
    if threshold_rows:
        blockers.append("threshold_relaxation_violation")
    if training_rows:
        blockers.append("training_violation")
    if model_download_rows:
        blockers.append("model_download_violation")
    if model_registry_rows:
        blockers.append("model_registry_edit_violation")
    result["cleanup_routes_added"] = False
    result["thresholds_relaxed"] = False
    result["training_started"] = False
    result["separator_training_started"] = False
    result["release_ready"] = False
    result["blockers"] = sorted(set(blockers))
    result["ok"] = not result["blockers"]
    if output_json is not None:
        write_json(output_json, result)
    if summary_md is not None:
        write_summary_md(summary_md, result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate no-reference minus route expansion artifacts.")
    parser.add_argument("--discovery-json", type=Path, required=True)
    parser.add_argument("--expansion-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--summary-md", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = validate_minus_route_expansion(
        discovery_json=args.discovery_json,
        expansion_dirs=args.expansion_dir,
        output_json=args.output_json,
        summary_md=args.summary_md,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
