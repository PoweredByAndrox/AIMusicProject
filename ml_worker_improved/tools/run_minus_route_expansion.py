from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.discover_no_reference_minus_routes import (  # noqa: E402
    SAFETY_FLAGS,
    discover_minus_routes,
    write_discovery_outputs,
)
from tools.no_reference_candidate_job_registry import (  # noqa: E402
    CandidateJobConfig,
    build_candidate_job_registry,
)
from tools.no_reference_score_matrix import pair_compatibility_score  # noqa: E402
from tools.run_local_separator_candidate_job import (  # noqa: E402
    run_local_separator_candidate_job,
    validate_input_audio,
)
from tools.run_no_reference_candidate_wave import (  # noqa: E402
    build_pairs,
    job_manifest_to_assets,
    stable_id,
    write_json,
    write_jsonl,
)
from tools.run_no_reference_windowed_judge import write_outputs_streaming  # noqa: E402


DETERMINISTIC_CREATED_AT = "1970-01-01T00:00:00Z"
ALLOWED_PROFILES = {"safe", "balanced", "hard_case"}
PROFILE_TO_REGISTRY_PROFILE = {
    "safe": "safe",
    "balanced": "balanced",
    "hard_case": "exploration",
}
BLOCKERS = (
    "severe_phase_collapse",
    "severe_vocal_bleed_in_minus",
    "phase_comb_filtering",
)
JobRunner = Callable[..., dict[str, Any]]
JudgeWriter = Callable[..., dict[str, Any]]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return cleaned.strip("_") or "route"


def profile_to_registry_profile(profile: str) -> str:
    if profile not in ALLOWED_PROFILES:
        raise ValueError(f"Unsupported minus route expansion profile: {profile}")
    return PROFILE_TO_REGISTRY_PROFILE[profile]


def selected_routes(routes: Iterable[dict[str, Any]], max_routes: int) -> list[dict[str, Any]]:
    safe_routes = [
        row
        for row in routes
        if row.get("available") is True
        and row.get("supports_minus") is True
        and not str(row.get("forbidden_reason") or "")
        and row.get("release_ready") is not True
    ]
    safe_routes.sort(
        key=lambda row: (
            0 if row.get("supports_both") is True else 1,
            str(row.get("route_family") or ""),
            str(row.get("route_id") or ""),
        )
    )
    deduped: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    for row in safe_routes:
        group_id = str(row.get("shared_execution_group_id") or row.get("route_id") or row.get("candidate_config_id") or "")
        if group_id in seen_groups:
            continue
        seen_groups.add(group_id)
        deduped.append(row)
    limit = max(0, int(max_routes))
    return deduped[:limit] if limit else deduped


def load_or_create_discovery(
    *,
    output_dir: Path,
    route_discovery_json: Path | None,
    registry_configs: Iterable[CandidateJobConfig] | None,
    registry_profile: str,
) -> dict[str, Any]:
    if route_discovery_json is not None and route_discovery_json.exists():
        return load_json(route_discovery_json)
    manifest = discover_minus_routes(profile=registry_profile, registry_configs=registry_configs)
    write_discovery_outputs(output_dir, manifest)
    return manifest


def config_by_id(configs: Iterable[CandidateJobConfig]) -> dict[str, CandidateJobConfig]:
    return {config.candidate_config_id: config for config in configs}


def current_registry_route_blocker(route: dict[str, Any], config: CandidateJobConfig) -> str:
    route_id = str(route.get("route_id") or "")
    source_model_id = str(route.get("source_model_id") or "")
    source_tool = str(route.get("source_tool") or "")
    if route_id != config.route_id:
        return f"registry_route_mismatch:route_id:{route_id or 'missing'}"
    if source_model_id and source_model_id != config.source_model_id:
        return f"registry_route_mismatch:source_model_id:{source_model_id}"
    if source_tool and source_tool != config.source_tool:
        return f"registry_route_mismatch:source_tool:{source_tool}"
    if config.available is not True or config.entrypoint_type == "unavailable":
        reason = config.unavailable_reason or "unavailable_entrypoint"
        return f"registry_route_unavailable:{reason}"
    if config.forbidden_reason:
        return f"registry_forbidden_route:{config.forbidden_reason}"
    if config.can_run_without_download is not True or route.get("can_run_without_download") is not True:
        return "route_requires_download_or_missing_guard"
    if config.can_run_without_registry_edit is not True or route.get("can_run_without_registry_edit") is not True:
        return "route_requires_registry_edit_or_missing_guard"
    if config.produces_minus is not True:
        return "registry_route_does_not_produce_minus"
    return ""


def route_failure_row(route: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "route_id": str(route.get("route_id") or ""),
        "candidate_config_id": str(route.get("candidate_config_id") or ""),
        "generation_status": "skipped",
        "skip_reason": reason,
        "accepted_minus": False,
        "release_ready": False,
        **SAFETY_FLAGS,
    }


def count_blocker(rows: Iterable[dict[str, Any]], blocker: str) -> int:
    return sum(1 for row in rows if blocker in [str(value) for value in row.get("minus_blockers") or row.get("reasons") or []])


def augment_candidate_rows(
    rows: list[dict[str, Any]],
    judge_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    judge_by_id = {str(row.get("candidate_id") or ""): row for row in judge_rows}
    augmented: list[dict[str, Any]] = []
    for row in rows:
        judged = judge_by_id.get(str(row.get("candidate_id") or ""))
        if judged is None:
            decision = "not_judged_missing_vocal_pair"
            blockers = ["missing_safe_mode_candidate"] if not row.get("vocal_path") else []
            accepted = False
        else:
            decision = str(judged.get("minus_decision") or judged.get("decision") or "")
            blockers = [str(value) for value in judged.get("minus_blockers") or judged.get("reasons") or [] if str(value) != "no_reference_proxy_passed"]
            accepted = decision == "accept_candidate" and not blockers
        augmented.append(
            {
                **row,
                "minus_judge_decision": decision,
                "minus_blockers": blockers,
                "accepted_minus": bool(accepted),
                "release_ready": False,
                **SAFETY_FLAGS,
            }
        )
    return augmented


def pair_compatibility_rows(candidate_rows: Iterable[dict[str, Any]], judge_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    judged_by_id = {str(row.get("candidate_id") or ""): row for row in judge_rows}
    rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        if not row.get("vocal_path") or not row.get("minus_path"):
            continue
        metrics = judged_by_id.get(str(row.get("candidate_id") or ""), row)
        pair = pair_compatibility_score(metrics, metrics)
        rows.append(
            {
                "track_id": str(row.get("track_id") or ""),
                "candidate_id": str(row.get("candidate_id") or ""),
                "route_id": str(row.get("route_id") or ""),
                "candidate_config_id": str(row.get("candidate_config_id") or ""),
                **pair,
                "can_publish_as_result": False,
                "can_mark_release_ready": False,
                "release_ready": False,
            }
        )
    return rows


def build_summary(
    *,
    discovery: dict[str, Any],
    selected: list[dict[str, Any]],
    job_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    judge_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    runtime_sec: float,
) -> dict[str, Any]:
    discovery_summary = discovery.get("summary") if isinstance(discovery.get("summary"), dict) else {}
    accepted = [row for row in candidate_rows if row.get("accepted_minus") is True]
    selected_ids = [str(row.get("route_id") or "") for row in selected]
    quality_ids = sorted({str(row.get("route_id") or row.get("candidate_config_id") or "") for row in accepted})
    rejected_ids = sorted({route_id for route_id in selected_ids if route_id and route_id not in set(quality_ids)})
    release_ready_count = sum(1 for row in [*candidate_rows, *judge_rows, *pair_rows, *job_rows] if row.get("release_ready") is True)
    return {
        "discovered_routes_total": int(discovery_summary.get("discovered_routes_total") or 0),
        "available_routes_total": int(discovery_summary.get("available_routes_total") or 0),
        "unavailable_routes_total": int(discovery_summary.get("unavailable_routes_total") or 0),
        "route_space_limited": bool(discovery_summary.get("route_space_limited") is True),
        "route_space_exhausted": not quality_ids,
        "selected_route_ids": selected_ids,
        "jobs_attempted": len(job_rows),
        "jobs_generated": sum(1 for row in job_rows if str(row.get("generation_status") or "") in {"generated", "generated_with_shape_normalization", "generated_with_late_timeout"}),
        "jobs_failed": sum(1 for row in job_rows if str(row.get("generation_status") or "").startswith("failed")),
        "jobs_skipped": len(failures),
        "minus_candidates_generated": sum(1 for row in candidate_rows if row.get("minus_path")),
        "accepted_minus_candidates": len(accepted),
        "severe_phase_collapse_count": count_blocker(candidate_rows, "severe_phase_collapse"),
        "severe_vocal_bleed_in_minus_count": count_blocker(candidate_rows, "severe_vocal_bleed_in_minus"),
        "phase_comb_filtering_count": count_blocker(candidate_rows, "phase_comb_filtering"),
        "pair_compatible_count": sum(1 for row in pair_rows if row.get("pair_decision") == "pair_compatible"),
        "audio_preview_ready_count": 0,
        "analysis_ready_stems_count": 0,
        "blocked_stems_count": len([row for row in candidate_rows if row.get("accepted_minus") is not True]),
        "quality_improved_route_ids": quality_ids,
        "rejected_route_ids": rejected_ids,
        "release_ready_count": release_ready_count,
        "runtime_sec": round(runtime_sec, 6),
        **SAFETY_FLAGS,
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# No-Reference Minus Route Expansion Summary",
        "",
        f"- discovered_routes_total: {summary.get('discovered_routes_total', 0)}",
        f"- available_routes_total: {summary.get('available_routes_total', 0)}",
        f"- unavailable_routes_total: {summary.get('unavailable_routes_total', 0)}",
        f"- route_space_limited: {str(summary.get('route_space_limited') is True).lower()}",
        f"- route_space_exhausted: {str(summary.get('route_space_exhausted') is True).lower()}",
        f"- jobs_attempted: {summary.get('jobs_attempted', 0)}",
        f"- jobs_generated: {summary.get('jobs_generated', 0)}",
        f"- accepted_minus_candidates: {summary.get('accepted_minus_candidates', 0)}",
        f"- severe_phase_collapse_count: {summary.get('severe_phase_collapse_count', 0)}",
        f"- severe_vocal_bleed_in_minus_count: {summary.get('severe_vocal_bleed_in_minus_count', 0)}",
        f"- phase_comb_filtering_count: {summary.get('phase_comb_filtering_count', 0)}",
        f"- release_ready_count: {summary.get('release_ready_count', 0)}",
        f"- training_started: {str(summary.get('training_started') is True).lower()}",
        f"- model_downloaded: {str(summary.get('model_downloaded') is True).lower()}",
        f"- model_registry_edited: {str(summary.get('model_registry_edited') is True).lower()}",
        f"- thresholds_relaxed: {str(summary.get('thresholds_relaxed') is True).lower()}",
        f"- cleanup_routes_added: {str(summary.get('cleanup_routes_added') is True).lower()}",
        f"- release_ready: {str(summary.get('release_ready') is True).lower()}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_minus_route_expansion(
    *,
    input_audio: Path,
    output_dir: Path,
    route_discovery_json: Path | None = None,
    profile: str = "safe",
    max_routes: int = 6,
    max_runtime_sec: int = 2400,
    registry_configs: Iterable[CandidateJobConfig] | None = None,
    job_runner: JobRunner = run_local_separator_candidate_job,
    judge_writer: JudgeWriter = write_outputs_streaming,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_profile = profile_to_registry_profile(profile)
    configs = list(registry_configs) if registry_configs is not None else build_candidate_job_registry(profile=registry_profile)
    configs_by_id = config_by_id(configs)
    discovery = load_or_create_discovery(
        output_dir=output_dir,
        route_discovery_json=route_discovery_json,
        registry_configs=configs,
        registry_profile=registry_profile,
    )
    routes = list(discovery.get("routes") or [])
    selected = selected_routes(routes, max_routes=max_routes)
    input_validation = validate_input_audio(input_audio)
    run_id = stable_id(
        "minus_route_expansion",
        {
            "input_audio": str(input_audio),
            "profile": profile,
            "selected_route_ids": [row.get("route_id") for row in selected],
        },
    )
    track_id = "".join(ch if ch.isalnum() else "_" for ch in input_audio.stem).strip("_") or "track"
    job_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    assets: list[dict[str, Any]] = []
    if not input_validation["ok"]:
        failures.append(route_failure_row({"route_id": "input_audio"}, str(input_validation["reason"])))
    for route in selected:
        if not input_validation["ok"]:
            break
        config_id = str(route.get("candidate_config_id") or "")
        config = configs_by_id.get(config_id)
        if config is None:
            failures.append(route_failure_row(route, "candidate_config_missing_from_registry"))
            continue
        registry_blocker = current_registry_route_blocker(route, config)
        if registry_blocker:
            failures.append(route_failure_row(route, registry_blocker))
            continue
        job_dir = output_dir / "jobs" / safe_name(str(route.get("route_id") or config_id))
        manifest = job_runner(
            input_audio=input_audio,
            candidate_config_id=config_id,
            output_dir=job_dir,
            job_manifest_json=job_dir / "job_manifest.json",
            job_progress_jsonl=job_dir / "job_progress.jsonl",
            registry_configs=configs,
            profile=registry_profile,
            timeout_sec=min(int(config.timeout_sec), int(max_runtime_sec)) if max_runtime_sec > 0 else int(config.timeout_sec),
            allow_shape_normalization=True,
        )
        manifest.update(
            {
                "route_id": str(route.get("route_id") or config.route_id),
                "route_family": str(route.get("route_family") or config.route_family),
                "can_run_without_download": bool(config.can_run_without_download is True and route.get("can_run_without_download") is True),
                "can_run_without_registry_edit": bool(config.can_run_without_registry_edit is True and route.get("can_run_without_registry_edit") is True),
                "forbidden_reason": str(route.get("forbidden_reason") or ""),
                "release_ready": False,
            }
        )
        job_rows.append(manifest)
        materialized_modes = tuple(str(value) for value in route.get("materialized_candidate_modes") or config.materialized_candidate_modes or (config.target_mode,))
        job_assets, failure = job_manifest_to_assets(
            run_id=run_id,
            track_id=track_id,
            input_audio=input_audio,
            profile=profile,
            manifest=manifest,
            materialized_modes=materialized_modes,
        )
        assets.extend(job_assets)
        if failure is not None:
            failures.append(failure)

    candidate_rows = build_pairs(assets, max_candidates=max(0, int(max_routes)) or len(assets))
    judge_input_rows = [row for row in candidate_rows if row.get("vocal_path") and row.get("minus_path")]
    judge_result = {"rows": [], "summary": {}}
    if judge_input_rows:
        judge_result = judge_writer(
            judge_input_rows,
            output_jsonl=output_dir / "minus_route_judge_minus.jsonl",
            summary_md=output_dir / "minus_route_judge_minus_summary.md",
            cache_jsonl=output_dir / "minus_route_judge_cache.jsonl",
            window_seconds=12.0,
            max_windows_per_candidate=5,
            window_step_seconds=0.0,
            score_mode="minus",
        )
    else:
        write_jsonl(output_dir / "minus_route_judge_minus.jsonl", [])
        (output_dir / "minus_route_judge_minus_summary.md").write_text("# Minus Route Judge\n\n- candidates_judged: 0\n", encoding="utf-8")
        (output_dir / "minus_route_judge_cache.jsonl").write_text("", encoding="utf-8")
    judged_rows = list(judge_result.get("rows") or [])
    candidate_rows = augment_candidate_rows(candidate_rows, judged_rows)
    pair_rows = pair_compatibility_rows(candidate_rows, judged_rows)
    summary = build_summary(
        discovery=discovery,
        selected=selected,
        job_rows=job_rows,
        candidate_rows=candidate_rows,
        judge_rows=judged_rows,
        pair_rows=pair_rows,
        failures=failures,
        runtime_sec=time.monotonic() - started,
    )
    manifest = {
        "schema_version": 1,
        "created_at": DETERMINISTIC_CREATED_AT,
        "run_id": run_id,
        "input_audio_path": str(input_audio),
        "output_dir": str(output_dir),
        "profile": profile,
        "registry_profile": registry_profile,
        "max_routes": int(max_routes),
        "max_runtime_sec": int(max_runtime_sec),
        "route_discovery_json": str(route_discovery_json or output_dir / "minus_route_discovery.json"),
        "selected_routes": selected,
        "summary": summary,
        **SAFETY_FLAGS,
    }
    write_json(output_dir / "minus_route_expansion_manifest.json", manifest)
    write_jsonl(output_dir / "minus_route_jobs.jsonl", job_rows)
    write_jsonl(output_dir / "minus_route_candidate_pool.jsonl", candidate_rows)
    write_jsonl(output_dir / "minus_route_pair_compatibility.jsonl", pair_rows)
    write_jsonl(output_dir / "minus_route_failures.jsonl", failures)
    write_summary_md(output_dir / "minus_route_expansion_summary.md", summary)
    return {
        "manifest_path": str(output_dir / "minus_route_expansion_manifest.json"),
        "summary": summary,
        "candidate_rows": candidate_rows,
        "judge_rows": judged_rows,
        "pair_rows": pair_rows,
        "job_rows": job_rows,
        "failures": failures,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bounded no-reference minus route expansion.")
    parser.add_argument("--input-audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route-discovery-json", type=Path)
    parser.add_argument("--profile", choices=tuple(sorted(ALLOWED_PROFILES)), default="safe")
    parser.add_argument("--max-routes", type=int, default=6)
    parser.add_argument("--max-runtime-sec", type=int, default=2400)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_minus_route_expansion(
        input_audio=args.input_audio,
        output_dir=args.output_dir,
        route_discovery_json=args.route_discovery_json,
        profile=args.profile,
        max_routes=args.max_routes,
        max_runtime_sec=args.max_runtime_sec,
    )
    print(json.dumps({"manifest_path": result["manifest_path"], **result["summary"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
