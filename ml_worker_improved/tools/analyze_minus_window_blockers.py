from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_no_reference_ml_dataset import iter_jsonl, stable_unique, write_jsonl  # noqa: E402


SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "separator_training_started": False,
    "model_downloaded": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "melody_transcription_ran": False,
    "recommendation_index_updated": False,
    "release_ready": False,
}

SEVERE_PHASE_COLLAPSE = "severe_phase_collapse"
SEVERE_VOCAL_BLEED = "severe_vocal_bleed_in_minus"
SEVERE_PHASE_COMB_FILTERING = "severe_phase_comb_filtering"
PHASE_COMB_FILTERING = "phase_comb_filtering"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    text = str(value).strip()
    if not text:
        return []
    if "," in text or "+" in text:
        normalized = text.replace("+", ",")
        return [part.strip() for part in normalized.split(",") if part.strip()]
    return [text]


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def first_non_empty(*values: Any, default: str = "") -> str:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return default


def merge_candidate_metadata(candidate_row: dict[str, Any] | None, judge_row: dict[str, Any]) -> dict[str, Any]:
    merged = dict(candidate_row or {})
    for key, value in judge_row.items():
        if value is None or value == "":
            continue
        merged[key] = value
    return merged


def window_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = row.get("window_metrics")
    if isinstance(metrics, list) and metrics:
        return [item for item in metrics if isinstance(item, dict)]
    windows = row.get("windows")
    if isinstance(windows, list) and windows and all(isinstance(item, dict) for item in windows):
        return [item for item in windows if isinstance(item, dict)]
    return [dict(row)]


def blockers_for_window(window: dict[str, Any], parent: dict[str, Any]) -> list[str]:
    for key in ("minus_blockers", "blockers", "failure_reasons"):
        if key in window:
            return stable_unique(as_list(window.get(key)))
    blockers: list[str] = []
    for key in ("minus_blockers", "blockers", "failure_reasons"):
        blockers.extend(as_list(parent.get(key)))
    return stable_unique(blockers)


def blocker_signature(blockers: Iterable[str]) -> str:
    normalized = sorted(stable_unique(str(blocker) for blocker in blockers if str(blocker)))
    return "+".join(normalized) if normalized else "clean"


def window_index_for(index: int, window: dict[str, Any]) -> int:
    for key in ("window_index", "index"):
        if key in window:
            try:
                return int(window.get(key))
            except (TypeError, ValueError):
                break
    return int(index)


def window_start_for(window: dict[str, Any], parent: dict[str, Any]) -> float:
    for key in ("window_start_sec", "start_sec"):
        if key in window:
            return round(as_float(window.get(key)), 6)
    if "start_sec" in parent:
        return round(as_float(parent.get("start_sec")), 6)
    return 0.0


def window_end_for(start_sec: float, window: dict[str, Any], parent: dict[str, Any]) -> float:
    for key in ("window_end_sec", "end_sec"):
        if key in window:
            return round(as_float(window.get(key)), 6)
    duration = 0.0
    for key in ("window_duration_sec", "duration_sec", "window_seconds"):
        if key in window:
            duration = as_float(window.get(key))
            break
    if duration == 0.0:
        duration = as_float(parent.get("window_seconds"), as_float(parent.get("duration_sec"), 0.0))
    return round(float(start_sec) + float(duration), 6)


def score_components_for(window: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    for source in (window, parent):
        value = source.get("minus_score_components")
        if isinstance(value, dict):
            return dict(value)
    return {}


def score_for(window: dict[str, Any], parent: dict[str, Any]) -> float:
    for source in (window, parent):
        for key in ("minus_score", "window_fitness", "overall_no_reference_score"):
            if key in source:
                return round(as_float(source.get(key)), 6)
    return 0.0


def candidate_map_for(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = first_existing_path(run_dir, ["candidate_pool.jsonl", "minus_route_candidate_pool.jsonl"])
    if path is None:
        return {}
    rows = list(iter_jsonl(path))
    return {str(row.get("candidate_id") or ""): row for row in rows if str(row.get("candidate_id") or "")}


def jsonl_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    return sum(1 for _ in iter_jsonl(path))


def input_artifact_inventory(run_dir: Path) -> dict[str, dict[str, Any]]:
    names = [
        "run_manifest.json",
        "runtime_summary.md",
        "candidate_pool.jsonl",
        "minus_route_candidate_pool.jsonl",
        "candidate_wave_jobs.jsonl",
        "candidate_job_plan.jsonl",
        "hard_case_route_policy.json",
        "hard_case_route_policy_summary.md",
        "minus_windowed_judge.jsonl",
        "minus_route_judge_minus.jsonl",
        "vocal_windowed_judge.jsonl",
        "selected_candidates.jsonl",
        "pair_compatibility.jsonl",
        "preview_manifest.json",
        "music_intelligence_input_manifest.json",
    ]
    inventory: dict[str, dict[str, Any]] = {}
    for name in names:
        path = run_dir / name
        item = {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}
        if path.suffix == ".jsonl":
            item["row_count"] = jsonl_count(path)
        elif path.suffix == ".json":
            item["json_readable"] = bool(read_json(path))
        elif path.suffix == ".md":
            item["text_readable"] = path.exists()
        inventory[name] = item
    return inventory


def first_existing_path(run_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        path = run_dir / name
        if path.exists():
            return path
    return None


def minus_route_id_for(merged: dict[str, Any]) -> str:
    return first_non_empty(
        merged.get("minus_route_id"),
        merged.get("route_id"),
        merged.get("candidate_config_id"),
        default="unknown_route",
    )


def route_role_for(merged: dict[str, Any], minus_route_id: str) -> str:
    candidate_route_id = str(merged.get("route_id") or "")
    pair_route_id = str(merged.get("pair_route_id") or "")
    if pair_route_id or ("+" in candidate_route_id and candidate_route_id != minus_route_id):
        return "pair_minus_component"
    return "direct_minus_candidate"


def profile_rows_for(run_dir: Path) -> list[dict[str, Any]]:
    judge_path = first_existing_path(run_dir, ["minus_windowed_judge.jsonl", "minus_route_judge_minus.jsonl"])
    if judge_path is None:
        return []
    run_manifest = read_json(run_dir / "run_manifest.json")
    candidates = candidate_map_for(run_dir)
    rows: list[dict[str, Any]] = []
    for judge_row in iter_jsonl(judge_path):
        candidate_id = str(judge_row.get("candidate_id") or "")
        merged = merge_candidate_metadata(candidates.get(candidate_id), judge_row)
        windows = window_items(judge_row)
        for index, window in enumerate(windows):
            blockers = blockers_for_window(window, judge_row)
            signature = blocker_signature(blockers)
            window_index = window_index_for(index, window)
            start_sec = window_start_for(window, judge_row)
            end_sec = window_end_for(start_sec, window, judge_row)
            severe_phase_comb = SEVERE_PHASE_COMB_FILTERING in blockers or PHASE_COMB_FILTERING in blockers
            minus_route_id = minus_route_id_for(merged)
            rows.append(
                {
                    "run_id": first_non_empty(merged.get("run_id"), run_manifest.get("run_id"), run_dir.name),
                    "track_id": first_non_empty(merged.get("track_id"), run_manifest.get("track_id"), run_dir.name),
                    "candidate_id": candidate_id,
                    "route_id": minus_route_id,
                    "candidate_route_id": first_non_empty(merged.get("route_id"), default=minus_route_id),
                    "vocal_route_id": first_non_empty(merged.get("vocal_route_id")),
                    "minus_route_id": minus_route_id,
                    "pair_route_id": first_non_empty(merged.get("pair_route_id")),
                    "route_role": route_role_for(merged, minus_route_id),
                    "candidate_family": first_non_empty(merged.get("candidate_family"), default="unknown_family"),
                    "source_model_id": first_non_empty(merged.get("source_model_id"), default="unknown_model"),
                    "source_tool": first_non_empty(merged.get("source_tool"), default="unknown_tool"),
                    "window_index": window_index,
                    "window_start_sec": start_sec,
                    "window_end_sec": end_sec,
                    "minus_judge_decision": first_non_empty(
                        window.get("minus_decision"),
                        window.get("decision"),
                        judge_row.get("minus_decision"),
                        judge_row.get("decision"),
                        default="unknown",
                    ),
                    "blockers": blockers,
                    "severe_phase_collapse": SEVERE_PHASE_COLLAPSE in blockers,
                    "severe_vocal_bleed_in_minus": SEVERE_VOCAL_BLEED in blockers,
                    "severe_phase_comb_filtering": severe_phase_comb,
                    "blocker_count": len(blockers),
                    "blocker_signature": signature,
                    "score_components": score_components_for(window, judge_row),
                    "minus_score": score_for(window, judge_row),
                    "raw_minus_path": first_non_empty(merged.get("raw_minus_path"), merged.get("minus_path")),
                    "normalized_minus_path": first_non_empty(merged.get("normalized_minus_path"), merged.get("minus_path")),
                    "shape_normalization_applied": as_bool(merged.get("shape_normalization_applied")),
                    "shape_normalization_reason": first_non_empty(merged.get("shape_normalization_reason")),
                    "audio_shape_info": merged.get("audio_shape_info") if isinstance(merged.get("audio_shape_info"), dict) else {},
                    "release_ready": False,
                }
            )
    rows.sort(key=lambda row: (row["candidate_id"], row["window_index"], row["route_id"], row["blocker_signature"]))
    return rows


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "window_count": len(rows),
        "blocked_windows": sum(1 for row in rows if int(row.get("blocker_count") or 0) > 0),
        "clean_windows": sum(1 for row in rows if int(row.get("blocker_count") or 0) == 0),
        "severe_phase_collapse_count": sum(1 for row in rows if row.get("severe_phase_collapse") is True),
        "severe_vocal_bleed_in_minus_count": sum(1 for row in rows if row.get("severe_vocal_bleed_in_minus") is True),
        "severe_phase_comb_filtering_count": sum(1 for row in rows if row.get("severe_phase_comb_filtering") is True),
        "blocker_signatures": sorted({str(row.get("blocker_signature") or "clean") for row in rows}),
    }


def average_blocker_count(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return round(sum(float(row.get("blocker_count") or 0) for row in rows) / float(len(rows)), 6)


def build_clusters(rows: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    signature_routes: dict[str, set[str]] = defaultdict(set)
    blocker_routes: dict[str, set[str]] = defaultdict(set)
    blocker_models: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        route_id = str(row.get("route_id") or "")
        window_key = str(row.get("window_index"))
        by_candidate[candidate_id].append(row)
        by_route[route_id].append(row)
        by_window[window_key].append(row)
        signature_routes[str(row.get("blocker_signature") or "clean")].add(route_id)
        for blocker in as_list(row.get("blockers")):
            blocker_routes[blocker].add(route_id)
            blocker_models[blocker][str(row.get("source_model_id") or "unknown_model")] += 1

    normalized_rows = [row for row in rows if row.get("shape_normalization_applied") is True]
    raw_rows = [row for row in rows if row.get("shape_normalization_applied") is not True]
    normalization_correlated = bool(
        normalized_rows
        and raw_rows
        and average_blocker_count(normalized_rows) > average_blocker_count(raw_rows)
    )
    summary = {
        "run_dir": str(run_dir),
        "candidates_analyzed": len({str(row.get("candidate_id") or "") for row in rows if str(row.get("candidate_id") or "")}),
        "routes_analyzed": len({str(row.get("route_id") or "") for row in rows if str(row.get("route_id") or "")}),
        "windows_analyzed": len(rows),
        "severe_phase_collapse_total": sum(1 for row in rows if row.get("severe_phase_collapse") is True),
        "severe_vocal_bleed_in_minus_total": sum(1 for row in rows if row.get("severe_vocal_bleed_in_minus") is True),
        "severe_phase_comb_filtering_total": sum(1 for row in rows if row.get("severe_phase_comb_filtering") is True),
        "blocker_signatures": sorted({str(row.get("blocker_signature") or "clean") for row in rows}),
        "global_all_windows": bool(rows) and all(int(row.get("blocker_count") or 0) > 0 for row in rows),
        "localized_windows": bool(rows)
        and any(int(row.get("blocker_count") or 0) == 0 for row in rows)
        and any(int(row.get("blocker_count") or 0) > 0 for row in rows),
        "route_specific": sorted(
            blocker for blocker, routes in blocker_routes.items() if len(routes) == 1 and len(by_route) > 1
        ),
        "shared_across_routes": sorted(
            signature for signature, routes in signature_routes.items() if signature != "clean" and len(routes) > 1
        ),
        "normalization_correlated": normalization_correlated,
        "normalization_evidence": {
            "shape_normalized_windows": len(normalized_rows),
            "non_normalized_windows": len(raw_rows),
            "shape_normalized_avg_blocker_count": average_blocker_count(normalized_rows),
            "non_normalized_avg_blocker_count": average_blocker_count(raw_rows),
        },
        "source_model_correlated": {
            blocker: dict(sorted(counter.items())) for blocker, counter in sorted(blocker_models.items())
        },
        "input_artifacts": input_artifact_inventory(run_dir),
        **SAFETY_FLAGS,
    }
    clusters = {
        "summary": summary,
        "by_candidate": {key: summarize_group(value) for key, value in sorted(by_candidate.items())},
        "by_route": {key: summarize_group(value) for key, value in sorted(by_route.items())},
        "by_window": {
            key: {
                **summarize_group(value),
                "window_start_sec": min(float(row.get("window_start_sec") or 0.0) for row in value),
                "window_end_sec": max(float(row.get("window_end_sec") or 0.0) for row in value),
                "routes": sorted({str(row.get("route_id") or "") for row in value}),
            }
            for key, value in sorted(by_window.items(), key=lambda item: int(item[0]))
        },
    }
    return clusters


def build_window_blocker_profile(run_dir: Path) -> dict[str, Any]:
    rows = profile_rows_for(run_dir)
    clusters = build_clusters(rows, run_dir)
    return {"rows": rows, "clusters": clusters, "summary": clusters["summary"]}


def build_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Minus Window Blocker Profile",
        "",
        f"- candidates_analyzed: {summary.get('candidates_analyzed', 0)}",
        f"- routes_analyzed: {summary.get('routes_analyzed', 0)}",
        f"- windows_analyzed: {summary.get('windows_analyzed', 0)}",
        f"- severe_phase_collapse_total: {summary.get('severe_phase_collapse_total', 0)}",
        f"- severe_vocal_bleed_in_minus_total: {summary.get('severe_vocal_bleed_in_minus_total', 0)}",
        f"- severe_phase_comb_filtering_total: {summary.get('severe_phase_comb_filtering_total', 0)}",
        f"- blocker_signatures: {', '.join(summary.get('blocker_signatures') or [])}",
        f"- global_all_windows: {str(summary.get('global_all_windows') is True).lower()}",
        f"- localized_windows: {str(summary.get('localized_windows') is True).lower()}",
        f"- normalization_correlated: {str(summary.get('normalization_correlated') is True).lower()}",
        "- release_ready: false",
        "- melody_transcription_ran: false",
        "- recommendation_index_updated: false",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "minus_window_blocker_profile.jsonl", result["rows"])
    write_json(output_dir / "minus_window_blocker_clusters.json", result["clusters"])
    (output_dir / "minus_window_blocker_summary.md").write_text(
        build_summary_markdown(result["summary"]),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile no-reference minus blockers at candidate/window level.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_window_blocker_profile(args.run_dir)
    write_outputs(result, args.output_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
