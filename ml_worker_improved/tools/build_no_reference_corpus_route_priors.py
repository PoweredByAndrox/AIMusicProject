from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402


ACCEPT_DECISIONS = {"accept_candidate", "accepted", "pair_compatible"}
LEAKAGE_BLOCKERS = {"severe_vocal_bleed_in_minus", "vocal_leakage_in_minus"}
LEAKAGE_REASONS = {"severe_vocal_bleed_in_minus", "vocal_leakage_in_minus"}
SELF_JUDGE_BLOCKED_REASON = "self_judge_proxy_blocked"
SELF_JUDGE_LEAKAGE_ISSUES = {"vocal_stem_corr_proxy", "vocal_projection_proxy"}
DERIVED_MINUS_ROUTE_SUFFIXES = ("_derived_non_vocal_sum", "_derived_minus")
MINUS_SOURCE_POOL_MIN_SCORE = 0.80
MINUS_SOURCE_POOL_MAX_VOCAL_LEAKAGE = 0.35
MINUS_SOURCE_POOL_MIN_MIXTURE_CONSISTENCY = 0.85

SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "audio_exported": False,
    "audio_copied": False,
    "release_ready": False,
}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def stable_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def sorted_counter(counter: Counter[str], limit: int = 10) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit])


def is_holdout(row: dict[str, Any]) -> bool:
    split = row.get("evaluation_split") or row.get("feedback_split") or row.get("split")
    return bool(row.get("holdout") is True or (isinstance(split, str) and split.lower() == "holdout"))


def decision_is_accept(decision: str) -> bool:
    return str(decision) in ACCEPT_DECISIONS


def decision_is_blocked(decision: str) -> bool:
    text = str(decision)
    return "blocked" in text or text.startswith("reject")


def pair_is_blocked(row: dict[str, Any]) -> bool:
    return decision_is_blocked(str(row.get("pair_decision") or "")) or bool(as_list(row.get("pair_blockers")))


def result_judge(row: dict[str, Any]) -> dict[str, Any]:
    return as_dict(row.get("result_judge"))


def self_judge(row: dict[str, Any]) -> dict[str, Any]:
    return as_dict(row.get("self_judge"))


def self_judge_blocks_minus(row: dict[str, Any]) -> bool:
    judge = self_judge(row)
    if not judge:
        return False
    if judge.get("passed") is False:
        return True
    if as_float(judge.get("blocker_count")) > 0:
        return True
    return "blocked" in str(judge.get("status") or "")


def self_judge_issue_labels(row: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    top_blockers = self_judge(row).get("top_blockers", [])
    if not isinstance(top_blockers, list):
        return labels
    for blocker in top_blockers:
        if isinstance(blocker, str):
            labels.append(blocker)
            continue
        if not isinstance(blocker, dict):
            continue
        labels.extend(as_list(blocker.get("issues")))
    return stable_unique(labels)


def infer_route_from_path(path_text: str, fallback: str) -> str:
    if not path_text:
        return fallback
    normalized = path_text.replace("\\", "/")
    path = PurePosixPath(normalized)
    parent = str(path.parent.name or "").strip()
    if parent and parent not in {".", "/", "vocal", "vocals", "minus", "instrumental"}:
        return parent
    stem = str(path.stem or "").strip()
    return stem or fallback


def canonical_route_id(route_id: str, stem_type: str) -> str:
    text = str(route_id or "").strip()
    if stem_type == "minus":
        for suffix in DERIVED_MINUS_ROUTE_SUFFIXES:
            if text.endswith(suffix) and len(text) > len(suffix):
                return text[: -len(suffix)]
    return text


def route_id_for(row: dict[str, Any], stem_type: str) -> str:
    explicit_keys = (
        f"{stem_type}_route_id",
        f"{stem_type}_route",
        f"{stem_type}_candidate_family",
        f"{stem_type}_model_id",
    )
    for key in explicit_keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return canonical_route_id(value.strip(), stem_type)
    if stem_type == "vocal":
        return canonical_route_id(infer_route_from_path(str(row.get("vocal_path") or ""), "unknown_vocal_route"), stem_type)
    if stem_type == "minus":
        for value in (
            row.get("model_name"),
            row.get("model_id"),
            result_judge(row).get("model_name"),
            result_judge(row).get("model_id"),
            result_judge(row).get("candidate_id"),
            self_judge(row).get("candidate_label"),
        ):
            if isinstance(value, str) and value.strip():
                return canonical_route_id(value.strip(), stem_type)
        return canonical_route_id(
            infer_route_from_path(str(row.get("minus_path") or row.get("instrumental_path") or ""), "unknown_minus_route"),
            stem_type,
        )
    return canonical_route_id(str(row.get("pair_route_id") or row.get("candidate_id") or "unknown_pair_route"), stem_type)


def row_score_mode(row: dict[str, Any]) -> str:
    mode = str(first_present(row.get("score_mode"), result_judge(row).get("score_mode"), row.get("target_stem")) or "").strip().lower()
    return mode if mode in {"vocal", "minus", "both"} else "both"


def stem_decision(row: dict[str, Any], stem_type: str) -> str:
    if stem_type == "vocal":
        return str(first_present(row.get("vocal_decision"), result_judge(row).get("vocal_decision"), row.get("decision"), result_judge(row).get("decision")) or "")
    if stem_type == "minus":
        if pair_is_blocked(row):
            return str(row.get("pair_decision") or "blocked_pair_incompatible")
        if self_judge_blocks_minus(row):
            return "blocked_self_judge"
        return str(first_present(row.get("minus_decision"), result_judge(row).get("minus_decision"), row.get("decision"), result_judge(row).get("decision")) or "")
    return str(first_present(row.get("pair_decision"), result_judge(row).get("pair_decision"), row.get("decision"), result_judge(row).get("decision")) or "")


def stem_score(row: dict[str, Any], stem_type: str) -> float:
    if stem_type == "vocal":
        return as_float(first_present(row.get("vocal_score"), result_judge(row).get("vocal_score")))
    if stem_type == "minus":
        return as_float(first_present(row.get("minus_score"), result_judge(row).get("minus_score")))
    return as_float(first_present(row.get("pair_compatibility_score"), result_judge(row).get("pair_compatibility_score")))


def stem_blockers(row: dict[str, Any], stem_type: str) -> list[str]:
    if stem_type == "vocal":
        return stable_unique(as_list(row.get("vocal_blockers")) + as_list(result_judge(row).get("vocal_blockers")))
    if stem_type == "minus":
        blockers = stable_unique(
            as_list(row.get("minus_blockers"))
            + as_list(result_judge(row).get("minus_blockers"))
            + as_list(row.get("pair_blockers"))
            + as_list(result_judge(row).get("pair_blockers"))
            + [reason for reason in stable_unique(as_list(row.get("reasons")) + as_list(result_judge(row).get("reasons"))) if reason in LEAKAGE_REASONS]
        )
        if self_judge_blocks_minus(row):
            blockers = stable_unique([*blockers, SELF_JUDGE_BLOCKED_REASON, *self_judge_issue_labels(row)])
            if SELF_JUDGE_LEAKAGE_ISSUES & set(blockers):
                blockers = stable_unique([*blockers, "vocal_leakage_in_minus"])
        return blockers
    return as_list(row.get("pair_blockers"))


def observed_stems_for_row(row: dict[str, Any]) -> list[str]:
    mode = row_score_mode(row)
    if mode == "vocal":
        return ["vocal"]
    if mode == "minus":
        return ["minus"]
    return ["vocal", "minus", "pair"]


def build_observation(row: dict[str, Any], stem_type: str) -> dict[str, Any]:
    nested_result = result_judge(row)
    reasons = stable_unique(as_list(row.get("reasons")) + as_list(nested_result.get("reasons")))
    if stem_type == "minus" and pair_is_blocked(row):
        reasons = stable_unique([*reasons, "pair_incompatible"])
    if stem_type == "minus" and self_judge_blocks_minus(row):
        reasons = stable_unique([*reasons, SELF_JUDGE_BLOCKED_REASON])
        if SELF_JUDGE_LEAKAGE_ISSUES & set(self_judge_issue_labels(row)):
            reasons = stable_unique([*reasons, "vocal_leakage_in_minus"])
    return {
        "track_id": str(row.get("track_id") or ""),
        "candidate_id": str(first_present(row.get("candidate_id"), row.get("canonical_candidate_id"), nested_result.get("candidate_id")) or ""),
        "stem_type": stem_type,
        "route_id": route_id_for(row, stem_type),
        "mix_path": str(first_present(row.get("mix_path"), nested_result.get("mix_path")) or ""),
        "vocal_path": str(first_present(row.get("vocal_path"), nested_result.get("vocal_path")) or ""),
        "minus_path": str(first_present(row.get("minus_path"), row.get("instrumental_path"), nested_result.get("minus_path"), nested_result.get("instrumental_path")) or ""),
        "score_mode": row_score_mode(row),
        "decision": stem_decision(row, stem_type),
        "score": stem_score(row, stem_type),
        "blockers": stem_blockers(row, stem_type),
        "reasons": reasons,
        "vocal_leakage_in_minus_score": as_float(first_present(row.get("vocal_leakage_in_minus_score"), nested_result.get("vocal_leakage_in_minus_score"))),
        "mixture_consistency_score": as_float(first_present(row.get("mixture_consistency_score"), nested_result.get("mixture_consistency_score"))),
        "pair_decision": str(first_present(row.get("pair_decision"), nested_result.get("pair_decision")) or ""),
        "vocal_decision": str(first_present(row.get("vocal_decision"), nested_result.get("vocal_decision")) or ""),
        "minus_decision": str(first_present(row.get("minus_decision"), nested_result.get("minus_decision")) or ""),
    }


def has_minus_leakage_failure(observation: dict[str, Any]) -> bool:
    labels = set(as_list(observation.get("blockers"))) | set(as_list(observation.get("reasons")))
    return bool(labels & (LEAKAGE_BLOCKERS | LEAKAGE_REASONS))


def minus_source_pool_evidence(observations: list[dict[str, Any]]) -> dict[str, Any]:
    supported: list[dict[str, Any]] = []
    for observation in observations:
        score = as_float(observation.get("score"))
        vocal_leakage = as_float(observation.get("vocal_leakage_in_minus_score"))
        mixture_consistency = as_float(observation.get("mixture_consistency_score"))
        if score < MINUS_SOURCE_POOL_MIN_SCORE:
            continue
        if vocal_leakage > MINUS_SOURCE_POOL_MAX_VOCAL_LEAKAGE:
            continue
        if mixture_consistency < MINUS_SOURCE_POOL_MIN_MIXTURE_CONSISTENCY:
            continue
        if has_minus_leakage_failure(observation):
            continue
        supported.append(observation)

    if not supported:
        return {
            "source_pool_eligible": False,
            "source_pool_status": "no_minus_source_candidate_supported_for_repair",
            "source_pool_reasons": ["no_low_leakage_high_score_minus_source_observed"],
            "best_source_track_id": "",
            "best_source_candidate_id": "",
            "best_source_mix_path": "",
            "best_source_vocal_path": "",
            "best_source_audio_path": "",
            "best_source_score": 0.0,
            "best_source_vocal_leakage_in_minus_score": 0.0,
            "best_source_mixture_consistency_score": 0.0,
        }

    best = sorted(
        supported,
        key=lambda item: (
            -as_float(item.get("score")),
            as_float(item.get("vocal_leakage_in_minus_score")),
            -as_float(item.get("mixture_consistency_score")),
            str(item.get("candidate_id") or ""),
        ),
    )[0]
    return {
        "source_pool_eligible": True,
        "source_pool_status": "minus_source_candidate_supported_for_repair",
        "source_pool_reasons": [
            "low_vocal_leakage_minus_source_available",
            "high_minus_score_minus_source_available",
            "repair_source_only_not_delivery_candidate",
        ],
        "best_source_track_id": str(best.get("track_id") or ""),
        "best_source_candidate_id": str(best.get("candidate_id") or ""),
        "best_source_mix_path": str(best.get("mix_path") or ""),
        "best_source_vocal_path": str(best.get("vocal_path") or ""),
        "best_source_audio_path": str(best.get("minus_path") or ""),
        "best_source_score": round(as_float(best.get("score")), 6),
        "best_source_vocal_leakage_in_minus_score": round(as_float(best.get("vocal_leakage_in_minus_score")), 6),
        "best_source_mixture_consistency_score": round(as_float(best.get("mixture_consistency_score")), 6),
    }


def source_pool_evidence_for(stem_type: str, observations: list[dict[str, Any]]) -> dict[str, Any]:
    if stem_type == "minus":
        return minus_source_pool_evidence(observations)
    return {
        "source_pool_eligible": False,
        "source_pool_status": "not_applicable_for_stem_type",
        "source_pool_reasons": [],
        "best_source_track_id": "",
        "best_source_candidate_id": "",
        "best_source_mix_path": "",
        "best_source_vocal_path": "",
        "best_source_audio_path": "",
        "best_source_score": 0.0,
        "best_source_vocal_leakage_in_minus_score": 0.0,
        "best_source_mixture_consistency_score": 0.0,
    }


def recommendation_for(
    *,
    stem_type: str,
    candidate_count: int,
    accepted_count: int,
    blocked_count: int,
    blockers: Counter[str],
    reasons: Counter[str],
    min_evidence: int,
) -> tuple[str, bool, list[str]]:
    if candidate_count < min_evidence:
        return "insufficient_evidence", False, [f"candidate_count<{min_evidence}"]
    if stem_type == "minus":
        leakage_failures = sum(blockers.get(label, 0) for label in LEAKAGE_BLOCKERS) + sum(reasons.get(label, 0) for label in LEAKAGE_REASONS)
        pair_failures = reasons.get("pair_incompatible", 0)
        if accepted_count == 0 and blocked_count > 0 and leakage_failures >= max(1, blocked_count):
            return (
                "blocked_as_minus_baseline",
                True,
                ["repeated_vocal_leakage_in_minus", "do_not_promote_best_of_bad_minus"],
            )
        if accepted_count == 0 and blocked_count > 0 and pair_failures >= max(1, blocked_count):
            return (
                "blocked_as_minus_baseline",
                True,
                ["repeated_pair_incompatible_minus", "do_not_promote_best_of_bad_minus"],
            )
    if stem_type == "vocal" and accepted_count > 0 and accepted_count >= blocked_count:
        return "keep_as_vocal_candidate_source", False, ["vocal_route_has_accepted_no_reference_rows"]
    if stem_type == "pair" and accepted_count == 0 and blocked_count > 0:
        return "pair_reconciliation_required", False, ["pair_rows_blocked_after_mode_specific_scoring"]
    if accepted_count > 0:
        return "candidate_source_supported", False, ["route_has_accepted_no_reference_rows"]
    if blocked_count >= candidate_count:
        return "blocked_route", False, ["all_non_holdout_rows_blocked"]
    return "mixed_evidence_monitor", False, ["mixed_route_evidence"]


def build_route_row(
    *,
    stem_type: str,
    route_id: str,
    observations: list[dict[str, Any]],
    holdout_observations: list[dict[str, Any]],
    min_evidence: int,
) -> dict[str, Any]:
    decisions = Counter(str(item["decision"]) for item in observations)
    blockers = Counter(blocker for item in observations for blocker in as_list(item.get("blockers")))
    reasons = Counter(reason for item in observations for reason in as_list(item.get("reasons")))
    accepted_count = sum(1 for item in observations if decision_is_accept(str(item["decision"])))
    blocked_count = sum(1 for item in observations if decision_is_blocked(str(item["decision"])))
    scores = [as_float(item.get("score")) for item in observations]
    leakages = [as_float(item.get("vocal_leakage_in_minus_score")) for item in observations if stem_type == "minus"]
    source_pool_evidence = source_pool_evidence_for(stem_type, observations)
    recommendation, requires_new_source, recommendation_reasons = recommendation_for(
        stem_type=stem_type,
        candidate_count=len(observations),
        accepted_count=accepted_count,
        blocked_count=blocked_count,
        blockers=blockers,
        reasons=reasons,
        min_evidence=min_evidence,
    )
    return {
        "schema_version": 1,
        "mode": "no_reference_corpus_route_prior",
        "stem_type": stem_type,
        "route_id": route_id,
        "candidate_count": len(observations),
        "holdout_candidate_count": len(holdout_observations),
        "observed_track_count": len({str(item["track_id"]) for item in observations if item.get("track_id")}),
        "candidate_ids": stable_unique(str(item["candidate_id"]) for item in observations),
        "accepted_count": accepted_count,
        "blocked_count": blocked_count,
        "accept_rate": round(accepted_count / len(observations), 6) if observations else 0.0,
        "average_score": round(sum(scores) / len(scores), 6) if scores else 0.0,
        "max_vocal_leakage_in_minus_score": round(max(leakages), 6) if leakages else 0.0,
        "average_vocal_leakage_in_minus_score": round(sum(leakages) / len(leakages), 6) if leakages else 0.0,
        "decision_counts": sorted_counter(decisions),
        "top_blockers": sorted_counter(blockers),
        "top_failure_reasons": sorted_counter(reasons),
        "recommendation": recommendation,
        "requires_new_source_candidate": requires_new_source,
        "recommendation_reasons": recommendation_reasons,
        **source_pool_evidence,
        "delivery_status": "diagnostic_only",
        "delivery_allowed": False,
        "presentation_status": "route_prior_not_listen_pack",
        "release_status": "not_release_ready",
        "forbidden_actions": [
            "do_not_publish_route_prior_as_audio_result",
            "do_not_relax_thresholds",
            "do_not_train_without_verified_references",
            "do_not_mark_release_ready",
        ],
    }


def build_summary(rows: list[dict[str, Any]], route_rows: list[dict[str, Any]], holdout_rows: int) -> dict[str, Any]:
    vocal_accept_minus_blocked = sum(
        1
        for row in rows
        if not is_holdout(row)
        and decision_is_accept(str(row.get("vocal_decision") or ""))
        and decision_is_blocked(str(row.get("minus_decision") or ""))
    )
    pair_compatible_minus_blocked = sum(
        1
        for row in rows
        if not is_holdout(row)
        and str(row.get("pair_decision") or "") == "pair_compatible"
        and decision_is_blocked(str(row.get("minus_decision") or ""))
    )
    route_recommendations = Counter(str(row.get("recommendation") or "") for row in route_rows)
    return {
        "input_rows": len(rows),
        "holdout_rows_skipped_for_priors": holdout_rows,
        "route_rows": len(route_rows),
        "stem_route_counts": dict(sorted(Counter(str(row.get("stem_type") or "") for row in route_rows).items())),
        "route_recommendations": sorted_counter(route_recommendations),
        "vocal_accept_minus_blocked_count": vocal_accept_minus_blocked,
        "pair_compatible_minus_blocked_count": pair_compatible_minus_blocked,
        "minus_source_pool_eligible_count": sum(
            1
            for row in route_rows
            if row.get("stem_type") == "minus" and row.get("source_pool_eligible") is True
        ),
        "requires_new_minus_source_candidate": any(
            row.get("stem_type") == "minus" and row.get("requires_new_source_candidate") is True for row in route_rows
        ),
    }


def build_corpus_route_priors(
    rows: Sequence[dict[str, Any]],
    *,
    source_paths: Sequence[str],
    min_evidence: int = 2,
) -> dict[str, Any]:
    if min_evidence <= 0:
        raise ValueError("min_evidence must be > 0")
    all_rows = list(rows)
    route_observations: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    holdout_observations: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    holdout_count = 0
    for row in all_rows:
        holdout = is_holdout(row)
        if holdout:
            holdout_count += 1
        for stem_type in observed_stems_for_row(row):
            observation = build_observation(row, stem_type)
            key = (stem_type, str(observation["route_id"]))
            if holdout:
                holdout_observations[key].append(observation)
            else:
                route_observations[key].append(observation)

    route_keys = sorted(set(route_observations) | set(holdout_observations))
    route_rows = [
        build_route_row(
            stem_type=stem_type,
            route_id=route_id,
            observations=route_observations.get((stem_type, route_id), []),
            holdout_observations=holdout_observations.get((stem_type, route_id), []),
            min_evidence=min_evidence,
        )
        for stem_type, route_id in route_keys
    ]
    route_rows = sorted(route_rows, key=lambda row: (str(row["stem_type"]), str(row["route_id"])))
    summary = build_summary(all_rows, route_rows, holdout_count)
    manifest = {
        "schema_version": 1,
        "mode": "no_reference_corpus_route_priors",
        "source_paths": [str(path) for path in source_paths],
        "min_evidence": int(min_evidence),
        "summary": summary,
        "safety": dict(SAFETY_FLAGS),
        "delivery_status": "diagnostic_only",
        "delivery_allowed": False,
        "presentation_status": "route_prior_not_listen_pack",
        "release_status": "not_release_ready",
        "listenable": False,
        "deliverable": False,
        "training_allowed": False,
        "fine_tune_allowed": False,
    }
    return {"rows": route_rows, "manifest": manifest}


def write_summary_md(path: Path, manifest: dict[str, Any]) -> None:
    summary = manifest["summary"]
    lines = [
        "# No-Reference Corpus Route Priors",
        "",
        f"- input_rows: {summary['input_rows']}",
        f"- holdout_rows_skipped_for_priors: {summary['holdout_rows_skipped_for_priors']}",
        f"- route_rows: {summary['route_rows']}",
        f"- vocal_accept_minus_blocked_count: {summary['vocal_accept_minus_blocked_count']}",
        f"- pair_compatible_minus_blocked_count: {summary['pair_compatible_minus_blocked_count']}",
        f"- minus_source_pool_eligible_count: {summary['minus_source_pool_eligible_count']}",
        f"- requires_new_minus_source_candidate: {str(summary['requires_new_minus_source_candidate']).lower()}",
        "",
        "## Route Recommendations",
        "",
    ]
    for key, value in summary["route_recommendations"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Safety", ""])
    for key, value in manifest["safety"].items():
        lines.append(f"- {key}: {str(value).lower()}")
    lines.extend(
        [
            "",
            "## Delivery",
            "",
            f"- delivery_status: {manifest['delivery_status']}",
            f"- delivery_allowed: {str(manifest['delivery_allowed']).lower()}",
            f"- presentation_status: {manifest['presentation_status']}",
            f"- release_status: {manifest['release_status']}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    result: dict[str, Any],
    *,
    output_jsonl: Path,
    manifest_json: Path,
    summary_md: Path,
) -> None:
    write_jsonl(output_jsonl, result["rows"])
    write_json(manifest_json, result["manifest"])
    write_summary_md(summary_md, result["manifest"])


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_no_reference_corpus_route_priors.py",
        entrypoint_path="/app/tools/build_no_reference_corpus_route_priors.py",
    )
    parser = argparse.ArgumentParser(description="Build no-reference corpus route priors from mode-specific judge JSONL rows.")
    parser.add_argument("--judge-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--min-evidence", type=int, default=2)
    args = parser.parse_args(argv)

    started = time.monotonic()
    print(
        json.dumps(
            {
                "event": "no_reference_corpus_route_priors_start",
                "judge_jsonl_count": len(args.judge_jsonl),
                "output_jsonl": str(args.output_jsonl),
                "manifest_json": str(args.manifest_json),
                "summary_md": str(args.summary_md),
                "acceleration_mode": "external_only_cpu_metadata",
                "gpu_inference_skipped": True,
                "gpu_inference_skip_reason": "route_prior_builder_reads_existing_judge_metadata_only",
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    for judge_path in args.judge_jsonl:
        rows.extend(iter_jsonl(judge_path))
    result = build_corpus_route_priors(
        rows,
        source_paths=[str(path) for path in args.judge_jsonl],
        min_evidence=args.min_evidence,
    )
    write_outputs(result, output_jsonl=args.output_jsonl, manifest_json=args.manifest_json, summary_md=args.summary_md)
    print(
        json.dumps(
            {
                "event": "no_reference_corpus_route_priors_end",
                "input_rows": result["manifest"]["summary"]["input_rows"],
                "route_rows": result["manifest"]["summary"]["route_rows"],
                "requires_new_minus_source_candidate": result["manifest"]["summary"]["requires_new_minus_source_candidate"],
                "elapsed_sec": round(time.monotonic() - started, 3),
                "output_jsonl": str(args.output_jsonl),
                "manifest_json": str(args.manifest_json),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
