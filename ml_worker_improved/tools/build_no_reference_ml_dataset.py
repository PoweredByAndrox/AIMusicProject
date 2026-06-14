from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "separator_training_started": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "release_ready": False,
    "audio_read_performed": False,
}

ACCEPT_DECISIONS = {"accept_candidate", "pair_compatible"}
BLOCK_DECISIONS = {"blocked_hard_case", "blocked_pair_incompatible"}
SUPPORTED_MODES = {"vocal", "minus", "both", "window"}
SEVERE_REASON_TERMS = {
    "bad_mixture_reconstruction",
    "vocal_leakage_in_minus",
    "instrumental_leakage_in_vocal",
    "silence_hallucination",
    "robotic_or_noise_artifact",
    "robotic_artifact",
    "musical_noise",
    "thin_or_overcleaned_vocal",
    "thin_or_overcleaned",
    "transient_damage",
    "phase_or_stereo_inconsistency",
    "phase_collapse",
    "music_damage",
    "high_band_damage",
    "pumping_or_holes",
}

FEATURE_NAMES = [
    "overall_no_reference_score",
    "confidence",
    "vocal_score",
    "minus_score",
    "pair_compatibility_score",
    "artifact_score",
    "consistency_score",
    "defect_reduction_score",
    "mixture_consistency_score",
    "phase_consistency_score",
    "vocal_phase_consistency_score",
    "instrumental_phase_consistency_score",
    "pair_phase_consistency_score",
    "vocal_leakage_in_minus_score",
    "instrumental_leakage_in_vocal_score",
    "vocal_activity_confidence",
    "silence_hallucination_score",
    "robotic_artifact_score",
    "thin_vocal_score",
    "transient_damage_score",
    "high_band_damage_score",
    "pumping_or_holes_score",
    "spectral_balance_damage_score",
    "energy_balance_score",
    "window_fitness",
    "window_start_sec",
    "window_duration_sec",
    "window_count",
    "fitness_before",
    "fitness_after",
    "improvement_delta",
    "base_window_fitness",
    "alternate_window_fitness",
    "base_window_score",
    "preview_window_score",
    "score_delta",
    "blocker_delta",
    "reason_count",
    "severe_reason_count",
    "forbidden_action_count",
    "blocker_count",
    "can_execute_blend_now",
    "blocker_check_passed",
    "requires_crossfade",
    "preview_differs_from_base",
    "is_complete_pair",
]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def stable_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def split_for_track(track_id: str) -> str:
    bucket = int(hashlib.sha256(str(track_id).encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket < 7:
        return "train"
    if bucket < 8:
        return "dev"
    return "test"


def normalize_mode(value: Any) -> str | None:
    mode = str(value or "").strip().lower()
    if mode in {"vocal", "vocals"}:
        return "vocal"
    if mode in {"minus", "instrumental"}:
        return "minus"
    if mode == "both":
        return "both"
    if mode == "window":
        return "window"
    return None


def label_from_decision(value: Any) -> int | None:
    decision = str(value or "").strip()
    if decision in ACCEPT_DECISIONS:
        return 1
    if decision in BLOCK_DECISIONS:
        return 0
    return None


def score_for_mode(row: dict[str, Any], mode: str) -> float:
    if mode == "vocal":
        return as_float(row.get("vocal_score"), as_float(row.get("overall_no_reference_score")))
    if mode == "minus":
        return as_float(row.get("minus_score"), as_float(row.get("overall_no_reference_score")))
    if mode == "both":
        return as_float(row.get("pair_compatibility_score"), as_float(row.get("overall_no_reference_score")))
    return as_float(row.get("window_fitness"), as_float(row.get("overall_no_reference_score"), as_float(row.get("score_delta"))))


def merged_metrics(row: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(row)
    if extra:
        payload.update(extra)
    return payload


def extract_features(row: dict[str, Any]) -> dict[str, float]:
    reasons = stable_unique(
        as_list(row.get("reasons"))
        + as_list(row.get("base_window_reasons"))
        + as_list(row.get("preview_window_reasons"))
        + as_list(row.get("failure_labels"))
        + as_list(row.get("failed_checks"))
    )
    blockers = stable_unique(
        as_list(row.get("vocal_blockers"))
        + as_list(row.get("minus_blockers"))
        + as_list(row.get("pair_blockers"))
    )
    features = {name: as_float(row.get(name)) for name in FEATURE_NAMES}
    features["reason_count"] = float(len(reasons))
    features["severe_reason_count"] = float(len([reason for reason in reasons if reason in SEVERE_REASON_TERMS]))
    features["forbidden_action_count"] = float(len(as_list(row.get("forbidden_actions"))))
    features["blocker_count"] = float(len(blockers))
    features["can_execute_blend_now"] = 1.0 if as_bool(row.get("can_execute_blend_now")) else 0.0
    features["blocker_check_passed"] = 1.0 if as_bool(row.get("blocker_check_passed")) else 0.0
    features["requires_crossfade"] = 1.0 if as_bool(row.get("requires_crossfade")) else 0.0
    features["preview_differs_from_base"] = 1.0 if as_bool(row.get("preview_differs_from_base")) else 0.0
    features["is_complete_pair"] = 1.0 if as_bool(row.get("is_complete_pair")) else 0.0
    return {name: round(float(features[name]), 8) for name in FEATURE_NAMES}


def source_refs(row: dict[str, Any], source_type: str) -> list[str]:
    refs = [source_type]
    for key in ("candidate_manifest", "source_manifest", "mix_path", "vocal_path", "instrumental_path", "minus_path"):
        value = row.get(key)
        if value:
            refs.append(f"{key}:{value}")
    return stable_unique(refs)


def make_example(
    *,
    source_type: str,
    row: dict[str, Any],
    mode: str,
    label: int,
    policy_action: str,
    strict_mode_label: bool,
    suffix: str = "",
) -> dict[str, Any]:
    track_id = str(row.get("track_id") or "")
    candidate_id = str(
        row.get("candidate_id")
        or row.get("selected_candidate_id")
        or row.get("base_candidate_id")
        or row.get("input_candidate_id")
        or ""
    )
    payload_id = stable_hash(
        {
            "source_type": source_type,
            "track_id": track_id,
            "candidate_id": candidate_id,
            "mode": mode,
            "suffix": suffix,
            "row": row,
        }
    )[:16]
    script_score = score_for_mode(row, mode)
    return {
        "example_id": f"{source_type}:{track_id}:{candidate_id}:{mode}:{payload_id}",
        "track_id": track_id,
        "candidate_id": candidate_id,
        "mode": mode,
        "label": int(label),
        "policy_action": str(policy_action or ("accept_candidate" if label else "block_candidate")),
        "split": split_for_track(track_id),
        "script_score": round(script_score, 6),
        "features": extract_features(row),
        "source_artifact_type": source_type,
        "source_artifact_refs": source_refs(row, source_type),
        "weak_label_source": "script_decision",
        "strict_mode_label": bool(strict_mode_label),
        "reference_required": False,
        "human_csv_required": False,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "separator_training_started": False,
        "release_ready": False,
        "delivery_allowed": False,
    }


def has_conflicting_labels(examples: list[dict[str, Any]]) -> set[tuple[str, str, str, str]]:
    labels_by_key: dict[tuple[str, str, str, str], set[int]] = defaultdict(set)
    for row in examples:
        key = (
            str(row.get("source_artifact_type") or ""),
            str(row.get("track_id") or ""),
            str(row.get("candidate_id") or ""),
            str(row.get("mode") or ""),
        )
        labels_by_key[key].add(int(row.get("label") or 0))
    return {key for key, labels in labels_by_key.items() if len(labels) > 1}


def add_windowed_examples(rows: list[dict[str, Any]], examples: list[dict[str, Any]], pairwise: list[dict[str, Any]]) -> dict[str, int]:
    stats = {"skipped_ambiguous_examples": 0, "legacy_missing_score_mode_rows": 0}
    by_track_mode: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = label_from_decision(row.get("decision"))
        if label is None or not row.get("track_id") or not row.get("candidate_id"):
            stats["skipped_ambiguous_examples"] += 1
            continue
        explicit_mode = normalize_mode(row.get("score_mode"))
        if explicit_mode is not None:
            modes = [(explicit_mode, True, "script_decision")]
        else:
            stats["legacy_missing_score_mode_rows"] += 1
            modes = [
                ("window", True, "legacy_windowed_global_decision"),
                ("vocal", False, "legacy_windowed_component_proxy"),
                ("minus", False, "legacy_windowed_component_proxy"),
                ("both", False, "legacy_windowed_component_proxy"),
            ]
        for mode, strict, weak_source in modes:
            example = make_example(
                source_type="windowed_judge",
                row=row,
                mode=mode,
                label=label,
                policy_action="accept_candidate" if label else "block_candidate",
                strict_mode_label=strict,
            )
            example["weak_label_source"] = weak_source
            examples.append(example)
            by_track_mode[(example["track_id"], mode)].append(example)
        for index, window in enumerate(row.get("window_metrics") or []):
            if not isinstance(window, dict):
                continue
            window_label = label_from_decision(window.get("decision"))
            if window_label is None:
                window_label = label
            merged = merged_metrics(row, window)
            merged["candidate_id"] = row.get("candidate_id")
            window_example = make_example(
                source_type="windowed_judge_window",
                row=merged,
                mode="window",
                label=window_label,
                policy_action="accept_candidate" if window_label else "block_candidate",
                strict_mode_label=True,
                suffix=str(index),
            )
            window_example["weak_label_source"] = "window_metric_decision"
            examples.append(window_example)
            by_track_mode[(window_example["track_id"], "window")].append(window_example)
    for (track_id, mode), mode_examples in sorted(by_track_mode.items()):
        accepted = [row for row in mode_examples if int(row["label"]) == 1]
        blocked = [row for row in mode_examples if int(row["label"]) == 0]
        for good in accepted[:2]:
            for bad in blocked[:2]:
                pairwise.append(
                    {
                        "pair_id": f"windowed:{track_id}:{mode}:{stable_hash([good['example_id'], bad['example_id']])[:12]}",
                        "track_id": track_id,
                        "mode": mode,
                        "preferred_example_id": good["example_id"],
                        "rejected_example_id": bad["example_id"],
                        "preferred_score": good["script_score"],
                        "rejected_score": bad["script_score"],
                        "split": split_for_track(track_id),
                        "source_artifact_refs": stable_unique(good["source_artifact_refs"] + bad["source_artifact_refs"]),
                    }
                )
    return stats


def add_optimizer_examples(rows: list[dict[str, Any]], examples: list[dict[str, Any]]) -> None:
    for row in rows:
        decision = str(row.get("final_decision") or "")
        if not row.get("track_id"):
            continue
        label = 0 if decision == "blocked_hard_case" else 1
        payload = dict(row)
        payload["overall_no_reference_score"] = row.get("fitness_after", row.get("fitness_before", 0.0))
        examples.append(
            make_example(
                source_type="optimizer_result",
                row=payload,
                mode="window",
                label=label,
                policy_action=decision or "optimizer_unknown",
                strict_mode_label=False,
            )
        )


def add_segment_examples(rows: list[dict[str, Any]], examples: list[dict[str, Any]], pairwise: list[dict[str, Any]]) -> None:
    for row in rows:
        mode = normalize_mode(row.get("source_stem")) or "window"
        label = 0
        example = make_example(
            source_type="segment_blend_plan",
            row=row,
            mode=mode,
            label=label,
            policy_action=str(row.get("segment_blend_action") or "segment_plan_diagnostic_only"),
            strict_mode_label=True,
        )
        example["weak_label_source"] = "segment_plan_diagnostic_not_delivery_accept"
        examples.append(example)
        if as_float(row.get("improvement_delta")) > 0:
            pairwise.append(
                {
                    "pair_id": f"segment:{row.get('track_id')}:{mode}:{stable_hash(row)[:12]}",
                    "track_id": str(row.get("track_id") or ""),
                    "mode": mode,
                    "preferred_example_id": f"alternate:{row.get('alternate_candidate_id')}",
                    "rejected_example_id": f"base:{row.get('base_candidate_id')}",
                    "preferred_score": as_float(row.get("alternate_window_fitness")),
                    "rejected_score": as_float(row.get("base_window_fitness")),
                    "split": split_for_track(str(row.get("track_id") or "")),
                    "source_artifact_refs": source_refs(row, "segment_blend_plan"),
                }
            )


def add_preview_examples(rows: list[dict[str, Any]], examples: list[dict[str, Any]], pairwise: list[dict[str, Any]]) -> None:
    for row in rows:
        mode = normalize_mode(row.get("source_stem")) or "both"
        label = 1 if as_bool(row.get("can_publish_as_result")) and as_bool(row.get("can_mark_release_ready")) else 0
        payload = dict(row)
        payload["overall_no_reference_score"] = row.get("preview_window_score", row.get("score_delta", 0.0))
        payload["window_fitness"] = payload["overall_no_reference_score"]
        example = make_example(
            source_type="segment_blend_preview_window_judge",
            row=payload,
            mode=mode,
            label=label,
            policy_action=str(row.get("allowed_next_step") or row.get("preview_decision") or "preview_diagnostic_only"),
            strict_mode_label=True,
        )
        example["weak_label_source"] = "preview_contract_not_delivery_accept"
        examples.append(example)
        if str(row.get("preview_decision") or "") == "preview_improved_window" and as_float(row.get("score_delta")) > 0:
            pairwise.append(
                {
                    "pair_id": f"preview:{row.get('track_id')}:{mode}:{stable_hash(row)[:12]}",
                    "track_id": str(row.get("track_id") or ""),
                    "mode": mode,
                    "preferred_example_id": f"preview:{row.get('alternate_candidate_id')}",
                    "rejected_example_id": f"base:{row.get('base_candidate_id')}",
                    "preferred_score": as_float(row.get("preview_window_score")),
                    "rejected_score": as_float(row.get("base_window_score")),
                    "split": split_for_track(str(row.get("track_id") or "")),
                    "source_artifact_refs": source_refs(row, "segment_blend_preview_window_judge"),
                }
            )


def prune_conflicts(examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    conflicts = has_conflicting_labels(examples)
    if not conflicts:
        return examples, 0
    pruned = []
    for row in examples:
        key = (
            str(row.get("source_artifact_type") or ""),
            str(row.get("track_id") or ""),
            str(row.get("candidate_id") or ""),
            str(row.get("mode") or ""),
        )
        if key not in conflicts:
            pruned.append(row)
    return pruned, len(examples) - len(pruned)


def build_manifest(
    *,
    examples: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
    skipped_ambiguous: int,
    skipped_conflicting: int,
    legacy_missing_score_mode_rows: int,
    input_artifacts: dict[str, str],
) -> dict[str, Any]:
    by_mode = Counter(str(row.get("mode") or "") for row in examples)
    by_split = Counter(str(row.get("split") or "") for row in examples)
    by_source = Counter(str(row.get("source_artifact_type") or "") for row in examples)
    track_split: dict[str, str] = {}
    for row in examples:
        track_split[str(row["track_id"])] = str(row["split"])
    fingerprint_payload = {
        "examples": examples,
        "pairwise": pairwise,
        "feature_names": FEATURE_NAMES,
        "input_artifacts": input_artifacts,
    }
    return {
        "dataset_type": "local_no_reference_weak_label_critic",
        "examples_total": len(examples),
        "pairwise_examples_total": len(pairwise),
        "examples_by_mode": dict(sorted(by_mode.items())),
        "examples_by_split": dict(sorted(by_split.items())),
        "examples_by_source": dict(sorted(by_source.items())),
        "tracks_total": len(track_split),
        "train_dev_test_split": dict(sorted(by_split.items())),
        "split_unit": "track_id",
        "track_split": dict(sorted(track_split.items())),
        "feature_list": list(FEATURE_NAMES),
        "skipped_ambiguous_examples": int(skipped_ambiguous),
        "skipped_conflicting_examples": int(skipped_conflicting),
        "legacy_missing_score_mode_rows": int(legacy_missing_score_mode_rows),
        "mode_specific_label_warning": (
            "Rows without score_mode are included as weak legacy component-proxy examples "
            "with strict_mode_label=false; they are not reference truth."
        ),
        "dataset_fingerprint": stable_hash(fingerprint_payload),
        "input_artifacts": input_artifacts,
        **SAFETY_FLAGS,
    }


def build_dataset(
    *,
    candidate_pool_rows: list[dict[str, Any]],
    windowed_judge_rows: list[dict[str, Any]],
    optimizer_result_rows: list[dict[str, Any]],
    segment_blend_plan_rows: list[dict[str, Any]],
    preview_window_judge_rows: list[dict[str, Any]],
    input_artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    examples: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    candidate_by_key = {
        (str(row.get("track_id") or ""), str(row.get("candidate_id") or "")): row
        for row in candidate_pool_rows
        if row.get("track_id") and row.get("candidate_id")
    }
    enriched_windowed_rows = []
    for row in windowed_judge_rows:
        key = (str(row.get("track_id") or ""), str(row.get("candidate_id") or ""))
        enriched = dict(candidate_by_key.get(key, {}))
        enriched.update(row)
        enriched_windowed_rows.append(enriched)
    windowed_stats = add_windowed_examples(enriched_windowed_rows, examples, pairwise)
    add_optimizer_examples(optimizer_result_rows, examples)
    add_segment_examples(segment_blend_plan_rows, examples, pairwise)
    add_preview_examples(preview_window_judge_rows, examples, pairwise)
    examples, skipped_conflicting = prune_conflicts(examples)
    examples.sort(key=lambda row: (row["track_id"], row["mode"], row["candidate_id"], row["example_id"]))
    pairwise.sort(key=lambda row: (row["track_id"], row["mode"], row["pair_id"]))
    manifest = build_manifest(
        examples=examples,
        pairwise=pairwise,
        skipped_ambiguous=windowed_stats["skipped_ambiguous_examples"],
        skipped_conflicting=skipped_conflicting,
        legacy_missing_score_mode_rows=windowed_stats["legacy_missing_score_mode_rows"],
        input_artifacts=input_artifacts or {},
    )
    return {"examples": examples, "pairwise_examples": pairwise, "manifest": manifest}


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "no_reference_ml_examples.jsonl", result["examples"])
    write_jsonl(output_dir / "no_reference_pairwise_examples.jsonl", result["pairwise_examples"])
    (output_dir / "no_reference_ml_dataset_manifest.json").write_text(
        json.dumps(result["manifest"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build metadata-only weak-label dataset for local no-reference critics.")
    parser.add_argument("--candidate-pool-jsonl", type=Path, required=True)
    parser.add_argument("--windowed-judge-jsonl", type=Path, required=True)
    parser.add_argument("--optimizer-result-jsonl", type=Path, required=True)
    parser.add_argument("--segment-blend-plan-jsonl", type=Path, required=True)
    parser.add_argument("--segment-blend-preview-window-judge-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    input_artifacts = {
        "candidate_pool_jsonl": str(args.candidate_pool_jsonl),
        "windowed_judge_jsonl": str(args.windowed_judge_jsonl),
        "optimizer_result_jsonl": str(args.optimizer_result_jsonl),
        "segment_blend_plan_jsonl": str(args.segment_blend_plan_jsonl),
        "segment_blend_preview_window_judge_jsonl": str(args.segment_blend_preview_window_judge_jsonl),
    }
    result = build_dataset(
        candidate_pool_rows=list(iter_jsonl(args.candidate_pool_jsonl)),
        windowed_judge_rows=list(iter_jsonl(args.windowed_judge_jsonl)),
        optimizer_result_rows=list(iter_jsonl(args.optimizer_result_jsonl)),
        segment_blend_plan_rows=list(iter_jsonl(args.segment_blend_plan_jsonl)),
        preview_window_judge_rows=list(iter_jsonl(args.segment_blend_preview_window_judge_jsonl)),
        input_artifacts=input_artifacts,
    )
    write_outputs(result, args.output_dir)
    print(json.dumps(result["manifest"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
