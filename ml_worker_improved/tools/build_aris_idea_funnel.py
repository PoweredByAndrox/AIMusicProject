#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for import_root in (APP_ROOT, TOOLS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from artifact_name_policy import validate_artifact_relative_path  # noqa: E402
from docker_runtime import require_docker_runtime  # noqa: E402

START_TIME = time.monotonic()
FUNNEL_NAME = "funnel.json"
IDEAS_NAME = "ideas.jsonl"
MANIFEST_NAME = "manifest.json"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"

ALLOWED_CANDIDATE_FAMILIES = ["demucs", "htdemucs", "scnet", "bs_roformer", "melband", "mdx23c", "bs_mamba2"]
BLOCKED_OLD_DIRECTIONS = {
    "qwen_controller": "cut_for_now_more_control_logic_same_bad_stems",
    "diffusion_generative_repair": "cut_for_now_faithfulness_risk",
    "repeat_teacher_mask_adapter": "cut_as_primary_previous_holdout_failures",
    "threshold_relaxation": "forbidden_no_relaxed_gates_as_success",
    "manual_listening_required": "forbidden_user_must_not_listen_per_track",
    "per_track_hardcode": "forbidden_no_track_specific_shortcuts",
    "preview_as_success": "forbidden_accepted_for_listen_pack_preview_is_not_final_result",
    "fallback_as_success": "forbidden_fallback_cannot_be_delivery_strategy",
}

IDEA_TEMPLATES = [
    {
        "route_id": "paired_vocal_minus_model_zoo_ladder",
        "title": "Generate vocal and minus as a paired full-track model-zoo ladder",
        "signals": {
            "binary_audio_unavailable": 1000,
            "minus_candidate_missing": 400,
            "objective_gate_missing": 40,
            "holdout_row_failed": 8,
            "holdout_failed": 8,
        },
        "candidate_families": ["scnet", "bs_roformer", "mdx23c", "bs_mamba2", "melband", "htdemucs"],
        "required_gates": [
            "reference_or_no_reference_gate",
            "pair_consistency_gate",
            "mixture_closure_gate",
            "minus_detail_gate",
            "vocal_bleed_gate",
        ],
        "why_new": "Switches from metadata/preview recovery to paired GPU inference candidates that must produce both stems before any delivery gate can pass.",
    },
    {
        "route_id": "holdout_error_map_segment_router",
        "title": "Build reference-conditioned segment error maps before routing models",
        "signals": {
            "holdout_row_failed": 35,
            "holdout_metric_below_threshold": 35,
            "holdout_failed": 35,
            "holdout_failed_evaluations": 35,
            "transient_preservation_below_detail_gate": 12,
            "instrumental_integrity_below_detail_gate": 12,
        },
        "candidate_families": ["scnet", "bs_roformer", "mdx23c", "bs_mamba2"],
        "required_gates": [
            "reference_alignment_gate",
            "segment_error_map_coverage_gate",
            "holdout_no_track_overlap_gate",
            "no_reference_generalization_gate",
        ],
        "why_new": "Uses failed holdout evidence to target segment/model routing before inference, instead of adding another controller layer over the same bad stems.",
    },
    {
        "route_id": "minus_detail_preservation_ladder",
        "title": "Route minus candidates through detail-preserving families first",
        "signals": {
            "minus_detail_gate_blocked": 45,
            "transient_preservation_below_detail_gate": 35,
            "instrumental_integrity_below_detail_gate": 25,
            "severe_music_damage": 25,
            "mixture_consistency_score_below_detail_gate": 20,
        },
        "candidate_families": ["bs_roformer", "mdx23c", "bs_mamba2", "htdemucs"],
        "required_gates": [
            "transient_detail_gate",
            "instrumental_integrity_gate",
            "mixture_consistency_gate",
            "vocal_residue_reject_gate",
        ],
        "why_new": "Treats minus detail as a first-class candidate generation target, not as a posthoc cleanup or vocal-guided fallback.",
    },
    {
        "route_id": "cross_family_pair_consensus_selector",
        "title": "Select stem pairs by cross-family complementarity, not single-stem previews",
        "signals": {
            "pair_consistency_gate_failed": 50,
            "mixture_consistency_score_below_detail_gate": 30,
            "best_matrix_row_missing": 20,
            "matrix_validation_not_passed": 20,
            "success_rate_not_strictly_above_threshold": 15,
        },
        "candidate_families": ["scnet", "bs_roformer", "melband", "mdx23c", "bs_mamba2", "demucs"],
        "required_gates": [
            "pair_complementarity_gate",
            "mixture_closure_gate",
            "cross_family_disagreement_gate",
            "no_best_of_bad_delivery_gate",
        ],
        "why_new": "Stops treating independent accepted previews as a final pair and ranks only vocal+minus combinations with objective pair evidence.",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            yield payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _resolved(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _ensure_output_inside_rag(*, artifact_rag_root: Path, output_dir: Path) -> tuple[Path, Path]:
    root = _resolved(artifact_rag_root)
    output = _resolved(output_dir)
    if output != root and root not in output.parents:
        raise ValueError("output_dir must be inside artifact_rag_root")
    try:
        validate_artifact_relative_path(output.relative_to(root).as_posix())
    except ValueError as exc:
        raise ValueError("output_dir must use short Windows-safe relative path inside artifact_rag_root") from exc
    return root, output


def _ensure_path_inside_rag(*, artifact_rag_root: Path, path: Path | str, label: str, base_dir: Path | None = None) -> Path:
    if not str(path or "").strip():
        raise ValueError(f"{label} is required")
    candidate = Path(path)
    if not candidate.is_absolute() and base_dir is not None:
        candidate = base_dir / candidate
    root = _resolved(artifact_rag_root)
    resolved = _resolved(candidate)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} must be inside artifact_rag_root")
    return resolved


def _progress_row(event: str, **fields: Any) -> dict[str, Any]:
    row = {
        "event": event,
        "phase": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_rag_planning",
        "gpu_inference_required": False,
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "idea_funnel_planning_only",
        "delivery_allowed": False,
        "release_ready": False,
    }
    row.update(fields)
    return row


def append_progress(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_progress_row(event, **fields), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _histogram_from_top_plan(top_plan: dict[str, Any]) -> Counter[str]:
    histogram: Counter[str] = Counter()
    for row in top_plan.get("priority_failures") or []:
        if isinstance(row, dict):
            failure = str(row.get("failure") or "").strip()
            if failure:
                histogram[failure] += int(row.get("count") or 0)
    return histogram


def _histogram_from_matrix_manifest(*, artifact_rag_root: Path, matrix_manifest_path: Path, matrix_manifest: dict[str, Any]) -> Counter[str]:
    histogram: Counter[str] = Counter()
    rejected_path_raw = str(matrix_manifest.get("rejected_jsonl") or "")
    if not rejected_path_raw:
        return histogram
    rejected_path = _ensure_path_inside_rag(
        artifact_rag_root=artifact_rag_root,
        path=rejected_path_raw,
        label="rejected_jsonl",
        base_dir=matrix_manifest_path.parent,
    )
    if not rejected_path.exists():
        return histogram
    for row in iter_jsonl(rejected_path):
        checks = row.get("failed_checks") or []
        if isinstance(checks, str):
            checks = [checks]
        for check in checks:
            label = str(check).strip()
            if label:
                histogram[label] += 1
    return histogram


def _rank_ideas(histogram: Counter[str]) -> list[dict[str, Any]]:
    ideas: list[dict[str, Any]] = []
    for template in IDEA_TEMPLATES:
        signals = template["signals"]
        score = sum(int(histogram.get(label, 0)) * int(weight) for label, weight in signals.items())
        matched = {label: int(histogram.get(label, 0)) for label in signals if histogram.get(label, 0)}
        ideas.append(
            {
                "schema": "aris_idea_funnel_route.v1",
                "route_id": template["route_id"],
                "title": template["title"],
                "rank_score": int(score),
                "matched_failures": matched,
                "candidate_families": template["candidate_families"],
                "required_gates": template["required_gates"],
                "why_new": template["why_new"],
                "execution_mode": "gpu_candidate_generation_required_before_scoring",
                "delivery_allowed": False,
                "release_ready": False,
                "presentation_status": "blocked_from_listen_pack",
                "forbidden_as_delivery": ["metadata_only", "fallback", "preview_only", "best_of_bad_candidate"],
            }
        )
    return sorted(ideas, key=lambda row: (-int(row["rank_score"]), str(row["route_id"])))


def _summary_text(funnel: dict[str, Any]) -> str:
    lines = [
        "# ARIS Strict Idea Funnel",
        "",
        "Status: planning_only; no audio/model bytes; no listen result.",
        "",
        "## Top Routes",
        "",
    ]
    lines.extend(f"- `{row['route_id']}`: score {row['rank_score']}" for row in funnel["top_routes"])
    lines.extend(["", "## Blocked Old Directions", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in funnel["blocked_old_directions"].items())
    lines.extend(["", "## Next Safe Step", "", "Run only the selected route work order through Docker GPU inference and strict gates."])
    return "\n".join(lines) + "\n"


def build_aris_idea_funnel(
    *,
    artifact_rag_root: Path,
    top_plan_json: Path,
    matrix_manifest_json: Path | None,
    output_dir: Path,
    command_args: Sequence[str] | None = None,
) -> dict[str, Any]:
    root, output = _ensure_output_inside_rag(artifact_rag_root=artifact_rag_root, output_dir=output_dir)
    top_plan_path = _ensure_path_inside_rag(artifact_rag_root=root, path=top_plan_json, label="top_plan_json")
    matrix_manifest_path = (
        _ensure_path_inside_rag(artifact_rag_root=root, path=matrix_manifest_json, label="matrix_manifest_json")
        if matrix_manifest_json is not None
        else None
    )
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    append_progress(progress_path, "run_start", top_plan_json=str(top_plan_path), matrix_manifest_json=str(matrix_manifest_path or ""))

    top_plan = read_json(top_plan_path)
    matrix_manifest = read_json(matrix_manifest_path) if matrix_manifest_path is not None and matrix_manifest_path.exists() else {}
    histogram = _histogram_from_top_plan(top_plan)
    if matrix_manifest_path is not None:
        histogram.update(
            _histogram_from_matrix_manifest(
                artifact_rag_root=root,
                matrix_manifest_path=matrix_manifest_path,
                matrix_manifest=matrix_manifest,
            )
        )
    ideas = _rank_ideas(histogram)
    append_progress(progress_path, "ideas_ranked", idea_count=len(ideas), failure_count=len(histogram))

    funnel = {
        "schema": "aris_strict_idea_funnel.v1",
        "status": "planning_only",
        "source_top_plan_json": str(top_plan_path),
        "source_matrix_manifest_json": str(matrix_manifest_path or ""),
        "confirmed_problem": {
            "not_a_threshold_strictness_problem": True,
            "current_routes_fail_holdout_minus_integrity_mixture_transient": True,
            "old_accepted_for_listen_pack_can_be_preview_only": True,
            "more_controller_logic_without_better_stems_is_cut": True,
        },
        "failure_histogram": dict(sorted(histogram.items(), key=lambda item: (-item[1], item[0]))),
        "candidate_families": list(ALLOWED_CANDIDATE_FAMILIES),
        "blocked_old_directions": dict(BLOCKED_OLD_DIRECTIONS),
        "top_routes": ideas[:2],
        "all_routes": ideas,
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "release_ready": False,
        "can_publish_as_result": False,
        "execution_contract": {
            "rag_controlled_outputs_only": True,
            "docker_only": True,
            "gpu_inference_required_for_candidate_generation": True,
            "metadata_only_rows_blocked": True,
            "rejected_artifacts_not_deliverable": True,
            "manual_listening_not_required": True,
        },
        "outputs": {
            "funnel_json": str(output / FUNNEL_NAME),
            "ideas_jsonl": str(output / IDEAS_NAME),
            "summary_md": str(output / SUMMARY_NAME),
            "progress_jsonl": str(progress_path),
        },
    }
    manifest = {
        "schema": "aris_strict_idea_funnel_manifest.v1",
        "manifest_type": "aris_strict_idea_funnel",
        "status": "planning_only",
        "artifact_rag_root": str(root),
        "output_dir": str(output),
        "top_plan_json": str(top_plan_path),
        "matrix_manifest_json": str(matrix_manifest_path or ""),
        "funnel_json": str(output / FUNNEL_NAME),
        "ideas_jsonl": str(output / IDEAS_NAME),
        "summary_md": str(output / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "top_route_ids": [row["route_id"] for row in ideas[:2]],
        "idea_count": len(ideas),
        "failure_count": len(histogram),
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "release_ready": False,
        "can_publish_as_result": False,
        "audio_read": False,
        "audio_exported": False,
        "models_changed": False,
        "thresholds_relaxed": False,
        "command_args": list(command_args or []),
    }
    write_json(output / FUNNEL_NAME, funnel)
    write_jsonl(output / IDEAS_NAME, ideas)
    write_json(output / MANIFEST_NAME, manifest)
    (output / SUMMARY_NAME).write_text(_summary_text(funnel), encoding="utf-8")
    append_progress(progress_path, "run_finish", status=manifest["status"], manifest_json=str(output / MANIFEST_NAME))
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict ARIS idea funnel from RAG failure and matrix evidence.")
    parser.add_argument("--artifact-rag-root", type=Path, required=True)
    parser.add_argument("--top-plan-json", type=Path, required=True)
    parser.add_argument("--matrix-manifest-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_aris_idea_funnel.py",
        entrypoint_path="/app/tools/build_aris_idea_funnel.py",
    )
    args = parse_args(argv)
    manifest = build_aris_idea_funnel(
        artifact_rag_root=args.artifact_rag_root,
        top_plan_json=args.top_plan_json,
        matrix_manifest_json=args.matrix_manifest_json,
        output_dir=args.output_dir,
        command_args=list(argv or sys.argv[1:]),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
