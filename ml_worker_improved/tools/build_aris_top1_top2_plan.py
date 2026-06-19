from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from artifact_name_policy import validate_artifact_relative_path
from docker_runtime import require_docker_runtime


START_TIME = time.monotonic()
PLAN_NAME = "plan.json"
MANIFEST_NAME = "manifest.json"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"

ALLOWED_CANDIDATE_FAMILIES = ["demucs", "htdemucs", "scnet", "bs_roformer", "melband", "mdx23c", "bs_mamba2"]
EXPLICITLY_CUT_DIRECTIONS = {
    "qwen_controller": "cut_for_now_more_control_logic_same_bad_stems",
    "diffusion_generative_repair": "cut_for_now_faithfulness_risk",
    "repeat_teacher_mask_adapter": "cut_as_primary_previous_holdout_failures",
}
FAILURE_TOKENS = [
    "vocal_bleed",
    "minus_detail_gate_blocked",
    "improvement_below_threshold",
    "mixture_consistency_score_below_detail_gate",
    "instrumental_integrity_below_detail_gate",
    "holdout_row_failed",
    "holdout_metric_below_threshold",
    "holdout_failed",
    "holdout_failed_evaluations",
    "sdr_below_threshold",
    "best_matrix_row_missing",
    "matrix_validation_not_passed",
    "missing_best_config_id",
    "success_rate_not_strictly_above_threshold",
    "severe_music_damage",
    "transient_preservation_below_detail_gate",
    "windowed_minus_not_accepted",
    "no_accepted_phase_repair_variants",
    "minus_warning_comparison_no_improved_windows",
    "audio_missing",
    "model_snapshot_not_cached",
    "model_cache_unavailable",
    "cuda_memory_guard_failed",
    "preflight_not_passed",
    "adapter_not_better_than_baseline",
    "adapter_parse_errors",
    "adapter_dir_missing",
]


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
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
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_manifest_build",
        "gpu_inference_required": False,
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "rag_failure_histogram_metadata_only",
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


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _strings(item)
    elif value is not None:
        yield str(value)


def _row_text(row: dict[str, Any]) -> str:
    fields = [row.get("excerpt"), row.get("relative_path"), row.get("zip_member"), row.get("status_fields")]
    return " ".join(text for field in fields for text in _strings(field))


def _stem_for_row(row: dict[str, Any]) -> str:
    status = row.get("status_fields") if isinstance(row.get("status_fields"), dict) else {}
    for key in ("stem", "stem_type", "target_stem"):
        stem = str(status.get(key) or row.get(key) or "").strip().lower()
        if stem:
            return "minus" if stem == "instrumental" else stem
    text = _row_text(row).lower()
    for stem in ("vocal", "minus", "drums", "bass", "guitar", "piano", "other"):
        if re.search(rf"\b{re.escape(stem)}\b", text):
            return stem
    return "unknown"


def _failure_counts(rows: Iterable[dict[str, Any]]) -> tuple[Counter[str], dict[str, Counter[str]], int, list[dict[str, Any]]]:
    total: Counter[str] = Counter()
    by_stem: dict[str, Counter[str]] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []
    row_count = 0
    for row in rows:
        row_count += 1
        text = _row_text(row)
        stem = _stem_for_row(row)
        row_hits: Counter[str] = Counter()
        for token in FAILURE_TOKENS:
            count = len(re.findall(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", text))
            if count:
                total[token] += count
                by_stem[stem][token] += count
                row_hits[token] = count
        if row_hits and len(examples) < 20:
            examples.append(
                {
                    "relative_path": str(row.get("relative_path") or ""),
                    "stem": stem,
                    "failures": dict(sorted(row_hits.items())),
                }
            )
    return total, by_stem, row_count, examples


def _counter_rows(counter: Counter[str], *, limit: int = 15) -> list[dict[str, Any]]:
    return [
        {"failure": failure, "count": int(count)}
        for failure, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _matrix_entry(stem: str, targets: Sequence[str]) -> dict[str, Any]:
    if stem in {"vocal", "minus"}:
        grammar = (
            ["base_family_export", "backing_vocal_preserve", "instrument_bleed_reject", "family_blend_candidate", "exact_reference_or_no_reference_gate"]
            if stem == "vocal"
            else [
                "base_family_export",
                "mixture_consistency_preserve",
                "transient_detail_preserve",
                "vocal_bleed_residual_reject",
                "family_blend_candidate",
                "exact_reference_or_no_reference_gate",
            ]
        )
    else:
        grammar = ["base_family_export", "semantic_presence_gate", "mixture_residual_gate", "family_blend_candidate"]
    return {
        "candidate_families": list(ALLOWED_CANDIDATE_FAMILIES),
        "grammar": grammar,
        "targets": list(targets),
    }


def _build_plan(*, current: dict[str, Any], manifest_refs_jsonl: Path, output_dir: Path) -> dict[str, Any]:
    total, by_stem, row_count, examples = _failure_counts(iter_jsonl(manifest_refs_jsonl))
    stem_priorities = {stem: _counter_rows(counter, limit=8) for stem, counter in sorted(by_stem.items())}
    matrix_stems = ["vocal", "minus", "drums", "bass", "other", "guitar", "piano"]
    matrix = {
        stem: _matrix_entry(stem, [row["failure"] for row in stem_priorities.get(stem, [])[:6]])
        for stem in matrix_stems
    }
    return {
        "schema": "aris_top1_top2_rag_plan.v1",
        "route_id": "failure_conditioned_refinement",
        "status": "planning_only",
        "source_rag": {
            "rag_id": str(current.get("rag_id") or ""),
            "current_retention_mode": str(current.get("current_retention_mode") or ""),
            "binary_audio_model_bytes_available": current.get("binary_audio_model_bytes_available") is True,
            "manifest_refs_jsonl": str(manifest_refs_jsonl),
        },
        "row_count_scanned": row_count,
        "candidate_families": list(ALLOWED_CANDIDATE_FAMILIES),
        "priority_failures": _counter_rows(total),
        "stem_priorities": stem_priorities,
        "top1_fixed_inference_refinement_matrix": matrix,
        "top2_reference_first_corpus_plan": {
            "minimum_unit": "verified_full_6s_stem_group",
            "required_stems": ["vocal", "minus", "drums", "bass", "other", "guitar", "piano"],
            "split": "leakage_safe_train_holdout_no_track_overlap",
            "pseudo_teacher": "training_only_never_holdout_ground_truth",
        },
        "explicitly_cut_directions": dict(EXPLICITLY_CUT_DIRECTIONS),
        "evidence_examples": examples,
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "release_ready": False,
        "can_publish_as_result": False,
        "outputs": {
            "plan_json": str(output_dir / PLAN_NAME),
            "summary_md": str(output_dir / SUMMARY_NAME),
            "progress_jsonl": str(output_dir / PROGRESS_NAME),
        },
    }


def _summary_text(plan: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = [
        "# TOP-1/TOP-2 ARIS RAG Plan",
        "",
        "Status: planning_only; no audio/model bytes; no listen result.",
        "",
        f"- source_rag: `{plan['source_rag']['rag_id']}`",
        f"- row_count_scanned: `{plan['row_count_scanned']}`",
        f"- binary_audio_model_bytes_available: `{str(manifest['binary_audio_model_bytes_available']).lower()}`",
        "",
        "## Priority Failures",
        "",
    ]
    lines.extend(f"- `{row['failure']}`: {row['count']}" for row in plan["priority_failures"])
    lines.extend(["", "## Directions Cut Now", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in plan["explicitly_cut_directions"].items())
    lines.extend(["", "## Next Safe Step", "", "Run the refinement matrix only against RAG-controlled candidate pools with objective gate evidence."])
    return "\n".join(lines) + "\n"


def build_aris_top1_top2_plan(
    *,
    artifact_rag_root: Path,
    output_dir: Path,
    current_json: Path | None = None,
    manifest_refs_jsonl: Path | None = None,
    command_args: Sequence[str] | None = None,
) -> dict[str, Any]:
    root, output = _ensure_output_inside_rag(artifact_rag_root=artifact_rag_root, output_dir=output_dir)
    current_path = _ensure_path_inside_rag(
        artifact_rag_root=root,
        path=current_json or root / "current.json",
        label="current_json",
    )
    current = read_json(current_path)
    refs_path = _ensure_path_inside_rag(
        artifact_rag_root=root,
        path=manifest_refs_jsonl or root / str(current.get("rag_id") or "") / "manifest_refs.jsonl",
        label="manifest_refs_jsonl",
    )
    if not refs_path.exists():
        raise FileNotFoundError(f"manifest_refs_jsonl missing: {refs_path}")
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    append_progress(progress_path, "run_start", current_json=str(current_path), manifest_refs_jsonl=str(refs_path), output_dir=str(output))
    plan = _build_plan(current=current, manifest_refs_jsonl=refs_path, output_dir=output)
    append_progress(progress_path, "refs_scanned", row_count=plan["row_count_scanned"], priority_failure_count=len(plan["priority_failures"]))
    manifest = {
        "schema": "aris_top1_top2_rag_plan_manifest.v1",
        "manifest_type": "aris_top1_top2_rag_plan",
        "status": "planning_only",
        "artifact_rag_root": str(root),
        "output_dir": str(output),
        "current_json": str(current_path),
        "manifest_refs_jsonl": str(refs_path),
        "plan_json": str(output / PLAN_NAME),
        "summary_md": str(output / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "row_count_scanned": plan["row_count_scanned"],
        "candidate_families": list(ALLOWED_CANDIDATE_FAMILIES),
        "binary_audio_model_bytes_available": current.get("binary_audio_model_bytes_available") is True,
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
    write_json(output / PLAN_NAME, plan)
    write_json(output / MANIFEST_NAME, manifest)
    (output / SUMMARY_NAME).write_text(_summary_text(plan, manifest), encoding="utf-8")
    append_progress(progress_path, "run_finish", status=manifest["status"], manifest_json=str(output / MANIFEST_NAME))
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Docker-only TOP-1/TOP-2 ARIS plan from artifact RAG refs.")
    parser.add_argument("--artifact-rag-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--current-json", type=Path, default=None)
    parser.add_argument("--manifest-refs-jsonl", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_aris_top1_top2_plan.py",
        entrypoint_path="/app/tools/build_aris_top1_top2_plan.py",
    )
    args = parse_args(argv)
    manifest = build_aris_top1_top2_plan(
        artifact_rag_root=args.artifact_rag_root,
        output_dir=args.output_dir,
        current_json=args.current_json,
        manifest_refs_jsonl=args.manifest_refs_jsonl,
        command_args=list(argv or sys.argv[1:]),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
