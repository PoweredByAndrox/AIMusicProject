#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime  # noqa: E402
from tools.render_no_reference_minus_segment_switch_candidate import (  # noqa: E402
    as_float,
    candidate_id_from_manifest,
    candidate_minus_path,
    candidate_row_from_manifest,
    candidate_vocal_path,
    choose_safe_alternate,
    repairable_blocked_segment_count,
    segment_issues,
    self_judge_blockers,
    window_values,
)
from tools.run_no_reference_minus_delivery_pipeline import (  # noqa: E402
    choose_base_candidate_id,
    choose_base_candidate_id_from_score_matrix,
    read_self_judge_json,
)


SAFETY_FLAGS: dict[str, bool] = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "audio_copied": False,
    "audio_exported": False,
    "diagnostic_audio_exported": False,
    "release_ready": False,
}

FORBIDDEN_ACTIONS = [
    "do_not_require_reference_stems",
    "do_not_require_human_csv",
    "do_not_start_finetune",
    "do_not_train",
    "do_not_add_models",
    "do_not_add_cleanup_debleed_denoise",
    "do_not_relax_gates",
    "do_not_export_diagnostic_audio_as_result",
    "do_not_accept_best_of_bad",
    "do_not_mark_release_ready",
    "do_not_publish_as_final_result",
]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def path_missing(path_text: str) -> bool:
    return not path_text or not Path(path_text).exists()


def load_self_judge_payloads(paths: Sequence[Path]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in paths:
        payload = read_self_judge_json(path)
        payload["source_self_judge_manifest_path"] = str(path)
        payloads.append(payload)
    return payloads


def select_base_candidate_id(
    *,
    payloads: list[dict[str, Any]],
    explicit_base_candidate_id: str | None,
    candidate_score_matrix_json: Path | None,
) -> tuple[str, str]:
    if explicit_base_candidate_id:
        return explicit_base_candidate_id, "explicit_base_candidate_id"
    if candidate_score_matrix_json is not None:
        matrix_candidate_id = choose_base_candidate_id_from_score_matrix(
            matrix_json=candidate_score_matrix_json,
            self_judge_payloads=payloads,
        )
        if matrix_candidate_id:
            return matrix_candidate_id, "candidate_score_matrix"
    return choose_base_candidate_id(payloads), "self_judge_heuristic"


def selected_segment_row(
    *,
    base_candidate_id: str,
    base_segment: dict[str, Any],
    selected_candidate_id: str,
    selected_segment: dict[str, Any],
    selection_reason: str,
    selected_requires_post_repair: bool,
    candidates_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    start_sec, end_sec = window_values(base_segment)
    selected_row = candidates_by_id.get(selected_candidate_id, {})
    return {
        "segment_id": str(base_segment.get("segment_id") or ""),
        "window_sec": [round(start_sec, 6), round(end_sec, 6)],
        "window_start_sec": round(start_sec, 6),
        "window_duration_sec": round(end_sec - start_sec, 6),
        "base_candidate_id": base_candidate_id,
        "selected_candidate_id": selected_candidate_id,
        "base_issues": segment_issues(base_segment),
        "selected_issues": segment_issues(selected_segment),
        "base_severity_score": as_float(base_segment.get("severity_score"), default=0.0),
        "selected_severity_score": as_float(selected_segment.get("severity_score"), default=0.0),
        "selected_segment_blocker": bool(selected_segment.get("blocker")),
        "selected_requires_post_repair": selected_requires_post_repair,
        "selection_reason": selection_reason,
        "selected_minus_path": candidate_minus_path(selected_row),
        "selected_vocal_path": candidate_vocal_path(selected_row),
    }


def unresolved_segment_row(base_segment: dict[str, Any]) -> dict[str, Any]:
    start_sec, end_sec = window_values(base_segment)
    return {
        "segment_id": str(base_segment.get("segment_id") or ""),
        "window_sec": [round(start_sec, 6), round(end_sec, 6)],
        "window_start_sec": round(start_sec, 6),
        "window_duration_sec": round(end_sec - start_sec, 6),
        "base_issues": segment_issues(base_segment),
        "base_severity_score": as_float(base_segment.get("severity_score"), default=0.0),
        "blocked_reason": "no_safe_non_blocking_alternate_segment",
    }


def missing_audio_inputs(
    *,
    base_candidate_id: str,
    base_row: dict[str, Any],
    selected_segments: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    checks = [
        (base_candidate_id, "source_audio_path", str(base_row.get("source_audio_path") or base_row.get("mix_path") or "")),
        (base_candidate_id, "base_vocal_path", candidate_vocal_path(base_row)),
        (base_candidate_id, "base_minus_path", candidate_minus_path(base_row)),
    ]
    for segment in selected_segments:
        candidate_id = str(segment.get("selected_candidate_id") or "")
        checks.append((candidate_id, "selected_vocal_path", str(segment.get("selected_vocal_path") or "")))
        checks.append((candidate_id, "selected_minus_path", str(segment.get("selected_minus_path") or "")))
    return [
        {"candidate_id": candidate_id, "path_role": role, "path": path}
        for candidate_id, role, path in checks
        if path_missing(path)
    ]


def recommended_render_args(
    *,
    self_judge_manifest_paths: Sequence[Path],
    render_output_dir: Path,
    base_candidate_id: str,
    crossfade_ms: int,
    allow_repairable_blocked_alternate: bool,
) -> list[str]:
    args = ["python", "/app/tools/render_no_reference_minus_segment_switch_candidate.py"]
    for path in self_judge_manifest_paths:
        args.extend(["--self-judge-manifest", str(path)])
    args.extend(
        [
            "--output-dir",
            str(render_output_dir),
            "--manifest-json",
            str(render_output_dir / "minus_segment_switch_manifest.json"),
            "--summary-md",
            str(render_output_dir / "minus_segment_switch_summary.md"),
            "--base-candidate-id",
            base_candidate_id,
            "--crossfade-ms",
            str(crossfade_ms),
        ]
    )
    if allow_repairable_blocked_alternate:
        args.append("--allow-repairable-blocked-alternate")
    return args


def base_plan_manifest(
    *,
    self_judge_manifest_paths: Sequence[Path],
    candidate_score_matrix_json: Path | None,
    render_output_dir: Path,
    base_candidate_id: str,
    base_selection_source: str,
    allow_repairable_blocked_alternate: bool,
) -> dict[str, Any]:
    return {
        "manifest_type": "no_reference_minus_segment_switch_plan",
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "diagnostic_plan_not_for_listen_pack",
        "release_status": "not_release_ready",
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "stem_type": "minus",
        "base_candidate_id": base_candidate_id,
        "base_selection_source": base_selection_source,
        "candidate_score_matrix_json": str(candidate_score_matrix_json or ""),
        "source_self_judge_manifest_paths": [str(path) for path in self_judge_manifest_paths],
        "render_output_dir": str(render_output_dir),
        "allow_repairable_blocked_alternate": allow_repairable_blocked_alternate,
        "expected_acceleration_mode": "external_only_cpu_dsp_planning",
        "gpu_inference_required": False,
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "plans_minus_segment_switch_from_existing_self_judge_manifests",
        "safety": dict(SAFETY_FLAGS),
        "forbidden_actions": list(FORBIDDEN_ACTIONS),
    }


def build_plan(
    *,
    self_judge_manifest_paths: Sequence[Path],
    candidate_score_matrix_json: Path | None,
    explicit_base_candidate_id: str | None,
    render_output_dir: Path,
    crossfade_ms: int = 80,
    allow_repairable_blocked_alternate: bool = False,
) -> dict[str, Any]:
    payloads = load_self_judge_payloads(self_judge_manifest_paths)
    base_candidate_id, base_selection_source = select_base_candidate_id(
        payloads=payloads,
        explicit_base_candidate_id=explicit_base_candidate_id,
        candidate_score_matrix_json=candidate_score_matrix_json,
    )
    manifest = base_plan_manifest(
        self_judge_manifest_paths=self_judge_manifest_paths,
        candidate_score_matrix_json=candidate_score_matrix_json,
        render_output_dir=render_output_dir,
        base_candidate_id=base_candidate_id,
        base_selection_source=base_selection_source,
        allow_repairable_blocked_alternate=allow_repairable_blocked_alternate,
    )
    manifests_by_id = {candidate_id_from_manifest(payload): payload for payload in payloads if candidate_id_from_manifest(payload)}
    candidate_rows = [candidate_row_from_manifest(payload) for payload in payloads]
    candidates_by_id = {str(row.get("candidate_id") or ""): row for row in candidate_rows if row.get("candidate_id")}
    base_payload = manifests_by_id.get(base_candidate_id)
    base_row = candidates_by_id.get(base_candidate_id)
    if base_payload is None or base_row is None:
        return {
            **manifest,
            "plan_status": "blocked_missing_base_candidate",
            "render_allowed": False,
            "dominant_failure_reason": "missing_base_candidate",
            "base_blocker_segment_count": 0,
            "selected_segment_count": 0,
            "unresolved_segment_count": 0,
            "selected_segments": [],
            "unresolved_segments": [],
            "missing_audio_inputs": [],
            "recommended_render_args": [],
        }

    base_blockers = sorted(self_judge_blockers(base_payload), key=lambda segment: str(segment.get("segment_id") or ""))
    alternate_manifests = [
        payload for payload in payloads if candidate_id_from_manifest(payload) and candidate_id_from_manifest(payload) != base_candidate_id
    ]
    selected_segments: list[dict[str, Any]] = []
    unresolved_segments: list[dict[str, Any]] = []
    for base_segment in base_blockers:
        alternate = choose_safe_alternate(
            base_segment,
            alternate_manifests,
            allow_repairable_blocked_alternate=allow_repairable_blocked_alternate,
        )
        if alternate is None:
            unresolved_segments.append(unresolved_segment_row(base_segment))
            continue
        selected_candidate_id, selected_segment, selection_reason, selected_requires_post_repair = alternate
        selected_segments.append(
            selected_segment_row(
                base_candidate_id=base_candidate_id,
                base_segment=base_segment,
                selected_candidate_id=selected_candidate_id,
                selected_segment=selected_segment,
                selection_reason=selection_reason,
                selected_requires_post_repair=selected_requires_post_repair,
                candidates_by_id=candidates_by_id,
            )
        )

    if not base_blockers:
        return {
            **manifest,
            "plan_status": "no_base_blocker_segments",
            "render_allowed": False,
            "dominant_failure_reason": "no_minus_blocker_to_switch",
            "base_blocker_segment_count": 0,
            "selected_segment_count": 0,
            "unresolved_segment_count": 0,
            "repairable_blocked_segment_count": 0,
            "selected_segments": [],
            "unresolved_segments": [],
            "missing_audio_inputs": [],
            "recommended_render_args": [],
        }

    missing_inputs = missing_audio_inputs(
        base_candidate_id=base_candidate_id,
        base_row=base_row,
        selected_segments=selected_segments,
    )
    render_allowed = bool(selected_segments and not unresolved_segments and not missing_inputs)
    if unresolved_segments:
        plan_status = "blocked_unresolved_segments"
        failure_reason = "no_safe_non_blocking_alternate_segment"
    elif missing_inputs:
        plan_status = "blocked_missing_audio_paths"
        failure_reason = "missing_audio_path_for_render"
    else:
        plan_status = "ready_for_render"
        failure_reason = ""
    return {
        **manifest,
        "plan_status": plan_status,
        "render_allowed": render_allowed,
        "dominant_failure_reason": failure_reason,
        "track_id": str(base_row.get("track_id") or ""),
        "source_audio_path": str(base_row.get("source_audio_path") or base_row.get("mix_path") or ""),
        "base_blocker_segment_count": len(base_blockers),
        "selected_segment_count": len(selected_segments),
        "unresolved_segment_count": len(unresolved_segments),
        "repairable_blocked_segment_count": repairable_blocked_segment_count(selected_segments),
        "selected_segments": selected_segments,
        "unresolved_segments": unresolved_segments,
        "missing_audio_inputs": missing_inputs,
        "recommended_render_args": recommended_render_args(
            self_judge_manifest_paths=self_judge_manifest_paths,
            render_output_dir=render_output_dir,
            base_candidate_id=base_candidate_id,
            crossfade_ms=crossfade_ms,
            allow_repairable_blocked_alternate=allow_repairable_blocked_alternate,
        )
        if render_allowed
        else [],
    }


def build_summary_md(manifest: dict[str, Any]) -> str:
    lines = [
        "# No-Reference Minus Segment Switch Plan",
        "",
        f"- plan_status: {manifest.get('plan_status')}",
        f"- render_allowed: {str(manifest.get('render_allowed') is True).lower()}",
        f"- base_candidate_id: {manifest.get('base_candidate_id')}",
        f"- base_selection_source: {manifest.get('base_selection_source')}",
        f"- base_blocker_segment_count: {manifest.get('base_blocker_segment_count')}",
        f"- selected_segment_count: {manifest.get('selected_segment_count')}",
        f"- unresolved_segment_count: {manifest.get('unresolved_segment_count')}",
        f"- delivery_allowed: {str(manifest.get('delivery_allowed') is True).lower()}",
        f"- release_ready: {str(manifest.get('release_ready') is True).lower()}",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan a no-reference minus segment-switch render from existing self-judge manifests."
    )
    parser.add_argument("--self-judge-manifest", type=Path, action="append", required=True)
    parser.add_argument("--candidate-score-matrix-json", type=Path)
    parser.add_argument("--base-candidate-id")
    parser.add_argument("--render-output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--crossfade-ms", type=int, default=80)
    parser.add_argument("--allow-repairable-blocked-alternate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="plan_no_reference_minus_segment_switch.py",
        entrypoint_path="/app/tools/plan_no_reference_minus_segment_switch.py",
    )
    args = parse_args(argv)
    manifest = build_plan(
        self_judge_manifest_paths=args.self_judge_manifest,
        candidate_score_matrix_json=args.candidate_score_matrix_json,
        explicit_base_candidate_id=args.base_candidate_id,
        render_output_dir=args.render_output_dir,
        crossfade_ms=args.crossfade_ms,
        allow_repairable_blocked_alternate=args.allow_repairable_blocked_alternate,
    )
    write_json(args.output_json, manifest)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(build_summary_md(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "plan_status": manifest.get("plan_status"),
                "render_allowed": manifest.get("render_allowed") is True,
                "base_candidate_id": manifest.get("base_candidate_id"),
                "selected_segment_count": manifest.get("selected_segment_count"),
                "unresolved_segment_count": manifest.get("unresolved_segment_count"),
                "output_json": str(args.output_json),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
