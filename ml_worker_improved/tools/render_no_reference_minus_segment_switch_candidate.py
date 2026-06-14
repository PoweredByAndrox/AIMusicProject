from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import soundfile as sf


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.render_no_reference_full_track_segment_switch_candidate import render_full_track_stem  # noqa: E402
from tools.render_no_reference_segment_switch_candidate import (  # noqa: E402
    SAFETY_FLAGS,
    candidate_minus_path,
    candidate_vocal_path,
    write_json,
)


BASE_FORBIDDEN_ACTIONS = [
    "do_not_require_reference_stems",
    "do_not_require_reference_vocal_minus",
    "do_not_require_human_csv",
    "do_not_start_finetune",
    "do_not_train",
    "do_not_add_models",
    "do_not_edit_model_registry",
    "do_not_add_cleanup_debleed_denoise",
    "do_not_relax_gates",
    "do_not_export_diagnostic_audio_as_result",
    "do_not_accept_without_post_render_self_judge",
]

REPAIRABLE_BLOCKED_ALTERNATE_ISSUES = {
    "vocal_projection_proxy",
    "vocal_stem_corr_proxy",
}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    payload = dict(payload)
    payload["source_self_judge_manifest_path"] = str(path)
    return payload


def as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def candidate_id_from_manifest(manifest: dict[str, Any]) -> str:
    candidate = manifest.get("candidate")
    if isinstance(candidate, dict):
        label = candidate.get("label") or candidate.get("candidate_id")
        if isinstance(label, str) and label:
            return label
    label = manifest.get("candidate_label") or manifest.get("candidate_id")
    return str(label or "")


def candidate_row_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    candidate = manifest.get("candidate")
    if not isinstance(candidate, dict):
        candidate = {}
    candidate_id = candidate_id_from_manifest(manifest)
    source_audio_path = str(manifest.get("source_audio_path") or "")
    source_sha = str(manifest.get("source_audio_sha256") or "")
    track_id = str(manifest.get("track_id") or source_sha[:12] or Path(source_audio_path).stem)
    instrumental_path = str(candidate.get("instrumental_path") or candidate.get("minus_path") or "")
    vocal_path = str(candidate.get("vocal_path") or "")
    return {
        "track_id": track_id,
        "candidate_id": candidate_id,
        "mix_path": source_audio_path,
        "source_audio_path": source_audio_path,
        "vocal_path": vocal_path,
        "selected_vocal_path": vocal_path,
        "instrumental_path": instrumental_path,
        "minus_path": instrumental_path,
        "selected_minus_path": instrumental_path,
        "source_self_judge_manifest_path": str(manifest.get("source_self_judge_manifest_path") or ""),
        "is_complete_pair": bool(source_audio_path and vocal_path and instrumental_path),
    }


def segments_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    segments = manifest.get("segments")
    if not isinstance(segments, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        segment_id = str(segment.get("segment_id") or "")
        if segment_id:
            result[segment_id] = segment
    return result


def window_values(segment: dict[str, Any]) -> tuple[float, float]:
    window = segment.get("window_sec")
    if not isinstance(window, list) or len(window) != 2:
        raise ValueError(f"Invalid self-judge segment window: {window!r}")
    start = as_float(window[0])
    end = as_float(window[1])
    if end <= start:
        raise ValueError(f"Invalid self-judge segment window: {window!r}")
    return start, end


def same_window(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_start, left_end = window_values(left)
    right_start, right_end = window_values(right)
    return round(left_start, 3) == round(right_start, 3) and round(left_end, 3) == round(right_end, 3)


def self_judge_blockers(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [segment for segment in segments_by_id(manifest).values() if bool(segment.get("blocker"))]


def segment_issues(segment: dict[str, Any]) -> list[str]:
    issues = segment.get("issues")
    if not isinstance(issues, list):
        return []
    return [str(issue) for issue in issues if str(issue)]


def is_repairable_blocked_alternate(segment: dict[str, Any]) -> bool:
    issues = set(segment_issues(segment))
    return bool(issues & REPAIRABLE_BLOCKED_ALTERNATE_ISSUES)


def choose_safe_alternate(
    base_segment: dict[str, Any],
    alternate_manifests: list[dict[str, Any]],
    *,
    allow_repairable_blocked_alternate: bool = False,
) -> tuple[str, dict[str, Any], str, bool] | None:
    segment_id = str(base_segment.get("segment_id") or "")
    base_severity = as_float(base_segment.get("severity_score"), default=0.0)
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    repairable_blocked_candidates: list[tuple[float, str, dict[str, Any]]] = []
    for manifest in alternate_manifests:
        candidate_id = candidate_id_from_manifest(manifest)
        if not candidate_id:
            continue
        segment = segments_by_id(manifest).get(segment_id)
        if segment is None:
            continue
        if not same_window(base_segment, segment):
            continue
        severity = as_float(segment.get("severity_score"), default=0.0)
        if bool(segment.get("blocker")):
            if (
                allow_repairable_blocked_alternate
                and severity < base_severity
                and is_repairable_blocked_alternate(segment)
            ):
                repairable_blocked_candidates.append((severity, candidate_id, segment))
            continue
        candidates.append((severity, candidate_id, segment))
    if candidates:
        _, candidate_id, segment = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
        return candidate_id, segment, "alternate_segment_has_no_self_judge_blocker", False
    if repairable_blocked_candidates:
        _, candidate_id, segment = sorted(repairable_blocked_candidates, key=lambda item: (item[0], item[1]))[0]
        return candidate_id, segment, "alternate_segment_has_lower_self_judge_severity_requires_post_repair", True
    return None


def repairable_blocked_segment_count(selected_segments: list[dict[str, Any]]) -> int:
    return sum(1 for segment in selected_segments if segment.get("selected_requires_post_repair") is True)


def base_manifest(self_judge_manifests: list[dict[str, Any]], base_candidate_id: str, output_dir: Path) -> dict[str, Any]:
    return {
        "manifest_type": "no_reference_minus_segment_switch_candidate",
        "track_id": (
            candidate_row_from_manifest(self_judge_manifests[0])["track_id"] if self_judge_manifests else ""
        ),
        "base_candidate_id": base_candidate_id,
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "preview_only_not_for_listen_pack",
        "release_status": "not_release_ready",
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "requires_post_render_minus_self_judge": True,
        "requires_dense_post_reconciliation_judge": True,
        "selected_segment_count": 0,
        "overlay_segment_count": 0,
        "blocked_segment_count": 0,
        "repairable_blocked_segment_count": 0,
        "selected_segments": [],
        "blocked_segments": [],
        "rendered_vocal_companion_path": "",
        "rendered_minus_candidate_path": "",
        "candidate_row_jsonl": str(output_dir / "minus_segment_switch_candidate_row.jsonl"),
        "safety_flags": dict(SAFETY_FLAGS),
        "forbidden_actions": list(BASE_FORBIDDEN_ACTIONS),
        "audio_copied": False,
        "audio_exported": False,
        "audio_exported_as_result": False,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "thresholds_relaxed": False,
        "cleanup_routes_added": False,
    }


def build_minus_segment_switch_candidate(
    self_judge_manifests: list[dict[str, Any]],
    *,
    output_dir: Path,
    base_candidate_id: str,
    crossfade_ms: int = 80,
    allow_repairable_blocked_alternate: bool = False,
) -> dict[str, Any]:
    manifest = base_manifest(self_judge_manifests, base_candidate_id, output_dir)
    manifests_by_id = {
        candidate_id_from_manifest(item): item
        for item in self_judge_manifests
        if candidate_id_from_manifest(item)
    }
    candidate_rows = [candidate_row_from_manifest(item) for item in self_judge_manifests]
    candidates_by_id = {str(row.get("candidate_id") or ""): row for row in candidate_rows if row.get("candidate_id")}
    base_self_judge = manifests_by_id.get(base_candidate_id)
    base_row = candidates_by_id.get(base_candidate_id)
    if base_self_judge is None or base_row is None:
        return {
            **manifest,
            "render_status": "blocked_missing_base_candidate",
            "dominant_failure_reason": "missing_base_candidate",
        }

    base_blockers = sorted(self_judge_blockers(base_self_judge), key=lambda segment: str(segment.get("segment_id") or ""))
    if not base_blockers:
        return {
            **manifest,
            "render_status": "blocked_no_base_blocker_segments",
            "dominant_failure_reason": "no_minus_blocker_to_switch",
        }

    alternate_manifests = [
        item for item in self_judge_manifests if candidate_id_from_manifest(item) and candidate_id_from_manifest(item) != base_candidate_id
    ]
    selected_segments: list[dict[str, Any]] = []
    blocked_segments: list[dict[str, Any]] = []
    for base_segment in base_blockers:
        segment_id = str(base_segment.get("segment_id") or "")
        start_sec, end_sec = window_values(base_segment)
        alternate = choose_safe_alternate(
            base_segment,
            alternate_manifests,
            allow_repairable_blocked_alternate=allow_repairable_blocked_alternate,
        )
        if alternate is None:
            blocked_segments.append(
                {
                    "segment_id": segment_id,
                    "window_sec": [round(start_sec, 6), round(end_sec, 6)],
                    "base_issues": segment_issues(base_segment),
                    "base_severity_score": as_float(base_segment.get("severity_score"), default=0.0),
                    "blocked_reason": "no_safe_non_blocking_alternate_segment",
                }
            )
            continue
        selected_candidate_id, selected_segment, selection_reason, selected_requires_post_repair = alternate
        selected_segments.append(
            {
                "segment_id": segment_id,
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
                "selected_minus_path": candidate_minus_path(candidates_by_id[selected_candidate_id]),
            }
        )

    manifest = {
        **manifest,
        "selected_segment_count": len(selected_segments),
        "blocked_segment_count": len(blocked_segments),
        "repairable_blocked_segment_count": repairable_blocked_segment_count(selected_segments),
        "allow_repairable_blocked_alternate": allow_repairable_blocked_alternate,
        "selected_segments": selected_segments,
        "blocked_segments": blocked_segments,
        "source_self_judge_manifest_paths": [
            str(item.get("source_self_judge_manifest_path") or "") for item in self_judge_manifests
        ],
    }
    if not selected_segments:
        return {
            **manifest,
            "render_status": "blocked_no_safe_alternate_segments",
            "dominant_failure_reason": "no_safe_non_blocking_alternate_segment",
        }

    base_vocal_path = candidate_vocal_path(base_row)
    missing_vocal_candidates = []
    if not base_vocal_path:
        missing_vocal_candidates.append(base_candidate_id)
    for segment in selected_segments:
        selected_candidate_id = str(segment.get("selected_candidate_id") or "")
        selected_row = candidates_by_id.get(selected_candidate_id, {})
        if not candidate_vocal_path(selected_row):
            missing_vocal_candidates.append(selected_candidate_id)
    if missing_vocal_candidates:
        return {
            **manifest,
            "render_status": "blocked_missing_vocal_companion_for_segment_switch",
            "dominant_failure_reason": "missing_vocal_companion_for_segment_switch",
            "missing_vocal_candidate_ids": sorted(set(missing_vocal_candidates)),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    vocal_output = output_dir / "minus_segment_switch_vocal_companion.wav"
    minus_output = output_dir / "minus_segment_switch_candidate.wav"
    vocal_rate, vocal_frames, vocal_overlays = render_full_track_stem(
        base_path=base_vocal_path,
        selected_windows=selected_segments,
        candidates_by_id=candidates_by_id,
        path_getter=candidate_vocal_path,
        output_path=vocal_output,
        crossfade_ms=crossfade_ms,
    )
    minus_rate, minus_frames, minus_overlays = render_full_track_stem(
        base_path=candidate_minus_path(base_row),
        selected_windows=selected_segments,
        candidates_by_id=candidates_by_id,
        path_getter=candidate_minus_path,
        output_path=minus_output,
        crossfade_ms=crossfade_ms,
    )
    base_vocal_info = sf.info(base_vocal_path)
    base_minus_info = sf.info(candidate_minus_path(base_row))
    duration_sec = round(float(minus_frames) / float(minus_rate), 6) if minus_rate else 0.0
    candidate_id = f"minus_segment_switch_{base_candidate_id}"
    candidate_row = {
        "track_id": manifest["track_id"],
        "candidate_id": candidate_id,
        "mix_path": str(base_row.get("mix_path") or ""),
        "source_audio_path": str(base_row.get("source_audio_path") or base_row.get("mix_path") or ""),
        "vocal_path": str(vocal_output),
        "instrumental_path": str(minus_output),
        "minus_path": str(minus_output),
        "selected_vocal_path": str(vocal_output),
        "selected_minus_path": str(minus_output),
        "candidate_family": "minus_segment_switch",
        "candidate_origin": "existing_self_judge_minus_window_overlay",
        "base_candidate_id": base_candidate_id,
        "final_decision": "minus_reconciliation_candidate",
        "requires_post_render_minus_self_judge": True,
        "requires_dense_post_reconciliation_judge": True,
        "needs_pair_reconciliation": True,
        "segment_blend_needed": False,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "delivery_allowed": False,
        "delivery_status": "blocked",
        "presentation_status": "preview_only_not_for_listen_pack",
        "release_status": "not_release_ready",
        "forbidden_actions": list(BASE_FORBIDDEN_ACTIONS),
        "is_complete_pair": bool(base_row.get("mix_path") and vocal_output),
    }
    candidate_row_jsonl = output_dir / "minus_segment_switch_candidate_row.jsonl"
    write_jsonl(candidate_row_jsonl, [candidate_row])
    return {
        **manifest,
        "render_status": "rendered_minus_segment_switch_candidate",
        "dominant_failure_reason": "",
        "audio_exported": True,
        "rendered_vocal_companion_path": str(vocal_output),
        "rendered_minus_candidate_path": str(minus_output),
        "sample_rate": minus_rate,
        "vocal_sample_rate": vocal_rate,
        "frames": int(minus_frames),
        "vocal_frames": int(vocal_frames),
        "duration_sec": duration_sec,
        "duration_preserved": bool(minus_frames == int(base_minus_info.frames)),
        "vocal_duration_preserved": bool(vocal_frames == int(base_vocal_info.frames)),
        "sample_rate_preserved": bool(minus_rate == int(base_minus_info.samplerate)),
        "vocal_sample_rate_preserved": bool(vocal_rate == int(base_vocal_info.samplerate)),
        "channels_preserved": True,
        "overlay_segment_count": int(minus_overlays),
        "vocal_overlay_segment_count": int(vocal_overlays),
        "candidate_row_jsonl": str(candidate_row_jsonl),
        "candidate_row": candidate_row,
    }


def build_summary(manifest: dict[str, Any]) -> str:
    lines = [
        "# No-Reference Minus Segment Switch Candidate Summary",
        "",
        f"- render_status: {manifest.get('render_status')}",
        f"- track_id: {manifest.get('track_id')}",
        f"- base_candidate_id: {manifest.get('base_candidate_id')}",
        f"- selected_segment_count: {manifest.get('selected_segment_count')}",
        f"- overlay_segment_count: {manifest.get('overlay_segment_count')}",
        f"- blocked_segment_count: {manifest.get('blocked_segment_count')}",
        f"- repairable_blocked_segment_count: {manifest.get('repairable_blocked_segment_count')}",
        f"- duration_sec: {manifest.get('duration_sec', 0.0)}",
        f"- duration_preserved: {str(manifest.get('duration_preserved') is True).lower()}",
        f"- delivery_allowed: {str(manifest.get('delivery_allowed') is True).lower()}",
        f"- can_mark_release_ready: {str(manifest.get('can_mark_release_ready') is True).lower()}",
        f"- requires_post_render_minus_self_judge: {str(manifest.get('requires_post_render_minus_self_judge') is True).lower()}",
        "",
        "## Safety",
        "",
    ]
    for key in (
        "training_started",
        "fine_tune_started",
        "models_changed",
        "model_registry_edited",
        "thresholds_relaxed",
        "cleanup_routes_added",
        "audio_exported_as_result",
    ):
        lines.append(f"- {key}: {str(manifest.get(key) is True).lower()}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a no-reference minus-only segment-switch candidate from existing self-judge manifests."
    )
    parser.add_argument("--self-judge-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--base-candidate-id", required=True)
    parser.add_argument("--crossfade-ms", type=int, default=80)
    parser.add_argument(
        "--allow-repairable-blocked-alternate",
        action="store_true",
        help=(
            "Allow a blocked alternate segment only when it has lower self-judge severity "
            "and repairable vocal leakage issues. Output remains blocked until post-render gates pass."
        ),
    )
    args = parser.parse_args(argv)

    manifest = build_minus_segment_switch_candidate(
        [read_json(path) for path in args.self_judge_manifest],
        output_dir=args.output_dir,
        base_candidate_id=args.base_candidate_id,
        crossfade_ms=args.crossfade_ms,
        allow_repairable_blocked_alternate=args.allow_repairable_blocked_alternate,
    )
    write_json(args.manifest_json, manifest)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(build_summary(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                key: manifest.get(key)
                for key in (
                    "render_status",
                    "selected_segment_count",
                    "overlay_segment_count",
                    "blocked_segment_count",
                    "duration_preserved",
                    "delivery_allowed",
                    "can_publish_as_result",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
