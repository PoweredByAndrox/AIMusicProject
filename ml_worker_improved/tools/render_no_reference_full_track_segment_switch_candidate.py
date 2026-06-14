from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.render_no_reference_pair_segment_switch_candidate import best_safe_windows_for_score  # noqa: E402
from tools.render_no_reference_segment_switch_candidate import (  # noqa: E402
    SAFETY_FLAGS,
    candidate_minus_path,
    candidate_vocal_path,
    fit_channels,
    iter_jsonl,
    read_audio,
    resample_audio,
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
    "do_not_accept_without_dense_post_reconciliation_judge",
]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def match_length(audio: np.ndarray, frames: int) -> np.ndarray:
    if audio.shape[0] == frames:
        return audio
    if audio.shape[0] > frames:
        return audio[:frames]
    padded = np.zeros((frames, audio.shape[1]), dtype=np.float32)
    padded[: audio.shape[0]] = audio
    return padded


def load_like_base(path: str, *, base_rate: int, base_shape: tuple[int, int]) -> np.ndarray:
    audio, sample_rate = read_audio(path)
    if sample_rate != base_rate:
        audio = resample_audio(audio, sample_rate, base_rate)
    audio = fit_channels(audio, base_shape[1])
    return match_length(audio.astype(np.float32), base_shape[0])


def overlay_window(
    base: np.ndarray,
    alternate: np.ndarray,
    *,
    sample_rate: int,
    start_sec: float,
    duration_sec: float,
    crossfade_ms: int,
) -> None:
    start = max(0, min(base.shape[0], int(round(start_sec * sample_rate))))
    end = max(start, min(base.shape[0], int(round((start_sec + duration_sec) * sample_rate))))
    if end <= start:
        return
    length = end - start
    weights = np.ones((length, 1), dtype=np.float32)
    fade_frames = max(0, min(int(round(sample_rate * crossfade_ms / 1000.0)), length // 2))
    if fade_frames > 0:
        weights[:fade_frames, 0] = np.linspace(0.0, 1.0, fade_frames, endpoint=False, dtype=np.float32)
        weights[-fade_frames:, 0] = np.linspace(1.0, 0.0, fade_frames, dtype=np.float32)
    base[start:end] = (base[start:end] * (1.0 - weights)) + (alternate[start:end] * weights)


def render_full_track_stem(
    *,
    base_path: str,
    selected_windows: list[dict[str, Any]],
    candidates_by_id: dict[str, dict[str, Any]],
    path_getter,
    output_path: Path,
    crossfade_ms: int,
) -> tuple[int, int, int]:
    rendered, sample_rate = read_audio(base_path)
    rendered = rendered.astype(np.float32)
    base_shape = (rendered.shape[0], rendered.shape[1])
    audio_cache: dict[str, np.ndarray] = {}
    overlays = 0
    for window in selected_windows:
        candidate_id = str(window.get("selected_candidate_id") or window.get("candidate_id") or "")
        if not candidate_id or candidate_id not in candidates_by_id:
            continue
        alternate_path = path_getter(candidates_by_id[candidate_id])
        if not alternate_path or alternate_path == base_path:
            continue
        if candidate_id not in audio_cache:
            audio_cache[candidate_id] = load_like_base(alternate_path, base_rate=sample_rate, base_shape=base_shape)
        overlay_window(
            rendered,
            audio_cache[candidate_id],
            sample_rate=sample_rate,
            start_sec=float(window["window_start_sec"]),
            duration_sec=float(window["window_duration_sec"]),
            crossfade_ms=crossfade_ms,
        )
        overlays += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), rendered, sample_rate, subtype="FLOAT")
    return sample_rate, int(rendered.shape[0]), overlays


def base_manifest(candidate_rows: list[dict[str, Any]], base_candidate_id: str, output_dir: Path) -> dict[str, Any]:
    return {
        "manifest_type": "no_reference_full_track_segment_switch_candidate",
        "track_id": str(candidate_rows[0].get("track_id") if candidate_rows else ""),
        "base_candidate_id": base_candidate_id,
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "preview_only_not_for_listen_pack",
        "release_status": "not_release_ready",
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "requires_dense_post_reconciliation_judge": True,
        "selected_window_count": 0,
        "overlay_window_count": 0,
        "blocked_window_count": 0,
        "selected_windows": [],
        "blocked_windows": [],
        "rendered_vocal_candidate_path": "",
        "rendered_minus_candidate_path": "",
        "candidate_row_jsonl": str(output_dir / "full_track_segment_switch_candidate_row.jsonl"),
        "safety_flags": dict(SAFETY_FLAGS),
        "forbidden_actions": list(BASE_FORBIDDEN_ACTIONS),
        "audio_copied": False,
        "audio_exported": False,
        "audio_exported_as_result": False,
        "training_started": False,
        "models_changed": False,
        "thresholds_relaxed": False,
        "cleanup_routes_added": False,
    }


def build_full_track_segment_switch_candidate(
    candidate_rows: list[dict[str, Any]],
    windowed_judge_rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    base_candidate_id: str,
    min_window_score: float = 0.72,
    crossfade_ms: int = 80,
    selection_score_field: str = "minus_score",
    selection_decision_field: str | None = None,
    allow_select_alternative: bool = False,
    block_severe_reasons: bool = True,
    allow_low_activity_proxy_gap: bool = False,
) -> dict[str, Any]:
    candidates_by_id = {str(row.get("candidate_id") or ""): row for row in candidate_rows if row.get("candidate_id")}
    manifest = base_manifest(candidate_rows, base_candidate_id, output_dir)
    base_row = candidates_by_id.get(base_candidate_id)
    if base_row is None:
        return {
            **manifest,
            "render_status": "blocked_missing_base_candidate",
            "dominant_failure_reason": "missing_base_candidate",
        }

    selected, missing = best_safe_windows_for_score(
        windowed_judge_rows,
        min_window_score=min_window_score,
        score_field=selection_score_field,
        decision_field=selection_decision_field,
        allow_select_alternative=allow_select_alternative,
        block_severe_reasons=block_severe_reasons,
        allow_low_activity_proxy_gap=allow_low_activity_proxy_gap,
    )
    selected = [
        {
            **row,
            "selected_candidate_id": str(row["candidate_id"]),
            "selected_vocal_path": candidate_vocal_path(candidates_by_id.get(str(row["candidate_id"]), {})),
            "selected_minus_path": candidate_minus_path(candidates_by_id.get(str(row["candidate_id"]), {})),
        }
        for row in selected
        if str(row.get("candidate_id") or "") in candidates_by_id
    ]
    alternate_selected = [row for row in selected if str(row.get("selected_candidate_id") or "") != base_candidate_id]
    manifest = {
        **manifest,
        "selected_window_count": len(selected),
        "blocked_window_count": len(missing),
        "selected_windows": selected,
        "blocked_windows": missing,
    }
    if not alternate_selected:
        return {
            **manifest,
            "render_status": "blocked_no_alternate_window_selected",
            "dominant_failure_reason": "no_full_track_overlay_improvement",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    vocal_output = output_dir / "full_track_segment_switch_vocal_candidate.wav"
    minus_output = output_dir / "full_track_segment_switch_minus_candidate.wav"
    vocal_rate, vocal_frames, vocal_overlays = render_full_track_stem(
        base_path=candidate_vocal_path(base_row),
        selected_windows=alternate_selected,
        candidates_by_id=candidates_by_id,
        path_getter=candidate_vocal_path,
        output_path=vocal_output,
        crossfade_ms=crossfade_ms,
    )
    minus_rate, minus_frames, minus_overlays = render_full_track_stem(
        base_path=candidate_minus_path(base_row),
        selected_windows=alternate_selected,
        candidates_by_id=candidates_by_id,
        path_getter=candidate_minus_path,
        output_path=minus_output,
        crossfade_ms=crossfade_ms,
    )
    frames = min(vocal_frames, minus_frames)
    duration_sec = round(float(frames) / float(vocal_rate), 6) if vocal_rate else 0.0
    candidate_id = f"full_track_segment_switch_{base_candidate_id}"
    candidate_row = {
        "track_id": manifest["track_id"],
        "candidate_id": candidate_id,
        "mix_path": str(base_row.get("mix_path") or ""),
        "vocal_path": str(vocal_output),
        "instrumental_path": str(minus_output),
        "minus_path": str(minus_output),
        "selected_vocal_path": str(vocal_output),
        "selected_minus_path": str(minus_output),
        "candidate_family": "full_track_segment_switch",
        "candidate_origin": "existing_candidates_full_track_window_overlay",
        "base_candidate_id": base_candidate_id,
        "final_decision": "pair_reconciliation_candidate",
        "requires_dense_post_reconciliation_judge": True,
        "needs_pair_reconciliation": False,
        "segment_blend_needed": False,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "delivery_allowed": False,
        "delivery_status": "blocked",
        "presentation_status": "preview_only_not_for_listen_pack",
        "release_status": "not_release_ready",
        "forbidden_actions": list(BASE_FORBIDDEN_ACTIONS),
        "is_complete_pair": True,
    }
    candidate_row_jsonl = output_dir / "full_track_segment_switch_candidate_row.jsonl"
    write_jsonl(candidate_row_jsonl, [candidate_row])
    base_vocal_info = sf.info(candidate_vocal_path(base_row))
    base_minus_info = sf.info(candidate_minus_path(base_row))
    return {
        **manifest,
        "render_status": "rendered_full_track_candidate",
        "dominant_failure_reason": "",
        "audio_exported": True,
        "rendered_vocal_candidate_path": str(vocal_output),
        "rendered_minus_candidate_path": str(minus_output),
        "sample_rate": vocal_rate,
        "minus_sample_rate": minus_rate,
        "frames": frames,
        "duration_sec": duration_sec,
        "duration_preserved": bool(vocal_frames == int(base_vocal_info.frames) and minus_frames == int(base_minus_info.frames)),
        "sample_rate_preserved": bool(vocal_rate == int(base_vocal_info.samplerate) and minus_rate == int(base_minus_info.samplerate)),
        "channels_preserved": True,
        "overlay_window_count": int(max(vocal_overlays, minus_overlays)),
        "candidate_row_jsonl": str(candidate_row_jsonl),
        "candidate_row": candidate_row,
    }


def build_summary(manifest: dict[str, Any]) -> str:
    lines = [
        "# No-Reference Full-Track Segment Switch Candidate Summary",
        "",
        f"- render_status: {manifest.get('render_status')}",
        f"- track_id: {manifest.get('track_id')}",
        f"- base_candidate_id: {manifest.get('base_candidate_id')}",
        f"- selected_window_count: {manifest.get('selected_window_count')}",
        f"- overlay_window_count: {manifest.get('overlay_window_count')}",
        f"- blocked_window_count: {manifest.get('blocked_window_count')}",
        f"- duration_sec: {manifest.get('duration_sec', 0.0)}",
        f"- duration_preserved: {str(manifest.get('duration_preserved') is True).lower()}",
        f"- delivery_allowed: {str(manifest.get('delivery_allowed') is True).lower()}",
        f"- can_mark_release_ready: {str(manifest.get('can_mark_release_ready') is True).lower()}",
        f"- requires_dense_post_reconciliation_judge: {str(manifest.get('requires_dense_post_reconciliation_judge') is True).lower()}",
        "",
        "## Safety",
        "",
    ]
    for key in (
        "training_started",
        "models_changed",
        "thresholds_relaxed",
        "cleanup_routes_added",
        "audio_exported_as_result",
    ):
        lines.append(f"- {key}: {str(manifest.get(key) is True).lower()}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render full-duration no-reference segment-switch candidate from existing candidates.")
    parser.add_argument("--candidate-pool-jsonl", type=Path, required=True)
    parser.add_argument("--windowed-judge-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--base-candidate-id", required=True)
    parser.add_argument("--min-window-score", type=float, default=0.72)
    parser.add_argument("--crossfade-ms", type=int, default=80)
    parser.add_argument("--selection-score-field", default="minus_score")
    parser.add_argument("--selection-decision-field", default="")
    parser.add_argument("--allow-select-alternative", action="store_true")
    parser.add_argument("--allow-low-activity-proxy-gap", action="store_true")
    parser.add_argument("--ignore-severe-reasons-for-selection", action="store_true")
    args = parser.parse_args(argv)
    manifest = build_full_track_segment_switch_candidate(
        list(iter_jsonl(args.candidate_pool_jsonl)),
        list(iter_jsonl(args.windowed_judge_jsonl)),
        output_dir=args.output_dir,
        base_candidate_id=args.base_candidate_id,
        min_window_score=args.min_window_score,
        crossfade_ms=args.crossfade_ms,
        selection_score_field=args.selection_score_field,
        selection_decision_field=args.selection_decision_field or None,
        allow_select_alternative=args.allow_select_alternative,
        block_severe_reasons=not args.ignore_severe_reasons_for_selection,
        allow_low_activity_proxy_gap=args.allow_low_activity_proxy_gap,
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
                    "selected_window_count",
                    "overlay_window_count",
                    "blocked_window_count",
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
