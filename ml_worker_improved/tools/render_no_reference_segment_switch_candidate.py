from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf
from scipy import signal


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


SEVERE_REASON_PREFIXES = ("severe_",)
SEVERE_REASONS = {
    "vocal_leakage_in_minus",
    "phase_or_stereo_inconsistency",
    "silence_hallucination",
    "robotic_or_noise_artifact",
    "transient_damage",
    "bad_mixture_reconstruction",
}

SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "diagnostic_audio_exported": False,
    "audio_copied": False,
    "audio_exported": False,
    "can_publish_as_result": False,
    "can_mark_release_ready": False,
}


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    text = str(raw_path)
    if text.startswith("/app/"):
        candidate = APP_ROOT / text.removeprefix("/app/")
        if candidate.exists():
            return candidate
    return path


def read_audio(path_text: str) -> tuple[np.ndarray, int]:
    path = resolve_path(path_text)
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if audio.size == 0:
        raise ValueError(f"Empty audio file: {path}")
    return audio, int(sample_rate)


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32)
    gcd = math.gcd(source_rate, target_rate)
    up = target_rate // gcd
    down = source_rate // gcd
    return signal.resample_poly(audio, up, down, axis=0).astype(np.float32)


def fit_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    if audio.shape[1] == channels:
        return audio
    if audio.shape[1] == 1:
        return np.repeat(audio, channels, axis=1)
    return audio[:, :channels]


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def has_severe_reason(window: dict[str, Any]) -> bool:
    reasons = as_list(window.get("reasons"))
    return any(reason.startswith(SEVERE_REASON_PREFIXES) or reason in SEVERE_REASONS for reason in reasons)


def metric(window: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(window.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def window_start(window: dict[str, Any]) -> float:
    for key in ("window_start_sec", "start_sec", "start"):
        if window.get(key) is not None:
            return metric(window, key)
    return 0.0


def window_duration(window: dict[str, Any]) -> float:
    for key in ("window_duration_sec", "duration_sec", "duration"):
        if window.get(key) is not None:
            return metric(window, key)
    return 0.0


def window_key(window: dict[str, Any]) -> tuple[float, float]:
    return (round(window_start(window), 6), round(window_duration(window), 6))


def candidate_minus_path(row: dict[str, Any]) -> str:
    return str(row.get("instrumental_path") or row.get("minus_path") or "")


def candidate_vocal_path(row: dict[str, Any]) -> str:
    return str(row.get("vocal_path") or "")


def best_safe_windows(
    judge_rows: list[dict[str, Any]],
    *,
    min_window_score: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    choices_by_window: dict[tuple[float, float], list[dict[str, Any]]] = {}
    blocked: list[dict[str, Any]] = []
    for row in sorted(judge_rows, key=lambda item: str(item.get("candidate_id") or "")):
        candidate_id = str(row.get("candidate_id") or "")
        for window in as_list_of_dicts(row.get("window_metrics")):
            key = window_key(window)
            if key[1] <= 0:
                continue
            score = metric(window, "minus_score", metric(window, "window_fitness"))
            enriched = {
                "candidate_id": candidate_id,
                "window_start_sec": key[0],
                "window_duration_sec": key[1],
                "minus_score": round(score, 6),
                "window_fitness": round(metric(window, "window_fitness", score), 6),
                "reasons": as_list(window.get("reasons")),
            }
            if score < min_window_score or has_severe_reason(window):
                blocked.append(enriched)
                continue
            choices_by_window.setdefault(key, []).append(enriched)
    selected: list[dict[str, Any]] = []
    for key, choices in sorted(choices_by_window.items()):
        selected.append(
            max(
                choices,
                key=lambda item: (
                    float(item["minus_score"]),
                    float(item["window_fitness"]),
                    str(item["candidate_id"]),
                ),
            )
        )
    expected_keys = sorted({window_key(window) for row in judge_rows for window in as_list_of_dicts(row.get("window_metrics")) if window_key(window)[1] > 0})
    selected_keys = {(row["window_start_sec"], row["window_duration_sec"]) for row in selected}
    missing = [
        {"window_start_sec": key[0], "window_duration_sec": key[1], "reason": "no_safe_minus_candidate_for_window"}
        for key in expected_keys
        if key not in selected_keys
    ]
    return selected, missing


def as_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def render_minus(
    selected_windows: list[dict[str, Any]],
    candidates_by_id: dict[str, dict[str, Any]],
    *,
    output_path: Path,
    target_sample_rate: int | None = None,
    crossfade_ms: int = 80,
) -> tuple[int, int]:
    first_candidate = candidates_by_id[str(selected_windows[0]["selected_candidate_id"])]
    first_audio, sample_rate = read_audio(candidate_minus_path(first_candidate))
    target_rate = int(target_sample_rate or sample_rate)
    first_audio = resample_audio(first_audio, sample_rate, target_rate)
    channels = first_audio.shape[1]
    total_duration = max(float(row["window_start_sec"]) + float(row["window_duration_sec"]) for row in selected_windows)
    total_frames = max(1, int(round(total_duration * target_rate)))
    rendered = np.zeros((total_frames, channels), dtype=np.float32)
    weight = np.zeros((total_frames, 1), dtype=np.float32)
    fade_frames = max(0, int(round((crossfade_ms / 1000.0) * target_rate)))

    audio_cache: dict[str, np.ndarray] = {str(selected_windows[0]["selected_candidate_id"]): first_audio}
    for window in selected_windows:
        candidate_id = str(window["selected_candidate_id"])
        if candidate_id not in audio_cache:
            audio, source_rate = read_audio(candidate_minus_path(candidates_by_id[candidate_id]))
            audio_cache[candidate_id] = fit_channels(resample_audio(audio, source_rate, target_rate), channels)
        source = audio_cache[candidate_id]
        start = max(0, int(round(float(window["window_start_sec"]) * target_rate)))
        duration = max(1, int(round(float(window["window_duration_sec"]) * target_rate)))
        end = min(total_frames, start + duration)
        if end <= start:
            continue
        segment = source[start:end]
        if segment.shape[0] < end - start:
            padded = np.zeros((end - start, channels), dtype=np.float32)
            padded[: segment.shape[0]] = segment
            segment = padded
        curve = np.ones((end - start, 1), dtype=np.float32)
        if fade_frames > 0 and curve.shape[0] > 2:
            fade = min(fade_frames, curve.shape[0] // 2)
            ramp_in = np.linspace(0.0, 1.0, fade, endpoint=True, dtype=np.float32).reshape(-1, 1)
            ramp_out = np.linspace(1.0, 0.0, fade, endpoint=True, dtype=np.float32).reshape(-1, 1)
            curve[:fade] *= ramp_in
            curve[-fade:] *= ramp_out
        rendered[start:end] += segment.astype(np.float32) * curve
        weight[start:end] += curve
    rendered = rendered / np.maximum(weight, 1.0e-6)
    rendered = np.clip(rendered, -1.0, 1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), rendered, target_rate, subtype="PCM_24")
    return target_rate, int(rendered.shape[0])


def build_segment_switch_candidate(
    candidate_rows: list[dict[str, Any]],
    minus_windowed_judge_rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    min_window_score: float = 0.72,
    crossfade_ms: int = 80,
) -> dict[str, Any]:
    candidates_by_id = {str(row.get("candidate_id") or ""): row for row in candidate_rows if row.get("candidate_id")}
    selected, missing = best_safe_windows(minus_windowed_judge_rows, min_window_score=min_window_score)
    selected = [
        {
            **row,
            "selected_candidate_id": row["candidate_id"],
            "selected_minus_path": candidate_minus_path(candidates_by_id.get(str(row["candidate_id"]), {})),
        }
        for row in selected
        if str(row["candidate_id"]) in candidates_by_id
    ]
    base_manifest = {
        "manifest_type": "no_reference_segment_switch_candidate",
        "track_id": str(candidate_rows[0].get("track_id") if candidate_rows else ""),
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "preview_only_not_for_listen_pack",
        "release_status": "not_release_ready",
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "requires_post_render_judge": True,
        "selected_window_count": len(selected),
        "blocked_window_count": len(missing),
        "selected_windows": selected,
        "blocked_windows": missing,
        "safety_flags": dict(SAFETY_FLAGS),
        "audio_copied": False,
        "training_started": False,
        "models_changed": False,
        "thresholds_relaxed": False,
        "cleanup_routes_added": False,
        "rendered_minus_preview_path": "",
        "candidate_row_jsonl": "",
    }
    if not selected or missing:
        return {
            **base_manifest,
            "render_status": "blocked_incomplete_window_coverage",
            "audio_exported": False,
            "dominant_failure_reason": "incomplete_safe_minus_window_coverage",
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_path = output_dir / "segment_switch_minus_preview.flac"
    sample_rate, frames = render_minus(
        selected,
        candidates_by_id,
        output_path=rendered_path,
        crossfade_ms=crossfade_ms,
    )
    vocal_path = choose_vocal_path(candidate_rows, minus_windowed_judge_rows)
    candidate_row = {
        "track_id": base_manifest["track_id"],
        "candidate_id": "segment_switch_minus_preview",
        "mix_path": str(candidate_rows[0].get("mix_path") if candidate_rows else ""),
        "vocal_path": vocal_path,
        "instrumental_path": str(rendered_path),
        "minus_path": str(rendered_path),
        "candidate_family": "segment_switch_minus_preview",
        "candidate_origin": "existing_candidates_window_switch",
        "is_complete_pair": bool(vocal_path),
    }
    candidate_row_jsonl = output_dir / "segment_switch_candidate_row.jsonl"
    with candidate_row_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(candidate_row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        **base_manifest,
        "render_status": "rendered_preview_candidate",
        "audio_exported": True,
        "dominant_failure_reason": "",
        "rendered_minus_preview_path": str(rendered_path),
        "sample_rate": sample_rate,
        "frames": frames,
        "duration_sec": round(float(frames) / float(sample_rate), 6) if sample_rate else 0.0,
        "candidate_row_jsonl": str(candidate_row_jsonl),
        "candidate_row": candidate_row,
    }


def choose_vocal_path(candidate_rows: list[dict[str, Any]], judge_rows: list[dict[str, Any]]) -> str:
    scores: dict[str, float] = {}
    for row in judge_rows:
        candidate_id = str(row.get("candidate_id") or "")
        windows = as_list_of_dicts(row.get("window_metrics"))
        if not windows:
            continue
        scores[candidate_id] = sum(metric(window, "window_fitness") for window in windows) / float(len(windows))
    ranked_ids = sorted(scores, key=lambda candidate_id: (scores[candidate_id], candidate_id), reverse=True)
    by_id = {str(row.get("candidate_id") or ""): row for row in candidate_rows}
    for candidate_id in ranked_ids:
        vocal_path = candidate_vocal_path(by_id.get(candidate_id, {}))
        if vocal_path:
            return vocal_path
    for row in candidate_rows:
        vocal_path = candidate_vocal_path(row)
        if vocal_path:
            return vocal_path
    return ""


def build_summary(manifest: dict[str, Any]) -> str:
    selected_by_candidate = Counter(str(row.get("selected_candidate_id") or "") for row in manifest.get("selected_windows", []))
    lines = [
        "# No-Reference Segment Switch Candidate",
        "",
        f"- render_status: {manifest['render_status']}",
        f"- selected_window_count: {manifest['selected_window_count']}",
        f"- blocked_window_count: {manifest['blocked_window_count']}",
        f"- delivery_allowed: {str(manifest['delivery_allowed']).lower()}",
        f"- can_publish_as_result: {str(manifest['can_publish_as_result']).lower()}",
        f"- can_mark_release_ready: {str(manifest['can_mark_release_ready']).lower()}",
        f"- requires_post_render_judge: {str(manifest['requires_post_render_judge']).lower()}",
        f"- rendered_minus_preview_path: {manifest.get('rendered_minus_preview_path') or ''}",
        "",
        "## Selected Windows By Candidate",
        "",
    ]
    for candidate_id, count in sorted(selected_by_candidate.items()):
        lines.append(f"- {candidate_id}: {count}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a preview-only no-reference minus candidate by switching existing candidates per safe window.")
    parser.add_argument("--candidate-pool-jsonl", type=Path, required=True)
    parser.add_argument("--minus-windowed-judge-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--min-window-score", type=float, default=0.72)
    parser.add_argument("--crossfade-ms", type=int, default=80)
    args = parser.parse_args(argv)
    manifest = build_segment_switch_candidate(
        list(iter_jsonl(args.candidate_pool_jsonl)),
        list(iter_jsonl(args.minus_windowed_judge_jsonl)),
        output_dir=args.output_dir,
        min_window_score=args.min_window_score,
        crossfade_ms=args.crossfade_ms,
    )
    write_json(args.manifest_json, manifest)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(build_summary(manifest), encoding="utf-8")
    print(json.dumps({key: manifest.get(key) for key in ("render_status", "selected_window_count", "blocked_window_count", "delivery_allowed", "can_publish_as_result")}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
