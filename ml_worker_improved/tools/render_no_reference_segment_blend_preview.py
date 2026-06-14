from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.run_no_reference_optimizer import as_list, stable_unique  # noqa: E402


BASE_FORBIDDEN_ACTIONS = [
    "do_not_require_reference_stems",
    "do_not_require_reference_vocal_minus",
    "do_not_require_human_csv",
    "do_not_start_finetune",
    "do_not_train",
    "do_not_add_models",
    "do_not_add_cleanup_debleed_denoise",
    "do_not_relax_gates",
    "do_not_mark_release_ready",
    "do_not_publish_as_final_result",
]

SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "original_files_overwritten": False,
}

MANIFEST_VERSION = "no_reference_segment_blend_preview_v1"


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_audio(path_text: str) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path_text, always_2d=True, dtype="float32")
    if audio.size == 0:
        raise ValueError(f"Empty audio file: {path_text}")
    return audio, int(sample_rate)


def write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    divisor = math.gcd(int(source_rate), int(target_rate))
    up = int(target_rate) // divisor
    down = int(source_rate) // divisor
    return resample_poly(audio, up, down, axis=0).astype(np.float32)


def match_audio_shape(audio: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_frames, target_channels = target_shape
    if audio.shape[1] != target_channels:
        raise ValueError(f"Incompatible channel count after resample: {audio.shape[1]} != {target_channels}")
    if audio.shape[0] > target_frames:
        return audio[:target_frames].astype(np.float32, copy=False)
    if audio.shape[0] < target_frames:
        pad = np.zeros((target_frames - audio.shape[0], target_channels), dtype=np.float32)
        return np.concatenate([audio, pad], axis=0)
    return audio.astype(np.float32, copy=False)


def slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return text.strip("._-")[:120] or "unknown"


def short_slug(value: str, *, prefix_length: int = 36) -> str:
    base = slug(value)[:prefix_length].strip("._-") or "unknown"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{base}_{digest}"


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def normalize_source_stem(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"vocal", "vocals"}:
        return "vocal"
    if text in {"minus", "instrumental"}:
        return "minus"
    if text == "both":
        return "both"
    return "unknown"


def should_blend_stem(source_stem: str, target_stem: str) -> bool:
    return source_stem == "both" or source_stem == target_stem


def blend_audio(
    base: np.ndarray,
    alternate: np.ndarray,
    sample_rate: int,
    *,
    start_sec: float,
    duration_sec: float,
    crossfade_ms: int,
    alternate_weight: float = 1.0,
) -> np.ndarray:
    if base.shape != alternate.shape:
        raise ValueError(f"Incompatible audio shapes: base={base.shape}, alternate={alternate.shape}")
    if not 0.0 <= float(alternate_weight) <= 1.0:
        raise ValueError("alternate_weight must be in [0, 1]")
    result = np.array(base, copy=True)
    start = max(0, min(base.shape[0], int(round(start_sec * sample_rate))))
    end = max(start, min(base.shape[0], int(round((start_sec + duration_sec) * sample_rate))))
    if end <= start:
        return result
    length = end - start
    fade_frames = max(0, min(int(round((crossfade_ms / 1000.0) * sample_rate)), length // 2))
    weights = np.ones((length, 1), dtype=np.float32)
    if fade_frames > 0:
        weights[:fade_frames, 0] = np.linspace(0.0, 1.0, fade_frames, endpoint=False, dtype=np.float32)
        weights[-fade_frames:, 0] = np.linspace(1.0, 0.0, fade_frames, endpoint=False, dtype=np.float32)
    weights *= float(alternate_weight)
    result[start:end] = (base[start:end] * (1.0 - weights)) + (alternate[start:end] * weights)
    return result.astype(np.float32)


def render_stem(
    *,
    base_path: str,
    alternate_path: str,
    output_path: Path,
    source_stem: str,
    target_stem: str,
    start_sec: float,
    duration_sec: float,
    crossfade_ms: int,
    alternate_weight: float = 1.0,
) -> dict[str, Any]:
    base, base_rate = read_audio(base_path)
    alternate, alternate_rate = read_audio(alternate_path)
    resample_applied = False
    if base_rate != alternate_rate:
        alternate = resample_audio(alternate, alternate_rate, base_rate)
        alternate = match_audio_shape(alternate, base.shape)
        resample_applied = True
    if should_blend_stem(source_stem, target_stem):
        rendered = blend_audio(
            base,
            alternate,
            base_rate,
            start_sec=start_sec,
            duration_sec=duration_sec,
            crossfade_ms=crossfade_ms,
            alternate_weight=alternate_weight,
        )
    else:
        rendered = np.array(base, copy=True)
    write_audio(output_path, rendered, base_rate)
    output_info = sf.info(str(output_path))
    base_info = sf.info(str(base_path))
    return {
        "sample_rate_preserved": int(output_info.samplerate) == int(base_info.samplerate),
        "channels_preserved": int(output_info.channels) == int(base_info.channels),
        "duration_preserved": int(output_info.frames) == int(base_info.frames),
        "sample_rate_resample_applied": resample_applied,
    }


def row_key(row: dict[str, Any]) -> tuple[str, float, float, str, str, str]:
    return (
        str(row.get("track_id") or ""),
        as_float(row.get("window_start_sec")),
        as_float(row.get("window_duration_sec")),
        str(row.get("source_stem") or ""),
        str(row.get("base_candidate_id") or ""),
        str(row.get("alternate_candidate_id") or ""),
    )


def ready_rows(contract_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [row for row in contract_rows if str(row.get("render_status") or "") == "render_ready"],
        key=row_key,
    )


def preview_paths(row: dict[str, Any], output_dir: Path, index: int) -> tuple[Path, Path]:
    prefix = "__".join(
        [
            f"{index:04d}",
            short_slug(str(row.get("track_id") or ""), prefix_length=44),
            short_slug(str(row.get("base_candidate_id") or ""), prefix_length=28),
            short_slug(str(row.get("alternate_candidate_id") or ""), prefix_length=28),
        ]
    )
    return output_dir / f"{prefix}__vocal_preview.flac", output_dir / f"{prefix}__minus_preview.flac"


def assert_preview_path(output_dir: Path, output_path: Path, original_paths: Sequence[str]) -> None:
    resolved_output_dir = output_dir.resolve()
    resolved_output = output_path.resolve()
    if resolved_output.parent != resolved_output_dir:
        raise ValueError(f"Preview output path must stay in {resolved_output_dir}: {resolved_output}")
    for original in original_paths:
        if original and resolved_output == Path(original).resolve():
            raise ValueError(f"Preview output would overwrite original file: {resolved_output}")


def render_preview_row(row: dict[str, Any], output_dir: Path, ordinal: int) -> dict[str, Any]:
    source_stem = normalize_source_stem(row.get("source_stem"))
    vocal_output, minus_output = preview_paths(row, output_dir, ordinal)
    original_paths = [
        str(row.get("base_vocal_path") or ""),
        str(row.get("base_minus_path") or ""),
        str(row.get("alt_vocal_path") or ""),
        str(row.get("alt_minus_path") or ""),
    ]
    assert_preview_path(output_dir, vocal_output, original_paths)
    assert_preview_path(output_dir, minus_output, original_paths)
    start_sec = as_float(row.get("window_start_sec"))
    duration_sec = as_float(row.get("window_duration_sec"))
    crossfade_ms = int(row.get("crossfade_ms")) if row.get("crossfade_ms") is not None else 80
    alternate_weight = as_float(row.get("alternate_weight"), default=1.0)
    vocal_shape = render_stem(
        base_path=str(row.get("base_vocal_path") or ""),
        alternate_path=str(row.get("alt_vocal_path") or ""),
        output_path=vocal_output,
        source_stem=source_stem,
        target_stem="vocal",
        start_sec=start_sec,
        duration_sec=duration_sec,
        crossfade_ms=crossfade_ms,
        alternate_weight=alternate_weight,
    )
    minus_shape = render_stem(
        base_path=str(row.get("base_minus_path") or ""),
        alternate_path=str(row.get("alt_minus_path") or ""),
        output_path=minus_output,
        source_stem=source_stem,
        target_stem="minus",
        start_sec=start_sec,
        duration_sec=duration_sec,
        crossfade_ms=crossfade_ms,
        alternate_weight=alternate_weight,
    )
    return {
        "track_id": str(row.get("track_id") or ""),
        "base_candidate_id": str(row.get("base_candidate_id") or ""),
        "alternate_candidate_id": str(row.get("alternate_candidate_id") or ""),
        "rendered_vocal_preview_path": str(vocal_output),
        "rendered_minus_preview_path": str(minus_output),
        "source_stem": source_stem,
        "window_start_sec": round(start_sec, 6),
        "window_duration_sec": round(duration_sec, 6),
        "crossfade_ms": crossfade_ms,
        "alternate_weight": round(alternate_weight, 6),
        "render_status": "render_ready",
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "requires_post_render_judge": True,
        "sample_rate_preserved": bool(vocal_shape["sample_rate_preserved"] and minus_shape["sample_rate_preserved"]),
        "sample_rate_resample_applied": bool(vocal_shape["sample_rate_resample_applied"] or minus_shape["sample_rate_resample_applied"]),
        "channels_preserved": bool(vocal_shape["channels_preserved"] and minus_shape["channels_preserved"]),
        "duration_preserved": bool(vocal_shape["duration_preserved"] and minus_shape["duration_preserved"]),
        "safety_flags": dict(SAFETY_FLAGS),
        "forbidden_actions": stable_unique(BASE_FORBIDDEN_ACTIONS + as_list(row.get("forbidden_actions"))),
    }


def build_summary(rows: list[dict[str, Any]], *, render_ready_count: int, total_contract_rows: int) -> dict[str, Any]:
    return {
        "render_ready_rows": render_ready_count,
        "previews_rendered": len(rows),
        "previews_skipped": max(0, total_contract_rows - len(rows)),
        "rendered_vocal_preview_path": rows[0]["rendered_vocal_preview_path"] if rows else "",
        "rendered_minus_preview_path": rows[0]["rendered_minus_preview_path"] if rows else "",
        "duration_preserved": all(row.get("duration_preserved") is True for row in rows) if rows else False,
        "sample_rate_preserved": all(row.get("sample_rate_preserved") is True for row in rows) if rows else False,
        "channels_preserved": all(row.get("channels_preserved") is True for row in rows) if rows else False,
        "can_publish_as_result_count": sum(1 for row in rows if row.get("can_publish_as_result") is True),
        "can_mark_release_ready_count": sum(1 for row in rows if row.get("can_mark_release_ready") is True),
        "requires_post_render_judge_count": sum(1 for row in rows if row.get("requires_post_render_judge") is True),
        "safety": dict(SAFETY_FLAGS),
    }


def build_preview_result(
    dry_run_rows: list[dict[str, Any]],
    contract_rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    max_render_count: int = 1,
) -> dict[str, Any]:
    del dry_run_rows
    candidates = ready_rows(contract_rows)
    limit = max(0, int(max_render_count))
    selected = candidates[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [render_preview_row(row, output_dir, index + 1) for index, row in enumerate(selected)]
    summary = build_summary(rows, render_ready_count=len(candidates), total_contract_rows=len(contract_rows))
    return {
        "manifest_version": MANIFEST_VERSION,
        "rows": rows,
        "summary": summary,
    }


def write_manifest_json(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# No-Reference Segment Blend Preview Summary",
        "",
        f"- render_ready_rows: {summary['render_ready_rows']}",
        f"- previews_rendered: {summary['previews_rendered']}",
        f"- previews_skipped: {summary['previews_skipped']}",
        f"- rendered_vocal_preview_path: {summary['rendered_vocal_preview_path']}",
        f"- rendered_minus_preview_path: {summary['rendered_minus_preview_path']}",
        f"- duration_preserved: {str(summary['duration_preserved']).lower()}",
        f"- sample_rate_preserved: {str(summary['sample_rate_preserved']).lower()}",
        f"- channels_preserved: {str(summary['channels_preserved']).lower()}",
        f"- can_publish_as_result_count: {summary['can_publish_as_result_count']}",
        f"- can_mark_release_ready_count: {summary['can_mark_release_ready_count']}",
        f"- requires_post_render_judge_count: {summary['requires_post_render_judge_count']}",
        "",
        "## Safety",
        "",
    ]
    for key, value in summary["safety"].items():
        lines.append(f"- {key}: {str(value).lower()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(result: dict[str, Any], *, manifest_json: Path, summary_md: Path) -> None:
    write_manifest_json(manifest_json, result)
    write_summary_md(summary_md, result["summary"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render one no-reference segment blend preview from render-ready dry-run rows.")
    parser.add_argument("--dry-run-jsonl", type=Path, required=True)
    parser.add_argument("--render-contract-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--max-render-count", type=int, default=1)
    args = parser.parse_args(argv)
    result = build_preview_result(
        list(iter_jsonl(args.dry_run_jsonl)),
        list(iter_jsonl(args.render_contract_jsonl)),
        args.output_dir,
        max_render_count=args.max_render_count,
    )
    write_outputs(result, manifest_json=args.manifest_json, summary_md=args.summary_md)
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
