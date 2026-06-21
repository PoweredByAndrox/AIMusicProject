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

from tools.render_no_reference_segment_blend_preview import (  # noqa: E402
    as_float,
    blend_audio,
    match_audio_shape,
    read_audio,
    resample_audio,
    write_audio,
)


SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "audio_rendered": True,
    "can_publish_as_result": False,
    "can_mark_release_ready": False,
    "release_ready": False,
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


def ready_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("render_status") == "render_ready"
        and row.get("can_publish_as_result") is False
        and row.get("can_mark_release_ready") is False
        and str(row.get("selected_target_stem") or row.get("source_stem") or "") in {"minus", "both"}
    ]
    return sorted(selected, key=lambda row: (as_float(row.get("window_start_sec")), as_float(row.get("window_duration_sec")), str(row.get("alternate_candidate_id") or "")))


def intervals_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_start = as_float(left.get("window_start_sec"))
    left_end = left_start + as_float(left.get("window_duration_sec"))
    right_start = as_float(right.get("window_start_sec"))
    right_end = right_start + as_float(right.get("window_duration_sec"))
    return left_start < right_end and right_start < left_end


def validate_nonoverlap(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        for other in rows[index + 1 :]:
            if intervals_overlap(row, other):
                raise ValueError(
                    "Render-ready multi-window blend rows overlap: "
                    f"{row.get('window_start_sec')} and {other.get('window_start_sec')}"
                )


def unique_required(rows: list[dict[str, Any]], key: str) -> str:
    values = {str(row.get(key) or "") for row in rows if row.get(key)}
    if len(values) != 1:
        raise ValueError(f"Expected exactly one {key}; got {sorted(values)}")
    return next(iter(values))


def apply_window(result, sample_rate: int, row: dict[str, Any]):
    alternate_path = str(row.get("alt_minus_path") or "")
    if not alternate_path:
        raise ValueError(f"Render-ready row has no alt_minus_path: {row}")
    alternate, alternate_rate = read_audio(alternate_path)
    if alternate_rate != sample_rate:
        alternate = resample_audio(alternate, alternate_rate, sample_rate)
    alternate = match_audio_shape(alternate, result.shape)
    crossfade_ms = int(row.get("crossfade_ms")) if row.get("crossfade_ms") is not None else 80
    return blend_audio(
        result,
        alternate,
        sample_rate,
        start_sec=as_float(row.get("window_start_sec")),
        duration_sec=as_float(row.get("window_duration_sec")),
        crossfade_ms=crossfade_ms,
        alternate_weight=as_float(row.get("alternate_weight"), default=1.0),
    )


def build_multi_window_preview(contract_rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    selected = ready_rows(contract_rows)
    if not selected:
        raise ValueError("No render-ready minus/both rows available for multi-window blend")
    validate_nonoverlap(selected)
    base_minus_path = unique_required(selected, "base_minus_path")
    base_vocal_path = unique_required(selected, "base_vocal_path")
    track_id = unique_required(selected, "track_id")
    base_candidate_id = unique_required(selected, "base_candidate_id")

    result, sample_rate = read_audio(base_minus_path)
    for row in selected:
        result = apply_window(result, sample_rate, row)

    audio_dir = output_dir / "audio"
    minus_path = audio_dir / "minus.flac"
    write_audio(minus_path, result, sample_rate)
    info = sf.info(str(minus_path))
    blocked_or_skipped = len(contract_rows) - len(selected)
    return {
        "manifest_version": 1,
        "render_status": "rendered_preview_only",
        "track_id": track_id,
        "base_candidate_id": base_candidate_id,
        "alternate_candidate_ids": sorted({str(row.get("alternate_candidate_id") or "") for row in selected}),
        "base_minus_path": base_minus_path,
        "base_vocal_path": base_vocal_path,
        "rendered_minus_path": str(minus_path),
        "applied_window_count": len(selected),
        "blocked_or_skipped_window_count": blocked_or_skipped,
        "applied_windows": [
            {
                "alternate_candidate_id": str(row.get("alternate_candidate_id") or ""),
                "window_start_sec": round(as_float(row.get("window_start_sec")), 6),
                "window_duration_sec": round(as_float(row.get("window_duration_sec")), 6),
                "crossfade_ms": int(row.get("crossfade_ms")) if row.get("crossfade_ms") is not None else 80,
                "source_stem": str(row.get("source_stem") or ""),
            }
            for row in selected
        ],
        "audio_info": {
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "frames": int(info.frames),
            "duration_sec": round(float(info.frames) / float(info.samplerate), 6) if info.samplerate else 0.0,
        },
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "release_ready": False,
        "safety": dict(SAFETY_FLAGS),
    }


def write_outputs(result: dict[str, Any], manifest_json: Path) -> None:
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render one full-length no-reference multi-window minus blend preview.")
    parser.add_argument("--render-contract-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_multi_window_preview(list(iter_jsonl(args.render_contract_jsonl)), args.output_dir)
    write_outputs(result, args.manifest_json)
    print(json.dumps({key: result[key] for key in ("render_status", "applied_window_count", "rendered_minus_path", "release_ready")}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
