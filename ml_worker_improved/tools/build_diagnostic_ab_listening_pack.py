from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf


SCHEMA_VERSION = 1
MANIFEST_NAME = "diagnostic_ab_listening_manifest.json"
PROGRESS_NAME = "diagnostic_ab_listening_progress.jsonl"


class ProgressWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started_at = time.monotonic()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def mark(self, stage: str, *, status: str = "progress", **metadata: Any) -> None:
        event = {
            "type": "diagnostic_ab_listening_pack_stage",
            "stage": stage,
            "status": status,
            "elapsed_sec": 0.0,
            "since_start_sec": round(time.monotonic() - self.started_at, 6),
            "metadata": metadata,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return loaded


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        loaded = json.loads(line)
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        records.append(loaded)
    return records


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _project_path(value: str, *, base_dir: Path) -> Path:
    normalized = value.replace("\\", "/")
    match = re.match(r"^[A-Za-z]:/AIMusicProject/(.+)$", normalized)
    if match:
        return Path("/app") / match.group(1)
    path = Path(normalized)
    if path.is_absolute():
        return path
    return base_dir / path


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    safe = safe.strip("._-")
    return safe or "audio"


def _window_seconds(record: dict[str, Any]) -> tuple[float, float]:
    raw = record.get("window_sec")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
        raise ValueError(f"Segment {record.get('segment_id', '<unknown>')} has no two-value window_sec")
    start_sec = float(raw[0])
    end_sec = float(raw[1])
    if start_sec < 0.0 or end_sec < 0.0:
        raise ValueError(f"Segment {record.get('segment_id', '<unknown>')} has negative window_sec={raw}")
    if end_sec != 0.0 and end_sec <= start_sec:
        raise ValueError(f"Segment {record.get('segment_id', '<unknown>')} has invalid window_sec={raw}")
    return start_sec, end_sec


def _time_token(value: float) -> str:
    return f"{value:.3f}".replace("-", "m").replace(".", "p")


def _read_window(path: Path, start_sec: float, end_sec: float) -> tuple[np.ndarray, int, int, int]:
    info = sf.info(str(path))
    start_frame = int(round(start_sec * info.samplerate))
    if end_sec == 0.0:
        frame_count = info.frames - start_frame
    else:
        end_frame = int(round(end_sec * info.samplerate))
        frame_count = end_frame - start_frame
    if frame_count <= 0:
        raise ValueError(f"Invalid or empty window for {path}: start_sec={start_sec}, end_sec={end_sec}")
    audio, sample_rate = sf.read(
        str(path),
        start=start_frame,
        frames=frame_count,
        always_2d=True,
        dtype="float64",
    )
    if sample_rate != info.samplerate:
        raise ValueError(f"Unexpected sample rate while reading {path}: {sample_rate} != {info.samplerate}")
    if len(audio) != frame_count:
        raise ValueError(f"Short read for {path}: expected {frame_count} frames, got {len(audio)}")
    return audio, sample_rate, start_frame, frame_count


def _write_flac(path: Path, audio: np.ndarray, sample_rate: int, subtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, format="FLAC", subtype=subtype)


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Audio shape mismatch: {a.shape} != {b.shape}")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def _read_written(path: Path) -> np.ndarray:
    audio, _ = sf.read(str(path), always_2d=True, dtype="float64")
    return audio


def _load_comparison_segments(comparison_manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Path | None]:
    manifest = load_json(comparison_manifest_path)
    base_dir = comparison_manifest_path.parent
    jsonl_value = manifest.get("segment_comparison_jsonl")
    if isinstance(jsonl_value, str) and jsonl_value:
        jsonl_path = _project_path(jsonl_value, base_dir=base_dir)
        return manifest, load_jsonl(jsonl_path), jsonl_path
    for key in ("segments", "segment_comparisons"):
        raw_segments = manifest.get(key)
        if isinstance(raw_segments, list):
            segments = []
            for index, item in enumerate(raw_segments, start=1):
                if not isinstance(item, dict):
                    raise ValueError(f"Expected JSON object in {key}[{index}]")
                segments.append(item)
            return manifest, segments, None
    raise ValueError("Comparison manifest has neither segment_comparison_jsonl nor segments")


def _segment_sort_key(indexed_record: tuple[int, dict[str, Any]]) -> tuple[int, float, int]:
    index, record = indexed_record
    winner = str(record.get("winner", "tie"))
    winner_rank = {"candidate": 0, "tie": 1, "baseline": 2}.get(winner, 1)
    score_delta = float(record.get("score_delta", 0.0))
    if winner == "candidate":
        score_order = -score_delta
    elif winner == "baseline":
        score_order = score_delta
    else:
        score_order = abs(score_delta)
    return winner_rank, score_order, index


def _combined_subtype(baseline_subtype: str, candidate_subtype: str) -> str:
    if baseline_subtype == candidate_subtype:
        return baseline_subtype
    return "PCM_24"


def build_ab_listening_pack(
    *,
    comparison_manifest_path: Path | str,
    baseline_audio_path: Path | str,
    candidate_audio_path: Path | str,
    output_dir: Path | str,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    silence_sec: float = 0.5,
) -> dict[str, Any]:
    comparison_path = Path(comparison_manifest_path)
    baseline_path = Path(baseline_audio_path)
    candidate_path = Path(candidate_audio_path)
    output_path = Path(output_dir)
    clips_dir = output_path / "clips"
    progress_path = output_path / PROGRESS_NAME
    progress = ProgressWriter(progress_path)

    if silence_sec < 0.0:
        raise ValueError("silence_sec must be non-negative")

    progress.mark(
        "start",
        comparison_manifest_path=str(comparison_path),
        baseline_audio_path=str(baseline_path),
        candidate_audio_path=str(candidate_path),
        output_dir=str(output_path),
        gpu_inference_skipped=True,
        acceleration_mode="cpu_dsp_packaging",
    )
    comparison_manifest, segments, segment_jsonl_path = _load_comparison_segments(comparison_path)
    if not segments:
        raise ValueError(f"No segments in {comparison_path}")

    baseline_info = sf.info(str(baseline_path))
    candidate_info = sf.info(str(candidate_path))
    if baseline_info.samplerate != candidate_info.samplerate:
        raise ValueError(
            "Sample rate mismatch: "
            f"baseline={baseline_info.samplerate}, candidate={candidate_info.samplerate}"
        )
    if baseline_info.channels != candidate_info.channels:
        raise ValueError(f"Channel mismatch: baseline={baseline_info.channels}, candidate={candidate_info.channels}")

    baseline_safe = _safe_name(baseline_label)
    candidate_safe = _safe_name(candidate_label)
    single_baseline_subtype = baseline_info.subtype
    single_candidate_subtype = candidate_info.subtype
    combined_subtype = _combined_subtype(single_baseline_subtype, single_candidate_subtype)
    max_a_diff = 0.0
    max_b_diff = 0.0
    clip_records: list[dict[str, Any]] = []

    ranked_segments = sorted(enumerate(segments, start=1), key=_segment_sort_key)
    progress.mark(
        "load_segments",
        status="completed",
        segment_count=len(ranked_segments),
        segment_comparison_jsonl=str(segment_jsonl_path) if segment_jsonl_path is not None else None,
    )

    for rank, (source_index, segment) in enumerate(ranked_segments, start=1):
        start_sec, end_sec = _window_seconds(segment)
        segment_id = _safe_name(str(segment.get("segment_id") or f"segment_{source_index:04d}"))
        clip_dir = clips_dir / (
            f"{rank:04d}_{_safe_name(str(segment.get('winner', 'tie')))}_"
            f"{segment_id}_{_time_token(start_sec)}-{_time_token(end_sec)}"
        )
        progress.mark(
            "write_clip",
            rank=rank,
            source_index=source_index,
            segment_id=segment_id,
            window_sec=[start_sec, end_sec],
            winner=segment.get("winner"),
        )

        baseline_audio, baseline_sr, start_frame, frame_count = _read_window(baseline_path, start_sec, end_sec)
        candidate_audio, candidate_sr, candidate_start_frame, candidate_frame_count = _read_window(
            candidate_path,
            start_sec,
            end_sec,
        )
        if baseline_sr != candidate_sr:
            raise ValueError(f"Sample rate mismatch in segment {segment_id}: {baseline_sr} != {candidate_sr}")
        if baseline_audio.shape != candidate_audio.shape:
            raise ValueError(f"Window shape mismatch in segment {segment_id}: {baseline_audio.shape} != {candidate_audio.shape}")
        if start_frame != candidate_start_frame or frame_count != candidate_frame_count:
            raise ValueError(f"Window frame mismatch in segment {segment_id}")

        a_path = clip_dir / f"A_{baseline_safe}.flac"
        b_path = clip_dir / f"B_{candidate_safe}.flac"
        ab_path = clip_dir / f"AB_{baseline_safe}_then_{candidate_safe}.flac"
        ba_path = clip_dir / f"BA_{candidate_safe}_then_{baseline_safe}.flac"
        silence = np.zeros((int(round(silence_sec * baseline_sr)), baseline_audio.shape[1]), dtype=np.float64)

        _write_flac(a_path, baseline_audio, baseline_sr, single_baseline_subtype)
        _write_flac(b_path, candidate_audio, candidate_sr, single_candidate_subtype)
        _write_flac(ab_path, np.vstack([baseline_audio, silence, candidate_audio]), baseline_sr, combined_subtype)
        _write_flac(ba_path, np.vstack([candidate_audio, silence, baseline_audio]), baseline_sr, combined_subtype)

        a_roundtrip_diff = _max_abs_diff(_read_written(a_path), baseline_audio)
        b_roundtrip_diff = _max_abs_diff(_read_written(b_path), candidate_audio)
        max_a_diff = max(max_a_diff, a_roundtrip_diff)
        max_b_diff = max(max_b_diff, b_roundtrip_diff)

        clip_records.append(
            {
                "rank": rank,
                "source_index": source_index,
                "segment_id": segment_id,
                "window_sec": [start_sec, end_sec],
                "start_frame": start_frame,
                "frame_count": frame_count,
                "winner": segment.get("winner", "tie"),
                "score_delta": segment.get("score_delta"),
                "defect_labels": segment.get("defect_labels", []),
                "paths": {
                    "A_baseline": str(a_path),
                    "B_candidate": str(b_path),
                    "AB_baseline_then_candidate": str(ab_path),
                    "BA_candidate_then_baseline": str(ba_path),
                },
                "subtypes": {
                    "A_baseline": sf.info(str(a_path)).subtype,
                    "B_candidate": sf.info(str(b_path)).subtype,
                    "AB_baseline_then_candidate": sf.info(str(ab_path)).subtype,
                    "BA_candidate_then_baseline": sf.info(str(ba_path)).subtype,
                },
                "roundtrip_verification": {
                    "A_max_abs_diff": a_roundtrip_diff,
                    "B_max_abs_diff": b_roundtrip_diff,
                    "source_window_decoded_before_write": True,
                    "single_clip_sample_transform_applied": False,
                },
            }
        )
        progress.mark(
            "roundtrip_verify",
            status="completed",
            rank=rank,
            segment_id=segment_id,
            A_max_abs_diff=a_roundtrip_diff,
            B_max_abs_diff=b_roundtrip_diff,
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "mode": "diagnostic_ab_listening_gate_pack_no_gain",
        "comparison_manifest_path": str(comparison_path),
        "segment_comparison_jsonl": str(segment_jsonl_path) if segment_jsonl_path is not None else None,
        "comparison_overall": comparison_manifest.get("overall", {}),
        "baseline": {
            "label": baseline_label,
            "audio_path": str(baseline_path),
            "sample_rate": baseline_info.samplerate,
            "channels": baseline_info.channels,
            "frames": baseline_info.frames,
            "format": baseline_info.format,
            "subtype": baseline_info.subtype,
        },
        "candidate": {
            "label": candidate_label,
            "audio_path": str(candidate_path),
            "sample_rate": candidate_info.samplerate,
            "channels": candidate_info.channels,
            "frames": candidate_info.frames,
            "format": candidate_info.format,
            "subtype": candidate_info.subtype,
        },
        "audio_write_contract": {
            "gain_applied": False,
            "normalization_applied": False,
            "limiting_applied": False,
            "fade_applied": False,
            "silence_inserted_between_combined_clips_sec": silence_sec,
            "baseline_source_subtype": baseline_info.subtype,
            "candidate_source_subtype": candidate_info.subtype,
            "single_clip_subtype_policy": "preserve_source_subtype",
            "combined_clip_subtype_policy": (
                "preserve_matching_source_subtype_else_pcm24"
                if baseline_info.subtype != candidate_info.subtype
                else "preserve_matching_source_subtype"
            ),
            "combined_clip_subtype": combined_subtype,
            "max_A_roundtrip_abs_diff": max_a_diff,
            "max_B_roundtrip_abs_diff": max_b_diff,
        },
        "delivery_contract": {
            "delivery_allowed": False,
            "listen_pack_allowed": False,
            "diagnostic_human_review_allowed": True,
            "presentation_status": "diagnostic_ab_listening_not_for_delivery",
            "release_status": "diagnostic_human_review_pack_only",
            "requires_human_ab_verdict": True,
            "rejected_candidates_must_not_be_presented_as_results": True,
        },
        "progress_jsonl": str(progress_path),
        "clip_count": len(clip_records),
        "clips": clip_records,
    }
    write_json(output_path / MANIFEST_NAME, manifest)
    progress.mark(
        "write_manifest",
        status="completed",
        manifest_path=str(output_path / MANIFEST_NAME),
        clip_count=len(clip_records),
        max_A_roundtrip_abs_diff=max_a_diff,
        max_B_roundtrip_abs_diff=max_b_diff,
    )
    progress.mark("end", status="completed")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a no-gain diagnostic A/B listening pack from segment comparison windows.",
    )
    parser.add_argument("comparison_manifest", type=Path)
    parser.add_argument("baseline_audio", type=Path)
    parser.add_argument("candidate_audio", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--silence-sec", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_ab_listening_pack(
        comparison_manifest_path=args.comparison_manifest,
        baseline_audio_path=args.baseline_audio,
        candidate_audio_path=args.candidate_audio,
        output_dir=args.output_dir,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        silence_sec=args.silence_sec,
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "clip_count": manifest["clip_count"],
                "max_A_roundtrip_abs_diff": manifest["audio_write_contract"]["max_A_roundtrip_abs_diff"],
                "max_B_roundtrip_abs_diff": manifest["audio_write_contract"]["max_B_roundtrip_abs_diff"],
                "delivery_allowed": manifest["delivery_contract"]["delivery_allowed"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
