#!/usr/bin/env python3
"""Audit whether a candidate differs enough from its source to justify listen handoff.

This is a metadata/DSP gate only. It does not repair audio, relax objective
thresholds, or publish listen output.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.stem_region_repair_common import read_audio_stereo


def _append_progress(path: Path, phase: str, status: str, **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_unix": time.time(), "phase": phase, "status": status, **extra}, sort_keys=True) + "\n")
        handle.flush()


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(np.asarray(audio, dtype=np.float64)))))


def _ratio_db(numerator: float, denominator: float) -> float:
    return float(20.0 * math.log10((float(numerator) + 1.0e-12) / (float(denominator) + 1.0e-12)))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.sqrt(np.sum(np.square(left, dtype=np.float64))) * np.sqrt(np.sum(np.square(right, dtype=np.float64))))
    if denom <= 1.0e-12:
        return 0.0
    return float(np.sum(np.asarray(left, dtype=np.float64) * np.asarray(right, dtype=np.float64)) / denom)


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return str(left) == str(right)


def _base_manifest(
    *,
    source_path: Path,
    candidate_path: Path,
    output_dir: Path,
    source_manifest_path: Path | None,
    candidate_manifest_path: Path | None,
) -> dict[str, Any]:
    return {
        "schema": "mac_candidate_audibility_delta.v1",
        "source_audio": str(source_path),
        "candidate_audio": str(candidate_path),
        "source_manifest": str(source_manifest_path or ""),
        "candidate_manifest": str(candidate_manifest_path or ""),
        "output_dir": str(output_dir),
        "manifest_path": str(output_dir / "manifest.json"),
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "audibility_delta_diagnostic_only",
        "contract": {
            "diagnostic_only": True,
            "metadata_gate_only": True,
            "no_listen_delivery": True,
            "no_audio_written": True,
        },
    }


def _window_metrics(
    *,
    source: np.ndarray,
    diff: np.ndarray,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
    min_active_window_ratio: float,
    min_abs_diff_rms: float,
    top_k: int,
) -> tuple[list[dict[str, Any]], float, int, float]:
    frame_count = int(source.shape[0])
    window_frames = max(1, int(round(float(window_sec) * sample_rate)))
    hop_frames = max(1, int(round(float(hop_sec) * sample_rate)))
    rows: list[dict[str, Any]] = []
    active_starts: set[int] = set()
    for start in range(0, max(1, frame_count - window_frames + 1), hop_frames):
        end = min(frame_count, start + window_frames)
        source_rms = _rms(source[start:end])
        diff_rms = _rms(diff[start:end])
        ratio = float(diff_rms / (source_rms + 1.0e-12))
        row = {
            "start": round(float(start) / sample_rate, 6),
            "end": round(float(end) / sample_rate, 6),
            "source_rms": round(source_rms, 10),
            "diff_rms": round(diff_rms, 10),
            "diff_to_source_ratio": round(ratio, 10),
            "diff_rel_db_vs_source": round(_ratio_db(diff_rms, source_rms), 6),
        }
        rows.append(row)
        if diff_rms >= min_abs_diff_rms and ratio >= min_active_window_ratio:
            active_starts.add(start)
    rows.sort(key=lambda item: (float(item["diff_to_source_ratio"]), float(item["diff_rms"])), reverse=True)
    active_changed_sec = round(float(len(active_starts) * hop_frames) / sample_rate, 6)
    max_ratio = float(rows[0]["diff_to_source_ratio"]) if rows else 0.0
    return rows[: max(1, int(top_k))], active_changed_sec, len(active_starts), max_ratio


def run_audibility_audit(
    *,
    source_audio_path: Path,
    candidate_audio_path: Path,
    output_dir: Path,
    source_manifest_path: Path | None = None,
    candidate_manifest_path: Path | None = None,
    window_sec: float = 0.5,
    hop_sec: float = 0.25,
    min_fulltrack_rel_db: float = -45.0,
    min_active_window_ratio: float = 0.03,
    min_active_changed_sec: float = 1.0,
    min_abs_diff_rms: float = 1.0e-4,
    top_k: int = 12,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    source_path = Path(source_audio_path)
    candidate_path = Path(candidate_audio_path)
    _append_progress(
        progress_path,
        "preflight",
        "start",
        acceleration_mode="cpu_dsp_scoring",
        gpu_inference_skipped=True,
        gpu_inference_skipped_reason="audibility_delta_audit_uses_existing_audio",
        source_audio=str(source_path),
        candidate_audio=str(candidate_path),
    )

    source_audio, source_rate = read_audio_stereo(source_path)
    candidate_audio, candidate_rate = read_audio_stereo(candidate_path)
    if int(source_rate) != int(candidate_rate):
        manifest = {
            **_base_manifest(
                source_path=source_path,
                candidate_path=candidate_path,
                output_dir=output_dir,
                source_manifest_path=source_manifest_path,
                candidate_manifest_path=candidate_manifest_path,
            ),
            "status": "audibility_delta_blocked",
            "verdict": "fail",
            "blocker_reasons": ["audibility_sample_rate_mismatch"],
            "metrics": {
                "source_sample_rate": int(source_rate),
                "candidate_sample_rate": int(candidate_rate),
            },
            "thresholds": {
                "min_fulltrack_rel_db": float(min_fulltrack_rel_db),
                "min_active_window_ratio": float(min_active_window_ratio),
                "min_active_changed_sec": float(min_active_changed_sec),
                "min_abs_diff_rms": float(min_abs_diff_rms),
                "window_sec": float(window_sec),
                "hop_sec": float(hop_sec),
            },
            "top_windows": [],
        }
        _write_json(output_dir / "manifest.json", manifest)
        _append_progress(progress_path, "manifest", "end", verdict="fail", blocker_reasons=manifest["blocker_reasons"])
        return manifest
    frame_count = min(int(source_audio.shape[0]), int(candidate_audio.shape[0]))
    source_audio = np.asarray(source_audio[:frame_count], dtype=np.float32)
    candidate_audio = np.asarray(candidate_audio[:frame_count], dtype=np.float32)
    diff = candidate_audio - source_audio

    source_rms = _rms(source_audio)
    candidate_rms = _rms(candidate_audio)
    diff_rms = _rms(diff)
    diff_rel_db = _ratio_db(diff_rms, source_rms)
    top_windows, active_changed_sec, active_window_count, max_window_ratio = _window_metrics(
        source=source_audio,
        diff=diff,
        sample_rate=int(source_rate),
        window_sec=window_sec,
        hop_sec=hop_sec,
        min_active_window_ratio=min_active_window_ratio,
        min_abs_diff_rms=min_abs_diff_rms,
        top_k=top_k,
    )
    fulltrack_pass = diff_rms >= min_abs_diff_rms and diff_rel_db >= float(min_fulltrack_rel_db)
    localized_pass = active_changed_sec >= float(min_active_changed_sec) and max_window_ratio >= float(min_active_window_ratio)
    same_path = _same_path(source_path, candidate_path)
    passed = bool((fulltrack_pass or localized_pass) and not same_path and frame_count > 0)
    blocker_reasons: list[str] = []
    if same_path:
        blocker_reasons.append("source_candidate_same_path")
    if frame_count <= 0:
        blocker_reasons.append("audibility_audio_empty")
    if not passed:
        blocker_reasons.append("audibility_delta_below_gate")

    manifest = {
        **_base_manifest(
            source_path=source_path,
            candidate_path=candidate_path,
            output_dir=output_dir,
            source_manifest_path=source_manifest_path,
            candidate_manifest_path=candidate_manifest_path,
        ),
        "status": "audibility_delta_passed" if passed else "audibility_delta_blocked",
        "verdict": "pass" if passed else "fail",
        "thresholds": {
            "min_fulltrack_rel_db": float(min_fulltrack_rel_db),
            "min_active_window_ratio": float(min_active_window_ratio),
            "min_active_changed_sec": float(min_active_changed_sec),
            "min_abs_diff_rms": float(min_abs_diff_rms),
            "window_sec": float(window_sec),
            "hop_sec": float(hop_sec),
        },
        "metrics": {
            "sample_rate": int(source_rate),
            "frames_compared": frame_count,
            "duration_sec": round(float(frame_count) / int(source_rate), 6) if source_rate else 0.0,
            "source_rms": round(source_rms, 10),
            "candidate_rms": round(candidate_rms, 10),
            "diff_rms": round(diff_rms, 10),
            "diff_rel_db_vs_source": round(diff_rel_db, 6),
            "correlation": round(_correlation(source_audio, candidate_audio), 10),
            "active_window_count": active_window_count,
            "active_changed_sec": active_changed_sec,
            "max_window_diff_ratio": round(max_window_ratio, 10),
            "fulltrack_delta_pass": bool(fulltrack_pass),
            "localized_delta_pass": bool(localized_pass),
        },
        "top_windows": top_windows,
        "blocker_reasons": blocker_reasons,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _append_progress(
        progress_path,
        "manifest",
        "end",
        verdict=manifest["verdict"],
        diff_rel_db_vs_source=manifest["metrics"]["diff_rel_db_vs_source"],
        active_changed_sec=active_changed_sec,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audio", type=Path, required=True)
    parser.add_argument("--candidate-audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--window-sec", type=float, default=0.5)
    parser.add_argument("--hop-sec", type=float, default=0.25)
    parser.add_argument("--min-fulltrack-rel-db", type=float, default=-45.0)
    parser.add_argument("--min-active-window-ratio", type=float, default=0.03)
    parser.add_argument("--min-active-changed-sec", type=float, default=1.0)
    parser.add_argument("--min-abs-diff-rms", type=float, default=1.0e-4)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_audibility_audit(
        source_audio_path=args.source_audio,
        candidate_audio_path=args.candidate_audio,
        output_dir=args.out,
        source_manifest_path=args.source_manifest,
        candidate_manifest_path=args.candidate_manifest,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        min_fulltrack_rel_db=args.min_fulltrack_rel_db,
        min_active_window_ratio=args.min_active_window_ratio,
        min_active_changed_sec=args.min_active_changed_sec,
        min_abs_diff_rms=args.min_abs_diff_rms,
        top_k=args.top_k,
    )
    print(json.dumps({"manifest": str(args.out / "manifest.json"), "status": manifest["status"], "verdict": manifest["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
