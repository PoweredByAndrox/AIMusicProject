#!/usr/bin/env python3
"""Apply a trained Mac stem refiner to a MelBand pitch-guard failure."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.mac_melband_pitch_guard import (
    pitch_shift_resample_same_length,
    score_best_value,
    score_pitch_shifted_vocal_leak,
    score_region_count,
    score_target_issue_improved_from_values,
)
from tools.stem_region_repair_common import ISSUE_VOCAL_LEAK, make_region_mask, read_audio_stereo, write_audio, write_json
from tools.stem_region_repair_model import (
    apply_region_repair_model,
    load_region_repair_checkpoint,
    resolve_region_repair_training_device,
)


def _append_progress(path: Path, phase: str, status: str, **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_unix": time.time(), "phase": phase, "status": status, **extra}, sort_keys=True) + "\n")
        handle.flush()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(audio, dtype=np.float64)))))


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _resolve_audio_path(value: str, *, guard_dir: Path) -> Path:
    path = Path(str(value))
    try:
        source_exists = _path_exists(path)
    except OSError:
        source_exists = False
    if source_exists:
        return path
    candidate = guard_dir / "audio" / path.name
    try:
        candidate_exists = _path_exists(candidate)
    except OSError:
        candidate_exists = False
    if candidate_exists:
        return candidate
    raise FileNotFoundError(f"audio path not found: {value}")


def _fit_audio(*arrays: np.ndarray) -> list[np.ndarray]:
    frames = min(array.shape[0] for array in arrays)
    return [np.asarray(array[:frames], dtype=np.float32) for array in arrays]


def _build_region_mask_and_source(
    *,
    vocals_audio: np.ndarray,
    sample_rate: int,
    regions: list[Mapping[str, Any]],
    pad_sec: float,
    fade_sec: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    frame_count = int(vocals_audio.shape[0])
    region_mask = np.zeros(frame_count, dtype=np.float32)
    source_condition = np.zeros_like(vocals_audio, dtype=np.float32)
    applied_regions: list[dict[str, Any]] = []
    for region in regions:
        start_sec = max(0.0, float(region.get("start", 0.0)) - float(pad_sec))
        end_sec = min(float(frame_count) / float(sample_rate), float(region.get("end", 0.0)) + float(pad_sec))
        if end_sec <= start_sec:
            continue
        local_mask = make_region_mask(
            frame_count=frame_count,
            sample_rate=sample_rate,
            start_sec=start_sec,
            end_sec=end_sec,
            fade_sec=fade_sec,
        )
        semitones = float(region.get("semitones", 0.0))
        probe = pitch_shift_resample_same_length(vocals_audio, semitones)
        source_condition += probe * local_mask[:, None]
        region_mask = np.maximum(region_mask, local_mask)
        applied_regions.append(
            {
                "start": round(start_sec, 6),
                "end": round(end_sec, 6),
                "semitones": semitones,
                "source_score": region.get("score"),
                "fail_mode": region.get("fail_mode"),
            }
        )
    return region_mask.astype(np.float32), source_condition.astype(np.float32), applied_regions


def _target_issue_improved(
    *,
    before_region_count: int,
    after_region_count: int,
    before_best_score: float,
    after_best_score: float,
    after_verdict: str,
) -> bool:
    return score_target_issue_improved_from_values(
        before_region_count=before_region_count,
        after_region_count=after_region_count,
        before_best_score=before_best_score,
        after_best_score=after_best_score,
        after_verdict=after_verdict,
    )


def run_refiner_replay(
    *,
    mel_guard_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    require_mps: bool = True,
    max_regions: int = 8,
    region_pad_sec: float = 0.08,
    fade_sec: float = 0.03,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    _append_progress(progress_path, "preflight", "start", require_mps=bool(require_mps))
    device = resolve_region_repair_training_device("mps")
    if bool(require_mps) and device.type != "mps":
        manifest = {
            "schema": "mac_melband_refiner_replay.v1",
            "status": "blocked_mps_unavailable",
            "delivery_allowed": False,
            "presentation_status": "diagnostic_refiner_replay_only",
            "resolved_device": str(device),
        }
        write_json(output_dir / "manifest.json", manifest)
        _append_progress(progress_path, "preflight", "blocked_mps_unavailable", resolved_device=str(device))
        return manifest

    guard_manifest = _load_json(Path(mel_guard_dir) / "manifest.json")
    audio_map = dict(guard_manifest.get("audio") or {})
    vocals_path = _resolve_audio_path(str(audio_map.get("vocals")), guard_dir=Path(mel_guard_dir))
    instrumental_path = _resolve_audio_path(str(audio_map.get("instrumental")), guard_dir=Path(mel_guard_dir))
    raw_path = _resolve_audio_path(str(audio_map.get("raw_ref")), guard_dir=Path(mel_guard_dir))
    _append_progress(progress_path, "preflight", "end", resolved_device=str(device), checkpoint=str(checkpoint_path))

    before_score = score_pitch_shifted_vocal_leak(
        vocals_path=vocals_path,
        instrumental_path=instrumental_path,
        output_dir=output_dir / "score" / "before",
        config=(guard_manifest.get("score") or {}).get("config"),
    )
    regions = list(before_score.get("regions") or [])[: int(max_regions)]
    if not regions:
        manifest = {
            "schema": "mac_melband_refiner_replay.v1",
            "status": "skipped_no_guard_regions",
            "delivery_allowed": False,
            "presentation_status": "diagnostic_refiner_replay_only",
            "before_score": before_score,
        }
        write_json(output_dir / "manifest.json", manifest)
        _append_progress(progress_path, "score_before", "skipped_no_guard_regions")
        return manifest

    vocals_audio, vocals_rate = read_audio_stereo(vocals_path)
    instrumental_audio, instrumental_rate = read_audio_stereo(instrumental_path)
    raw_audio, raw_rate = read_audio_stereo(raw_path)
    if len({vocals_rate, instrumental_rate, raw_rate}) != 1:
        raise ValueError("mac_refiner_replay_sample_rate_mismatch")
    vocals_audio, instrumental_audio, raw_audio = _fit_audio(vocals_audio, instrumental_audio, raw_audio)
    region_mask, source_condition, applied_regions = _build_region_mask_and_source(
        vocals_audio=vocals_audio,
        sample_rate=vocals_rate,
        regions=regions,
        pad_sec=region_pad_sec,
        fade_sec=fade_sec,
    )
    if not np.any(region_mask > 1.0e-4):
        raise ValueError("mac_refiner_replay_empty_region_mask")

    _append_progress(progress_path, "apply_refiner", "start", region_count=len(applied_regions), device=str(device))
    model, checkpoint_payload = load_region_repair_checkpoint(checkpoint_path)
    repaired_instrumental, apply_diag = apply_region_repair_model(
        model=model,
        mix_audio=raw_audio,
        bad_stem_audio=instrumental_audio,
        source_condition_audio=source_condition,
        region_mask=region_mask,
        sample_rate=vocals_rate,
        issue_type=ISSUE_VOCAL_LEAK,
        device=str(device),
    )
    delta = repaired_instrumental - instrumental_audio
    repaired_vocals = np.clip(vocals_audio - delta, -1.0, 1.0).astype(np.float32)
    repaired_instrumental = np.clip(repaired_instrumental, -1.0, 1.0).astype(np.float32)
    audio_dir = output_dir / "audio"
    vocals_out = write_audio(audio_dir / "v.flac", repaired_vocals, vocals_rate)
    instrumental_out = write_audio(audio_dir / "i.flac", repaired_instrumental, vocals_rate)
    raw_out = write_audio(audio_dir / "raw.flac", raw_audio, vocals_rate)
    _append_progress(progress_path, "apply_refiner", "end", changed_region_ratio=apply_diag.get("changed_region_ratio"))

    after_score = score_pitch_shifted_vocal_leak(
        vocals_path=vocals_out,
        instrumental_path=instrumental_out,
        output_dir=output_dir / "score" / "after",
        config=before_score.get("config"),
    )
    reconstruction_rms = _rms(repaired_vocals + repaired_instrumental - raw_audio)
    original_reconstruction_rms = _rms(vocals_audio + instrumental_audio - raw_audio)
    before_region_count = score_region_count(before_score)
    after_region_count = score_region_count(after_score)
    before_selected_region_count = len(before_score.get("regions") or [])
    after_selected_region_count = len(after_score.get("regions") or [])
    before_best = score_best_value(before_score)
    after_best = score_best_value(after_score)
    target_issue_improved = _target_issue_improved(
        before_region_count=before_region_count,
        after_region_count=after_region_count,
        before_best_score=before_best,
        after_best_score=after_best,
        after_verdict=str(after_score.get("verdict")),
    )
    status = "repaired_passed_pitch_guard" if after_score.get("verdict") == "pass" else "repaired_but_still_blocked"
    training_history = dict(checkpoint_payload.get("training_history") or {})
    manifest = {
        "schema": "mac_melband_refiner_replay.v1",
        "status": status,
        "source_guard_dir": str(mel_guard_dir),
        "checkpoint": str(checkpoint_path),
        "resolved_device": str(device),
        "checkpoint_training_device": training_history.get("resolved_device"),
        "audio": {
            "vocals": str(vocals_out),
            "instrumental": str(instrumental_out),
            "raw_ref": str(raw_out),
        },
        "applied_regions": applied_regions,
        "before_score": before_score,
        "after_score": after_score,
        "metrics": {
            "before_region_count": before_region_count,
            "after_region_count": after_region_count,
            "before_selected_region_count": before_selected_region_count,
            "after_selected_region_count": after_selected_region_count,
            "before_best_score": before_best,
            "after_best_score": after_best,
            "target_issue_improved": bool(target_issue_improved),
            "original_reconstruction_rms": original_reconstruction_rms,
            "reconstruction_rms": reconstruction_rms,
            "instrumental_delta_rms": _rms(delta),
            "apply_diagnostics": apply_diag,
        },
        "contract": {
            "diagnostic_only": True,
            "no_listen_delivery": True,
            "delivery_allowed": False,
            "usable_for_promotion": False,
            "mixture_consistent_delta_transfer": True,
        },
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "diagnostic_refiner_replay_only",
    }
    write_json(output_dir / "manifest.json", manifest)
    _append_progress(
        progress_path,
        "manifest",
        "end",
        run_status=status,
        before_region_count=before_region_count,
        after_region_count=after_region_count,
        target_issue_improved=target_issue_improved,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mel-guard-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-regions", type=int, default=8)
    parser.add_argument("--require-mps", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_refiner_replay(
        mel_guard_dir=args.mel_guard_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.out,
        require_mps=bool(args.require_mps),
        max_regions=int(args.max_regions),
    )
    print(json.dumps({"manifest": str(args.out / "manifest.json"), "status": manifest["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
