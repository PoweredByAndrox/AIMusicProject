from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf
from scipy import signal


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


MANIFEST_NAME = "vocal_silence_hallucination_repair_manifest.json"
SUMMARY_NAME = "vocal_silence_hallucination_repair_summary.md"
RENDER_STATUS = "rendered_vocal_source_quiet_silence_repair_candidate"
START_TIME = time.monotonic()
EPS = 1.0e-9

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
    "release_ready": False,
}


def log_event(phase: str, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "phase": phase,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "gpu_inference_skipped": True,
        "gpu_inference_skipped_reason": "deterministic_cpu_dsp_source_quiet_vocal_repair",
    }
    payload.update(fields)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    if audio.size == 0:
        raise ValueError(f"Empty audio file: {path}")
    return audio.astype(np.float32), int(sample_rate)


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    gcd = math.gcd(source_rate, target_rate)
    up = target_rate // gcd
    down = source_rate // gcd
    return signal.resample_poly(audio, up, down, axis=0).astype(np.float32)


def fit_length(audio: np.ndarray, frame_count: int) -> np.ndarray:
    if audio.shape[0] == frame_count:
        return audio
    if audio.shape[0] > frame_count:
        return audio[:frame_count]
    padding = np.zeros((frame_count - audio.shape[0], audio.shape[1]), dtype=audio.dtype)
    return np.concatenate([audio, padding], axis=0)


def mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float64)
    return audio.mean(axis=1).astype(np.float64)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio))) + EPS)


def frame_rms(audio: np.ndarray, window_frames: int) -> np.ndarray:
    values: list[float] = []
    for start in range(0, audio.size, window_frames):
        segment = audio[start : min(start + window_frames, audio.size)]
        if segment.size:
            values.append(rms(segment))
    return np.asarray(values or [0.0], dtype=np.float64)


def contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask.astype(bool).tolist()):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, int(mask.size)))
    return runs


def source_quiet_vocal_windows(
    *,
    mix_audio: np.ndarray,
    vocal_audio: np.ndarray,
    sample_rate: int,
    window_sec: float,
    min_run_sec: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    frame_count = min(mix_audio.shape[0], vocal_audio.shape[0])
    mix_mono = mono(fit_length(mix_audio, frame_count))
    vocal_mono = mono(fit_length(vocal_audio, frame_count))
    window_frames = max(1, int(round(float(sample_rate) * float(window_sec))))
    mix_windows = frame_rms(mix_mono, window_frames)
    vocal_windows = frame_rms(vocal_mono, window_frames)
    mix_active_threshold = max(float(np.percentile(mix_windows, 30)) * 0.6, rms(mix_mono) * 0.05, EPS)
    vocal_active_threshold = max(float(np.percentile(vocal_windows, 55)) * 0.7, rms(vocal_mono) * 0.08, EPS)
    quiet_mix = mix_windows <= mix_active_threshold
    active_vocal = vocal_windows > vocal_active_threshold
    raw_mask = quiet_mix & active_vocal
    min_run_windows = max(1, int(math.ceil(float(min_run_sec) / float(window_sec))))
    decisions: list[dict[str, Any]] = []
    for start_window, end_window in contiguous_true_runs(raw_mask):
        if end_window - start_window < min_run_windows:
            continue
        start_frame = start_window * window_frames
        end_frame = min(frame_count, end_window * window_frames)
        if end_frame <= start_frame:
            continue
        decisions.append(
            {
                "start_sec": round(float(start_frame) / float(sample_rate), 6),
                "end_sec": round(float(end_frame) / float(sample_rate), 6),
                "start_window": int(start_window),
                "end_window": int(end_window),
                "window_count": int(end_window - start_window),
                "cleanup_kind": "source_quiet_vocal_silence_hallucination",
                "mix_rms_mean": round(float(np.mean(mix_windows[start_window:end_window])), 9),
                "vocal_rms_mean": round(float(np.mean(vocal_windows[start_window:end_window])), 9),
            }
        )
    metrics = {
        "mix_active_threshold": round(float(mix_active_threshold), 9),
        "vocal_active_threshold": round(float(vocal_active_threshold), 9),
        "raw_source_quiet_vocal_window_count": float(np.count_nonzero(raw_mask)),
        "selected_source_quiet_vocal_window_count": float(sum(decision["window_count"] for decision in decisions)),
        "window_sec": round(float(window_sec), 6),
        "min_run_sec": round(float(min_run_sec), 6),
    }
    return decisions, metrics


def apply_window_attenuation(
    audio: np.ndarray,
    *,
    decisions: list[dict[str, Any]],
    sample_rate: int,
    attenuation_gain: float,
    fade_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    gain = np.ones(audio.shape[0], dtype=np.float32)
    fade_frames = max(1, int(round(float(sample_rate) * float(fade_sec))))
    target_gain = float(max(0.0, min(1.0, attenuation_gain)))
    for decision in decisions:
        start = max(0, int(round(float(decision["start_sec"]) * sample_rate)))
        end = min(audio.shape[0], int(round(float(decision["end_sec"]) * sample_rate)))
        if end <= start:
            continue
        length = end - start
        local = np.full(length, target_gain, dtype=np.float32)
        local_fade = min(fade_frames, max(1, length // 2))
        if local_fade > 1:
            local[:local_fade] = np.linspace(1.0, target_gain, local_fade, dtype=np.float32)
            local[-local_fade:] = np.linspace(target_gain, 1.0, local_fade, dtype=np.float32)
        gain[start:end] = np.minimum(gain[start:end], local)
    return (audio * gain[:, None]).astype(np.float32), gain


def safe_candidate_id(base_candidate_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(base_candidate_id or "vocal_candidate")).strip("_")
    return f"source_quiet_vocal_silence_repair__{normalized or 'vocal_candidate'}"


def source_ref(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.exists() else "",
    }


def blocked_manifest(
    *,
    reason: str,
    mix_path: Path,
    vocal_path: Path,
    instrumental_path: Path,
    output_dir: Path,
    track_id: str,
    base_candidate_id: str,
    detection_metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "manifest_type": "no_reference_vocal_silence_hallucination_repair",
        "repair_status": reason,
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "preview_only_not_for_listen_pack",
        "release_status": "not_release_ready",
        "dominant_failure_reason": reason,
        "stem_type": "vocal",
        "track_id": track_id,
        "base_candidate_id": base_candidate_id,
        "candidate_id": "",
        "candidate_row_jsonl": "",
        "rendered_vocal_candidate_path": "",
        "suppressed_window_count": 0,
        "suppressed_duration_sec": 0.0,
        "window_decisions": [],
        "detection_metrics": detection_metrics,
        "source_mix": source_ref(mix_path),
        "source_vocal": source_ref(vocal_path),
        "source_instrumental": source_ref(instrumental_path),
        "output_dir": str(output_dir),
        "safety_flags": dict(SAFETY_FLAGS),
        "can_mark_release_ready": False,
        "can_publish_as_result": False,
    }


def accepted_repair_manifest(
    *,
    mix_path: Path,
    vocal_path: Path,
    instrumental_path: Path,
    output_dir: Path,
    track_id: str,
    base_candidate_id: str,
    candidate_id: str,
    output_vocal_path: Path,
    candidate_row_jsonl: Path,
    decisions: list[dict[str, Any]],
    detection_metrics: dict[str, float],
    attenuation_gain: float,
    fade_sec: float,
) -> dict[str, Any]:
    suppressed_duration = sum(max(0.0, float(decision["end_sec"]) - float(decision["start_sec"])) for decision in decisions)
    return {
        "manifest_type": "no_reference_vocal_silence_hallucination_repair",
        "repair_status": "rendered_source_quiet_vocal_silence_hallucination_repair",
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "preview_only_not_for_listen_pack",
        "release_status": "not_release_ready",
        "dominant_failure_reason": "",
        "stem_type": "vocal",
        "track_id": track_id,
        "base_candidate_id": base_candidate_id,
        "candidate_id": candidate_id,
        "render_status": RENDER_STATUS,
        "candidate_row_jsonl": str(candidate_row_jsonl),
        "rendered_vocal_candidate_path": str(output_vocal_path),
        "requires_post_render_vocal_judge": True,
        "requires_dense_post_reconciliation_judge": False,
        "suppressed_window_count": int(sum(int(decision["window_count"]) for decision in decisions)),
        "suppressed_duration_sec": round(float(suppressed_duration), 6),
        "attenuation_gain": round(float(attenuation_gain), 6),
        "fade_sec": round(float(fade_sec), 6),
        "window_decisions": decisions,
        "detection_metrics": detection_metrics,
        "source_mix": source_ref(mix_path),
        "source_vocal": source_ref(vocal_path),
        "source_instrumental": source_ref(instrumental_path),
        "output_dir": str(output_dir),
        "safety_flags": dict(SAFETY_FLAGS),
        "can_mark_release_ready": False,
        "can_publish_as_result": False,
    }


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# No-Reference Vocal Silence Hallucination Repair Summary",
        "",
        f"- repair_status: {manifest.get('repair_status')}",
        f"- delivery_status: {manifest.get('delivery_status')}",
        f"- delivery_allowed: {str(manifest.get('delivery_allowed') is True).lower()}",
        f"- presentation_status: {manifest.get('presentation_status')}",
        f"- release_status: {manifest.get('release_status')}",
        f"- candidate_id: {manifest.get('candidate_id')}",
        f"- suppressed_window_count: {manifest.get('suppressed_window_count')}",
        f"- suppressed_duration_sec: {manifest.get('suppressed_duration_sec')}",
        f"- rendered_vocal_candidate_path: {manifest.get('rendered_vocal_candidate_path')}",
        "",
        "## Safety",
        "",
    ]
    safety = manifest.get("safety_flags") if isinstance(manifest.get("safety_flags"), dict) else {}
    for key in sorted(SAFETY_FLAGS):
        lines.append(f"- {key}: {str(safety.get(key) is True).lower()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_vocal_silence_hallucination_repair(
    *,
    mix_path: Path,
    vocal_path: Path,
    instrumental_path: Path,
    output_dir: Path,
    track_id: str,
    base_candidate_id: str,
    window_sec: float = 0.05,
    min_run_sec: float = 0.20,
    attenuation_gain: float = 0.0,
    fade_sec: float = 0.01,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_event(
        "load_audio",
        "start",
        mix_path=str(mix_path),
        vocal_path=str(vocal_path),
        instrumental_path=str(instrumental_path),
        output_dir=str(output_dir),
    )
    mix_audio, mix_rate = read_audio(mix_path)
    vocal_audio, vocal_rate = read_audio(vocal_path)
    read_audio(instrumental_path)
    mix_for_detection = fit_length(resample_audio(mix_audio, mix_rate, vocal_rate), vocal_audio.shape[0])
    log_event("load_audio", "end", sample_rate=vocal_rate, frame_count=int(vocal_audio.shape[0]))

    decisions, detection_metrics = source_quiet_vocal_windows(
        mix_audio=mix_for_detection,
        vocal_audio=vocal_audio,
        sample_rate=vocal_rate,
        window_sec=window_sec,
        min_run_sec=min_run_sec,
    )
    log_event(
        "detect_source_quiet_vocal_windows",
        "end",
        selected_window_count=sum(int(decision["window_count"]) for decision in decisions),
        raw_window_count=int(detection_metrics["raw_source_quiet_vocal_window_count"]),
    )
    if not decisions:
        manifest = blocked_manifest(
            reason="blocked_no_source_quiet_vocal_activity_windows",
            mix_path=mix_path,
            vocal_path=vocal_path,
            instrumental_path=instrumental_path,
            output_dir=output_dir,
            track_id=track_id,
            base_candidate_id=base_candidate_id,
            detection_metrics=detection_metrics,
        )
        write_json(output_dir / MANIFEST_NAME, manifest)
        write_summary(output_dir / SUMMARY_NAME, manifest)
        return manifest

    repaired_vocal, gain = apply_window_attenuation(
        vocal_audio,
        decisions=decisions,
        sample_rate=vocal_rate,
        attenuation_gain=attenuation_gain,
        fade_sec=fade_sec,
    )
    output_vocal_path = output_dir / "source_quiet_vocal_silence_repair_candidate.wav"
    sf.write(output_vocal_path, repaired_vocal, vocal_rate, subtype="FLOAT")
    candidate_id = safe_candidate_id(base_candidate_id)
    candidate_row_jsonl = output_dir / "source_quiet_vocal_silence_repair_candidate.jsonl"
    candidate_row = {
        "track_id": track_id,
        "candidate_id": candidate_id,
        "mix_path": str(mix_path),
        "source_audio_path": str(mix_path),
        "vocal_path": str(output_vocal_path),
        "selected_vocal_path": str(output_vocal_path),
        "instrumental_path": str(instrumental_path),
        "minus_path": str(instrumental_path),
        "selected_minus_path": str(instrumental_path),
        "candidate_family": "source_quiet_vocal_silence_hallucination_repair",
        "candidate_origin": "deterministic_no_reference_source_quiet_vocal_repair",
        "base_candidate_id": base_candidate_id,
        "candidate_manifest": str(output_dir / MANIFEST_NAME),
        "render_status": RENDER_STATUS,
        "score_mode": "vocal",
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "preview_only_not_for_listen_pack",
        "release_status": "not_release_ready",
        "can_mark_release_ready": False,
        "can_publish_as_result": False,
        "requires_post_render_vocal_judge": True,
        "requires_dense_post_reconciliation_judge": False,
        "segment_blend_needed": False,
        "window_decisions": decisions,
        "suppressed_window_count": int(sum(int(decision["window_count"]) for decision in decisions)),
        "suppression_gain_min": round(float(np.min(gain)), 6),
        "safety_flags": dict(SAFETY_FLAGS),
        "forbidden_actions": [
            "do_not_require_reference_stems",
            "do_not_require_human_csv",
            "do_not_start_finetune",
            "do_not_train",
            "do_not_add_models",
            "do_not_add_cleanup_debleed_denoise",
            "do_not_relax_gates",
            "do_not_export_diagnostic_audio_as_result",
            "do_not_accept_best_of_bad",
        ],
    }
    write_jsonl(candidate_row_jsonl, [candidate_row])
    manifest = accepted_repair_manifest(
        mix_path=mix_path,
        vocal_path=vocal_path,
        instrumental_path=instrumental_path,
        output_dir=output_dir,
        track_id=track_id,
        base_candidate_id=base_candidate_id,
        candidate_id=candidate_id,
        output_vocal_path=output_vocal_path,
        candidate_row_jsonl=candidate_row_jsonl,
        decisions=decisions,
        detection_metrics=detection_metrics,
        attenuation_gain=attenuation_gain,
        fade_sec=fade_sec,
    )
    write_json(output_dir / MANIFEST_NAME, manifest)
    write_summary(output_dir / SUMMARY_NAME, manifest)
    log_event("write_candidate", "end", output_manifest=str(output_dir / MANIFEST_NAME), candidate_row_jsonl=str(candidate_row_jsonl))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a no-reference vocal candidate with source-quiet silence hallucination repair."
    )
    parser.add_argument("--mix-path", type=Path, required=True)
    parser.add_argument("--vocal-path", type=Path, required=True)
    parser.add_argument("--instrumental-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--base-candidate-id", required=True)
    parser.add_argument("--window-sec", type=float, default=0.05)
    parser.add_argument("--min-run-sec", type=float, default=0.20)
    parser.add_argument("--attenuation-gain", type=float, default=0.0)
    parser.add_argument("--fade-sec", type=float, default=0.01)
    args = parser.parse_args(argv)

    manifest = build_vocal_silence_hallucination_repair(
        mix_path=args.mix_path,
        vocal_path=args.vocal_path,
        instrumental_path=args.instrumental_path,
        output_dir=args.output_dir,
        track_id=args.track_id,
        base_candidate_id=args.base_candidate_id,
        window_sec=args.window_sec,
        min_run_sec=args.min_run_sec,
        attenuation_gain=args.attenuation_gain,
        fade_sec=args.fade_sec,
    )
    print(
        json.dumps(
            {
                key: manifest.get(key)
                for key in (
                    "repair_status",
                    "delivery_status",
                    "delivery_allowed",
                    "presentation_status",
                    "release_status",
                    "suppressed_window_count",
                    "candidate_row_jsonl",
                    "rendered_vocal_candidate_path",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if str(manifest.get("repair_status") or "").startswith("rendered_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
