from __future__ import annotations

import argparse
import json
import math
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


@dataclass
class Segment:
    index: int
    start: float
    end: float
    text: str


def read_job(root: Path, job_id: str) -> dict:
    path = root / "data" / "jobs" / job_id / "job.json"
    return json.loads(path.read_text(encoding="utf-8"))


def container_to_host_path(project_root: Path, container_path: str) -> Path:
    prefix = "/app/data/"
    if container_path.startswith(prefix):
        return project_root / "data" / container_path[len(prefix) :]
    return project_root / container_path


def load_segments(job_root: Path, job: dict) -> list[Segment]:
    lyrics_path = None
    for key in ("lyrics_wordlevel_json", "lyrics_json"):
        value = job.get("artifacts", {}).get(key)
        if value:
            lyrics_path = container_to_host_path(job_root, value)
            break
    if lyrics_path is None or not lyrics_path.exists():
        raise FileNotFoundError("lyrics json artifact not found")

    payload = json.loads(lyrics_path.read_text(encoding="utf-8"))
    segments: list[Segment] = []
    for idx, item in enumerate(payload.get("segments", []), start=1):
        text = " ".join(str(item.get("text", "")).split())
        segments.append(
            Segment(
                index=idx,
                start=float(item["start"]),
                end=float(item["end"]),
                text=text,
            )
        )
    return segments


def resolve_audio_path(job_root: Path, job: dict, key: str) -> Path:
    value = job.get("artifacts", {}).get(key)
    if not value:
        raise KeyError(f"artifact {key!r} not found in job {job['job_id']}")
    path = container_to_host_path(job_root, value)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        frame_count = wf.getnframes()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(frame_count)

    if sample_width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483648.0
    else:
        raise RuntimeError(f"unsupported sample width: {sample_width}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sr


def slice_audio(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    start_idx = max(0, int(start * sr))
    end_idx = min(audio.shape[0], int(end * sr))
    if end_idx <= start_idx:
        return np.zeros(1, dtype=np.float64)
    return audio[start_idx:end_idx]


def compute_metrics(audio: np.ndarray, sr: int) -> dict:
    if audio.size == 0:
        audio = np.zeros(1, dtype=np.float64)

    rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
    peak = float(np.max(np.abs(audio)))
    crest = float(peak / (rms + 1e-12))

    windowed = audio * np.hanning(audio.shape[0]) if audio.shape[0] > 8 else audio
    spec = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(windowed.shape[0], d=1.0 / sr)
    total = float(np.sum(spec) + 1e-12)

    centroid = float(np.sum(freqs * spec) / total)
    hi4 = float(np.sum(spec[freqs >= 4000]) / total)
    hi8 = float(np.sum(spec[freqs >= 8000]) / total)
    band2_5 = float(np.sum(spec[(freqs >= 2000) & (freqs < 5000)]) / total)
    band5_12 = float(np.sum(spec[(freqs >= 5000) & (freqs < 12000)]) / total)

    return {
        "rms": rms,
        "peak": peak,
        "crest": crest,
        "centroid_hz": centroid,
        "hi4_ratio": hi4,
        "hi8_ratio": hi8,
        "band2_5_ratio": band2_5,
        "band5_12_ratio": band5_12,
    }


def summarize_segment(
    segment: Segment,
    variants: dict[str, dict[str, np.ndarray | int]],
) -> dict:
    metrics: dict[str, dict] = {}
    for variant_name, payload in variants.items():
        sr = int(payload["sr"])
        vocals = slice_audio(payload["vocals"], sr, segment.start, segment.end)
        mix = slice_audio(payload["instrumental_mix"], sr, segment.start, segment.end)
        other = slice_audio(payload["other"], sr, segment.start, segment.end)

        vocals_metrics = compute_metrics(vocals, sr)
        mix_metrics = compute_metrics(mix, sr)
        other_metrics = compute_metrics(other, sr)

        vocal_to_mix_db = float(20.0 * math.log10((vocals_metrics["rms"] + 1e-9) / (mix_metrics["rms"] + 1e-9)))
        metrics[variant_name] = {
            "vocals": vocals_metrics,
            "instrumental_mix": mix_metrics,
            "other": other_metrics,
            "vocal_to_mix_db": vocal_to_mix_db,
        }

    old = metrics["old"]
    tuned = metrics["tuned"]
    open_plus = metrics["open_plus"]

    issues: list[str] = []
    if tuned["vocals"]["centroid_hz"] < old["vocals"]["centroid_hz"] + 80:
        issues.append("tuned_vocals_still_dull")
    if tuned["vocal_to_mix_db"] < old["vocal_to_mix_db"] - 0.4:
        issues.append("tuned_vocals_more_buried")
    if open_plus["instrumental_mix"]["hi4_ratio"] - tuned["instrumental_mix"]["hi4_ratio"] > 0.02 and open_plus["vocal_to_mix_db"] < tuned["vocal_to_mix_db"] - 0.2:
        issues.append("open_plus_backing_masks_vocals")
    if open_plus["vocals"]["hi8_ratio"] - tuned["vocals"]["hi8_ratio"] > 0.03 and open_plus["vocals"]["crest"] < tuned["vocals"]["crest"] - 0.6:
        issues.append("open_plus_possible_harshness")

    severity = 0.0
    severity += max(0.0, 5600.0 - tuned["vocals"]["centroid_hz"]) / 120.0
    severity += max(0.0, -tuned["vocal_to_mix_db"] - 7.0) * 1.8
    severity += max(0.0, 0.52 - tuned["vocals"]["hi4_ratio"]) * 35.0
    if "open_plus_backing_masks_vocals" in issues:
        severity += 3.5
    if "open_plus_possible_harshness" in issues:
        severity += 2.5

    recommendation = "keep_tuned"
    if "tuned_vocals_still_dull" in issues and "open_plus_backing_masks_vocals" not in issues:
        recommendation = "use_open_plus"
    elif "open_plus_backing_masks_vocals" in issues:
        recommendation = "hybrid_fix_vocals_only"
    elif "tuned_vocals_more_buried" in issues:
        recommendation = "boost_vocal_only"

    return {
        "segment_index": segment.index,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "metrics": metrics,
        "issues": issues,
        "severity": severity,
        "recommendation": recommendation,
    }


def build_report(root: Path, old_job_id: str, tuned_job_id: str, open_job_id: str) -> tuple[Path, Path]:
    old_job = read_job(root, old_job_id)
    tuned_job = read_job(root, tuned_job_id)
    open_job = read_job(root, open_job_id)

    segments = load_segments(root, tuned_job)
    variants: dict[str, dict[str, np.ndarray | int]] = {}
    for name, job in (("old", old_job), ("tuned", tuned_job), ("open_plus", open_job)):
        vocals, sr = read_wav(resolve_audio_path(root, job, "vocals"))
        instrumental, sr2 = read_wav(resolve_audio_path(root, job, "instrumental_mix"))
        other, sr3 = read_wav(resolve_audio_path(root, job, "other"))
        if sr != sr2 or sr != sr3:
            raise RuntimeError("sample rates mismatch between artifacts")
        variants[name] = {
            "sr": sr,
            "vocals": vocals,
            "instrumental_mix": instrumental,
            "other": other,
        }

    segment_rows = [summarize_segment(segment, variants) for segment in segments]
    ranked = sorted(segment_rows, key=lambda item: item["severity"], reverse=True)

    out_dir = root / ".tmp" / "stem-diagnostics" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "segment_report.json"
    json_path.write_text(
        json.dumps(
            {
                "jobs": {"old": old_job_id, "tuned": tuned_job_id, "open_plus": open_job_id},
                "segment_count": len(segment_rows),
                "ranked_segments": ranked,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    md_lines = [
        "# Stem Diagnostics",
        "",
        f"- old: `{old_job_id}`",
        f"- tuned: `{tuned_job_id}`",
        f"- open_plus: `{open_job_id}`",
        "",
        "## Top Problem Segments",
        "",
    ]

    for item in ranked[:5]:
        tuned = item["metrics"]["tuned"]
        open_plus = item["metrics"]["open_plus"]
        text = item["text"][:160].strip()
        md_lines.extend(
            [
                f"### Segment {item['segment_index']} `{item['start']:.2f}-{item['end']:.2f}`",
                "",
                f"- text: {text}",
                f"- severity: {item['severity']:.2f}",
                f"- issues: {', '.join(item['issues']) or 'none'}",
                f"- recommendation: `{item['recommendation']}`",
                f"- tuned vocals centroid: `{tuned['vocals']['centroid_hz']:.0f} Hz`",
                f"- tuned vocal/mix: `{tuned['vocal_to_mix_db']:.2f} dB`",
                f"- open_plus vocals centroid: `{open_plus['vocals']['centroid_hz']:.0f} Hz`",
                f"- open_plus vocal/mix: `{open_plus['vocal_to_mix_db']:.2f} dB`",
                "",
            ]
        )

    md_path = out_dir / "segment_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--old-job", required=True)
    parser.add_argument("--tuned-job", required=True)
    parser.add_argument("--open-job", required=True)
    args = parser.parse_args()

    json_path, md_path = build_report(args.root, args.old_job, args.tuned_job, args.open_job)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
