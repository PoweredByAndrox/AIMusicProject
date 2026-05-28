from __future__ import annotations

import argparse
import json
import math
import secrets
import time
import urllib.error
import urllib.request
import wave
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


STEM_KEYS = ["vocals", "drums", "bass", "guitar", "piano", "other", "instrumental_mix"]
BACKING_STEMS = ["drums", "bass", "guitar", "piano", "other", "instrumental_mix"]


def container_to_host_path(project_root: Path, container_path: str) -> Path:
    prefix = "/app/data/"
    if container_path.startswith(prefix):
        return project_root / "data" / container_path[len(prefix) :]
    return project_root / container_path


def host_to_container_path(project_root: Path, host_path: Path) -> str:
    resolved_root = (project_root / "data").resolve()
    resolved_path = host_path.resolve()
    relative = resolved_path.relative_to(resolved_root).as_posix()
    return f"/app/data/{relative}"


def read_job(project_root: Path, job_id: str) -> dict:
    path = project_root / "data" / "jobs" / job_id / "job.json"
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_path(project_root: Path, job: dict, key: str) -> Path:
    value = job.get("artifacts", {}).get(key)
    if not value:
        raise KeyError(f"artifact {key!r} missing for job {job['job_id']}")
    path = container_to_host_path(project_root, value)
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
        raise RuntimeError(f"unsupported sample width {sample_width}")

    if channels > 1:
        data = data.reshape(-1, channels)
    else:
        data = data.reshape(-1, 1)
    return data, sr


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(pcm.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def to_index(sr: int, seconds: float) -> int:
    return max(0, int(round(seconds * sr)))


def ensure_same_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape[0] == b.shape[0]:
        return a, b
    size = min(a.shape[0], b.shape[0])
    return a[:size], b[:size]


def resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    if audio.shape[0] < 2:
        return audio.copy()

    duration = (audio.shape[0] - 1) / float(src_sr)
    new_length = max(2, int(round(duration * dst_sr)) + 1)
    src_times = np.linspace(0.0, duration, audio.shape[0], endpoint=True)
    dst_times = np.linspace(0.0, duration, new_length, endpoint=True)
    channels = [np.interp(dst_times, src_times, audio[:, idx]) for idx in range(audio.shape[1])]
    return np.stack(channels, axis=1)


def apply_crossfade_replace(dest: np.ndarray, src: np.ndarray, start_sec: float, end_sec: float, sr: int, fade_sec: float) -> None:
    start = to_index(sr, start_sec)
    end = min(dest.shape[0], to_index(sr, end_sec))
    if end <= start:
        return

    src, dest = ensure_same_length(src, dest)
    fade = min(int(fade_sec * sr), max(0, (end - start) // 2))
    core_start = start + fade
    core_end = end - fade

    if fade > 0:
        fade_in = np.linspace(0.0, 1.0, fade, endpoint=False).reshape(-1, 1)
        fade_out = np.linspace(1.0, 0.0, fade, endpoint=False).reshape(-1, 1)
        dest[start:core_start] = dest[start:core_start] * (1.0 - fade_in) + src[start:core_start] * fade_in
        dest[core_end:end] = dest[core_end:end] * (1.0 - fade_out) + src[core_end:end] * fade_out

    if core_end > core_start:
        dest[core_start:core_end] = src[core_start:core_end]


def apply_gain(dest: np.ndarray, start_sec: float, end_sec: float, sr: int, gain_db: float, fade_sec: float) -> None:
    start = to_index(sr, start_sec)
    end = min(dest.shape[0], to_index(sr, end_sec))
    if end <= start:
        return
    gain = 10 ** (gain_db / 20.0)
    fade = min(int(fade_sec * sr), max(0, (end - start) // 2))
    segment = dest[start:end].copy()

    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, endpoint=False).reshape(-1, 1)
        segment[:fade] *= (1.0 + (gain - 1.0) * ramp)
        segment[-fade:] *= (gain - (gain - 1.0) * ramp)
    if end - start > 2 * fade:
        segment[fade : end - start - fade] *= gain
    elif fade == 0:
        segment *= gain
    dest[start:end] = segment


def local_vocal_presence_patch(
    segment: np.ndarray,
    sr: int,
    gain_db: float = 1.1,
    presence_mul: float = 1.22,
    air_mul: float = 1.18,
    sparkle_mul: float = 1.08,
) -> np.ndarray:
    if segment.shape[0] < 8:
        return segment.copy()

    if segment.ndim == 1:
        segment = segment.reshape(-1, 1)

    freqs = np.fft.rfftfreq(segment.shape[0], d=1.0 / sr)
    gain = np.ones_like(freqs)
    gain[(freqs >= 250) & (freqs < 900)] *= 0.92
    gain[(freqs >= 2400) & (freqs < 5200)] *= presence_mul
    gain[(freqs >= 7000) & (freqs < 12500)] *= air_mul
    gain[freqs >= 12500] *= sparkle_mul

    patched_channels: list[np.ndarray] = []
    for channel in range(segment.shape[1]):
        spec = np.fft.rfft(segment[:, channel])
        patched = np.fft.irfft(spec * gain, n=segment.shape[0]).real
        patched *= 10 ** (gain_db / 20.0)
        peak = float(np.max(np.abs(patched)) + 1e-9)
        if peak > 0.995:
            patched *= 0.995 / peak
        patched_channels.append(patched)
    return np.stack(patched_channels, axis=1)


def segment_metrics(vocals: np.ndarray, mix: np.ndarray, sr: int) -> dict:
    vocals, mix = ensure_same_length(vocals, mix)
    if vocals.shape[0] == 0:
        return {
            "vocal_to_mix_db": -120.0,
            "centroid_hz": 0.0,
            "hi4_ratio": 0.0,
            "peak": 0.0,
            "vocal_rms": 0.0,
            "mix_rms": 0.0,
        }

    vocal_rms = float(np.sqrt(np.mean(np.square(vocals)) + 1e-12))
    mix_rms = float(np.sqrt(np.mean(np.square(mix)) + 1e-12))

    if vocals.ndim > 1:
        mono = np.mean(vocals, axis=1)
    else:
        mono = vocals
    windowed = mono * np.hanning(mono.shape[0]) if mono.shape[0] > 8 else mono
    spec = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(windowed.shape[0], d=1.0 / sr)
    total = float(np.sum(spec) + 1e-12)

    return {
        "vocal_to_mix_db": float(20.0 * math.log10((vocal_rms + 1e-9) / (mix_rms + 1e-9))),
        "centroid_hz": float(np.sum(freqs * spec) / total),
        "hi4_ratio": float(np.sum(spec[freqs >= 4000]) / total),
        "peak": float(np.max(np.abs(vocals)) + 1e-9),
        "vocal_rms": vocal_rms,
        "mix_rms": mix_rms,
    }


def apply_local_patch(
    dest: np.ndarray,
    start_sec: float,
    end_sec: float,
    sr: int,
    fade_sec: float,
    overlay_source: np.ndarray | None = None,
    patch_kwargs: dict | None = None,
) -> None:
    start = to_index(sr, start_sec)
    end = min(dest.shape[0], to_index(sr, end_sec))
    if end <= start:
        return

    source = dest if overlay_source is None else overlay_source
    patch_kwargs = patch_kwargs or {}
    patched = local_vocal_presence_patch(source[start:end].copy(), sr, **patch_kwargs)
    src = dest.copy()
    src[start:end] = patched
    apply_crossfade_replace(dest, src, start_sec, end_sec, sr, fade_sec)


def choose_targeted_phrase_repair(
    tuned_audio: dict[str, np.ndarray],
    open_audio: dict[str, np.ndarray],
    sr: int,
    start_sec: float,
    end_sec: float,
    fade_sec: float,
) -> dict:
    start = to_index(sr, start_sec)
    end = to_index(sr, end_sec)

    base_metrics = segment_metrics(tuned_audio["vocals"][start:end], tuned_audio["instrumental_mix"][start:end], sr)
    best: dict | None = None

    for gain_db in (2.0, 2.2, 2.4):
        for presence_mul in (1.34, 1.38):
            for air_mul in (1.22, 1.26, 1.30):
                for mix_duck_db in (-1.5, -1.75):
                    candidate_vocals = tuned_audio["vocals"].copy()
                    candidate_mix = tuned_audio["instrumental_mix"].copy()
                    candidate_other = tuned_audio["other"].copy()
                    candidate_drums = tuned_audio["drums"].copy()
                    candidate_guitar = tuned_audio["guitar"].copy()
                    candidate_piano = tuned_audio["piano"].copy()

                    apply_local_patch(
                        candidate_vocals,
                        start_sec,
                        end_sec,
                        sr,
                        fade_sec,
                        overlay_source=open_audio["vocals"],
                        patch_kwargs={
                            "gain_db": gain_db,
                            "presence_mul": presence_mul,
                            "air_mul": air_mul,
                            "sparkle_mul": 1.10,
                        },
                    )
                    other_duck_db = min(mix_duck_db + 0.25, -0.8)
                    drums_duck_db = max(mix_duck_db + 0.75, -1.0)
                    harmonic_duck_db = -0.8

                    apply_gain(candidate_mix, start_sec, end_sec, sr, gain_db=mix_duck_db, fade_sec=fade_sec)
                    apply_gain(candidate_other, start_sec, end_sec, sr, gain_db=other_duck_db, fade_sec=fade_sec)
                    apply_gain(candidate_drums, start_sec, end_sec, sr, gain_db=drums_duck_db, fade_sec=fade_sec)
                    apply_gain(candidate_guitar, start_sec, end_sec, sr, gain_db=harmonic_duck_db, fade_sec=fade_sec)
                    apply_gain(candidate_piano, start_sec, end_sec, sr, gain_db=harmonic_duck_db, fade_sec=fade_sec)

                    metrics = segment_metrics(candidate_vocals[start:end], candidate_mix[start:end], sr)
                    score = (
                        4.5 * (metrics["vocal_to_mix_db"] - base_metrics["vocal_to_mix_db"])
                        + 0.022 * (metrics["centroid_hz"] - base_metrics["centroid_hz"])
                        + 18.0 * (metrics["hi4_ratio"] - base_metrics["hi4_ratio"])
                        - 0.010 * max(0.0, metrics["centroid_hz"] - 2425.0)
                        - 220.0 * max(0.0, metrics["peak"] - 0.995)
                    )
                    candidate = {
                        "score": score,
                        "baseline_metrics": base_metrics,
                        "predicted_metrics": metrics,
                        "vocals_patch": {
                            "gain_db": gain_db,
                            "presence_mul": presence_mul,
                            "air_mul": air_mul,
                            "sparkle_mul": 1.10,
                        },
                        "stem_gains": {
                            "instrumental_mix": mix_duck_db,
                            "other": other_duck_db,
                            "drums": drums_duck_db,
                            "guitar": harmonic_duck_db,
                            "piano": harmonic_duck_db,
                        },
                    }
                    if best is None or candidate["score"] > best["score"]:
                        best = candidate

    if best is None:
        raise RuntimeError("failed to resolve targeted phrase repair")
    return best


def load_report(report_path: Path) -> dict:
    return json.loads(report_path.read_text(encoding="utf-8"))


def load_aligned_external_vocals(
    source_path: Path,
    sr: int,
    total_length: int,
    anchor_start_sec: float,
) -> np.ndarray:
    clip_audio, clip_sr = read_wav(source_path)
    if clip_sr != sr:
        clip_audio = resample_audio(clip_audio, clip_sr, sr)
    aligned = np.zeros((total_length, clip_audio.shape[1]), dtype=np.float64)
    start = max(0, to_index(sr, anchor_start_sec))
    end = min(total_length, start + clip_audio.shape[0])
    if end <= start:
        raise RuntimeError("external vocals alignment window is empty")
    aligned[start:end] = clip_audio[: end - start]
    return aligned


def normalize_open_windows(windows: Sequence[tuple[float, float]]) -> list[dict]:
    normalized: list[dict] = []
    for start, end in windows:
        if end <= start:
            continue
        normalized.append(
            {
                "type": "replace_with_open_plus",
                "start": float(start),
                "end": float(end),
                "fade_sec": 0.18,
                "stems": STEM_KEYS,
                "reason": "manual extra open_plus window",
            }
        )
    return normalized


def normalize_tandem_windows(windows: Sequence[tuple[float, float]]) -> list[dict]:
    normalized: list[dict] = []
    for start, end in windows:
        if end <= start:
            continue
        normalized.append(
            {
                "type": "local_tandem_repair",
                "start": float(start),
                "end": float(end),
                "fade_sec": 0.18,
                "reason": "manual local tandem repair window",
            }
        )
    return normalized


def choose_local_tandem_repair(
    tuned_audio: dict[str, np.ndarray],
    open_audio: dict[str, np.ndarray],
    sr: int,
    start_sec: float,
    end_sec: float,
    fade_sec: float,
    external_vocals: dict[str, np.ndarray] | None = None,
) -> dict:
    start = to_index(sr, start_sec)
    end = to_index(sr, end_sec)
    if end <= start:
        raise RuntimeError("invalid tandem repair window")

    base_vocals = open_audio["vocals"][start:end]
    base_mix = open_audio["instrumental_mix"][start:end]
    base_metrics = segment_metrics(base_vocals, base_mix, sr)
    best: dict | None = None

    vocal_patch_options = [
        {"mode": "none"},
        {"mode": "mild", "gain_db": 0.0, "presence_mul": 1.06, "air_mul": 1.04, "sparkle_mul": 1.04},
        {"mode": "mid", "gain_db": 0.0, "presence_mul": 1.10, "air_mul": 1.06, "sparkle_mul": 1.05},
    ]
    duck_presets = [
        {"instrumental_mix": 0.0, "drums": 0.0, "other": 0.0, "guitar": 0.0, "piano": 0.0, "bass": 0.0},
        {"instrumental_mix": -0.75, "drums": -0.5, "other": -0.75, "guitar": -0.26, "piano": -0.26, "bass": 0.0},
        {"instrumental_mix": -1.25, "drums": -1.0, "other": -1.0, "guitar": -0.44, "piano": -0.44, "bass": 0.0},
        {"instrumental_mix": -1.75, "drums": -1.0, "other": -1.5, "guitar": -0.61, "piano": -0.61, "bass": 0.0},
        {"instrumental_mix": -2.25, "drums": -1.5, "other": -1.5, "guitar": -0.79, "piano": -0.79, "bass": 0.0},
    ]

    vocal_sources = {
        "open": open_audio["vocals"],
        "tuned": tuned_audio["vocals"],
    }
    for name, audio in (external_vocals or {}).items():
        vocal_sources[name] = audio

    for vocal_source, source_audio in vocal_sources.items():
        source_vocals = source_audio[start:end].copy()
        for back_source in ("tuned", "open"):
            back_map = tuned_audio if back_source == "tuned" else open_audio
            for patch in vocal_patch_options:
                candidate_vocals = source_vocals.copy()
                if patch["mode"] != "none":
                    candidate_vocals = local_vocal_presence_patch(
                        candidate_vocals,
                        sr,
                        gain_db=patch["gain_db"],
                        presence_mul=patch["presence_mul"],
                        air_mul=patch["air_mul"],
                        sparkle_mul=patch["sparkle_mul"],
                    )
                for stem_gains in duck_presets:
                    candidate_mix = np.zeros_like(candidate_vocals)
                    for stem in ("drums", "bass", "guitar", "piano", "other"):
                        source_segment = back_map[stem][start:end].copy()
                        gain_db = stem_gains.get(stem, 0.0)
                        if abs(gain_db) > 1e-9:
                            source_segment *= 10 ** (gain_db / 20.0)
                        candidate_mix += source_segment

                    metrics = segment_metrics(candidate_vocals, candidate_mix, sr)
                    score = (
                        4.0 * metrics["vocal_to_mix_db"]
                        + 0.016 * min(metrics["centroid_hz"], 6200.0)
                        + 28.0 * metrics["hi4_ratio"]
                        - 0.012 * max(0.0, metrics["centroid_hz"] - 6600.0)
                        - 220.0 * max(0.0, metrics["peak"] - 0.995)
                        - 0.20 * abs(stem_gains["instrumental_mix"])
                        - 0.12 * abs(stem_gains["drums"])
                        - 0.10 * abs(stem_gains["other"])
                    )

                    candidate = {
                        "score": score,
                        "baseline_metrics": base_metrics,
                        "predicted_metrics": metrics,
                        "vocal_source": vocal_source,
                        "back_source": back_source,
                        "vocals_patch": patch,
                        "stem_gains": stem_gains,
                    }
                    if best is None or candidate["score"] > best["score"]:
                        best = candidate

    if best is None:
        raise RuntimeError("failed to resolve local tandem repair")
    best["fade_sec"] = fade_sec
    return best


def build_actions(
    report: dict,
    extra_open_windows: Sequence[tuple[float, float]] | None = None,
    extra_tandem_windows: Sequence[tuple[float, float]] | None = None,
) -> list[dict]:
    actions: list[dict] = []
    for item in report["ranked_segments"]:
        start = float(item["start"])
        end = float(item["end"])
        duration = end - start
        issues = set(item.get("issues", []))
        recommendation = item.get("recommendation")
        metrics = item.get("metrics", {})
        tuned = metrics.get("tuned", {})
        open_plus = metrics.get("open_plus", {})
        tuned_v2m = float(tuned.get("vocal_to_mix_db", 0.0))
        open_v2m = float(open_plus.get("vocal_to_mix_db", 0.0))

        if recommendation == "use_open_plus":
            if duration < 2.4 and open_v2m - tuned_v2m < 0.25:
                actions.append(
                    {
                        "type": "targeted_phrase_repair",
                        "start": start,
                        "end": end,
                        "fade_sec": 0.12,
                        "reason": "short dull phrase where open_plus does not buy enough vocal separation",
                    }
                )
            else:
                actions.append(
                    {
                        "type": "replace_with_open_plus",
                        "start": start,
                        "end": end,
                        "fade_sec": 0.18,
                        "stems": STEM_KEYS,
                        "reason": "segment diagnosed as clearer with open_plus",
                    }
                )
        elif "open_plus_possible_harshness" in issues:
            actions.append(
                {
                    "type": "keep_tuned",
                    "start": start,
                    "end": end,
                    "reason": "open_plus gets too sharp here",
                }
            )

    actions.extend(normalize_open_windows(extra_open_windows or []))
    actions.extend(normalize_tandem_windows(extra_tandem_windows or []))
    deduped: list[dict] = []
    seen: set[tuple[str, float, float]] = set()
    for item in sorted(actions, key=lambda action: (action["start"], action["end"], action["type"])):
        key = (item["type"], round(item["start"], 3), round(item["end"], 3))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def api_json_request(url: str, payload: dict | None = None) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def rerun_lyrics_for_hybrid(
    api_base: str,
    source_job: dict,
    hybrid_vocals_path: str,
    hybrid_job: dict,
) -> dict:
    request_payload = {
        "track_name": hybrid_job["request"]["track_name"],
        "input_path": hybrid_vocals_path,
        "preset_name": "lyrics_sync_from_vocals",
        "notes": f"Derived lyrics rerun for {hybrid_job['job_id']}",
        "enable_lyrics": True,
        "lyrics_language": source_job["request"].get("lyrics_language"),
        "lyrics_reference_path": source_job["request"].get("lyrics_reference_path"),
    }
    created = api_json_request(f"{api_base.rstrip('/')}/api/v1/jobs", request_payload)
    job_id = created["job_id"]
    deadline = time.time() + 600
    while time.time() < deadline:
        current = api_json_request(f"{api_base.rstrip('/')}/api/v1/jobs/{job_id}")
        if current["status"] in {"completed", "failed"}:
            return current
        time.sleep(2.0)
    raise TimeoutError(f"lyrics rerun job {job_id} did not finish in time")


def build_hybrid(
    project_root: Path,
    base_job_id: str,
    overlay_job_id: str,
    report_path: Path,
    extra_open_windows: Sequence[tuple[float, float]] | None = None,
    extra_tandem_windows: Sequence[tuple[float, float]] | None = None,
    external_vocals_path: Path | None = None,
    external_vocals_anchor_start: float | None = None,
    rerun_lyrics: bool = False,
    api_base: str = "http://localhost:8011",
) -> tuple[Path, Path, str]:
    tuned_job = read_job(project_root, base_job_id)
    open_job = read_job(project_root, overlay_job_id)
    report = load_report(report_path)
    actions = build_actions(
        report,
        extra_open_windows=extra_open_windows,
        extra_tandem_windows=extra_tandem_windows,
    )

    output_id = f"segment_hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_root = project_root / "data" / "outputs" / output_id
    hybrid_dir = out_root / "segment_hybrid" / "01_hybrid"
    hybrid_dir.mkdir(parents=True, exist_ok=True)

    tuned_audio_map: dict[str, np.ndarray] = {}
    open_audio_map: dict[str, np.ndarray] = {}
    sample_rate: int | None = None
    for stem in STEM_KEYS:
        tuned_audio, sr = read_wav(artifact_path(project_root, tuned_job, stem))
        open_audio, open_sr = read_wav(artifact_path(project_root, open_job, stem))
        if sr != open_sr:
            open_audio = resample_audio(open_audio, open_sr, sr)
        tuned_audio, open_audio = ensure_same_length(tuned_audio, open_audio)
        tuned_audio_map[stem] = tuned_audio
        open_audio_map[stem] = open_audio
        sample_rate = sr if sample_rate is None else sample_rate

    external_vocals_sources: dict[str, np.ndarray] = {}
    if external_vocals_path is not None:
        if external_vocals_anchor_start is None:
            raise RuntimeError("external vocals anchor start is required when external vocals path is provided")
        external_vocals_sources["bs"] = load_aligned_external_vocals(
            external_vocals_path,
            sample_rate or 44100,
            tuned_audio_map["vocals"].shape[0],
            external_vocals_anchor_start,
        )

    resolved_actions: list[dict] = []
    for action in actions:
        resolved = dict(action)
        if resolved["type"] == "targeted_phrase_repair":
            resolved.update(
                choose_targeted_phrase_repair(
                    tuned_audio_map,
                    open_audio_map,
                    sample_rate or 44100,
                    resolved["start"],
                    resolved["end"],
                    resolved["fade_sec"],
                )
            )
        elif resolved["type"] == "local_tandem_repair":
            resolved.update(
                choose_local_tandem_repair(
                    tuned_audio_map,
                    open_audio_map,
                    sample_rate or 44100,
                    resolved["start"],
                    resolved["end"],
                    resolved["fade_sec"],
                    external_vocals=external_vocals_sources or None,
                )
            )
        resolved_actions.append(resolved)

    manifest = {
        "hybrid_output_id": output_id,
        "base_job": base_job_id,
        "overlay_job": overlay_job_id,
        "report_path": str(report_path),
        "actions": resolved_actions,
        "stems": {},
    }

    for stem in STEM_KEYS:
        tuned_audio = tuned_audio_map[stem]
        open_audio = open_audio_map[stem]
        sr = sample_rate or 44100
        hybrid = tuned_audio.copy()

        for action in resolved_actions:
            action_type = action["type"]
            if action_type == "replace_with_open_plus" and stem in action.get("stems", []):
                apply_crossfade_replace(hybrid, open_audio, action["start"], action["end"], sr, action["fade_sec"])
            elif action_type == "targeted_phrase_repair" and stem == "vocals":
                apply_local_patch(
                    hybrid,
                    action["start"],
                    action["end"],
                    sr,
                    action["fade_sec"],
                    overlay_source=open_audio,
                    patch_kwargs=action["vocals_patch"],
                )
            elif action_type == "targeted_phrase_repair" and stem in action.get("stem_gains", {}):
                apply_gain(
                    hybrid,
                    action["start"],
                    action["end"],
                    sr,
                    gain_db=action["stem_gains"][stem],
                    fade_sec=action["fade_sec"],
                )
            elif action_type == "local_tandem_repair":
                if action["vocal_source"] == "open":
                    vocal_source_audio = open_audio_map["vocals"]
                elif action["vocal_source"] == "tuned":
                    vocal_source_audio = tuned_audio_map["vocals"]
                else:
                    vocal_source_audio = external_vocals_sources[action["vocal_source"]]
                back_source_map = open_audio_map if action["back_source"] == "open" else tuned_audio_map
                if stem == "vocals":
                    patch = action.get("vocals_patch", {"mode": "none"})
                    if patch.get("mode") == "none":
                        apply_crossfade_replace(
                            hybrid,
                            vocal_source_audio,
                            action["start"],
                            action["end"],
                            sr,
                            action["fade_sec"],
                        )
                    else:
                        patch_kwargs = {
                            "gain_db": patch["gain_db"],
                            "presence_mul": patch["presence_mul"],
                            "air_mul": patch["air_mul"],
                            "sparkle_mul": patch["sparkle_mul"],
                        }
                        apply_local_patch(
                            hybrid,
                            action["start"],
                            action["end"],
                            sr,
                            action["fade_sec"],
                            overlay_source=vocal_source_audio,
                            patch_kwargs=patch_kwargs,
                        )
                elif stem in BACKING_STEMS:
                    apply_crossfade_replace(
                        hybrid,
                        back_source_map[stem],
                        action["start"],
                        action["end"],
                        sr,
                        action["fade_sec"],
                    )
                    gain_db = action.get("stem_gains", {}).get(stem, 0.0)
                    if abs(gain_db) > 1e-9:
                        apply_gain(
                            hybrid,
                            action["start"],
                            action["end"],
                            sr,
                            gain_db=gain_db,
                            fade_sec=action["fade_sec"],
                        )

        out_path = hybrid_dir / f"{stem}.wav"
        write_wav(out_path, hybrid, sr)
        manifest["stems"][stem] = str(out_path)

    manifest_path = out_root / "segment_hybrid_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    job_manifest_path = out_root / "job_manifest.json"
    job_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    hybrid_job_id = secrets.token_hex(6)
    job_root = project_root / "data" / "jobs" / hybrid_job_id
    job_root.mkdir(parents=True, exist_ok=True)
    run_log_entry = {
        "service": "segment-hybrid-builder",
        "status": "completed",
        "task": "segment_local_hybrid_stitch",
        "target_stem": "hybrid_mix",
        "command": [
            "python",
            "scripts/build_segment_hybrid.py",
            "--base-job",
            base_job_id,
            "--overlay-job",
            overlay_job_id,
            "--report-json",
            str(report_path),
        ],
        "output_files": [host_to_container_path(project_root, Path(path)) for path in manifest["stems"].values()],
        "named_outputs": {stem: host_to_container_path(project_root, Path(path)) for stem, path in manifest["stems"].items()},
        "metadata": {
            "base_job": base_job_id,
            "overlay_job": overlay_job_id,
            "actions": resolved_actions,
            "hybrid_output_id": output_id,
        },
        "notes": "Segment-local hybrid stitch completed from the selected base and overlay renders.",
        "resolved_input_path": tuned_job["request"]["input_path"],
        "build_from_paths": {
            "base_job": base_job_id,
            "overlay_job": overlay_job_id,
        },
    }
    hybrid_job = {
        "job_id": hybrid_job_id,
        "status": "completed",
        "request": {
            "track_name": f"{tuned_job['request']['track_name']} [segment hybrid]",
            "input_path": tuned_job["request"]["input_path"],
            "preset_name": "segment_hybrid_auto",
            "notes": "Auto-built segment-local hybrid stitched from a base render plus a focused overlay render",
            "enable_lyrics": True,
            "lyrics_language": tuned_job["request"].get("lyrics_language"),
            "lyrics_reference_path": tuned_job["request"].get("lyrics_reference_path"),
        },
        "preset": {
            "name": "segment_hybrid_auto",
            "description": "Synthetic completed result composed from a base render, overlay windows, and automatically tuned local repairs on weak phrase segments.",
            "targets": STEM_KEYS + ["lyrics_json", "lyrics_txt", "lyrics_lrc", "lyrics_srt", "job_manifest.json"],
            "phases": [
                {
                    "id": "segment_hybrid",
                    "label": "Segment-local hybrid stitch",
                    "steps": [
                        {
                            "stem": "hybrid_mix",
                            "service": "segment-hybrid-builder",
                            "task": "segment_local_hybrid_stitch",
                            "preset": "segment_hybrid_auto",
                        }
                    ],
                }
            ],
        },
        "invocations": [
            {
                "phase_id": "segment_hybrid",
                "stem": "hybrid_mix",
                "service": "segment-hybrid-builder",
                "task": "segment_local_hybrid_stitch",
                "model": None,
                "preset": "segment_hybrid_auto",
                "prompt": None,
                "input_bus": None,
                "build_from": {
                    "include": ["base", "overlay", "lyrics"],
                },
                "raw_step": {
                    "stem": "hybrid_mix",
                    "service": "segment-hybrid-builder",
                    "task": "segment_local_hybrid_stitch",
                    "preset": "segment_hybrid_auto",
                },
            }
        ],
        "run_log": [run_log_entry],
        "output_root": host_to_container_path(project_root, out_root),
        "artifacts": {
            **{stem: host_to_container_path(project_root, Path(path)) for stem, path in manifest["stems"].items()},
            "lyrics_wordlevel_json": tuned_job["artifacts"].get("lyrics_wordlevel_json", tuned_job["artifacts"].get("lyrics_json")),
            "lyrics_wordlevel_txt": tuned_job["artifacts"].get("lyrics_wordlevel_txt", tuned_job["artifacts"].get("lyrics_txt")),
            "lyrics_wordlevel_lrc": tuned_job["artifacts"].get("lyrics_wordlevel_lrc", tuned_job["artifacts"].get("lyrics_lrc")),
            "lyrics_wordlevel_srt": tuned_job["artifacts"].get("lyrics_wordlevel_srt", tuned_job["artifacts"].get("lyrics_srt")),
            "lyrics_json": tuned_job["artifacts"].get("lyrics_json", tuned_job["artifacts"].get("lyrics_wordlevel_json")),
            "lyrics_txt": tuned_job["artifacts"].get("lyrics_txt", tuned_job["artifacts"].get("lyrics_wordlevel_txt")),
            "lyrics_lrc": tuned_job["artifacts"].get("lyrics_lrc", tuned_job["artifacts"].get("lyrics_wordlevel_lrc")),
            "lyrics_srt": tuned_job["artifacts"].get("lyrics_srt", tuned_job["artifacts"].get("lyrics_wordlevel_srt")),
            "job_manifest.json": host_to_container_path(project_root, job_manifest_path),
        },
    }
    if rerun_lyrics:
        lyrics_job = rerun_lyrics_for_hybrid(
            api_base,
            tuned_job,
            hybrid_job["artifacts"]["vocals"],
            hybrid_job,
        )
        hybrid_job["artifacts"]["lyrics_wordlevel_json"] = lyrics_job["artifacts"].get(
            "lyrics_wordlevel_json", lyrics_job["artifacts"].get("lyrics_json")
        )
        hybrid_job["artifacts"]["lyrics_wordlevel_txt"] = lyrics_job["artifacts"].get(
            "lyrics_wordlevel_txt", lyrics_job["artifacts"].get("lyrics_txt")
        )
        hybrid_job["artifacts"]["lyrics_wordlevel_lrc"] = lyrics_job["artifacts"].get(
            "lyrics_wordlevel_lrc", lyrics_job["artifacts"].get("lyrics_lrc")
        )
        hybrid_job["artifacts"]["lyrics_wordlevel_srt"] = lyrics_job["artifacts"].get(
            "lyrics_wordlevel_srt", lyrics_job["artifacts"].get("lyrics_srt")
        )
        hybrid_job["artifacts"]["lyrics_json"] = lyrics_job["artifacts"].get(
            "lyrics_json", hybrid_job["artifacts"].get("lyrics_wordlevel_json")
        )
        hybrid_job["artifacts"]["lyrics_txt"] = lyrics_job["artifacts"].get(
            "lyrics_txt", hybrid_job["artifacts"].get("lyrics_wordlevel_txt")
        )
        hybrid_job["artifacts"]["lyrics_lrc"] = lyrics_job["artifacts"].get(
            "lyrics_lrc", hybrid_job["artifacts"].get("lyrics_wordlevel_lrc")
        )
        hybrid_job["artifacts"]["lyrics_srt"] = lyrics_job["artifacts"].get(
            "lyrics_srt", hybrid_job["artifacts"].get("lyrics_wordlevel_srt")
        )
        hybrid_job["run_log"].append(
            {
                "service": "lyrics-worker",
                "status": lyrics_job["status"],
                "task": "import_wordlevel_from_hybrid_vocals",
                "target_stem": "lyrics_wordlevel",
                "notes": "Fresh lyrics rerun was executed on the hybrid vocals output so timing/text matches the stitched audio.",
                "derived_from_job": lyrics_job["job_id"],
                "resolved_input_path": hybrid_job["artifacts"]["vocals"],
            }
        )
    (job_root / "job.json").write_text(json.dumps(hybrid_job, indent=2), encoding="utf-8")
    return manifest_path, hybrid_dir, hybrid_job_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--base-job")
    parser.add_argument("--overlay-job")
    parser.add_argument("--tuned-job")
    parser.add_argument("--open-job")
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--extra-open-window", action="append", default=[], help="window in start:end seconds to force open_plus")
    parser.add_argument("--extra-tandem-window", action="append", default=[], help="window in start:end seconds to force local tandem repair")
    parser.add_argument("--external-vocals-path", type=Path, help="external vocals wav to align into the hybrid search space")
    parser.add_argument("--external-vocals-anchor-start", type=float, help="absolute track start time for external vocals clip")
    parser.add_argument("--rerun-lyrics", action="store_true")
    parser.add_argument("--api-base", default="http://localhost:8011")
    args = parser.parse_args()

    extra_open_windows: list[tuple[float, float]] = []
    for raw in args.extra_open_window:
        start_s, end_s = raw.split(":", 1)
        extra_open_windows.append((float(start_s), float(end_s)))
    extra_tandem_windows: list[tuple[float, float]] = []
    for raw in args.extra_tandem_window:
        start_s, end_s = raw.split(":", 1)
        extra_tandem_windows.append((float(start_s), float(end_s)))

    base_job_id = args.base_job or args.tuned_job
    overlay_job_id = args.overlay_job or args.open_job
    if not base_job_id or not overlay_job_id:
        raise SystemExit("Either --base-job/--overlay-job or legacy --tuned-job/--open-job must be provided")

    manifest_path, hybrid_dir, hybrid_job_id = build_hybrid(
        args.root,
        base_job_id,
        overlay_job_id,
        args.report_json,
        extra_open_windows=extra_open_windows,
        extra_tandem_windows=extra_tandem_windows,
        external_vocals_path=args.external_vocals_path,
        external_vocals_anchor_start=args.external_vocals_anchor_start,
        rerun_lyrics=args.rerun_lyrics,
        api_base=args.api_base,
    )
    print(json.dumps({"manifest": str(manifest_path), "hybrid_dir": str(hybrid_dir), "hybrid_job_id": hybrid_job_id}, indent=2))


if __name__ == "__main__":
    main()
