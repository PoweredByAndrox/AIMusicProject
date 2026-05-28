from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np


def read_job(project_root: Path, job_id: str) -> dict:
    path = project_root / "data" / "jobs" / job_id / "job.json"
    return json.loads(path.read_text(encoding="utf-8"))


def container_to_host_path(project_root: Path, container_path: str) -> Path:
    prefix = "/app/data/"
    if container_path.startswith(prefix):
        return project_root / "data" / container_path[len(prefix) :]
    return project_root / container_path


def _parse_wav_header_compat(path: Path) -> tuple[int, int, int, int, bytes]:
    """Manual RIFF/WAVE parser that tolerates IEEE_FLOAT (format=3).

    Python's stdlib ``wave.open(rb)`` only accepts wFormatTag=1, so
    IEEE_FLOAT wavs (which the workers happily write via _write_float_wav)
    cannot be analyzed without this fallback. Returns
    ``(format_code, channels, samplerate, sample_width_bytes, raw_data_bytes)``.
    Supports format codes 1 (PCM) and 3 (IEEE_FLOAT) plus
    WAVE_FORMAT_EXTENSIBLE (0xFFFE) unwrapped to inner SubFormat.
    """
    import struct as _struct

    raw_format_code: int | None = None
    channels: int | None = None
    samplerate: int | None = None
    sample_width: int | None = None
    data_bytes: bytes | None = None

    with path.open("rb") as handle:
        riff = handle.read(12)
        if len(riff) < 12 or riff[0:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise RuntimeError(f"Not a RIFF/WAVE file: {path}")
        while True:
            header = handle.read(8)
            if len(header) < 8:
                break
            chunk_id = header[0:4]
            chunk_size = _struct.unpack("<I", header[4:8])[0]
            chunk_data = handle.read(chunk_size)
            if chunk_size % 2 == 1:
                handle.read(1)
            if chunk_id == b"fmt ":
                if len(chunk_data) < 16:
                    raise RuntimeError(f"fmt chunk too short in {path}")
                raw_format_code = _struct.unpack("<H", chunk_data[0:2])[0]
                channels = _struct.unpack("<H", chunk_data[2:4])[0]
                samplerate = _struct.unpack("<I", chunk_data[4:8])[0]
                bits_per_sample = _struct.unpack("<H", chunk_data[14:16])[0]
                sample_width = bits_per_sample // 8
                if raw_format_code == 0xFFFE and len(chunk_data) >= 40:
                    raw_format_code = _struct.unpack("<H", chunk_data[24:26])[0]
            elif chunk_id == b"data":
                data_bytes = chunk_data

    if (
        raw_format_code is None or channels is None or samplerate is None
        or sample_width is None or data_bytes is None
    ):
        raise RuntimeError(f"Incomplete WAV header in {path}")
    return raw_format_code, int(channels), int(samplerate), int(sample_width), data_bytes


def _decode_wav_payload(
    format_code: int, sample_width: int, raw: bytes, channels: int
) -> np.ndarray:
    if format_code == 3:
        if sample_width == 4:
            samples = np.frombuffer(raw, dtype="<f4").astype(np.float64)
        elif sample_width == 8:
            samples = np.frombuffer(raw, dtype="<f8").astype(np.float64)
        else:
            raise RuntimeError(f"Unsupported IEEE_FLOAT sample width: {sample_width}")
    elif format_code == 1:
        if sample_width == 2:
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
        elif sample_width == 3:
            triplets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            ints = (
                triplets[:, 0].astype(np.int32)
                | (triplets[:, 1].astype(np.int32) << 8)
                | (triplets[:, 2].astype(np.int32) << 16)
            )
            sign_bit = 1 << 23
            ints = np.where(ints & sign_bit, ints - (1 << 24), ints)
            samples = ints.astype(np.float64) / float((1 << 23) - 1)
        elif sample_width == 4:
            samples = np.frombuffer(raw, dtype="<i4").astype(np.float64) / float((1 << 31) - 1)
        else:
            raise RuntimeError(f"Unsupported PCM sample width: {sample_width}")
    else:
        raise RuntimeError(f"Unsupported WAV format code {format_code}")
    return samples


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    format_code, channels, sr, sample_width, raw = _parse_wav_header_compat(path)
    data = _decode_wav_payload(format_code, sample_width, raw, channels)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sr


def read_wav_stereo(path: Path) -> tuple[np.ndarray, int]:
    format_code, channels, sr, sample_width, raw = _parse_wav_header_compat(path)
    data = _decode_wav_payload(format_code, sample_width, raw, channels)
    if channels <= 1:
        mono = data.reshape(-1)
        stereo = np.vstack([mono, mono])
        return stereo, sr

    data = data.reshape(-1, channels).T
    if data.shape[0] > 2:
        data = data[:2, :]
    return data, sr


def decode_audio(path: Path) -> tuple[np.ndarray, int]:
    if path.suffix.lower() == ".wav":
        return read_wav(path)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        temp_path = Path(handle.name)

    try:
        ffmpeg_path = shutil_which("ffmpeg")
        if ffmpeg_path:
            command = [
                ffmpeg_path,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                str(temp_path),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
        else:
            docker_decode_audio(path, temp_path)
        return read_wav(temp_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def decode_audio_stereo(path: Path) -> tuple[np.ndarray, int]:
    if path.suffix.lower() == ".wav":
        return read_wav_stereo(path)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        temp_path = Path(handle.name)

    try:
        ffmpeg_path = shutil_which("ffmpeg")
        if ffmpeg_path:
            command = [
                ffmpeg_path,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                str(temp_path),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
        else:
            docker_decode_audio(path, temp_path)
        return read_wav_stereo(temp_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def shutil_which(executable: str) -> str | None:
    for directory in os.getenv("PATH", "").split(os.pathsep):
        candidate = Path(directory) / executable
        if candidate.exists():
            return str(candidate)
        if os.name == "nt":
            candidate_exe = Path(directory) / f"{executable}.exe"
            if candidate_exe.exists():
                return str(candidate_exe)
    return None


def docker_decode_audio(source_path: Path, target_path: Path) -> None:
    win_source = str(source_path.resolve()).replace("\\", "/")
    win_target = str(target_path.resolve()).replace("\\", "/")
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{win_source}:/tmp/in:ro",
        "-v",
        f"{win_target}:/tmp/out.wav",
        "jrottenberg/ffmpeg:6.1-scratch",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "/tmp/in",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "/tmp/out.wav",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    if audio.shape[0] < 2:
        return audio.copy()

    duration = (audio.shape[0] - 1) / float(src_sr)
    new_length = max(2, int(round(duration * dst_sr)) + 1)
    src_times = np.linspace(0.0, duration, audio.shape[0], endpoint=True)
    dst_times = np.linspace(0.0, duration, new_length, endpoint=True)
    return np.interp(dst_times, src_times, audio)


def align_pair(a: np.ndarray, a_sr: int, b: np.ndarray, b_sr: int, target_sr: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    resolved_sr = target_sr or min(a_sr, b_sr)
    aa = resample_audio(a, a_sr, resolved_sr)
    bb = resample_audio(b, b_sr, resolved_sr)
    size = min(aa.shape[0], bb.shape[0])
    return aa[:size], bb[:size], resolved_sr


def _ensure_stereo_metric_audio(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        return np.vstack([arr, arr])
    if arr.shape[0] == 1:
        return np.vstack([arr[0], arr[0]])
    if arr.shape[0] > 2:
        return arr[:2]
    return arr


def resample_audio_stereo(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    arr = _ensure_stereo_metric_audio(audio)
    if src_sr == dst_sr:
        return arr
    return np.vstack([resample_audio(channel, src_sr, dst_sr) for channel in arr])


def align_pair_stereo(
    a: np.ndarray,
    a_sr: int,
    b: np.ndarray,
    b_sr: int,
    target_sr: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    resolved_sr = target_sr or min(a_sr, b_sr)
    aa = resample_audio_stereo(a, a_sr, resolved_sr)
    bb = resample_audio_stereo(b, b_sr, resolved_sr)
    size = min(aa.shape[1], bb.shape[1])
    return aa[:, :size], bb[:, :size], resolved_sr


def spectral_metrics(audio: np.ndarray, sr: int) -> dict[str, float]:
    if audio.size == 0:
        audio = np.zeros(1, dtype=np.float64)

    rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
    peak = float(np.max(np.abs(audio)) + 1e-12)
    windowed = audio * np.hanning(audio.shape[0]) if audio.shape[0] > 8 else audio
    spec = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(windowed.shape[0], d=1.0 / sr)
    total = float(np.sum(spec) + 1e-12)

    return {
        "rms": rms,
        "peak": peak,
        "crest": float(peak / (rms + 1e-12)),
        "centroid_hz": float(np.sum(freqs * spec) / total),
        "hi4_ratio": float(np.sum(spec[freqs >= 4000]) / total),
        "hi8_ratio": float(np.sum(spec[freqs >= 8000]) / total),
        "vocal_band_ratio": float(np.sum(spec[(freqs >= 200) & (freqs < 4000)]) / total),
    }


def frame_instability_metrics(audio: np.ndarray, sr: int, win_sec: float = 0.080, hop_sec: float = 0.020) -> dict[str, float]:
    frame = max(256, int(sr * win_sec))
    hop = max(128, int(sr * hop_sec))
    if audio.shape[0] < frame:
        audio = np.pad(audio, (0, frame - audio.shape[0]))

    centroids: list[float] = []
    hi4: list[float] = []
    flux: list[float] = []
    envelopes: list[float] = []
    prev_spec: np.ndarray | None = None

    for start in range(0, audio.shape[0] - frame + 1, hop):
        chunk = audio[start : start + frame]
        envelopes.append(float(np.sqrt(np.mean(np.square(chunk)) + 1e-12)))
        chunk = chunk * np.hanning(frame)
        spec = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(frame, d=1.0 / sr)
        total = float(np.sum(spec) + 1e-12)
        centroids.append(float(np.sum(freqs * spec) / total))
        hi4.append(float(np.sum(spec[freqs >= 4000]) / total))
        if prev_spec is not None:
            flux.append(float(np.mean(np.abs(spec - prev_spec))))
        prev_spec = spec

    centroid_arr = np.asarray(centroids, dtype=np.float64)
    hi4_arr = np.asarray(hi4, dtype=np.float64)
    flux_arr = np.asarray(flux if flux else [0.0], dtype=np.float64)
    env_arr = np.asarray(envelopes, dtype=np.float64)
    env_arr = env_arr - float(np.mean(env_arr))

    if env_arr.shape[0] > 8:
        mod_spec = np.abs(np.fft.rfft(env_arr * np.hanning(env_arr.shape[0])))
        mod_freqs = np.fft.rfftfreq(env_arr.shape[0], d=hop_sec)
        mod_total = float(np.sum(mod_spec) + 1e-12)
        mod_4_12 = float(np.sum(mod_spec[(mod_freqs >= 4.0) & (mod_freqs < 12.0)]) / mod_total)
    else:
        mod_4_12 = 0.0

    return {
        "centroid_std": float(np.std(centroid_arr)),
        "centroid_mad": float(np.median(np.abs(centroid_arr - np.median(centroid_arr)))),
        "hi4_std": float(np.std(hi4_arr)),
        "flux_mean": float(np.mean(flux_arr)),
        "mod_4_12_ratio": mod_4_12,
    }


def source_similarity_metrics(stem_audio: np.ndarray, stem_sr: int, source_audio: np.ndarray, source_sr: int) -> dict[str, float]:
    stem, source, _ = align_pair(stem_audio, stem_sr, source_audio, source_sr)
    diff = stem - source
    source_rms = float(np.sqrt(np.mean(np.square(source)) + 1e-12))
    diff_rms = float(np.sqrt(np.mean(np.square(diff)) + 1e-12))
    corr = float(np.dot(stem, source) / (np.linalg.norm(stem) * np.linalg.norm(source) + 1e-12))
    return {
        "source_similarity_corr": corr,
        "diff_from_source_dbfs": float(20.0 * math.log10(diff_rms + 1e-12)),
        "diff_vs_source_rms_db": float(20.0 * math.log10((diff_rms + 1e-12) / (source_rms + 1e-12))),
    }


def vocal_bleed_metrics(stem_audio: np.ndarray, stem_sr: int, vocal_audio: np.ndarray, vocal_sr: int) -> dict[str, float]:
    stem, vocal, _ = align_pair(stem_audio, stem_sr, vocal_audio, vocal_sr)
    vv = float(np.dot(vocal, vocal) + 1e-12)
    alpha = float(np.dot(stem, vocal) / vv)
    projection = alpha * vocal
    leak_rms = float(np.sqrt(np.mean(np.square(projection)) + 1e-12))
    total_rms = float(np.sqrt(np.mean(np.square(stem)) + 1e-12))
    return {
        "alpha": alpha,
        "bleed_db_rel_total": float(20.0 * math.log10((leak_rms + 1e-12) / (total_rms + 1e-12))),
    }


def center_vocal_bleed_metrics(
    stem_audio: np.ndarray,
    stem_sr: int,
    vocal_audio: np.ndarray,
    vocal_sr: int,
) -> dict[str, float]:
    stem, vocal, _ = align_pair_stereo(stem_audio, stem_sr, vocal_audio, vocal_sr)
    stem_mid = 0.5 * (stem[0] + stem[1])
    vocal_mid = 0.5 * (vocal[0] + vocal[1])
    vv = float(np.dot(vocal_mid, vocal_mid) + 1e-12)
    alpha = float(np.dot(stem_mid, vocal_mid) / vv)
    projection = alpha * vocal_mid
    leak_rms = float(np.sqrt(np.mean(np.square(projection)) + 1e-12))
    total_rms = float(np.sqrt(np.mean(np.square(stem_mid)) + 1e-12))
    return {
        "alpha": alpha,
        "bleed_db_rel_total": float(20.0 * math.log10((leak_rms + 1e-12) / (total_rms + 1e-12))),
    }


def metrics_block(audio: np.ndarray, sr: int) -> dict[str, float]:
    block = {}
    block.update(spectral_metrics(audio, sr))
    block.update(frame_instability_metrics(audio, sr))
    block["sample_rate"] = float(sr)
    return block


def _other_instability_features(metrics: dict[str, float]) -> dict[str, float]:
    rms = float(metrics.get("rms", 0.0))
    centroid_hz = float(metrics.get("centroid_hz", 0.0))
    centroid_std = float(metrics.get("centroid_std", 0.0))
    hi4_std = float(metrics.get("hi4_std", 0.0))
    flux_mean = float(metrics.get("flux_mean", 0.0))
    mod_4_12_ratio = float(metrics.get("mod_4_12_ratio", 0.0))
    return {
        "rms": rms,
        "centroid_rel_std": float(centroid_std / max(800.0, centroid_hz)),
        "hi4_std": hi4_std,
        "flux_per_rms": float(flux_mean / (rms + 1e-12)),
        "mod_4_12_ratio": mod_4_12_ratio,
    }


def other_instability_compare(
    final_metrics: dict[str, float],
    demucs_metrics: dict[str, float],
) -> dict[str, Any]:
    final_features = _other_instability_features(final_metrics)
    demucs_features = _other_instability_features(demucs_metrics)

    demucs_rms = demucs_features["rms"]
    final_rms = final_features["rms"]
    compare_mode = "ratio"
    if demucs_rms < 0.010 or demucs_rms < final_rms * 0.70 or max(final_rms, demucs_rms) < 0.020:
        compare_mode = "absolute"

    ratios = {
        "centroid_std_ratio": float(final_features["centroid_rel_std"] / (demucs_features["centroid_rel_std"] + 1e-12)),
        "hi4_std_ratio": float(final_features["hi4_std"] / (demucs_features["hi4_std"] + 1e-12)),
        "flux_ratio": float(final_features["flux_per_rms"] / (demucs_features["flux_per_rms"] + 1e-12)),
        "mod_4_12_ratio": float(final_features["mod_4_12_ratio"] / (demucs_features["mod_4_12_ratio"] + 1e-12)),
    }
    return {
        "compare_mode": compare_mode,
        "demucs_is_near_silent": bool(demucs_rms < 0.006),
        "final": final_features,
        "demucs": demucs_features,
        "ratios": ratios,
    }


def other_instability_penalty(compare: dict[str, Any]) -> float:
    final_features = compare["final"]
    ratios = compare["ratios"]
    if compare["compare_mode"] == "absolute":
        return float(
            max(0.0, (final_features["centroid_rel_std"] - 0.85) * 40.0)
            + max(0.0, (final_features["hi4_std"] - 0.24) * 75.0)
            + max(0.0, (final_features["flux_per_rms"] - 8.0) * 18.0)
            + max(0.0, (final_features["mod_4_12_ratio"] - 0.34) * 80.0)
        )
    return float(
        max(0.0, (ratios["centroid_std_ratio"] - 1.08) * 48.0)
        + max(0.0, (ratios["hi4_std_ratio"] - 1.10) * 62.0)
        + max(0.0, (ratios["flux_ratio"] - 1.20) * 48.0)
        + max(0.0, (ratios["mod_4_12_ratio"] - 1.12) * 48.0)
    )


def other_usefulness_reward(compare: dict[str, Any], bleed_db: float) -> float:
    final_features = compare["final"]
    demucs_features = compare["demucs"]
    if final_features["rms"] < 0.006 or bleed_db > -28.0:
        return 0.0
    reward = 0.0
    reward += min(6.0, max(0.0, final_features["rms"] - demucs_features["rms"]) * 220.0)
    if compare["compare_mode"] == "absolute":
        reward += min(3.0, max(0.0, 0.95 - final_features["centroid_rel_std"]) * 6.0)
    else:
        reward += min(3.0, max(0.0, 1.08 - compare["ratios"]["centroid_std_ratio"]) * 10.0)
    return float(reward)


def find_lyrics_json_path(project_root: Path, job: dict[str, Any]) -> Path | None:
    for key in ("lyrics_wordlevel_json", "lyrics_json"):
        candidate = job.get("artifacts", {}).get(key)
        if candidate:
            path = container_to_host_path(project_root, candidate)
            if path.exists():
                return path
    output_root = container_to_host_path(project_root, job["output_root"])
    for name in ("lyrics_wordlevel.json", "lyrics.json"):
        matches = list(output_root.rglob(name))
        if matches:
            return matches[0]
    return None


def load_lyrics_segments(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_segments = payload.get("segments") or []
    segments: list[dict[str, Any]] = []
    for raw in raw_segments:
        start = float(raw.get("start", 0.0) or 0.0)
        end = float(raw.get("end", start) or start)
        words = []
        for word in raw.get("words") or []:
            w_start = float(word.get("start", start) or start)
            w_end = float(word.get("end", w_start) or w_start)
            text = str(word.get("text", "") or "").strip()
            if not text:
                continue
            words.append({"start": w_start, "end": w_end, "text": text})
        text = str(raw.get("text", "") or "").strip() or " ".join(item["text"] for item in words)
        if end <= start:
            end = max(start + 0.12, max((item["end"] for item in words), default=start + 0.12))
        if not text and not words:
            continue
        segments.append({"start": start, "end": end, "text": text, "words": words})
    return segments


def slice_audio(audio: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    start_index = max(0, int(round(start_sec * sr)))
    end_index = max(start_index + 1, int(round(end_sec * sr)))
    if start_index >= audio.shape[0]:
        return np.zeros(1, dtype=np.float64)
    return audio[start_index : min(end_index, audio.shape[0])]


def _segment_window_metrics(
    loaded: dict[str, tuple[np.ndarray, int]],
    segment: dict[str, Any],
) -> dict[str, Any]:
    start = max(0.0, float(segment["start"]) - 0.12)
    end = max(start + 0.15, float(segment["end"]) + 0.12)

    final_mix, final_mix_sr = loaded["instrumental_mix"]
    demucs_mix, demucs_mix_sr = loaded["demucs_instrumental_mix"]
    final_vocals, final_vocals_sr = loaded["vocals"]
    demucs_vocals, demucs_vocals_sr = loaded["demucs_vocals"]
    final_other, final_other_sr = loaded["other"]
    demucs_other, demucs_other_sr = loaded["demucs_other"]

    final_mix_window = slice_audio(final_mix, final_mix_sr, start, end)
    demucs_mix_window = slice_audio(demucs_mix, demucs_mix_sr, start, end)
    final_vocals_window = slice_audio(final_vocals, final_vocals_sr, start, end)
    demucs_vocals_window = slice_audio(demucs_vocals, demucs_vocals_sr, start, end)
    final_other_window = slice_audio(final_other, final_other_sr, start, end)
    demucs_other_window = slice_audio(demucs_other, demucs_other_sr, start, end)

    final_bleed = max(
        vocal_bleed_metrics(final_mix_window, final_mix_sr, final_vocals_window, final_vocals_sr)["bleed_db_rel_total"],
        vocal_bleed_metrics(final_mix_window, final_mix_sr, demucs_vocals_window, demucs_vocals_sr)["bleed_db_rel_total"],
    )
    demucs_bleed = max(
        vocal_bleed_metrics(demucs_mix_window, demucs_mix_sr, final_vocals_window, final_vocals_sr)["bleed_db_rel_total"],
        vocal_bleed_metrics(demucs_mix_window, demucs_mix_sr, demucs_vocals_window, demucs_vocals_sr)["bleed_db_rel_total"],
    )

    final_other_summary = metrics_block(final_other_window, final_other_sr)
    demucs_other_summary = metrics_block(demucs_other_window, demucs_other_sr)
    other_compare = other_instability_compare(final_other_summary, demucs_other_summary)
    warble = other_compare["ratios"]
    other_bleed = vocal_bleed_metrics(final_other_window, final_other_sr, final_vocals_window, final_vocals_sr)[
        "bleed_db_rel_total"
    ]

    final_vocal_metrics = spectral_metrics(final_vocals_window, final_vocals_sr)
    demucs_vocal_metrics = spectral_metrics(demucs_vocals_window, demucs_vocals_sr)
    openness_delta = spectral_metrics(final_mix_window, final_mix_sr)["centroid_hz"] - spectral_metrics(
        demucs_mix_window, demucs_mix_sr
    )["centroid_hz"]

    local_penalty = (
        max(0.0, (final_bleed + 28.0) * 3.0)
        + other_instability_penalty(other_compare)
        + max(
            0.0,
            20.0
            if (
                demucs_vocal_metrics["rms"] < final_vocal_metrics["rms"] * 0.95
                and demucs_vocal_metrics["centroid_hz"] < final_vocal_metrics["centroid_hz"] - 160.0
            )
            else 0.0,
        )
    )

    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "text": segment["text"],
        "word_count": len(segment.get("words") or []),
        "final_mix_vocal_bleed_db": round(final_bleed, 6),
        "demucs_mix_vocal_bleed_db": round(demucs_bleed, 6),
        "warble_centroid_ratio": round(warble["centroid_std_ratio"], 6),
        "warble_hi4_ratio": round(warble["hi4_std_ratio"], 6),
        "warble_flux_ratio": round(warble["flux_ratio"], 6),
        "other_compare_mode": other_compare["compare_mode"],
        "other_final_rms": round(other_compare["final"]["rms"], 6),
        "other_demucs_rms": round(other_compare["demucs"]["rms"], 6),
        "other_flux_per_rms": round(other_compare["final"]["flux_per_rms"], 6),
        "other_centroid_rel_std": round(other_compare["final"]["centroid_rel_std"], 6),
        "other_mod_4_12_ratio": round(other_compare["final"]["mod_4_12_ratio"], 6),
        "other_vocal_bleed_db": round(other_bleed, 6),
        "final_vocal_rms": round(final_vocal_metrics["rms"], 6),
        "demucs_vocal_rms": round(demucs_vocal_metrics["rms"], 6),
        "final_vocal_centroid_hz": round(final_vocal_metrics["centroid_hz"], 6),
        "demucs_vocal_centroid_hz": round(demucs_vocal_metrics["centroid_hz"], 6),
        "openness_centroid_delta_hz": round(openness_delta, 6),
        "difficulty_score": round(float(local_penalty), 6),
    }


def derive_focus_windows(
    loaded: dict[str, tuple[np.ndarray, int]],
    lyrics_segments: list[dict[str, Any]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    if not lyrics_segments:
        return []
    windows = [_segment_window_metrics(loaded, segment) for segment in lyrics_segments]
    windows.sort(key=lambda item: item["difficulty_score"], reverse=True)
    return windows[:limit]


def _activity_mask_from_audio(audio: np.ndarray, sr: int, hop_sec: float = 0.02, smooth_sec: float = 0.08) -> tuple[np.ndarray, float]:
    hop = max(1, int(sr * hop_sec))
    window = max(hop, int(sr * smooth_sec))
    if audio.shape[0] < window:
        audio = np.pad(audio, (0, window - audio.shape[0]))
    env = []
    for start in range(0, audio.shape[0] - window + 1, hop):
        chunk = audio[start : start + window]
        env.append(float(np.sqrt(np.mean(np.square(chunk)) + 1e-12)))
    arr = np.asarray(env if env else [0.0], dtype=np.float64)
    low = float(np.percentile(arr, 35))
    high = float(np.percentile(arr, 97))
    if high <= low + 1e-9:
        norm = np.zeros_like(arr)
    else:
        norm = np.clip((arr - low) / (high - low), 0.0, 1.0)
    return norm.astype(np.float64, copy=False), hop_sec


def _centered_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    size = min(aa.shape[0], bb.shape[0])
    if size < 2:
        return 0.0
    aa = aa[:size] - float(np.mean(aa[:size]))
    bb = bb[:size] - float(np.mean(bb[:size]))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12)
    return float(np.dot(aa, bb) / denom)


def _compute_alignment_indices(reference_len: int, target_len: int, lag_samples: int) -> tuple[int, int, int]:
    if lag_samples >= 0:
        reference_start = lag_samples
        target_start = 0
    else:
        reference_start = 0
        target_start = -lag_samples
    size = min(reference_len - reference_start, target_len - target_start)
    return reference_start, target_start, max(1, size)


def estimate_alignment_lag(
    reference_audio: np.ndarray,
    reference_sr: int,
    target_audio: np.ndarray,
    target_sr: int,
    *,
    max_lag_sec: float = 20.0,
    hop_sec: float = 0.05,
    smooth_sec: float = 0.12,
) -> dict[str, float]:
    resolved_sr = min(reference_sr, target_sr, 12000)
    ref = resample_audio(reference_audio, reference_sr, resolved_sr)
    target = resample_audio(target_audio, target_sr, resolved_sr)
    ref_env, env_hop_sec = _activity_mask_from_audio(ref, resolved_sr, hop_sec=hop_sec, smooth_sec=smooth_sec)
    target_env, _ = _activity_mask_from_audio(target, resolved_sr, hop_sec=hop_sec, smooth_sec=smooth_sec)

    max_lag_frames = max(1, int(round(max_lag_sec / env_hop_sec)))
    min_overlap_frames = max(16, int(round(8.0 / env_hop_sec)))

    best = {"lag_frames": 0, "score": -1.0, "overlap_frames": 0}
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        reference_start, target_start, size = _compute_alignment_indices(ref_env.shape[0], target_env.shape[0], lag)
        if size < min_overlap_frames:
            continue
        ref_slice = ref_env[reference_start : reference_start + size]
        target_slice = target_env[target_start : target_start + size]
        score = _centered_cosine_similarity(ref_slice, target_slice)
        if score > best["score"]:
            best = {"lag_frames": lag, "score": score, "overlap_frames": size}

    return {
        "resolved_sr": float(resolved_sr),
        "hop_sec": float(env_hop_sec),
        "lag_frames": float(best["lag_frames"]),
        "lag_sec": float(best["lag_frames"] * env_hop_sec),
        "alignment_corr": float(best["score"]),
        "overlap_sec": float(best["overlap_frames"] * env_hop_sec),
    }


def align_reference_pair(
    reference_audio: np.ndarray,
    reference_sr: int,
    target_audio: np.ndarray,
    target_sr: int,
    *,
    max_lag_sec: float = 20.0,
    target_align_sr: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, float]]:
    resolved_sr = target_align_sr or min(reference_sr, target_sr)
    ref = resample_audio(reference_audio, reference_sr, resolved_sr)
    target = resample_audio(target_audio, target_sr, resolved_sr)
    lag_info = estimate_alignment_lag(ref, resolved_sr, target, resolved_sr, max_lag_sec=max_lag_sec)
    lag_samples = int(round(lag_info["lag_sec"] * resolved_sr))
    reference_start, target_start, size = _compute_alignment_indices(ref.shape[0], target.shape[0], lag_samples)
    ref_aligned = ref[reference_start : reference_start + size]
    target_aligned = target[target_start : target_start + size]
    info = {
        "lag_sec": float(lag_samples / float(resolved_sr)),
        "lag_samples": float(lag_samples),
        "alignment_corr": float(lag_info["alignment_corr"]),
        "overlap_sec": float(size / float(resolved_sr)),
        "target_start_sec": float(target_start / float(resolved_sr)),
        "reference_start_sec": float(reference_start / float(resolved_sr)),
    }
    return ref_aligned, target_aligned, resolved_sr, info


def reference_match_metrics(
    target_audio: np.ndarray,
    target_sr: int,
    reference_audio: np.ndarray,
    reference_sr: int,
    *,
    vocals_audio: np.ndarray | None = None,
    vocals_sr: int | None = None,
    max_lag_sec: float = 20.0,
) -> dict[str, float | dict[str, float]]:
    reference_aligned, target_aligned, resolved_sr, info = align_reference_pair(
        reference_audio,
        reference_sr,
        target_audio,
        target_sr,
        max_lag_sec=max_lag_sec,
    )
    vv = float(np.dot(reference_aligned, reference_aligned) + 1e-12)
    gain = float(np.dot(target_aligned, reference_aligned) / vv)
    matched_reference = gain * reference_aligned
    residual = target_aligned - matched_reference

    target_metrics = spectral_metrics(target_aligned, resolved_sr)
    reference_metrics = spectral_metrics(matched_reference, resolved_sr)
    residual_metrics = spectral_metrics(residual, resolved_sr)
    reference_rms = float(np.sqrt(np.mean(np.square(matched_reference)) + 1e-12))
    residual_rms = float(np.sqrt(np.mean(np.square(residual)) + 1e-12))

    metrics: dict[str, float | dict[str, float]] = {
        "gain_match_db": float(20.0 * math.log10(abs(gain) + 1e-12)),
        "corr": float(_centered_cosine_similarity(target_aligned, matched_reference)),
        "diff_vs_reference_rms_db": float(20.0 * math.log10((residual_rms + 1e-12) / (reference_rms + 1e-12))),
        "target_centroid_hz": target_metrics["centroid_hz"],
        "reference_centroid_hz": reference_metrics["centroid_hz"],
        "centroid_delta_hz": float(target_metrics["centroid_hz"] - reference_metrics["centroid_hz"]),
        "target_hi4_ratio": target_metrics["hi4_ratio"],
        "reference_hi4_ratio": reference_metrics["hi4_ratio"],
        "hi4_ratio_delta": float(target_metrics["hi4_ratio"] - reference_metrics["hi4_ratio"]),
        "residual_rms": residual_rms,
        "residual_hi4_ratio": residual_metrics["hi4_ratio"],
        "residual_centroid_hz": residual_metrics["centroid_hz"],
        "alignment": info,
    }

    if vocals_audio is not None and vocals_sr is not None:
        vocals_resampled = resample_audio(vocals_audio, vocals_sr, resolved_sr)
        target_start = int(round(info["target_start_sec"] * resolved_sr))
        vocals_aligned = vocals_resampled[target_start : target_start + target_aligned.shape[0]]
        if vocals_aligned.shape[0] < target_aligned.shape[0]:
            vocals_aligned = np.pad(vocals_aligned, (0, target_aligned.shape[0] - vocals_aligned.shape[0]))
        metrics["residual_vocal_projection"] = vocal_bleed_metrics(residual, resolved_sr, vocals_aligned, resolved_sr)

    return metrics


def _reference_window_metrics(
    target_audio: np.ndarray,
    target_sr: int,
    reference_audio: np.ndarray,
    reference_sr: int,
    vocals_audio: np.ndarray,
    vocals_sr: int,
    *,
    start_sec: float,
    end_sec: float,
) -> dict[str, Any]:
    target_window = slice_audio(target_audio, target_sr, start_sec, end_sec)
    reference_window = slice_audio(reference_audio, reference_sr, start_sec, end_sec)
    vocals_window = slice_audio(vocals_audio, vocals_sr, start_sec, end_sec)
    metrics = reference_match_metrics(
        target_window,
        target_sr,
        reference_window,
        reference_sr,
        vocals_audio=vocals_window,
        vocals_sr=vocals_sr,
        max_lag_sec=1.2,
    )
    residual_vocal = float(
        ((metrics.get("residual_vocal_projection") or {}).get("bleed_db_rel_total", -120.0))
    )
    diff_db = float(metrics["diff_vs_reference_rms_db"])
    corr = float(metrics["corr"])
    difficulty = (
        max(0.0, (diff_db + 18.0) * 6.5)
        + max(0.0, (0.93 - corr) * 80.0)
        + max(0.0, abs(float(metrics["centroid_delta_hz"])) - 180.0) / 32.0
        + max(0.0, abs(float(metrics["hi4_ratio_delta"])) - 0.03) * 140.0
        + max(0.0, (residual_vocal + 26.0) * 3.5)
    )
    return {
        "start": round(start_sec, 3),
        "end": round(end_sec, 3),
        "text": "reference-minus mismatch",
        "reference_diff_vs_rms_db": round(diff_db, 6),
        "reference_corr": round(corr, 6),
        "reference_centroid_delta_hz": round(float(metrics["centroid_delta_hz"]), 6),
        "reference_hi4_ratio_delta": round(float(metrics["hi4_ratio_delta"]), 6),
        "reference_residual_vocal_bleed_db": round(residual_vocal, 6),
        "difficulty_score": round(float(difficulty), 6),
    }


def derive_reference_windows(
    final_mix_audio: np.ndarray,
    final_mix_sr: int,
    reference_audio: np.ndarray,
    reference_sr: int,
    vocals_audio: np.ndarray,
    vocals_sr: int,
    *,
    limit: int = 5,
    window_sec: float = 4.0,
    hop_sec: float = 1.5,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    reference_aligned, final_aligned, resolved_sr, info = align_reference_pair(
        reference_audio,
        reference_sr,
        final_mix_audio,
        final_mix_sr,
        max_lag_sec=20.0,
    )
    target_start_sec = float(info["target_start_sec"])
    vocals_resampled = resample_audio(vocals_audio, vocals_sr, resolved_sr)
    target_start_index = int(round(target_start_sec * resolved_sr))
    vocals_aligned = vocals_resampled[target_start_index : target_start_index + final_aligned.shape[0]]
    if vocals_aligned.shape[0] < final_aligned.shape[0]:
        vocals_aligned = np.pad(vocals_aligned, (0, final_aligned.shape[0] - vocals_aligned.shape[0]))

    windows: list[dict[str, Any]] = []
    total_duration = float(final_aligned.shape[0] / resolved_sr)
    max_start = max(0.0, total_duration - window_sec)
    start = 0.0
    while start <= max_start + 1e-6:
        end = min(total_duration, start + window_sec)
        metrics = _reference_window_metrics(
            final_aligned,
            resolved_sr,
            reference_aligned,
            resolved_sr,
            vocals_aligned,
            resolved_sr,
            start_sec=start,
            end_sec=end,
        )
        metrics["start"] = round(metrics["start"] + target_start_sec, 3)
        metrics["end"] = round(metrics["end"] + target_start_sec, 3)
        windows.append(metrics)
        start += hop_sec

    windows.sort(key=lambda item: item["difficulty_score"], reverse=True)
    return windows[:limit], info


def lyrics_drift_metrics(vocals_audio: np.ndarray, vocals_sr: int, lyrics_segments: list[dict[str, Any]]) -> dict[str, float]:
    if not lyrics_segments:
        return {}

    activity, hop_sec = _activity_mask_from_audio(vocals_audio, vocals_sr)
    if activity.size == 0:
        return {}

    word_mask = np.zeros(activity.shape[0], dtype=np.float64)
    loose_mask = np.zeros(activity.shape[0], dtype=np.float64)
    for segment in lyrics_segments:
        windows = segment.get("words") or [segment]
        for item in windows:
            start = max(0.0, float(item["start"]))
            end = max(start + 0.04, float(item["end"]))
            start_index = max(0, int(start / hop_sec))
            end_index = min(activity.shape[0], int(math.ceil(end / hop_sec)))
            if end_index > start_index:
                word_mask[start_index:end_index] = 1.0

            loose_start = max(0.0, start - 0.18)
            loose_end = end + 0.18
            loose_start_index = max(0, int(loose_start / hop_sec))
            loose_end_index = min(activity.shape[0], int(math.ceil(loose_end / hop_sec)))
            if loose_end_index > loose_start_index:
                loose_mask[loose_start_index:loose_end_index] = 1.0

    total_activity = float(np.sum(activity) + 1e-12)
    covered_activity = float(np.sum(activity * word_mask))
    loose_activity = float(np.sum(activity * loose_mask) + 1e-12)
    uncovered_activity = float(np.sum(activity * np.clip(1.0 - loose_mask, 0.0, 1.0)))

    active_binary = (activity >= 0.35).astype(np.float64)
    high_activity_total = float(np.sum(active_binary) + 1e-12)
    high_activity_covered = float(np.sum(active_binary * loose_mask))

    return {
        "coverage_ratio": round(covered_activity / total_activity, 6),
        "focus_ratio": round(covered_activity / loose_activity, 6),
        "uncovered_activity_ratio": round(uncovered_activity / total_activity, 6),
        "high_activity_coverage_ratio": round(high_activity_covered / high_activity_total, 6),
    }


def _merge_focus_windows(
    reference_windows: list[dict[str, Any]],
    lyric_windows: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    candidates = sorted(
        [*reference_windows, *lyric_windows],
        key=lambda item: float(item.get("difficulty_score", 0.0)),
        reverse=True,
    )
    for candidate in candidates:
        start = float(candidate.get("start", 0.0))
        end = float(candidate.get("end", start))
        overlaps_existing = False
        for existing in merged:
            existing_start = float(existing.get("start", 0.0))
            existing_end = float(existing.get("end", existing_start))
            overlap = max(0.0, min(end, existing_end) - max(start, existing_start))
            if overlap > min(end - start, existing_end - existing_start) * 0.45:
                overlaps_existing = True
                break
        if not overlaps_existing:
            merged.append(candidate)
        if len(merged) >= limit:
            break
    return merged


def analyze_job(
    project_root: Path,
    job_id: str,
    focus_windows: list[dict[str, Any]] | None = None,
    reference_instrumental_path: Path | None = None,
    force_reference_target: bool = False,
) -> dict:
    job = read_job(project_root, job_id)
    artifacts = job.get("artifacts", {})

    required = [
        "instrumental_mix",
        "vocals",
        "demucs_vocals",
        "demucs_other",
        "demucs_instrumental_mix",
        "bs_resurrected_instrumental_mix",
        "bs_instrumental_mix",
        "other",
    ]
    missing = [key for key in required if key not in artifacts]
    if missing:
        raise KeyError(f"job is missing required artifacts: {missing}")

    resolved = {key: container_to_host_path(project_root, artifacts[key]) for key in required}
    source_path = container_to_host_path(project_root, job["request"]["input_path"])

    loaded: dict[str, tuple[np.ndarray, int]] = {key: decode_audio(path) for key, path in resolved.items()}
    stereo_loaded: dict[str, tuple[np.ndarray, int]] = {}
    for key in ("instrumental_mix", "demucs_instrumental_mix", "vocals", "demucs_vocals"):
        stereo_loaded[key] = decode_audio_stereo(resolved[key])
    source_audio, source_sr = decode_audio(source_path)
    reference_audio: np.ndarray | None = None
    reference_sr: int | None = None
    resolved_reference_path: Path | None = None
    if reference_instrumental_path:
        resolved_reference_path = reference_instrumental_path.resolve()
        reference_audio, reference_sr = decode_audio(resolved_reference_path)

    report: dict[str, dict] = {
        "job_id": job_id,
        "preset_name": job["request"]["preset_name"],
        "source_path": str(source_path),
        "paths": {key: str(path) for key, path in resolved.items()},
        "stems": {},
    }

    lyrics_path = find_lyrics_json_path(project_root, job)
    lyrics_segments = load_lyrics_segments(lyrics_path) if lyrics_path else []
    if lyrics_path:
        report["lyrics_path"] = str(lyrics_path)
    if resolved_reference_path:
        report["reference_instrumental_path"] = str(resolved_reference_path)

    for key, (audio, sr) in loaded.items():
        stem_report = metrics_block(audio, sr)
        if key in {"instrumental_mix", "demucs_instrumental_mix", "bs_resurrected_instrumental_mix", "bs_instrumental_mix"}:
            vocal_refs = {
                "final_vocals": vocal_bleed_metrics(audio, sr, loaded["vocals"][0], loaded["vocals"][1]),
                "demucs_vocals": vocal_bleed_metrics(audio, sr, loaded["demucs_vocals"][0], loaded["demucs_vocals"][1]),
            }
            stem_report["vocal_bleed"] = vocal_refs
            stem_report["source_similarity"] = source_similarity_metrics(audio, sr, source_audio, source_sr)
            if reference_audio is not None and reference_sr is not None:
                stem_report["reference_match"] = reference_match_metrics(
                    audio,
                    sr,
                    reference_audio,
                    reference_sr,
                    vocals_audio=loaded["vocals"][0],
                    vocals_sr=loaded["vocals"][1],
                )
        report["stems"][key] = stem_report

    final_mix = report["stems"]["instrumental_mix"]
    demucs_mix = report["stems"]["demucs_instrumental_mix"]
    bs_seed = report["stems"]["bs_resurrected_instrumental_mix"]
    bs_raw = report["stems"]["bs_instrumental_mix"]
    final_vocals = report["stems"]["vocals"]
    demucs_vocals = report["stems"]["demucs_vocals"]
    demucs_other = report["stems"]["demucs_other"]
    tandem_other = report["stems"]["other"]

    final_bleed = max(
        final_mix["vocal_bleed"]["final_vocals"]["bleed_db_rel_total"],
        final_mix["vocal_bleed"]["demucs_vocals"]["bleed_db_rel_total"],
    )
    final_center_bleed = max(
        center_vocal_bleed_metrics(
            stereo_loaded["instrumental_mix"][0],
            stereo_loaded["instrumental_mix"][1],
            stereo_loaded["vocals"][0],
            stereo_loaded["vocals"][1],
        )["bleed_db_rel_total"],
        center_vocal_bleed_metrics(
            stereo_loaded["instrumental_mix"][0],
            stereo_loaded["instrumental_mix"][1],
            stereo_loaded["demucs_vocals"][0],
            stereo_loaded["demucs_vocals"][1],
        )["bleed_db_rel_total"],
    )
    demucs_center_bleed = max(
        center_vocal_bleed_metrics(
            stereo_loaded["demucs_instrumental_mix"][0],
            stereo_loaded["demucs_instrumental_mix"][1],
            stereo_loaded["vocals"][0],
            stereo_loaded["vocals"][1],
        )["bleed_db_rel_total"],
        center_vocal_bleed_metrics(
            stereo_loaded["demucs_instrumental_mix"][0],
            stereo_loaded["demucs_instrumental_mix"][1],
            stereo_loaded["demucs_vocals"][0],
            stereo_loaded["demucs_vocals"][1],
        )["bleed_db_rel_total"],
    )
    seed_bleed = max(
        bs_seed["vocal_bleed"]["final_vocals"]["bleed_db_rel_total"],
        bs_seed["vocal_bleed"]["demucs_vocals"]["bleed_db_rel_total"],
    )
    raw_bleed = max(
        bs_raw["vocal_bleed"]["final_vocals"]["bleed_db_rel_total"],
        bs_raw["vocal_bleed"]["demucs_vocals"]["bleed_db_rel_total"],
    )

    other_compare = other_instability_compare(tandem_other, demucs_other)
    warble_ratio = other_compare["ratios"]
    other_bleed = vocal_bleed_metrics(
        loaded["other"][0],
        loaded["other"][1],
        loaded["vocals"][0],
        loaded["vocals"][1],
    )["bleed_db_rel_total"]

    report["summary"] = {
        "final_mix_vocal_bleed_db": final_bleed,
        "final_mix_center_vocal_bleed_db": final_center_bleed,
        "demucs_mix_vocal_bleed_db": max(
            demucs_mix["vocal_bleed"]["final_vocals"]["bleed_db_rel_total"],
            demucs_mix["vocal_bleed"]["demucs_vocals"]["bleed_db_rel_total"],
        ),
        "demucs_mix_center_vocal_bleed_db": demucs_center_bleed,
        "bs_seed_vocal_bleed_db": seed_bleed,
        "bs_raw_vocal_bleed_db": raw_bleed,
        "demucs_mix_dullness_delta": {
            "centroid_hz": float(final_mix["centroid_hz"] - demucs_mix["centroid_hz"]),
            "hi4_ratio": float(final_mix["hi4_ratio"] - demucs_mix["hi4_ratio"]),
        },
        "demucs_vocals_vs_final_vocals": {
            "rms_db_delta": float(20.0 * math.log10((demucs_vocals["rms"] + 1e-12) / (final_vocals["rms"] + 1e-12))),
            "centroid_hz_delta": float(demucs_vocals["centroid_hz"] - final_vocals["centroid_hz"]),
            "hi4_ratio_delta": float(demucs_vocals["hi4_ratio"] - final_vocals["hi4_ratio"]),
        },
        "other_warble_ratio": warble_ratio,
        "other_compare": other_compare,
        "other_vocal_bleed_db": other_bleed,
        "bs_raw_source_similarity": bs_raw["source_similarity"],
    }
    reference_windows: list[dict[str, Any]] = []
    if reference_audio is not None and reference_sr is not None:
        final_reference = final_mix.get("reference_match") or {}
        demucs_reference = demucs_mix.get("reference_match") or {}
        seed_reference = bs_seed.get("reference_match") or {}
        raw_reference = bs_raw.get("reference_match") or {}
        computed_reference_usable = bool(
            max(
                float(final_reference.get("corr", 0.0)),
                float(demucs_reference.get("corr", 0.0)),
            )
            >= 0.62
            and max(
                float((final_reference.get("alignment") or {}).get("alignment_corr", 0.0)),
                float((demucs_reference.get("alignment") or {}).get("alignment_corr", 0.0)),
            )
            >= 0.50
        )
        reference_usable = bool(computed_reference_usable or force_reference_target)
        report["summary"]["reference_compare"] = {
            "usable": reference_usable,
            "computed_usable": computed_reference_usable,
            "forced": force_reference_target,
            "final": final_reference,
            "demucs": demucs_reference,
            "bs_seed": seed_reference,
            "bs_raw": raw_reference,
        }
        reference_windows, reference_alignment = derive_reference_windows(
            loaded["instrumental_mix"][0],
            loaded["instrumental_mix"][1],
            reference_audio,
            reference_sr,
            loaded["vocals"][0],
            loaded["vocals"][1],
            limit=5,
        )
        report["summary"]["reference_alignment"] = reference_alignment

    resolved_focus_windows = focus_windows or _merge_focus_windows(
        reference_windows,
        derive_focus_windows(loaded, lyrics_segments),
        limit=5,
    )
    if resolved_focus_windows:
        local_windows = []
        for window in resolved_focus_windows:
            if window.get("text") == "reference-minus mismatch" or "reference_diff_vs_rms_db" in window:
                local_windows.append(window)
            else:
                local_windows.append(
                    _segment_window_metrics(
                        loaded,
                        {
                            "start": window["start"],
                            "end": window["end"],
                            "text": window.get("text", ""),
                            "words": window.get("words", []),
                        },
                    )
                )
        report["focus_windows"] = resolved_focus_windows
        report["segment_local"] = local_windows
    if reference_windows:
        report["reference_windows"] = reference_windows

    lyrics_drift = lyrics_drift_metrics(loaded["vocals"][0], loaded["vocals"][1], lyrics_segments)
    if lyrics_drift:
        report["summary"]["lyrics_drift"] = lyrics_drift

    issues: list[str] = []
    if final_bleed > -24.0:
        issues.append("final_instrumental_has_strong_vocal_bleed")
    elif final_bleed > -30.0:
        issues.append("final_instrumental_has_moderate_vocal_bleed")

    if final_center_bleed > -22.0:
        issues.append("final_instrumental_has_strong_center_vocal_bleed")
    elif final_center_bleed > -27.0:
        issues.append("final_instrumental_has_moderate_center_vocal_bleed")

    if final_mix["centroid_hz"] - demucs_mix["centroid_hz"] > 350.0 and final_mix["hi4_ratio"] - demucs_mix["hi4_ratio"] > 0.04:
        issues.append("demucs_instrumental_is_dull_vs_final")

    if demucs_vocals["rms"] < final_vocals["rms"] * 0.9 and demucs_vocals["centroid_hz"] < final_vocals["centroid_hz"] - 180.0:
        issues.append("demucs_vocals_are_too_soft_and_dull_for_asr")

    if other_instability_penalty(other_compare) > 12.0:
        issues.append("tandem_other_has_warble_like_instability")

    if bs_raw["source_similarity"]["source_similarity_corr"] > 0.92 and bs_raw["source_similarity"]["diff_vs_source_rms_db"] < -10.0:
        issues.append("bs_raw_instrumental_is_too_close_to_full_track")

    if seed_bleed > final_bleed + 2.0:
        issues.append("bs_seed_is_more_vocal_contaminated_than_final_mix")

    reference_compare = report["summary"].get("reference_compare") or {}
    if reference_compare:
        reference_usable = bool(reference_compare.get("usable"))
        computed_reference_usable = bool(reference_compare.get("computed_usable", reference_usable))
        final_reference = reference_compare.get("final") or {}
        demucs_reference = reference_compare.get("demucs") or {}
        if not computed_reference_usable and not bool(reference_compare.get("forced")):
            issues.append("reference_minus_is_probably_not_same_arrangement_or_master")
        elif not computed_reference_usable and bool(reference_compare.get("forced")):
            issues.append("reference_target_is_forced_despite_low_match_confidence")
        else:
            if float(final_reference.get("diff_vs_reference_rms_db", -24.0)) > -10.0:
                issues.append("final_instrumental_is_far_from_reference_minus")
            if float(final_reference.get("corr", 1.0)) < 0.76:
                issues.append("final_instrumental_has_low_reference_correlation")
            if float(final_reference.get("diff_vs_reference_rms_db", -120.0)) > float(demucs_reference.get("diff_vs_reference_rms_db", -120.0)) + 1.0:
                issues.append("final_instrumental_is_worse_than_demucs_against_reference")

    if lyrics_drift:
        if (
            lyrics_drift["coverage_ratio"] < 0.72
            or lyrics_drift["focus_ratio"] < 0.82
            or lyrics_drift["high_activity_coverage_ratio"] < 0.88
        ):
            issues.append("lyrics_word_timing_has_high_drift")
        elif lyrics_drift["coverage_ratio"] < 0.84 or lyrics_drift["focus_ratio"] < 0.90:
            issues.append("lyrics_word_timing_is_loose")

    report["issues"] = issues
    report["score"] = score_report(report)
    return report


def score_report(report: dict) -> dict[str, object]:
    summary = report["summary"]
    stems = report["stems"]
    final_mix = stems["instrumental_mix"]
    final_source = final_mix["source_similarity"]
    other_compare = summary.get("other_compare") or {
        "compare_mode": "ratio",
        "final": {"rms": 0.0, "centroid_rel_std": 0.0, "hi4_std": 0.0, "flux_per_rms": 0.0, "mod_4_12_ratio": 0.0},
        "demucs": {"rms": 0.0, "centroid_rel_std": 0.0, "hi4_std": 0.0, "flux_per_rms": 0.0, "mod_4_12_ratio": 0.0},
        "ratios": summary["other_warble_ratio"],
    }
    warble = other_compare["ratios"]
    openness = summary["demucs_mix_dullness_delta"]
    vocal_delta = summary["demucs_vocals_vs_final_vocals"]
    local_windows = report.get("segment_local") or []
    lyrics_drift = summary.get("lyrics_drift") or {}
    reference_compare = summary.get("reference_compare") or {}
    reference_usable = bool(reference_compare.get("usable"))
    reference_final = reference_compare.get("final") or {}
    reference_demucs = reference_compare.get("demucs") or {}

    local_bleed_penalty = 0.0
    local_warble_penalty = 0.0
    local_openness_reward = 0.0
    local_reference_penalty = 0.0
    local_reference_reward = 0.0
    if local_windows:
        top_windows = sorted(local_windows, key=lambda item: item["difficulty_score"], reverse=True)[:3]
        stem_like_windows = [item for item in top_windows if "final_mix_vocal_bleed_db" in item]
        if stem_like_windows:
            local_bleed_penalty = sum(
                max(0.0, (item["final_mix_vocal_bleed_db"] + 28.0) * 2.5) for item in stem_like_windows
            ) / len(stem_like_windows)
            local_warble_penalty = sum(
                (
                    max(0.0, (item["other_centroid_rel_std"] - 0.85) * 28.0)
                    + max(0.0, (item["other_mod_4_12_ratio"] - 0.34) * 55.0)
                    + max(0.0, (item["other_flux_per_rms"] - 8.0) * 14.0)
                    if item.get("other_compare_mode") == "absolute"
                    else (
                        max(0.0, (item["warble_centroid_ratio"] - 1.08) * 26.0)
                        + max(0.0, (item["warble_hi4_ratio"] - 1.10) * 32.0)
                        + max(0.0, (item["warble_flux_ratio"] - 1.20) * 28.0)
                    )
                )
                for item in stem_like_windows
            ) / len(stem_like_windows)
            local_openness_reward = sum(
                max(0.0, item["openness_centroid_delta_hz"]) / 180.0 for item in stem_like_windows
            ) / len(stem_like_windows)
        reference_like_windows = [item for item in top_windows if "reference_diff_vs_rms_db" in item]
        if reference_like_windows and reference_usable:
            local_reference_penalty = sum(
                max(0.0, (float(item["reference_diff_vs_rms_db"]) + 18.0) * 5.0)
                + max(0.0, (0.93 - float(item["reference_corr"])) * 65.0)
                + max(0.0, (float(item["reference_residual_vocal_bleed_db"]) + 26.0) * 2.5)
                for item in reference_like_windows
            ) / len(reference_like_windows)
            local_reference_reward = sum(
                min(4.5, max(0.0, -18.0 - float(item["reference_diff_vs_rms_db"])))
                + min(4.0, max(0.0, float(item["reference_corr"]) - 0.90) * 20.0)
                for item in reference_like_windows
            ) / len(reference_like_windows)

    penalties = {
        "vocal_bleed": max(0.0, (summary["final_mix_vocal_bleed_db"] + 30.0) * 4.0),
        "center_vocal_bleed": max(0.0, (summary.get("final_mix_center_vocal_bleed_db", -30.0) + 28.0) * 5.5),
        "bleed_vs_demucs": max(
            0.0,
            (summary["final_mix_vocal_bleed_db"] - summary["demucs_mix_vocal_bleed_db"] - 2.0) * 3.5,
        ),
        "center_bleed_vs_demucs": max(
            0.0,
            (
                float(summary.get("final_mix_center_vocal_bleed_db", -120.0))
                - float(summary.get("demucs_mix_center_vocal_bleed_db", -120.0))
                - 1.2
            )
            * 4.5,
        ),
        "warble_centroid": (
            max(0.0, (other_compare["final"]["centroid_rel_std"] - 0.85) * 40.0)
            if other_compare["compare_mode"] == "absolute"
            else max(0.0, (warble["centroid_std_ratio"] - 1.08) * 48.0)
        ),
        "warble_hi4": (
            max(0.0, (other_compare["final"]["hi4_std"] - 0.24) * 75.0)
            if other_compare["compare_mode"] == "absolute"
            else max(0.0, (warble["hi4_std_ratio"] - 1.10) * 62.0)
        ),
        "warble_flux": (
            max(0.0, (other_compare["final"]["flux_per_rms"] - 8.0) * 18.0)
            + max(0.0, (other_compare["final"]["mod_4_12_ratio"] - 0.34) * 80.0)
            if other_compare["compare_mode"] == "absolute"
            else max(0.0, (warble["flux_ratio"] - 1.20) * 48.0)
            + max(0.0, (warble["mod_4_12_ratio"] - 1.12) * 48.0)
        ),
        "source_corr": max(0.0, (final_source["source_similarity_corr"] - 0.92) * 120.0),
        "source_diff": max(0.0, (-10.0 - final_source["diff_vs_source_rms_db"]) * 2.0),
        "local_bleed": local_bleed_penalty,
        "local_warble": local_warble_penalty,
        "lyrics_drift_coverage": max(0.0, (0.88 - float(lyrics_drift.get("coverage_ratio", 0.88))) * 90.0),
        "lyrics_drift_focus": max(0.0, (0.92 - float(lyrics_drift.get("focus_ratio", 0.92))) * 80.0),
        "lyrics_drift_uncovered": max(
            0.0,
            (float(lyrics_drift.get("uncovered_activity_ratio", 0.08)) - 0.08) * 110.0,
        ),
        "lyrics_drift_high_activity": max(
            0.0,
            (0.94 - float(lyrics_drift.get("high_activity_coverage_ratio", 0.94))) * 120.0,
        ),
        "reference_diff": 0.0,
        "reference_corr": 0.0,
        "reference_vocal_residual": 0.0,
        "reference_vs_demucs": 0.0,
        "local_reference": local_reference_penalty,
    }
    if reference_usable:
        penalties["reference_diff"] = max(
            0.0,
            (float(reference_final.get("diff_vs_reference_rms_db", -18.0)) + 18.0) * 6.0,
        )
        penalties["reference_corr"] = max(
            0.0,
            (0.90 - float(reference_final.get("corr", 0.90))) * 85.0,
        )
        penalties["reference_vocal_residual"] = max(
            0.0,
            (
                float(
                    ((reference_final.get("residual_vocal_projection") or {}).get("bleed_db_rel_total", -28.0))
                )
                + 28.0
            )
            * 4.0,
        )
        penalties["reference_vs_demucs"] = max(
            0.0,
            (
                float(reference_final.get("diff_vs_reference_rms_db", -120.0))
                - float(reference_demucs.get("diff_vs_reference_rms_db", -120.0))
                - 0.8
            )
            * 5.0,
        )
    rewards = {
        "openness_centroid": min(12.0, max(0.0, openness["centroid_hz"]) / 90.0),
        "openness_hi4": min(10.0, max(0.0, openness["hi4_ratio"]) * 120.0),
        "vocal_clarity_rms": min(5.0, max(0.0, -vocal_delta["rms_db_delta"]) * 2.0),
        "vocal_clarity_centroid": min(4.0, max(0.0, -vocal_delta["centroid_hz_delta"]) / 120.0),
        "local_openness": min(6.0, local_openness_reward),
        "other_usefulness": other_usefulness_reward(other_compare, float(summary.get("other_vocal_bleed_db", -120.0))),
        "center_bleed_improvement": min(
            5.0,
            max(
                0.0,
                float(summary.get("demucs_mix_center_vocal_bleed_db", -120.0))
                - float(summary.get("final_mix_center_vocal_bleed_db", -120.0)),
            )
            * 1.5,
        ),
        "reference_closeness": 0.0,
        "reference_local": local_reference_reward,
    }
    if reference_usable:
        rewards["reference_closeness"] = (
            min(10.0, max(0.0, -18.0 - float(reference_final.get("diff_vs_reference_rms_db", -18.0))))
            + min(8.0, max(0.0, float(reference_final.get("corr", 0.0)) - 0.88) * 20.0)
        )

    total_penalty = float(sum(penalties.values()))
    total_reward = float(sum(rewards.values()))
    score = 100.0 + total_reward - total_penalty

    return {
        "composite_score": round(score, 6),
        "total_penalty": round(total_penalty, 6),
        "total_reward": round(total_reward, 6),
        "penalties": {key: round(value, 6) for key, value in penalties.items()},
        "rewards": {key: round(value, 6) for key, value in rewards.items()},
    }


def write_report(project_root: Path, report: dict) -> tuple[Path, Path]:
    out_dir = project_root / ".tmp" / "job-diagnostics" / report["job_id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "diagnostics.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = report["summary"]
    score = report.get("score", {})
    lines = [
        "# Job Diagnostics",
        "",
        f"- job: `{report['job_id']}`",
        f"- preset: `{report['preset_name']}`",
        f"- source: `{report['source_path']}`",
        f"- issues: {', '.join(report['issues']) or 'none'}",
        f"- composite score: `{score.get('composite_score', 0.0):.3f}`",
        "",
        "## Summary",
        "",
        f"- final vocal bleed: `{summary['final_mix_vocal_bleed_db']:.2f} dB`",
        f"- final center vocal bleed: `{summary['final_mix_center_vocal_bleed_db']:.2f} dB`",
        f"- demucs vocal bleed: `{summary['demucs_mix_vocal_bleed_db']:.2f} dB`",
        f"- demucs center vocal bleed: `{summary['demucs_mix_center_vocal_bleed_db']:.2f} dB`",
        f"- bs seed vocal bleed: `{summary['bs_seed_vocal_bleed_db']:.2f} dB`",
        f"- bs raw vocal bleed: `{summary['bs_raw_vocal_bleed_db']:.2f} dB`",
        f"- demucs mix dullness delta centroid: `{summary['demucs_mix_dullness_delta']['centroid_hz']:.1f} Hz`",
        f"- demucs mix dullness delta hi4: `{summary['demucs_mix_dullness_delta']['hi4_ratio']:.4f}`",
        f"- demucs vocals rms delta: `{summary['demucs_vocals_vs_final_vocals']['rms_db_delta']:.2f} dB`",
        f"- demucs vocals centroid delta: `{summary['demucs_vocals_vs_final_vocals']['centroid_hz_delta']:.1f} Hz`",
        f"- other compare mode: `{summary['other_compare']['compare_mode']}`",
        f"- other rms: `{summary['other_compare']['final']['rms']:.4f}` vs demucs `{summary['other_compare']['demucs']['rms']:.4f}`",
        f"- other warble centroid ratio: `{summary['other_warble_ratio']['centroid_std_ratio']:.3f}`",
        f"- other warble hi4 ratio: `{summary['other_warble_ratio']['hi4_std_ratio']:.3f}`",
        f"- bs raw source corr: `{summary['bs_raw_source_similarity']['source_similarity_corr']:.4f}`",
        f"- bs raw diff vs source rms: `{summary['bs_raw_source_similarity']['diff_vs_source_rms_db']:.2f} dB`",
        "",
    ]
    if summary.get("reference_compare"):
        reference_compare = summary["reference_compare"]
        final_reference = reference_compare.get("final") or {}
        demucs_reference = reference_compare.get("demucs") or {}
        lines.extend(
            [
                "## Reference Minus",
                "",
                f"- usable: `{str(reference_compare.get('usable', False)).lower()}`",
                f"- computed usable: `{str(reference_compare.get('computed_usable', reference_compare.get('usable', False))).lower()}`",
                f"- forced target: `{str(reference_compare.get('forced', False)).lower()}`",
                f"- final diff vs ref rms: `{float(final_reference.get('diff_vs_reference_rms_db', 0.0)):.2f} dB`",
                f"- final corr: `{float(final_reference.get('corr', 0.0)):.3f}`",
                f"- final residual vocal bleed: `{float(((final_reference.get('residual_vocal_projection') or {}).get('bleed_db_rel_total', 0.0))):.2f} dB`",
                f"- demucs diff vs ref rms: `{float(demucs_reference.get('diff_vs_reference_rms_db', 0.0)):.2f} dB`",
                f"- demucs corr: `{float(demucs_reference.get('corr', 0.0)):.3f}`",
                f"- alignment lag: `{float((summary.get('reference_alignment') or {}).get('lag_sec', 0.0)):.3f} s`",
                f"- alignment corr: `{float((summary.get('reference_alignment') or {}).get('alignment_corr', 0.0)):.3f}`",
                "",
            ]
        )
    if summary.get("lyrics_drift"):
        lines.extend(
            [
                "## Lyrics Drift",
                "",
                f"- coverage ratio: `{summary['lyrics_drift']['coverage_ratio']:.3f}`",
                f"- focus ratio: `{summary['lyrics_drift']['focus_ratio']:.3f}`",
                f"- uncovered activity ratio: `{summary['lyrics_drift']['uncovered_activity_ratio']:.3f}`",
                f"- high activity coverage ratio: `{summary['lyrics_drift']['high_activity_coverage_ratio']:.3f}`",
                "",
            ]
        )
    if report.get("segment_local"):
        lines.extend(["## Worst Windows", ""])
        for index, window in enumerate(report["segment_local"][:3], start=1):
            if "reference_diff_vs_rms_db" in window:
                lines.append(
                    f"- window {index}: `{window['start']:.2f}-{window['end']:.2f}` | ref diff `{window['reference_diff_vs_rms_db']:.2f} dB` | "
                    f"ref corr `{window['reference_corr']:.3f}` | residual vocal `{window['reference_residual_vocal_bleed_db']:.2f} dB` | text `{window['text']}`"
                )
            else:
                lines.append(
                    f"- window {index}: `{window['start']:.2f}-{window['end']:.2f}` | bleed `{window['final_mix_vocal_bleed_db']:.2f} dB` | "
                    f"warble `{window['warble_centroid_ratio']:.3f}/{window['warble_hi4_ratio']:.3f}` | text `{window['text']}`"
                )
        lines.append("")
    lines.extend(
        [
            "## Score",
            "",
            f"- total reward: `{score.get('total_reward', 0.0):.3f}`",
            f"- total penalty: `{score.get('total_penalty', 0.0):.3f}`",
        ]
    )
    for key, value in (score.get("rewards") or {}).items():
        lines.append(f"- reward `{key}`: `{value:.3f}`")
    for key, value in (score.get("penalties") or {}).items():
        lines.append(f"- penalty `{key}`: `{value:.3f}`")
    lines.extend(
        [
            "",
        ]
    )

    md_path = out_dir / "diagnostics.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--job", required=True)
    parser.add_argument("--reference-instrumental", type=Path, default=None)
    parser.add_argument("--force-reference-target", action="store_true")
    args = parser.parse_args()

    report = analyze_job(
        args.root,
        args.job,
        reference_instrumental_path=args.reference_instrumental,
        force_reference_target=args.force_reference_target,
    )
    json_path, md_path = write_report(args.root, report)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
