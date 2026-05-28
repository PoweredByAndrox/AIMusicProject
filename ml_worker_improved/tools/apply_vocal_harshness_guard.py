from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402

import librosa
import numpy as np
import soundfile as sf


EPSILON = 1e-12
TARGET_HARSHNESS_LABELS = frozenset(
    {
        "high_noise_or_robotic_artifact",
        "high_band_unsupported_leakage",
        "unsupported_anchor_leakage",
        "excessive_reverb_tail_smear",
    }
)
VOCAL_LOSS_LABELS = frozenset(
    {
        "vocal_dropout",
        "vocal_tail_loss",
        "local_vocal_body_dip",
        "backing_vocal_loss_risk",
    }
)


@dataclass(frozen=True)
class HarshnessGuardTuning:
    high_band_start_hz: float = 6200.0
    high_band_end_hz: float = 15000.0
    anchor_tolerance: float = 1.12
    min_high_band_floor: float = 0.58
    base_strength: float = 0.52
    max_strength: float = 0.78
    vocal_loss_strength_scale: float = 0.42
    min_segment_severity: float = 0.55
    high_noise_flatness_threshold: float = 0.16
    fade_sec: float = 0.12
    n_fft: int = 4096
    hop_length: int = 512


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True)
    if audio.size == 0:
        raise RuntimeError(f"Audio is empty: {path}")
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def match_length(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = min(int(left.shape[0]), int(right.shape[0]))
    if length <= 0:
        raise RuntimeError("Audio inputs have no overlapping frames")
    return left[:length], right[:length]


def rms(audio: np.ndarray) -> float:
    data = np.asarray(audio, dtype=np.float32)
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data, dtype=np.float32))))


def peak(audio: np.ndarray) -> float:
    data = np.asarray(audio, dtype=np.float32)
    if data.size == 0:
        return 0.0
    return float(np.max(np.abs(data)))


def band_rms(audio: np.ndarray, sample_rate: int, *, low_hz: float, high_hz: float) -> float:
    mono = np.asarray(audio, dtype=np.float32).mean(axis=1)
    if mono.size == 0:
        return 0.0
    spectrum = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(mono.shape[0], d=1.0 / float(sample_rate))
    mask = (freqs >= float(low_hz)) & (freqs <= float(high_hz))
    if not np.any(mask):
        return 0.0
    band = np.fft.irfft(spectrum * mask.astype(np.float32), n=mono.shape[0])
    return rms(band[:, np.newaxis])


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def candidate_by_profile(adaptive_manifest: dict[str, Any], profile_name: str) -> dict[str, Any]:
    candidates = adaptive_manifest.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Adaptive manifest does not contain candidates")
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("profile_name") == profile_name:
            return candidate
    raise ValueError(f"Candidate profile not found: {profile_name}")


def selected_profile_name(adaptive_manifest: dict[str, Any]) -> str:
    selected = adaptive_manifest.get("selected_profile")
    if not isinstance(selected, str) or not selected:
        raise ValueError("Adaptive manifest does not contain selected_profile")
    return selected


def safe_artifact_name(value: str) -> str:
    safe_name = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)
    return safe_name.strip("._") or "profile"


def segment_labels(segment: dict[str, Any]) -> set[str]:
    labels = segment.get("labels")
    if not isinstance(labels, list):
        return set()
    return {str(label) for label in labels if isinstance(label, str)}


def segment_metrics(segment: dict[str, Any]) -> dict[str, Any]:
    metrics = segment.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def window_strength(labels: set[str], metrics: dict[str, Any], severity: float, tuning: HarshnessGuardTuning) -> float:
    strength = float(tuning.base_strength)
    if "high_noise_or_robotic_artifact" in labels:
        strength += 0.14
    if "high_band_unsupported_leakage" in labels:
        strength += 0.10
    if "unsupported_anchor_leakage" in labels:
        strength += 0.08
    if "excessive_reverb_tail_smear" in labels:
        strength += 0.12

    flatness = metrics.get("high_noise_flatness_6000_16000")
    if isinstance(flatness, (int, float)) and float(flatness) > float(tuning.high_noise_flatness_threshold):
        strength += min(0.14, (float(flatness) - float(tuning.high_noise_flatness_threshold)) * 0.22)
    if severity > 2.0:
        strength += min(0.10, (float(severity) - 2.0) * 0.025)
    if labels & VOCAL_LOSS_LABELS:
        strength *= float(tuning.vocal_loss_strength_scale)
    return float(np.clip(strength, 0.0, float(tuning.max_strength)))


def build_window_decisions(base_candidate: dict[str, Any], tuning: HarshnessGuardTuning) -> list[dict[str, Any]]:
    report = base_candidate.get("segment_issue_report")
    if not isinstance(report, dict):
        return []
    segments = report.get("segments")
    if not isinstance(segments, list):
        return []

    decisions: list[dict[str, Any]] = []
    for index, raw_segment in enumerate(segments):
        if not isinstance(raw_segment, dict):
            continue
        labels = segment_labels(raw_segment)
        metrics = segment_metrics(raw_segment)
        severity = float(raw_segment.get("severity_score", 0.0))
        flatness = metrics.get("high_noise_flatness_6000_16000")
        has_flatness_evidence = (
            isinstance(flatness, (int, float))
            and float(flatness) > float(tuning.high_noise_flatness_threshold)
            and {"presence_loss", "air_detail_loss"} & labels
        )
        if not (labels & TARGET_HARSHNESS_LABELS or has_flatness_evidence):
            continue
        if severity < float(tuning.min_segment_severity):
            continue

        strength = window_strength(labels, metrics, severity, tuning)
        if strength <= 0.0:
            continue
        decisions.append(
            {
                "segment_index": int(index),
                "start_sec": float(raw_segment["start_sec"]),
                "end_sec": float(raw_segment["end_sec"]),
                "severity_score": round(float(severity), 6),
                "labels": sorted(labels),
                "strength": round(float(strength), 6),
                "high_noise_flatness_6000_16000": (
                    round(float(flatness), 6) if isinstance(flatness, (int, float)) else None
                ),
            }
        )
    return decisions


def smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    width = max(float(edge1) - float(edge0), 1e-8)
    t = np.clip((values - float(edge0)) / width, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_frame_route(
    frame_count: int,
    *,
    sample_rate: int,
    hop_length: int,
    decisions: list[dict[str, Any]],
    fade_sec: float,
) -> np.ndarray:
    frame_times = librosa.frames_to_time(np.arange(frame_count), sr=sample_rate, hop_length=hop_length)
    route = np.zeros(frame_count, dtype=np.float32)
    fade = max(0.001, float(fade_sec))
    for decision in decisions:
        start = float(decision["start_sec"])
        end = float(decision["end_sec"])
        strength = float(decision["strength"])
        left = smoothstep(start - fade, start, frame_times)
        right = 1.0 - smoothstep(end, end + fade, frame_times)
        route = np.maximum(route, np.clip(left * right * strength, 0.0, strength).astype(np.float32))
    return np.clip(route, 0.0, 1.0).astype(np.float32)


def apply_harshness_guard(
    base_audio: np.ndarray,
    anchor_audio: np.ndarray,
    *,
    sample_rate: int,
    decisions: list[dict[str, Any]],
    tuning: HarshnessGuardTuning,
) -> tuple[np.ndarray, dict[str, Any]]:
    base, anchor = match_length(np.asarray(base_audio, dtype=np.float32), np.asarray(anchor_audio, dtype=np.float32))
    if base.shape[1] != anchor.shape[1]:
        raise ValueError("Base and anchor channel counts must match")

    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=int(tuning.n_fft)).astype(np.float32)
    high_band_mask = (
        (freqs >= float(tuning.high_band_start_hz))
        & (freqs <= min(float(tuning.high_band_end_hz), float(sample_rate) / 2.0))
    )
    if not np.any(high_band_mask):
        return base.copy(), {"window_count": len(decisions), "route_mean": 0.0, "high_band_bin_count": 0}

    output_channels: list[np.ndarray] = []
    route_means: list[float] = []
    for channel_index in range(base.shape[1]):
        base_channel = base[:, channel_index]
        anchor_channel = anchor[:, channel_index]
        base_stft = librosa.stft(base_channel, n_fft=int(tuning.n_fft), hop_length=int(tuning.hop_length))
        anchor_stft = librosa.stft(anchor_channel, n_fft=int(tuning.n_fft), hop_length=int(tuning.hop_length))
        base_mag = np.abs(base_stft).astype(np.float32)
        anchor_mag = np.abs(anchor_stft).astype(np.float32)
        route = build_frame_route(
            base_stft.shape[1],
            sample_rate=sample_rate,
            hop_length=int(tuning.hop_length),
            decisions=decisions,
            fade_sec=float(tuning.fade_sec),
        )
        route_2d = route[np.newaxis, :] * high_band_mask[:, np.newaxis].astype(np.float32)

        anchor_ceiling = anchor_mag * float(tuning.anchor_tolerance)
        guarded_mag = np.maximum(
            np.minimum(base_mag, anchor_ceiling),
            base_mag * float(tuning.min_high_band_floor),
        ).astype(np.float32)
        target_mag = base_mag.copy()
        target_mag[high_band_mask, :] = guarded_mag[high_band_mask, :]
        mixed_mag = (base_mag * (1.0 - route_2d) + target_mag * route_2d).astype(np.float32)

        phase = np.exp(1j * np.angle(base_stft))
        guarded_stft = mixed_mag * phase
        output_channel = librosa.istft(
            guarded_stft,
            hop_length=int(tuning.hop_length),
            length=base_channel.shape[0],
        ).astype(np.float32)
        output_channels.append(output_channel)
        route_means.append(float(np.mean(route)))

    guarded = np.stack(output_channels, axis=1).astype(np.float32, copy=False)
    guarded = np.clip(guarded, -0.999, 0.999).astype(np.float32)
    debug = {
        "window_count": int(len(decisions)),
        "route_mean": round(float(np.mean(route_means)) if route_means else 0.0, 6),
        "route_max": round(float(max((float(d["strength"]) for d in decisions), default=0.0)), 6),
        "high_band_bin_count": int(np.sum(high_band_mask)),
        "base_rms": round(rms(base), 6),
        "guarded_rms": round(rms(guarded), 6),
        "base_peak": round(peak(base), 6),
        "guarded_peak": round(peak(guarded), 6),
        "base_high_band_rms": round(
            band_rms(base, sample_rate, low_hz=float(tuning.high_band_start_hz), high_hz=float(tuning.high_band_end_hz)),
            6,
        ),
        "guarded_high_band_rms": round(
            band_rms(
                guarded,
                sample_rate,
                low_hz=float(tuning.high_band_start_hz),
                high_hz=float(tuning.high_band_end_hz),
            ),
            6,
        ),
    }
    return guarded, debug


def write_listen_window_clips(
    *,
    listen_dir: Path,
    raw_audio: np.ndarray | None,
    base_audio: np.ndarray,
    guarded_audio: np.ndarray,
    sample_rate: int,
    decisions: list[dict[str, Any]],
    max_windows: int,
) -> list[dict[str, Any]]:
    clips_dir = listen_dir / "harshness_guard_windows"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_records: list[dict[str, Any]] = []
    ranked_decisions = sorted(decisions, key=lambda item: float(item.get("severity_score", 0.0)), reverse=True)
    for rank, decision in enumerate(ranked_decisions[: max(0, int(max_windows))], start=1):
        start_sec = float(decision["start_sec"])
        end_sec = float(decision["end_sec"])
        start = max(0, int(round(start_sec * sample_rate)))
        end = min(int(base_audio.shape[0]), int(round(end_sec * sample_rate)))
        if end <= start:
            continue
        label_text = "_".join(str(label) for label in decision.get("labels", [])[:2]) or "harshness"
        prefix = f"window_{rank:02d}_{label_text}_{start_sec:.2f}_{end_sec:.2f}"
        base_path = clips_dir / f"{prefix}_base.flac"
        guarded_path = clips_dir / f"{prefix}_guarded.flac"
        sf.write(str(base_path), base_audio[start:end], sample_rate, subtype="PCM_24")
        sf.write(str(guarded_path), guarded_audio[start:end], sample_rate, subtype="PCM_24")
        record: dict[str, Any] = {
            "rank": rank,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "labels": decision.get("labels", []),
            "strength": decision.get("strength"),
            "base_clip": str(base_path),
            "guarded_clip": str(guarded_path),
        }
        if raw_audio is not None:
            raw_path = clips_dir / f"{prefix}_raw.flac"
            sf.write(str(raw_path), raw_audio[start:end], sample_rate, subtype="PCM_24")
            record["raw_clip"] = str(raw_path)
        clip_records.append(record)
    write_json(clips_dir / "harshness_guard_window_clips.json", {"clips": clip_records})
    return clip_records


def run_harshness_guard(
    *,
    adaptive_manifest_path: Path,
    output_dir: Path,
    listen_dir: Path | None,
    base_profile_name: str | None,
    anchor_profile_name: str,
    tuning: HarshnessGuardTuning,
    max_listen_windows: int,
) -> dict[str, Any]:
    start_wall = time.time()
    adaptive_manifest = load_json(adaptive_manifest_path)
    base_name = base_profile_name or selected_profile_name(adaptive_manifest)
    base_candidate = candidate_by_profile(adaptive_manifest, base_name)
    anchor_candidate = candidate_by_profile(adaptive_manifest, anchor_profile_name)

    base_path = Path(str(base_candidate["vocals_path"]))
    anchor_path = Path(str(anchor_candidate["vocals_path"]))
    source_audio_path = Path(str(adaptive_manifest["source_audio_path"]))
    base_audio, base_sr = read_audio(base_path)
    anchor_audio, anchor_sr = read_audio(anchor_path)
    if base_sr != anchor_sr:
        raise ValueError("Base and anchor sample rates must match")

    decisions = build_window_decisions(base_candidate, tuning)
    guarded_audio, debug = apply_harshness_guard(
        base_audio,
        anchor_audio,
        sample_rate=base_sr,
        decisions=decisions,
        tuning=tuning,
    )
    base_audio, guarded_audio = match_length(base_audio, guarded_audio)

    output_dir.mkdir(parents=True, exist_ok=True)
    vocals_path = output_dir / "vocals.flac"
    sf.write(str(vocals_path), guarded_audio, base_sr, subtype="PCM_24")

    raw_audio: np.ndarray | None = None
    try:
        raw_audio, raw_sr = read_audio(source_audio_path)
        if raw_sr == base_sr:
            raw_audio, _ = match_length(raw_audio, guarded_audio)
        else:
            raw_audio = None
    except Exception:
        raw_audio = None

    manifest = {
        "schema_version": 1,
        "mode": "vocal_harshness_guard",
        "source_adaptive_manifest_path": str(adaptive_manifest_path),
        "source_audio_path": str(source_audio_path),
        "source_audio_sha256": adaptive_manifest.get("source_audio_sha256"),
        "window_sec": adaptive_manifest.get("window_sec"),
        "base_profile": base_name,
        "anchor_profile": anchor_profile_name,
        "base_vocals_path": str(base_path),
        "anchor_vocals_path": str(anchor_path),
        "base_vocals_sha256": sha256_file(base_path),
        "anchor_vocals_sha256": sha256_file(anchor_path),
        "output_vocals_path": str(vocals_path),
        "output_vocals_sha256": sha256_file(vocals_path),
        "tuning": tuning.__dict__,
        "window_decisions": decisions,
        "debug": debug,
        "audio": {
            "sample_rate": int(base_sr),
            "frames": int(guarded_audio.shape[0]),
            "duration_sec": round(float(guarded_audio.shape[0]) / float(base_sr), 6),
        },
        "artifacts": {"vocals_flac": str(vocals_path)},
        "elapsed_sec": round(time.time() - start_wall, 3),
    }
    manifest_path = output_dir / "vocal_harshness_guard_manifest.json"
    write_json(manifest_path, manifest)

    if listen_dir is not None:
        listen_dir.mkdir(parents=True, exist_ok=True)
        if raw_audio is not None:
            raw_ref_path = listen_dir / "00_raw_ref.flac"
            sf.write(str(raw_ref_path), raw_audio, base_sr, subtype="PCM_24")
        shutil.copyfile(base_path, listen_dir / "01_base_selected_vocals.flac")
        shutil.copyfile(vocals_path, listen_dir / "02_harshness_guard_vocals.flac")
        anchor_listen_filename = f"03_anchor_{safe_artifact_name(anchor_profile_name)}.flac"
        shutil.copyfile(anchor_path, listen_dir / anchor_listen_filename)
        clip_records = write_listen_window_clips(
            listen_dir=listen_dir,
            raw_audio=raw_audio,
            base_audio=base_audio,
            guarded_audio=guarded_audio,
            sample_rate=base_sr,
            decisions=decisions,
            max_windows=max_listen_windows,
        )
        write_json(
            listen_dir / "candidate_map.json",
            {
                "01_base_selected_vocals.flac": str(base_path),
                "02_harshness_guard_vocals.flac": str(vocals_path),
                anchor_listen_filename: str(anchor_path),
                "harshness_guard_windows": str(listen_dir / "harshness_guard_windows"),
            },
        )
        (listen_dir / "00_README.txt").write_text(
            (
                "Listen only here.\n"
                "01 = base selected vocals.\n"
                "02 = high-band harshness guard.\n"
                f"03 = {anchor_profile_name} anchor reference.\n"
            ),
            encoding="utf-8",
        )
        manifest["listen_dir"] = str(listen_dir)
        manifest["listen_window_clips"] = clip_records
        write_json(manifest_path, manifest)

    return manifest


def main() -> None:
    require_docker_runtime(
        tool_name="apply_vocal_harshness_guard.py",
        entrypoint_path="/app/tools/apply_vocal_harshness_guard.py",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("adaptive_manifest_path")
    parser.add_argument("output_dir")
    parser.add_argument("--listen-dir")
    parser.add_argument("--base-profile")
    parser.add_argument("--anchor-profile", default="bs_artifact_clean")
    parser.add_argument("--high-band-start-hz", type=float, default=6200.0)
    parser.add_argument("--high-band-end-hz", type=float, default=15000.0)
    parser.add_argument("--anchor-tolerance", type=float, default=1.12)
    parser.add_argument("--min-high-band-floor", type=float, default=0.58)
    parser.add_argument("--base-strength", type=float, default=0.52)
    parser.add_argument("--max-strength", type=float, default=0.78)
    parser.add_argument("--vocal-loss-strength-scale", type=float, default=0.42)
    parser.add_argument("--min-segment-severity", type=float, default=0.55)
    parser.add_argument("--high-noise-flatness-threshold", type=float, default=0.16)
    parser.add_argument("--fade-sec", type=float, default=0.12)
    parser.add_argument("--max-listen-windows", type=int, default=8)
    args = parser.parse_args()

    manifest = run_harshness_guard(
        adaptive_manifest_path=Path(args.adaptive_manifest_path),
        output_dir=Path(args.output_dir),
        listen_dir=Path(args.listen_dir) if args.listen_dir else None,
        base_profile_name=args.base_profile,
        anchor_profile_name=str(args.anchor_profile),
        tuning=HarshnessGuardTuning(
            high_band_start_hz=float(args.high_band_start_hz),
            high_band_end_hz=float(args.high_band_end_hz),
            anchor_tolerance=float(args.anchor_tolerance),
            min_high_band_floor=float(args.min_high_band_floor),
            base_strength=float(args.base_strength),
            max_strength=float(args.max_strength),
            vocal_loss_strength_scale=float(args.vocal_loss_strength_scale),
            min_segment_severity=float(args.min_segment_severity),
            high_noise_flatness_threshold=float(args.high_noise_flatness_threshold),
            fade_sec=float(args.fade_sec),
        ),
        max_listen_windows=int(args.max_listen_windows),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(Path(args.output_dir) / "vocal_harshness_guard_manifest.json"),
                "output_vocals_path": manifest["output_vocals_path"],
                "window_count": len(manifest["window_decisions"]),
                "debug": manifest["debug"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
