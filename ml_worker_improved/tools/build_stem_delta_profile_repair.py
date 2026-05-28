from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

TOOLS_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402
import run_ground_truth_quality_gate as quality_gate  # noqa: E402


MODEL_ID = "stem_delta_profile_repair"
EPSILON = 1e-9

PROFILE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("sub", 20.0, 80.0),
    ("kick_body", 80.0, 180.0),
    ("punch", 180.0, 420.0),
    ("snare_body", 420.0, 1100.0),
    ("presence", 1100.0, 3200.0),
    ("attack", 3200.0, 7000.0),
    ("air", 7000.0, 14000.0),
    ("ultra_air", 14000.0, 22050.0),
)


@dataclass(frozen=True)
class StemDeltaProfileConfig:
    n_fft: int = 4096
    hop_length: int = 1024
    max_abs_gain_db: float = 7.0
    profile_mix: float = 1.0
    min_error_improvement_db: float = 0.05
    peak_ceiling: float = 0.98
    max_report_windows: int = 20
    use_percussive_gate: bool = True
    allow_negative_gains: bool = False
    use_source_percussive_reservoir: bool = False
    source_reservoir_mix: float = 0.22
    source_reservoir_min_hz: float = 20.0
    source_reservoir_max_hz: float = 12000.0
    source_reservoir_min_source_percussive_confidence: float = 0.68
    source_reservoir_min_candidate_percussive_confidence: float = 0.32


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_manifest_path(manifest_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return manifest_path.parent / candidate


def resolve_stem_path(manifest_path: Path, stem_name: str) -> Path:
    payload = read_json(manifest_path)
    stems = payload.get("stems")
    if not isinstance(stems, dict):
        raise RuntimeError(f"Manifest has no stems object: {manifest_path}")
    stem = stems.get(stem_name)
    if not isinstance(stem, dict) or not stem.get("path"):
        raise RuntimeError(f"Manifest has no requested stem path: stem={stem_name} manifest={manifest_path}")
    stem_path = resolve_manifest_path(manifest_path, str(stem["path"]))
    if not stem_path.exists():
        raise FileNotFoundError(stem_path)
    return stem_path


def resolve_ground_truth_mix_path(ground_truth_manifest: Path) -> Path:
    payload = read_json(ground_truth_manifest)
    mix = payload.get("mix")
    if isinstance(mix, dict) and mix.get("path"):
        return resolve_manifest_path(ground_truth_manifest, str(mix["path"]))
    raise RuntimeError(f"Ground-truth manifest has no mix.path: {ground_truth_manifest}")


def read_stereo_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    return audio.astype(np.float32, copy=False), int(sample_rate)


def fit_length(audio: np.ndarray, frames: int) -> np.ndarray:
    if audio.shape[0] == frames:
        return audio
    if audio.shape[0] > frames:
        return audio[:frames]
    pad = np.zeros((frames - audio.shape[0], audio.shape[1]), dtype=audio.dtype)
    return np.vstack([audio, pad])


def rms_dbfs(audio: np.ndarray) -> float:
    return round(float(20.0 * np.log10(float(np.sqrt(np.mean(np.square(audio)))) + EPSILON)), 6)


def audio_file_summary(path: Path) -> dict[str, Any]:
    info = sf.info(str(path))
    audio, _ = read_stereo_audio(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "duration_sec": round(float(info.frames / info.samplerate), 6),
        "frames": int(info.frames),
        "channels": int(info.channels),
        "sample_rate": int(info.samplerate),
        "rms_dbfs": rms_dbfs(audio),
        "peak_dbfs": round(float(20.0 * np.log10(float(np.max(np.abs(audio))) + EPSILON)), 6),
    }


def safe_stft_channel(audio: np.ndarray, config: StemDeltaProfileConfig) -> np.ndarray:
    n_fft = min(config.n_fft, max(256, int(2 ** np.floor(np.log2(max(256, audio.shape[0]))))))
    hop_length = min(config.hop_length, max(64, n_fft // 4))
    return librosa.stft(audio.astype(np.float32), n_fft=n_fft, hop_length=hop_length, center=True)


def average_magnitude(audio: np.ndarray, config: StemDeltaProfileConfig) -> tuple[np.ndarray, np.ndarray]:
    stfts = [safe_stft_channel(audio[:, channel], config) for channel in range(audio.shape[1])]
    magnitude = np.mean([np.abs(stft) for stft in stfts], axis=0)
    sample_n_fft = (magnitude.shape[0] - 1) * 2
    return magnitude.astype(np.float32), np.fft.rfftfreq(sample_n_fft, d=1.0)


def db_from_linear(value: np.ndarray | float) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(value, EPSILON))


def band_indices(frequencies_hz: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    return (frequencies_hz >= low_hz) & (frequencies_hz < high_hz)


def build_stem_delta_profile(
    candidate_audio: np.ndarray,
    teacher_repair_audio: np.ndarray,
    reference_audio: np.ndarray,
    sample_rate: int,
    stem_name: str,
    config: StemDeltaProfileConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not 0.0 <= config.profile_mix <= 1.0:
        raise ValueError("profile_mix must be within [0, 1]")
    if config.max_abs_gain_db <= 0.0:
        raise ValueError("max_abs_gain_db must be positive")

    frames = min(candidate_audio.shape[0], teacher_repair_audio.shape[0], reference_audio.shape[0])
    candidate_audio = fit_length(candidate_audio, frames)
    teacher_repair_audio = fit_length(teacher_repair_audio, frames)
    reference_audio = fit_length(reference_audio, frames)

    candidate_mag, frequency_ratio = average_magnitude(candidate_audio, config)
    repair_mag, _ = average_magnitude(teacher_repair_audio, config)
    reference_mag, _ = average_magnitude(reference_audio, config)
    frequencies_hz = frequency_ratio * float(sample_rate)

    band_profiles: list[dict[str, Any]] = []
    for band_name, low_hz, high_hz in PROFILE_BANDS:
        mask = band_indices(frequencies_hz, low_hz, min(high_hz, sample_rate / 2.0))
        if not np.any(mask):
            continue
        candidate_band = candidate_mag[mask]
        repair_band = repair_mag[mask]
        reference_band = reference_mag[mask]
        raw_gain_db = float(np.median(db_from_linear((repair_band + EPSILON) / (candidate_band + EPSILON))))
        candidate_error_db = float(db_from_linear(np.mean(np.abs(candidate_band - reference_band))))
        repair_error_db = float(db_from_linear(np.mean(np.abs(repair_band - reference_band))))
        error_improvement_db = candidate_error_db - repair_error_db
        learned_gain_db = raw_gain_db if error_improvement_db >= config.min_error_improvement_db else 0.0
        learned_gain_db = float(np.clip(learned_gain_db * config.profile_mix, -config.max_abs_gain_db, config.max_abs_gain_db))
        band_profiles.append(
            {
                "name": band_name,
                "low_hz": low_hz,
                "high_hz": min(high_hz, sample_rate / 2.0),
                "gain_db": round(learned_gain_db, 6),
                "raw_teacher_gain_db": round(raw_gain_db, 6),
                "candidate_reference_error_db": round(candidate_error_db, 6),
                "teacher_reference_error_db": round(repair_error_db, 6),
                "error_improvement_db": round(float(error_improvement_db), 6),
            }
        )

    frame_error_before = np.mean(np.abs(candidate_mag - reference_mag), axis=0)
    frame_error_after = np.mean(np.abs(repair_mag - reference_mag), axis=0)
    frame_improvement = frame_error_before - frame_error_after
    top_indices = np.argsort(frame_improvement)[::-1][: config.max_report_windows]
    frame_times = librosa.frames_to_time(np.arange(candidate_mag.shape[1]), sr=sample_rate, hop_length=config.hop_length)
    top_windows = [
        {
            "time_sec": round(float(frame_times[index]), 3),
            "error_improvement_linear": round(float(frame_improvement[index]), 8),
            "candidate_error_linear": round(float(frame_error_before[index]), 8),
            "teacher_error_linear": round(float(frame_error_after[index]), 8),
        }
        for index in top_indices
        if frame_improvement[index] > 0.0
    ]

    profile = {
        "schema_version": 1,
        "profile_id": f"{MODEL_ID}:{stem_name}",
        "stem": stem_name,
        "sample_rate": sample_rate,
        "contract": {
            "application_reference_free": True,
            "target_reference_used_at_application": False,
            "profile_source_reference_assisted": True,
            "not_timestamp_hardcoded": True,
            "per_track_hardcode_allowed": False,
            "delivery_allowed_without_reference": False,
            "not_a_no_reference_production_route": True,
        },
        "config": {
            "n_fft": config.n_fft,
            "hop_length": config.hop_length,
            "max_abs_gain_db": config.max_abs_gain_db,
            "profile_mix": config.profile_mix,
            "min_error_improvement_db": config.min_error_improvement_db,
            "peak_ceiling": config.peak_ceiling,
            "use_percussive_gate": config.use_percussive_gate,
            "allow_negative_gains": config.allow_negative_gains,
            "use_source_percussive_reservoir": config.use_source_percussive_reservoir,
            "source_reservoir_mix": config.source_reservoir_mix,
            "source_reservoir_min_hz": config.source_reservoir_min_hz,
            "source_reservoir_max_hz": config.source_reservoir_max_hz,
            "source_reservoir_min_source_percussive_confidence": config.source_reservoir_min_source_percussive_confidence,
            "source_reservoir_min_candidate_percussive_confidence": config.source_reservoir_min_candidate_percussive_confidence,
        },
        "bands": band_profiles,
    }
    diagnostics = {
        "stem": stem_name,
        "rms_dbfs": {
            "candidate": rms_dbfs(candidate_audio),
            "teacher_repair": rms_dbfs(teacher_repair_audio),
            "reference": rms_dbfs(reference_audio),
        },
        "top_error_improvement_windows": top_windows,
    }
    return profile, diagnostics


def frequency_gain_db_vector(profile: dict[str, Any], sample_rate: int, n_bins: int, config: StemDeltaProfileConfig) -> np.ndarray:
    n_fft = max(2, (n_bins - 1) * 2)
    frequencies_hz = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    gain_db = np.zeros(n_bins, dtype=np.float32)
    for band in profile.get("bands", []):
        mask = band_indices(frequencies_hz, float(band["low_hz"]), float(band["high_hz"]))
        band_gain_db = float(band["gain_db"])
        if not config.allow_negative_gains:
            band_gain_db = max(0.0, band_gain_db)
        gain_db[mask] = band_gain_db
    return gain_db


def apply_stem_delta_profile(
    candidate_audio: np.ndarray,
    sample_rate: int,
    profile: dict[str, Any],
    config: StemDeltaProfileConfig,
    source_audio: np.ndarray | None = None,
) -> np.ndarray:
    if config.use_source_percussive_reservoir and source_audio is None:
        raise ValueError("source_audio is required when use_source_percussive_reservoir=True")
    if not 0.0 <= config.source_reservoir_mix <= 1.0:
        raise ValueError("source_reservoir_mix must be within [0, 1]")
    source_audio = fit_length(source_audio, candidate_audio.shape[0]) if source_audio is not None else None
    output_channels = []
    for channel in range(candidate_audio.shape[1]):
        stft = safe_stft_channel(candidate_audio[:, channel], config)
        gain_db = frequency_gain_db_vector(profile, sample_rate, stft.shape[0], config)[:, None]
        harmonic, percussive = librosa.decompose.hpss(stft)
        if config.use_percussive_gate:
            harmonic_mag = np.abs(harmonic)
            percussive_mag = np.abs(percussive)
            gate = percussive_mag / (harmonic_mag + percussive_mag + EPSILON)
            effective_gain_db = gain_db * np.clip(gate, 0.0, 1.0)
        else:
            effective_gain_db = gain_db
        repaired_stft = stft * np.power(10.0, effective_gain_db / 20.0).astype(np.float32)
        if config.use_source_percussive_reservoir and source_audio is not None:
            source_stft = safe_stft_channel(source_audio[:, channel], config)
            source_harmonic, source_percussive = librosa.decompose.hpss(source_stft)
            source_harmonic_mag = np.abs(source_harmonic)
            candidate_percussive_mag = np.abs(percussive)
            source_percussive_mag = np.abs(source_percussive)
            missing_percussive_ratio = np.maximum(source_percussive_mag - candidate_percussive_mag, 0.0) / (
                source_percussive_mag + EPSILON
            )
            source_percussive_confidence = source_percussive_mag / (
                source_harmonic_mag + source_percussive_mag + EPSILON
            )
            candidate_percussive_confidence = candidate_percussive_mag / (
                np.abs(harmonic) + candidate_percussive_mag + EPSILON
            )
            source_confidence_gate = np.clip(
                (source_percussive_confidence - config.source_reservoir_min_source_percussive_confidence)
                / max(EPSILON, 1.0 - config.source_reservoir_min_source_percussive_confidence),
                0.0,
                1.0,
            )
            candidate_confidence_gate = np.clip(
                (candidate_percussive_confidence - config.source_reservoir_min_candidate_percussive_confidence)
                / max(EPSILON, 1.0 - config.source_reservoir_min_candidate_percussive_confidence),
                0.0,
                1.0,
            )
            reservoir_confidence_gate = source_confidence_gate * candidate_confidence_gate
            n_fft = max(2, (stft.shape[0] - 1) * 2)
            frequencies_hz = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
            reservoir_band = (
                (frequencies_hz >= config.source_reservoir_min_hz)
                & (frequencies_hz <= min(config.source_reservoir_max_hz, sample_rate / 2.0))
            ).astype(np.float32)[:, None]
            repaired_stft = repaired_stft + (
                config.source_reservoir_mix
                * source_percussive
                * missing_percussive_ratio
                * reservoir_band
                * reservoir_confidence_gate
            )
        repaired_channel = librosa.istft(
            repaired_stft,
            hop_length=min(config.hop_length, max(64, ((stft.shape[0] - 1) * 2) // 4)),
            length=candidate_audio.shape[0],
            center=True,
        )
        output_channels.append(repaired_channel.astype(np.float32))
    repaired = np.stack(output_channels, axis=1)
    peak = float(np.max(np.abs(repaired)))
    if peak > config.peak_ceiling:
        repaired = repaired * (config.peak_ceiling / peak)
    return np.clip(repaired, -1.0, 1.0).astype(np.float32)


def build_listen_pack(
    listen_dir: Path,
    reference_stem_path: Path,
    candidate_stem_path: Path,
    teacher_repair_path: Path,
    profile_repair_path: Path,
    stem_name: str,
) -> None:
    listen_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        f"00_reference_teacher_{stem_name}.flac": reference_stem_path,
        f"01_base_candidate_{stem_name}.flac": candidate_stem_path,
        f"02_teacher_residual_repair_{stem_name}.flac": teacher_repair_path,
        f"03_profile_repair_reference_free_application_{stem_name}.flac": profile_repair_path,
    }
    for filename, source in mapping.items():
        shutil.copyfile(source, listen_dir / filename)
    (listen_dir / "README.txt").write_text(
        "00 = verified reference teacher stem.\n"
        "01 = generated base candidate before learned profile repair.\n"
        "02 = supervised residual repair teacher artifact.\n"
        "03 = same candidate with learned profile applied without reading reference during application.\n",
        encoding="utf-8",
    )


def build_manifest(
    *,
    ground_truth_manifest: Path,
    base_candidate_manifest: Path,
    teacher_repair_manifest: Path,
    source_mix_path: Path,
    output_stem_path: Path,
    profile_path: Path,
    diagnostic_path: Path,
    stem_name: str,
    config: StemDeltaProfileConfig,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_audio_path": str(source_mix_path),
        "model_id": MODEL_ID,
        "model_name": MODEL_ID,
        "model": {"model_id": MODEL_ID, "family": "learned_stem_delta_profile"},
        "contract": {
            "full_track": True,
            "preview": False,
            "benchmark_compatible": True,
            "fallback_used": False,
            "hidden_inference": False,
            "registered_model_candidate": False,
            "target_reference_assisted": True,
            "profile_source_reference_assisted": True,
            "target_reference_used_at_application": False,
            "application_reference_free": True,
            "delivery_allowed_without_reference": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
            "quality_gate_required": True,
        },
        "profile_application": {
            "stem": stem_name,
            "profile_path": str(profile_path),
            "diagnostic_report_path": str(diagnostic_path),
            "not_a_no_reference_production_route": True,
            "rule_scope": "generic_frequency_band_delta_profile",
            "uses_reference_during_application": False,
        },
        "inputs": {
            "ground_truth_manifest": str(ground_truth_manifest),
            "base_candidate_manifest": str(base_candidate_manifest),
            "teacher_repair_manifest": str(teacher_repair_manifest),
        },
        "config": {
            "n_fft": config.n_fft,
            "hop_length": config.hop_length,
            "max_abs_gain_db": config.max_abs_gain_db,
            "profile_mix": config.profile_mix,
            "min_error_improvement_db": config.min_error_improvement_db,
            "peak_ceiling": config.peak_ceiling,
            "use_percussive_gate": config.use_percussive_gate,
            "allow_negative_gains": config.allow_negative_gains,
            "use_source_percussive_reservoir": config.use_source_percussive_reservoir,
            "source_reservoir_mix": config.source_reservoir_mix,
            "source_reservoir_min_hz": config.source_reservoir_min_hz,
            "source_reservoir_max_hz": config.source_reservoir_max_hz,
            "source_reservoir_min_source_percussive_confidence": config.source_reservoir_min_source_percussive_confidence,
            "source_reservoir_min_candidate_percussive_confidence": config.source_reservoir_min_candidate_percussive_confidence,
        },
        "stems": {
            stem_name: {
                "path": str(output_stem_path),
                "sha256": sha256_file(output_stem_path),
                "role": "profile_repaired_reference_free_application_stem",
                "profile_source_reference_assisted": True,
                "target_reference_used_at_application": False,
            }
        },
    }


def build_stem_delta_profile_repair(
    *,
    ground_truth_manifest: Path,
    base_candidate_manifest: Path,
    teacher_repair_manifest: Path,
    output_dir: Path,
    stem_name: str,
    config: StemDeltaProfileConfig,
    listen_dir: Path | None = None,
) -> dict[str, Any]:
    require_docker_runtime(
        tool_name="build_stem_delta_profile_repair.py",
        entrypoint_path="/app/tools/build_stem_delta_profile_repair.py",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_mix_path = resolve_ground_truth_mix_path(ground_truth_manifest)
    reference_stem_path = resolve_stem_path(ground_truth_manifest, stem_name)
    candidate_stem_path = resolve_stem_path(base_candidate_manifest, stem_name)
    teacher_repair_stem_path = resolve_stem_path(teacher_repair_manifest, stem_name)

    reference_audio, reference_sample_rate = read_stereo_audio(reference_stem_path)
    candidate_audio, candidate_sample_rate = read_stereo_audio(candidate_stem_path)
    teacher_repair_audio, teacher_sample_rate = read_stereo_audio(teacher_repair_stem_path)
    source_audio, source_sample_rate = read_stereo_audio(source_mix_path)
    if len({reference_sample_rate, candidate_sample_rate, teacher_sample_rate, source_sample_rate}) != 1:
        raise ValueError("reference, candidate, and teacher repair sample rates must match")
    frames = min(reference_audio.shape[0], candidate_audio.shape[0], teacher_repair_audio.shape[0], source_audio.shape[0])
    reference_audio = fit_length(reference_audio, frames)
    candidate_audio = fit_length(candidate_audio, frames)
    teacher_repair_audio = fit_length(teacher_repair_audio, frames)
    source_audio = fit_length(source_audio, frames)

    profile, diagnostics = build_stem_delta_profile(
        candidate_audio=candidate_audio,
        teacher_repair_audio=teacher_repair_audio,
        reference_audio=reference_audio,
        sample_rate=reference_sample_rate,
        stem_name=stem_name,
        config=config,
    )
    repaired = apply_stem_delta_profile(
        candidate_audio,
        reference_sample_rate,
        profile,
        config,
        source_audio=source_audio if config.use_source_percussive_reservoir else None,
    )

    output_stem_path = output_dir / f"{stem_name}.flac"
    sf.write(str(output_stem_path), repaired, reference_sample_rate, subtype="PCM_24")

    profile_path = output_dir / "stem_delta_profile.json"
    diagnostic_path = output_dir / "stem_delta_profile_diagnostic_report.json"
    write_json(profile_path, profile)
    write_json(
        diagnostic_path,
        {
            "schema_version": 1,
            "model_id": MODEL_ID,
            "stem": stem_name,
            "inputs": {
                "ground_truth_manifest": str(ground_truth_manifest),
                "base_candidate_manifest": str(base_candidate_manifest),
                "teacher_repair_manifest": str(teacher_repair_manifest),
            },
            "input_audio": {
                "reference": audio_file_summary(reference_stem_path),
                "base_candidate": audio_file_summary(candidate_stem_path),
                "teacher_repair": audio_file_summary(teacher_repair_stem_path),
                "profile_repair": audio_file_summary(output_stem_path),
            },
            "diagnostics": diagnostics,
            "profile": profile,
        },
    )

    manifest = build_manifest(
        ground_truth_manifest=ground_truth_manifest,
        base_candidate_manifest=base_candidate_manifest,
        teacher_repair_manifest=teacher_repair_manifest,
        source_mix_path=source_mix_path,
        output_stem_path=output_stem_path,
        profile_path=profile_path,
        diagnostic_path=diagnostic_path,
        stem_name=stem_name,
        config=config,
    )
    manifest_path = output_dir / "stem_manifest.json"
    write_json(manifest_path, manifest)
    if listen_dir is not None:
        build_listen_pack(
            listen_dir=listen_dir,
            reference_stem_path=reference_stem_path,
            candidate_stem_path=candidate_stem_path,
            teacher_repair_path=teacher_repair_stem_path,
            profile_repair_path=output_stem_path,
            stem_name=stem_name,
        )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Learn a stem delta profile from a reference-guided teacher artifact and apply it without reading reference during application."
    )
    parser.add_argument("--ground-truth-manifest", type=Path, required=True)
    parser.add_argument("--base-candidate-manifest", type=Path, required=True)
    parser.add_argument("--teacher-repair-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--n-fft", type=int, default=StemDeltaProfileConfig.n_fft)
    parser.add_argument("--hop-length", type=int, default=StemDeltaProfileConfig.hop_length)
    parser.add_argument("--max-abs-gain-db", type=float, default=StemDeltaProfileConfig.max_abs_gain_db)
    parser.add_argument("--profile-mix", type=float, default=StemDeltaProfileConfig.profile_mix)
    parser.add_argument("--min-error-improvement-db", type=float, default=StemDeltaProfileConfig.min_error_improvement_db)
    parser.add_argument("--peak-ceiling", type=float, default=StemDeltaProfileConfig.peak_ceiling)
    parser.add_argument("--disable-percussive-gate", action="store_true")
    parser.add_argument("--allow-negative-gains", action="store_true")
    parser.add_argument("--use-source-percussive-reservoir", action="store_true")
    parser.add_argument("--source-reservoir-mix", type=float, default=StemDeltaProfileConfig.source_reservoir_mix)
    parser.add_argument("--source-reservoir-min-hz", type=float, default=StemDeltaProfileConfig.source_reservoir_min_hz)
    parser.add_argument("--source-reservoir-max-hz", type=float, default=StemDeltaProfileConfig.source_reservoir_max_hz)
    parser.add_argument(
        "--source-reservoir-min-source-percussive-confidence",
        type=float,
        default=StemDeltaProfileConfig.source_reservoir_min_source_percussive_confidence,
    )
    parser.add_argument(
        "--source-reservoir-min-candidate-percussive-confidence",
        type=float,
        default=StemDeltaProfileConfig.source_reservoir_min_candidate_percussive_confidence,
    )
    return parser


def main() -> None:
    require_docker_runtime(
        tool_name="build_stem_delta_profile_repair.py",
        entrypoint_path="/app/tools/build_stem_delta_profile_repair.py",
    )
    args = build_arg_parser().parse_args()
    manifest = build_stem_delta_profile_repair(
        ground_truth_manifest=args.ground_truth_manifest,
        base_candidate_manifest=args.base_candidate_manifest,
        teacher_repair_manifest=args.teacher_repair_manifest,
        output_dir=args.output_dir,
        stem_name=args.stem,
        listen_dir=args.listen_dir,
        config=StemDeltaProfileConfig(
            n_fft=int(args.n_fft),
            hop_length=int(args.hop_length),
            max_abs_gain_db=float(args.max_abs_gain_db),
            profile_mix=float(args.profile_mix),
            min_error_improvement_db=float(args.min_error_improvement_db),
            peak_ceiling=float(args.peak_ceiling),
            use_percussive_gate=not bool(args.disable_percussive_gate),
            allow_negative_gains=bool(args.allow_negative_gains),
            use_source_percussive_reservoir=bool(args.use_source_percussive_reservoir),
            source_reservoir_mix=float(args.source_reservoir_mix),
            source_reservoir_min_hz=float(args.source_reservoir_min_hz),
            source_reservoir_max_hz=float(args.source_reservoir_max_hz),
            source_reservoir_min_source_percussive_confidence=float(
                args.source_reservoir_min_source_percussive_confidence
            ),
            source_reservoir_min_candidate_percussive_confidence=float(
                args.source_reservoir_min_candidate_percussive_confidence
            ),
        ),
    )
    print(json.dumps({"manifest_path": str(args.output_dir / "stem_manifest.json"), "contract": manifest["contract"]}, indent=2))


if __name__ == "__main__":
    main()
