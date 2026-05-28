from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
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


@dataclass(frozen=True)
class TonalRepairConfig:
    n_fft: int = 4096
    hop_length: int = 1024
    min_tonal_hz: float = 160.0
    max_tonal_hz: float = 4200.0
    harmonic_ratio_floor: float = 0.54
    suppression_strength: float = 0.78
    min_tonal_gain: float = 0.22
    transient_guard_strength: float = 0.85
    transient_guard_floor: float = 0.46
    low_body_hz: float = 130.0
    high_air_hz: float = 7600.0


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if audio.shape[1] > 2:
        audio = audio[:, :2]
    return audio.astype(np.float32, copy=False), int(sample_rate)


def _write_flac(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, subtype="PCM_24")


def _rms_dbfs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -120.0
    value = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)) + 1e-12)
    return 20.0 * math.log10(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _config_from_manifest(manifest_path: Path) -> TonalRepairConfig:
    manifest = _load_json(manifest_path)
    config_data = manifest.get("config")
    if not isinstance(config_data, dict):
        raise ValueError(f"replay manifest does not contain config: {manifest_path}")
    return TonalRepairConfig(
        n_fft=int(config_data.get("n_fft", TonalRepairConfig.n_fft)),
        hop_length=int(config_data.get("hop_length", TonalRepairConfig.hop_length)),
        min_tonal_hz=float(config_data.get("min_tonal_hz", TonalRepairConfig.min_tonal_hz)),
        max_tonal_hz=float(config_data.get("max_tonal_hz", TonalRepairConfig.max_tonal_hz)),
        harmonic_ratio_floor=float(config_data.get("harmonic_ratio_floor", TonalRepairConfig.harmonic_ratio_floor)),
        suppression_strength=float(config_data.get("suppression_strength", TonalRepairConfig.suppression_strength)),
        min_tonal_gain=float(config_data.get("min_tonal_gain", TonalRepairConfig.min_tonal_gain)),
        transient_guard_strength=float(
            config_data.get("transient_guard_strength", TonalRepairConfig.transient_guard_strength)
        ),
        transient_guard_floor=float(config_data.get("transient_guard_floor", TonalRepairConfig.transient_guard_floor)),
        low_body_hz=float(config_data.get("low_body_hz", TonalRepairConfig.low_body_hz)),
        high_air_hz=float(config_data.get("high_air_hz", TonalRepairConfig.high_air_hz)),
    )


def _input_path_from_manifest(manifest_path: Path) -> Path:
    manifest = _load_json(manifest_path)
    input_data = manifest.get("input")
    if not isinstance(input_data, dict) or not input_data.get("path"):
        raise ValueError(f"replay manifest does not contain input.path: {manifest_path}")
    return Path(str(input_data["path"]))


def _output_label_from_manifest(manifest_path: Path) -> str:
    manifest = _load_json(manifest_path)
    output_data = manifest.get("output")
    if not isinstance(output_data, dict) or not output_data.get("path"):
        raise ValueError(f"replay manifest does not contain output.path: {manifest_path}")
    return Path(str(output_data["path"])).name


def _expected_output_sha_from_manifest(manifest_path: Path) -> str:
    manifest = _load_json(manifest_path)
    output_data = manifest.get("output")
    if not isinstance(output_data, dict) or not output_data.get("sha256"):
        raise ValueError(f"replay manifest does not contain output.sha256: {manifest_path}")
    return str(output_data["sha256"])


def _safe_stft(audio: np.ndarray, config: TonalRepairConfig) -> np.ndarray:
    n_fft = min(config.n_fft, max(256, 2 ** int(math.floor(math.log2(max(256, audio.shape[0]))))))
    hop_length = min(config.hop_length, max(64, n_fft // 4))
    return librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)


def _frame_guard_from_components(harmonic_mag: np.ndarray, percussive_mag: np.ndarray, config: TonalRepairConfig) -> np.ndarray:
    percussive_energy = np.sum(percussive_mag, axis=0)
    harmonic_energy = np.sum(harmonic_mag, axis=0)
    percussive_ratio = percussive_energy / (percussive_energy + harmonic_energy + 1e-9)
    guard = np.clip(
        (percussive_ratio - config.transient_guard_floor) / max(1e-6, 1.0 - config.transient_guard_floor),
        0.0,
        1.0,
    )
    if guard.size > 1:
        guard = np.maximum(guard, np.r_[guard[0], guard[:-1]] * 0.65)
        guard = np.maximum(guard, np.r_[guard[1:], guard[-1]] * 0.65)
    return np.clip(guard * config.transient_guard_strength, 0.0, 1.0)


def _frequency_weight(frequencies: np.ndarray, config: TonalRepairConfig) -> np.ndarray:
    tonal_band = (frequencies >= config.min_tonal_hz) & (frequencies <= config.max_tonal_hz)
    body_guard = np.clip((frequencies - config.low_body_hz) / max(1.0, config.min_tonal_hz - config.low_body_hz), 0.0, 1.0)
    air_guard = 1.0 - np.clip((frequencies - config.max_tonal_hz) / max(1.0, config.high_air_hz - config.max_tonal_hz), 0.0, 1.0)
    return tonal_band.astype(np.float32) * body_guard.astype(np.float32) * air_guard.astype(np.float32)


def _repair_channel(audio: np.ndarray, sample_rate: int, config: TonalRepairConfig) -> tuple[np.ndarray, dict[str, Any]]:
    stft = _safe_stft(audio, config)
    n_fft = (stft.shape[0] - 1) * 2
    hop_length = min(config.hop_length, max(64, n_fft // 4))
    magnitude = np.abs(stft)
    harmonic_mag, percussive_mag = librosa.decompose.hpss(magnitude, margin=(1.0, 2.8))
    harmonic_ratio = harmonic_mag / (harmonic_mag + percussive_mag + 1e-9)

    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    frequency_weight = _frequency_weight(frequencies, config).reshape(-1, 1)
    harmonic_excess = np.clip(
        (harmonic_ratio - config.harmonic_ratio_floor) / max(1e-6, 1.0 - config.harmonic_ratio_floor),
        0.0,
        1.0,
    )
    frame_guard = _frame_guard_from_components(harmonic_mag, percussive_mag, config).reshape(1, -1)
    tonal_mask = harmonic_excess * frequency_weight * (1.0 - frame_guard)
    gain = np.clip(1.0 - config.suppression_strength * tonal_mask, config.min_tonal_gain, 1.0)
    repaired_stft = stft * gain
    repaired = librosa.istft(repaired_stft, hop_length=hop_length, length=audio.shape[0])
    return repaired.astype(np.float32), {
        "mean_tonal_mask": round(float(np.mean(tonal_mask)), 8),
        "max_tonal_mask": round(float(np.max(tonal_mask)), 8),
        "mean_gain": round(float(np.mean(gain)), 8),
        "min_gain": round(float(np.min(gain)), 8),
        "mean_transient_guard": round(float(np.mean(frame_guard)), 8),
    }


def _fit_length(audio: np.ndarray, frames: int) -> np.ndarray:
    if audio.shape[0] == frames:
        return audio
    if audio.shape[0] > frames:
        return audio[:frames]
    pad = np.zeros((frames - audio.shape[0], audio.shape[1]), dtype=audio.dtype)
    return np.concatenate([audio, pad], axis=0)


def _slice_audio(audio: np.ndarray, sample_rate: int, window: tuple[float, float] | None) -> np.ndarray:
    if window is None:
        return audio
    start_sec, end_sec = window
    start = max(0, int(round(start_sec * sample_rate)))
    end = min(audio.shape[0], int(round(end_sec * sample_rate)))
    return audio[start:end]


def _tonal_metrics(audio: np.ndarray, sample_rate: int, config: TonalRepairConfig) -> dict[str, float]:
    if audio.shape[0] < 1024:
        return {
            "rms_dbfs": round(_rms_dbfs(audio), 6),
            "percussive_minus_harmonic_db": 0.0,
            "chroma_peak_share": 0.0,
            "narrowband_peak_share": 0.0,
        }
    mono = audio.mean(axis=1)
    max_frames = sample_rate * 30
    if mono.shape[0] > max_frames:
        mono = mono[:max_frames]
    harmonic, percussive = librosa.effects.hpss(mono)
    stft_mag = np.abs(librosa.stft(mono, n_fft=min(config.n_fft, 4096), hop_length=min(config.hop_length, 1024))) + 1e-9
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=(stft_mag.shape[0] - 1) * 2)
    tonal_band = (frequencies >= config.min_tonal_hz) & (frequencies <= config.max_tonal_hz)
    tonal_mag = stft_mag[tonal_band]
    narrowband_peak_share = 0.0
    if tonal_mag.size:
        narrowband_peak_share = float(np.mean(np.max(tonal_mag, axis=0) / (np.sum(tonal_mag, axis=0) + 1e-9)))
    chroma = librosa.feature.chroma_stft(S=stft_mag, sr=sample_rate)
    chroma_peak_share = float(np.mean(np.max(chroma, axis=0) / (np.sum(chroma, axis=0) + 1e-9)))
    return {
        "rms_dbfs": round(_rms_dbfs(audio), 6),
        "harmonic_dbfs": round(_rms_dbfs(harmonic.reshape(-1, 1)), 6),
        "percussive_dbfs": round(_rms_dbfs(percussive.reshape(-1, 1)), 6),
        "percussive_minus_harmonic_db": round(
            _rms_dbfs(percussive.reshape(-1, 1)) - _rms_dbfs(harmonic.reshape(-1, 1)),
            6,
        ),
        "chroma_peak_share": round(chroma_peak_share, 8),
        "narrowband_peak_share": round(narrowband_peak_share, 8),
    }


def repair_drum_tonal_leakage(
    drum_stem_path: Path,
    output_dir: Path,
    *,
    listen_dir: Path | None = None,
    output_label: str = "02_drum_tonal_leakage_repaired.flac",
    input_label: str = "01_input_drums.flac",
    validation_windows: dict[str, tuple[float, float]] | None = None,
    config: TonalRepairConfig | None = None,
) -> dict[str, Any]:
    config = config or TonalRepairConfig()
    if config.n_fft < 256:
        raise ValueError("n_fft must be at least 256.")
    if not 0.0 <= config.suppression_strength <= 1.0:
        raise ValueError("suppression_strength must be in [0, 1].")
    if not 0.0 < config.min_tonal_gain <= 1.0:
        raise ValueError("min_tonal_gain must be in (0, 1].")
    if config.min_tonal_hz >= config.max_tonal_hz:
        raise ValueError("min_tonal_hz must be lower than max_tonal_hz.")

    input_audio, sample_rate = _read_audio(drum_stem_path)
    repaired_channels: list[np.ndarray] = []
    channel_reports: list[dict[str, Any]] = []
    for channel_index in range(input_audio.shape[1]):
        repaired_channel, channel_report = _repair_channel(input_audio[:, channel_index], sample_rate, config)
        repaired_channels.append(repaired_channel)
        channel_reports.append({"channel": channel_index, **channel_report})
    repaired_audio = np.stack(repaired_channels, axis=1)
    repaired_audio = _fit_length(repaired_audio, input_audio.shape[0])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_label
    _write_flac(output_path, repaired_audio, sample_rate)

    listen_pack: dict[str, str] = {}
    if listen_dir is not None:
        listen_dir.mkdir(parents=True, exist_ok=True)
        input_copy_path = listen_dir / input_label
        output_copy_path = listen_dir / output_label
        shutil.copy2(drum_stem_path, input_copy_path)
        shutil.copy2(output_path, output_copy_path)
        listen_pack[input_label] = str(input_copy_path)
        listen_pack[output_label] = str(output_copy_path)

    windows = validation_windows or {}
    metric_windows: dict[str, tuple[float, float] | None] = {"full": None, **windows}
    metrics: dict[str, Any] = {}
    for label, window in metric_windows.items():
        input_slice = _slice_audio(input_audio, sample_rate, window)
        output_slice = _slice_audio(repaired_audio, sample_rate, window)
        metrics[label] = {
            "input": _tonal_metrics(input_slice, sample_rate, config),
            "output": _tonal_metrics(output_slice, sample_rate, config),
        }
        metrics[label]["delta"] = {
            "rms_db": round(metrics[label]["output"]["rms_dbfs"] - metrics[label]["input"]["rms_dbfs"], 6),
            "percussive_minus_harmonic_db": round(
                metrics[label]["output"]["percussive_minus_harmonic_db"]
                - metrics[label]["input"]["percussive_minus_harmonic_db"],
                6,
            ),
            "chroma_peak_share": round(
                metrics[label]["output"]["chroma_peak_share"] - metrics[label]["input"]["chroma_peak_share"],
                8,
            ),
            "narrowband_peak_share": round(
                metrics[label]["output"]["narrowband_peak_share"] - metrics[label]["input"]["narrowband_peak_share"],
                8,
            ),
        }

    manifest = {
        "contract": {
            "route": "drum_tonal_leakage_repair_v1",
            "accepted_by_listening": False,
            "fallback_used": False,
            "not_timestamp_hardcoded": True,
            "purpose": "Suppress stable pitched/harmonic leakage in DRUM while preserving transient/body evidence.",
        },
        "input": {
            "path": str(drum_stem_path),
            "sha256": _sha256(drum_stem_path),
        },
        "output": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "sample_rate": sample_rate,
            "frames": int(repaired_audio.shape[0]),
            "rms_dbfs": round(_rms_dbfs(repaired_audio), 6),
        },
        "config": {
            "n_fft": config.n_fft,
            "hop_length": config.hop_length,
            "min_tonal_hz": config.min_tonal_hz,
            "max_tonal_hz": config.max_tonal_hz,
            "harmonic_ratio_floor": config.harmonic_ratio_floor,
            "suppression_strength": config.suppression_strength,
            "min_tonal_gain": config.min_tonal_gain,
            "transient_guard_strength": config.transient_guard_strength,
            "transient_guard_floor": config.transient_guard_floor,
            "low_body_hz": config.low_body_hz,
            "high_air_hz": config.high_air_hz,
        },
        "channel_reports": channel_reports,
        "metrics": metrics,
        "listen_pack": listen_pack,
    }
    manifest_path = output_dir / "drum_tonal_leakage_repair_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    candidate_manifest = {
        "schema_version": 1,
        "model_id": "drum_tonal_leakage_repair",
        "model_name": "drum_tonal_leakage_repair",
        "model": {
            "model_id": "drum_tonal_leakage_repair",
            "family": "no_reference_drum_tonal_cleanup",
        },
        "contract": {
            "full_track": True,
            "preview": False,
            "benchmark_compatible": True,
            "fallback_used": False,
            "hidden_inference": False,
            "registered_model_candidate": False,
            "target_reference_assisted": False,
            "target_reference_used_at_application": False,
            "application_reference_free": True,
            "delivery_allowed_without_reference": True,
            "delivery_allowed_without_quality_gate": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
            "quality_gate_required": True,
        },
        "route": {
            "mode": "drum_tonal_leakage_repair_v1",
            "diagnostic_manifest_path": str(manifest_path),
        },
        "inputs": {
            "drum_stem_path": str(drum_stem_path),
        },
        "stems": {
            "drums": {
                "path": str(output_path),
                "sha256": _sha256(output_path),
                "role": "drum_tonal_leakage_repaired_candidate",
            }
        },
    }
    (output_dir / "stem_manifest.json").write_text(json.dumps(candidate_manifest, indent=2), encoding="utf-8")
    return manifest


def replay_drum_tonal_leakage_repair(
    replay_manifest_path: Path,
    output_dir: Path,
    *,
    listen_dir: Path | None = None,
) -> dict[str, Any]:
    replay_manifest_path = replay_manifest_path.resolve()
    drum_stem_path = _input_path_from_manifest(replay_manifest_path)
    config = _config_from_manifest(replay_manifest_path)
    output_label = _output_label_from_manifest(replay_manifest_path)
    expected_sha256 = _expected_output_sha_from_manifest(replay_manifest_path)
    manifest = repair_drum_tonal_leakage(
        drum_stem_path,
        output_dir,
        listen_dir=listen_dir,
        output_label=output_label,
        config=config,
    )
    actual_sha256 = str(manifest["output"]["sha256"])
    replay_report = {
        "source_manifest": str(replay_manifest_path),
        "input_path": str(drum_stem_path),
        "expected_output_sha256": expected_sha256,
        "actual_output_sha256": actual_sha256,
        "sha256_matched": actual_sha256 == expected_sha256,
    }
    manifest["replay"] = replay_report
    manifest_path = output_dir / "drum_tonal_leakage_repair_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not replay_report["sha256_matched"]:
        raise ValueError(
            "replayed drum tonal leakage repair does not match expected sha256: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    return manifest


def _parse_validation_window(value: str) -> tuple[str, tuple[float, float]]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("validation window must use LABEL:START_SEC:END_SEC")
    label = parts[0].strip()
    if not label:
        raise argparse.ArgumentTypeError("validation window label cannot be empty")
    start_sec = float(parts[1])
    end_sec = float(parts[2])
    if end_sec <= start_sec:
        raise argparse.ArgumentTypeError("validation window end must be greater than start")
    return label, (start_sec, end_sec)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair pitched tonal leakage in a DRUM stem.")
    parser.add_argument("drum_stem_path", nargs="?", type=Path)
    parser.add_argument("output_dir", nargs="?", type=Path)
    parser.add_argument("--replay-manifest", type=Path)
    parser.add_argument("--replay-output-dir", type=Path)
    parser.add_argument("--listen-dir", type=Path)
    parser.add_argument("--output-label", default="02_drum_tonal_leakage_repaired.flac")
    parser.add_argument("--input-label", default="01_input_drums.flac")
    parser.add_argument("--validation-window", action="append", type=_parse_validation_window, default=[])
    parser.add_argument("--min-tonal-hz", type=float, default=TonalRepairConfig.min_tonal_hz)
    parser.add_argument("--max-tonal-hz", type=float, default=TonalRepairConfig.max_tonal_hz)
    parser.add_argument("--suppression-strength", type=float, default=TonalRepairConfig.suppression_strength)
    parser.add_argument("--min-tonal-gain", type=float, default=TonalRepairConfig.min_tonal_gain)
    parser.add_argument("--harmonic-ratio-floor", type=float, default=TonalRepairConfig.harmonic_ratio_floor)
    parser.add_argument("--transient-guard-strength", type=float, default=TonalRepairConfig.transient_guard_strength)
    return parser


def main() -> None:
    require_docker_runtime(
        tool_name="drum_tonal_leakage_repair.py",
        entrypoint_path="/app/tools/drum_tonal_leakage_repair.py",
    )
    args = _build_arg_parser().parse_args()
    if args.replay_manifest is not None:
        if args.replay_output_dir is None:
            raise ValueError("--replay-output-dir is required with --replay-manifest")
        manifest = replay_drum_tonal_leakage_repair(
            args.replay_manifest,
            args.replay_output_dir,
            listen_dir=args.listen_dir,
        )
        print(json.dumps({"output": manifest["output"], "replay": manifest["replay"]}, indent=2))
        return
    if args.drum_stem_path is None or args.output_dir is None:
        raise ValueError("drum_stem_path and output_dir are required outside replay mode")
    validation_windows = {label: window for label, window in args.validation_window}
    manifest = repair_drum_tonal_leakage(
        args.drum_stem_path,
        args.output_dir,
        listen_dir=args.listen_dir,
        output_label=str(args.output_label),
        input_label=str(args.input_label),
        validation_windows=validation_windows,
        config=TonalRepairConfig(
            min_tonal_hz=float(args.min_tonal_hz),
            max_tonal_hz=float(args.max_tonal_hz),
            suppression_strength=float(args.suppression_strength),
            min_tonal_gain=float(args.min_tonal_gain),
            harmonic_ratio_floor=float(args.harmonic_ratio_floor),
            transient_guard_strength=float(args.transient_guard_strength),
        ),
    )
    print(json.dumps({"output": manifest["output"], "metrics": manifest["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
