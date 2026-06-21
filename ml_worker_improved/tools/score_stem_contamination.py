"""Score forbidden source contamination inside a generated stem."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.stem_contamination_model import band_limited_audio, predict_no_reference_probability


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), sample_rate)
    return path


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))


def _dbfs(value: float) -> float:
    if value <= 1e-12:
        return -240.0
    return 20.0 * math.log10(value)


def _relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _projection_leak(stem_audio: np.ndarray, foreign_audio: np.ndarray) -> np.ndarray:
    min_len = min(len(stem_audio), len(foreign_audio))
    if min_len <= 0:
        return np.zeros((0, stem_audio.shape[1] if stem_audio.ndim == 2 else 1), dtype=np.float32)
    stem = stem_audio[:min_len].astype(np.float64)
    foreign = foreign_audio[:min_len].astype(np.float64)
    denom = float(np.sum(foreign * foreign))
    if denom <= 1e-12:
        return np.zeros_like(stem, dtype=np.float32)
    coeff = float(np.sum(stem * foreign) / denom)
    return (foreign * coeff).astype(np.float32)


def _band_limited_audio(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    return band_limited_audio(audio, sample_rate, low_hz, high_hz)


def _no_reference_vocal_in_minus_score(
    *,
    stem_audio: np.ndarray,
    sample_rate: int,
    detection_floor_db: float,
) -> tuple[str, np.ndarray, float]:
    leak_audio = _band_limited_audio(stem_audio, sample_rate, 700.0, 1200.0)
    leak_dbfs = _dbfs(_rms(leak_audio))
    verdict = "fail" if leak_dbfs > detection_floor_db else "pass"
    return verdict, leak_audio, leak_dbfs


def _blocked_unknown_result(
    *,
    tested_stem: str,
    foreign_source: str,
    result_path: Path,
    reason: str,
) -> dict[str, Any]:
    result = {
        "tested_stem": tested_stem,
        "foreign_source": foreign_source,
        "verdict": "blocked_unknown",
        "leak_audio": "",
        "regions": [],
        "reason": reason,
        "detector_mode": "none",
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _write_no_reference_result(
    *,
    tested_stem: str,
    foreign_source: str,
    output_dir: Path,
    result_path: Path,
    stem_rate: int,
    verdict: str,
    leak_audio: np.ndarray,
    leak_dbfs: float,
    detection_floor_db: float,
    detector_mode: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    leak_path = ""
    regions: list[dict[str, float]] = []
    if verdict == "fail":
        audio_path = _write_audio(output_dir / "leaks" / f"{tested_stem}_{foreign_source}.flac", leak_audio, stem_rate)
        leak_path = _relative_path(audio_path, output_dir)
        duration = float(len(leak_audio) / stem_rate)
        regions = [{"start": 0.0, "end": round(duration, 6)}]
    result = {
        "tested_stem": tested_stem,
        "foreign_source": foreign_source,
        "verdict": verdict,
        "leak_audio": leak_path,
        "regions": regions,
        "leak_dbfs": round(leak_dbfs, 4),
        "detection_floor_db": detection_floor_db,
        "detector_mode": detector_mode,
        "spectral_band_hz": [700.0, 1200.0],
    }
    if extra:
        result.update(extra)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def score_stem_contamination(
    *,
    tested_stem_path: Path,
    tested_stem: str,
    foreign_source: str,
    output_dir: Path,
    foreign_reference_path: Path | None = None,
    model_manifest_path: Path | None = None,
    detection_floor_db: float = -60.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{tested_stem}_{foreign_source}.json"

    if foreign_reference_path is None:
        if tested_stem != "minus" or foreign_source != "vocals":
            return _blocked_unknown_result(
                tested_stem=tested_stem,
                foreign_source=foreign_source,
                result_path=result_path,
                reason="no_reference_detector_not_available",
            )

        stem_audio, stem_rate = _read_audio(tested_stem_path)
        if model_manifest_path is not None:
            try:
                prediction = predict_no_reference_probability(
                    model_manifest_path=model_manifest_path,
                    audio=stem_audio,
                    sample_rate=stem_rate,
                    tested_stem=tested_stem,
                    foreign_source=foreign_source,
                )
            except Exception as exc:
                return _blocked_unknown_result(
                    tested_stem=tested_stem,
                    foreign_source=foreign_source,
                    result_path=result_path,
                    reason=f"no_reference_model_unavailable:{type(exc).__name__}",
                )
            leak_audio = _band_limited_audio(stem_audio, stem_rate, 700.0, 1200.0)
            leak_dbfs = _dbfs(_rms(leak_audio))
            probability = float(prediction["probability"])
            threshold = float(prediction["threshold"])
            verdict = "fail" if probability >= threshold else "pass"
            return _write_no_reference_result(
                tested_stem=tested_stem,
                foreign_source=foreign_source,
                output_dir=output_dir,
                result_path=result_path,
                stem_rate=stem_rate,
                verdict=verdict,
                leak_audio=leak_audio,
                leak_dbfs=leak_dbfs,
                detection_floor_db=detection_floor_db,
                detector_mode="no_reference_neural_v1",
                extra={
                    "contamination_probability": round(probability, 6),
                    "probability_threshold": threshold,
                    "model_id": prediction["model_id"],
                    "model_manifest": prediction["model_manifest"],
                    "features": prediction["features"],
                    "feature_names": prediction["feature_names"],
                    "shadow_only": True,
                },
            )

        verdict, leak_audio, leak_dbfs = _no_reference_vocal_in_minus_score(
            stem_audio=stem_audio,
            sample_rate=stem_rate,
            detection_floor_db=detection_floor_db,
        )
        return _write_no_reference_result(
            tested_stem=tested_stem,
            foreign_source=foreign_source,
            output_dir=output_dir,
            result_path=result_path,
            stem_rate=stem_rate,
            verdict=verdict,
            leak_audio=leak_audio,
            leak_dbfs=leak_dbfs,
            detection_floor_db=detection_floor_db,
            detector_mode="no_reference_spectral_v1",
        )

    stem_audio, stem_rate = _read_audio(tested_stem_path)
    foreign_audio, foreign_rate = _read_audio(foreign_reference_path)
    if stem_rate != foreign_rate:
        raise ValueError(f"sample rate mismatch: {stem_rate} != {foreign_rate}")

    leak_audio = _projection_leak(stem_audio, foreign_audio)
    leak_dbfs = _dbfs(_rms(leak_audio))
    verdict = "fail" if leak_dbfs > detection_floor_db else "pass"
    leak_path = ""
    regions: list[dict[str, float]] = []
    if verdict == "fail":
        audio_path = _write_audio(output_dir / "leaks" / f"{tested_stem}_{foreign_source}.flac", leak_audio, stem_rate)
        leak_path = _relative_path(audio_path, output_dir)
        duration = float(len(leak_audio) / stem_rate)
        regions = [{"start": 0.0, "end": round(duration, 6)}]

    result = {
        "tested_stem": tested_stem,
        "foreign_source": foreign_source,
        "verdict": verdict,
        "leak_audio": leak_path,
        "regions": regions,
        "leak_dbfs": round(leak_dbfs, 4),
        "detection_floor_db": detection_floor_db,
        "detector_mode": "reference_projection_v1",
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--tested-stem", required=True)
    parser.add_argument("--foreign-source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--foreign-reference")
    parser.add_argument("--model-manifest")
    parser.add_argument("--detection-floor-db", type=float, default=-60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = score_stem_contamination(
        tested_stem_path=Path(args.stem),
        tested_stem=args.tested_stem,
        foreign_source=args.foreign_source,
        output_dir=Path(args.out),
        foreign_reference_path=Path(args.foreign_reference) if args.foreign_reference else None,
        model_manifest_path=Path(args.model_manifest) if args.model_manifest else None,
        detection_floor_db=args.detection_floor_db,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
