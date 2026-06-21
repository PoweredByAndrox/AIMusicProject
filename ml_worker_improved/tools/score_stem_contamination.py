"""Score forbidden source contamination inside a generated stem."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


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


def score_stem_contamination(
    *,
    tested_stem_path: Path,
    tested_stem: str,
    foreign_source: str,
    output_dir: Path,
    foreign_reference_path: Path | None = None,
    detection_floor_db: float = -60.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{tested_stem}_{foreign_source}.json"

    if foreign_reference_path is None:
        result = {
            "tested_stem": tested_stem,
            "foreign_source": foreign_source,
            "verdict": "blocked_unknown",
            "leak_audio": "",
            "regions": [],
            "reason": "foreign_reference_required_for_fixture_scorer",
        }
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result

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
        detection_floor_db=args.detection_floor_db,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
