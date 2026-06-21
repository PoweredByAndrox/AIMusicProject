"""Build and validate synthetic stem contamination dataset rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.stem_contamination_common import validate_contamination_label_row, write_jsonl


def validate_stem_contamination_row(row: Mapping[str, Any]) -> None:
    validate_contamination_label_row(row)
    required = ("row_id", "input_stem", "target_leak", "base_stem", "foreign_source", "label_type")
    missing = [key for key in required if not row.get(key)]
    if missing:
        raise ValueError(f"missing contamination row fields: {', '.join(missing)}")


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), sample_rate)
    return path


def _gain_from_db(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def build_synthetic_contamination_pair(
    *,
    row_id: str,
    base_stem: str,
    base_path: Path,
    foreign_source: str,
    foreign_path: Path,
    output_dir: Path,
    contamination_db: float,
) -> dict[str, Any]:
    base_audio, base_rate = _read_audio(base_path)
    foreign_audio, foreign_rate = _read_audio(foreign_path)
    if base_rate != foreign_rate:
        raise ValueError(f"sample rate mismatch: {base_rate} != {foreign_rate}")
    min_len = min(len(base_audio), len(foreign_audio))
    if min_len <= 0:
        raise ValueError("cannot build contamination pair from empty audio")
    base_audio = base_audio[:min_len]
    foreign_audio = foreign_audio[:min_len]
    injected = foreign_audio * _gain_from_db(contamination_db)
    dirty = base_audio + injected

    input_path = _write_audio(output_dir / "audio" / f"{row_id}_in.flac", dirty, base_rate)
    leak_path = _write_audio(output_dir / "audio" / f"{row_id}_leak.flac", injected, base_rate)
    row = {
        "row_id": row_id,
        "input_stem": str(input_path),
        "target_leak": str(leak_path),
        "base_stem": base_stem,
        "foreign_source": foreign_source,
        "contamination_db": contamination_db,
        "label_type": "synthetic_clean",
        "is_ground_truth": True,
    }
    validate_stem_contamination_row(row)
    return row


def build_dataset_from_manifest(manifest_path: Path, output_dir: Path, contamination_db: float) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    pairs = manifest.get("pairs") or []
    rows = [
        build_synthetic_contamination_pair(
            row_id=f"d{idx + 1:06d}",
            base_stem=str(pair["base_stem"]),
            base_path=Path(pair["base_path"]),
            foreign_source=str(pair["foreign_source"]),
            foreign_path=Path(pair["foreign_path"]),
            output_dir=output_dir,
            contamination_db=contamination_db,
        )
        for idx, pair in enumerate(pairs)
    ]
    return write_jsonl(output_dir / "rows.jsonl", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--contamination-db", type=float, default=-35.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows_path = build_dataset_from_manifest(Path(args.manifest), Path(args.out), args.contamination_db)
    print(rows_path)


if __name__ == "__main__":
    main()
