#!/usr/bin/env python3

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

SCRIPT_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = SCRIPT_ROOT.parent / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402


DEFAULT_EXCLUDED_STEMS = ("vocals", "vocal", "minus", "instrumental", "instrumentals")
PEAK_MODES = {"normalize", "clip"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_stem_audio(stem_path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(stem_path), always_2d=True)
    if audio.size == 0:
        raise RuntimeError(f"Stem audio is empty: {stem_path}")
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def parse_stem_weights(raw_weights: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for raw in raw_weights:
        if "=" not in str(raw):
            raise ValueError(f"Invalid stem weight {raw!r}; expected stem=value")
        stem_name, value_text = str(raw).split("=", 1)
        stem_name = stem_name.strip()
        if not stem_name:
            raise ValueError(f"Invalid stem weight {raw!r}; empty stem name")
        try:
            value = float(value_text)
        except ValueError as exc:
            raise ValueError(f"Invalid stem weight {raw!r}; value must be numeric") from exc
        if not math.isfinite(value) or value < 0.0 or value > 2.0:
            raise ValueError(f"Invalid stem weight {raw!r}; value must be finite and between 0.0 and 2.0")
        weights[stem_name] = value
    return weights


def resolve_manifest_stems(
    manifest: dict[str, Any],
    excluded_stems: set[str],
    stem_weights: dict[str, float] | None = None,
) -> list[tuple[str, Path]]:
    stem_weights = stem_weights or {}
    stems = manifest.get("stems")
    if not isinstance(stems, dict):
        raise RuntimeError("Manifest has no stems object")

    selected: list[tuple[str, Path]] = []
    for stem_name, stem_info in stems.items():
        if stem_name in excluded_stems:
            continue
        if not isinstance(stem_info, dict) or not stem_info.get("path"):
            raise RuntimeError(f"Stem entry has no path: {stem_name}")
        stem_path = Path(str(stem_info["path"]))
        if not stem_path.exists():
            raise FileNotFoundError(stem_path)
        selected.append((stem_name, stem_path))

    if not selected:
        raise RuntimeError("No stems selected for instrumental build")
    selected_names = {stem_name for stem_name, _stem_path in selected}
    missing_weight_stems = sorted(set(stem_weights) - selected_names)
    if missing_weight_stems:
        raise ValueError(f"Stem weight references missing stem(s): {', '.join(missing_weight_stems)}")
    return selected


def build_instrumental(
    stem_paths: list[tuple[str, Path]],
    *,
    stem_weights: dict[str, float] | None = None,
    peak_mode: str = "normalize",
) -> tuple[np.ndarray, int, dict[str, Any]]:
    if peak_mode not in PEAK_MODES:
        raise ValueError(f"Unsupported peak_mode {peak_mode!r}; expected one of {', '.join(sorted(PEAK_MODES))}")
    stem_weights = stem_weights or {}
    mix: np.ndarray | None = None
    sample_rate: int | None = None
    included_stems: list[str] = []
    source_paths: dict[str, str] = {}
    applied_weights: dict[str, float] = {}

    for stem_name, stem_path in stem_paths:
        audio, stem_sample_rate = read_stem_audio(stem_path)
        if mix is None:
            mix = np.zeros_like(audio, dtype=np.float32)
            sample_rate = stem_sample_rate
        elif stem_sample_rate != sample_rate:
            raise RuntimeError(
                f"Stem sample-rate mismatch: expected={sample_rate} actual={stem_sample_rate} stem={stem_name}"
            )
        elif audio.shape != mix.shape:
            raise RuntimeError(
                f"Stem shape mismatch: expected={mix.shape} actual={audio.shape} stem={stem_name}"
            )

        weight = float(stem_weights.get(stem_name, 1.0))
        mix = mix + audio * weight
        included_stems.append(stem_name)
        source_paths[stem_name] = str(stem_path)
        applied_weights[stem_name] = round(weight, 9)

    if mix is None or sample_rate is None:
        raise RuntimeError("No audio was mixed")

    peak_before = float(np.max(np.abs(mix))) if mix.size else 0.0
    normalization_gain = 1.0
    if peak_mode == "normalize" and peak_before > 1.0:
        normalization_gain = 1.0 / peak_before
        mix = mix * normalization_gain
    elif peak_mode == "clip":
        mix = np.clip(mix, -1.0, 1.0)

    summary = {
        "included_stems": included_stems,
        "source_paths": source_paths,
        "stem_weights": applied_weights,
        "peak_mode": peak_mode,
        "sample_rate": int(sample_rate),
        "frames": int(mix.shape[0]),
        "channels": int(mix.shape[1]) if mix.ndim == 2 else 1,
        "peak_before_normalization": round(peak_before, 9),
        "normalization_gain": round(normalization_gain, 9),
        "peak_after_normalization": round(float(np.max(np.abs(mix))) if mix.size else 0.0, 9),
    }
    return mix.astype(np.float32, copy=False), sample_rate, summary


def main() -> int:
    require_docker_runtime(
        tool_name="build_instrumental_from_stem_manifest.py",
        entrypoint_path="/app/scripts/build_instrumental_from_stem_manifest.py",
    )
    parser = argparse.ArgumentParser(description="Build an instrumental candidate by summing non-vocal manifest stems.")
    parser.add_argument("--stem-manifest", required=True, type=Path)
    parser.add_argument("--output-audio", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--exclude-stem", action="append", default=list(DEFAULT_EXCLUDED_STEMS))
    parser.add_argument("--stem-weight", action="append", default=[])
    parser.add_argument("--peak-mode", choices=sorted(PEAK_MODES), default="normalize")
    parser.add_argument("--subtype", default="PCM_24")
    args = parser.parse_args()

    if not args.stem_manifest.exists():
        raise FileNotFoundError(args.stem_manifest)

    excluded_stems = {str(stem).strip() for stem in args.exclude_stem if str(stem).strip()}
    stem_weights = parse_stem_weights(args.stem_weight)
    manifest = load_json(args.stem_manifest)
    selected_stems = resolve_manifest_stems(manifest, excluded_stems, stem_weights=stem_weights)
    instrumental, sample_rate, summary = build_instrumental(
        selected_stems,
        stem_weights=stem_weights,
        peak_mode=str(args.peak_mode),
    )

    args.output_audio.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output_audio), instrumental, sample_rate, subtype=str(args.subtype))

    report = {
        "schema_version": 1,
        "execution": {
            "tool": "ml_worker_improved/scripts/build_instrumental_from_stem_manifest.py",
            "intended_runtime": "docker ml-worker",
        },
        "inputs": {
            "stem_manifest": str(args.stem_manifest),
            "excluded_stems": sorted(excluded_stems),
            "stem_weights": dict(sorted(stem_weights.items())),
            "peak_mode": str(args.peak_mode),
        },
        "output": {
            "audio_path": str(args.output_audio),
            "subtype": str(args.subtype),
        },
        "instrumental": summary,
        "notes": [
            "This follows the current project convention used by _build_instrumental_mix_from_stage1_non_vocals: sum non-vocal stems and peak-normalize only when needed.",
        ],
    }

    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
