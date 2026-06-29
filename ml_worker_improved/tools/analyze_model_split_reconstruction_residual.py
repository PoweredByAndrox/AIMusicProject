#!/usr/bin/env python3
"""Localize model-split bad-reconstruction residuals into concrete stems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from tools.stem_contamination_common import MINUS_STEM, STEMS, normalize_stem_name
    from tools.stem_region_repair_common import ISSUE_BAD_RECONSTRUCTION, read_audio_stereo, write_json, write_jsonl
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from stem_contamination_common import MINUS_STEM, STEMS, normalize_stem_name
    from stem_region_repair_common import ISSUE_BAD_RECONSTRUCTION, read_audio_stereo, write_json, write_jsonl


EPSILON = 1.0e-8
DELIVERY_PAIR_STEMS = ("vocals", MINUS_STEM)


def _fit_like(audio: np.ndarray, frames: int) -> np.ndarray:
    if audio.shape[0] == frames:
        return audio.astype(np.float32)
    if audio.shape[0] > frames:
        return audio[:frames].astype(np.float32)
    pad = np.zeros((frames - audio.shape[0], audio.shape[1]), dtype=np.float32)
    return np.concatenate([audio.astype(np.float32), pad], axis=0)


def _read_stem_paths(candidate: Mapping[str, Any]) -> dict[str, Path]:
    stem_paths = candidate.get("stem_paths")
    if not isinstance(stem_paths, Mapping):
        return {}
    result: dict[str, Path] = {}
    for stem, path in stem_paths.items():
        normalized = normalize_stem_name(stem)
        if normalized in set(STEMS) | {MINUS_STEM} and path:
            result[normalized] = Path(str(path))
    return result


def _route_signature(candidate: Mapping[str, Any]) -> str:
    composition = candidate.get("target_specific_composition")
    if not isinstance(composition, Mapping):
        return ""
    routes = composition.get("routes")
    if not isinstance(routes, list):
        return ""
    parts: list[str] = []
    for route in routes:
        if not isinstance(route, Mapping):
            continue
        lane = str(route.get("lane") or "").strip()
        route_id = str(route.get("route_id") or route.get("route_alias") or "").strip()
        if lane or route_id:
            parts.append(f"{lane}:{route_id}")
    return "|".join(parts)


def _candidate_mix_path(candidate: Mapping[str, Any], stem_paths: Mapping[str, Path]) -> Path | None:
    for key in ("mix_path", "window_mix_path", "source_mix_path"):
        if candidate.get(key):
            return Path(str(candidate[key]))
    for stem_path in stem_paths.values():
        for parent in [stem_path.parent, *stem_path.parents]:
            candidate = parent / "mix.flac"
            if candidate.exists():
                return candidate
    return None


def _load_audio_map(paths: Mapping[str, Path]) -> tuple[dict[str, np.ndarray], int]:
    audio_map: dict[str, np.ndarray] = {}
    sample_rate = 0
    for stem, path in paths.items():
        if not path.exists():
            continue
        audio, rate = read_audio_stereo(path)
        if sample_rate and int(rate) != int(sample_rate):
            raise ValueError(f"sample rate mismatch for stem {stem}: {rate} != {sample_rate}")
        sample_rate = int(rate)
        audio_map[stem] = audio
    return audio_map, sample_rate


def _sum_stems(audio_map: Mapping[str, np.ndarray], stems: tuple[str, ...] | list[str], frames: int) -> np.ndarray | None:
    present = [stem for stem in stems if stem in audio_map]
    if not present:
        return None
    total = np.zeros((frames, 2), dtype=np.float32)
    for stem in present:
        total += _fit_like(audio_map[stem], frames)
    return total


def _projection_scores(residual: np.ndarray, audio_map: Mapping[str, np.ndarray], frames: int) -> dict[str, float]:
    residual = residual[:frames].astype(np.float32)
    residual_norm = float(np.sqrt(np.sum(np.square(residual)))) + EPSILON
    scores: dict[str, float] = {}
    for stem in list(STEMS) + [MINUS_STEM]:
        if stem not in audio_map:
            continue
        audio = _fit_like(audio_map[stem], frames)
        audio_norm = float(np.sqrt(np.sum(np.square(audio)))) + EPSILON
        scores[stem] = float(abs(np.sum(residual * audio)) / (residual_norm * audio_norm))
    return dict(sorted(scores.items(), key=lambda item: item[0]))


def _region_from_residual(residual: np.ndarray, sample_rate: int) -> dict[str, float]:
    energy = np.sqrt(np.mean(np.square(residual), axis=1)).astype(np.float32)
    if energy.size == 0 or float(np.max(energy)) <= EPSILON:
        return {"start_sec": 0.0, "end_sec": 0.0}
    threshold = max(float(np.max(energy)) * 0.25, float(np.mean(energy)) + float(np.std(energy)))
    active = np.flatnonzero(energy >= threshold)
    if active.size == 0:
        active = np.flatnonzero(energy > 0.0)
    if active.size == 0:
        return {"start_sec": 0.0, "end_sec": 0.0}
    return {
        "start_sec": round(float(active[0]) / float(sample_rate), 6),
        "end_sec": round(float(active[-1] + 1) / float(sample_rate), 6),
    }


def localize_candidate_reconstruction_residual(
    candidate: Mapping[str, Any],
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    stem_paths = _read_stem_paths(candidate)
    mix_path = _candidate_mix_path(candidate, stem_paths)
    base = {
        "schema": "model_split_reconstruction_residual.v1",
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "window_id": str(candidate.get("window_id") or ""),
        "issue_type": ISSUE_BAD_RECONSTRUCTION,
        "route_family": _route_signature(candidate),
        "affected_stem": "",
        "residual_region": {"start_sec": 0.0, "end_sec": 0.0},
        "projection_scores": {},
    }
    if mix_path is None or not mix_path.exists() or not stem_paths:
        result = {**base, "verdict": "unavailable", "reason": "missing_mix_or_stems"}
        if output_dir is not None:
            write_json(output_dir / "loc.json", result)
        return result

    mix, sample_rate = read_audio_stereo(mix_path)
    audio_map, stem_rate = _load_audio_map(stem_paths)
    if not audio_map or int(stem_rate) != int(sample_rate):
        result = {**base, "verdict": "unavailable", "reason": "missing_audio_or_sample_rate_mismatch"}
        if output_dir is not None:
            write_json(output_dir / "loc.json", result)
        return result
    frames = min([mix.shape[0], *[audio.shape[0] for audio in audio_map.values()]])
    mix = mix[:frames]

    delivery_sum = _sum_stems(audio_map, list(DELIVERY_PAIR_STEMS), frames)
    six_sum = _sum_stems(audio_map, STEMS, frames)
    residual_surface = "delivery_pair"
    reconstructed = delivery_sum
    if reconstructed is None and six_sum is not None:
        residual_surface = "six_stem_sum"
        reconstructed = six_sum
    if reconstructed is None:
        result = {**base, "verdict": "unavailable", "reason": "missing_reconstruction_surface"}
        if output_dir is not None:
            write_json(output_dir / "loc.json", result)
        return result

    residual = (mix - reconstructed).astype(np.float32)
    residual_rms = float(np.sqrt(np.mean(np.square(residual))))
    mix_rms = float(np.sqrt(np.mean(np.square(mix)))) + EPSILON
    scores = _projection_scores(residual, audio_map, frames)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    affected_stem = ranked[0][0] if ranked and ranked[0][1] > 0.05 and residual_rms > EPSILON else ""
    verdict = "localized" if affected_stem else "unlocalized"
    result = {
        **base,
        "verdict": verdict,
        "reason": "" if affected_stem else "no_confident_projection",
        "affected_stem": affected_stem,
        "residual_surface": residual_surface,
        "residual_region": _region_from_residual(residual, sample_rate),
        "residual_rms": residual_rms,
        "relative_residual_rms": float(residual_rms / mix_rms),
        "projection_scores": scores,
    }
    if output_dir is not None:
        write_json(output_dir / "loc.json", result)
    return result


def analyze_run_reconstruction_residuals(
    *,
    run_json_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    payload = json.loads(Path(run_json_path).read_text(encoding="utf-8"))
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("run_json must contain candidates list")
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if "reconstruction" not in str(candidate.get("verdict") or "") and "reconstruction" not in json.dumps(candidate):
            continue
        rows.append(localize_candidate_reconstruction_residual(candidate))
    manifest = {
        "schema": "model_split_reconstruction_residual_manifest.v1",
        "source_run_json": str(run_json_path),
        "row_count": len(rows),
        "localized_count": sum(1 for row in rows if row.get("verdict") == "localized"),
        "rows_jsonl": str(output_dir / "rows.jsonl"),
        "delivery_allowed": False,
    }
    write_jsonl(output_dir / "rows.jsonl", rows)
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest = analyze_run_reconstruction_residuals(run_json_path=args.run_json, output_dir=args.out)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
