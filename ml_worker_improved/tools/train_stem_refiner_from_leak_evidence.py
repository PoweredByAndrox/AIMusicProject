#!/usr/bin/env python3
"""Train a stem-region refiner from blocked candidate leak evidence.

This is intentionally pseudo-teacher training: recovered leak audio is a repair
signal, not verified ground truth and not usable for promotion by itself.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from tools.stem_region_repair_common import make_region_mask, read_audio_stereo, validate_issue_type, write_audio, write_json
    from tools.stem_region_repair_model import (
        StemRegionRepairTrainConfig,
        apply_region_repair_model,
        save_region_repair_checkpoint,
        train_region_repair_model_from_arrays,
        write_training_history,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from stem_region_repair_common import make_region_mask, read_audio_stereo, validate_issue_type, write_audio, write_json
    from stem_region_repair_model import (
        StemRegionRepairTrainConfig,
        apply_region_repair_model,
        save_region_repair_checkpoint,
        train_region_repair_model_from_arrays,
        write_training_history,
    )


def _fit_like(audio: np.ndarray, frames: int) -> np.ndarray:
    if audio.shape[0] == frames:
        return audio.astype(np.float32)
    if audio.shape[0] > frames:
        return audio[:frames].astype(np.float32)
    pad = np.zeros((frames - audio.shape[0], audio.shape[1]), dtype=np.float32)
    return np.concatenate([audio.astype(np.float32), pad], axis=0)


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))


def _region_mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    weights = mask[:, None].astype(np.float32)
    denom = max(float(np.sum(weights)) * float(a.shape[1]), 1.0)
    return float(np.sum(np.square((a - b) * weights)) / denom)


def _outside_delta_mse(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> float:
    weights = (1.0 - mask[:, None]).astype(np.float32)
    denom = max(float(np.sum(weights)) * float(before.shape[1]), 1.0)
    return float(np.sum(np.square((after - before) * weights)) / denom)


def _band_limited_audio(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    if len(audio) <= 0:
        return np.zeros_like(audio, dtype=np.float32)
    working = audio.astype(np.float64)
    spectrum = np.fft.rfft(working, axis=0)
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / float(sample_rate))
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    filtered = np.zeros_like(spectrum)
    filtered[mask, :] = spectrum[mask, :]
    return np.fft.irfft(filtered, n=len(audio), axis=0).astype(np.float32)


def _build_forensic_envelope_target(
    *,
    bad_audio: np.ndarray,
    source_audio: np.ndarray,
    sample_rate: int,
    regions: list[dict[str, Any]],
    attenuation: float,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    if bad_audio.shape != source_audio.shape:
        raise ValueError("forensic envelope target shape mismatch")
    bad_hf = _band_limited_audio(bad_audio, sample_rate, low_hz, high_hz)
    source_hf = _band_limited_audio(source_audio, sample_rate, low_hz, high_hz)
    source_env = np.sqrt(np.mean(np.square(source_hf.astype(np.float32)), axis=1)).astype(np.float32)
    peak = float(np.percentile(source_env, 95.0)) if source_env.size else 0.0
    if peak <= 1.0e-9:
        return bad_audio.astype(np.float32)
    source_mask = np.clip(source_env / peak, 0.0, 1.0)
    region_mask = np.zeros((bad_audio.shape[0],), dtype=np.float32)
    duration_sec = bad_audio.shape[0] / float(sample_rate)
    for region in regions:
        if not isinstance(region, dict):
            continue
        start_sec = max(0.0, min(duration_sec, float(region.get("start", 0.0))))
        end_sec = max(start_sec, min(duration_sec, float(region.get("end", duration_sec))))
        start = int(round(start_sec * sample_rate))
        end = int(round(end_sec * sample_rate))
        region_mask[start:end] = 1.0
    if float(np.max(region_mask)) <= 0.0:
        region_mask[:] = 1.0
    attenuation_curve = np.clip(float(attenuation), 0.0, 1.0) * source_mask * region_mask
    smooth_frames = max(1, int(round(0.02 * float(sample_rate))))
    if smooth_frames > 1:
        kernel = np.hanning((smooth_frames * 2) + 1).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0.0:
            kernel /= kernel_sum
            attenuation_curve = np.convolve(attenuation_curve, kernel, mode="same").astype(np.float32)
    target = bad_audio.astype(np.float32) - (bad_hf.astype(np.float32) * attenuation_curve[:, None])
    return np.clip(target, -1.0, 1.0).astype(np.float32)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _host_path(path: str, repo_root: Path | None = None) -> Path:
    raw = str(path or "").strip()
    if raw.startswith("/repo/") and repo_root is not None:
        return repo_root / raw.removeprefix("/repo/")
    return Path(raw)


def _candidate_repo_root(candidate_dir: Path) -> Path | None:
    parts = list(candidate_dir.resolve().parts)
    if "artifacts" not in parts:
        return None
    idx = parts.index("artifacts")
    return Path(*parts[:idx]) if idx > 0 else None


def _candidate_stem_path(candidate_dir: Path, manifest: dict[str, Any], stem: str) -> Path:
    repo_root = _candidate_repo_root(candidate_dir)
    stem_paths = manifest.get("stem_paths")
    if isinstance(stem_paths, dict) and stem_paths.get(stem):
        path = _host_path(str(stem_paths[stem]), repo_root)
        if path.is_file():
            return path
    path = candidate_dir / "stems" / f"{stem}.flac"
    if not path.is_file():
        raise FileNotFoundError(f"missing candidate stem: {path}")
    return path


def _find_leak_results(
    *,
    score: dict[str, Any],
    stem: str,
    foreign_sources: set[str],
) -> list[dict[str, Any]]:
    contamination = score.get("contamination_score")
    leak_results = contamination.get("leak_results") if isinstance(contamination, dict) else None
    if not isinstance(leak_results, list):
        return []
    found: list[dict[str, Any]] = []
    for item in leak_results:
        if not isinstance(item, dict):
            continue
        if str(item.get("tested_stem") or "") != stem:
            continue
        if str(item.get("foreign_source") or "") not in foreign_sources:
            continue
        if str(item.get("verdict") or "") != "fail":
            continue
        leak_audio = str(item.get("leak_audio") or "").strip()
        if leak_audio:
            found.append(item)
    return found


def _load_leak_sum(
    *,
    candidate_dir: Path,
    leak_results: list[dict[str, Any]],
    frames: int,
    sample_rate: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    leak_sum = np.zeros((frames, 2), dtype=np.float32)
    evidence: list[dict[str, Any]] = []
    for item in leak_results:
        leak_path = candidate_dir / "score" / str(item["leak_audio"])
        if not leak_path.is_file():
            raise FileNotFoundError(f"missing leak audio: {leak_path}")
        leak_audio, leak_rate = read_audio_stereo(leak_path)
        if int(leak_rate) != int(sample_rate):
            raise ValueError(f"sample rate mismatch for leak audio: {leak_path}")
        leak_audio = _fit_like(leak_audio, frames)
        leak_sum += leak_audio
        evidence.append(
            {
                "tested_stem": str(item.get("tested_stem") or ""),
                "foreign_source": str(item.get("foreign_source") or ""),
                "leak_audio": str(leak_path),
                "leak_dbfs": item.get("leak_dbfs"),
                "regions": item.get("regions") or [],
            }
        )
    return leak_sum.astype(np.float32), evidence


def _forensic_proxy_result(*, forensic_pair_json: Path, stem: str, foreign_sources: set[str]) -> dict[str, Any]:
    pair = _load_json(forensic_pair_json)
    tested_stem = str(pair.get("tested_stem") or "").strip()
    source_stem = str(pair.get("source_stem") or "").strip()
    if tested_stem != stem:
        raise ValueError(f"forensic pair tested stem mismatch: {tested_stem}!={stem}")
    if source_stem not in foreign_sources:
        raise ValueError(f"forensic pair source stem {source_stem} is not in requested foreign sources")
    if str(pair.get("verdict") or "") != "fail":
        raise ValueError("forensic pair must be a failed pair")
    proxy_audio = str(pair.get("forensic_proxy_audio") or "").strip()
    if not proxy_audio:
        raise ValueError("forensic pair is missing forensic_proxy_audio")
    return {
        "tested_stem": tested_stem,
        "foreign_source": source_stem,
        "verdict": "fail",
        "leak_audio": str(forensic_pair_json.parent / proxy_audio),
        "source_path": str(pair.get("source_path") or ""),
        "leak_dbfs": pair.get("tested_hf_relative_db"),
        "regions": pair.get("regions") or [],
        "evidence_type": "cross_stem_forensic_proxy",
    }


def _region_from_leaks(leak_results: list[dict[str, Any]], duration_sec: float) -> dict[str, float]:
    starts: list[float] = []
    ends: list[float] = []
    for item in leak_results:
        regions = item.get("regions")
        if not isinstance(regions, list):
            continue
        for region in regions:
            if not isinstance(region, dict):
                continue
            starts.append(max(0.0, float(region.get("start", 0.0))))
            ends.append(min(duration_sec, float(region.get("end", duration_sec))))
    if not starts or not ends:
        return {"start_sec": 0.0, "end_sec": duration_sec}
    return {"start_sec": min(starts), "end_sec": max(ends)}


def _sibling_sum(candidate_dir: Path, manifest: dict[str, Any], *, stem: str, frames: int, sample_rate: int) -> np.ndarray | None:
    stem_paths = manifest.get("stem_paths")
    if not isinstance(stem_paths, dict):
        return None
    repo_root = _candidate_repo_root(candidate_dir)
    total: np.ndarray | None = None
    for sibling, raw_path in sorted(stem_paths.items()):
        if sibling == stem or sibling == "minus":
            continue
        path = _host_path(str(raw_path), repo_root)
        if not path.is_file():
            continue
        audio, rate = read_audio_stereo(path)
        if int(rate) != int(sample_rate):
            raise ValueError(f"sample rate mismatch for sibling stem: {path}")
        audio = _fit_like(audio, frames)
        total = audio if total is None else total + audio
    return total.astype(np.float32) if total is not None else None


def _stem_audio_or_none(
    candidate_dir: Path,
    manifest: dict[str, Any],
    *,
    stem: str,
    frames: int,
    sample_rate: int,
) -> np.ndarray | None:
    try:
        path = _candidate_stem_path(candidate_dir, manifest, stem)
    except FileNotFoundError:
        return None
    audio, rate = read_audio_stereo(path)
    if int(rate) != int(sample_rate):
        raise ValueError(f"sample rate mismatch for pair stem: {path}")
    return _fit_like(audio, frames)


def _mix_conservation_sibling(
    candidate_dir: Path,
    manifest: dict[str, Any],
    *,
    stem: str,
    frames: int,
    sample_rate: int,
) -> tuple[np.ndarray | None, str, str]:
    delivery_pair_sibling = {"vocals": "minus", "minus": "vocals"}.get(stem)
    if delivery_pair_sibling:
        pair_audio = _stem_audio_or_none(
            candidate_dir,
            manifest,
            stem=delivery_pair_sibling,
            frames=frames,
            sample_rate=sample_rate,
        )
        if pair_audio is not None:
            return pair_audio.astype(np.float32), "delivery_pair", delivery_pair_sibling
    siblings = _sibling_sum(candidate_dir, manifest, stem=stem, frames=frames, sample_rate=sample_rate)
    return siblings, "six_stem_sum" if siblings is not None else "", ""


def _forensic_source_audio(
    *,
    candidate_dir: Path,
    manifest: dict[str, Any],
    leak_result: dict[str, Any],
    frames: int,
    sample_rate: int,
) -> tuple[np.ndarray | None, str]:
    source_stem = str(leak_result.get("foreign_source") or "").strip()
    if not source_stem:
        return None, ""
    repo_root = _candidate_repo_root(candidate_dir)
    source_path = Path()
    raw_source_path = str(leak_result.get("source_path") or "").strip()
    if raw_source_path:
        source_path = _host_path(raw_source_path, repo_root)
    if not source_path.is_file():
        try:
            source_path = _candidate_stem_path(candidate_dir, manifest, source_stem)
        except FileNotFoundError:
            return None, ""
    source_audio, source_rate = read_audio_stereo(source_path)
    if int(source_rate) != int(sample_rate):
        raise ValueError(f"sample rate mismatch for forensic source audio: {source_path}")
    return _fit_like(source_audio, frames), str(source_path)


def _load_leak_training_example(
    *,
    candidate_dir: Path,
    stem: str,
    foreign_source_set: set[str],
    leak_gain: float,
    config: StemRegionRepairTrainConfig,
    forensic_pair_json: Path | None,
    forensic_envelope_target: bool,
    forensic_envelope_attenuation: float,
    forensic_high_low_hz: float,
    forensic_high_high_hz: float,
) -> dict[str, Any]:
    manifest = _load_json(candidate_dir / "manifest.json")
    score = _load_json(candidate_dir / "score.json")
    stem_path = _candidate_stem_path(candidate_dir, manifest, stem)
    bad, sample_rate = read_audio_stereo(stem_path)
    frames = bad.shape[0]
    mix_path = candidate_dir.parent / "mix.flac"
    if not mix_path.is_file():
        raise FileNotFoundError(f"missing window mix: {mix_path}")
    mix, mix_rate = read_audio_stereo(mix_path)
    if int(mix_rate) != int(sample_rate):
        raise ValueError("sample rate mismatch between mix and stem")
    mix = _fit_like(mix, frames)

    if forensic_pair_json is not None:
        leak_results = [
            _forensic_proxy_result(
                forensic_pair_json=forensic_pair_json,
                stem=stem,
                foreign_sources=foreign_source_set,
            )
        ]
    else:
        leak_results = _find_leak_results(score=score, stem=stem, foreign_sources=foreign_source_set)
    if not leak_results:
        raise ValueError(f"no failed leak evidence for {stem}:{sorted(foreign_source_set)} in {candidate_dir}")
    if forensic_pair_json is not None:
        leak_sum = np.zeros((frames, 2), dtype=np.float32)
        leak_evidence: list[dict[str, Any]] = []
        for item in leak_results:
            leak_path = Path(str(item["leak_audio"]))
            if not leak_path.is_file():
                raise FileNotFoundError(f"missing forensic proxy audio: {leak_path}")
            leak_audio, leak_rate = read_audio_stereo(leak_path)
            if int(leak_rate) != int(sample_rate):
                raise ValueError(f"sample rate mismatch for forensic proxy audio: {leak_path}")
            leak_audio = _fit_like(leak_audio, frames)
            leak_sum += leak_audio
            leak_evidence.append(
                {
                    "tested_stem": str(item.get("tested_stem") or ""),
                    "foreign_source": str(item.get("foreign_source") or ""),
                    "leak_audio": str(leak_path),
                    "leak_dbfs": item.get("leak_dbfs"),
                    "regions": item.get("regions") or [],
                    "evidence_type": "cross_stem_forensic_proxy",
                    "candidate_dir": str(candidate_dir),
                }
            )
    else:
        leak_sum, leak_evidence = _load_leak_sum(
            candidate_dir=candidate_dir,
            leak_results=leak_results,
            frames=frames,
            sample_rate=sample_rate,
        )
        for item in leak_evidence:
            item["candidate_dir"] = str(candidate_dir)
    region = _region_from_leaks(leak_results, frames / float(sample_rate))
    use_source_condition = (
        float(config.forensic_source_decorrelation_loss_weight) > 0.0
        or float(config.forensic_source_envelope_loss_weight) > 0.0
        or bool(config.forensic_source_conditioning)
    )
    forensic_source, forensic_source_path = (
        _forensic_source_audio(
            candidate_dir=candidate_dir,
            manifest=manifest,
            leak_result=leak_results[0],
            frames=frames,
            sample_rate=sample_rate,
        )
        if use_source_condition
        else (None, "")
    )
    if forensic_pair_json is not None and forensic_envelope_target:
        target = _build_forensic_envelope_target(
            bad_audio=bad,
            source_audio=leak_sum,
            sample_rate=sample_rate,
            regions=leak_results[0].get("regions") if isinstance(leak_results[0].get("regions"), list) else [],
            attenuation=forensic_envelope_attenuation,
            low_hz=forensic_high_low_hz,
            high_hz=forensic_high_high_hz,
        )
    else:
        target = np.clip(bad - (float(leak_gain) * leak_sum), -1.0, 1.0).astype(np.float32)
    mask = make_region_mask(
        frame_count=frames,
        sample_rate=sample_rate,
        start_sec=float(region["start_sec"]),
        end_sec=float(region["end_sec"]),
    )
    siblings, reconstruction_surface, mix_preserve_companion_stem = _mix_conservation_sibling(
        candidate_dir,
        manifest,
        stem=stem,
        frames=frames,
        sample_rate=sample_rate,
    )
    return {
        "candidate_dir": candidate_dir,
        "sample_rate": int(sample_rate),
        "frames": int(frames),
        "mix": mix.astype(np.float32),
        "bad": bad.astype(np.float32),
        "target": target.astype(np.float32),
        "mask": mask.astype(np.float32),
        "leak_sum": leak_sum.astype(np.float32),
        "leak_evidence": leak_evidence,
        "region": region,
        "siblings": siblings.astype(np.float32) if siblings is not None else None,
        "reconstruction_surface": reconstruction_surface,
        "mix_preserve_companion_stem": mix_preserve_companion_stem,
        "forensic_source": forensic_source.astype(np.float32) if forensic_source is not None else None,
        "forensic_source_path": forensic_source_path,
    }


def _concat_examples(examples: Sequence[dict[str, Any]], key: str, *, pad_frames: int) -> np.ndarray | None:
    chunks: list[np.ndarray] = []
    for index, example in enumerate(examples):
        audio = example.get(key)
        if audio is None:
            return None
        chunks.append(np.asarray(audio, dtype=np.float32))
        if index != len(examples) - 1 and pad_frames > 0:
            chunks.append(np.zeros((pad_frames, 2), dtype=np.float32))
    return np.concatenate(chunks, axis=0).astype(np.float32) if chunks else None


def _concat_masks(examples: Sequence[dict[str, Any]], *, pad_frames: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for index, example in enumerate(examples):
        chunks.append(np.asarray(example["mask"], dtype=np.float32))
        if index != len(examples) - 1 and pad_frames > 0:
            chunks.append(np.zeros((pad_frames,), dtype=np.float32))
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _train_from_leak_examples(
    *,
    examples: list[dict[str, Any]],
    output_dir: Path,
    stem: str,
    issue_type: str,
    foreign_source_set: set[str],
    leak_gain: float,
    config: StemRegionRepairTrainConfig,
    forensic_envelope_target: bool,
    forensic_envelope_attenuation: float,
    forensic_high_low_hz: float,
    forensic_high_high_hz: float,
    transfer_leak_to_foreign_source: bool,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("at least one leak training example is required")
    sample_rate = int(examples[0]["sample_rate"])
    for example in examples:
        if int(example["sample_rate"]) != sample_rate:
            raise ValueError("batch leak evidence sample-rate mismatch")
    pad_frames = max(0, int(round(0.05 * float(sample_rate)))) if len(examples) > 1 else 0
    mix = _concat_examples(examples, "mix", pad_frames=pad_frames)
    bad = _concat_examples(examples, "bad", pad_frames=pad_frames)
    target = _concat_examples(examples, "target", pad_frames=pad_frames)
    leak_sum = _concat_examples(examples, "leak_sum", pad_frames=pad_frames)
    mask = _concat_masks(examples, pad_frames=pad_frames)
    siblings = _concat_examples(examples, "siblings", pad_frames=pad_frames)
    forensic_source = _concat_examples(examples, "forensic_source", pad_frames=pad_frames)
    if mix is None or bad is None or target is None or leak_sum is None:
        raise ValueError("batch leak evidence audio assembly failed")

    primary = examples[0]
    region = dict(primary["region"])
    reconstruction_surface = str(primary.get("reconstruction_surface") or "")
    mix_preserve_companion_stem = str(primary.get("mix_preserve_companion_stem") or "")
    forensic_source_path = str(primary.get("forensic_source_path") or "")
    leak_evidence: list[dict[str, Any]] = []
    for example in examples:
        leak_evidence.extend(example["leak_evidence"])

    before_mse = _region_mse(bad, target, mask)
    model, history = train_region_repair_model_from_arrays(
        mix_audio=mix,
        bad_stem_audio=bad,
        target_stem_audio=target,
        sibling_stem_audio=siblings,
        target_mix_audio=mix if siblings is not None else None,
        forensic_source_audio=forensic_source,
        region_mask=mask,
        sample_rate=sample_rate,
        issue_type=issue_type,
        config=config,
    )
    repaired, diagnostics = apply_region_repair_model(
        model=model,
        mix_audio=mix,
        bad_stem_audio=bad,
        region_mask=mask,
        sample_rate=sample_rate,
        issue_type=issue_type,
        source_condition_audio=forensic_source if bool(config.forensic_source_conditioning) else None,
    )
    after_mse = _region_mse(repaired, target, mask)
    outside_mse = _outside_delta_mse(bad, repaired, mask)

    first_bad = np.asarray(primary["bad"], dtype=np.float32)
    first_target = np.asarray(primary["target"], dtype=np.float32)
    first_mask = np.asarray(primary["mask"], dtype=np.float32)
    first_mix = np.asarray(primary["mix"], dtype=np.float32)
    first_source = np.asarray(primary["forensic_source"], dtype=np.float32) if primary.get("forensic_source") is not None else None
    first_repaired, first_diagnostics = apply_region_repair_model(
        model=model,
        mix_audio=first_mix,
        bad_stem_audio=first_bad,
        region_mask=first_mask,
        sample_rate=sample_rate,
        issue_type=issue_type,
        source_condition_audio=first_source if bool(config.forensic_source_conditioning) else None,
    )
    first_before_mse = _region_mse(first_bad, first_target, first_mask)
    first_after_mse = _region_mse(first_repaired, first_target, first_mask)

    audio_dir = output_dir / "audio"
    bad_out = write_audio(audio_dir / "bad.flac", first_bad, sample_rate)
    leak_out = write_audio(audio_dir / "leak.flac", np.asarray(primary["leak_sum"], dtype=np.float32), sample_rate)
    target_out = write_audio(audio_dir / "pseudo.flac", first_target, sample_rate)
    repaired_out = write_audio(audio_dir / "repaired.flac", first_repaired, sample_rate)

    transfer_companion_stem = ""
    if transfer_leak_to_foreign_source and not mix_preserve_companion_stem and len(foreign_source_set) == 1:
        transfer_companion_stem = sorted(foreign_source_set)[0]
        mix_preserve_companion_stem = transfer_companion_stem
    history = {
        **history,
        "region": region,
        "training_row": {
            "row_id": "leak_ev_000001",
            "target_stem": stem,
            "root_cause_row_id": "",
            "reconstruction_surface": reconstruction_surface or "unavailable",
            "target_surface": "pseudo_teacher_minus_leak",
        },
        "training_rows": [
            {
                "row_id": f"leak_ev_{index + 1:06d}",
                "candidate_dir": str(example["candidate_dir"]),
                "target_stem": stem,
                "root_cause_row_id": "",
                "reconstruction_surface": str(example.get("reconstruction_surface") or "unavailable"),
                "target_surface": "pseudo_teacher_minus_leak",
            }
            for index, example in enumerate(examples)
        ],
        "training_example_count": len(examples),
        "label_type": "pseudo_teacher",
        "is_ground_truth": False,
        "leak_gain": float(leak_gain),
        "forensic_envelope_target": bool(forensic_envelope_target),
        "forensic_envelope_attenuation": float(forensic_envelope_attenuation),
        "forensic_high_band_hz": [float(forensic_high_low_hz), float(forensic_high_high_hz)],
        "forensic_source_decorrelation_loss_weight": float(config.forensic_source_decorrelation_loss_weight),
        "forensic_source_envelope_loss_weight": float(config.forensic_source_envelope_loss_weight),
        "forensic_envelope_window_ms": float(config.forensic_envelope_window_ms),
        "forensic_source_conditioning": bool(config.forensic_source_conditioning),
        "forensic_source_audio_path": forensic_source_path,
    }
    if mix_preserve_companion_stem:
        history["mix_preserve_companion_stem"] = mix_preserve_companion_stem
    if transfer_companion_stem:
        history["leak_transfer_to_foreign_source"] = True
    if bool(config.forensic_source_conditioning) and len(foreign_source_set) == 1:
        history["forensic_source_condition_stem"] = sorted(foreign_source_set)[0]
    checkpoint_path = save_region_repair_checkpoint(model=model, path=output_dir / "model" / "model.pt", training_history=history)
    history_path = write_training_history(output_dir / "history.json", history)
    result = {
        "schema": "stem_refiner_leak_evidence_training.v1",
        "candidate_dir": str(examples[0]["candidate_dir"]),
        "candidate_dirs": [str(example["candidate_dir"]) for example in examples],
        "training_example_count": len(examples),
        "stem": stem,
        "issue_type": issue_type,
        "foreign_sources": sorted(foreign_source_set),
        "label_type": "pseudo_teacher",
        "is_ground_truth": False,
        "usable_for_promotion": False,
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "checkpoint": str(checkpoint_path),
        "history_json": str(history_path),
        "bad_stem_path": str(bad_out),
        "leak_audio_path": str(leak_out),
        "pseudo_target_path": str(target_out),
        "repaired_stem_path": str(repaired_out),
        "region": region,
        "leak_gain": float(leak_gain),
        "forensic_envelope_target": bool(forensic_envelope_target),
        "forensic_envelope_attenuation": float(forensic_envelope_attenuation),
        "forensic_high_band_hz": [float(forensic_high_low_hz), float(forensic_high_high_hz)],
        "forensic_source_decorrelation_loss_weight": float(config.forensic_source_decorrelation_loss_weight),
        "forensic_source_envelope_loss_weight": float(config.forensic_source_envelope_loss_weight),
        "forensic_envelope_window_ms": float(config.forensic_envelope_window_ms),
        "forensic_source_conditioning": bool(config.forensic_source_conditioning),
        "forensic_source_audio_path": forensic_source_path,
        "mix_preserve_companion_stem": mix_preserve_companion_stem,
        "leak_transfer_to_foreign_source": bool(transfer_companion_stem),
        "leak_rms": _rms(leak_sum),
        "target_region_mse_before": before_mse,
        "target_region_mse_after": after_mse,
        "primary_target_region_mse_before": first_before_mse,
        "primary_target_region_mse_after": first_after_mse,
        "target_issue_improved": after_mse < before_mse,
        "outside_region_mse_delta": outside_mse,
        "weights_changed": float(history.get("final_loss", 0.0)) < float(history.get("initial_loss", 0.0)),
        "train_config": asdict(config),
        "leak_evidence": leak_evidence,
        "diagnostics": {**diagnostics, "primary": first_diagnostics},
        "contract": {
            "pseudo_teacher_signal": True,
            "pseudo_teacher_is_ground_truth": False,
            "can_promote_without_gates": False,
            "no_threshold_relaxation": True,
        },
    }
    write_json(output_dir / "manifest.json", result)
    return result


def train_from_leak_evidence(
    *,
    candidate_dir: Path,
    candidate_dirs: list[Path] | None = None,
    output_dir: Path,
    stem: str,
    issue_type: str,
    foreign_sources: list[str],
    leak_gain: float,
    config: StemRegionRepairTrainConfig,
    forensic_pair_json: Path | None = None,
    forensic_envelope_target: bool = False,
    forensic_envelope_attenuation: float = 0.85,
    forensic_high_low_hz: float = 1200.0,
    forensic_high_high_hz: float = 10000.0,
    transfer_leak_to_foreign_source: bool = False,
) -> dict[str, Any]:
    issue_type = validate_issue_type(issue_type)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_json(candidate_dir / "manifest.json")
    score = _load_json(candidate_dir / "score.json")
    foreign_source_set = {str(item).strip() for item in foreign_sources if str(item).strip()}
    if not foreign_source_set:
        raise ValueError("at least one foreign source is required")
    ordered_candidate_dirs = [candidate_dir]
    for extra_dir in candidate_dirs or []:
        if extra_dir not in ordered_candidate_dirs:
            ordered_candidate_dirs.append(extra_dir)
    if len(ordered_candidate_dirs) > 1:
        examples = [
            _load_leak_training_example(
                candidate_dir=example_dir,
                stem=stem,
                foreign_source_set=foreign_source_set,
                leak_gain=leak_gain,
                config=config,
                forensic_pair_json=forensic_pair_json if index == 0 else None,
                forensic_envelope_target=forensic_envelope_target,
                forensic_envelope_attenuation=forensic_envelope_attenuation,
                forensic_high_low_hz=forensic_high_low_hz,
                forensic_high_high_hz=forensic_high_high_hz,
            )
            for index, example_dir in enumerate(ordered_candidate_dirs)
        ]
        return _train_from_leak_examples(
            examples=examples,
            output_dir=output_dir,
            stem=stem,
            issue_type=issue_type,
            foreign_source_set=foreign_source_set,
            leak_gain=leak_gain,
            config=config,
            forensic_envelope_target=forensic_envelope_target,
            forensic_envelope_attenuation=forensic_envelope_attenuation,
            forensic_high_low_hz=forensic_high_low_hz,
            forensic_high_high_hz=forensic_high_high_hz,
            transfer_leak_to_foreign_source=transfer_leak_to_foreign_source,
        )

    stem_path = _candidate_stem_path(candidate_dir, manifest, stem)
    bad, sample_rate = read_audio_stereo(stem_path)
    frames = bad.shape[0]
    mix_path = candidate_dir.parent / "mix.flac"
    if not mix_path.is_file():
        raise FileNotFoundError(f"missing window mix: {mix_path}")
    mix, mix_rate = read_audio_stereo(mix_path)
    if int(mix_rate) != int(sample_rate):
        raise ValueError("sample rate mismatch between mix and stem")
    mix = _fit_like(mix, frames)

    if forensic_pair_json is not None:
        leak_results = [
            _forensic_proxy_result(
                forensic_pair_json=forensic_pair_json,
                stem=stem,
                foreign_sources=foreign_source_set,
            )
        ]
    else:
        leak_results = _find_leak_results(score=score, stem=stem, foreign_sources=foreign_source_set)
    if not leak_results:
        raise ValueError(f"no failed leak evidence for {stem}:{sorted(foreign_source_set)}")
    if forensic_pair_json is not None:
        leak_sum = np.zeros((frames, 2), dtype=np.float32)
        leak_evidence: list[dict[str, Any]] = []
        for item in leak_results:
            leak_path = Path(str(item["leak_audio"]))
            if not leak_path.is_file():
                raise FileNotFoundError(f"missing forensic proxy audio: {leak_path}")
            leak_audio, leak_rate = read_audio_stereo(leak_path)
            if int(leak_rate) != int(sample_rate):
                raise ValueError(f"sample rate mismatch for forensic proxy audio: {leak_path}")
            leak_audio = _fit_like(leak_audio, frames)
            leak_sum += leak_audio
            leak_evidence.append(
                {
                    "tested_stem": str(item.get("tested_stem") or ""),
                    "foreign_source": str(item.get("foreign_source") or ""),
                    "leak_audio": str(leak_path),
                    "leak_dbfs": item.get("leak_dbfs"),
                    "regions": item.get("regions") or [],
                    "evidence_type": "cross_stem_forensic_proxy",
                }
            )
    else:
        leak_sum, leak_evidence = _load_leak_sum(
            candidate_dir=candidate_dir,
            leak_results=leak_results,
            frames=frames,
            sample_rate=sample_rate,
        )
    region = _region_from_leaks(leak_results, frames / float(sample_rate))
    use_source_condition = (
        float(config.forensic_source_decorrelation_loss_weight) > 0.0
        or float(config.forensic_source_envelope_loss_weight) > 0.0
        or bool(config.forensic_source_conditioning)
    )
    forensic_source, forensic_source_path = (
        _forensic_source_audio(
            candidate_dir=candidate_dir,
            manifest=manifest,
            leak_result=leak_results[0],
            frames=frames,
            sample_rate=sample_rate,
        )
        if use_source_condition
        else (None, "")
    )
    if forensic_pair_json is not None and forensic_envelope_target:
        target = _build_forensic_envelope_target(
            bad_audio=bad,
            source_audio=leak_sum,
            sample_rate=sample_rate,
            regions=leak_results[0].get("regions") if isinstance(leak_results[0].get("regions"), list) else [],
            attenuation=forensic_envelope_attenuation,
            low_hz=forensic_high_low_hz,
            high_hz=forensic_high_high_hz,
        )
    else:
        target = np.clip(bad - (float(leak_gain) * leak_sum), -1.0, 1.0).astype(np.float32)
    mask = make_region_mask(
        frame_count=frames,
        sample_rate=sample_rate,
        start_sec=float(region["start_sec"]),
        end_sec=float(region["end_sec"]),
    )
    siblings, reconstruction_surface, mix_preserve_companion_stem = _mix_conservation_sibling(
        candidate_dir,
        manifest,
        stem=stem,
        frames=frames,
        sample_rate=sample_rate,
    )

    before_mse = _region_mse(bad, target, mask)
    model, history = train_region_repair_model_from_arrays(
        mix_audio=mix,
        bad_stem_audio=bad,
        target_stem_audio=target,
        sibling_stem_audio=siblings,
        target_mix_audio=mix if siblings is not None else None,
        forensic_source_audio=forensic_source,
        region_mask=mask,
        sample_rate=sample_rate,
        issue_type=issue_type,
        config=config,
    )
    repaired, diagnostics = apply_region_repair_model(
        model=model,
        mix_audio=mix,
        bad_stem_audio=bad,
        region_mask=mask,
        sample_rate=sample_rate,
        issue_type=issue_type,
        source_condition_audio=forensic_source if bool(config.forensic_source_conditioning) else None,
    )
    after_mse = _region_mse(repaired, target, mask)
    outside_mse = _outside_delta_mse(bad, repaired, mask)

    audio_dir = output_dir / "audio"
    bad_out = write_audio(audio_dir / "bad.flac", bad, sample_rate)
    leak_out = write_audio(audio_dir / "leak.flac", leak_sum, sample_rate)
    target_out = write_audio(audio_dir / "pseudo.flac", target, sample_rate)
    repaired_out = write_audio(audio_dir / "repaired.flac", repaired, sample_rate)
    history = {
        **history,
        "region": region,
        "training_row": {
            "row_id": "leak_ev_000001",
            "target_stem": stem,
            "root_cause_row_id": "",
            "reconstruction_surface": reconstruction_surface or "unavailable",
            "target_surface": "pseudo_teacher_minus_leak",
        },
        "label_type": "pseudo_teacher",
        "is_ground_truth": False,
        "leak_gain": float(leak_gain),
        "forensic_envelope_target": bool(forensic_pair_json is not None and forensic_envelope_target),
        "forensic_envelope_attenuation": float(forensic_envelope_attenuation),
        "forensic_high_band_hz": [float(forensic_high_low_hz), float(forensic_high_high_hz)],
        "forensic_source_decorrelation_loss_weight": float(config.forensic_source_decorrelation_loss_weight),
        "forensic_source_envelope_loss_weight": float(config.forensic_source_envelope_loss_weight),
        "forensic_envelope_window_ms": float(config.forensic_envelope_window_ms),
        "forensic_source_conditioning": bool(config.forensic_source_conditioning),
        "forensic_source_audio_path": forensic_source_path,
    }
    transfer_companion_stem = ""
    if transfer_leak_to_foreign_source and not mix_preserve_companion_stem and len(foreign_source_set) == 1:
        transfer_companion_stem = sorted(foreign_source_set)[0]
        mix_preserve_companion_stem = transfer_companion_stem
    if mix_preserve_companion_stem:
        history["mix_preserve_companion_stem"] = mix_preserve_companion_stem
    if transfer_companion_stem:
        history["leak_transfer_to_foreign_source"] = True
    if bool(config.forensic_source_conditioning) and len(foreign_source_set) == 1:
        history["forensic_source_condition_stem"] = sorted(foreign_source_set)[0]
    checkpoint_path = save_region_repair_checkpoint(model=model, path=output_dir / "model" / "model.pt", training_history=history)
    history_path = write_training_history(output_dir / "history.json", history)
    result = {
        "schema": "stem_refiner_leak_evidence_training.v1",
        "candidate_dir": str(candidate_dir),
        "stem": stem,
        "issue_type": issue_type,
        "foreign_sources": sorted(foreign_source_set),
        "label_type": "pseudo_teacher",
        "is_ground_truth": False,
        "usable_for_promotion": False,
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "checkpoint": str(checkpoint_path),
        "history_json": str(history_path),
        "bad_stem_path": str(bad_out),
        "leak_audio_path": str(leak_out),
        "pseudo_target_path": str(target_out),
        "repaired_stem_path": str(repaired_out),
        "region": region,
        "leak_gain": float(leak_gain),
        "forensic_envelope_target": bool(forensic_pair_json is not None and forensic_envelope_target),
        "forensic_envelope_attenuation": float(forensic_envelope_attenuation),
        "forensic_high_band_hz": [float(forensic_high_low_hz), float(forensic_high_high_hz)],
        "forensic_source_decorrelation_loss_weight": float(config.forensic_source_decorrelation_loss_weight),
        "forensic_source_envelope_loss_weight": float(config.forensic_source_envelope_loss_weight),
        "forensic_envelope_window_ms": float(config.forensic_envelope_window_ms),
        "forensic_source_conditioning": bool(config.forensic_source_conditioning),
        "forensic_source_audio_path": forensic_source_path,
        "mix_preserve_companion_stem": mix_preserve_companion_stem,
        "leak_transfer_to_foreign_source": bool(transfer_companion_stem),
        "leak_rms": _rms(leak_sum),
        "target_region_mse_before": before_mse,
        "target_region_mse_after": after_mse,
        "target_issue_improved": after_mse < before_mse,
        "outside_region_mse_delta": outside_mse,
        "weights_changed": float(history.get("final_loss", 0.0)) < float(history.get("initial_loss", 0.0)),
        "train_config": asdict(config),
        "leak_evidence": leak_evidence,
        "diagnostics": diagnostics,
        "contract": {
            "pseudo_teacher_signal": True,
            "pseudo_teacher_is_ground_truth": False,
            "can_promote_without_gates": False,
            "no_threshold_relaxation": True,
        },
    }
    write_json(output_dir / "manifest.json", result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--issue-type", required=True)
    parser.add_argument("--foreign-source", action="append", default=[], required=True)
    parser.add_argument("--forensic-pair-json", type=Path)
    parser.add_argument("--forensic-envelope-target", action="store_true")
    parser.add_argument("--forensic-envelope-attenuation", type=float, default=0.85)
    parser.add_argument("--forensic-high-low-hz", type=float, default=1200.0)
    parser.add_argument("--forensic-high-high-hz", type=float, default=10000.0)
    parser.add_argument("--leak-gain", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--dilated-layers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--region-loss-weight", type=float, default=1.0)
    parser.add_argument("--outside-loss-weight", type=float, default=0.2)
    parser.add_argument("--mix-conservation-loss-weight", type=float, default=0.5)
    parser.add_argument("--stereo-preserve-loss-weight", type=float, default=0.2)
    parser.add_argument("--forensic-source-decorrelation-loss-weight", type=float, default=0.0)
    parser.add_argument("--forensic-source-envelope-loss-weight", type=float, default=0.0)
    parser.add_argument("--forensic-envelope-window-ms", type=float, default=20.0)
    parser.add_argument("--forensic-source-conditioning", action="store_true")
    parser.add_argument("--transfer-leak-to-foreign-source", action="store_true")
    parser.add_argument("--curriculum-target-boost", type=float, default=1.0)
    parser.add_argument("--curriculum-hard-fraction", type=float, default=0.0)
    parser.add_argument("--curriculum-hard-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-delta", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    candidate_dirs = list(args.candidate_dir)
    manifest = train_from_leak_evidence(
        candidate_dir=candidate_dirs[0],
        candidate_dirs=candidate_dirs[1:],
        output_dir=args.out,
        stem=str(args.stem).strip().lower(),
        issue_type=str(args.issue_type).strip().lower(),
        foreign_sources=[str(item).strip().lower() for item in args.foreign_source],
        leak_gain=float(args.leak_gain),
        forensic_pair_json=args.forensic_pair_json,
        forensic_envelope_target=bool(args.forensic_envelope_target),
        forensic_envelope_attenuation=float(args.forensic_envelope_attenuation),
        forensic_high_low_hz=float(args.forensic_high_low_hz),
        forensic_high_high_hz=float(args.forensic_high_high_hz),
        transfer_leak_to_foreign_source=bool(args.transfer_leak_to_foreign_source),
        config=StemRegionRepairTrainConfig(
            epochs=int(args.epochs),
            hidden_channels=int(args.hidden_channels),
            dilated_layers=int(args.dilated_layers),
            learning_rate=float(args.learning_rate),
            region_loss_weight=float(args.region_loss_weight),
            outside_loss_weight=float(args.outside_loss_weight),
            mix_conservation_loss_weight=float(args.mix_conservation_loss_weight),
            stereo_preserve_loss_weight=float(args.stereo_preserve_loss_weight),
            forensic_source_decorrelation_loss_weight=float(args.forensic_source_decorrelation_loss_weight),
            forensic_source_envelope_loss_weight=float(args.forensic_source_envelope_loss_weight),
            forensic_envelope_window_ms=float(args.forensic_envelope_window_ms),
            forensic_source_conditioning=bool(args.forensic_source_conditioning),
            curriculum_target_boost=float(args.curriculum_target_boost),
            curriculum_hard_fraction=float(args.curriculum_hard_fraction),
            curriculum_hard_loss_weight=float(args.curriculum_hard_loss_weight),
            max_delta=float(args.max_delta),
            device=str(args.device),
        ),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
