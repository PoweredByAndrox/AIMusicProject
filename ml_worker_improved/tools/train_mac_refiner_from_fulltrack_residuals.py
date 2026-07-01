#!/usr/bin/env python3
"""Train a Mac/MPS refiner on residual full-track pitch-guard failures.

Input is a failed full-track diagnostic stage, such as mel_full002. The script
builds a short-path guard from the replay audio after the previous checkpoint,
then trains a pseudo-teacher residual checkpoint on the remaining detected
pitch-shifted vocal regions. It is diagnostic-only and cannot publish listen
audio.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROOT_SCORE_STAGE_SCHEMAS = {
    "mac_mel_residual_chunk_composition.v1",
    "mac_melband_refiner_training.v1",
}


def _append_progress(path: Path, phase: str, status: str, **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_unix": time.time(), "phase": phase, "status": status, **extra}, sort_keys=True) + "\n")
        handle.flush()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if isinstance(item, Mapping):
                rows.append(dict(item))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def run_training(**kwargs: Any) -> dict[str, Any]:
    from tools.train_mac_refiner_from_mel_guard import run_training as _run

    return _run(**kwargs)


def run_damage_audit(**kwargs: Any) -> dict[str, Any]:
    from tools.mac_audio_damage_audit import run_damage_audit as _run

    return _run(**kwargs)


def run_pitch_score(**kwargs: Any) -> dict[str, Any]:
    from tools.mac_melband_pitch_guard import score_pitch_shifted_vocal_leak as _run

    return _run(**kwargs)


def _resolve_existing_path(value: Any, *, base_dir: Path) -> Path:
    path = Path(str(value or ""))
    if path.is_file():
        return path
    if not path.is_absolute():
        candidate = base_dir / path
        if candidate.is_file():
            return candidate
    tail = tuple(path.parts[-2:]) if len(path.parts) >= 2 else ()
    allowed_audio_names = {"v.flac", "i.flac", "raw.flac"}
    if tail == ("replay", "manifest.json") or (len(tail) == 2 and tail[0] == "audio" and tail[1] in allowed_audio_names):
        candidate = base_dir / Path(*tail)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"residual training source file missing: {value}")


def _resolve_optional_regions_path(value: Any, *, score_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_file():
        return path
    if not path.is_absolute():
        candidate = score_dir / path
        if candidate.is_file():
            return candidate
    candidate = score_dir / path.name
    if path.name == "regions.jsonl" and candidate.is_file():
        return candidate
    return None


def _base_manifest(*, failed_stage_dir: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "schema": "mac_mel_residual_refiner_training.v1",
        "source_failed_stage_dir": str(failed_stage_dir),
        "output_dir": str(output_dir),
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "residual_training_diagnostic_only",
        "label_type": "pseudo_teacher",
        "is_ground_truth": False,
        "usable_for_promotion": False,
        "contract": {
            "diagnostic_only": True,
            "pseudo_teacher_only": True,
            "no_listen_delivery": True,
            "no_threshold_relaxation": True,
            "source_is_failed_fulltrack_stage": True,
        },
    }


def _write_blocked(
    *,
    output_dir: Path,
    progress_path: Path,
    failed_stage_dir: Path,
    status: str,
    failed_checks: list[str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        **_base_manifest(failed_stage_dir=failed_stage_dir, output_dir=output_dir),
        "status": status,
        "failed_checks": list(failed_checks),
        **dict(extra or {}),
    }
    _write_json(output_dir / "manifest.json", manifest)
    _append_progress(progress_path, "manifest", status, failed_checks=failed_checks)
    return manifest


def _copy_audio_aliases(*, replay_audio: Mapping[str, Any], replay_dir: Path, guard_dir: Path) -> dict[str, str]:
    audio_dir = guard_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    aliases = {
        "vocals": "v.flac",
        "instrumental": "i.flac",
        "raw_ref": "raw.flac",
    }
    copied: dict[str, str] = {}
    for key, alias in aliases.items():
        source = _resolve_existing_path(replay_audio.get(key), base_dir=replay_dir)
        target = audio_dir / alias
        shutil.copy2(source, target)
        copied[key] = str(target)
    return copied


def _resolve_stage_sources(
    *,
    failed_stage_dir: Path,
    stage_manifest_path: Path,
    stage_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, Path, dict[str, str]]:
    stage_schema = str(stage_manifest.get("schema") or "")
    if stage_schema in ROOT_SCORE_STAGE_SCHEMAS:
        score_path = failed_stage_dir / "score" / "after" / "pitch.json"
        if not score_path.is_file():
            raise FileNotFoundError(f"residual training source file missing: {score_path}")
        return {}, score_path, failed_stage_dir, {
            "source_stage_manifest": str(stage_manifest_path),
            "source_replay_manifest": "",
            "source_stage_schema": stage_schema,
        }

    replay_manifest_path = _resolve_existing_path(
        stage_manifest.get("replay_manifest") or "replay/manifest.json",
        base_dir=failed_stage_dir,
    )
    replay_manifest = _load_json(replay_manifest_path)
    return replay_manifest, replay_manifest_path.parent / "score" / "after" / "pitch.json", replay_manifest_path.parent, {
        "source_stage_manifest": str(stage_manifest_path),
        "source_replay_manifest": str(replay_manifest_path),
        "source_stage_schema": stage_schema,
    }


def _normalized_probe_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.1f}"
    text = str(value or "").strip()
    try:
        return f"{float(text):.1f}"
    except ValueError:
        return text


def _region_matches_batch(region: Mapping[str, Any], region_batch: Mapping[str, Any] | None) -> bool:
    if not isinstance(region_batch, Mapping) or str(region_batch.get("axis") or "all") == "all":
        return True
    axis = str(region_batch.get("axis") or "")
    value = region_batch.get("value")
    if axis == "probe":
        region_value = region.get("probe_semitones", region.get("semitones"))
        return _normalized_probe_value(region_value) == _normalized_probe_value(value)
    if axis == "fail_mode":
        return str(region.get("fail_mode") or "") == str(value)
    return False


def _filter_regions_for_batch(regions: list[Any], region_batch: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    filtered = [
        dict(region)
        for region in regions
        if isinstance(region, Mapping) and _region_matches_batch(region, region_batch)
    ]
    if not isinstance(region_batch, Mapping) or "chunk_start" not in region_batch:
        return filtered
    start = max(0, int(region_batch.get("chunk_start") or 0))
    limit = max(1, int(region_batch.get("chunk_limit") or len(filtered) or 1))
    return filtered[start : start + limit]


def _is_scoped_batch(region_batch: Mapping[str, Any] | None) -> bool:
    return isinstance(region_batch, Mapping) and str(region_batch.get("axis") or "all") != "all"


def build_residual_guard_from_failed_stage(
    *,
    failed_stage_dir: Path,
    output_dir: Path,
    region_batch: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    stage_manifest_path = Path(failed_stage_dir) / "manifest.json"
    stage_manifest = _load_json(stage_manifest_path)
    if bool(stage_manifest.get("delivery_allowed")) or bool(stage_manifest.get("listen_pack_created")):
        raise ValueError("failed stage cannot be delivery/listen artifact")
    replay_manifest, score_path, audio_base_dir, source_provenance = _resolve_stage_sources(
        failed_stage_dir=Path(failed_stage_dir),
        stage_manifest_path=stage_manifest_path,
        stage_manifest=stage_manifest,
    )
    residual_score = _load_json(score_path)
    source_all_regions_path = _resolve_optional_regions_path(
        residual_score.get("all_regions_path"),
        score_dir=score_path.parent,
    )
    source_regions = (
        _load_jsonl(source_all_regions_path)
        if source_all_regions_path is not None
        else list(residual_score.get("regions") or [])
    )
    source_region_mode = "all_regions_jsonl" if source_all_regions_path is not None else "selected_regions"
    batch = dict(region_batch or {})
    regions = _filter_regions_for_batch(source_regions, batch)
    if str(residual_score.get("verdict")) != "fail" or not regions:
        has_source_regions = bool(source_regions)
        has_batch = bool(batch) and str(batch.get("axis") or "all") != "all"
        status = "blocked_no_matching_residual_batch_regions" if has_source_regions and has_batch else "blocked_no_residual_pitch_regions"
        failed_check = "no_matching_residual_batch_regions" if has_source_regions and has_batch else "no_residual_pitch_regions"
        return {
            "status": status,
            "failed_checks": [failed_check],
            "stage_manifest": stage_manifest,
            "replay_manifest": replay_manifest,
            "residual_score": residual_score,
            "taxonomy_batch": batch,
            "source_region_mode": source_region_mode,
            "source_all_regions_path": str(source_all_regions_path) if source_all_regions_path is not None else "",
            "source_residual_region_count": len(source_regions),
            "residual_region_count": len(regions),
        }
    replay_audio = dict(stage_manifest.get("audio") or replay_manifest.get("audio") or {})
    guard_dir = output_dir / "guard"
    copied_audio = _copy_audio_aliases(replay_audio=replay_audio, replay_dir=audio_base_dir, guard_dir=guard_dir)
    source_total_fail_regions = int(residual_score.get("total_fail_regions") or len(source_regions))
    source_selected_region_count = int(residual_score.get("selected_region_count") or len(source_regions))
    guard_manifest = {
        "schema": "mac_mel_residual_guard.v1",
        "source_failed_stage_dir": str(failed_stage_dir),
        **source_provenance,
        "audio": copied_audio,
        "score": {
            "config": dict(residual_score.get("config") or {}),
            "source_after_score": str(score_path),
            "regions": regions,
            "total_fail_regions": source_total_fail_regions,
            "selected_region_count": source_selected_region_count,
            "regions_capped": bool(residual_score.get("regions_capped") or source_total_fail_regions > source_selected_region_count),
            "failure_taxonomy": dict(residual_score.get("failure_taxonomy") or {}),
            "best_probe": dict(residual_score.get("best_probe") or {}),
            "batch_region_count": len(regions),
            "source_region_mode": source_region_mode,
            "source_all_regions_path": str(source_all_regions_path) if source_all_regions_path is not None else "",
            "source_selected_region_count": source_selected_region_count,
        },
        "region_batch": batch,
        "source_region_mode": source_region_mode,
        "source_all_regions_path": str(source_all_regions_path) if source_all_regions_path is not None else "",
        "source_residual_region_count": len(source_regions),
        "batch_region_count": len(regions),
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "residual_guard_diagnostic_only",
    }
    _write_json(guard_dir / "manifest.json", guard_manifest)
    return {
        "status": "ok",
        "guard_dir": str(guard_dir),
        "guard_manifest": guard_manifest,
        "stage_manifest": stage_manifest,
        "replay_manifest": replay_manifest,
        "residual_score": residual_score,
        "taxonomy_batch": batch,
        "source_region_mode": source_region_mode,
        "source_all_regions_path": str(source_all_regions_path) if source_all_regions_path is not None else "",
        "source_residual_region_count": len(source_regions),
        "residual_region_count": len(regions),
    }


def _fit_arrays(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    frames = min(int(array.shape[0]) for array in arrays)
    return tuple(np.asarray(array[:frames], dtype=np.float32) for array in arrays)


def _rms(audio: np.ndarray) -> float:
    array = np.asarray(audio, dtype=np.float64)
    if array.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(array))))


def _region_influence(
    *,
    frame_count: int,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
    fade_sec: float,
) -> tuple[np.ndarray, int, int]:
    start = max(0, min(int(round(float(start_sec) * float(sample_rate))), int(frame_count)))
    end = max(start, min(int(round(float(end_sec) * float(sample_rate))), int(frame_count)))
    influence = np.zeros(int(frame_count), dtype=np.float32)
    if end <= start:
        return influence, start, end
    influence[start:end] = 1.0
    fade_frames = max(0, int(round(float(fade_sec) * float(sample_rate))))
    if fade_frames > 0:
        fade_frames = min(fade_frames, max(1, (end - start) // 2))
        ramp = np.linspace(0.0, 1.0, fade_frames, endpoint=False, dtype=np.float32)
        influence[start : start + fade_frames] *= ramp
        influence[end - fade_frames : end] *= ramp[::-1]
    return influence, start, end


def _apply_damage_limited_delta(
    *,
    reference_vocals: np.ndarray,
    reference_instrumental: np.ndarray,
    candidate_instrumental: np.ndarray,
    sample_rate: int,
    damage_regions: Sequence[Mapping[str, Any]],
    max_region_delta_ratio: float,
    safety_margin: float = 0.97,
    fade_sec: float = 0.03,
    raw_audio: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if raw_audio is None:
        reference_vocals, reference_instrumental, candidate_instrumental = _fit_arrays(
            reference_vocals,
            reference_instrumental,
            candidate_instrumental,
        )
        mix_target = reference_vocals + reference_instrumental
    else:
        reference_vocals, reference_instrumental, candidate_instrumental, raw_audio = _fit_arrays(
            reference_vocals,
            reference_instrumental,
            candidate_instrumental,
            raw_audio,
        )
        mix_target = raw_audio

    frame_count = int(reference_instrumental.shape[0])
    delta = candidate_instrumental - reference_instrumental
    scale_by_frame = np.ones(frame_count, dtype=np.float32)
    applied_regions: list[dict[str, Any]] = []
    limit = max(0.0, float(max_region_delta_ratio))
    margin = max(0.0, float(safety_margin))
    for region in damage_regions:
        if not isinstance(region, Mapping):
            continue
        reasons = [str(reason) for reason in region.get("blocker_reasons") or []]
        source_delta_ratio = float(region.get("delta_ratio") or 0.0)
        if source_delta_ratio <= 0.0:
            continue
        if "instrumental_region_overprocessed" not in reasons and source_delta_ratio <= limit:
            continue
        scale = min(1.0, max(0.0, (limit * margin) / max(source_delta_ratio, 1.0e-9)))
        if scale >= 0.999999:
            continue
        influence, start, end = _region_influence(
            frame_count=frame_count,
            sample_rate=int(sample_rate),
            start_sec=float(region.get("start") or 0.0),
            end_sec=float(region.get("end") or 0.0),
            fade_sec=float(fade_sec),
        )
        if end <= start:
            continue
        region_scale = 1.0 - ((1.0 - scale) * influence)
        scale_by_frame = np.minimum(scale_by_frame, region_scale.astype(np.float32, copy=False))
        applied_regions.append(
            {
                "start": round(float(start) / float(sample_rate), 6),
                "end": round(float(end) / float(sample_rate), 6),
                "source_delta_ratio": round(float(source_delta_ratio), 6),
                "scale": round(float(scale), 6),
                "blocker_reasons": reasons,
            }
        )

    limited_delta = delta * scale_by_frame[:, None]
    limited_instrumental = reference_instrumental + limited_delta
    limited_vocals = mix_target - limited_instrumental
    return limited_vocals.astype(np.float32), limited_instrumental.astype(np.float32), applied_regions


def _write_damage_limited_manifest(output_dir: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    manifest = {
        "schema": "mac_residual_damage_limited_repair.v1",
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "damage_limited_repair_diagnostic_only",
        **dict(payload),
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def run_damage_limited_repair(
    *,
    reference_vocals_path: Path,
    reference_instrumental_path: Path,
    reference_raw_path: Path,
    candidate_instrumental_path: Path,
    guard_score: Mapping[str, Any],
    initial_damage_audit: Mapping[str, Any],
    output_dir: Path,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    from tools.stem_region_repair_common import read_audio_stereo, write_audio

    output_dir.mkdir(parents=True, exist_ok=True)
    local_progress = output_dir / "progress.jsonl"
    parent_progress = progress_path if progress_path is not None else local_progress
    _append_progress(parent_progress, "damage_limited_repair", "start", output_dir=str(output_dir))
    _append_progress(local_progress, "preflight", "start")
    try:
        reference_vocals, vocals_rate = read_audio_stereo(reference_vocals_path)
        reference_instrumental, instrumental_rate = read_audio_stereo(reference_instrumental_path)
        candidate_instrumental, candidate_rate = read_audio_stereo(candidate_instrumental_path)
        raw_audio, raw_rate = read_audio_stereo(reference_raw_path)
    except Exception as exc:
        manifest = _write_damage_limited_manifest(
            output_dir,
            {
                "status": "damage_limited_repair_blocked_audio_read_failed",
                "blocker_reasons": ["damage_limited_audio_read_failed"],
                "error": str(exc),
            },
        )
        _append_progress(parent_progress, "damage_limited_repair", "blocked", repair_status=manifest["status"])
        return manifest

    if len({int(vocals_rate), int(instrumental_rate), int(candidate_rate), int(raw_rate)}) != 1:
        manifest = _write_damage_limited_manifest(
            output_dir,
            {
                "status": "damage_limited_repair_blocked_sample_rate_mismatch",
                "blocker_reasons": ["damage_limited_sample_rate_mismatch"],
                "sample_rates": {
                    "vocals": int(vocals_rate),
                    "reference_instrumental": int(instrumental_rate),
                    "candidate_instrumental": int(candidate_rate),
                    "raw_ref": int(raw_rate),
                },
            },
        )
        _append_progress(parent_progress, "damage_limited_repair", "blocked", repair_status=manifest["status"])
        return manifest

    damage_config = dict(initial_damage_audit.get("config") or {})
    max_delta_ratio = float(damage_config.get("max_region_delta_ratio", 0.42))
    damage_regions = [
        dict(region)
        for region in initial_damage_audit.get("regions") or []
        if isinstance(region, Mapping) and region.get("blocker_reasons")
    ]
    limited_vocals, limited_instrumental, applied_regions = _apply_damage_limited_delta(
        reference_vocals=reference_vocals,
        reference_instrumental=reference_instrumental,
        candidate_instrumental=candidate_instrumental,
        raw_audio=raw_audio,
        sample_rate=int(vocals_rate),
        damage_regions=damage_regions,
        max_region_delta_ratio=max_delta_ratio,
    )
    if not applied_regions:
        manifest = _write_damage_limited_manifest(
            output_dir,
            {
                "status": "damage_limited_repair_blocked_no_repairable_regions",
                "blocker_reasons": ["damage_limited_no_repairable_regions"],
                "initial_damage_audit": dict(initial_damage_audit),
            },
        )
        _append_progress(parent_progress, "damage_limited_repair", "blocked", repair_status=manifest["status"])
        return manifest

    audio_dir = output_dir / "audio"
    vocals_out = write_audio(audio_dir / "v.flac", limited_vocals, int(vocals_rate))
    instrumental_out = write_audio(audio_dir / "i.flac", limited_instrumental, int(vocals_rate))
    raw_out = write_audio(audio_dir / "raw.flac", raw_audio, int(vocals_rate))
    reconstruction_rms = _rms(limited_vocals + limited_instrumental - raw_audio[: limited_vocals.shape[0]])
    _append_progress(
        local_progress,
        "audio",
        "end",
        applied_region_count=len(applied_regions),
        reconstruction_rms=round(float(reconstruction_rms), 10),
    )

    score_dir = output_dir / "score" / "after"
    score_config = dict(guard_score.get("config") or {})

    def _score_progress(event: dict[str, Any]) -> None:
        _append_progress(local_progress, "pitch_score", str(event.get("event") or "progress"), **event)

    after_score = run_pitch_score(
        vocals_path=Path(vocals_out),
        instrumental_path=Path(instrumental_out),
        output_dir=score_dir,
        config=score_config,
        progress_callback=_score_progress,
    )
    _append_progress(
        local_progress,
        "pitch_score",
        "end",
        verdict=after_score.get("verdict"),
        total_fail_regions=after_score.get("total_fail_regions"),
    )
    if str(after_score.get("verdict")) != "pass":
        manifest = _write_damage_limited_manifest(
            output_dir,
            {
                "status": "damage_limited_repair_pitch_failed",
                "blocker_reasons": list(after_score.get("blocker_reasons") or ["pitch_guard_failed"]),
                "audio": {"vocals": str(vocals_out), "instrumental": str(instrumental_out), "raw_ref": str(raw_out)},
                "applied_regions": applied_regions,
                "after_score": dict(after_score),
                "initial_damage_audit": dict(initial_damage_audit),
                "reconstruction_rms": round(float(reconstruction_rms), 10),
            },
        )
        _append_progress(parent_progress, "damage_limited_repair", "end", repair_status=manifest["status"])
        return manifest

    damage_dir = output_dir / "damage"
    damage_score_path = damage_dir / "score.json"
    audit_regions = list(initial_damage_audit.get("regions") or guard_score.get("regions") or [])
    _write_json(
        damage_score_path,
        {
            "regions": audit_regions,
            "config": score_config,
            "source_after_score": str(score_dir / "pitch.json"),
            "source_damage_audit_regions": True,
        },
    )
    damage_audit = run_damage_audit(
        reference_instrumental_path=reference_instrumental_path,
        candidate_instrumental_path=Path(instrumental_out),
        score_json_path=damage_score_path,
        output_dir=damage_dir,
        config=damage_config,
    )
    status = "damage_limited_repair_passed" if str(damage_audit.get("verdict")) == "pass" else "damage_limited_repair_damage_failed"
    manifest = _write_damage_limited_manifest(
        output_dir,
        {
            "status": status,
            "blocker_reasons": list(damage_audit.get("blocker_reasons") or []),
            "audio": {"vocals": str(vocals_out), "instrumental": str(instrumental_out), "raw_ref": str(raw_out)},
            "applied_regions": applied_regions,
            "after_score": dict(after_score),
            "initial_damage_audit": dict(initial_damage_audit),
            "damage_audit": dict(damage_audit),
            "damage_audit_manifest": str(damage_dir / "damage.json"),
            "reconstruction_rms": round(float(reconstruction_rms), 10),
            "contract": {
                "diagnostic_only": True,
                "mixture_consistent_delta_transfer": True,
                "damage_regions_only": True,
                "no_threshold_relaxation": True,
            },
        },
    )
    _append_progress(
        parent_progress,
        "damage_limited_repair",
        "end",
        repair_status=manifest["status"],
        pitch_verdict=after_score.get("verdict"),
        damage_verdict=damage_audit.get("verdict"),
    )
    return manifest


def _training_status(training_manifest: Mapping[str, Any], region_batch: Mapping[str, Any] | None = None) -> str:
    if str(training_manifest.get("status")) == "trained_repair_passed_pitch_guard":
        if _is_scoped_batch(region_batch):
            return "residual_batch_trained_passed_pitch_guard"
        return "residual_trained_passed_pitch_guard"
    return "residual_trained_still_blocked"


def _damage_failed_status(region_batch: Mapping[str, Any] | None = None) -> str:
    if _is_scoped_batch(region_batch):
        return "residual_batch_trained_damage_failed"
    return "residual_trained_damage_failed"


def _damage_limited_passed_status(region_batch: Mapping[str, Any] | None = None) -> str:
    if _is_scoped_batch(region_batch):
        return "residual_batch_trained_damage_limited_passed"
    return "residual_trained_damage_limited_passed"


def run_residual_training(
    *,
    failed_stage_dir: Path,
    output_dir: Path,
    target_gain: float = 2.5,
    epochs: int = 260,
    require_mps: bool = True,
    max_regions: int = 8,
    region_batch: Mapping[str, Any] | None = None,
    probe_semitones: Sequence[float] | None = None,
    objective_profile: str = "projection_teacher",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    batch = dict(region_batch or {})
    _append_progress(
        progress_path,
        "preflight",
        "start",
        failed_stage_dir=str(failed_stage_dir),
        require_mps=bool(require_mps),
        region_batch=batch,
        objective_profile=str(objective_profile),
        probe_semitones=[float(value) for value in probe_semitones] if probe_semitones else [],
    )
    guard_result = build_residual_guard_from_failed_stage(
        failed_stage_dir=Path(failed_stage_dir),
        output_dir=output_dir,
        region_batch=batch,
    )
    if guard_result.get("status") != "ok":
        return _write_blocked(
            output_dir=output_dir,
            progress_path=progress_path,
            failed_stage_dir=Path(failed_stage_dir),
            status=str(guard_result.get("status")),
            failed_checks=list(guard_result.get("failed_checks") or ["residual_guard_failed"]),
            extra={
                "taxonomy_batch": dict(guard_result.get("taxonomy_batch") or batch),
                "source_region_mode": str(guard_result.get("source_region_mode") or "selected_regions"),
                "source_all_regions_path": str(guard_result.get("source_all_regions_path") or ""),
                "source_residual_region_count": int(guard_result.get("source_residual_region_count") or 0),
                "residual_region_count": int(guard_result.get("residual_region_count") or 0),
            },
        )

    guard_dir = Path(str(guard_result["guard_dir"]))
    residual_region_count = int(guard_result.get("residual_region_count") or 0)
    source_residual_region_count = int(guard_result.get("source_residual_region_count") or residual_region_count)
    _append_progress(
        progress_path,
        "train",
        "start",
        residual_region_count=residual_region_count,
        source_residual_region_count=source_residual_region_count,
        target_gain=float(target_gain),
        region_batch=dict(guard_result.get("taxonomy_batch") or {}),
        objective_profile=str(objective_profile),
        probe_semitones=[float(value) for value in probe_semitones] if probe_semitones else [],
    )
    training_manifest = run_training(
        mel_guard_dir=guard_dir,
        output_dir=output_dir / "train",
        target_gain=float(target_gain),
        epochs=int(epochs),
        require_mps=bool(require_mps),
        max_regions=int(max_regions),
        probe_semitones=probe_semitones,
        objective_profile=str(objective_profile),
    )
    _append_progress(progress_path, "train", "end", train_status=training_manifest.get("status"))
    if bool(training_manifest.get("delivery_allowed")) or bool(training_manifest.get("listen_pack_created")):
        return _write_blocked(
            output_dir=output_dir,
            progress_path=progress_path,
            failed_stage_dir=Path(failed_stage_dir),
            status="blocked_invalid_training_contract",
            failed_checks=["training_delivery_contract_invalid"],
            extra={"training_status": training_manifest.get("status")},
        )

    final_audio = dict(training_manifest.get("audio") or {})
    damage_audit: dict[str, Any] | None = None
    initial_damage_audit: dict[str, Any] | None = None
    damage_limited_repair: dict[str, Any] | None = None
    damage_audit_manifest = ""
    if str(training_manifest.get("status")) == "trained_repair_passed_pitch_guard":
        training_audio = training_manifest.get("audio")
        guard_manifest = guard_result.get("guard_manifest")
        candidate_instrumental = Path(str((training_audio if isinstance(training_audio, Mapping) else {}).get("instrumental") or ""))
        reference_audio = (guard_manifest if isinstance(guard_manifest, Mapping) else {}).get("audio") or {}
        reference_instrumental = Path(str((reference_audio if isinstance(reference_audio, Mapping) else {}).get("instrumental") or ""))
        if not candidate_instrumental.is_file() or not reference_instrumental.is_file():
            return _write_blocked(
                output_dir=output_dir,
                progress_path=progress_path,
                failed_stage_dir=Path(failed_stage_dir),
                status="blocked_missing_damage_audit_audio",
                failed_checks=["damage_audit_audio_missing"],
                extra={
                    "training_status": training_manifest.get("status"),
                    "candidate_instrumental": str(candidate_instrumental),
                    "reference_instrumental": str(reference_instrumental),
                },
            )
        damage_dir = output_dir / "damage"
        damage_score_path = damage_dir / "score.json"
        guard_score = dict((guard_manifest if isinstance(guard_manifest, Mapping) else {}).get("score") or {})
        _write_json(
            damage_score_path,
            {
                "regions": list(guard_score.get("regions") or []),
                "config": dict(guard_score.get("config") or {}),
                "source_after_score": str(guard_score.get("source_after_score") or ""),
                "source_region_mode": str(guard_score.get("source_region_mode") or ""),
                "source_all_regions_path": str(guard_score.get("source_all_regions_path") or ""),
            },
        )
        _append_progress(
            progress_path,
            "damage_audit",
            "start",
            reference_instrumental=str(reference_instrumental),
            candidate_instrumental=str(candidate_instrumental),
            score_json=str(damage_score_path),
        )
        damage_audit = run_damage_audit(
            reference_instrumental_path=reference_instrumental,
            candidate_instrumental_path=candidate_instrumental,
            score_json_path=damage_score_path,
            output_dir=damage_dir,
        )
        damage_audit_manifest = str(damage_dir / "damage.json")
        _append_progress(
            progress_path,
            "damage_audit",
            "end",
            verdict=damage_audit.get("verdict"),
            failed_region_count=damage_audit.get("failed_region_count"),
            blocker_reasons=list(damage_audit.get("blocker_reasons") or []),
        )
        initial_damage_audit = dict(damage_audit)
        if str(damage_audit.get("verdict")) != "pass":
            reference_vocals = Path(str((reference_audio if isinstance(reference_audio, Mapping) else {}).get("vocals") or ""))
            reference_raw = Path(str((reference_audio if isinstance(reference_audio, Mapping) else {}).get("raw_ref") or ""))
            damage_limited_repair = run_damage_limited_repair(
                reference_vocals_path=reference_vocals,
                reference_instrumental_path=reference_instrumental,
                reference_raw_path=reference_raw,
                candidate_instrumental_path=candidate_instrumental,
                guard_score=guard_score,
                initial_damage_audit=damage_audit,
                output_dir=output_dir / "dlim",
                progress_path=progress_path,
            )
            if str(damage_limited_repair.get("status")) == "damage_limited_repair_passed":
                final_audio = dict(damage_limited_repair.get("audio") or final_audio)
                damage_audit = dict(damage_limited_repair.get("damage_audit") or {})
                damage_audit_manifest = str(damage_limited_repair.get("damage_audit_manifest") or "")

    metrics = dict(training_manifest.get("metrics") or {})
    scope = "taxonomy_batch" if _is_scoped_batch(guard_result.get("taxonomy_batch")) else "residual_guard"
    failed_checks: list[str] = []
    status = _training_status(training_manifest, guard_result.get("taxonomy_batch"))
    if damage_limited_repair is not None and str(damage_limited_repair.get("status")) == "damage_limited_repair_passed":
        status = _damage_limited_passed_status(guard_result.get("taxonomy_batch"))
    elif damage_audit is not None and str(damage_audit.get("verdict")) != "pass":
        status = _damage_failed_status(guard_result.get("taxonomy_batch"))
        failed_checks.append("instrumental_damage_audit_failed")
    manifest = {
        **_base_manifest(failed_stage_dir=Path(failed_stage_dir), output_dir=output_dir),
        "status": status,
        "failed_checks": failed_checks,
        "residual_region_count": residual_region_count,
        "source_residual_region_count": source_residual_region_count,
        "taxonomy_batch": dict(guard_result.get("taxonomy_batch") or {}),
        "source_region_mode": str(guard_result.get("source_region_mode") or "selected_regions"),
        "source_all_regions_path": str(guard_result.get("source_all_regions_path") or ""),
        "resolution_scope": scope,
        "full_holdout_resolution_status": "not_evaluated",
        "target_gain": float(target_gain),
        "epochs": int(epochs),
        "objective_profile": str(objective_profile),
        "probe_semitones": [float(value) for value in probe_semitones] if probe_semitones else [],
        "guard_manifest": str(guard_dir / "manifest.json"),
        "training_manifest": str(output_dir / "train" / "manifest.json"),
        "checkpoint": training_manifest.get("checkpoint"),
        "training_status": training_manifest.get("status"),
        "resolved_device": training_manifest.get("resolved_device"),
        "audio": final_audio,
        "initial_damage_audit": dict(initial_damage_audit or {}),
        "damage_audit": dict(damage_audit or {}),
        "damage_audit_manifest": damage_audit_manifest,
        "damage_limited_repair": dict(damage_limited_repair or {}),
        "metrics": metrics,
        "before_region_count": metrics.get("before_region_count"),
        "after_region_count": metrics.get("after_region_count"),
        "target_issue_improved": metrics.get("target_issue_improved"),
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "residual_training_diagnostic_only",
    }
    _write_json(output_dir / "manifest.json", manifest)
    _append_progress(progress_path, "manifest", "end", run_status=manifest["status"], after_region_count=manifest.get("after_region_count"))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-stage-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-gain", type=float, default=2.5)
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--max-regions", type=int, default=8)
    parser.add_argument("--probe-semitone", action="append", type=float, dest="probe_semitones")
    parser.add_argument(
        "--objective-profile",
        default="projection_teacher",
        choices=[
            "projection_teacher",
            "multi_probe_anti_vocal",
            "pitch_region_count_attack_v2",
            "pitch_region_detector_objective_v3",
            "pitch_region_detector_damage_guarded_v1",
            "pitch_region_detector_damage_aware_v2",
            "pitch_region_detector_damage_margin_v3",
        ],
    )
    parser.add_argument("--region-batch-json", default="")
    parser.add_argument("--require-mps", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_residual_training(
        failed_stage_dir=args.failed_stage_dir,
        output_dir=args.out,
        target_gain=float(args.target_gain),
        epochs=int(args.epochs),
        require_mps=bool(args.require_mps),
        max_regions=int(args.max_regions),
        region_batch=json.loads(args.region_batch_json) if str(args.region_batch_json or "").strip() else None,
        probe_semitones=args.probe_semitones,
        objective_profile=str(args.objective_profile),
    )
    print(json.dumps({"manifest": str(args.out / "manifest.json"), "status": manifest["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
