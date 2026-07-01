#!/usr/bin/env python3
"""Compose diagnostic Mac residual chunk repairs and rescore the full guard.

This tool is diagnostic-only. It does not run model inference and does not
publish listen output. It applies each residual chunk's instrumental delta only
inside that chunk's recorded applied regions, then reruns the full pitch guard.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.mac_melband_pitch_guard import (
    score_best_value,
    score_pitch_shifted_vocal_leak,
    score_region_count,
    score_target_issue_improved_from_values,
)
from tools.stem_region_repair_common import make_region_mask, read_audio_stereo, write_audio


def _append_progress(path: Path, phase: str, status: str, **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_unix": time.time(), "phase": phase, "status": status, **extra}, sort_keys=True) + "\n")
        handle.flush()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(audio, dtype=np.float64)))))


def _fit_audio(*arrays: np.ndarray) -> list[np.ndarray]:
    frames = min(array.shape[0] for array in arrays)
    return [np.asarray(array[:frames], dtype=np.float32) for array in arrays]


def _resolve_existing_file(value: Any, *, base_dir: Path, accepted_tails: Sequence[tuple[str, ...]] = ()) -> Path:
    path = Path(str(value or ""))
    if path.is_file():
        return path
    if str(value or "") and not path.is_absolute():
        candidate = base_dir / path
        if candidate.is_file():
            return candidate
    parts = tuple(path.parts)
    for tail in accepted_tails:
        if len(parts) >= len(tail) and parts[-len(tail) :] == tuple(tail):
            candidate = base_dir / Path(*tail)
            if candidate.is_file():
                return candidate
    if path.name:
        candidate = base_dir / path.name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing file: {value}")


def _resolve_audio_path(value: Any, *, audio_base_dir: Path) -> Path:
    path = Path(str(value or ""))
    if path.is_file():
        return path
    if str(value or "") and not path.is_absolute():
        candidate = audio_base_dir / path
        if candidate.is_file():
            return candidate
    if path.name:
        candidate = audio_base_dir / path.name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing audio file: {value}")


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return str(left) == str(right)


def _base_manifest(*, failed_stage_dir: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "schema": "mac_mel_residual_chunk_composition.v1",
        "source_failed_stage_dir": str(failed_stage_dir),
        "output_dir": str(output_dir),
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "chunk_composition_diagnostic_only",
        "contract": {
            "diagnostic_only": True,
            "no_listen_delivery": True,
            "no_threshold_relaxation": True,
            "chunk_delta_composition": True,
            "mixture_consistent_delta_transfer": True,
            "model_inference": False,
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


def _load_failed_stage_audio(*, failed_stage_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Path]]:
    stage_manifest = _load_json(failed_stage_dir / "manifest.json")
    replay_manifest_path = _resolve_existing_file(
        stage_manifest.get("replay_manifest") or "replay/manifest.json",
        base_dir=failed_stage_dir,
        accepted_tails=(("replay", "manifest.json"),),
    )
    replay_manifest = _load_json(replay_manifest_path)
    score_path = replay_manifest_path.parent / "score" / "after" / "pitch.json"
    before_score = _load_json(score_path) if score_path.is_file() else dict(replay_manifest.get("after_score") or {})
    replay_audio = dict(stage_manifest.get("audio") or replay_manifest.get("audio") or {})
    audio_base_dir = replay_manifest_path.parent / "audio"
    audio_paths = {
        "vocals": _resolve_audio_path(replay_audio.get("vocals"), audio_base_dir=audio_base_dir),
        "instrumental": _resolve_audio_path(replay_audio.get("instrumental"), audio_base_dir=audio_base_dir),
        "raw_ref": _resolve_audio_path(replay_audio.get("raw_ref"), audio_base_dir=audio_base_dir),
    }
    return stage_manifest, replay_manifest, before_score, audio_paths


def _load_residual_chunk(residual_manifest_path: Path) -> dict[str, Any]:
    residual_manifest = _load_json(Path(residual_manifest_path))
    if bool(residual_manifest.get("delivery_allowed")) or bool(residual_manifest.get("listen_pack_created")):
        raise ValueError("residual_delivery_contract_invalid")
    train_manifest_path = _resolve_existing_file(
        residual_manifest.get("training_manifest") or "train/manifest.json",
        base_dir=Path(residual_manifest_path).parent,
        accepted_tails=(("train", "manifest.json"),),
    )
    training_manifest = _load_json(train_manifest_path)
    if bool(training_manifest.get("delivery_allowed")) or bool(training_manifest.get("listen_pack_created")):
        raise ValueError("training_delivery_contract_invalid")
    audio = dict(training_manifest.get("audio") or residual_manifest.get("audio") or {})
    audio_base_dir = train_manifest_path.parent / "audio"
    instrumental_path = _resolve_audio_path(audio.get("instrumental"), audio_base_dir=audio_base_dir)
    applied_regions = [
        dict(region)
        for region in list(training_manifest.get("applied_regions") or residual_manifest.get("applied_regions") or [])
        if isinstance(region, Mapping)
    ]
    if not applied_regions:
        raise ValueError("residual_applied_regions_missing")
    return {
        "manifest_path": str(residual_manifest_path),
        "manifest": residual_manifest,
        "training_manifest_path": str(train_manifest_path),
        "training_manifest": training_manifest,
        "instrumental_path": instrumental_path,
        "applied_regions": applied_regions,
    }


def _residual_source_matches(residual_manifest: Mapping[str, Any], *, failed_stage_dir: Path) -> bool:
    source = residual_manifest.get("source_failed_stage_dir")
    if not source:
        return False
    return _same_path(Path(str(source)), Path(failed_stage_dir))


def _mask_from_regions(*, frame_count: int, sample_rate: int, regions: Sequence[Mapping[str, Any]], fade_sec: float) -> np.ndarray:
    mask = np.zeros(int(frame_count), dtype=np.float32)
    for region in regions:
        start_sec = float(region.get("start", region.get("start_sec", 0.0)) or 0.0)
        end_sec = float(region.get("end", region.get("end_sec", 0.0)) or 0.0)
        if end_sec <= start_sec:
            continue
        local = make_region_mask(
            frame_count=frame_count,
            sample_rate=sample_rate,
            start_sec=start_sec,
            end_sec=end_sec,
            fade_sec=float(fade_sec),
        )
        mask = np.maximum(mask, local)
    return mask.astype(np.float32)


def _target_issue_improved(*, before_score: Mapping[str, Any], after_score: Mapping[str, Any]) -> bool:
    return score_target_issue_improved_from_values(
        before_region_count=score_region_count(before_score),
        after_region_count=score_region_count(after_score),
        before_best_score=score_best_value(before_score),
        after_best_score=score_best_value(after_score),
        after_verdict=str(after_score.get("verdict")),
    )


def _score_progress_callback(progress_path: Path):
    def _callback(event: Mapping[str, Any]) -> None:
        payload = dict(event)
        status = str(payload.pop("event", "progress"))
        _append_progress(progress_path, "score_after", status, **payload)

    return _callback


def run_chunk_composition(
    *,
    failed_stage_dir: Path,
    residual_manifest_paths: Sequence[Path],
    output_dir: Path,
    fade_sec: float = 0.03,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    residual_paths = [Path(path) for path in residual_manifest_paths]
    _append_progress(
        progress_path,
        "preflight",
        "start",
        residual_count=len(residual_paths),
        gpu_inference_skipped=True,
        gpu_inference_skipped_reason="chunk_repair_composition_uses_existing_residual_audio",
        acceleration_mode="cpu_dsp_scoring",
    )
    if not residual_paths:
        return _write_blocked(
            output_dir=output_dir,
            progress_path=progress_path,
            failed_stage_dir=Path(failed_stage_dir),
            status="blocked_no_residual_manifests",
            failed_checks=["no_residual_manifests"],
        )

    stage_manifest, replay_manifest, before_score, source_audio_paths = _load_failed_stage_audio(failed_stage_dir=Path(failed_stage_dir))
    if bool(stage_manifest.get("delivery_allowed")) or bool(stage_manifest.get("listen_pack_created")):
        return _write_blocked(
            output_dir=output_dir,
            progress_path=progress_path,
            failed_stage_dir=Path(failed_stage_dir),
            status="blocked_invalid_failed_stage_contract",
            failed_checks=["failed_stage_delivery_contract_invalid"],
        )
    if bool(replay_manifest.get("delivery_allowed")) or bool(replay_manifest.get("listen_pack_created")):
        return _write_blocked(
            output_dir=output_dir,
            progress_path=progress_path,
            failed_stage_dir=Path(failed_stage_dir),
            status="blocked_invalid_replay_contract",
            failed_checks=["replay_delivery_contract_invalid"],
        )

    vocals_audio, vocals_rate = read_audio_stereo(source_audio_paths["vocals"])
    instrumental_audio, instrumental_rate = read_audio_stereo(source_audio_paths["instrumental"])
    raw_audio, raw_rate = read_audio_stereo(source_audio_paths["raw_ref"])
    if len({vocals_rate, instrumental_rate, raw_rate}) != 1:
        raise ValueError("chunk_composition_source_sample_rate_mismatch")
    vocals_audio, instrumental_audio, raw_audio = _fit_audio(vocals_audio, instrumental_audio, raw_audio)
    sample_rate = int(vocals_rate)
    delta_sum = np.zeros_like(instrumental_audio, dtype=np.float32)
    chunk_summaries: list[dict[str, Any]] = []
    _append_progress(progress_path, "compose", "start", chunk_count=len(residual_paths), sample_rate=sample_rate)

    for index, residual_path in enumerate(residual_paths):
        _append_progress(progress_path, "chunk", "start", chunk_index=index, manifest=str(residual_path))
        try:
            chunk = _load_residual_chunk(residual_path)
        except ValueError as exc:
            reason = str(exc)
            failed_check = reason if reason.endswith("_invalid") else reason
            status = "blocked_invalid_residual_contract" if "delivery_contract_invalid" in reason else "blocked_invalid_residual_chunk"
            return _write_blocked(
                output_dir=output_dir,
                progress_path=progress_path,
                failed_stage_dir=Path(failed_stage_dir),
                status=status,
                failed_checks=[failed_check],
                extra={"residual_manifest": str(residual_path), "chunk_index": index},
            )
        if not _residual_source_matches(dict(chunk["manifest"]), failed_stage_dir=Path(failed_stage_dir)):
            return _write_blocked(
                output_dir=output_dir,
                progress_path=progress_path,
                failed_stage_dir=Path(failed_stage_dir),
                status="blocked_residual_source_mismatch",
                failed_checks=["residual_source_failed_stage_mismatch"],
                extra={
                    "residual_manifest": str(residual_path),
                    "chunk_index": index,
                    "residual_source_failed_stage_dir": dict(chunk["manifest"]).get("source_failed_stage_dir"),
                    "source_failed_stage_dir": str(failed_stage_dir),
                },
            )
        candidate_instrumental, candidate_rate = read_audio_stereo(chunk["instrumental_path"])
        if int(candidate_rate) != sample_rate:
            raise ValueError("chunk_composition_residual_sample_rate_mismatch")
        base_fit, candidate_fit = _fit_audio(instrumental_audio, candidate_instrumental)
        mask = _mask_from_regions(
            frame_count=base_fit.shape[0],
            sample_rate=sample_rate,
            regions=chunk["applied_regions"],
            fade_sec=float(fade_sec),
        )
        if not np.any(mask > 1.0e-4):
            raise ValueError("chunk_composition_empty_region_mask")
        delta = candidate_fit - base_fit
        if delta.shape[0] < delta_sum.shape[0]:
            padded_delta = np.zeros_like(delta_sum)
            padded_mask = np.zeros(delta_sum.shape[0], dtype=np.float32)
            padded_delta[: delta.shape[0]] = delta
            padded_mask[: mask.shape[0]] = mask
            delta = padded_delta
            mask = padded_mask
        delta_sum += delta[: delta_sum.shape[0]] * mask[: delta_sum.shape[0], None]
        residual_manifest = dict(chunk["manifest"])
        chunk_summaries.append(
            {
                "chunk_index": index,
                "residual_manifest": str(residual_path),
                "training_manifest": str(chunk["training_manifest_path"]),
                "status": residual_manifest.get("status"),
                "taxonomy_batch": dict(residual_manifest.get("taxonomy_batch") or {}),
                "applied_region_count": len(chunk["applied_regions"]),
                "delta_rms": _rms(delta[: delta_sum.shape[0]] * mask[: delta_sum.shape[0], None]),
            }
        )
        _append_progress(progress_path, "chunk", "end", chunk_index=index, applied_region_count=len(chunk["applied_regions"]))

    composed_instrumental = np.clip(instrumental_audio + delta_sum, -1.0, 1.0).astype(np.float32)
    composed_vocals = np.clip(raw_audio - composed_instrumental, -1.0, 1.0).astype(np.float32)
    audio_dir = output_dir / "audio"
    vocals_out = write_audio(audio_dir / "v.flac", composed_vocals, sample_rate)
    instrumental_out = write_audio(audio_dir / "i.flac", composed_instrumental, sample_rate)
    raw_out = write_audio(audio_dir / "raw.flac", raw_audio, sample_rate)
    _append_progress(progress_path, "compose", "end", chunk_count=len(chunk_summaries), delta_rms=_rms(delta_sum))

    after_score = score_pitch_shifted_vocal_leak(
        vocals_path=vocals_out,
        instrumental_path=instrumental_out,
        output_dir=output_dir / "score" / "after",
        config=before_score.get("config") if isinstance(before_score, Mapping) else None,
        progress_callback=_score_progress_callback(progress_path),
    )
    target_issue_improved = _target_issue_improved(before_score=before_score, after_score=after_score)
    after_verdict = str(after_score.get("verdict"))
    status = "composition_passed_pitch_guard" if after_verdict == "pass" else "composition_blocked_pitch_guard"
    before_region_count = score_region_count(before_score)
    after_region_count = score_region_count(after_score)
    manifest = {
        **_base_manifest(failed_stage_dir=Path(failed_stage_dir), output_dir=output_dir),
        "status": status,
        "failed_checks": [] if after_verdict == "pass" else ["pitch_guard_failed"],
        "source_replay_manifest": str(Path(failed_stage_dir) / "replay" / "manifest.json"),
        "residual_manifest_count": len(residual_paths),
        "applied_chunk_count": len(chunk_summaries),
        "chunks": chunk_summaries,
        "audio": {
            "vocals": str(vocals_out),
            "instrumental": str(instrumental_out),
            "raw_ref": str(raw_out),
        },
        "before_score": before_score,
        "after_score": after_score,
        "metrics": {
            "before_region_count": before_region_count,
            "after_region_count": after_region_count,
            "before_best_score": score_best_value(before_score),
            "after_best_score": score_best_value(after_score),
            "target_issue_improved": bool(target_issue_improved),
            "instrumental_delta_rms": _rms(delta_sum),
            "reconstruction_rms": _rms(composed_vocals + composed_instrumental - raw_audio),
            "original_reconstruction_rms": _rms(vocals_audio + instrumental_audio - raw_audio),
        },
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "chunk_composition_diagnostic_only",
    }
    _write_json(output_dir / "manifest.json", manifest)
    _append_progress(
        progress_path,
        "manifest",
        "end",
        run_status=status,
        before_region_count=before_region_count,
        after_region_count=after_region_count,
        target_issue_improved=bool(target_issue_improved),
    )
    return manifest


def _residual_manifests_from_root(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.glob("r*/manifest.json") if path.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-stage-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--residual-manifest", type=Path, action="append", default=[])
    parser.add_argument("--residual-root", type=Path)
    parser.add_argument("--fade-sec", type=float, default=0.03)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    residuals = list(args.residual_manifest or [])
    if args.residual_root is not None:
        residuals.extend(_residual_manifests_from_root(args.residual_root))
    manifest = run_chunk_composition(
        failed_stage_dir=args.failed_stage_dir,
        residual_manifest_paths=residuals,
        output_dir=args.out,
        fade_sec=float(args.fade_sec),
    )
    print(json.dumps({"manifest": str(args.out / "manifest.json"), "status": manifest["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
