#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from mac_melband_pitch_guard import (  # noqa: E402
    score_best_value,
    score_pitch_shifted_vocal_leak,
    score_region_count,
    score_target_issue_improved,
)
from melband_pitch_attribution_adapter import (  # noqa: E402
    OBJECTIVE_NAME,
    PitchAttributionAdapterConfig,
    apply_pitch_attribution_adapter,
    resolve_pitch_attribution_device,
    train_pitch_attribution_adapter_from_arrays,
    validate_pitch_attribution_label_contract,
)
from stem_region_repair_common import read_audio_stereo, write_audio, write_json  # noqa: E402
from train_mac_refiner_from_mel_guard import (  # noqa: E402
    _build_training_source_condition,
    _fit_audio,
    _load_json,
    _projection_coeff,
    _resolve_audio_path,
    _rms,
    _score_has_guard_regions,
)

START_TIME = time.monotonic()


def _append_progress(path: Path, phase: str, status: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "elapsed_sec": round(time.monotonic() - START_TIME, 6),
        "phase": str(phase),
        "status": str(status),
        **fields,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _best_score(score: dict[str, Any]) -> float:
    return score_best_value(score)


def build_regionwise_target_delta(
    *,
    instrumental_audio: np.ndarray,
    source_condition: np.ndarray,
    region_mask: np.ndarray,
    regions: Sequence[dict[str, Any]],
    target_gain: float,
    sample_rate: int | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    instrumental = np.asarray(instrumental_audio, dtype=np.float32)
    source = np.asarray(source_condition, dtype=np.float32)
    mask = np.asarray(region_mask, dtype=np.float32).reshape(-1)
    frame_count = min(int(instrumental.shape[0]), int(source.shape[0]), int(mask.shape[0]))
    channels = min(int(instrumental.shape[1]), int(source.shape[1]))
    target_delta = np.zeros((frame_count, channels), dtype=np.float32)
    coeff_rows: list[dict[str, Any]] = []
    for region_index, region in enumerate(regions):
        if "start_frame" in region or "end_frame" in region:
            start_frame = int(region.get("start_frame", 0))
            end_frame = int(region.get("end_frame", 0))
        else:
            if sample_rate is None:
                raise ValueError("sample_rate is required for second-based regions")
            start_frame = int(round(float(region.get("start", 0.0)) * float(sample_rate)))
            end_frame = int(round(float(region.get("end", 0.0)) * float(sample_rate)))
        start_i = max(0, min(frame_count, start_frame))
        end_i = max(start_i, min(frame_count, end_frame))
        if end_i <= start_i:
            continue
        local_mask = mask[start_i:end_i]
        if float(np.max(local_mask)) <= 0.0:
            continue
        inst_seg = instrumental[start_i:end_i, :channels]
        src_seg = source[start_i:end_i, :channels]
        coeff = _projection_coeff(inst_seg, src_seg, local_mask)
        coeff = max(-2.5, min(2.5, coeff * float(target_gain)))
        target_delta[start_i:end_i, :channels] += (coeff * src_seg * local_mask[:, None]).astype(np.float32)
        coeff_rows.append(
            {
                "region_index": int(region_index),
                "start_frame": int(start_i),
                "end_frame": int(end_i),
                "start_sec": round(float(start_i) / float(sample_rate), 6) if sample_rate else None,
                "end_sec": round(float(end_i) / float(sample_rate), 6) if sample_rate else None,
                "projection_coeff": float(coeff),
            }
        )
    return np.clip(target_delta, -1.0, 1.0).astype(np.float32), coeff_rows


def _status_for_scores(before_score: dict[str, Any], after_score: dict[str, Any]) -> tuple[str, str, str, bool]:
    target_issue_improved = score_target_issue_improved(before_score=before_score, after_score=after_score)
    if str(after_score.get("verdict")) == "pass":
        return "trained_adapter_passed_pitch_guard", "resolved", "validate_or_replay", True
    if target_issue_improved:
        return "trained_adapter_improved_still_blocked", "adapter_gap", "continue_separator_adapter_training", True
    return "trained_adapter_model_gap", "model_gap", "change_separator_objective_or_architecture", False


def run_training(
    *,
    mel_guard_dir: Path,
    output_dir: Path,
    target_gain: float = 1.0,
    epochs: int = 160,
    require_mps: bool = True,
    device: str = "mps",
    adapter_mode: str = "global",
    max_regions: int = 16,
    probe_semitones: Sequence[float] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    _append_progress(
        progress_path,
        "preflight",
        "start",
        objective=OBJECTIVE_NAME,
        device_requested=str(device),
        require_mps=bool(require_mps),
        delivery_allowed=False,
    )
    try:
        resolved_device = resolve_pitch_attribution_device("mps" if bool(require_mps) else str(device))
    except Exception as exc:
        manifest = {
            "schema": "melband_pitch_attribution_adapter_guard_training.v1",
            "objective": OBJECTIVE_NAME,
            "status": "blocked_mps_unavailable" if bool(require_mps) else "blocked_device_unavailable",
            "training_level": "separator_adapter",
            "must_train_weights": True,
            "resolved_device": "unavailable",
            "device_error": str(exc),
            "delivery_status": "blocked",
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "training_diagnostic_only",
        }
        write_json(output_dir / "manifest.json", manifest)
        _append_progress(progress_path, "preflight", "blocked", reason=manifest["status"], error=str(exc))
        return manifest

    validate_pitch_attribution_label_contract(label_type="pseudo_teacher", is_ground_truth=False)
    guard_manifest = _load_json(Path(mel_guard_dir) / "manifest.json")
    audio_map = dict(guard_manifest.get("audio") or {})
    vocals_path = _resolve_audio_path(str(audio_map.get("vocals")), guard_dir=Path(mel_guard_dir))
    instrumental_path = _resolve_audio_path(str(audio_map.get("instrumental")), guard_dir=Path(mel_guard_dir))
    raw_path = _resolve_audio_path(str(audio_map.get("raw_ref")), guard_dir=Path(mel_guard_dir))
    if _score_has_guard_regions(guard_manifest.get("score")):
        before_score = dict(guard_manifest["score"])
        write_json(output_dir / "score" / "before" / "pitch.json", before_score)
        _append_progress(
            progress_path,
            "before_score",
            "reused_guard_manifest",
            region_count=score_region_count(before_score),
            selected_region_count=len(before_score.get("regions") or []),
            verdict=before_score.get("verdict"),
        )
    else:
        _append_progress(progress_path, "before_score", "scan_start")
        before_score = score_pitch_shifted_vocal_leak(
            vocals_path=vocals_path,
            instrumental_path=instrumental_path,
            output_dir=output_dir / "score" / "before",
            config=(guard_manifest.get("score") or {}).get("config"),
            progress_callback=lambda event: _append_progress(progress_path, "before_score", "progress", **event),
        )
        _append_progress(
            progress_path,
            "before_score",
            "scan_end",
            region_count=score_region_count(before_score),
            selected_region_count=len(before_score.get("regions") or []),
            verdict=before_score.get("verdict"),
        )

    regions = list(before_score.get("regions") or [])[: int(max_regions)]
    if not regions:
        manifest = {
            "schema": "melband_pitch_attribution_adapter_guard_training.v1",
            "objective": OBJECTIVE_NAME,
            "status": "skipped_no_guard_regions",
            "training_level": "separator_adapter",
            "must_train_weights": True,
            "delivery_status": "blocked",
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "training_diagnostic_only",
            "before_score": before_score,
        }
        write_json(output_dir / "manifest.json", manifest)
        return manifest

    vocals_audio, vocals_rate = read_audio_stereo(vocals_path)
    instrumental_audio, instrumental_rate = read_audio_stereo(instrumental_path)
    raw_audio, raw_rate = read_audio_stereo(raw_path)
    if len({vocals_rate, instrumental_rate, raw_rate}) != 1:
        raise ValueError("pitch_attribution_adapter_sample_rate_mismatch")
    vocals_audio, instrumental_audio, raw_audio = _fit_audio(vocals_audio, instrumental_audio, raw_audio)
    region_mask, source_condition, applied_regions, source_condition_mode, normalized_probe_semitones = (
        _build_training_source_condition(
            vocals_audio=vocals_audio,
            sample_rate=vocals_rate,
            regions=regions,
            pad_sec=0.08,
            fade_sec=0.03,
            probe_semitones=probe_semitones,
        )
    )
    coeff = _projection_coeff(instrumental_audio, source_condition, region_mask)
    coeff = max(-2.5, min(2.5, coeff * float(target_gain)))
    if str(adapter_mode) == "conditioned":
        target_delta, region_projection_coeffs = build_regionwise_target_delta(
            instrumental_audio=instrumental_audio,
            source_condition=source_condition,
            region_mask=region_mask,
            regions=applied_regions,
            target_gain=float(target_gain),
            sample_rate=vocals_rate,
        )
    else:
        target_delta = np.clip(coeff * source_condition * region_mask[:, None], -1.0, 1.0).astype(np.float32)
        region_projection_coeffs = []
    audio_dir = output_dir / "audio"
    write_audio(audio_dir / "raw.flac", raw_audio, vocals_rate)
    write_audio(audio_dir / "src.flac", source_condition, vocals_rate)
    write_audio(audio_dir / "d.flac", target_delta, vocals_rate)

    config = PitchAttributionAdapterConfig(
        epochs=int(epochs),
        learning_rate=0.05,
        attribution_loss_weight=1.0,
        outside_loss_weight=0.8,
        mixture_loss_weight=0.05,
        max_probe_gain=2.5,
        adapter_mode=str(adapter_mode),
        conditioned_hidden_channels=16,
        device=str(resolved_device),
    )
    _append_progress(
        progress_path,
        "train",
        "start",
        device=str(resolved_device),
        epochs=int(epochs),
        projection_coeff=coeff,
        adapter_mode=str(adapter_mode),
        region_count=len(regions),
        active_region_fraction=float(np.mean(region_mask, dtype=np.float64)),
    )
    model, train_diagnostics = train_pitch_attribution_adapter_from_arrays(
        mix_audio=raw_audio,
        vocals_audio=vocals_audio,
        instrumental_audio=instrumental_audio,
        source_probe_audio=source_condition,
        region_mask=region_mask,
        target_delta_audio=target_delta,
        config=config,
        progress_path=progress_path,
    )
    _append_progress(
        progress_path,
        "train",
        "end",
        final_loss=train_diagnostics.get("final_loss"),
        weights_changed=train_diagnostics.get("weights_changed"),
    )
    repaired_vocals, repaired_instrumental, delta, apply_diag = apply_pitch_attribution_adapter(
        model=model,
        mix_audio=raw_audio,
        vocals_audio=vocals_audio,
        instrumental_audio=instrumental_audio,
        source_probe_audio=source_condition,
        region_mask=region_mask,
        config=config,
    )
    written_vocals = np.clip(repaired_vocals, -1.0, 1.0).astype(np.float32)
    written_instrumental = np.clip(repaired_instrumental, -1.0, 1.0).astype(np.float32)
    vocals_out = write_audio(audio_dir / "v.flac", written_vocals, vocals_rate)
    instrumental_out = write_audio(audio_dir / "i.flac", written_instrumental, vocals_rate)
    checkpoint_path = output_dir / "model.pt"
    torch.save(
        {
            "schema": "melband_pitch_attribution_adapter_guard_training.v1",
            "objective": OBJECTIVE_NAME,
            "training_level": "separator_adapter",
            "adapter_mode": str(adapter_mode),
            "config": asdict(config),
            "region_projection_coeffs": region_projection_coeffs,
            "state_dict": model.state_dict(),
            "channels": int(vocals_audio.shape[1]),
        },
        str(checkpoint_path),
    )
    _append_progress(progress_path, "after_score", "scan_start")
    after_score = score_pitch_shifted_vocal_leak(
        vocals_path=vocals_out,
        instrumental_path=instrumental_out,
        output_dir=output_dir / "score" / "after",
        config=before_score.get("config"),
        progress_callback=lambda event: _append_progress(progress_path, "after_score", "progress", **event),
    )
    _append_progress(
        progress_path,
        "after_score",
        "scan_end",
        region_count=score_region_count(after_score),
        selected_region_count=len(after_score.get("regions") or []),
        verdict=after_score.get("verdict"),
    )
    status, root_cause, next_training_action, target_issue_improved = _status_for_scores(before_score, after_score)
    before_count = score_region_count(before_score)
    after_count = score_region_count(after_score)
    before_selected_count = len(before_score.get("regions") or [])
    after_selected_count = len(after_score.get("regions") or [])
    manifest = {
        "schema": "melband_pitch_attribution_adapter_guard_training.v1",
        "objective": OBJECTIVE_NAME,
        "status": status,
        "source_guard_dir": str(mel_guard_dir),
        "checkpoint": str(checkpoint_path),
        "resolved_device": str(resolved_device),
        "training_level": "separator_adapter",
        "target_model_family": "melband_roformer_vocals_or_adapter",
        "must_train_weights": True,
        "weights_changed": bool(train_diagnostics.get("weights_changed")),
        "label_type": "pseudo_teacher",
        "is_ground_truth": False,
        "usable_for_promotion": False,
        "target_gain": float(target_gain),
        "projection_coeff": coeff,
        "region_projection_coeffs": region_projection_coeffs,
        "adapter_mode": str(adapter_mode),
        "root_cause": root_cause,
        "next_training_action": next_training_action,
        "source_condition_mode": source_condition_mode,
        "probe_semitones": normalized_probe_semitones,
        "applied_regions": applied_regions,
        "audio": {
            "vocals": str(vocals_out),
            "instrumental": str(instrumental_out),
            "raw_ref": str(audio_dir / "raw.flac"),
            "source_probe": str(audio_dir / "src.flac"),
            "target_delta": str(audio_dir / "d.flac"),
        },
        "before_score": before_score,
        "after_score": after_score,
        "metrics": {
            "before_region_count": before_count,
            "after_region_count": after_count,
            "before_selected_region_count": before_selected_count,
            "after_selected_region_count": after_selected_count,
            "region_count_reduced": bool(after_count < before_count),
            "before_best_score": _best_score(before_score),
            "after_best_score": _best_score(after_score),
            "target_issue_improved": bool(target_issue_improved),
            "reconstruction_rms": _rms(written_vocals + written_instrumental - raw_audio),
            "delta_rms": _rms(delta),
            "apply_diagnostics": apply_diag,
        },
        "training": train_diagnostics,
        "contract": {
            "pseudo_teacher_only": True,
            "diagnostic_only": True,
            "no_listen_delivery": True,
            "cannot_override_gates": True,
            "mixture_consistent_by_construction": True,
            "delivery_allowed": False,
            "usable_for_promotion": False,
        },
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "training_diagnostic_only",
    }
    write_json(output_dir / "manifest.json", manifest)
    _append_progress(
        progress_path,
        "manifest",
        "end",
        run_status=status,
        before_region_count=before_count,
        after_region_count=after_count,
        target_issue_improved=bool(target_issue_improved),
        delivery_allowed=False,
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mel-guard-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-gain", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--max-regions", type=int, default=16)
    parser.add_argument("--probe-semitone", action="append", type=float, dest="probe_semitones")
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--adapter-mode", default="global", choices=["global", "conditioned"])
    parser.add_argument("--require-mps", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_training(
        mel_guard_dir=args.mel_guard_dir,
        output_dir=args.out,
        target_gain=float(args.target_gain),
        epochs=int(args.epochs),
        require_mps=bool(args.require_mps),
        device=str(args.device),
        adapter_mode=str(args.adapter_mode),
        max_regions=int(args.max_regions),
        probe_semitones=args.probe_semitones,
    )
    print(json.dumps({"manifest": str(args.out / "manifest.json"), "status": manifest["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
