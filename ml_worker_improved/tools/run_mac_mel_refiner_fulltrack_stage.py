#!/usr/bin/env python3
"""Run the promoted Mac MelBand refiner checkpoint on the full track.

This is the diagnostic stage after the bounded benchmark. It never creates a
listen/result pack and cannot set delivery_allowed=true.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _append_progress(path: Path, phase: str, status: str, **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_unix": time.time(), "phase": phase, "status": status, **extra}, sort_keys=True) + "\n")
        handle.flush()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def resolve_region_repair_training_device(requested_device: str):
    from tools.stem_region_repair_model import resolve_region_repair_training_device as _resolve

    return _resolve(requested_device)


def run_refiner_replay(**kwargs: Any) -> dict[str, Any]:
    from tools.apply_mac_stem_refiner_to_mel_guard import run_refiner_replay as _run

    return _run(**kwargs)


def read_audio_stereo(path: Path) -> tuple[np.ndarray, int]:
    from tools.stem_region_repair_common import read_audio_stereo as _read

    return _read(path)


def _safe_checkpoint_alias(value: Any) -> str:
    alias = str(value or "").strip()
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    if not alias or len(alias) > 16 or any(char not in allowed for char in alias):
        raise ValueError(f"unsafe selected checkpoint id: {value!r}")
    return alias


def _base_manifest(*, benchmark_manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "schema": "mac_mel_refiner_fulltrack_stage.v1",
        "source_benchmark_manifest": str(benchmark_manifest_path),
        "output_dir": str(output_dir),
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "fulltrack_diagnostic_only",
        "contract": {
            "diagnostic_only": True,
            "no_listen_delivery": True,
            "no_threshold_relaxation": True,
            "selected_from_bounded_benchmark": True,
            "requires_benchmark_failure_taxonomy": True,
        },
    }


def _write_blocked(
    *,
    output_dir: Path,
    progress_path: Path,
    benchmark_manifest_path: Path,
    status: str,
    failed_checks: list[str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        **_base_manifest(benchmark_manifest_path=benchmark_manifest_path, output_dir=output_dir),
        "status": status,
        "failed_checks": list(failed_checks),
        **dict(extra or {}),
    }
    write_json(output_dir / "manifest.json", manifest)
    _append_progress(progress_path, "manifest", status, failed_checks=failed_checks)
    return manifest


def _rms(audio: np.ndarray) -> float:
    array = np.asarray(audio, dtype=np.float64)
    if array.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(array))))


def _stereo_width(audio: np.ndarray) -> float:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 2:
        return 0.0
    mid = array[:, 0] + array[:, 1]
    side = array[:, 0] - array[:, 1]
    return float(_rms(side) / max(_rms(mid), 1.0e-9))


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _count_map(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int] = {}
    for key, count in value.items():
        if isinstance(count, (int, float)):
            result[str(key)] = max(0, int(count))
    return dict(sorted(result.items()))


def _score_region_count(score: Mapping[str, Any] | None) -> int:
    if not isinstance(score, Mapping):
        return 0
    total_fail_regions = score.get("total_fail_regions")
    if isinstance(total_fail_regions, (int, float)):
        return max(0, int(total_fail_regions))
    taxonomy = score.get("failure_taxonomy")
    if isinstance(taxonomy, Mapping):
        taxonomy_total = taxonomy.get("total_fail_regions")
        if isinstance(taxonomy_total, (int, float)):
            return max(0, int(taxonomy_total))
    return len(score.get("regions") or [])


def _score_failure_taxonomy(score: Mapping[str, Any] | None) -> dict[str, Any]:
    taxonomy = score.get("failure_taxonomy") if isinstance(score, Mapping) else None
    taxonomy_map = taxonomy if isinstance(taxonomy, Mapping) else {}
    total_fail_regions = _score_region_count(score)
    selected_count = taxonomy_map.get("selected_region_count")
    if not isinstance(selected_count, (int, float)):
        selected_count = len(score.get("regions") or []) if isinstance(score, Mapping) else 0
    return {
        "total_fail_regions": int(total_fail_regions),
        "selected_region_count": max(0, int(selected_count)),
        "regions_capped": bool(taxonomy_map.get("regions_capped") or (int(total_fail_regions) > int(selected_count))),
        "by_fail_mode": _count_map(taxonomy_map.get("by_fail_mode")),
        "by_probe": _count_map(taxonomy_map.get("by_probe")),
    }


def _fulltrack_failure_taxonomy_summary(replay_manifest: Mapping[str, Any], *, max_regions: int) -> dict[str, Any]:
    before_score = replay_manifest.get("before_score") if isinstance(replay_manifest.get("before_score"), Mapping) else None
    after_score = replay_manifest.get("after_score") if isinstance(replay_manifest.get("after_score"), Mapping) else None
    before_taxonomy = _score_failure_taxonomy(before_score)
    after_taxonomy = _score_failure_taxonomy(after_score)
    before_regions = int(before_taxonomy["total_fail_regions"])
    after_regions = int(after_taxonomy["total_fail_regions"])
    return {
        "schema": "mac_mel_refiner_fulltrack_taxonomy_summary.v1",
        "delivery_allowed": False,
        "presentation_status": "fulltrack_taxonomy_diagnostic_only",
        "max_regions_requested": int(max_regions),
        "applied_region_count": len(replay_manifest.get("applied_regions") or []),
        "before_failure_taxonomy": before_taxonomy,
        "after_failure_taxonomy": after_taxonomy,
        "before_region_count": before_regions,
        "after_region_count": after_regions,
        "region_delta": before_regions - after_regions,
        "remaining_failure_count": after_regions,
        "coverage_status": "fulltrack_clear" if after_regions == 0 else "fulltrack_remaining_failures",
    }


def _has_benchmark_failure_taxonomy(benchmark_manifest: Mapping[str, Any]) -> bool:
    taxonomy = benchmark_manifest.get("failure_taxonomy_summary")
    return isinstance(taxonomy, Mapping) and str(taxonomy.get("schema") or "") == "mac_mel_refiner_benchmark_taxonomy_summary.v1"


def _audio_evidence(audio_map: Mapping[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for key in ("vocals", "instrumental", "raw_ref"):
        value = audio_map.get(key)
        if not value:
            continue
        audio, sample_rate = read_audio_stereo(Path(str(value)))
        evidence[key] = {
            "sample_rate": int(sample_rate),
            "frames": int(audio.shape[0]),
            "rms": round(_rms(audio), 9),
            "stereo_width": round(_stereo_width(audio), 9),
        }
    return evidence


def _damage_checks(replay_manifest: Mapping[str, Any], audio_evidence: Mapping[str, Any]) -> tuple[str, list[str]]:
    failed: list[str] = []
    after_score = dict(replay_manifest.get("after_score") or {})
    if str(after_score.get("verdict")) != "pass":
        failed.append("pitch_guard_failed")
    metrics = dict(replay_manifest.get("metrics") or {})
    for metric_name in ("original_reconstruction_rms", "reconstruction_rms", "instrumental_delta_rms"):
        if metric_name in metrics and not _finite(metrics.get(metric_name)):
            failed.append(f"{metric_name}_not_finite")
    for stem_name, stem_evidence in audio_evidence.items():
        if int(stem_evidence.get("frames") or 0) <= 0:
            failed.append(f"{stem_name}_empty_audio")
        if not _finite(stem_evidence.get("rms")) or float(stem_evidence.get("rms") or 0.0) <= 0.0:
            failed.append(f"{stem_name}_silent_or_invalid")
        if not _finite(stem_evidence.get("stereo_width")):
            failed.append(f"{stem_name}_stereo_width_invalid")
    status = "diagnostic_passed" if not failed else "diagnostic_blocked_pitch_guard" if "pitch_guard_failed" in failed else "diagnostic_blocked_damage"
    return status, failed


def run_fulltrack_diagnostic_stage(
    *,
    benchmark_manifest_path: Path,
    mel_guard_dir: Path,
    checkpoint_root: Path,
    output_dir: Path,
    require_mps: bool = True,
    max_regions: int = 8,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    _append_progress(progress_path, "preflight", "start", require_mps=bool(require_mps))
    benchmark_manifest = _load_json(Path(benchmark_manifest_path))
    if benchmark_manifest.get("outcome") != "promote_to_next_diagnostic_stage":
        return _write_blocked(
            output_dir=output_dir,
            progress_path=progress_path,
            benchmark_manifest_path=benchmark_manifest_path,
            status="blocked_benchmark_not_promoted",
            failed_checks=["benchmark_not_promoted"],
            extra={"benchmark_outcome": benchmark_manifest.get("outcome")},
        )
    if bool(benchmark_manifest.get("delivery_allowed")) or bool(benchmark_manifest.get("listen_pack_created")):
        return _write_blocked(
            output_dir=output_dir,
            progress_path=progress_path,
            benchmark_manifest_path=benchmark_manifest_path,
            status="blocked_invalid_benchmark_contract",
            failed_checks=["benchmark_delivery_contract_invalid"],
            extra={"benchmark_delivery_allowed": benchmark_manifest.get("delivery_allowed")},
        )
    if not _has_benchmark_failure_taxonomy(benchmark_manifest):
        return _write_blocked(
            output_dir=output_dir,
            progress_path=progress_path,
            benchmark_manifest_path=benchmark_manifest_path,
            status="blocked_benchmark_missing_failure_taxonomy",
            failed_checks=["benchmark_missing_failure_taxonomy_summary"],
            extra={"benchmark_outcome": benchmark_manifest.get("outcome")},
        )
    selected_checkpoint_id = _safe_checkpoint_alias(benchmark_manifest.get("selected_checkpoint_id"))
    device = resolve_region_repair_training_device("mps")
    if bool(require_mps) and getattr(device, "type", str(device)) != "mps":
        return _write_blocked(
            output_dir=output_dir,
            progress_path=progress_path,
            benchmark_manifest_path=benchmark_manifest_path,
            status="blocked_mps_unavailable",
            failed_checks=["mps_unavailable"],
            extra={"selected_checkpoint_id": selected_checkpoint_id, "resolved_device": str(device)},
        )
    checkpoint_path = Path(checkpoint_root) / selected_checkpoint_id / "model.pt"
    if not checkpoint_path.is_file():
        return _write_blocked(
            output_dir=output_dir,
            progress_path=progress_path,
            benchmark_manifest_path=benchmark_manifest_path,
            status="blocked_missing_checkpoint",
            failed_checks=["missing_selected_checkpoint"],
            extra={"selected_checkpoint_id": selected_checkpoint_id, "checkpoint": str(checkpoint_path)},
        )

    replay_dir = output_dir / "replay"
    _append_progress(progress_path, "replay", "start", selected_checkpoint_id=selected_checkpoint_id, checkpoint=str(checkpoint_path))
    replay_manifest = run_refiner_replay(
        mel_guard_dir=Path(mel_guard_dir),
        checkpoint_path=checkpoint_path,
        output_dir=replay_dir,
        require_mps=bool(require_mps),
        max_regions=int(max_regions),
    )
    replay_manifest_path = replay_dir / "manifest.json"
    _append_progress(progress_path, "replay", "end", replay_status=replay_manifest.get("status"))

    audio_map = dict(replay_manifest.get("audio") or {})
    evidence = _audio_evidence(audio_map) if audio_map else {}
    status, failed_checks = _damage_checks(replay_manifest, evidence)
    failure_taxonomy_summary = _fulltrack_failure_taxonomy_summary(replay_manifest, max_regions=int(max_regions))
    manifest = {
        **_base_manifest(benchmark_manifest_path=benchmark_manifest_path, output_dir=output_dir),
        "status": status,
        "failed_checks": failed_checks,
        "selected_checkpoint_id": selected_checkpoint_id,
        "checkpoint": str(checkpoint_path),
        "source_guard_dir": str(mel_guard_dir),
        "resolved_device": str(device),
        "benchmark_outcome": benchmark_manifest.get("outcome"),
        "replay_manifest": str(replay_manifest_path),
        "replay_status": replay_manifest.get("status"),
        "after_pitch_verdict": (dict(replay_manifest.get("after_score") or {})).get("verdict"),
        "metrics": dict(replay_manifest.get("metrics") or {}),
        "failure_taxonomy_summary": failure_taxonomy_summary,
        "audio": audio_map,
        "audio_evidence": evidence,
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "fulltrack_diagnostic_only",
    }
    write_json(output_dir / "manifest.json", manifest)
    _append_progress(progress_path, "manifest", "end", run_status=status, failed_checks=failed_checks)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--mel-guard-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-regions", type=int, default=8)
    parser.add_argument("--require-mps", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_fulltrack_diagnostic_stage(
        benchmark_manifest_path=args.benchmark_manifest,
        mel_guard_dir=args.mel_guard_dir,
        checkpoint_root=args.checkpoint_root,
        output_dir=args.out,
        require_mps=bool(args.require_mps),
        max_regions=int(args.max_regions),
    )
    print(json.dumps({"manifest": str(args.out / "manifest.json"), "status": manifest["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
