#!/usr/bin/env python3
"""Validate a two-stage Mac MelBand refiner chain.

This is TOP-2 validation evidence. It composes two checkpoints but never emits
listen/result delivery. If validation uses the same guard that produced the
residual training checkpoint, the run is marked as training-overlap blocked even
when audio gates pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

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


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def run_refiner_replay(**kwargs: Any) -> dict[str, Any]:
    from tools.apply_mac_stem_refiner_to_mel_guard import run_refiner_replay as _run

    return _run(**kwargs)


def _base_manifest(*, mel_guard_dir: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "schema": "mac_mel_refiner_chain_validation.v1",
        "source_guard_dir": str(mel_guard_dir),
        "output_dir": str(output_dir),
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "chain_validation_diagnostic_only",
        "contract": {
            "diagnostic_only": True,
            "no_listen_delivery": True,
            "no_threshold_relaxation": True,
            "chain_validation_only": True,
        },
    }


def _status_from_checks(*, final_verdict: str, training_overlap: bool) -> tuple[str, list[str]]:
    failed: list[str] = []
    if final_verdict != "pass":
        failed.append("pitch_guard_failed")
    if training_overlap:
        failed.append("training_overlap")
    if "training_overlap" in failed:
        return "validation_blocked_training_overlap", failed
    if "pitch_guard_failed" in failed:
        return "validation_blocked_pitch_guard", failed
    return "validation_passed", []


def _resolve_path(value: Any, *, base_dir: Path) -> Path:
    path = Path(str(value or ""))
    if path.exists():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"missing path: {value}")


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return str(left) == str(right)


def _training_overlap(*, mel_guard_dir: Path, residual_training_manifest_path: Path | None) -> tuple[bool, dict[str, Any]]:
    if residual_training_manifest_path is None:
        return False, {}
    residual_manifest = _load_json(Path(residual_training_manifest_path))
    failed_stage_dir = residual_manifest.get("source_failed_stage_dir")
    if not failed_stage_dir:
        return False, {"residual_training_manifest": str(residual_training_manifest_path)}
    failed_manifest_path = Path(str(failed_stage_dir)) / "manifest.json"
    if not failed_manifest_path.is_file():
        return False, {"residual_training_manifest": str(residual_training_manifest_path), "failed_stage_manifest_missing": str(failed_manifest_path)}
    failed_manifest = _load_json(failed_manifest_path)
    trained_guard = failed_manifest.get("source_guard_dir")
    if not trained_guard:
        return False, {"residual_training_manifest": str(residual_training_manifest_path), "failed_stage_manifest": str(failed_manifest_path)}
    overlap = _same_path(Path(mel_guard_dir), Path(str(trained_guard)))
    return overlap, {
        "residual_training_manifest": str(residual_training_manifest_path),
        "failed_stage_manifest": str(failed_manifest_path),
        "trained_guard_dir": str(trained_guard),
        "validation_guard_dir": str(mel_guard_dir),
    }


def _write_intermediate_guard(*, first_replay_manifest: Mapping[str, Any], output_dir: Path) -> Path:
    guard_dir = output_dir / "guard_s2"
    audio = dict(first_replay_manifest.get("audio") or {})
    if not audio:
        raise ValueError("first stage did not produce audio map")
    guard_manifest = {
        "schema": "mac_mel_chain_intermediate_guard.v1",
        "audio": {
            "vocals": str(audio["vocals"]),
            "instrumental": str(audio["instrumental"]),
            "raw_ref": str(audio["raw_ref"]),
        },
        "score": {"config": dict((first_replay_manifest.get("after_score") or {}).get("config") or {})},
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "chain_intermediate_guard_only",
    }
    _write_json(guard_dir / "manifest.json", guard_manifest)
    return guard_dir


def run_chain_validation(
    *,
    mel_guard_dir: Path,
    first_checkpoint: Path,
    residual_checkpoint: Path,
    output_dir: Path,
    residual_training_manifest_path: Path | None = None,
    require_mps: bool = True,
    max_regions: int = 8,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    _append_progress(progress_path, "preflight", "start", require_mps=bool(require_mps))
    if not Path(first_checkpoint).is_file():
        raise FileNotFoundError(f"missing first checkpoint: {first_checkpoint}")
    if not Path(residual_checkpoint).is_file():
        raise FileNotFoundError(f"missing residual checkpoint: {residual_checkpoint}")
    overlap, overlap_evidence = _training_overlap(
        mel_guard_dir=Path(mel_guard_dir),
        residual_training_manifest_path=Path(residual_training_manifest_path) if residual_training_manifest_path else None,
    )

    _append_progress(progress_path, "stage1", "start", checkpoint=str(first_checkpoint))
    stage1 = run_refiner_replay(
        mel_guard_dir=Path(mel_guard_dir),
        checkpoint_path=Path(first_checkpoint),
        output_dir=output_dir / "s1",
        require_mps=bool(require_mps),
        max_regions=int(max_regions),
    )
    _append_progress(progress_path, "stage1", "end", stage_status=stage1.get("status"), verdict=(stage1.get("after_score") or {}).get("verdict"))
    guard_s2 = _write_intermediate_guard(first_replay_manifest=stage1, output_dir=output_dir)

    _append_progress(progress_path, "stage2", "start", checkpoint=str(residual_checkpoint))
    stage2 = run_refiner_replay(
        mel_guard_dir=guard_s2,
        checkpoint_path=Path(residual_checkpoint),
        output_dir=output_dir / "s2",
        require_mps=bool(require_mps),
        max_regions=int(max_regions),
    )
    final_verdict = str((stage2.get("after_score") or {}).get("verdict") or "")
    status, failed_checks = _status_from_checks(final_verdict=final_verdict, training_overlap=overlap)
    _append_progress(progress_path, "stage2", "end", stage_status=stage2.get("status"), verdict=final_verdict)

    manifest = {
        **_base_manifest(mel_guard_dir=Path(mel_guard_dir), output_dir=output_dir),
        "status": status,
        "failed_checks": failed_checks,
        "first_checkpoint": str(first_checkpoint),
        "residual_checkpoint": str(residual_checkpoint),
        "residual_training_manifest": str(residual_training_manifest_path) if residual_training_manifest_path else None,
        "training_overlap": bool(overlap),
        "training_overlap_evidence": overlap_evidence,
        "stage1_manifest": str(output_dir / "s1" / "manifest.json"),
        "stage2_manifest": str(output_dir / "s2" / "manifest.json"),
        "stage1_status": stage1.get("status"),
        "stage2_status": stage2.get("status"),
        "stage1_after_verdict": (stage1.get("after_score") or {}).get("verdict"),
        "stage2_after_verdict": final_verdict,
        "stage1_metrics": dict(stage1.get("metrics") or {}),
        "stage2_metrics": dict(stage2.get("metrics") or {}),
        "audio": dict(stage2.get("audio") or {}),
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "chain_validation_diagnostic_only",
    }
    _write_json(output_dir / "manifest.json", manifest)
    _append_progress(progress_path, "manifest", "end", run_status=status, failed_checks=failed_checks)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mel-guard-dir", type=Path, required=True)
    parser.add_argument("--first-checkpoint", type=Path, required=True)
    parser.add_argument("--residual-checkpoint", type=Path, required=True)
    parser.add_argument("--residual-training-manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-regions", type=int, default=8)
    parser.add_argument("--require-mps", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_chain_validation(
        mel_guard_dir=args.mel_guard_dir,
        first_checkpoint=args.first_checkpoint,
        residual_checkpoint=args.residual_checkpoint,
        residual_training_manifest_path=args.residual_training_manifest,
        output_dir=args.out,
        require_mps=bool(args.require_mps),
        max_regions=int(args.max_regions),
    )
    print(json.dumps({"manifest": str(args.out / "manifest.json"), "status": manifest["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
