from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402

import numpy as np
import soundfile as sf


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _safe_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def _peak(audio: np.ndarray) -> float:
    return float(np.max(np.abs(audio))) if audio.size else 0.0


def _dbfs(amplitude: float) -> float:
    return float(20.0 * math.log10(max(float(amplitude), 1e-12)))


def _amplitude_from_dbfs(dbfs: float) -> float:
    return float(10.0 ** (float(dbfs) / 20.0))


def _audio_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, float | int]:
    peak = _peak(audio)
    rms = _safe_rms(audio)
    return {
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]) if audio.ndim == 2 else 1,
        "frames": int(audio.shape[0]) if audio.ndim >= 1 else 0,
        "duration_sec": round(float(audio.shape[0] / sample_rate), 6) if sample_rate > 0 else 0.0,
        "rms": round(rms, 9),
        "rms_dbfs": round(_dbfs(rms), 3),
        "peak": round(peak, 9),
        "peak_dbfs": round(_dbfs(peak), 3),
    }


def _validate_audio(
    audio: np.ndarray,
    *,
    sample_rate: int,
    min_duration_sec: float,
    min_rms_dbfs: float,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if sample_rate <= 0:
        findings.append({"severity": "blocker", "label": "invalid_sample_rate", "evidence": sample_rate})
    if audio.ndim != 2 or audio.shape[1] < 1:
        findings.append({"severity": "blocker", "label": "invalid_channel_shape", "evidence": list(audio.shape)})
    if not np.all(np.isfinite(audio)):
        findings.append({"severity": "blocker", "label": "non_finite_audio_samples"})

    duration_sec = float(audio.shape[0] / sample_rate) if sample_rate > 0 else 0.0
    if duration_sec < float(min_duration_sec):
        findings.append(
            {
                "severity": "blocker",
                "label": "audio_too_short",
                "evidence": {"duration_sec": round(duration_sec, 6), "min_duration_sec": float(min_duration_sec)},
            }
        )

    rms_dbfs = _dbfs(_safe_rms(audio))
    if rms_dbfs < float(min_rms_dbfs):
        findings.append(
            {
                "severity": "blocker",
                "label": "audio_too_quiet",
                "evidence": {"rms_dbfs": round(rms_dbfs, 3), "min_rms_dbfs": float(min_rms_dbfs)},
            }
        )
    return findings


def apply_headroom(audio: np.ndarray, *, target_peak_dbfs: float) -> tuple[np.ndarray, float, bool]:
    target_peak = _amplitude_from_dbfs(target_peak_dbfs)
    current_peak = _peak(audio)
    if current_peak <= 0.0 or current_peak <= target_peak:
        return audio.astype(np.float32, copy=False), 1.0, False
    gain = target_peak / current_peak
    return (audio * gain).astype(np.float32), float(gain), True


def write_quality_controlled_stem(
    input_path: Path,
    output_dir: Path,
    *,
    stem_name: str,
    target_peak_dbfs: float,
    min_duration_sec: float,
    min_rms_dbfs: float,
) -> dict[str, Any]:
    audio, sample_rate = _read_audio(input_path)
    findings = _validate_audio(
        audio,
        sample_rate=sample_rate,
        min_duration_sec=min_duration_sec,
        min_rms_dbfs=min_rms_dbfs,
    )
    blocker_findings = [finding for finding in findings if finding.get("severity") == "blocker"]
    if blocker_findings:
        raise ValueError(f"Stem quality control failed for {input_path}: {blocker_findings}")

    controlled_audio, headroom_gain, headroom_applied = apply_headroom(audio, target_peak_dbfs=target_peak_dbfs)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem_name}_qc.flac"
    sf.write(str(output_path), controlled_audio, sample_rate, subtype="PCM_24")

    manifest = {
        "status": "completed",
        "pipeline": {
            "mode": "stem_quality_control_v1",
            "strategy": "fail_fast_validation_plus_downward_peak_headroom",
        },
        "stem_name": stem_name,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "target_peak_dbfs": float(target_peak_dbfs),
        "headroom_gain": round(float(headroom_gain), 9),
        "headroom_applied": bool(headroom_applied),
        "findings": findings,
        "metrics": {
            "before": _audio_metrics(audio, sample_rate),
            "after": _audio_metrics(controlled_audio, sample_rate),
        },
    }
    manifest_path = output_dir / "stem_quality_control_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _copy_audio(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)


def write_listen_pack(
    *,
    manifest: dict[str, Any],
    listen_dir: Path,
    raw_ref_path: Path | None,
    source_label: str,
    output_label: str,
) -> dict[str, str]:
    listen_dir.mkdir(parents=True, exist_ok=True)
    candidate_map: dict[str, str] = {}
    if raw_ref_path is not None:
        raw_target = listen_dir / "00_raw_ref.flac"
        _copy_audio(raw_ref_path, raw_target)
        candidate_map[raw_target.name] = str(raw_target)

    input_path = Path(str(manifest["input_path"]))
    output_path = Path(str(manifest["output_path"]))
    source_target = listen_dir / source_label
    output_target = listen_dir / output_label
    _copy_audio(input_path, source_target)
    _copy_audio(output_path, output_target)
    candidate_map[source_target.name] = str(source_target)
    candidate_map[output_target.name] = str(output_target)

    (listen_dir / "candidate_map.json").write_text(
        json.dumps(candidate_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (listen_dir / "00_README.txt").write_text(
        "\n".join(
            [
                "Stem quality-control listen pack.",
                "00_raw_ref.flac: optional raw/source reference when provided.",
                f"{source_label}: original stem before quality control.",
                f"{output_label}: quality-controlled stem after fail-fast validation and headroom.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return candidate_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply production quality-control gates to an exported stem.")
    parser.add_argument("input_stem_path")
    parser.add_argument("output_dir")
    parser.add_argument("--stem-name", required=True)
    parser.add_argument("--target-peak-dbfs", type=float, default=-1.0)
    parser.add_argument("--min-duration-sec", type=float, default=1.0)
    parser.add_argument("--min-rms-dbfs", type=float, default=-80.0)
    parser.add_argument("--listen-dir")
    parser.add_argument("--raw-ref-path")
    parser.add_argument("--source-label", default="01_original_stem.flac")
    parser.add_argument("--output-label", default="02_quality_controlled_stem.flac")
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="stem_quality_control.py",
        entrypoint_path="/app/tools/stem_quality_control.py",
    )
    args = parse_args()
    input_path = Path(args.input_stem_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input stem is missing: {input_path}")

    manifest = write_quality_controlled_stem(
        input_path,
        output_dir,
        stem_name=str(args.stem_name),
        target_peak_dbfs=float(args.target_peak_dbfs),
        min_duration_sec=float(args.min_duration_sec),
        min_rms_dbfs=float(args.min_rms_dbfs),
    )

    if args.listen_dir:
        raw_ref_path = Path(args.raw_ref_path).resolve() if args.raw_ref_path else None
        if raw_ref_path is not None and not raw_ref_path.exists():
            raise FileNotFoundError(f"Raw reference is missing: {raw_ref_path}")
        listen_map = write_listen_pack(
            manifest=manifest,
            listen_dir=Path(args.listen_dir).resolve(),
            raw_ref_path=raw_ref_path,
            source_label=str(args.source_label),
            output_label=str(args.output_label),
        )
        manifest["listen_pack"] = {"listen_dir": str(Path(args.listen_dir).resolve()), "candidate_map": listen_map}
        Path(str(manifest["manifest_path"])).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
