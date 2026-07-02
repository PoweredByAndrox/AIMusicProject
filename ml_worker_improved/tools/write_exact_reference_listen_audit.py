from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402
from waveform_forensic_compare import (
    DEFAULT_DEFECT_THRESHOLD_DBFS,
    DEFAULT_ENVELOPE_WINDOW_SEC,
    DEFAULT_MAX_DEFECT_RANGES,
    DEFAULT_MIN_DEFECT_DURATION_SEC,
    _load_named_reference_stems,
    write_waveform_forensic_evidence_pack,
)


EPSILON = 1.0e-12
DEFAULT_MIN_SDR_DB = 20.0
DEFAULT_MIN_COSINE_PERCENT = 95.0
DEFAULT_MAX_RESIDUAL_PEAK_DBFS = -45.0
DEFAULT_STREAM_BLOCK_FRAMES = 262_144


def _load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=False, dtype="float32")
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _normalize_audio_shape(reference_audio: np.ndarray, candidate_audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference_audio)
    candidate = np.asarray(candidate_audio)
    if reference.ndim > 2 or candidate.ndim > 2:
        raise ValueError("expected mono or stereo audio arrays")
    if reference.ndim != candidate.ndim:
        raise ValueError(f"channel-rank mismatch: reference={reference.ndim}, candidate={candidate.ndim}")
    if reference.ndim == 2 and reference.shape[1] != candidate.shape[1]:
        raise ValueError(f"channel-count mismatch: reference={reference.shape[1]}, candidate={candidate.shape[1]}")
    frame_count = min(reference.shape[0], candidate.shape[0])
    if frame_count <= 0:
        raise ValueError("reference and candidate must contain at least one frame")
    return reference[:frame_count], candidate[:frame_count]


def _read_streaming_metrics_and_evidence_head(
    *,
    reference_path: str | Path,
    candidate_path: str | Path,
    max_evidence_frames: int | None,
    block_frames: int = DEFAULT_STREAM_BLOCK_FRAMES,
) -> dict[str, Any]:
    if block_frames <= 0:
        raise ValueError(f"block_frames must be positive, got {block_frames}")
    if max_evidence_frames is not None and int(max_evidence_frames) <= 0:
        raise ValueError(f"max_evidence_frames must be positive when provided, got {max_evidence_frames}")

    with sf.SoundFile(str(reference_path), mode="r") as reference_file, sf.SoundFile(
        str(candidate_path),
        mode="r",
    ) as candidate_file:
        if reference_file.samplerate != candidate_file.samplerate:
            raise ValueError(
                f"sample-rate mismatch: reference={reference_file.samplerate}, "
                f"candidate={candidate_file.samplerate}"
            )
        if reference_file.channels != candidate_file.channels:
            raise ValueError(
                f"channel-count mismatch: reference={reference_file.channels}, candidate={candidate_file.channels}"
            )

        sample_rate = int(reference_file.samplerate)
        reference_frame_count = int(reference_file.frames)
        candidate_frame_count = int(candidate_file.frames)
        compared_frame_count = min(reference_frame_count, candidate_frame_count)
        if compared_frame_count <= 0:
            raise ValueError("reference and candidate must contain at least one frame")

        evidence_frame_limit = (
            min(compared_frame_count, int(max_evidence_frames))
            if max_evidence_frames is not None
            else compared_frame_count
        )
        reference_head_chunks: list[np.ndarray] = []
        candidate_head_chunks: list[np.ndarray] = []
        reference_energy = 0.0
        candidate_energy = 0.0
        residual_energy = 0.0
        dot_product = 0.0
        residual_peak = 0.0
        residual_value_count = 0
        frames_read = 0

        while frames_read < compared_frame_count:
            frames_to_read = min(int(block_frames), compared_frame_count - frames_read)
            reference_block = reference_file.read(frames_to_read, dtype="float32", always_2d=False)
            candidate_block = candidate_file.read(frames_to_read, dtype="float32", always_2d=False)
            reference_block = np.asarray(reference_block, dtype=np.float32)
            candidate_block = np.asarray(candidate_block, dtype=np.float32)
            residual_block = reference_block - candidate_block

            reference_energy += float(np.sum(np.square(reference_block), dtype=np.float64))
            candidate_energy += float(np.sum(np.square(candidate_block), dtype=np.float64))
            residual_energy += float(np.sum(np.square(residual_block), dtype=np.float64))
            dot_product += float(
                np.dot(
                    reference_block.reshape(-1).astype(np.float64, copy=False),
                    candidate_block.reshape(-1).astype(np.float64, copy=False),
                )
            )
            residual_peak = max(residual_peak, float(np.max(np.abs(residual_block))))
            residual_value_count += int(residual_block.size)

            if frames_read < evidence_frame_limit:
                keep_frames = min(evidence_frame_limit - frames_read, reference_block.shape[0])
                reference_head_chunks.append(reference_block[:keep_frames].copy())
                candidate_head_chunks.append(candidate_block[:keep_frames].copy())
            frames_read += int(reference_block.shape[0])

    reference_head = np.concatenate(reference_head_chunks, axis=0) if reference_head_chunks else np.empty(0)
    candidate_head = np.concatenate(candidate_head_chunks, axis=0) if candidate_head_chunks else np.empty(0)
    residual_rms = math.sqrt((residual_energy / max(residual_value_count, 1)) + EPSILON)
    cosine_denominator = math.sqrt(reference_energy) * math.sqrt(candidate_energy) + EPSILON
    metrics = {
        "sdr_db": float(10.0 * math.log10((reference_energy + EPSILON) / (residual_energy + EPSILON))),
        "cosine_percent": float(100.0 * dot_product / cosine_denominator),
        "residual_rms_dbfs": float(20.0 * math.log10(max(residual_rms, EPSILON))),
        "residual_peak_dbfs": _dbfs_from_peak(residual_peak),
        "compared_frame_count": int(compared_frame_count),
        "reference_frame_count": int(reference_frame_count),
        "candidate_frame_count": int(candidate_frame_count),
        "truncated_to_overlap": bool(reference_frame_count != candidate_frame_count),
    }
    return {
        "sample_rate": sample_rate,
        "reference_frame_count": int(reference_frame_count),
        "candidate_frame_count": int(candidate_frame_count),
        "compared_frame_count": int(compared_frame_count),
        "evidence_frame_count": int(reference_head.shape[0]),
        "reference_evidence_audio": reference_head,
        "candidate_evidence_audio": candidate_head,
        "metrics": metrics,
    }


def _dbfs_from_peak(value: float) -> float:
    peak = max(abs(float(value)), EPSILON)
    return float(20.0 * math.log10(peak))


def _dbfs_from_rms(audio: np.ndarray) -> float:
    rms = math.sqrt(float(np.mean(np.square(audio), dtype=np.float64)) + EPSILON)
    return float(20.0 * math.log10(max(rms, EPSILON)))


def _cosine_percent(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    numerator = float(np.dot(reference_flat, candidate_flat))
    denominator = float(np.linalg.norm(reference_flat) * np.linalg.norm(candidate_flat) + EPSILON)
    return float(100.0 * numerator / denominator)


def _sdr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    residual = reference - candidate
    reference_energy = float(np.sum(np.square(reference), dtype=np.float64))
    residual_energy = float(np.sum(np.square(residual), dtype=np.float64))
    return float(10.0 * math.log10((reference_energy + EPSILON) / (residual_energy + EPSILON)))


def _build_failed_checks(
    *,
    sdr_db: float,
    cosine_percent: float,
    residual_peak_dbfs: float,
    forensic_status: str,
    min_sdr_db: float,
    min_cosine_percent: float,
    max_residual_peak_dbfs: float,
) -> list[str]:
    failed_checks: list[str] = []
    if sdr_db < min_sdr_db:
        failed_checks.append("sdr_below_threshold")
    if cosine_percent < min_cosine_percent:
        failed_checks.append("cosine_below_threshold")
    if residual_peak_dbfs > max_residual_peak_dbfs:
        failed_checks.append("residual_peak_above_threshold")
    if forensic_status != "passed":
        failed_checks.append("waveform_forensic_status_failed")
    return failed_checks


def write_exact_reference_listen_audit(
    *,
    reference_path: str | Path,
    candidate_path: str | Path,
    output_dir: str | Path,
    stem_name: str,
    min_sdr_db: float = DEFAULT_MIN_SDR_DB,
    min_cosine_percent: float = DEFAULT_MIN_COSINE_PERCENT,
    max_residual_peak_dbfs: float = DEFAULT_MAX_RESIDUAL_PEAK_DBFS,
    defect_threshold_dbfs: float = DEFAULT_DEFECT_THRESHOLD_DBFS,
    min_defect_duration_sec: float = DEFAULT_MIN_DEFECT_DURATION_SEC,
    envelope_window_sec: float = DEFAULT_ENVELOPE_WINDOW_SEC,
    max_defect_ranges: int = DEFAULT_MAX_DEFECT_RANGES,
    max_offset_samples: int = 0,
    reference_stems: dict[str, np.ndarray] | None = None,
    max_evidence_frames: int | None = None,
) -> dict[str, Any]:
    streamed = _read_streaming_metrics_and_evidence_head(
        reference_path=reference_path,
        candidate_path=candidate_path,
        max_evidence_frames=max_evidence_frames,
    )
    reference_sample_rate = int(streamed["sample_rate"])
    reference_audio = streamed["reference_evidence_audio"]
    candidate_audio = streamed["candidate_evidence_audio"]
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    evidence_dir = target_dir / "forensic_evidence"
    evidence_report = write_waveform_forensic_evidence_pack(
        reference_audio=reference_audio,
        candidate_audio=candidate_audio,
        sample_rate=reference_sample_rate,
        output_dir=evidence_dir,
        stem_name=stem_name,
        defect_threshold_dbfs=defect_threshold_dbfs,
        min_defect_duration_sec=min_defect_duration_sec,
        envelope_window_sec=envelope_window_sec,
        max_defect_ranges=max_defect_ranges,
        max_offset_samples=max_offset_samples,
        apply_gain_correction=False,
        reference_stems=reference_stems,
        max_evidence_frames=max_evidence_frames,
    )

    metrics = dict(streamed["metrics"])
    evidence_scope = dict(evidence_report.get("artifact_scope", {}))
    if metrics["compared_frame_count"] > streamed["evidence_frame_count"]:
        evidence_scope = {
            "mode": "head_truncated",
            "aligned_frame_count": int(metrics["compared_frame_count"]),
            "artifact_frame_count": int(streamed["evidence_frame_count"]),
            "max_evidence_frames": int(max_evidence_frames) if max_evidence_frames is not None else None,
            "waveform_forensic_scope": "evidence_head_only",
            "exact_metrics_scope": "full_overlap_streaming",
        }
        evidence_report["artifact_scope"] = evidence_scope
        report_rel_path = evidence_report.get("evidence_artifacts", {}).get("report_json")
        if report_rel_path:
            evidence_report_path = evidence_dir / str(report_rel_path)
            if evidence_report_path.exists():
                persisted_evidence = json.loads(evidence_report_path.read_text(encoding="utf-8"))
                persisted_evidence["artifact_scope"] = evidence_scope
                evidence_report_path.write_text(
                    json.dumps(persisted_evidence, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
    failed_checks = _build_failed_checks(
        sdr_db=metrics["sdr_db"],
        cosine_percent=metrics["cosine_percent"],
        residual_peak_dbfs=metrics["residual_peak_dbfs"],
        forensic_status=str(evidence_report["waveform_forensics"]["status"]),
        min_sdr_db=min_sdr_db,
        min_cosine_percent=min_cosine_percent,
        max_residual_peak_dbfs=max_residual_peak_dbfs,
    )
    if evidence_scope.get("waveform_forensic_scope") == "evidence_head_only":
        failed_checks.append("waveform_forensic_scope_truncated")
    delivery_status = "accepted" if not failed_checks else "rejected"
    audit_report = {
        "schema_version": 1,
        "comparison_mode": "exact_reference_listen_audit",
        "stem_name": stem_name,
        "reference": {
            "path": str(reference_path),
            "sample_rate": reference_sample_rate,
            "frame_count": int(metrics["reference_frame_count"]),
        },
        "candidate": {
            "path": str(candidate_path),
            "sample_rate": reference_sample_rate,
            "frame_count": int(metrics["candidate_frame_count"]),
        },
        "thresholds": {
            "min_sdr_db": float(min_sdr_db),
            "min_cosine_percent": float(min_cosine_percent),
            "max_residual_peak_dbfs": float(max_residual_peak_dbfs),
        },
        "metrics": metrics,
        "forensic_evidence": {
            "status": "available",
            "output_dir": "forensic_evidence",
            "evidence_artifacts": evidence_report["evidence_artifacts"],
            "artifact_scope": evidence_scope,
            "waveform_status": evidence_report["waveform_forensics"]["status"],
            "defect_ranges": evidence_report["waveform_forensics"]["defect_ranges"],
        },
        "delivery": {
            "status": delivery_status,
            "reason": (
                "exact_reference_threshold_passed"
                if delivery_status == "accepted"
                else "exact_reference_threshold_failed"
            ),
            "failed_checks": failed_checks,
        },
    }
    (target_dir / "exact_reference_listen_audit.json").write_text(
        json.dumps(audit_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return audit_report


def main() -> int:
    require_docker_runtime(
        tool_name="write_exact_reference_listen_audit.py",
        entrypoint_path="/app/tools/write_exact_reference_listen_audit.py",
    )
    parser = argparse.ArgumentParser(description="Audit one listen candidate against an exact reference stem.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stem-name", required=True)
    parser.add_argument("--min-sdr-db", type=float, default=DEFAULT_MIN_SDR_DB)
    parser.add_argument("--min-cosine-percent", type=float, default=DEFAULT_MIN_COSINE_PERCENT)
    parser.add_argument("--max-residual-peak-dbfs", type=float, default=DEFAULT_MAX_RESIDUAL_PEAK_DBFS)
    parser.add_argument("--defect-threshold-dbfs", type=float, default=DEFAULT_DEFECT_THRESHOLD_DBFS)
    parser.add_argument("--min-defect-duration-sec", type=float, default=DEFAULT_MIN_DEFECT_DURATION_SEC)
    parser.add_argument("--envelope-window-sec", type=float, default=DEFAULT_ENVELOPE_WINDOW_SEC)
    parser.add_argument("--max-defect-ranges", type=int, default=DEFAULT_MAX_DEFECT_RANGES)
    parser.add_argument("--max-offset-samples", type=int, default=0)
    parser.add_argument("--max-evidence-frames", type=int)
    parser.add_argument("--reference-stem", action="append", default=[], help="Named source stem in NAME=PATH form.")
    args = parser.parse_args()

    sample_rate = int(sf.info(args.reference).samplerate)
    reference_stems = _load_named_reference_stems(args.reference_stem, expected_sample_rate=sample_rate)
    report = write_exact_reference_listen_audit(
        reference_path=args.reference,
        candidate_path=args.candidate,
        output_dir=args.output_dir,
        stem_name=args.stem_name,
        min_sdr_db=args.min_sdr_db,
        min_cosine_percent=args.min_cosine_percent,
        max_residual_peak_dbfs=args.max_residual_peak_dbfs,
        defect_threshold_dbfs=args.defect_threshold_dbfs,
        min_defect_duration_sec=args.min_defect_duration_sec,
        envelope_window_sec=args.envelope_window_sec,
        max_defect_ranges=args.max_defect_ranges,
        max_offset_samples=args.max_offset_samples,
        reference_stems=reference_stems,
        max_evidence_frames=args.max_evidence_frames,
    )
    print(
        json.dumps(
            {
                "status": report["delivery"]["status"],
                "reason": report["delivery"]["reason"],
                "output": str(Path(args.output_dir) / "exact_reference_listen_audit.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
