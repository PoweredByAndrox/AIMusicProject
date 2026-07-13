#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.artifact_status_contract import sha256_file  # noqa: E402
from tools.docker_runtime import require_docker_runtime  # noqa: E402


SCHEMA = "minus_reference_semantic_audit.v1"
SEMANTIC_JSON_NAME = "sem.json"
PROGRESS_JSONL_NAME = "progress.jsonl"
EXPECTED_ACCELERATION_MODE = "external_only_cpu_dsp_scoring"
GPU_SKIP_REASON = "minus_reference_semantic_audit_reads_existing_audio_only"
DB_FLOOR = -240.0
EPSILON = 1.0e-12
START_TIME = time.monotonic()


@dataclass(frozen=True)
class DetectorSpec:
    detector_id: str
    source_vocal_path: Path
    reference_vocal_path: Path


@dataclass(frozen=True)
class AudioBuffer:
    path: Path
    mono: np.ndarray
    sample_rate: int

    @property
    def frames(self) -> int:
        return int(self.mono.size)

    @property
    def rms(self) -> float:
        return rms(self.mono)


def read_audio(path: Path) -> AudioBuffer:
    data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = np.mean(data, axis=1).astype(np.float32) if data.size else np.zeros(0, dtype=np.float32)
    return AudioBuffer(path=path, mono=mono, sample_rate=int(sample_rate or 0))


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    values = audio.astype(np.float64, copy=False)
    return math.sqrt(float(np.mean(values * values)))


def ratio_db(numerator_rms: float, denominator_rms: float) -> float:
    if numerator_rms <= EPSILON or denominator_rms <= EPSILON:
        return DB_FLOOR
    return round(max(DB_FLOOR, 20.0 * math.log10(numerator_rms / denominator_rms)), 6)


def resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if audio.size == 0 or src_rate <= 0 or dst_rate <= 0:
        return np.zeros(0, dtype=np.float32)
    if src_rate == dst_rate:
        return audio.astype(np.float32, copy=False)
    target_frames = max(1, int(round(audio.size * float(dst_rate) / float(src_rate))))
    source_positions = np.linspace(0.0, audio.size - 1, num=audio.size, endpoint=True, dtype=np.float64)
    target_positions = np.linspace(0.0, audio.size - 1, num=target_frames, endpoint=True, dtype=np.float64)
    return np.interp(target_positions, source_positions, audio.astype(np.float64, copy=False)).astype(np.float32)


def resample_and_align(audio: np.ndarray, src_rate: int, *, target_rate: int, target_frames: int) -> np.ndarray:
    if target_frames <= 0:
        return np.zeros(0, dtype=np.float32)
    resampled = resample_linear(audio, src_rate, target_rate)
    if resampled.size == target_frames:
        return resampled
    if resampled.size > target_frames:
        return resampled[:target_frames].astype(np.float32, copy=False)
    padded = np.zeros(target_frames, dtype=np.float32)
    padded[: resampled.size] = resampled
    return padded


def stable_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "")
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def median_or_floor(values: Sequence[float]) -> float:
    if not values:
        return DB_FLOOR
    return round(float(np.median(np.asarray(values, dtype=np.float64))), 6)


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 6),
        "expected_acceleration_mode": EXPECTED_ACCELERATION_MODE,
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": GPU_SKIP_REASON,
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "minus_reference_semantic_audit_no_training_route",
        "delivery_allowed": False,
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def json_ref(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.is_file() else "",
    }


def support_decision(*, gap_db: float, reference_db: float, min_gap_db: float, max_reference_vocal_share_db: float) -> str:
    if gap_db >= min_gap_db and reference_db <= max_reference_vocal_share_db:
        return "accepted"
    if gap_db <= -min_gap_db:
        return "roles_reversed"
    return "unresolved"


def detector_row(
    detector: DetectorSpec,
    *,
    source_mix: AudioBuffer,
    minus_reference: AudioBuffer,
    min_gap_db: float,
    max_reference_vocal_share_db: float,
) -> dict[str, Any]:
    source_vocal = read_audio(detector.source_vocal_path)
    reference_vocal = read_audio(detector.reference_vocal_path)
    source_aligned = resample_and_align(
        source_vocal.mono,
        source_vocal.sample_rate,
        target_rate=source_mix.sample_rate,
        target_frames=source_mix.frames,
    )
    reference_aligned = resample_and_align(
        reference_vocal.mono,
        reference_vocal.sample_rate,
        target_rate=minus_reference.sample_rate,
        target_frames=minus_reference.frames,
    )
    source_db = ratio_db(rms(source_aligned), source_mix.rms)
    reference_db = ratio_db(rms(reference_aligned), minus_reference.rms)
    gap_db = round(source_db - reference_db, 6)
    decision = support_decision(
        gap_db=gap_db,
        reference_db=reference_db,
        min_gap_db=min_gap_db,
        max_reference_vocal_share_db=max_reference_vocal_share_db,
    )
    return {
        "detector_id": detector.detector_id,
        "support_decision": decision,
        "source_reference_gap_db": gap_db,
        "source_vocal": {
            "path": str(detector.source_vocal_path),
            "sha256": sha256_file(detector.source_vocal_path),
            "sample_rate": source_vocal.sample_rate,
            "frames_after_resample": source_mix.frames,
            "vocal_to_mix_rms_db": source_db,
        },
        "reference_vocal": {
            "path": str(detector.reference_vocal_path),
            "sha256": sha256_file(detector.reference_vocal_path),
            "sample_rate": reference_vocal.sample_rate,
            "frames_after_resample": minus_reference.frames,
            "vocal_to_mix_rms_db": reference_db,
        },
    }


def blocked_result(
    *,
    source_mix_path: Path,
    minus_reference_path: Path,
    detectors: list[dict[str, Any]],
    reasons: Sequence[str],
    min_gap_db: float,
    max_reference_vocal_share_db: float,
) -> dict[str, Any]:
    unique_ids = stable_unique(row.get("detector_id") for row in detectors)
    reason_list = stable_unique(str(reason) for reason in reasons)
    return {
        "schema": SCHEMA,
        "decision": "blocked",
        "reasons": reason_list,
        "training_allowed": False,
        "delivery_allowed": False,
        "source_mix": json_ref(source_mix_path),
        "minus_reference": json_ref(minus_reference_path),
        "detector_count": len(unique_ids),
        "detectors": detectors,
        "evidence": {
            "source_median_vocal_to_mix_rms_db": DB_FLOOR,
            "reference_median_vocal_to_mix_rms_db": DB_FLOOR,
            "median_source_reference_gap_db": 0.0,
            "accepted_support_detector_ids": [],
            "roles_reversed_support_detector_ids": [],
            "unresolved_support_detector_ids": unique_ids,
            "unique_detector_ids": unique_ids,
            "duplicate_detector_ids": sorted(
                detector_id
                for detector_id, count in Counter(row.get("detector_id") for row in detectors).items()
                if detector_id and count > 1
            ),
        },
        "thresholds": {
            "min_gap_db": float(min_gap_db),
            "max_reference_vocal_share_db": float(max_reference_vocal_share_db),
        },
        "expected_acceleration_mode": EXPECTED_ACCELERATION_MODE,
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": GPU_SKIP_REASON,
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "minus_reference_semantic_audit_no_training_route",
        "asr_dependency": False,
        "audio_copied": False,
        "audio_exported": False,
        "release_ready": False,
        "can_publish_as_result": False,
    }


def audit_minus_reference_pair_semantics(
    *,
    source_mix_path: Path,
    minus_reference_path: Path,
    detector_specs: Sequence[DetectorSpec],
    output_dir: Path,
    min_gap_db: float = 6.0,
    max_reference_vocal_share_db: float = -20.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_JSONL_NAME
    if progress_path.exists():
        progress_path.unlink()

    progress(
        progress_path,
        "start",
        schema=SCHEMA,
        source_mix=json_ref(source_mix_path),
        minus_reference=json_ref(minus_reference_path),
        detector_count_requested=len(detector_specs),
    )

    detector_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    if not source_mix_path.is_file():
        blockers.append("source_mix_missing")
    if not minus_reference_path.is_file():
        blockers.append("minus_reference_missing")
    for detector in detector_specs:
        if not detector.source_vocal_path.is_file():
            blockers.append(f"source_vocal_missing:{detector.detector_id}")
        if not detector.reference_vocal_path.is_file():
            blockers.append(f"reference_vocal_missing:{detector.detector_id}")
    if blockers:
        result = blocked_result(
            source_mix_path=source_mix_path,
            minus_reference_path=minus_reference_path,
            detectors=detector_rows,
            reasons=blockers,
            min_gap_db=min_gap_db,
            max_reference_vocal_share_db=max_reference_vocal_share_db,
        )
        write_json(output_dir / SEMANTIC_JSON_NAME, result)
        progress(
            progress_path,
            "finish",
            decision=result["decision"],
            reasons=result["reasons"],
            detector_count=result["detector_count"],
            training_allowed=result["training_allowed"],
        )
        return result

    try:
        source_mix = read_audio(source_mix_path)
        minus_reference = read_audio(minus_reference_path)
    except Exception as exc:  # pragma: no cover - defensive fail-closed path
        result = blocked_result(
            source_mix_path=source_mix_path,
            minus_reference_path=minus_reference_path,
            detectors=detector_rows,
            reasons=[f"audio_read_failed:{type(exc).__name__}"],
            min_gap_db=min_gap_db,
            max_reference_vocal_share_db=max_reference_vocal_share_db,
        )
        write_json(output_dir / SEMANTIC_JSON_NAME, result)
        progress(
            progress_path,
            "finish",
            decision=result["decision"],
            reasons=result["reasons"],
            detector_count=result["detector_count"],
            training_allowed=result["training_allowed"],
        )
        return result

    if source_mix.rms <= EPSILON:
        blockers.append("source_mix_silent")
    if minus_reference.rms <= EPSILON:
        blockers.append("minus_reference_silent")

    for detector in detector_specs:
        try:
            row = detector_row(
                detector,
                source_mix=source_mix,
                minus_reference=minus_reference,
                min_gap_db=min_gap_db,
                max_reference_vocal_share_db=max_reference_vocal_share_db,
            )
            detector_rows.append(row)
            progress(
                progress_path,
                "detector_scored",
                detector_id=row["detector_id"],
                support_decision=row["support_decision"],
                source_vocal_to_mix_rms_db=row["source_vocal"]["vocal_to_mix_rms_db"],
                reference_vocal_to_mix_rms_db=row["reference_vocal"]["vocal_to_mix_rms_db"],
                source_reference_gap_db=row["source_reference_gap_db"],
            )
        except Exception as exc:
            blockers.append(f"detector_scoring_failed:{detector.detector_id}:{type(exc).__name__}")

    unique_detector_ids = stable_unique(row["detector_id"] for row in detector_rows)
    detector_count = len(unique_detector_ids)
    if detector_count < 2:
        blockers.append("insufficient_unique_detectors")

    if blockers:
        result = blocked_result(
            source_mix_path=source_mix_path,
            minus_reference_path=minus_reference_path,
            detectors=detector_rows,
            reasons=blockers,
            min_gap_db=min_gap_db,
            max_reference_vocal_share_db=max_reference_vocal_share_db,
        )
        write_json(output_dir / SEMANTIC_JSON_NAME, result)
        progress(
            progress_path,
            "finish",
            decision=result["decision"],
            reasons=result["reasons"],
            detector_count=result["detector_count"],
            training_allowed=result["training_allowed"],
        )
        return result

    source_values = [float(row["source_vocal"]["vocal_to_mix_rms_db"]) for row in detector_rows]
    reference_values = [float(row["reference_vocal"]["vocal_to_mix_rms_db"]) for row in detector_rows]
    gap_values = [float(row["source_reference_gap_db"]) for row in detector_rows]
    support_sets: dict[str, list[str]] = {"accepted": [], "roles_reversed": [], "unresolved": []}
    for row in detector_rows:
        support_sets.setdefault(str(row["support_decision"]), []).append(str(row["detector_id"]))
    accepted_support_ids = stable_unique(support_sets["accepted"])
    roles_reversed_support_ids = stable_unique(support_sets["roles_reversed"])
    unresolved_support_ids = stable_unique(support_sets["unresolved"])

    source_median = median_or_floor(source_values)
    reference_median = median_or_floor(reference_values)
    gap_median = median_or_floor(gap_values)
    duplicate_detector_ids = sorted(
        detector_id
        for detector_id, count in Counter(row["detector_id"] for row in detector_rows).items()
        if detector_id and count > 1
    )

    reasons: list[str] = []
    if (
        source_median - reference_median >= min_gap_db
        and reference_median <= max_reference_vocal_share_db
        and len(accepted_support_ids) >= 2
    ):
        decision = "accepted"
    elif reference_median - source_median >= min_gap_db and len(roles_reversed_support_ids) >= 2:
        decision = "roles_reversed"
        reasons.append("reference_source_gap_reversed")
    else:
        decision = "unresolved"
        if gap_median > -min_gap_db and gap_median < min_gap_db:
            reasons.append("median_gap_below_threshold")
        if reference_median > max_reference_vocal_share_db:
            reasons.append("reference_vocal_share_above_max")
        if len(accepted_support_ids) < 2 and len(roles_reversed_support_ids) < 2:
            reasons.append("insufficient_matching_detector_quorum")

    result = {
        "schema": SCHEMA,
        "decision": decision,
        "reasons": stable_unique(reasons),
        "training_allowed": decision == "accepted",
        "delivery_allowed": False,
        "source_mix": json_ref(source_mix_path),
        "minus_reference": json_ref(minus_reference_path),
        "detector_count": detector_count,
        "detectors": detector_rows,
        "evidence": {
            "source_median_vocal_to_mix_rms_db": source_median,
            "reference_median_vocal_to_mix_rms_db": reference_median,
            "median_source_reference_gap_db": gap_median,
            "accepted_support_detector_ids": accepted_support_ids,
            "roles_reversed_support_detector_ids": roles_reversed_support_ids,
            "unresolved_support_detector_ids": unresolved_support_ids,
            "unique_detector_ids": unique_detector_ids,
            "duplicate_detector_ids": duplicate_detector_ids,
        },
        "thresholds": {
            "min_gap_db": float(min_gap_db),
            "max_reference_vocal_share_db": float(max_reference_vocal_share_db),
        },
        "expected_acceleration_mode": EXPECTED_ACCELERATION_MODE,
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": GPU_SKIP_REASON,
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "minus_reference_semantic_audit_no_training_route",
        "asr_dependency": False,
        "audio_copied": False,
        "audio_exported": False,
        "release_ready": False,
        "can_publish_as_result": False,
    }
    write_json(output_dir / SEMANTIC_JSON_NAME, result)
    progress(
        progress_path,
        "finish",
        decision=decision,
        reasons=result["reasons"],
        detector_count=detector_count,
        source_median_vocal_to_mix_rms_db=source_median,
        reference_median_vocal_to_mix_rms_db=reference_median,
        median_source_reference_gap_db=gap_median,
        training_allowed=result["training_allowed"],
    )
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed semantic audit for source-mix / minus-reference role correctness."
    )
    parser.add_argument("source_mix_path", type=Path)
    parser.add_argument("minus_reference_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--detector",
        action="append",
        nargs=3,
        metavar=("DETECTOR_ID", "SOURCE_VOCAL_PATH", "REFERENCE_VOCAL_PATH"),
        default=[],
        required=True,
        help="Repeat for each detector: unique detector id, source vocal path, reference vocal path.",
    )
    parser.add_argument("--min-gap-db", type=float, default=6.0)
    parser.add_argument("--max-reference-vocal-share-db", type=float, default=-20.0)
    return parser.parse_args(argv)


def parse_detectors(raw_detectors: Sequence[Sequence[str]]) -> list[DetectorSpec]:
    return [
        DetectorSpec(
            detector_id=str(items[0]),
            source_vocal_path=Path(str(items[1])),
            reference_vocal_path=Path(str(items[2])),
        )
        for items in raw_detectors
    ]


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="audit_minus_reference_pair_semantics.py",
        entrypoint_path="/app/tools/audit_minus_reference_pair_semantics.py",
    )
    args = parse_args(argv)
    result = audit_minus_reference_pair_semantics(
        source_mix_path=args.source_mix_path,
        minus_reference_path=args.minus_reference_path,
        detector_specs=parse_detectors(args.detector),
        output_dir=args.output_dir,
        min_gap_db=float(args.min_gap_db),
        max_reference_vocal_share_db=float(args.max_reference_vocal_share_db),
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "detector_count": result["detector_count"],
                "training_allowed": result["training_allowed"],
                "delivery_allowed": result["delivery_allowed"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["decision"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
