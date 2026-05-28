from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
MANIFEST_NAME = "instrumental_preference_dataset_manifest.json"
SEGMENTS_NAME = "segments.jsonl"
FEEDBACK_NAME = "feedback.jsonl"
PROGRESS_NAME = "instrumental_preference_progress.jsonl"

POSITIVE_RATINGS = {"accept", "good", "better"}
NEGATIVE_RATINGS = {"reject", "bad", "worse"}
RATINGS = sorted(POSITIVE_RATINGS | NEGATIVE_RATINGS)

RESIDUAL_VOCAL_LABELS = {
    "residual_vocal_bleed",
    "vocal_bleed",
    "leaky_full_mix_risk",
    "kick_punch_bleed",
    "drop_melody_leakage",
    "low_band_bleed",
    "wide_stereo_non_vocal_leakage",
    "unsupported_anchor_leakage",
}
INSTRUMENTAL_PRESERVATION_LABELS = {
    "instrumental_muffling",
    "thin_or_overcleaned",
    "presence_loss",
    "air_detail_loss",
    "local_vocal_body_dip",
    "vocal_dropout",
    "vocal_tail_loss",
}
NOISE_ARTIFACT_LABELS = {
    "high_noise_or_robotic_artifact",
    "robotic_artifact",
    "watery_artifacts",
}


def _now_event(stage: str, *, status: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "type": "instrumental_preference_dataset_stage",
        "stage": stage,
        "status": status,
        "elapsed_sec": 0.0,
        "since_start_sec": 0.0,
        "metadata": metadata or {},
    }


class ProgressWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started_at = time.monotonic()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def mark(self, stage: str, *, status: str = "progress", **metadata: Any) -> None:
        event = _now_event(stage, status=status, metadata=metadata)
        event["since_start_sec"] = round(time.monotonic() - self.started_at, 6)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return loaded


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_path(value: Any) -> Path | None:
    if isinstance(value, str) and value:
        return Path(value)
    return None


def preferred_instrumental_path(stem_manifest: dict[str, Any]) -> Path:
    stems = stem_manifest.get("stems")
    if isinstance(stems, dict):
        instrumental = stems.get("instrumental")
        if isinstance(instrumental, dict):
            path = _as_path(instrumental.get("path"))
            if path is not None:
                return path

    artifacts = stem_manifest.get("artifacts")
    if isinstance(artifacts, dict):
        for key in ("instrumental_flac", "instrumental_path"):
            path = _as_path(artifacts.get(key))
            if path is not None:
                return path

    direct_path = _as_path(stem_manifest.get("instrumental_path"))
    if direct_path is not None:
        return direct_path

    raise ValueError("Preferred stem manifest has no instrumental path")


def preferred_profile_name(stem_manifest: dict[str, Any], override: str | None = None) -> str:
    if override:
        return override
    for key in ("model_id", "model_name", "profile_name"):
        value = stem_manifest.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown_instrumental_candidate"


def audio_duration_sec(stem_manifest: dict[str, Any]) -> float | None:
    audio = stem_manifest.get("audio")
    if isinstance(audio, dict) and isinstance(audio.get("duration_sec"), (int, float)):
        value = float(audio["duration_sec"])
        return value if value > 0.0 and math.isfinite(value) else None
    return None


def _normalized_label_list(labels: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for label in labels:
        if not isinstance(label, str):
            continue
        clean_label = label.strip()
        if not clean_label or clean_label in seen:
            continue
        seen.add(clean_label)
        normalized.append(clean_label)
    return normalized


def _training_targets(labels: Sequence[str]) -> list[str]:
    targets: list[str] = []
    label_set = set(labels)
    if label_set & RESIDUAL_VOCAL_LABELS:
        targets.append("suppress_residual_vocal")
    if label_set & INSTRUMENTAL_PRESERVATION_LABELS:
        targets.append("preserve_instrumental_energy")
    if label_set & NOISE_ARTIFACT_LABELS:
        targets.append("avoid_artifact_noise")
    if not targets:
        targets.append("balance_instrumental_cleanliness_and_preservation")
    return targets


def _timed_blocker_segments(
    gate_manifest: dict[str, Any] | None,
    human_defect_labels: Sequence[str],
) -> list[dict[str, Any]]:
    if not gate_manifest:
        return []

    gate = gate_manifest.get("defect_effectiveness_gate")
    if not isinstance(gate, dict):
        return []
    blockers = gate.get("blockers")
    if not isinstance(blockers, list):
        return []

    by_window: dict[tuple[float, float], dict[str, Any]] = {}
    for blocker in blockers:
        if not isinstance(blocker, dict):
            continue
        start_sec = blocker.get("start_sec")
        end_sec = blocker.get("end_sec")
        if not isinstance(start_sec, (int, float)) or not isinstance(end_sec, (int, float)):
            continue
        start_value = round(float(start_sec), 6)
        end_value = round(float(end_sec), 6)
        if not math.isfinite(start_value) or not math.isfinite(end_value) or end_value <= start_value:
            continue
        labels = list(blocker.get("labels") if isinstance(blocker.get("labels"), list) else [])
        labels.extend(human_defect_labels)
        key = (start_value, end_value)
        record = by_window.setdefault(
            key,
            {
                "window_sec": [start_value, end_value],
                "defect_labels": [],
                "source": "gate_blocker",
                "gate_reasons": [],
                "severity_score": None,
            },
        )
        record["defect_labels"] = _normalized_label_list([*record["defect_labels"], *labels])
        reason = blocker.get("reason")
        if isinstance(reason, str) and reason not in record["gate_reasons"]:
            record["gate_reasons"].append(reason)
        severity = blocker.get("severity_score")
        if isinstance(severity, (int, float)) and math.isfinite(float(severity)):
            current = record["severity_score"]
            record["severity_score"] = max(float(severity), float(current or 0.0))

    return [by_window[key] for key in sorted(by_window)]


def _uniform_segments(
    *,
    duration_sec: float | None,
    segment_sec: float,
    human_defect_labels: Sequence[str],
) -> list[dict[str, Any]]:
    if duration_sec is None:
        return [
            {
                "window_sec": [0.0, 0.0],
                "defect_labels": list(human_defect_labels),
                "source": "full_track_without_duration",
                "gate_reasons": [],
                "severity_score": None,
            }
        ]
    if segment_sec <= 0.0 or not math.isfinite(segment_sec):
        raise ValueError("--segment-sec must be a positive finite value")

    segments: list[dict[str, Any]] = []
    start_sec = 0.0
    while start_sec < duration_sec - 1e-6:
        end_sec = min(duration_sec, start_sec + segment_sec)
        segments.append(
            {
                "window_sec": [round(start_sec, 6), round(end_sec, 6)],
                "defect_labels": list(human_defect_labels),
                "source": "uniform_fallback",
                "gate_reasons": [],
                "severity_score": None,
            }
        )
        start_sec = end_sec
    return segments


def build_segment_records(
    *,
    gate_manifest: dict[str, Any] | None,
    human_defect_labels: Sequence[str],
    duration_sec: float | None,
    segment_sec: float,
    preferred_profile: str,
    rating: str,
    confidence: float,
) -> list[dict[str, Any]]:
    segments = _timed_blocker_segments(gate_manifest, human_defect_labels)
    if not segments:
        segments = _uniform_segments(
            duration_sec=duration_sec,
            segment_sec=segment_sec,
            human_defect_labels=human_defect_labels,
        )

    records: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        labels = _normalized_label_list(segment["defect_labels"])
        records.append(
            {
                "segment_id": f"segment_{index:04d}",
                "window_sec": segment["window_sec"],
                "preferred_profile": preferred_profile,
                "rating": rating,
                "confidence": round(confidence, 6),
                "defect_labels": labels,
                "training_targets": _training_targets(labels),
                "source": segment["source"],
                "gate_reasons": segment["gate_reasons"],
                "severity_score": segment["severity_score"],
            }
        )
    return records


def _feedback_record(
    *,
    preferred_profile: str,
    rating: str,
    confidence: float,
    feedback_split: str,
    defect_labels: Sequence[str],
    source_audio_path: Path,
    preferred_instrumental: Path,
) -> dict[str, Any]:
    label_set = set(defect_labels)
    return {
        "mode": "instrumental_preference_feedback",
        "profile": preferred_profile,
        "selected_profile": preferred_profile,
        "target_stem": "instrumental",
        "rating": rating,
        "confidence": round(confidence, 6),
        "feedback_split": feedback_split,
        "human_verified": True,
        "listener_verified": True,
        "residual_vocal_bleed": bool(label_set & RESIDUAL_VOCAL_LABELS),
        "too_dry": bool(label_set & INSTRUMENTAL_PRESERVATION_LABELS),
        "artifact_noise": bool(label_set & NOISE_ARTIFACT_LABELS),
        "source_audio_path": str(source_audio_path),
        "source_audio_sha256": sha256_file(source_audio_path),
        "candidate_audio_path": str(preferred_instrumental),
        "candidate_audio_sha256": sha256_file(preferred_instrumental),
        "defect_labels": list(defect_labels),
    }


def _materialize_audio_segments(
    *,
    source_audio_path: Path,
    preferred_instrumental: Path,
    output_dir: Path,
    segments: list[dict[str, Any]],
    progress: ProgressWriter,
) -> list[dict[str, Any]]:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - exercised only in stripped runtimes
        raise RuntimeError("soundfile is required for --materialize-audio") from exc

    source_info = sf.info(str(source_audio_path))
    preferred_info = sf.info(str(preferred_instrumental))
    if source_info.samplerate != preferred_info.samplerate:
        raise ValueError("Source and preferred instrumental sample rates differ")

    sample_rate = int(source_info.samplerate)
    materialized: list[dict[str, Any]] = []
    for segment in segments:
        start_sec, end_sec = segment["window_sec"]
        if end_sec <= start_sec:
            continue
        start_frame = int(round(float(start_sec) * sample_rate))
        frame_count = max(1, int(round((float(end_sec) - float(start_sec)) * sample_rate)))
        source_audio, _ = sf.read(str(source_audio_path), start=start_frame, frames=frame_count, always_2d=True)
        candidate_audio, _ = sf.read(
            str(preferred_instrumental),
            start=start_frame,
            frames=frame_count,
            always_2d=True,
        )
        segment_dir = output_dir / "segments" / segment["segment_id"]
        segment_dir.mkdir(parents=True, exist_ok=True)
        source_clip = segment_dir / "source_mix.flac"
        candidate_clip = segment_dir / "preferred_instrumental.flac"
        sf.write(str(source_clip), source_audio, sample_rate)
        sf.write(str(candidate_clip), candidate_audio, sample_rate)
        materialized.append(
            {
                "segment_id": segment["segment_id"],
                "source_mix_clip": str(source_clip),
                "preferred_instrumental_clip": str(candidate_clip),
            }
        )
        progress.mark("materialize_segment", segment_id=segment["segment_id"])
    return materialized


def build_instrumental_preference_dataset(
    *,
    source_audio_path: Path,
    preferred_stem_manifest_path: Path,
    output_dir: Path,
    gate_manifest_path: Path | None = None,
    preferred_profile: str | None = None,
    human_defect_labels: Sequence[str] = (),
    rating: str = "good",
    confidence: float = 0.85,
    feedback_split: str = "train",
    segment_sec: float = 5.0,
    materialize_audio: bool = False,
) -> dict[str, Any]:
    if rating not in POSITIVE_RATINGS | NEGATIVE_RATINGS:
        raise ValueError(f"Unsupported rating: {rating}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressWriter(output_dir / PROGRESS_NAME)
    progress.mark(
        "start",
        source_audio_path=str(source_audio_path),
        preferred_stem_manifest_path=str(preferred_stem_manifest_path),
        gate_manifest_path=str(gate_manifest_path) if gate_manifest_path else None,
        expected_acceleration_mode="cpu_metadata_dataset_build",
    )

    stem_manifest = load_json(preferred_stem_manifest_path)
    gate_manifest = load_json(gate_manifest_path) if gate_manifest_path is not None else None
    preferred_instrumental = preferred_instrumental_path(stem_manifest)
    profile = preferred_profile_name(stem_manifest, preferred_profile)
    labels = _normalized_label_list(human_defect_labels)
    duration_sec = audio_duration_sec(stem_manifest)

    progress.mark("build_segments", preferred_profile=profile, duration_sec=duration_sec)
    segments = build_segment_records(
        gate_manifest=gate_manifest,
        human_defect_labels=labels,
        duration_sec=duration_sec,
        segment_sec=segment_sec,
        preferred_profile=profile,
        rating=rating,
        confidence=confidence,
    )

    materialized_segments: list[dict[str, Any]] = []
    if materialize_audio:
        progress.mark("materialize_audio", segment_count=len(segments))
        materialized_segments = _materialize_audio_segments(
            source_audio_path=source_audio_path,
            preferred_instrumental=preferred_instrumental,
            output_dir=output_dir,
            segments=segments,
            progress=progress,
        )

    segment_records_by_id = {record["segment_id"]: record for record in segments}
    for clip_record in materialized_segments:
        segment_records_by_id[clip_record["segment_id"]]["materialized_audio"] = clip_record
    aggregate_defect_labels = _normalized_label_list(
        [
            label
            for segment in segments
            for label in segment.get("defect_labels", [])
        ]
    )

    feedback_record = _feedback_record(
        preferred_profile=profile,
        rating=rating,
        confidence=confidence,
        feedback_split=feedback_split,
        defect_labels=aggregate_defect_labels,
        source_audio_path=source_audio_path,
        preferred_instrumental=preferred_instrumental,
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "mode": "instrumental_preference_dataset",
        "source_audio_path": str(source_audio_path),
        "source_audio_sha256": sha256_file(source_audio_path),
        "preferred_stem_manifest_path": str(preferred_stem_manifest_path),
        "gate_manifest_path": str(gate_manifest_path) if gate_manifest_path else None,
        "preferred_candidate": {
            "profile": profile,
            "instrumental_path": str(preferred_instrumental),
            "instrumental_sha256": sha256_file(preferred_instrumental),
            "stem_manifest_contract": stem_manifest.get("contract", {}),
        },
        "preference": {
            "rating": rating,
            "confidence": round(confidence, 6),
            "feedback_split": feedback_split,
            "human_verified": True,
        },
        "defect_taxonomy": {
            "human_reported": labels,
            "aggregate_observed": aggregate_defect_labels,
            "residual_vocal_labels": sorted(RESIDUAL_VOCAL_LABELS),
            "instrumental_preservation_labels": sorted(INSTRUMENTAL_PRESERVATION_LABELS),
            "noise_artifact_labels": sorted(NOISE_ARTIFACT_LABELS),
        },
        "segment_count": len(segments),
        "segments_jsonl": str(output_dir / SEGMENTS_NAME),
        "feedback_jsonl": str(output_dir / FEEDBACK_NAME),
        "progress_log_path": str(output_dir / PROGRESS_NAME),
        "materialized_audio": {
            "enabled": bool(materialize_audio),
            "segment_count": len(materialized_segments),
        },
        "delivery_contract": {
            "delivery_allowed": False,
            "delivery_status": "not_a_delivery_artifact",
            "presentation_status": "training_only_not_for_listen_pack",
            "release_status": "training_artifact_only",
            "listen_pack_allowed": False,
        },
        "execution_contract": {
            "route_name": "build_instrumental_preference_dataset",
            "docker_runtime_required": True,
            "expected_acceleration_mode": (
                "cpu_audio_slicing" if materialize_audio else "cpu_metadata_dataset_build"
            ),
            "gpu_inference_skipped": True,
            "gpu_inference_skip_reason": "dataset_manifest_build_no_model_inference",
            "per_track_hardcode_allowed": False,
            "fallback_used": False,
        },
        "next_training_use": {
            "recommended_route": "teacher_calibrated_instrumental_selector",
            "purpose": "learn segment-level tradeoff between residual vocal suppression and instrumental preservation",
            "not_sufficient_for_claim": "full separator fine-tune or accepted listen delivery",
        },
    }

    progress.mark("write_artifacts", segment_count=len(segments))
    write_jsonl(output_dir / SEGMENTS_NAME, segments)
    write_jsonl(output_dir / FEEDBACK_NAME, [feedback_record])
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress.mark(
        "finish",
        status="finish",
        manifest_path=str(output_dir / MANIFEST_NAME),
        segment_count=len(segments),
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_audio")
    parser.add_argument("output_dir")
    parser.add_argument("--preferred-stem-manifest", required=True)
    parser.add_argument("--gate-manifest")
    parser.add_argument("--preferred-profile")
    parser.add_argument("--defect-label", action="append", default=[])
    parser.add_argument("--rating", choices=RATINGS, default="good")
    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument("--feedback-split", default="train")
    parser.add_argument("--segment-sec", type=float, default=5.0)
    parser.add_argument("--materialize-audio", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_instrumental_preference_dataset(
        source_audio_path=Path(args.source_audio),
        preferred_stem_manifest_path=Path(args.preferred_stem_manifest),
        output_dir=Path(args.output_dir),
        gate_manifest_path=Path(args.gate_manifest) if args.gate_manifest else None,
        preferred_profile=args.preferred_profile,
        human_defect_labels=tuple(args.defect_label),
        rating=args.rating,
        confidence=float(args.confidence),
        feedback_split=args.feedback_split,
        segment_sec=float(args.segment_sec),
        materialize_audio=bool(args.materialize_audio),
    )
    print(json.dumps({"manifest_path": str(Path(args.output_dir) / MANIFEST_NAME), "segment_count": manifest["segment_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
