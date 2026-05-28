from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf


SCHEMA_VERSION = 1
MANIFEST_NAME = "teacher_calibrated_selector_manifest.json"
PLAN_NAME = "segment_selection_plan.jsonl"
PROGRESS_NAME = "teacher_calibrated_selector_progress.jsonl"
OUTPUT_AUDIO_NAME = "teacher_calibrated_instrumental.flac"

POSITIVE_RATINGS = {"accept", "good", "better"}
NEGATIVE_RATINGS = {"reject", "bad", "worse"}
RATINGS = sorted(POSITIVE_RATINGS | NEGATIVE_RATINGS)
SEGMENT_HUMAN_PRIOR_WEIGHT = 2.25

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


@dataclass(frozen=True)
class Candidate:
    profile: str
    manifest_path: Path
    instrumental_path: Path
    vocal_path: Path | None
    contract: dict[str, Any]


class ProgressWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started_at = time.monotonic()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def mark(self, stage: str, *, status: str = "progress", **metadata: Any) -> None:
        event = {
            "type": "teacher_calibrated_selector_stage",
            "stage": stage,
            "status": status,
            "elapsed_sec": 0.0,
            "since_start_sec": round(time.monotonic() - self.started_at, 6),
            "metadata": metadata,
        }
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        loaded = json.loads(line)
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        records.append(loaded)
    return records


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


def instrumental_path_from_manifest(manifest: dict[str, Any]) -> Path:
    stems = manifest.get("stems")
    if isinstance(stems, dict):
        instrumental = stems.get("instrumental")
        if isinstance(instrumental, dict):
            path = _as_path(instrumental.get("path"))
            if path is not None:
                return path

    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, dict):
        for key in ("instrumental_flac", "instrumental_path"):
            path = _as_path(artifacts.get(key))
            if path is not None:
                return path

    direct_path = _as_path(manifest.get("instrumental_path"))
    if direct_path is not None:
        return direct_path

    raise ValueError("Candidate stem manifest has no instrumental path")


def vocal_path_from_manifest(manifest: dict[str, Any]) -> Path | None:
    stems = manifest.get("stems")
    if isinstance(stems, dict):
        vocals = stems.get("vocals")
        if isinstance(vocals, dict):
            path = _as_path(vocals.get("path"))
            if path is not None:
                return path

    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, dict):
        for key in ("vocals_flac", "vocals_path", "vocal_flac", "vocal_path"):
            path = _as_path(artifacts.get(key))
            if path is not None:
                return path

    direct_path = _as_path(manifest.get("vocals_path")) or _as_path(manifest.get("vocal_path"))
    return direct_path


def candidate_profile_from_manifest(manifest: dict[str, Any], fallback: str) -> str:
    for key in ("model_id", "model_name", "profile_name"):
        value = manifest.get(key)
        if isinstance(value, str) and value:
            return value
    return fallback


def load_candidate(path: Path) -> Candidate:
    manifest = load_json(path)
    return Candidate(
        profile=candidate_profile_from_manifest(manifest, path.parent.name),
        manifest_path=path,
        instrumental_path=instrumental_path_from_manifest(manifest),
        vocal_path=vocal_path_from_manifest(manifest),
        contract=manifest.get("contract", {}) if isinstance(manifest.get("contract"), dict) else {},
    )


def _mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float64, copy=False)
    return np.mean(audio.astype(np.float64, copy=False), axis=1)


def _read_window(path: Path, start_sec: float, end_sec: float) -> tuple[np.ndarray, int]:
    info = sf.info(str(path))
    sample_rate = int(info.samplerate)
    start_frame = max(0, int(round(float(start_sec) * sample_rate)))
    end_frame = max(start_frame + 1, int(round(float(end_sec) * sample_rate)))
    frames = max(1, end_frame - start_frame)
    audio, sr = sf.read(str(path), start=start_frame, frames=frames, always_2d=True, dtype="float32")
    return _mono(audio), int(sr)


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


def _band_energy(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    if audio.size == 0:
        return 0.0
    windowed = audio * np.hanning(audio.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(audio.size, d=1.0 / sample_rate)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return 0.0
    return float(np.sqrt(np.mean(np.square(spectrum[mask]))) + 1e-12)


def _db_ratio(numerator: float, denominator: float) -> float:
    return float(20.0 * math.log10(max(numerator, 1e-12) / max(denominator, 1e-12)))


def _correlation_abs(left: np.ndarray, right: np.ndarray) -> float:
    frame_count = min(left.size, right.size)
    if frame_count == 0:
        return 0.0
    left_centered = left[:frame_count] - float(np.mean(left[:frame_count]))
    right_centered = right[:frame_count] - float(np.mean(right[:frame_count]))
    denominator = math.sqrt(
        max(float(np.dot(left_centered, left_centered)) * float(np.dot(right_centered, right_centered)), 1e-24)
    )
    if denominator <= 0.0:
        return 0.0
    return abs(float(np.dot(left_centered, right_centered)) / denominator)


def _projection_db_vs_reference(audio: np.ndarray, reference: np.ndarray) -> float:
    frame_count = min(audio.size, reference.size)
    if frame_count == 0:
        return -240.0
    audio_centered = audio[:frame_count] - float(np.mean(audio[:frame_count]))
    reference_centered = reference[:frame_count] - float(np.mean(reference[:frame_count]))
    reference_energy = float(np.dot(reference_centered, reference_centered))
    if reference_energy <= 1e-18:
        return -240.0
    projection_gain = float(np.dot(audio_centered, reference_centered)) / reference_energy
    projected = reference_centered * projection_gain
    return _db_ratio(_rms(projected), _rms(audio_centered))


def segment_features(
    source_audio_path: Path,
    candidate_path: Path,
    window_sec: Sequence[float],
    vocal_stem_path: Path | None = None,
) -> dict[str, float]:
    start_sec, end_sec = float(window_sec[0]), float(window_sec[1])
    source_audio, source_sr = _read_window(source_audio_path, start_sec, end_sec)
    candidate_audio, candidate_sr = _read_window(candidate_path, start_sec, end_sec)
    if source_sr != candidate_sr:
        raise ValueError(f"Sample rate mismatch: {source_audio_path}={source_sr}, {candidate_path}={candidate_sr}")

    source_rms = _rms(source_audio)
    candidate_rms = _rms(candidate_audio)
    source_presence = _band_energy(source_audio, source_sr, 1800.0, 6000.0)
    candidate_presence = _band_energy(candidate_audio, candidate_sr, 1800.0, 6000.0)
    source_air = _band_energy(source_audio, source_sr, 6000.0, 12000.0)
    candidate_air = _band_energy(candidate_audio, candidate_sr, 6000.0, 12000.0)
    features = {
        "rms": round(candidate_rms, 9),
        "source_rms": round(source_rms, 9),
        "rms_ratio": round(candidate_rms / max(source_rms, 1e-12), 9),
        "rms_ratio_db_vs_source": round(_db_ratio(candidate_rms, source_rms), 6),
        "presence_ratio_db_vs_source": round(_db_ratio(candidate_presence, source_presence), 6),
        "air_ratio_db_vs_source": round(_db_ratio(candidate_air, source_air), 6),
    }
    if vocal_stem_path is not None:
        vocal_audio, vocal_sr = _read_window(vocal_stem_path, start_sec, end_sec)
        if vocal_sr != candidate_sr:
            raise ValueError(f"Sample rate mismatch: {vocal_stem_path}={vocal_sr}, {candidate_path}={candidate_sr}")
        vocal_rms = _rms(vocal_audio)
        features.update(
            {
                "vocal_stem_rms": round(vocal_rms, 9),
                "vocal_stem_corr_abs": round(_correlation_abs(candidate_audio, vocal_audio), 9),
                "vocal_projection_db_vs_instrumental": round(
                    _projection_db_vs_reference(candidate_audio, vocal_audio),
                    6,
                ),
            }
        )
    return features


def feedback_priors(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    priors: dict[str, float] = {}
    for record in records:
        split = record.get("feedback_split") or record.get("split")
        if record.get("holdout") is True or (isinstance(split, str) and split.lower() == "holdout"):
            continue
        profile = record.get("profile") or record.get("selected_profile")
        if not isinstance(profile, str) or not profile:
            continue
        rating = record.get("rating")
        confidence = record.get("confidence", 1.0)
        confidence_value = max(0.0, min(1.0, float(confidence) if isinstance(confidence, (int, float)) else 1.0))
        delta = 0.0
        if rating in POSITIVE_RATINGS:
            delta += 0.18 * confidence_value
        elif rating in NEGATIVE_RATINGS:
            delta -= 0.24 * confidence_value
        if record.get("residual_vocal_bleed") is True:
            delta -= 0.03
        if record.get("too_dry") is True:
            delta -= 0.03
        if record.get("artifact_noise") is True:
            delta -= 0.05
        priors[profile] = round(priors.get(profile, 0.0) + delta, 6)
    return priors


def segment_feedback_priors(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    priors: dict[str, dict[str, float]] = {}
    for record in records:
        split = record.get("feedback_split") or record.get("split")
        if record.get("holdout") is True or (isinstance(split, str) and split.lower() == "holdout"):
            continue
        segment_id = record.get("segment_id")
        if not isinstance(segment_id, str) or not segment_id:
            continue

        confidence = record.get("confidence", 1.0)
        confidence_value = max(0.0, min(1.0, float(confidence) if isinstance(confidence, (int, float)) else 1.0))
        if confidence_value == 0.0:
            continue

        rating = record.get("rating")
        verdict = record.get("human_ab_verdict") or record.get("ab_verdict")
        preferred_profile = record.get("preferred_profile") or record.get("baseline_profile")
        rejected_profile = record.get("rejected_profile") or record.get("candidate_profile")
        profile = record.get("profile") or record.get("selected_profile")

        positive_profile: str | None = None
        negative_profile: str | None = None
        if verdict in {"baseline_better", "a_better", "preferred_better"}:
            positive_profile = preferred_profile if isinstance(preferred_profile, str) else None
            negative_profile = rejected_profile if isinstance(rejected_profile, str) else profile if isinstance(profile, str) else None
        elif verdict in {"candidate_better", "b_better", "tested_better"}:
            positive_profile = rejected_profile if isinstance(rejected_profile, str) else profile if isinstance(profile, str) else None
            negative_profile = preferred_profile if isinstance(preferred_profile, str) else None
        elif rating in POSITIVE_RATINGS:
            positive_profile = profile if isinstance(profile, str) else preferred_profile if isinstance(preferred_profile, str) else None
        elif rating in NEGATIVE_RATINGS:
            negative_profile = profile if isinstance(profile, str) else rejected_profile if isinstance(rejected_profile, str) else None
            if isinstance(preferred_profile, str):
                positive_profile = preferred_profile

        segment_priors = priors.setdefault(segment_id, {})
        delta = SEGMENT_HUMAN_PRIOR_WEIGHT * confidence_value
        if positive_profile:
            segment_priors[positive_profile] = round(segment_priors.get(positive_profile, 0.0) + delta, 6)
        if negative_profile:
            segment_priors[negative_profile] = round(segment_priors.get(negative_profile, 0.0) - delta, 6)

    return {segment_id: values for segment_id, values in priors.items() if values}


def _target_ratio_for_labels(labels: set[str]) -> tuple[float, str]:
    if labels & RESIDUAL_VOCAL_LABELS and not labels & INSTRUMENTAL_PRESERVATION_LABELS:
        return 0.34, "residual_vocal_suppression"
    if labels & INSTRUMENTAL_PRESERVATION_LABELS and not labels & RESIDUAL_VOCAL_LABELS:
        return 0.86, "instrumental_preservation"
    if labels & RESIDUAL_VOCAL_LABELS and labels & INSTRUMENTAL_PRESERVATION_LABELS:
        return 0.86, "balanced_no_vocal_proxy_preservation_first"
    return 0.64, "neutral_preference"


def _has_vocal_leakage_proxy(features: dict[str, float]) -> bool:
    vocal_corr = float(features.get("vocal_stem_corr_abs", 0.0))
    vocal_projection_db = float(features.get("vocal_projection_db_vs_instrumental", -240.0))
    return vocal_corr > 0.12 or vocal_projection_db > -22.0


def score_candidate(
    *,
    profile: str,
    features: dict[str, float],
    labels: Sequence[str],
    preferred_profile: str,
    priors: dict[str, float],
    segment_prior: float = 0.0,
) -> tuple[float, list[str]]:
    label_set = set(labels)
    target_ratio, mode = _target_ratio_for_labels(label_set)
    has_vocal_stem_proxy = (
        "vocal_stem_corr_abs" in features or "vocal_projection_db_vs_instrumental" in features
    )
    has_balanced_defects = bool(label_set & RESIDUAL_VOCAL_LABELS and label_set & INSTRUMENTAL_PRESERVATION_LABELS)
    vocal_leakage_proxy = _has_vocal_leakage_proxy(features)
    if has_balanced_defects and has_vocal_stem_proxy:
        if vocal_leakage_proxy:
            target_ratio = 0.72
            mode = "balanced_suppress_when_vocal_leakage_proxy_present"
        else:
            target_ratio = 0.86
            mode = "balanced_preserve_when_vocal_leakage_proxy_low"
    ratio = float(features.get("rms_ratio", 0.0))
    score = 0.0
    reasons = [f"target_mode={mode}", f"target_rms_ratio={target_ratio:.3f}"]

    ratio_error = abs(ratio - target_ratio)
    score -= ratio_error * 3.0
    reasons.append(f"rms_ratio_error_penalty=-{ratio_error * 3.0:.3f}")

    if label_set & INSTRUMENTAL_PRESERVATION_LABELS:
        floor_ratio = 0.58 if label_set & RESIDUAL_VOCAL_LABELS and vocal_leakage_proxy else 0.52
        if ratio < floor_ratio:
            penalty = (floor_ratio - ratio) * 2.4
            score -= penalty
            reasons.append(f"muffling_floor_penalty=-{penalty:.3f}")
        if ratio > 1.02:
            penalty = (ratio - 1.02) * 5.0
            score -= penalty
            reasons.append(f"full_mix_like_penalty=-{penalty:.3f}")
        presence_db = float(features.get("presence_ratio_db_vs_source", 0.0))
        if presence_db < -9.0:
            penalty = min(2.5, (-9.0 - presence_db) * 0.08)
            score -= penalty
            reasons.append(f"presence_loss_penalty=-{penalty:.3f}")
        air_db = float(features.get("air_ratio_db_vs_source", 0.0))
        if air_db < -12.0:
            penalty = min(2.0, (-12.0 - air_db) * 0.05)
            score -= penalty
            reasons.append(f"air_loss_penalty=-{penalty:.3f}")

    if label_set & RESIDUAL_VOCAL_LABELS:
        ratio_bleed_threshold = 0.72
        if has_balanced_defects and (not has_vocal_stem_proxy or not vocal_leakage_proxy):
            ratio_bleed_threshold = 1.02
        if ratio > ratio_bleed_threshold:
            penalty = (ratio - ratio_bleed_threshold) * 4.2
            score -= penalty
            reasons.append(f"residual_vocal_bleed_proxy_penalty=-{penalty:.3f}")
        if ratio < 0.18:
            penalty = (0.18 - ratio) * 1.8
            score -= penalty
            reasons.append(f"over_suppression_penalty=-{penalty:.3f}")
        vocal_corr = float(features.get("vocal_stem_corr_abs", 0.0))
        if vocal_corr > 0.12:
            penalty = min(2.0, (vocal_corr - 0.12) * 5.0)
            score -= penalty
            reasons.append(f"vocal_stem_corr_penalty=-{penalty:.3f}")
        vocal_projection_db = float(features.get("vocal_projection_db_vs_instrumental", -240.0))
        if vocal_projection_db > -22.0:
            penalty = min(1.5, (vocal_projection_db + 22.0) * 0.08)
            score -= penalty
            reasons.append(f"vocal_projection_penalty=-{penalty:.3f}")

    if label_set & NOISE_ARTIFACT_LABELS:
        presence_db = float(features.get("presence_ratio_db_vs_source", 0.0))
        if presence_db > 1.0:
            penalty = min(2.5, (presence_db - 1.0) * 0.25)
            score -= penalty
            reasons.append(f"artifact_presence_penalty=-{penalty:.3f}")

    prior = priors.get(profile, 0.0)
    if profile == preferred_profile:
        prior += 0.04
    score += prior
    if prior:
        reasons.append(f"feedback_prior={prior:.3f}")
    if segment_prior:
        score += segment_prior
        reasons.append(f"segment_feedback_prior={segment_prior:.3f}")

    return round(score, 6), reasons


def _source_audio_from_dataset(dataset_manifest: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override
    path = _as_path(dataset_manifest.get("source_audio_path"))
    if path is None:
        raise ValueError("Dataset manifest has no source_audio_path; pass --source-audio")
    return path


def _preferred_profile(dataset_manifest: dict[str, Any]) -> str:
    preferred = dataset_manifest.get("preferred_candidate")
    if isinstance(preferred, dict) and isinstance(preferred.get("profile"), str):
        return preferred["profile"]
    return "unknown_preferred_profile"


def _load_dataset_segments(dataset_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    segments_path = _as_path(dataset_manifest.get("segments_jsonl"))
    if segments_path is None:
        raise ValueError("Dataset manifest has no segments_jsonl")
    records = load_jsonl(segments_path)
    if not records:
        raise ValueError(f"No segment records in {segments_path}")
    return records


def _load_dataset_feedback(dataset_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    feedback_path = _as_path(dataset_manifest.get("feedback_jsonl"))
    if feedback_path is None or not feedback_path.exists():
        return []
    return load_jsonl(feedback_path)


def select_segments(
    *,
    source_audio_path: Path,
    segments: Sequence[dict[str, Any]],
    candidates: Sequence[Candidate],
    preferred_profile: str,
    priors: dict[str, float],
    segment_priors: dict[str, dict[str, float]] | None = None,
    progress: ProgressWriter | None = None,
) -> list[dict[str, Any]]:
    if not candidates:
        raise ValueError("At least one candidate stem manifest is required")

    selections: list[dict[str, Any]] = []
    for segment in segments:
        window_sec = segment.get("window_sec")
        if (
            not isinstance(window_sec, list)
            or len(window_sec) != 2
            or not all(isinstance(value, (int, float)) for value in window_sec)
        ):
            raise ValueError(f"Invalid segment window: {window_sec!r}")
        labels = segment.get("defect_labels")
        if not isinstance(labels, list):
            labels = []
        segment_id = str(segment.get("segment_id"))
        segment_prior_values = segment_priors.get(segment_id, {}) if segment_priors is not None else {}

        scored_candidates: list[dict[str, Any]] = []
        for candidate in candidates:
            features = segment_features(source_audio_path, candidate.instrumental_path, window_sec, candidate.vocal_path)
            score, reasons = score_candidate(
                profile=candidate.profile,
                features=features,
                labels=[str(label) for label in labels if isinstance(label, str)],
                preferred_profile=preferred_profile,
                priors=priors,
                segment_prior=float(segment_prior_values.get(candidate.profile, 0.0)),
            )
            scored_candidates.append(
                {
                    "profile": candidate.profile,
                    "instrumental_path": str(candidate.instrumental_path),
                    "score": score,
                    "features": features,
                    "score_reasons": reasons,
                }
            )

        scored_candidates.sort(key=lambda item: (-float(item["score"]), str(item["profile"])))
        selected = scored_candidates[0]
        selection = {
            "segment_id": segment.get("segment_id"),
            "window_sec": [round(float(window_sec[0]), 6), round(float(window_sec[1]), 6)],
            "defect_labels": [str(label) for label in labels if isinstance(label, str)],
            "training_targets": segment.get("training_targets", []),
            "preferred_profile": preferred_profile,
            "selected_profile": selected["profile"],
            "selected_instrumental_path": selected["instrumental_path"],
            "selected_score": selected["score"],
            "candidate_scores": scored_candidates,
            "selector_contract": {
                "proxy_only": True,
                "requires_downstream_quality_gate": True,
                "not_delivery_result": True,
            },
        }
        selections.append(selection)
        if progress is not None:
            progress.mark(
                "score_segment",
                segment_id=str(selection["segment_id"]),
                selected_profile=str(selection["selected_profile"]),
                candidate_count=len(scored_candidates),
            )

    return selections


def _read_full_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    return audio, int(sample_rate)


def _render_output_subtype(base_audio_path: Path, output_path: Path) -> str | None:
    if output_path.suffix.lower() != ".flac":
        return None
    subtype = sf.info(str(base_audio_path)).subtype
    if subtype in {"PCM_S8", "PCM_16", "PCM_24"}:
        return subtype
    return None


def render_selected_instrumental(
    *,
    output_path: Path,
    candidates_by_profile: dict[str, Candidate],
    preferred_profile: str,
    selections: Sequence[dict[str, Any]],
    crossfade_ms: float,
    progress: ProgressWriter,
) -> Path:
    base_candidate = candidates_by_profile.get(preferred_profile)
    if base_candidate is None:
        base_candidate = next(iter(candidates_by_profile.values()))

    base_audio, sample_rate = _read_full_audio(base_candidate.instrumental_path)
    rendered = np.array(base_audio, copy=True)
    crossfade_frames = max(0, int(round(sample_rate * max(0.0, crossfade_ms) / 1000.0)))

    for selection in selections:
        selected_profile = str(selection["selected_profile"])
        selected_candidate = candidates_by_profile[selected_profile]
        if selected_candidate.instrumental_path == base_candidate.instrumental_path:
            continue
        selected_audio, selected_sr = _read_full_audio(selected_candidate.instrumental_path)
        if selected_sr != sample_rate:
            raise ValueError(
                f"Sample rate mismatch for selected candidate {selected_profile}: {selected_sr} != {sample_rate}"
            )
        window_sec = selection["window_sec"]
        start_frame = max(0, int(round(float(window_sec[0]) * sample_rate)))
        end_frame = min(rendered.shape[0], int(round(float(window_sec[1]) * sample_rate)))
        if end_frame <= start_frame:
            continue
        end_frame = min(end_frame, selected_audio.shape[0])
        replacement = selected_audio[start_frame:end_frame]
        if replacement.shape[1] != rendered.shape[1]:
            replacement = np.tile(replacement[:, :1], (1, rendered.shape[1]))

        if crossfade_frames <= 0:
            rendered[start_frame:end_frame] = replacement
        else:
            local_frames = end_frame - start_frame
            fade = min(crossfade_frames, max(0, local_frames // 2))
            if fade:
                fade_in = np.linspace(0.0, 1.0, fade, dtype=np.float32).reshape(-1, 1)
                rendered[start_frame : start_frame + fade] = (
                    rendered[start_frame : start_frame + fade] * (1.0 - fade_in)
                    + replacement[:fade] * fade_in
                )
                fade_out = np.linspace(1.0, 0.0, fade, dtype=np.float32).reshape(-1, 1)
                rendered[end_frame - fade : end_frame] = (
                    replacement[-fade:] * fade_out
                    + rendered[end_frame - fade : end_frame] * (1.0 - fade_out)
                )
                rendered[start_frame + fade : end_frame - fade] = replacement[fade:-fade]
            else:
                rendered[start_frame:end_frame] = replacement
        progress.mark("render_segment", segment_id=str(selection["segment_id"]), selected_profile=selected_profile)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_subtype = _render_output_subtype(base_candidate.instrumental_path, output_path)
    if output_subtype is None:
        sf.write(str(output_path), rendered, sample_rate)
    else:
        sf.write(str(output_path), rendered, sample_rate, format="FLAC", subtype=output_subtype)
    return output_path


def build_teacher_calibrated_selector(
    *,
    dataset_manifest_path: Path,
    output_dir: Path,
    candidate_stem_manifest_paths: Sequence[Path],
    source_audio_path: Path | None = None,
    render_audio: bool = False,
    crossfade_ms: float = 15.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressWriter(output_dir / PROGRESS_NAME)
    progress.mark(
        "start",
        dataset_manifest_path=str(dataset_manifest_path),
        candidate_count=len(candidate_stem_manifest_paths),
        expected_acceleration_mode="cpu_audio_selector",
        render_audio=bool(render_audio),
    )

    dataset_manifest = load_json(dataset_manifest_path)
    source_path = _source_audio_from_dataset(dataset_manifest, source_audio_path)
    preferred_profile = _preferred_profile(dataset_manifest)
    segments = _load_dataset_segments(dataset_manifest)
    feedback = _load_dataset_feedback(dataset_manifest)
    priors = feedback_priors(feedback)
    per_segment_priors = segment_feedback_priors(feedback)
    candidates = [load_candidate(path) for path in candidate_stem_manifest_paths]
    candidates_by_profile = {candidate.profile: candidate for candidate in candidates}
    if len(candidates_by_profile) != len(candidates):
        raise ValueError("Candidate profile names must be unique")
    if preferred_profile not in candidates_by_profile:
        raise ValueError(f"Preferred profile {preferred_profile!r} is missing from candidate manifests")

    progress.mark("score_segments", segment_count=len(segments), preferred_profile=preferred_profile)
    selections = select_segments(
        source_audio_path=source_path,
        segments=segments,
        candidates=candidates,
        preferred_profile=preferred_profile,
        priors=priors,
        segment_priors=per_segment_priors,
        progress=progress,
    )

    rendered_path: Path | None = None
    if render_audio:
        progress.mark("render_audio", segment_count=len(selections), crossfade_ms=float(crossfade_ms))
        rendered_path = render_selected_instrumental(
            output_path=output_dir / OUTPUT_AUDIO_NAME,
            candidates_by_profile=candidates_by_profile,
            preferred_profile=preferred_profile,
            selections=selections,
            crossfade_ms=float(crossfade_ms),
            progress=progress,
        )

    write_jsonl(output_dir / PLAN_NAME, selections)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "mode": "teacher_calibrated_instrumental_selector",
        "dataset_manifest_path": str(dataset_manifest_path),
        "source_audio_path": str(source_path),
        "source_audio_sha256": sha256_file(source_path),
        "preferred_profile": preferred_profile,
        "candidate_count": len(candidates),
        "segment_count": len(selections),
        "feedback_priors": priors,
        "segment_feedback_priors": per_segment_priors,
        "candidate_manifests": [
            {
                "profile": candidate.profile,
                "stem_manifest_path": str(candidate.manifest_path),
                "instrumental_path": str(candidate.instrumental_path),
                "instrumental_sha256": sha256_file(candidate.instrumental_path),
                "vocal_path": str(candidate.vocal_path) if candidate.vocal_path is not None else None,
                "vocal_sha256": sha256_file(candidate.vocal_path) if candidate.vocal_path is not None else None,
                "contract": candidate.contract,
            }
            for candidate in candidates
        ],
        "segment_selection": selections,
        "segment_selection_plan_jsonl": str(output_dir / PLAN_NAME),
        "artifacts": {
            "diagnostic_instrumental_flac": str(rendered_path) if rendered_path is not None else None,
            "progress_log_path": str(output_dir / PROGRESS_NAME),
        },
        "rendered_audio_write_contract": {
            "rendered_audio_path": str(rendered_path) if rendered_path is not None else None,
            "base_profile": preferred_profile,
            "base_source_subtype": sf.info(str(candidates_by_profile[preferred_profile].instrumental_path)).subtype,
            "output_subtype": sf.info(str(rendered_path)).subtype if rendered_path is not None else None,
            "subtype_policy": "preserve_base_flac_pcm_subtype_when_supported",
            "gain_applied": False,
            "normalization_applied": False,
            "limiting_applied": False,
        },
        "delivery_contract": {
            "delivery_allowed": False,
            "delivery_status": "not_a_delivery_artifact",
            "presentation_status": "diagnostic_selector_not_for_listen_pack",
            "release_status": "diagnostic_training_selector_only",
            "listen_pack_allowed": False,
            "requires_downstream_quality_gate": True,
        },
        "execution_contract": {
            "route_name": "teacher_calibrated_instrumental_selector",
            "docker_runtime_required": True,
            "expected_acceleration_mode": "cpu_audio_selector",
            "gpu_inference_skipped": True,
            "gpu_inference_skip_reason": "selector_consumes_precomputed_model_candidates",
            "per_track_hardcode_allowed": False,
            "fallback_used": False,
            "proxy_only_scoring": True,
        },
        "next_gate_required": {
            "route": "instrumental_quality_gate_or_reference_holdout",
            "reason": "selector output is proxy-ranked and diagnostic until objective/listening gate accepts it",
        },
    }
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress.mark(
        "finish",
        status="finish",
        manifest_path=str(output_dir / MANIFEST_NAME),
        rendered_audio=str(rendered_path) if rendered_path is not None else None,
    )
    return manifest


def parse_candidate_manifest_spec(raw: str) -> Path:
    if "=" in raw:
        _, raw_path = raw.split("=", 1)
        return Path(raw_path)
    return Path(raw)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_manifest")
    parser.add_argument("output_dir")
    parser.add_argument("--candidate-stem-manifest", action="append", required=True)
    parser.add_argument("--source-audio")
    parser.add_argument("--render-audio", action="store_true")
    parser.add_argument("--crossfade-ms", type=float, default=15.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    candidate_paths = [parse_candidate_manifest_spec(raw) for raw in args.candidate_stem_manifest]
    manifest = build_teacher_calibrated_selector(
        dataset_manifest_path=Path(args.dataset_manifest),
        output_dir=Path(args.output_dir),
        candidate_stem_manifest_paths=candidate_paths,
        source_audio_path=Path(args.source_audio) if args.source_audio else None,
        render_audio=bool(args.render_audio),
        crossfade_ms=float(args.crossfade_ms),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(Path(args.output_dir) / MANIFEST_NAME),
                "segment_count": manifest["segment_count"],
                "rendered_audio": manifest["artifacts"]["diagnostic_instrumental_flac"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
