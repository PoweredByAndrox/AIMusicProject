from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf
from scipy import signal


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.no_reference_score_matrix import score_candidate, validate_mode  # noqa: E402


DECISIONS = {
    "accept_candidate",
    "select_alternative_candidate",
    "needs_segment_blend",
    "blocked_hard_case",
}

STEM_DIRECTORY_NAMES = {"vocal", "vocals", "minus", "instrumental", "melody", "drums", "bass", "other"}

BASE_FORBIDDEN_ACTIONS = [
    "do_not_require_reference_stems",
    "do_not_require_human_csv",
    "do_not_start_finetune",
    "do_not_train",
    "do_not_add_models",
    "do_not_add_cleanup_debleed_denoise",
    "do_not_relax_gates",
    "do_not_export_diagnostic_audio_as_result",
    "do_not_accept_best_of_bad",
]

SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "diagnostic_audio_exported": False,
    "audio_copied": False,
    "audio_exported": False,
    "audio_read_for_judging": True,
}

EPS = 1.0e-9
PASSTHROUGH_REMOVAL_RATIO_FULL_SCALE = 0.08
PASSTHROUGH_MIX_SIMILARITY_THRESHOLD = 0.92


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    text = str(raw_path)
    if text.startswith("/app/"):
        candidate = APP_ROOT / text.removeprefix("/app/")
        if candidate.exists():
            return candidate
    return path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def stable_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    if audio.size == 0:
        raise ValueError(f"Empty audio file: {path}")
    return audio, int(sample_rate)


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    gcd = math.gcd(source_rate, target_rate)
    up = target_rate // gcd
    down = source_rate // gcd
    return signal.resample_poly(audio, up, down, axis=0).astype(np.float32)


def align_audio(arrays: list[np.ndarray]) -> list[np.ndarray]:
    channels = max(array.shape[1] for array in arrays)
    expanded: list[np.ndarray] = []
    for array in arrays:
        if array.shape[1] == channels:
            expanded.append(array)
        elif array.shape[1] == 1:
            expanded.append(np.repeat(array, channels, axis=1))
        else:
            expanded.append(array[:, :channels])
    min_len = min(array.shape[0] for array in expanded)
    return [array[:min_len].astype(np.float64) for array in expanded]


def mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float64)
    return audio.mean(axis=1).astype(np.float64)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio))) + EPS)


def cosine_abs(left: np.ndarray, right: np.ndarray) -> float:
    left = left.reshape(-1)
    right = right.reshape(-1)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right)) + EPS
    return float(abs(np.dot(left, right)) / denom)


def projection_gain(target: np.ndarray, basis: np.ndarray) -> float:
    target = target.reshape(-1)
    basis = basis.reshape(-1)
    return float(abs(np.dot(target, basis)) / (float(np.dot(basis, basis)) + EPS))


def minus_vocal_leakage_proxy(mix_mono: np.ndarray, vocal_mono: np.ndarray, instrumental_mono: np.ndarray) -> dict[str, Any]:
    """Estimate vocal bleed in minus without trusting a weak generated vocal basis."""
    mix_energy = rms(mix_mono)
    vocal_proxy_energy_ratio = rms(vocal_mono) / (mix_energy + EPS)
    removed_by_minus = mix_mono - instrumental_mono
    removed_energy_ratio = rms(removed_by_minus) / (mix_energy + EPS)
    residual_leakage = cosine_abs(instrumental_mono, removed_by_minus) if removed_energy_ratio >= 0.03 else 0.0
    vocal_basis_projection = projection_gain(instrumental_mono, vocal_mono)
    vocal_basis_supported = vocal_proxy_energy_ratio >= 0.08
    mix_passthrough_similarity = cosine_abs(instrumental_mono, mix_mono)
    passthrough_under_removal = clip01(
        (PASSTHROUGH_REMOVAL_RATIO_FULL_SCALE - removed_energy_ratio) / PASSTHROUGH_REMOVAL_RATIO_FULL_SCALE
    )
    passthrough_leakage = (
        mix_passthrough_similarity * passthrough_under_removal
        if (
            not vocal_basis_supported
            and removed_energy_ratio < PASSTHROUGH_REMOVAL_RATIO_FULL_SCALE
            and mix_passthrough_similarity > PASSTHROUGH_MIX_SIMILARITY_THRESHOLD
        )
        else 0.0
    )
    raw_leakage = max(
        residual_leakage,
        clip01(vocal_basis_projection) if vocal_basis_supported else 0.0,
        passthrough_leakage,
    )
    return {
        "raw_leakage": clip01(raw_leakage),
        "vocal_proxy_energy_ratio": round(float(vocal_proxy_energy_ratio), 6),
        "minus_removed_energy_ratio": round(float(removed_energy_ratio), 6),
        "minus_residual_leakage_score": clip01(residual_leakage),
        "vocal_basis_projection_score": clip01(vocal_basis_projection),
        "vocal_basis_supported": bool(vocal_basis_supported),
        "mix_passthrough_similarity": clip01(mix_passthrough_similarity),
        "passthrough_under_removal_score": passthrough_under_removal,
        "passthrough_leakage_score": clip01(passthrough_leakage),
    }


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    phat = float(successes) / float(total)
    denominator = 1.0 + z * z / float(total)
    centre = phat + z * z / (2.0 * float(total))
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * float(total))) / float(total))
    return max(0.0, (centre - margin) / denominator)


def clip01(value: float) -> float:
    return round(float(max(0.0, min(1.0, value))), 6)


def band_energy(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    audio = mono(audio)
    if audio.size < 8:
        return 0.0
    spectrum = np.fft.rfft(audio * np.hanning(audio.size))
    freqs = np.fft.rfftfreq(audio.size, 1.0 / sample_rate)
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.square(np.abs(spectrum[mask]))))


def spectral_flatness(audio: np.ndarray) -> float:
    audio = mono(audio)
    if audio.size < 8 or rms(audio) < EPS * 10:
        return 0.0
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(audio.size))) + EPS
    return float(np.exp(np.mean(np.log(spectrum))) / (np.mean(spectrum) + EPS))


def window_rms(audio: np.ndarray, sample_rate: int, window_sec: float = 0.05) -> np.ndarray:
    audio = mono(audio)
    window = max(1, int(sample_rate * window_sec))
    usable = (audio.size // window) * window
    if usable == 0:
        return np.array([rms(audio)])
    return np.sqrt(np.mean(np.square(audio[:usable].reshape(-1, window)), axis=1)) + EPS


def phase_consistency(audio: np.ndarray) -> float:
    if audio.ndim < 2 or audio.shape[1] < 2:
        return 1.0
    return clip01((cosine_abs(audio[:, 0], audio[:, 1]) + 0.15) / 1.15)


def judge_audio(mix: np.ndarray, vocal: np.ndarray, instrumental: np.ndarray, sample_rate: int) -> dict[str, Any]:
    mix, vocal, instrumental = align_audio([mix, vocal, instrumental])
    reconstruction = vocal + instrumental
    residual = mix - reconstruction
    mix_mono = mono(mix)
    vocal_mono = mono(vocal)
    instrumental_mono = mono(instrumental)

    residual_ratio = rms(residual) / (rms(mix) + EPS)
    mixture_consistency_score = clip01(1.0 - residual_ratio)

    minus_leakage = minus_vocal_leakage_proxy(mix_mono, vocal_mono, instrumental_mono)
    vocal_projection_in_minus = float(minus_leakage["raw_leakage"])
    instrumental_projection_in_vocal = projection_gain(vocal_mono, instrumental_mono)
    vocal_leakage_in_minus_score = clip01((vocal_projection_in_minus - 0.12) / 0.88)
    instrumental_leakage_in_vocal_score = clip01((instrumental_projection_in_vocal - 0.18) / 0.82)

    mix_windows = window_rms(mix, sample_rate)
    vocal_windows = window_rms(vocal, sample_rate)
    mix_active_threshold = max(np.percentile(mix_windows, 30) * 0.6, rms(mix) * 0.05, EPS)
    vocal_active_threshold = max(np.percentile(vocal_windows, 55) * 0.7, rms(vocal) * 0.08, EPS)
    active_mix = mix_windows > mix_active_threshold
    active_vocal = vocal_windows > vocal_active_threshold
    vocal_activity_confidence = clip01(float(active_vocal.sum()) / float(max(1, active_mix.sum())))
    quiet_mix = mix_windows <= mix_active_threshold
    hallucination_ratio = wilson_lower_bound(int((quiet_mix & active_vocal).sum()), int(quiet_mix.sum()))
    quiet_energy_ratio = float(np.mean(vocal_windows[quiet_mix])) / (float(np.mean(mix_windows[active_mix])) + EPS) if np.any(quiet_mix) and np.any(active_mix) else 0.0
    silence_hallucination_score = clip01(max(hallucination_ratio, quiet_energy_ratio))

    flatness = spectral_flatness(vocal)
    robotic_artifact_score = clip01((flatness - 0.35) / 0.65)

    low_mid = band_energy(vocal, sample_rate, 150.0, 1200.0)
    presence = band_energy(vocal, sample_rate, 3500.0, 9000.0)
    thin_vocal_score = clip01((presence / (low_mid + EPS) - 0.85) / 2.0)

    transient_residual_ratio = rms(np.diff(mono(residual), prepend=0.0)) / (rms(np.diff(mix_mono, prepend=0.0)) + EPS)
    transient_damage_score = clip01((transient_residual_ratio - 0.55) / 1.10)

    mix_phase_score = phase_consistency(mix)
    vocal_phase_score = phase_consistency(vocal)
    instrumental_phase_score = phase_consistency(instrumental)
    phase_score = clip01(min(mix_phase_score, vocal_phase_score, instrumental_phase_score))

    severe_reasons: list[str] = []
    if mixture_consistency_score < 0.55:
        severe_reasons.append("bad_mixture_reconstruction")
    if vocal_leakage_in_minus_score > 0.60:
        severe_reasons.append("vocal_leakage_in_minus")
    if instrumental_leakage_in_vocal_score > 0.72:
        severe_reasons.append("instrumental_leakage_in_vocal")
    if silence_hallucination_score > 0.75:
        severe_reasons.append("silence_hallucination")
    if robotic_artifact_score > 0.82:
        severe_reasons.append("robotic_or_noise_artifact")
    if thin_vocal_score > 0.78:
        severe_reasons.append("thin_or_overcleaned_vocal")
    if transient_damage_score > 0.82:
        severe_reasons.append("transient_damage")
    if phase_score < 0.30:
        severe_reasons.append("phase_or_stereo_inconsistency")

    overall = clip01(
        0.30 * mixture_consistency_score
        + 0.14 * (1.0 - vocal_leakage_in_minus_score)
        + 0.12 * (1.0 - instrumental_leakage_in_vocal_score)
        + 0.10 * vocal_activity_confidence
        + 0.09 * (1.0 - silence_hallucination_score)
        + 0.08 * (1.0 - robotic_artifact_score)
        + 0.07 * (1.0 - thin_vocal_score)
        + 0.05 * (1.0 - transient_damage_score)
        + 0.05 * phase_score
    )
    if severe_reasons:
        decision = "blocked_hard_case"
    elif overall >= 0.72:
        decision = "accept_candidate"
    elif overall >= 0.55:
        decision = "select_alternative_candidate"
    else:
        decision = "blocked_hard_case"
        severe_reasons.append("low_no_reference_score")
    confidence = clip01((0.30 if severe_reasons else 0.45) + 0.55 * abs(overall - 0.50) * 2.0)
    reasons = stable_unique(severe_reasons or ["no_reference_proxy_passed"])

    return {
        "mixture_consistency_score": mixture_consistency_score,
        "window_count": int(len(mix_windows)),
        "vocal_leakage_in_minus_score": vocal_leakage_in_minus_score,
        "vocal_leakage_proxy_mode": "residual_or_reliable_vocal_proxy",
        "vocal_proxy_energy_ratio": minus_leakage["vocal_proxy_energy_ratio"],
        "minus_removed_energy_ratio": minus_leakage["minus_removed_energy_ratio"],
        "minus_residual_leakage_score": minus_leakage["minus_residual_leakage_score"],
        "vocal_basis_projection_score": minus_leakage["vocal_basis_projection_score"],
        "vocal_basis_supported": minus_leakage["vocal_basis_supported"],
        "mix_passthrough_similarity": minus_leakage["mix_passthrough_similarity"],
        "passthrough_leakage_score": minus_leakage["passthrough_leakage_score"],
        "instrumental_leakage_in_vocal_score": instrumental_leakage_in_vocal_score,
        "vocal_activity_confidence": vocal_activity_confidence,
        "silence_hallucination_score": silence_hallucination_score,
        "robotic_artifact_score": robotic_artifact_score,
        "thin_vocal_score": thin_vocal_score,
        "transient_damage_score": transient_damage_score,
        "phase_consistency_score": phase_score,
        "mix_phase_consistency_score": mix_phase_score,
        "vocal_phase_consistency_score": vocal_phase_score,
        "instrumental_phase_consistency_score": instrumental_phase_score,
        "pair_phase_consistency_score": phase_score,
        "overall_no_reference_score": overall,
        "confidence": confidence,
        "decision": decision,
        "reasons": reasons,
    }


def candidate_row_from_manifest(path: Path, ordinal: int) -> dict[str, Any]:
    payload = read_json(path)
    flat: list[tuple[str, Any]] = []

    def walk(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{prefix}.{key}" if prefix else key)
        elif isinstance(value, list):
            for index, child in enumerate(value[:20]):
                walk(child, f"{prefix}.{index}" if prefix else str(index))
        else:
            flat.append((prefix, value))

    walk(payload)

    def first_path(patterns: Sequence[str]) -> str:
        for key, value in flat:
            text = str(value)
            if not (text.endswith(".wav") or text.endswith(".flac") or text.endswith(".mp3")):
                continue
            key_lower = key.lower()
            text_lower = text.lower()
            if any(pattern in key_lower or pattern in text_lower for pattern in patterns):
                return text
        return ""

    def stem_path(*names: str) -> str:
        stems = payload.get("stems")
        if not isinstance(stems, dict):
            return ""
        for name in names:
            entry = stems.get(name)
            if isinstance(entry, dict):
                raw_path = entry.get("path") or entry.get("audio_path") or entry.get("file")
            else:
                raw_path = entry
            if raw_path:
                return str(raw_path)
        return ""

    def layout_track_id() -> str:
        if path.parent.parent.name.lower() in STEM_DIRECTORY_NAMES:
            return path.parent.parent.parent.name
        if path.parent.name.lower() in STEM_DIRECTORY_NAMES:
            return path.parent.parent.name
        return path.parent.name

    embedded_row = payload.get("candidate_row")
    embedded_row = embedded_row if isinstance(embedded_row, dict) else {}

    def embedded_value(*keys: str) -> str:
        for key in keys:
            value = embedded_row.get(key)
            if value:
                return str(value)
        return ""

    mix_path = (
        embedded_value("mix_path", "source_audio_path")
        or str(payload.get("source_audio_path") or payload.get("mix_path") or payload.get("raw_audio_path") or "")
    ) or first_path(
        ["source_audio", "raw_audio", "audio_path", "mix_path", "source.mp3"]
    )
    vocal_path = (
        embedded_value("selected_vocal_path", "vocal_path")
        or stem_path("vocals", "vocal")
        or first_path(["stems.vocals.path", "vocals.flac", "vocal_path"])
    )
    instrumental_path = (
        embedded_value("selected_minus_path", "instrumental_path", "minus_path", "rendered_minus_candidate_path")
        or str(payload.get("rendered_minus_candidate_path") or payload.get("selected_minus_path") or "")
        or stem_path("minus", "instrumental", "instrumentals")
    ) or first_path(
        ["repaired_minus", "base_minus", "instrumental", "minus_path", "minus.flac"]
    )
    track_id = str(payload.get("track_id") or payload.get("track_hash") or layout_track_id())
    return {
        "track_id": track_id,
        "candidate_id": str(
            payload.get("candidate_id")
            or embedded_value("candidate_id")
            or payload.get("model_id")
            or payload.get("pipeline", {}).get("candidate_mode")
            or f"manifest_{ordinal:04d}"
        ),
        "mix_path": mix_path,
        "vocal_path": vocal_path,
        "instrumental_path": instrumental_path,
        "candidate_manifest": str(path),
    }


def load_candidate_rows(candidate_jsonl: Path | None, candidate_manifests: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if candidate_jsonl is not None:
        rows.extend(iter_jsonl(candidate_jsonl))
    for ordinal, manifest_path in enumerate(candidate_manifests, start=1):
        rows.append(candidate_row_from_manifest(manifest_path, ordinal))
    return rows


MINUS_DETAIL_GATE_THRESHOLDS = {
    "instrumental_integrity": 0.70,
    "transient_preservation": 0.65,
    "spectral_balance": 0.65,
    "phase_stereo_stability": 0.55,
    "mixture_consistency_score": 0.70,
}


def finite_metric(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def build_minus_musical_detail_preservation_gate(row: dict[str, Any]) -> dict[str, Any]:
    components = row.get("minus_score_components")
    if not isinstance(components, dict):
        components = {}
    metrics = {
        "instrumental_integrity": finite_metric(components, "instrumental_integrity"),
        "transient_preservation": finite_metric(components, "transient_preservation"),
        "spectral_balance": finite_metric(components, "spectral_balance"),
        "phase_stereo_stability": finite_metric(components, "phase_stereo_stability"),
        "mixture_consistency_score": finite_metric(row, "mixture_consistency_score"),
        "minus_score": finite_metric(row, "minus_score"),
        "transient_damage_score": finite_metric(row, "transient_damage_score"),
    }
    blockers: list[dict[str, Any]] = []
    for metric_name, threshold in MINUS_DETAIL_GATE_THRESHOLDS.items():
        value = metrics[metric_name]
        if value < threshold:
            blockers.append(
                {
                    "label": f"{metric_name}_below_detail_gate",
                    "reason": "minus_candidate_loses_music_detail",
                    "metric": metric_name,
                    "value": round(value, 6),
                    "threshold": threshold,
                }
            )
    return {
        "schema_version": 1,
        "mode": "minus_musical_detail_preservation_gate",
        "status": "proxy_pass" if not blockers else "proxy_blocked",
        "evidence_type": "no_reference_minus_detail_proxy",
        "required_metrics": list(MINUS_DETAIL_GATE_THRESHOLDS),
        "metrics": {key: round(value, 6) for key, value in metrics.items()},
        "thresholds": dict(MINUS_DETAIL_GATE_THRESHOLDS),
        "blockers": blockers,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
    }


def apply_minus_detail_gate_override(row: dict[str, Any]) -> None:
    gate = row.get("minus_musical_detail_preservation_gate")
    if not isinstance(gate, dict) or gate.get("status") == "proxy_pass":
        return
    blockers = [
        str(blocker.get("label") or "minus_detail_gate_blocked")
        for blocker in gate.get("blockers", [])
        if isinstance(blocker, dict)
    ]
    row["decision"] = "blocked_hard_case"
    row["minus_decision"] = "blocked_hard_case"
    row["minus_blockers"] = stable_unique(as_list(row.get("minus_blockers")) + ["minus_detail_gate_blocked"] + blockers)
    row["reasons"] = stable_unique(as_list(row.get("reasons")) + ["minus_detail_gate_blocked"])
    if not row.get("minus_dominant_defect"):
        row["minus_dominant_defect"] = "music_damage"


def judge_candidate_row(row: dict[str, Any], *, score_mode: str | None = None) -> dict[str, Any]:
    required = ["track_id", "candidate_id", "mix_path", "vocal_path", "instrumental_path"]
    missing = [key for key in required if not row.get(key)]
    if missing:
        raise ValueError(f"Candidate row is missing required no-reference fields {missing}: {row}")
    mix_path = resolve_path(row["mix_path"])
    vocal_path = resolve_path(row["vocal_path"])
    instrumental_path = resolve_path(row["instrumental_path"])
    mix, sample_rate = read_audio(mix_path)
    vocal, vocal_rate = read_audio(vocal_path)
    instrumental, instrumental_rate = read_audio(instrumental_path)
    vocal = resample_audio(vocal, vocal_rate, sample_rate)
    instrumental = resample_audio(instrumental, instrumental_rate, sample_rate)
    metrics = judge_audio(mix, vocal, instrumental, sample_rate)
    matrix_fields = score_candidate(metrics, score_mode) if score_mode is not None else {}
    judged = {
        "track_id": str(row["track_id"]),
        "candidate_id": str(row["candidate_id"]),
        "mix_path": str(row["mix_path"]),
        "vocal_path": str(row["vocal_path"]),
        "instrumental_path": str(row["instrumental_path"]),
        "candidate_manifest": str(row.get("candidate_manifest") or ""),
        "judge_mode": "no_reference",
        **metrics,
        **matrix_fields,
        "forbidden_actions": stable_unique(BASE_FORBIDDEN_ACTIONS + as_list(row.get("forbidden_actions"))),
    }
    if score_mode == "minus":
        judged["minus_musical_detail_preservation_gate"] = build_minus_musical_detail_preservation_gate(judged)
        apply_minus_detail_gate_override(judged)
    return judged


def judge_candidate_rows(rows: list[dict[str, Any]], *, score_mode: str | None = None) -> dict[str, Any]:
    normalized_score_mode = validate_mode(score_mode) if score_mode is not None else None
    judged = [
        judge_candidate_row(row, score_mode=normalized_score_mode)
        for row in sorted(rows, key=lambda r: (str(r.get("track_id", "")), str(r.get("candidate_id", ""))))
    ]
    summary = build_summary(judged, score_mode=normalized_score_mode)
    return {"rows": judged, "summary": summary}


def summary_decision(row: dict[str, Any], score_mode: str | None = None) -> str:
    if score_mode == "vocal":
        return str(row.get("vocal_decision") or row.get("decision") or "")
    if score_mode == "minus":
        return str(row.get("minus_decision") or row.get("decision") or "")
    if score_mode == "both":
        pair_decision = str(row.get("pair_decision") or "")
        if pair_decision == "pair_compatible":
            return "accept_candidate"
        if pair_decision:
            return pair_decision
    return str(row.get("decision") or "")


def build_summary(rows: list[dict[str, Any]], *, score_mode: str | None = None) -> dict[str, Any]:
    normalized_score_mode = validate_mode(score_mode) if score_mode is not None else None
    decision_counts = Counter(summary_decision(row, normalized_score_mode) for row in rows)
    reasons = Counter(reason for row in rows for reason in as_list(row.get("reasons")))
    usable = decision_counts.get("accept_candidate", 0)
    summary = {
        "total_tracks": len({row["track_id"] for row in rows}),
        "candidates_judged": len(rows),
        "decision_counts": dict(sorted(decision_counts.items())),
        "accepted_candidates": usable,
        "blocked_hard_case": decision_counts.get("blocked_hard_case", 0),
        "top_failure_reasons": dict(reasons.most_common(10)),
        "usable_without_stems": usable,
        "safety": dict(SAFETY_FLAGS),
    }
    if normalized_score_mode is not None:
        summary["score_mode"] = normalized_score_mode
    return summary


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# No-Reference Result Summary",
        "",
        f"- total_tracks: {summary['total_tracks']}",
        f"- candidates_judged: {summary['candidates_judged']}",
        f"- accepted_candidates: {summary['accepted_candidates']}",
        f"- blocked_hard_case: {summary['blocked_hard_case']}",
        f"- usable_without_stems: {summary['usable_without_stems']}",
        "",
        "## Decisions",
        "",
    ]
    if summary.get("score_mode"):
        lines.insert(2, f"- score_mode: {summary['score_mode']}")
    for key, value in summary["decision_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Top Failure Reasons", ""])
    for key, value in summary["top_failure_reasons"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Safety", ""])
    for key, value in summary["safety"].items():
        lines.append(f"- {key}: {str(value).lower()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "no_reference_result_judge.jsonl", result["rows"])
    write_summary_md(output_dir / "no_reference_result_summary.md", result["summary"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Judge generated vocal/minus candidates without reference stems.")
    parser.add_argument("--candidate-jsonl", type=Path)
    parser.add_argument("--candidate-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-mode", choices=("vocal", "minus", "both"))
    args = parser.parse_args(argv)
    rows = load_candidate_rows(args.candidate_jsonl, args.candidate_manifest)
    if not rows:
        raise SystemExit("No candidate rows supplied")
    result = judge_candidate_rows(rows, score_mode=args.score_mode)
    write_outputs(result, args.output_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
