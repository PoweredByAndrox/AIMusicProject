from __future__ import annotations

import math
from typing import Any, Iterable


SUPPORTED_MODES = {"vocal", "minus", "both"}

SCORE_WEIGHTS: dict[str, dict[str, float]] = {
    "vocal": {
        "vocal_naturalness": 0.28,
        "instrumental_bleed_absence": 0.22,
        "artifact_absence": 0.16,
        "vocal_body_preservation": 0.12,
        "vocal_tail_preservation": 0.10,
        "phase_stereo_stability": 0.07,
        "temporal_stability": 0.05,
    },
    "minus": {
        "vocal_leakage_absence": 0.30,
        "instrumental_integrity": 0.20,
        "artifact_absence": 0.15,
        "transient_preservation": 0.12,
        "spectral_balance": 0.10,
        "phase_stereo_stability": 0.08,
        "temporal_stability": 0.05,
    },
    "both": {
        "mixture_reconstruction": 0.40,
        "phase_alignment": 0.20,
        "duration_channel_sr_compatibility": 0.15,
        "energy_balance": 0.15,
        "no_cross_stem_conflict": 0.10,
    },
}

ACCEPT_THRESHOLD = 0.72
REVIEW_THRESHOLD = 0.55
MINUS_REQUIRED_EVIDENCE_METRICS = (
    ("vocal_leakage_in_minus_score", "missing_minus_vocal_leakage_evidence"),
    ("mixture_consistency_score", "missing_minus_instrumental_integrity_evidence"),
)
VOCAL_NUMERIC_SCORE_METRICS = (
    "instrumental_leakage_in_vocal_score",
    "mixture_consistency_score",
    "phase_consistency_score",
    "robotic_artifact_score",
    "silence_hallucination_score",
    "thin_vocal_score",
    "transient_damage_score",
    "vocal_activity_confidence",
    "vocal_phase_consistency_score",
    "vocal_tail_loss_score",
)
MINUS_NUMERIC_SCORE_METRICS = (
    "high_band_damage_score",
    "instrumental_phase_consistency_score",
    "minus_phase_consistency_score",
    "mixture_consistency_score",
    "phase_consistency_score",
    "pumping_or_holes_score",
    "robotic_artifact_score",
    "spectral_balance_damage_score",
    "transient_damage_score",
    "vocal_leakage_in_minus_score",
)
PAIR_NUMERIC_SCORE_METRICS = (
    "channels",
    "duration",
    "duration_sec",
    "duration_seconds",
    "energy_balance_score",
    "instrumental_leakage_in_vocal_score",
    "mixture_consistency_score",
    "pair_phase_consistency_score",
    "phase_consistency_score",
    "sample_rate",
    "vocal_leakage_in_minus_score",
)
MODE_NUMERIC_SCORE_METRICS = {
    "vocal": VOCAL_NUMERIC_SCORE_METRICS,
    "minus": MINUS_NUMERIC_SCORE_METRICS,
    "both": PAIR_NUMERIC_SCORE_METRICS,
}


def clip01(value: float) -> float:
    number = finite_float(value)
    if number is None:
        return 0.0
    return round(max(0.0, min(1.0, number)), 6)


def validate_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported no-reference score mode: {mode!r}; expected one of both, minus, vocal")
    return normalized


def as_float(value: Any, default: float = 0.0) -> float:
    number = finite_float(value)
    if number is not None:
        return number
    fallback = finite_float(default)
    return fallback if fallback is not None else 0.0


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def stable_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def metric(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    if key in payload:
        return as_float(payload.get(key), 0.0)
    audio_info = payload.get("audio_info")
    if isinstance(audio_info, dict) and key in audio_info:
        return as_float(audio_info.get(key), 0.0)
    return as_float(default, 0.0)


def has_metric(payload: dict[str, Any], key: str) -> bool:
    if key in payload:
        return True
    audio_info = payload.get("audio_info")
    return isinstance(audio_info, dict) and key in audio_info


def metric_value(payload: dict[str, Any], key: str) -> tuple[bool, Any]:
    if key in payload:
        return True, payload.get(key)
    audio_info = payload.get("audio_info")
    if isinstance(audio_info, dict) and key in audio_info:
        return True, audio_info.get(key)
    return False, None


def has_finite_metric(payload: dict[str, Any], key: str) -> bool:
    exists, value = metric_value(payload, key)
    return exists and finite_float(value) is not None


def invalid_numeric_metric_blockers(payload: dict[str, Any], mode: str) -> list[str]:
    mode = validate_mode(mode)
    blockers: list[str] = []
    for key in MODE_NUMERIC_SCORE_METRICS[mode]:
        exists, value = metric_value(payload, key)
        if exists and finite_float(value) is None:
            blockers.append(f"invalid_no_reference_metric:{key}")
    return stable_unique(blockers)


def missing_required_evidence(payload: dict[str, Any], mode: str) -> list[str]:
    mode = validate_mode(mode)
    if mode != "minus":
        return []
    return [
        blocker
        for metric_name, blocker in MINUS_REQUIRED_EVIDENCE_METRICS
        if not has_finite_metric(payload, metric_name)
    ]


def mode_phase_metric(payload: dict[str, Any], mode: str) -> float:
    mode = validate_mode(mode)
    if mode == "vocal":
        return metric(payload, "vocal_phase_consistency_score", metric(payload, "phase_consistency_score", 1.0))
    if mode == "minus":
        return metric(
            payload,
            "instrumental_phase_consistency_score",
            metric(payload, "minus_phase_consistency_score", metric(payload, "phase_consistency_score", 1.0)),
        )
    return metric(payload, "pair_phase_consistency_score", metric(payload, "phase_consistency_score", 1.0))


def has_mode_specific_phase(payload: dict[str, Any], mode: str) -> bool:
    mode = validate_mode(mode)
    if mode == "vocal":
        return has_metric(payload, "vocal_phase_consistency_score")
    if mode == "minus":
        return has_metric(payload, "instrumental_phase_consistency_score") or has_metric(payload, "minus_phase_consistency_score")
    return has_metric(payload, "pair_phase_consistency_score")


def _reasons(payload: dict[str, Any]) -> set[str]:
    return set(as_list(payload.get("reasons")) + as_list(payload.get("failure_labels")) + as_list(payload.get("failed_checks")))


def _decision(score: float, blockers: list[str]) -> str:
    if blockers:
        return "blocked_hard_case"
    if score >= ACCEPT_THRESHOLD:
        return "accept_candidate"
    if score >= REVIEW_THRESHOLD:
        return "select_alternative_candidate"
    return "blocked_hard_case"


def _vocal_components(metrics: dict[str, Any]) -> dict[str, float]:
    reasons = _reasons(metrics)
    hallucination = metric(metrics, "silence_hallucination_score")
    robotic = metric(metrics, "robotic_artifact_score")
    transient = metric(metrics, "transient_damage_score")
    thin = metric(metrics, "thin_vocal_score")
    tail_loss = max(metric(metrics, "vocal_tail_loss_score"), 1.0 if "vocal_tail_loss" in reasons else 0.0)
    phase = mode_phase_metric(metrics, "vocal")
    return {
        "vocal_naturalness": clip01(min(metric(metrics, "vocal_activity_confidence", 1.0), 1.0 - hallucination, 1.0 - robotic * 0.50)),
        "instrumental_bleed_absence": clip01(1.0 - metric(metrics, "instrumental_leakage_in_vocal_score")),
        "artifact_absence": clip01(1.0 - max(robotic, transient)),
        "vocal_body_preservation": clip01(1.0 - thin),
        "vocal_tail_preservation": clip01(1.0 - tail_loss),
        "phase_stereo_stability": clip01(phase),
        "temporal_stability": clip01(1.0 - max(transient, hallucination * 0.50)),
    }


def _minus_components(metrics: dict[str, Any]) -> dict[str, float]:
    reasons = _reasons(metrics)
    robotic = metric(metrics, "robotic_artifact_score")
    transient = metric(metrics, "transient_damage_score")
    high_band_damage = max(metric(metrics, "high_band_damage_score"), 1.0 if "high_band_damage" in reasons else 0.0)
    pumping = max(metric(metrics, "pumping_or_holes_score"), 1.0 if "pumping_or_holes" in reasons else 0.0)
    mixture = metric(metrics, "mixture_consistency_score", 1.0)
    return {
        "vocal_leakage_absence": clip01(1.0 - metric(metrics, "vocal_leakage_in_minus_score")),
        "instrumental_integrity": clip01(min(mixture, 1.0 - pumping)),
        "artifact_absence": clip01(1.0 - max(robotic, high_band_damage * 0.70)),
        "transient_preservation": clip01(1.0 - transient),
        "spectral_balance": clip01(1.0 - max(high_band_damage, metric(metrics, "spectral_balance_damage_score"))),
        "phase_stereo_stability": clip01(mode_phase_metric(metrics, "minus")),
        "temporal_stability": clip01(1.0 - max(transient * 0.70, pumping)),
    }


def _weighted_score(components: dict[str, float], weights: dict[str, float]) -> float:
    return clip01(sum(weights[name] * components[name] for name in weights))


def detect_blockers(metrics: dict[str, Any], mode: str) -> list[str]:
    mode = validate_mode(mode)
    reasons = _reasons(metrics)
    if mode == "vocal":
        blockers = invalid_numeric_metric_blockers(metrics, "vocal")
        if metric(metrics, "instrumental_leakage_in_vocal_score") > 0.72 or "instrumental_leakage_in_vocal" in reasons:
            blockers.append("severe_instrumental_bleed_in_vocal")
        if metric(metrics, "robotic_artifact_score") > 0.82 or reasons.intersection({"robotic_or_noise_artifact", "robotic_artifact", "musical_noise"}):
            blockers.append("severe_robotic_musical_noise")
        if metric(metrics, "thin_vocal_score") > 0.78 or reasons.intersection({"thin_or_overcleaned_vocal", "thin_or_overcleaned"}):
            blockers.append("severe_vocal_body_loss")
        if metric(metrics, "vocal_tail_loss_score") > 0.75 or "vocal_tail_loss" in reasons:
            blockers.append("severe_vocal_tail_cut")
        legacy_pair_phase_reason = "phase_or_stereo_inconsistency" in reasons and not has_mode_specific_phase(metrics, "vocal")
        if mode_phase_metric(metrics, "vocal") < 0.30 or legacy_pair_phase_reason:
            blockers.append("severe_phase_comb_filtering")
        if metric(metrics, "vocal_activity_confidence", 1.0) < 0.25 and metric(metrics, "mixture_consistency_score", 1.0) > 0.55:
            blockers.append("vocal_missing_on_active_region")
        return stable_unique(blockers)
    if mode == "minus":
        blockers = invalid_numeric_metric_blockers(metrics, "minus") + missing_required_evidence(metrics, "minus")
        if metric(metrics, "vocal_leakage_in_minus_score") > 0.60 or "vocal_leakage_in_minus" in reasons:
            blockers.append("severe_vocal_bleed_in_minus")
        if metric(metrics, "mixture_consistency_score", 1.0) < 0.55 or "music_damage" in reasons:
            blockers.append("severe_music_damage")
        if metric(metrics, "transient_damage_score") > 0.82 or "transient_damage" in reasons:
            blockers.append("severe_transient_smearing")
        if mode_phase_metric(metrics, "minus") < 0.30 or "phase_collapse" in reasons:
            blockers.append("severe_phase_collapse")
        if metric(metrics, "pumping_or_holes_score") > 0.75 or "pumping_or_holes" in reasons:
            blockers.append("severe_pumping_or_holes")
        if metric(metrics, "high_band_damage_score") > 0.75 or "high_band_damage" in reasons:
            blockers.append("severe_high_band_damage")
        return stable_unique(blockers)
    return list(pair_compatibility_score(metrics, metrics).get("pair_blockers", []))


def dominant_defect(metrics: dict[str, Any], mode: str) -> str | None:
    mode = validate_mode(mode)
    reasons = _reasons(metrics)
    if mode == "vocal":
        ordered = [
            ("instrumental_bleed_in_vocal", metric(metrics, "instrumental_leakage_in_vocal_score")),
            ("robotic_musical_noise", metric(metrics, "robotic_artifact_score")),
            ("vocal_body_loss", max(metric(metrics, "thin_vocal_score"), 1.0 if reasons.intersection({"thin_or_overcleaned_vocal", "thin_or_overcleaned"}) else 0.0)),
            ("vocal_tail_loss", max(metric(metrics, "vocal_tail_loss_score"), 1.0 if "vocal_tail_loss" in reasons else 0.0)),
            ("phase_comb_filtering", 1.0 - mode_phase_metric(metrics, "vocal")),
            ("vocal_missing_on_active_region", 1.0 - metric(metrics, "vocal_activity_confidence", 1.0)),
        ]
    elif mode == "minus":
        ordered = [
            ("vocal_bleed_in_minus", metric(metrics, "vocal_leakage_in_minus_score")),
            ("music_damage", 1.0 - metric(metrics, "mixture_consistency_score", 1.0)),
            ("transient_smearing", metric(metrics, "transient_damage_score")),
            ("phase_collapse", 1.0 - mode_phase_metric(metrics, "minus")),
            ("pumping_or_holes", max(metric(metrics, "pumping_or_holes_score"), 1.0 if "pumping_or_holes" in reasons else 0.0)),
            ("high_band_damage", max(metric(metrics, "high_band_damage_score"), 1.0 if "high_band_damage" in reasons else 0.0)),
            ("spectral_balance_damage", metric(metrics, "spectral_balance_damage_score")),
        ]
    else:
        blockers = detect_blockers(metrics, mode)
        return blockers[0] if blockers else None
    defect, value = max(ordered, key=lambda item: (item[1], -ordered.index(item)))
    return defect if value >= 0.45 else None


ACTION_RULES: dict[tuple[str, str], dict[str, list[str]]] = {
    ("vocal", "vocal_body_loss"): {
        "allowed_actions": ["rollback_to_fuller_vocal_candidate", "body_preserving_blend_plan"],
        "forbidden_actions": ["stronger_cleanup", "stronger_debleed", "highpass_aggressive", "denoise_stack"],
    },
    ("vocal", "vocal_tail_loss"): {
        "allowed_actions": ["tail_preserving_candidate", "segment_tail_blend_plan"],
        "forbidden_actions": ["hard_vad_gate", "aggressive_silence_gate"],
    },
    ("vocal", "robotic_musical_noise"): {
        "allowed_actions": ["less_aggressive_candidate", "rollback", "artifact_safe_candidate"],
        "forbidden_actions": ["more_denoise", "cleanup_stack"],
    },
    ("vocal", "instrumental_bleed_in_vocal"): {
        "allowed_actions": ["cleaner_vocal_candidate", "segment_vocal_switch"],
        "forbidden_actions": ["global_aggressive_cleanup"],
    },
    ("vocal", "phase_comb_filtering"): {
        "allowed_actions": ["bounded_phase_alignment_candidate", "stereo_consistency_candidate", "phase_safe_candidate_rollback"],
        "forbidden_actions": ["stronger_cleanup", "stronger_debleed", "denoise_stack", "hard_vad_gate", "threshold_relaxation"],
    },
    ("minus", "vocal_bleed_in_minus"): {
        "allowed_actions": ["choose_candidate_with_lower_vocal_leakage", "minus_specific_candidate_selection", "frequency_aware_minus_selection"],
        "forbidden_actions": ["stronger_vocal_cleanup", "denoise_stack", "accept_candidate_only_because_vocal_improved"],
    },
    ("minus", "music_damage"): {
        "allowed_actions": ["rollback_to_music_preserving_minus", "less_aggressive_minus_candidate"],
        "forbidden_actions": ["over_suppress_mid_band", "more_debleed"],
    },
    ("minus", "transient_smearing"): {
        "allowed_actions": ["transient_preserving_candidate", "rollback"],
        "forbidden_actions": ["smoothing_stack", "heavy_overlap_blend_without_check"],
    },
    ("minus", "phase_collapse"): {
        "allowed_actions": ["bounded_phase_alignment_candidate", "stereo_consistency_candidate", "phase_safe_candidate_rollback"],
        "forbidden_actions": ["stronger_cleanup", "more_debleed", "denoise_stack", "threshold_relaxation"],
    },
}


def allowed_actions_for_defect(defect: str, mode: str) -> dict[str, list[str]]:
    mode = validate_mode(mode)
    rule = ACTION_RULES.get((mode, str(defect)))
    if rule is None:
        return {"allowed_actions": ["select_existing_candidate_variant"], "forbidden_actions": ["do_not_add_cleanup_debleed_denoise", "do_not_train"]}
    return {"allowed_actions": list(rule["allowed_actions"]), "forbidden_actions": list(rule["forbidden_actions"])}


def score_candidate(metrics: dict[str, Any], mode: str) -> dict[str, Any]:
    mode = validate_mode(mode)
    if mode == "both":
        return pair_compatibility_score(metrics, metrics)
    if mode == "vocal":
        components = _vocal_components(metrics)
        score = _weighted_score(components, SCORE_WEIGHTS["vocal"])
        blockers = detect_blockers(metrics, "vocal")
        defect = dominant_defect(metrics, "vocal")
        return {
            "score_mode": "vocal",
            "vocal_score": score,
            "vocal_score_components": components,
            "vocal_blockers": blockers,
            "vocal_dominant_defect": defect,
            "vocal_decision": _decision(score, blockers),
        }
    components = _minus_components(metrics)
    score = _weighted_score(components, SCORE_WEIGHTS["minus"])
    blockers = detect_blockers(metrics, "minus")
    defect = dominant_defect(metrics, "minus")
    return {
        "score_mode": "minus",
        "minus_score": score,
        "minus_score_components": components,
        "minus_blockers": blockers,
        "minus_dominant_defect": defect,
        "minus_decision": _decision(score, blockers),
    }


def _identity_value(payload: dict[str, Any], *keys: str) -> tuple[str | None, Any]:
    for key in keys:
        exists, value = metric_value(payload, key)
        if exists:
            return key, value
    return None, None


def _identity_finite_number(payload: dict[str, Any], *keys: str) -> tuple[str | None, float | None, bool]:
    key, value = _identity_value(payload, *keys)
    if key is None:
        return None, None, False
    number = finite_float(value)
    return key, number, number is None


def _compatibility_identity_score(vocal_candidate: dict[str, Any], minus_candidate: dict[str, Any]) -> tuple[float, list[str]]:
    blockers: list[str] = []
    score = 1.0
    for key, blocker in (("sample_rate", "sample_rate_mismatch"), ("channels", "channel_mismatch")):
        _, left, left_invalid = _identity_finite_number(vocal_candidate, key)
        _, right, right_invalid = _identity_finite_number(minus_candidate, key)
        if left_invalid or right_invalid:
            blockers.append(f"invalid_no_reference_metric:{key}")
            score = 0.0
            continue
        if left is not None and right is not None and left != right:
            blockers.append(blocker)
            score = min(score, 0.0)
    left_key, left_duration, left_duration_invalid = _identity_finite_number(vocal_candidate, "duration_sec", "duration_seconds", "duration")
    right_key, right_duration, right_duration_invalid = _identity_finite_number(minus_candidate, "duration_sec", "duration_seconds", "duration")
    for key, invalid in ((left_key, left_duration_invalid), (right_key, right_duration_invalid)):
        if key is not None and invalid:
            blockers.append(f"invalid_no_reference_metric:{key}")
            score = 0.0
    if left_duration is not None and right_duration is not None and abs(left_duration - right_duration) > 0.05:
        blockers.append("duration_mismatch")
        score = min(score, 0.0)
    return score, blockers


def pair_compatibility_score(vocal_candidate: dict[str, Any], minus_candidate: dict[str, Any]) -> dict[str, Any]:
    identity_score, blockers = _compatibility_identity_score(vocal_candidate, minus_candidate)
    blockers.extend(invalid_numeric_metric_blockers(vocal_candidate, "both"))
    blockers.extend(invalid_numeric_metric_blockers(minus_candidate, "both"))
    mixture = clip01(min(metric(vocal_candidate, "mixture_consistency_score", 1.0), metric(minus_candidate, "mixture_consistency_score", 1.0)))
    phase = clip01(min(mode_phase_metric(vocal_candidate, "both"), mode_phase_metric(minus_candidate, "both")))
    vocal_leakage = max(metric(vocal_candidate, "vocal_leakage_in_minus_score"), metric(minus_candidate, "vocal_leakage_in_minus_score"))
    inst_leakage = max(metric(vocal_candidate, "instrumental_leakage_in_vocal_score"), metric(minus_candidate, "instrumental_leakage_in_vocal_score"))
    no_cross_conflict = clip01(1.0 - max(vocal_leakage, inst_leakage))
    energy_balance = clip01(min(metric(vocal_candidate, "energy_balance_score", 1.0), metric(minus_candidate, "energy_balance_score", 1.0)))
    if mixture < 0.55:
        blockers.append("severe_mixture_reconstruction_drift")
    if phase < 0.30:
        blockers.append("phase_collapse")
    components = {
        "mixture_reconstruction": mixture,
        "phase_alignment": phase,
        "duration_channel_sr_compatibility": clip01(identity_score),
        "energy_balance": energy_balance,
        "no_cross_stem_conflict": no_cross_conflict,
    }
    score = _weighted_score(components, SCORE_WEIGHTS["both"])
    blockers = stable_unique(blockers)
    if blockers:
        decision = "blocked_pair_incompatible"
    elif score >= ACCEPT_THRESHOLD:
        decision = "pair_compatible"
    elif score >= REVIEW_THRESHOLD:
        decision = "needs_pair_reconciliation"
    else:
        decision = "blocked_pair_incompatible"
    return {
        "score_mode": "both",
        "pair_compatibility_score": score,
        "pair_score_components": components,
        "pair_blockers": blockers,
        "pair_decision": decision,
    }
