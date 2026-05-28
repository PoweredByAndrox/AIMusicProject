from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402


@dataclass(frozen=True)
class CandidateProfile:
    name: str
    pure_base: bool
    ambience_strength: float
    ambience_max_gain_db: float
    ambience_percentile_window_sec: float
    texture_strength: float
    texture_percentile_window_sec: float
    texture_max_floor_ratio: float
    enable_raw_kick_repair: bool
    prior_score: float
    intent: str


@dataclass(frozen=True)
class CandidateScore:
    profile_name: str
    score: float
    reasons: list[str]
    metrics: dict[str, Any]
    focus_metrics: dict[str, Any]
    manifest_path: str
    output_dir: str
    listen_dir: str | None


def default_candidate_profiles() -> tuple[CandidateProfile, ...]:
    return (
        CandidateProfile(
            name="pure_base",
            pure_base=True,
            ambience_strength=0.0,
            ambience_max_gain_db=0.0,
            ambience_percentile_window_sec=8.0,
            texture_strength=0.0,
            texture_percentile_window_sec=8.0,
            texture_max_floor_ratio=0.0,
            enable_raw_kick_repair=False,
            prior_score=0.55,
            intent="Use the Instrumental Resurrection base minus exactly, with no synthetic repair layer.",
        ),
        CandidateProfile(
            name="strict_no_texture",
            pure_base=False,
            ambience_strength=0.08,
            ambience_max_gain_db=0.25,
            ambience_percentile_window_sec=4.5,
            texture_strength=0.0,
            texture_percentile_window_sec=8.0,
            texture_max_floor_ratio=0.0,
            enable_raw_kick_repair=False,
            prior_score=0.08,
            intent="Prefer minimum synthetic layer and avoid texture artifacts.",
        ),
        CandidateProfile(
            name="soft_delta",
            pure_base=False,
            ambience_strength=0.16,
            ambience_max_gain_db=0.45,
            ambience_percentile_window_sec=4.0,
            texture_strength=0.18,
            texture_percentile_window_sec=6.0,
            texture_max_floor_ratio=0.08,
            enable_raw_kick_repair=False,
            prior_score=0.04,
            intent="Restore light background texture while staying vocal-safe.",
        ),
        CandidateProfile(
            name="vocal_safe_balanced",
            pure_base=False,
            ambience_strength=0.22,
            ambience_max_gain_db=0.65,
            ambience_percentile_window_sec=3.0,
            texture_strength=0.34,
            texture_percentile_window_sec=4.5,
            texture_max_floor_ratio=0.16,
            enable_raw_kick_repair=False,
            prior_score=0.0,
            intent="Use the existing balanced flexible repair defaults.",
        ),
    )


def build_repair_command(
    *,
    python_executable: str,
    repair_script: Path,
    raw_audio_path: Path,
    base_minus_path: Path,
    output_dir: Path,
    listen_dir: Path | None,
    listen_label: str,
    focus_windows: list[str],
    profile: CandidateProfile,
) -> list[str]:
    command = [
        python_executable,
        str(repair_script),
        str(raw_audio_path),
        str(base_minus_path),
        str(output_dir),
        "--ambience-strength",
        str(profile.ambience_strength),
        "--ambience-max-gain-db",
        str(profile.ambience_max_gain_db),
        "--ambience-percentile-window-sec",
        str(profile.ambience_percentile_window_sec),
        "--texture-strength",
        str(profile.texture_strength),
        "--texture-percentile-window-sec",
        str(profile.texture_percentile_window_sec),
        "--texture-max-floor-ratio",
        str(profile.texture_max_floor_ratio),
    ]

    if profile.enable_raw_kick_repair:
        command.append("--enable-raw-kick-repair")

    if profile.pure_base:
        command.append("--pure-base")

    if listen_dir is not None:
        command.extend(["--listen-dir", str(listen_dir), "--listen-label", listen_label])

    for focus_window in focus_windows:
        command.extend(["--focus-window", focus_window])

    return command


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return loaded


def load_feedback_adjustments(feedback_store_path: Path | None) -> dict[str, float]:
    if feedback_store_path is None or not feedback_store_path.exists():
        return {}

    adjustments: dict[str, float] = {}
    for line_number, raw_line in enumerate(
        feedback_store_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSONL feedback record at {feedback_store_path}:{line_number}"
            ) from exc

        profile_name = record.get("profile") or record.get("selected_profile")
        if not isinstance(profile_name, str) or not profile_name:
            continue
        split = record.get("feedback_split") or record.get("split")
        if record.get("holdout") is True or (isinstance(split, str) and split.lower() == "holdout"):
            continue

        delta = 0.0
        rating = record.get("rating")
        if rating in {"accept", "good", "better"}:
            delta += 0.20
        elif rating in {"reject", "bad", "worse"}:
            delta -= 0.28

        numeric_score = record.get("score")
        if isinstance(numeric_score, (int, float)):
            clamped_score = max(-1.0, min(1.0, float(numeric_score)))
            delta += clamped_score * 0.15

        if record.get("artifact_noise") is True:
            delta -= 0.12
        if record.get("too_dry") is True:
            delta -= 0.10

        adjustments[profile_name] = adjustments.get(profile_name, 0.0) + delta

    return adjustments


def metric_as_float(metrics: dict[str, Any], name: str, default: float = 0.0) -> float:
    value = metrics.get(name, default)
    if isinstance(value, (int, float)):
        numeric_value = float(value)
        return numeric_value if math.isfinite(numeric_value) else default
    return default


def has_finite_metric(metrics: dict[str, Any], name: str) -> bool:
    value = metrics.get(name)
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


MINUS_VOCAL_RESIDUE_CORR_THRESHOLD = 0.18
MINUS_VOCAL_RESIDUE_RMS_RATIO_THRESHOLD_DB = -18.0
MINUS_PRESENCE_DIP_THRESHOLD_DB = -0.35
MINUS_AIR_DIP_THRESHOLD_DB = -0.50
MINUS_RELEASE_REQUIRED_METRICS = (
    "vocal_residue_corr",
    "vocal_to_instrumental_rms_ratio_db",
    "presence_band_ratio_drift_db",
    "air_band_ratio_drift_db",
)


def build_minus_release_gate(metrics: dict[str, Any], focus_metrics: dict[str, Any]) -> dict[str, Any]:
    blockers: list[dict[str, Any]] = []
    missing_metrics = [
        key
        for key in MINUS_RELEASE_REQUIRED_METRICS
        if key not in metrics or not has_finite_metric(metrics, key)
    ]
    if missing_metrics:
        blockers.append(
            {
                "label": "missing_minus_release_metrics",
                "reason": "missing_minus_release_metrics",
                "required_metrics": list(MINUS_RELEASE_REQUIRED_METRICS),
                "missing_metrics": missing_metrics,
            }
        )
    vocal_residue_corr = metric_as_float(metrics, "vocal_residue_corr", 0.0)
    vocal_ratio_db = metric_as_float(metrics, "vocal_to_instrumental_rms_ratio_db", -120.0)
    presence_drift_db = metric_as_float(metrics, "presence_band_ratio_drift_db", 0.0)
    air_drift_db = metric_as_float(metrics, "air_band_ratio_drift_db", 0.0)

    if (
        vocal_residue_corr >= MINUS_VOCAL_RESIDUE_CORR_THRESHOLD
        and vocal_ratio_db >= MINUS_VOCAL_RESIDUE_RMS_RATIO_THRESHOLD_DB
    ):
        blockers.append(
            {
                "label": "vocal_residue_in_minus",
                "reason": "selected_minus_retains_vocal_residue",
                "vocal_residue_corr": round(vocal_residue_corr, 8),
                "vocal_to_instrumental_rms_ratio_db": round(vocal_ratio_db, 6),
            }
        )
    if presence_drift_db <= MINUS_PRESENCE_DIP_THRESHOLD_DB:
        blockers.append(
            {
                "label": "instrumental_presence_dip",
                "reason": "selected_minus_loses_presence_band",
                "presence_band_ratio_drift_db": round(presence_drift_db, 6),
            }
        )
    if air_drift_db <= MINUS_AIR_DIP_THRESHOLD_DB:
        blockers.append(
            {
                "label": "instrumental_air_dip",
                "reason": "selected_minus_loses_air_band",
                "air_band_ratio_drift_db": round(air_drift_db, 6),
            }
        )

    focus_blockers: list[dict[str, Any]] = []
    for label, focus_record in focus_metrics.items():
        if not isinstance(focus_record, dict):
            continue
        drift = focus_record.get("drift_vs_base")
        if not isinstance(drift, dict):
            continue
        focus_presence_drift_db = metric_as_float(drift, "presence_band_ratio_drift_db", 0.0)
        focus_air_drift_db = metric_as_float(drift, "air_band_ratio_drift_db", 0.0)
        if (
            focus_presence_drift_db <= MINUS_PRESENCE_DIP_THRESHOLD_DB
            or focus_air_drift_db <= MINUS_AIR_DIP_THRESHOLD_DB
        ):
            focus_blockers.append(
                {
                    "focus_window": str(label),
                    "reason": "focused_window_instrumental_dip",
                    "presence_band_ratio_drift_db": round(focus_presence_drift_db, 6),
                    "air_band_ratio_drift_db": round(focus_air_drift_db, 6),
                }
            )

    blockers.extend(focus_blockers)
    status = "blocked_for_listening" if blockers else "release_ready"
    return {
        "schema_version": 1,
        "mode": "minus_release_gate",
        "status": status,
        "release_status": "diagnostic_only" if blockers else "release_ready",
        "thresholds": {
            "vocal_residue_corr": MINUS_VOCAL_RESIDUE_CORR_THRESHOLD,
            "vocal_to_instrumental_rms_ratio_db": MINUS_VOCAL_RESIDUE_RMS_RATIO_THRESHOLD_DB,
            "presence_dip_db": MINUS_PRESENCE_DIP_THRESHOLD_DB,
            "air_dip_db": MINUS_AIR_DIP_THRESHOLD_DB,
        },
        "summary": {
            "blocker_count": int(len(blockers)),
            "focus_blocker_count": int(len(focus_blockers)),
            "vocal_residue_corr": round(vocal_residue_corr, 8),
            "vocal_to_instrumental_rms_ratio_db": round(vocal_ratio_db, 6),
            "presence_band_ratio_drift_db": round(presence_drift_db, 6),
            "air_band_ratio_drift_db": round(air_drift_db, 6),
        },
        "blockers": blockers,
    }


def focus_penalty(focus_metrics: dict[str, Any]) -> tuple[float, list[str]]:
    penalty = 0.0
    reasons: list[str] = []
    for label, focus_record in focus_metrics.items():
        if not isinstance(focus_record, dict):
            continue
        drift = focus_record.get("drift_vs_base")
        if not isinstance(drift, dict):
            continue

        added_layer_db = metric_as_float(drift, "added_layer_vs_base_rms_db", -90.0)
        air_drift_db = abs(metric_as_float(drift, "air_band_ratio_drift_db", 0.0))
        presence_drift_db = abs(metric_as_float(drift, "presence_band_ratio_drift_db", 0.0))

        if added_layer_db > -44.0:
            local_penalty = (added_layer_db + 44.0) * 0.10
            penalty += local_penalty
            reasons.append(f"{label}: focused added layer is loud ({added_layer_db:.2f} dB)")
        if air_drift_db > 0.60:
            local_penalty = (air_drift_db - 0.60) * 0.18
            penalty += local_penalty
            reasons.append(f"{label}: focused air drift is high ({air_drift_db:.2f} dB)")
        if presence_drift_db > 0.45:
            local_penalty = (presence_drift_db - 0.45) * 0.16
            penalty += local_penalty
            reasons.append(
                f"{label}: focused presence drift is high ({presence_drift_db:.2f} dB)"
            )

    return penalty, reasons


def score_candidate(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    output_dir: Path,
    profile: CandidateProfile,
    feedback_adjustments: dict[str, float],
) -> CandidateScore:
    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Candidate {profile.name} manifest has no metrics object")

    focus_metrics = manifest.get("focus_metrics", {})
    if not isinstance(focus_metrics, dict):
        focus_metrics = {}

    rms_drift_db = abs(metric_as_float(metrics, "rms_drift_db_vs_base"))
    presence_drift_db = metric_as_float(metrics, "presence_band_ratio_drift_db")
    air_drift_db = metric_as_float(metrics, "air_band_ratio_drift_db")
    added_layer_db = metric_as_float(metrics, "added_layer_vs_base_rms_db", -90.0)
    kick_attack_drift_db = abs(metric_as_float(metrics, "kick_attack_band_ratio_drift_db"))
    kick_punch_drift_db = abs(metric_as_float(metrics, "kick_punch_band_ratio_drift_db"))
    minus_release_gate = build_minus_release_gate(metrics, focus_metrics)

    score = profile.prior_score + feedback_adjustments.get(profile.name, 0.0)
    reasons = [f"profile prior {profile.prior_score:+.3f}"]

    if profile.name in feedback_adjustments:
        reasons.append(f"feedback adjustment {feedback_adjustments[profile.name]:+.3f}")

    presence_bonus = min(max(presence_drift_db, 0.0), 0.18) * 1.20
    air_bonus = min(max(air_drift_db, 0.0), 0.30) * 0.80
    score += presence_bonus + air_bonus
    reasons.append(f"presence restoration bonus {presence_bonus:+.3f}")
    reasons.append(f"air restoration bonus {air_bonus:+.3f}")

    if added_layer_db > -48.0:
        penalty = (added_layer_db + 48.0) * 0.13
        score -= penalty
        reasons.append(f"synthetic layer loudness penalty -{penalty:.3f}")
    elif added_layer_db < -66.0 and not profile.pure_base:
        penalty = (-66.0 - added_layer_db) * 0.025
        score -= penalty
        reasons.append(f"too-dry layer penalty -{penalty:.3f}")
    elif profile.pure_base:
        reasons.append("pure base profile intentionally has no repair layer")
    else:
        reasons.append("added layer is inside conservative loudness band")

    rms_penalty = rms_drift_db * 4.0
    kick_penalty = (kick_attack_drift_db + kick_punch_drift_db) * 0.35
    score -= rms_penalty + kick_penalty
    reasons.append(f"rms drift penalty -{rms_penalty:.3f}")
    reasons.append(f"kick drift penalty -{kick_penalty:.3f}")

    for blocker in minus_release_gate["blockers"]:
        label = blocker.get("label")
        if label == "vocal_residue_in_minus":
            corr_excess = max(
                0.0,
                float(blocker["vocal_residue_corr"]) - MINUS_VOCAL_RESIDUE_CORR_THRESHOLD,
            )
            ratio_excess = max(
                0.0,
                float(blocker["vocal_to_instrumental_rms_ratio_db"]) - MINUS_VOCAL_RESIDUE_RMS_RATIO_THRESHOLD_DB,
            )
            penalty = corr_excess * 3.2 + ratio_excess * 0.08
            score -= penalty
            reasons.append(f"vocal residue in minus penalty -{penalty:.3f}")
        elif label == "instrumental_presence_dip":
            penalty = (MINUS_PRESENCE_DIP_THRESHOLD_DB - float(blocker["presence_band_ratio_drift_db"])) * 0.90
            score -= penalty
            reasons.append(f"instrumental presence dip penalty -{penalty:.3f}")
        elif label == "instrumental_air_dip":
            penalty = (MINUS_AIR_DIP_THRESHOLD_DB - float(blocker["air_band_ratio_drift_db"])) * 0.65
            score -= penalty
            reasons.append(f"instrumental air dip penalty -{penalty:.3f}")
        elif blocker.get("reason") == "focused_window_instrumental_dip":
            presence_dip = max(
                0.0,
                MINUS_PRESENCE_DIP_THRESHOLD_DB - float(blocker["presence_band_ratio_drift_db"]),
            )
            air_dip = max(
                0.0,
                MINUS_AIR_DIP_THRESHOLD_DB - float(blocker["air_band_ratio_drift_db"]),
            )
            penalty = presence_dip * 0.70 + air_dip * 0.50
            score -= penalty
            reasons.append(f"focused instrumental dip penalty -{penalty:.3f}")

    local_focus_penalty, focus_reasons = focus_penalty(focus_metrics)
    score -= local_focus_penalty
    reasons.extend(focus_reasons)
    if local_focus_penalty:
        reasons.append(f"focus-window penalty -{local_focus_penalty:.3f}")

    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
    repaired_artifact = artifacts.get("repaired_minus_flac", {})
    if not isinstance(repaired_artifact, dict) or not repaired_artifact.get("path"):
        raise ValueError(f"Candidate {profile.name} manifest has no repaired_minus_flac path")

    return CandidateScore(
        profile_name=profile.name,
        score=round(score, 6),
        reasons=reasons,
        metrics=metrics,
        focus_metrics=focus_metrics,
        manifest_path=str(manifest_path),
        output_dir=str(output_dir),
        listen_dir=manifest.get("listen_dir"),
    )


def _run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def select_profiles(
    requested_profiles: list[str] | None,
    available_profiles: tuple[CandidateProfile, ...],
) -> list[CandidateProfile]:
    if not requested_profiles:
        return list(available_profiles)

    by_name = {profile.name: profile for profile in available_profiles}
    unknown_profiles = sorted(set(requested_profiles) - set(by_name))
    if unknown_profiles:
        names = ", ".join(unknown_profiles)
        raise ValueError(f"Unknown adaptive profile(s): {names}")
    return [by_name[name] for name in requested_profiles]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def copy_selected_artifacts(
    *,
    selected_manifest: dict[str, Any],
    output_dir: Path,
    listen_dir: Path | None,
    release_gate: dict[str, Any],
) -> dict[str, Any]:
    artifacts = selected_manifest["artifacts"]
    repaired_path = Path(artifacts["repaired_minus_flac"]["path"])
    selected_minus_path = output_dir / "02_adaptive_selected_minus.flac"
    shutil.copyfile(repaired_path, selected_minus_path)

    delivery_allowed = release_gate.get("release_status") == "release_ready"
    copied_paths = {
        "selected_minus_flac": str(selected_minus_path),
        "delivery_status": "accepted" if delivery_allowed else "blocked",
        "delivery_allowed": delivery_allowed,
        "presentation_status": (
            "accepted_for_listen_pack" if delivery_allowed else "blocked_from_listen_pack"
        ),
        "release_status": release_gate.get("release_status", "diagnostic_only"),
    }
    if listen_dir is not None and delivery_allowed:
        listen_dir.mkdir(parents=True, exist_ok=True)
        listen_minus_path = listen_dir / "02_adaptive_selected_minus.flac"
        shutil.copyfile(repaired_path, listen_minus_path)
        copied_paths["listen_selected_minus_flac"] = str(listen_minus_path)

    return copied_paths


def run_adaptive_repair(args: argparse.Namespace) -> dict[str, Any]:
    raw_audio_path = Path(args.raw_audio_path)
    base_minus_path = Path(args.base_minus_path)
    output_dir = Path(args.output_dir)
    listen_dir = Path(args.listen_dir) if args.listen_dir else None
    repair_script = Path(args.repair_script)
    feedback_store_path = Path(args.feedback_store) if args.feedback_store else None

    if not raw_audio_path.exists():
        raise FileNotFoundError(raw_audio_path)
    if not base_minus_path.exists():
        raise FileNotFoundError(base_minus_path)
    if not repair_script.exists():
        raise FileNotFoundError(repair_script)

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_root_dir = output_dir / "candidates"
    candidate_root_dir.mkdir(parents=True, exist_ok=True)

    profiles = select_profiles(args.profile, default_candidate_profiles())
    feedback_adjustments = load_feedback_adjustments(feedback_store_path)

    candidate_scores: list[CandidateScore] = []
    candidate_manifests: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        candidate_output_dir = candidate_root_dir / profile.name
        candidate_listen_dir = listen_dir / profile.name if listen_dir is not None else None
        command = build_repair_command(
            python_executable=args.python_executable,
            repair_script=repair_script,
            raw_audio_path=raw_audio_path,
            base_minus_path=base_minus_path,
            output_dir=candidate_output_dir,
            listen_dir=candidate_listen_dir,
            listen_label=f"02_{profile.name}_minus.flac",
            focus_windows=args.focus_window,
            profile=profile,
        )
        _run_command(command)

        manifest_path = candidate_output_dir / "stem_manifest.json"
        manifest = load_json(manifest_path)
        candidate_manifests[profile.name] = manifest
        candidate_scores.append(
            score_candidate(
                manifest=manifest,
                manifest_path=manifest_path,
                output_dir=candidate_output_dir,
                profile=profile,
                feedback_adjustments=feedback_adjustments,
            )
        )

    selected_score = max(candidate_scores, key=lambda candidate: candidate.score)
    selected_manifest = candidate_manifests[selected_score.profile_name]
    selected_release_gate = build_minus_release_gate(selected_score.metrics, selected_score.focus_metrics)
    copied_artifacts = copy_selected_artifacts(
        selected_manifest=selected_manifest,
        output_dir=output_dir,
        listen_dir=listen_dir,
        release_gate=selected_release_gate,
    )
    delivery_allowed = copied_artifacts["delivery_allowed"]

    adaptive_manifest = {
        "schema_version": 1,
        "mode": "adaptive_minus_repair_controller",
        "raw_audio_path": str(raw_audio_path),
        "base_minus_path": str(base_minus_path),
        "output_dir": str(output_dir),
        "listen_dir": str(listen_dir) if listen_dir is not None else None,
        "repair_script": str(repair_script),
        "feedback_store_path": str(feedback_store_path) if feedback_store_path else None,
        "feedback_adjustments": feedback_adjustments,
        "candidate_profiles": [
            {
                "name": profile.name,
                "intent": profile.intent,
                "pure_base": profile.pure_base,
                "ambience_strength": profile.ambience_strength,
                "ambience_max_gain_db": profile.ambience_max_gain_db,
                "ambience_percentile_window_sec": profile.ambience_percentile_window_sec,
                "texture_strength": profile.texture_strength,
                "texture_percentile_window_sec": profile.texture_percentile_window_sec,
                "texture_max_floor_ratio": profile.texture_max_floor_ratio,
                "enable_raw_kick_repair": profile.enable_raw_kick_repair,
                "prior_score": profile.prior_score,
            }
            for profile in profiles
        ],
        "candidates": [
            {
                "profile_name": candidate.profile_name,
                "score": candidate.score,
                "reasons": candidate.reasons,
                "metrics": candidate.metrics,
                "focus_metrics": candidate.focus_metrics,
                "manifest_path": candidate.manifest_path,
                "output_dir": candidate.output_dir,
                "listen_dir": candidate.listen_dir,
                "minus_release_gate": build_minus_release_gate(candidate.metrics, candidate.focus_metrics),
            }
            for candidate in sorted(
                candidate_scores,
                key=lambda candidate: candidate.score,
                reverse=True,
            )
        ],
        "selected_profile": selected_score.profile_name,
        "selected_score": selected_score.score,
        "selected_manifest_path": selected_score.manifest_path,
        "selected_minus_release_gate": selected_release_gate,
        "release_status": selected_release_gate["release_status"],
        "delivery_status": "accepted" if delivery_allowed else "blocked",
        "delivery_allowed": delivery_allowed,
        "presentation_status": (
            "accepted_for_listen_pack" if delivery_allowed else "blocked_from_listen_pack"
        ),
        "selected_artifacts": copied_artifacts,
        "learning_contract": {
            "type": "feedback_weighted_profile_selection",
            "feedback_record_schema": {
                "profile": "selected profile name",
                "rating": "accept|reject|good|bad|better|worse",
                "score": "optional numeric value from -1.0 to 1.0",
                "artifact_noise": "optional boolean",
                "too_dry": "optional boolean",
            },
            "limits": [
                "This controller adapts profile selection from persisted feedback.",
                "It does not train or fine-tune the neural separation model.",
                "It does not claim SDR/SI-SDR without isolated ground truth stems.",
            ],
        },
    }

    write_json(output_dir / "adaptive_manifest.json", adaptive_manifest)
    return adaptive_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adaptive candidate search over flexible minus repair profiles.",
    )
    parser.add_argument("raw_audio_path", help="Full song audio path.")
    parser.add_argument("base_minus_path", help="Base instrumental reference path.")
    parser.add_argument("output_dir", help="Directory for adaptive repair artifacts.")
    parser.add_argument("--listen-dir", default=None, help="Optional listen pack directory.")
    parser.add_argument(
        "--profile",
        action="append",
        choices=[profile.name for profile in default_candidate_profiles()],
        help="Candidate profile to run. Repeat to limit the search space.",
    )
    parser.add_argument(
        "--feedback-store",
        default=None,
        help="Optional JSONL feedback store used to adjust future profile selection.",
    )
    parser.add_argument(
        "--focus-window",
        action="append",
        default=[],
        help="Forwarded as label:start:end to candidate repair runs.",
    )
    parser.add_argument(
        "--repair-script",
        default="/app/tools/apply_flexible_minus_repair.py",
        help="Path to apply_flexible_minus_repair.py inside the runtime.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to run the repair script.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="adaptive_minus_repair.py",
        entrypoint_path="/app/tools/adaptive_minus_repair.py",
    )
    args = parse_args(argv)
    adaptive_manifest = run_adaptive_repair(args)
    print(
        json.dumps(
            {
                "manifest_path": str(Path(args.output_dir) / "adaptive_manifest.json"),
                "selected_profile": adaptive_manifest["selected_profile"],
                "selected_score": adaptive_manifest["selected_score"],
                "selected_artifacts": adaptive_manifest["selected_artifacts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
