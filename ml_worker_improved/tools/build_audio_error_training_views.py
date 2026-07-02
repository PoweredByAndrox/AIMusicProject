#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.audit_audio_error_dataset_provenance import is_real_value, is_synthetic_candidate_id, stable_unique  # noqa: E402
from tools.train_audio_error_critic import (  # noqa: E402
    ACCEPT_DECISIONS,
    PAIR_ACCEPT_STATUSES,
    REPAIRED_MANIFEST,
    REPAIRED_MAPS_INDEX,
    REPAIRED_ROWS,
    REPAIR_REPORT,
    SAFETY_FLAGS,
    SCALAR_FEATURE_COLUMNS,
    as_float,
    as_list,
    feature_dict,
    has_scalar_features,
)


VIEW_FILES = {
    "minus_critic": "minus_critic_training_view.jsonl",
    "vocal_critic": "vocal_critic_training_view.jsonl",
    "pair_critic": "pair_critic_training_view.jsonl",
    "overall_accept": "overall_accept_training_view.jsonl",
}
MANIFEST_FILE = "audio_error_training_views_manifest.json"
SUMMARY_FILE = "audio_error_training_views_summary.md"
OWNED_OUTPUT_FILES = (MANIFEST_FILE, SUMMARY_FILE, *VIEW_FILES.values())
ACCEPTED_PREVIEW_SOURCE_DIR = "accepted_preview_sources"
RALPH_SOURCE_DIR = "ralph_manifest_sources"
OWNED_OUTPUT_DIRS = (ACCEPTED_PREVIEW_SOURCE_DIR, RALPH_SOURCE_DIR)

STEM_JUDGE_FILES = {
    "minus": ("minus_windowed_judge.jsonl", "minus_judge.jsonl", "no_reference_minus_windowed_judge.jsonl"),
    "vocal": ("vocal_windowed_judge.jsonl", "vocal_judge.jsonl", "no_reference_vocal_windowed_judge.jsonl"),
}
REJECT_DECISION_PREFIXES = ("blocked", "reject", "failed", "needs_", "select_alternative")
VIEW_READINESS_RULES = {
    "minus_critic": {"min_safe": 100, "min_positive": 20, "min_negative": 20},
    "vocal_critic": {"min_safe": 100, "min_positive": 20, "min_negative": 20},
    "pair_critic": {"min_safe": 50, "min_positive": 1, "min_negative": 1},
    "overall_accept": {"min_safe": 100, "min_positive": 1, "min_negative": 1},
}
RUN_MARKER_FILES = ("candidate_pool.jsonl", "minus_windowed_judge.jsonl", "vocal_windowed_judge.jsonl")
ACCEPTED_PREVIEW_MINUS_JUDGE_FILE = "no_reference_minus_windowed_judge.jsonl"
ACOUSTIC_SCALAR_FEATURE_COLUMNS = tuple(name for name in SCALAR_FEATURE_COLUMNS if not name.startswith("window_"))
PROJECT_PATH_MOUNT_PREFIXES = (
    ("D:/AIMusicProject/artifacts", "AIMUSIC_ARTIFACTS_MOUNT", "/host_artifacts"),
    ("/d/AIMusicProject/artifacts", "AIMUSIC_ARTIFACTS_MOUNT", "/host_artifacts"),
    ("/Users/andrejviluha/AIMusicProject_win/artifacts", "AIMUSIC_ARTIFACTS_MOUNT", "/host_artifacts"),
    ("D:/AIMusicProject/testing_stuff_music", "AIMUSIC_TESTING_STUFF_MOUNT", "/host_testing_stuff_music"),
    ("/d/AIMusicProject/testing_stuff_music", "AIMUSIC_TESTING_STUFF_MOUNT", "/host_testing_stuff_music"),
    ("/Users/andrejviluha/AIMusicProject_win/testing_stuff_music", "AIMUSIC_TESTING_STUFF_MOUNT", "/host_testing_stuff_music"),
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OWNED_OUTPUT_FILES:
        path = output_dir / name
        if path.exists():
            path.unlink()
    for name in OWNED_OUTPUT_DIRS:
        path = output_dir / name
        if path.exists():
            shutil.rmtree(path)


def find_run_dirs(run_roots: Sequence[Path]) -> list[Path]:
    discovered: set[Path] = set()
    for root in run_roots:
        if not root.exists():
            continue
        if root.is_file():
            root = root.parent
        if any((root / marker).exists() for marker in RUN_MARKER_FILES):
            discovered.add(root)
        for marker in RUN_MARKER_FILES:
            for path in root.rglob(marker):
                discovered.add(path.parent)
    return sorted(discovered, key=lambda path: str(path))


def normalize_runtime_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = text.replace("\\", "/")
    normalized_lower = normalized.lower()
    for source_prefix, env_name, default_mount in PROJECT_PATH_MOUNT_PREFIXES:
        prefix = source_prefix.rstrip("/")
        prefix_lower = prefix.lower()
        if normalized_lower == prefix_lower:
            return os.environ.get(env_name, default_mount).rstrip("/")
        if normalized_lower.startswith(prefix_lower + "/"):
            mount = os.environ.get(env_name, default_mount).rstrip("/")
            return mount + normalized[len(prefix) :]
    return text


def row_path(row: dict[str, Any], names: Sequence[str]) -> str:
    for name in names:
        value = row.get(name)
        if value:
            return normalize_runtime_path(value)
    return ""


def runtime_candidate_paths(candidate: dict[str, Any]) -> dict[str, str]:
    return {
        "mix_path": row_path(candidate, ("mix_path", "input_audio_path", "input_audio", "source_mix_path")),
        "vocal_candidate_path": row_path(
            candidate,
            ("vocal_candidate_path", "vocal_path", "selected_vocal_path", "normalized_vocal_path", "raw_vocal_path"),
        ),
        "minus_candidate_path": row_path(
            candidate,
            (
                "minus_candidate_path",
                "minus_path",
                "instrumental_path",
                "selected_minus_path",
                "normalized_minus_path",
                "normalized_instrumental_path",
                "raw_minus_path",
            ),
        ),
    }


def load_runtime_candidates(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    mapping: dict[tuple[str, str], dict[str, Any]] = {}
    for row in iter_jsonl(run_dir / "candidate_pool.jsonl"):
        track_id = str(row.get("track_id") or "")
        candidate_id = str(row.get("candidate_id") or "")
        if track_id and candidate_id:
            mapping[(track_id, candidate_id)] = row
    return mapping


def load_runtime_pair_status(run_dir: Path) -> dict[tuple[str, str], tuple[str, str]]:
    mapping: dict[tuple[str, str], tuple[str, str]] = {}
    path = run_dir / "pair_compatibility.jsonl"
    for row in iter_jsonl(path):
        track_id = str(row.get("track_id") or "")
        ids = stable_unique(
            str(value)
            for value in (
                row.get("candidate_id"),
                row.get("selected_candidate_id"),
                row.get("selected_vocal_candidate_id"),
                row.get("selected_minus_candidate_id"),
            )
            if str(value or "")
        )
        status = str(row.get("pair_compatibility_status") or row.get("pair_decision") or row.get("compatibility_status") or row.get("decision") or "")
        for candidate_id in ids:
            if track_id and candidate_id:
                mapping[(track_id, candidate_id)] = (status, path.name)
    return mapping


def window_decision_for(stem_type: str, judge_row: dict[str, Any], window: dict[str, Any]) -> str:
    for key in (f"{stem_type}_decision", "decision", "judge_decision"):
        value = window.get(key)
        if value:
            return str(value)
    for key in (f"{stem_type}_decision", "decision", "judge_decision"):
        value = judge_row.get(key)
        if value:
            return str(value)
    return ""


def window_blockers_for(stem_type: str, window: dict[str, Any]) -> list[str]:
    return stable_unique(value for key in ("blockers", f"{stem_type}_blockers") for value in as_list(window.get(key)))


def clip01(value: Any) -> float:
    return max(0.0, min(1.0, as_float(value)))


def optional_score(window: dict[str, Any], key: str) -> float | None:
    if key not in window:
        return None
    return clip01(window.get(key))


def inverse_score(window: dict[str, Any], key: str) -> float:
    score = optional_score(window, key)
    return 0.0 if score is None else 1.0 - score


def max_score(window: dict[str, Any], keys: Sequence[str]) -> float:
    values = [score for key in keys if (score := optional_score(window, key)) is not None]
    return max(values) if values else 0.0


def phase_collapse_from_no_reference_scores(window: dict[str, Any]) -> float:
    scores = [
        score
        for key in ("phase_consistency_score", "instrumental_phase_consistency_score")
        if (score := optional_score(window, key)) is not None
    ]
    return 0.0 if not scores else 1.0 - min(scores)


def no_reference_runtime_scalar_features(stem_type: str, window: dict[str, Any]) -> dict[str, float]:
    del stem_type
    mixture_error = inverse_score(window, "mixture_consistency_score")
    phase_collapse = phase_collapse_from_no_reference_scores(window)
    vocal_bleed = clip01(window.get("vocal_leakage_in_minus_score"))
    instrumental_bleed = clip01(window.get("instrumental_leakage_in_vocal_score"))
    artifact_noise = max_score(window, ("robotic_artifact_score", "silence_hallucination_score", "thin_vocal_score"))
    transient_damage = clip01(window.get("transient_damage_score"))
    missing_detail = max(mixture_error, inverse_score(window, "instrumental_phase_consistency_score"), clip01(window.get("thin_vocal_score")))
    residual_energy = max(mixture_error, phase_collapse, vocal_bleed, instrumental_bleed, artifact_noise, transient_damage)
    return {
        "residual_energy": residual_energy,
        "mixture_consistency_error": mixture_error,
        "phase_collapse_score": phase_collapse,
        "stereo_width_score": 0.0,
        "vocal_bleed_proxy_score": vocal_bleed,
        "instrumental_bleed_proxy_score": instrumental_bleed,
        "missing_detail_score": missing_detail,
        "transient_damage_score": transient_damage,
        "artifact_noise_score": artifact_noise,
    }


def runtime_scalar_features(stem_type: str, window: dict[str, Any]) -> dict[str, float]:
    no_reference_features = no_reference_runtime_scalar_features(stem_type, window)
    return {
        name: as_float(window.get(name)) if name in window else no_reference_features[name]
        for name in ACOUSTIC_SCALAR_FEATURE_COLUMNS
    }


def ralph_drift_damage(window: dict[str, Any]) -> float:
    return max(
        clip01(max(0.0, -as_float(window.get("rms_drift_db_vs_base")))),
        clip01(max(0.0, -as_float(window.get("presence_band_ratio_drift_db")))),
        clip01(max(0.0, -as_float(window.get("air_band_ratio_drift_db")))),
    )


def ralph_scalar_features(window: dict[str, Any]) -> dict[str, float]:
    vocal_bleed = clip01(abs(as_float(window.get("vocal_residue_corr"))))
    missing_detail = ralph_drift_damage(window)
    residual_energy = max(vocal_bleed, missing_detail)
    return {
        "residual_energy": residual_energy,
        "mixture_consistency_error": vocal_bleed,
        "phase_collapse_score": 0.0,
        "stereo_width_score": 0.0,
        "vocal_bleed_proxy_score": vocal_bleed,
        "instrumental_bleed_proxy_score": 0.0,
        "missing_detail_score": missing_detail,
        "transient_damage_score": 0.0,
        "artifact_noise_score": 0.0,
    }


def manifest_ref_path(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if isinstance(value, dict):
        return normalize_runtime_path(value.get("path"))
    return normalize_runtime_path(value)


def accepted_listen_preview_manifest(payload: dict[str, Any]) -> bool:
    return (
        payload.get("delivery_allowed") is True
        and str(payload.get("delivery_status") or "") == "accepted"
        and str(payload.get("presentation_status") or "") == "accepted_for_listen_pack"
        and str(payload.get("stem_type") or payload.get("score_mode") or "") == "minus"
    )


def listen_preview_scalar_features(
    *,
    segment: dict[str, Any],
    listen_manifest: dict[str, Any],
    accepted_row: dict[str, Any],
    result_row: dict[str, Any],
) -> dict[str, float]:
    metrics = segment.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    severities = segment.get("issue_severities")
    if not isinstance(severities, dict):
        severities = {}
    gate = listen_manifest.get("minus_musical_detail_preservation_gate")
    gate_metrics = gate.get("metrics") if isinstance(gate, dict) and isinstance(gate.get("metrics"), dict) else {}
    accepted_gate = accepted_row.get("minus_musical_detail_preservation_gate")
    accepted_gate_metrics = (
        accepted_gate.get("metrics")
        if isinstance(accepted_gate, dict) and isinstance(accepted_gate.get("metrics"), dict)
        else {}
    )

    mixture_consistency = clip01(
        result_row.get("mixture_consistency_score")
        or accepted_gate_metrics.get("mixture_consistency_score")
        or gate_metrics.get("mixture_consistency_score")
    )
    if mixture_consistency <= 0.0:
        mixture_consistency = clip01(gate_metrics.get("mixture_consistency_score"))
    phase_stability = clip01(result_row.get("instrumental_phase_consistency_score") or gate_metrics.get("phase_stereo_stability"))
    spectral_balance = clip01(gate_metrics.get("spectral_balance"))
    instrumental_integrity = clip01(gate_metrics.get("instrumental_integrity"))
    source_corr_damage = 1.0 - clip01(metrics.get("source_candidate_corr_abs"))
    source_delta = clip01(metrics.get("source_candidate_delta_rms_ratio"))
    vocal_corr = clip01(metrics.get("vocal_stem_corr_abs"))
    vocal_projection = clip01(severities.get("vocal_projection"))
    vocal_bleed = max(clip01(result_row.get("vocal_leakage_in_minus_score")), vocal_corr, vocal_projection)
    instrumental_bleed = clip01(result_row.get("instrumental_leakage_in_vocal_score"))
    artifact_noise = max_score(
        result_row,
        ("robotic_artifact_score", "silence_hallucination_score", "thin_vocal_score"),
    )
    transient_damage = max(clip01(result_row.get("transient_damage_score")), clip01(gate_metrics.get("transient_damage_score")))
    missing_detail = max(
        clip01(severities.get("air_loss")),
        clip01(severities.get("presence_loss")),
        clip01(severities.get("low_rms_ratio")),
        1.0 - spectral_balance if spectral_balance > 0.0 else 0.0,
        1.0 - instrumental_integrity if instrumental_integrity > 0.0 else 0.0,
        source_delta * 0.5,
    )
    mixture_error = max(1.0 - mixture_consistency if mixture_consistency > 0.0 else 0.0, source_corr_damage)
    phase_collapse = 1.0 - phase_stability if phase_stability > 0.0 else 0.0
    residual_energy = max(
        mixture_error,
        phase_collapse,
        vocal_bleed,
        instrumental_bleed,
        missing_detail,
        transient_damage,
        artifact_noise,
    )
    return {
        "residual_energy": residual_energy,
        "mixture_consistency_error": mixture_error,
        "phase_collapse_score": phase_collapse,
        "stereo_width_score": phase_stability,
        "vocal_bleed_proxy_score": vocal_bleed,
        "instrumental_bleed_proxy_score": instrumental_bleed,
        "missing_detail_score": missing_detail,
        "transient_damage_score": transient_damage,
        "artifact_noise_score": artifact_noise,
    }


def segment_window_bounds(segment: dict[str, Any]) -> tuple[float, float]:
    window = segment.get("window_sec")
    if isinstance(window, list) and len(window) >= 2:
        return round(as_float(window[0]), 6), round(as_float(window[1]), 6)
    return window_bounds(segment, segment)


def load_result_judge_row(listen_manifest: dict[str, Any]) -> dict[str, Any]:
    path = manifest_ref_path(listen_manifest, "source_result_judge_jsonl")
    if not path:
        return {}
    for row in iter_jsonl(Path(path)):
        if str(row.get("candidate_id") or "") == str(listen_manifest.get("candidate_id") or ""):
            return row
    return {}


def accepted_listen_preview_rows(listen_preview_manifests: Sequence[Path], output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_root = output_dir / ACCEPTED_PREVIEW_SOURCE_DIR
    for manifest_index, manifest_path in enumerate(listen_preview_manifests, start=1):
        listen_manifest = read_json(manifest_path)
        if not accepted_listen_preview_manifest(listen_manifest):
            continue
        accepted_rows = listen_manifest.get("accepted_listen_preview_rows")
        accepted_row = accepted_rows[0] if isinstance(accepted_rows, list) and accepted_rows and isinstance(accepted_rows[0], dict) else {}
        candidate_id = str(listen_manifest.get("candidate_id") or accepted_row.get("candidate_id") or "")
        self_judge_path = manifest_ref_path(listen_manifest, "source_self_judge_manifest")
        self_judge_manifest = read_json(Path(self_judge_path)) if self_judge_path else {}
        segments = self_judge_manifest.get("segments")
        if not isinstance(segments, list):
            continue
        result_row = load_result_judge_row(listen_manifest)
        source_dir = source_root / f"lp{manifest_index:03d}"
        source_judge_file = source_dir / ACCEPTED_PREVIEW_MINUS_JUDGE_FILE
        judge_windows: list[dict[str, Any]] = []
        for segment_index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                continue
            start, end = segment_window_bounds(segment)
            if end <= start:
                continue
            scalar_features = listen_preview_scalar_features(
                segment=segment,
                listen_manifest=listen_manifest,
                accepted_row=accepted_row,
                result_row=result_row,
            )
            blockers = stable_unique(as_list(segment.get("issues")) + as_list(accepted_row.get("minus_blockers")) + as_list(listen_manifest.get("minus_blockers")))
            decision = str(accepted_row.get("minus_decision") or listen_manifest.get("minus_decision") or result_row.get("minus_decision") or "accept_candidate")
            judge_window = {
                "start_sec": start,
                "end_sec": end,
                "minus_decision": decision,
                "minus_blockers": blockers,
                **scalar_features,
            }
            judge_windows.append(judge_window)
            window_id = "__".join(
                [
                    str(listen_manifest.get("track_id") or "missing_track_id"),
                    candidate_id or "missing_candidate_id",
                    "minus",
                    f"alp{manifest_index:03d}",
                    f"{segment_index:04d}",
                ]
            )
            rows.append(
                {
                    "dataset_id": "accepted_listen_preview",
                    "track_id": str(listen_manifest.get("track_id") or accepted_row.get("track_id") or ""),
                    "run_id": str(listen_manifest.get("manifest_type") or "no_reference_minus_listen_preview"),
                    "window_id": window_id,
                    "window_start_sec": start,
                    "window_end_sec": end,
                    "mix_path": normalize_runtime_path(listen_manifest.get("mix_path")),
                    "vocal_candidate_path": normalize_runtime_path(listen_manifest.get("vocal_path")),
                    "minus_candidate_path": normalize_runtime_path(listen_manifest.get("selected_minus_path") or listen_manifest.get("minus_path") or listen_manifest.get("instrumental_path")),
                    "candidate_id": candidate_id,
                    "route_id": str(listen_manifest.get("judge_mode") or "no_reference_minus_listen_preview"),
                    "stem_type": "minus",
                    "blockers": blockers,
                    "judge_decision": decision,
                    "pair_compatibility_status": "",
                    **scalar_features,
                    "error_map_paths": {},
                    "train_split": "train",
                    "safe_for_training": False,
                    "unsafe_reason": "accepted_listen_preview_source_row_not_repaired_dataset",
                    "target_views": ["minus_critic"],
                    "source_run_dir": str(source_dir),
                    "source_candidate_family": "accepted_listen_preview_minus",
                    "candidate_family": "accepted_listen_preview_minus",
                    "source_judge_file": ACCEPTED_PREVIEW_MINUS_JUDGE_FILE,
                    "source_pair_file": "",
                    "source_selected_file": manifest_ref_path(listen_manifest, "source_selector_jsonl"),
                    "selected_candidate_evidence": True,
                    "repair_source_files": stable_unique(
                        [
                            str(manifest_path),
                            self_judge_path,
                            manifest_ref_path(listen_manifest, "source_result_judge_jsonl"),
                            manifest_ref_path(listen_manifest, "source_selector_jsonl"),
                        ]
                    ),
                    **SAFETY_FLAGS,
                }
            )
        if judge_windows:
            write_jsonl(
                source_judge_file,
                [
                    {
                        "track_id": str(listen_manifest.get("track_id") or accepted_row.get("track_id") or ""),
                        "candidate_id": candidate_id,
                        "window_metrics": judge_windows,
                    }
                ],
            )
    return rows


def ralph_input_paths(manifest: dict[str, Any]) -> dict[str, str]:
    inputs = manifest.get("inputs") if isinstance(manifest.get("inputs"), dict) else {}
    normalized = inputs.get("normalized_inputs") if isinstance(inputs.get("normalized_inputs"), dict) else {}
    merged = {**inputs, **normalized}
    return {
        "mix_path": normalize_runtime_path(merged.get("raw_audio_path")),
        "vocal_candidate_path": normalize_runtime_path(merged.get("primary_vocals_path") or merged.get("secondary_vocals_path")),
        "base_minus_path": normalize_runtime_path(merged.get("base_minus_path")),
        "guide_minus_path": normalize_runtime_path(merged.get("guide_minus_path")),
    }


def ralph_candidate_windows(manifest: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    del manifest
    evaluation = candidate.get("evaluation") if isinstance(candidate.get("evaluation"), dict) else {}
    focus = evaluation.get("focus_evaluation") if isinstance(evaluation.get("focus_evaluation"), dict) else {}
    summary = evaluation.get("summary") if isinstance(evaluation.get("summary"), dict) else {}
    for value in (
        focus.get("windows"),
        summary.get("top_vocal_residue_windows"),
    ):
        if isinstance(value, list) and value:
            return [window for window in value if isinstance(window, dict)]
    return []


def label_values(value: Any) -> list[str]:
    if isinstance(value, list):
        labels: list[str] = []
        for item in value:
            if isinstance(item, dict):
                labels.append(str(item.get("label") or ""))
            else:
                labels.append(str(item or ""))
        return stable_unique(labels)
    if isinstance(value, dict):
        return stable_unique([str(value.get("label") or "")])
    return stable_unique(as_list(value))


def ralph_blockers(evaluation: dict[str, Any]) -> list[str]:
    status = str(evaluation.get("status") or "").strip().lower()
    if status == "accepted":
        return []
    return label_values(evaluation.get("blockers"))


def ralph_decision(evaluation: dict[str, Any]) -> str:
    return "accept_candidate" if str(evaluation.get("status") or "").strip().lower() == "accepted" else "blocked_hard_case"


def ralph_window_evidence(window: dict[str, Any]) -> dict[str, Any]:
    return {
        "vocal_residue_corr": as_float(window.get("vocal_residue_corr")),
        "vocal_projection_ratio_db": as_float(window.get("vocal_projection_ratio_db")),
        "rms_drift_db_vs_base": as_float(window.get("rms_drift_db_vs_base")),
        "presence_band_ratio_drift_db": as_float(window.get("presence_band_ratio_drift_db")),
        "air_band_ratio_drift_db": as_float(window.get("air_band_ratio_drift_db")),
        "critic_decision": str(window.get("critic_decision") or ""),
        "predicted_blockers": label_values(window.get("predicted_blockers")),
        "focus_reason": str(window.get("focus_reason") or ""),
    }


def ralph_manifest_rows(ralph_manifests: Sequence[Path], output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_root = output_dir / RALPH_SOURCE_DIR
    for manifest_index, manifest_path in enumerate(ralph_manifests, start=1):
        manifest = read_json(manifest_path)
        inputs = ralph_input_paths(manifest)
        source_dir = source_root / f"rm{manifest_index:03d}"
        source_judge_file = source_dir / ACCEPTED_PREVIEW_MINUS_JUDGE_FILE
        source_judge_rows: list[dict[str, Any]] = []
        filters = manifest.get("focus_window_filters") if isinstance(manifest.get("focus_window_filters"), dict) else {}
        candidates = manifest.get("candidates")
        if not isinstance(candidates, list):
            continue
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            evaluation = candidate.get("evaluation") if isinstance(candidate.get("evaluation"), dict) else {}
            candidate_name = str(candidate.get("candidate_name") or evaluation.get("candidate_name") or f"ralph_candidate_{candidate_index:03d}")
            minus_path = normalize_runtime_path(candidate.get("artifact_path"))
            blockers = ralph_blockers(evaluation)
            decision = ralph_decision(evaluation)
            window_metrics: list[dict[str, Any]] = []
            for window_index, window in enumerate(ralph_candidate_windows(manifest, candidate)):
                start, end = window_bounds({}, window)
                if end <= start:
                    continue
                track_id = str(window.get("track_id") or filters.get("track_id") or "ralph_focus")
                scalar_features = ralph_scalar_features(window)
                evidence = ralph_window_evidence(window)
                judge_window = {
                    "track_id": track_id,
                    "start_sec": start,
                    "end_sec": end,
                    "minus_decision": decision,
                    "minus_blockers": blockers,
                    **scalar_features,
                    "ralph_window_evidence": evidence,
                }
                window_metrics.append(judge_window)
                rows.append(
                    {
                        "dataset_id": "ralph_manifest_hardcase",
                        "track_id": track_id,
                        "run_id": Path(str(manifest_path.parent)).name,
                        "window_id": "__".join(
                            [
                                track_id or "missing_track_id",
                                candidate_name or "missing_candidate_id",
                                "minus",
                                f"rm{manifest_index:03d}",
                                f"{candidate_index:03d}",
                                f"{window_index:04d}",
                            ]
                        ),
                        "window_start_sec": start,
                        "window_end_sec": end,
                        "mix_path": inputs["mix_path"],
                        "vocal_candidate_path": inputs["vocal_candidate_path"],
                        "minus_candidate_path": minus_path,
                        "candidate_id": candidate_name,
                        "route_id": candidate_name,
                        "stem_type": "minus",
                        "blockers": blockers,
                        "judge_decision": decision,
                        "pair_compatibility_status": "",
                        **scalar_features,
                        "error_map_paths": {},
                        "train_split": "train",
                        "safe_for_training": False,
                        "unsafe_reason": "ralph_manifest_score_only_source_row",
                        "target_views": ["minus_critic"],
                        "source_run_dir": str(source_dir),
                        "source_candidate_family": "ralph_minus_improvement",
                        "candidate_family": "ralph_minus_improvement",
                        "source_judge_file": ACCEPTED_PREVIEW_MINUS_JUDGE_FILE,
                        "source_pair_file": "",
                        "source_selected_file": normalize_runtime_path(manifest.get("focus_window_source")),
                        "selected_candidate_evidence": candidate_name == str(manifest.get("selected_candidate") or ""),
                        "hardcase": True,
                        "repair_source_files": stable_unique([str(manifest_path), normalize_runtime_path(manifest.get("focus_window_source"))]),
                        "source_evidence": {
                            "source": "ralph_minus_improvement_manifest",
                            "delivery_status": str(manifest.get("delivery_status") or ""),
                            "delivery_allowed": bool(manifest.get("delivery_allowed")),
                            "presentation_status": str(manifest.get("presentation_status") or ""),
                            "candidate_status": str(evaluation.get("status") or ""),
                            "focus_evaluation_used_for_delivery": bool(manifest.get("focus_evaluation_used_for_delivery")),
                            "ralph_window_evidence": evidence,
                        },
                        **SAFETY_FLAGS,
                    }
                )
            if window_metrics:
                source_judge_rows.append(
                    {
                        "track_id": str(window_metrics[0].get("track_id") or filters.get("track_id") or "ralph_focus"),
                        "candidate_id": candidate_name,
                        "window_metrics": window_metrics,
                    }
                )
        if source_judge_rows:
            write_jsonl(source_judge_file, source_judge_rows)
    return rows


def runtime_window_rows(run_roots: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in find_run_dirs(run_roots):
        candidates = load_runtime_candidates(run_dir)
        pair_status = load_runtime_pair_status(run_dir)
        for stem_type, filenames in STEM_JUDGE_FILES.items():
            for filename in filenames:
                judge_path = run_dir / filename
                if not judge_path.exists():
                    continue
                for judge_row in iter_jsonl(judge_path):
                    track_id = str(judge_row.get("track_id") or "")
                    candidate_id = str(judge_row.get("candidate_id") or judge_row.get("selected_candidate_id") or "")
                    candidate = candidates.get((track_id, candidate_id), {})
                    candidate = {**judge_row, **candidate} if candidate else dict(judge_row)
                    paths = runtime_candidate_paths(candidate)
                    windows = judge_row.get("window_metrics")
                    if not isinstance(windows, list):
                        continue
                    for window in windows:
                        if not isinstance(window, dict):
                            continue
                        start, end = window_bounds(judge_row, window)
                        status, pair_file = pair_status.get((track_id, candidate_id), ("", ""))
                        scalar_features = runtime_scalar_features(stem_type, window)
                        window_id = "__".join(
                            [
                                track_id or "missing_track_id",
                                candidate_id or "missing_candidate_id",
                                stem_type,
                                f"{start:.3f}-{end:.3f}",
                                stable_hash({"run": str(run_dir), "candidate": candidate_id, "stem": stem_type, "start": start, "end": end})[:10],
                            ]
                        )
                        rows.append(
                            {
                                "dataset_id": "runtime_harvest",
                                "track_id": track_id,
                                "run_id": str(judge_row.get("run_id") or candidate.get("run_id") or run_dir.name),
                                "window_id": window_id,
                                "window_start_sec": start,
                                "window_end_sec": end,
                                **paths,
                                "candidate_id": candidate_id,
                                "route_id": str(
                                    judge_row.get("route_id")
                                    or candidate.get("route_id")
                                    or candidate.get("minus_route_id")
                                    or candidate.get("model_id")
                                    or candidate.get("minus_model_id")
                                    or candidate.get("candidate_config_id")
                                    or ""
                                ),
                                "stem_type": stem_type,
                                "blockers": window_blockers_for(stem_type, window),
                                "judge_decision": window_decision_for(stem_type, judge_row, window),
                                "pair_compatibility_status": status,
                                **scalar_features,
                                "error_map_paths": {},
                                "train_split": "train",
                                "safe_for_training": False,
                                "unsafe_reason": "runtime_source_row_not_repaired_dataset",
                                "source_run_dir": str(run_dir),
                                "source_candidate_family": str(candidate.get("candidate_family") or candidate.get("route_family") or ""),
                                "source_judge_file": filename,
                                "source_pair_file": pair_file,
                                **SAFETY_FLAGS,
                            }
                        )
                break
    return rows


def path_exists(value: Any) -> bool:
    text = normalize_runtime_path(value)
    return bool(text and Path(text).exists())


def normalized_error_map_npz_path(row: dict[str, Any]) -> Path | None:
    paths = row.get("error_map_paths")
    if not isinstance(paths, dict):
        return None
    text = normalize_runtime_path(paths.get("npz"))
    if not text:
        return None
    return Path(text)


def row_with_normalized_error_map_paths(row: dict[str, Any]) -> dict[str, Any]:
    paths = row.get("error_map_paths")
    if not isinstance(paths, dict):
        return row
    normalized_npz = normalize_runtime_path(paths.get("npz"))
    if not normalized_npz or normalized_npz == str(paths.get("npz") or ""):
        return row
    normalized_row = dict(row)
    normalized_paths = dict(paths)
    normalized_paths["npz"] = normalized_npz
    normalized_row["error_map_paths"] = normalized_paths
    return normalized_row


def view_row_id(index: int, row: dict[str, Any]) -> str:
    for key in ("window_id", "row_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return f"source_row_{index:06d}"


def npz_text(row: dict[str, Any]) -> str:
    path = normalized_error_map_npz_path(row)
    return str(path) if path is not None else ""


def has_error_map_or_scalar_features(row: dict[str, Any]) -> bool:
    path = normalized_error_map_npz_path(row)
    if path is not None and path.exists():
        return True
    return has_scalar_features(row)


def has_acoustic_feature_evidence(row: dict[str, Any]) -> bool:
    path = normalized_error_map_npz_path(row)
    if path is not None and path.exists():
        return True
    for name in ACOUSTIC_SCALAR_FEATURE_COLUMNS:
        if name in row and abs(as_float(row.get(name))) > 1.0e-12:
            return True
    return False


def error_map_evidence_present(row: dict[str, Any]) -> bool:
    path = normalized_error_map_npz_path(row)
    return path is not None and path.exists()


def view_feature_dict(row: dict[str, Any], *, map_cache: dict[str, dict[str, float]]) -> dict[str, Any]:
    normalized_row = row_with_normalized_error_map_paths(row)
    features = feature_dict(normalized_row, map_cache=map_cache)
    features["acoustic_feature_evidence_present"] = 1.0 if has_acoustic_feature_evidence(row) else 0.0
    features["error_map_evidence_present"] = 1.0 if error_map_evidence_present(row) else 0.0
    return features


def valid_window(row: dict[str, Any]) -> bool:
    try:
        return float(row.get("window_end_sec") or 0.0) > float(row.get("window_start_sec") or 0.0)
    except (TypeError, ValueError):
        return False


def normalized_source_files(row: dict[str, Any]) -> list[str]:
    sources: list[str] = []
    for key in ("source_judge_file", "source_pair_file", "source_selected_file", "source_run_dir"):
        value = row.get(key)
        if is_real_value(value):
            sources.append(str(value))
    sources.extend(str(value) for value in as_list(row.get("repair_source_files")) if str(value))
    return stable_unique(sources)


def source_judge_file_matches(row: dict[str, Any], stem_type: str) -> bool:
    source = str(row.get("source_judge_file") or "").strip()
    if not source:
        return False
    source_name = Path(source).name
    return source_name in STEM_JUDGE_FILES.get(stem_type, ())


def source_judge_artifact_path(row: dict[str, Any], stem_type: str) -> Path | None:
    if not source_judge_file_matches(row, stem_type):
        return None
    run_dir = str(row.get("source_run_dir") or "").strip()
    if not run_dir:
        return None
    return Path(run_dir) / Path(str(row.get("source_judge_file") or "")).name


def judge_candidate_ids(row: dict[str, Any]) -> set[str]:
    keys = ("candidate_id", "selected_candidate_id", "source_candidate_id", "input_candidate_id")
    return {str(row.get(key)) for key in keys if is_real_value(row.get(key))}


def window_bounds(parent: dict[str, Any], window: dict[str, Any]) -> tuple[float, float]:
    start = as_float(window.get("start_sec"), as_float(parent.get("window_start_sec"), 0.0))
    if "end_sec" in window:
        end = as_float(window.get("end_sec"), start)
    else:
        duration = as_float(window.get("duration_sec"), as_float(parent.get("window_duration_sec"), 0.0))
        end = start + max(0.0, duration)
    return round(start, 6), round(end, 6)


def source_window_for(
    row: dict[str, Any],
    stem_type: str,
    *,
    source_cache: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any] | None, bool]:
    path = source_judge_artifact_path(row, stem_type)
    if path is None or not path.exists():
        return None, False
    cache_key = str(path)
    if cache_key not in source_cache:
        source_cache[cache_key] = list(iter_jsonl(path))
    track_id = str(row.get("track_id") or "")
    candidate_id = str(row.get("candidate_id") or "")
    expected_start = round(as_float(row.get("window_start_sec")), 6)
    expected_end = round(as_float(row.get("window_end_sec")), 6)
    for judge_row in source_cache[cache_key]:
        source_track = str(judge_row.get("track_id") or "")
        if track_id and source_track and source_track != track_id:
            continue
        if candidate_id not in judge_candidate_ids(judge_row):
            continue
        windows = judge_row.get("window_metrics")
        if not isinstance(windows, list):
            continue
        for window in windows:
            if not isinstance(window, dict):
                continue
            start, end = window_bounds(judge_row, window)
            if start == expected_start and end == expected_end:
                return window, True
    return None, True


def source_window_decision(stem_type: str, row: dict[str, Any], window: dict[str, Any] | None) -> str:
    if isinstance(window, dict):
        for key in (f"{stem_type}_decision", "decision", "judge_decision"):
            value = window.get(key)
            if value:
                return str(value)
    return str(row.get("judge_decision") or "")


def source_window_blockers(stem_type: str, row: dict[str, Any], window: dict[str, Any] | None) -> list[str]:
    if isinstance(window, dict):
        blockers = stable_unique(
            value
            for key in ("blockers", f"{stem_type}_blockers")
            for value in as_list(window.get(key))
        )
        if blockers:
            return blockers
    return as_list(row.get("blockers"))


def exact_window_evidence_reasons(
    row: dict[str, Any],
    stem_type: str,
    *,
    source_window_required: bool = False,
    source_window: dict[str, Any] | None = None,
) -> list[str]:
    reasons: list[str] = []
    if str(row.get("stem_type") or "") != stem_type:
        reasons.append(f"stem_type_not_{stem_type}")
    if not source_judge_file_matches(row, stem_type):
        reasons.append(f"missing_exact_{stem_type}_window_judge_evidence")
    elif source_window_required and source_window is None:
        reasons.append(f"missing_exact_{stem_type}_window_judge_evidence")
    if not is_real_value(row.get("judge_decision")):
        reasons.append("missing_window_judge_decision")
    if not valid_window(row):
        reasons.append("invalid_window_boundaries")
    return reasons


def common_reasons(row: dict[str, Any], *, stem_type: str | None = None) -> list[str]:
    reasons: list[str] = []
    candidate_id = row.get("candidate_id")
    route_id = row.get("route_id")
    if not is_real_value(candidate_id):
        reasons.append("missing_candidate_id")
    elif is_synthetic_candidate_id(candidate_id):
        reasons.append("synthetic_candidate_id")
    if str(route_id or "").strip().lower() == "unknown":
        reasons.append("route_id_unknown")
    elif not is_real_value(route_id):
        reasons.append("missing_route_id")
    if row.get("release_ready") is True:
        reasons.append("release_ready_true")
    if row.get("training_started") is True or row.get("separator_training_started") is True:
        reasons.append("training_started_true")
    if not path_exists(row.get("mix_path")):
        reasons.append("missing_audio_file:mix_path")
    if stem_type in (None, "vocal") and not path_exists(row.get("vocal_candidate_path")):
        reasons.append("missing_audio_file:vocal_candidate_path")
    if stem_type in (None, "minus") and not path_exists(row.get("minus_candidate_path")):
        reasons.append("missing_audio_file:minus_candidate_path")
    if not has_error_map_or_scalar_features(row):
        reasons.append("missing_error_maps_or_scalar_features")
    elif not has_acoustic_feature_evidence(row):
        reasons.append("missing_acoustic_feature_evidence")
    return stable_unique(reasons)


def stem_label(row: dict[str, Any]) -> int | None:
    decision = str(row.get("judge_decision") or "").strip()
    blockers = as_list(row.get("blockers"))
    if decision in ACCEPT_DECISIONS and not blockers:
        return 1
    if blockers or decision.startswith(REJECT_DECISION_PREFIXES):
        return 0
    return None


def pair_label(row: dict[str, Any]) -> int | None:
    status = str(row.get("pair_compatibility_status") or "").strip()
    if not status:
        return None
    return 1 if status in PAIR_ACCEPT_STATUSES else 0


def selected_candidate_evidence(row: dict[str, Any]) -> bool:
    if row.get("selected_candidate_evidence") is True:
        return True
    source = str(row.get("source_selected_file") or "").strip()
    return Path(source).name == "selected_candidates.jsonl"


def overall_label(row: dict[str, Any]) -> int | None:
    if not selected_candidate_evidence(row):
        return None
    label = stem_label(row)
    if label is None:
        return pair_label(row)
    return label


def stem_candidate_path(row: dict[str, Any], stem_type: str) -> str:
    key = "minus_candidate_path" if stem_type == "minus" else "vocal_candidate_path"
    return normalize_runtime_path(row.get(key))


def primary_candidate_path(row: dict[str, Any]) -> str:
    return normalize_runtime_path(row.get("minus_candidate_path") or row.get("vocal_candidate_path"))


def base_view_row(
    *,
    source_row: dict[str, Any],
    index: int,
    view_name: str,
    target_name: str,
    label: int | None,
    label_source: str,
    safe: bool,
    unsafe_reasons: Sequence[str],
    candidate_audio_path: str,
    map_cache: dict[str, dict[str, float]],
    blocker_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    resolved_blockers = list(blocker_labels if blocker_labels is not None else as_list(source_row.get("blockers")))
    source_run_dir = normalize_runtime_path(source_row.get("source_run_dir"))
    source_candidate_family = str(source_row.get("source_candidate_family") or "")
    candidate_family = str(
        source_row.get("candidate_family")
        or source_candidate_family
        or source_row.get("route_family")
        or source_row.get("route_family_or_unknown")
        or ""
    )
    hardcase = source_row.get("hardcase") is True or (label == 0 and bool(resolved_blockers))
    payload = {
        "view_name": view_name,
        "target_name": target_name,
        "row_id": view_row_id(index, source_row),
        "track_id": str(source_row.get("track_id") or ""),
        "run_id": str(source_row.get("run_id") or ""),
        "candidate_id": str(source_row.get("candidate_id") or ""),
        "route_id": str(source_row.get("route_id") or ""),
        "stem_type": str(source_row.get("stem_type") or ""),
        "window_start_sec": as_float(source_row.get("window_start_sec")),
        "window_end_sec": as_float(source_row.get("window_end_sec")),
        "candidate_audio_path": candidate_audio_path,
        "mix_path": normalize_runtime_path(source_row.get("mix_path")),
        "error_map_path": npz_text(source_row),
        "feature_values": view_feature_dict(source_row, map_cache=map_cache),
        "label": label if safe else None,
        "label_source": label_source if safe else "",
        "blocker_labels": resolved_blockers,
        "candidate_family": candidate_family,
        "source_candidate_family": source_candidate_family,
        "runtime_source_row": str(source_row.get("dataset_id") or "") == "runtime_harvest",
        "source_run_dir": source_run_dir,
        "source_judge_file": str(source_row.get("source_judge_file") or ""),
        "source_pair_file": str(source_row.get("source_pair_file") or ""),
        "source_selected_file": str(source_row.get("source_selected_file") or ""),
        "hardcase": bool(hardcase),
        "hardcase_holdout": source_row.get("hardcase_holdout") is True,
        "provenance_source_files": normalized_source_files(source_row),
        "safe_for_this_view": bool(safe),
        "unsafe_reason_for_this_view": ";".join(stable_unique(unsafe_reasons)),
        "release_ready": False,
    }
    if isinstance(source_row.get("source_evidence"), dict):
        payload["source_evidence"] = source_row["source_evidence"]
    return payload


def target_views_for(row: dict[str, Any]) -> set[str]:
    values = stable_unique(str(value) for value in as_list(row.get("target_views")) if str(value or ""))
    return set(values)


def row_targets_view(row: dict[str, Any], view_name: str) -> bool:
    targets = target_views_for(row)
    return not targets or view_name in targets


def build_stem_view_row(
    index: int,
    row: dict[str, Any],
    *,
    stem_type: str,
    map_cache: dict[str, dict[str, float]],
    source_cache: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    source_window, source_window_required = source_window_for(row, stem_type, source_cache=source_cache)
    evidence_row = {
        **row,
        "judge_decision": source_window_decision(stem_type, row, source_window),
        "blockers": source_window_blockers(stem_type, row, source_window),
    }
    label = stem_label(evidence_row)
    reasons = common_reasons(row, stem_type=stem_type)
    reasons.extend(
        exact_window_evidence_reasons(
            evidence_row,
            stem_type,
            source_window_required=source_window_required,
            source_window=source_window,
        )
    )
    if label is None:
        reasons.append("missing_window_label")
    safe = not stable_unique(reasons)
    source = Path(str(row.get("source_judge_file") or "")).name
    return base_view_row(
        source_row=row,
        index=index,
        view_name=f"{stem_type}_critic",
        target_name=stem_type,
        label=label,
        label_source=source,
        safe=safe,
        unsafe_reasons=reasons,
        candidate_audio_path=stem_candidate_path(row, stem_type),
        map_cache=map_cache,
        blocker_labels=as_list(evidence_row.get("blockers")),
    )


def build_pair_view_row(index: int, row: dict[str, Any], *, map_cache: dict[str, dict[str, float]]) -> dict[str, Any]:
    label = pair_label(row)
    reasons = common_reasons(row, stem_type=None)
    if not is_real_value(row.get("pair_compatibility_status")):
        reasons.append("missing_pair_compatibility_status")
    if not is_real_value(row.get("source_pair_file")):
        reasons.append("missing_exact_pair_compatibility_evidence")
    if label is None:
        reasons.append("missing_pair_label")
    safe = not stable_unique(reasons)
    source = Path(str(row.get("source_pair_file") or "")).name
    return base_view_row(
        source_row=row,
        index=index,
        view_name="pair_critic",
        target_name="pair",
        label=label,
        label_source=source,
        safe=safe,
        unsafe_reasons=reasons,
        candidate_audio_path=primary_candidate_path(row),
        map_cache=map_cache,
    )


def build_overall_view_row(index: int, row: dict[str, Any], *, map_cache: dict[str, dict[str, float]]) -> dict[str, Any]:
    label = overall_label(row)
    reasons = common_reasons(row, stem_type=None)
    if not selected_candidate_evidence(row):
        reasons.append("missing_selected_candidate_evidence")
    if not is_real_value(row.get("pair_compatibility_status")):
        reasons.append("missing_pair_compatibility_status")
    if not is_real_value(row.get("source_pair_file")):
        reasons.append("missing_exact_pair_compatibility_evidence")
    if label is None:
        reasons.append("missing_artifact_proven_accept_or_block_label")
    safe = not stable_unique(reasons)
    return base_view_row(
        source_row=row,
        index=index,
        view_name="overall_accept",
        target_name="overall_accept",
        label=label,
        label_source=str(row.get("source_selected_file") or ""),
        safe=safe,
        unsafe_reasons=reasons,
        candidate_audio_path=primary_candidate_path(row),
        map_cache=map_cache,
    )


def readiness_blockers(view_name: str, safe_rows: Sequence[dict[str, Any]]) -> list[str]:
    rules = VIEW_READINESS_RULES[view_name]
    positives = sum(1 for row in safe_rows if row.get("label") == 1)
    negatives = sum(1 for row in safe_rows if row.get("label") == 0)
    blockers: list[str] = []
    if len(safe_rows) < int(rules["min_safe"]):
        blockers.append("safe_rows_below_minimum")
    if positives < int(rules["min_positive"]):
        blockers.append("missing_positive_class" if positives == 0 else "positive_rows_below_minimum")
    if negatives < int(rules["min_negative"]):
        blockers.append("missing_negative_class" if negatives == 0 else "negative_rows_below_minimum")
    return blockers


def summarize_view(view_name: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    safe_rows = [row for row in rows if row.get("safe_for_this_view") is True]
    unsafe_rows = [row for row in rows if row.get("safe_for_this_view") is not True]
    positives = sum(1 for row in safe_rows if row.get("label") == 1)
    negatives = sum(1 for row in safe_rows if row.get("label") == 0)
    reasons = Counter(
        reason
        for row in unsafe_rows
        for reason in str(row.get("unsafe_reason_for_this_view") or "").split(";")
        if reason
    )
    blockers = readiness_blockers(view_name, safe_rows)
    return {
        "rows_total": len(rows),
        "safe_rows": len(safe_rows),
        "unsafe_rows": len(unsafe_rows),
        "positive_labels": positives,
        "negative_labels": negatives,
        "unsafe_reason_counts": dict(sorted(reasons.items())),
        "readiness_blockers": blockers,
        "can_train": not blockers,
    }


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Audio Error Training Views",
        "",
        f"- dataset_id: `{manifest['dataset_id']}`",
        f"- rows_total: {manifest['rows_total']}",
        f"- strict_negative_rows_found: {manifest['strict_negative_rows_found']}",
        f"- strict_positive_rows_found: {manifest['strict_positive_rows_found']}",
        "- training_started: false",
        "- separator_training_started: false",
        "- release_ready: false",
        "- melody_transcription_ran: false",
        "- recommendation_index_updated: false",
        "",
    ]
    for view_name, view in manifest["views"].items():
        lines.extend(
            [
                f"## {view_name}",
                "",
                f"- rows_total: {view['rows_total']}",
                f"- safe_rows: {view['safe_rows']}",
                f"- unsafe_rows: {view['unsafe_rows']}",
                f"- positive_labels: {view['positive_labels']}",
                f"- negative_labels: {view['negative_labels']}",
                f"- can_train: {str(view['can_train']).lower()}",
                f"- readiness_blockers: `{json.dumps(view['readiness_blockers'], sort_keys=True)}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_audio_error_training_views(
    *,
    dataset_dir: Path,
    output_dir: Path,
    run_roots: Sequence[Path] = (),
    listen_preview_manifests: Sequence[Path] = (),
    ralph_manifests: Sequence[Path] = (),
) -> dict[str, Any]:
    manifest_path = dataset_dir / REPAIRED_MANIFEST
    rows_path = dataset_dir / REPAIRED_ROWS
    maps_index_path = dataset_dir / REPAIRED_MAPS_INDEX
    repair_report_path = dataset_dir / REPAIR_REPORT
    missing = [path.name for path in (manifest_path, rows_path, maps_index_path, repair_report_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing_repaired_dataset_files:{','.join(missing)}")

    prepare_output_dir(output_dir)
    source_manifest = read_json(manifest_path)
    repair_report = read_json(repair_report_path)
    repaired_rows = list(iter_jsonl(rows_path))
    runtime_rows = runtime_window_rows(run_roots)
    accepted_preview_rows = accepted_listen_preview_rows(listen_preview_manifests, output_dir)
    ralph_rows = ralph_manifest_rows(ralph_manifests, output_dir)
    source_rows = repaired_rows + runtime_rows + accepted_preview_rows + ralph_rows
    map_cache: dict[str, dict[str, float]] = {}
    source_cache: dict[str, list[dict[str, Any]]] = {}
    views: dict[str, list[dict[str, Any]]] = {view_name: [] for view_name in VIEW_FILES}

    for index, row in enumerate(source_rows):
        stem_type = str(row.get("stem_type") or "")
        if stem_type in ("minus", "vocal") and row_targets_view(row, f"{stem_type}_critic"):
            views[f"{stem_type}_critic"].append(
                build_stem_view_row(
                    index,
                    row,
                    stem_type=stem_type,
                    map_cache=map_cache,
                    source_cache=source_cache,
                )
            )
        if row_targets_view(row, "pair_critic"):
            views["pair_critic"].append(build_pair_view_row(index, row, map_cache=map_cache))
        if row_targets_view(row, "overall_accept"):
            views["overall_accept"].append(build_overall_view_row(index, row, map_cache=map_cache))

    for view_name, filename in VIEW_FILES.items():
        rows = sorted(
            views[view_name],
            key=lambda item: (
                str(item.get("track_id") or ""),
                str(item.get("candidate_id") or ""),
                str(item.get("stem_type") or ""),
                float(item.get("window_start_sec") or 0.0),
                str(item.get("row_id") or ""),
            ),
        )
        views[view_name] = rows
        write_jsonl(output_dir / filename, rows)

    view_summaries = {view_name: summarize_view(view_name, rows) for view_name, rows in views.items()}
    manifest = {
        "schema": "audio_error_training_views.v1",
        "dataset_dir": str(dataset_dir),
        "dataset_id": str(source_manifest.get("dataset_id") or ""),
        "rows_total": len(source_rows),
        "repaired_rows_total": len(repaired_rows),
        "source_run_rows_added": len(runtime_rows),
        "accepted_listen_preview_rows_added": len(accepted_preview_rows),
        "ralph_manifest_rows_added": len(ralph_rows),
        "run_roots": [str(path) for path in run_roots],
        "listen_preview_manifests": [str(path) for path in listen_preview_manifests],
        "ralph_manifests": [str(path) for path in ralph_manifests],
        "source_rows_safe_for_training": sum(1 for row in repaired_rows if row.get("safe_for_training") is True),
        "source_rows_unsafe": sum(1 for row in repaired_rows if row.get("safe_for_training") is not True),
        "source_repair_report": {
            "safe_for_training_count_after": repair_report.get("safe_for_training_count_after"),
            "unsafe_count_after": repair_report.get("unsafe_count_after"),
        },
        "views": view_summaries,
        "strict_negative_rows_found": sum(view["negative_labels"] for view in view_summaries.values()),
        "strict_positive_rows_found": sum(view["positive_labels"] for view in view_summaries.values()),
        "can_train_minus_critic": view_summaries["minus_critic"]["can_train"],
        "can_train_vocal_critic": view_summaries["vocal_critic"]["can_train"],
        "can_train_pair_critic": view_summaries["pair_critic"]["can_train"],
        "can_train_overall_accept": view_summaries["overall_accept"]["can_train"],
        "unsafe_rows_used_as_labels": False,
        "final_status_alone_labels_used": False,
        **SAFETY_FLAGS,
    }
    write_json(output_dir / MANIFEST_FILE, manifest)
    write_summary(output_dir / SUMMARY_FILE, manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build target-specific audio error critic training views.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-roots", type=Path, nargs="*", default=[])
    parser.add_argument("--listen-preview-manifests", type=Path, nargs="*", default=[])
    parser.add_argument("--ralph-manifests", type=Path, nargs="*", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_audio_error_training_views(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        run_roots=args.run_roots,
        listen_preview_manifests=args.listen_preview_manifests,
        ralph_manifests=args.ralph_manifests,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
