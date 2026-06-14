#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.audit_audio_error_dataset_provenance import is_real_value, is_synthetic_candidate_id, stable_unique

try:  # pragma: no cover - availability is verified in Docker tests.
    import joblib  # type: ignore
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # type: ignore
    from sklearn.feature_extraction import DictVectorizer  # type: ignore
    from sklearn.pipeline import make_pipeline  # type: ignore

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    joblib = None
    RandomForestClassifier = None
    RandomForestRegressor = None
    DictVectorizer = None
    make_pipeline = None
    SKLEARN_AVAILABLE = False


REPAIRED_MANIFEST = "repaired_audio_error_grounding_manifest.json"
REPAIRED_ROWS = "repaired_audio_error_window_dataset.jsonl"
REPAIRED_MAPS_INDEX = "repaired_audio_error_maps_index.jsonl"
REPAIR_REPORT = "provenance_repair_report.json"

MODEL_FILE = "audio_error_critic_model.joblib"
MANIFEST_FILE = "audio_error_critic_manifest.json"
METRICS_FILE = "audio_error_critic_metrics.json"
EVAL_FILE = "audio_error_critic_eval.jsonl"
SUMMARY_FILE = "audio_error_critic_summary.md"
SCORED_FILE = "scored_audio_error_windows.jsonl"
PROGRESS_FILE = "audio_error_critic_progress.jsonl"

SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "separator_training_started": False,
    "model_downloaded": False,
    "model_registry_edited": False,
    "thresholds_relaxed": False,
    "cleanup_routes_added": False,
    "release_ready": False,
    "melody_transcription_ran": False,
    "recommendation_index_updated": False,
}

ACCEPT_DECISIONS = {"accept_candidate", "accepted_for_listen_pack", "preview_ready"}
PAIR_ACCEPT_STATUSES = {"pair_compatible", "compatible", "accepted", "preview_ready"}
SCALAR_FEATURE_COLUMNS = (
    "residual_energy",
    "mixture_consistency_error",
    "phase_collapse_score",
    "stereo_width_score",
    "vocal_bleed_proxy_score",
    "instrumental_bleed_proxy_score",
    "missing_detail_score",
    "transient_damage_score",
    "artifact_noise_score",
    "window_start_sec",
    "window_end_sec",
    "window_duration_sec",
)
CATEGORICAL_FEATURE_COLUMNS = ("route_id", "candidate_family", "stem_type")
MAP_NAMES = (
    "waveform_residual",
    "stft_residual_magnitude",
    "phase_difference_map",
    "stereo_width_map",
    "vocal_bleed_proxy_map",
    "instrumental_bleed_proxy_map",
    "missing_detail_proxy_map",
    "transient_loss_proxy_map",
    "artifact_noise_proxy_map",
    "reconstruction_error_map",
)
MAP_STAT_NAMES = ("mean", "max", "p95", "std", "temporal_centroid", "frequency_centroid", "low_band_energy", "mid_band_energy", "high_band_energy")
TARGET_COLUMNS = (
    "judge_accept_probability",
    "blocked_probability",
    "vocal_bleed_in_minus_score",
    "phase_collapse_score",
    "pair_compatibility_score",
    "overall_audio_error_score",
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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return float(number)


def clip01(value: Any) -> float:
    return float(max(0.0, min(1.0, as_float(value))))


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def path_exists(value: Any) -> bool:
    text = str(value or "")
    return bool(text and Path(text).exists())


def pair_is_compatible(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in PAIR_ACCEPT_STATUSES


def judge_is_accept(row: dict[str, Any]) -> bool:
    return (
        str(row.get("judge_decision") or "").strip() in ACCEPT_DECISIONS
        and not as_list(row.get("blockers"))
        and pair_is_compatible(row.get("pair_compatibility_status"))
    )


def has_scalar_features(row: dict[str, Any]) -> bool:
    required = [name for name in SCALAR_FEATURE_COLUMNS if name != "window_duration_sec"]
    for name in required:
        if name not in row:
            return False
        try:
            float(row[name])
        except (TypeError, ValueError):
            return False
    return True


def error_map_npz_path(row: dict[str, Any]) -> Path | None:
    paths = row.get("error_map_paths")
    if not isinstance(paths, dict):
        return None
    text = str(paths.get("npz") or "")
    if not text:
        return None
    return Path(text)


def strict_exclusion_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if row.get("safe_for_training") is not True:
        reasons.append("safe_for_training_false")
    candidate_id = row.get("candidate_id")
    route_id = row.get("route_id")
    pair_status = row.get("pair_compatibility_status")
    if not is_real_value(candidate_id):
        reasons.append("missing_candidate_id")
    elif is_synthetic_candidate_id(candidate_id):
        reasons.append("synthetic_candidate_id")
    if str(route_id or "").strip().lower() == "unknown":
        reasons.append("route_id_unknown")
    elif not is_real_value(route_id):
        reasons.append("missing_route_id")
    if not is_real_value(pair_status):
        reasons.append("missing_pair_compatibility_status")
    for field in ("mix_path", "vocal_candidate_path", "minus_candidate_path"):
        if not path_exists(row.get(field)):
            reasons.append(f"missing_audio_file:{field}")
    if row.get("release_ready") is True:
        reasons.append("release_ready_true")
    if row.get("training_started") is True or row.get("separator_training_started") is True:
        reasons.append("training_started_true")
    if not is_real_value(row.get("judge_decision")):
        reasons.append("missing_judge_decision")
    try:
        if float(row.get("window_end_sec") or 0.0) <= float(row.get("window_start_sec") or 0.0):
            reasons.append("invalid_window_boundaries")
    except (TypeError, ValueError):
        reasons.append("invalid_window_boundaries")
    npz_path = error_map_npz_path(row)
    if not ((npz_path is not None and npz_path.exists()) or has_scalar_features(row)):
        reasons.append("missing_error_maps_or_scalar_features")
    return stable_unique(reasons)


def _normalized_positive_weights(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    values = np.abs(values)
    total = float(np.sum(values))
    if total <= 1.0e-12:
        return np.zeros_like(values, dtype=np.float64)
    return values / total


def map_summary_features(path: Path, *, max_cells: int = 2_000_000) -> dict[str, float]:
    if not path.exists():
        return {}
    arrays = np.load(path)
    features: dict[str, float] = {}
    for name in MAP_NAMES:
        if name not in arrays.files:
            continue
        data = np.asarray(arrays[name], dtype=np.float32)
        if data.size == 0 or data.size > max_cells:
            continue
        if data.ndim == 1:
            data = data[None, :]
        if data.ndim > 2:
            data = data.reshape(data.shape[0], -1)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            continue
        prefix = f"map_{name}"
        features[f"{prefix}_mean"] = float(np.mean(finite))
        features[f"{prefix}_max"] = float(np.max(finite))
        features[f"{prefix}_p95"] = float(np.percentile(finite, 95))
        features[f"{prefix}_std"] = float(np.std(finite))
        weights = _normalized_positive_weights(data)
        if weights.size:
            freq_idx = np.arange(weights.shape[0], dtype=np.float64)[:, None]
            time_idx = np.arange(weights.shape[1], dtype=np.float64)[None, :]
            freq_den = max(1.0, float(weights.shape[0] - 1))
            time_den = max(1.0, float(weights.shape[1] - 1))
            features[f"{prefix}_frequency_centroid"] = float(np.sum(weights * freq_idx) / freq_den)
            features[f"{prefix}_temporal_centroid"] = float(np.sum(weights * time_idx) / time_den)
            low = data[: max(1, data.shape[0] // 3), :]
            mid = data[max(1, data.shape[0] // 3) : max(2, 2 * data.shape[0] // 3), :]
            high = data[max(2, 2 * data.shape[0] // 3) :, :]
            features[f"{prefix}_low_band_energy"] = float(np.mean(np.square(low))) if low.size else 0.0
            features[f"{prefix}_mid_band_energy"] = float(np.mean(np.square(mid))) if mid.size else 0.0
            features[f"{prefix}_high_band_energy"] = float(np.mean(np.square(high))) if high.size else 0.0
    return features


def feature_dict(row: dict[str, Any], *, map_cache: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
    start = as_float(row.get("window_start_sec"))
    end = as_float(row.get("window_end_sec"))
    features: dict[str, Any] = {
        "window_start_sec": start,
        "window_end_sec": end,
        "window_duration_sec": max(0.0, end - start),
        "route_id": str(row.get("route_id") or ""),
        "candidate_family": str(row.get("candidate_family") or row.get("source_candidate_family") or ""),
        "stem_type": str(row.get("stem_type") or "unknown"),
    }
    for name in SCALAR_FEATURE_COLUMNS:
        if name not in features:
            features[name] = as_float(row.get(name))
    npz_path = error_map_npz_path(row)
    if npz_path is not None and npz_path.exists():
        cache_key = str(npz_path)
        if map_cache is not None and cache_key in map_cache:
            features.update(map_cache[cache_key])
        else:
            stats = map_summary_features(npz_path)
            if map_cache is not None:
                map_cache[cache_key] = stats
            features.update(stats)
    return features


def target_dict(row: dict[str, Any]) -> dict[str, float | int]:
    accept = 1 if judge_is_accept(row) else 0
    error_values = [
        clip01(row.get("mixture_consistency_error")),
        clip01(row.get("phase_collapse_score")),
        clip01(row.get("vocal_bleed_proxy_score")),
        clip01(row.get("instrumental_bleed_proxy_score")),
        clip01(row.get("missing_detail_score")),
        clip01(row.get("transient_damage_score")),
        clip01(row.get("artifact_noise_score")),
    ]
    return {
        "judge_accept_label": accept,
        "blocked_label": 0 if accept else 1,
        "vocal_bleed_in_minus_score": clip01(row.get("vocal_bleed_proxy_score")),
        "phase_collapse_score": clip01(row.get("phase_collapse_score")),
        "pair_compatibility_score": 1.0 if pair_is_compatible(row.get("pair_compatibility_status")) else 0.0,
        "overall_audio_error_score": round(float(sum(error_values) / len(error_values)), 6),
    }


def group_key(row: dict[str, Any]) -> str:
    for field in ("track_id", "run_id", "candidate_id"):
        value = str(row.get(field) or "")
        if value:
            return value
    return str(row.get("window_id") or "")


def assign_group_splits(rows: list[dict[str, Any]]) -> dict[str, str]:
    groups = sorted({row["group_key"] for row in rows})
    if not groups:
        return {}
    if len(groups) == 1:
        return {groups[0]: "train"}
    if len(groups) == 2:
        return {groups[0]: "train", groups[1]: "test"}
    dev_count = 1 if len(groups) >= 5 else 0
    test_count = max(1, int(round(len(groups) * 0.2)))
    train_count = max(1, len(groups) - dev_count - test_count)
    assignments: dict[str, str] = {}
    for index, group in enumerate(groups):
        if index < train_count:
            assignments[group] = "train"
        elif index < train_count + dev_count:
            assignments[group] = "dev"
        else:
            assignments[group] = "test"
    return assignments


def load_strict_training_rows(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / REPAIRED_MANIFEST
    rows_path = dataset_dir / REPAIRED_ROWS
    maps_index_path = dataset_dir / REPAIRED_MAPS_INDEX
    repair_report_path = dataset_dir / REPAIR_REPORT
    required_paths = [manifest_path, rows_path, maps_index_path, repair_report_path]
    missing = [path.name for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing_repaired_dataset_files:{','.join(missing)}")
    manifest = read_json(manifest_path)
    repair_report = read_json(repair_report_path)
    raw_rows = list(iter_jsonl(rows_path))
    map_cache: dict[str, dict[str, float]] = {}
    usable: list[dict[str, Any]] = []
    excluded_reasons: Counter[str] = Counter()
    for index, row in enumerate(raw_rows):
        reasons = strict_exclusion_reasons(row)
        if reasons:
            for reason in reasons:
                excluded_reasons[reason] += 1
            continue
        targets = target_dict(row)
        usable.append(
            {
                "row_index": index,
                "dataset_id": str(row.get("dataset_id") or manifest.get("dataset_id") or ""),
                "track_id": str(row.get("track_id") or ""),
                "run_id": str(row.get("run_id") or ""),
                "candidate_id": str(row.get("candidate_id") or ""),
                "route_id": str(row.get("route_id") or ""),
                "stem_type": str(row.get("stem_type") or ""),
                "window_id": str(row.get("window_id") or ""),
                "window_start_sec": as_float(row.get("window_start_sec")),
                "window_end_sec": as_float(row.get("window_end_sec")),
                "judge_decision": str(row.get("judge_decision") or ""),
                "pair_compatibility_status": str(row.get("pair_compatibility_status") or ""),
                "blockers": as_list(row.get("blockers")),
                "features": feature_dict(row, map_cache=map_cache),
                "targets": targets,
                "label": int(targets["judge_accept_label"]),
                "group_key": group_key(row),
                "source_row": row,
            }
        )
    split_by_group = assign_group_splits(usable)
    for row in usable:
        row["split"] = split_by_group.get(row["group_key"], "train")
    usable.sort(key=lambda item: (item["track_id"], item["candidate_id"], item["stem_type"], item["window_start_sec"], item["window_id"]))
    return {
        "manifest": manifest,
        "repair_report": repair_report,
        "raw_rows": raw_rows,
        "usable_rows": usable,
        "excluded_reason_counts": dict(sorted(excluded_reasons.items())),
    }


def dump_bundle(bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if joblib is None:
        raise RuntimeError("joblib_unavailable")
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> dict[str, Any]:
    if joblib is None:
        raise RuntimeError("joblib_unavailable")
    return joblib.load(path)


def classifier_accept_probability(classifier: Any, features: dict[str, Any]) -> float:
    probabilities = classifier.predict_proba([features])[0]
    classes = list(classifier.classes_) if hasattr(classifier, "classes_") else []
    if 1 in classes:
        return clip01(probabilities[classes.index(1)])
    return clip01(max(probabilities))


def predict_with_bundle(bundle: dict[str, Any], row: dict[str, Any] | None = None, *, features: dict[str, Any] | None = None) -> dict[str, float]:
    if features is None:
        if row is None:
            raise ValueError("row_or_features_required")
        features = feature_dict(row)
    accept_probability = classifier_accept_probability(bundle["classifier"], features)
    predictions = {
        "judge_accept_probability": round(accept_probability, 6),
        "blocked_probability": round(1.0 - accept_probability, 6),
    }
    for target_name, regressor in bundle["regressors"].items():
        predictions[target_name] = round(clip01(regressor.predict([features])[0]), 6)
    return predictions


def _confusion_metrics(labels: list[int], scores: list[float]) -> dict[str, float | int]:
    if not labels:
        return {
            "rows": 0,
            "accept_block_accuracy": None,
            "precision": None,
            "recall": None,
            "false_accept_rate": None,
            "false_reject_rate": None,
        }
    tp = tn = fp = fn = 0
    for label, score in zip(labels, scores):
        prediction = 1 if score >= 0.5 else 0
        if prediction == 1 and label == 1:
            tp += 1
        elif prediction == 0 and label == 0:
            tn += 1
        elif prediction == 1 and label == 0:
            fp += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    false_accept_rate = fp / (fp + tn) if fp + tn else 0.0
    false_reject_rate = fn / (fn + tp) if fn + tp else 0.0
    return {
        "rows": len(labels),
        "accept_block_accuracy": round((tp + tn) / len(labels), 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "false_accept_rate": round(false_accept_rate, 6),
        "false_reject_rate": round(false_reject_rate, 6),
    }


def maybe_auroc(labels: list[int], scores: list[float]) -> float | None:
    if len(set(labels)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore

        return round(float(roc_auc_score(labels, scores)), 6)
    except Exception:  # pragma: no cover - sklearn metrics availability follows sklearn.
        return None


def calibration_summary(labels: list[int], scores: list[float], bins: int = 5) -> list[dict[str, float | int]]:
    if not labels:
        return []
    rows: list[dict[str, float | int]] = []
    for bucket in range(bins):
        low = bucket / bins
        high = (bucket + 1) / bins
        selected = [
            (label, score)
            for label, score in zip(labels, scores)
            if low <= score < high or (bucket == bins - 1 and score == 1.0)
        ]
        if not selected:
            rows.append({"bin": bucket, "rows": 0, "confidence": None, "accuracy": None})
            continue
        rows.append(
            {
                "bin": bucket,
                "rows": len(selected),
                "confidence": round(sum(score for _, score in selected) / len(selected), 6),
                "accuracy": round(sum(label for label, _ in selected) / len(selected), 6),
            }
        )
    return rows


def compute_metrics(rows: list[dict[str, Any]], predictions: list[dict[str, float]]) -> dict[str, Any]:
    labels = [int(row["targets"]["judge_accept_label"]) for row in rows]
    accept_scores = [float(prediction["judge_accept_probability"]) for prediction in predictions]
    base = _confusion_metrics(labels, accept_scores)
    phase_errors = [
        abs(float(prediction["phase_collapse_score"]) - float(row["targets"]["phase_collapse_score"]))
        for row, prediction in zip(rows, predictions)
    ]
    bleed_errors = [
        abs(float(prediction["vocal_bleed_in_minus_score"]) - float(row["targets"]["vocal_bleed_in_minus_score"]))
        for row, prediction in zip(rows, predictions)
    ]
    pair_correct = [
        (1 if float(prediction["pair_compatibility_score"]) >= 0.5 else 0)
        == int(round(float(row["targets"]["pair_compatibility_score"])))
        for row, prediction in zip(rows, predictions)
    ]
    metrics: dict[str, Any] = {
        **base,
        "blocker_auroc": maybe_auroc(labels, accept_scores),
        "phase_collapse_regression_mae": round(sum(phase_errors) / len(phase_errors), 6) if phase_errors else None,
        "vocal_bleed_regression_mae": round(sum(bleed_errors) / len(bleed_errors), 6) if bleed_errors else None,
        "pair_compatibility_accuracy": round(sum(1 for item in pair_correct if item) / len(pair_correct), 6) if pair_correct else None,
        "calibration_summary": calibration_summary(labels, accept_scores),
    }
    return metrics


def grouped_metrics(rows: list[dict[str, Any]], predictions: list[dict[str, float]], key: str) -> dict[str, Any]:
    grouped: dict[str, tuple[list[dict[str, Any]], list[dict[str, float]]]] = {}
    for row, prediction in zip(rows, predictions):
        value = str(row.get(key) or "")
        if value not in grouped:
            grouped[value] = ([], [])
        grouped[value][0].append(row)
        grouped[value][1].append(prediction)
    return {name: compute_metrics(group_rows, group_predictions) for name, (group_rows, group_predictions) in sorted(grouped.items())}


def null_metrics(reason: str) -> dict[str, Any]:
    return {
        "decision": "critic_training_skipped_insufficient_data",
        "critic_usable_for_shadow_scoring": False,
        "critic_not_usable_reason": reason,
        "accept_block_accuracy": None,
        "precision": None,
        "recall": None,
        "false_accept_rate": None,
        "false_reject_rate": None,
        "blocker_auroc": None,
        "phase_collapse_regression_mae": None,
        "vocal_bleed_regression_mae": None,
        "pair_compatibility_accuracy": None,
        "calibration_summary": [],
        "per_track_metrics": {},
        "per_route_metrics": {},
        **SAFETY_FLAGS,
    }


def write_summary(path: Path, manifest: dict[str, Any], metrics: dict[str, Any]) -> None:
    lines = [
        "# Audio Error Critic Baseline",
        "",
        f"- model_id: `{manifest['model_id']}`",
        f"- decision: {manifest['decision']}",
        f"- rows_total: {manifest['rows_total']}",
        f"- rows_used_for_training: {manifest['rows_used_for_training']}",
        f"- rows_excluded: {manifest['rows_excluded']}",
        f"- train_rows: {manifest['train_rows']}",
        f"- dev_rows: {manifest['dev_rows']}",
        f"- test_rows: {manifest['test_rows']}",
        f"- model_quality_confidence: {manifest['model_quality_confidence']}",
        f"- critic_usable_for_shadow_scoring: {str(manifest['critic_usable_for_shadow_scoring']).lower()}",
        f"- critic_not_usable_reason: {manifest['critic_not_usable_reason']}",
        f"- accept_block_accuracy: {metrics.get('accept_block_accuracy')}",
        f"- false_accept_rate: {metrics.get('false_accept_rate')}",
        f"- false_reject_rate: {metrics.get('false_reject_rate')}",
        f"- phase_collapse_mae: {metrics.get('phase_collapse_regression_mae')}",
        f"- vocal_bleed_mae: {metrics.get('vocal_bleed_regression_mae')}",
        f"- pair_compatibility_accuracy: {metrics.get('pair_compatibility_accuracy')}",
        "- shadow_only: true",
        "- unsafe_rows_used: false",
        "- separator_training_started: false",
        "- release_ready: false",
        "- melody_transcription_ran: false",
        "- recommendation_index_updated: false",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def progress(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row = {
        **payload,
        "gpu_inference_skipped": True,
        "reason": payload.get("reason") or "cpu_sklearn_shadow_critic_training_only",
        "release_ready": False,
    }
    with (output_dir / PROGRESS_FILE).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)


def base_manifest(
    *,
    dataset_dir: Path,
    output_dir: Path,
    data: dict[str, Any],
    min_safe_rows: int,
    model_kind: str,
    group_split: str,
    shadow_only: bool,
    decision: str,
    reason: str,
) -> dict[str, Any]:
    rows = data["usable_rows"]
    split_counts = Counter(row.get("split") for row in rows)
    labels = Counter(int(row["label"]) for row in rows)
    confidence = "low" if len(rows) < 500 else "normal"
    model_id = "audio_error_critic_" + stable_hash(
        {
            "dataset_id": data["manifest"].get("dataset_id"),
            "usable_row_ids": [row["window_id"] for row in rows],
            "model_kind": model_kind,
            "decision": decision,
        }
    )[:12]
    return {
        "model_id": model_id,
        "model_type": f"sklearn_{model_kind}_audio_error_critic" if decision == "critic_training_completed" else "skipped_no_model",
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "rows_total": len(data["raw_rows"]),
        "safe_rows_available": sum(1 for row in data["raw_rows"] if row.get("safe_for_training") is True),
        "rows_used_for_training": len(rows) if decision == "critic_training_completed" else 0,
        "strict_rows_available": len(rows),
        "rows_excluded": len(data["raw_rows"]) - len(rows),
        "train_rows": int(split_counts.get("train", 0)) if decision == "critic_training_completed" else 0,
        "dev_rows": int(split_counts.get("dev", 0)) if decision == "critic_training_completed" else 0,
        "test_rows": int(split_counts.get("test", 0)) if decision == "critic_training_completed" else 0,
        "group_split_strategy": group_split,
        "feature_columns": list(SCALAR_FEATURE_COLUMNS) + list(CATEGORICAL_FEATURE_COLUMNS) + [f"map_*_{name}" for name in MAP_STAT_NAMES],
        "target_columns": list(TARGET_COLUMNS),
        "excluded_reason_counts": data["excluded_reason_counts"],
        "label_counts": {str(key): int(value) for key, value in sorted(labels.items())},
        "minimum_safe_rows": int(min_safe_rows),
        "unsafe_rows_used": False,
        "shadow_only": bool(shadow_only),
        "decision": decision,
        "critic_training_completed": decision == "critic_training_completed",
        "critic_usable_for_shadow_scoring": decision == "critic_training_completed",
        "critic_not_usable_reason": reason,
        "model_quality_confidence": confidence if decision == "critic_training_completed" else "not_applicable",
        "can_promote_to_production_gate": False,
        **SAFETY_FLAGS,
    }


def train_audio_error_critic(
    *,
    dataset_dir: Path,
    output_dir: Path,
    min_safe_rows: int = 100,
    model_kind: str = "random_forest",
    group_split: str = "track_or_candidate",
    shadow_only: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (MANIFEST_FILE, METRICS_FILE, EVAL_FILE, SUMMARY_FILE, SCORED_FILE, PROGRESS_FILE):
        path = output_dir / name
        if path.exists():
            path.unlink()
    model_path = output_dir / MODEL_FILE
    if model_path.exists():
        model_path.unlink()
    progress(output_dir, {"phase": "start", "dataset_dir": str(dataset_dir), "model_kind": model_kind})
    data = load_strict_training_rows(dataset_dir)
    rows = data["usable_rows"]
    reason = ""
    if len(rows) < min_safe_rows:
        reason = "safe_rows_below_minimum"
    elif len({int(row["label"]) for row in rows}) < 2:
        reason = "missing_positive_or_negative_class"
    elif not SKLEARN_AVAILABLE:
        reason = "sklearn_or_joblib_unavailable"
    elif model_kind != "random_forest":
        reason = "unsupported_model_kind"
    if reason:
        manifest = base_manifest(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            data=data,
            min_safe_rows=min_safe_rows,
            model_kind=model_kind,
            group_split=group_split,
            shadow_only=shadow_only,
            decision="critic_training_skipped_insufficient_data",
            reason=reason,
        )
        metrics = null_metrics(reason)
        write_json(output_dir / MANIFEST_FILE, manifest)
        write_json(output_dir / METRICS_FILE, metrics)
        write_jsonl(output_dir / EVAL_FILE, [])
        write_jsonl(output_dir / SCORED_FILE, [])
        write_summary(output_dir / SUMMARY_FILE, manifest, metrics)
        progress(output_dir, {"phase": "skipped", "critic_not_usable_reason": reason, "rows_total": len(data["raw_rows"]), "strict_rows_available": len(rows)})
        return manifest

    train_rows = [row for row in rows if row.get("split") == "train"] or rows
    x_train = [row["features"] for row in train_rows]
    y_train = [int(row["label"]) for row in train_rows]
    classifier = make_pipeline(
        DictVectorizer(sparse=False),
        RandomForestClassifier(n_estimators=64, random_state=0, class_weight="balanced", min_samples_leaf=1),
    )
    classifier.fit(x_train, y_train)
    regressors: dict[str, Any] = {}
    for target_name in (
        "vocal_bleed_in_minus_score",
        "phase_collapse_score",
        "pair_compatibility_score",
        "overall_audio_error_score",
    ):
        regressor = make_pipeline(
            DictVectorizer(sparse=False),
            RandomForestRegressor(n_estimators=64, random_state=0, min_samples_leaf=1),
        )
        regressor.fit(x_train, [float(row["targets"][target_name]) for row in train_rows])
        regressors[target_name] = regressor
    manifest = base_manifest(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        data=data,
        min_safe_rows=min_safe_rows,
        model_kind=model_kind,
        group_split=group_split,
        shadow_only=shadow_only,
        decision="critic_training_completed",
        reason="",
    )
    bundle = {
        "model_id": manifest["model_id"],
        "model_type": manifest["model_type"],
        "feature_columns": manifest["feature_columns"],
        "target_columns": manifest["target_columns"],
        "classifier": classifier,
        "regressors": regressors,
        "shadow_only": True,
        "unsafe_rows_used": False,
        **SAFETY_FLAGS,
    }
    dump_bundle(bundle, model_path)
    predictions = [predict_with_bundle(bundle, features=row["features"]) for row in rows]
    metrics = {
        **compute_metrics(rows, predictions),
        "per_track_metrics": grouped_metrics(rows, predictions, "track_id"),
        "per_route_metrics": grouped_metrics(rows, predictions, "route_id"),
        "decision": manifest["decision"],
        "critic_usable_for_shadow_scoring": True,
        "critic_not_usable_reason": "",
        **SAFETY_FLAGS,
    }
    eval_rows = []
    for row, prediction in zip(rows, predictions):
        payload = evaluation_row(row, prediction)
        payload["model_id"] = manifest["model_id"]
        eval_rows.append(payload)
    write_json(output_dir / MANIFEST_FILE, manifest)
    write_json(output_dir / METRICS_FILE, metrics)
    write_jsonl(output_dir / EVAL_FILE, eval_rows)
    write_jsonl(output_dir / SCORED_FILE, eval_rows)
    write_summary(output_dir / SUMMARY_FILE, manifest, metrics)
    progress(output_dir, {"phase": "complete", "model_id": manifest["model_id"], "rows_used_for_training": manifest["rows_used_for_training"]})
    return manifest


def evaluation_row(row: dict[str, Any], prediction: dict[str, float]) -> dict[str, Any]:
    return {
        "model_id": "",
        "dataset_id": row["dataset_id"],
        "track_id": row["track_id"],
        "run_id": row["run_id"],
        "candidate_id": row["candidate_id"],
        "route_id": row["route_id"],
        "stem_type": row["stem_type"],
        "window_id": row["window_id"],
        "window_start_sec": row["window_start_sec"],
        "window_end_sec": row["window_end_sec"],
        "split": row["split"],
        "judge_decision": row["judge_decision"],
        "pair_compatibility_status": row["pair_compatibility_status"],
        "label": int(row["label"]),
        "target_blocked_label": int(row["targets"]["blocked_label"]),
        **prediction,
        "critic_score_available": True,
        "shadow_only": True,
        "unsafe_rows_used": False,
        **SAFETY_FLAGS,
    }


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid_bool:{value}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a shadow-only audio error critic from strict safe repaired rows.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-safe-rows", type=int, default=100)
    parser.add_argument("--model-kind", default="random_forest")
    parser.add_argument("--group-split", default="track_or_candidate")
    parser.add_argument("--shadow-only", type=parse_bool, default=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = train_audio_error_critic(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        min_safe_rows=args.min_safe_rows,
        model_kind=args.model_kind,
        group_split=args.group_split,
        shadow_only=args.shadow_only,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
