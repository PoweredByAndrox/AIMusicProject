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

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import joblib  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_extraction import DictVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.multiclass import OneVsRestClassifier  # noqa: E402
from sklearn.preprocessing import MultiLabelBinarizer  # noqa: E402

from tools.audit_audio_error_dataset_provenance import is_real_value, is_synthetic_candidate_id  # noqa: E402


MANIFEST_FILE = "target_audio_error_critics_manifest.json"
SUMMARY_FILE = "target_audio_error_critic_summary.md"
EVAL_FILE = "target_audio_error_critic_eval.jsonl"
THRESHOLDS_FILE = "target_audio_error_critic_thresholds.json"
RELIABILITY_REPORT_FILE = "target_audio_error_critic_reliability_report.json"
MINUS_MODEL_FILE = "minus_audio_error_critic.joblib"
VOCAL_MODEL_FILE = "vocal_audio_error_critic.joblib"
MINUS_METRICS_FILE = "minus_audio_error_critic_metrics.json"
VOCAL_METRICS_FILE = "vocal_audio_error_critic_metrics.json"

TARGETS = {
    "minus": {
        "view_name": "minus_critic",
        "view_file": "minus_critic_training_view.jsonl",
        "model_file": MINUS_MODEL_FILE,
        "metrics_file": MINUS_METRICS_FILE,
        "label_source": "minus_windowed_judge.jsonl",
        "stem_type": "minus",
        "model_id_prefix": "minus_accept_block_classifier",
    },
    "vocal": {
        "view_name": "vocal_critic",
        "view_file": "vocal_critic_training_view.jsonl",
        "model_file": VOCAL_MODEL_FILE,
        "metrics_file": VOCAL_METRICS_FILE,
        "label_source": "vocal_windowed_judge.jsonl",
        "stem_type": "vocal",
        "model_id_prefix": "vocal_accept_block_classifier",
    },
}

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
)
CATEGORICAL_FEATURE_COLUMNS = ("route_id", "candidate_family", "stem_type")
TARGET_COLUMNS = ("accept_block_label",)
ACOUSTIC_FEATURE_METADATA_COLUMNS = set(CATEGORICAL_FEATURE_COLUMNS) | {"window_start_sec", "window_end_sec", "window_duration_sec"}
SAFETY_FLAGS = {
    "unsafe_rows_used": False,
    "pair_rows_used": False,
    "overall_rows_used": False,
    "shadow_only": True,
    "separator_training_started": False,
    "release_ready": False,
    "melody_transcription_ran": False,
    "recommendation_index_updated": False,
    "model_downloaded": False,
    "thresholds_relaxed": False,
    "cleanup_routes_added": False,
    "runtime_gates_changed": False,
}


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


def parse_targets(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        items = [item.strip().lower() for item in value.split(",") if item.strip()]
    else:
        items = [str(item).strip().lower() for item in value if str(item).strip()]
    invalid = sorted(set(items) - set(TARGETS))
    if invalid:
        raise ValueError(f"unsupported_targets:{','.join(invalid)}")
    return tuple(items or ("minus", "vocal"))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:
        return default
    return result


def clip01(value: Any) -> float:
    return max(0.0, min(1.0, as_float(value)))


def row_path_exists(value: Any) -> bool:
    if not is_real_value(value):
        return False
    return Path(str(value)).exists()


def target_row_exclusion_reasons(row: dict[str, Any], target: str) -> list[str]:
    cfg = TARGETS[target]
    reasons: list[str] = []
    if row.get("safe_for_this_view") is not True:
        reasons.append("safe_for_this_view_false")
    target_name = str(row.get("target_name") or "")
    if row.get("view_name") != cfg["view_name"] or target_name not in {cfg["view_name"], target}:
        reasons.append("target_view_mismatch")
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
    if not row_path_exists(row.get("candidate_audio_path")):
        reasons.append("missing_audio_file:candidate_audio_path")
    if not row_path_exists(row.get("mix_path")):
        reasons.append("missing_audio_file:mix_path")
    if row.get("label") not in (0, 1):
        reasons.append("missing_label")
    label_source = str(row.get("label_source") or "")
    if label_source != cfg["label_source"]:
        reasons.append(f"missing_exact_{cfg['stem_type']}_window_judge_evidence")
    if row.get("release_ready") is True:
        reasons.append("release_ready_true")
    if row.get("separator_training_started") is True:
        reasons.append("separator_training_started_true")
    if row.get("unsafe_reason_for_this_view"):
        reasons.append("unsafe_reason_present")
    try:
        if float(row.get("window_end_sec") or 0.0) <= float(row.get("window_start_sec") or 0.0):
            reasons.append("invalid_window_boundaries")
    except (TypeError, ValueError):
        reasons.append("invalid_window_boundaries")
    features = row.get("feature_values")
    if not isinstance(features, dict) or not features:
        reasons.append("missing_feature_values")
    elif not has_acoustic_feature_values(features):
        reasons.append("missing_acoustic_feature_values")
    return sorted(set(reasons))


def has_acoustic_feature_values(features: dict[str, Any]) -> bool:
    for name, value in features.items():
        if name in ACOUSTIC_FEATURE_METADATA_COLUMNS:
            continue
        if abs(as_float(value)) > 1.0e-12:
            return True
    return False


def feature_dict(row: dict[str, Any], target: str) -> dict[str, Any]:
    values = row.get("feature_values") if isinstance(row.get("feature_values"), dict) else {}
    start = as_float(row.get("window_start_sec", values.get("window_start_sec")))
    end = as_float(row.get("window_end_sec", values.get("window_end_sec")))
    features: dict[str, Any] = {
        "route_id": str(row.get("route_id") or values.get("route_id") or ""),
        "candidate_family": str(values.get("candidate_family") or row.get("candidate_family") or row.get("source_candidate_family") or ""),
        "stem_type": str(row.get("stem_type") or values.get("stem_type") or TARGETS[target]["stem_type"]),
        "window_start_sec": start,
        "window_end_sec": end,
        "window_duration_sec": max(0.0, end - start),
    }
    for name in SCALAR_FEATURE_COLUMNS:
        if name not in features:
            features[name] = as_float(values.get(name, row.get(name)))
    return features


def split_group_value(row: dict[str, Any], strategy: str) -> str:
    track_id = str(row.get("track_id") or "")
    candidate_id = str(row.get("candidate_id") or "")
    strategy = str(strategy or "candidate_id").strip().lower()
    if strategy == "candidate_id" and candidate_id:
        return f"candidate:{candidate_id}"
    if strategy == "track_id" and track_id:
        return f"track:{track_id}"
    if strategy == "track_or_candidate" and track_id:
        return f"track:{track_id}"
    if candidate_id:
        return f"candidate:{candidate_id}"
    if track_id:
        return f"track:{track_id}"
    return f"candidate:{candidate_id}"


def group_key(row: dict[str, Any], strategy: str = "candidate_id") -> str:
    return split_group_value(row, strategy)


def assign_group_splits(
    rows: list[dict[str, Any]],
    strategy: str = "candidate_id",
    *,
    min_negative_test_rows: int = 0,
    min_negative_dev_rows: int = 0,
    min_negative_train_rows: int = 0,
) -> dict[str, str]:
    for row in rows:
        row["group_key"] = group_key(row, strategy)
    groups = sorted({str(row["group_key"]) for row in rows})
    if len(groups) <= 1:
        return {groups[0]: "train"} if groups else {}
    ordered = sorted(groups, key=lambda value: stable_hash(value))
    test_count = max(1, len(ordered) // 5)
    dev_count = 1 if len(ordered) >= 5 else 0
    split_by_group: dict[str, str] = {}
    negative_by_group: Counter[str] = Counter()
    for row in rows:
        if int(row.get("label") or 0) == 0:
            negative_by_group[str(row["group_key"])] += 1
    total_negative_rows = sum(negative_by_group.values())
    assigned_test_negative_rows = 0
    assigned_dev_negative_rows = 0
    assigned_test_groups = 0
    assigned_dev_groups = 0
    for index, group in enumerate(ordered):
        group_negatives = int(negative_by_group.get(group, 0))
        remaining_if_not_train = total_negative_rows - assigned_test_negative_rows - assigned_dev_negative_rows - group_negatives
        can_reserve_train = remaining_if_not_train >= int(min_negative_train_rows)
        if assigned_test_groups < test_count or (
            assigned_test_negative_rows < int(min_negative_test_rows)
            and index < len(ordered) - 1
            and can_reserve_train
        ):
            split_by_group[group] = "test"
            assigned_test_groups += 1
            assigned_test_negative_rows += group_negatives
        elif assigned_dev_groups < dev_count or (
            assigned_dev_negative_rows < int(min_negative_dev_rows)
            and index < len(ordered) - 1
            and can_reserve_train
        ):
            split_by_group[group] = "dev"
            assigned_dev_groups += 1
            assigned_dev_negative_rows += group_negatives
        else:
            split_by_group[group] = "train"
    if all(split == "test" for split in split_by_group.values()):
        split_by_group[ordered[-1]] = "train"
    return split_by_group


def _ids_by_split(rows: list[dict[str, Any]], field: str) -> dict[str, list[str]]:
    result: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
    for row in rows:
        split = str(row.get("split") or "train")
        if split not in result:
            result[split] = set()
        value = str(row.get(field) or "")
        if value:
            result[split].add(value)
    return {split: sorted(values) for split, values in sorted(result.items())}


def _counter_by_split(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = {"train": Counter(), "dev": Counter(), "test": Counter()}
    for row in rows:
        split = str(row.get("split") or "train")
        result.setdefault(split, Counter())
        value = str(row.get(field) if row.get(field) is not None else "")
        result[split][value] += 1
    return {split: dict(sorted(counter.items())) for split, counter in sorted(result.items())}


def split_audit(rows: list[dict[str, Any]], *, split_strategy: str, min_negative_test_rows: int = 0) -> dict[str, Any]:
    groups_by_split = _ids_by_split(rows, "group_key")
    candidate_ids = _ids_by_split(rows, "candidate_id")
    track_ids = _ids_by_split(rows, "track_id")
    train_candidates = set(candidate_ids.get("train", []))
    test_candidates = set(candidate_ids.get("test", []))
    train_tracks = set(track_ids.get("train", []))
    test_tracks = set(track_ids.get("test", []))
    test_negative_rows = sum(1 for row in rows if str(row.get("split") or "") == "test" and int(row.get("label") or 0) == 0)
    return {
        "split_strategy": split_strategy,
        "split_group_key": split_strategy,
        "train_groups": groups_by_split.get("train", []),
        "dev_groups": groups_by_split.get("dev", []),
        "test_groups": groups_by_split.get("test", []),
        "train_candidate_ids": candidate_ids.get("train", []),
        "dev_candidate_ids": candidate_ids.get("dev", []),
        "test_candidate_ids": candidate_ids.get("test", []),
        "candidate_id_overlap_train_test": bool(train_candidates & test_candidates),
        "track_id_overlap_train_test": bool(train_tracks & test_tracks),
        "route_id_distribution_by_split": _counter_by_split(rows, "route_id"),
        "label_distribution_by_split": _counter_by_split(rows, "label"),
        "test_negative_rows": int(test_negative_rows),
        "evaluation_low_confidence": int(test_negative_rows) < int(min_negative_test_rows),
    }


def load_target_rows(
    views_dir: Path,
    target: str,
    group_split: str = "candidate_id",
    *,
    min_negative_test_rows: int = 0,
    min_negative_dev_rows: int = 0,
    min_negative_train_rows: int = 0,
) -> dict[str, Any]:
    cfg = TARGETS[target]
    view_path = views_dir / cfg["view_file"]
    if not view_path.exists():
        raise FileNotFoundError(f"missing_view_file:{cfg['view_file']}")
    raw_rows = list(iter_jsonl(view_path))
    usable: list[dict[str, Any]] = []
    excluded_reasons: Counter[str] = Counter()
    for index, row in enumerate(raw_rows):
        reasons = target_row_exclusion_reasons(row, target)
        if reasons:
            for reason in reasons:
                excluded_reasons[reason] += 1
            continue
        usable.append(
            {
                "row_index": index,
                "view_name": cfg["view_name"],
                "target_name": cfg["view_name"],
                "row_id": str(row.get("row_id") or ""),
                "track_id": str(row.get("track_id") or ""),
                "run_id": str(row.get("run_id") or ""),
                "candidate_id": str(row.get("candidate_id") or ""),
                "route_id": str(row.get("route_id") or ""),
                "stem_type": str(row.get("stem_type") or cfg["stem_type"]),
                "window_start_sec": as_float(row.get("window_start_sec")),
                "window_end_sec": as_float(row.get("window_end_sec")),
                "candidate_audio_path": str(row.get("candidate_audio_path") or ""),
                "mix_path": str(row.get("mix_path") or ""),
                "label_source": str(row.get("label_source") or ""),
                "blocker_labels": [str(item) for item in (row.get("blocker_labels") or [])],
                "features": feature_dict(row, target),
                "label": int(row["label"]),
                "group_key": group_key(row, group_split),
                "source_row": row,
            }
        )
    split_by_group = assign_group_splits(
        usable,
        group_split,
        min_negative_test_rows=min_negative_test_rows,
        min_negative_dev_rows=min_negative_dev_rows,
        min_negative_train_rows=min_negative_train_rows,
    )
    for row in usable:
        row["split"] = split_by_group.get(row["group_key"], "train")
    usable.sort(key=lambda item: (item["track_id"], item["candidate_id"], item["window_start_sec"], item["row_id"]))
    return {
        "raw_rows": raw_rows,
        "usable_rows": usable,
        "excluded_reason_counts": dict(sorted(excluded_reasons.items())),
    }


def model_for_kind(kind: str, class_weight: str | dict[int, float] | None = "balanced") -> Any:
    if kind == "random_forest":
        return RandomForestClassifier(n_estimators=80, random_state=1729, class_weight=class_weight, min_samples_leaf=1)
    if kind == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=1729)
    raise ValueError(f"unsupported_model_kind:{kind}")


def parse_class_weight(value: str | None) -> str | dict[int, float] | None:
    if value is None:
        return "balanced"
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "false", "0"}:
        return None
    if normalized == "balanced":
        return "balanced"
    raise ValueError(f"unsupported_class_weight:{value}")


def parse_blockers(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in value if str(item).strip()]
    return tuple(dict.fromkeys(items))


def sample_weights_for_rows(
    rows: list[dict[str, Any]],
    *,
    negative_class_weight: float = 1.0,
    hard_negative_blockers: Sequence[str] = (),
) -> tuple[list[float], int]:
    blockers = set(hard_negative_blockers)
    weights: list[float] = []
    hard_negative_rows_used = 0
    for row in rows:
        label = int(row.get("label") or 0)
        weight = 1.0
        if label == 0:
            weight *= max(1.0, float(negative_class_weight))
            if blockers and blockers.intersection(set(row.get("blocker_labels") or [])):
                weight *= 2.0
                hard_negative_rows_used += 1
        weights.append(float(weight))
    return weights, hard_negative_rows_used


def accept_probability(classifier: Any, matrix: Any) -> list[float]:
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(matrix)
        classes = list(getattr(classifier, "classes_", []))
        if 1 in classes:
            index = classes.index(1)
            return [clip01(row[index]) for row in probabilities]
        return [0.0 for _ in range(matrix.shape[0])]
    return [float(value) for value in classifier.predict(matrix)]


def apply_probability_calibration(bundle: dict[str, Any], scores: list[float]) -> list[float]:
    calibration = bundle.get("calibration")
    if not isinstance(calibration, dict) or calibration.get("status") != "calibrated":
        return [clip01(score) for score in scores]
    method = str(calibration.get("method") or "")
    calibrator = calibration.get("model")
    if calibrator is None:
        return [clip01(score) for score in scores]
    if method == "sigmoid" and hasattr(calibrator, "predict_proba"):
        matrix = [[clip01(score)] for score in scores]
        probabilities = calibrator.predict_proba(matrix)
        classes = list(getattr(calibrator, "classes_", []))
        if 1 in classes:
            index = classes.index(1)
            return [clip01(row[index]) for row in probabilities]
    if method == "isotonic" and hasattr(calibrator, "transform"):
        return [clip01(score) for score in calibrator.transform([clip01(score) for score in scores])]
    return [clip01(score) for score in scores]


def blocker_predictions(bundle: dict[str, Any], matrix: Any) -> list[list[str]]:
    classifier = bundle.get("blocker_classifier")
    binarizer = bundle.get("blocker_binarizer")
    if classifier is None or binarizer is None:
        return [[] for _ in range(matrix.shape[0])]
    raw = classifier.predict(matrix)
    if len(getattr(raw, "shape", ())) == 1:
        raw = raw.reshape(-1, 1)
    labels = binarizer.inverse_transform(raw)
    return [[str(item) for item in row] for row in labels]


def predict_bundle(bundle: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    matrix = bundle["vectorizer"].transform([row["features"] for row in rows])
    accept_scores = apply_probability_calibration(bundle, accept_probability(bundle["accept_classifier"], matrix))
    blockers = blocker_predictions(bundle, matrix)
    predictions: list[dict[str, Any]] = []
    for score, predicted_blockers in zip(accept_scores, blockers):
        predictions.append(
            {
                "predicted_accept_probability": round(clip01(score), 6),
                "predicted_block_probability": round(1.0 - clip01(score), 6),
                "predicted_blockers": predicted_blockers,
                "critic_decision": "accept_candidate" if score >= 0.5 else "blocked_hard_case",
                "confidence": round(abs(clip01(score) - 0.5) * 2.0, 6),
            }
        )
    return predictions


def thresholded_metrics(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    accept_threshold: float,
    block_threshold: float,
) -> dict[str, Any]:
    total = len(rows)
    tp = tn = fp = fn = 0
    accept_count = block_count = abstain_count = 0
    accepted_positive_rows = 0
    decisions: list[str] = []
    labels = [int(row.get("label") or 0) for row in rows]
    scores = [clip01(prediction.get("predicted_accept_probability")) for prediction in predictions]
    for label, prediction in zip(labels, predictions):
        accept_score = clip01(prediction.get("predicted_accept_probability"))
        block_score = clip01(prediction.get("predicted_block_probability", 1.0 - accept_score))
        if accept_score >= accept_threshold:
            decision = "accept_candidate"
            accept_count += 1
            if label == 1:
                tp += 1
                accepted_positive_rows += 1
            else:
                fp += 1
        elif block_score >= block_threshold:
            decision = "blocked_hard_case"
            block_count += 1
            if label == 0:
                tn += 1
            else:
                fn += 1
        else:
            decision = "abstain"
            abstain_count += 1
            if label == 1:
                fn += 1
        decisions.append(decision)
    positives = sum(1 for label in labels if label == 1)
    negatives = sum(1 for label in labels if label == 0)
    false_accept_rate = fp / negatives if negatives else 0.0
    false_reject_rate = fn / positives if positives else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else 0.0
    auroc = None
    if len(set(labels)) > 1:
        auroc = round(float(roc_auc_score(labels, scores)), 6)
    brier_score = None
    if labels:
        brier_score = round(sum((float(label) - score) ** 2 for label, score in zip(labels, scores)) / len(labels), 6)
    return {
        "rows": total,
        "accept_threshold": round(float(accept_threshold), 6),
        "block_threshold": round(float(block_threshold), 6),
        "accept_count": accept_count,
        "block_count": block_count,
        "abstain_rows": abstain_count,
        "abstain_rate": round(abstain_count / total, 6) if total else 0.0,
        "accepted_positive_rows": accepted_positive_rows,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "false_accept_rate": round(false_accept_rate, 6),
        "false_reject_rate": round(false_reject_rate, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "auroc": auroc,
        "brier_score": brier_score,
        "decisions": decisions,
    }


def select_thresholds(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    max_false_accept_rate: float,
    threshold_objective: str = "false_accept_first",
    enable_abstain_zone: bool = True,
    abstain_margin: float = 0.10,
) -> dict[str, Any]:
    if threshold_objective != "false_accept_first":
        raise ValueError(f"unsupported_threshold_objective:{threshold_objective}")
    scores = sorted({clip01(prediction.get("predicted_accept_probability")) for prediction in predictions})
    candidates = sorted(set([0.0, 0.5, 1.000001] + scores))
    best: dict[str, Any] | None = None
    for threshold in candidates:
        if enable_abstain_zone:
            block_threshold = max(0.0, min(1.0, float(threshold) - float(abstain_margin)))
        else:
            block_threshold = max(0.0, min(1.0, 1.0 - float(threshold)))
        metrics = thresholded_metrics(rows, predictions, accept_threshold=threshold, block_threshold=block_threshold)
        metrics["accept_threshold"] = round(float(threshold), 6)
        metrics["block_threshold"] = round(float(block_threshold), 6)
        within_limit = metrics["false_accept_rate"] <= float(max_false_accept_rate)
        score_tuple = (
            0 if within_limit else 1,
            metrics["false_accept_rate"],
            -metrics["accepted_positive_rows"],
            metrics["abstain_rows"],
            -metrics["accept_threshold"],
        )
        if best is None or score_tuple < best["_score_tuple"]:
            best = {**metrics, "_score_tuple": score_tuple}
    assert best is not None
    best.pop("_score_tuple", None)
    best["threshold_objective"] = threshold_objective
    best["max_false_accept_rate"] = float(max_false_accept_rate)
    best["enable_abstain_zone"] = bool(enable_abstain_zone)
    best["abstain_margin"] = float(abstain_margin)
    return best


def confusion_metrics(labels: list[int], accept_scores: list[float]) -> dict[str, Any]:
    if not labels:
        return {
            "rows": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "false_accept_rate": None,
            "false_reject_rate": None,
            "auroc": None,
        }
    tp = tn = fp = fn = 0
    for label, score in zip(labels, accept_scores):
        predicted = 1 if score >= 0.5 else 0
        if predicted == 1 and label == 1:
            tp += 1
        elif predicted == 0 and label == 0:
            tn += 1
        elif predicted == 1 and label == 0:
            fp += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    false_accept_rate = fp / (fp + tn) if fp + tn else 0.0
    false_reject_rate = fn / (fn + tp) if fn + tp else 0.0
    auroc = None
    if len(set(labels)) > 1:
        auroc = round(float(roc_auc_score(labels, accept_scores)), 6)
    return {
        "rows": len(labels),
        "accuracy": round((tp + tn) / len(labels), 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "false_accept_rate": round(false_accept_rate, 6),
        "false_reject_rate": round(false_reject_rate, 6),
        "auroc": auroc,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def calibration_summary(labels: list[int], accept_scores: list[float], bins: int = 5) -> list[dict[str, Any]]:
    if not labels:
        return []
    rows: list[dict[str, Any]] = []
    for bucket in range(bins):
        lower = bucket / bins
        upper = (bucket + 1) / bins
        pairs = [(label, score) for label, score in zip(labels, accept_scores) if (lower <= score < upper or (bucket == bins - 1 and score <= upper))]
        if not pairs:
            rows.append({"bin": bucket, "lower": lower, "upper": upper, "count": 0, "mean_predicted_accept": None, "actual_accept_rate": None})
            continue
        rows.append(
            {
                "bin": bucket,
                "lower": lower,
                "upper": upper,
                "count": len(pairs),
                "mean_predicted_accept": round(sum(score for _, score in pairs) / len(pairs), 6),
                "actual_accept_rate": round(sum(label for label, _ in pairs) / len(pairs), 6),
            }
        )
    return rows


def grouped_metrics(rows: list[dict[str, Any]], predictions: list[dict[str, Any]], key: str) -> dict[str, Any]:
    by_key: dict[str, tuple[list[int], list[float]]] = {}
    for row, prediction in zip(rows, predictions):
        labels, scores = by_key.setdefault(str(row.get(key) or ""), ([], []))
        labels.append(int(row["label"]))
        scores.append(float(prediction["predicted_accept_probability"]))
    return {name: confusion_metrics(labels, scores) for name, (labels, scores) in sorted(by_key.items())}


def blocker_metrics(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    truth = [set(row.get("blocker_labels") or []) for row in rows]
    predicted = [set(prediction.get("predicted_blockers") or []) for prediction in predictions]
    if not any(truth):
        return {"available": False, "precision": None, "recall": None}
    tp = fp = fn = 0
    for expected, actual in zip(truth, predicted):
        tp += len(expected & actual)
        fp += len(actual - expected)
        fn += len(expected - actual)
    return {
        "available": True,
        "precision": round(tp / (tp + fp), 6) if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 6) if tp + fn else 0.0,
    }


def compute_target_metrics(target: str, rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    scores = [float(prediction["predicted_accept_probability"]) for prediction in predictions]
    metrics = {
        **confusion_metrics(labels, scores),
        "target": target,
        "target_name": TARGETS[target]["view_name"],
        "per_route_metrics": grouped_metrics(rows, predictions, "route_id"),
        "per_track_metrics": grouped_metrics(rows, predictions, "track_id"),
        "calibration_summary": calibration_summary(labels, scores),
        "blocker_metrics": blocker_metrics(rows, predictions),
        "model_quality_confidence": "low" if len(rows) < 500 else "normal",
        **SAFETY_FLAGS,
    }
    return metrics


def false_accept_audit(rows: list[dict[str, Any]], predictions: list[dict[str, Any]], key: str, *, accept_threshold: float, block_threshold: float) -> dict[str, Any]:
    grouped: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for row, prediction in zip(rows, predictions):
        name = str(row.get(key) or "")
        row_group, prediction_group = grouped.setdefault(name, ([], []))
        row_group.append(row)
        prediction_group.append(prediction)
    return {
        name: {
            item: value
            for item, value in thresholded_metrics(
                group_rows,
                group_predictions,
                accept_threshold=accept_threshold,
                block_threshold=block_threshold,
            ).items()
            if item != "decisions"
        }
        for name, (group_rows, group_predictions) in sorted(grouped.items())
    }


def readiness_reason(rows: list[dict[str, Any]], *, min_safe_rows: int, min_positive: int, min_negative: int) -> str:
    labels = Counter(int(row["label"]) for row in rows)
    if len(rows) < min_safe_rows:
        return "safe_rows_below_minimum"
    if labels.get(1, 0) < min_positive or labels.get(0, 0) < min_negative:
        return "missing_positive_or_negative_class"
    return ""


def train_one_target(
    *,
    target: str,
    data: dict[str, Any],
    output_dir: Path,
    model_kind: str,
    min_safe_rows: int,
    min_positive: int,
    min_negative: int,
    max_shadow_false_accept_rate: float,
    min_negative_test_rows: int,
    class_weight: str | dict[int, float] | None,
    negative_class_weight: float,
    hard_negative_blockers: Sequence[str],
    threshold_objective: str,
    enable_abstain_zone: bool,
    abstain_margin: float,
    split_strategy: str,
) -> dict[str, Any]:
    cfg = TARGETS[target]
    rows = data["usable_rows"]
    labels = Counter(int(row["label"]) for row in rows)
    reason = readiness_reason(rows, min_safe_rows=min_safe_rows, min_positive=min_positive, min_negative=min_negative)
    base = {
        "target": target,
        "target_name": cfg["view_name"],
        "rows_available": len(data["raw_rows"]),
        "rows_used": 0,
        "positive_labels": int(labels.get(1, 0)),
        "negative_labels": int(labels.get(0, 0)),
        "training_completed": False,
        "usable_for_shadow_scoring": False,
        "reason_if_not_usable": reason,
        "model_id": "",
        "model_file": cfg["model_file"],
        "model_type": "skipped_no_model",
        "metrics": {
            "target": target,
            "target_name": cfg["view_name"],
            "rows": 0,
            "false_accept_rate": None,
            "false_reject_rate": None,
            "model_quality_confidence": "not_applicable",
            "critic_usable_for_shadow_scoring": False,
            "reason_if_not_usable": reason,
            **SAFETY_FLAGS,
        },
        "eval_rows": [],
        "bundle": None,
    }
    if reason:
        write_json(output_dir / cfg["metrics_file"], base["metrics"])
        return base

    train_rows = [row for row in rows if row.get("split") == "train"]
    if not train_rows:
        train_rows = rows
    dev_rows = [row for row in rows if row.get("split") == "dev"]
    test_rows = [row for row in rows if row.get("split") == "test"]
    eval_rows = test_rows or [row for row in rows if row.get("split") in {"test", "dev"}] or rows
    vectorizer = DictVectorizer(sparse=True)
    train_matrix = vectorizer.fit_transform([row["features"] for row in train_rows])
    train_labels = [int(row["label"]) for row in train_rows]
    sample_weights, hard_negative_rows_used = sample_weights_for_rows(
        train_rows,
        negative_class_weight=negative_class_weight,
        hard_negative_blockers=hard_negative_blockers,
    )
    classifier = model_for_kind(model_kind, class_weight=class_weight)
    classifier.fit(train_matrix, train_labels, sample_weight=sample_weights)
    blocker_classifier = None
    blocker_binarizer = None
    blocker_sets = [row["blocker_labels"] for row in train_rows]
    if any(blocker_sets):
        blocker_binarizer = MultiLabelBinarizer()
        blocker_targets = blocker_binarizer.fit_transform(blocker_sets)
        blocker_classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=40, random_state=1931, class_weight="balanced"))
        blocker_classifier.fit(train_matrix, blocker_targets)
    model_id = f"{cfg['model_id_prefix']}_{stable_hash([row['row_id'] for row in rows] + [model_kind])[:12]}"
    bundle = {
        "model_id": model_id,
        "target": target,
        "target_name": cfg["view_name"],
        "model_kind": model_kind,
        "vectorizer": vectorizer,
        "accept_classifier": classifier,
        "blocker_classifier": blocker_classifier,
        "blocker_binarizer": blocker_binarizer,
        "feature_columns": list(SCALAR_FEATURE_COLUMNS) + list(CATEGORICAL_FEATURE_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "class_weight": "balanced" if class_weight == "balanced" else class_weight,
        "negative_class_weight": float(negative_class_weight),
        "hard_negative_blockers": list(hard_negative_blockers),
        "calibration": {"status": "skipped", "method": "none", "calibration_skipped_reason": "not_requested"},
        **SAFETY_FLAGS,
    }
    threshold_rows = dev_rows or train_rows
    threshold_predictions = predict_bundle(bundle, threshold_rows)
    thresholds = select_thresholds(
        threshold_rows,
        threshold_predictions,
        max_false_accept_rate=max_shadow_false_accept_rate,
        threshold_objective=threshold_objective,
        enable_abstain_zone=enable_abstain_zone,
        abstain_margin=abstain_margin,
    )
    predictions = predict_bundle(bundle, eval_rows)
    threshold_metrics = thresholded_metrics(
        eval_rows,
        predictions,
        accept_threshold=float(thresholds["accept_threshold"]),
        block_threshold=float(thresholds["block_threshold"]),
    )
    for prediction, decision in zip(predictions, threshold_metrics.get("decisions", [])):
        prediction["critic_decision"] = decision
        prediction["abstained"] = decision == "abstain"
    metrics = compute_target_metrics(target, eval_rows, predictions)
    audit = split_audit(rows, split_strategy=split_strategy, min_negative_test_rows=min_negative_test_rows)
    usability_reasons: list[str] = []
    if audit["candidate_id_overlap_train_test"]:
        usability_reasons.append("candidate_id_overlap_train_test")
    if int(audit["test_negative_rows"]) < int(min_negative_test_rows):
        usability_reasons.append("test_negative_rows_below_minimum")
    if threshold_metrics["false_accept_rate"] is None:
        usability_reasons.append("missing_false_accept_rate")
    elif float(threshold_metrics["false_accept_rate"]) > float(max_shadow_false_accept_rate):
        usability_reasons.append("false_accept_rate_above_max")
    calibration_status = "skipped"
    calibration_skipped_reason = "not_requested"
    metrics.update(
        {
            "model_id": model_id,
            "rows_used": len(rows),
            "rows_available": len(data["raw_rows"]),
            "positive_labels": int(labels.get(1, 0)),
            "negative_labels": int(labels.get(0, 0)),
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "test_rows": len(test_rows),
            "test_negative_rows": int(audit["test_negative_rows"]),
            "false_accept_rate": threshold_metrics["false_accept_rate"],
            "false_reject_rate": threshold_metrics["false_reject_rate"],
            "true_positive": threshold_metrics["true_positive"],
            "true_negative": threshold_metrics["true_negative"],
            "false_positive": threshold_metrics["false_positive"],
            "false_negative": threshold_metrics["false_negative"],
            "precision": threshold_metrics["precision"],
            "recall": threshold_metrics["recall"],
            "auroc": threshold_metrics["auroc"],
            "brier_score": threshold_metrics["brier_score"],
            "chosen_accept_threshold": threshold_metrics["accept_threshold"],
            "chosen_block_threshold": threshold_metrics["block_threshold"],
            "abstain_rate": threshold_metrics["abstain_rate"],
            "accept_count": threshold_metrics["accept_count"],
            "block_count": threshold_metrics["block_count"],
            "abstain_rows": threshold_metrics["abstain_rows"],
            "hard_negative_rows_used": hard_negative_rows_used,
            "split_audit": audit,
            "candidate_id_overlap_train_test": audit["candidate_id_overlap_train_test"],
            "track_id_overlap_train_test": audit["track_id_overlap_train_test"],
            "calibration_method": "none",
            "calibration_status": calibration_status,
            "calibration_skipped_reason": calibration_skipped_reason,
            "per_route_false_accept_audit": false_accept_audit(
                eval_rows,
                predictions,
                "route_id",
                accept_threshold=float(threshold_metrics["accept_threshold"]),
                block_threshold=float(threshold_metrics["block_threshold"]),
            ),
            "per_track_false_accept_audit": false_accept_audit(
                eval_rows,
                predictions,
                "track_id",
                accept_threshold=float(threshold_metrics["accept_threshold"]),
                block_threshold=float(threshold_metrics["block_threshold"]),
            ),
            "critic_usable_for_shadow_scoring": not usability_reasons,
            "reason_if_not_usable": ";".join(usability_reasons),
        }
    )
    thresholds_payload = {
        "target": target,
        "target_name": cfg["view_name"],
        "accept_threshold": float(threshold_metrics["accept_threshold"]),
        "block_threshold": float(threshold_metrics["block_threshold"]),
        "threshold_objective": threshold_objective,
        "max_false_accept_rate": float(max_shadow_false_accept_rate),
        "enable_abstain_zone": bool(enable_abstain_zone),
        "abstain_margin": float(abstain_margin),
        "selected_on_split": "dev" if dev_rows else "train",
        "threshold_selection_metrics": {item: value for item, value in thresholds.items() if item != "decisions"},
    }
    bundle["thresholds"] = thresholds_payload
    joblib.dump(bundle, output_dir / cfg["model_file"])
    write_json(output_dir / cfg["metrics_file"], metrics)
    return {
        **base,
        "rows_used": len(rows),
        "training_completed": True,
        "usable_for_shadow_scoring": bool(metrics["critic_usable_for_shadow_scoring"]),
        "reason_if_not_usable": str(metrics.get("reason_if_not_usable") or ""),
        "model_id": model_id,
        "model_type": f"sklearn_{model_kind}",
        "metrics": metrics,
        "thresholds": thresholds_payload,
        "split_audit": audit,
        "eval_rows": eval_rows,
        "predictions": predictions,
        "bundle": bundle,
    }


def evaluation_payload(row: dict[str, Any], prediction: dict[str, Any], *, target: str, model_id: str) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "view_name": TARGETS[target]["view_name"],
        "target_name": TARGETS[target]["view_name"],
        "row_id": row["row_id"],
        "track_id": row["track_id"],
        "candidate_id": row["candidate_id"],
        "route_id": row["route_id"],
        "stem_type": row["stem_type"],
        "window_start_sec": row["window_start_sec"],
        "window_end_sec": row["window_end_sec"],
        "true_label": int(row["label"]),
        "predicted_accept_probability": prediction["predicted_accept_probability"],
        "predicted_block_probability": prediction["predicted_block_probability"],
        "predicted_blockers": prediction["predicted_blockers"],
        "critic_decision": prediction["critic_decision"],
        "confidence": prediction["confidence"],
        "split": row.get("split", ""),
        "shadow_only": True,
        "release_ready": False,
        "separator_training_started": False,
    }


def manifest_from_results(
    *,
    views_dir: Path,
    output_dir: Path,
    targets: tuple[str, ...],
    model_kind: str,
    group_split: str,
    shadow_only: bool,
    data_by_target: dict[str, dict[str, Any]],
    result_by_target: dict[str, dict[str, Any]],
    min_safe_rows: int,
    min_positive: int,
    min_negative: int,
    min_negative_test_rows: int,
    max_false_accept_rate: float,
) -> dict[str, Any]:
    model_ids = {}
    model_types = {}
    for target in targets:
        result = result_by_target[target]
        cfg = TARGETS[target]
        model_ids[f"{target}_accept_block_classifier"] = result["model_id"]
        model_types[f"{target}_accept_block_classifier"] = result["model_type"]
        if result["bundle"] and result["bundle"].get("blocker_classifier") is not None:
            model_ids[f"{target}_blocker_multilabel_classifier"] = result["model_id"] + "_blockers"
            model_types[f"{target}_blocker_multilabel_classifier"] = f"sklearn_{model_kind}_multilabel"
        else:
            model_ids[f"{target}_blocker_multilabel_classifier"] = ""
            model_types[f"{target}_blocker_multilabel_classifier"] = "not_trained_no_blocker_labels"

    manifest = {
        "schema": "target_audio_error_critics.v1",
        "views_dir": str(views_dir),
        "output_dir": str(output_dir),
        "model_ids": model_ids,
        "model_types": model_types,
        "targets_requested": list(targets),
        "minus_rows_available": len(data_by_target.get("minus", {}).get("raw_rows", [])),
        "minus_rows_used": result_by_target.get("minus", {}).get("rows_used", 0),
        "minus_positive_labels": result_by_target.get("minus", {}).get("positive_labels", 0),
        "minus_negative_labels": result_by_target.get("minus", {}).get("negative_labels", 0),
        "vocal_rows_available": len(data_by_target.get("vocal", {}).get("raw_rows", [])),
        "vocal_rows_used": result_by_target.get("vocal", {}).get("rows_used", 0),
        "vocal_positive_labels": result_by_target.get("vocal", {}).get("positive_labels", 0),
        "vocal_negative_labels": result_by_target.get("vocal", {}).get("negative_labels", 0),
        "group_split_strategy": group_split,
        "feature_columns": list(SCALAR_FEATURE_COLUMNS) + list(CATEGORICAL_FEATURE_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "minimum_readiness": {"safe_rows": min_safe_rows, "positive_labels": min_positive, "negative_labels": min_negative},
        "minimum_reliability": {"test_negative_rows": min_negative_test_rows, "max_false_accept_rate": max_false_accept_rate},
        "excluded_reason_counts": {TARGETS[target]["view_name"]: data_by_target[target]["excluded_reason_counts"] for target in targets},
        "minus_critic_training_completed": result_by_target.get("minus", {}).get("training_completed", False),
        "vocal_critic_training_completed": result_by_target.get("vocal", {}).get("training_completed", False),
        "minus_critic_usable_for_shadow_scoring": result_by_target.get("minus", {}).get("usable_for_shadow_scoring", False),
        "vocal_critic_usable_for_shadow_scoring": result_by_target.get("vocal", {}).get("usable_for_shadow_scoring", False),
        "reason_if_minus_not_usable": result_by_target.get("minus", {}).get("reason_if_not_usable", "target_not_requested"),
        "reason_if_vocal_not_usable": result_by_target.get("vocal", {}).get("reason_if_not_usable", "target_not_requested"),
        "minus_test_negative_rows": result_by_target.get("minus", {}).get("metrics", {}).get("test_negative_rows", 0),
        "vocal_test_negative_rows": result_by_target.get("vocal", {}).get("metrics", {}).get("test_negative_rows", 0),
        "thresholds_file": THRESHOLDS_FILE,
        "reliability_report_file": RELIABILITY_REPORT_FILE,
        "shadow_only": bool(shadow_only),
        "can_promote_to_production_gate": False,
        "critic_training_started": True,
        **SAFETY_FLAGS,
    }
    manifest["shadow_only"] = bool(shadow_only)
    return manifest


def thresholds_from_results(output_dir: Path, targets: tuple[str, ...], result_by_target: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "target_audio_error_critic_thresholds.v1",
        "output_dir": str(output_dir),
        "targets": {
            target: result_by_target.get(target, {}).get(
                "thresholds",
                {
                    "target": target,
                    "target_name": TARGETS[target]["view_name"],
                    "accept_threshold": None,
                    "block_threshold": None,
                    "threshold_objective": "not_selected",
                },
            )
            for target in targets
        },
        **SAFETY_FLAGS,
    }


def reliability_report_from_results(
    *,
    output_dir: Path,
    targets: tuple[str, ...],
    result_by_target: dict[str, dict[str, Any]],
    min_negative_test_rows: int,
    max_false_accept_rate: float,
) -> dict[str, Any]:
    target_reports: dict[str, Any] = {}
    for target in targets:
        result = result_by_target.get(target, {})
        metrics = result.get("metrics", {})
        target_reports[target] = {
            "target": target,
            "target_name": TARGETS[target]["view_name"],
            "rows_used": result.get("rows_used", 0),
            "positives": result.get("positive_labels", 0),
            "negatives": result.get("negative_labels", 0),
            "train_rows": metrics.get("train_rows", 0),
            "dev_rows": metrics.get("dev_rows", 0),
            "test_rows": metrics.get("test_rows", 0),
            "test_negative_rows": metrics.get("test_negative_rows", 0),
            "candidate_id_overlap_train_test": metrics.get("candidate_id_overlap_train_test", False),
            "track_id_overlap_train_test": metrics.get("track_id_overlap_train_test", False),
            "chosen_accept_threshold": metrics.get("chosen_accept_threshold"),
            "chosen_block_threshold": metrics.get("chosen_block_threshold"),
            "abstain_rate": metrics.get("abstain_rate"),
            "false_accept_rate": metrics.get("false_accept_rate"),
            "false_reject_rate": metrics.get("false_reject_rate"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "auroc": metrics.get("auroc"),
            "brier_score": metrics.get("brier_score"),
            "calibration_method": metrics.get("calibration_method", "none"),
            "calibration_status": metrics.get("calibration_status", "skipped"),
            "calibration_skipped_reason": metrics.get("calibration_skipped_reason", "not_requested"),
            "hard_negative_rows_used": metrics.get("hard_negative_rows_used", 0),
            "per_route_false_accept_audit": metrics.get("per_route_false_accept_audit", {}),
            "per_track_false_accept_audit": metrics.get("per_track_false_accept_audit", {}),
            "split_audit": metrics.get("split_audit", result.get("split_audit", {})),
            "critic_usable_for_shadow_scoring": result.get("usable_for_shadow_scoring", False),
            "reason_if_not_usable": result.get("reason_if_not_usable", ""),
        }
    return {
        "schema": "target_audio_error_critic_reliability_report.v1",
        "output_dir": str(output_dir),
        "max_false_accept_rate": float(max_false_accept_rate),
        "min_negative_test_rows": int(min_negative_test_rows),
        "targets": target_reports,
        **SAFETY_FLAGS,
    }


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Target Audio Error Critics",
        "",
        f"- views_dir: `{manifest['views_dir']}`",
        f"- minus_rows_used: {manifest['minus_rows_used']}",
        f"- vocal_rows_used: {manifest['vocal_rows_used']}",
        f"- minus_critic_training_completed: {str(manifest['minus_critic_training_completed']).lower()}",
        f"- vocal_critic_training_completed: {str(manifest['vocal_critic_training_completed']).lower()}",
        f"- minus_critic_usable_for_shadow_scoring: {str(manifest['minus_critic_usable_for_shadow_scoring']).lower()}",
        f"- vocal_critic_usable_for_shadow_scoring: {str(manifest['vocal_critic_usable_for_shadow_scoring']).lower()}",
        f"- minus_test_negative_rows: {manifest.get('minus_test_negative_rows', 0)}",
        f"- vocal_test_negative_rows: {manifest.get('vocal_test_negative_rows', 0)}",
        f"- max_false_accept_rate: {manifest.get('minimum_reliability', {}).get('max_false_accept_rate')}",
        "- unsafe_rows_used: false",
        "- pair_rows_used: false",
        "- overall_rows_used: false",
        "- shadow_only: true",
        "- separator_training_started: false",
        "- release_ready: false",
        "- melody_transcription_ran: false",
        "- recommendation_index_updated: false",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def train_target_audio_error_critics(
    *,
    views_dir: Path,
    output_dir: Path,
    targets: Sequence[str] = ("minus", "vocal"),
    model_kind: str = "random_forest",
    group_split: str = "candidate_id",
    shadow_only: bool = True,
    min_safe_rows: int = 100,
    min_positive: int = 20,
    min_negative: int = 20,
    max_shadow_false_accept_rate: float | None = None,
    max_false_accept_rate: float = 0.10,
    min_negative_test_rows: int = 20,
    class_weight: str | dict[int, float] | None = "balanced",
    negative_class_weight: float = 3.0,
    hard_negative_blockers: Sequence[str] = (
        "severe_phase_collapse",
        "severe_vocal_bleed_in_minus",
        "severe_phase_comb_filtering",
    ),
    threshold_objective: str = "false_accept_first",
    enable_abstain_zone: bool = True,
    abstain_margin: float = 0.10,
) -> dict[str, Any]:
    selected_targets = parse_targets(targets)
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_max_false_accept_rate = float(max_shadow_false_accept_rate if max_shadow_false_accept_rate is not None else max_false_accept_rate)
    class_weight_value = parse_class_weight(class_weight if isinstance(class_weight, str) or class_weight is None else "balanced")
    hard_negative_blockers = parse_blockers(hard_negative_blockers)
    data_by_target = {
        target: load_target_rows(
            views_dir,
            target,
            group_split=group_split,
            min_negative_test_rows=min_negative_test_rows,
            min_negative_dev_rows=min_negative_test_rows,
            min_negative_train_rows=min_negative,
        )
        for target in selected_targets
    }
    result_by_target = {
        target: train_one_target(
            target=target,
            data=data_by_target[target],
            output_dir=output_dir,
            model_kind=model_kind,
            min_safe_rows=min_safe_rows,
            min_positive=min_positive,
            min_negative=min_negative,
            max_shadow_false_accept_rate=effective_max_false_accept_rate,
            min_negative_test_rows=min_negative_test_rows,
            class_weight=class_weight_value,
            negative_class_weight=negative_class_weight,
            hard_negative_blockers=hard_negative_blockers,
            threshold_objective=threshold_objective,
            enable_abstain_zone=enable_abstain_zone,
            abstain_margin=abstain_margin,
            split_strategy=group_split,
        )
        for target in selected_targets
    }
    eval_rows: list[dict[str, Any]] = []
    for target, result in result_by_target.items():
        for row, prediction in zip(result.get("eval_rows", []), result.get("predictions", [])):
            eval_rows.append(evaluation_payload(row, prediction, target=target, model_id=result["model_id"]))
    write_jsonl(output_dir / EVAL_FILE, eval_rows)
    manifest = manifest_from_results(
        views_dir=views_dir,
        output_dir=output_dir,
        targets=selected_targets,
        model_kind=model_kind,
        group_split=group_split,
        shadow_only=shadow_only,
        data_by_target=data_by_target,
        result_by_target=result_by_target,
        min_safe_rows=min_safe_rows,
        min_positive=min_positive,
        min_negative=min_negative,
        min_negative_test_rows=min_negative_test_rows,
        max_false_accept_rate=effective_max_false_accept_rate,
    )
    thresholds = thresholds_from_results(output_dir, selected_targets, result_by_target)
    reliability_report = reliability_report_from_results(
        output_dir=output_dir,
        targets=selected_targets,
        result_by_target=result_by_target,
        min_negative_test_rows=min_negative_test_rows,
        max_false_accept_rate=effective_max_false_accept_rate,
    )
    write_json(output_dir / THRESHOLDS_FILE, thresholds)
    write_json(output_dir / RELIABILITY_REPORT_FILE, reliability_report)
    write_json(output_dir / MANIFEST_FILE, manifest)
    write_summary(output_dir / SUMMARY_FILE, manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train shadow-only target-specific minus/vocal audio error critics.")
    parser.add_argument("--views-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", default="minus,vocal")
    parser.add_argument("--model-kind", choices=("random_forest", "logistic_regression"), default="random_forest")
    parser.add_argument("--group-split", default="candidate_id")
    parser.add_argument("--shadow-only", type=parse_bool, default=True)
    parser.add_argument("--min-safe-rows", type=int, default=100)
    parser.add_argument("--min-positive", type=int, default=20)
    parser.add_argument("--min-negative", type=int, default=20)
    parser.add_argument("--max-shadow-false-accept-rate", type=float, default=None)
    parser.add_argument("--max-false-accept-rate", type=float, default=0.10)
    parser.add_argument("--min-negative-test-rows", type=int, default=20)
    parser.add_argument("--class-weight", default="balanced")
    parser.add_argument("--negative-class-weight", type=float, default=3.0)
    parser.add_argument("--hard-negative-blockers", default="severe_phase_collapse,severe_vocal_bleed_in_minus,severe_phase_comb_filtering")
    parser.add_argument("--threshold-objective", default="false_accept_first")
    parser.add_argument("--enable-abstain-zone", type=parse_bool, default=True)
    parser.add_argument("--abstain-margin", type=float, default=0.10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    train_target_audio_error_critics(
        views_dir=args.views_dir,
        output_dir=args.output_dir,
        targets=parse_targets(args.targets),
        model_kind=args.model_kind,
        group_split=args.group_split,
        shadow_only=args.shadow_only,
        min_safe_rows=args.min_safe_rows,
        min_positive=args.min_positive,
        min_negative=args.min_negative,
        max_shadow_false_accept_rate=args.max_shadow_false_accept_rate,
        max_false_accept_rate=args.max_false_accept_rate,
        min_negative_test_rows=args.min_negative_test_rows,
        class_weight=args.class_weight,
        negative_class_weight=args.negative_class_weight,
        hard_negative_blockers=parse_blockers(args.hard_negative_blockers),
        threshold_objective=args.threshold_objective,
        enable_abstain_zone=args.enable_abstain_zone,
        abstain_margin=args.abstain_margin,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
