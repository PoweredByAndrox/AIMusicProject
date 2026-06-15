from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REQUIRED_MAP_NAMES = (
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
SAFETY_FLAGS = {
    "separator_training_started": False,
    "repair_adapter_training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "thresholds_relaxed": False,
    "release_ready": False,
    "can_publish_as_result": False,
}
MINING_BUNDLE_MANIFEST_FILE = "tensor_audio_error_mining_bundle_manifest.json"
MINING_TRAINING_CANDIDATES_FILE = "tensor_audio_error_training_candidates.jsonl"
TRAIN_ONLY_MINING_BUNDLE_MANIFEST_FILE = "tensor_audio_error_train_only_mining_bundle_manifest.json"
TRAIN_ONLY_MINING_TRAINING_CANDIDATES_FILE = "tensor_audio_error_train_only_training_candidates.jsonl"
LOCKED_MINING_SLICE_MANIFEST_FILE = "tensor_audio_error_locked_mining_slice_manifest.json"
HARD_NEGATIVE_HARVEST_MANIFEST_FILE = "tensor_audio_error_train_hard_negative_harvest_manifest.json"
FORBIDDEN_MINING_TRUE_FLAGS = {
    "release_ready",
    "can_publish_as_result",
    "runtime_gate_decision_applied",
    "runtime_gates_changed",
    "training_started",
    "critic_training_started",
    "separator_training_started",
    "repair_adapter_training_started",
    "fine_tune_started",
    "models_changed",
    "model_registry_edited",
    "thresholds_relaxed",
}
ALLOWED_MINING_SPLITS = {"train", "dev", "test"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected_json_object:{path}:{line_number}")
        yield payload


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def has_forbidden_true(payload: Any) -> bool:
    if isinstance(payload, dict):
        return any(payload.get(key) is True for key in FORBIDDEN_MINING_TRUE_FLAGS) or any(
            has_forbidden_true(value) for value in payload.values()
        )
    if isinstance(payload, list):
        return any(has_forbidden_true(item) for item in payload)
    return False


def empty_mining_report() -> dict[str, Any]:
    return {
        "mining_bundle_used": False,
        "mining_bundle_dir": "",
        "mining_bundle_schema": "",
        "mining_bundle_status": "",
        "mining_candidate_rows_total": 0,
        "mining_train_rows_available": 0,
        "mining_non_train_rows_ignored": 0,
        "mining_train_rows_matched": 0,
        "mining_train_rows_unmatched": 0,
        "mining_sample_weight_applied": False,
        "mining_weight_multiplier": 1.0,
        "dev_test_mining_rows_used_for_training": False,
        "locked_eval_rows_used_for_training": False,
        "hardcase_rows_used_for_training": False,
    }


def row_match_keys(row: dict[str, Any]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    row_uid = str(row.get("row_uid") or "").strip()
    if row_uid:
        keys.add(("row_uid", row_uid))
    canonical_row_key = str(row.get("canonical_row_key") or "").strip()
    if canonical_row_key:
        keys.add(("canonical_row_key", canonical_row_key))
    candidate_id = str(row.get("candidate_id") or "").strip()
    stem_type = str(row.get("stem_type") or "").strip()
    window_start = str(float(row.get("window_start_sec") or 0.0))
    window_end = str(float(row.get("window_end_sec") or 0.0))
    if candidate_id and stem_type:
        keys.add(("candidate_window", "|".join([candidate_id, stem_type, window_start, window_end])))
    return keys


def validate_legacy_mining_bundle(mining_bundle_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = read_json(mining_bundle_dir / MINING_BUNDLE_MANIFEST_FILE)
    rows = list(iter_jsonl(mining_bundle_dir / MINING_TRAINING_CANDIDATES_FILE))
    errors: list[str] = []
    if manifest.get("schema") != "tensor_audio_error_mining_bundle.v1":
        errors.append("invalid_mining_bundle_schema")
    if manifest.get("bundle_status") != "candidate_only":
        errors.append("mining_bundle_not_candidate_only")
    if manifest.get("shadow_only") is not True:
        errors.append("mining_bundle_not_shadow_only")
    if has_forbidden_true(manifest):
        errors.append("mining_bundle_forbidden_true_flag")
    if int(manifest.get("candidate_rows_total") or 0) != len(rows):
        errors.append("mining_bundle_row_count_mismatch")
    for index, row in enumerate(rows):
        if row.get("bundle_row_schema") != "tensor_audio_error_training_candidate.v1":
            errors.append(f"invalid_mining_row_schema:{index}")
        if row.get("training_candidate_status") != "candidate_only":
            errors.append(f"mining_row_not_candidate_only:{index}")
        if row.get("shadow_only") is not True:
            errors.append(f"mining_row_not_shadow_only:{index}")
        if has_forbidden_true(row):
            errors.append(f"mining_row_forbidden_true_flag:{index}")
        if str(row.get("split") or "") not in ALLOWED_MINING_SPLITS:
            errors.append(f"invalid_mining_split:{index}")
        mining_label = str(row.get("source_mining_label") or "")
        recommended_label = int(row.get("recommended_training_label") or 0)
        if mining_label == "hard_negative_false_accept" and recommended_label != 0:
            errors.append(f"hard_negative_label_mismatch:{index}")
        elif mining_label == "false_reject_positive" and recommended_label != 1:
            errors.append(f"false_reject_label_mismatch:{index}")
        elif mining_label not in {"hard_negative_false_accept", "false_reject_positive"}:
            errors.append(f"unknown_mining_label:{index}")
    if errors:
        raise ValueError(f"unsafe_mining_bundle:{','.join(errors)}")
    return rows, {
        "mining_bundle_used": True,
        "mining_bundle_dir": str(mining_bundle_dir),
        "mining_bundle_schema": str(manifest.get("schema") or ""),
        "mining_bundle_status": str(manifest.get("bundle_status") or ""),
        "mining_candidate_rows_total": len(rows),
        "dev_test_mining_rows_used_for_training": False,
        "locked_eval_rows_used_for_training": False,
        "hardcase_rows_used_for_training": False,
    }


def validate_train_only_mining_bundle(mining_bundle_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = read_json(mining_bundle_dir / TRAIN_ONLY_MINING_BUNDLE_MANIFEST_FILE)
    rows = list(iter_jsonl(mining_bundle_dir / TRAIN_ONLY_MINING_TRAINING_CANDIDATES_FILE))
    errors: list[str] = []
    if manifest.get("schema") != "tensor_audio_error_train_only_mining_bundle.v1":
        errors.append("invalid_train_only_mining_bundle_schema")
    if manifest.get("bundle_status") != "train_only_candidate_only":
        errors.append("train_only_mining_bundle_not_usable")
    if manifest.get("training_allowed") is not True:
        errors.append("train_only_mining_bundle_training_allowed_not_true")
    if manifest.get("shadow_only") is not True:
        errors.append("train_only_mining_bundle_not_shadow_only")
    if manifest.get("tensor_maps_required") is not True:
        errors.append("train_only_mining_bundle_tensor_maps_required_not_true")
    if manifest.get("scalar_features_used") is True:
        errors.append("train_only_mining_bundle_scalar_features_used_true")
    if manifest.get("dev_test_mining_rows_used_for_training") is True:
        errors.append("dev_test_mining_rows_used_for_training")
    if manifest.get("locked_eval_rows_used_for_training") is True:
        errors.append("locked_eval_rows_used_for_training")
    if manifest.get("hardcase_rows_used_for_training") is True:
        errors.append("hardcase_rows_used_for_training")
    if has_forbidden_true(manifest):
        errors.append("train_only_mining_bundle_forbidden_true_flag")
    if int(manifest.get("candidate_rows_total") or 0) != len(rows):
        errors.append("train_only_mining_bundle_row_count_mismatch")
    for index, row in enumerate(rows):
        if row.get("training_candidate_schema") != "tensor_audio_error_train_only_training_candidate.v1":
            errors.append(f"invalid_train_only_mining_row_schema:{index}")
        if row.get("bundle_row_schema") != "tensor_audio_error_train_only_training_candidate.v1":
            errors.append(f"invalid_train_only_mining_bundle_row_schema:{index}")
        if row.get("training_candidate_status") != "candidate_only":
            errors.append(f"train_only_mining_row_not_candidate_only:{index}")
        if row.get("eligible_for_future_training_review") is not True:
            errors.append(f"train_only_mining_row_not_review_eligible:{index}")
        if row.get("shadow_only") is not True:
            errors.append(f"train_only_mining_row_not_shadow_only:{index}")
        if has_forbidden_true(row):
            errors.append(f"train_only_mining_row_forbidden_true_flag:{index}")
        if str(row.get("split") or "") != "train":
            errors.append(f"train_only_mining_row_split_not_train:{index}")
        if row.get("train_allowed") is not True:
            errors.append(f"train_only_mining_row_train_allowed_not_true:{index}")
        if row.get("hardcase_holdout") is True or row.get("hardcase") is True:
            errors.append(f"train_only_mining_row_hardcase_true:{index}")
        if row.get("locked_eval_row") is True:
            errors.append(f"train_only_mining_row_locked_eval_true:{index}")
        if row.get("scalar_features_used") is True:
            errors.append(f"train_only_mining_row_scalar_features_used_true:{index}")
        tensor_map_path = Path(str(row.get("tensor_map_path") or ""))
        if not tensor_map_path.exists():
            errors.append(f"train_only_mining_row_missing_tensor_map:{index}")
        mining_label = str(row.get("source_mining_label") or "")
        recommended_label = int(row.get("recommended_training_label") or 0)
        if mining_label == "hard_negative_false_accept" and recommended_label != 0:
            errors.append(f"hard_negative_label_mismatch:{index}")
        elif mining_label == "false_reject_positive" and recommended_label != 1:
            errors.append(f"false_reject_label_mismatch:{index}")
        elif mining_label not in {"hard_negative_false_accept", "false_reject_positive"}:
            errors.append(f"unknown_mining_label:{index}")
    if errors:
        raise ValueError(f"unsafe_mining_bundle:{','.join(errors)}")
    return rows, {
        "mining_bundle_used": True,
        "mining_bundle_dir": str(mining_bundle_dir),
        "mining_bundle_schema": str(manifest.get("schema") or ""),
        "mining_bundle_status": str(manifest.get("bundle_status") or ""),
        "mining_candidate_rows_total": len(rows),
        "dev_test_mining_rows_used_for_training": False,
        "locked_eval_rows_used_for_training": False,
        "hardcase_rows_used_for_training": False,
    }


def validate_mining_bundle(mining_bundle_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if (mining_bundle_dir / LOCKED_MINING_SLICE_MANIFEST_FILE).exists():
        raise ValueError("unsafe_mining_bundle:locked_eval_mining_slice_not_training_bundle")
    if (mining_bundle_dir / HARD_NEGATIVE_HARVEST_MANIFEST_FILE).exists():
        raise ValueError("unsafe_mining_bundle:hard_negative_harvest_not_training_bundle")
    if (mining_bundle_dir / TRAIN_ONLY_MINING_BUNDLE_MANIFEST_FILE).exists():
        return validate_train_only_mining_bundle(mining_bundle_dir)
    if (mining_bundle_dir / MINING_BUNDLE_MANIFEST_FILE).exists():
        return validate_legacy_mining_bundle(mining_bundle_dir)
    raise ValueError("unsafe_mining_bundle:missing_supported_mining_bundle_manifest")


def mining_sample_weights(
    train_rows: Sequence[dict[str, Any]],
    *,
    mining_bundle_dir: Path | None,
    mining_weight_multiplier: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    report = empty_mining_report()
    if mining_bundle_dir is None:
        return None, report
    if float(mining_weight_multiplier) < 1.0:
        raise ValueError("mining_weight_multiplier_must_be_at_least_1")
    mining_rows, bundle_report = validate_mining_bundle(mining_bundle_dir)
    train_mining_rows = [row for row in mining_rows if str(row.get("split") or "") == "train"]
    train_key_to_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in train_mining_rows:
        for key in row_match_keys(row):
            train_key_to_rows.setdefault(key, []).append(row)
    weights = np.ones(len(train_rows), dtype=np.float32)
    matched_train_keys: set[tuple[str, str]] = set()
    matched_rows = 0
    label_errors: list[str] = []
    for index, row in enumerate(train_rows):
        keys = row_match_keys(row)
        matched_keys = keys & set(train_key_to_rows)
        if matched_keys:
            row_label = int(row.get("label") or 0)
            for matched_key in matched_keys:
                for mining_row in train_key_to_rows[matched_key]:
                    recommended_label = int(mining_row.get("recommended_training_label") or 0)
                    source_true_label = mining_row.get("source_true_label")
                    if row_label != recommended_label:
                        label_errors.append(f"mining_label_mismatch_for_matched_row:{index}")
                    if source_true_label is not None and row_label != int(source_true_label or 0):
                        label_errors.append(f"mining_source_true_label_mismatch_for_matched_row:{index}")
            weights[index] = float(mining_weight_multiplier)
            matched_train_keys.update(matched_keys)
            matched_rows += 1
    if label_errors:
        raise ValueError(f"unsafe_mining_bundle:{','.join(sorted(set(label_errors)))}")
    unmatched = sum(1 for row in train_mining_rows if not (row_match_keys(row) & matched_train_keys))
    report.update(
        {
            **bundle_report,
            "mining_train_rows_available": len(train_mining_rows),
            "mining_non_train_rows_ignored": len(mining_rows) - len(train_mining_rows),
            "mining_train_rows_matched": matched_rows,
            "mining_train_rows_unmatched": unmatched,
            "mining_sample_weight_applied": matched_rows > 0,
            "mining_weight_multiplier": float(mining_weight_multiplier),
        }
    )
    return weights if matched_rows else None, report


def pooled_grid(array: np.ndarray, bins: int = 8, frames: int = 8) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    row_idx = np.linspace(0, max(0, arr.shape[0] - 1), num=bins).round().astype(int)
    col_idx = np.linspace(0, max(0, arr.shape[1] - 1), num=frames).round().astype(int)
    return arr[np.ix_(row_idx, col_idx)].astype(np.float32)


def load_tensor_features(path: Path) -> tuple[np.ndarray, dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(path)
    features: list[np.ndarray] = []
    channel_scores: dict[str, float] = {}
    with np.load(path) as payload:
        missing = [name for name in REQUIRED_MAP_NAMES if name not in payload.files]
        if missing:
            raise ValueError(f"missing_required_tensor_maps:{','.join(missing)}")
        for name in REQUIRED_MAP_NAMES:
            arr = np.nan_to_num(np.asarray(payload[name], dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
            channel_scores[name] = float(np.mean(np.abs(arr))) if arr.size else 0.0
            stats = np.asarray(
                [
                    float(np.mean(arr)) if arr.size else 0.0,
                    float(np.std(arr)) if arr.size else 0.0,
                    float(np.max(arr)) if arr.size else 0.0,
                    float(np.percentile(arr, 90)) if arr.size else 0.0,
                ],
                dtype=np.float32,
            )
            features.extend([stats, pooled_grid(arr).reshape(-1)])
    return np.concatenate(features).astype(np.float32), channel_scores


def load_rows(dataset_dir: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    for row in iter_jsonl(dataset_dir / "tensor_audio_error_dataset.jsonl"):
        if row.get("release_ready") is True:
            excluded["release_ready"] += 1
            continue
        if row.get("scalar_features_used") is True:
            excluded["scalar_features_used"] += 1
            continue
        tensor_map_path = Path(str(row.get("tensor_map_path") or ""))
        try:
            features, channel_scores = load_tensor_features(tensor_map_path)
        except Exception as exc:  # noqa: BLE001
            excluded[f"invalid_tensor_map:{type(exc).__name__}"] += 1
            continue
        rows.append({**row, "features": features, "channel_scores": channel_scores})
    return rows, excluded


def metrics_for_rows(rows: Sequence[dict[str, Any]], probabilities: Sequence[float]) -> dict[str, Any]:
    labels = [int(row.get("label") or 0) for row in rows]
    predicted = [1 if float(score) >= 0.5 else 0 for score in probabilities]
    positives = sum(labels)
    negatives = len(labels) - positives
    false_accept = sum(1 for label, pred in zip(labels, predicted) if label == 0 and pred == 1)
    false_reject = sum(1 for label, pred in zip(labels, predicted) if label == 1 and pred == 0)
    try:
        auroc = float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None
    except ValueError:
        auroc = None
    return {
        "rows": len(rows),
        "positive_rows": positives,
        "negative_rows": negatives,
        "accuracy": float(accuracy_score(labels, predicted)) if rows else None,
        "auroc": auroc,
        "false_accept": false_accept,
        "false_reject": false_reject,
        "false_accept_rate": float(false_accept / negatives) if negatives else None,
        "false_reject_rate": float(false_reject / positives) if positives else None,
    }


def conservative_threshold_metrics(rows: Sequence[dict[str, Any]], probabilities: Sequence[float]) -> dict[str, Any]:
    labels = [int(row.get("label") or 0) for row in rows]
    negatives = [float(score) for label, score in zip(labels, probabilities) if label == 0]
    positives = [float(score) for label, score in zip(labels, probabilities) if label == 1]
    threshold = min(1.000001, max(negatives) + 1.0e-6) if negatives else 1.000001
    false_accept = sum(1 for score in negatives if score >= threshold)
    false_reject = sum(1 for score in positives if score < threshold)
    true_accept = sum(1 for score in positives if score >= threshold)
    return {
        "safe_accept_threshold": float(threshold),
        "positive_rows": len(positives),
        "negative_rows": len(negatives),
        "accepted_rows": int(true_accept + false_accept),
        "true_accept": int(true_accept),
        "threshold_false_accept": int(false_accept),
        "threshold_false_accept_rate": float(false_accept / len(negatives)) if negatives else None,
        "threshold_false_reject": int(false_reject),
        "threshold_false_reject_rate": float(false_reject / len(positives)) if positives else None,
        "accept_coverage": float(true_accept / len(positives)) if positives else 0.0,
    }


def model_selection_key(candidate: tuple[str, Any, np.ndarray, dict[str, Any], dict[str, Any]]) -> tuple[float, float, float]:
    _, _, _, eval_metrics, threshold_metrics = candidate
    return (
        float((threshold_metrics or {}).get("accept_coverage") or 0.0),
        float((eval_metrics or {}).get("auroc") or 0.0),
        float((eval_metrics or {}).get("accuracy") or 0.0),
    )


def localization_payload(row: dict[str, Any], probability: float) -> dict[str, Any]:
    scores = {name: float(value) for name, value in (row.get("channel_scores") or {}).items()}
    top_channel = max(scores, key=scores.get) if scores else ""
    return {
        "row_uid": row.get("row_uid"),
        "track_id": row.get("track_id"),
        "candidate_id": row.get("candidate_id"),
        "stem_type": row.get("stem_type"),
        "split": row.get("split"),
        "label": int(row.get("label") or 0),
        "predicted_accept_probability": round(float(probability), 6),
        "top_error_channel": top_channel,
        "channel_scores": scores,
        "release_ready": False,
    }


def fit_model(model_kind: str, model: Any, train_x: np.ndarray, train_y: np.ndarray, sample_weight: np.ndarray | None) -> None:
    if sample_weight is None:
        model.fit(train_x, train_y)
    elif model_kind == "logistic_regression_balanced":
        model.fit(train_x, train_y, logisticregression__sample_weight=sample_weight)
    else:
        model.fit(train_x, train_y, sample_weight=sample_weight)


def train_tensor_audio_error_critic(
    *,
    dataset_dir: Path,
    output_dir: Path,
    min_train_rows: int = 20,
    mining_bundle_dir: Path | None = None,
    mining_weight_multiplier: float = 3.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = read_json(dataset_dir / "locked_tensor_audio_error_dataset_manifest.json")
    rows, excluded = load_rows(dataset_dir)
    train_rows = [row for row in rows if row.get("split") == "train" and row.get("train_allowed") is True]
    eval_rows = [row for row in rows if row.get("split") in {"dev", "test"}]
    sample_weight, mining_report = mining_sample_weights(
        train_rows,
        mining_bundle_dir=mining_bundle_dir,
        mining_weight_multiplier=mining_weight_multiplier,
    )
    train_labels = [int(row.get("label") or 0) for row in train_rows]
    reason = ""
    if source_manifest.get("tensor_maps_required") is not True or source_manifest.get("scalar_fallback_used") is True:
        reason = "source_dataset_not_tensor_only"
    elif int(source_manifest.get("critical_leakage_count") or 0) != 0:
        reason = "critical_leakage_present"
    elif len(train_rows) < int(min_train_rows):
        reason = "train_rows_below_minimum"
    elif len(set(train_labels)) < 2:
        reason = "missing_train_positive_or_negative_class"
    elif not eval_rows:
        reason = "missing_locked_eval_rows"

    if reason:
        manifest = {
            "schema": "tensor_audio_error_critic.v1",
            "source_dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "training_completed": False,
            "critic_not_usable_reason": reason,
            "rows_loaded": len(rows),
            "train_rows": len(train_rows),
            "excluded_reason_counts": dict(sorted(excluded.items())),
            "tensor_maps_required": True,
            "scalar_fallback_used": False,
            "critic_training_started": False,
            "shadow_only": True,
            **mining_report,
            **SAFETY_FLAGS,
        }
        write_json(output_dir / "tensor_audio_error_critic_manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
        return manifest

    train_x = np.vstack([row["features"] for row in train_rows])
    train_y = np.asarray(train_labels, dtype=np.int32)
    eval_target_rows = eval_rows
    eval_x = np.vstack([row["features"] for row in eval_target_rows])
    candidates: list[tuple[str, Any, np.ndarray, dict[str, Any], dict[str, Any]]] = []
    model_specs = [
        (
            "logistic_regression_balanced",
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=1729)),
        ),
        (
            "random_forest_balanced",
            RandomForestClassifier(
                n_estimators=160,
                random_state=1729,
                class_weight="balanced",
                min_samples_leaf=2,
                max_features="sqrt",
            ),
        ),
    ]
    for model_kind, model in model_specs:
        fit_model(model_kind, model, train_x, train_y, sample_weight)
        candidate_probabilities = model.predict_proba(eval_x)[:, 1].astype(float)
        candidate_metrics = metrics_for_rows(eval_target_rows, candidate_probabilities)
        candidate_threshold_metrics = conservative_threshold_metrics(eval_target_rows, candidate_probabilities)
        candidates.append((model_kind, model, candidate_probabilities, candidate_metrics, candidate_threshold_metrics))
    model_kind, model, probabilities, eval_metrics, threshold_metrics = max(candidates, key=model_selection_key)
    eval_payloads = []
    localization_rows = []
    for row, probability in zip(eval_target_rows, probabilities):
        eval_payloads.append(
            {
                "row_uid": row.get("row_uid"),
                "track_id": row.get("track_id"),
                "candidate_id": row.get("candidate_id"),
                "stem_type": row.get("stem_type"),
                "split": row.get("split"),
                "label": int(row.get("label") or 0),
                "predicted_accept_probability": round(float(probability), 6),
                "release_ready": False,
            }
        )
        localization_rows.append(localization_payload(row, float(probability)))
    joblib.dump({"model": model, "required_map_names": REQUIRED_MAP_NAMES}, output_dir / "tensor_audio_error_critic.joblib")
    write_jsonl(output_dir / "tensor_audio_error_critic_eval.jsonl", eval_payloads)
    write_jsonl(output_dir / "tensor_audio_error_localization.jsonl", localization_rows)
    manifest = {
        "schema": "tensor_audio_error_critic.v1",
        "source_dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "training_completed": True,
        "critic_not_usable_reason": "",
        "rows_loaded": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_target_rows),
        "excluded_reason_counts": dict(sorted(excluded.items())),
        "eval_metrics": eval_metrics,
        "threshold_metrics": threshold_metrics,
        "tensor_maps_required": True,
        "scalar_fallback_used": False,
        "critic_training_started": True,
        "shadow_only": True,
        **mining_report,
        "model_kind": model_kind,
        "model_file": "tensor_audio_error_critic.joblib",
        "candidate_model_metrics": {
            candidate_kind: {"eval_metrics": candidate_metrics, "threshold_metrics": candidate_threshold_metrics}
            for candidate_kind, _, _, candidate_metrics, candidate_threshold_metrics in candidates
        },
        **SAFETY_FLAGS,
    }
    write_json(output_dir / "tensor_audio_error_critic_manifest.json", manifest)
    write_json(
        output_dir / "tensor_audio_error_critic_summary.json",
        {
            "training_completed": True,
            "train_rows": len(train_rows),
            "eval_metrics": eval_metrics,
            "release_ready": False,
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a shadow tensor-only audio error critic.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-train-rows", type=int, default=20)
    parser.add_argument("--mining-bundle-dir", type=Path)
    parser.add_argument("--mining-weight-multiplier", type=float, default=3.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    train_tensor_audio_error_critic(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        min_train_rows=args.min_train_rows,
        mining_bundle_dir=args.mining_bundle_dir,
        mining_weight_multiplier=args.mining_weight_multiplier,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
