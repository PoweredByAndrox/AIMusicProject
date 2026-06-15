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
    return {
        "safe_accept_threshold": float(threshold),
        "threshold_false_accept": int(false_accept),
        "threshold_false_accept_rate": float(false_accept / len(negatives)) if negatives else None,
        "threshold_false_reject": int(false_reject),
        "threshold_false_reject_rate": float(false_reject / len(positives)) if positives else None,
    }


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


def train_tensor_audio_error_critic(*, dataset_dir: Path, output_dir: Path, min_train_rows: int = 20) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = read_json(dataset_dir / "locked_tensor_audio_error_dataset_manifest.json")
    rows, excluded = load_rows(dataset_dir)
    train_rows = [row for row in rows if row.get("split") == "train" and row.get("train_allowed") is True]
    eval_rows = [row for row in rows if row.get("split") in {"dev", "test"}]
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
            **SAFETY_FLAGS,
        }
        write_json(output_dir / "tensor_audio_error_critic_manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
        return manifest

    train_x = np.vstack([row["features"] for row in train_rows])
    train_y = np.asarray(train_labels, dtype=np.int32)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=1729))
    model.fit(train_x, train_y)
    eval_target_rows = eval_rows if eval_rows else train_rows
    eval_x = np.vstack([row["features"] for row in eval_target_rows])
    probabilities = model.predict_proba(eval_x)[:, 1].astype(float)
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
    eval_metrics = metrics_for_rows(eval_target_rows, probabilities)
    threshold_metrics = conservative_threshold_metrics(eval_target_rows, probabilities)
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
        "model_file": "tensor_audio_error_critic.joblib",
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    train_tensor_audio_error_critic(dataset_dir=args.dataset_dir, output_dir=args.output_dir, min_train_rows=args.min_train_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
