from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_no_reference_ml_dataset import FEATURE_NAMES, SAFETY_FLAGS, iter_jsonl, stable_hash

try:  # pragma: no cover - exercised in Docker when sklearn is present.
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None

try:  # pragma: no cover - availability is environment-dependent.
    from sklearn.dummy import DummyClassifier  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.pipeline import make_pipeline  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    DummyClassifier = None
    LogisticRegression = None
    StandardScaler = None
    make_pipeline = None
    SKLEARN_AVAILABLE = False


MODEL_FILES = {
    "vocal_critic": "vocal_critic.joblib",
    "minus_critic": "minus_critic.joblib",
    "pair_critic": "pair_critic.joblib",
    "policy_controller": "policy_controller.joblib",
}


def feature_vector(example_or_row: dict[str, Any], feature_list: Sequence[str] = FEATURE_NAMES) -> list[float]:
    features = example_or_row.get("features") if isinstance(example_or_row.get("features"), dict) else example_or_row
    return [float(features.get(name, 0.0) or 0.0) for name in feature_list]


def load_examples(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def load_pairwise(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(iter_jsonl(path))


def dump_bundle(bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if joblib is not None:
        joblib.dump(bundle, path)
        return
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)


def load_bundle(path: Path) -> dict[str, Any]:
    if joblib is not None:
        return joblib.load(path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def estimator_classes(estimator: Any) -> list[Any]:
    if hasattr(estimator, "classes_"):
        return list(estimator.classes_)
    if hasattr(estimator, "steps") and estimator.steps:
        tail = estimator.steps[-1][1]
        if hasattr(tail, "classes_"):
            return list(tail.classes_)
    return []


def class_counts(labels: Iterable[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(labels).items(), key=lambda item: str(item[0]))}


def fit_binary_model(rows: list[dict[str, Any]], *, model_name: str) -> dict[str, Any]:
    feature_list = list(FEATURE_NAMES)
    labels = [int(row.get("label") or 0) for row in rows]
    train_rows = [row for row in rows if row.get("split") == "train"] or rows
    train_labels = [int(row.get("label") or 0) for row in train_rows]
    x_train = [feature_vector(row, feature_list) for row in train_rows]
    if SKLEARN_AVAILABLE and x_train:
        if len(set(train_labels)) > 1:
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
            )
            model_type = "sklearn_pipeline_standard_scaler_logistic_regression"
        else:
            estimator = DummyClassifier(strategy="constant", constant=train_labels[0] if train_labels else 0)
            model_type = "sklearn_dummy_constant"
        estimator.fit(x_train, train_labels)
        return {
            "model_name": model_name,
            "task_type": "binary_critic",
            "model_type": model_type,
            "estimator": estimator,
            "feature_list": feature_list,
            "class_counts": class_counts(labels),
        }
    positive_scores = [float(row.get("script_score") or 0.0) for row in train_rows if int(row.get("label") or 0) == 1]
    negative_scores = [float(row.get("script_score") or 0.0) for row in train_rows if int(row.get("label") or 0) == 0]
    if positive_scores and negative_scores:
        threshold = (sum(positive_scores) / len(positive_scores) + sum(negative_scores) / len(negative_scores)) / 2.0
    elif positive_scores:
        threshold = min(positive_scores)
    else:
        threshold = 1.01
    return {
        "model_name": model_name,
        "task_type": "binary_critic",
        "model_type": "threshold_fallback",
        "threshold": round(float(threshold), 6),
        "feature_list": feature_list,
        "class_counts": class_counts(labels),
    }


def fit_policy_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feature_list = list(FEATURE_NAMES)
    train_rows = [row for row in rows if row.get("split") == "train"] or rows
    labels = [str(row.get("policy_action") or "block_candidate") for row in rows]
    train_labels = [str(row.get("policy_action") or "block_candidate") for row in train_rows]
    x_train = [feature_vector(row, feature_list) for row in train_rows]
    if SKLEARN_AVAILABLE and x_train:
        if len(set(train_labels)) > 1:
            estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=0))
            model_type = "sklearn_pipeline_standard_scaler_logistic_regression_multiclass"
        else:
            estimator = DummyClassifier(strategy="constant", constant=train_labels[0] if train_labels else "block_candidate")
            model_type = "sklearn_dummy_constant"
        estimator.fit(x_train, train_labels)
        return {
            "model_name": "policy_controller",
            "task_type": "policy_controller",
            "model_type": model_type,
            "estimator": estimator,
            "feature_list": feature_list,
            "class_counts": class_counts(labels),
        }
    majority = Counter(train_labels).most_common(1)[0][0] if train_labels else "block_candidate"
    return {
        "model_name": "policy_controller",
        "task_type": "policy_controller",
        "model_type": "majority_policy_fallback",
        "majority_action": majority,
        "feature_list": feature_list,
        "class_counts": class_counts(labels),
    }


def predict_binary_score(bundle: dict[str, Any], features: dict[str, float] | list[float]) -> float:
    vector = features if isinstance(features, list) else feature_vector({"features": features}, bundle.get("feature_list") or FEATURE_NAMES)
    if "estimator" in bundle:
        estimator = bundle["estimator"]
        if hasattr(estimator, "predict_proba"):
            probabilities = estimator.predict_proba([vector])[0]
            classes = estimator_classes(estimator)
            if 1 in classes:
                return round(float(probabilities[classes.index(1)]), 6)
            return round(float(max(probabilities)), 6)
        return round(float(estimator.predict([vector])[0]), 6)
    threshold = float(bundle.get("threshold", 1.01))
    script_score = float(vector[FEATURE_NAMES.index("overall_no_reference_score")]) if "overall_no_reference_score" in FEATURE_NAMES else 0.0
    return 1.0 if script_score >= threshold else 0.0


def predict_policy_action(bundle: dict[str, Any], features: dict[str, float] | list[float]) -> tuple[str, float]:
    vector = features if isinstance(features, list) else feature_vector({"features": features}, bundle.get("feature_list") or FEATURE_NAMES)
    if "estimator" in bundle:
        estimator = bundle["estimator"]
        if hasattr(estimator, "predict_proba"):
            probabilities = estimator.predict_proba([vector])[0]
            classes = estimator_classes(estimator)
            if not classes:
                return "block_candidate", 0.0
            index = max(range(len(probabilities)), key=lambda idx: probabilities[idx])
            return str(classes[index]), round(float(probabilities[index]), 6)
        return str(estimator.predict([vector])[0]), 1.0
    return str(bundle.get("majority_action") or "block_candidate"), 1.0


def binary_metrics(rows: list[dict[str, Any]], bundle: dict[str, Any]) -> dict[str, float | int]:
    if not rows:
        return {
            "examples": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "false_accept_rate": 0.0,
            "false_block_rate": 0.0,
        }
    tp = tn = fp = fn = 0
    for row in rows:
        score = predict_binary_score(bundle, row.get("features", {}))
        pred = 1 if score >= 0.5 else 0
        label = int(row.get("label") or 0)
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 1 and label == 0:
            fp += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    false_accept_rate = fp / (fp + tn) if fp + tn else 0.0
    false_block_rate = fn / (fn + tp) if fn + tp else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "examples": len(rows),
        "accuracy": round((tp + tn) / len(rows), 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "false_accept_rate": round(false_accept_rate, 6),
        "false_block_rate": round(false_block_rate, 6),
    }


def metrics_by_split(rows: list[dict[str, Any]], bundle: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    return {
        split: binary_metrics([row for row in rows if row.get("split") == split], bundle)
        for split in ("train", "dev", "test")
    } | {"all": binary_metrics(rows, bundle)}


def data_hash(examples: list[dict[str, Any]], pairwise: list[dict[str, Any]]) -> str:
    return stable_hash({"examples": examples, "pairwise": pairwise, "feature_list": FEATURE_NAMES})


def train_critics(*, examples_jsonl: Path, pairwise_jsonl: Path, model_dir: Path) -> dict[str, Any]:
    examples = load_examples(examples_jsonl)
    pairwise = load_pairwise(pairwise_jsonl)
    rows_by_model = {
        "vocal_critic": [row for row in examples if row.get("mode") == "vocal"],
        "minus_critic": [row for row in examples if row.get("mode") == "minus"],
        "pair_critic": [row for row in examples if row.get("mode") == "both"],
    }
    bundles: dict[str, dict[str, Any]] = {}
    model_metrics: dict[str, Any] = {}
    for model_name, rows in rows_by_model.items():
        bundle = fit_binary_model(rows, model_name=model_name)
        bundles[model_name] = bundle
        model_metrics[model_name] = metrics_by_split(rows, bundle)
        dump_bundle(bundle, model_dir / MODEL_FILES[model_name])
    policy_bundle = fit_policy_model(examples)
    bundles["policy_controller"] = policy_bundle
    dump_bundle(policy_bundle, model_dir / MODEL_FILES["policy_controller"])
    split_counts = Counter(str(row.get("split") or "") for row in examples)
    mode_counts = Counter(str(row.get("mode") or "") for row in examples)
    manifest = {
        "model_family": "local_no_reference_weak_label_critic",
        "model_type": "sklearn-compatible local classifiers" if SKLEARN_AVAILABLE else "deterministic threshold fallback",
        "training_data_hash": data_hash(examples, pairwise),
        "feature_list": list(FEATURE_NAMES),
        "train_dev_test_split": dict(sorted(split_counts.items())),
        "examples_by_mode": dict(sorted(mode_counts.items())),
        "pairwise_examples_total": len(pairwise),
        "models": {
            name: {
                "path": str(model_dir / MODEL_FILES[name]),
                "model_type": bundle.get("model_type"),
                "class_counts": bundle.get("class_counts"),
                "metrics": model_metrics.get(name, {}),
            }
            for name, bundle in bundles.items()
        },
        "safety": dict(SAFETY_FLAGS),
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train local sklearn-compatible no-reference critics.")
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--pairwise-jsonl", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = train_critics(examples_jsonl=args.examples_jsonl, pairwise_jsonl=args.pairwise_jsonl, model_dir=args.model_dir)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
