from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_no_reference_ml_dataset import FEATURE_NAMES, split_for_track, stable_hash, write_jsonl  # noqa: E402
from tools.eval_no_reference_critic import evaluate_critics  # noqa: E402
from tools.train_no_reference_critic import train_critics  # noqa: E402


SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "separator_training_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "release_ready": False,
}


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def find_outcome_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("learning_outcome.jsonl") if path.is_file())


def load_outcomes(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in find_outcome_files(root):
        for row in iter_jsonl(path):
            rows.append({**row, "source_learning_outcome_jsonl": str(path)})
    return sorted(rows, key=lambda row: (str(row.get("track_id") or ""), str(row.get("candidate_id") or ""), str(row.get("source_learning_outcome_jsonl") or "")))


def verified_examples_from_outcome(row: dict[str, Any]) -> list[dict[str, Any]]:
    if row.get("verified_examples") and isinstance(row["verified_examples"], list):
        source_rows = [item for item in row["verified_examples"] if isinstance(item, dict)]
    else:
        source_rows = [row]
    examples: list[dict[str, Any]] = []
    for source in source_rows:
        if source.get("verified_outcome") is not True or source.get("raw_prediction_only") is True:
            continue
        mode = str(source.get("mode") or source.get("score_mode") or "").strip().lower()
        if mode not in {"vocal", "minus", "both"}:
            continue
        features = source.get("features")
        if not isinstance(features, dict):
            continue
        label = int(source.get("label") or source.get("outcome_label") or 0)
        track_id = str(source.get("track_id") or row.get("track_id") or "")
        candidate_id = str(source.get("candidate_id") or row.get("candidate_id") or "")
        feature_payload = {name: float(features.get(name, 0.0) or 0.0) for name in FEATURE_NAMES}
        example_id = f"runtime_outcome:{track_id}:{candidate_id}:{mode}:{stable_hash(source)[:12]}"
        examples.append(
            {
                "example_id": example_id,
                "track_id": track_id,
                "candidate_id": candidate_id,
                "mode": mode,
                "label": label,
                "policy_action": str(source.get("policy_action") or ("accept_candidate" if label else "block_candidate")),
                "split": split_for_track(track_id),
                "script_score": float(source.get("script_score") or features.get("overall_no_reference_score") or 0.0),
                "features": feature_payload,
                "source_artifact_type": "runtime_learning_outcome",
                "source_artifact_refs": [str(row.get("source_learning_outcome_jsonl") or "")],
                "weak_label_source": "verified_runtime_outcome",
                "strict_mode_label": True,
                "reference_required": False,
                "human_csv_required": False,
                "training_started": False,
                "fine_tune_started": False,
                "models_changed": False,
                "model_registry_edited": False,
                "separator_training_started": False,
                "release_ready": False,
                "delivery_allowed": False,
                "disagreement_type": str(source.get("disagreement_type") or "no_disagreement"),
            }
        )
    return examples


def build_training_examples(outcomes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    examples: list[dict[str, Any]] = []
    raw_excluded = 0
    for row in outcomes:
        if row.get("raw_prediction_only") is True or row.get("verified_outcome") is not True:
            raw_excluded += 1
        examples.extend(verified_examples_from_outcome(row))
    examples.sort(key=lambda item: (item["track_id"], item["mode"], item["candidate_id"], item["example_id"]))
    return examples, raw_excluded


def max_false_accept_rate(summary: dict[str, Any]) -> float:
    values: list[float] = []
    for model in (summary.get("models") or {}).values():
        if not isinstance(model, dict):
            continue
        for section in ("learned", "hybrid"):
            metrics = model.get(section)
            if isinstance(metrics, dict):
                values.append(float(metrics.get("false_accept_rate") or 0.0))
    return max(values) if values else 1.0


def disagreement_outcome_accuracy(examples: list[dict[str, Any]], summary: dict[str, Any]) -> float:
    disagreement_examples = [row for row in examples if row.get("disagreement_type") not in {"", "no_disagreement"}]
    if not disagreement_examples:
        return 0.0
    model_scores = [
        float((model.get("hybrid") or {}).get("accuracy") or 0.0)
        for model in (summary.get("models") or {}).values()
        if isinstance(model, dict)
    ]
    return round(sum(model_scores) / len(model_scores), 6) if model_scores else 0.0


def build_comparison(
    *,
    current_eval: dict[str, Any],
    next_eval: dict[str, Any],
    examples: list[dict[str, Any]],
    current_model_dir: Path,
    output_model_dir: Path,
) -> dict[str, Any]:
    current_false_accept = max_false_accept_rate(current_eval)
    next_false_accept = max_false_accept_rate(next_eval)
    current_rank = float(current_eval.get("pairwise_ranking_accuracy") or 0.0)
    next_rank = float(next_eval.get("pairwise_ranking_accuracy") or 0.0)
    current_disagreement = disagreement_outcome_accuracy(examples, current_eval)
    next_disagreement = disagreement_outcome_accuracy(examples, next_eval)
    false_accept_regressed = next_false_accept > current_false_accept
    ranking_regressed = next_rank < current_rank
    disagreement_regressed = next_disagreement < current_disagreement
    blocker_safety_regressed = bool(next_eval.get("blocker_safety_regressed") is True)
    approved = not any([false_accept_regressed, ranking_regressed, disagreement_regressed, blocker_safety_regressed])
    return {
        "current_model_dir": str(current_model_dir),
        "next_model_dir": str(output_model_dir),
        "current": {
            "max_false_accept_rate": round(current_false_accept, 6),
            "pairwise_ranking_accuracy": round(current_rank, 6),
            "disagreement_outcome_accuracy": round(current_disagreement, 6),
        },
        "next": {
            "max_false_accept_rate": round(next_false_accept, 6),
            "pairwise_ranking_accuracy": round(next_rank, 6),
            "disagreement_outcome_accuracy": round(next_disagreement, 6),
        },
        "false_accept_rate_regressed": false_accept_regressed,
        "ranking_accuracy_regressed": ranking_regressed,
        "disagreement_outcome_accuracy_regressed": disagreement_regressed,
        "blocker_safety_regressed": blocker_safety_regressed,
        "promotion_approved": approved,
        "release_ready": False,
        **SAFETY_FLAGS,
    }


def run_learning_cycle(*, outcomes_root: Path, current_model_dir: Path, output_model_dir: Path, eval_output_dir: Path) -> dict[str, Any]:
    outcomes = load_outcomes(outcomes_root)
    examples, raw_excluded = build_training_examples(outcomes)
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    training_dir = eval_output_dir / "training_data"
    examples_jsonl = training_dir / "verified_runtime_examples.jsonl"
    pairwise_jsonl = training_dir / "verified_runtime_pairwise.jsonl"
    write_jsonl(examples_jsonl, examples)
    write_jsonl(pairwise_jsonl, [])

    if not examples:
        manifest = {
            "outcomes_root": str(outcomes_root),
            "outcome_rows_total": len(outcomes),
            "raw_prediction_rows_excluded": raw_excluded,
            "verified_examples_total": 0,
            "training_started": False,
            "promotion_approved": False,
            "reason": "no_verified_outcomes",
            **{key: value for key, value in SAFETY_FLAGS.items() if key != "training_started"},
        }
        write_json(eval_output_dir / "learning_cycle_manifest.json", manifest)
        write_json(eval_output_dir / "evaluation_comparison.json", {**manifest, "promotion_approved": False})
        return manifest

    train_manifest = train_critics(examples_jsonl=examples_jsonl, pairwise_jsonl=pairwise_jsonl, model_dir=output_model_dir)
    next_eval = evaluate_critics(examples_jsonl=examples_jsonl, pairwise_jsonl=pairwise_jsonl, model_dir=output_model_dir, output_dir=eval_output_dir / "next")
    if all((current_model_dir / name).exists() for name in ("vocal_critic.joblib", "minus_critic.joblib", "pair_critic.joblib")):
        current_eval = evaluate_critics(examples_jsonl=examples_jsonl, pairwise_jsonl=pairwise_jsonl, model_dir=current_model_dir, output_dir=eval_output_dir / "current")
    else:
        current_eval = {"models": {}, "pairwise_ranking_accuracy": 0.0, "missing_current_model": True}
    comparison = build_comparison(
        current_eval=current_eval,
        next_eval=next_eval,
        examples=examples,
        current_model_dir=current_model_dir,
        output_model_dir=output_model_dir,
    )
    write_json(eval_output_dir / "evaluation_comparison.json", comparison)
    manifest = {
        "outcomes_root": str(outcomes_root),
        "outcome_rows_total": len(outcomes),
        "raw_prediction_rows_excluded": raw_excluded,
        "verified_examples_total": len(examples),
        "examples_jsonl": str(examples_jsonl),
        "pairwise_jsonl": str(pairwise_jsonl),
        "critic_training_started": True,
        "training_started": False,
        "train_manifest": train_manifest,
        "promotion_approved": comparison["promotion_approved"],
        **{key: value for key, value in SAFETY_FLAGS.items() if key != "training_started"},
    }
    write_json(eval_output_dir / "learning_cycle_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a verified-outcome-only no-reference critic learning cycle.")
    parser.add_argument("--outcomes-root", type=Path, required=True)
    parser.add_argument("--current-model-dir", type=Path, required=True)
    parser.add_argument("--output-model-dir", type=Path, required=True)
    parser.add_argument("--eval-output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = run_learning_cycle(
        outcomes_root=args.outcomes_root,
        current_model_dir=args.current_model_dir,
        output_model_dir=args.output_model_dir,
        eval_output_dir=args.eval_output_dir,
    )
    print(json.dumps({"verified_examples_total": manifest["verified_examples_total"], "promotion_approved": manifest["promotion_approved"], "release_ready": False}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
