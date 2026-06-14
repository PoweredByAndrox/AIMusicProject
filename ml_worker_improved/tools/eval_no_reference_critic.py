from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_no_reference_ml_dataset import FEATURE_NAMES, iter_jsonl
from tools.train_no_reference_critic import MODEL_FILES, binary_metrics, load_bundle, predict_binary_score


def load_examples(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def load_pairwise(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(iter_jsonl(path))


def script_baseline_metrics(rows: list[dict[str, Any]], threshold: float = 0.72) -> dict[str, float | int]:
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
        pred = 1 if float(row.get("script_score") or 0.0) >= threshold else 0
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


def hybrid_metrics(rows: list[dict[str, Any]], bundle: dict[str, Any], threshold: float = 0.72) -> dict[str, float | int]:
    if not rows:
        return script_baseline_metrics(rows, threshold=threshold)
    patched: list[dict[str, Any]] = []
    for row in rows:
        learned = predict_binary_score(bundle, row.get("features", {}))
        hybrid = 0.5 * float(row.get("script_score") or 0.0) + 0.5 * learned
        patched.append({**row, "script_score": hybrid})
    return script_baseline_metrics(patched, threshold=threshold)


def calibration_error(rows: list[dict[str, Any]], bundle: dict[str, Any], bins: int = 5) -> float:
    if not rows:
        return 0.0
    bucket_error = 0.0
    for bucket in range(bins):
        low = bucket / bins
        high = (bucket + 1) / bins
        scored = []
        for row in rows:
            score = predict_binary_score(bundle, row.get("features", {}))
            if low <= score < high or (bucket == bins - 1 and score == 1.0):
                scored.append((score, int(row.get("label") or 0)))
        if not scored:
            continue
        confidence = sum(score for score, _ in scored) / len(scored)
        accuracy = sum(label for _, label in scored) / len(scored)
        bucket_error += (len(scored) / len(rows)) * abs(confidence - accuracy)
    return round(bucket_error, 6)


def pairwise_ranking_accuracy(pairwise: list[dict[str, Any]], examples_by_id: dict[str, dict[str, Any]], bundles: dict[str, dict[str, Any]]) -> float:
    if not pairwise:
        return 0.0
    correct = 0
    total = 0
    for row in pairwise:
        mode = str(row.get("mode") or "both")
        bundle = bundles.get("pair_critic" if mode == "both" else f"{mode}_critic")
        preferred = examples_by_id.get(str(row.get("preferred_example_id") or ""))
        rejected = examples_by_id.get(str(row.get("rejected_example_id") or ""))
        if bundle is not None and preferred is not None and rejected is not None:
            preferred_score = predict_binary_score(bundle, preferred.get("features", {}))
            rejected_score = predict_binary_score(bundle, rejected.get("features", {}))
        else:
            preferred_score = float(row.get("preferred_score") or 0.0)
            rejected_score = float(row.get("rejected_score") or 0.0)
        correct += 1 if preferred_score >= rejected_score else 0
        total += 1
    return round(correct / total, 6) if total else 0.0


def evaluate_critics(*, examples_jsonl: Path, pairwise_jsonl: Path, model_dir: Path, output_dir: Path) -> dict[str, Any]:
    examples = load_examples(examples_jsonl)
    pairwise = load_pairwise(pairwise_jsonl)
    bundles = {
        "vocal_critic": load_bundle(model_dir / MODEL_FILES["vocal_critic"]),
        "minus_critic": load_bundle(model_dir / MODEL_FILES["minus_critic"]),
        "pair_critic": load_bundle(model_dir / MODEL_FILES["pair_critic"]),
    }
    mode_to_model = {"vocal": "vocal_critic", "minus": "minus_critic", "both": "pair_critic"}
    summary: dict[str, Any] = {
        "examples_total": len(examples),
        "pairwise_examples_total": len(pairwise),
        "feature_list": list(FEATURE_NAMES),
        "models": {},
    }
    for mode, model_name in mode_to_model.items():
        rows = [row for row in examples if row.get("mode") == mode]
        bundle = bundles[model_name]
        learned = binary_metrics(rows, bundle)
        script = script_baseline_metrics(rows)
        hybrid = hybrid_metrics(rows, bundle)
        summary["models"][model_name] = {
            "mode": mode,
            "learned": learned,
            "script_baseline": script,
            "hybrid": hybrid,
            "calibration_error": calibration_error(rows, bundle),
            "hybrid_beats_script_baseline": bool(
                hybrid.get("false_accept_rate", 1.0) <= script.get("false_accept_rate", 1.0)
                and hybrid.get("f1", 0.0) >= script.get("f1", 0.0)
            ),
        }
    examples_by_id = {str(row.get("example_id") or ""): row for row in examples}
    summary["pairwise_ranking_accuracy"] = pairwise_ranking_accuracy(pairwise, examples_by_id, bundles)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary_md(output_dir / "evaluation_summary.md", summary)
    return summary


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# No-Reference Local Critic Evaluation",
        "",
        f"- examples_total: {summary['examples_total']}",
        f"- pairwise_examples_total: {summary['pairwise_examples_total']}",
        f"- pairwise_ranking_accuracy: {summary['pairwise_ranking_accuracy']}",
        "",
        "## Models",
        "",
    ]
    for name, metrics in summary["models"].items():
        learned = metrics["learned"]
        script = metrics["script_baseline"]
        hybrid = metrics["hybrid"]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- mode: {metrics['mode']}",
                f"- learned_accuracy: {learned['accuracy']}",
                f"- learned_f1: {learned['f1']}",
                f"- learned_false_accept_rate: {learned['false_accept_rate']}",
                f"- learned_false_block_rate: {learned['false_block_rate']}",
                f"- script_false_accept_rate: {script['false_accept_rate']}",
                f"- hybrid_false_accept_rate: {hybrid['false_accept_rate']}",
                f"- calibration_error: {metrics['calibration_error']}",
                f"- hybrid_beats_script_baseline: {str(metrics['hybrid_beats_script_baseline']).lower()}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate local no-reference critic bundles.")
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--pairwise-jsonl", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = evaluate_critics(
        examples_jsonl=args.examples_jsonl,
        pairwise_jsonl=args.pairwise_jsonl,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
