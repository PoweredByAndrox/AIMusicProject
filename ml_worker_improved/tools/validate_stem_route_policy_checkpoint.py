from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .stem_route_policy_common import ACTION_BLOCK_UNKNOWN, ACTIONS, normalize_policy_action
    from .stem_route_policy_model import load_stem_route_policy_checkpoint, score_policy_row
except ImportError:  # pragma: no cover - direct script execution
    from stem_route_policy_common import ACTION_BLOCK_UNKNOWN, ACTIONS, normalize_policy_action  # type: ignore
    from stem_route_policy_model import load_stem_route_policy_checkpoint, score_policy_row  # type: ignore


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    return path


def _baseline_accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    counts = Counter(normalize_policy_action(str(row.get("oracle_action") or "")) for row in rows)
    baseline = counts.most_common(1)[0][0] if counts else ACTION_BLOCK_UNKNOWN
    return sum(1 for row in rows if normalize_policy_action(str(row.get("oracle_action") or "")) == baseline) / float(len(rows))


def validate_stem_route_policy_checkpoint(
    *,
    checkpoint_path: Path,
    rows_jsonl_path: Path,
    output_dir: Path,
    before_after_jsonl_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_stem_route_policy_checkpoint(checkpoint_path)
    rows = [row for row in _read_jsonl(rows_jsonl_path) if bool(row.get("usable_for_training", True))]
    predictions: list[dict[str, Any]] = []
    correct = 0
    false_accept_count = 0
    top3_correct = 0
    quarantine_total = 0
    quarantine_correct = 0
    for row in rows:
        scored = score_policy_row(model, row)
        oracle = normalize_policy_action(str(row.get("oracle_action") or ""))
        predicted = normalize_policy_action(str(scored.get("action") or ""))
        probs = scored.get("action_probs") if isinstance(scored.get("action_probs"), dict) else {}
        top3 = sorted(ACTIONS, key=lambda action: float(probs.get(action, 0.0)), reverse=True)[:3]
        correct += int(predicted == oracle)
        top3_correct += int(oracle in top3)
        quarantine_total += int(predicted == "quarantine_route")
        quarantine_correct += int(predicted == "quarantine_route" and oracle == "quarantine_route")
        if bool(scored.get("delivery_allowed")):
            false_accept_count += 1
        predictions.append(
            {
                "row_id": row.get("row_id"),
                "oracle_action": oracle,
                "predicted_action": predicted,
                "action_probs": probs,
                "match_level": scored.get("match_level"),
                "correct": predicted == oracle,
            }
        )
    row_count = len(rows)
    accuracy = correct / float(row_count) if row_count else 0.0
    top3_recall = top3_correct / float(row_count) if row_count else 0.0
    quarantine_precision = quarantine_correct / float(quarantine_total) if quarantine_total else 1.0
    baseline_accuracy = _baseline_accuracy(rows)

    replay_rows = _read_jsonl(before_after_jsonl_path)
    improved_count = sum(1 for row in replay_rows if bool(row.get("improved")))
    unchanged_count = sum(1 for row in replay_rows if not bool(row.get("improved")))
    failed_checks: list[str] = []
    if false_accept_count:
        failed_checks.append("false_accept_path_exists")
    if row_count == 0:
        failed_checks.append("missing_holdout_rows")
    if accuracy < baseline_accuracy:
        failed_checks.append("worse_than_heuristic_baseline")
    if before_after_jsonl_path is not None and improved_count == 0:
        failed_checks.append("no_counterfactual_improvement")
    if bool(model.get("delivery_allowed")):
        failed_checks.append("policy_model_claims_delivery_allowed")
    promotion_allowed = not failed_checks

    _write_jsonl(output_dir / "predictions.jsonl", predictions)
    manifest = {
        "schema": "stem_route_policy_validation_manifest.v1",
        "checkpoint_path": str(checkpoint_path),
        "rows_jsonl": str(rows_jsonl_path),
        "before_after_jsonl": str(before_after_jsonl_path or ""),
        "manifest_path": str(output_dir / "manifest.json"),
        "predictions_jsonl": str(output_dir / "predictions.jsonl"),
        "row_count": row_count,
        "top1_action_accuracy": accuracy,
        "top3_action_recall": top3_recall,
        "heuristic_baseline_accuracy": baseline_accuracy,
        "quarantine_precision": quarantine_precision,
        "false_accept_count": false_accept_count,
        "accepted_candidate_recall": 0.0,
        "candidate_count_reduction": 0.0,
        "issue_specific_improvement_count": improved_count,
        "unchanged_replay_count": unchanged_count,
        "promotion_allowed": promotion_allowed,
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "failed_checks": failed_checks,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--rows-jsonl", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--before-after-jsonl", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = validate_stem_route_policy_checkpoint(
        checkpoint_path=args.checkpoint,
        rows_jsonl_path=args.rows_jsonl,
        output_dir=args.out,
        before_after_jsonl_path=args.before_after_jsonl,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
