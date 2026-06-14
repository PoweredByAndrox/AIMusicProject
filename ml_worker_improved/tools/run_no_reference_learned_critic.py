from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_no_reference_ml_dataset import (
    SEVERE_REASON_TERMS,
    as_list,
    extract_features,
    iter_jsonl,
    normalize_mode,
    score_for_mode,
    stable_unique,
    write_jsonl,
)
from tools.train_no_reference_critic import MODEL_FILES, load_bundle, predict_binary_score, predict_policy_action


BLOCKING_DECISIONS = {"blocked_hard_case", "blocked_pair_incompatible", "needs_pair_reconciliation"}
BLOCKING_POLICY_ACTION = "block_by_safety_gate"


def load_model_dir(model_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        "vocal_critic": load_bundle(model_dir / MODEL_FILES["vocal_critic"]),
        "minus_critic": load_bundle(model_dir / MODEL_FILES["minus_critic"]),
        "pair_critic": load_bundle(model_dir / MODEL_FILES["pair_critic"]),
        "policy_controller": load_bundle(model_dir / MODEL_FILES["policy_controller"]),
    }


def safety_reasons(row: dict[str, Any]) -> list[str]:
    reasons = stable_unique(
        as_list(row.get("reasons"))
        + as_list(row.get("failure_labels"))
        + as_list(row.get("failed_checks"))
        + as_list(row.get("vocal_blockers"))
        + as_list(row.get("minus_blockers"))
        + as_list(row.get("pair_blockers"))
    )
    blockers: list[str] = []
    decision = str(row.get("decision") or row.get("pair_decision") or "")
    if decision in BLOCKING_DECISIONS:
        blockers.append(f"script_decision:{decision}")
    blockers.extend(reason for reason in reasons if reason in SEVERE_REASON_TERMS or reason.startswith("severe_"))
    forbidden = [
        action
        for action in as_list(row.get("forbidden_actions"))
        if "release" in action or "publish" in action or "delivery" in action
    ]
    blockers.extend(f"forbidden_action:{action}" for action in forbidden)
    return stable_unique(blockers)


def selected_mode(row: dict[str, Any]) -> str:
    return normalize_mode(row.get("score_mode")) or normalize_mode(row.get("source_stem")) or "both"


def score_row(row: dict[str, Any], bundles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    features = extract_features(row)
    learned_vocal_score = predict_binary_score(bundles["vocal_critic"], features)
    learned_minus_score = predict_binary_score(bundles["minus_critic"], features)
    learned_pair_score = predict_binary_score(bundles["pair_critic"], features)
    policy_action, policy_confidence = predict_policy_action(bundles["policy_controller"], features)
    mode = selected_mode(row)
    if mode == "vocal":
        learned_score = learned_vocal_score
    elif mode == "minus":
        learned_score = learned_minus_score
    else:
        learned_score = learned_pair_score
    script_score = round(score_for_mode(row, mode), 6)
    hybrid_score = round(0.5 * script_score + 0.5 * learned_score, 6)
    blockers = safety_reasons(row)
    safety_blocked = bool(blockers)
    if safety_blocked:
        policy_action = BLOCKING_POLICY_ACTION
        policy_confidence = max(policy_confidence, 1.0)
    return {
        **row,
        "learned_vocal_score": round(learned_vocal_score, 6),
        "learned_minus_score": round(learned_minus_score, 6),
        "learned_pair_score": round(learned_pair_score, 6),
        "learned_score": round(learned_score, 6),
        "learned_policy_action": policy_action,
        "learned_confidence": round(policy_confidence, 6),
        "script_score": script_score,
        "hybrid_score": hybrid_score,
        "safety_blocked": safety_blocked,
        "safety_blockers": blockers,
        "delivery_allowed": False,
        "presentation_status": "diagnostic_learned_critic_only",
        "release_ready": False,
        "training_started": False,
        "separator_training_started": False,
    }


def score_rows(rows: Iterable[dict[str, Any]], *, model_dir: Path) -> list[dict[str, Any]]:
    bundles = load_model_dir(model_dir)
    return [score_row(row, bundles) for row in rows]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score candidates with local no-reference learned critics plus script safety gates.")
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = score_rows(iter_jsonl(args.candidate_jsonl), model_dir=args.model_dir)
    write_jsonl(args.output_jsonl, rows)
    print(
        json.dumps(
            {
                "rows_scored": len(rows),
                "safety_blocked": sum(1 for row in rows if row.get("safety_blocked") is True),
                "output_jsonl": str(args.output_jsonl),
                "release_ready": False,
                "delivery_allowed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
