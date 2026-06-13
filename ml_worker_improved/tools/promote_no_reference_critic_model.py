from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.train_no_reference_critic import MODEL_FILES  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def metric(payload: dict[str, Any], section: str, key: str, default: float = 0.0) -> float:
    section_payload = payload.get(section)
    if not isinstance(section_payload, dict):
        return default
    if key not in section_payload or section_payload.get(key) is None:
        return default
    try:
        return float(section_payload.get(key))
    except (TypeError, ValueError):
        return default


def evaluate_promotion(comparison: dict[str, Any]) -> dict[str, Any]:
    current_false_accept = metric(comparison, "current", "max_false_accept_rate", 1.0)
    next_false_accept = metric(comparison, "next", "max_false_accept_rate", 1.0)
    current_rank = metric(comparison, "current", "pairwise_ranking_accuracy", 0.0)
    next_rank = metric(comparison, "next", "pairwise_ranking_accuracy", 0.0)
    current_disagreement = metric(comparison, "current", "disagreement_outcome_accuracy", 0.0)
    next_disagreement = metric(comparison, "next", "disagreement_outcome_accuracy", 0.0)
    false_accept_regressed = bool(comparison.get("false_accept_rate_regressed", next_false_accept > current_false_accept))
    ranking_regressed = bool(comparison.get("ranking_accuracy_regressed", next_rank < current_rank))
    disagreement_regressed = bool(comparison.get("disagreement_outcome_accuracy_regressed", next_disagreement < current_disagreement))
    safety_regressed = bool(comparison.get("blocker_safety_regressed") is True or comparison.get("safety_regressed") is True)
    approved_by_comparison = bool(comparison.get("promotion_approved", True))
    approved = approved_by_comparison and not any([false_accept_regressed, ranking_regressed, disagreement_regressed, safety_regressed])
    reasons: list[str] = []
    if not approved_by_comparison:
        reasons.append("comparison_not_approved")
    if false_accept_regressed:
        reasons.append("false_accept_rate_regressed")
    if ranking_regressed:
        reasons.append("ranking_accuracy_regressed")
    if disagreement_regressed:
        reasons.append("disagreement_outcome_accuracy_regressed")
    if safety_regressed:
        reasons.append("blocker_safety_regressed")
    return {
        "promotion_status": "approved" if approved else "rejected",
        "promotion_approved": approved,
        "false_accept_rate_regressed": false_accept_regressed,
        "ranking_accuracy_regressed": ranking_regressed,
        "disagreement_outcome_accuracy_regressed": disagreement_regressed,
        "safety_regressed": safety_regressed,
        "rejection_reasons": reasons,
        "release_ready": False,
        "can_mark_release_ready": False,
    }


def copy_model_version(candidate_model_dir: Path, active_model_dir: Path) -> list[str]:
    active_model_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for filename in list(MODEL_FILES.values()) + ["model_manifest.json"]:
        source = candidate_model_dir / filename
        if not source.exists():
            raise FileNotFoundError(f"candidate_model_file_missing:{source}")
        target = active_model_dir / filename
        shutil.copy2(source, target)
        copied.append(str(target))
    return copied


def promote_model(
    *,
    comparison_json: Path,
    candidate_model_dir: Path,
    active_model_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    comparison = read_json(comparison_json)
    decision = evaluate_promotion(comparison)
    copied_files: list[str] = []
    if decision["promotion_approved"]:
        copied_files = copy_model_version(candidate_model_dir, active_model_dir)
    manifest = {
        "comparison_json": str(comparison_json),
        "candidate_model_dir": str(candidate_model_dir),
        "active_model_dir": str(active_model_dir),
        "copied_files": copied_files,
        **decision,
    }
    write_json(output_dir / "promotion_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote a no-reference critic model only when evaluation safety gates pass.")
    parser.add_argument("--comparison-json", type=Path, required=True)
    parser.add_argument("--candidate-model-dir", type=Path, required=True)
    parser.add_argument("--active-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = promote_model(
        comparison_json=args.comparison_json,
        candidate_model_dir=args.candidate_model_dir,
        active_model_dir=args.active_model_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps({"promotion_status": manifest["promotion_status"], "release_ready": False}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
