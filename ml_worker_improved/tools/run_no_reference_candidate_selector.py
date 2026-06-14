from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.no_reference_score_matrix import validate_mode  # noqa: E402


ACCEPT_SCORE_THRESHOLD = 0.72
MIN_CONFIDENCE = 0.55
CLEAR_WIN_MARGIN = 0.06
SEGMENT_BLEND_MARGIN = 0.04

SEVERE_REASONS = {
    "bad_mixture_reconstruction",
    "vocal_leakage_in_minus",
    "instrumental_leakage_in_vocal",
    "silence_hallucination",
    "robotic_or_noise_artifact",
    "thin_or_overcleaned_vocal",
    "transient_damage",
    "phase_or_stereo_inconsistency",
    "low_no_reference_score",
}

BASE_FORBIDDEN_ACTIONS = [
    "do_not_require_reference_stems",
    "do_not_require_human_csv",
    "do_not_start_finetune",
    "do_not_train",
    "do_not_add_models",
    "do_not_add_cleanup_debleed_denoise",
    "do_not_relax_gates",
    "do_not_export_diagnostic_audio_as_result",
    "do_not_accept_best_of_bad",
    "do_not_execute_segment_blend_without_approval",
]

SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "diagnostic_audio_exported": False,
    "audio_copied": False,
    "audio_exported": False,
    "segment_blend_executed": False,
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def stable_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def row_score(row: dict[str, Any], score_mode: str = "both") -> float:
    score_mode = validate_mode(score_mode)
    if score_mode == "vocal":
        return float(row.get("vocal_score") or row.get("overall_no_reference_score") or 0.0)
    if score_mode == "minus":
        return float(row.get("minus_score") or row.get("overall_no_reference_score") or 0.0)
    return float(row.get("pair_compatibility_score") or row.get("overall_no_reference_score") or 0.0)


def row_decision(row: dict[str, Any], score_mode: str = "both") -> str:
    score_mode = validate_mode(score_mode)
    if score_mode in {"vocal", "minus"}:
        return str(row.get(f"{score_mode}_decision") or row.get("decision") or "")
    pair_decision = str(row.get("pair_decision") or "")
    if pair_decision == "pair_compatible":
        return "accept_candidate"
    if pair_decision:
        return pair_decision
    return str(row.get("decision") or "")


def row_confidence(row: dict[str, Any]) -> float:
    return float(row.get("confidence") or 0.0)


def row_reasons(row: dict[str, Any]) -> list[str]:
    reasons = as_list(row.get("reasons"))
    gate = row.get("minus_musical_detail_preservation_gate")
    if isinstance(gate, dict) and str(gate.get("status") or "") not in {"", "proxy_pass"}:
        reasons.append("minus_detail_gate_blocked")
        blockers = gate.get("blockers")
        if isinstance(blockers, list):
            for blocker in blockers:
                if isinstance(blocker, dict):
                    reasons.extend(as_list(blocker.get("reason") or blocker.get("label")))
    return stable_unique(reasons)


def mode_blockers(row: dict[str, Any], score_mode: str = "both") -> list[str]:
    score_mode = validate_mode(score_mode)
    blocker_key = {
        "vocal": "vocal_blockers",
        "minus": "minus_blockers",
        "both": "pair_blockers",
    }[score_mode]
    if blocker_key in row and row.get(blocker_key) is not None:
        return as_list(row.get(blocker_key))
    return []


def has_severe_blocker(row: dict[str, Any], score_mode: str = "both") -> bool:
    if validate_mode(score_mode) == "minus":
        gate = row.get("minus_musical_detail_preservation_gate")
        if isinstance(gate, dict) and str(gate.get("status") or "") not in {"", "proxy_pass"}:
            return True
    blockers = mode_blockers(row, score_mode)
    if blockers:
        return True
    blocker_key = {
        "vocal": "vocal_blockers",
        "minus": "minus_blockers",
        "both": "pair_blockers",
    }[validate_mode(score_mode)]
    if blocker_key in row and row.get(blocker_key) is not None:
        return False
    return any(reason in SEVERE_REASONS for reason in row_reasons(row))


def is_acceptable(row: dict[str, Any], score_mode: str = "both") -> bool:
    return (
        row_decision(row, score_mode) == "accept_candidate"
        and row_score(row, score_mode) >= ACCEPT_SCORE_THRESHOLD
        and row_confidence(row) >= MIN_CONFIDENCE
        and not has_severe_blocker(row, score_mode)
    )


def has_segment_complement(rows: list[dict[str, Any]]) -> bool:
    segment_reasons = {
        reason
        for row in rows
        for reason in row_reasons(row)
        if reason.startswith("segment_winner_")
    }
    return len(segment_reasons) >= 2


def ranked_candidate(row: dict[str, Any], rank: int, score_mode: str = "both") -> dict[str, Any]:
    return {
        "rank": rank,
        "candidate_id": str(row.get("candidate_id") or ""),
        "score_mode": validate_mode(score_mode),
        "selection_score": row_score(row, score_mode),
        "overall_no_reference_score": float(row.get("overall_no_reference_score") or 0.0),
        "confidence": row_confidence(row),
        "decision": row_decision(row, score_mode),
        "reasons": row_reasons(row),
    }


def select_track(track_id: str, candidates: list[dict[str, Any]], *, score_mode: str = "both") -> dict[str, Any]:
    score_mode = validate_mode(score_mode)
    ordered = sorted(
        candidates,
        key=lambda row: (-row_score(row, score_mode), -row_confidence(row), str(row.get("candidate_id") or "")),
    )
    ranked = [ranked_candidate(row, index + 1, score_mode) for index, row in enumerate(ordered)]
    acceptable = [row for row in ordered if is_acceptable(row, score_mode)]
    all_reasons = stable_unique(reason for row in ordered for reason in row_reasons(row))
    forbidden_actions = stable_unique(BASE_FORBIDDEN_ACTIONS + [action for row in ordered for action in as_list(row.get("forbidden_actions"))])

    if not acceptable:
        reasons = stable_unique(["no_acceptable_candidate"] + all_reasons)
        if any(row_confidence(row) < MIN_CONFIDENCE for row in ordered):
            reasons = stable_unique(["low_confidence"] + reasons)
        return {
            "track_id": track_id,
            "score_mode": score_mode,
            "decision": "blocked_hard_case",
            "selected_candidate_id": None,
            "selected_candidate": None,
            "candidate_count": len(ordered),
            "accepted_candidate_count": 0,
            "ranked_candidates": ranked,
            "confidence": max((row_confidence(row) for row in ordered), default=0.0),
            "reasons": reasons,
            "segment_blend_plan": None,
            "forbidden_actions": forbidden_actions,
        }

    top = acceptable[0]
    if row_confidence(top) < MIN_CONFIDENCE:
        return {
            "track_id": track_id,
            "score_mode": score_mode,
            "decision": "blocked_hard_case",
            "selected_candidate_id": None,
            "selected_candidate": None,
            "candidate_count": len(ordered),
            "accepted_candidate_count": len(acceptable),
            "ranked_candidates": ranked,
            "confidence": row_confidence(top),
            "reasons": stable_unique(["low_confidence"] + all_reasons),
            "segment_blend_plan": None,
            "forbidden_actions": forbidden_actions,
        }

    if len(acceptable) >= 2:
        gap = row_score(acceptable[0], score_mode) - row_score(acceptable[1], score_mode)
        if gap <= SEGMENT_BLEND_MARGIN and has_segment_complement(acceptable[:2]):
            return {
                "track_id": track_id,
                "score_mode": score_mode,
                "decision": "needs_segment_blend",
                "selected_candidate_id": None,
                "selected_candidate": None,
                "candidate_count": len(ordered),
                "accepted_candidate_count": len(acceptable),
                "ranked_candidates": ranked,
                "confidence": min(row_confidence(acceptable[0]), row_confidence(acceptable[1])),
                "reasons": stable_unique(["segment_level_candidate_disagreement"] + all_reasons),
                "segment_blend_plan": {
                    "status": "plan_only_not_executed",
                    "candidate_ids": [str(acceptable[0].get("candidate_id") or ""), str(acceptable[1].get("candidate_id") or "")],
                    "reason": "close_complementary_candidates",
                },
                "forbidden_actions": forbidden_actions,
            }

    first_input_candidate = sorted(candidates, key=lambda row: str(row.get("candidate_id") or ""))[0]
    top_id = str(top.get("candidate_id") or "")
    first_id = str(first_input_candidate.get("candidate_id") or "")
    nearest_other_score = row_score(acceptable[1], score_mode) if len(acceptable) > 1 else max(
        (row_score(row, score_mode) for row in ordered if str(row.get("candidate_id") or "") != top_id),
        default=0.0,
    )
    clear_win = (row_score(top, score_mode) - nearest_other_score) >= CLEAR_WIN_MARGIN
    decision = "select_alternative_candidate" if (len(candidates) > 1 and (top_id != first_id or clear_win)) else "accept_candidate"

    return {
        "track_id": track_id,
        "score_mode": score_mode,
        "decision": decision,
        "selected_candidate_id": top_id,
        "selected_candidate": top,
        "candidate_count": len(ordered),
        "accepted_candidate_count": len(acceptable),
        "ranked_candidates": ranked,
        "confidence": row_confidence(top),
        "reasons": stable_unique(["selected_by_no_reference_score"] + row_reasons(top)),
        "segment_blend_plan": None,
        "forbidden_actions": forbidden_actions,
    }


def select_candidates(rows: list[dict[str, Any]], *, score_mode: str = "both") -> dict[str, Any]:
    score_mode = validate_mode(score_mode)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        track_id = str(row.get("track_id") or "")
        if not track_id:
            raise ValueError(f"Judge row is missing track_id: {row}")
        grouped[track_id].append(row)
    selected = [select_track(track_id, grouped[track_id], score_mode=score_mode) for track_id in sorted(grouped)]
    return {"rows": selected, "summary": build_summary(selected, score_mode=score_mode)}


def build_summary(rows: list[dict[str, Any]], *, score_mode: str = "both") -> dict[str, Any]:
    decision_counts = Counter(str(row.get("decision") or "") for row in rows)
    candidates_judged = sum(int(row.get("candidate_count") or 0) for row in rows)
    reasons = Counter(reason for row in rows for reason in row_reasons(row))
    return {
        "score_mode": validate_mode(score_mode),
        "total_tracks": len(rows),
        "candidates_judged": candidates_judged,
        "decision_counts": dict(sorted(decision_counts.items())),
        "accepted_candidates": decision_counts.get("accept_candidate", 0),
        "selected_alternatives": decision_counts.get("select_alternative_candidate", 0),
        "segment_blend_needed": decision_counts.get("needs_segment_blend", 0),
        "blocked_hard_case": decision_counts.get("blocked_hard_case", 0),
        "usable_without_stems": decision_counts.get("accept_candidate", 0) + decision_counts.get("select_alternative_candidate", 0),
        "top_failure_reasons": dict(reasons.most_common(10)),
        "safety": dict(SAFETY_FLAGS),
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# No-Reference Result Summary",
        "",
        f"- score_mode: {summary.get('score_mode', 'both')}",
        f"- total_tracks: {summary['total_tracks']}",
        f"- candidates_judged: {summary['candidates_judged']}",
        f"- accepted_candidates: {summary['accepted_candidates']}",
        f"- selected_alternatives: {summary['selected_alternatives']}",
        f"- segment_blend_needed: {summary['segment_blend_needed']}",
        f"- blocked_hard_case: {summary['blocked_hard_case']}",
        f"- usable_without_stems: {summary['usable_without_stems']}",
        "",
        "## Decisions",
        "",
    ]
    for key, value in summary["decision_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Top Failure Reasons", ""])
    for key, value in summary["top_failure_reasons"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Safety", ""])
    for key, value in summary["safety"].items():
        lines.append(f"- {key}: {str(value).lower()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "no_reference_candidate_selection.jsonl", result["rows"])
    write_summary_md(output_dir / "no_reference_result_summary.md", result["summary"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select no-reference vocal/minus candidates without exporting audio.")
    parser.add_argument("--judge-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-mode", choices=("vocal", "minus", "both"), default="both")
    args = parser.parse_args(argv)

    rows = list(iter_jsonl(args.judge_jsonl))
    if not rows:
        raise SystemExit("No no-reference judge rows supplied")
    result = select_candidates(rows, score_mode=args.score_mode)
    write_outputs(result, args.output_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
