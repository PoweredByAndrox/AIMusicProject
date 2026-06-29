from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

try:
    from .stem_route_policy_common import (
        ACTION_BLOCK_UNKNOWN,
        ACTION_PRIORITIZE_ROUTE,
        ACTION_SWITCH_MINUS_LANE,
        ACTION_SWITCH_STEM_LANE,
        ACTION_SWITCH_VOCAL_LANE,
        extract_route_signature,
        normalize_policy_action,
    )
except ImportError:  # pragma: no cover - direct script execution
    from stem_route_policy_common import (  # type: ignore
        ACTION_BLOCK_UNKNOWN,
        ACTION_PRIORITIZE_ROUTE,
        ACTION_SWITCH_MINUS_LANE,
        ACTION_SWITCH_STEM_LANE,
        ACTION_SWITCH_VOCAL_LANE,
        extract_route_signature,
        normalize_policy_action,
    )


VERDICT_GATE_ORDER = {
    "blocked_unknown": 0,
    "blocked_missing_stem": 1,
    "blocked_bad_reconstruction": 2,
    "blocked_damage": 3,
    "blocked_leak": 4,
    "blocked_voice_presence": 4,
    "blocked_contamination": 4,
    "blocked_cross_stem_forensic": 5,
    "auto_accept": 10,
}


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def _verdict_rank(verdict: str) -> int:
    return VERDICT_GATE_ORDER.get(str(verdict or ""), 0)


def _lane_routes_from_signature(signature: str) -> dict[str, str]:
    lanes: dict[str, str] = {}
    for part in str(signature or "").split("|"):
        if ":" not in part:
            continue
        lane, route = part.split(":", 1)
        lane = lane.strip()
        route = route.strip()
        if lane and route:
            lanes[lane] = route
    return lanes


def _action_from_signature_change(before_signature: str, after_signature: str) -> str:
    before = _lane_routes_from_signature(before_signature)
    after = _lane_routes_from_signature(after_signature)
    if before.get("minus_lane") != after.get("minus_lane") and after.get("minus_lane"):
        return ACTION_SWITCH_MINUS_LANE
    if before.get("stem_lane") != after.get("stem_lane") and after.get("stem_lane"):
        return ACTION_SWITCH_STEM_LANE
    if before.get("vocal_lane") != after.get("vocal_lane") and after.get("vocal_lane"):
        return ACTION_SWITCH_VOCAL_LANE
    return ACTION_PRIORITIZE_ROUTE


def _candidate_groups_by_window(run_json_path: Path | None) -> dict[str, list[dict[str, Any]]]:
    run = _read_json(run_json_path)
    groups: dict[str, list[dict[str, Any]]] = {}
    for candidate in run.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        window_id = str(candidate.get("window_id") or "")
        groups.setdefault(window_id, []).append(candidate)
    return groups


def _observed_counterfactual_for_row(
    row: dict[str, Any],
    *,
    candidates_by_window: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    window_id = str(row.get("window_id") or "")
    candidate_id = str(row.get("candidate_id") or "")
    before_verdict = str((row.get("gate_verdicts") or {}).get("candidate_verdict") or "")
    before_rank = _verdict_rank(before_verdict)
    before_signature = str(row.get("route_signature") or "")
    alternatives = [
        candidate
        for candidate in candidates_by_window.get(window_id, [])
        if str(candidate.get("candidate_id") or "") != candidate_id
    ]
    improved: list[dict[str, Any]] = []
    for candidate in alternatives:
        after_verdict = str(candidate.get("verdict") or "")
        after_rank = _verdict_rank(after_verdict)
        after_signature = extract_route_signature(candidate)
        if after_rank <= before_rank:
            continue
        if not after_signature or after_signature == before_signature:
            continue
        improved.append(candidate)
    if not improved:
        return None
    best = max(
        improved,
        key=lambda candidate: (
            _verdict_rank(str(candidate.get("verdict") or "")),
            str(candidate.get("candidate_id") or ""),
        ),
    )
    after_signature = extract_route_signature(best)
    return {
        "after_candidate_id": str(best.get("candidate_id") or ""),
        "after_verdict": str(best.get("verdict") or ""),
        "after_route_signature": after_signature,
        "action": _action_from_signature_change(before_signature, after_signature),
        "evidence_source": "observed_run_candidate",
    }


def run_route_action_counterfactuals(
    *,
    rows_jsonl_path: Path,
    output_dir: Path,
    run_json_path: Path | None = None,
    max_actions: int = 64,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    started = time.monotonic()
    _append_jsonl(progress_path, {"event": "start", "rows_jsonl": str(rows_jsonl_path), "artifact_rag_write_allowed": False})
    rows = _read_jsonl(rows_jsonl_path)
    selected_rows = rows[: max(0, int(max_actions))]
    candidates_by_window = _candidate_groups_by_window(run_json_path)
    actions_path = output_dir / "actions.jsonl"
    before_after_path = output_dir / "before_after.jsonl"
    if actions_path.exists():
        actions_path.unlink()
    if before_after_path.exists():
        before_after_path.unlink()
    observed_improvement_count = 0
    for index, row in enumerate(selected_rows, start=1):
        observed = _observed_counterfactual_for_row(row, candidates_by_window=candidates_by_window)
        action = normalize_policy_action(
            str((observed or {}).get("action") or row.get("oracle_action") or row.get("current_action") or ACTION_BLOCK_UNKNOWN)
        )
        if observed is not None:
            observed_improvement_count += 1
        action_row = {
            "schema": "stem_route_policy_counterfactual_action.v1",
            "action_id": f"a{index:06d}",
            "policy_row_id": row.get("row_id"),
            "root_cause_row_id": row.get("source_root_cause_row_id", ""),
            "candidate_id": row.get("candidate_id"),
            "window_id": row.get("window_id"),
            "route_signature": row.get("route_signature"),
            "action": action,
            "evidence_only": True,
            "listen_pack_created": False,
            "delivery_allowed": False,
        }
        before_after = {
            "schema": "stem_route_policy_counterfactual_before_after.v1",
            "row_id": row.get("row_id"),
            "policy_row_id": row.get("row_id"),
            "root_cause_row_id": row.get("source_root_cause_row_id", ""),
            "source_row_id": row.get("source_root_cause_row_id", ""),
            "candidate_id": row.get("candidate_id"),
            "action": action,
            "before_verdict": (row.get("gate_verdicts") or {}).get("candidate_verdict", ""),
            "after_verdict": (observed or {}).get("after_verdict", (row.get("gate_verdicts") or {}).get("candidate_verdict", "")),
            "before_route_signature": row.get("route_signature"),
            "after_route_signature": (observed or {}).get("after_route_signature", row.get("route_signature")),
            "after_candidate_id": (observed or {}).get("after_candidate_id", ""),
            "improved": observed is not None,
            "counterfactual_replay_required": observed is None,
            "evidence_source": (observed or {}).get("evidence_source", "none"),
            "evidence_only": True,
            "delivery_allowed": False,
            "listen_pack_created": False,
        }
        _append_jsonl(actions_path, action_row)
        _append_jsonl(before_after_path, before_after)
        _append_jsonl(
            progress_path,
            {
                "event": "action_recorded",
                "action_id": action_row["action_id"],
                "policy_row_id": row.get("row_id"),
                "elapsed_sec": round(time.monotonic() - started, 6),
            },
        )
    manifest = {
        "schema": "stem_route_policy_counterfactual_manifest.v1",
        "manifest_path": str(output_dir / "manifest.json"),
        "rows_jsonl": str(rows_jsonl_path),
        "actions_jsonl": str(actions_path),
        "before_after_jsonl": str(before_after_path),
        "progress_jsonl": str(progress_path),
        "run_json": str(run_json_path or ""),
        "row_count": len(rows),
        "action_count": len(selected_rows),
        "observed_counterfactual_improvement_count": observed_improvement_count,
        "artifact_rag_write_allowed": False,
        "listen_pack_created": False,
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "counterfactual_labels_require_actual_before_after_evidence": True,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _append_jsonl(progress_path, {"event": "end", "action_count": len(selected_rows), "elapsed_sec": round(time.monotonic() - started, 6)})
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-jsonl", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run-json", type=Path)
    parser.add_argument("--max-actions", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_route_action_counterfactuals(
        rows_jsonl_path=args.rows_jsonl,
        output_dir=args.out,
        run_json_path=args.run_json,
        max_actions=args.max_actions,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
