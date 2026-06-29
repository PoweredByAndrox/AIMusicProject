from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from .stem_route_policy_common import (
        ACTION_BLOCK_UNKNOWN,
        ACTION_QUARANTINE_ROUTE,
        COUNTERFACTUAL_LABEL_SOURCES,
        ISSUE_BAD_RECONSTRUCTION,
        ISSUE_INFRA_FAILURE,
        extract_lane_routes,
        extract_route_signature,
        normalize_issue_type,
        normalize_policy_action,
        safe_policy_row_id,
        validate_policy_dataset_row,
    )
except ImportError:  # pragma: no cover - direct script execution
    from stem_route_policy_common import (  # type: ignore
        ACTION_BLOCK_UNKNOWN,
        ACTION_QUARANTINE_ROUTE,
        COUNTERFACTUAL_LABEL_SOURCES,
        ISSUE_BAD_RECONSTRUCTION,
        ISSUE_INFRA_FAILURE,
        extract_lane_routes,
        extract_route_signature,
        normalize_issue_type,
        normalize_policy_action,
        safe_policy_row_id,
        validate_policy_dataset_row,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _candidate_index(run: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for candidate in run.get("candidates") or []:
        if not isinstance(candidate, Mapping):
            continue
        candidate_id = str(candidate.get("candidate_id") or "")
        window_id = str(candidate.get("window_id") or "")
        indexed[(window_id, candidate_id)] = candidate
        indexed[("", candidate_id)] = candidate
    return indexed


def _first_value(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if isinstance(value, list):
        return str(value[0] if value else "")
    return str(value or "")


def _safe_features(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _load_score(candidate: Mapping[str, Any]) -> dict[str, Any]:
    score_path = str(candidate.get("score_path") or "").strip()
    if not score_path:
        return {}
    path = Path(score_path)
    if not path.is_file():
        return {}
    try:
        return _read_json(path)
    except Exception:
        return {}


def _index_counterfactual_evidence(rows: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        for key in (
            str(row.get("policy_row_id") or ""),
            str(row.get("row_id") or ""),
            str(row.get("root_cause_row_id") or ""),
            str(row.get("source_row_id") or ""),
        ):
            if key:
                indexed[key] = row
    return indexed


def _oracle_from_evidence(
    *,
    queue_row: Mapping[str, Any],
    counterfactual: Mapping[str, Any] | None,
) -> tuple[str, str]:
    if counterfactual is not None and bool(counterfactual.get("improved")):
        action = normalize_policy_action(str(counterfactual.get("action") or counterfactual.get("oracle_action") or ""))
        if action != ACTION_BLOCK_UNKNOWN:
            return action, "counterfactual_replay"
    issue_type = normalize_issue_type(str(queue_row.get("issue_type") or ""))
    if issue_type == ISSUE_INFRA_FAILURE or bool(queue_row.get("quarantined_for_next_run")):
        return ACTION_QUARANTINE_ROUTE, "root_cause_queue"
    if issue_type == ISSUE_BAD_RECONSTRUCTION:
        return ACTION_BLOCK_UNKNOWN, "no_counterfactual_evidence"
    return ACTION_BLOCK_UNKNOWN, "no_counterfactual_evidence"


def _split_rows(rows: list[Mapping[str, Any]]) -> dict[str, list[str]]:
    train: list[str] = []
    holdout: list[str] = []
    for index, row in enumerate(rows):
        target = holdout if index % 5 == 4 else train
        target.append(str(row["row_id"]))
    return {"train": train, "holdout": holdout}


def build_stem_route_policy_dataset(
    *,
    run_json_path: Path,
    queue_jsonl_path: Path,
    output_dir: Path,
    counterfactual_evidence_jsonl_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run = _read_json(run_json_path)
    queue_rows = _read_jsonl(queue_jsonl_path)
    candidates = _candidate_index(run)
    counterfactual_rows = _read_jsonl(counterfactual_evidence_jsonl_path) if counterfactual_evidence_jsonl_path else []
    counterfactual_by_key = _index_counterfactual_evidence(counterfactual_rows)

    rows: list[dict[str, Any]] = []
    unsafe_row_count = 0
    for index, queue_row in enumerate(queue_rows, start=1):
        window_id = _first_value(queue_row, "window_ids") or str(queue_row.get("window_id") or "")
        candidate_id = _first_value(queue_row, "candidate_ids") or str(queue_row.get("candidate_id") or "")
        candidate = candidates.get((window_id, candidate_id)) or candidates.get(("", candidate_id)) or {}
        score = _load_score(candidate)
        counterfactual = (
            counterfactual_by_key.get(str(queue_row.get("row_id") or ""))
            or counterfactual_by_key.get(candidate_id)
        )
        oracle_action, oracle_label_source = _oracle_from_evidence(queue_row=queue_row, counterfactual=counterfactual)
        route_signature = str(queue_row.get("route_signature") or extract_route_signature(candidate) or "").strip()
        row = {
            "schema": "stem_route_policy_row.v1",
            "row_id": safe_policy_row_id(index),
            "source_run": str(run_json_path),
            "source_root_cause_row_id": str(queue_row.get("row_id") or ""),
            "window_id": window_id,
            "candidate_id": candidate_id,
            "route_signature": route_signature,
            "lane_routes": extract_lane_routes(candidate) or _safe_features(queue_row.get("lane_routes")),
            "issue_type": normalize_issue_type(str(queue_row.get("issue_type") or candidate.get("verdict") or "")),
            "root_cause": str(queue_row.get("root_cause") or "").strip(),
            "affected_stem": str(queue_row.get("affected_stem") or "").strip(),
            "gate_verdicts": {
                "candidate_verdict": str(candidate.get("verdict") or queue_row.get("issue_type") or ""),
                "candidate_delivery_allowed": bool(candidate.get("delivery_allowed", False)),
                "candidate_presentation_status": str(candidate.get("presentation_status") or ""),
            },
            "forensic_features": _safe_features(candidate.get("cross_stem_forensic_score") or score.get("cross_stem_forensic_score")),
            "damage_features": _safe_features(candidate.get("damage_score") or score.get("damage_score")),
            "reconstruction_features": _safe_features(candidate.get("reconstruction") or score.get("reconstruction")),
            "current_action": normalize_policy_action(str(candidate.get("route_policy_action") or ACTION_BLOCK_UNKNOWN)),
            "oracle_action": oracle_action,
            "oracle_label_source": oracle_label_source,
            "usable_for_training": True,
            "usable_for_promotion": False,
            "is_ground_truth": bool(queue_row.get("is_ground_truth", False)),
            "label_type": str(queue_row.get("label_type") or "blocked_candidate_mining"),
        }
        try:
            validate_policy_dataset_row(row)
        except ValueError:
            unsafe_row_count += 1
            row["usable_for_training"] = False
            row["validation_error"] = "unsafe_policy_dataset_row"
        rows.append(row)

    rows_jsonl = output_dir / "rows.jsonl"
    splits_json = output_dir / "splits.json"
    manifest_json = output_dir / "manifest.json"
    splits = _split_rows([row for row in rows if row.get("usable_for_training")])
    _write_jsonl(rows_jsonl, rows)
    _write_json(splits_json, {"schema": "stem_route_policy_splits.v1", **splits})
    action_counts = Counter(str(row.get("oracle_action") or "") for row in rows)
    manifest = {
        "schema": "stem_route_policy_dataset_manifest.v1",
        "manifest_path": str(manifest_json),
        "rows_jsonl": str(rows_jsonl),
        "splits_json": str(splits_json),
        "source_run_json": str(run_json_path),
        "source_queue_jsonl": str(queue_jsonl_path),
        "counterfactual_evidence_jsonl": str(counterfactual_evidence_jsonl_path or ""),
        "row_count": len(rows),
        "usable_training_row_count": sum(1 for row in rows if row.get("usable_for_training")),
        "usable_for_promotion_row_count": 0,
        "unsafe_row_count": unsafe_row_count,
        "action_counts": dict(sorted(action_counts.items())),
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "artifact_rag_write_allowed": False,
        "counterfactual_label_sources": sorted(COUNTERFACTUAL_LABEL_SOURCES),
    }
    _write_json(manifest_json, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-json", required=True, type=Path)
    parser.add_argument("--queue-jsonl", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--counterfactual-evidence-jsonl", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_stem_route_policy_dataset(
        run_json_path=args.run_json,
        queue_jsonl_path=args.queue_jsonl,
        output_dir=args.out,
        counterfactual_evidence_jsonl_path=args.counterfactual_evidence_jsonl,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
