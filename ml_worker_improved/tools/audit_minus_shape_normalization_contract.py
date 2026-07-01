from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


SAFETY_FLAGS = {
    "delivery_allowed": False,
    "can_publish_as_result": False,
    "release_ready": False,
    "model_downloaded": False,
    "model_registry_edited": False,
    "training_started": False,
    "thresholds_relaxed": False,
    "cleanup_routes_added": False,
    "audio_read": False,
    "audio_exported": False,
}
RAW_CANDIDATE_ID_MAX_LEN = 32
RAW_CANDIDATE_SUFFIX = "__raw_shape"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def raw_candidate_id(candidate_id: str) -> str:
    base = "".join(
        ch if ch.isascii() and (ch.isalnum() or ch in {"_", "-"}) else "_"
        for ch in candidate_id
    ).strip("_-")
    base = base or "candidate"
    direct_id = f"{base}{RAW_CANDIDATE_SUFFIX}"
    if len(direct_id) <= RAW_CANDIDATE_ID_MAX_LEN:
        return direct_id
    digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:8]
    prefix_len = RAW_CANDIDATE_ID_MAX_LEN - len(RAW_CANDIDATE_SUFFIX) - len(digest) - 1
    prefix = base[:prefix_len].strip("_-") or "candidate"
    return f"{prefix}_{digest}{RAW_CANDIDATE_SUFFIX}"


def path_exists(path_text: str) -> bool:
    return bool(path_text) and Path(path_text).exists()


def cluster_summary(forensics_dir: Path | None) -> dict[str, Any]:
    if forensics_dir is None:
        return {}
    clusters = read_json(forensics_dir / "minus_window_blocker_clusters.json")
    summary = clusters.get("summary")
    return summary if isinstance(summary, dict) else {}


def normalization_issue_suspected(summary: dict[str, Any]) -> bool:
    if summary.get("normalization_correlated") is True:
        return True
    evidence = summary.get("normalization_evidence")
    if not isinstance(evidence, dict):
        return False
    normalized = float(evidence.get("shape_normalized_avg_blocker_count") or 0.0)
    raw = float(evidence.get("non_normalized_avg_blocker_count") or 0.0)
    return normalized > raw and normalized > 0.0


def build_raw_rejudge_row(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    if row.get("shape_normalization_applied") is not True:
        return None, "shape_normalization_not_applied"
    raw_minus = str(row.get("raw_minus_path") or "").strip()
    normalized_minus = str(row.get("normalized_minus_path") or row.get("minus_path") or row.get("instrumental_path") or "").strip()
    raw_vocal = str(row.get("raw_vocal_path") or "").strip()
    if not raw_minus:
        return None, "missing_raw_minus_path"
    if raw_minus == normalized_minus:
        return None, "raw_minus_matches_normalized_minus"
    if not path_exists(raw_minus):
        return None, "raw_minus_path_missing_on_disk"
    if not raw_vocal:
        return None, "missing_raw_vocal_path"
    if not path_exists(raw_vocal):
        return None, "raw_vocal_path_missing_on_disk"

    candidate_id = str(row.get("candidate_id") or "")
    return {
        **row,
        "candidate_id": raw_candidate_id(candidate_id),
        "source_candidate_id": candidate_id,
        "candidate_origin": "raw_shape_normalization_contract_audit",
        "candidate_family": "shape_normalization_raw_audit",
        "vocal_path": raw_vocal,
        "minus_path": raw_minus,
        "instrumental_path": raw_minus,
        "selected_minus_path": raw_minus,
        "shape_normalization_audit_mode": "raw_outputs_before_shape_normalization",
        "raw_shape_audit_source_row": str(row.get("source_jsonl") or ""),
        "quality_status": "requires_raw_shape_minus_judge",
        **SAFETY_FLAGS,
    }, ""


def recommended_next_step(*, raw_rejudge_count: int, issue_suspected: bool) -> str:
    if raw_rejudge_count > 0 and issue_suspected:
        return "run_raw_shape_minus_judge"
    if raw_rejudge_count > 0:
        return "optional_raw_shape_minus_judge"
    if issue_suspected:
        return "plan_shape_normalization_contract_fix"
    return "plan_segment_level_minus_repair"


def build_shape_normalization_audit(
    *,
    route_dir: Path,
    forensics_dir: Path | None = None,
) -> dict[str, Any]:
    candidate_pool_path = route_dir / "minus_route_candidate_pool.jsonl"
    rows = list(iter_jsonl(candidate_pool_path))
    for row in rows:
        row["source_jsonl"] = str(candidate_pool_path)

    raw_rejudge_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        raw_row, reason = build_raw_rejudge_row(row)
        if raw_row is None:
            skipped.append(
                {
                    "candidate_id": str(row.get("candidate_id") or ""),
                    "route_id": str(row.get("route_id") or ""),
                    "skip_reason": reason,
                    "shape_normalization_applied": row.get("shape_normalization_applied") is True,
                    "release_ready": False,
                }
            )
            continue
        raw_rejudge_rows.append(raw_row)

    clusters = cluster_summary(forensics_dir)
    issue_suspected = normalization_issue_suspected(clusters)
    skip_reasons = Counter(str(row["skip_reason"]) for row in skipped)
    summary = {
        "candidate_count": len(rows),
        "shape_normalized_candidates": sum(1 for row in rows if row.get("shape_normalization_applied") is True),
        "raw_rejudge_candidates": len(raw_rejudge_rows),
        "skipped_raw_rejudge_candidates": len(skipped),
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "normalization_issue_suspected": issue_suspected,
        "normalization_evidence": clusters.get("normalization_evidence") if isinstance(clusters.get("normalization_evidence"), dict) else {},
        "recommended_next_step": recommended_next_step(
            raw_rejudge_count=len(raw_rejudge_rows),
            issue_suspected=issue_suspected,
        ),
        "route_dir": str(route_dir),
        "forensics_dir": str(forensics_dir or ""),
        **SAFETY_FLAGS,
    }
    return {"summary": summary, "raw_rejudge_rows": raw_rejudge_rows, "skipped_rows": skipped}


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Minus Shape Normalization Audit",
        "",
        f"- candidate_count: {summary['candidate_count']}",
        f"- shape_normalized_candidates: {summary['shape_normalized_candidates']}",
        f"- raw_rejudge_candidates: {summary['raw_rejudge_candidates']}",
        f"- skipped_raw_rejudge_candidates: {summary['skipped_raw_rejudge_candidates']}",
        f"- normalization_issue_suspected: {str(summary['normalization_issue_suspected']).lower()}",
        f"- recommended_next_step: {summary['recommended_next_step']}",
        f"- delivery_allowed: {str(summary['delivery_allowed']).lower()}",
        f"- release_ready: {str(summary['release_ready']).lower()}",
        "",
        "## Skip Reasons",
        "",
    ]
    for key, value in summary["skip_reasons"].items():
        lines.append(f"- {key}: {value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    write_json(output_dir / "shape_normalization_audit.json", result)
    write_jsonl(output_dir / "raw_rejudge_candidates.jsonl", result["raw_rejudge_rows"])
    write_jsonl(output_dir / "raw_rejudge_skipped.jsonl", result["skipped_rows"])
    write_summary_md(output_dir / "shape_normalization_audit_summary.md", result["summary"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit minus route shape-normalization contract and build raw rejudge rows.")
    parser.add_argument("--route-dir", type=Path, required=True)
    parser.add_argument("--forensics-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = build_shape_normalization_audit(route_dir=args.route_dir, forensics_dir=args.forensics_dir)
    write_outputs(result, args.output_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
