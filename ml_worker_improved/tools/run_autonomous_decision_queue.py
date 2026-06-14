from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.artifact_status_contract import (
    infer_dominant_failure_reason,
    infer_stem_type,
    stable_track_hash,
    stable_track_split,
)
from tools.autonomous_audio_judge import (
    as_list,
    judge_case,
    load_golden_track_hashes,
    read_json,
)


SAFETY_FLAGS = {
    "audio_copied": False,
    "training_started": False,
    "policy_changed": False,
    "models_changed": False,
    "weights_added": False,
    "cleanup_routes_added": False,
    "thresholds_changed": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"jsonl_row_must_be_object:{path}:{line_number}")
            yield payload


def load_review_rows(review_pack_jsonl: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(review_pack_jsonl):
        case_id = str(row.get("benchmark_case_id") or "").strip()
        if case_id:
            rows[case_id] = row
    return rows


def candidate_manifest_to_benchmark_row(path: Path, ordinal: int) -> dict[str, Any]:
    payload = read_json(path)
    track_hash = stable_track_hash(
        payload.get("track_hash"),
        payload.get("source_audio_path"),
        payload.get("audio_path"),
        str(path),
    )
    dominant = infer_dominant_failure_reason(payload)
    return {
        "benchmark_case_id": f"candidate_manifest_{ordinal:04d}_{track_hash[:12]}",
        "source_hard_case_id": "",
        "track_hash": track_hash,
        "split": stable_track_split(track_hash),
        "stem_type": infer_stem_type(payload, path),
        "dominant_failure_reason": dominant,
        "failure_labels": as_list(payload.get("failure_labels")) or [dominant],
        "failed_checks": as_list(payload.get("failed_checks")),
        "severity_score": 0,
        "window_start_sec": None,
        "window_end_sec": None,
        "source_manifest": str(path),
        "recommended_review_type": "autonomous_manifest_judge",
    }


def count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(Counter(str(row.get(field) or "") for row in rows).most_common())


def build_summary(rows: list[dict[str, Any]], dashboard: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "autonomous_decision_summary.v1",
        "row_count": len(rows),
        "decisions_by_type": count_by(rows, "decision"),
        "confidence_bands": count_by(rows, "confidence_band"),
        "judge_modes": count_by(rows, "judge_mode"),
        "human_review_required_count": sum(1 for row in rows if row.get("human_review_required") is True),
        "optional_human_spotcheck_count": sum(1 for row in rows if row.get("optional_human_spotcheck") is True),
        "dashboard_summary": dashboard.get("summary", {}) if isinstance(dashboard, dict) else {},
        "safety": dict(SAFETY_FLAGS),
    }


def build_manifest(
    *,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    hard_case_benchmark_jsonl: Path,
    review_pack_jsonl: Path,
    golden_jsonl: Path,
    dashboard_json: Path,
    candidate_manifests: list[Path],
) -> dict[str, Any]:
    input_files = {
        "hard_case_benchmark_jsonl": str(hard_case_benchmark_jsonl),
        "review_pack_jsonl": str(review_pack_jsonl),
        "golden_jsonl": str(golden_jsonl),
        "dashboard_json": str(dashboard_json),
        "candidate_manifests": [str(path) for path in candidate_manifests],
    }
    input_sha256 = {
        key: sha256_file(Path(value))
        for key, value in input_files.items()
        if isinstance(value, str) and Path(value).exists()
    }
    if candidate_manifests:
        input_sha256["candidate_manifests"] = {
            str(path): sha256_file(path) for path in candidate_manifests if path.exists()
        }
    return {
        "schema": "autonomous_judge_manifest.v1",
        "row_count": len(rows),
        "decision_counts": summary["decisions_by_type"],
        "confidence_band_counts": summary["confidence_bands"],
        "judge_mode_counts": summary["judge_modes"],
        "human_review_required_count": summary["human_review_required_count"],
        "optional_human_spotcheck_count": summary["optional_human_spotcheck_count"],
        "input_files": input_files,
        "input_sha256": input_sha256,
        "safety": dict(SAFETY_FLAGS),
    }


def build_decision_queue(
    *,
    hard_case_benchmark_jsonl: Path,
    review_pack_jsonl: Path,
    golden_jsonl: Path,
    dashboard_json: Path,
    candidate_manifests: list[Path] | None = None,
) -> dict[str, Any]:
    review_rows = load_review_rows(review_pack_jsonl)
    golden_hashes = load_golden_track_hashes(golden_jsonl)
    dashboard = read_json(dashboard_json) if dashboard_json.exists() else {}
    benchmark_rows = list(iter_jsonl(hard_case_benchmark_jsonl))
    for index, path in enumerate(candidate_manifests or [], start=1):
        benchmark_rows.append(candidate_manifest_to_benchmark_row(path, index))

    decision_rows = [
        judge_case(
            row,
            review_rows.get(str(row.get("benchmark_case_id") or "")),
            golden_track_hashes=golden_hashes,
        )
        for row in benchmark_rows
    ]
    summary = build_summary(decision_rows, dashboard)
    manifest = build_manifest(
        rows=decision_rows,
        summary=summary,
        hard_case_benchmark_jsonl=hard_case_benchmark_jsonl,
        review_pack_jsonl=review_pack_jsonl,
        golden_jsonl=golden_jsonl,
        dashboard_json=dashboard_json,
        candidate_manifests=candidate_manifests or [],
    )
    return {"rows": decision_rows, "summary": summary, "manifest": manifest}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Autonomous Decision Summary",
        "",
        f"- `total_autonomous_decisions`: {summary['row_count']}",
        f"- `human_review_required_count`: {summary['human_review_required_count']}",
        f"- `optional_human_spotcheck_count`: {summary['optional_human_spotcheck_count']}",
        "",
        "## Decisions By Type",
        "",
    ]
    for key, value in summary["decisions_by_type"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Confidence Bands", ""])
    for key, value in summary["confidence_bands"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Judge Modes", ""])
    for key, value in summary["judge_modes"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Safety", ""])
    for key, value in summary["safety"].items():
        lines.append(f"- `{key}`: {str(value).lower()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "autonomous_decision_queue.jsonl", result["rows"])
    write_summary_md(output_dir / "autonomous_decision_summary.md", result["summary"])
    (output_dir / "autonomous_judge_manifest.json").write_text(
        json.dumps(result["manifest"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an autonomous control-plane audio decision queue.")
    parser.add_argument("--hard-case-benchmark-jsonl", type=Path, required=True)
    parser.add_argument("--review-pack-jsonl", type=Path, required=True)
    parser.add_argument("--golden-jsonl", type=Path, required=True)
    parser.add_argument("--dashboard-json", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, action="append", default=[])
    parser.add_argument("--hard-case-corpus-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    result = build_decision_queue(
        hard_case_benchmark_jsonl=args.hard_case_benchmark_jsonl,
        review_pack_jsonl=args.review_pack_jsonl,
        golden_jsonl=args.golden_jsonl,
        dashboard_json=args.dashboard_json,
        candidate_manifests=args.candidate_manifest,
    )
    write_outputs(result, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
