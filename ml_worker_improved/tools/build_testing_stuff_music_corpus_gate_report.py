#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


MODE_SPECIFIC_NEXT_GATES = {
    "vocal": "vocal_judge_selector_listen_preview",
    "minus": "minus_self_result_selector_listen_preview",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def stable_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def candidate_rows_from_matrix(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("ranked_candidates", "rows", "candidates"):
        rows = matrix.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def candidate_stem(candidate: dict[str, Any], matrix: dict[str, Any]) -> str:
    return str(
        candidate.get("stem_type")
        or candidate.get("target_stem")
        or candidate.get("score_mode")
        or matrix.get("score_mode")
        or ""
    )


def candidate_is_accepted(candidate: dict[str, Any]) -> bool:
    blockers = candidate.get("stem_semantic_blockers")
    if isinstance(blockers, list) and any(str(item) for item in blockers):
        return False
    if candidate.get("semantic_quality_verified") is False:
        return False
    return (
        str(candidate.get("delivery_status") or "") == "accepted"
        and candidate.get("delivery_allowed") is True
        and str(candidate.get("presentation_status") or "") == "accepted_for_listen_pack"
        and str(candidate.get("release_status") or "") == "listen_preview_only"
        and candidate.get("release_ready") is not True
        and candidate.get("can_mark_release_ready") is not True
        and candidate.get("can_publish_as_release") is not True
    )


def best_score_candidates_by_track_stem(score_matrix_jsons: Sequence[Path]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    accepted: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for matrix_path in score_matrix_jsons:
        matrix = read_json(matrix_path)
        for candidate in candidate_rows_from_matrix(matrix):
            stem = candidate_stem(candidate, matrix)
            track_id = str(candidate.get("track_id") or "")
            if not track_id or not stem or not candidate_is_accepted(candidate):
                continue
            enriched = dict(candidate)
            enriched["_score_matrix_path"] = str(matrix_path)
            accepted[(track_id, stem)].append(enriched)
    for key, candidates in accepted.items():
        accepted[key] = sorted(candidates, key=lambda row: float(row.get("selection_score") or 0.0), reverse=True)
    return accepted


def plan_summary(corpus_plan_jsonl: Path) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    rows = list(iter_jsonl(corpus_plan_jsonl))
    stems_by_track: dict[str, list[str]] = {}
    for row in rows:
        track_id = str(row.get("track_id") or "")
        if not track_id:
            continue
        target_stems = row.get("target_stems")
        stems_by_track[track_id] = [str(stem) for stem in target_stems] if isinstance(target_stems, list) else []
    return rows, stems_by_track


def execution_rows(execution_results_jsonls: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in execution_results_jsonls:
        for row in iter_jsonl(path):
            enriched = dict(row)
            enriched["_execution_results_jsonl"] = str(path)
            rows.append(enriched)
    return rows


def execution_candidate_path(row: dict[str, Any]) -> str:
    candidate_manifest = row.get("candidate_manifest")
    if isinstance(candidate_manifest, dict):
        return str(candidate_manifest.get("path") or "")
    return ""


def normalize_artifact_path(path: str) -> str:
    return str(path or "").replace("\\", "/").rstrip("/")


def candidate_matches_execution(candidate: dict[str, Any], row: dict[str, Any]) -> bool:
    model_id = str(row.get("model_id") or row.get("model_name") or "")
    candidate_model = str(candidate.get("model_id") or candidate.get("model_name") or "")
    if model_id and candidate_model and model_id == candidate_model:
        return True
    candidate_manifest_path = normalize_artifact_path(execution_candidate_path(row))
    selected_audio_path = normalize_artifact_path(str(candidate.get("selected_audio_path") or ""))
    if candidate_manifest_path and selected_audio_path:
        manifest_dir = candidate_manifest_path.rsplit("/", 1)[0]
        if selected_audio_path == manifest_dir or selected_audio_path.startswith(manifest_dir + "/"):
            return True
    return False


def accepted_candidate_for_execution(
    row: dict[str, Any],
    accepted_by_track_stem: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any] | None:
    track_id = str(row.get("track_id") or "")
    stem = str(row.get("target_stem") or row.get("stem_type") or "")
    candidates = accepted_by_track_stem.get((track_id, stem), [])
    for candidate in candidates:
        if candidate_matches_execution(candidate, row):
            return candidate
    return None


def report_row_from_execution(row: dict[str, Any], accepted_by_track_stem: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[str, Any]:
    track_id = str(row.get("track_id") or "")
    stem = str(row.get("target_stem") or row.get("stem_type") or "")
    model_id = str(row.get("model_id") or row.get("model_name") or "")
    execution_status = str(row.get("execution_status") or "")
    base = {
        "track_id": track_id,
        "stem_type": stem,
        "model_id": model_id,
        "execution_status": execution_status,
        "candidate_manifest_path": execution_candidate_path(row),
        "execution_results_jsonl": str(row.get("_execution_results_jsonl") or ""),
        "delivery_allowed": False,
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
    }
    accepted = accepted_candidate_for_execution(row, accepted_by_track_stem)
    if accepted is not None:
        return {
            **base,
            "gate_status": str(accepted.get("next_action") or "accepted_for_listen_pack_not_release"),
            "required_next_gate": "",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_status": "listen_preview_only",
            "selected_audio_path": str(accepted.get("selected_audio_path") or ""),
            "score_matrix_path": str(accepted.get("_score_matrix_path") or ""),
            "selection_score": accepted.get("selection_score"),
            "semantic_quality_verified": accepted.get("semantic_quality_verified"),
            "stem_semantic_score": accepted.get("stem_semantic_score"),
            "stem_semantic_blockers": accepted.get("stem_semantic_blockers") or [],
        }
    if execution_status != "completed":
        return {
            **base,
            "gate_status": f"execution_{execution_status or 'unknown'}",
            "required_next_gate": "complete_candidate_generation",
            "delivery_status": "blocked",
            "presentation_status": "diagnostic_execution_not_for_listen_pack",
            "release_status": "not_release_ready",
            "selected_audio_path": "",
        }
    if stem in MODE_SPECIFIC_NEXT_GATES:
        return {
            **base,
            "gate_status": "diagnostic_only_missing_mode_specific_gate",
            "required_next_gate": MODE_SPECIFIC_NEXT_GATES[stem],
            "delivery_status": "blocked",
            "presentation_status": "diagnostic_candidate_not_for_listen_pack",
            "release_status": "not_release_ready",
            "selected_audio_path": "",
        }
    return {
        **base,
        "gate_status": "diagnostic_only_missing_stem_score_matrix",
        "required_next_gate": "stem_listen_semantic_score_matrix",
        "delivery_status": "blocked",
        "presentation_status": "diagnostic_candidate_not_for_listen_pack",
        "release_status": "not_release_ready",
        "selected_audio_path": "",
    }


def build_summary(rows: list[dict[str, Any]], *, plan_rows: list[dict[str, Any]]) -> dict[str, Any]:
    gate_counts = Counter(str(row.get("gate_status") or "") for row in rows)
    delivery_allowed = sum(1 for row in rows if row.get("delivery_allowed") is True)
    diagnostic_only = sum(1 for row in rows if str(row.get("gate_status") or "").startswith("diagnostic_only"))
    return {
        "schema": "testing_stuff_music_corpus_gate_report_summary.v1",
        "planned_track_count": len(plan_rows),
        "observed_track_count": len({str(row.get("track_id") or "") for row in rows if row.get("track_id")}),
        "candidate_count": len(rows),
        "accepted_listen_preview_candidates": delivery_allowed,
        "diagnostic_only_candidates": diagnostic_only,
        "missing_mode_specific_gate_count": gate_counts.get("diagnostic_only_missing_mode_specific_gate", 0),
        "missing_stem_score_matrix_count": gate_counts.get("diagnostic_only_missing_stem_score_matrix", 0),
        "execution_blocked_count": sum(1 for row in rows if str(row.get("gate_status") or "").startswith("execution_")),
        "gate_status_counts": dict(sorted(gate_counts.items())),
        "delivery_allowed": False,
        "release_ready": False,
    }


def build_report(
    *,
    corpus_plan_jsonl: Path,
    execution_results_jsonls: Sequence[Path],
    score_matrix_jsons: Sequence[Path],
) -> dict[str, Any]:
    plan_rows, stems_by_track = plan_summary(corpus_plan_jsonl)
    accepted_by_track_stem = best_score_candidates_by_track_stem(score_matrix_jsons)
    rows = [
        report_row_from_execution(row, accepted_by_track_stem)
        for row in execution_rows(execution_results_jsonls)
    ]
    rows.sort(key=lambda row: (str(row.get("track_id") or ""), str(row.get("stem_type") or ""), str(row.get("model_id") or "")))
    summary = build_summary(rows, plan_rows=plan_rows)
    return {
        "schema": "testing_stuff_music_corpus_gate_report.v1",
        "manifest_type": "testing_stuff_music_corpus_gate_report",
        "delivery_allowed": False,
        "presentation_status": "diagnostic_corpus_gate_report_not_for_listen_pack",
        "release_status": "not_release_ready",
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "source_corpus_plan_jsonl": source_ref(corpus_plan_jsonl),
        "source_execution_results_jsonls": [source_ref(path) for path in execution_results_jsonls],
        "source_score_matrix_jsons": [source_ref(path) for path in score_matrix_jsons],
        "planned_target_stems_by_track": stems_by_track,
        "summary": summary,
        "rows": rows,
        "safety": {
            "reference_required": False,
            "human_csv_required": False,
            "training_started": False,
            "fine_tune_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "cleanup_routes_added": False,
            "thresholds_relaxed": False,
            "release_ready": False,
        },
    }


def build_summary_md(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Testing Stuff Music Corpus Gate Report",
        "",
        f"- candidate_count: {summary['candidate_count']}",
        f"- accepted_listen_preview_candidates: {summary['accepted_listen_preview_candidates']}",
        f"- diagnostic_only_candidates: {summary['diagnostic_only_candidates']}",
        f"- missing_mode_specific_gate_count: {summary['missing_mode_specific_gate_count']}",
        f"- missing_stem_score_matrix_count: {summary['missing_stem_score_matrix_count']}",
        f"- delivery_allowed: {str(report.get('delivery_allowed') is True).lower()}",
        f"- release_ready: {str(report.get('release_ready') is True).lower()}",
        "",
        "## Gate Status Counts",
        "",
    ]
    for status, count in summary["gate_status_counts"].items():
        lines.append(f"- {status}: {count}")
    lines.extend(["", "## Rows", ""])
    for row in report["rows"]:
        lines.append(
            "- "
            f"{row.get('track_id')} {row.get('stem_type')} {row.get('model_id')} "
            f"status={row.get('gate_status')} next={row.get('required_next_gate') or '-'}"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fail-closed gate report for testing_stuff_music corpus candidate waves."
    )
    parser.add_argument("--corpus-plan-jsonl", type=Path, required=True)
    parser.add_argument("--execution-results-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--score-matrix-json", type=Path, action="append", default=[])
    parser.add_argument("--output-manifest-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        corpus_plan_jsonl=args.corpus_plan_jsonl,
        execution_results_jsonls=args.execution_results_jsonl,
        score_matrix_jsons=args.score_matrix_json,
    )
    write_json(args.output_manifest_json, report)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(build_summary_md(report), encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
