#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime  # noqa: E402

CANDIDATES_JSONL_NAME = "no_reference_vocal_windowed_judge_candidates.jsonl"
MANIFEST_JSON_NAME = "no_reference_vocal_windowed_judge_candidates_manifest.json"
SUMMARY_MD_NAME = "no_reference_vocal_windowed_judge_candidates_summary.md"

SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "release_ready": False,
}
FORBIDDEN_ACTIONS = ["present_as_listen_result", "mark_release_ready", "start_training"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def source_ref(path: Path) -> dict[str, str]:
    return {"path": str(path)}


def normalized_target_stem(row: dict[str, Any]) -> str:
    raw = str(row.get("target_stem") or "").lower()
    if raw in {"vocal", "vocals"}:
        return "vocal"
    if raw in {"minus", "instrumental", "instrumentals"}:
        return "minus"
    candidate = row.get("candidate_manifest") if isinstance(row.get("candidate_manifest"), dict) else {}
    raw = str(candidate.get("target_stem") or "").lower()
    if raw in {"vocal", "vocals"}:
        return "vocal"
    if raw in {"minus", "instrumental", "instrumentals"}:
        return "minus"
    return raw


def completed_candidate_state(row: dict[str, Any]) -> dict[str, Any] | None:
    if str(row.get("execution_status") or "") != "completed":
        return None
    candidate = row.get("candidate_manifest")
    if not isinstance(candidate, dict) or candidate.get("exists") is not True:
        return None
    audio_path = str(candidate.get("target_audio_path") or "")
    manifest_path = str(candidate.get("path") or row.get("expected_candidate_manifest") or "")
    if not audio_path or not manifest_path:
        return None
    stem = normalized_target_stem(row)
    if stem not in {"vocal", "minus"}:
        return None
    return {
        "track_id": str(row.get("track_id") or ""),
        "source_audio_path": str(row.get("source_audio_path") or row.get("mix_path") or ""),
        "target_stem": stem,
        "model_id": str(row.get("model_id") or row.get("route_name") or stem),
        "route_family": str(row.get("route_family") or ""),
        "audio_path": audio_path,
        "candidate_manifest": manifest_path,
        "execution_result": row,
    }


def collect_completed_candidates(execution_results_jsonls: Sequence[Path]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in execution_results_jsonls:
        for row in iter_jsonl(path):
            candidate = completed_candidate_state(row)
            if candidate is not None:
                candidate["source_execution_results_jsonl"] = str(path)
                candidates.append(candidate)
    return candidates


def build_pair_rows(candidates: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    by_track: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"vocal": [], "minus": []})
    for candidate in candidates:
        track_id = str(candidate.get("track_id") or "")
        if not track_id:
            continue
        by_track[track_id][str(candidate["target_stem"])].append(candidate)

    rows: list[dict[str, Any]] = []
    unpaired = {"vocal": [], "minus": []}
    for track_id in sorted(by_track):
        vocals = sorted(by_track[track_id]["vocal"], key=lambda item: (str(item["model_id"]), str(item["audio_path"])))
        minuses = sorted(by_track[track_id]["minus"], key=lambda item: (str(item["model_id"]), str(item["audio_path"])))
        if not vocals:
            unpaired["minus"].append(track_id)
            continue
        if not minuses:
            unpaired["vocal"].append(track_id)
            continue
        for vocal in vocals:
            for minus in minuses:
                candidate_id = f"{vocal['model_id']}__{minus['model_id']}"
                rows.append(
                    {
                        "schema": "testing_stuff_music_vocal_windowed_judge_candidate_row.v1",
                        "track_id": track_id,
                        "candidate_id": candidate_id,
                        "mix_path": vocal["source_audio_path"] or minus["source_audio_path"],
                        "source_audio_path": vocal["source_audio_path"] or minus["source_audio_path"],
                        "vocal_path": vocal["audio_path"],
                        "instrumental_path": minus["audio_path"],
                        "minus_path": minus["audio_path"],
                        "candidate_manifest": vocal["candidate_manifest"],
                        "source_manifest": vocal["candidate_manifest"],
                        "score_mode": "vocal",
                        "stem_type": "vocal",
                        "vocal_model_id": vocal["model_id"],
                        "minus_model_id": minus["model_id"],
                        "vocal_route_id": vocal["model_id"],
                        "minus_route_id": minus["model_id"],
                        "route_family": "corpus_completed_vocal_minus_pair",
                        "source_vocal_execution_results_jsonl": vocal["source_execution_results_jsonl"],
                        "source_minus_execution_results_jsonl": minus["source_execution_results_jsonl"],
                        "delivery_allowed": False,
                        "presentation_status": "diagnostic_candidate_for_vocal_judge_only",
                        "release_status": "not_release_ready",
                        "release_ready": False,
                        "forbidden_actions": list(FORBIDDEN_ACTIONS),
                    }
                )
    return rows, unpaired


def build_summary(
    *,
    execution_results_jsonls: Sequence[Path],
    candidates: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    unpaired: dict[str, list[str]],
) -> dict[str, Any]:
    return {
        "schema": "testing_stuff_music_corpus_vocal_judge_inputs_summary.v1",
        "source_execution_results_jsonls": [source_ref(path) for path in execution_results_jsonls],
        "completed_candidate_count": len(candidates),
        "candidate_count": len(rows),
        "paired_track_count": len({str(row.get("track_id") or "") for row in rows}),
        "unpaired_vocal_track_count": len(unpaired["vocal"]),
        "unpaired_minus_track_count": len(unpaired["minus"]),
        "unpaired_vocal_track_ids": sorted(unpaired["vocal"]),
        "unpaired_minus_track_ids": sorted(unpaired["minus"]),
        "expected_downstream": "run_no_reference_windowed_judge --score-mode vocal",
        "expected_acceleration_mode": "external_only_cpu_dsp_scoring",
        "gpu_inference_required": False,
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "builds_windowed_judge_inputs_from_existing_completed_candidates",
        "delivery_allowed": False,
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "safety": dict(SAFETY_FLAGS),
    }


def build_manifest(rows: Sequence[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "testing_stuff_music_corpus_vocal_judge_inputs_manifest.v1",
        "manifest_type": "testing_stuff_music_corpus_vocal_judge_inputs",
        "delivery_allowed": False,
        "presentation_status": "diagnostic_candidate_rows_not_for_listen_pack",
        "release_status": "not_release_ready",
        "release_ready": False,
        "candidate_count": len(rows),
        "summary": summary,
        "safety": dict(SAFETY_FLAGS),
    }


def build_summary_md(manifest: dict[str, Any]) -> str:
    summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
    lines = [
        "# Testing Stuff Music Vocal Judge Inputs",
        "",
        f"- candidate_count: {summary.get('candidate_count', 0)}",
        f"- paired_track_count: {summary.get('paired_track_count', 0)}",
        f"- unpaired_vocal_track_count: {summary.get('unpaired_vocal_track_count', 0)}",
        f"- unpaired_minus_track_count: {summary.get('unpaired_minus_track_count', 0)}",
        f"- expected_downstream: {summary.get('expected_downstream')}",
        f"- expected_acceleration_mode: {summary.get('expected_acceleration_mode')}",
        "- delivery_allowed: false",
        "- release_ready: false",
        "",
    ]
    return "\n".join(lines)


def build_inputs(*, execution_results_jsonls: Sequence[Path]) -> dict[str, Any]:
    candidates = collect_completed_candidates(execution_results_jsonls)
    rows, unpaired = build_pair_rows(candidates)
    summary = build_summary(
        execution_results_jsonls=execution_results_jsonls,
        candidates=candidates,
        rows=rows,
        unpaired=unpaired,
    )
    return {"rows": rows, "summary": summary, "manifest": build_manifest(rows, summary)}


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    write_jsonl(output_dir / CANDIDATES_JSONL_NAME, result["rows"])
    write_json(output_dir / MANIFEST_JSON_NAME, result["manifest"])
    (output_dir / SUMMARY_MD_NAME).parent.mkdir(parents=True, exist_ok=True)
    (output_dir / SUMMARY_MD_NAME).write_text(build_summary_md(result["manifest"]), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build vocal windowed-judge candidate rows from completed testing_stuff_music corpus execution rows."
    )
    parser.add_argument("--execution-results-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_testing_stuff_music_corpus_vocal_judge_inputs.py",
        entrypoint_path="/app/tools/build_testing_stuff_music_corpus_vocal_judge_inputs.py",
    )
    args = parse_args(argv)
    result = build_inputs(execution_results_jsonls=args.execution_results_jsonl)
    write_outputs(result, args.output_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
