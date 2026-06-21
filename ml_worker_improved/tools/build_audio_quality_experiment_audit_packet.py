from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.audio_quality_critic_common import (
    BLOCKED_FILE,
    MANIFEST_FILE,
    PROGRESS_FILE,
    SAFETY_FLAGS,
    append_progress,
    write_json,
    write_jsonl,
)
from tools.docker_runtime import require_docker_runtime

PACKET_SCHEMA = "audio_quality_experiment_audit_packet.v1"
PATHS_FILE = "paths.jsonl"
PROMPT_FILE = "reviewer_prompt.md"

AUDIT_CHECKLIST = """# Audio Quality Critic Experiment Audit Packet

You are an experiment integrity auditor. Read ALL files listed below and check
for the following fraud patterns. Do not infer PASS from this packet. A PASS
requires an independent review report.

## Files To Read

{path_lines}

## Audit Checklist

### A. Ground Truth Provenance
1. Where does ground truth / reference / target come from?
2. Is it loaded from verified data, or generated/derived from model outputs?
3. If derived, is it explicitly labeled as proxy evaluation?
4. FAIL if generated or pseudo labels are treated as ground truth.

### B. Score Normalization
1. Is any metric divided by max/min/mean of the model's own output?
2. Are raw scores preserved alongside normalized scores?
3. FAIL if normalization uses prediction statistics as denominator.

### C. Result File Existence
1. Do all referenced result files exist?
2. Do claimed metric keys exist in those files?
3. Do numbers match the files exactly?

### D. Dead Code Detection
1. Are metric functions actually called?
2. Does their output appear in result files?
3. WARN if metrics exist but are unused.

### E. Scope Assessment
1. How many tracks/windows/configurations were actually tested?
2. Is the scope enough for any claimed improvement?
3. WARN if scope language exceeds evidence.

### F. Evaluation Type Classification
Classify evaluation as real_gt, human_eval, self_supervised_proxy,
synthetic_proxy, simulation_only, or mixed.

## Required Output

Write EXPERIMENT_AUDIT.json and EXPERIMENT_AUDIT.md only after independent
review. The JSON must include overall_verdict, integrity_status, auditor, and
checks. Promotion validation requires overall_verdict=PASS and
integrity_status=pass.
"""


def _path_row(category: str, path: Path) -> dict[str, Any]:
    return {
        "schema": "audio_quality_experiment_audit_path.v1",
        "category": category,
        "path": str(path),
        "exists": path.exists(),
    }


def _blocked_row(reasons: Sequence[str]) -> dict[str, Any]:
    return {
        "schema": "audio_quality_experiment_audit_packet_blocked_row.v1",
        "blocked_reasons": sorted(set(str(reason) for reason in reasons if reason)),
        **SAFETY_FLAGS,
    }


def _prompt_for_paths(path_rows: Sequence[dict[str, Any]]) -> str:
    if not path_rows:
        path_lines = "- No audit evidence paths are available."
    else:
        path_lines = "\n".join(f"- {row['category']}: {row['path']}" for row in path_rows)
    return AUDIT_CHECKLIST.format(path_lines=path_lines)


def build_audio_quality_experiment_audit_packet(
    *,
    output_dir: Path,
    model_manifest_json: Path,
    validation_json: Path,
    dataset_manifest_json: Path | None = None,
    score_manifest_json: Path | None = None,
    holdout_scores_jsonl: Path | None = None,
    extra_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    start_time = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_FILE
    if progress_path.exists():
        progress_path.unlink()
    append_progress(
        progress_path,
        "start",
        start_time=start_time,
        model_manifest_json=str(model_manifest_json),
        validation_json=str(validation_json),
        expected_acceleration_mode="external_only_cpu_experiment_audit_packet",
        gpu_inference_skipped=True,
        gpu_inference_skip_reason="audit_packet_metadata_only",
    )

    errors: list[str] = []
    requested_rows: list[dict[str, Any]] = []
    required = (
        ("model_manifest", model_manifest_json, "model_manifest_missing"),
        ("validation_result", validation_json, "validation_json_missing"),
    )
    optional = [
        ("dataset_manifest", dataset_manifest_json, "dataset_manifest_missing"),
        ("score_manifest", score_manifest_json, "score_manifest_missing"),
        ("holdout_scores", holdout_scores_jsonl, "holdout_scores_missing"),
    ]
    for category, path, missing_reason in required:
        row = _path_row(category, path)
        requested_rows.append(row)
        if not row["exists"]:
            errors.append(missing_reason)
    for category, maybe_path, missing_reason in optional:
        if maybe_path is None:
            continue
        row = _path_row(category, maybe_path)
        requested_rows.append(row)
        if not row["exists"]:
            errors.append(missing_reason)
    for index, path in enumerate(extra_paths, start=1):
        row = _path_row("extra_evidence", path)
        row["extra_index"] = index
        requested_rows.append(row)
        if not row["exists"]:
            errors.append("extra_evidence_missing")

    path_rows = [row for row in requested_rows if row["exists"] and not errors]
    blocked_rows = [_blocked_row(errors)] if errors else []
    packet_status = "blocked" if errors else "ready_for_independent_review"

    write_jsonl(output_dir / PATHS_FILE, path_rows)
    write_jsonl(output_dir / BLOCKED_FILE, blocked_rows)
    (output_dir / PROMPT_FILE).write_text(_prompt_for_paths(path_rows), encoding="utf-8")
    manifest = {
        "schema": PACKET_SCHEMA,
        "manifest_type": "audio_quality_experiment_audit_packet",
        "packet_status": packet_status,
        "errors": sorted(set(errors)),
        "path_count": len(path_rows),
        "blocked_count": len(blocked_rows),
        "reviewer_required": True,
        "experiment_audit_pass": False,
        "promotion_allowed": False,
        "training_started": False,
        "presentation_status": "audio_quality_experiment_audit_packet_not_for_listen_pack",
        "outputs": {
            "paths_jsonl": str(output_dir / PATHS_FILE),
            "blocked_jsonl": str(output_dir / BLOCKED_FILE),
            "reviewer_prompt_md": str(output_dir / PROMPT_FILE),
            "manifest_json": str(output_dir / MANIFEST_FILE),
            "progress_jsonl": str(progress_path),
        },
        "safety": {
            "independent_reviewer_required": True,
            "packet_is_not_experiment_audit_pass": True,
            "no_self_issued_promotion": True,
            "not_release_artifact": True,
        },
        **SAFETY_FLAGS,
    }
    write_json(output_dir / MANIFEST_FILE, manifest)
    append_progress(
        progress_path,
        "finish",
        start_time=start_time,
        packet_status=packet_status,
        path_count=len(path_rows),
        blocked_count=len(blocked_rows),
        experiment_audit_pass=False,
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an AQC experiment-audit reviewer packet.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-manifest-json", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--dataset-manifest-json", type=Path)
    parser.add_argument("--score-manifest-json", type=Path)
    parser.add_argument("--holdout-scores-jsonl", type=Path)
    parser.add_argument("--extra-path", type=Path, action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_audio_quality_experiment_audit_packet.py",
        entrypoint_path="/app/tools/build_audio_quality_experiment_audit_packet.py",
    )
    args = parse_args(argv)
    manifest = build_audio_quality_experiment_audit_packet(
        output_dir=args.output_dir,
        model_manifest_json=args.model_manifest_json,
        validation_json=args.validation_json,
        dataset_manifest_json=args.dataset_manifest_json,
        score_manifest_json=args.score_manifest_json,
        holdout_scores_jsonl=args.holdout_scores_jsonl,
        extra_paths=args.extra_path,
    )
    print(
        json.dumps(
            {
                "manifest_json": str(args.output_dir / MANIFEST_FILE),
                "packet_status": manifest["packet_status"],
                "path_count": manifest["path_count"],
                "blocked_count": manifest["blocked_count"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["packet_status"] == "ready_for_independent_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())
