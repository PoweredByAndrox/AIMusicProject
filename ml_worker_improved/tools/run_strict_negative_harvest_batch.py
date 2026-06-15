#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.train_audio_error_critic import SAFETY_FLAGS  # noqa: E402


AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a", ".aiff", ".aif", ".ogg")
MANIFEST_FILE = "negative_harvest_manifest.json"
RUNS_FILE = "negative_harvest_runs.jsonl"
SUMMARY_FILE = "negative_harvest_summary.md"
PROGRESS_FILE = "negative_harvest_progress.jsonl"
OWNED_OUTPUT_FILES = (MANIFEST_FILE, RUNS_FILE, SUMMARY_FILE, PROGRESS_FILE)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid_bool:{value}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OWNED_OUTPUT_FILES:
        path = output_dir / name
        if path.exists():
            path.unlink()


def progress(output_dir: Path, payload: dict[str, Any]) -> None:
    row = {**payload, "release_ready": False}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / PROGRESS_FILE).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)


def slugify(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "_", path.stem).strip("_").lower()
    return stem[:64] or "track"


def discover_audio_files(input_audio_dir: Path, max_tracks: int) -> list[Path]:
    if not input_audio_dir.exists():
        return []
    files = sorted(path for path in input_audio_dir.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)
    if max_tracks > 0:
        files = files[:max_tracks]
    return files


def first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return ""


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def jsonl_count(path: Path) -> int:
    return sum(1 for _ in iter_jsonl(path))


def strict_window_counts(path: Path, *, stem_type: str) -> tuple[int, int, Counter[str], int]:
    positive = 0
    negative = 0
    blockers: Counter[str] = Counter()
    row_count = 0
    for row in iter_jsonl(path):
        row_count += 1
        parent_decision = first_text(row.get("judge_decision"), row.get("decision"), row.get(f"{stem_type}_decision"))
        windows = row.get("window_metrics")
        if not isinstance(windows, list):
            continue
        for window in windows:
            if not isinstance(window, dict):
                continue
            decision = first_text(window.get("judge_decision"), window.get("decision"), window.get(f"{stem_type}_decision"), parent_decision)
            window_blockers: list[str] = []
            for key in ("blockers", f"{stem_type}_blockers"):
                window_blockers.extend(as_list(window.get(key)))
            if decision == "accept_candidate" and not window_blockers:
                positive += 1
            else:
                negative += 1
            blockers.update(window_blockers)
    return positive, negative, blockers, row_count


def preview_is_ready(path: Path) -> bool:
    payload = read_json(path)
    return payload.get("preview_ready") is True or str(payload.get("status") or "") == "preview_ready"


def harvest_example_counts(output_dir: Path) -> dict[str, Any]:
    run_dirs = sorted(path for path in output_dir.glob("run_*") if path.is_dir())
    status_counts: Counter[str] = Counter()
    blocker_counts: Counter[str] = Counter()
    candidate_pool_rows = 0
    vocal_judge_rows = 0
    minus_judge_rows = 0
    pair_rows = 0
    preview_ready_count = 0
    strict_positive_windows = 0
    strict_negative_windows = 0
    for run_dir in run_dirs:
        manifest = read_json(run_dir / "run_manifest.json")
        status = first_text(manifest.get("output_status"), manifest.get("final_runtime_status"), manifest.get("status"))
        if status:
            status_counts[status] += 1
        candidate_pool_rows += jsonl_count(run_dir / "candidate_pool.jsonl")
        pair_rows += jsonl_count(run_dir / "pair_compatibility.jsonl")
        if preview_is_ready(run_dir / "preview_manifest.json"):
            preview_ready_count += 1
        for stem_type, filename in (("vocal", "vocal_windowed_judge.jsonl"), ("minus", "minus_windowed_judge.jsonl")):
            positive, negative, blockers, row_count = strict_window_counts(run_dir / filename, stem_type=stem_type)
            strict_positive_windows += positive
            strict_negative_windows += negative
            blocker_counts.update(blockers)
            if stem_type == "vocal":
                vocal_judge_rows += row_count
            else:
                minus_judge_rows += row_count
    return {
        "run_dirs": len(run_dirs),
        "candidate_pool_rows": candidate_pool_rows,
        "vocal_judge_rows": vocal_judge_rows,
        "minus_judge_rows": minus_judge_rows,
        "pair_rows": pair_rows,
        "preview_ready_count": preview_ready_count,
        "strict_positive_windows": strict_positive_windows,
        "strict_negative_windows": strict_negative_windows,
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "final_status_counts": dict(sorted(status_counts.items())),
    }


def runtime_command(
    *,
    input_audio: Path,
    run_dir: Path,
    model_dir: Path,
    profile: str,
    candidate_diversity_profile: str,
    enable_conditional_hard_case_routes: bool,
    enable_audio_preview_render: bool,
    require_preview_audio_for_preview_ready: bool,
    continue_on_failure: bool,
) -> list[str]:
    return [
        sys.executable,
        "tools/run_autonomous_no_reference_pipeline.py",
        "--input-audio",
        str(input_audio),
        "--output-dir",
        str(run_dir),
        "--model-dir",
        str(model_dir),
        "--profile",
        profile,
        "--generate-candidates",
        "true",
        "--candidate-profile",
        profile,
        "--candidate-execution-mode",
        "auto",
        "--allow-shape-normalization",
        "true",
        "--candidate-diversity-profile",
        candidate_diversity_profile,
        "--max-candidates",
        "8",
        "--max-candidate-runtime-sec",
        "1800",
        "--continue-on-candidate-failure",
        "true" if continue_on_failure else "false",
        "--enable-conditional-hard-case-routes",
        "true" if enable_conditional_hard_case_routes else "false",
        "--conditional-route-policy",
        "hard_case_only" if enable_conditional_hard_case_routes else "disabled",
        "--max-conditional-route-jobs",
        "2",
        "--conditional-route-runtime-sec",
        "1800",
        "--enable-audio-preview-render",
        "true" if enable_audio_preview_render else "false",
        "--require-preview-audio-for-preview-ready",
        "true" if require_preview_audio_for_preview_ready else "false",
    ]


def summarize_runs(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "tracks_planned": len(rows),
        "tracks_succeeded": sum(1 for row in rows if row.get("execution_status") == "completed"),
        "tracks_failed": sum(1 for row in rows if row.get("execution_status") == "failed"),
        "tracks_planned_only": sum(1 for row in rows if row.get("execution_status") == "planned_only"),
    }


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Strict Negative Harvest Batch",
        "",
        f"- input_audio_dir: `{manifest['input_audio_dir']}`",
        f"- tracks_discovered: {manifest['tracks_discovered']}",
        f"- tracks_planned: {manifest['tracks_planned']}",
        f"- tracks_succeeded: {manifest['tracks_succeeded']}",
        f"- tracks_failed: {manifest['tracks_failed']}",
        f"- tracks_planned_only: {manifest['tracks_planned_only']}",
        f"- execute: {str(manifest['execute']).lower()}",
        "- training_started: false",
        "- separator_training_started: false",
        "- release_ready: false",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_strict_negative_harvest_batch(
    *,
    input_audio_dir: Path,
    output_dir: Path,
    model_dir: Path,
    max_tracks: int,
    profile: str,
    candidate_diversity_profile: str,
    enable_conditional_hard_case_routes: bool,
    enable_audio_preview_render: bool,
    require_preview_audio_for_preview_ready: bool,
    continue_on_failure: bool,
    execute: bool = True,
) -> dict[str, Any]:
    prepare_output_dir(output_dir)
    progress(
        output_dir,
        {
            "phase": "start",
            "input_audio_dir": str(input_audio_dir),
            "model_dir": str(model_dir),
            "profile": profile,
            "candidate_diversity_profile": candidate_diversity_profile,
            "execute": bool(execute),
            "expected_acceleration_mode": "gpu_inference_for_runtime_candidates",
            "training_started": False,
            "separator_training_started": False,
        },
    )
    audio_files = discover_audio_files(input_audio_dir, max_tracks)
    progress(output_dir, {"phase": "discovered_tracks", "tracks_discovered": len(audio_files)})
    rows: list[dict[str, Any]] = []
    for index, audio_path in enumerate(audio_files, start=1):
        run_dir = output_dir / f"run_{index:03d}_{slugify(audio_path)}"
        command = runtime_command(
            input_audio=audio_path,
            run_dir=run_dir,
            model_dir=model_dir,
            profile=profile,
            candidate_diversity_profile=candidate_diversity_profile,
            enable_conditional_hard_case_routes=enable_conditional_hard_case_routes,
            enable_audio_preview_render=enable_audio_preview_render,
            require_preview_audio_for_preview_ready=require_preview_audio_for_preview_ready,
            continue_on_failure=continue_on_failure,
        )
        row: dict[str, Any] = {
            "track_index": index,
            "input_audio_path": str(audio_path),
            "run_output_dir": str(run_dir),
            "command": command,
            "execution_status": "planned_only",
            "return_code": None,
            "candidate_artifact_requirements": {
                "candidate_id": True,
                "route_id": True,
                "stem_type": True,
                "source_model_id": True,
                "source_tool": True,
                "candidate_audio_path": True,
                "normalized_path_if_any": True,
                "window_start_end": True,
                "judge_decision": True,
                "blockers": True,
                "provenance_source_file": True,
            },
            **SAFETY_FLAGS,
        }
        if execute:
            progress(
                output_dir,
                {
                    "phase": "runtime_track_start",
                    "track_index": index,
                    "input_audio_path": str(audio_path),
                    "run_output_dir": str(run_dir),
                    "command": command,
                },
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(command, cwd=APP_ROOT, text=True, capture_output=True, check=False)
            row["return_code"] = completed.returncode
            row["stdout_tail"] = completed.stdout[-4000:]
            row["stderr_tail"] = completed.stderr[-4000:]
            row["execution_status"] = "completed" if completed.returncode == 0 else "failed"
            progress(
                output_dir,
                {
                    "phase": "runtime_track_end",
                    "track_index": index,
                    "input_audio_path": str(audio_path),
                    "run_output_dir": str(run_dir),
                    "return_code": completed.returncode,
                    "execution_status": row["execution_status"],
                },
            )
            if completed.returncode != 0 and not continue_on_failure:
                rows.append(row)
                break
        else:
            progress(
                output_dir,
                {
                    "phase": "runtime_track_planned",
                    "track_index": index,
                    "input_audio_path": str(audio_path),
                    "run_output_dir": str(run_dir),
                    "gpu_inference_skipped": True,
                    "reason": "execute_false_plan_only",
                },
            )
        rows.append(row)

    run_summary = summarize_runs(rows)
    example_counts = harvest_example_counts(output_dir)
    manifest = {
        "schema": "strict_negative_harvest_batch.v1",
        "input_audio_dir": str(input_audio_dir),
        "output_dir": str(output_dir),
        "model_dir": str(model_dir),
        "profile": profile,
        "candidate_diversity_profile": candidate_diversity_profile,
        "tracks_discovered": len(audio_files),
        "execute": bool(execute),
        "strict_negative_rows_found": example_counts["strict_negative_windows"],
        "strict_positive_rows_found": example_counts["strict_positive_windows"],
        "harvest_example_counts": example_counts,
        **run_summary,
        **SAFETY_FLAGS,
    }
    write_json(output_dir / MANIFEST_FILE, manifest)
    write_jsonl(output_dir / RUNS_FILE, rows)
    write_summary(output_dir / SUMMARY_FILE, manifest)
    progress(output_dir, {"phase": "complete", **manifest})
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a controlled strict negative no-reference harvest batch.")
    parser.add_argument("--input-audio-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--max-tracks", type=int, default=20)
    parser.add_argument("--profile", choices=("safe", "balanced"), default="balanced")
    parser.add_argument("--candidate-diversity-profile", choices=("safe", "balanced", "exploration"), default="balanced")
    parser.add_argument("--enable-conditional-hard-case-routes", type=parse_bool, default=True)
    parser.add_argument("--enable-audio-preview-render", type=parse_bool, default=True)
    parser.add_argument("--require-preview-audio-for-preview-ready", type=parse_bool, default=True)
    parser.add_argument("--continue-on-failure", type=parse_bool, default=True)
    parser.add_argument("--execute", type=parse_bool, default=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_strict_negative_harvest_batch(
        input_audio_dir=args.input_audio_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        max_tracks=args.max_tracks,
        profile=args.profile,
        candidate_diversity_profile=args.candidate_diversity_profile,
        enable_conditional_hard_case_routes=args.enable_conditional_hard_case_routes,
        enable_audio_preview_render=args.enable_audio_preview_render,
        require_preview_audio_for_preview_ready=args.require_preview_audio_for_preview_ready,
        continue_on_failure=args.continue_on_failure,
        execute=args.execute,
    )
    return 0 if manifest["tracks_failed"] == 0 or args.continue_on_failure else 1


if __name__ == "__main__":
    raise SystemExit(main())
