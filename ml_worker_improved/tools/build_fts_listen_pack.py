#!/usr/bin/env python3
"""Build a short listen pack for a gated full-track switch minus candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import soundfile as sf

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.artifact_name_policy import validate_artifact_relative_path, validate_artifact_segment  # noqa: E402
from tools.docker_runtime import require_docker_runtime  # noqa: E402

MANIFEST_NAME = "pack.json"
SELECTED_NAME = "selected.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
ARTIFACT_AUDIO_DIR = "audio"
START_TIME = time.monotonic()
BLOCKED_TRUE_FLAGS = (
    "release_ready",
    "can_publish_as_result",
    "training_started",
    "separator_training_started",
    "models_changed",
    "model_registry_edited",
    "thresholds_relaxed",
    "cleanup_routes_added",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_packaging",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "fts_listen_pack_copies_judged_audio_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "fts_listen_pack_no_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def collect_true_flag_failures(prefix: str, payload: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    for key in BLOCKED_TRUE_FLAGS:
        if payload.get(key) is True:
            failed.append(f"{prefix}_{key}_true")
    safety = payload.get("safety")
    if isinstance(safety, dict):
        for key in BLOCKED_TRUE_FLAGS:
            if safety.get(key) is True:
                failed.append(f"{prefix}_safety_{key}_true")
    contract = payload.get("contract")
    if isinstance(contract, dict):
        for key in BLOCKED_TRUE_FLAGS:
            if contract.get(key) is True:
                failed.append(f"{prefix}_contract_{key}_true")
    return failed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_metadata(path: Path) -> dict[str, Any]:
    info = sf.info(str(path))
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "duration_sec": round(float(info.frames) / float(info.samplerate), 6) if info.samplerate else 0.0,
    }


def rows_accept_minus(rows: Sequence[dict[str, Any]]) -> bool:
    if not rows:
        return False
    for row in rows:
        decision = str(row.get("minus_decision") or row.get("decision") or "")
        if decision != "accept_candidate":
            return False
    return True


def safety_flags() -> dict[str, bool]:
    return {
        "reference_required": False,
        "human_csv_required": False,
        "training_started": False,
        "separator_training_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "thresholds_relaxed": False,
        "cleanup_routes_added": False,
        "release_ready": False,
    }


def summary_text(manifest: dict[str, Any], selected: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# full-track switch listen pack",
        "",
        f"- delivery_status: `{manifest['delivery_status']}`",
        f"- selected_count: `{manifest['selected_count']}`",
        f"- release_ready: `{str(manifest['release_ready']).lower()}`",
        f"- can_publish_as_result: `{str(manifest['can_publish_as_result']).lower()}`",
        "",
        "## Audio",
        "",
    ]
    for row in selected:
        lines.append(f"- `{row['artifact_relative_path']}`")
    if manifest["failed_checks"]:
        lines.extend(["", "## Failed Checks", ""])
        lines.extend(f"- `{check}`" for check in manifest["failed_checks"])
    return "\n".join(lines) + "\n"


def build_fts_listen_pack(
    *,
    switch_manifest_json: Path,
    result_judge_jsonl: Path,
    windowed_judge_jsonl: Path,
    targeted_cmp_json: Path,
    output_dir: Path,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / ARTIFACT_AUDIO_DIR
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", switch_manifest_json=str(switch_manifest_json))

    switch_manifest = read_json(switch_manifest_json)
    targeted_cmp = read_json(targeted_cmp_json)
    failed: list[str] = []
    failed.extend(collect_true_flag_failures("switch", switch_manifest))
    failed.extend(collect_true_flag_failures("targeted_cmp", targeted_cmp))
    if switch_manifest.get("manifest_type") != "no_reference_full_track_segment_switch_candidate":
        failed.append("switch_manifest_type_unexpected")
    if switch_manifest.get("render_status") != "rendered_full_track_candidate":
        failed.append("switch_not_rendered_full_track_candidate")
    if switch_manifest.get("duration_preserved") is not True:
        failed.append("switch_duration_not_preserved")
    if switch_manifest.get("blocked_window_count") not in (0, None):
        failed.append("switch_blocked_windows_present")
    source_audio = Path(str(switch_manifest.get("rendered_minus_candidate_path") or ""))
    if not source_audio.exists() or not source_audio.is_file():
        failed.append(f"switch_minus_audio_missing:{source_audio}")
    if not result_judge_jsonl.exists():
        failed.append("result_judge_missing")
        result_rows: list[dict[str, Any]] = []
    else:
        result_rows = iter_jsonl(result_judge_jsonl)
    if not windowed_judge_jsonl.exists():
        failed.append("windowed_judge_missing")
        windowed_rows: list[dict[str, Any]] = []
    else:
        windowed_rows = iter_jsonl(windowed_judge_jsonl)
    if result_rows and not rows_accept_minus(result_rows):
        failed.append("result_judge_not_accept_minus")
    if windowed_rows and not rows_accept_minus(windowed_rows):
        failed.append("windowed_judge_not_accept_minus")
    if targeted_cmp.get("schema") != "targeted_minus_window_cmp.v1":
        failed.append("targeted_cmp_schema_unexpected")
    if targeted_cmp.get("comparison_status") != "completed":
        failed.append("targeted_cmp_not_completed")
    if int(targeted_cmp.get("regressed_count") or 0) > 0:
        failed.append("targeted_cmp_regressed_windows_present")

    selected: list[dict[str, Any]] = []
    failed = stable_unique(failed)
    accepted = not failed
    if accepted:
        validate_artifact_segment("min")
        filename = "m001.wav"
        validate_artifact_segment(filename)
        target = output_dir / ARTIFACT_AUDIO_DIR / "min" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_audio, target)
        metadata = audio_metadata(target)
        selected.append(
            {
                "schema": "full_track_switch_listen_audio.v1",
                "selection_rank": 1,
                "route_intent": "minus_full_track_switch_preview",
                "stem_type": "minus",
                "artifact_alias": "m001",
                "artifact_filename": filename,
                "artifact_relative_path": "audio/min/m001.wav",
                "listen_audio_path": str(target),
                "listen_audio_sha256": sha256_file(target),
                "source_audio_path": str(source_audio),
                "source_switch_manifest_json": str(switch_manifest_json),
                "result_judge_jsonl": str(result_judge_jsonl),
                "windowed_judge_jsonl": str(windowed_judge_jsonl),
                "targeted_cmp_json": str(targeted_cmp_json),
                "track_alias": str(switch_manifest.get("track_id") or "unknown"),
                "quality_status": "listen_preview_only_targeted_cmp_pass",
                "semantic_quality_verified": False,
                "delivery_status": "accepted",
                "delivery_allowed": True,
                "presentation_status": "accepted_for_listen_pack",
                "release_status": "listen_preview_only",
                "release_ready": False,
                "can_publish_as_result": False,
                **metadata,
            }
        )
    elif audio_dir.exists():
        shutil.rmtree(audio_dir)

    manifest = {
        "schema": "full_track_switch_listen_pack.v1",
        "delivery_status": "accepted" if accepted else "blocked",
        "delivery_allowed": accepted,
        "presentation_status": "accepted_for_listen_pack" if accepted else "blocked_from_listen_pack",
        "release_status": "listen_preview_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "deliverable": False,
        "listenable": accepted,
        "selected_count": len(selected),
        "listen_audio_files": selected,
        "switch_manifest_json": str(switch_manifest_json),
        "result_judge_jsonl": str(result_judge_jsonl),
        "windowed_judge_jsonl": str(windowed_judge_jsonl),
        "targeted_cmp_json": str(targeted_cmp_json),
        "failed_checks": failed,
        "selected_jsonl": str(output_dir / SELECTED_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "artifact_name_policy": {
            "schema": "short_windows_safe_artifact_names.v1",
            "audio_dir": ARTIFACT_AUDIO_DIR,
            "group_dir_strategy": "min",
            "filename_strategy": "indexed_candidate_alias",
            "raw_track_or_candidate_ids_in_path": False,
            "provenance_in_manifest": True,
        },
        "contract": {
            "audio_copied_without_modification": True,
            "selection_only": True,
            "post_render_result_judge_required": True,
            "post_render_windowed_judge_required": True,
            "targeted_cmp_no_regression_required": True,
            "training_started": False,
            "separator_training_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "thresholds_relaxed": False,
            "release_ready_forced_false": True,
            "dirty_stems_never_ground_truth": True,
        },
        "safety": safety_flags(),
        "not_claimed": ["release_ready", "publishable_result", "semantic_ground_truth"],
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / SELECTED_NAME, selected)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(summary_text(manifest, selected), encoding="utf-8")
    progress(progress_path, "finish", delivery_status=manifest["delivery_status"], selected_count=len(selected), failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a gated listen pack for full-track switch minus preview.")
    parser.add_argument("--switch-manifest-json", type=Path, required=True)
    parser.add_argument("--result-judge-jsonl", type=Path, required=True)
    parser.add_argument("--windowed-judge-jsonl", type=Path, required=True)
    parser.add_argument("--targeted-cmp-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(tool_name="build_fts_listen_pack.py", entrypoint_path="/app/tools/build_fts_listen_pack.py")
    args = parse_args(argv)
    manifest = build_fts_listen_pack(
        switch_manifest_json=args.switch_manifest_json,
        result_judge_jsonl=args.result_judge_jsonl,
        windowed_judge_jsonl=args.windowed_judge_jsonl,
        targeted_cmp_json=args.targeted_cmp_json,
        output_dir=args.output_dir,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "delivery_status": manifest["delivery_status"],
                "selected_count": manifest["selected_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["delivery_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
