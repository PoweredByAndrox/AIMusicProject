#!/usr/bin/env python3
"""Render full-length candidate-family blends for listen checking."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402
from train_teacher_mask_separator import read_audio, resample_audio, validate_stem_name, write_flac  # noqa: E402
from artifact_name_policy import ARTIFACT_AUDIO_DIR, indexed_alias, indexed_audio_filename, relative_artifact_path  # noqa: E402


START_TIME = time.monotonic()
MANIFEST_NAME = "pack.json"
SELECTED_NAME = "selected.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid jsonl row: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object row")
            yield row


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_dsp_full_length_blend",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "candidate_family_blend_uses_existing_source_audio",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "listen_pack_no_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _duration(path: Path) -> float:
    audio, sample_rate = read_audio(path)
    return round(float(audio.shape[0]) / float(sample_rate), 6)


def _row_allowed(row: dict[str, Any], *, stem: str) -> bool:
    return (
        row.get("trainable") is not False
        and row.get("release_ready") is not True
        and row.get("delivery_allowed") is not True
        and row.get("pseudo_teacher") is not True
        and row.get("source_candidate_is_ground_truth") is not True
        and str(row.get("stem_type") or row.get("stem") or "") == stem
        and (row.get("reference_verified") is True or row.get("human_verified") is True)
    )


def _row_blockers(row: dict[str, Any], *, stem: str) -> list[str]:
    failed: list[str] = []
    if row.get("release_ready") is True:
        failed.append("release_ready_input_row")
    if row.get("delivery_allowed") is True:
        failed.append("delivery_allowed_input_row")
    if row.get("pseudo_teacher") is True:
        failed.append("pseudo_teacher_input_row")
    if row.get("source_candidate_is_ground_truth") is True:
        failed.append("source_candidate_marked_ground_truth")
    if str(row.get("stem_type") or row.get("stem") or "") != stem:
        failed.append("stem_mismatch")
    if not (row.get("reference_verified") is True or row.get("human_verified") is True):
        failed.append("unverified_input_row")
    return failed


def _load_sources(rows_by_family: dict[str, dict[str, Any]], families: Sequence[str]) -> tuple[list[np.ndarray], int, float, float]:
    sources: list[np.ndarray] = []
    sample_rate: int | None = None
    source_duration = math.inf
    target_duration = math.inf
    for family in families:
        source_path = Path(str(rows_by_family[family].get("source_audio_path") or ""))
        if not source_path.exists():
            raise FileNotFoundError(f"source audio missing: {source_path}")
        audio, rate = read_audio(source_path)
        if sample_rate is None:
            sample_rate = rate
        elif rate != sample_rate:
            audio = resample_audio(audio, rate, sample_rate)
        sources.append(audio)
        source_duration = min(source_duration, float(audio.shape[0]) / float(sample_rate))
        target_path = Path(str(rows_by_family[family].get("target_audio_path") or ""))
        if target_path.exists():
            target_duration = min(target_duration, _duration(target_path))
    if sample_rate is None:
        raise ValueError("no source audio loaded")
    min_frames = min(source.shape[0] for source in sources)
    min_channels = min(source.shape[1] for source in sources)
    return [source[:min_frames, :min_channels] for source in sources], sample_rate, round(source_duration, 6), round(target_duration, 6)


def build_candidate_family_full_length_listen_pack(
    *,
    audio_training_examples_jsonl: Path,
    adapter_json: Path,
    output_dir: Path,
    stem: str,
    track_ids: set[str] | None = None,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    stem = validate_stem_name(stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    listen_dir = output_dir / ARTIFACT_AUDIO_DIR
    if listen_dir.exists():
        shutil.rmtree(listen_dir)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    adapter = read_json(adapter_json)
    families = [str(item) for item in adapter.get("families") or [] if str(item)]
    weights_map = adapter.get("weights") if isinstance(adapter.get("weights"), dict) else {}
    if len(families) < 1:
        raise ValueError("adapter families missing")
    weights = [float(weights_map.get(family, 0.0)) for family in families]
    if not math.isclose(sum(weights), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("adapter weights must sum to 1.0")
    progress(progress_path, "start", adapter_json=str(adapter_json), examples=str(audio_training_examples_jsonl), families=families)
    requested_tracks = {str(item) for item in (track_ids or set()) if str(item)}
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    failed_checks: list[str] = []
    for row in iter_jsonl(audio_training_examples_jsonl):
        track_id = str(row.get("track_id") or "")
        family = str(row.get("source_candidate_family") or "")
        if requested_tracks and track_id not in requested_tracks:
            continue
        if family not in families:
            continue
        failed_checks.extend(_row_blockers(row, stem=stem))
        if _row_allowed(row, stem=stem):
            grouped.setdefault(track_id, {})[family] = row
    selected: list[dict[str, Any]] = []
    for track_id, rows_by_family in sorted(grouped.items()):
        missing = [family for family in families if family not in rows_by_family]
        if missing:
            failed_checks.append("required_family_missing")
            continue
        sources, sample_rate, source_duration, target_duration = _load_sources(rows_by_family, families)
        blend = np.zeros_like(sources[0], dtype=np.float64)
        for source, weight in zip(sources, weights):
            blend += source.astype(np.float64, copy=False) * weight
        artifact_alias = indexed_alias("v", len(selected) + 1)
        artifact_filename = indexed_audio_filename("v", len(selected) + 1)
        out_path = listen_dir / artifact_filename
        write_flac(out_path, blend, sample_rate)
        selected.append(
            {
                "schema": "candidate_family_full_length_listen_audio.v1",
                "track_id": track_id,
                "source_track_id": track_id,
                "source_config_id": str(adapter.get("config_id") or ""),
                "artifact_alias": artifact_alias,
                "artifact_filename": artifact_filename,
                "artifact_relative_path": relative_artifact_path(out_path, output_dir),
                "stem_type": stem,
                "families": families,
                "weights": {family: round(weight, 10) for family, weight in zip(families, weights)},
                "duration_sec": source_duration,
                "source_duration_sec": source_duration,
                "target_duration_sec": target_duration if target_duration != math.inf else None,
                "listen_audio_path": str(out_path),
                "listen_audio_sha256": sha256_file(out_path),
                "delivery_status": "accepted",
                "delivery_allowed": True,
                "presentation_status": "accepted_for_listen_pack",
                "release_status": "listen_preview_only",
                "release_ready": False,
                "can_publish_as_result": False,
            }
        )
    if requested_tracks and set(grouped) != requested_tracks:
        failed_checks.append("requested_track_missing")
    failed_checks = sorted(set(failed_checks))
    accepted = bool(selected) and not failed_checks
    manifest = {
        "schema": "candidate_family_full_length_listen_pack_manifest.v1",
        "delivery_status": "accepted" if accepted else "blocked",
        "delivery_allowed": accepted,
        "presentation_status": "accepted_for_listen_pack" if accepted else "blocked_hard_case",
        "release_status": "listen_preview_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "listenable": accepted,
        "deliverable": False,
        "failed_checks": failed_checks,
        "audio_training_examples_jsonl": str(audio_training_examples_jsonl),
        "adapter_json": str(adapter_json),
        "output_dir": str(output_dir),
        "selected_jsonl": str(output_dir / SELECTED_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "stem_type": stem,
        "listen_audio_count": len(selected) if accepted else 0,
        "listen_audio_files": selected if accepted else [],
        "artifact_name_policy": {
            "schema": "short_windows_safe_artifact_names.v1",
            "audio_dir": ARTIFACT_AUDIO_DIR,
            "filename_strategy": "indexed_alias",
            "raw_track_or_config_ids_in_path": False,
            "provenance_in_manifest": True,
        },
        "thresholds_relaxed": False,
        "runtime_gates_changed": False,
        "fallback_used": False,
        "model_registry_edited": False,
        "command_args": command_args or [],
        "contract": {
            "full_length_source_audio_used": True,
            "target_audio_not_used_for_length": True,
            "release_ready_forced_false": True,
            "short_windows_safe_artifact_names": True,
            "raw_ids_kept_in_manifest_only": True,
        },
    }
    write_jsonl(output_dir / SELECTED_NAME, selected if accepted else [])
    write_json(output_dir / MANIFEST_NAME, manifest)
    write_json(output_dir / SUMMARY_NAME, {"listen_audio_count": manifest["listen_audio_count"], "failed_checks": failed_checks})
    progress(progress_path, "finish", delivery_status=manifest["delivery_status"], listen_audio_count=manifest["listen_audio_count"], failed_checks=failed_checks)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render full-length candidate-family blend listen pack.")
    parser.add_argument("--audio-training-examples-jsonl", type=Path, required=True)
    parser.add_argument("--adapter-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="vocal")
    parser.add_argument("--track-id", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(tool_name="build_candidate_family_full_length_listen_pack.py", entrypoint_path="/app/tools/build_candidate_family_full_length_listen_pack.py")
    args = parse_args(argv)
    manifest = build_candidate_family_full_length_listen_pack(
        audio_training_examples_jsonl=args.audio_training_examples_jsonl,
        adapter_json=args.adapter_json,
        output_dir=args.output_dir,
        stem=str(args.stem),
        track_ids={str(item) for item in args.track_id},
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(json.dumps({"manifest_path": str(args.output_dir / MANIFEST_NAME), "delivery_status": manifest["delivery_status"], "listen_audio_count": manifest["listen_audio_count"]}, sort_keys=True))
    return 0 if manifest["delivery_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
