#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.extract_audio_error_maps import ERROR_MAP_NAMES, extract_error_maps_for_window, write_error_maps


SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "separator_training_started": False,
    "model_downloaded": False,
    "model_registry_edited": False,
    "models_changed": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "melody_transcription_ran": False,
    "recommendation_index_updated": False,
    "release_ready": False,
}
ACCEPT_DECISIONS = {"accept_candidate", "pair_compatible", "accepted_for_listen_pack", "preview_ready"}
BLOCK_DECISIONS = {"blocked_hard_case", "blocked_pair_incompatible", "failed_safety_contract"}
JUDGE_FILES_BY_STEM = {
    "vocal": ("vocal_windowed_judge.jsonl", "vocal_judge.jsonl", "no_reference_vocal_windowed_judge.jsonl"),
    "minus": ("minus_windowed_judge.jsonl", "minus_judge.jsonl", "no_reference_minus_windowed_judge.jsonl"),
}
RUN_MARKER_FILES = (
    "candidate_pool.jsonl",
    "vocal_windowed_judge.jsonl",
    "minus_windowed_judge.jsonl",
    "vocal_judge.jsonl",
    "minus_judge.jsonl",
    "pair_compatibility.jsonl",
)
OWNED_OUTPUT_FILES = (
    "audio_error_grounding_manifest.json",
    "audio_error_window_dataset.jsonl",
    "audio_error_maps_index.jsonl",
    "audio_error_dataset_summary.md",
    "audio_error_grounding_progress.jsonl",
)
DATASET_ROW_FIELDS = (
    "dataset_id",
    "track_id",
    "run_id",
    "window_id",
    "window_start_sec",
    "window_end_sec",
    "mix_path",
    "vocal_candidate_path",
    "minus_candidate_path",
    "candidate_id",
    "route_id",
    "stem_type",
    "blockers",
    "judge_decision",
    "pair_compatibility_status",
    "residual_energy",
    "mixture_consistency_error",
    "phase_collapse_score",
    "stereo_width_score",
    "vocal_bleed_proxy_score",
    "instrumental_bleed_proxy_score",
    "missing_detail_score",
    "transient_damage_score",
    "artifact_noise_score",
    "error_map_paths",
    "train_split",
    "safe_for_training",
    "unsafe_reason",
)


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def split_for_track(track_id: str) -> str:
    bucket = int(hashlib.sha256(str(track_id).encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket < 7:
        return "train"
    if bucket < 8:
        return "dev"
    return "test"


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    if isinstance(value, set):
        return sorted(str(item) for item in value if str(item))
    text = str(value)
    return [text] if text else []


def stable_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def is_accept_decision(value: Any) -> bool:
    return str(value or "").strip() in ACCEPT_DECISIONS


def is_blocked_decision(value: Any) -> bool:
    decision = str(value or "").strip()
    return decision in BLOCK_DECISIONS or decision.startswith("blocked")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def find_run_dirs(run_roots: Sequence[Path]) -> list[Path]:
    discovered: set[Path] = set()
    for root in run_roots:
        if not root.exists():
            continue
        if root.is_file():
            root = root.parent
        if any((root / marker).exists() for marker in RUN_MARKER_FILES):
            discovered.add(root)
        for marker in RUN_MARKER_FILES:
            for path in root.rglob(marker):
                discovered.add(path.parent)
    return sorted(discovered, key=lambda path: str(path))


def load_jsonl_first_existing(run_dir: Path, names: Sequence[str]) -> tuple[list[dict[str, Any]], str]:
    for name in names:
        path = run_dir / name
        if path.exists():
            return list(iter_jsonl(path)), name
    return [], ""


def load_candidate_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows, _ = load_jsonl_first_existing(run_dir, ("candidate_pool.jsonl", "no_reference_candidate_pool.jsonl"))
    return rows


def pair_status_by_candidate(rows: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    mapping: dict[tuple[str, str], str] = {}
    for row in rows:
        track_id = str(row.get("track_id") or "")
        candidate_ids = stable_unique(
            [
                str(row.get("candidate_id") or ""),
                str(row.get("selected_candidate_id") or ""),
                str(row.get("selected_vocal_candidate_id") or ""),
                str(row.get("selected_minus_candidate_id") or ""),
            ]
        )
        status = str(row.get("pair_compatibility_status") or row.get("pair_decision") or row.get("compatibility_status") or row.get("decision") or "")
        for candidate_id in candidate_ids:
            if track_id and candidate_id:
                mapping[(track_id, candidate_id)] = status
    return mapping


def row_path(row: dict[str, Any], candidate: dict[str, Any], names: Sequence[str]) -> str:
    for name in names:
        value = row.get(name)
        if value:
            return str(value)
    for name in names:
        value = candidate.get(name)
        if value:
            return str(value)
    return ""


def candidate_paths(row: dict[str, Any], candidate: dict[str, Any]) -> dict[str, str]:
    return {
        "mix_path": row_path(row, candidate, ("mix_path", "input_audio_path", "input_audio", "source_mix_path")),
        "vocal_candidate_path": row_path(
            row,
            candidate,
            ("vocal_candidate_path", "vocal_path", "selected_vocal_path", "normalized_vocal_path"),
        ),
        "minus_candidate_path": row_path(
            row,
            candidate,
            (
                "minus_candidate_path",
                "minus_path",
                "instrumental_path",
                "selected_minus_path",
                "normalized_minus_path",
                "normalized_instrumental_path",
            ),
        ),
    }


def candidate_id_for(row: dict[str, Any], candidate: dict[str, Any]) -> str:
    for key in ("candidate_id", "selected_candidate_id", "source_candidate_id", "input_candidate_id"):
        if row.get(key):
            return str(row[key])
        if candidate.get(key):
            return str(candidate[key])
    return ""


def window_metrics_for(row: dict[str, Any], *, window_sec: float) -> list[dict[str, Any]]:
    windows = row.get("window_metrics")
    if isinstance(windows, list) and windows:
        return [window for window in windows if isinstance(window, dict)]
    return []


def blockers_for(stem_type: str, row: dict[str, Any], window: dict[str, Any]) -> list[str]:
    generic_keys = ["blockers"]
    stem_keys = []
    if stem_type == "vocal":
        stem_keys.append("vocal_blockers")
    elif stem_type == "minus":
        stem_keys.append("minus_blockers")
    window_keys = generic_keys + stem_keys
    return stable_unique(value for key in window_keys for value in as_list(window.get(key)))


def judge_decision_for(stem_type: str, row: dict[str, Any], window: dict[str, Any]) -> str:
    keys = [f"{stem_type}_decision", "decision", "judge_decision"]
    for key in keys:
        value = window.get(key)
        if value:
            return str(value)
    return ""


def check_audio_paths(paths: dict[str, str]) -> list[str]:
    reasons: list[str] = []
    for field in ("mix_path", "vocal_candidate_path", "minus_candidate_path"):
        value = paths.get(field) or ""
        if not value:
            reasons.append(f"missing_{field}")
            continue
        if not Path(value).exists():
            reasons.append(f"missing_audio_file:{field}")
    return reasons


def check_required_provenance(
    *,
    track_id: str,
    run_id: str,
    candidate_id: str,
    route_id: str,
    judge_decision: str,
    pair_status: str,
    window_has_metrics: bool,
) -> list[str]:
    reasons: list[str] = []
    if not track_id:
        reasons.append("missing_track_id")
    if not run_id:
        reasons.append("missing_run_id")
    if not candidate_id:
        reasons.append("missing_candidate_id")
    if not route_id:
        reasons.append("missing_route_id")
    if not judge_decision:
        reasons.append("missing_window_judge_decision")
    if not pair_status:
        reasons.append("missing_pair_compatibility_status")
    if not window_has_metrics:
        reasons.append("missing_window_metrics")
    return reasons


def map_path_payload(npz_path: Path) -> dict[str, str]:
    payload = {"npz": str(npz_path)}
    for name in ERROR_MAP_NAMES:
        payload[name] = f"{npz_path}#{name}"
    return payload


def progress(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row = {**payload, "release_ready": False}
    with (output_dir / "audio_error_grounding_progress.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)


def prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OWNED_OUTPUT_FILES:
        path = output_dir / name
        if path.exists():
            path.unlink()
    maps_dir = output_dir / "error_maps"
    if maps_dir.exists():
        for path in maps_dir.glob("*.npz"):
            path.unlink()


def build_window_row(
    *,
    dataset_id: str,
    run_dir: Path,
    output_dir: Path,
    stem_type: str,
    row: dict[str, Any],
    candidate: dict[str, Any],
    window: dict[str, Any],
    pair_status_map: dict[tuple[str, str], str],
    map_options: dict[str, int],
    map_cache: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    track_id = str(row.get("track_id") or candidate.get("track_id") or "")
    run_id = str(row.get("run_id") or candidate.get("run_id") or "")
    candidate_id = candidate_id_for(row, candidate)
    route_id = str(
        row.get("route_id")
        or row.get(f"{stem_type}_route_id")
        or candidate.get("route_id")
        or candidate.get(f"{stem_type}_route_id")
        or candidate.get("model_id")
        or ""
    )
    start_sec = as_float(window.get("start_sec"), as_float(row.get("window_start_sec"), 0.0))
    duration_sec = as_float(window.get("duration_sec"), as_float(row.get("window_duration_sec"), 0.0))
    end_sec = start_sec + max(0.0, duration_sec)
    blockers = blockers_for(stem_type, row, window)
    judge_decision = judge_decision_for(stem_type, row, window)
    pair_status = pair_status_map.get((track_id, candidate_id), str(row.get("pair_compatibility_status") or ""))
    paths = candidate_paths(row, candidate)
    window_id = "__".join(
        [
            track_id or "missing_track_id",
            candidate_id or "missing_candidate_id",
            stem_type,
            f"{start_sec:.3f}-{end_sec:.3f}",
            stable_hash({"run": run_id, "candidate": candidate_id, "stem": stem_type, "start": start_sec, "end": end_sec})[
                :10
            ],
        ]
    )
    unsafe_reasons = check_audio_paths(paths)
    unsafe_reasons.extend(
        check_required_provenance(
            track_id=track_id,
            run_id=run_id,
            candidate_id=candidate_id,
            route_id=route_id,
            judge_decision=judge_decision,
            pair_status=pair_status,
            window_has_metrics=bool(window),
        )
    )
    if end_sec <= start_sec:
        unsafe_reasons.append("invalid_window_boundaries")
    metrics = {
        "residual_energy": 0.0,
        "mixture_consistency_error": 0.0,
        "phase_collapse_score": as_float(window.get("phase_collapse_score"), 0.0),
        "stereo_width_score": as_float(window.get("stereo_width_score"), 0.0),
        "vocal_bleed_proxy_score": as_float(window.get("vocal_bleed_proxy_score"), 0.0),
        "instrumental_bleed_proxy_score": as_float(window.get("instrumental_bleed_proxy_score"), 0.0),
        "missing_detail_score": as_float(window.get("missing_detail_score"), 0.0),
        "transient_damage_score": as_float(window.get("transient_damage_score"), 0.0),
        "artifact_noise_score": as_float(window.get("artifact_noise_score"), 0.0),
    }
    error_map_paths: dict[str, str] = {}
    map_index_row = None
    if not unsafe_reasons:
        try:
            cache_key = stable_hash(
                {
                    "paths": paths,
                    "window_start_sec": start_sec,
                    "window_end_sec": end_sec,
                    "map_options": map_options,
                }
            )
            cached = map_cache.get(cache_key)
            if cached is None:
                extraction = extract_error_maps_for_window(
                    mix_path=Path(paths["mix_path"]),
                    vocal_path=Path(paths["vocal_candidate_path"]),
                    minus_path=Path(paths["minus_candidate_path"]),
                    window_start_sec=start_sec,
                    window_end_sec=end_sec,
                    n_fft=map_options["n_fft"],
                    hop_length=map_options["hop_length"],
                    map_max_bins=map_options["map_max_bins"],
                    map_max_frames=map_options["map_max_frames"],
                    max_waveform_samples=map_options["max_waveform_samples"],
                )
                npz_path = write_error_maps(
                    output_dir=output_dir / "error_maps",
                    dataset_id=dataset_id,
                    row_id=cache_key[:16],
                    maps=extraction["maps"],
                    metadata={
                        **extraction["metadata"],
                        "track_id": track_id,
                        "run_id": run_id,
                        "candidate_id": candidate_id,
                        "window_start_sec": start_sec,
                        "window_end_sec": end_sec,
                    },
                )
                cached = {"metrics": extraction["metrics"], "npz_path": str(npz_path)}
                map_cache[cache_key] = cached
            metrics.update(cached["metrics"])
            npz_path = Path(str(cached["npz_path"]))
            error_map_paths = map_path_payload(npz_path)
            map_index_row = {
                "dataset_id": dataset_id,
                "track_id": track_id,
                "run_id": run_id,
                "candidate_id": candidate_id,
                "window_id": window_id,
                "stem_type": stem_type,
                "npz_path": str(npz_path),
                "map_names": list(ERROR_MAP_NAMES),
                "release_ready": False,
            }
        except Exception as exc:  # pragma: no cover - corrupt external artifact path.
            unsafe_reasons.append(f"error_map_extraction_failed:{type(exc).__name__}:{exc}")
    safe_for_training = not unsafe_reasons
    payload = {
        "dataset_id": dataset_id,
        "track_id": track_id,
        "run_id": run_id,
        "window_id": window_id,
        "window_start_sec": round(start_sec, 6),
        "window_end_sec": round(end_sec, 6),
        "mix_path": paths["mix_path"],
        "vocal_candidate_path": paths["vocal_candidate_path"],
        "minus_candidate_path": paths["minus_candidate_path"],
        "candidate_id": candidate_id,
        "route_id": route_id,
        "stem_type": stem_type,
        "blockers": blockers,
        "judge_decision": judge_decision,
        "pair_compatibility_status": pair_status,
        **metrics,
        "error_map_paths": error_map_paths,
        "train_split": split_for_track(track_id),
        "safe_for_training": safe_for_training,
        "unsafe_reason": ";".join(stable_unique(unsafe_reasons)),
        "source_run_dir": str(run_dir),
        "source_candidate_family": str(
            row.get("source_candidate_family") or candidate.get("source_candidate_family") or row.get("route_family") or ""
        ),
        "example_label": "positive" if is_accept_decision(judge_decision) and not blockers else "negative_hard",
        **SAFETY_FLAGS,
    }
    ordered = {field: payload[field] for field in DATASET_ROW_FIELDS}
    for key in sorted(set(payload) - set(ordered)):
        ordered[key] = payload[key]
    return ordered, map_index_row


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": round(min(values), 10),
        "max": round(max(values), 10),
        "mean": round(statistics.fmean(values), 10),
    }


def summarize(rows: list[dict[str, Any]], maps_index: list[dict[str, Any]], run_dirs: list[Path], dataset_id: str) -> dict[str, Any]:
    blocked_rows = [
        row
        for row in rows
        if row["blockers"] or is_blocked_decision(row["judge_decision"]) or row["example_label"] == "negative_hard"
    ]
    accepted_rows = [
        row for row in rows if is_accept_decision(row["judge_decision"]) and not row["blockers"] and row["safe_for_training"]
    ]
    phase_rows = [
        row
        for row in rows
        if any("phase" in blocker or "stereo" in blocker for blocker in row["blockers"])
        or "phase_collapse" in " ".join(row["blockers"])
    ]
    vocal_bleed_rows = [
        row
        for row in rows
        if any("vocal_bleed" in blocker or "vocal_leakage" in blocker for blocker in row["blockers"])
    ]
    safe_rows = [row for row in rows if row["safe_for_training"]]
    unique_map_paths = {str(row.get("npz_path") or "") for row in maps_index if row.get("npz_path")}
    manifest = {
        "dataset_id": dataset_id,
        "run_roots_expanded": [str(path) for path in run_dirs],
        "tracks_included": len({row["track_id"] for row in rows}),
        "candidates_included": len({(row["track_id"], row["candidate_id"]) for row in rows}),
        "windows_total": len(rows),
        "accepted_windows": len(accepted_rows),
        "blocked_windows": len(blocked_rows),
        "phase_collapse_windows": len(phase_rows),
        "vocal_bleed_windows": len(vocal_bleed_rows),
        "mixture_consistency_error_stats": stats([float(row["mixture_consistency_error"]) for row in safe_rows]),
        "error_maps_written": len(unique_map_paths),
        "error_map_index_rows": len(maps_index),
        "safe_for_training_count": len(safe_rows),
        "unsafe_count": len(rows) - len(safe_rows),
        "dataset_row_fields": list(DATASET_ROW_FIELDS),
        "error_map_names": list(ERROR_MAP_NAMES),
        "ground_truth_stems_used": False,
        "original_mix_used_as_reference": False,
        "original_mix_used_as_mixture_constraint": True,
        **SAFETY_FLAGS,
    }
    return manifest


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Audio Error Grounding Dataset",
        "",
        f"- dataset_id: `{manifest['dataset_id']}`",
        f"- tracks_included: {manifest['tracks_included']}",
        f"- candidates_included: {manifest['candidates_included']}",
        f"- windows_total: {manifest['windows_total']}",
        f"- accepted_windows: {manifest['accepted_windows']}",
        f"- blocked_windows: {manifest['blocked_windows']}",
        f"- phase_collapse_windows: {manifest['phase_collapse_windows']}",
        f"- vocal_bleed_windows: {manifest['vocal_bleed_windows']}",
        f"- mixture_consistency_error_stats: `{json.dumps(manifest['mixture_consistency_error_stats'], sort_keys=True)}`",
        f"- error_maps_written: {manifest['error_maps_written']}",
        f"- safe_for_training_count: {manifest['safe_for_training_count']}",
        f"- unsafe_count: {manifest['unsafe_count']}",
        "- ground_truth_stems_used: false",
        "- original_mix_used_as_mixture_constraint: true",
        "- training_started: false",
        "- release_ready: false",
        "- melody_transcription_ran: false",
        "- recommendation_index_updated: false",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_audio_error_grounding_dataset(
    *,
    run_roots: Sequence[Path],
    output_dir: Path,
    window_sec: float = 12.0,
    hop_sec: float = 12.0,
    n_fft: int = 1024,
    hop_length: int = 256,
    map_max_bins: int = 128,
    map_max_frames: int = 256,
    max_waveform_samples: int = 4096,
) -> dict[str, Any]:
    prepare_output_dir(output_dir)
    progress(output_dir, {"phase": "start", "run_roots": [str(path) for path in run_roots], "gpu_inference_skipped": True, "reason": "cpu_dsp_error_map_extraction_only"})
    run_dirs = find_run_dirs(run_roots)
    dataset_id = "audio_error_grounding_" + stable_hash(
        {"run_dirs": [str(path) for path in run_dirs], "window_sec": window_sec, "hop_sec": hop_sec}
    )[:12]
    rows: list[dict[str, Any]] = []
    maps_index: list[dict[str, Any]] = []
    map_options = {
        "n_fft": int(n_fft),
        "hop_length": int(hop_length),
        "map_max_bins": int(map_max_bins),
        "map_max_frames": int(map_max_frames),
        "max_waveform_samples": int(max_waveform_samples),
    }
    map_cache: dict[str, dict[str, Any]] = {}
    for run_dir in run_dirs:
        progress(output_dir, {"phase": "run_dir_start", "run_dir": str(run_dir)})
        candidate_rows = load_candidate_rows(run_dir)
        candidate_map = {
            (str(row.get("track_id") or ""), str(row.get("candidate_id") or "")): row
            for row in candidate_rows
            if row.get("candidate_id")
        }
        pair_rows, pair_file = load_jsonl_first_existing(run_dir, ("pair_compatibility.jsonl", "pair_compatibility_rows.jsonl"))
        pair_map = pair_status_by_candidate(pair_rows)
        preview_contract = read_json(run_dir / "preview_render_contract.json")
        if preview_contract.get("release_ready") is True:
            progress(output_dir, {"phase": "source_preview_release_ready_ignored", "run_dir": str(run_dir)})
        for stem_type, filenames in JUDGE_FILES_BY_STEM.items():
            judge_rows, judge_file = load_jsonl_first_existing(run_dir, filenames)
            for judge_row in judge_rows:
                track_id = str(judge_row.get("track_id") or "")
                candidate_id = candidate_id_for(judge_row, {})
                candidate = candidate_map.get((track_id, candidate_id), {})
                for window in window_metrics_for(judge_row, window_sec=window_sec):
                    row, map_index_row = build_window_row(
                        dataset_id=dataset_id,
                        run_dir=run_dir,
                        output_dir=output_dir,
                        stem_type=stem_type,
                        row=judge_row,
                        candidate=candidate,
                        window=window,
                        pair_status_map=pair_map,
                        map_options=map_options,
                        map_cache=map_cache,
                    )
                    row["source_judge_file"] = judge_file
                    row["source_pair_file"] = pair_file
                    rows.append(row)
                    if map_index_row is not None:
                        maps_index.append(map_index_row)
                    if len(rows) == 1 or len(rows) % 25 == 0:
                        progress(
                            output_dir,
                            {
                                "phase": "window_done",
                                "run_dir": str(run_dir),
                                "stem_type": stem_type,
                                "candidate_id": row["candidate_id"],
                                "window_id": row["window_id"],
                                "dataset_rows_so_far": len(rows),
                                "error_maps_so_far": len(maps_index),
                                "safe_for_training": row["safe_for_training"],
                            },
                        )
            progress(
                output_dir,
                {
                    "phase": "judge_file_done",
                    "run_dir": str(run_dir),
                    "stem_type": stem_type,
                    "judge_file": judge_file,
                    "judge_rows": len(judge_rows),
                    "dataset_rows_so_far": len(rows),
                    "error_maps_so_far": len(maps_index),
                },
            )
    rows.sort(key=lambda row: (row["track_id"], row["candidate_id"], row["stem_type"], row["window_start_sec"], row["window_id"]))
    maps_index.sort(key=lambda row: (row["track_id"], row["candidate_id"], row["stem_type"], row["window_id"]))
    manifest = summarize(rows, maps_index, run_dirs, dataset_id)
    write_json(output_dir / "audio_error_grounding_manifest.json", manifest)
    write_jsonl(output_dir / "audio_error_window_dataset.jsonl", rows)
    write_jsonl(output_dir / "audio_error_maps_index.jsonl", maps_index)
    write_summary(output_dir / "audio_error_dataset_summary.md", manifest)
    progress(output_dir, {"phase": "complete", **manifest})
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a no-reference audio error grounding dataset from runtime artifacts.")
    parser.add_argument("--run-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-sec", type=float, default=12.0)
    parser.add_argument("--hop-sec", type=float, default=12.0)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--map-max-bins", type=int, default=128)
    parser.add_argument("--map-max-frames", type=int, default=256)
    parser.add_argument("--max-waveform-samples", type=int, default=4096)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_audio_error_grounding_dataset(
        run_roots=args.run_roots,
        output_dir=args.output_dir,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        map_max_bins=args.map_max_bins,
        map_max_frames=args.map_max_frames,
        max_waveform_samples=args.max_waveform_samples,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
