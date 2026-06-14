from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "diagnostic_audio_exported": False,
    "audio_copied": False,
    "audio_exported": False,
    "audio_read": False,
}

DEFAULT_MAX_PAIRS_PER_TRACK = 4
DEFAULT_MAX_TOTAL_PAIRS = 12

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3")
SKIP_PATH_PARTS = ("added_layer", "delta_raw", "raw_ref", "listen_clip")
SKIP_DIR_PREFIXES = ("control_plane_", "no_reference_", "_archive")
SKIP_DIR_NAMES = {"clips", "__pycache__", ".pytest_cache"}
MANIFEST_FILENAMES = {
    "adaptive_manifest.json",
    "adaptive_vocal_manifest.json",
    "candidate_map.json",
    "registry_candidate_execution_manifest.json",
    "selected_delivery_gate_manifest.json",
    "stem_manifest.json",
    "validation_summary.json",
}
PREVIEW_CANDIDATE_JSONL_FILENAMES = {
    "preview_vs_base_candidates.jsonl",
    "no_reference_segment_blend_preview_judge_input.jsonl",
}
PREVIEW_WINDOW_JUDGE_GLOB = "*segment_blend_preview_window_judge*.jsonl"
SKIP_JSON_KEYS = {
    "segments",
    "events",
    "stage_profiler",
    "segment_issue_report",
    "segment_quality_report",
    "ranked_windows",
    "window_reports",
}


@dataclass(frozen=True)
class Asset:
    track_id: str
    mix_path: str
    path: str
    manifest_path: str
    candidate_id: str
    candidate_family: str
    candidate_origin: str
    score_hints: dict[str, Any]


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        result: list[tuple[str, Any]] = []
        for key, child in value.items():
            if str(key) in SKIP_JSON_KEYS:
                continue
            result.extend(flatten(child, f"{prefix}.{key}" if prefix else str(key)))
        return result
    if isinstance(value, list):
        result = []
        for index, child in enumerate(value[:25]):
            result.extend(flatten(child, f"{prefix}.{index}" if prefix else str(index)))
        return result
    return [(prefix, value)]


def is_audio_path(value: Any) -> bool:
    text = str(value)
    return text.lower().endswith(AUDIO_SUFFIXES)


def path_exists(path: str) -> bool:
    return Path(path).exists()


def stable_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def sanitize(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "candidate"


def track_id_from_mix_path(mix_path: str, fallback: str) -> str:
    parts = Path(mix_path).parts
    for part in reversed(parts):
        if "_source_input_" in part:
            return sanitize(part.split("_source_input_", 1)[0])
    if parts and any(part in {"/host_testing_stuff_music", "host_testing_stuff_music"} for part in parts):
        return sanitize(Path(mix_path).stem)
    if len(parts) >= 2 and Path(mix_path).name.lower().startswith("source."):
        return sanitize(parts[-2])
    return sanitize(fallback)


def source_paths(flat: list[tuple[str, Any]]) -> list[str]:
    patterns = ("source_audio_path", "raw_audio_path", "audio_path", "mix_path")
    paths: list[str] = []
    for key, value in flat:
        if not is_audio_path(value):
            continue
        key_lower = key.lower()
        text_lower = str(value).lower()
        name_lower = Path(str(value)).name.lower()
        if "raw_ref" in name_lower or "00_raw_ref" in name_lower:
            continue
        if any(pattern in key_lower for pattern in patterns) or text_lower.endswith("/source.mp3") or text_lower.endswith("\\source.mp3"):
            paths.append(str(value))
    return stable_unique(paths)


def vocal_paths(flat: list[tuple[str, Any]]) -> list[str]:
    paths: list[str] = []
    for key, value in flat:
        if not is_audio_path(value):
            continue
        key_lower = key.lower()
        text_lower = str(value).lower()
        name_lower = Path(str(value)).name.lower()
        if any(part in text_lower for part in SKIP_PATH_PARTS):
            continue
        if any(token in name_lower for token in ("minus", "instrumental")):
            continue
        key_is_vocal_field = any(
            token in key_lower
            for token in (
                "selected_vocals",
                "vocals_path",
                "vocals_flac",
                "stems.vocals.path",
                "listen_selected_vocals",
            )
        )
        name_is_vocal_file = "vocal" in name_lower or name_lower == "vocals.flac"
        if key_is_vocal_field or name_is_vocal_file:
            paths.append(str(value))
    return stable_unique(paths)


def minus_paths(flat: list[tuple[str, Any]]) -> list[str]:
    paths: list[str] = []
    for key, value in flat:
        if not is_audio_path(value):
            continue
        key_lower = key.lower()
        text_lower = str(value).lower()
        name_lower = Path(str(value)).name.lower()
        if any(part in text_lower for part in SKIP_PATH_PARTS):
            continue
        if any(token in key_lower or token in name_lower for token in ("minus", "instrumental", "repaired_minus", "base_minus")):
            paths.append(str(value))
    return stable_unique(paths)


def score_hints(payload: dict[str, Any]) -> dict[str, Any]:
    hints: dict[str, Any] = {}
    for key in ("selected_score", "score", "delivery_status", "delivery_allowed", "presentation_status", "release_status"):
        if key in payload:
            hints[key] = payload[key]
    if isinstance(payload.get("metrics"), dict):
        for key, value in payload["metrics"].items():
            if isinstance(value, (int, float, str, bool)):
                hints[f"metric.{key}"] = value
    return hints


def candidate_name(payload: dict[str, Any], manifest_path: Path, kind: str) -> str:
    pipeline = payload.get("pipeline") if isinstance(payload.get("pipeline"), dict) else {}
    for value in (
        payload.get("candidate_id"),
        payload.get("selected_profile"),
        pipeline.get("candidate_mode"),
        payload.get("mode"),
        manifest_path.parent.name,
    ):
        if value:
            return sanitize(f"{kind}_{value}")
    return sanitize(f"{kind}_{manifest_path.parent.name}")


def assets_from_manifest(manifest_path: Path, payload: dict[str, Any]) -> tuple[list[Asset], list[Asset]]:
    flat = flatten(payload)
    mixes = [path for path in source_paths(flat) if path_exists(path)]
    if not mixes:
        return [], []
    mix_path = sorted(mixes)[0]
    fallback_track = manifest_path.parent.name
    track_id = track_id_from_mix_path(mix_path, fallback_track)
    hints = score_hints(payload)
    origin = "manifest_scan"
    vocal_assets = [
        Asset(
            track_id=track_id,
            mix_path=mix_path,
            path=path,
            manifest_path=str(manifest_path),
            candidate_id=candidate_name(payload, manifest_path, "vocal"),
            candidate_family="vocal",
            candidate_origin=origin,
            score_hints=hints,
        )
        for path in sorted(vocal_paths(flat))
        if path_exists(path)
    ]
    minus_assets = [
        Asset(
            track_id=track_id,
            mix_path=mix_path,
            path=path,
            manifest_path=str(manifest_path),
            candidate_id=candidate_name(payload, manifest_path, "minus"),
            candidate_family="minus",
            candidate_origin=origin,
            score_hints=hints,
        )
        for path in sorted(minus_paths(flat))
        if path_exists(path)
    ]
    return vocal_assets, minus_assets


def merge_hints(vocal: Asset, minus: Asset) -> dict[str, Any]:
    hints: dict[str, Any] = {}
    if vocal.score_hints:
        hints["vocal"] = vocal.score_hints
    if minus.score_hints:
        hints["minus"] = minus.score_hints
    return hints


def pair_row(track_id: str, vocal: Asset, minus: Asset) -> dict[str, Any]:
    candidate_id = sanitize(f"{vocal.candidate_id}__{minus.candidate_id}")
    return {
        "track_id": track_id,
        "candidate_id": candidate_id,
        "mix_path": vocal.mix_path,
        "vocal_path": vocal.path,
        "minus_path": minus.path,
        "instrumental_path": minus.path,
        "source_manifest": f"{vocal.manifest_path};{minus.manifest_path}",
        "candidate_manifest": f"{vocal.manifest_path};{minus.manifest_path}",
        "candidate_family": f"{vocal.candidate_family}+{minus.candidate_family}",
        "candidate_origin": "existing_generated_artifact_pair",
        "score_hints": merge_hints(vocal, minus),
        "is_complete_pair": True,
        "skip_reason": "",
    }


def row_audio_path(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def preview_candidate_origin(row: dict[str, Any]) -> str:
    joined_paths = " ".join(str(row.get(key) or "").lower() for key in ("vocal_path", "instrumental_path", "minus_path"))
    if "segment_blend_previews" in joined_paths:
        return "existing_segment_blend_preview_pair"
    return "existing_no_reference_candidate_jsonl_pair"


def preview_candidate_score_hints(row: dict[str, Any]) -> dict[str, Any]:
    hints: dict[str, Any] = {}
    for key in (
        "can_publish_as_result",
        "can_mark_release_ready",
        "preview_decision",
        "score_delta",
        "allowed_next_step",
        "render_status",
        "source_stem",
        "window_start_sec",
        "window_duration_sec",
    ):
        if key in row:
            hints[key] = row.get(key)
    return hints


def preview_candidate_row(path: Path, row: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    mix_path = row_audio_path(row, "mix_path", "source_mix_path", "source_audio_path", "audio_path")
    vocal_path = row_audio_path(row, "vocal_path", "selected_vocal_path", "rendered_vocal_preview_path")
    minus_path = row_audio_path(row, "instrumental_path", "minus_path", "selected_minus_path", "rendered_minus_preview_path")
    track_id = sanitize(str(row.get("track_id") or track_id_from_mix_path(mix_path, path.parent.name)))
    candidate_id = sanitize(str(row.get("candidate_id") or path.parent.name))
    base = {
        "track_id": track_id,
        "candidate_id": candidate_id,
        "mix_path": mix_path,
        "vocal_path": vocal_path,
        "minus_path": minus_path,
        "instrumental_path": minus_path,
        "source_manifest": str(path),
        "candidate_manifest": str(path),
        "candidate_family": "preview_candidate_pair",
        "candidate_origin": preview_candidate_origin(row),
        "score_hints": preview_candidate_score_hints(row),
    }
    missing = []
    if not mix_path or not path_exists(mix_path):
        missing.append("mix_path")
    if not vocal_path or not path_exists(vocal_path):
        missing.append("vocal_path")
    if not minus_path or not path_exists(minus_path):
        missing.append("minus_path")
    if missing:
        return None, {
            **base,
            "is_complete_pair": False,
            "skip_reason": "missing_" + "_and_".join(missing),
        }
    return {
        **base,
        "is_complete_pair": True,
        "skip_reason": "",
    }, None


def preview_window_judge_candidate_id(row: dict[str, Any]) -> str:
    parts = [
        "preview_improved_window",
        str(row.get("source_stem") or "unknown"),
        str(row.get("window_start_sec") or "0"),
        str(row.get("base_candidate_id") or "base"),
        str(row.get("alternate_candidate_id") or "alternate"),
    ]
    return sanitize("__".join(parts))


def preview_window_judge_score_hints(row: dict[str, Any]) -> dict[str, Any]:
    hints: dict[str, Any] = {}
    for key in (
        "preview_decision",
        "allowed_next_step",
        "score_delta",
        "blocker_delta",
        "base_window_score",
        "preview_window_score",
        "source_stem",
        "window_start_sec",
        "window_duration_sec",
        "can_publish_as_result",
        "can_mark_release_ready",
    ):
        if key in row:
            hints[key] = row.get(key)
    return hints


def preview_window_judge_noop_reason(row: dict[str, Any]) -> str:
    if row.get("preview_differs_from_base") is False:
        return "noop_self_blend_preview_window"
    if str(row.get("allowed_next_step") or "") == "fix_candidate_identity/path-distinctness":
        return "noop_self_blend_preview_window"
    if str(row.get("preview_decision") or "") == "preview_noop_self_blend":
        return "noop_self_blend_preview_window"
    return ""


def preview_window_judge_row(path: Path, row: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    mix_path = row_audio_path(row, "mix_path")
    vocal_path = row_audio_path(row, "preview_vocal_path", "rendered_vocal_preview_path")
    minus_path = row_audio_path(row, "preview_minus_path", "rendered_minus_preview_path")
    track_id = sanitize(str(row.get("track_id") or track_id_from_mix_path(mix_path, path.parent.name)))
    candidate_id = preview_window_judge_candidate_id(row)
    base = {
        "track_id": track_id,
        "candidate_id": candidate_id,
        "mix_path": mix_path,
        "vocal_path": vocal_path,
        "minus_path": minus_path,
        "instrumental_path": minus_path,
        "source_manifest": str(path),
        "candidate_manifest": str(path),
        "candidate_family": "preview_candidate_pair",
        "candidate_origin": "existing_segment_blend_preview_improved_window",
        "score_hints": preview_window_judge_score_hints(row),
    }
    noop_reason = preview_window_judge_noop_reason(row)
    if noop_reason:
        return None, {
            **base,
            "is_complete_pair": False,
            "skip_reason": noop_reason,
        }
    if str(row.get("preview_decision") or "") != "preview_improved_window":
        return None, {
            **base,
            "is_complete_pair": False,
            "skip_reason": "not_preview_improved_window",
        }
    if row.get("can_publish_as_result") is True or row.get("can_mark_release_ready") is True:
        return None, {
            **base,
            "is_complete_pair": False,
            "skip_reason": "unsafe_preview_promotion_flags",
        }
    missing = []
    if not mix_path or not path_exists(mix_path):
        missing.append("mix_path")
    if not vocal_path or not path_exists(vocal_path):
        missing.append("vocal_path")
    if not minus_path or not path_exists(minus_path):
        missing.append("minus_path")
    if missing:
        return None, {
            **base,
            "is_complete_pair": False,
            "skip_reason": "missing_" + "_and_".join(missing),
        }
    return {
        **base,
        "is_complete_pair": True,
        "skip_reason": "",
    }, None


def seed_candidate_pool_row(path: Path, row: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    minus_path = row_audio_path(row, "minus_path", "instrumental_path")
    base = {
        "track_id": sanitize(str(row.get("track_id") or track_id_from_mix_path(str(row.get("mix_path") or ""), path.stem))),
        "candidate_id": sanitize(str(row.get("candidate_id") or row.get("selected_candidate_id") or path.stem)),
        "mix_path": str(row.get("mix_path") or ""),
        "vocal_path": str(row.get("vocal_path") or ""),
        "minus_path": minus_path,
        "instrumental_path": minus_path,
        "source_manifest": str(row.get("source_manifest") or row.get("candidate_manifest") or path),
        "candidate_manifest": str(row.get("candidate_manifest") or row.get("source_manifest") or path),
        "candidate_family": str(row.get("candidate_family") or "vocal+minus"),
        "candidate_origin": str(row.get("candidate_origin") or "explicit_candidate_pool_seed"),
        "score_hints": row.get("score_hints") if isinstance(row.get("score_hints"), dict) else {},
    }
    if row.get("is_complete_pair") is False:
        return None, {
            **base,
            "is_complete_pair": False,
            "skip_reason": str(row.get("skip_reason") or "seed_marked_incomplete"),
        }
    missing = []
    if not base["mix_path"] or not path_exists(base["mix_path"]):
        missing.append("mix_path")
    if not base["vocal_path"] or not path_exists(base["vocal_path"]):
        missing.append("vocal_path")
    if not base["minus_path"] or not path_exists(base["minus_path"]):
        missing.append("minus_path")
    if missing:
        return None, {
            **base,
            "is_complete_pair": False,
            "skip_reason": "missing_" + "_and_".join(missing),
        }
    return {
        **base,
        "is_complete_pair": True,
        "skip_reason": "",
    }, None


def iter_preview_candidate_paths(root: Path) -> Iterable[Path]:
    paths: list[Path] = []
    for filename in sorted(PREVIEW_CANDIDATE_JSONL_FILENAMES):
        paths.extend(root.rglob(filename))
    return sorted(set(paths))


def iter_preview_window_judge_paths(root: Path) -> Iterable[Path]:
    return sorted(root.rglob(PREVIEW_WINDOW_JUDGE_GLOB))


def collect_preview_candidate_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for path in iter_preview_candidate_paths(root):
        for raw_row in iter_jsonl(path):
            row, skip = preview_candidate_row(path, raw_row)
            if row is not None:
                rows.append(row)
            if skip is not None:
                skipped.append(skip)
    for path in iter_preview_window_judge_paths(root):
        for raw_row in iter_jsonl(path):
            row, skip = preview_window_judge_row(path, raw_row)
            if row is not None:
                rows.append(row)
            if skip is not None:
                skipped.append(skip)
    rows.sort(key=lambda item: (item["track_id"], item["candidate_id"], item["vocal_path"], item["minus_path"]))
    skipped.sort(key=lambda item: (item["track_id"], item["candidate_id"], item["skip_reason"]))
    return rows, skipped


def collect_seed_candidate_pool_rows(paths: Sequence[Path] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for path in sorted(paths or []):
        for raw_row in iter_jsonl(path):
            row, skip = seed_candidate_pool_row(path, raw_row)
            if row is not None:
                rows.append(row)
            if skip is not None:
                skipped.append(skip)
    rows.sort(key=lambda item: (item["track_id"], item["candidate_id"], item["vocal_path"], item["minus_path"]))
    skipped.sort(key=lambda item: (item["track_id"], item["candidate_id"], item["skip_reason"]))
    return rows, skipped


def skipped_rows_for_incomplete(track_id: str, vocals: list[Asset], minuses: list[Asset]) -> list[dict[str, Any]]:
    if vocals and not minuses:
        return [
            {
                "track_id": track_id,
                "candidate_id": vocal.candidate_id,
                "mix_path": vocal.mix_path,
                "vocal_path": vocal.path,
                "minus_path": "",
                "source_manifest": vocal.manifest_path,
                "candidate_family": vocal.candidate_family,
                "candidate_origin": vocal.candidate_origin,
                "score_hints": vocal.score_hints,
                "is_complete_pair": False,
                "skip_reason": "missing_minus_path",
            }
            for vocal in vocals
        ]
    if minuses and not vocals:
        return [
            {
                "track_id": track_id,
                "candidate_id": minus.candidate_id,
                "mix_path": minus.mix_path,
                "vocal_path": "",
                "minus_path": minus.path,
                "source_manifest": minus.manifest_path,
                "candidate_family": minus.candidate_family,
                "candidate_origin": minus.candidate_origin,
                "score_hints": minus.score_hints,
                "is_complete_pair": False,
                "skip_reason": "missing_vocal_path",
            }
            for minus in minuses
        ]
    return []


def collect_assets(lab_artifacts_root: Path) -> tuple[dict[str, list[Asset]], dict[str, list[Asset]]]:
    vocals_by_track: dict[str, list[Asset]] = defaultdict(list)
    minuses_by_track: dict[str, list[Asset]] = defaultdict(list)
    for manifest_path in iter_manifest_paths(lab_artifacts_root):
        payload = read_json(manifest_path)
        if payload is None:
            continue
        vocals, minuses = assets_from_manifest(manifest_path, payload)
        for asset in vocals:
            vocals_by_track[asset.track_id].append(asset)
        for asset in minuses:
            minuses_by_track[asset.track_id].append(asset)
    return vocals_by_track, minuses_by_track


def should_scan_json(path: Path) -> bool:
    if path.name not in MANIFEST_FILENAMES and "manifest" not in path.name.lower():
        return False
    for part in path.parts:
        if part in SKIP_DIR_NAMES:
            return False
        if any(part.startswith(prefix) for prefix in SKIP_DIR_PREFIXES):
            return False
    return True


def iter_manifest_paths(root: Path) -> Iterable[Path]:
    return (path for path in sorted(root.rglob("*.json")) if should_scan_json(path))


def collect_candidate_pool(
    lab_artifacts_root: Path,
    max_pairs_per_track: int = DEFAULT_MAX_PAIRS_PER_TRACK,
    max_total_pairs: int = DEFAULT_MAX_TOTAL_PAIRS,
    seed_candidate_pool_jsonls: Sequence[Path] | None = None,
) -> dict[str, Any]:
    vocals_by_track, minuses_by_track = collect_assets(lab_artifacts_root)
    preview_rows, preview_skipped = collect_preview_candidate_rows(lab_artifacts_root)
    seed_rows, seed_skipped = collect_seed_candidate_pool_rows(seed_candidate_pool_jsonls)
    preview_by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in preview_rows:
        preview_by_track[str(row["track_id"])].append(row)
    seed_by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        seed_by_track[str(row["track_id"])].append(row)
    all_tracks = sorted(set(vocals_by_track) | set(minuses_by_track) | set(preview_by_track) | set(seed_by_track))
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = list(preview_skipped) + list(seed_skipped)
    seen_pairs: set[tuple[str, str, str, str]] = set()
    deduplicated_pairs = 0
    capped_pairs = 0

    for track_id in all_tracks:
        vocals = sorted(vocals_by_track.get(track_id, []), key=lambda asset: (asset.path, asset.manifest_path, asset.candidate_id))
        minuses = sorted(minuses_by_track.get(track_id, []), key=lambda asset: (asset.path, asset.manifest_path, asset.candidate_id))
        skipped.extend(skipped_rows_for_incomplete(track_id, vocals, minuses))
        track_rows: list[dict[str, Any]] = []
        for vocal in vocals:
            for minus in minuses:
                mix_path = sorted({vocal.mix_path, minus.mix_path})[0]
                key = (track_id, mix_path, vocal.path, minus.path)
                if key in seen_pairs:
                    deduplicated_pairs += 1
                    continue
                seen_pairs.add(key)
                track_rows.append(pair_row(track_id, vocal, minus))
        for seed_row in sorted(seed_by_track.get(track_id, []), key=lambda row: (row["candidate_id"], row["vocal_path"], row["minus_path"])):
            key = (track_id, seed_row["mix_path"], seed_row["vocal_path"], seed_row["minus_path"])
            if key in seen_pairs:
                deduplicated_pairs += 1
                continue
            seen_pairs.add(key)
            track_rows.append(seed_row)
        for preview_row in sorted(preview_by_track.get(track_id, []), key=lambda row: (row["candidate_id"], row["vocal_path"], row["minus_path"])):
            key = (track_id, preview_row["mix_path"], preview_row["vocal_path"], preview_row["minus_path"])
            if key in seen_pairs:
                deduplicated_pairs += 1
                continue
            seen_pairs.add(key)
            track_rows.append(preview_row)
        track_rows = sorted(track_rows, key=lambda row: (row["candidate_id"], row["vocal_path"], row["minus_path"]))
        if max_pairs_per_track > 0 and len(track_rows) > max_pairs_per_track:
            capped_pairs += len(track_rows) - max_pairs_per_track
            track_rows = track_rows[:max_pairs_per_track]
        rows.extend(track_rows)

    rows = sorted(rows, key=lambda row: (row["track_id"], row["candidate_id"], row["vocal_path"], row["minus_path"]))
    capped_total_pairs = 0
    if max_total_pairs > 0 and len(rows) > max_total_pairs:
        capped_total_pairs = len(rows) - max_total_pairs
        rows = rows[:max_total_pairs]
    skipped = sorted(skipped, key=lambda row: (row["track_id"], row["candidate_id"], row["skip_reason"]))
    summary = build_summary(rows, skipped, deduplicated_pairs, capped_pairs, capped_total_pairs, len(seed_rows))
    return {"rows": rows, "skipped": skipped, "summary": summary}


def build_summary(
    rows: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    deduplicated_pairs: int,
    capped_pairs: int,
    capped_total_pairs: int,
    seed_candidate_pairs: int = 0,
) -> dict[str, Any]:
    candidates_by_track = Counter(row["track_id"] for row in rows)
    tracks_discovered = len(set(candidates_by_track) | {row["track_id"] for row in skipped})
    return {
        "total_candidate_rows_discovered": len(rows) + len(skipped) + deduplicated_pairs + capped_pairs + capped_total_pairs,
        "complete_candidate_pairs": len(rows),
        "skipped_incomplete": len(skipped),
        "deduplicated_pairs": deduplicated_pairs,
        "seed_candidate_pairs": seed_candidate_pairs,
        "capped_pairs": capped_pairs,
        "capped_total_pairs": capped_total_pairs,
        "tracks_discovered": tracks_discovered,
        "tracks_with_1_candidate": sum(1 for count in candidates_by_track.values() if count == 1),
        "tracks_with_2plus_candidates": sum(1 for count in candidates_by_track.values() if count >= 2),
        "skip_reasons": dict(sorted(Counter(row["skip_reason"] for row in skipped).items())),
        "safety": dict(SAFETY_FLAGS),
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# No-Reference Candidate Pool Summary",
        "",
        f"- total_candidate_rows_discovered: {summary['total_candidate_rows_discovered']}",
        f"- complete_candidate_pairs: {summary['complete_candidate_pairs']}",
        f"- skipped_incomplete: {summary['skipped_incomplete']}",
        f"- deduplicated_pairs: {summary['deduplicated_pairs']}",
        f"- seed_candidate_pairs: {summary.get('seed_candidate_pairs', 0)}",
        f"- capped_pairs: {summary['capped_pairs']}",
        f"- capped_total_pairs: {summary['capped_total_pairs']}",
        f"- tracks_discovered: {summary['tracks_discovered']}",
        f"- tracks_with_1_candidate: {summary['tracks_with_1_candidate']}",
        f"- tracks_with_2plus_candidates: {summary['tracks_with_2plus_candidates']}",
        "",
        "## Skip Reasons",
        "",
    ]
    for key, value in summary["skip_reasons"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Safety", ""])
    for key, value in summary["safety"].items():
        lines.append(f"- {key}: {str(value).lower()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(result: dict[str, Any], output_jsonl: Path, summary_md: Path, skipped_jsonl: Path) -> None:
    write_jsonl(output_jsonl, result["rows"])
    write_jsonl(skipped_jsonl, result["skipped"])
    write_summary_md(summary_md, result["summary"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect existing generated vocal/minus pairs for no-reference optimization.")
    parser.add_argument("--lab-artifacts-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--skipped-jsonl", type=Path, required=True)
    parser.add_argument("--candidate-pool-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--max-pairs-per-track", type=int, default=DEFAULT_MAX_PAIRS_PER_TRACK)
    parser.add_argument("--max-total-pairs", type=int, default=DEFAULT_MAX_TOTAL_PAIRS)
    args = parser.parse_args(argv)

    result = collect_candidate_pool(
        args.lab_artifacts_root,
        max_pairs_per_track=args.max_pairs_per_track,
        max_total_pairs=args.max_total_pairs,
        seed_candidate_pool_jsonls=args.candidate_pool_jsonl,
    )
    write_outputs(result, args.output_jsonl, args.summary_md, args.skipped_jsonl)
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
