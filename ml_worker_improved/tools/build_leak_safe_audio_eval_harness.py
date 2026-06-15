from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


VIEW_FILES = {
    "minus_critic": "minus_critic_training_view.jsonl",
    "vocal_critic": "vocal_critic_training_view.jsonl",
}
SPLITS = ("train", "dev", "test")
HARDCASE_BLOCKERS = {
    "severe_phase_collapse",
    "severe_vocal_bleed_in_minus",
    "severe_music_damage",
    "severe_instrumental_bleed_in_vocal",
}
SAFETY_FLAGS = {
    "training_started": False,
    "critic_training_started": False,
    "separator_training_started": False,
    "repair_adapter_training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "thresholds_relaxed": False,
    "release_ready": False,
    "can_publish_as_result": False,
}
START_TIME = time.monotonic()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected_json_object:{path}:{line_number}")
        yield payload


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def progress(path: Path, event: str, *, stage: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "stage": stage,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "cpu_audio_fingerprint_eval",
        "gpu_inference_skipped": True,
        "gpu_inference_skipped_reason": "leak_safe_eval_uses_audio_decode_hashing_and_model2c_shadow_scoring_only",
        "release_ready": False,
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coarse_envelope_hash(mono: np.ndarray, duration_sec: float, *, blocks: int = 64) -> str:
    if mono.size == 0:
        values = [0] * blocks
    else:
        peak = float(np.max(np.abs(mono)))
        normalized = mono / peak if peak > 1.0e-12 else mono
        chunks = np.array_split(np.abs(normalized), blocks)
        values = [int(round(float(chunk.mean() if chunk.size else 0.0) * 255.0)) for chunk in chunks]
    payload = {
        "duration_bucket_sec": round(float(duration_sec), 1),
        "envelope_blocks": values,
    }
    return stable_hash(payload)


def extract_audio_fingerprint(audio_path: Path) -> dict[str, Any]:
    path = Path(audio_path)
    base = {
        "audio_path": str(path),
        "exists": path.exists(),
        "fingerprint_status": "missing",
        "sample_rate": None,
        "channels": None,
        "frames": None,
        "duration_sec": None,
        "exact_audio_sha256": "",
        "near_duplicate_audio_hash": "",
        "release_ready": False,
    }
    if not path.exists():
        return base
    try:
        info = sf.info(str(path))
        audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    except Exception as exc:  # noqa: BLE001 - artifact validator must record unreadable paths without fabricating hashes.
        return {**base, "exists": True, "fingerprint_status": "unreadable", "error": f"{type(exc).__name__}:{exc}"}
    frames = int(audio.shape[0])
    channels = int(audio.shape[1]) if audio.ndim == 2 else 1
    duration_sec = float(frames / float(sample_rate)) if sample_rate else 0.0
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    return {
        **base,
        "exists": True,
        "fingerprint_status": "ok",
        "sample_rate": int(sample_rate),
        "channels": channels,
        "frames": frames,
        "duration_sec": duration_sec,
        "format": str(getattr(info, "format", "") or ""),
        "exact_audio_sha256": path_sha256(path),
        "near_duplicate_audio_hash": _coarse_envelope_hash(mono.astype(np.float32), duration_sec),
    }


def infer_family(row: dict[str, Any], *, field: str, fallback: str) -> str:
    explicit = str(row.get(field) or "").strip()
    if explicit:
        return explicit
    source_config = row.get("source_separator_config")
    if isinstance(source_config, dict):
        for key in ("source_model_family", "model_family", "model_name", "name", "config_id"):
            value = str(source_config.get(key) or "").strip()
            if value:
                return value
    route_id = str(row.get("route_id") or "").strip()
    candidate_id = str(row.get("candidate_id") or "").strip()
    probe = f"{route_id} {candidate_id}".lower()
    for family in ("htdemucs", "demucs", "scnet", "mdx", "roformer", "bs_roformer", "spleeter"):
        if family in probe:
            return family
    if route_id:
        return route_id.split("__", 1)[0]
    return fallback


def row_uid_for_view(view_name: str, row: dict[str, Any]) -> str:
    explicit = str(row.get("row_uid") or "").strip()
    if explicit:
        return explicit
    payload = {
        "view_name": view_name,
        "row_id": str(row.get("row_id") or ""),
        "track_id": str(row.get("track_id") or ""),
        "run_id": str(row.get("run_id") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "stem_type": str(row.get("stem_type") or ""),
        "window_start_sec": as_float(row.get("window_start_sec")),
        "window_end_sec": as_float(row.get("window_end_sec")),
    }
    return stable_hash(payload)


def canonical_row_key(row: dict[str, Any]) -> str:
    return stable_hash(
        {
            "row_id": str(row.get("row_id") or ""),
            "track_id": str(row.get("track_id") or ""),
            "run_id": str(row.get("run_id") or ""),
            "candidate_id": str(row.get("candidate_id") or ""),
            "stem_type": str(row.get("stem_type") or ""),
            "window_start_sec": as_float(row.get("window_start_sec")),
            "window_end_sec": as_float(row.get("window_end_sec")),
        }
    )


def is_hardcase(row: dict[str, Any]) -> bool:
    blockers = {str(item) for item in row.get("blockers", row.get("blocker_labels", [])) or []}
    if blockers & HARDCASE_BLOCKERS:
        return True
    return int(row.get("label") or 0) == 0 and bool(blockers)


def collect_training_view_rows(views_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for view_name, filename in VIEW_FILES.items():
        path = views_dir / filename
        if not path.exists():
            continue
        for index, row in enumerate(iter_jsonl(path)):
            blockers = [str(item) for item in row.get("blocker_labels", row.get("blockers", [])) or []]
            target = "minus" if view_name.startswith("minus") else "vocal"
            rows.append(
                {
                    "row_uid": row_uid_for_view(view_name, row),
                    "canonical_row_key": canonical_row_key(row),
                    "source_view_file": str(path),
                    "source_row_index": index,
                    "view_name": view_name,
                    "target": target,
                    "track_id": str(row.get("track_id") or ""),
                    "run_id": str(row.get("run_id") or ""),
                    "candidate_id": str(row.get("candidate_id") or ""),
                    "route_id": str(row.get("route_id") or ""),
                    "candidate_family": infer_family(row, field="candidate_family", fallback="unknown_candidate_family"),
                    "source_model_family": infer_family(row, field="source_model_family", fallback="unknown_source_model_family"),
                    "stem_type": str(row.get("stem_type") or target),
                    "window_start_sec": as_float(row.get("window_start_sec")),
                    "window_end_sec": as_float(row.get("window_end_sec")),
                    "mix_path": str(row.get("mix_path") or ""),
                    "candidate_audio_path": str(row.get("candidate_audio_path") or ""),
                    "label": int(row.get("label") or 0),
                    "label_source": str(row.get("label_source") or ""),
                    "blockers": blockers,
                    "safe_for_training": row.get("safe_for_this_view") is True,
                    "release_ready": False,
                    "source_row": row,
                }
            )
    return rows


def attach_audio_fingerprints(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cache: dict[str, dict[str, Any]] = {}
    roles_by_path: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for role, key in (("input_mix", "mix_path"), ("candidate_stem", "candidate_audio_path")):
            raw_path = str(row.get(key) or "").strip()
            if not raw_path:
                continue
            roles_by_path[raw_path].add(role)
            if raw_path not in cache:
                cache[raw_path] = extract_audio_fingerprint(Path(raw_path))

    enriched: list[dict[str, Any]] = []
    for row in rows:
        mix_fp = cache.get(str(row.get("mix_path") or ""), {})
        candidate_fp = cache.get(str(row.get("candidate_audio_path") or ""), {})
        enriched.append(
            {
                **row,
                "input_fingerprint_status": mix_fp.get("fingerprint_status", "missing"),
                "input_exact_audio_sha256": str(mix_fp.get("exact_audio_sha256") or ""),
                "input_near_duplicate_audio_hash": str(mix_fp.get("near_duplicate_audio_hash") or ""),
                "input_audio_fingerprint": str(mix_fp.get("near_duplicate_audio_hash") or mix_fp.get("exact_audio_sha256") or ""),
                "candidate_fingerprint_status": candidate_fp.get("fingerprint_status", "missing"),
                "candidate_exact_audio_sha256": str(candidate_fp.get("exact_audio_sha256") or ""),
                "candidate_near_duplicate_audio_hash": str(candidate_fp.get("near_duplicate_audio_hash") or ""),
                "hardcase": is_hardcase(row),
                "release_ready": False,
            }
        )

    index_rows = []
    for audio_path, payload in sorted(cache.items()):
        index_rows.append(
            {
                **payload,
                "roles": sorted(roles_by_path[audio_path]),
                "audio_path": audio_path,
                "release_ready": False,
            }
        )
    return enriched, index_rows


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def grouping_keys(row: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for prefix, field in (
        ("track", "track_id"),
        ("input_exact", "input_exact_audio_sha256"),
        ("input_near", "input_near_duplicate_audio_hash"),
        ("candidate", "candidate_id"),
        ("candidate_exact", "candidate_exact_audio_sha256"),
        ("candidate_near", "candidate_near_duplicate_audio_hash"),
    ):
        value = str(row.get(field) or "").strip()
        if value:
            keys.append(f"{prefix}:{value}")
    return keys


def build_leakage_groups(rows: Sequence[dict[str, Any]]) -> dict[int, list[int]]:
    union_find = UnionFind(len(rows))
    owner_by_key: dict[str, int] = {}
    for index, row in enumerate(rows):
        for key in grouping_keys(row):
            if key in owner_by_key:
                union_find.union(owner_by_key[key], index)
            else:
                owner_by_key[key] = index
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        groups[union_find.find(index)].append(index)
    return dict(groups)


def assign_leak_safe_splits(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    groups = build_leakage_groups(rows)
    ordered_groups = sorted(groups.values(), key=lambda indexes: stable_hash([rows[i]["row_uid"] for i in indexes]))
    hardcase_group_keys = [
        tuple(indexes)
        for indexes in ordered_groups
        if any(rows[index].get("hardcase") is True for index in indexes)
    ]
    hardcase_holdout_count = max(1, len(hardcase_group_keys) // 3) if hardcase_group_keys else 0
    hardcase_holdout_group_keys = set(hardcase_group_keys[:hardcase_holdout_count])
    split_by_group: dict[tuple[int, ...], str] = {}
    for indexes in ordered_groups:
        key = tuple(indexes)
        if key in hardcase_holdout_group_keys:
            split_by_group[key] = "test"

    remaining = [indexes for indexes in ordered_groups if tuple(indexes) not in split_by_group]
    remaining_count = len(remaining)
    test_target = max(1, int(round(len(ordered_groups) * 0.2))) if len(ordered_groups) > 1 else 0
    dev_target = max(1, int(round(len(ordered_groups) * 0.15))) if len(ordered_groups) >= 4 else 0
    current_test = sum(1 for split in split_by_group.values() if split == "test")
    for indexes in remaining:
        key = tuple(indexes)
        if current_test < test_target:
            split_by_group[key] = "test"
            current_test += 1
        elif dev_target > 0 and sum(1 for split in split_by_group.values() if split == "dev") < dev_target:
            split_by_group[key] = "dev"
        else:
            split_by_group[key] = "train"

    if remaining_count and all(split != "train" for split in split_by_group.values()):
        last_non_hardcase = next((tuple(indexes) for indexes in reversed(remaining)), tuple(remaining[-1]))
        split_by_group[last_non_hardcase] = "train"
    train_has_negative = any(
        split_by_group.get(tuple(indexes)) == "train" and any(int(rows[index].get("label") or 0) == 0 for index in indexes)
        for indexes in ordered_groups
    )
    if not train_has_negative:
        move_to_train = next(
            (
                tuple(indexes)
                for indexes in reversed(ordered_groups)
                if tuple(indexes) not in hardcase_holdout_group_keys
                and any(int(rows[index].get("label") or 0) == 0 for index in indexes)
            ),
            None,
        )
        if move_to_train is not None:
            split_by_group[move_to_train] = "train"

    enriched: list[dict[str, Any]] = []
    for indexes in ordered_groups:
        split = split_by_group[tuple(indexes)]
        group_id = stable_hash([rows[index]["row_uid"] for index in indexes])[:16]
        is_holdout_group = tuple(indexes) in hardcase_holdout_group_keys
        for index in indexes:
            row = dict(rows[index])
            row["leakage_group_id"] = group_id
            row["split"] = split
            row["hardcase_holdout"] = bool(is_holdout_group and row.get("hardcase") is True)
            row["train_allowed"] = split == "train" and row.get("safe_for_training") is True and row.get("hardcase_holdout") is not True
            row["release_ready"] = False
            row.pop("source_row", None)
            enriched.append(row)
    return sorted(enriched, key=lambda item: (item["split"], item["track_id"], item["candidate_id"], item["view_name"], item["window_start_sec"]))


def split_values(rows: Sequence[dict[str, Any]], field: str) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {split: set() for split in SPLITS}
    for row in rows:
        split = str(row.get("split") or "train")
        value = str(row.get(field) or "").strip()
        if value:
            values.setdefault(split, set()).add(value)
    return values


def overlap_details(values: dict[str, set[str]]) -> dict[str, list[str]]:
    details: dict[str, list[str]] = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = sorted(values.get(left, set()) & values.get(right, set()))
            if overlap:
                details[f"{left}_vs_{right}"] = overlap[:50]
    return details


def leakage_check(rows: Sequence[dict[str, Any]], field: str, *, critical_if_overlap: bool) -> dict[str, Any]:
    details = overlap_details(split_values(rows, field))
    return {
        "field": field,
        "overlap_count": sum(len(values) for values in details.values()),
        "overlaps": details,
        "critical": bool(details) and bool(critical_if_overlap),
        "warning": bool(details) and not bool(critical_if_overlap),
    }


def intervals_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_start = as_float(left.get("window_start_sec"))
    left_end = as_float(left.get("window_end_sec"))
    right_start = as_float(right.get("window_start_sec"))
    right_end = as_float(right.get("window_end_sec"))
    return max(left_start, right_start) < min(left_end, right_end)


def window_overlap_check(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(str(row.get("track_id") or ""), str(row.get("candidate_id") or ""), str(row.get("stem_type") or ""))].append(row)
    examples: list[dict[str, Any]] = []
    for key_rows in by_key.values():
        ordered = sorted(key_rows, key=lambda item: (as_float(item.get("window_start_sec")), str(item.get("row_uid") or "")))
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                if str(left.get("split") or "") == str(right.get("split") or ""):
                    continue
                if intervals_overlap(left, right):
                    examples.append(
                        {
                            "left_row_uid": left.get("row_uid"),
                            "left_split": left.get("split"),
                            "right_row_uid": right.get("row_uid"),
                            "right_split": right.get("split"),
                            "track_id": left.get("track_id"),
                            "candidate_id": left.get("candidate_id"),
                        }
                    )
    return {
        "field": "window_interval",
        "overlap_count": len(examples),
        "examples": examples[:50],
        "critical": bool(examples),
        "warning": False,
    }


def detect_leakage(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    checks = {
        "track_id_overlap": leakage_check(rows, "track_id", critical_if_overlap=True),
        "input_audio_fingerprint_overlap": leakage_check(rows, "input_audio_fingerprint", critical_if_overlap=True),
        "candidate_id_overlap": leakage_check(rows, "candidate_id", critical_if_overlap=True),
        "same_candidate_train_test": leakage_check(rows, "candidate_id", critical_if_overlap=True),
        "candidate_audio_hash_overlap": leakage_check(rows, "candidate_exact_audio_sha256", critical_if_overlap=True),
        "near_duplicate_audio_hash_overlap": leakage_check(rows, "candidate_near_duplicate_audio_hash", critical_if_overlap=True),
        "route_id_overlap": leakage_check(rows, "route_id", critical_if_overlap=False),
        "candidate_family_overlap": leakage_check(rows, "candidate_family", critical_if_overlap=False),
        "source_model_family_overlap": leakage_check(rows, "source_model_family", critical_if_overlap=False),
        "window_overlap": window_overlap_check(rows),
    }
    critical_count = sum(1 for check in checks.values() if check.get("critical") is True)
    warning_count = sum(1 for check in checks.values() if check.get("warning") is True)
    return {
        "schema": "leakage_report.v1",
        "critical_leakage_count": critical_count,
        "warning_leakage_count": warning_count,
        "checks": checks,
        **SAFETY_FLAGS,
    }


def summarize_counts(rows: Sequence[dict[str, Any]], field: str) -> dict[str, int]:
    values = []
    for row in rows:
        value = row.get(field)
        values.append(str(value) if value is not None and str(value) != "" else "missing")
    return dict(sorted(Counter(values).items()))


def build_hardcase_holdout(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    hardcase_rows = [
        {
            "row_uid": row["row_uid"],
            "track_id": row["track_id"],
            "candidate_id": row["candidate_id"],
            "route_id": row["route_id"],
            "split": row["split"],
            "blockers": row.get("blockers", []),
            "train_allowed": False,
            "release_ready": False,
        }
        for row in rows
        if row.get("hardcase_holdout") is True
    ]
    return {
        "schema": "hardcase_holdout_manifest.v1",
        "locked": True,
        "hardcase_rows_count": len(hardcase_rows),
        "hardcase_rows": hardcase_rows,
        "hardcase_blockers": sorted(HARDCASE_BLOCKERS),
        "train_allowed": False,
        **SAFETY_FLAGS,
    }


def _metrics_from_predictions(rows: Sequence[dict[str, Any]], predictions: Sequence[float]) -> dict[str, Any]:
    labels = [int(row.get("label") or 0) for row in rows]
    scores = [
        float(prediction.get("predicted_accept_probability") if isinstance(prediction, dict) else prediction)
        for prediction in predictions
    ]
    positives = sum(labels)
    negatives = len(labels) - positives
    threshold = 0.5
    false_accept = sum(1 for label, score in zip(labels, scores) if label == 0 and score >= threshold)
    false_reject = sum(1 for label, score in zip(labels, scores) if label == 1 and score < threshold)
    return {
        "rows": len(rows),
        "positive_rows": positives,
        "negative_rows": negatives,
        "false_accept": false_accept,
        "false_reject": false_reject,
        "false_accept_rate": float(false_accept / negatives) if negatives else None,
        "false_reject_rate": float(false_reject / positives) if positives else None,
    }


def reevaluate_model2c_locked_split(
    *,
    model_dir: Path | None,
    views_dir: Path,
    locked_rows: Sequence[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    if model_dir is None:
        return {"status": "skipped_missing_model_dir", **SAFETY_FLAGS}
    if not model_dir.exists():
        return {"status": "skipped_model_dir_not_found", "model_dir": str(model_dir), **SAFETY_FLAGS}
    try:
        from tools.evaluate_target_audio_error_critics import load_bundle
        from tools.train_target_audio_error_critics import TARGETS, load_target_rows, predict_bundle
    except Exception as exc:  # noqa: BLE001
        return {"status": "skipped_import_error", "error": f"{type(exc).__name__}:{exc}", **SAFETY_FLAGS}

    locked_split_by_uid = {str(row["row_uid"]): str(row.get("split") or "") for row in locked_rows}
    locked_split_by_key = {str(row.get("canonical_row_key") or ""): str(row.get("split") or "") for row in locked_rows}
    targets: dict[str, Any] = {}
    for target, cfg in TARGETS.items():
        model_file = cfg.get("model_file", "")
        model_path = model_dir / model_file
        if not model_path.exists():
            targets[target] = {"status": "skipped_model_file_missing", "model_file": model_file, **SAFETY_FLAGS}
            continue
        try:
            bundle, calibration_source = load_bundle(model_dir=model_dir, calibration_dir=None, target=target, model_file=model_file)
            loaded_rows = load_target_rows(views_dir, target)["usable_rows"]
            eval_rows: list[dict[str, Any]] = []
            for row in loaded_rows:
                source_row = row.get("source_row") if isinstance(row.get("source_row"), dict) else row
                uid = row_uid_for_view(str(cfg.get("view_name") or f"{target}_critic"), source_row)
                key = canonical_row_key(source_row)
                if locked_split_by_uid.get(uid) == "test" or locked_split_by_key.get(key) == "test":
                    eval_rows.append(row)
            predictions = predict_bundle(bundle, eval_rows) if eval_rows else []
            targets[target] = {
                "status": "evaluated",
                "model_file": model_file,
                "model_id": str(bundle.get("model_id") or ""),
                "calibration_source": calibration_source,
                **_metrics_from_predictions(eval_rows, predictions),
                **SAFETY_FLAGS,
            }
        except Exception as exc:  # noqa: BLE001
            targets[target] = {"status": "failed_diagnostic_eval", "error": f"{type(exc).__name__}:{exc}", **SAFETY_FLAGS}
    result = {
        "schema": "model_2c_locked_re_eval.v1",
        "status": "evaluated" if any(item.get("status") == "evaluated" for item in targets.values()) else "skipped_no_target_evaluated",
        "model_dir": str(model_dir),
        "views_dir": str(views_dir),
        "threshold_mode": "base_probability_0.5_shadow",
        "targets": targets,
        **SAFETY_FLAGS,
    }
    write_json(output_dir / "model_2c_locked_re_eval.json", result)
    return result


def write_locked_eval_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Leak-Safe Locked Audio Evaluation",
        "",
        f"- rows_total: {manifest['rows_total']}",
        f"- split_counts: `{json.dumps(manifest['split_counts'], sort_keys=True)}`",
        f"- critical_leakage_count: {manifest['critical_leakage_count']}",
        f"- warning_leakage_count: {manifest['warning_leakage_count']}",
        f"- hardcase_rows_count: {manifest['hardcase_rows_count']}",
        f"- model_2c_locked_re_eval_status: `{manifest['model_2c_locked_re_eval']['status']}`",
        "- training_started: false",
        "- critic_training_started: false",
        "- separator_training_started: false",
        "- repair_adapter_training_started: false",
        "- release_ready: false",
        "- can_publish_as_result: false",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_leak_safe_audio_eval_harness(
    *,
    views_dir: Path,
    output_dir: Path,
    model_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "leak_safe_eval_progress.jsonl"
    progress(progress_path, "start", stage="build_leak_safe_eval_harness", views_dir=str(views_dir), model_dir=str(model_dir or ""))
    source_manifest_path = views_dir / "audio_error_training_views_manifest.json"
    source_manifest = read_json(source_manifest_path) if source_manifest_path.exists() else {}
    progress(progress_path, "start", stage="collect_training_view_rows")
    collected_rows = collect_training_view_rows(views_dir)
    progress(progress_path, "finish", stage="collect_training_view_rows", rows=len(collected_rows))
    progress(progress_path, "start", stage="audio_fingerprints")
    fingerprinted_rows, fingerprint_index = attach_audio_fingerprints(collected_rows)
    progress(progress_path, "finish", stage="audio_fingerprints", audio_files=len(fingerprint_index))
    progress(progress_path, "start", stage="assign_split")
    split_rows = assign_leak_safe_splits(fingerprinted_rows)
    progress(progress_path, "finish", stage="assign_split", rows=len(split_rows))
    progress(progress_path, "start", stage="detect_leakage")
    leakage_report = detect_leakage(split_rows)
    progress(progress_path, "finish", stage="detect_leakage", critical_leakage_count=leakage_report["critical_leakage_count"])
    hardcase_holdout = build_hardcase_holdout(split_rows)
    progress(progress_path, "start", stage="model2c_locked_re_eval")
    model_eval = reevaluate_model2c_locked_split(model_dir=model_dir, views_dir=views_dir, locked_rows=split_rows, output_dir=output_dir)
    progress(progress_path, "finish", stage="model2c_locked_re_eval", status=model_eval.get("status"))

    write_jsonl(output_dir / "audio_fingerprint_index.jsonl", fingerprint_index)
    write_jsonl(output_dir / "leak_safe_split_rows.jsonl", split_rows)
    write_json(output_dir / "leakage_report.json", leakage_report)
    write_json(output_dir / "hardcase_holdout_manifest.json", hardcase_holdout)

    split_counts = {split: sum(1 for row in split_rows if row.get("split") == split) for split in SPLITS}
    manifest = {
        "schema": "leak_safe_audio_eval_harness.v1",
        "views_dir": str(views_dir),
        "source_training_views_manifest": str(source_manifest_path) if source_manifest_path.exists() else "",
        "source_dataset_id": str(source_manifest.get("dataset_id") or ""),
        "output_dir": str(output_dir),
        "rows_total": len(split_rows),
        "fingerprinted_audio_files": len(fingerprint_index),
        "missing_or_unreadable_audio_files": sum(
            1 for row in fingerprint_index if row.get("fingerprint_status") != "ok"
        ),
        "split_counts": split_counts,
        "label_counts": summarize_counts(split_rows, "label"),
        "view_counts": summarize_counts(split_rows, "view_name"),
        "critical_leakage_count": leakage_report["critical_leakage_count"],
        "warning_leakage_count": leakage_report["warning_leakage_count"],
        "hardcase_rows_count": hardcase_holdout["hardcase_rows_count"],
        "leak_safe_split_manifest_path": str(output_dir / "leak_safe_split_manifest.json"),
        "split_rows_path": str(output_dir / "leak_safe_split_rows.jsonl"),
        "audio_fingerprint_index_path": str(output_dir / "audio_fingerprint_index.jsonl"),
        "leakage_report_path": str(output_dir / "leakage_report.json"),
        "hardcase_holdout_manifest_path": str(output_dir / "hardcase_holdout_manifest.json"),
        "locked_eval_summary_path": str(output_dir / "locked_eval_summary.md"),
        "progress_path": str(progress_path),
        "future_r1_r2_r3_r4_must_use_this_split": True,
        "model_2c_re_eval_required": model_dir is not None,
        "model_2c_locked_re_eval": model_eval,
        **SAFETY_FLAGS,
    }
    write_json(output_dir / "leak_safe_split_manifest.json", manifest)
    write_locked_eval_summary(output_dir / "locked_eval_summary.md", manifest)
    progress(progress_path, "finish", stage="build_leak_safe_eval_harness", output_manifest=str(output_dir / "leak_safe_split_manifest.json"))
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a leak-safe audio fingerprint evaluation harness.")
    parser.add_argument("--views-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    build_leak_safe_audio_eval_harness(views_dir=args.views_dir, output_dir=args.output_dir, model_dir=args.model_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
