#!/usr/bin/env python3
"""Build track-disjoint pseudo-teacher pairs for a shared minus repair adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402
from mac_windows_mount_preflight import (  # noqa: E402
    DEFAULT_WINDOWS_PROJECT_ROOT,
    preflight_mac_windows_project_mount,
)
from pseudo_teacher_contract import PSEUDO_TEACHER_SCHEMA  # noqa: E402

SCHEMA = "minus_repair_pseudo_teacher_dataset.v1"
PROGRESS_NAME = "progress.jsonl"
TRAIN_ROWS_NAME = "train.jsonl"
HOLDOUT_ROWS_NAME = "holdout.jsonl"
MANIFEST_NAME = "manifest.json"
VARIANTS = ("identity", "vocal_leak", "detail_dip", "combined")
EPSILON = 1.0e-8
START_TIME = time.monotonic()
SAFE_PROJECT_SUFFIX_ANCHORS = ("artifacts", "lab_artifacts", "artifact_rag")
PROJECT_ROOT_ALIASES = ("/work",)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_dsp_dataset_build",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "dataset_build_uses_existing_mps_candidates",
        **fields,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    if audio.shape[0] <= 0:
        raise ValueError(f"empty audio: {path}")
    return audio.astype(np.float32, copy=False), int(sample_rate)


def write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), int(sample_rate), format="FLAC", subtype="PCM_24")


def consumer_path(path: Path, *, producer_project_root: Path, consumer_project_root: Path) -> Path:
    resolved_path = path.resolve()
    resolved_producer_root = producer_project_root.resolve()
    try:
        relative_path = resolved_path.relative_to(resolved_producer_root)
    except ValueError as exc:
        raise ValueError(f"path is outside producer project root: {path}") from exc
    return consumer_project_root / relative_path


def normalized_path_parts(path_value: str | Path) -> tuple[str, ...]:
    normalized = str(path_value or "").strip().replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return tuple(part for part in normalized.split("/") if part and part != ".")


def project_relative_suffix(path_value: str | Path, *, known_roots: Sequence[str | Path]) -> Path:
    parts = normalized_path_parts(path_value)
    if not parts:
        raise ValueError("path is empty")
    for root in known_roots:
        root_parts = normalized_path_parts(root)
        if root_parts and len(parts) > len(root_parts) and parts[: len(root_parts)] == root_parts:
            return Path(*parts[len(root_parts) :])
    for anchor in SAFE_PROJECT_SUFFIX_ANCHORS:
        if anchor in parts:
            anchor_index = parts.index(anchor)
            suffix_parts = parts[anchor_index:]
            if len(suffix_parts) >= 2:
                return Path(*suffix_parts)
    raise ValueError(f"path does not expose a safe project-relative suffix: {path_value}")


def clip_path_suffixes(
    track_row: dict[str, Any],
    *,
    track_id: str,
    producer_project_root: Path,
    consumer_project_root: Path,
    windows_project_root: str,
) -> set[str]:
    known_roots: tuple[str | Path, ...] = (
        producer_project_root,
        consumer_project_root,
        windows_project_root,
        *PROJECT_ROOT_ALIASES,
    )
    suffixes = {
        project_relative_suffix(track_row[key], known_roots=known_roots).as_posix()
        for key in ("windows_clip_path", "mac_clip_path")
        if str(track_row.get(key) or "").strip()
    }
    if not suffixes:
        raise ValueError(f"conditioning proxies require exact clip path provenance for {track_id}")
    return suffixes


def resolve_conditioning_proxies(
    *,
    track_id: str,
    track_row: dict[str, Any],
    conditioning_proxy_roots: Sequence[Path],
    producer_project_root: Path,
    consumer_project_root: Path,
    windows_project_root: str,
) -> list[dict[str, Any]]:
    if not conditioning_proxy_roots:
        return []
    known_roots: tuple[str | Path, ...] = (
        producer_project_root,
        consumer_project_root,
        windows_project_root,
        *PROJECT_ROOT_ALIASES,
    )
    expected_clip_suffixes = clip_path_suffixes(
        track_row,
        track_id=track_id,
        producer_project_root=producer_project_root,
        consumer_project_root=consumer_project_root,
        windows_project_root=windows_project_root,
    )
    proxies: list[dict[str, Any]] = []
    seen_model_ids: set[str] = set()
    seen_vocal_sha256: set[str] = set()
    for proxy_root in conditioning_proxy_roots:
        proxy_manifest_path = proxy_root / track_id / "run" / "stem_manifest.json"
        proxy_manifest = read_json(proxy_manifest_path)
        model_info = proxy_manifest.get("model") if isinstance(proxy_manifest.get("model"), dict) else {}
        manifest_model_id = str(proxy_manifest.get("model_id") or "").strip()
        nested_model_id = str(model_info.get("model_id") or "").strip()
        if manifest_model_id and nested_model_id and manifest_model_id != nested_model_id:
            raise ValueError(f"conditioning proxy has conflicting model ids: {proxy_manifest_path}")
        model_id = manifest_model_id or nested_model_id
        if not model_id:
            raise ValueError(f"conditioning proxy missing model id: {proxy_manifest_path}")
        if model_id in seen_model_ids:
            raise ValueError(f"duplicate conditioning proxy model id for {track_id}: {model_id}")
        source_audio_path = str(proxy_manifest.get("source_audio_path") or "").strip()
        if not source_audio_path:
            raise ValueError(f"conditioning proxy missing source audio path: {proxy_manifest_path}")
        source_audio_suffix = project_relative_suffix(source_audio_path, known_roots=known_roots).as_posix()
        if source_audio_suffix not in expected_clip_suffixes:
            raise ValueError(
                f"conditioning proxy source mismatch for {track_id}: {proxy_manifest_path} -> {source_audio_suffix}"
            )
        stems = proxy_manifest.get("stems") if isinstance(proxy_manifest.get("stems"), dict) else {}
        vocal_stem = stems.get("vocals") if isinstance(stems.get("vocals"), dict) else {}
        target_artifact = (
            proxy_manifest.get("target_stem_artifact")
            if isinstance(proxy_manifest.get("target_stem_artifact"), dict)
            else {}
        )
        role_provenance = {
            str(proxy_manifest.get("target_product_stem") or "").strip().lower(),
            str(proxy_manifest.get("target_stem") or "").strip().lower(),
            str(target_artifact.get("product_stem_type") or "").strip().lower(),
            str(target_artifact.get("user_facing_stem_role") or "").strip().lower(),
            str(target_artifact.get("target_stem") or "").strip().lower(),
            "vocals" if vocal_stem else "",
        }
        if not (role_provenance & {"vocal", "vocals"}):
            raise ValueError(f"conditioning proxy missing vocal role provenance: {proxy_manifest_path}")
        if str(vocal_stem.get("role") or "").strip().lower() != "target":
            raise ValueError(f"conditioning proxy missing target vocal role: {proxy_manifest_path}")
        source_provenance = str(target_artifact.get("source") or vocal_stem.get("source") or "").strip()
        if not source_provenance and str(model_info.get("target_output") or "").strip().lower() in {"vocal", "vocals"}:
            source_provenance = f"model_target_output:{model_id}"
        if not source_provenance:
            raise ValueError(f"conditioning proxy missing source provenance: {proxy_manifest_path}")
        target_audio_candidates = [
            str(target_artifact.get("path") or "").strip(),
            str(vocal_stem.get("path") or "").strip(),
            str(proxy_manifest.get("target_product_audio_path") or "").strip(),
            str(proxy_manifest.get("target_audio_path") or "").strip(),
        ]
        target_audio_candidates = [value for value in target_audio_candidates if value]
        if not target_audio_candidates:
            raise ValueError(f"conditioning proxy missing vocal target path: {proxy_manifest_path}")
        target_audio_suffixes = {
            project_relative_suffix(candidate, known_roots=known_roots).as_posix()
            for candidate in target_audio_candidates
        }
        if len(target_audio_suffixes) != 1:
            raise ValueError(f"conditioning proxy has conflicting vocal target paths: {proxy_manifest_path}")
        vocal_sha256 = str(vocal_stem.get("sha256") or "").strip()
        if not vocal_sha256:
            raise ValueError(f"conditioning proxy missing vocal sha256: {proxy_manifest_path}")
        if vocal_sha256 in seen_vocal_sha256:
            raise ValueError(f"duplicate conditioning proxy vocal sha256 for {track_id}: {vocal_sha256}")
        seen_model_ids.add(model_id)
        seen_vocal_sha256.add(vocal_sha256)
        target_audio_suffix = next(iter(target_audio_suffixes))
        producer_vocal_path = producer_project_root / Path(target_audio_suffix)
        if not producer_vocal_path.exists():
            raise ValueError(f"conditioning proxy vocal target is missing on producer mount: {producer_vocal_path}")
        proxies.append(
            {
                "model_id": model_id,
                "vocal_sha256": vocal_sha256,
                "source_audio_suffix": source_audio_suffix,
                "source_provenance": source_provenance,
                "producer_manifest_path": proxy_manifest_path,
                "published_manifest_path": consumer_path(
                    proxy_manifest_path,
                    producer_project_root=producer_project_root,
                    consumer_project_root=consumer_project_root,
                ),
                "producer_audio_path": producer_vocal_path,
                "published_audio_path": consumer_path(
                    producer_vocal_path,
                    producer_project_root=producer_project_root,
                    consumer_project_root=consumer_project_root,
                ),
            }
        )
    proxies.sort(key=lambda item: item["model_id"])
    return proxies


def deterministic_rng(seed: int, track_id: str, variant: str) -> np.random.Generator:
    payload = f"{int(seed)}:{track_id}:{variant}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)
    return np.random.default_rng(derived_seed)


def vocal_activity_envelope(vocal_audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mono = np.mean(vocal_audio, axis=1, dtype=np.float64)
    frame_length = max(1, int(round(0.040 * float(sample_rate))))
    hop_length = max(1, int(round(0.010 * float(sample_rate))))
    starts = np.arange(0, mono.shape[0], hop_length, dtype=np.int64)
    ends = np.minimum(starts + frame_length, mono.shape[0])
    cumulative = np.concatenate(([0.0], np.cumsum(np.square(mono), dtype=np.float64)))
    frame_energy = (cumulative[ends] - cumulative[starts]) / np.maximum(ends - starts, 1)
    frame_rms = np.sqrt(np.maximum(frame_energy, 0.0))
    floor = float(np.quantile(frame_rms, 0.20))
    ceiling = float(np.quantile(frame_rms, 0.95))
    if ceiling <= floor + EPSILON:
        return np.zeros((mono.shape[0],), dtype=np.float32)
    normalized = np.clip((frame_rms - floor) / (ceiling - floor), 0.0, 1.0)
    centers = np.minimum(starts + frame_length // 2, mono.shape[0] - 1)
    sample_positions = np.arange(mono.shape[0], dtype=np.float64)
    envelope = np.interp(sample_positions, centers.astype(np.float64), normalized, left=normalized[0], right=normalized[-1])
    return np.power(np.clip(envelope, 0.0, 1.0), 0.75).astype(np.float32)


def rms(audio: np.ndarray) -> float:
    return math.sqrt(float(np.mean(np.square(np.asarray(audio, dtype=np.float64)))) + EPSILON)


def corrupt_minus(
    *,
    base_minus: np.ndarray,
    vocal_proxy: np.ndarray,
    envelope: np.ndarray,
    variant: str,
    rng: np.random.Generator,
    leak_ratio_range: tuple[float, float],
    dip_depth_range: tuple[float, float],
    max_peak: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported corruption variant: {variant}")
    target = np.asarray(base_minus, dtype=np.float32).copy()
    source = target.copy()
    leak_ratio = 0.0
    dip_depth = 0.0
    if variant in {"vocal_leak", "combined"}:
        leak_ratio = float(rng.uniform(*leak_ratio_range))
        proxy = np.asarray(vocal_proxy, dtype=np.float32)
        proxy = proxy - np.mean(proxy, axis=0, keepdims=True)
        proxy_rms = rms(proxy)
        desired_rms = rms(target) * leak_ratio
        source += proxy * (desired_rms / max(proxy_rms, EPSILON)) * envelope[:, None]
    if variant in {"detail_dip", "combined"}:
        dip_depth = float(rng.uniform(*dip_depth_range))
        source *= 1.0 - dip_depth * envelope[:, None]
    joint_peak = max(float(np.max(np.abs(source))), float(np.max(np.abs(target))), EPSILON)
    common_scale = min(1.0, float(max_peak) / joint_peak)
    source *= common_scale
    target *= common_scale
    return source.astype(np.float32), target.astype(np.float32), {
        "variant": variant,
        "leak_ratio": round(leak_ratio, 8),
        "dip_depth": round(dip_depth, 8),
        "common_peak_scale": round(common_scale, 8),
        "vocal_activity_fraction": round(float(np.mean(envelope >= 0.25)), 8),
    }


def validate_candidate_manifest(manifest: dict[str, Any], manifest_path: Path) -> None:
    target_stem = str(manifest.get("target_stem") or "").strip()
    model_output_stem = str(manifest.get("model_output_stem_type") or target_stem).strip()
    if target_stem != "instrumental" or model_output_stem != "instrumental":
        raise ValueError(f"candidate is not a dedicated instrumental model output: {manifest_path}")
    contract = manifest.get("contract") if isinstance(manifest.get("contract"), dict) else {}
    if contract.get("fallback_used") is True:
        raise ValueError(f"candidate used fallback: {manifest_path}")
    execution = manifest.get("execution_contract") if isinstance(manifest.get("execution_contract"), dict) else {}
    runtime = execution.get("inference_runtime") if isinstance(execution.get("inference_runtime"), dict) else {}
    if runtime.get("vendor_cpu_fallback_detected") is True:
        raise ValueError(f"candidate used hidden CPU inference fallback: {manifest_path}")


def build_minus_repair_pseudo_teacher_dataset(
    *,
    clip_batch_manifest: Path,
    candidate_root: Path,
    output_dir: Path,
    mac_windows_project_mount: Path,
    conditioning_proxy_roots: Sequence[Path] = (),
    consumer_project_root: Path | None = None,
    windows_project_root: str = DEFAULT_WINDOWS_PROJECT_ROOT,
    seed: int = 20260711,
    leak_ratio_min: float = 0.08,
    leak_ratio_max: float = 0.24,
    dip_depth_min: float = 0.10,
    dip_depth_max: float = 0.38,
    max_peak: float = 0.98,
) -> dict[str, Any]:
    if not (0.0 <= leak_ratio_min <= leak_ratio_max):
        raise ValueError("invalid leak ratio range")
    if not (0.0 <= dip_depth_min <= dip_depth_max < 1.0):
        raise ValueError("invalid dip depth range")
    mount_preflight = preflight_mac_windows_project_mount(
        mac_windows_project_mount=mac_windows_project_mount,
        output_path=output_dir,
        windows_project_root=windows_project_root,
    )
    if mount_preflight.get("preflight_status") != "passed":
        raise RuntimeError(str(mount_preflight.get("blocked_reason") or "mount preflight failed"))
    consumer_project_root = consumer_project_root or mac_windows_project_mount
    conditioning_proxy_roots = [Path(root) for root in conditioning_proxy_roots]
    clip_manifest = read_json(clip_batch_manifest)
    tracks = clip_manifest.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        raise ValueError("clip batch manifest has no tracks")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(
        progress_path,
        "start",
        track_count=len(tracks),
        candidate_root=str(candidate_root),
        conditioning_proxy_root_count=len(conditioning_proxy_roots),
    )
    train_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    track_records: list[dict[str, Any]] = []
    split_track_ids: dict[str, set[str]] = {"train": set(), "holdout": set()}
    for track_index, raw_track in enumerate(tracks, start=1):
        if not isinstance(raw_track, dict):
            raise ValueError("clip track row must be an object")
        track_id = str(raw_track.get("track_id") or "").strip()
        split = str(raw_track.get("split") or "").strip().lower()
        if not track_id or split not in split_track_ids:
            raise ValueError(f"invalid track id/split at index {track_index}")
        split_track_ids[split].add(track_id)
        conditioning_proxies = resolve_conditioning_proxies(
            track_id=track_id,
            track_row=raw_track,
            conditioning_proxy_roots=conditioning_proxy_roots,
            producer_project_root=mac_windows_project_mount,
            consumer_project_root=consumer_project_root,
            windows_project_root=windows_project_root,
        )
        run_dir = candidate_root / track_id / "run"
        candidate_manifest_path = run_dir / "stem_manifest.json"
        candidate_manifest = read_json(candidate_manifest_path)
        validate_candidate_manifest(candidate_manifest, candidate_manifest_path)
        minus_path = run_dir / "instrumental.flac"
        vocal_path = run_dir / "vocals.flac"
        base_minus, sample_rate = read_audio(minus_path)
        vocal_proxy, vocal_rate = read_audio(vocal_path)
        if sample_rate != vocal_rate:
            raise ValueError(f"candidate sample-rate mismatch for {track_id}: {sample_rate} != {vocal_rate}")
        frame_count = min(base_minus.shape[0], vocal_proxy.shape[0])
        base_minus = base_minus[:frame_count]
        vocal_proxy = vocal_proxy[:frame_count]
        envelope = vocal_activity_envelope(vocal_proxy, sample_rate)
        track_rows: list[dict[str, Any]] = []
        for variant_index, variant in enumerate(VARIANTS, start=1):
            rng = deterministic_rng(seed, track_id, variant)
            source, target, corruption = corrupt_minus(
                base_minus=base_minus,
                vocal_proxy=vocal_proxy,
                envelope=envelope,
                variant=variant,
                rng=rng,
                leak_ratio_range=(float(leak_ratio_min), float(leak_ratio_max)),
                dip_depth_range=(float(dip_depth_min), float(dip_depth_max)),
                max_peak=float(max_peak),
            )
            example_id = f"e{variant_index:02d}"
            example_dir = output_dir / "audio" / track_id / example_id
            source_path = example_dir / "src.flac"
            target_path = example_dir / "tgt.flac"
            write_audio(source_path, source, sample_rate)
            write_audio(target_path, target, sample_rate)
            published_source_path = consumer_path(
                source_path,
                producer_project_root=mac_windows_project_mount,
                consumer_project_root=consumer_project_root,
            )
            published_target_path = consumer_path(
                target_path,
                producer_project_root=mac_windows_project_mount,
                consumer_project_root=consumer_project_root,
            )
            published_candidate_manifest_path = consumer_path(
                candidate_manifest_path,
                producer_project_root=mac_windows_project_mount,
                consumer_project_root=consumer_project_root,
            )
            row = {
                "schema": PSEUDO_TEACHER_SCHEMA,
                "id": f"mrp__{track_id}__{example_id}",
                "track_id": track_id,
                "training_group_id": track_id,
                "stem_type": "minus",
                "source_audio_path": str(published_source_path),
                "target_audio_path": str(published_target_path),
                "target_audio_sha256": sha256_file(target_path),
                "split": split,
                "trainable": split == "train",
                "pseudo_teacher": True,
                "pseudo_teacher_source": "synthetic_defect_from_dedicated_minus_candidate",
                "pseudo_teacher_targets_are_ground_truth": False,
                "human_verified": False,
                "reference_verified": False,
                "permission_status": "pseudo_teacher_runtime_artifact",
                "source_candidate_family": "fv7z_synthetic_minus_defect_v1",
                "source_candidate_is_ground_truth": False,
                "target_candidate_family": "melband_inst_fv7z",
                "candidate_execution_manifest_json": str(published_candidate_manifest_path),
                "corruption": corruption,
                "release_ready": False,
                "delivery_allowed": False,
                "can_publish_as_result": False,
            }
            if conditioning_proxies:
                row["conditioning_audio_paths"] = [
                    str(proxy["published_audio_path"]) for proxy in conditioning_proxies
                ]
                row["conditioning_model_ids"] = [proxy["model_id"] for proxy in conditioning_proxies]
            (train_rows if split == "train" else holdout_rows).append(row)
            track_rows.append(row)
            progress(
                progress_path,
                "example_written",
                track_id=track_id,
                split=split,
                example_id=example_id,
                variant=variant,
                conditioning_proxy_count=len(conditioning_proxies),
            )
        track_records.append(
            {
                "track_id": track_id,
                "split": split,
                "candidate_manifest": str(candidate_manifest_path),
                "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
                "base_minus_sha256": sha256_file(minus_path),
                "vocal_proxy_sha256": sha256_file(vocal_path),
                "sample_rate": sample_rate,
                "frame_count": frame_count,
                "example_count": len(track_rows),
                "conditioning_proxy_count": len(conditioning_proxies),
                "conditioning_model_ids": [proxy["model_id"] for proxy in conditioning_proxies],
                "conditioning_vocal_sha256": [proxy["vocal_sha256"] for proxy in conditioning_proxies],
                "conditioning_source_audio_suffixes": [
                    proxy["source_audio_suffix"] for proxy in conditioning_proxies
                ],
                "conditioning_audio_paths": [str(proxy["published_audio_path"]) for proxy in conditioning_proxies],
                "conditioning_proxy_manifests": [
                    str(proxy["published_manifest_path"]) for proxy in conditioning_proxies
                ],
                "conditioning_source_provenance": [
                    proxy["source_provenance"] for proxy in conditioning_proxies
                ],
            }
        )
    split_overlap = sorted(split_track_ids["train"] & split_track_ids["holdout"])
    if split_overlap:
        raise ValueError(f"train/holdout track overlap: {split_overlap}")
    train_path = output_dir / TRAIN_ROWS_NAME
    holdout_path = output_dir / HOLDOUT_ROWS_NAME
    write_jsonl(train_path, train_rows)
    write_jsonl(holdout_path, holdout_rows)
    published_output_dir = consumer_path(
        output_dir,
        producer_project_root=mac_windows_project_mount,
        consumer_project_root=consumer_project_root,
    )
    published_train_path = published_output_dir / TRAIN_ROWS_NAME
    published_holdout_path = published_output_dir / HOLDOUT_ROWS_NAME
    manifest = {
        "schema": SCHEMA,
        "clip_batch_manifest": str(clip_batch_manifest),
        "candidate_root": str(candidate_root),
        "output_dir": str(published_output_dir),
        "producer_output_dir": str(output_dir),
        "producer_project_root": str(mac_windows_project_mount),
        "consumer_project_root": str(consumer_project_root),
        "mount_preflight": mount_preflight,
        "conditioning_proxy_root_count": len(conditioning_proxy_roots),
        "conditioning_proxy_roots": [str(root) for root in conditioning_proxy_roots],
        "conditioning_track_count": sum(
            1 for record in track_records if int(record.get("conditioning_proxy_count", 0)) > 0
        ),
        "seed": int(seed),
        "variants": list(VARIANTS),
        "ranges": {
            "leak_ratio": [float(leak_ratio_min), float(leak_ratio_max)],
            "dip_depth": [float(dip_depth_min), float(dip_depth_max)],
            "max_peak": float(max_peak),
        },
        "train_track_ids": sorted(split_track_ids["train"]),
        "holdout_track_ids": sorted(split_track_ids["holdout"]),
        "track_split_overlap": split_overlap,
        "track_count": len(track_records),
        "train_example_count": len(train_rows),
        "holdout_example_count": len(holdout_rows),
        "tracks": track_records,
        "train_rows_jsonl": str(published_train_path),
        "holdout_rows_jsonl": str(published_holdout_path),
        "contract": {
            "shared_weight_training": True,
            "track_disjoint_holdout": True,
            "per_track_timestamp_hardcode": False,
            "dedicated_minus_model_candidates_only": True,
            "pseudo_teacher_targets_are_ground_truth": False,
            "holdout_rows_trainable": False,
            "requires_real_candidate_holdout_audit": True,
            "optional_multi_proxy_conditioning": True,
            "conditioning_proxy_source_requires_exact_clip_suffix_match": True,
        },
        "training_started": False,
        "models_changed": False,
        "delivery_allowed": False,
        "release_ready": False,
        "presentation_status": "training_dataset_not_for_listen_pack",
        "progress_jsonl": str(published_output_dir / PROGRESS_NAME),
    }
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress(
        progress_path,
        "finish",
        train_example_count=len(train_rows),
        holdout_example_count=len(holdout_rows),
        train_track_count=len(split_track_ids["train"]),
        holdout_track_count=len(split_track_ids["holdout"]),
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-batch-manifest", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mac-windows-project-mount", type=Path, required=True)
    parser.add_argument("--conditioning-proxy-root", type=Path, action="append", default=[])
    parser.add_argument("--consumer-project-root", type=Path)
    parser.add_argument("--windows-project-root", default=DEFAULT_WINDOWS_PROJECT_ROOT)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--leak-ratio-min", type=float, default=0.08)
    parser.add_argument("--leak-ratio-max", type=float, default=0.24)
    parser.add_argument("--dip-depth-min", type=float, default=0.10)
    parser.add_argument("--dip-depth-max", type=float, default=0.38)
    parser.add_argument("--max-peak", type=float, default=0.98)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_minus_repair_pseudo_teacher_dataset.py",
        entrypoint_path="/app/tools/build_minus_repair_pseudo_teacher_dataset.py",
    )
    args = build_arg_parser().parse_args(argv)
    manifest = build_minus_repair_pseudo_teacher_dataset(
        clip_batch_manifest=args.clip_batch_manifest,
        candidate_root=args.candidate_root,
        output_dir=args.output_dir,
        mac_windows_project_mount=args.mac_windows_project_mount,
        conditioning_proxy_roots=list(args.conditioning_proxy_root),
        consumer_project_root=args.consumer_project_root,
        windows_project_root=str(args.windows_project_root),
        seed=int(args.seed),
        leak_ratio_min=float(args.leak_ratio_min),
        leak_ratio_max=float(args.leak_ratio_max),
        dip_depth_min=float(args.dip_depth_min),
        dip_depth_max=float(args.dip_depth_max),
        max_peak=float(args.max_peak),
    )
    print(json.dumps({"manifest_path": str(args.output_dir / MANIFEST_NAME), "train_example_count": manifest["train_example_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
