"""Fail-closed multi-proxy quality audit for dedicated minus candidates.

The audit combines independent vocal-model evidence with source-conservation
metrics and the pitch-robust vocal leak guard. It is diagnostic-only: passing
this proxy gate is necessary for autonomous delivery, but an exact reference
or another accepted production policy gate is still required for release.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
from scipy import signal

from instrumental_self_judge_gate import (
    ProgressWriter,
    classify_issues,
    segment_metrics,
    sha256_file,
    write_json,
    write_jsonl,
)
from mac_melband_pitch_guard import score_pitch_shifted_vocal_leak


SCHEMA = "minus_quality_ensemble.v2"
DETAIL_ISSUES = {
    "low_rms_muffling_proxy",
    "presence_loss_proxy",
    "air_loss_proxy",
}
DETAIL_SEVERITY_KEYS = {
    "low_rms_muffling_proxy": "low_rms_ratio",
    "presence_loss_proxy": "presence_loss",
    "air_loss_proxy": "air_loss",
}
DIRECT_VOCAL_ISSUES = {
    "vocal_stem_corr_proxy",
    "vocal_projection_proxy",
}
FULL_MIX_ISSUE = "full_mix_like_risk_proxy"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _leaf_name(raw_path: str) -> str:
    return raw_path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _relative_to_producer_root(raw_path: str, producer_root: str | Path) -> Path | None:
    normalized_path = raw_path.replace("\\", "/").rstrip("/")
    normalized_root = str(producer_root).replace("\\", "/").rstrip("/")
    if not normalized_root:
        return None
    if normalized_path == normalized_root:
        relative_text = ""
    elif normalized_path.startswith(f"{normalized_root}/"):
        relative_text = normalized_path[len(normalized_root) + 1 :]
    else:
        return None
    relative_parts = PurePosixPath(relative_text).parts
    if any(part in {"", ".", ".."} for part in relative_parts):
        return None
    return Path(*relative_parts)


def _resolve_manifest_path(
    manifest_path: Path,
    raw_path: Any,
    *,
    runtime_project_root: Path | None = None,
    producer_project_roots: Sequence[str | Path] = (),
) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    direct = Path(raw_path)
    if direct.exists():
        return direct
    if runtime_project_root is not None:
        runtime_root = runtime_project_root.resolve()
        for producer_root in producer_project_roots:
            relative = _relative_to_producer_root(raw_path, producer_root)
            if relative is None:
                continue
            mapped = (runtime_root / relative).resolve()
            try:
                mapped.relative_to(runtime_root)
            except ValueError:
                continue
            if mapped.exists():
                return mapped
    sibling = manifest_path.parent / _leaf_name(raw_path)
    if sibling.exists():
        return sibling
    return None


def _model_id(manifest: Mapping[str, Any]) -> str | None:
    candidates = (
        manifest.get("model_id"),
        _nested(manifest, "model", "model_id"),
        manifest.get("model_name"),
        _nested(manifest, "model", "name"),
    )
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _declares_vocal_role(manifest: Mapping[str, Any]) -> bool:
    target_values = (
        manifest.get("target_stem"),
        manifest.get("target_product_stem"),
        _nested(manifest, "model", "target_output"),
    )
    normalized_targets = {
        str(value).strip().lower()
        for value in target_values
        if isinstance(value, str) and value.strip()
    }
    if normalized_targets.intersection({"vocals", "vocal"}):
        return True
    non_exclusive_targets = {"multi_stem", "multistem", "all_stems", "stems"}
    if normalized_targets.difference(non_exclusive_targets):
        return bool(normalized_targets.intersection({"vocals", "vocal"}))
    stem_record = _nested(manifest, "stems", "vocals")
    if not isinstance(stem_record, Mapping):
        return False
    stem_role = stem_record.get("role")
    return stem_role is None or str(stem_role).strip().lower() == "target"


def _proxy_audio_path(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    runtime_project_root: Path | None = None,
    producer_project_roots: Sequence[str | Path] = (),
) -> Path | None:
    candidates = (
        _nested(manifest, "stems", "vocals", "path"),
        _nested(manifest, "artifacts", "vocals_flac"),
        manifest.get("target_audio_path"),
        manifest.get("target_product_audio_path"),
        manifest.get("target_model_output_audio_path"),
    )
    for raw_path in candidates:
        resolved = _resolve_manifest_path(
            manifest_path,
            raw_path,
            runtime_project_root=runtime_project_root,
            producer_project_roots=producer_project_roots,
        )
        if resolved is not None:
            return resolved
    return None


def _proxy_source_path(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    runtime_project_root: Path | None = None,
    producer_project_roots: Sequence[str | Path] = (),
) -> Path | None:
    candidates = (
        _nested(manifest, "source_reference", "path"),
        manifest.get("source_audio_path"),
        _nested(manifest, "artifacts", "raw_ref_flac"),
        _nested(manifest, "stems", "raw_ref", "path"),
    )
    for raw_path in candidates:
        resolved = _resolve_manifest_path(
            manifest_path,
            raw_path,
            runtime_project_root=runtime_project_root,
            producer_project_roots=producer_project_roots,
        )
        if resolved is not None:
            return resolved
    return None


def _load_proxy(
    manifest_path: Path,
    source_sha256: str | None,
    index: int,
    *,
    runtime_project_root: Path | None = None,
    producer_project_roots: Sequence[str | Path] = (),
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    model_id = _model_id(manifest)
    audio_path = _proxy_audio_path(
        manifest_path,
        manifest,
        runtime_project_root=runtime_project_root,
        producer_project_roots=producer_project_roots,
    )
    source_path = _proxy_source_path(
        manifest_path,
        manifest,
        runtime_project_root=runtime_project_root,
        producer_project_roots=producer_project_roots,
    )
    audio_sha256 = sha256_file(audio_path) if audio_path is not None else None
    source_proxy_sha256 = sha256_file(source_path) if source_path is not None else None
    expected_audio_sha256 = _nested(manifest, "stems", "vocals", "sha256")
    reasons: list[str] = []
    if not _declares_vocal_role(manifest):
        reasons.append("manifest_does_not_declare_vocal_role")
    if model_id is None:
        reasons.append("model_identity_missing")
    if audio_path is None:
        reasons.append("vocal_audio_missing")
    if source_path is None:
        reasons.append("proxy_source_audio_missing")
    elif source_sha256 is None or source_proxy_sha256 != source_sha256:
        reasons.append("proxy_source_sha256_mismatch")
    if (
        isinstance(expected_audio_sha256, str)
        and audio_sha256 is not None
        and expected_audio_sha256.lower() != audio_sha256.lower()
    ):
        reasons.append("vocal_audio_sha256_mismatch")
    return {
        "proxy_id": f"p{index:03d}",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "model_id": model_id,
        "audio_path": str(audio_path) if audio_path is not None else None,
        "audio_sha256": audio_sha256,
        "source_audio_path": str(source_path) if source_path is not None else None,
        "source_audio_sha256": source_proxy_sha256,
        "valid": not reasons,
        "invalid_reasons": reasons,
    }


def _select_independent_proxies(proxies: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    seen_audio: set[str] = set()
    for proxy in proxies:
        if not proxy["valid"]:
            excluded.append({"proxy_id": proxy["proxy_id"], "reasons": list(proxy["invalid_reasons"])})
            continue
        model_id = str(proxy["model_id"])
        audio_sha256 = str(proxy["audio_sha256"])
        reasons: list[str] = []
        if model_id in seen_models:
            reasons.append("duplicate_model_identity")
        if audio_sha256 in seen_audio:
            reasons.append("duplicate_proxy_audio_sha256")
        if reasons:
            excluded.append({"proxy_id": proxy["proxy_id"], "reasons": reasons})
            continue
        seen_models.add(model_id)
        seen_audio.add(audio_sha256)
        selected.append(proxy)
    return selected, excluded


def _segments(source_audio_path: Path, segment_sec: float) -> list[dict[str, Any]]:
    if segment_sec <= 0.0:
        raise ValueError("segment_sec must be positive")
    duration = float(sf.info(str(source_audio_path)).duration)
    rows: list[dict[str, Any]] = []
    start_sec = 0.0
    while start_sec < duration - 0.001:
        end_sec = min(duration, start_sec + segment_sec)
        rows.append(
            {
                "segment_id": f"s{len(rows):04d}",
                "window_sec": [round(start_sec, 6), round(end_sec, 6)],
            }
        )
        start_sec = end_sec
    return rows


def _resample_for_pitch(source_path: Path, target_rate: int, scratch_path: Path) -> tuple[Path, dict[str, Any]]:
    source_info = sf.info(str(source_path))
    source_rate = int(source_info.samplerate)
    if source_rate == target_rate:
        return source_path, {
            "source_sample_rate": source_rate,
            "pitch_sample_rate": target_rate,
            "resampled_for_pitch": False,
        }
    audio, _ = sf.read(str(source_path), dtype="float32", always_2d=True)
    divisor = math.gcd(source_rate, target_rate)
    resampled = signal.resample_poly(
        np.asarray(audio, dtype=np.float32),
        target_rate // divisor,
        source_rate // divisor,
        axis=0,
    ).astype(np.float32, copy=False)
    scratch_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(scratch_path), resampled, target_rate, subtype="PCM_24")
    return scratch_path, {
        "source_sample_rate": source_rate,
        "pitch_sample_rate": target_rate,
        "resampled_for_pitch": True,
    }


def _match_audio_to_source(
    audio: np.ndarray,
    *,
    audio_rate: int,
    source_frames: int,
    source_channels: int,
    source_rate: int,
) -> np.ndarray:
    matched = np.asarray(audio, dtype=np.float32)
    if audio_rate != source_rate:
        divisor = math.gcd(audio_rate, source_rate)
        matched = signal.resample_poly(
            matched,
            source_rate // divisor,
            audio_rate // divisor,
            axis=0,
        ).astype(np.float32, copy=False)
    if matched.shape[1] == 1 and source_channels > 1:
        matched = np.repeat(matched, source_channels, axis=1)
    elif matched.shape[1] > source_channels:
        matched = matched[:, :source_channels]
    elif matched.shape[1] < source_channels:
        repeats = int(math.ceil(source_channels / float(matched.shape[1])))
        matched = np.tile(matched, (1, repeats))[:, :source_channels]
    if matched.shape[0] > source_frames:
        matched = matched[:source_frames]
    elif matched.shape[0] < source_frames:
        matched = np.pad(matched, ((0, source_frames - matched.shape[0]), (0, 0)))
    return matched.astype(np.float32, copy=False)


def _build_proxy_residual_references(
    *,
    source_audio_path: Path,
    proxies: Sequence[dict[str, Any]],
    scratch_dir: Path,
) -> list[dict[str, Any]]:
    source_audio, source_rate = sf.read(
        str(source_audio_path),
        dtype="float32",
        always_2d=True,
    )
    source_audio = np.asarray(source_audio, dtype=np.float32)
    records: list[dict[str, Any]] = []
    for proxy in proxies:
        proxy_audio_path = Path(str(proxy["audio_path"]))
        proxy_audio, proxy_rate = sf.read(
            str(proxy_audio_path),
            dtype="float32",
            always_2d=True,
        )
        matched_proxy = _match_audio_to_source(
            proxy_audio,
            audio_rate=int(proxy_rate),
            source_frames=int(source_audio.shape[0]),
            source_channels=int(source_audio.shape[1]),
            source_rate=int(source_rate),
        )
        residual_path = scratch_dir / f"{proxy['proxy_id']}.wav"
        residual = source_audio - matched_proxy
        sf.write(str(residual_path), residual, int(source_rate), format="WAV", subtype="FLOAT")
        records.append(
            {
                "proxy_id": str(proxy["proxy_id"]),
                "model_id": str(proxy["model_id"]),
                "path": residual_path,
                "method": "source_minus_vocal_proxy",
            }
        )
    return records


def _overlaps(left: Sequence[float], right: Sequence[float]) -> bool:
    return float(left[0]) < float(right[1]) and float(right[0]) < float(left[1])


def _load_pitch_regions(result: Mapping[str, Any], fallback_path: Path) -> list[dict[str, Any]]:
    raw_path = result.get("all_regions_path")
    regions_path = Path(str(raw_path)) if isinstance(raw_path, str) else fallback_path
    if not regions_path.is_file() and fallback_path.is_file():
        regions_path = fallback_path
    if not regions_path.is_file():
        return [dict(region) for region in result.get("regions") or [] if isinstance(region, Mapping)]
    regions: list[dict[str, Any]] = []
    with regions_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            loaded = json.loads(line)
            if not isinstance(loaded, dict):
                raise ValueError(f"Expected JSON object at {regions_path}:{line_number}")
            regions.append(loaded)
    return regions


def _map_windows_path(path: Path, runtime_project_root: Path | None, windows_project_root: str | None) -> str | None:
    if runtime_project_root is None or windows_project_root is None:
        return None
    try:
        relative = path.resolve().relative_to(runtime_project_root.resolve())
    except ValueError:
        return None
    return str(PureWindowsPath(windows_project_root).joinpath(*relative.parts))


def _mount_probe(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_path = output_dir / f"probe_{os.getpid()}.tmp"
    token = f"minus-quality-ensemble:{os.getpid()}"
    probe_path.write_text(token, encoding="utf-8")
    observed = probe_path.read_text(encoding="utf-8")
    if observed != token:
        raise RuntimeError(f"Artifact mount probe read mismatch at {probe_path}")
    probe_path.unlink()
    return {
        "probe_path": str(probe_path),
        "write_read_verified": True,
        "cleanup_verified": not probe_path.exists(),
    }


def audit_minus_quality(
    *,
    source_audio_path: Path,
    candidate_minus_path: Path,
    proxy_manifest_paths: Sequence[Path],
    output_dir: Path,
    segment_sec: float = 1.0,
    min_independent_proxies: int = 2,
    pitch_config: Mapping[str, Any] | None = None,
    runtime_project_root: Path | None = None,
    windows_project_root: str | None = None,
    producer_project_roots: Sequence[str | Path] = (),
) -> dict[str, Any]:
    if min_independent_proxies < 2:
        raise ValueError("min_independent_proxies must be at least 2")
    if not source_audio_path.is_file():
        raise FileNotFoundError(source_audio_path)
    if not candidate_minus_path.is_file():
        raise FileNotFoundError(candidate_minus_path)
    if not proxy_manifest_paths:
        raise ValueError("At least one proxy manifest is required")
    if (runtime_project_root is None) != (windows_project_root is None):
        raise ValueError("runtime_project_root and windows_project_root must be provided together")
    if producer_project_roots and runtime_project_root is None:
        raise ValueError("producer_project_roots require runtime_project_root")
    resolved_producer_roots = [str(root) for root in producer_project_roots]
    if windows_project_root is not None and windows_project_root not in resolved_producer_roots:
        resolved_producer_roots.append(windows_project_root)
    if runtime_project_root is not None:
        runtime_root = runtime_project_root.resolve()
        try:
            relative_output = output_dir.resolve().relative_to(runtime_root)
        except ValueError as exc:
            raise ValueError("output_dir must be inside runtime_project_root") from exc
        if not relative_output.parts or relative_output.parts[0] not in {
            "artifacts",
            "lab_artifacts",
            "artifact_rag",
        }:
            raise ValueError("output_dir must be inside an approved project artifact root")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    rows_path = output_dir / "rows.jsonl"
    matrix_path = output_dir / "matrix.json"
    progress_path = output_dir / "progress.jsonl"
    progress = ProgressWriter(progress_path)
    mount_probe = _mount_probe(output_dir)
    progress.mark(
        "start",
        source_audio_path=str(source_audio_path),
        candidate_minus_path=str(candidate_minus_path),
        proxy_manifest_count=len(proxy_manifest_paths),
        expected_acceleration_mode="cpu_dsp_quality_audit",
        gpu_inference_skipped=True,
        gpu_inference_skip_reason="audit_consumes_precomputed_model_outputs",
        producer_project_roots=resolved_producer_roots,
    )

    source_sha256 = sha256_file(source_audio_path)
    candidate_sha256 = sha256_file(candidate_minus_path)
    loaded_proxies = [
        _load_proxy(
            Path(path),
            source_sha256,
            index,
            runtime_project_root=runtime_project_root,
            producer_project_roots=resolved_producer_roots,
        )
        for index, path in enumerate(proxy_manifest_paths, start=1)
    ]
    independent_proxies, excluded_proxies = _select_independent_proxies(loaded_proxies)
    progress.mark(
        "validate_proxies",
        valid_proxy_count=sum(1 for proxy in loaded_proxies if proxy["valid"]),
        independent_proxy_count=len(independent_proxies),
        excluded_proxy_count=len(excluded_proxies),
    )

    candidate_rate = int(sf.info(str(candidate_minus_path)).samplerate)
    scratch_dir: Path | None = None
    pitch_records: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="minus_quality_") as scratch_name:
        scratch_dir = Path(scratch_name)
        for proxy in independent_proxies:
            proxy_id = str(proxy["proxy_id"])
            proxy_audio_path = Path(str(proxy["audio_path"]))
            pitch_input_path, resample_record = _resample_for_pitch(
                proxy_audio_path,
                candidate_rate,
                scratch_dir / f"{proxy_id}.flac",
            )

            def on_pitch_progress(event: dict[str, Any], *, current_proxy_id: str = proxy_id) -> None:
                progress.mark("pitch_progress", proxy_id=current_proxy_id, **event)

            progress.mark("pitch_start", proxy_id=proxy_id, model_id=proxy["model_id"])
            pitch_result = score_pitch_shifted_vocal_leak(
                vocals_path=pitch_input_path,
                instrumental_path=candidate_minus_path,
                output_dir=output_dir / "pitch" / proxy_id,
                config=pitch_config,
                progress_callback=on_pitch_progress,
            )
            all_pitch_regions = _load_pitch_regions(
                pitch_result,
                output_dir / "pitch" / proxy_id / "regions.jsonl",
            )
            pitch_records[proxy_id] = {
                **resample_record,
                "verdict": pitch_result.get("verdict"),
                "blocker_reasons": list(pitch_result.get("blocker_reasons") or []),
                "regions": all_pitch_regions,
                "total_fail_regions": int(pitch_result.get("total_fail_regions") or 0),
                "best_probe": pitch_result.get("best_probe"),
                "manifest_path": str(output_dir / "pitch" / proxy_id / "pitch.json"),
            }
            progress.mark(
                "pitch_finish",
                proxy_id=proxy_id,
                verdict=pitch_result.get("verdict"),
                total_fail_regions=int(pitch_result.get("total_fail_regions") or 0),
            )
    pitch_scratch_cleanup_verified = scratch_dir is not None and not scratch_dir.exists()
    if not pitch_scratch_cleanup_verified:
        raise RuntimeError(f"Container-local scratch cleanup failed: {scratch_dir}")

    detail_scratch = tempfile.TemporaryDirectory(prefix="minus_detail_reference_")
    detail_scratch_dir = Path(detail_scratch.name)
    residual_references = _build_proxy_residual_references(
        source_audio_path=source_audio_path,
        proxies=independent_proxies,
        scratch_dir=detail_scratch_dir,
    )
    rows: list[dict[str, Any]] = []
    for segment in _segments(source_audio_path, segment_sec):
        segment_id = str(segment["segment_id"])
        window_sec = list(segment["window_sec"])
        base_metrics = segment_metrics(
            source_audio_path=source_audio_path,
            candidate_instrumental_path=candidate_minus_path,
            window_sec=window_sec,
            candidate_vocal_path=None,
            diagnostic_non_vocal_support_path=None,
            selector_segment_features=None,
        )
        base_issues, _base_severities = classify_issues(base_metrics)
        source_diagnostic_detail_issues = sorted(DETAIL_ISSUES.intersection(base_issues))
        detail_votes: dict[str, list[str]] = {issue: [] for issue in sorted(DETAIL_ISSUES)}
        detail_reference_evidence: list[dict[str, Any]] = []
        for reference in residual_references:
            detail_metrics = segment_metrics(
                source_audio_path=Path(reference["path"]),
                candidate_instrumental_path=candidate_minus_path,
                window_sec=window_sec,
                candidate_vocal_path=None,
                diagnostic_non_vocal_support_path=None,
                selector_segment_features=None,
            )
            detail_reference_issues, detail_severities = classify_issues(detail_metrics)
            confirmed_reference_issues = sorted(DETAIL_ISSUES.intersection(detail_reference_issues))
            for issue in confirmed_reference_issues:
                detail_votes[issue].append(str(reference["proxy_id"]))
            detail_reference_evidence.append(
                {
                    "proxy_id": reference["proxy_id"],
                    "model_id": reference["model_id"],
                    "method": reference["method"],
                    "detail_issues": confirmed_reference_issues,
                    "metrics": {
                        key: detail_metrics[key]
                        for key in (
                            "rms_ratio",
                            "rms_ratio_db_vs_source",
                            "source_candidate_corr_abs",
                            "source_candidate_delta_rms_ratio",
                            "presence_ratio_db_vs_source",
                            "air_ratio_db_vs_source",
                        )
                    },
                    "issue_severities": {
                        DETAIL_SEVERITY_KEYS[issue]: detail_severities.get(
                            DETAIL_SEVERITY_KEYS[issue],
                            0.0,
                        )
                        for issue in confirmed_reference_issues
                    },
                }
            )
        proxy_evidence: list[dict[str, Any]] = []
        direct_votes: list[str] = []
        full_mix_votes: list[str] = []
        pitch_votes: list[str] = []
        for proxy in independent_proxies:
            proxy_id = str(proxy["proxy_id"])
            metrics = segment_metrics(
                source_audio_path=source_audio_path,
                candidate_instrumental_path=candidate_minus_path,
                window_sec=window_sec,
                candidate_vocal_path=Path(str(proxy["audio_path"])),
                diagnostic_non_vocal_support_path=None,
                selector_segment_features=None,
            )
            issues, severities = classify_issues(metrics)
            direct_issues = sorted(DIRECT_VOCAL_ISSUES.intersection(issues))
            full_mix_flag = FULL_MIX_ISSUE in issues
            pitch_regions = pitch_records.get(proxy_id, {}).get("regions") or []
            pitch_flag = any(
                isinstance(region, Mapping)
                and isinstance(region.get("start"), (int, float))
                and isinstance(region.get("end"), (int, float))
                and _overlaps(window_sec, [float(region["start"]), float(region["end"])])
                for region in pitch_regions
            )
            if direct_issues:
                direct_votes.append(proxy_id)
            if full_mix_flag:
                full_mix_votes.append(proxy_id)
            if pitch_flag:
                pitch_votes.append(proxy_id)
            proxy_evidence.append(
                {
                    "proxy_id": proxy_id,
                    "model_id": proxy["model_id"],
                    "metrics": {
                        key: metrics[key]
                        for key in (
                            "vocal_stem_corr_abs",
                            "vocal_projection_db_vs_instrumental",
                            "source_candidate_corr_abs",
                            "source_candidate_delta_rms_ratio",
                        )
                        if key in metrics
                    },
                    "direct_vocal_issues": direct_issues,
                    "full_mix_like": full_mix_flag,
                    "pitch_shifted_leak": pitch_flag,
                    "issue_severities": {
                        key: value
                        for key, value in severities.items()
                        if key in {"full_mix_like_risk", "vocal_stem_corr", "vocal_projection"}
                    },
                }
            )

        confirmed_issues: list[str] = []
        uncertain_issues: list[str] = []
        for issue, votes in detail_votes.items():
            if len(votes) >= min_independent_proxies:
                confirmed_issues.append(issue)
            elif votes:
                uncertain_issues.append(f"single_proxy_{issue}")
        vote_groups = (
            (direct_votes, "vocal_leak_proxy_consensus", "single_proxy_vocal_leak_evidence"),
            (full_mix_votes, "full_mix_like_proxy_consensus", "single_proxy_full_mix_like_evidence"),
            (pitch_votes, "pitch_shifted_vocal_leak_consensus", "single_proxy_pitch_shifted_leak_evidence"),
        )
        for votes, confirmed_label, uncertain_label in vote_groups:
            if len(votes) >= min_independent_proxies:
                confirmed_issues.append(confirmed_label)
            elif votes:
                uncertain_issues.append(uncertain_label)
        severity_values: list[float] = []
        for evidence in detail_reference_evidence:
            severity_values.extend(float(value) for value in evidence["issue_severities"].values())
        for evidence in proxy_evidence:
            severity_values.extend(float(value) for value in evidence["issue_severities"].values())
        row = {
            "segment_id": segment_id,
            "window_sec": window_sec,
            "source_conservation": {
                key: base_metrics[key]
                for key in (
                    "rms_ratio",
                    "rms_ratio_db_vs_source",
                    "source_candidate_corr_abs",
                    "source_candidate_delta_rms_ratio",
                    "presence_ratio_db_vs_source",
                    "air_ratio_db_vs_source",
                )
            },
            "detail_reference": {
                "mode": "proxy_residual_consensus",
                "source_diagnostic_only_issues": source_diagnostic_detail_issues,
                "proxy_residual_evidence": detail_reference_evidence,
            },
            "proxy_evidence": proxy_evidence,
            "votes": {
                "detail_loss": detail_votes,
                "direct_vocal": direct_votes,
                "full_mix_like": full_mix_votes,
                "pitch_shifted_vocal": pitch_votes,
                "required": min_independent_proxies,
            },
            "confirmed_issues": sorted(set(confirmed_issues)),
            "uncertain_issues": sorted(set(uncertain_issues)),
            "blocker": bool(confirmed_issues),
            "uncertain": bool(uncertain_issues),
            "severity_score": round(max(severity_values, default=0.0), 6),
        }
        rows.append(row)
        progress.mark(
            "segment",
            segment_id=segment_id,
            blocker=row["blocker"],
            uncertain=row["uncertain"],
            confirmed_issues=row["confirmed_issues"],
        )

    detail_scratch.cleanup()
    detail_scratch_cleanup_verified = not detail_scratch_dir.exists()
    if not detail_scratch_cleanup_verified:
        raise RuntimeError(f"Detail-reference scratch cleanup failed: {detail_scratch_dir}")
    scratch_cleanup_verified = pitch_scratch_cleanup_verified and detail_scratch_cleanup_verified

    blocker_rows = [row for row in rows if row["blocker"]]
    uncertain_rows = [row for row in rows if row["uncertain"]]
    unknown_pitch_proxies = sorted(
        proxy_id
        for proxy_id, record in pitch_records.items()
        if record.get("verdict") not in {"pass", "fail"}
    )
    quality_blocker_reasons: list[str] = []
    if len(independent_proxies) < min_independent_proxies:
        quality_blocker_reasons.append("insufficient_independent_vocal_proxies")
    if blocker_rows:
        quality_blocker_reasons.append("confirmed_quality_defects")
    if uncertain_rows:
        quality_blocker_reasons.append("inconclusive_single_proxy_evidence")
    if unknown_pitch_proxies:
        quality_blocker_reasons.append("pitch_guard_unknown")
    quality_gate_passed = not quality_blocker_reasons
    if blocker_rows:
        status = "proxy_blocked"
    elif quality_blocker_reasons:
        status = "proxy_unknown"
    else:
        status = "proxy_pass"

    issue_counts: dict[str, int] = {}
    for row in blocker_rows:
        for issue in row["confirmed_issues"]:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    uncertain_issue_counts: dict[str, int] = {}
    for row in uncertain_rows:
        for issue in row["uncertain_issues"]:
            uncertain_issue_counts[issue] = uncertain_issue_counts.get(issue, 0) + 1

    matrix = {
        "schema": SCHEMA,
        "detail_reference": {
            "mode": "proxy_residual_consensus",
            "method": "source_minus_vocal_proxy",
            "required_votes": min_independent_proxies,
            "source_comparison_is_diagnostic_only": True,
        },
        "independence": {
            "required": min_independent_proxies,
            "selected_proxy_ids": [proxy["proxy_id"] for proxy in independent_proxies],
            "selected_model_ids": [proxy["model_id"] for proxy in independent_proxies],
            "selected_audio_sha256": [proxy["audio_sha256"] for proxy in independent_proxies],
            "excluded": excluded_proxies,
        },
        "pitch": pitch_records,
        "issue_counts": issue_counts,
        "uncertain_issue_counts": uncertain_issue_counts,
    }
    write_json(matrix_path, matrix)
    write_jsonl(rows_path, rows)

    manifest = {
        "schema": SCHEMA,
        "mode": "multi_proxy_residual_minus_quality_audit",
        "source": {
            "path": str(source_audio_path),
            "sha256": source_sha256,
        },
        "candidate_minus": {
            "path": str(candidate_minus_path),
            "sha256": candidate_sha256,
        },
        "proxies": loaded_proxies,
        "overall": {
            "status": status,
            "quality_gate_passed": quality_gate_passed,
            "quality_blocker_reasons": quality_blocker_reasons,
            "segment_count": len(rows),
            "confirmed_blocker_count": len(blocker_rows),
            "uncertain_segment_count": len(uncertain_rows),
            "issue_counts": issue_counts,
            "uncertain_issue_counts": uncertain_issue_counts,
            "top_blockers": sorted(
                blocker_rows,
                key=lambda row: (-float(row["severity_score"]), str(row["segment_id"])),
            )[:10],
        },
        "artifacts": {
            "manifest": str(manifest_path),
            "rows": str(rows_path),
            "matrix": str(matrix_path),
            "progress": str(progress_path),
        },
        "artifact_name_policy": {
            "path_segment_limit": 32,
            "raw_ids_in_paths": False,
            "provenance_in_manifest": True,
        },
        "mount_contract": {
            **mount_probe,
            "runtime_project_root": str(runtime_project_root) if runtime_project_root is not None else None,
            "windows_project_root": windows_project_root,
            "producer_project_roots": resolved_producer_roots,
            "manifest_path_mapping_enabled": bool(runtime_project_root and resolved_producer_roots),
            "runtime_output_path": str(output_dir),
            "windows_output_path": _map_windows_path(output_dir, runtime_project_root, windows_project_root),
            "scratch_cleanup_verified": scratch_cleanup_verified,
            "pitch_scratch_cleanup_verified": pitch_scratch_cleanup_verified,
            "detail_reference_scratch_cleanup_verified": detail_scratch_cleanup_verified,
            "scratch_storage": "container_local_temporary_directory",
        },
        "execution_contract": {
            "route_name": "audit_minus_quality_ensemble",
            "expected_acceleration_mode": "cpu_dsp_quality_audit",
            "gpu_inference_skipped": True,
            "gpu_inference_skip_reason": "audit_consumes_precomputed_model_outputs",
            "fallback_used": False,
            "per_track_hardcode_allowed": False,
            "dedicated_minus_candidate_required": True,
            "non_vocal_stem_sum_allowed": False,
            "detail_reference_mode": "proxy_residual_consensus",
            "source_conservation_role": "diagnostic_only_not_detail_blocker",
        },
        "delivery_contract": {
            "delivery_allowed": False,
            "delivery_status": "not_a_delivery_artifact",
            "presentation_status": "diagnostic_quality_evidence_only",
            "listen_pack_allowed": False,
            "acceptance_blocker_reasons": ["exact_reference_or_accepted_policy_gate_required"],
        },
    }
    write_json(manifest_path, manifest)
    progress.mark(
        "finish",
        status=status,
        quality_gate_passed=quality_gate_passed,
        confirmed_blocker_count=len(blocker_rows),
        uncertain_segment_count=len(uncertain_rows),
        manifest_path=str(manifest_path),
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_audio")
    parser.add_argument("candidate_minus")
    parser.add_argument("output_dir")
    parser.add_argument("--proxy-manifest", action="append", required=True)
    parser.add_argument("--segment-sec", type=float, default=1.0)
    parser.add_argument("--min-independent-proxies", type=int, default=2)
    parser.add_argument("--pitch-config")
    parser.add_argument("--runtime-project-root")
    parser.add_argument("--windows-project-root")
    parser.add_argument("--producer-project-root", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    pitch_config = _load_json(Path(args.pitch_config)) if args.pitch_config else None
    manifest = audit_minus_quality(
        source_audio_path=Path(args.source_audio),
        candidate_minus_path=Path(args.candidate_minus),
        proxy_manifest_paths=[Path(path) for path in args.proxy_manifest],
        output_dir=Path(args.output_dir),
        segment_sec=float(args.segment_sec),
        min_independent_proxies=int(args.min_independent_proxies),
        pitch_config=pitch_config,
        runtime_project_root=Path(args.runtime_project_root) if args.runtime_project_root else None,
        windows_project_root=str(args.windows_project_root) if args.windows_project_root else None,
        producer_project_roots=[str(path) for path in args.producer_project_root],
    )
    print(
        json.dumps(
            {
                "manifest_path": manifest["artifacts"]["manifest"],
                "status": manifest["overall"]["status"],
                "quality_gate_passed": manifest["overall"]["quality_gate_passed"],
                "confirmed_blocker_count": manifest["overall"]["confirmed_blocker_count"],
                "delivery_allowed": manifest["delivery_contract"]["delivery_allowed"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
