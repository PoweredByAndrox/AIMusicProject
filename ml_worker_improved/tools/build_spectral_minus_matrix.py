"""Build spectral non-vocal-detail restoration candidates from quality evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import istft, stft


TOOLS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TOOLS_ROOT.parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from build_quality_guided_minus_matrix import (  # noqa: E402
    _manifest_rows_path,
    _model_ids,
    _mount_probe,
    _validate_inside_project,
    _validate_quality_manifest,
    read_json,
    read_jsonl,
    select_repair_windows,
    sha256_file,
    write_json,
    write_jsonl,
)
from docker_runtime import require_docker_runtime  # noqa: E402
from tools.render_no_reference_full_track_segment_switch_candidate import (  # noqa: E402
    load_like_base,
    read_audio,
)


SCHEMA = "spectral_minus_matrix.v1"
DEFAULT_RESTORE_STRENGTHS = (0.25, 0.5, 0.75, 1.0)
DEFAULT_MASK_POWERS = (1.0, 2.0)
EPSILON = 1e-8


class ProgressWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started_at = time.monotonic()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def mark(self, event: str, **fields: Any) -> None:
        row = {
            "event": event,
            "elapsed_sec": round(time.monotonic() - self.started_at, 3),
            "expected_acceleration_mode": "cpu_dsp_spectral_consensus_restore",
            "gpu_inference_skipped": True,
            "gpu_inference_skip_reason": "repair_consumes_precomputed_mps_model_outputs",
        }
        row.update(fields)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()


def build_sample_gate(
    frame_count: int,
    *,
    sample_rate: int,
    windows: Sequence[Mapping[str, Any]],
    crossfade_ms: int,
) -> np.ndarray:
    gate = np.zeros(frame_count, dtype=np.float32)
    fade_frames = max(0, int(round(sample_rate * crossfade_ms / 1000.0)))
    for window in windows:
        start_sec, end_sec = (float(value) for value in window["window_sec"])
        start = max(0, min(frame_count, int(round(start_sec * sample_rate))))
        end = max(start, min(frame_count, int(round(end_sec * sample_rate))))
        if end <= start:
            continue
        weights = np.ones(end - start, dtype=np.float32)
        fade = min(fade_frames, (end - start) // 2)
        if fade > 0 and start > 0:
            weights[:fade] = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
        if fade > 0 and end < frame_count:
            weights[-fade:] = np.linspace(1.0, 0.0, fade, endpoint=True, dtype=np.float32)
        gate[start:end] = np.maximum(gate[start:end], weights)
    return gate


def compute_consensus_support(
    delta_magnitude: np.ndarray,
    proxy_magnitudes: Sequence[np.ndarray],
    *,
    active_frames: np.ndarray,
    consensus_quantile: float,
    min_proxy_scale: float,
    max_proxy_scale: float,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    if not proxy_magnitudes:
        raise ValueError("At least one vocal proxy magnitude is required")
    if active_frames.ndim != 1 or active_frames.shape[0] != delta_magnitude.shape[1]:
        raise ValueError("active_frames must match STFT time frames")
    scale_rows: list[dict[str, float]] = []
    supports: list[np.ndarray] = []
    active = active_frames.astype(bool)
    for proxy_magnitude in proxy_magnitudes:
        if proxy_magnitude.shape != delta_magnitude.shape:
            raise ValueError("proxy magnitude shape mismatch")
        delta_active = delta_magnitude[:, active]
        proxy_active = proxy_magnitude[:, active]
        numerator = float(np.sum(delta_active * proxy_active, dtype=np.float64))
        denominator = float(np.sum(np.square(proxy_active), dtype=np.float64)) + EPSILON
        raw_scale = numerator / denominator
        scale = float(np.clip(raw_scale, min_proxy_scale, max_proxy_scale))
        scaled_proxy = scale * proxy_magnitude
        support = np.clip(scaled_proxy / np.maximum(delta_magnitude, EPSILON), 0.0, 1.0).astype(np.float32)
        supports.append(support)
        scale_rows.append(
            {
                "raw_scale": round(raw_scale, 6),
                "applied_scale": round(scale, 6),
                "support_mean_active": round(float(np.mean(support[:, active])), 6),
            }
        )
    consensus = np.quantile(
        np.stack(supports, axis=0),
        float(consensus_quantile),
        axis=0,
    ).astype(np.float32)
    return np.clip(consensus, 0.0, 1.0), scale_rows


def prepare_spectral_components(
    base_audio: np.ndarray,
    alternate_audio: np.ndarray,
    proxy_audios: Sequence[np.ndarray],
    *,
    sample_rate: int,
    windows: Sequence[Mapping[str, Any]],
    consensus_quantile: float,
    min_proxy_scale: float,
    max_proxy_scale: float,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(proxy_audios) < 3:
        raise ValueError("At least three independent vocal proxies are required")
    win_length = n_fft
    sample_gate = build_sample_gate(
        base_audio.shape[0],
        sample_rate=sample_rate,
        windows=windows,
        crossfade_ms=0,
    )
    components: list[dict[str, Any]] = []
    channel_evidence: list[dict[str, Any]] = []
    for channel_index in range(base_audio.shape[1]):
        _, times, base_z = stft(
            base_audio[:, channel_index],
            fs=sample_rate,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            boundary="zeros",
        )
        _, _, alternate_z = stft(
            alternate_audio[:, channel_index],
            fs=sample_rate,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            boundary="zeros",
        )
        delta_z = base_z - alternate_z
        delta_magnitude = np.abs(delta_z).astype(np.float32)
        proxy_magnitudes: list[np.ndarray] = []
        for proxy_audio in proxy_audios:
            _, _, proxy_z = stft(
                proxy_audio[:, min(channel_index, proxy_audio.shape[1] - 1)],
                fs=sample_rate,
                nperseg=win_length,
                noverlap=win_length - hop_length,
                nfft=n_fft,
                boundary="zeros",
            )
            proxy_magnitudes.append(np.abs(proxy_z).astype(np.float32))
        active_frames = np.interp(
            times,
            np.arange(sample_gate.shape[0], dtype=np.float64) / float(sample_rate),
            sample_gate,
            left=0.0,
            right=0.0,
        ) > 0.5
        if not np.any(active_frames):
            raise RuntimeError("selected quality windows produced no active STFT frames")
        consensus, scale_rows = compute_consensus_support(
            delta_magnitude,
            proxy_magnitudes,
            active_frames=active_frames,
            consensus_quantile=consensus_quantile,
            min_proxy_scale=min_proxy_scale,
            max_proxy_scale=max_proxy_scale,
        )
        components.append(
            {
                "base_z": base_z,
                "alternate_z": alternate_z,
                "delta_z": delta_z,
                "consensus": consensus,
            }
        )
        channel_evidence.append(
            {
                "channel_index": channel_index,
                "proxy_scales": scale_rows,
                "consensus_mean_active": round(float(np.mean(consensus[:, active_frames])), 6),
                "consensus_max_active": round(float(np.max(consensus[:, active_frames])), 6),
            }
        )
    return components, {"channels": channel_evidence, "n_fft": n_fft, "hop_length": hop_length}


def render_spectral_candidate(
    base_audio: np.ndarray,
    components: Sequence[Mapping[str, Any]],
    *,
    sample_rate: int,
    windows: Sequence[Mapping[str, Any]],
    restore_strength: float,
    mask_power: float,
    crossfade_ms: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> tuple[np.ndarray, dict[str, float]]:
    if not 0.0 <= restore_strength <= 1.0:
        raise ValueError("restore_strength must be in [0, 1]")
    if mask_power <= 0.0:
        raise ValueError("mask_power must be positive")
    spectral_channels: list[np.ndarray] = []
    restore_weight_means: list[float] = []
    for component in components:
        consensus = np.asarray(component["consensus"], dtype=np.float32)
        restore_weight = np.power(np.clip(1.0 - consensus, 0.0, 1.0), mask_power).astype(np.float32)
        candidate_z = component["alternate_z"] + restore_strength * restore_weight * component["delta_z"]
        _, channel = istft(
            candidate_z,
            fs=sample_rate,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            nfft=n_fft,
            input_onesided=True,
            boundary=True,
        )
        if channel.shape[0] < base_audio.shape[0]:
            channel = np.pad(channel, (0, base_audio.shape[0] - channel.shape[0]))
        spectral_channels.append(channel[: base_audio.shape[0]].astype(np.float32))
        restore_weight_means.append(float(np.mean(restore_weight)))
    spectral_target = np.stack(spectral_channels, axis=1)
    sample_gate = build_sample_gate(
        base_audio.shape[0],
        sample_rate=sample_rate,
        windows=windows,
        crossfade_ms=crossfade_ms,
    )[:, None]
    rendered = base_audio * (1.0 - sample_gate) + spectral_target * sample_gate
    peak = float(np.max(np.abs(rendered), initial=0.0))
    active_samples = np.broadcast_to(sample_gate > 0.0, rendered.shape)
    clipped_samples = active_samples & (np.abs(rendered) > 0.999)
    clipped_ratio = float(np.mean(clipped_samples))
    rendered[active_samples] = np.clip(rendered[active_samples], -0.999, 0.999)
    rendered = rendered.astype(np.float32)
    return rendered, {
        "restore_weight_mean": round(float(np.mean(restore_weight_means)), 6),
        "pre_clip_peak": round(peak, 6),
        "clipped_sample_ratio": round(clipped_ratio, 9),
    }


def _proxy_records(
    quality_manifest: Mapping[str, Any],
    *,
    runtime_project_root: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    seen_audio: set[str] = set()
    for proxy in quality_manifest.get("proxies", []):
        if not isinstance(proxy, Mapping) or proxy.get("valid") is not True:
            continue
        model_id = str(proxy.get("model_id") or "")
        raw_path = str(proxy.get("audio_path") or "")
        if not model_id or not raw_path:
            continue
        path = _validate_inside_project(Path(raw_path), runtime_project_root, f"proxy:{model_id}")
        if not path.is_file():
            raise FileNotFoundError(path)
        audio_sha = sha256_file(path)
        if str(proxy.get("audio_sha256") or "") != audio_sha:
            raise ValueError(f"proxy audio sha256 mismatch: {model_id}")
        if model_id in seen_models or audio_sha in seen_audio:
            raise ValueError(f"proxy independence violation: {model_id}")
        seen_models.add(model_id)
        seen_audio.add(audio_sha)
        records.append({"model_id": model_id, "path": path, "sha256": audio_sha})
    if len(records) < 3:
        raise ValueError("At least three independent valid vocal proxies are required")
    return records


def _windows_path(path: Path, runtime_project_root: Path, windows_project_root: str) -> str:
    relative = path.resolve().relative_to(runtime_project_root.resolve())
    return str(PureWindowsPath(windows_project_root).joinpath(*relative.parts))


def build_spectral_minus_matrix(
    *,
    source_audio_path: Path,
    base_minus_path: Path,
    alternate_minus_path: Path,
    base_quality_manifest_path: Path,
    alternate_quality_manifest_path: Path,
    output_dir: Path,
    runtime_project_root: Path,
    windows_project_root: str,
    restore_strengths: Sequence[float] = DEFAULT_RESTORE_STRENGTHS,
    mask_powers: Sequence[float] = DEFAULT_MASK_POWERS,
    consensus_quantile: float = 0.5,
    min_proxy_scale: float = 0.25,
    max_proxy_scale: float = 4.0,
    min_direct_vote_reduction: int = 1,
    max_alternate_delta_rms_ratio: float = 0.35,
    crossfade_ms: int = 80,
) -> dict[str, Any]:
    runtime_root = runtime_project_root.resolve()
    source_path = _validate_inside_project(source_audio_path, runtime_root, "source_audio_path")
    base_path = _validate_inside_project(base_minus_path, runtime_root, "base_minus_path")
    alternate_path = _validate_inside_project(alternate_minus_path, runtime_root, "alternate_minus_path")
    base_manifest_path = _validate_inside_project(base_quality_manifest_path, runtime_root, "base_quality_manifest")
    alternate_manifest_path = _validate_inside_project(
        alternate_quality_manifest_path,
        runtime_root,
        "alternate_quality_manifest",
    )
    output_path = _validate_inside_project(output_dir, runtime_root, "output_dir")
    relative_output = output_path.relative_to(runtime_root)
    if not relative_output.parts or relative_output.parts[0] not in {"artifacts", "lab_artifacts", "artifact_rag"}:
        raise ValueError("output_dir must be inside an approved project artifact root")
    for path in (source_path, base_path, alternate_path, base_manifest_path, alternate_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not 0.0 <= consensus_quantile <= 1.0:
        raise ValueError("consensus_quantile must be in [0, 1]")

    output_path.mkdir(parents=True, exist_ok=True)
    progress_path = output_path / "progress.jsonl"
    manifest_path = output_path / "manifest.json"
    rows_path = output_path / "rows.jsonl"
    progress = ProgressWriter(progress_path)
    probe = _mount_probe(output_path)
    progress.mark("start", expected_candidate_count=len(restore_strengths) * len(mask_powers))

    base_manifest = read_json(base_manifest_path)
    alternate_manifest = read_json(alternate_manifest_path)
    _validate_quality_manifest(base_manifest_path, base_manifest, source_audio_path=source_path, candidate_path=base_path)
    _validate_quality_manifest(
        alternate_manifest_path,
        alternate_manifest,
        source_audio_path=source_path,
        candidate_path=alternate_path,
    )
    if _model_ids(base_manifest) != _model_ids(alternate_manifest):
        raise ValueError("quality manifests must use the same independent proxy model set")
    selected_windows, rejected_windows = select_repair_windows(
        read_jsonl(_manifest_rows_path(base_manifest_path, base_manifest)),
        read_jsonl(_manifest_rows_path(alternate_manifest_path, alternate_manifest)),
        min_direct_vote_reduction=min_direct_vote_reduction,
        max_alternate_delta_rms_ratio=max_alternate_delta_rms_ratio,
    )
    if not selected_windows:
        raise RuntimeError("no_quality_supported_repair_windows")
    proxy_records = _proxy_records(base_manifest, runtime_project_root=runtime_root)
    progress.mark(
        "inputs_validated",
        selected_window_count=len(selected_windows),
        proxy_count=len(proxy_records),
    )

    base_audio, sample_rate = read_audio(str(base_path))
    alternate_audio = load_like_base(
        str(alternate_path),
        base_rate=sample_rate,
        base_shape=(base_audio.shape[0], base_audio.shape[1]),
    )
    proxy_audios = [
        load_like_base(
            str(record["path"]),
            base_rate=sample_rate,
            base_shape=(base_audio.shape[0], base_audio.shape[1]),
        )
        for record in proxy_records
    ]
    components, spectral_evidence = prepare_spectral_components(
        base_audio,
        alternate_audio,
        proxy_audios,
        sample_rate=sample_rate,
        windows=selected_windows,
        consensus_quantile=consensus_quantile,
        min_proxy_scale=min_proxy_scale,
        max_proxy_scale=max_proxy_scale,
    )
    progress.mark("spectral_components_ready", channel_count=len(components))

    audio_dir = output_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows: list[dict[str, Any]] = []
    candidate_index = 0
    for mask_power in mask_powers:
        for restore_strength in restore_strengths:
            candidate_index += 1
            candidate_id = f"s{candidate_index:03d}"
            progress.mark(
                "candidate_start",
                candidate_id=candidate_id,
                restore_strength=restore_strength,
                mask_power=mask_power,
            )
            rendered, render_metrics = render_spectral_candidate(
                base_audio,
                components,
                sample_rate=sample_rate,
                windows=selected_windows,
                restore_strength=float(restore_strength),
                mask_power=float(mask_power),
                crossfade_ms=crossfade_ms,
            )
            candidate_path = audio_dir / f"{candidate_id}.flac"
            sf.write(str(candidate_path), rendered, sample_rate, subtype="PCM_24")
            candidate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "path": str(candidate_path),
                    "windows_path": _windows_path(candidate_path, runtime_root, windows_project_root),
                    "sha256": sha256_file(candidate_path),
                    "restore_strength": float(restore_strength),
                    "mask_power": float(mask_power),
                    "render_metrics": render_metrics,
                    "delivery_allowed": False,
                    "presentation_status": "diagnostic_candidate_not_for_listen_pack",
                }
            )
            progress.mark("candidate_finish", candidate_id=candidate_id, **render_metrics)
    write_jsonl(rows_path, candidate_rows)

    manifest = {
        "schema": SCHEMA,
        "mode": "quality_guided_spectral_non_vocal_detail_restore_matrix",
        "source": {"path": str(source_path), "sha256": sha256_file(source_path)},
        "base_minus": {"path": str(base_path), "sha256": sha256_file(base_path)},
        "alternate_minus": {"path": str(alternate_path), "sha256": sha256_file(alternate_path)},
        "quality_evidence": {
            "base_manifest": str(base_manifest_path),
            "base_manifest_sha256": sha256_file(base_manifest_path),
            "alternate_manifest": str(alternate_manifest_path),
            "alternate_manifest_sha256": sha256_file(alternate_manifest_path),
            "selected_windows": selected_windows,
            "rejected_windows": rejected_windows,
        },
        "vocal_proxy_ensemble": [
            {"model_id": record["model_id"], "path": str(record["path"]), "sha256": record["sha256"]}
            for record in proxy_records
        ],
        "spectral_evidence": spectral_evidence,
        "matrix": {
            "restore_strengths": [float(value) for value in restore_strengths],
            "mask_powers": [float(value) for value in mask_powers],
            "candidate_count": len(candidate_rows),
            "candidates": candidate_rows,
        },
        "parameters": {
            "consensus_quantile": consensus_quantile,
            "min_proxy_scale": min_proxy_scale,
            "max_proxy_scale": max_proxy_scale,
            "crossfade_ms": crossfade_ms,
            "per_track_timestamps_used": False,
        },
        "execution_contract": {
            "runtime_environment": "apple_container",
            "expected_acceleration_mode": "cpu_dsp_spectral_consensus_restore",
            "gpu_inference_skipped": True,
            "gpu_inference_skip_reason": "repair_consumes_precomputed_mps_model_outputs",
            "fallback_used": False,
            "dedicated_minus_baseline_required": True,
            "alternate_role": "diagnostic_vocal_residual_repair_only",
            "non_vocal_stem_sum_allowed": False,
        },
        "mount_contract": {
            **probe,
            "runtime_project_root": str(runtime_root),
            "windows_project_root": windows_project_root,
            "runtime_output_path": str(output_path),
            "windows_output_path": _windows_path(output_path, runtime_root, windows_project_root),
        },
        "artifacts": {"manifest": str(manifest_path), "rows": str(rows_path), "progress": str(progress_path)},
        "delivery_contract": {
            "delivery_allowed": False,
            "delivery_status": "candidate_matrix_only",
            "presentation_status": "diagnostic_candidates_not_for_listen_pack",
            "listen_pack_allowed": False,
            "required_next_gates": ["five_proxy_quality_audit", "content_word_audit", "exact_reference_or_policy_gate"],
        },
    }
    write_json(manifest_path, manifest)
    progress.mark("finish", candidate_count=len(candidate_rows), manifest_path=str(manifest_path))
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_audio")
    parser.add_argument("base_minus")
    parser.add_argument("alternate_minus")
    parser.add_argument("base_quality_manifest")
    parser.add_argument("alternate_quality_manifest")
    parser.add_argument("output_dir")
    parser.add_argument("--runtime-project-root", required=True)
    parser.add_argument("--windows-project-root", required=True)
    parser.add_argument("--restore-strength", action="append", type=float)
    parser.add_argument("--mask-power", action="append", type=float)
    parser.add_argument("--consensus-quantile", type=float, default=0.5)
    parser.add_argument("--min-proxy-scale", type=float, default=0.25)
    parser.add_argument("--max-proxy-scale", type=float, default=4.0)
    parser.add_argument("--min-direct-vote-reduction", type=int, default=1)
    parser.add_argument("--max-alternate-delta-rms-ratio", type=float, default=0.35)
    parser.add_argument("--crossfade-ms", type=int, default=80)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_spectral_minus_matrix.py",
        entrypoint_path="/app/tools/build_spectral_minus_matrix.py",
    )
    args = parse_args(argv)
    manifest = build_spectral_minus_matrix(
        source_audio_path=Path(args.source_audio),
        base_minus_path=Path(args.base_minus),
        alternate_minus_path=Path(args.alternate_minus),
        base_quality_manifest_path=Path(args.base_quality_manifest),
        alternate_quality_manifest_path=Path(args.alternate_quality_manifest),
        output_dir=Path(args.output_dir),
        runtime_project_root=Path(args.runtime_project_root),
        windows_project_root=str(args.windows_project_root),
        restore_strengths=tuple(args.restore_strength or DEFAULT_RESTORE_STRENGTHS),
        mask_powers=tuple(args.mask_power or DEFAULT_MASK_POWERS),
        consensus_quantile=float(args.consensus_quantile),
        min_proxy_scale=float(args.min_proxy_scale),
        max_proxy_scale=float(args.max_proxy_scale),
        min_direct_vote_reduction=int(args.min_direct_vote_reduction),
        max_alternate_delta_rms_ratio=float(args.max_alternate_delta_rms_ratio),
        crossfade_ms=int(args.crossfade_ms),
    )
    print(
        json.dumps(
            {
                "manifest_path": manifest["artifacts"]["manifest"],
                "candidate_count": manifest["matrix"]["candidate_count"],
                "selected_window_count": len(manifest["quality_evidence"]["selected_windows"]),
                "delivery_allowed": manifest["delivery_contract"]["delivery_allowed"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
