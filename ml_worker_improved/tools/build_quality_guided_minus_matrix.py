"""Build a fail-closed segment-repair matrix from two minus quality manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TOOLS_ROOT.parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402
from tools.render_no_reference_full_track_segment_switch_candidate import (  # noqa: E402
    load_like_base,
    read_audio,
)


SCHEMA = "quality_guided_minus_matrix.v1"
VOCAL_ISSUE = "vocal_leak_proxy_consensus"
DETAIL_ISSUES = {
    "low_rms_muffling_proxy",
    "presence_loss_proxy",
    "air_loss_proxy",
}
DEFAULT_BLEND_VALUES = (0.5, 0.75, 1.0)
DEFAULT_GAIN_MODES = ("none", "window_rms_match")
EPSILON = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


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
            "expected_acceleration_mode": "cpu_dsp_quality_guided_segment_repair",
            "gpu_inference_skipped": True,
            "gpu_inference_skip_reason": "repair_consumes_precomputed_mps_model_outputs",
        }
        row.update(fields)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()


def _manifest_rows_path(manifest_path: Path, manifest: Mapping[str, Any]) -> Path:
    raw_path = manifest.get("artifacts", {}).get("rows") if isinstance(manifest.get("artifacts"), Mapping) else None
    if isinstance(raw_path, str) and raw_path:
        direct = Path(raw_path)
        if direct.is_file():
            return direct
        sibling = manifest_path.parent / Path(raw_path.replace("\\", "/")).name
        if sibling.is_file():
            return sibling
    raise FileNotFoundError(f"Quality rows are missing for {manifest_path}")


def _model_ids(manifest: Mapping[str, Any]) -> list[str]:
    return sorted(
        str(proxy.get("model_id") or "")
        for proxy in manifest.get("proxies", [])
        if isinstance(proxy, Mapping) and proxy.get("valid") is True and proxy.get("model_id")
    )


def _validate_quality_manifest(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    source_audio_path: Path,
    candidate_path: Path,
) -> None:
    if str(manifest.get("schema") or "") != "minus_quality_ensemble.v2":
        raise ValueError(f"Unsupported quality manifest schema: {manifest_path}")
    expected_source_sha = str(manifest.get("source", {}).get("sha256") or "")
    expected_candidate_sha = str(manifest.get("candidate_minus", {}).get("sha256") or "")
    if expected_source_sha != sha256_file(source_audio_path):
        raise ValueError(f"source sha256 mismatch: {manifest_path}")
    if expected_candidate_sha != sha256_file(candidate_path):
        raise ValueError(f"candidate sha256 mismatch: {manifest_path}")


def _row_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        segment_id = str(row.get("segment_id") or "")
        if not segment_id or segment_id in result:
            raise ValueError(f"Quality rows require unique non-empty segment_id: {segment_id!r}")
        result[segment_id] = row
    return result


def _window(row: Mapping[str, Any]) -> tuple[float, float]:
    raw = row.get("window_sec")
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"Invalid quality window: {raw!r}")
    start = float(raw[0])
    end = float(raw[1])
    if start < 0.0 or end <= start:
        raise ValueError(f"Invalid quality window: {raw!r}")
    return start, end


def _direct_vote_count(row: Mapping[str, Any]) -> int:
    votes = row.get("votes")
    direct = votes.get("direct_vocal") if isinstance(votes, Mapping) else None
    return len(direct) if isinstance(direct, list) else 0


def _confirmed_issues(row: Mapping[str, Any]) -> set[str]:
    raw = row.get("confirmed_issues")
    return {str(issue) for issue in raw} if isinstance(raw, list) else set()


def select_repair_windows(
    base_rows: Sequence[Mapping[str, Any]],
    alternate_rows: Sequence[Mapping[str, Any]],
    *,
    min_direct_vote_reduction: int,
    max_alternate_delta_rms_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    alternate_by_id = _row_index(alternate_rows)
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for base_row in base_rows:
        segment_id = str(base_row.get("segment_id") or "")
        base_issues = _confirmed_issues(base_row)
        if VOCAL_ISSUE not in base_issues:
            continue
        alternate_row = alternate_by_id.get(segment_id)
        reasons: list[str] = []
        if alternate_row is None:
            reasons.append("alternate_segment_missing")
        else:
            if _window(base_row) != _window(alternate_row):
                reasons.append("window_mismatch")
            alternate_issues = _confirmed_issues(alternate_row)
            if alternate_issues.intersection(DETAIL_ISSUES):
                reasons.append("alternate_has_confirmed_detail_damage")
            base_votes = _direct_vote_count(base_row)
            alternate_votes = _direct_vote_count(alternate_row)
            if alternate_votes > base_votes - min_direct_vote_reduction:
                reasons.append("insufficient_direct_vocal_vote_reduction")
            source_conservation = alternate_row.get("source_conservation")
            delta_ratio = (
                float(source_conservation.get("source_candidate_delta_rms_ratio") or 0.0)
                if isinstance(source_conservation, Mapping)
                else math.inf
            )
            if delta_ratio > max_alternate_delta_rms_ratio:
                reasons.append("alternate_delta_rms_ratio_too_high")
        start, end = _window(base_row)
        record = {
            "segment_id": segment_id,
            "window_sec": [start, end],
            "base_direct_vocal_votes": _direct_vote_count(base_row),
            "alternate_direct_vocal_votes": _direct_vote_count(alternate_row or {}),
            "base_confirmed_issues": sorted(base_issues),
            "alternate_confirmed_issues": sorted(_confirmed_issues(alternate_row or {})),
        }
        if reasons:
            rejected.append({**record, "rejection_reasons": reasons})
        else:
            selected.append(record)
    return selected, rejected


def _window_gain(
    base_segment: np.ndarray,
    alternate_segment: np.ndarray,
    *,
    mode: str,
    max_gain_db: float,
) -> tuple[float, dict[str, float]]:
    base_rms = float(np.sqrt(np.mean(np.square(base_segment), dtype=np.float64)) + EPSILON)
    alternate_rms = float(np.sqrt(np.mean(np.square(alternate_segment), dtype=np.float64)) + EPSILON)
    requested_gain = 1.0 if mode == "none" else base_rms / alternate_rms
    max_gain = 10.0 ** (max_gain_db / 20.0)
    requested_gain = min(requested_gain, max_gain)
    alternate_peak = float(np.max(np.abs(alternate_segment), initial=0.0))
    peak_limited_gain = min(requested_gain, 0.999 / max(alternate_peak, EPSILON))
    applied_gain = max(0.0, peak_limited_gain)
    return applied_gain, {
        "base_rms": base_rms,
        "alternate_rms": alternate_rms,
        "requested_gain_db": round(20.0 * math.log10(max(requested_gain, EPSILON)), 6),
        "applied_gain_db": round(20.0 * math.log10(max(applied_gain, EPSILON)), 6),
        "alternate_peak": alternate_peak,
        "peak_gain_limited": applied_gain + 1e-9 < requested_gain,
    }


def render_candidate(
    base_audio: np.ndarray,
    alternate_audio: np.ndarray,
    *,
    sample_rate: int,
    windows: Sequence[Mapping[str, Any]],
    blend: float,
    gain_mode: str,
    max_gain_db: float,
    crossfade_ms: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not 0.0 < blend <= 1.0:
        raise ValueError("blend must be in (0, 1]")
    if gain_mode not in DEFAULT_GAIN_MODES:
        raise ValueError(f"Unsupported gain mode: {gain_mode}")
    rendered = np.asarray(base_audio, dtype=np.float32).copy()
    alternate = np.asarray(alternate_audio, dtype=np.float32)
    window_records: list[dict[str, Any]] = []
    for window in windows:
        start_sec, end_sec = _window(window)
        start = max(0, min(rendered.shape[0], int(round(start_sec * sample_rate))))
        end = max(start, min(rendered.shape[0], int(round(end_sec * sample_rate))))
        if end <= start:
            continue
        base_segment = rendered[start:end].copy()
        alternate_segment = alternate[start:end]
        gain, gain_record = _window_gain(
            base_segment,
            alternate_segment,
            mode=gain_mode,
            max_gain_db=max_gain_db,
        )
        target = base_segment * (1.0 - blend) + alternate_segment * gain * blend
        weights = np.ones((end - start, 1), dtype=np.float32)
        fade_frames = min(int(round(sample_rate * crossfade_ms / 1000.0)), (end - start) // 2)
        if fade_frames > 0 and start > 0:
            weights[:fade_frames, 0] = np.linspace(0.0, 1.0, fade_frames, endpoint=False, dtype=np.float32)
        if fade_frames > 0 and end < rendered.shape[0]:
            weights[-fade_frames:, 0] = np.linspace(1.0, 0.0, fade_frames, endpoint=True, dtype=np.float32)
        rendered[start:end] = base_segment * (1.0 - weights) + target * weights
        window_records.append(
            {
                "segment_id": str(window.get("segment_id") or ""),
                "window_sec": [start_sec, end_sec],
                "blend": blend,
                "gain_mode": gain_mode,
                **gain_record,
            }
        )
    return rendered, window_records


def _validate_inside_project(path: Path, runtime_project_root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(runtime_project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must be inside runtime_project_root") from exc
    return resolved


def _windows_path(path: Path, runtime_project_root: Path, windows_project_root: str) -> str:
    relative = path.resolve().relative_to(runtime_project_root.resolve())
    return str(PureWindowsPath(windows_project_root).joinpath(*relative.parts))


def _mount_probe(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "probe.tmp"
    token = f"quality-guided-minus:{time.time_ns()}"
    path.write_text(token, encoding="utf-8")
    if path.read_text(encoding="utf-8") != token:
        raise RuntimeError(f"Artifact mount probe mismatch: {path}")
    path.unlink()
    return {"write_read_verified": True, "cleanup_verified": not path.exists(), "probe_path": str(path)}


def build_quality_guided_minus_matrix(
    *,
    source_audio_path: Path,
    base_minus_path: Path,
    alternate_minus_path: Path,
    base_quality_manifest_path: Path,
    alternate_quality_manifest_path: Path,
    output_dir: Path,
    runtime_project_root: Path,
    windows_project_root: str,
    blends: Sequence[float] = DEFAULT_BLEND_VALUES,
    gain_modes: Sequence[str] = DEFAULT_GAIN_MODES,
    min_direct_vote_reduction: int = 1,
    max_alternate_delta_rms_ratio: float = 0.35,
    max_gain_db: float = 3.0,
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
    if len(set(float(value) for value in blends)) != len(blends):
        raise ValueError("blend values must be unique")
    if len(set(gain_modes)) != len(gain_modes):
        raise ValueError("gain modes must be unique")

    output_path.mkdir(parents=True, exist_ok=True)
    progress_path = output_path / "progress.jsonl"
    manifest_path = output_path / "manifest.json"
    rows_path = output_path / "rows.jsonl"
    progress = ProgressWriter(progress_path)
    probe = _mount_probe(output_path)
    progress.mark("start", expected_candidate_count=len(blends) * len(gain_modes))

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
    base_rows = read_jsonl(_manifest_rows_path(base_manifest_path, base_manifest))
    alternate_rows = read_jsonl(_manifest_rows_path(alternate_manifest_path, alternate_manifest))
    selected_windows, rejected_windows = select_repair_windows(
        base_rows,
        alternate_rows,
        min_direct_vote_reduction=min_direct_vote_reduction,
        max_alternate_delta_rms_ratio=max_alternate_delta_rms_ratio,
    )
    progress.mark(
        "select_windows",
        selected_window_count=len(selected_windows),
        rejected_window_count=len(rejected_windows),
    )
    if not selected_windows:
        raise RuntimeError("no_quality_supported_repair_windows")

    base_audio, sample_rate = read_audio(str(base_path))
    alternate_audio = load_like_base(
        str(alternate_path),
        base_rate=sample_rate,
        base_shape=(base_audio.shape[0], base_audio.shape[1]),
    )
    audio_dir = output_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows: list[dict[str, Any]] = []
    candidate_index = 0
    for gain_mode in gain_modes:
        for blend in blends:
            candidate_index += 1
            candidate_id = f"c{candidate_index:03d}"
            progress.mark("candidate_start", candidate_id=candidate_id, blend=blend, gain_mode=gain_mode)
            rendered, window_records = render_candidate(
                base_audio,
                alternate_audio,
                sample_rate=sample_rate,
                windows=selected_windows,
                blend=float(blend),
                gain_mode=str(gain_mode),
                max_gain_db=max_gain_db,
                crossfade_ms=crossfade_ms,
            )
            candidate_path = audio_dir / f"{candidate_id}.flac"
            sf.write(str(candidate_path), rendered, sample_rate, subtype="PCM_24")
            row = {
                "candidate_id": candidate_id,
                "path": str(candidate_path),
                "windows_path": _windows_path(candidate_path, runtime_root, windows_project_root),
                "sha256": sha256_file(candidate_path),
                "blend": float(blend),
                "gain_mode": str(gain_mode),
                "window_records": window_records,
                "delivery_allowed": False,
                "presentation_status": "diagnostic_candidate_not_for_listen_pack",
            }
            candidate_rows.append(row)
            progress.mark("candidate_finish", candidate_id=candidate_id, sha256=row["sha256"])

    write_jsonl(rows_path, candidate_rows)
    manifest = {
        "schema": SCHEMA,
        "mode": "quality_manifest_guided_segment_repair_matrix",
        "source": {"path": str(source_path), "sha256": sha256_file(source_path)},
        "base_minus": {"path": str(base_path), "sha256": sha256_file(base_path)},
        "alternate_minus": {"path": str(alternate_path), "sha256": sha256_file(alternate_path)},
        "quality_evidence": {
            "base_manifest": str(base_manifest_path),
            "base_manifest_sha256": sha256_file(base_manifest_path),
            "alternate_manifest": str(alternate_manifest_path),
            "alternate_manifest_sha256": sha256_file(alternate_manifest_path),
            "proxy_model_ids": _model_ids(base_manifest),
            "selected_windows": selected_windows,
            "rejected_windows": rejected_windows,
        },
        "matrix": {
            "blend_values": [float(value) for value in blends],
            "gain_modes": list(gain_modes),
            "candidate_count": len(candidate_rows),
            "candidates": candidate_rows,
        },
        "parameters": {
            "min_direct_vote_reduction": min_direct_vote_reduction,
            "max_alternate_delta_rms_ratio": max_alternate_delta_rms_ratio,
            "max_gain_db": max_gain_db,
            "crossfade_ms": crossfade_ms,
            "per_track_timestamps_used": False,
        },
        "execution_contract": {
            "runtime_environment": "apple_container",
            "expected_acceleration_mode": "cpu_dsp_quality_guided_segment_repair",
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
    parser.add_argument("--blend", action="append", type=float)
    parser.add_argument("--gain-mode", action="append", choices=DEFAULT_GAIN_MODES)
    parser.add_argument("--min-direct-vote-reduction", type=int, default=1)
    parser.add_argument("--max-alternate-delta-rms-ratio", type=float, default=0.35)
    parser.add_argument("--max-gain-db", type=float, default=3.0)
    parser.add_argument("--crossfade-ms", type=int, default=80)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_quality_guided_minus_matrix.py",
        entrypoint_path="/app/tools/build_quality_guided_minus_matrix.py",
    )
    args = parse_args(argv)
    manifest = build_quality_guided_minus_matrix(
        source_audio_path=Path(args.source_audio),
        base_minus_path=Path(args.base_minus),
        alternate_minus_path=Path(args.alternate_minus),
        base_quality_manifest_path=Path(args.base_quality_manifest),
        alternate_quality_manifest_path=Path(args.alternate_quality_manifest),
        output_dir=Path(args.output_dir),
        runtime_project_root=Path(args.runtime_project_root),
        windows_project_root=str(args.windows_project_root),
        blends=tuple(args.blend or DEFAULT_BLEND_VALUES),
        gain_modes=tuple(args.gain_mode or DEFAULT_GAIN_MODES),
        min_direct_vote_reduction=int(args.min_direct_vote_reduction),
        max_alternate_delta_rms_ratio=float(args.max_alternate_delta_rms_ratio),
        max_gain_db=float(args.max_gain_db),
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
