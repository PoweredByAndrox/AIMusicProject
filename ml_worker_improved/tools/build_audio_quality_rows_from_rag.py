from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.audio_quality_critic_common import (
    PROGRESS_FILE,
    SAFETY_FLAGS,
    append_progress,
    as_float,
    as_list,
    iter_jsonl,
    normalize_stem_type,
    read_json,
    split_for_track,
    stable_hash,
    write_json,
    write_jsonl,
)
from tools.docker_runtime import require_docker_runtime


ROWS_FILE = "rows.jsonl"
BLOCKED_FILE = "blocked.jsonl"
MANIFEST_FILE = "manifest.json"
TENSOR_DIR = "tensors"
MANIFEST_SCHEMA = "audio_quality_rag_rows_manifest.v1"
NO_REFERENCE_GATE_SCHEMA = "aris_top_route_no_reference_gate_manifest.v1"
NO_REFERENCE_GATE_TYPE = "aris_top_route_no_reference_gate"
NUMERIC_FEATURE_TERMS = (
    "score",
    "ratio",
    "confidence",
    "stability",
    "integrity",
    "preservation",
    "balance",
    "similarity",
    "damage",
    "leakage",
    "phase",
)


def _unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _rag_idx_dir(rag_current_json: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    current = read_json(rag_current_json)
    idx_rel = str(current.get("idx") or "")
    if not idx_rel:
        raise ValueError("rag_current_missing_idx")
    idx_path = rag_current_json.parent / idx_rel
    idx = read_json(idx_path)
    return current, idx_path.parent, idx


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(iter_jsonl(path))


def _load_roots(idx_dir: Path, idx: dict[str, Any]) -> dict[str, dict[str, Any]]:
    outputs = idx.get("outputs") if isinstance(idx.get("outputs"), dict) else {}
    roots_file = str(outputs.get("roots") or "roots.jsonl")
    roots = _load_jsonl(idx_dir / roots_file)
    return {str(row.get("root_alias") or ""): row for row in roots if row.get("root_alias")}


def _load_manifest_refs(idx_dir: Path, idx: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = idx.get("outputs") if isinstance(idx.get("outputs"), dict) else {}
    refs_file = str(outputs.get("manifest_refs") or "manifest_refs.jsonl")
    return _load_jsonl(idx_dir / refs_file)


def _gate_root_aliases(manifest_refs: Sequence[dict[str, Any]]) -> list[str]:
    aliases: list[str] = []
    for ref in manifest_refs:
        status = ref.get("status_fields") if isinstance(ref.get("status_fields"), dict) else {}
        if str(ref.get("relative_path") or "") != "manifest.json":
            continue
        if status.get("schema") == NO_REFERENCE_GATE_SCHEMA or status.get("manifest_type") == NO_REFERENCE_GATE_TYPE:
            aliases.append(str(ref.get("root_alias") or ""))
    return _unique(aliases)


def _resolve_gate_dir(root: dict[str, Any], rag_current_json: Path) -> Path:
    original = Path(str(root.get("original_path") or ""))
    if original.exists():
        return original
    requested = str(root.get("requested") or "")
    if requested:
        candidate = rag_current_json.parent / requested
        if candidate.exists():
            return candidate
    return original


def _refs_for_root(manifest_refs: Sequence[dict[str, Any]], root_alias: str) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for ref in manifest_refs:
        if str(ref.get("root_alias") or "") != root_alias:
            continue
        refs.append(
            {
                "rag_id": "",
                "root": str(ref.get("root") or ""),
                "root_alias": root_alias,
                "chunk_id": str(ref.get("chunk_id") or ""),
                "relative_path": str(ref.get("relative_path") or ""),
                "sha256": str(ref.get("sha256") or ""),
                "zip_member": str(ref.get("zip_member") or ""),
            }
        )
    return refs


def _flatten_numeric_features(payload: Any, *, prefix: str = "") -> dict[str, float]:
    features: dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key)
            next_prefix = f"{prefix}.{key_text}" if prefix else key_text
            if "threshold" in next_prefix.lower():
                continue
            if isinstance(value, (dict, list)):
                features.update(_flatten_numeric_features(value, prefix=next_prefix))
            elif isinstance(value, bool):
                continue
            elif isinstance(value, (int, float)) and any(term in next_prefix.lower() for term in NUMERIC_FEATURE_TERMS):
                number = float(value)
                if math.isfinite(number):
                    features[next_prefix] = round(number, 8)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            features.update(_flatten_numeric_features(value, prefix=f"{prefix}.{index}" if prefix else str(index)))
    return features


def _judge_payload(row: dict[str, Any], name: str) -> dict[str, Any]:
    payload = row.get(name)
    return payload if isinstance(payload, dict) else {}


def _path_from_judges(row: dict[str, Any], *names: str) -> str:
    for judge_name in ("pair_judge", "minus_judge", "vocal_judge"):
        judge = _judge_payload(row, judge_name)
        for name in names:
            value = str(judge.get(name) or "")
            if value:
                return value
    for name in names:
        value = str(row.get(name) or "")
        if value:
            return value
    return ""


def _collect_blockers(row: dict[str, Any]) -> list[str]:
    blockers: list[Any] = []
    blockers.extend(as_list(row.get("failed_checks")))
    if str(row.get("no_reference_gate") or "") != "passed":
        blockers.append("no_reference_gate_failed")
    if str(row.get("pair_consistency_gate") or "") != "passed":
        blockers.append("pair_consistency_gate_failed")
    for judge_name in ("minus_judge", "vocal_judge", "pair_judge"):
        judge = _judge_payload(row, judge_name)
        blockers.extend(as_list(judge.get("minus_blockers")))
        blockers.extend(as_list(judge.get("vocal_blockers")))
        blockers.extend(as_list(judge.get("pair_blockers")))
        gate = judge.get("minus_musical_detail_preservation_gate")
        if isinstance(gate, dict):
            for item in as_list(gate.get("blockers")):
                if isinstance(item, dict):
                    blockers.append(item.get("label") or item.get("reason"))
                else:
                    blockers.append(item)
    return _unique(blockers)


def _issue_tags(row: dict[str, Any], blockers: Sequence[str]) -> list[str]:
    tags: list[Any] = list(blockers)
    for judge_name in ("minus_judge", "vocal_judge"):
        judge = _judge_payload(row, judge_name)
        tags.append(judge.get("minus_dominant_defect"))
        tags.append(judge.get("vocal_dominant_defect"))
    if str(row.get("no_reference_gate") or "") == "failed":
        tags.append("no_reference_gate_failed")
    if str(row.get("pair_consistency_gate") or "") == "failed":
        tags.append("pair_consistency_gate_failed")
    return _unique(tags)


def _proxy_decision(row: dict[str, Any], accepted: bool) -> str:
    pair = _judge_payload(row, "pair_judge")
    minus = _judge_payload(row, "minus_judge")
    for key in ("pair_decision", "decision"):
        value = str(pair.get(key) or "")
        if value:
            return value
    for key in ("minus_decision", "decision"):
        value = str(minus.get(key) or "")
        if value:
            return value
    return "accept_candidate" if accepted else "blocked_hard_case"


def _targets(row: dict[str, Any], accepted: bool) -> dict[str, float]:
    minus = _judge_payload(row, "minus_judge")
    vocal = _judge_payload(row, "vocal_judge")
    pair = _judge_payload(row, "pair_judge")
    confidence = max(as_float(minus.get("confidence")), as_float(vocal.get("confidence")), as_float(pair.get("confidence")))
    phase = max(as_float(minus.get("phase_consistency_score")), as_float(pair.get("pair_phase_consistency_score")))
    instrumental_integrity = as_float((minus.get("minus_score_components") or {}).get("instrumental_integrity"), 1.0)
    return {
        "accept_probability": 1.0 if accepted else 0.0,
        "vocal_bleed_in_minus": as_float(minus.get("vocal_leakage_in_minus_score")),
        "instrumental_bleed_in_vocal": as_float(vocal.get("instrumental_leakage_in_vocal_score")),
        "vocal_hallucination": as_float(vocal.get("silence_hallucination_score") or minus.get("silence_hallucination_score")),
        "robotic_artifact": as_float(vocal.get("robotic_artifact_score") or minus.get("robotic_artifact_score")),
        "thin_vocal": as_float(vocal.get("thin_vocal_score") or minus.get("thin_vocal_score")),
        "music_damage": round(max(0.0, 1.0 - instrumental_integrity), 6),
        "transient_damage": as_float(minus.get("transient_damage_score")),
        "phase_stereo_collapse": round(max(0.0, 1.0 - phase), 6),
        "uncertainty": round(max(0.0, 1.0 - confidence), 6),
    }


def _envelope(samples: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray, float]:
    if samples.ndim == 1:
        samples = samples[:, None]
    mono = samples.astype(np.float32).mean(axis=1)
    if mono.size == 0:
        return np.zeros((bins,), dtype=np.float32), np.zeros((bins,), dtype=np.float32), 0.0
    chunks = np.array_split(mono, bins)
    rms = np.asarray([math.sqrt(float(np.mean(np.square(chunk)))) if chunk.size else 0.0 for chunk in chunks], dtype=np.float32)
    peak = np.asarray([float(np.max(np.abs(chunk))) if chunk.size else 0.0 for chunk in chunks], dtype=np.float32)
    corr = 1.0
    if samples.shape[1] >= 2 and samples.shape[0] > 1:
        left = samples[:, 0].astype(np.float32)
        right = samples[:, 1].astype(np.float32)
        if float(np.std(left)) > 1.0e-8 and float(np.std(right)) > 1.0e-8:
            corr = float(np.corrcoef(left, right)[0, 1])
    if not math.isfinite(corr):
        corr = 0.0
    return rms, peak, float(corr)


def _read_audio_envelope(path: Path, *, window_sec: float, bins: int) -> tuple[np.ndarray, np.ndarray, float]:
    import soundfile as sf

    info = sf.info(str(path))
    frames = max(1, min(int(info.frames), int(float(window_sec) * int(info.samplerate))))
    samples, _sample_rate = sf.read(str(path), frames=frames, dtype="float32", always_2d=True)
    return _envelope(np.asarray(samples, dtype=np.float32), bins)


def _write_tensor(
    *,
    row: dict[str, Any],
    output_dir: Path,
    row_index: int,
    window_sec: float,
    tensor_bins: int,
) -> tuple[str, str]:
    paths = {
        "mix": Path(str(row.get("mix_path") or "")),
        "vocal": Path(str(row.get("vocal_path") or "")),
        "minus": Path(str(row.get("minus_path") or "")),
    }
    missing = [name for name, path in paths.items() if not str(path) or not path.exists()]
    if missing:
        return "", "missing_audio_for_tensor:" + ",".join(missing)
    try:
        mix_rms, mix_peak, mix_corr = _read_audio_envelope(paths["mix"], window_sec=window_sec, bins=tensor_bins)
        vocal_rms, vocal_peak, vocal_corr = _read_audio_envelope(paths["vocal"], window_sec=window_sec, bins=tensor_bins)
        minus_rms, minus_peak, minus_corr = _read_audio_envelope(paths["minus"], window_sec=window_sec, bins=tensor_bins)
    except Exception as exc:  # noqa: BLE001
        return "", "tensor_extract_failed:" + type(exc).__name__
    tensor_path = output_dir / TENSOR_DIR / f"q{row_index:03d}.npz"
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    feature_vector = np.asarray([float(value) for _, value in sorted((row.get("features") or {}).items())], dtype=np.float32)
    np.savez_compressed(
        tensor_path,
        mix_rms=mix_rms,
        vocal_rms=vocal_rms,
        minus_rms=minus_rms,
        mix_peak=mix_peak,
        vocal_peak=vocal_peak,
        minus_peak=minus_peak,
        residual_rms_proxy=(mix_rms - vocal_rms - minus_rms).astype(np.float32),
        stereo_corr=np.asarray([mix_corr, vocal_corr, minus_corr], dtype=np.float32),
        proxy_metric_vector=feature_vector,
    )
    return str(tensor_path), ""


def _row_from_gate_judge(
    *,
    judge_row: dict[str, Any],
    gate_dir: Path,
    root_alias: str,
    rag_id: str,
    rag_refs: Sequence[dict[str, Any]],
    row_index: int,
    output_dir: Path,
    make_tensors: bool,
    window_sec: float,
    tensor_bins: int,
    source_lock_pack_json: Path | None,
) -> tuple[dict[str, Any], str]:
    blockers = _collect_blockers(judge_row)
    accepted = bool(judge_row.get("accepted") is True and not blockers and judge_row.get("no_reference_gate") == "passed")
    label_type = "weak_proxy" if accepted else "rejected_blocked"
    features = _flatten_numeric_features(judge_row)
    row = {
        "track_id": str(judge_row.get("track_id") or ""),
        "candidate_id": str(judge_row.get("candidate_id") or f"cand-{row_index:03d}"),
        "stem_type": normalize_stem_type("pair"),
        "window_start_sec": 0.0,
        "window_end_sec": float(window_sec),
        "mix_path": _path_from_judges(judge_row, "mix_path"),
        "vocal_path": _path_from_judges(judge_row, "vocal_path"),
        "minus_path": _path_from_judges(judge_row, "instrumental_path", "minus_path"),
        "tensor_path": "",
        "features": features,
        "route_id": NO_REFERENCE_GATE_TYPE,
        "label_type": label_type,
        "label": 1 if accepted else 0,
        "label_confidence": max(as_float(_judge_payload(judge_row, "minus_judge").get("confidence")), as_float(_judge_payload(judge_row, "vocal_judge").get("confidence"))),
        "split": split_for_track(str(judge_row.get("track_id") or "")),
        "source_manifest_refs": [str(gate_dir / "manifest.json"), str(gate_dir / "judges.jsonl")],
        "rag_refs": [{**ref, "rag_id": rag_id} for ref in rag_refs],
        "issue_tags": _issue_tags(judge_row, blockers),
        "blocker_reasons": blockers,
        "targets": _targets(judge_row, accepted),
        "proxy_decision": _proxy_decision(judge_row, accepted),
        "source_rag_root_alias": root_alias,
        "source_lock_pack_json": str(source_lock_pack_json or ""),
        "generated_stem_claims_ground_truth": False,
        "pseudo_teacher_targets_are_ground_truth": False,
        "delivery_allowed": False,
        "release_ready": False,
    }
    tensor_skip_reason = ""
    if make_tensors:
        tensor_path, tensor_skip_reason = _write_tensor(
            row=row,
            output_dir=output_dir,
            row_index=row_index,
            window_sec=window_sec,
            tensor_bins=tensor_bins,
        )
        row["tensor_path"] = tensor_path
    return row, tensor_skip_reason


def build_audio_quality_rows_from_rag(
    *,
    rag_current_json: Path,
    output_dir: Path,
    make_tensors: bool = True,
    window_sec: float = 6.0,
    tensor_bins: int = 64,
    max_roots: int = 0,
    source_lock_pack_json: Path | None = None,
) -> dict[str, Any]:
    start_time = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_FILE
    if progress_path.exists():
        progress_path.unlink()
    append_progress(
        progress_path,
        "start",
        start_time=start_time,
        rag_current_json=str(rag_current_json),
        make_tensors=bool(make_tensors),
        expected_acceleration_mode="cpu_audio_tensor_extraction",
        gpu_inference_skipped=True,
        gpu_inference_skip_reason="rag_row_bridge_no_model_inference",
    )
    current, idx_dir, idx = _rag_idx_dir(rag_current_json)
    roots = _load_roots(idx_dir, idx)
    manifest_refs = _load_manifest_refs(idx_dir, idx)
    aliases = _gate_root_aliases(manifest_refs)
    if max_roots > 0:
        aliases = aliases[: int(max_roots)]
    rows: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    tensor_skip_counts: dict[str, int] = {}
    for root_alias in aliases:
        root = roots.get(root_alias, {})
        gate_dir = _resolve_gate_dir(root, rag_current_json)
        manifest_json = gate_dir / "manifest.json"
        judges_jsonl = gate_dir / "judges.jsonl"
        if not manifest_json.exists() or not judges_jsonl.exists():
            reason = "missing_gate_files"
            blocked.append({"root_alias": root_alias, "gate_dir": str(gate_dir), "reason": reason, **SAFETY_FLAGS})
            continue
        gate_manifest = read_json(manifest_json)
        if gate_manifest.get("release_ready") is True or gate_manifest.get("delivery_allowed") is True:
            blocked.append({"root_alias": root_alias, "gate_dir": str(gate_dir), "reason": "unsafe_gate_manifest_flags", **SAFETY_FLAGS})
            continue
        rag_refs = _refs_for_root(manifest_refs, root_alias)
        for judge_row in iter_jsonl(judges_jsonl):
            row, tensor_skip_reason = _row_from_gate_judge(
                judge_row=judge_row,
                gate_dir=gate_dir,
                root_alias=root_alias,
                rag_id=str(current.get("rag_id") or ""),
                rag_refs=rag_refs,
                row_index=len(rows) + 1,
                output_dir=output_dir,
                make_tensors=make_tensors,
                window_sec=window_sec,
                tensor_bins=tensor_bins,
                source_lock_pack_json=source_lock_pack_json,
            )
            if tensor_skip_reason:
                tensor_skip_counts[tensor_skip_reason] = tensor_skip_counts.get(tensor_skip_reason, 0) + 1
            rows.append(row)
            append_progress(progress_path, "row", start_time=start_time, root_alias=root_alias, row_count=len(rows), label_type=row["label_type"])
    write_jsonl(output_dir / ROWS_FILE, rows)
    write_jsonl(output_dir / BLOCKED_FILE, blocked)
    label_counts: dict[str, int] = {}
    for row in rows:
        label_counts[row["label_type"]] = label_counts.get(row["label_type"], 0) + 1
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "manifest_type": "audio_quality_rag_rows",
        "rows_id": "aqr_" + stable_hash({"rag_id": current.get("rag_id"), "aliases": aliases, "window_sec": window_sec})[:12],
        "current_rag_id": str(current.get("rag_id") or ""),
        "rag_current_json": str(rag_current_json),
        "rag_idx_dir": str(idx_dir),
        "source_root_aliases": aliases,
        "source_lock_pack_json": str(source_lock_pack_json or ""),
        "row_count": len(rows),
        "tensor_row_count": sum(1 for row in rows if row.get("tensor_path")),
        "feature_only_row_count": sum(1 for row in rows if not row.get("tensor_path")),
        "blocked_source_count": len(blocked),
        "label_type_counts": dict(sorted(label_counts.items())),
        "tensor_skip_counts": dict(sorted(tensor_skip_counts.items())),
        "outputs": {
            "rows_jsonl": str(output_dir / ROWS_FILE),
            "blocked_jsonl": str(output_dir / BLOCKED_FILE),
            "manifest_json": str(output_dir / MANIFEST_FILE),
            "progress_jsonl": str(progress_path),
            "tensor_dir": str(output_dir / TENSOR_DIR),
        },
        "presentation_status": "audio_quality_rag_rows_not_for_listen_pack",
        "training_started": False,
        "promotion_allowed": False,
        "safety": {
            "label_source": "rag_no_reference_proxy_or_rejected_blocked_only",
            "generated_stems_never_ground_truth": True,
            "self_labeled_ground_truth": False,
            "short_tensor_aliases": True,
        },
        **SAFETY_FLAGS,
    }
    write_json(output_dir / MANIFEST_FILE, manifest)
    append_progress(progress_path, "finish", start_time=start_time, row_count=len(rows), tensor_row_count=manifest["tensor_row_count"])
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build audio quality critic weak rows from current RAG no-reference gate evidence.")
    parser.add_argument("--rag-current-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-tensor-extraction", action="store_true")
    parser.add_argument("--window-sec", type=float, default=6.0)
    parser.add_argument("--tensor-bins", type=int, default=64)
    parser.add_argument("--max-roots", type=int, default=0)
    parser.add_argument("--source-lock-pack-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(tool_name="build_audio_quality_rows_from_rag.py", entrypoint_path="/app/tools/build_audio_quality_rows_from_rag.py")
    args = parse_args(argv)
    manifest = build_audio_quality_rows_from_rag(
        rag_current_json=args.rag_current_json,
        output_dir=args.output_dir,
        make_tensors=not bool(args.skip_tensor_extraction),
        window_sec=float(args.window_sec),
        tensor_bins=int(args.tensor_bins),
        max_roots=int(args.max_roots),
        source_lock_pack_json=args.source_lock_pack_json,
    )
    print(json.dumps({"manifest_json": str(args.output_dir / MANIFEST_FILE), "row_count": manifest["row_count"], "tensor_row_count": manifest["tensor_row_count"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
