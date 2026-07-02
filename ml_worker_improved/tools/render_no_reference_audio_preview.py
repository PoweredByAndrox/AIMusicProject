from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


PREVIEW_STATUSES = {
    "accepted_pair_ready",
    "manifest_preview_ready",
    "audio_preview_ready",
    "preview_render_skipped",
    "preview_render_failed",
    "preview_blocked_safety_contract",
}

SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "separator_training_started": False,
    "model_downloaded": False,
    "model_registry_edited": False,
    "thresholds_relaxed": False,
    "cleanup_routes_added": False,
    "melody_transcription_ran": False,
    "recommendation_index_updated": False,
    "can_publish_as_result": False,
    "can_mark_release_ready": False,
    "release_ready": False,
}


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {value}")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return text.strip("._-")[:96] or "unknown"


def short_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]


def audio_metadata(path: Path) -> dict[str, Any]:
    info = sf.info(str(path))
    duration = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
    return {
        "duration_sec": round(duration, 6),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
    }


def is_inside(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def assert_preview_output_path(run_dir: Path, output_path: Path, source_paths: Sequence[str]) -> None:
    if not is_inside(run_dir, output_path):
        raise ValueError(f"preview_output_outside_run_dir:{output_path}")
    for source in source_paths:
        if source and output_path.resolve() == Path(source).resolve():
            raise ValueError(f"preview_would_overwrite_original:{output_path}")


def find_selected_pair(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    selected_rows = iter_jsonl(run_dir / "selected_candidates.jsonl")
    pair_rows = iter_jsonl(run_dir / "pair_compatibility.jsonl")
    candidate_rows = iter_jsonl(run_dir / "candidate_pool.jsonl")
    selected = selected_rows[0] if selected_rows else {}
    pair = pair_rows[0] if pair_rows else {}
    candidate = candidate_rows[0] if candidate_rows else {}
    return selected, pair, candidate


def preview_output_path(
    *,
    run_dir: Path,
    run_id: str,
    track_id: str,
    source_candidate_id: str,
    stem_type: str,
) -> Path:
    prefix = "__".join(
        [
            stable_slug(run_id),
            stable_slug(track_id),
            stable_slug(source_candidate_id),
            stem_type,
            short_id(f"{run_id}|{track_id}|{source_candidate_id}|{stem_type}"),
        ]
    )
    return run_dir / "audio_previews" / f"{prefix}.flac"


def render_segment(
    *,
    source_path: Path,
    output_path: Path,
    start_sec: float,
    duration_sec: float,
    render_mode: str,
) -> dict[str, Any]:
    audio, sample_rate = sf.read(str(source_path), always_2d=True, dtype="float32")
    if audio.size == 0:
        raise ValueError(f"empty_audio:{source_path}")
    start_frame = max(0, int(round(float(start_sec) * int(sample_rate))))
    if start_frame >= audio.shape[0]:
        raise ValueError(f"preview_start_beyond_audio:{source_path}")
    if render_mode == "full":
        end_frame = audio.shape[0]
    else:
        requested_frames = max(1, int(round(float(duration_sec) * int(sample_rate))))
        end_frame = min(audio.shape[0], start_frame + requested_frames)
    if end_frame <= start_frame:
        raise ValueError(f"empty_preview_segment:{source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), np.asarray(audio[start_frame:end_frame], dtype=np.float32), int(sample_rate))
    return audio_metadata(output_path)


def skipped_or_failed_row(
    *,
    run_id: str,
    track_id: str,
    stem_type: str,
    source_candidate_id: str,
    source_path: str,
    render_status: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "track_id": track_id,
        "preview_audio_id": stable_slug(f"{run_id}__{source_candidate_id}__{stem_type}"),
        "stem_type": stem_type,
        "source_candidate_id": source_candidate_id,
        "source_path": source_path,
        "rendered_path": "",
        "duration_sec": 0.0,
        "sample_rate": 0,
        "channels": 0,
        "render_status": render_status,
        "validation_status": render_status,
        "failure_reason": reason if render_status == "failed" else "",
        "skip_reason": reason if render_status == "skipped" else "",
        "release_ready": False,
    }


def rendered_row(
    *,
    run_id: str,
    track_id: str,
    stem_type: str,
    source_candidate_id: str,
    source_path: str,
    rendered_path: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "track_id": track_id,
        "preview_audio_id": stable_slug(f"{run_id}__{source_candidate_id}__{stem_type}"),
        "stem_type": stem_type,
        "source_candidate_id": source_candidate_id,
        "source_path": source_path,
        "rendered_path": str(rendered_path),
        "duration_sec": float(metadata["duration_sec"]),
        "sample_rate": int(metadata["sample_rate"]),
        "channels": int(metadata["channels"]),
        "render_status": "rendered",
        "validation_status": "validated",
        "failure_reason": "",
        "release_ready": False,
    }


def base_preview_manifest(
    *,
    run_dir: Path,
    run_manifest: dict[str, Any],
    selected: dict[str, Any],
    pair: dict[str, Any],
    preview_duration_sec: float,
    preview_start_sec: float,
    preview_render_mode: str,
    require_preview_audio_for_preview_ready: bool,
) -> dict[str, Any]:
    run_id = str(run_manifest.get("run_id") or Path(str(run_manifest.get("output_dir") or run_dir)).name)
    track_id = str(selected.get("track_id") or pair.get("track_id") or run_manifest.get("track_id") or Path(str(run_manifest.get("input_audio") or run_id)).stem)
    selected_vocal_path = str(selected.get("selected_vocal_path") or selected.get("vocal_path") or "")
    selected_minus_path = str(selected.get("selected_minus_path") or selected.get("instrumental_path") or selected.get("minus_path") or "")
    return {
        "run_id": run_id,
        "track_id": track_id,
        "preview_status": "manifest_preview_ready",
        "accepted_pair_ready": pair.get("pair_decision") == "pair_compatible",
        "manifest_preview_ready": bool(selected and pair.get("pair_decision") == "pair_compatible"),
        "audio_preview_ready": False,
        "preview_render_attempted": False,
        "preview_render_skipped": False,
        "preview_render_failed": False,
        "presentation_status": "diagnostic_preview_only",
        "selected_vocal_candidate_id": str(selected.get("selected_vocal_candidate_id") or pair.get("selected_vocal_candidate_id") or ""),
        "selected_minus_candidate_id": str(selected.get("selected_minus_candidate_id") or pair.get("selected_minus_candidate_id") or ""),
        "selected_vocal_path": selected_vocal_path,
        "selected_minus_path": selected_minus_path,
        "rendered_vocal_preview_path": "",
        "rendered_minus_preview_path": "",
        "rendered_pair_preview_path": "",
        "rendered_preview_audio": [],
        "rendered_preview_audio_count": 0,
        "duration_sec": round(float(preview_duration_sec), 6),
        "preview_start_sec": round(float(preview_start_sec), 6),
        "preview_render_mode": preview_render_mode,
        "require_preview_audio_for_preview_ready": bool(require_preview_audio_for_preview_ready),
        "sample_rate": 0,
        "channels": 0,
        "validation_status": "pending",
        "failure_reason": "",
        "skip_reason": "",
        **SAFETY_FLAGS,
    }


def finalize_manifest_from_rows(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    rendered = [row for row in rows if row.get("render_status") == "rendered" and row.get("validation_status") == "validated"]
    rendered_paths = [str(row.get("rendered_path") or "") for row in rendered if row.get("rendered_path")]
    manifest["rendered_preview_audio"] = rendered_paths
    manifest["rendered_preview_audio_count"] = len(rendered_paths)
    manifest["audio_preview_ready"] = len(rendered) >= 2
    manifest["validation_status"] = "validated" if manifest["audio_preview_ready"] else manifest.get("validation_status", "pending")
    if rendered:
        manifest["sample_rate"] = int(rendered[0].get("sample_rate") or 0)
        manifest["channels"] = int(rendered[0].get("channels") or 0)
        manifest["duration_sec"] = round(min(float(row.get("duration_sec") or 0.0) for row in rendered), 6)
    for row in rendered:
        if row.get("stem_type") == "vocal_preview":
            manifest["rendered_vocal_preview_path"] = str(row.get("rendered_path") or "")
        if row.get("stem_type") == "minus_preview":
            manifest["rendered_minus_preview_path"] = str(row.get("rendered_path") or "")
    return manifest


def write_preview_outputs(run_dir: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    contract = {
        "contract_version": "no_reference_audio_preview_render_v1",
        "run_id": manifest["run_id"],
        "track_id": manifest["track_id"],
        "preview_status": manifest["preview_status"],
        "rendered_preview_audio_count": len([row for row in rows if row.get("render_status") == "rendered"]),
        "rows": rows,
        "safety": dict(SAFETY_FLAGS),
    }
    write_json(run_dir / "preview_manifest.json", manifest)
    write_jsonl(run_dir / "rendered_preview_audio.jsonl", rows)
    write_json(run_dir / "preview_render_contract.json", contract)
    write_preview_summary(run_dir / "preview_render_summary.md", manifest, rows)


def write_preview_summary(path: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    rendered_count = sum(1 for row in rows if row.get("render_status") == "rendered")
    skipped_count = sum(1 for row in rows if row.get("render_status") == "skipped")
    failed_count = sum(1 for row in rows if row.get("render_status") == "failed")
    lines = [
        "# No-Reference Audio Preview Render Contract",
        "",
        f"- preview_status: {manifest['preview_status']}",
        f"- accepted_pair_ready: {str(manifest['accepted_pair_ready']).lower()}",
        f"- manifest_preview_ready: {str(manifest['manifest_preview_ready']).lower()}",
        f"- audio_preview_ready: {str(manifest['audio_preview_ready']).lower()}",
        f"- preview_render_attempted: {str(manifest['preview_render_attempted']).lower()}",
        f"- preview_render_skipped: {str(manifest['preview_render_skipped']).lower()}",
        f"- preview_render_failed: {str(manifest['preview_render_failed']).lower()}",
        f"- rendered_preview_audio_count: {rendered_count}",
        f"- skipped_preview_audio_count: {skipped_count}",
        f"- failed_preview_audio_count: {failed_count}",
        f"- rendered_vocal_preview_path: {manifest.get('rendered_vocal_preview_path') or ''}",
        f"- rendered_minus_preview_path: {manifest.get('rendered_minus_preview_path') or ''}",
        f"- validation_status: {manifest.get('validation_status') or ''}",
        f"- failure_reason: {manifest.get('failure_reason') or ''}",
        f"- skip_reason: {manifest.get('skip_reason') or ''}",
        "- can_publish_as_result: false",
        "- can_mark_release_ready: false",
        "- release_ready: false",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_audio_preview_contract(
    *,
    run_dir: Path,
    preview_duration_sec: float = 30.0,
    preview_start_sec: float | None = None,
    preview_render_mode: str = "bounded",
    enable_audio_preview_render: bool = True,
    require_preview_audio_for_preview_ready: bool = True,
) -> dict[str, Any]:
    if preview_render_mode not in {"bounded", "full"}:
        raise ValueError(f"Unsupported preview_render_mode: {preview_render_mode}")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = read_json(run_dir / "run_manifest.json")
    selected, pair, _candidate = find_selected_pair(run_dir)
    start_sec = 0.0 if preview_start_sec is None else max(0.0, float(preview_start_sec))
    manifest = base_preview_manifest(
        run_dir=run_dir,
        run_manifest=run_manifest,
        selected=selected,
        pair=pair,
        preview_duration_sec=float(preview_duration_sec),
        preview_start_sec=start_sec,
        preview_render_mode=preview_render_mode,
        require_preview_audio_for_preview_ready=require_preview_audio_for_preview_ready,
    )
    vocal_path = manifest["selected_vocal_path"]
    minus_path = manifest["selected_minus_path"]
    rows: list[dict[str, Any]] = []
    stems = [
        ("vocal_preview", manifest["selected_vocal_candidate_id"], vocal_path, "selected_vocal_audio_missing"),
        ("minus_preview", manifest["selected_minus_candidate_id"], minus_path, "selected_minus_audio_missing"),
    ]
    if not manifest["accepted_pair_ready"] or not manifest["manifest_preview_ready"]:
        reason = "pair_not_compatible_or_missing_selection"
        manifest.update(
            {
                "preview_status": "preview_blocked_safety_contract",
                "preview_render_failed": True,
                "validation_status": "blocked",
                "failure_reason": reason,
            }
        )
        for stem_type, candidate_id, source_path, _missing_reason in stems:
            rows.append(
                skipped_or_failed_row(
                    run_id=manifest["run_id"],
                    track_id=manifest["track_id"],
                    stem_type=stem_type,
                    source_candidate_id=candidate_id,
                    source_path=source_path,
                    render_status="failed",
                    reason=reason,
                )
            )
        write_preview_outputs(run_dir, manifest, rows)
        return manifest
    if not enable_audio_preview_render:
        reason = "audio_preview_render_disabled"
        manifest.update(
            {
                "preview_status": "preview_render_skipped",
                "preview_render_skipped": True,
                "validation_status": "skipped",
                "skip_reason": reason,
            }
        )
        for stem_type, candidate_id, source_path, _missing_reason in stems:
            rows.append(
                skipped_or_failed_row(
                    run_id=manifest["run_id"],
                    track_id=manifest["track_id"],
                    stem_type=stem_type,
                    source_candidate_id=candidate_id,
                    source_path=source_path,
                    render_status="skipped",
                    reason=reason,
                )
            )
        write_preview_outputs(run_dir, manifest, rows)
        return manifest
    manifest["preview_render_attempted"] = True
    failure_reasons: list[str] = []
    source_paths = [vocal_path, minus_path]
    for stem_type, candidate_id, source_path, missing_reason in stems:
        source = Path(source_path)
        if not source_path or not source.exists():
            failure_reasons.append(missing_reason)
            rows.append(
                skipped_or_failed_row(
                    run_id=manifest["run_id"],
                    track_id=manifest["track_id"],
                    stem_type=stem_type,
                    source_candidate_id=candidate_id,
                    source_path=source_path,
                    render_status="failed",
                    reason=missing_reason,
                )
            )
            continue
        output_path = preview_output_path(
            run_dir=run_dir,
            run_id=manifest["run_id"],
            track_id=manifest["track_id"],
            source_candidate_id=candidate_id,
            stem_type=stem_type,
        )
        try:
            assert_preview_output_path(run_dir, output_path, source_paths)
            metadata = render_segment(
                source_path=source,
                output_path=output_path,
                start_sec=start_sec,
                duration_sec=float(preview_duration_sec),
                render_mode=preview_render_mode,
            )
        except Exception as exc:  # pragma: no cover - defensive corrupt-user-artifact path.
            reason = f"{missing_reason}:{type(exc).__name__}:{exc}"
            failure_reasons.append(reason)
            rows.append(
                skipped_or_failed_row(
                    run_id=manifest["run_id"],
                    track_id=manifest["track_id"],
                    stem_type=stem_type,
                    source_candidate_id=candidate_id,
                    source_path=source_path,
                    render_status="failed",
                    reason=reason,
                )
            )
            continue
        rows.append(
            rendered_row(
                run_id=manifest["run_id"],
                track_id=manifest["track_id"],
                stem_type=stem_type,
                source_candidate_id=candidate_id,
                source_path=source_path,
                rendered_path=output_path,
                metadata=metadata,
            )
        )
    finalize_manifest_from_rows(manifest, rows)
    if manifest["audio_preview_ready"]:
        manifest["preview_status"] = "audio_preview_ready"
    elif failure_reasons:
        manifest.update(
            {
                "preview_status": "preview_render_failed",
                "preview_render_failed": True,
                "validation_status": "failed",
                "failure_reason": ";".join(dict.fromkeys(failure_reasons)),
            }
        )
    else:
        manifest.update(
            {
                "preview_status": "manifest_preview_ready",
                "validation_status": "manifest_only",
            }
        )
    if require_preview_audio_for_preview_ready and manifest["preview_status"] == "manifest_preview_ready":
        manifest["skip_reason"] = "preview_audio_required_but_not_rendered"
    write_preview_outputs(run_dir, manifest, rows)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render bounded no-reference vocal/minus audio previews for a runtime run directory.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preview-duration-sec", type=float, default=30.0)
    parser.add_argument("--preview-start-sec", type=float)
    parser.add_argument("--preview-render-mode", choices=("bounded", "full"), default="bounded")
    parser.add_argument("--enable-audio-preview-render", type=parse_bool, default=True)
    parser.add_argument("--require-preview-audio-for-preview-ready", type=parse_bool, default=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = render_audio_preview_contract(
        run_dir=args.run_dir,
        preview_duration_sec=args.preview_duration_sec,
        preview_start_sec=args.preview_start_sec,
        preview_render_mode=args.preview_render_mode,
        enable_audio_preview_render=args.enable_audio_preview_render,
        require_preview_audio_for_preview_ready=args.require_preview_audio_for_preview_ready,
    )
    print(
        json.dumps(
            {
                "preview_status": manifest["preview_status"],
                "audio_preview_ready": manifest["audio_preview_ready"],
                "rendered_preview_audio_count": manifest["rendered_preview_audio_count"],
                "release_ready": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
