from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import soundfile as sf


ALLOWED_STEM_TYPES = {
    "vocal",
    "minus",
    "instrumental",
    "piano",
    "bass",
    "drums",
    "other",
    "unknown",
}

ANALYSIS_READY_RUNTIME_STATUSES = {"preview_ready", "accepted_for_listen_pack"}

STEM_PATH_FIELDS: tuple[tuple[str, str, str, str], ...] = (
    ("vocal", "vocal_path", "normalized_vocal_path", "vocal"),
    ("minus", "minus_path", "normalized_minus_path", "minus"),
    ("instrumental", "instrumental_path", "normalized_instrumental_path", "instrumental"),
    ("piano", "piano_path", "normalized_piano_path", "piano"),
    ("bass", "bass_path", "normalized_bass_path", "bass"),
    ("drums", "drums_path", "normalized_drums_path", "drums"),
    ("other", "other_path", "normalized_other_path", "other"),
    ("unknown", "audio_path", "normalized_audio_path", "unknown"),
)

SELECTED_STEM_PATH_FIELDS: tuple[tuple[str, str, str, str], ...] = (
    ("vocal", "selected_vocal_path", "normalized_selected_vocal_path", "vocal"),
    ("minus", "selected_minus_path", "normalized_selected_minus_path", "minus"),
)


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
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if isinstance(value, dict):
            items = [str(key) for key, enabled in value.items() if enabled]
        elif isinstance(value, list):
            items = [str(item) for item in value]
        elif value:
            items = [str(value)]
        else:
            items = []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                result.append(item)
    return result


def stable_slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return text.strip("_") or "unknown"


def candidate_id_for(row: dict[str, Any], stem_type: str) -> str:
    for key in (
        "candidate_id",
        f"{stem_type}_candidate_id",
        f"selected_{stem_type}_candidate_id",
        "selected_vocal_candidate_id",
        "selected_minus_candidate_id",
    ):
        value = str(row.get(key) or "")
        if value:
            return value
    return "unknown_candidate"


def audio_metadata(path: str) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {
            "sample_rate": 0,
            "channels": 0,
            "duration_sec": 0.0,
            "metadata_blocker": "missing_audio_path",
        }
    try:
        info = sf.info(path)
    except Exception as exc:  # pragma: no cover - defensive against corrupt user artifacts.
        return {
            "sample_rate": 0,
            "channels": 0,
            "duration_sec": 0.0,
            "metadata_blocker": f"audio_metadata_error:{type(exc).__name__}",
        }
    duration = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "duration_sec": round(duration, 6),
        "metadata_blocker": "",
    }


def pair_status_for(
    *,
    pair_rows: list[dict[str, Any]],
    vocal_candidate_id: str,
    minus_candidate_id: str,
    candidate_id: str,
) -> str:
    if not pair_rows:
        return "not_evaluated"
    for row in pair_rows:
        selected_vocal = str(row.get("selected_vocal_candidate_id") or "")
        selected_minus = str(row.get("selected_minus_candidate_id") or "")
        if (
            (selected_vocal == vocal_candidate_id and selected_minus == minus_candidate_id)
            or selected_vocal == candidate_id
            or selected_minus == candidate_id
        ):
            return str(row.get("pair_decision") or row.get("runtime_status") or "evaluated")
    return "not_selected_pair"


def source_separator_config(row: dict[str, Any]) -> str:
    for key in ("source_separator_config", "separator_config_id", "candidate_config_id", "source_config_id"):
        value = str(row.get(key) or "")
        if value:
            return value
    return ""


def source_candidate_family(row: dict[str, Any]) -> str:
    for key in ("source_candidate_family", "candidate_family", "family", "candidate_config_family"):
        value = str(row.get(key) or "")
        if value:
            return value
    return ""


def base_blockers(runtime_manifest: dict[str, Any]) -> list[str]:
    return stable_unique(
        [
            runtime_manifest.get("failure_reason"),
            runtime_manifest.get("safety_blockers"),
        ]
    )


def row_blockers(row: dict[str, Any]) -> list[str]:
    return stable_unique(
        [
            row.get("blockers"),
            row.get("script_blockers"),
            row.get("minus_blockers"),
            row.get("failure_reasons"),
            row.get("skip_reason"),
            row.get("judge_reason"),
        ]
    )


def selected_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = iter_jsonl(run_dir / "selected_candidates.jsonl")
    candidate_metadata = {
        str(row.get("candidate_id") or ""): row
        for row in iter_jsonl(run_dir / "candidate_pool.jsonl")
        if str(row.get("candidate_id") or "")
    }
    result: list[dict[str, Any]] = []
    for row in rows:
        vocal_candidate_id = str(row.get("selected_vocal_candidate_id") or "")
        minus_candidate_id = str(row.get("selected_minus_candidate_id") or "")
        for stem_type, path_key, normalized_key, hint in SELECTED_STEM_PATH_FIELDS:
            audio_path = str(row.get(path_key) or "")
            if not audio_path:
                continue
            stem_row = dict(row)
            stem_row.update(
                {
                    "candidate_id": vocal_candidate_id if stem_type == "vocal" else minus_candidate_id,
                    "stem_type": stem_type,
                    "stem_audio_path": audio_path,
                    "stem_normalized_audio_path": str(row.get(normalized_key) or audio_path),
                    "instrument_hint": str(row.get("instrument_hint") or hint),
                    "selected_for_runtime": True,
                    "selected_vocal_candidate_id": vocal_candidate_id,
                    "selected_minus_candidate_id": minus_candidate_id,
                }
            )
            metadata = candidate_metadata.get(str(stem_row["candidate_id"]) or "")
            if metadata:
                for key in (
                    "source_separator_config",
                    "separator_config_id",
                    "candidate_config_id",
                    "source_config_id",
                    "source_candidate_family",
                    "candidate_family",
                    "family",
                    "candidate_config_family",
                ):
                    if key not in stem_row and metadata.get(key):
                        stem_row[key] = metadata[key]
            result.append(stem_row)
    return result


def candidate_pool_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = iter_jsonl(run_dir / "candidate_pool.jsonl") + iter_jsonl(run_dir / "candidate_pool_invalid_rows.jsonl")
    result: list[dict[str, Any]] = []
    for row in rows:
        emitted_paths: set[str] = set()
        for stem_type, path_key, normalized_key, hint in STEM_PATH_FIELDS:
            audio_path = str(row.get(path_key) or "")
            if not audio_path:
                continue
            if stem_type == "instrumental" and audio_path in emitted_paths:
                continue
            emitted_paths.add(audio_path)
            stem_row = dict(row)
            stem_row.update(
                {
                    "stem_type": stem_type,
                    "stem_audio_path": audio_path,
                    "stem_normalized_audio_path": str(row.get(normalized_key) or audio_path),
                    "instrument_hint": str(row.get("instrument_hint") or hint),
                    "selected_for_runtime": False,
                }
            )
            result.append(stem_row)
    return result


def dedupe_stem_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("candidate_id") or ""),
            str(row.get("stem_type") or ""),
            str(row.get("stem_audio_path") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def quality_status_for(
    *,
    runtime_status: str,
    selected_for_runtime: bool,
    release_ready_forbidden: bool,
    blockers: list[str],
    metadata_blocker: str,
    pair_compatibility_status: str,
) -> str:
    if release_ready_forbidden:
        return "blocked"
    if metadata_blocker:
        return "blocked"
    if not selected_for_runtime:
        return "blocked"
    if runtime_status not in ANALYSIS_READY_RUNTIME_STATUSES:
        return "blocked"
    if blockers:
        return "blocked"
    if pair_compatibility_status not in {"pair_compatible", "not_evaluated"}:
        return "blocked"
    return "accepted"


def analysis_block_reason_for(
    *,
    quality_status: str,
    release_ready_forbidden: bool,
    metadata_blocker: str,
    blockers: list[str],
    selected_for_runtime: bool,
    runtime_status: str,
    pair_compatibility_status: str,
) -> str:
    if quality_status == "accepted":
        return ""
    reasons: list[str] = []
    if release_ready_forbidden:
        reasons.append("release_ready_forbidden")
    if metadata_blocker:
        reasons.append(metadata_blocker)
    reasons.extend(blockers)
    if not selected_for_runtime:
        reasons.append("stem_not_selected_for_runtime_output")
    if runtime_status not in ANALYSIS_READY_RUNTIME_STATUSES:
        reasons.append(f"runtime_status:{runtime_status}")
    if pair_compatibility_status not in {"pair_compatible", "not_evaluated"}:
        reasons.append(f"pair_compatibility_status:{pair_compatibility_status}")
    return ";".join(stable_unique(reasons))


def build_stem_entry(
    *,
    row: dict[str, Any],
    runtime_manifest: dict[str, Any],
    pair_rows: list[dict[str, Any]],
    run_id: str,
    default_track_id: str,
    runtime_blockers: list[str],
) -> dict[str, Any]:
    stem_type = str(row.get("stem_type") or "unknown")
    if stem_type not in ALLOWED_STEM_TYPES:
        stem_type = "unknown"
    candidate_id = candidate_id_for(row, stem_type)
    audio_path = str(row.get("stem_audio_path") or "")
    normalized_audio_path = str(row.get("stem_normalized_audio_path") or audio_path)
    track_id = str(row.get("track_id") or default_track_id)
    vocal_candidate_id = str(row.get("selected_vocal_candidate_id") or row.get("candidate_id") or "")
    minus_candidate_id = str(row.get("selected_minus_candidate_id") or row.get("candidate_id") or "")
    pair_status = pair_status_for(
        pair_rows=pair_rows,
        vocal_candidate_id=vocal_candidate_id,
        minus_candidate_id=minus_candidate_id,
        candidate_id=candidate_id,
    )
    metadata = audio_metadata(normalized_audio_path or audio_path)
    release_ready_forbidden = runtime_manifest.get("release_ready") is True or row.get("release_ready") is True
    blockers = stable_unique(runtime_blockers + row_blockers(row))
    quality_status = quality_status_for(
        runtime_status=str(runtime_manifest.get("output_status") or ""),
        selected_for_runtime=row.get("selected_for_runtime") is True,
        release_ready_forbidden=release_ready_forbidden,
        blockers=blockers,
        metadata_blocker=str(metadata["metadata_blocker"]),
        pair_compatibility_status=pair_status,
    )
    block_reason = analysis_block_reason_for(
        quality_status=quality_status,
        release_ready_forbidden=release_ready_forbidden,
        metadata_blocker=str(metadata["metadata_blocker"]),
        blockers=blockers,
        selected_for_runtime=row.get("selected_for_runtime") is True,
        runtime_status=str(runtime_manifest.get("output_status") or ""),
        pair_compatibility_status=pair_status,
    )
    digest = hashlib.sha1(f"{run_id}|{candidate_id}|{stem_type}|{audio_path}".encode("utf-8")).hexdigest()[:10]
    return {
        "track_id": track_id,
        "run_id": run_id,
        "candidate_id": candidate_id,
        "stem_id": stable_slug(f"{run_id}__{candidate_id}__{stem_type}__{digest}"),
        "stem_type": stem_type,
        "instrument_hint": str(row.get("instrument_hint") or stem_type),
        "audio_path": audio_path,
        "normalized_audio_path": normalized_audio_path,
        "sample_rate": metadata["sample_rate"],
        "channels": metadata["channels"],
        "duration_sec": metadata["duration_sec"],
        "quality_status": quality_status,
        "blockers": blockers,
        "analysis_allowed": quality_status == "accepted" and not block_reason,
        "analysis_block_reason": block_reason,
        "source_separator_config": source_separator_config(row),
        "source_candidate_family": source_candidate_family(row),
        "pair_compatibility_status": pair_status,
        "release_ready": False,
    }


def build_music_intelligence_input_manifest(
    *,
    run_dir: Path,
    runtime_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_manifest = dict(runtime_manifest or read_json(run_dir / "run_manifest.json"))
    run_id = str(runtime_manifest.get("run_id") or Path(runtime_manifest.get("output_dir") or run_dir).name)
    default_track_id = str(runtime_manifest.get("track_id") or Path(str(runtime_manifest.get("input_audio") or run_id)).stem)
    pair_rows = iter_jsonl(run_dir / "pair_compatibility.jsonl")
    selected = selected_rows(run_dir)
    source_rows = selected if selected else candidate_pool_rows(run_dir)
    runtime_blockers = base_blockers(runtime_manifest)
    stems = [
        build_stem_entry(
            row=row,
            runtime_manifest=runtime_manifest,
            pair_rows=pair_rows,
            run_id=run_id,
            default_track_id=default_track_id,
            runtime_blockers=runtime_blockers,
        )
        for row in dedupe_stem_rows(source_rows)
    ]
    stems.sort(key=lambda row: (row["track_id"], row["candidate_id"], row["stem_type"], row["audio_path"]))
    analysis_ready_count = sum(1 for row in stems if row["analysis_allowed"] is True)
    blocked_count = len(stems) - analysis_ready_count
    return {
        "schema_version": 1,
        "run_id": run_id,
        "track_id": default_track_id,
        "run_dir": str(run_dir),
        "runtime_status": str(runtime_manifest.get("output_status") or ""),
        "stems": stems,
        "analysis_ready_stems_count": analysis_ready_count,
        "blocked_stems_count": blocked_count,
        "future_analysis_supported": False,
        "melody_transcription_ran": False,
        "recommendation_index_updated": False,
        "release_ready": False,
    }


def write_music_intelligence_input_manifest(
    *,
    run_dir: Path,
    output_path: Path | None = None,
    runtime_manifest: dict[str, Any] | None = None,
) -> Path:
    output_path = output_path or (run_dir / "music_intelligence_input_manifest.json")
    manifest = build_music_intelligence_input_manifest(run_dir=run_dir, runtime_manifest=runtime_manifest)
    write_json(output_path, manifest)
    return output_path


def music_intelligence_run_section(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    return {
        "manifest_written": bool(manifest_path.exists()),
        "manifest_path": str(manifest_path),
        "analysis_ready_stems_count": int(manifest.get("analysis_ready_stems_count") or 0),
        "blocked_stems_count": int(manifest.get("blocked_stems_count") or 0),
        "future_analysis_supported": False,
        "melody_transcription_ran": False,
        "recommendation_index_updated": False,
        "release_ready": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-json", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    output_path = Path(args.output_json) if args.output_json else run_dir / "music_intelligence_input_manifest.json"
    path = write_music_intelligence_input_manifest(run_dir=run_dir, output_path=output_path)
    manifest = read_json(path)
    print(
        json.dumps(
            {
                "manifest_path": str(path),
                "analysis_ready_stems_count": manifest.get("analysis_ready_stems_count", 0),
                "blocked_stems_count": manifest.get("blocked_stems_count", 0),
                "melody_transcription_ran": False,
                "recommendation_index_updated": False,
                "release_ready": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
